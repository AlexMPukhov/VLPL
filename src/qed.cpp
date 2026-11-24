// Spin-resolved strong-field QED (qed.hpp, docs/specs/qed.md, decision 93): deck, setup, the qed operator (stage Qed:
// T-BMT and the emission), the ballistic photon move (stage MoveDeposit), the split push of the species with spin or
// emission (stage Push: gather and push only; the move and deposit follow in stage MoveDeposit, decision 15).
#include "qed.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include "deck.hpp"
#include "fsutil.hpp"
#include "sim.hpp"
#if RIP_GPU
#include <cub/cub.cuh>
#endif

namespace {
template <class T> void ensure(Buf<T>& b, size_t n) { if (b.size() < n) b.resize(std::max(n, b.size() + b.size() / 2)); }

long scan(const int* C, int* O, long n, Buf<unsigned char>& tmp) {   // exclusive sum, returns the total
#if RIP_GPU
    size_t bytes = 0;
    gpu_check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, C, O, int(n)), "cub scan (size)");
    ensure(tmp, bytes);
    gpu_check(cub::DeviceScan::ExclusiveSum(tmp.data(), bytes, C, O, int(n)), "cub scan");
    int last[2] = {0, 0};
    gpu_check(cudaMemcpy(&last[0], C + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "scan last");
    gpu_check(cudaMemcpy(&last[1], O + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "scan last");
    return long(last[0]) + last[1];
#else
    (void)tmp;
    long s = 0;
    for (long p = 0; p < n; ++p) { O[p] = int(s); s += C[p]; }
    return s;
#endif
}

// ids (random-number streams) of particles that have none: a hash of (salt, position) when first seen (as ionization.cpp)
void assign_ids(Species& sp, unsigned long long salt) {
    long long* ID = sp.ptr<long long>(sp.slot("id"));
    const double *X = sp.ptr<double>(Species::X), *Y = sp.ptr<double>(Species::Y), *Z = sp.ptr<double>(Species::Z);
    const unsigned long long base = ion_mix(0x5eedULL + salt);
    par_for(long(sp.size()), [=] HD(long p) {
        if (ID[p] != 0) return;
        unsigned long long h = base, b;
        double v = X[p]; memcpy(&b, &v, 8); h = ion_mix(h ^ b);
        v = Y[p]; memcpy(&b, &v, 8); h = ion_mix(h ^ b);
        v = Z[p]; memcpy(&b, &v, 8); h = ion_mix(h ^ b);
        ID[p] = (long long)(h == 0 ? 1 : h);
    });
}
}  // namespace

struct QedSpecies {
    int index = -1;
    bool spin = false, emit = false;
    double qm = -1, anomaly = 0;
    int s0 = -1, chi = -1, id = -1;
    Buf<double> eb;                            // per particle: E, B at x^n, u^{n-1/2} (9)
    Buf<double> tot, hu, he;                   // tallies (8) and the spectra (bins x 5), local
};

struct Qed {
    QedConfig c;
    QedK K;
    Buf<double> fy, fc, fb;                    // device tables (fb: Breit-Wheeler T0, T3, Mb)
    std::vector<QedSpecies> sp;
    int photon = -1;                           // species index of the photons (-1: none)
    Buf<int> cnt, off;
    // Breit-Wheeler pairs (decision 96)
    BwK bk;
    int pair_e = -1, pair_p = -1;              // species indices (-1: no pairs)
    Buf<double> btot, bhx, bhe;                // tallies (8), spectra in x (bins x 4) and in eps (bins x 2), local
    Buf<int> keep, perm;
    Buf<double> ebph;                          // E, B, u at the photons (9 per photon)
    Buf<unsigned char> tmp;
    FILE* f = nullptr;
    ~Qed() { if (f) std::fclose(f); }
};

// ---------------------------------------------------------------- deck
SpinConfig spin_from_deck(const Deck& d, const std::string& b) {
    SpinConfig s;
    if (!d.has(b, "spin")) {
        if (d.has(b, "spin_anomaly")) throw std::runtime_error(b + ".spin_anomaly: needs spin");
        return s;
    }
    std::istringstream is(d.str(b, "spin"));
    for (int k = 0; k < 3; ++k)
        if (!(is >> s.s[k])) throw std::runtime_error(b + ".spin: three numbers (sx sy sz, |S| <= 1)");
    const double n = std::sqrt(s.s[0] * s.s[0] + s.s[1] * s.s[1] + s.s[2] * s.s[2]);
    if (n > 1 + 1e-12) throw std::runtime_error(b + ".spin: |S| <= 1");
    s.on = true;
    s.anomaly = d.num(b, "spin_anomaly", Kind::None, s.anomaly);
    return s;
}

QedConfig qed_from_deck(const Deck& d, double) {
    QedConfig q;
    if (!d.has_block("qed")) return q;
    const std::string b = "qed";
    q.on = true;
    const std::string m = d.str(b, "model", "monte_carlo");
    if (m == "monte_carlo") q.model = QedConfig::MonteCarlo;
    else if (m == "deterministic") q.model = QedConfig::Deterministic;
    else if (m == "none") q.model = QedConfig::None;
    else throw std::runtime_error("qed.model: monte_carlo, deterministic or none");
    std::istringstream is(d.str(b, "species", ""));
    for (std::string w; is >> w;) q.species.push_back(w);
    q.photons = d.str(b, "photons", "");
    q.photon_min = d.num(b, "photon_min_energy", Kind::None, 0.0) / 0.51099895000;   // MeV -> m c^2
    const std::string st = d.str(b, "spin_state", "mixed");
    if (st != "mixed" && st != "pure") throw std::runtime_error("qed.spin_state: mixed or pure");
    q.pure = st == "pure";
    q.precession = d.flag(b, "spin_precession", true);
    q.spin_flip = d.flag(b, "spin_flip", true);
    q.substep_prob = d.num(b, "substep_prob", Kind::None, 0.1);
    q.bins = int(d.integer(b, "spectrum_bins", 200));
    q.u_min = d.num(b, "spectrum_u_min", Kind::None, 1e-8);
    q.e_min_mev = d.num(b, "spectrum_e_min", Kind::None, 1e-3);
    q.e_max_mev = d.num(b, "spectrum_e_max", Kind::None, 1e7);
    q.spectrum_every = d.integer(b, "spectrum_every", 0);
    q.pair_e = d.str(b, "pair_electrons", "");
    q.pair_p = d.str(b, "pair_positrons", "");
    q.pair_selection = d.flag(b, "pair_selection", true);
    if (q.pair_e.empty() != q.pair_p.empty()) throw std::runtime_error("qed: pair_electrons and pair_positrons together");
    if (!(q.substep_prob > 0 && q.substep_prob <= 1)) throw std::runtime_error("qed.substep_prob: in (0, 1]");
    if (q.bins < 1 || !(q.u_min > 0 && q.u_min < 1) || !(q.e_min_mev > 0 && q.e_max_mev > q.e_min_mev))
        throw std::runtime_error("qed: spectrum_bins >= 1, 0 < spectrum_u_min < 1, 0 < spectrum_e_min < spectrum_e_max");
    return q;
}

static int find(const Config& c, const std::string& n) {
    for (size_t i = 0; i < c.species.size(); ++i)
        if (c.species[i].name == n) return int(i);
    return -1;
}

void qed_check(const Config& c) {
    for (const SpeciesConfig& s : c.species) {
        if (s.photon && s.spin.on) throw std::runtime_error("species " + s.name + ": kind = photon has no spin");
        if (!s.spin.on) continue;
        if (s.ion.on) throw std::runtime_error("species " + s.name + ": spin not with field ionization (charge states)");
        if (!s.mobile) throw std::runtime_error("species " + s.name + ": spin needs a mobile species");
    }
    bool any = c.qed.on;
    for (const SpeciesConfig& s : c.species) any = any || s.spin.on;
    if (!any) return;
    if (c.mover_implicit) throw std::runtime_error("qed / spin: not with numerics.mover = implicit");
    // boosted frame (decision 117): the rates are covariant (chi invariant, W dt' = invariant x proper time); the energies of
    // the tallies, the spectra and photon_min_energy are lab energies, the deck spin a lab rest-frame spin (qed_register)
    const QedConfig& q = c.qed;
    for (const std::string& n : q.species) {
        const int i = find(c, n);
        if (i < 0) throw std::runtime_error("qed.species: no species " + n);
        const SpeciesConfig& s = c.species[size_t(i)];
        if (s.photon || s.ion.on || std::abs(std::abs(s.q) - 1) > 1e-12 || std::abs(s.m - 1) > 1e-12 || !s.mobile)
            throw std::runtime_error("qed.species: " + n + " must be mobile electrons or positrons (|q| = 1, m = 1)");
    }
    if (!q.photons.empty()) {
        const int i = find(c, q.photons);
        if (i < 0 || !c.species[size_t(i)].photon) throw std::runtime_error("qed.photons: " + q.photons + " must be a species with kind = photon");
    }
    if (q.pairs()) {
        if (q.photons.empty()) throw std::runtime_error("qed.pair_electrons / pair_positrons: need qed.photons (the photons that decay)");
        if (q.model != QedConfig::MonteCarlo) throw std::runtime_error("qed: pairs need model = monte_carlo");
        if (q.pair_e == q.pair_p) throw std::runtime_error("qed: pair_electrons and pair_positrons must be two species");
        for (int k = 0; k < 2; ++k) {
            const std::string& n = k ? q.pair_p : q.pair_e;
            const int i = find(c, n);
            if (i < 0) throw std::runtime_error("qed: no pair species " + n);
            const SpeciesConfig& s = c.species[size_t(i)];
            const double qs = k ? 1.0 : -1.0;
            if (s.photon || s.ion.on || std::abs(s.q - qs) > 1e-12 || std::abs(s.m - 1) > 1e-12 || !s.mobile)
                throw std::runtime_error("qed: " + n + (k ? " must be mobile positrons (charge = 1, mass = 1)" : " must be mobile electrons (charge = -1, mass = 1)"));
        }
    }
}

bool qed_split(const Config& c, size_t i) {
    const SpeciesConfig& s = c.species[i];
    if (s.spin.on) return true;
    if (!c.qed.on || c.qed.model == QedConfig::None) return false;
    if (c.qed.pairs() && (s.name == c.qed.pair_e || s.name == c.qed.pair_p)) return true;   // the pairs appear in stage Qed
    for (const std::string& n : c.qed.species) if (n == s.name) return true;
    return false;
}

void qed_summary(FILE* f, const Config& c) {
    for (const SpeciesConfig& s : c.species) {
        if (s.spin.on)
            std::fprintf(f, "  spin of %s: initial (%g %g %g), anomaly %.10g, T-BMT %s\n", s.name.c_str(), s.spin.s[0], s.spin.s[1],
                         s.spin.s[2], s.spin.anomaly, !c.qed.on || c.qed.precession ? "on" : "off");
        if (s.photon) std::fprintf(f, "  species %s: photons (ballistic)\n", s.name.c_str());
    }
    if (!c.qed.on) return;
    const QedConfig& q = c.qed;
    std::string sp;
    for (const std::string& n : q.species) sp += " " + n;
    std::fprintf(f, "  qed: model %s, species%s, %s spin states, spin flip %s, photons %s (above %g MeV), substep_prob %g, hbar omega0 / m c^2 = %.6e\n",
                 q.model == QedConfig::MonteCarlo ? "monte_carlo (LCFA, spin- and polarization-resolved)"
                 : q.model == QedConfig::Deterministic ? "deterministic (P_alpha chi^2 g(chi))" : "none",
                 sp.c_str(), q.pure ? "pure" : "mixed", q.spin_flip ? "on" : "off", q.photons.empty() ? "tallied only" : q.photons.c_str(),
                 q.photon_min * 0.51099895000, q.substep_prob, qed_hw(c.lambda0));
    if (q.pairs())
        std::fprintf(f, "  qed: Breit-Wheeler pairs from %s (LCFA, resolved in the photon polarization and the pair spins, above 2 m c^2) "
                        "into %s (e-) and %s (e+), photon selection %s\n",
                     q.photons.c_str(), q.pair_e.c_str(), q.pair_p.c_str(), q.pair_selection ? "on" : "off");
    if (c.boost.on)
        std::fprintf(f, "  qed in the boosted frame (decision 117): rates covariant (chi invariant); photon_min_energy, the "
                        "energies of qed.txt and the spectra in the lab; the deck spin a lab rest-frame spin; qed.txt rows at boosted times\n");
}

// ---------------------------------------------------------------- the split push (stage Push)
void qed_push(Simulation& s, size_t i) {
    Qed& Q = *s.qed;
    QedSpecies* e = nullptr;
    for (QedSpecies& x : Q.sp) if (size_t(x.index) == i) e = &x;
    Species& sp = s.species[i];
    ensure(e->eb, 9 * sp.size() + 9);
    if (s.cfg.rz) rz_push_store(sp, *s.rz, s.cfg.pusher, s.cfg.order, e->eb.data());
    else push_store(sp, *s.fields, s.dom, s.cfg.pusher, s.cfg.order, e->eb.data());
}

// ---------------------------------------------------------------- the qed operator (stage Qed)
namespace {
struct SpinView { double *s0, *s1, *s2; };
template <bool Count>
void qed_kernel(long n, const QedK K, const QedOut O, const double* EB, double* X, double* Y, double* Z, double* UX, double* UY,
                double* UZ, const double* W, const long long* ID, SpinView SV, double* CHI, double xf, double qm, double a, bool prec,
                bool emit, unsigned long long ks, int* C, const int* OFF, long first) {
    const bool spin = SV.s0 != nullptr;
    par_for(n, [=] HD(long p) {
        if (X[p] >= xf) { if (Count) C[p] = 0; return; }     // entry columns of the moving window: not pushed
        const double* e = EB + 9 * p;
        double u[3] = {UX[p], UY[p], UZ[p]};
        double S[3] = {0, 0, 0};
        if (spin) { S[0] = SV.s0[p]; S[1] = SV.s1[p]; S[2] = SV.s2[p]; }
        if (spin && prec) {
            // u^n: the direction of u^{n-1/2} + u^{n+1/2}, the length the mean of theirs (a rotation in B keeps |u|: the chord
            // average alone is short by (omega_c dt)^2 / 8 and would turn S too fast)
            double um[3] = {e[6] + u[0], e[7] + u[1], e[8] + u[2]};
            const double n0 = sqrt(e[6] * e[6] + e[7] * e[7] + e[8] * e[8]), n1 = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
            const double nm = sqrt(um[0] * um[0] + um[1] * um[1] + um[2] * um[2]), f = nm > 0 ? 0.5 * (n0 + n1) / nm : 0.0;
            um[0] *= f; um[1] *= f; um[2] *= f;
            qed_tbmt(qm, a, e, e + 3, um, K.dt, S);
        }
        int made = 0;
        double chi = 0;
        if (emit) {
            const double x[3] = {X[p], Y[p], Z[p]};
            made = qed_emit<Count>(K, O, e, e + 3, u, S, W[p], x, ion_mix(ks ^ ion_mix((unsigned long long)ID[p])),
                                   (Count || !OFF) ? 0 : first + OFF[p], chi);
        }
        if (Count) { C[p] = made; return; }
        UX[p] = u[0]; UY[p] = u[1]; UZ[p] = u[2];
        if (spin) { SV.s0[p] = S[0]; SV.s1[p] = S[1]; SV.s2[p] = S[2]; }
        if (CHI) CHI[p] = chi;
    });
}

template <bool Count>
void bw_kernel(long n, const BwK K, const BwOut O, const double* EB, const double* X, const double* Y, const double* Z, const double* UX,
               const double* UY, const double* UZ, const double* W, const long long* ID, double* XI1, double* XI2, double* XI3, double* P0,
               double* P1, double* P2, double xf, unsigned long long ks, int* C, const int* OFF, long first) {
    par_for(n, [=] HD(long p) {
        if (X[p] >= xf) { if (Count) C[p] = 0; return; }
        const double* e = EB + 9 * p;
        const double u[3] = {UX[p], UY[p], UZ[p]}, x[3] = {X[p], Y[p], Z[p]};
        double xi[3] = {XI1[p], XI2[p], XI3[p]}, pe[3] = {P0[p], P1[p], P2[p]};
        const int d = qed_pair<Count>(K, O, e, e + 3, u, xi, pe, W[p], x, ion_mix(ks ^ ion_mix((unsigned long long)ID[p])),
                                      Count ? 0 : first + OFF[p]);
        if (Count) { C[p] = d; return; }
        XI1[p] = xi[0]; XI2[p] = xi[1]; XI3[p] = xi[2];
        P0[p] = pe[0]; P1[p] = pe[1]; P2[p] = pe[2];
    });
}

// E, B at the photons (x^n; the photons of this step were made at x^n too): the gather of push_store without a push (9 per photon)
void gather_photons(Simulation& s, Species& ph, double* EB) {
    if (s.cfg.rz) rz_push_store(ph, *s.rz, Pusher::None, s.cfg.order, EB);
    else push_store(ph, *s.fields, s.dom, Pusher::None, s.cfg.order, EB);
}

// Breit-Wheeler step of the photon species: count the decays, make room for the pairs, decay (writing the pairs), then remove
// the decayed photons (stable order of the rest)
void bw_step(Simulation& s, Qed& Q) {
    Species& ph = s.species[size_t(Q.photon)];
    const long n = long(ph.size());
    if (n == 0) return;
    assign_ids(ph, 0xb3eULL);                  // photons without an id (made without pairs, e.g. before a restart)
    ensure(Q.ebph, 9 * size_t(n) + 9);
    gather_photons(s, ph, Q.ebph.data());
    const double *X = ph.ptr<double>(Species::X), *Y = ph.ptr<double>(Species::Y), *Z = ph.ptr<double>(Species::Z);
    const double *UX = ph.ptr<double>(Species::UX), *UY = ph.ptr<double>(Species::UY), *UZ = ph.ptr<double>(Species::UZ);
    const double* W = ph.ptr<double>(Species::W);
    const long long* ID = ph.ptr<long long>(ph.slot("id"));
    const int x0 = ph.slot("xi1"), p0 = ph.slot("pe0");
    double *XI1 = ph.ptr<double>(x0), *XI2 = ph.ptr<double>(x0 + 1), *XI3 = ph.ptr<double>(x0 + 2);
    double *P0 = ph.ptr<double>(p0), *P1 = ph.ptr<double>(p0 + 1), *P2 = ph.ptr<double>(p0 + 2);
    const unsigned long long ks = ion_mix(ion_mix((unsigned long long)s.cfg.seed * 0x9E3779B97F4A7C15ULL + (unsigned long long)s.istep) ^
                                          ion_mix(0xb3e0ULL + (unsigned long long)Q.photon));
    ensure(Q.cnt, size_t(n)); ensure(Q.off, size_t(n));
    int* C = Q.cnt.data();
    int* OFF = Q.off.data();
    BwOut O;
    O.tot = Q.btot.data(); O.hx = Q.bhx.data(); O.he = Q.bhe.data();
    bw_kernel<true>(n, Q.bk, O, Q.ebph.data(), X, Y, Z, UX, UY, UZ, W, ID, XI1, XI2, XI3, P0, P1, P2, s.dom.xfreeze, ks, C, OFF, 0);
    const long total = scan(C, OFF, n, Q.tmp);
    if (total == 0) {                          // no decay: only the selection of xi (and the basis) of the photons
        bw_kernel<false>(n, Q.bk, O, Q.ebph.data(), X, Y, Z, UX, UY, UZ, W, ID, XI1, XI2, XI3, P0, P1, P2, s.dom.xfreeze, ks, C, OFF, 0);
        return;
    }
    Species& se = s.species[size_t(Q.pair_e)];
    Species& sq = s.species[size_t(Q.pair_p)];
    const long fe = long(se.extend(size_t(total))), fp = long(sq.extend(size_t(total)));
    BwOut::L* L[2] = {&O.e, &O.p};
    Species* S2[2] = {&se, &sq};
    for (int c = 0; c < 2; ++c) {
        Species& t = *S2[c];
        BwOut::L& o = *L[c];
        const long f0 = c ? fp : fe;
        o.X = t.ptr<double>(Species::X) + f0; o.Y = t.ptr<double>(Species::Y) + f0; o.Z = t.ptr<double>(Species::Z) + f0;
        o.UX = t.ptr<double>(Species::UX) + f0; o.UY = t.ptr<double>(Species::UY) + f0; o.UZ = t.ptr<double>(Species::UZ) + f0;
        o.W = t.ptr<double>(Species::W) + f0;
        const int ss = t.slot("s0"), si = t.slot("id");
        o.S0 = ss >= 0 ? t.ptr<double>(ss) + f0 : nullptr;
        o.S1 = ss >= 0 ? t.ptr<double>(ss + 1) + f0 : nullptr;
        o.S2 = ss >= 0 ? t.ptr<double>(ss + 2) + f0 : nullptr;
        o.ID = si >= 0 ? t.ptr<long long>(si) + f0 : nullptr;
    }
    bw_kernel<false>(n, Q.bk, O, Q.ebph.data(), X, Y, Z, UX, UY, UZ, W, ID, XI1, XI2, XI3, P0, P1, P2, s.dom.xfreeze, ks, C, OFF, 0);
    // remove the decayed photons: perm = the kept ones in order, then the decayed; then cut
    ensure(Q.keep, size_t(n)); ensure(Q.perm, size_t(n));
    int* KP = Q.keep.data();
    int* KO = Q.perm.data();
    par_for(n, [=] HD(long p) { KP[p] = 1 - C[p]; });
    const long nk = scan(KP, OFF, n, Q.tmp);   // OFF: position among the kept (C keeps the decay flags)
    par_for(n, [=] HD(long p) { KO[C[p] ? nk + (p - OFF[p]) : OFF[p]] = int(p); });
    ph.permute_device(KO);
    ph.resize(size_t(nk));
}

class QedOp : public Operator {
public:
    const char* name() const override { return "qed"; }
    Stage stage() const override { return Stage::Qed; }
    void apply(Simulation& s) override {
        Qed& Q = *s.qed;
        for (size_t k = 0; k < Q.sp.size(); ++k) {
            QedSpecies& e = Q.sp[k];
            Species& sp = s.species[size_t(e.index)];
            const long n = long(sp.size());
            if (n == 0) continue;
            const bool emit = e.emit && Q.c.model != QedConfig::None;
            if (emit) assign_ids(sp, 0x9edULL + (unsigned long long)e.index);
            SpinView SV{nullptr, nullptr, nullptr};
            if (e.spin) SV = {sp.ptr<double>(e.s0), sp.ptr<double>(e.s0 + 1), sp.ptr<double>(e.s0 + 2)};
            double *X = sp.ptr<double>(Species::X), *Y = sp.ptr<double>(Species::Y), *Z = sp.ptr<double>(Species::Z);
            double *UX = sp.ptr<double>(Species::UX), *UY = sp.ptr<double>(Species::UY), *UZ = sp.ptr<double>(Species::UZ);
            const double* W = sp.ptr<double>(Species::W);
            const long long* ID = emit ? sp.ptr<long long>(e.id) : nullptr;
            double* CHI = emit ? sp.ptr<double>(e.chi) : nullptr;
            const unsigned long long ks = ion_mix(ion_mix((unsigned long long)s.cfg.seed * 0x9E3779B97F4A7C15ULL + (unsigned long long)s.istep) ^
                                                  ion_mix(0x9ed0ULL + (unsigned long long)e.index));
            QedK K = Q.K;
            K.q = sp.q > 0 ? 1.0 : -1.0;
            K.spin = e.spin;
            QedOut O;
            O.tot = e.tot.data(); O.hu = e.hu.data(); O.he = e.he.data();
            const bool prec = Q.c.precession || !Q.c.on;
            long first = 0;
            int* C = nullptr;
            int* OFF = nullptr;
            if (emit && K.photons) {
                ensure(Q.cnt, size_t(n)); ensure(Q.off, size_t(n));
                C = Q.cnt.data(); OFF = Q.off.data();
                qed_kernel<true>(n, K, O, e.eb.data(), X, Y, Z, UX, UY, UZ, W, ID, SV, CHI, s.dom.xfreeze, e.qm, e.anomaly, prec, emit, ks, C, OFF, 0);
                const long total = scan(C, OFF, n, Q.tmp);
                Species& ph = s.species[size_t(Q.photon)];
                first = long(ph.extend(size_t(total)));
                O.X = ph.ptr<double>(Species::X); O.Y = ph.ptr<double>(Species::Y); O.Z = ph.ptr<double>(Species::Z);
                O.UX = ph.ptr<double>(Species::UX); O.UY = ph.ptr<double>(Species::UY); O.UZ = ph.ptr<double>(Species::UZ);
                O.W = ph.ptr<double>(Species::W);
                const int x0 = ph.slot("xi1"), p0 = ph.slot("pe0");
                O.XI1 = ph.ptr<double>(x0); O.XI2 = ph.ptr<double>(x0 + 1); O.XI3 = ph.ptr<double>(x0 + 2);
                O.P0 = ph.ptr<double>(p0); O.P1 = ph.ptr<double>(p0 + 1); O.P2 = ph.ptr<double>(p0 + 2);
                if (Q.pair_e >= 0) O.ID = ph.ptr<long long>(ph.slot("id"));
                // the pointers of the emitting species are unchanged (a different species grew)
            } else K.photons = false;
            qed_kernel<false>(n, K, O, e.eb.data(), X, Y, Z, UX, UY, UZ, W, ID, SV, CHI, s.dom.xfreeze, e.qm, e.anomaly, prec, emit, ks,
                              C, OFF, first);
        }
        if (Q.pair_e >= 0) bw_step(s, Q);           // Breit-Wheeler: the photons (also those of this step) decay
        const long every = Q.c.spectrum_every;
        if (every > 0 && (s.istep + 1) % every == 0) {      // after this step (collective)
            char b[64];
            std::snprintf(b, sizeof b, "/qed_spectrum_%06ld.txt", s.istep + 1);
            qed_write_spectra(s, s.cfg.dir + b);
            if (Q.pair_e >= 0) {
                std::snprintf(b, sizeof b, "/qed_pairs_%06ld.txt", s.istep + 1);
                qed_write_pair_spectra(s, s.cfg.dir + b);
            }
        }
    }
    void finish(Simulation& s) override { qed_write_spectra(s, s.cfg.dir + "/qed_spectrum.txt");
        if (s.qed->pair_e >= 0) qed_write_pair_spectra(s, s.cfg.dir + "/qed_pairs.txt");
    }
};

// photons: x <- x + dt u / |u| (stage MoveDeposit, after the photons of this step were created at x^n)
class PhotonMove : public Operator {
public:
    const char* name() const override { return "photon_move"; }
    Stage stage() const override { return Stage::MoveDeposit; }
    void apply(Simulation& s) override {
        Species& ph = s.species[size_t(s.qed->photon)];
        double *X = ph.ptr<double>(Species::X), *Y = ph.ptr<double>(Species::Y), *Z = ph.ptr<double>(Species::Z);
        const double *UX = ph.ptr<double>(Species::UX), *UY = ph.ptr<double>(Species::UY), *UZ = ph.ptr<double>(Species::UZ);
        const double dt = s.dom.dt();
        par_for(long(ph.size()), [=] HD(long p) {
            const double un = sqrt(UX[p] * UX[p] + UY[p] * UY[p] + UZ[p] * UZ[p]);
            if (!(un > 0)) return;
            X[p] += dt * UX[p] / un; Y[p] += dt * UY[p] / un; Z[p] += dt * UZ[p] / un;
        });
    }
};
}  // namespace

// ---------------------------------------------------------------- setup
void qed_register(Simulation& s, size_t i) {
    const SpeciesConfig& sc = s.cfg.species[i];
    Species& sp = s.species[i];
    if (sc.photon) {
        sp.photon = true;
        for (int k = 0; k < 3; ++k) sp.require("xi" + std::to_string(k + 1), Attr::F64, 0.0);
        for (int k = 0; k < 3; ++k) sp.require("pe" + std::to_string(k), Attr::F64, 0.0);
        if (s.cfg.qed.pairs() && sc.name == s.cfg.qed.photons) sp.require("id", Attr::I64, 0.0);   // random numbers of the decay
        return;
    }
    if (!qed_split(s.cfg, i)) return;
    if (sc.spin.on) {
        // boosted frame (decision 117): the deck spin is the rest-frame spin seen from the lab; the code keeps the rest-frame
        // spin reached by the pure boost from the simulation frame: converted with the species' lab momentum (exact for a
        // cold beam; the identity for a momentum along the boost axis or a plasma at rest in the lab)
        double s0[3] = {sc.spin.s[0], sc.spin.s[1], sc.spin.s[2]};
        if (s.cfg.boost.on) qed_spin_boost(sc.u0_lab, s0, s.cfg.boost.beta);
        for (int k = 0; k < 3; ++k) sp.require("s" + std::to_string(k), Attr::F64, s0[k]);
    }
    bool emit = false;
    for (const std::string& n : s.cfg.qed.species) emit = emit || n == sc.name;
    if (emit && s.cfg.qed.model != QedConfig::None) { sp.require("id", Attr::I64, 0.0); sp.require("chi", Attr::F64, 0.0); }
    if (s.cfg.qed.pairs() && (sc.name == s.cfg.qed.pair_e || sc.name == s.cfg.qed.pair_p)) sp.require("id", Attr::I64, 0.0);
}

bool qed_active(const Config& c) {
    for (size_t i = 0; i < c.species.size(); ++i) if (qed_split(c, i)) return true;
    return false;
}

void qed_setup(Simulation& s) {
    const Config& c = s.cfg;
    auto Q = std::make_shared<Qed>();
    Q->c = c.qed;
    const QedTables& T = qed_tables();
    Q->fy.upload(T.fy); Q->fc.upload(T.fc); Q->fb.upload(T.fb);
    QedK& K = Q->K;
    K.T = T.view_host();
    K.T.fy = Q->fy.data(); K.T.fc = Q->fc.data(); K.T.fb = Q->fb.data();   // device copies (a host pointer in a kernel: illegal address without HMM)
    K.hw = qed_hw(c.lambda0);
    K.c0f = QED_ALPHA / (std::sqrt(3.0) * M_PI * K.hw);
    K.dt = s.dom.dt();
    K.pmax = c.qed.substep_prob;
    K.model = c.qed.on ? int(c.qed.model) : 0;
    K.pure = c.qed.pure;
    K.flip = c.qed.spin_flip;
    K.ph_min = c.qed.photon_min;
    K.bins = c.qed.bins;
    K.lu0 = std::log(c.qed.u_min); K.dlu = -K.lu0 / K.bins;
    K.le0 = std::log(c.qed.e_min_mev / 0.51099895000); K.dle = std::log(c.qed.e_max_mev / c.qed.e_min_mev) / K.bins;
    if (c.boost.on) { K.bg = c.boost.gamma; K.bb = c.boost.beta; }   // lab photon energies (decision 117)
    Q->photon = c.qed.photons.empty() ? -1 : find(c, c.qed.photons);
    K.photons = Q->photon >= 0;
    if (c.qed.on && c.qed.pairs() && Q->photon >= 0) {
        Q->pair_e = find(c, c.qed.pair_e);
        Q->pair_p = find(c, c.qed.pair_p);
        BwK& b = Q->bk;
        b.T = K.T; b.hw = K.hw; b.c0f = K.c0f; b.dt = K.dt; b.pmax = K.pmax; b.pure = K.pure; b.sel = c.qed.pair_selection;
        b.spin_e = c.species[size_t(Q->pair_e)].spin.on; b.spin_p = c.species[size_t(Q->pair_p)].spin.on;
        b.bins = K.bins; b.le0 = K.le0; b.dle = K.dle; b.bg = K.bg; b.bb = K.bb;
        Q->btot.resize(8); Q->bhx.resize(4 * size_t(K.bins)); Q->bhe.resize(2 * size_t(K.bins));
    }
    s.split.assign(c.species.size(), 0);
    for (size_t i = 0; i < c.species.size(); ++i) {
        if (!qed_split(c, i)) continue;
        s.split[i] = 1;
        QedSpecies e;
        e.index = int(i);
        e.spin = c.species[i].spin.on;
        for (const std::string& n : c.qed.species) e.emit = e.emit || n == c.species[i].name;
        e.emit = e.emit && K.model != 0;
        e.qm = c.species[i].q / c.species[i].m;
        e.anomaly = c.species[i].spin.anomaly;
        const Species& sp = s.species[i];
        e.s0 = sp.slot("s0"); e.chi = sp.slot("chi"); e.id = sp.slot("id");
        e.tot.resize(8); e.hu.resize(5 * size_t(K.bins)); e.he.resize(5 * size_t(K.bins));
        Q->sp.push_back(std::move(e));
    }
    s.qed = Q;
    s.pipeline.add(std::make_unique<QedOp>());
    s.pipeline.add(make_move_deposit_operator(c.rz));
    if (Q->photon >= 0) s.pipeline.add(std::make_unique<PhotonMove>());
}

// ---------------------------------------------------------------- diagnostics
void qed_diagnostics(Simulation& s, bool root) {
    Qed& Q = *s.qed;
    // per species: sum w, sum w S (3), sum w |S|, sum w chi, max chi; tallies (8); photon species: N, energy
    std::vector<double> v, mx;
    for (QedSpecies& e : Q.sp) {
        Species& sp = s.species[size_t(e.index)];
        const long n = long(sp.size());
        const double* W = sp.ptr<double>(Species::W);
        v.push_back(par_sum(n, [=] HD(long p) { return W[p]; }));
        if (e.spin) {
            for (int k = 0; k < 3; ++k) {
                const double* S = sp.ptr<double>(e.s0 + k);
                v.push_back(par_sum(n, [=] HD(long p) { return W[p] * S[p]; }));
            }
            const double *A = sp.ptr<double>(e.s0), *B = sp.ptr<double>(e.s0 + 1), *Cc = sp.ptr<double>(e.s0 + 2);
            v.push_back(par_sum(n, [=] HD(long p) { return W[p] * sqrt(A[p] * A[p] + B[p] * B[p] + Cc[p] * Cc[p]); }));
        } else for (int k = 0; k < 4; ++k) v.push_back(0);
        if (e.chi >= 0) {
            const double* X = sp.ptr<double>(e.chi);
            v.push_back(par_sum(n, [=] HD(long p) { return W[p] * X[p]; }));
            mx.push_back(par_max_abs(n, [=] HD(long p) { return X[p]; }));
        } else { v.push_back(0); mx.push_back(0); }
        const std::vector<double> t = e.tot.host();
        v.insert(v.end(), t.begin(), t.begin() + 7);
    }
    double nph = 0, eph = 0;
    if (Q.photon >= 0) {
        Species& ph = s.species[size_t(Q.photon)];
        nph = double(ph.size());
        eph = ph.kinetic_energy();
    }
    v.push_back(nph); v.push_back(eph);
    const bool pairs = Q.pair_e >= 0;
    if (pairs) { const std::vector<double> t = Q.btot.host(); v.insert(v.end(), t.begin(), t.end()); }   // 8 pair tallies
    s.comm.allreduce_sum(v.data(), int(v.size()));
    for (double& x : mx) x = s.comm.allreduce_max(x);
    if (!root) return;
    if (!Q.f) {
        bool fresh;
        Q.f = open_log(s.cfg.dir + "/qed.txt", s.restart_step, fresh);
        if (fresh) {
            std::fprintf(Q.f, "# step  t");
            for (QedSpecies& e : Q.sp) {
                const char* n = s.cfg.species[size_t(e.index)].name.c_str();
                std::fprintf(Q.f, "  W_%s <S0>_%s <S1>_%s <S2>_%s <|S|>_%s <chi>_%s chi_max_%s N_rad_%s E_rad_%s <xi1>_%s <xi2>_%s <xi3>_%s "
                                  "P>1_%s E_photons_%s", n, n, n, n, n, n, n, n, n, n, n, n, n, n);
            }
            std::fprintf(Q.f, "  N_photons E_photons_now");
            if (pairs)
                std::fprintf(Q.f, "  N_pairs E_pairs P>1_pairs <S+.e2>_pairs <S+.n>_pairs <S-.e2->_pairs <S-.n>_pairs <xi3>_decayed   (pairs: "
                                  "sum w and sum w eps_gamma of the decayed photons, mean pair spins in their own bases (e2+ = n x e1+, "
                                  "e1+ along E + n x B; e2- = -e2+), mean xi3 of the decayed photons, all since the start)");
            std::fprintf(Q.f, "   (sums of weights; E in m c^2 times weight; spins in the particle axes; "
                              "N_rad, E_rad, <xi>, P>1: all emitted since the start)%s\n",
                         s.cfg.boost.on ? "   (boosted frame, decision 117: steps and times boosted; E_rad, E_photons, E_pairs lab energies; "
                                          "W, <S>, <chi>, N_photons, E_photons_now: the box at the boosted time)" : "");
        }
    }
    std::fprintf(Q.f, "%ld %.6e", s.istep, s.t);
    size_t o = 0;
    for (size_t k = 0; k < Q.sp.size(); ++k, o += 13) {    // 13 sums per species (above)
        const double w = v[o] > 0 ? v[o] : 1, nr = v[o + 6] > 0 ? v[o + 6] : 1;
        std::fprintf(Q.f, " %.10e %.10e %.10e %.10e %.10e %.6e %.6e %.10e %.10e %.8e %.8e %.8e %.0f %.10e", v[o], v[o + 1] / w,
                     v[o + 2] / w, v[o + 3] / w, v[o + 4] / w, v[o + 5] / w, mx[k], v[o + 6], v[o + 7], v[o + 8] / nr, v[o + 9] / nr,
                     v[o + 10] / nr, v[o + 11], v[o + 12]);
    }
    std::fprintf(Q.f, " %.0f %.10e", v[o], v[o + 1]);
    if (pairs) {
        const double* t = &v[o + 2];
        const double np = t[0] > 0 ? t[0] : 1;
        std::fprintf(Q.f, " %.10e %.10e %.0f %.8e %.8e %.8e %.8e %.8e", t[0], t[1], t[2], t[3] / np, t[4] / np, t[5] / np, t[6] / np, t[7] / np);
    }
    std::fprintf(Q.f, "\n");
    std::fflush(Q.f);
}

void qed_totals(Simulation& s, double out[2]) {
    out[0] = out[1] = 0;
    if (s.qed)
        for (QedSpecies& e : s.qed->sp) {
            const std::vector<double> t = e.tot.host();
            out[0] += t[1]; out[1] += t[6];
        }
    s.comm.allreduce_sum(out, 2);
}

void qed_pair_totals(Simulation& s, double out[2]) {
    out[0] = out[1] = 0;
    if (s.qed && s.qed->pair_e >= 0) {
        const std::vector<double> t = s.qed->btot.host();
        out[0] = t[0]; out[1] = t[1];
    }
    s.comm.allreduce_sum(out, 2);
}

double qed_events_over_one(Simulation& s) {
    double n = 0;
    if (s.qed) {
        for (QedSpecies& e : s.qed->sp) n += e.tot.host()[5];          // the P>1 column of each species
        if (s.qed->pair_e >= 0) n += s.qed->btot.host()[2];            // P>1_pairs
    }
    s.comm.allreduce_sum(&n, 1);
    return n;
}

void qed_write_pair_spectra(Simulation& s, const std::string& path) {
    if (!s.qed || s.qed->pair_e < 0) return;
    Qed& Q = *s.qed;
    const int nb = Q.bk.bins;
    std::vector<double> hx = Q.bhx.host(), he = Q.bhe.host();
    s.comm.allreduce_sum(hx.data(), int(hx.size()));
    s.comm.allreduce_sum(he.data(), int(he.size()));
    if (s.comm.rank != 0) return;
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "# Breit-Wheeler pairs since the start, step %ld, t = %.6e (1/omega0)\n", s.istep, s.t);
    std::fprintf(f, "# x_lo x_hi (x = eps_+ / eps_gamma)  dN (sum w)  <xi3> of the photons  <S+.e2> <S+.n>  |  eps_lo eps_hi (MeV)  dN dE "
                    "(sum w, sum w eps of the electrons and positrons, m c^2)\n");
    for (int b = 0; b < nb; ++b) {
        const double* a = &hx[4 * size_t(b)];
        const double* c = &he[2 * size_t(b)];
        const double na = a[0] > 0 ? a[0] : 1;
        std::fprintf(f, "%.6e %.6e %.10e %.6e %.6e %.6e  %.6e %.6e %.10e %.10e\n", double(b) / nb, double(b + 1) / nb, a[0], a[1] / na,
                     a[2] / na, a[3] / na, 0.51099895000 * std::exp(Q.bk.le0 + b * Q.bk.dle),
                     0.51099895000 * std::exp(Q.bk.le0 + (b + 1) * Q.bk.dle), c[0], c[1]);
    }
    std::fclose(f);
}

void qed_write_spectra(Simulation& s, const std::string& path) {
    if (!s.qed) return;
    Qed& Q = *s.qed;
    const int nb = Q.K.bins;
    for (size_t k = 0; k < Q.sp.size(); ++k) {
        QedSpecies& e = Q.sp[k];
        if (!e.emit) continue;
        std::vector<double> hu = e.hu.host(), he = e.he.host();
        s.comm.allreduce_sum(hu.data(), int(hu.size()));
        s.comm.allreduce_sum(he.data(), int(he.size()));
        if (s.comm.rank != 0) continue;
        const std::string nm = s.cfg.species[size_t(e.index)].name;
        const std::string p = Q.sp.size() > 1 ? path.substr(0, path.size() - 4) + "_" + nm + ".txt" : path;
        FILE* f = std::fopen(p.c_str(), "w");
        if (!f) continue;
        std::fprintf(f, "# photons emitted by %s since the start, step %ld, t = %.6e (1/omega0)\n", nm.c_str(), s.istep, s.t);
        std::fprintf(f, "# u_lo u_hi  dN (sum w)  dE (sum w eps, m c^2)  <xi1> <xi2> <xi3>  |  eps_lo eps_hi (MeV)  dN  dE  <xi1> <xi2> <xi3>\n");
        for (int b = 0; b < nb; ++b) {
            const double* a = &hu[5 * size_t(b)];
            const double* c = &he[5 * size_t(b)];
            const double na = a[0] > 0 ? a[0] : 1, nc = c[0] > 0 ? c[0] : 1;
            std::fprintf(f, "%.6e %.6e %.10e %.10e %.6e %.6e %.6e  %.6e %.6e %.10e %.10e %.6e %.6e %.6e\n",
                         std::exp(Q.K.lu0 + b * Q.K.dlu), std::exp(Q.K.lu0 + (b + 1) * Q.K.dlu), a[0], a[1], a[2] / na, a[3] / na, a[4] / na,
                         0.51099895000 * std::exp(Q.K.le0 + b * Q.K.dle), 0.51099895000 * std::exp(Q.K.le0 + (b + 1) * Q.K.dle),
                         c[0], c[1], c[2] / nc, c[3] / nc, c[4] / nc);
        }
        std::fclose(f);
    }
}
