// Field ionization (ionization.hpp, docs/specs/field_ionization.md, decision 86).
#include "ionization.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "atomic.hpp"
#include "fsutil.hpp"
#include "geometry.hpp"
#include "sim.hpp"
#if RIP_GPU
#include <cub/cub.cuh>
#endif

namespace {
constexpr double HARTREE_EV = 27.211386245988;       // atomic unit of energy [eV]
constexpr double E_ATOMIC = 5.14220674763e11;        // atomic unit of field [V/m]
constexpr double OMEGA_ATOMIC = 4.1341373335e16;     // atomic unit of frequency [1/s]
constexpr double MC2_EV = 0.51099895000e6;           // m_e c^2 [eV]
constexpr double C_SI = 299792458.0;

double factorial(int n) { double f = 1; for (int i = 2; i <= n; ++i) f *= i; return f; }

// ln of the prefactor C^2 f(l, m) I [atomic units], the power 2 n* - |m| - 1, F0 = (2 I)^(3/2)
void adk_coefficients(double ip_eV, int zafter, int l, int m, double& lnpre, double& power, double& F0) {
    const double I = ip_eV / HARTREE_EV, ns = zafter / std::sqrt(2 * I);
    const int mm = std::min(std::abs(m), l);
    const double lnC2 = 2 * ns * std::log(2.0) - std::log(ns) - std::lgamma(2 * ns);   // l* = n* - 1: Gamma(n* - l*) = 1
    const double f = (2 * l + 1) * factorial(l + mm) / (std::pow(2.0, mm) * factorial(mm) * factorial(l - mm));
    lnpre = lnC2 + std::log(f) + std::log(I);
    power = 2 * ns - mm - 1;
    F0 = std::pow(2 * I, 1.5);
}
}  // namespace

double ion_rate_si(double ip_eV, int zafter, int l, int m, bool tong_lin, double alpha, double E_Vm) {
    if (!(E_Vm > 0)) return 0;
    double lnpre, p, F0;
    adk_coefficients(ip_eV, zafter, l, m, lnpre, p, F0);
    const double F = E_Vm / E_ATOMIC, I = ip_eV / HARTREE_EV;
    double ln = lnpre + p * std::log(2 * F0 / F) - 2 * F0 / (3 * F);
    if (tong_lin) ln -= alpha * zafter * zafter * F / (I * F0);
    return OMEGA_ATOMIC * std::exp(ln);
}

IonLevel ion_level(double ip_eV, int zafter, int l, int m, bool tong_lin, double alpha, double lambda0, double dt) {
    double lnpre, p, F0;
    adk_coefficients(ip_eV, zafter, l, m, lnpre, p, F0);
    const double omega0 = 2 * M_PI * C_SI / lambda0, E0 = 2 * M_PI * MC2_EV / lambda0;   // E0 = m c omega0 / e [V/m]
    const double s = E0 / E_ATOMIC, I = ip_eV / HARTREE_EV;                               // F = s E
    IonLevel L{};
    L.power = p;
    L.logpre = lnpre + std::log(OMEGA_ATOMIC / omega0) + p * std::log(2 * F0 / s);
    L.expo = -2 * F0 / (3 * s);
    L.tl = tong_lin ? alpha * zafter * zafter * s / (I * F0) : 0.0;
    L.ip = ip_eV / MC2_EV;
    // the maximum of -p ln E + expo / E - tl E: tl E^2 + p E + expo = 0
    if (L.tl > 0) L.epeak = (-p + std::sqrt(p * p - 4 * L.tl * L.expo)) / (2 * L.tl);
    else L.epeak = p > 0 ? -L.expo / p : 1e300;
    // emin: w(emin) dt = 1e-18 (w increases below epeak)
    L.emin = 0;
    auto raw = [&](double E) { const double e = std::min(E, L.epeak); return std::exp(L.logpre - p * std::log(e) + L.expo / e - L.tl * e); };
    double lo = 1e-12, hi = std::min(L.epeak, 1e8);
    if (raw(hi) * dt < 1e-18) L.emin = hi;
    else if (raw(lo) * dt >= 1e-18) L.emin = 0;
    else {
        for (int it = 0; it < 200; ++it) {
            const double mid = std::sqrt(lo * hi);
            (raw(mid) * dt < 1e-18 ? lo : hi) = mid;
        }
        L.emin = lo;
    }
    return L;
}

// ---------------------------------------------------------------- the module
struct IonSpecies {
    int ion = -1, el = -1;             // species indices
    int z0 = 0, zmax = 0;
    bool current = true;
    double qe = -1;                    // charge of the electron species
    Buf<IonLevel> lev;
    std::vector<IonLevel> hlev;
    long events = 0;                   // local, cumulative
};

struct Ionization {
    std::vector<IonSpecies> sp;
    Buf<double> ef;                    // E at the ions (3 per ion)
    Buf<double> bf;                    // B at the ions (3 per ion; boosted frame only)
    bool rest = false;                 // boosted frame: the rate in the ion's rest frame (decision 90)
    Buf<int> cnt, off;
    Buf<unsigned char> tmp;
    // the displacements of this step's electrons: start, end (global positions), q_e w
    Buf<double> j0, j1, jqw;
    long nj = 0;
    Buf<double> eip, ework;            // per new electron: I_k w, q_e w d . E
    double e_ion = 0, e_cur = 0;       // local, cumulative: sum I_k w and the work of the displacement currents
    long capped = 0;                   // displacements shortened to half a cell (local)
    double dmax = 0;                   // that length
    FILE* f = nullptr;
    ~Ionization() { if (f) std::fclose(f); }
};

namespace {
template <class T> void ensure(Buf<T>& b, size_t n) { if (b.size() < n) b.resize(std::max(n, b.size() + b.size() / 2)); }
template <class T> void ensure_keep(Buf<T>& b, size_t n) { if (b.size() < n) b.grow_keep(std::max(n, b.size() + b.size() / 2)); }

// O[0 .. n) = exclusive sum of C; returns the total
long exclusive_scan(const int* C, int* O, long n, Buf<unsigned char>& tmp) {
#if RIP_GPU
    size_t bytes = 0;
    gpu_check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, C, O, int(n)), "cub scan (size)");
    ensure(tmp, bytes);
    gpu_check(cub::DeviceScan::ExclusiveSum(tmp.data(), bytes, C, O, int(n)), "cub scan");
    int last[2] = {0, 0};
    gpu_check(cudaMemcpy(&last[0], O + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "scan last");
    gpu_check(cudaMemcpy(&last[1], C + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "scan last");
    return long(last[0]) + last[1];
#else
    (void)tmp;
    long t = 0;
    for (long q = 0; q < n; ++q) { O[q] = int(t); t += C[q]; }
    return t;
#endif
}

// E^n at the particles of sp (3 per particle), and B if Bo is given; 0 in the entry columns of the moving window (not pushed
// yet either)
template <class G, class St, int O> void cart_field_at(Species& sp, Fields& f, const Domain& d, double* E, double* Bo) {
    const typename G::FieldViews F = G::views(f);
    const double ih0 = 1.0 / d.h[0], ih1 = 1.0 / d.h[1], ih2 = 1.0 / d.h[2], lo0 = d.lo(0), lo1 = d.lo(1), lo2 = d.lo(2), xf = d.xfreeze;
    const double *X = sp.ptr<double>(Species::X), *Y = sp.ptr<double>(Species::Y), *Z = sp.ptr<double>(Species::Z);
    par_for(long(sp.size()), [=] HD(long p) {
        double e[3] = {0, 0, 0}, b[3] = {0, 0, 0};
        if (X[p] < xf) {
            const double ih[3] = {ih0, ih1, ih2}, r[3] = {X[p] - lo0, Y[p] - lo1, Z[p] - lo2};
            G::template gather<St, O>(F, r, ih, e, b);
        }
        E[3 * p] = e[0]; E[3 * p + 1] = e[1]; E[3 * p + 2] = e[2];
        if (Bo) { Bo[3 * p] = b[0]; Bo[3 * p + 1] = b[1]; Bo[3 * p + 2] = b[2]; }
    });
}

// the Esirkepov current of the moves j0 -> j1 (global positions), charge q w each
template <class G, class St, int O> void cart_jumps(Fields& f, const Domain& d, const double* J0, const double* J1, const double* QW, long n) {
    const typename G::FieldViews F = G::views(f);
    const double ih0 = 1.0 / d.h[0], ih1 = 1.0 / d.h[1], ih2 = 1.0 / d.h[2], lo0 = d.lo(0), lo1 = d.lo(1), lo2 = d.lo(2), dt = d.dt();
    par_for(n, [=] HD(long q) {
        const double ih[3] = {ih0, ih1, ih2};
        const double r0[3] = {J0[3 * q] - lo0, J0[3 * q + 1] - lo1, J0[3 * q + 2] - lo2};
        const double r1[3] = {J1[3 * q] - lo0, J1[3 * q + 1] - lo1, J1[3 * q + 2] - lo2};
        const double qw = QW[q] / dt;
        const double coef[3] = {qw * ih1 * ih2, qw * ih0 * ih2, qw * ih0 * ih1};
        G::template deposit<St, O, true>(F, r0, r1, ih, coef);
    });
}

void field_at(Simulation& s, Species& sp, double* E, double* B) {
    if (s.cfg.rz) { rz_field_at(sp, *s.rz, E, s.cfg.order, B); return; }
    with_stagger(s.dom, s.cfg.order, [&](auto geo, auto st, auto o) {
        cart_field_at<decltype(geo), decltype(st), decltype(o)::value>(sp, *s.fields, s.dom, E, B);
    });
}

// ids of the particles that have none yet: a hash of (species, position) (the position when first seen: the load position).
// salt != 0 (the boosted frame: salt = step + 1) is mixed in too: the plasma enters through the injection plane in every step,
// and the step of its first ionization pass (the step after its injection) is the same for any decomposition.
void assign_ids(Species& sp, int index, unsigned long long salt) {
    const int ids = sp.slot("id");
    long long* ID = sp.ptr<long long>(ids);
    const double *X = sp.ptr<double>(Species::X), *Y = sp.ptr<double>(Species::Y), *Z = sp.ptr<double>(Species::Z);
    const unsigned long long base = salt ? ion_mix(ion_mix(0x5eedULL + (unsigned long long)index) ^ ion_mix(0xb005ULL + salt))
                                         : ion_mix(0x5eedULL + (unsigned long long)index);
    par_for(long(sp.size()), [=] HD(long p) {
        if (ID[p] != 0) return;
        unsigned long long h = base, b;
        double v = X[p]; memcpy(&b, &v, 8); h = ion_mix(h ^ b);
        v = Y[p]; memcpy(&b, &v, 8); h = ion_mix(h ^ b);
        v = Z[p]; memcpy(&b, &v, 8); h = ion_mix(h ^ b);
        ID[p] = (long long)(h == 0 ? 1 : h);
    });
}

void ionize_species(Simulation& s, Ionization& I, IonSpecies& e) {
    Species& ion = s.species[size_t(e.ion)];
    Species& el = s.species[size_t(e.el)];
    const long n = long(ion.size());
    if (n == 0) return;
    const bool rest = I.rest;
    assign_ids(ion, e.ion, rest ? (unsigned long long)s.istep + 1 : 0ULL);
    ensure(I.ef, size_t(3 * n)); ensure(I.cnt, size_t(n)); ensure(I.off, size_t(n));
    if (rest) ensure(I.bf, size_t(3 * n));
    field_at(s, ion, I.ef.data(), rest ? I.bf.data() : nullptr);
    int* Zs = ion.ptr<int>(ion.slot("charge_state"));
    const long long* ID = ion.ptr<long long>(ion.slot("id"));
    const IonLevel* L = e.lev.data();
    const int zmax = e.zmax;
    const double dt = s.dom.dt();
    const unsigned long long ks = ion_mix(ion_mix((unsigned long long)s.cfg.seed * 0x9E3779B97F4A7C15ULL + (unsigned long long)s.istep) ^
                                          ion_mix(0x10dULL + (unsigned long long)e.ion));
    const double* E = I.ef.data();
    const double* Bf = rest ? I.bf.data() : nullptr;
    const double *UX = ion.ptr<double>(Species::UX), *UY = ion.ptr<double>(Species::UY), *UZ = ion.ptr<double>(Species::UZ);
    int* C = I.cnt.data();
    par_for(n, [=] HD(long p) {
        if (rest) {          // boosted frame: |E| in the ion's rest frame over its proper time dt / gamma
            double g;
            const double a = ion_rest_field(E + 3 * p, Bf + 3 * p, UX[p], UY[p], UZ[p], g);
            C[p] = ion_sample(L, Zs[p], zmax, a, dt / g, ion_mix(ks ^ ion_mix((unsigned long long)ID[p])));
            return;
        }
        const double a = sqrt(E[3 * p] * E[3 * p] + E[3 * p + 1] * E[3 * p + 1] + E[3 * p + 2] * E[3 * p + 2]);
        C[p] = ion_sample(L, Zs[p], zmax, a, dt, ion_mix(ks ^ ion_mix((unsigned long long)ID[p])));
    });
    const double tot = par_sum(n, [=] HD(long p) { return double(C[p]); });
    if (!(tot > 0)) return;
    int* O = I.off.data();
    const long total = exclusive_scan(C, O, n, I.tmp);
    // the new electrons, in the order of their ions
    const long first = long(el.extend(size_t(total)));
    double *EX = el.ptr<double>(Species::X), *EY = el.ptr<double>(Species::Y), *EZ = el.ptr<double>(Species::Z);
    double *EUX = el.ptr<double>(Species::UX), *EUY = el.ptr<double>(Species::UY), *EUZ = el.ptr<double>(Species::UZ);
    double* EW = el.ptr<double>(Species::W);
    const double *X = ion.ptr<double>(Species::X), *Y = ion.ptr<double>(Species::Y), *Z = ion.ptr<double>(Species::Z);
    const double* W = ion.ptr<double>(Species::W);
    // with collisions the new electrons get their id at birth from the ion's id and the level (collisions.hpp: the shuffle in a
    // cell must not depend on positions that differ at round-off between decompositions)
    long long* EID = s.collisions && el.slot("id") >= 0 ? el.ptr<long long>(el.slot("id")) : nullptr;
    const long jb = I.nj;
    if (e.current) {
        ensure_keep(I.j0, size_t(3 * (jb + total))); ensure_keep(I.j1, size_t(3 * (jb + total))); ensure_keep(I.jqw, size_t(jb + total));
    }
    ensure(I.eip, size_t(total)); ensure(I.ework, size_t(total));
    double *J0 = e.current ? I.j0.data() : nullptr, *J1 = e.current ? I.j1.data() : nullptr, *JQ = e.current ? I.jqw.data() : nullptr;
    double *EIP = I.eip.data(), *EWK = I.ework.data();
    const double qe = e.qe, dmax = I.dmax;
    const bool cur = e.current;
    Buf<int> capb(1);
    int* CAP = capb.data();
    par_for(n, [=] HD(long p) {
        const int c = C[p];
        if (c == 0) return;
        const double ex = E[3 * p], ey = E[3 * p + 1], ez = E[3 * p + 2], e2 = ex * ex + ey * ey + ez * ez;
        const long o = O[p];
        // boosted frame: the field pays gamma_ion I_k (the rest-frame energy I_k of an ion moving at gamma_ion)
        const double gi = rest ? sqrt(1.0 + UX[p] * UX[p] + UY[p] * UY[p] + UZ[p] * UZ[p]) : 1.0;
        for (int j = 0; j < c; ++j) {
            const IonLevel& lv = L[Zs[p] + j];
            const long q = first + o + j;
            const double ip = rest ? gi * lv.ip : lv.ip;
            double sc = 0;                                  // d = sc E
            if (cur && e2 > 0) {
                sc = ip / (qe * e2);
                const double dl = fabs(sc) * sqrt(e2);
                if (dl > dmax) { sc *= dmax / dl; atomic_inc(CAP); }
            }
            EX[q] = X[p] + sc * ex; EY[q] = Y[p] + sc * ey; EZ[q] = Z[p] + sc * ez;
            EUX[q] = UX[p]; EUY[q] = UY[p]; EUZ[q] = UZ[p];
            EW[q] = W[p];
            if (EID) EID[q] = (long long)(ion_mix((unsigned long long)ID[p] ^ ion_mix(0xe1ec0000ULL + (unsigned long long)(Zs[p] + j))) | 1ULL);
            EIP[o + j] = ip * W[p];
            if (cur) {
                const long b = jb + o + j;
                J0[3 * b] = X[p]; J0[3 * b + 1] = Y[p]; J0[3 * b + 2] = Z[p];
                J1[3 * b] = EX[q]; J1[3 * b + 1] = EY[q]; J1[3 * b + 2] = EZ[q];
                JQ[b] = qe * W[p];
                // the work of the displacement as stored (rounded positions)
                EWK[o + j] = qe * W[p] * ((EX[q] - X[p]) * ex + (EY[q] - Y[p]) * ey + (EZ[q] - Z[p]) * ez);
            } else EWK[o + j] = 0;
        }
        Zs[p] += c;
    });
    if (cur) I.nj += total;
    I.e_ion += par_sum(total, [=] HD(long q) { return EIP[q]; });
    I.e_cur += par_sum(total, [=] HD(long q) { return EWK[q]; });
    int ncap = 0;
    capb.to_host(&ncap, 0, 1);
    I.capped += ncap;
    e.events += total;
}

class IonizeOp : public Operator {
public:
    const char* name() const override { return "ionize"; }
    Stage stage() const override { return Stage::Ionize; }
    void setup(Simulation& s) override {
        auto I = std::make_shared<Ionization>();
        const Config& c = s.cfg;
        for (size_t i = 0; i < c.species.size(); ++i) {
            const IonizationConfig& ic = c.species[i].ion;
            if (!ic.on) continue;
            IonSpecies e;
            e.ion = int(i); e.z0 = ic.z0; e.zmax = ic.zmax; e.current = ic.current;
            for (size_t j = 0; j < c.species.size(); ++j)
                if (c.species[j].name == ic.electrons) e.el = int(j);
            if (e.el < 0) throw std::runtime_error("species " + c.species[i].name + ": ionization_electrons = " + ic.electrons + ": no such species");
            if (e.el == e.ion) throw std::runtime_error("species " + c.species[i].name + ": ionization_electrons must be another species");
            e.qe = c.species[size_t(e.el)].q;
            if (!(e.qe < 0)) throw std::runtime_error("species " + c.species[i].name + ": ionization_electrons needs a negative charge");
            if (c.species[size_t(e.el)].ion.on) throw std::runtime_error("species " + ic.electrons + ": the electron species cannot be ionizable");
            for (int k = 0; k < ic.zmax; ++k)
                e.hlev.push_back(k < ic.z0 ? IonLevel{0, 0, 0, 0, 1e300, 1e300, 0}
                                           : ion_level(ic.ip_eV[size_t(k)], k + 1, ic.l[size_t(k)], ic.m, ic.tong_lin, ic.tl_alpha, c.lambda0, s.dom.dt()));
            e.lev.upload(e.hlev);
            s.species[i].charge_max = ic.zmax;
            I->sp.push_back(std::move(e));
        }
        const RadialGrid* g = c.rz ? &s.rz->grid() : nullptr;
        I->rest = c.boost.on;
        I->dmax = 0.5 * (c.rz ? std::min(s.dom.h[0], g->dr) : std::min(std::min(s.dom.h[0], s.dom.h[1]), c.flat ? 1e300 : s.dom.h[2]));
        s.ionization = I;
    }
    void apply(Simulation& s) override {
        Ionization& I = *s.ionization;
        for (IonSpecies& e : I.sp) ionize_species(s, I, e);
    }
};
}  // namespace

std::unique_ptr<Operator> make_ionization_operator() { return std::make_unique<IonizeOp>(); }

void ionization_register(Species& sp, const IonizationConfig& ic) {
    sp.zslot = sp.require("charge_state", Attr::I32, ic.z0);
    sp.require("id", Attr::I64, 0);
    sp.charge_max = ic.zmax;
}

void ionization_currents(Simulation& s) {
    if (!s.ionization || s.ionization->nj == 0) return;
    Ionization& I = *s.ionization;
    const long n = I.nj;
    I.nj = 0;
    if (s.cfg.rz) { rz_jumps(*s.rz, I.j0.data(), I.j1.data(), I.jqw.data(), n, s.cfg.order); return; }
    const double *J0 = I.j0.data(), *J1 = I.j1.data(), *JQ = I.jqw.data();
    with_stagger(s.dom, s.cfg.order, [&](auto geo, auto st, auto o) {
        cart_jumps<decltype(geo), decltype(st), decltype(o)::value>(*s.fields, s.dom, J0, J1, JQ, n);
    });
}

void ionization_diagnostics(Simulation& s, bool root, double wtot) {
    if (!s.ionization) return;
    Ionization& I = *s.ionization;
    std::vector<double> v = {I.e_ion, I.e_cur, double(I.capped)};
    for (IonSpecies& e : I.sp) {
        Species& ion = s.species[size_t(e.ion)];
        const int* Zs = ion.ptr<int>(ion.slot("charge_state"));
        const double* W = ion.ptr<double>(Species::W);
        v.push_back(double(e.events));
        for (int k = 0; k <= e.zmax; ++k)
            v.push_back(par_sum(long(ion.size()), [=] HD(long p) { return Zs[p] == k ? W[p] : 0.0; }));
    }
    s.comm.allreduce_sum(v.data(), int(v.size()));
    if (!root) return;
    if (!I.f) {
        bool fresh;
        I.f = open_log(s.cfg.dir + "/ionization.txt", s.restart_step, fresh);
        if (fresh) {
            std::fprintf(I.f, "# step  t  E_ion (sum I_k w)  E_cur (work of the ionization current)  W_field + W_kin + E_cur  capped");
            for (IonSpecies& e : I.sp) {
                const std::string& nm = s.cfg.species[size_t(e.ion)].name;
                std::fprintf(I.f, "  events_%s", nm.c_str());
                for (int k = 0; k <= e.zmax; ++k) std::fprintf(I.f, "  %s_%d", nm.c_str(), k);
            }
            std::fprintf(I.f, "   (charge states: sums of weights)\n");
        }
    }
    std::fprintf(I.f, "%ld %.6e %.12e %.12e %.12e %.0f", s.istep, s.t, v[0], v[1], wtot + v[1], v[2]);
    for (size_t q = 3; q < v.size(); ++q) std::fprintf(I.f, " %.10e", v[q]);
    std::fprintf(I.f, "\n");
    std::fflush(I.f);
}

void ionization_summary(FILE* f, const Config& c) {
    bool any = false;
    for (const SpeciesConfig& sc : c.species) {
        const IonizationConfig& ic = sc.ion;
        if (!ic.on) continue;
        any = true;
        std::fprintf(f, "  ionization of %s (%s): charge states %d .. %d, ADK (l* = n* - 1, |m| = %d)%s, electrons into %s, "
                        "ionization current %s\n", sc.name.c_str(), ic.element.empty() ? "potentials of the deck" : ic.element.c_str(),
                     ic.z0, ic.zmax, ic.m, ic.tong_lin ? (", Tong-Lin alpha = " + std::to_string(ic.tl_alpha)).c_str() : "",
                     ic.electrons.c_str(), ic.current ? "on" : "off");
        if (c.boost.on)
            std::fprintf(f, "    boosted frame: the rate from |E| in the ion's rest frame over its proper time dt / gamma_ion; the "
                            "ionization current takes gamma_ion I_k (E_ion in this frame)\n");
    }
    if (any) std::fprintf(f, "  ionization energies: %s\n", atomic_data_source());
}

IonizationTotals ionization_totals(Simulation& s) {
    IonizationTotals t;
    if (!s.ionization) return t;
    Ionization& I = *s.ionization;
    double v[4] = {I.e_ion, I.e_cur, 0, double(I.capped)};
    for (const IonSpecies& e : I.sp) v[2] += double(e.events);
    s.comm.allreduce_sum(v, 4);
    t.e_ion = v[0]; t.e_cur = v[1]; t.events = long(v[2]); t.capped = long(v[3]);
    return t;
}
