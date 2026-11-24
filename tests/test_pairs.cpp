// Breit-Wheeler pairs (decision 96, docs/specs/qed.md, src/qed.hpp qed_pair). References: the Bessel functions by their
// integral representation (qed_bessel_exact, checked against std::cyl_bessel_k in test_qed Q1a) and quadrature here, not the
// code's tables; Erber's asymptotics in closed form.
// P1 tables T0, T3 vs quadrature; Erber chi << 1 (3 sqrt3 / (16 sqrt2) chi e^(-8/(3 chi))) and chi >> 1 (0.3796 chi^(2/3));
//    the ratio of the rates of the two linear polarizations (-> 2 and 3/2); |S_+-| <= 1 for all x, chi, xi.
// P2 pair spectrum dN/dx (chi = 0.3, 1, 10; chi^2 per bin); energy exact and the momentum deficit per event.
// P3 rate of xi3 = +1 vs -1 photons; survival and the selection xi3(t) = tanh(Cg T3 t) of unpolarized photons (mixed, pure).
// P4 pair spins per x bin (unpolarized, linear along e1 / e2 / 45 degrees, circular; mixed and pure); P4b their orientation:
//    positron parallel, electron antiparallel to the rest-frame magnetic field (Seipt & King 2020).
// P5 energy balance of a cascade (electrons -> photons -> pairs -> photons, Cart3D and RZ); charge; P6 particle order; P7 decks.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>
#include "common.hpp"
#include "deck.hpp"
#include "qed.hpp"
#include "sim.hpp"

namespace {
const double LAM = 0.8e-6;
// ---------------------------------------------------------------- references
struct Bes { double k13, k23, ik13; };
Bes bes(double rho) { Bes b; qed_bessel_exact(rho, b.k13, b.k23, b.ik13); return b; }
double rho_of(double x, double chi) { return 2 / (3 * chi * x * (1 - x)); }
double B0_ref(double x, double chi) { const Bes b = bes(rho_of(x, chi)); return (x * x + (1 - x) * (1 - x)) / (x * (1 - x)) * b.k23 + b.ik13; }
// int_a^b f dx (Simpson in x; the integrands are smooth inside (0, 1))
double simpson(const std::function<double(double)>& f, double a, double b, int n) {
    const double h = (b - a) / n;
    double s = 0;
    for (int i = 0; i <= n; ++i) s += (i == 0 || i == n ? 1 : (i % 2 ? 4 : 2)) * f(a + i * h);
    return s * h / 3;
}
// int_0^1 f dx for a symmetric f: twice the half in ln x (the end x -> 0: rho -> inf)
double half_ln(const std::function<double(double)>& f, double chi, int n = 4000) {
    const double lo = std::log(std::max(1e-30, 2 / (3 * 800 * chi))), hi = std::log(0.5), h = (hi - lo) / n;
    double s = 0;
    for (int i = 0; i <= n; ++i) { const double x = std::exp(lo + i * h); s += (i == 0 || i == n ? 1 : (i % 2 ? 4 : 2)) * f(x) * x; }
    return 2 * s * h / 3;
}
double T0_ref(double chi) { return half_ln([&](double x) { return B0_ref(x, chi); }, chi); }
double T3_ref(double chi) { return half_ln([&](double x) { return bes(rho_of(x, chi)).k23; }, chi); }

// ---------------------------------------------------------------- host Monte-Carlo driver of qed_pair
// photon along x with energy om, B along z: n = x, e1 = n x B direction = -y (the force on a positron), e2 = n x e1 = -z
struct Lep { double X, Y, Z, UX, UY, UZ, W, S0, S1, S2; long long ID; };
struct Event { bool dec; double xi[3]; Lep e, p; };
struct BwMC {
    BwK K;
    std::vector<double> tot, hx, he;
    double om, chi, B, Cg;
    BwMC(double chi_, double om_, int bins, bool pure, bool sel = true) : om(om_), chi(chi_) {
        const double hw = qed_hw(LAM);
        K.T = qed_tables().view_host();
        K.hw = hw; K.c0f = QED_ALPHA / (std::sqrt(3.0) * M_PI * hw); K.pmax = 0.1; K.pure = pure; K.sel = sel;
        K.spin_e = K.spin_p = true; K.bins = bins; K.le0 = std::log(1e-3); K.dle = std::log(1e8) / bins;
        B = chi / (om * hw);
        Cg = K.c0f / om;
        tot.assign(8, 0); hx.assign(4 * size_t(bins), 0); he.assign(2 * size_t(bins), 0);
    }
    // one step of one photon with the Stokes vector xi (in the basis e1 = -y), kept in ev.xi
    Event trial(const double xi[3], unsigned long long key) {
        Event ev;
        BwOut O;
        O.tot = tot.data(); O.hx = hx.data(); O.he = he.data();
        auto set = [](BwOut::L& o, Lep& l) {
            o.X = &l.X; o.Y = &l.Y; o.Z = &l.Z; o.UX = &l.UX; o.UY = &l.UY; o.UZ = &l.UZ; o.W = &l.W; o.S0 = &l.S0; o.S1 = &l.S1; o.S2 = &l.S2;
            o.ID = &l.ID;
        };
        set(O.e, ev.e); set(O.p, ev.p);
        const double E[3] = {0, 0, 0}, Bv[3] = {0, 0, B}, u[3] = {om, 0, 0}, x[3] = {0, 0, 0};
        double pe[3] = {0, -1, 0};
        for (int k = 0; k < 3; ++k) ev.xi[k] = xi[k];
        ev.dec = qed_pair<false>(K, O, E, Bv, u, ev.xi, pe, 1.0, x, key, 0) != 0;
        return ev;
    }
};
unsigned long long key_of(long t, unsigned long long salt) { return ion_mix(ion_mix(salt) ^ ion_mix((unsigned long long)t)); }

// ---------------------------------------------------------------- P1 tables, Erber, polarization ratio, spin bound
void p_tables() {
    const QedTab T = qed_tables().view_host();
    double e0 = 0, e3 = 0;
    for (double chi : {0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0}) {
        double T0 = 0, T3 = 0, Mb = 0;
        qed_bw_integrals(T, chi, T0, T3, Mb);
        const double ef = std::exp(-8 / (3 * chi));
        e0 = std::max(e0, std::abs(T0 * ef / T0_ref(chi) - 1));
        e3 = std::max(e3, std::abs(T3 * ef / T3_ref(chi) - 1));
    }
    check("pairs P1a: Breit-Wheeler tables T0 = int B0 dx, T3 = int K23 dx vs quadrature (chi_gamma 0.05 .. 1000)",
          std::isfinite(e0 + e3) && e0 < 1e-4 && e3 < 1e-4, fmt("max rel. error T0 %.1e", e0) + fmt(", T3 %.1e", e3));
    auto rate = [&](double chi) { double T0 = 0, T3 = 0, Mb = 0; qed_bw_integrals(T, chi, T0, T3, Mb); return T0 * std::exp(-8 / (3 * chi)) / (std::sqrt(3.0) * M_PI); };
    const double cs = 3 * std::sqrt(3.0) / (16 * std::sqrt(2.0));
    const double ci = std::pow(3.0, 2.0 / 3) * 15 * std::pow(std::tgamma(2.0 / 3), 3) / (14 * std::sqrt(3.0) * M_PI * std::tgamma(1.0 / 3));
    const double s1 = rate(0.02) / (cs * 0.02 * std::exp(-8 / 0.06)) - 1, s2 = rate(0.05) / (cs * 0.05 * std::exp(-8 / 0.15)) - 1;
    const double l1 = rate(1e3) / (ci * std::pow(1e3, 2.0 / 3)) - 1, l2 = rate(1e4) / (ci * std::pow(1e4, 2.0 / 3)) - 1;
    check("pairs P1b: Erber asymptotics: chi << 1 3sqrt3/(16 sqrt2) chi e^(-8/(3chi)) (correction ~ chi), chi >> 1 0.3796 chi^(2/3) (~ chi^(-2/3))",
          std::abs(s1) < 6e-3 && s2 / s1 > 2.2 && s2 / s1 < 2.8 && std::abs(l2) < 6e-3 && l1 / l2 > 4.1 && l1 / l2 < 5.2,
          fmt("rate / asymptote - 1: chi 0.02 %.2e", s1) + fmt(", 0.05 %.2e", s2) + fmt("; 1e3 %.2e", l1) + fmt(", 1e4 %.2e", l2));
    auto ratio = [&](double chi) { double T0 = 0, T3 = 0, Mb = 0; qed_bw_integrals(T, chi, T0, T3, Mb); return (T0 + T3) / (T0 - T3); };
    const double r1 = (T0_ref(1.0) + T3_ref(1.0)) / (T0_ref(1.0) - T3_ref(1.0));
    check("pairs P1c: rate ratio of photons polarized along e2 / along e1 (xi3 = -1 / +1): -> 2 (chi -> 0), -> 3/2 (chi -> inf)",
          std::abs(ratio(0.02) - 2) < 0.05 && std::abs(ratio(1e4) - 1.5) < 0.02 && std::abs(ratio(1.0) / r1 - 1) < 1e-4,
          fmt("chi 0.02: %.4f", ratio(0.02)) + fmt(", 1: %.4f", ratio(1.0)) + fmt(", 1e4: %.4f", ratio(1e4)));
    // |b_+-| <= p0: the mean pair spins are physical for every photon state (saturated: pure pair states at the edge)
    double worst = 0;
    unsigned long long k = 7;
    for (double chi : {0.05, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0})
        for (int i = 1; i < 60; ++i) {
            const double x = i / 60.0, xm = 1 - x;
            if (rho_of(x, chi) > 600) continue;
            const Bes b = bes(rho_of(x, chi));
            const double B0 = (x * x + xm * xm) / (x * xm) * b.k23 + b.ik13;
            for (int t = 0; t < 200; ++t) {
                double v[3] = {2 * ion_uniform(k++) - 1, 2 * ion_uniform(k++) - 1, 2 * ion_uniform(k++) - 1};
                const double nv = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                for (double& c : v) c /= nv;
                const double p0 = B0 - v[2] * b.k23;
                const double bp[3] = {v[0] * b.k13 / xm, b.k13 / x - v[2] * b.k13 / xm, v[1] * ((x - xm) / (x * xm) * b.k23 + b.ik13 / x)};
                const double bm[3] = {v[0] * b.k13 / x, b.k13 / xm - v[2] * b.k13 / x, v[1] * ((xm - x) / (x * xm) * b.k23 + b.ik13 / xm)};
                worst = std::max({worst, std::sqrt(bp[0] * bp[0] + bp[1] * bp[1] + bp[2] * bp[2]) / p0,
                                  std::sqrt(bm[0] * bm[0] + bm[1] * bm[1] + bm[2] * bm[2]) / p0});
            }
        }
    check("pairs P1d: pair spins |S_+-| = |b_+-| / p0 <= 1 for all x, chi and pure photon states", worst <= 1 + 1e-12 && worst > 0.99,
          fmt("max %.6f", worst));
}

// ---------------------------------------------------------------- P2 pair spectrum, per-event kinematics
void p_spectrum() {
    const long N = 1500000;
    for (double chi : {0.3, 1.0, 10.0}) {
        const int bins = 24;
        BwMC m(chi, 2e4, bins, false);
        double T0 = 0, T3 = 0, Mb = 0;
        qed_bw_integrals(m.K.T, chi, T0, T3, Mb);
        m.K.dt = 0.05 / (m.Cg * T0 * std::exp(-8 / (3 * chi)));
        double eerr = 0, pdef_bad = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(max : eerr, pdef_bad)
#endif
        for (long t = 0; t < N; ++t) {
            const double xi[3] = {0, 0, 0};
            const Event ev = m.trial(xi, key_of(t, 41));
            if (!ev.dec) continue;
            const double ep = std::sqrt(1 + ev.p.UX * ev.p.UX + ev.p.UY * ev.p.UY + ev.p.UZ * ev.p.UZ);
            const double em = std::sqrt(1 + ev.e.UX * ev.e.UX + ev.e.UY * ev.e.UY + ev.e.UZ * ev.e.UZ);
            eerr = std::max(eerr, std::abs(ep + em - m.om) / m.om);
            const double def = m.om - (ev.p.UX + ev.e.UX), want = 1 / (2 * ep) + 1 / (2 * em);   // m^2 / (2 eps) each (eps >> 1)
            pdef_bad = std::max(pdef_bad, std::abs(def / want - 1));
        }
        double c2 = 0, worst = 0;
        int nb = 0;
        for (int b = 0; b < bins; ++b) {
            const double ref = simpson([&](double x) { return B0_ref(x, chi); }, std::max(1e-9, double(b) / bins), std::min(1 - 1e-9, double(b + 1) / bins), 40);
            const double mu = N * m.K.dt * m.Cg * ref;
            if (mu < 30) continue;
            const double z = (m.hx[4 * size_t(b)] - mu) / std::sqrt(mu);
            c2 += z * z; ++nb; worst = std::max(worst, std::abs(z));
        }
        check("pairs P2: pair spectrum dN/dx = Cg B0 (unpolarized photons, " + fmt("chi = %g", chi) + ", 1.5e6 trials); eps_+ + eps_- = eps_gamma",
              nb > 5 && c2 / nb < 2.0 && worst < 4.5 && eerr < 1e-12 && pdef_bad < 1e-3,
              fmt("chi^2/bin %.2f", c2 / nb) + fmt(", worst %.1f sigma", worst) + fmt(", %.0f pairs", m.tot[0]) + fmt("; energy %.0e", eerr) +
              fmt(", momentum deficit = m^2 (1/2eps+ + 1/2eps-) to %.0e", pdef_bad));
    }
}

// ---------------------------------------------------------------- P3 polarization dependence and selection
void p_polarization() {
    const double chi = 1.0;
    const long N = 2000000;
    double n[2] = {0, 0}, dt = 0, T0 = 0, T3 = 0, Mb = 0;
    for (int s = 0; s < 2; ++s) {
        BwMC m(chi, 2e4, 4, false, false);
        qed_bw_integrals(m.K.T, chi, T0, T3, Mb);
        const double ef = std::exp(-8 / (3 * chi));
        T0 *= ef; T3 *= ef;
        m.K.dt = dt = 0.05 / (m.Cg * T0);
        long c = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : c)
#endif
        for (long t = 0; t < N; ++t) {
            const double xi[3] = {0, 0, s ? -1.0 : 1.0};
            c += m.trial(xi, key_of(t, 43 + s)).dec;
        }
        n[s] = double(c);
    }
    BwMC m0(chi, 2e4, 4, false);
    const double mu[2] = {N * dt * m0.Cg * (T0 - T3), N * dt * m0.Cg * (T0 + T3)};
    const double z0 = (n[0] - mu[0]) / std::sqrt(mu[0]), z1 = (n[1] - mu[1]) / std::sqrt(mu[1]);
    check("pairs P3a: decay rates of photons polarized along e1 (xi3 = +1) and e2 (xi3 = -1) = Cg (T0 -+ T3) (chi = 1)",
          std::abs(z0) < 4 && std::abs(z1) < 4, fmt("ratio %.4f", n[1] / n[0]) + fmt(" (theory %.4f)", (T0 + T3) / (T0 - T3)) +
          fmt(", %.1f", z0) + fmt(" / %.1f sigma", z1));
    // survival and selection of unpolarized photons: P(t) = (e^(-Cg (T0 - T3) t) + e^(-Cg (T0 + T3) t)) / 2,
    // xi3 of the survivors = tanh(Cg T3 t)
    for (int pure = 0; pure < 2; ++pure) {
        BwMC m(chi, 2e4, 4, pure);
        const int np = 200000, ns = 40;
        m.K.pmax = 1e-3;                       // sub-steps: (1 - a Gamma)^k = e^(-Gamma t) to 1e-3
        m.K.dt = 2.0 / (m.Cg * T0) / ns;
        std::vector<double> xi(3 * size_t(np), 0.0);
        std::vector<char> alive(size_t(np), 1);
        double worst_p = 0, worst_x = 0, lastP = 0, lastX = 0;
        for (int k = 1; k <= ns; ++k) {
            long a = 0;
            double sx = 0, sx2 = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : a, sx, sx2)
#endif
            for (int p = 0; p < np; ++p) {
                if (!alive[size_t(p)]) continue;
                const Event ev = m.trial(&xi[3 * size_t(p)], key_of(long(k) * np + p, 47 + pure));
                if (ev.dec) { alive[size_t(p)] = 0; continue; }
                for (int c = 0; c < 3; ++c) xi[3 * size_t(p) + c] = ev.xi[c];
                ++a; sx += ev.xi[2]; sx2 += ev.xi[2] * ev.xi[2];
            }
            const double t = k * m.K.dt, P = double(a) / np, X = sx / a;
            const double Pw = 0.5 * (std::exp(-m.Cg * (T0 - T3) * t) + std::exp(-m.Cg * (T0 + T3) * t)), Xw = std::tanh(m.Cg * T3 * t);
            const double sP = std::sqrt(Pw * (1 - Pw) / np), sX = std::sqrt(std::max(sx2 / a - X * X, 1e-8) / a);
            worst_p = std::max(worst_p, std::abs(P - Pw) / (sP + 1e-4 * Pw));
            worst_x = std::max(worst_x, std::abs(X - Xw) / (sX + 2e-3 * Xw + 1e-6));
            lastP = P; lastX = X;
        }
        check(std::string("pairs P3b: survival and polarization of unpolarized photons: xi3 = tanh(Cg T3 t) (selection, ") + (pure ? "pure)" : "mixed)"),
              worst_p < 4.5 && worst_x < 4.5, fmt("at Cg T0 t = 2: survival %.4f", lastP) + fmt(", xi3 %.4f", lastX) +
              fmt(" (theory %.4f)", std::tanh(2 * T3 / T0)) + fmt("; worst %.1f", worst_p) + fmt(" / %.1f sigma", worst_x));
    }
}

// ---------------------------------------------------------------- P4 pair spins
void p_spins() {
    const double chi = 1.0;
    const int bins = 10;
    struct Case { double xi[3]; const char* name; };
    const Case cases[] = {{{0, 0, 0}, "unpolarized"}, {{0, 0, 1}, "xi3 = +1"}, {{0, 0, -1}, "xi3 = -1"}, {{1, 0, 0}, "xi1 = 1"}, {{0, 1, 0}, "xi2 = 1"}};
    for (int pure = 0; pure < 2; ++pure) {
        double worst = 0;
        std::string info;
        for (const Case& cs : cases) {
            BwMC m(chi, 2e4, 4, pure, false);
            double T0 = 0, T3 = 0, Mb = 0;
            qed_bw_integrals(m.K.T, chi, T0, T3, Mb);
            m.K.dt = 0.05 / (m.Cg * T0 * std::exp(-8 / (3 * chi)));
            const long N = pure ? 3000000 : 600000;
            // per bin: count, sums of S_+ and S_- in the global axes (6), squares (6)
            std::vector<double> acc(size_t(bins) * 13, 0.0);
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                std::vector<double> loc(acc.size(), 0.0);
#ifdef _OPENMP
#pragma omp for
#endif
                for (long t = 0; t < N; ++t) {
                    const Event ev = m.trial(cs.xi, key_of(t, 53));
                    if (!ev.dec) continue;
                    const double ep = std::sqrt(1 + ev.p.UX * ev.p.UX);
                    const int b = std::min(bins - 1, int(ep / m.om * bins));
                    const double v[6] = {ev.p.S0, ev.p.S1, ev.p.S2, ev.e.S0, ev.e.S1, ev.e.S2};
                    double* a = &loc[13 * size_t(b)];
                    a[0] += 1;
                    for (int c = 0; c < 6; ++c) { a[1 + c] += v[c]; a[7 + c] += v[c] * v[c]; }
                }
#ifdef _OPENMP
#pragma omp critical
#endif
                for (size_t i = 0; i < acc.size(); ++i) acc[i] += loc[i];
            }
            // theory: e1 = -y, e2 = -z, n = x; components of b_+ and b_- in (x, y, z) per bin = bin integral of b / of p0
            const double* X = cs.xi;
            for (int b = 0; b < bins; ++b) {
                const double* a = &acc[13 * size_t(b)];
                if (a[0] < 2000) continue;
                const double lo = std::max(1e-9, double(b) / bins), hi = std::min(1 - 1e-9, double(b + 1) / bins);
                auto in = [&](const std::function<double(double, const Bes&)>& f) {
                    return simpson([&](double x) { return f(x, bes(rho_of(x, chi))); }, lo, hi, 40);
                };
                const double p0 = in([&](double x, const Bes& c) { const double xm = 1 - x; return (x * x + xm * xm) / (x * xm) * c.k23 + c.ik13 - X[2] * c.k23; });
                // Seipt & King (2020), Chen et al. (2022): transverse terms with the positron along -e2 = +B' for xi = 0
                const double cp2 = in([&](double x, const Bes& c) { return -(c.k13 / x - X[2] * c.k13 / (1 - x)); });
                const double cp1 = in([&](double x, const Bes& c) { return -X[0] * c.k13 / (1 - x); });
                const double cpn = in([&](double x, const Bes& c) { const double xm = 1 - x; return X[1] * ((x - xm) / (x * xm) * c.k23 + c.ik13 / x); });
                const double cm2 = in([&](double x, const Bes& c) { return c.k13 / (1 - x) - X[2] * c.k13 / x; });
                const double cm1 = in([&](double x, const Bes& c) { return X[0] * c.k13 / x; });
                const double cmn = in([&](double x, const Bes& c) { const double xm = 1 - x; return X[1] * ((xm - x) / (x * xm) * c.k23 + c.ik13 / xm); });
                // global components: b = c2 e2 + c1 e1 + cn n, e1 = -y, e2 = -z
                const double want[6] = {cpn / p0, -cp1 / p0, -cp2 / p0, cmn / p0, -cm1 / p0, -cm2 / p0};
                for (int c = 0; c < 6; ++c) {
                    const double mean = a[1 + c] / a[0], var = std::max(a[7 + c] / a[0] - mean * mean, 0.0);
                    // mixed: each event carries the exact local mean (spread only by x inside the bin): compare to 1e-2
                    const double sig = pure ? std::sqrt(std::max(var, 1e-4) / a[0]) : 0.01 * std::abs(want[c]) + 2e-3;
                    worst = std::max(worst, std::abs(mean - want[c]) / sig);
                }
            }
        }
        // the mean positron spin along e2 of unpolarized photons over all x (printed)
        check(std::string("pairs P4: pair spins per x bin vs b_+- / p0 (unpolarized, linear e1 / e2 / 45 deg, circular; ") + (pure ? "pure)" : "mixed)"),
              worst < 4.5, fmt("worst %.2f sigma", worst) + info);
    }
}

// P4b the orientation (Seipt & King 2020: the pair preferably in the lower magnetic energy state): unpolarized photons in two
// fields, the rest-frame magnetic field B' = B - n x E of a lepton along n computed here from the fields (not from qed_basis):
// <S_+ . B'> > 0 (positron parallel), <S_- . B'> < 0 (electron antiparallel); x-averaged values vs the integrals of b_+- / p0.
void p_orientation() {
    const double chi = 1.0;
    BwMC m(chi, 2e4, 4, false, false);
    double T0 = 0, T3 = 0, Mb = 0;
    qed_bw_integrals(m.K.T, chi, T0, T3, Mb);
    m.K.dt = 0.05 / (m.Cg * T0 * std::exp(-8 / (3 * chi)));
    // reference: <S_+ . B'^> = int K13/x dx / int B0 dx over the decays (unpolarized; by symmetry the same for the electron)
    const double ref = half_ln([&](double x) { const Bes c = bes(rho_of(x, chi)); return 0.5 * (c.k13 / x + c.k13 / (1 - x)); }, chi) /
                       half_ln([&](double x) { return B0_ref(x, chi); }, chi);
    std::string info;
    bool ok = true;
    for (int f = 0; f < 2; ++f) {
        // f = 0: B along z (B' along +z); f = 1: the crossed field E = -B/2 y, B = B/2 z of a wave along -x (head-on, the same
        // chi; B' = B z)
        const double E[3] = {0, f ? -0.5 * m.B : 0.0, 0}, Bv[3] = {0, 0, f ? 0.5 * m.B : m.B}, n[3] = {1, 0, 0};
        const double nxE[3] = {n[1] * E[2] - n[2] * E[1], n[2] * E[0] - n[0] * E[2], n[0] * E[1] - n[1] * E[0]};
        double Bp[3] = {Bv[0] - nxE[0], Bv[1] - nxE[1], Bv[2] - nxE[2]};
        const double bn = std::sqrt(Bp[0] * Bp[0] + Bp[1] * Bp[1] + Bp[2] * Bp[2]);
        for (double& c : Bp) c /= bn;
        double sp = 0, sm = 0, cnt = 0;
        for (long t = 0; t < 400000; ++t) {
            Event ev;
            BwOut O;
            O.tot = m.tot.data(); O.hx = m.hx.data(); O.he = m.he.data();
            auto set = [](BwOut::L& o, Lep& l) {
                o.X = &l.X; o.Y = &l.Y; o.Z = &l.Z; o.UX = &l.UX; o.UY = &l.UY; o.UZ = &l.UZ; o.W = &l.W; o.S0 = &l.S0; o.S1 = &l.S1;
                o.S2 = &l.S2; o.ID = &l.ID;
            };
            set(O.e, ev.e); set(O.p, ev.p);
            const double u[3] = {m.om, 0, 0}, x[3] = {0, 0, 0};
            double xi[3] = {0, 0, 0}, pe[3] = {0, -1, 0};
            if (!qed_pair<false>(m.K, O, E, Bv, u, xi, pe, 1.0, x, key_of(t, 71 + f), 0)) continue;
            sp += ev.p.S0 * Bp[0] + ev.p.S1 * Bp[1] + ev.p.S2 * Bp[2];
            sm += ev.e.S0 * Bp[0] + ev.e.S1 * Bp[1] + ev.e.S2 * Bp[2];
            cnt += 1;
        }
        sp /= cnt; sm /= cnt;
        ok = ok && cnt > 1000 && sp > 0 && sm < 0 && std::abs(sp - ref) < 0.02 * ref && std::abs(sm + ref) < 0.02 * ref;
        info += fmt(f ? "; crossed field: %.4f" : "magnetic field: <S+.B'> %.4f", sp) + fmt(", <S-.B'> %.4f", sm);
    }
    check("pairs P4b: positron spin parallel, electron spin antiparallel to the rest-frame B' (Seipt & King 2020; unpolarized photons)",
          ok, info + fmt(" (theory +-%.4f)", ref));
}

// ---------------------------------------------------------------- P5 cascade energy balance, P6 order, P7 decks
Config deck_config(const std::string& text, const char* path) {
    if (FILE* f = std::fopen(path, "w")) { std::fputs(text.c_str(), f); std::fclose(f); }
    Deck dk = Deck::read(path);
    std::remove(path);
    return Config::from_deck(dk);
}
int idx(Simulation& s, const char* name) {
    for (size_t i = 0; i < s.cfg.species.size(); ++i) if (s.cfg.species[i].name == name) return int(i);
    return -1;
}
// electrons at gamma = 2000 across a constant B (chi_e = 20): photons up to ~2000 m c^2, chi_gamma up to ~20
// grid_cells: the cell sizes (default those of P5); numerics: extra numerics keys (solver, seed)
std::string cascade_deck(bool rz, const std::string& extra = "", const std::string& grid_cells = "", const std::string& numerics = " seed = 5\n") {
    const std::string cells = !grid_cells.empty() ? grid_cells : rz ? " dz = 0.02 um\n dr = 0.2 um\n" : " dx = 0.02 um\n dy = 0.04 um\n dz = 0.04 um\n";
    const std::string grid = rz ? "grid {\n geometry = rz\n modes = 2\n lambda0 = 0.8 um\n nz = 16\n nr = 64\n" + cells + " t_end = 1 ps\n}\n"
                                : "grid {\n lambda0 = 0.8 um\n nx = 8\n ny = 8\n nz = 8\n" + cells + " t_end = 1 ps\n}\n";
    const std::string beam = rz ? "species e {\n charge = -1\n mass = 1\n density = 1e-10\n rmax = 0.4 um\n ppc_r = 1\n ppc_theta = 8\n ux = 1999.99975\n spin = 0 0 1\n}\n"
                                : "species e {\n charge = -1\n mass = 1\n density = 1e-10\n uy = 1999.99975\n spin = 0 0 1\n}\n";
    return grid + "numerics {\n" + numerics + "}\n" + beam +
           "species pe {\n charge = -1\n mass = 1\n density = 0\n spin = 0 0 0\n}\nspecies pp {\n charge = 1\n mass = 1\n density = 0\n spin = 0 0 0\n}\n"
           "species ph {\n kind = photon\n}\nqed {\n species = e pe pp\n photons = ph\n pair_electrons = pe\n pair_positrons = pp\n" + extra +
           "}\noutput {\n dir = pairs_tmp_out\n}\n";
}
void set_b(Simulation& s, bool rz, double B0) {
    Field& F = rz ? s.rz->bz : s.fields->bx;
    H h(F);
    const int g = s.dom.ng;
    const int nk = rz ? 1 : s.dom.n[2] + 2 * g, k0 = rz ? 0 : -g, j0 = rz ? 0 : -g, nj = rz ? s.dom.n[1] : s.dom.n[1] + 2 * g;
    for (int k = k0; k < k0 + nk; ++k) for (int j = j0; j < j0 + nj; ++j) for (int i = -g; i < s.dom.n[0] + g; ++i) h(i, j, k) = B0;
    h.put(F);
}
void p_cascade(Comm& comm) {
    const double B0 = 20 / (2000 * qed_hw(LAM));
    for (int rz = 0; rz < 2; ++rz) {
        Config c = deck_config(cascade_deck(rz), "pairs_tmp.deck");
        Simulation s(c, comm);
        s.init();
        set_b(s, rz, B0);
        const char* nm[4] = {"e", "pe", "pp", "ph"};
        Species* sp[4];
        for (int k = 0; k < 4; ++k) sp[k] = &s.species[size_t(idx(s, nm[k]))];
        double e0[2] = {sp[0]->kinetic_energy(), 0};
        for (size_t p = 0; p < sp[0]->size(); ++p) e0[1] += sp[0]->get(Species::W)[p] * sp[0]->q;
        comm.allreduce_sum(e0, 2);
        const int nsteps = 40;
        for (int n = 0; n < nsteps; ++n) s.step();
        double v[8] = {0};
        for (int k = 0; k < 4; ++k) v[k] = sp[k]->kinetic_energy();
        for (int k = 0; k < 3; ++k) { const std::vector<double> w = sp[k]->get(Species::W); for (double x : w) v[4] += x * sp[k]->q; }
        { const std::vector<double> w = sp[2]->get(Species::W); for (double x : w) v[5] += x; }
        { const std::vector<double> w = sp[1]->get(Species::W); for (double x : w) v[6] += x; }
        v[7] = double(sp[2]->size());
        comm.allreduce_sum(v, 8);
        double rad[2], pr[2];
        qed_totals(s, rad);                            // all photons emitted, those into ph (all of them: no threshold)
        qed_pair_totals(s, pr);                        // sum w of the pairs, their photons' energy
        double lost = 0;
        for (int k = 0; k < 4; ++k) lost += sp[k]->lost_energy;
        comm.allreduce_sum(&lost, 1);
        // energy: kinetic of all leptons + photons + the rest energy 2 m c^2 of the pairs (+ what left the box)
        const double bal = (v[0] + v[1] + v[2] + v[3] + 2 * pr[0] + (rad[0] - rad[1]) + lost - e0[0]) / e0[0];
        check(std::string("pairs P5: cascade e- -> photons -> pairs (chi_e = 20, 40 steps, ") + (rz ? "RZ" : "Cart3D") +
                  "): energy balance with the pair rest energy, charge, pairs = positrons",
              std::abs(bal) < 1e-9 && std::abs(v[4] - e0[1]) < 1e-12 * std::abs(e0[1]) && pr[0] > 0 && std::abs(v[5] / pr[0] - 1) < 1e-12 &&
                  std::abs(v[6] / pr[0] - 1) < 1e-12,
              fmt("balance %.1e", bal) + fmt(", %.0f positrons", v[7]) + fmt(", pair weight / beam weight %.3f", pr[0] / std::abs(e0[1])) +
              fmt(", E_photons_decayed / E_0 %.3f", pr[1] / e0[0]));
    }
    // P6: the same run with the beam in reverse order: identical (sorted) positron momenta and spins
    std::vector<double> g[2];
    for (int rev = 0; rev < 2; ++rev) {
        Config c = deck_config(cascade_deck(false, " spin_state = pure\n"), "pairs_tmp.deck");
        Simulation s(c, comm);
        s.init();
        set_b(s, false, B0);
        Species& e = s.species[size_t(idx(s, "e"))];
        if (rev) {
            std::vector<int> perm(e.size());
            for (size_t p = 0; p < perm.size(); ++p) perm[p] = int(perm.size() - 1 - p);
            e.permute(perm);
        }
        for (int n = 0; n < 20; ++n) s.step();
        Species& pp = s.species[size_t(idx(s, "pp"))];
        const std::vector<double> ux = pp.get(Species::UX), uy = pp.get(Species::UY), s2 = pp.get(pp.slot("s2"));
        for (size_t p = 0; p < ux.size(); ++p) g[rev].push_back(ux[p] * ux[p] + uy[p] * uy[p] + 1e-3 * s2[p]);
        std::sort(g[rev].begin(), g[rev].end());
    }
    check("pairs P6: random numbers per particle id: the beam in reverse order gives identical pairs", g[0] == g[1] && !g[0].empty(),
          fmt("%.0f positrons", double(g[0].size())));
}

// P5b: the cascade of P5 with the iRIP field solver. The QED operator acts on the particles only; in the constant B of P5 the
// pair yield, the emitted energy and the decayed photons' energy at a fixed time depend on the QED rates and the sub-steps,
// not on the solver or the step. iRIP at the step of P5 must give the RIP-CC events (same random numbers, the same B at the
// particles); iRIP at 4 times that step with h_perp = c dt / 4 (RIP-CC rejects it: h_x^2 (1/h_y^2 + 1/h_z^2) = 8 in Cart3D,
// dz^2 / dr^2 = 16 in RZ) must agree with RIP-CC within the scatter of 4 seeds, with no event of probability > 1.
struct CascadeRun { double bal, pairs, erad, edec, over1, npos; };
CascadeRun run_cascade(Comm& comm, bool rz, const std::string& cells, const std::string& numerics, int nsteps) {
    const double B0 = 20 / (2000 * qed_hw(LAM));
    Config c = deck_config(cascade_deck(rz, "", cells, numerics), "pairs_tmp.deck");
    Simulation s(c, comm);
    s.init();
    set_b(s, rz, B0);
    const char* nm[4] = {"e", "pe", "pp", "ph"};
    Species* sp[4];
    for (int k = 0; k < 4; ++k) sp[k] = &s.species[size_t(idx(s, nm[k]))];
    double e0[2] = {sp[0]->kinetic_energy(), 0};
    for (size_t p = 0; p < sp[0]->size(); ++p) e0[1] += sp[0]->get(Species::W)[p];
    comm.allreduce_sum(e0, 2);
    for (int n = 0; n < nsteps; ++n) s.step();
    double v[5] = {0};
    for (int k = 0; k < 4; ++k) v[k] = sp[k]->kinetic_energy();
    v[4] = double(sp[2]->size());
    comm.allreduce_sum(v, 5);
    double rad[2], pr[2], lost = 0;
    qed_totals(s, rad);
    qed_pair_totals(s, pr);
    for (int k = 0; k < 4; ++k) lost += sp[k]->lost_energy;
    comm.allreduce_sum(&lost, 1);
    CascadeRun r;
    r.bal = (v[0] + v[1] + v[2] + v[3] + 2 * pr[0] + (rad[0] - rad[1]) + lost - e0[0]) / e0[0];
    r.pairs = pr[0] / e0[1];                           // pairs per beam electron (sums of weights)
    r.erad = rad[0] / e0[0];                           // energy of all photons emitted / initial beam energy
    r.edec = pr[1] / e0[0];                            // energy of the decayed photons / initial beam energy
    r.over1 = qed_events_over_one(s);
    r.npos = v[4];
    return r;
}
void p_cascade_irip(Comm& comm) {
    for (int rz = 0; rz < 2; ++rz) {
        const std::string g = rz ? "RZ" : "Cart3D";
        const std::string wide = rz ? " dz = 0.08 um\n dr = 0.02 um\n" : " dx = 0.08 um\n dy = 0.04 um\n dz = 0.04 um\n";
        // the same step: RIP-CC vs iRIP, seed 5
        const CascadeRun a = run_cascade(comm, rz, "", " seed = 5\n", 40);
        const CascadeRun b = run_cascade(comm, rz, "", " seed = 5\n solver = irip\n", 40);
        const double d = std::max({std::abs(b.pairs / a.pairs - 1), std::abs(b.erad / a.erad - 1), std::abs(b.edec / a.edec - 1)});
        check("pairs P5b: cascade of P5 with iRIP at the same step (" + g + "): the RIP-CC events, energy balance, no P > 1",
              d < 1e-6 && std::abs(b.bal) < 1e-9 && b.over1 == 0,
              fmt("pairs / beam electron %.5f vs %.5f", b.pairs, a.pairs) + fmt(", max relative difference %.1e", d) +
                  fmt(", balance %.1e", b.bal));
        // 4 times the step (h_perp = c dt / 4), 4 seeds each, the same time (40 / 10 steps)
        double m[2][3] = {{0}}, q[2][3] = {{0}}, worst_bal = 0, over1 = 0, npos = 0;
        const int ns = 4;
        for (int k = 0; k < ns; ++k) {
            const std::string seed = " seed = " + std::to_string(5 + k) + "\n";
            const CascadeRun r[2] = {run_cascade(comm, rz, "", seed, 40), run_cascade(comm, rz, wide, seed + " solver = irip\n", 10)};
            for (int j = 0; j < 2; ++j) {
                const double x[3] = {r[j].pairs, r[j].erad, r[j].edec};
                for (int i = 0; i < 3; ++i) { m[j][i] += x[i] / ns; q[j][i] += x[i] * x[i] / ns; }
                worst_bal = std::max(worst_bal, std::abs(r[j].bal));
            }
            over1 += r[1].over1;
            npos += r[1].npos / ns;
        }
        double z[3];                                   // difference in combined standard errors of the means
        for (int i = 0; i < 3; ++i) {
            double se2 = 0;
            for (int j = 0; j < 2; ++j) se2 += std::max(q[j][i] - m[j][i] * m[j][i], 0.0) * ns / (ns - 1) / ns;
            z[i] = (m[1][i] - m[0][i]) / std::sqrt(std::max(se2, 1e-300));
        }
        check("pairs P5b: cascade with iRIP at 4 x the step, h_perp = c dt / 4 (" + g + ", 4 seeds): pairs, emitted and decayed "
              "photon energy as RIP-CC within 4 standard errors, energy balance, no P > 1",
              std::abs(z[0]) < 4 && std::abs(z[1]) < 4 && std::abs(z[2]) < 4 && worst_bal < 1e-9 && over1 == 0,
              fmt("pairs / beam electron %.4f vs %.4f", m[1][0], m[0][0]) + fmt(" (%+.1f sigma)", z[0]) +
                  fmt(", E_rad / E_0 %.4f vs %.4f", m[1][1], m[0][1]) + fmt(" (%+.1f)", z[1]) +
                  fmt(", E_decayed / E_0 %.4f vs %.4f", m[1][2], m[0][2]) + fmt(" (%+.1f)", z[2]) +
                  fmt(", balance %.1e", worst_bal) + fmt(", P>1 %.0f", over1) + fmt(", %.0f positrons", npos));
    }
}

void p_deck() {
    auto bad = [](const std::string& species, const std::string& qed) {
        const std::string d = "grid {\n lambda0 = 0.8 um\n nx = 8\n ny = 4\n nz = 4\n dx = 0.02 um\n dy = 0.04 um\n dz = 0.04 um\n t_end = 1 fs\n}\n"
                              "species e {\n charge = -1\n mass = 1\n density = 1e-6\n}\nspecies ph {\n kind = photon\n}\n" + species + qed +
                              "output {\n dir = pairs_tmp_out\n}\n";
        try { deck_config(d, "pairs_tmp.deck"); } catch (const std::exception&) { return true; }
        return false;
    };
    const std::string pp = "species p {\n charge = 1\n mass = 1\n density = 0\n}\n";
    const bool a = bad(pp, "qed {\n species = e\n pair_electrons = e\n pair_positrons = p\n}\n");                    // no photons
    const bool b = bad(pp, "qed {\n species = e\n photons = ph\n pair_electrons = e\n}\n");                           // one of two
    const bool c = bad(pp, "qed {\n species = e\n photons = ph\n pair_electrons = p\n pair_positrons = e\n}\n");      // charges
    const bool d = bad(pp, "qed {\n species = e\n photons = ph\n model = deterministic\n pair_electrons = e\n pair_positrons = p\n}\n");
    const bool e = !bad(pp, "qed {\n species = e p\n photons = ph\n pair_electrons = e\n pair_positrons = p\n}\n");
    check("pairs P7: deck checks (pairs without photons, one pair species, wrong charges, deterministic rejected; valid deck accepted)",
          a && b && c && d && e, "");
}
}  // namespace

void test_pairs(Comm& comm) {
    p_tables();
    p_spectrum();
    p_polarization();
    p_spins();
    p_orientation();
    p_cascade(comm);
    p_cascade_irip(comm);
    p_deck();
}
