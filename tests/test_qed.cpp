// Spin-resolved QED (decision 93, docs/specs/qed.md, src/qed.hpp). References are independent of the code's tables: the
// Bessel functions from std::cyl_bessel_k, the integrals by quadrature here, g(chi) from the Baier-Katkov form with K_5/3.
// Q1 tables; Q2 photon spectrum vs C0 w at chi = 0.1, 1, 10; Q3 parallel minus antiparallel spin = 2 C0 g; Q4 photon Stokes
// vector vs Eq. 21 (and the xi1, xi2 terms); Q5 spin relaxation vs the analytic solution (Sokolov-Ternov limit along e2);
// Q6 energy loss vs P_alpha chi^2 g(chi); Q7 T-BMT (rotation vs RK4; precession in a constant B in Cart3D and RZ);
// Q8 energy balance electrons + photons, photon species; Q9 random numbers independent of the particle order; Q10 deck checks.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <numeric>
#include <vector>
#include "common.hpp"
#include "deck.hpp"
#include "qed.hpp"
#include "sim.hpp"

namespace {
const double LAM = 0.8e-6;
// ---------------------------------------------------------------- independent references
double K13(double y) { return y > 600 ? 0.0 : std::cyl_bessel_k(1.0 / 3, y); }   // (libstdc++ refuses large arguments)
double K23(double y) { return y > 600 ? 0.0 : std::cyl_bessel_k(2.0 / 3, y); }
double IK13(double y) {                                  // int_y^inf K13
    if (y > 700) return 0;
    if (y < 2) {                                         // pi/sqrt3 - int_0^y K13, x = s^3
        const int n = 2000;
        const double s1 = std::cbrt(y), h = s1 / n;
        double a = 0;
        for (int i = 0; i <= n; ++i) {
            const double s = i * h, f = i == 0 ? 0.0 : 3 * s * s * K13(s * s * s);   // -> 3 Gamma(1/3) 2^(-2/3) s at s -> 0
            a += (i == 0 || i == n ? 1 : (i % 2 ? 4 : 2)) * f;
        }
        return M_PI / std::sqrt(3.0) - a * h / 3;
    }
    const int n = 6000;
    const double h = 60.0 / n;
    double a = 0;
    for (int i = 0; i <= n; ++i) a += (i == 0 || i == n ? 1 : (i % 2 ? 4 : 2)) * K13(y + i * h);
    return a * h / 3;
}
double yof(double u, double chi) { return 2 * u / (3 * (1 - u) * chi); }
double w_ref(double u, double chi) { const double y = yof(u, chi); return (u * u - 2 * u + 2) / (1 - u) * K23(y) - IK13(y); }
// int_a^b f(u) du in ln u (Simpson)
double integ(const std::function<double(double)>& f, double a, double b, int n = 400) {
    const double la = std::log(a), h = (std::log(b) - la) / n;
    double s = 0;
    for (int i = 0; i <= n; ++i) {
        const double u = std::exp(la + i * h);
        if (u < 1) s += (i == 0 || i == n ? 1 : (i % 2 ? 4 : 2)) * f(u) * u;   // u -> 1: y -> inf, the integrands vanish
    }
    return s * h / 3;
}
// int_0^1 f(u) du in two parts (ln u, ln(1 - u)), as the singular ends need
double integ01(const std::function<double(double)>& f, double chi) {
    double s = integ(f, 1e-25 * chi, 0.5, 3000);
    const int n = 3000;
    const double lo = -50, hi = std::log(0.5), h = (hi - lo) / n;
    for (int i = 0; i <= n; ++i) {
        const double e = std::exp(lo + i * h);
        if (1 - e < 1) s += (i == 0 || i == n ? 1 : (i % 2 ? 4 : 2)) * f(1 - e) * e * h / 3;
    }
    return s;
}
// Baier-Katkov: g(chi) = 9 sqrt3 / (8 pi) int_0^inf [2 r^2 K53(r) / (2 + 3 chi r)^2 + 36 chi^2 r^3 K23(r) / (2 + 3 chi r)^4] dr
double g_bk(double chi) {
    const int n = 4000;
    const double lo = std::log(1e-8), hi = std::log(80.0), h = (hi - lo) / n;
    double s = 0;
    for (int i = 0; i <= n; ++i) {
        const double r = std::exp(lo + i * h), d = 2 + 3 * chi * r;
        const double f = 2 * r * r * std::cyl_bessel_k(5.0 / 3, r) / (d * d) + 36 * chi * chi * r * r * r * K23(r) / (d * d * d * d);
        s += (i == 0 || i == n ? 1 : (i % 2 ? 4 : 2)) * f * r;
    }
    return 9 * std::sqrt(3.0) / (8 * M_PI) * s * h / 3;
}

// ---------------------------------------------------------------- a host Monte-Carlo driver of qed_emit
// electron along x with gamma, B along z (chi), E = 0: ev = x, e1 = y (the force on q = -1), e2 = z
struct MC {
    QedK K;
    QedOut O;
    std::vector<double> tot, hu, he;
    double g, B, chi, C0;
    MC(double chi_, double gamma, int bins, double umin, bool pure, bool flip = true) : chi(chi_) {
        const double hw = qed_hw(LAM);
        K.T = qed_tables().view_host();
        K.hw = hw; K.c0f = QED_ALPHA / (std::sqrt(3.0) * M_PI * hw);
        K.q = -1; K.model = 1; K.pure = pure; K.flip = flip; K.spin = true; K.photons = false; K.pmax = 0.1;
        K.bins = bins; K.lu0 = std::log(umin); K.dlu = -K.lu0 / bins; K.le0 = std::log(1e-12); K.dle = 1;
        g = gamma;
        const double beta = std::sqrt(1 - 1 / (g * g));
        B = chi / (g * beta * hw);
        C0 = K.c0f / g;
        tot.assign(8, 0); hu.assign(5 * size_t(bins), 0); he.assign(5 * size_t(bins), 0);
        O.tot = tot.data(); O.hu = hu.data(); O.he = he.data();
    }
    // one trial: u, S reset to the given state unless keep; returns gamma lost
    double trial(double* u, double* S, unsigned long long key) {
        const double E[3] = {0, 0, 0}, Bv[3] = {0, 0, B}, x[3] = {0, 0, 0};
        const double g0 = std::sqrt(1 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        double chi_out;
        qed_emit<false>(K, O, E, Bv, u, S, 1.0, x, key, 0, chi_out);
        return g0 - std::sqrt(1 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    }
    double ux() const { return std::sqrt(g * g - 1); }
};
unsigned long long key_of(long t, unsigned long long salt) { return ion_mix(ion_mix(salt) ^ ion_mix((unsigned long long)t)); }

// ---------------------------------------------------------------- Q1 tables
void q_tables() {
    const QedTab T = qed_tables().view_host();
    double e = 0, e10 = 0;                               // all y; y <= 10 (where the photons are)
    for (double y : {1e-10, 1e-6, 1e-3, 0.01, 0.07, 0.3, 1.0, 2.5, 7.0, 20.0, 60.0, 150.0}) {
        double a, b, c;
        qed_bessel(T, y, a, b, c);
        const double d = std::max({std::abs(a / K13(y) - 1), std::abs(b / K23(y) - 1), std::abs(c / IK13(y) - 1)});
        e = std::max(e, d);
        if (y <= 10) e10 = std::max(e10, d);
    }
    check("qed Q1a: tables K13, K23, IntK13 (y = 1e-10 .. 150) = std::cyl_bessel_k", e < 3e-5 && e10 < 2e-6,
          fmt("max rel. error %.1e (y <= 10: ", e) + fmt("%.1e)", e10));
    double eg = 0, ew = 0;
    for (double chi : {0.01, 0.1, 1.0, 10.0, 100.0}) {
        double W, Gs, M, P;
        qed_integrals(T, chi, W, Gs, M, P);
        // g(chi) = gamma C0 P / (2/3 alpha chi^2 / hw) = sqrt3 / (2 pi) P / chi^2
        eg = std::max(eg, std::abs(std::sqrt(3.0) / (2 * M_PI) * P / (chi * chi) / g_bk(chi) - 1));
        ew = std::max(ew, std::abs(W / integ01([&](double u) { return w_ref(u, chi); }, chi) - 1));
    }
    check("qed Q1b: integrated rate W(chi) and g(chi) from the tables vs quadrature / Baier-Katkov (chi 0.01 .. 100)", std::isfinite(eg + ew) && eg < 1e-4 && ew < 1e-4,
          fmt("g: %.1e, W: %.1e", eg, ew));
    // Sokolov-Ternov: S_inf = -(K - G) / (Xi - W) -> -8 / (5 sqrt3) for chi -> 0 (spin along e2)
    const double chi = 1e-3;
    const double KG = integ01([&](double u) { return -u * u / (1 - u) * K13(yof(u, chi)); }, chi);
    const double XW = integ01([&](double u) { return -u * u / (1 - u) * K23(yof(u, chi)); }, chi);
    const double st = -KG / XW;
    check("qed Q1c: equilibrium spin along e2 at chi = 1e-3 = Sokolov-Ternov -8/(5 sqrt3) = -0.9238", std::isfinite(st) && std::abs(st + 8 / (5 * std::sqrt(3.0))) < 2e-3,
          fmt("S_inf %.5f", st));
}

// bin integrals of f over the code's u bins
std::vector<double> bin_int(const MC& m, const std::function<double(double)>& f) {
    std::vector<double> r(size_t(m.K.bins));
    for (int b = 0; b < m.K.bins; ++b) r[size_t(b)] = integ(f, std::exp(m.K.lu0 + b * m.K.dlu), std::exp(m.K.lu0 + (b + 1) * m.K.dlu), 60);
    return r;
}

// ---------------------------------------------------------------- Q2 spectrum, Q3 spin dependence, Q4 Stokes vector
void q_spectrum() {
    const long N = 1500000;
    for (double chi : {0.1, 1.0, 10.0}) {
        MC m(chi, 1e4, 24, 1e-6, false);
        double W, Gs, M, P;
        qed_integrals(m.K.T, chi, W, Gs, M, P);
        m.K.dt = 0.05 / (m.C0 * W);
        // trials in parallel: the histograms are atomic adds of integer counts (exact in any order)
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (long t = 0; t < N; ++t) {
            double u[3] = {m.ux(), 0, 0}, S[3] = {0, 0, 0};
            m.trial(u, S, key_of(t, 11));
        }
        const std::vector<double> ref = bin_int(m, [&](double u) { return w_ref(u, chi); });
        double c2 = 0, worst = 0;
        int nb = 0;
        for (int b = 0; b < m.K.bins; ++b) {
            const double mu = N * m.K.dt * m.C0 * ref[size_t(b)];
            if (mu < 30) continue;
            const double z = (m.hu[5 * size_t(b)] - mu) / std::sqrt(mu);
            c2 += z * z; ++nb; worst = std::max(worst, std::abs(z));
        }
        check("qed Q2: photon spectrum dN/du = C0 w_rad (unpolarized, " + fmt("chi = %g", chi) + ", 1.5e6 trials)", nb > 5 && c2 / nb < 2.0 && worst < 4.5,
              fmt("chi^2/bin %.2f", c2 / nb) + fmt(", worst %.1f sigma", worst) + fmt(", %.0f photons", m.tot[0]));
    }
    for (double chi : {0.1, 1.0}) {
        std::vector<double> h[2];
        double dt = 0;
        MC m0(chi, 1e4, 16, 1e-4, false);              // bins and C0
        const MC* mp = &m0;
        for (int sgn = 0; sgn < 2; ++sgn) {
            MC m(chi, 1e4, 16, 1e-4, false, false);              // spin fixed: no flip
            double W, Gs, M, P;
            qed_integrals(m.K.T, chi, W, Gs, M, P);
            m.K.dt = dt = 0.05 / (m.C0 * W);
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (long t = 0; t < N; ++t) {
                double u[3] = {m.ux(), 0, 0}, S[3] = {0, 0, sgn ? -1.0 : 1.0};
                m.trial(u, S, key_of(t, 13 + sgn));
            }
            h[sgn] = m.hu;
        }
        const std::vector<double> ref = bin_int(*mp, [&](double u) { return -2 * u * K13(yof(u, chi)); });   // 2 g . e2
        double c2 = 0, worst = 0;
        int nb = 0;
        for (int b = 0; b < mp->K.bins; ++b) {
            const double n1 = h[0][5 * size_t(b)], n2 = h[1][5 * size_t(b)], var = n1 + n2;
            if (var < 100) continue;
            const double z = (n1 - n2 - N * dt * mp->C0 * ref[size_t(b)]) / std::sqrt(var);
            c2 += z * z; ++nb; worst = std::max(worst, std::abs(z));
        }
        check("qed Q3: rate(S = +e2) - rate(S = -e2) = 2 C0 g_rad (" + fmt("chi = %g", chi) + ")", nb > 5 && c2 / nb < 2.0 && worst < 4.5,
              fmt("chi^2/bin %.2f", c2 / nb) + fmt(", worst %.1f sigma", worst));
    }
    // Q4: <xi> per u bin for S = +e2, -e2 (xi3, Eq. 21), S = e1 (xi1), S = ev (xi2); mixed (deterministic per event) and pure
    const double chi = 1.0;
    struct Case { double S[3]; int comp; const char* name; };
    const Case cases[] = {{{0, 0, 1}, 2, "xi3, S = +e2"}, {{0, 0, -1}, 2, "xi3, S = -e2"}, {{0, 1, 0}, 0, "xi1, S = e1"}, {{1, 0, 0}, 1, "xi2, S = ev"}};
    for (int pure = 0; pure < 2; ++pure)
        for (const Case& cs : cases) {
            MC m(chi, 1e4, 12, 1e-3, pure, false);
            double W, Gs, M, P;
            qed_integrals(m.K.T, chi, W, Gs, M, P);
            m.K.dt = 0.05 / (m.C0 * W);
            const long n = pure ? N : N / 5;
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (long t = 0; t < n; ++t) {
                double u[3] = {m.ux(), 0, 0}, S[3] = {cs.S[0], cs.S[1], cs.S[2]};
                m.trial(u, S, key_of(t, 17 + cs.comp));
            }
            const double Sv = cs.S[0], S1 = cs.S[1], S2 = cs.S[2];
            auto p0 = [&](double u) { return w_ref(u, chi) - u * K13(yof(u, chi)) * S2; };
            std::function<double(double)> px;
            if (cs.comp == 2) px = [&](double u) { const double y = yof(u, chi); return K23(y) - u / (1 - u) * K13(y) * S2; };
            else if (cs.comp == 0) px = [&](double u) { return u / (1 - u) * K13(yof(u, chi)) * S1; };
            else px = [&](double u) { const double y = yof(u, chi); return ((2 * u - u * u) / (1 - u) * K23(y) - u * IK13(y)) * Sv; };
            const std::vector<double> num = bin_int(m, px), den = bin_int(m, p0);
            double worst = 0;
            for (int b = 0; b < m.K.bins; ++b) {
                const double cnt = m.hu[5 * size_t(b)];
                if (cnt < 400) continue;
                const double got = m.hu[5 * size_t(b) + 2 + cs.comp] / cnt, want = num[size_t(b)] / den[size_t(b)];
                const double sig = pure ? std::sqrt(std::max(1 - want * want, 1e-4) / cnt) : 0.02 * std::abs(want) + 1e-3;
                worst = std::max(worst, std::abs(got - want) / sig);
            }
            check(std::string("qed Q4: photon Stokes vector per u bin vs Eq. 21 (") + cs.name + (pure ? ", pure states)" : ", mixed states)"),
                  worst < 4.5, fmt("worst %.2f sigma", worst));
        }
}

// ---------------------------------------------------------------- Q5 spin relaxation vs the analytic solution (Eq. 14)
void q_relax() {
    for (double chi : {1.0, 0.1})
        for (int pure = 0; pure < 2; ++pure) {
            MC m(chi, 1e4, 4, 1e-3, pure);
            double W, Gs, M, P;
            qed_integrals(m.K.T, chi, W, Gs, M, P);
            const double KG = integ01([&](double u) { return -u * u / (1 - u) * K13(yof(u, chi)); }, chi);
            const double XW = integ01([&](double u) { return -u * u / (1 - u) * K23(yof(u, chi)); }, chi);
            const double lam = -m.C0 * XW, Sinf = -KG / XW;              // S(t) = (S0 - Sinf) e^(-lam t) + Sinf
            // the energy is reset every step: at most ~0.05 emissions per step, so no sub-step runs at a lowered gamma
            const int np = chi > 0.5 ? 10000 : 4000;
            const double T = 2.0 / lam;
            const int nsteps = std::max(200, int(std::ceil(T * m.C0 * W / 0.05)));
            m.K.dt = T / nsteps;
            std::vector<double> S(3 * size_t(np), 0.0);
            for (int p = 0; p < np; ++p) S[3 * size_t(p) + 2] = 1.0;   // S0 = +e2
            double worst = 0, last = 0;
            for (int k = 1; k <= nsteps; ++k) {
                double sum = 0, sum2 = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum, sum2)
#endif
                for (int p = 0; p < np; ++p) {
                    double u[3] = {m.ux(), 0, 0};
                    m.trial(u, &S[3 * size_t(p)], key_of(long(k) * np + p, 23 + pure));
                    sum += S[3 * size_t(p) + 2]; sum2 += S[3 * size_t(p) + 2] * S[3 * size_t(p) + 2];
                }
                const double mean = sum / np, sig = std::sqrt(std::max(sum2 / np - mean * mean, 1e-6) / np);
                const double want = (1 - Sinf) * std::exp(-lam * k * m.K.dt) + Sinf;
                if (k % (nsteps / 10) == 0) worst = std::max(worst, std::abs(mean - want) / (sig + 2e-3 * std::abs(want)));
                last = mean;
            }
            check(std::string("qed Q5: spin along e2 relaxes as Eq. 14 (") + fmt("chi = %g", chi) + (pure ? ", pure)" : ", mixed)"),
                  std::isfinite(Sinf) && std::isfinite(last) && worst < 4.5, fmt("S(2/lambda) %.4f", last) + fmt(" (analytic %.4f)", (1 - Sinf) * std::exp(-2.0) + Sinf) +
                  fmt(", S_inf %.4f", Sinf) + fmt(", worst %.1f sigma", worst) + fmt(", %.0f steps", nsteps));
        }
}

// ---------------------------------------------------------------- Q6 energy loss vs P_alpha chi^2 g(chi)
void q_energy_loss() {
    for (double chi : {0.1, 1.0, 10.0}) {
        MC m(chi, 1e4, 4, 1e-3, false);
        double W, Gs, M, P;
        qed_integrals(m.K.T, chi, W, Gs, M, P);
        m.K.dt = 0.05 / (m.C0 * W);
        const long N = 1000000;
        double s = 0, s2 = 0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+ : s, s2)
#endif
        for (long t = 0; t < N; ++t) {
            double u[3] = {m.ux(), 0, 0}, S[3] = {0, 0, 0};
            const double d = m.trial(u, S, key_of(t, 31));
            s += d; s2 += d * d;
        }
        const double rate = s / (N * m.K.dt), sig = std::sqrt(s2 / N - (s / N) * (s / N)) / std::sqrt(double(N)) / m.K.dt;
        const double want = 2.0 / 3 * QED_ALPHA * chi * chi * g_bk(chi) / m.K.hw;    // P_alpha chi^2 g(chi) in m c^2 omega0
        // deterministic model: one short step
        QedK K = m.K;
        K.model = 2; K.dt = 1e-4 / (m.C0 * P);
        double u[3] = {m.ux(), 0, 0}, S[3] = {0, 0, 0}, chi_out;
        const double E[3] = {0, 0, 0}, B[3] = {0, 0, m.B}, x[3] = {0, 0, 0};
        qed_emit<false>(K, m.O, E, B, u, S, 1.0, x, 1, 0, chi_out);
        const double det = (m.g - std::sqrt(1 + u[0] * u[0])) / K.dt;
        check("qed Q6: energy loss <d eps/dt> = P_alpha chi^2 g(chi) (" + fmt("chi = %g", chi) + "; Monte Carlo and deterministic)",
              std::abs(rate - want) < 4 * sig && std::abs(det / want - 1) < 3e-4,
              fmt("MC / ref - 1 = %.2e", rate / want - 1) + fmt(" (sigma %.1e)", sig / want) + fmt(", deterministic %.1e", det / want - 1));
    }
}

// ---------------------------------------------------------------- Q7 T-BMT
void tbmt_rhs(double qm, double a, const double E[3], const double B[3], const double u[3], const double S[3], double d[3]) {
    const double g = std::sqrt(1 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), b[3] = {u[0] / g, u[1] / g, u[2] / g};
    const double bB = b[0] * B[0] + b[1] * B[1] + b[2] * B[2];
    double O[3];
    for (int k = 0; k < 3; ++k) {
        const double bxE = b[(k + 1) % 3] * E[(k + 2) % 3] - b[(k + 2) % 3] * E[(k + 1) % 3];
        O[k] = (a + 1 / g) * B[k] - a * g / (g + 1) * bB * b[k] - (a + 1 / (g + 1)) * bxE;
    }
    for (int k = 0; k < 3; ++k) d[k] = qm * (S[(k + 1) % 3] * O[(k + 2) % 3] - S[(k + 2) % 3] * O[(k + 1) % 3]);   // (q/m) S x Omega
}
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
void q_tbmt(Comm& comm) {
    // (a) the exact rotation of qed_tbmt vs RK4 of dS/dt = (q/m) S x Omega (random E, B, u, both charges, a = 0 and 1.16e-3)
    double e = 0;
    unsigned long long k = 5;
    auto U = [&]() { return 2 * ion_uniform(k++) - 1; };
    for (int t = 0; t < 40; ++t) {
        const double E[3] = {U(), U(), U()}, B[3] = {U(), U(), U()}, u[3] = {5 * U(), 5 * U(), 5 * U()};
        const double qm = t % 2 ? -1.0 : 1.0 / 1836, a = t % 4 < 2 ? 1.15965218128e-3 : 0.0, dt = 0.3;
        double S[3] = {U(), U(), U()}, R[3] = {S[0], S[1], S[2]};
        qed_tbmt(qm, a, E, B, u, dt, S);
        const int n = 2000;
        const double h = dt / n;
        for (int i = 0; i < n; ++i) {
            double k1[3], k2[3], k3[3], k4[3], x[3];
            tbmt_rhs(qm, a, E, B, u, R, k1);
            for (int c = 0; c < 3; ++c) x[c] = R[c] + 0.5 * h * k1[c];
            tbmt_rhs(qm, a, E, B, u, x, k2);
            for (int c = 0; c < 3; ++c) x[c] = R[c] + 0.5 * h * k2[c];
            tbmt_rhs(qm, a, E, B, u, x, k3);
            for (int c = 0; c < 3; ++c) x[c] = R[c] + h * k3[c];
            tbmt_rhs(qm, a, E, B, u, x, k4);
            for (int c = 0; c < 3; ++c) R[c] += h / 6 * (k1[c] + 2 * k2[c] + 2 * k3[c] + k4[c]);
        }
        e = std::max({e, std::abs(S[0] - R[0]), std::abs(S[1] - R[1]), std::abs(S[2] - R[2])});
    }
    check("qed Q7a: T-BMT rotation (any E, B, u, q/m, a) = RK4 of dS/dt = (q/m) S x Omega", e < 1e-11, fmt("max error %.1e", e));
    // (b) electrons gyrating in a constant B (along the particle axis 0): S turns at (a + 1/gamma) B = (a gamma + 1) omega_c,
    //     the angle between S and u grows at a gamma omega_c (Boris phase error (omega_c dt)^3 / 12 per step aside)
    for (int rz = 0; rz < 2; ++rz) {
        const double a = 1.15965218128e-3, uperp = 3.0, g = std::sqrt(1 + uperp * uperp);
        std::string deck = rz ? "grid {\n geometry = rz\n modes = 2\n lambda0 = 1 um\n nz = 16\n nr = 64\n dz = 0.05 um\n dr = 0.2 um\n t_end = 1 ps\n}\n"
                                "species e {\n charge = -1\n mass = 1\n density = 1e-12\n rmax = 0.4 um\n ppc_r = 1\n ppc_theta = 8\n ux = 3\n spin = 0 1 0\n}\n"
                              : "grid {\n lambda0 = 1 um\n nx = 8\n ny = 8\n nz = 8\n dx = 0.05 um\n dy = 0.1 um\n dz = 0.1 um\n t_end = 1 ps\n}\n"
                                "species e {\n charge = -1\n mass = 1\n density = 1e-12\n uy = 3\n spin = 0 1 0\n}\n";
        deck += "numerics {\n pusher = boris\n}\noutput {\n dir = qed_tmp_out\n}\n";
        Config c = deck_config(deck, "qed_tmp.deck");
        Simulation s(c, comm);
        s.init();
        const double B0 = 0.8;                             // omega_c dt = 0.08: Boris phase 4e-5 per step
        // B along the particle axis 0 (Cart3D: x, B_x; RZ: z, B_z mode 0); u along particle axis 1 or 2, S along axis 1
        Field& F = rz ? s.rz->bz : s.fields->bx;
        H h(F);
        const int nk = rz ? 1 : s.dom.n[2] + 2 * s.dom.ng, k0 = rz ? 0 : -s.dom.ng, j0 = rz ? 0 : -s.dom.ng, nj = rz ? s.dom.n[1] : s.dom.n[1] + 2 * s.dom.ng;
        for (int kk = k0; kk < k0 + nk; ++kk) for (int j = j0; j < j0 + nj; ++j) for (int i = -s.dom.ng; i < s.dom.n[0] + s.dom.ng; ++i) h(i, j, kk) = B0;
        h.put(F);
        const int nsteps = 120;
        for (int n = 0; n < nsteps; ++n) s.step();
        Species& e = s.species[size_t(idx(s, "e"))];
        const std::vector<double> s1 = e.get(e.slot("s1")), s2 = e.get(e.slot("s2")), uy = e.get(Species::UY), uz = e.get(Species::UZ);
        const double T = nsteps * s.dom.dt(), th = (a + 1 / g) * B0 * T;     // electron: dS/dt = (B/gamma) x0_hat x S ... (a + 1/gamma)
        double err = 0, rel = 0;
        for (size_t p = 0; p < s1.size(); ++p) {
            // rotation about +axis 0 by th (q < 0: w = -(q/m) Omega = +(a + 1/gamma) B)
            err = std::max(err, std::abs(s1[p] - std::cos(th)) + std::abs(s2[p] - std::sin(th)));
            const double phs = std::atan2(s2[p], s1[p]), phu = std::atan2(uz[p], uy[p]);
            // u turns by the Boris angle 2 atan(B dt / (2 gamma)) per step: S - u = (a + 1/gamma) B T - that
            const double ub = nsteps * 2 * std::atan(B0 * s.dom.dt() / (2 * g));
            rel = std::max(rel, std::abs(std::remainder(phs - phu - (th - ub), 2 * M_PI)));
        }
        check(std::string("qed Q7b: T-BMT in a constant B (") + (rz ? "RZ" : "Cart3D") + "): S turns at (a gamma + 1) omega_c, S - u angle at a gamma omega_c",
              err < 1e-9 && rel < 1e-9 && !s1.empty(), fmt("S error %.1e", err) + fmt(", S-u phase error %.1e rad", rel) + fmt(" of a B T = %.4f", a * B0 * T));
    }
}

// ---------------------------------------------------------------- Q8 energy balance with a photon species, Q9 particle order
std::string mc_deck(bool photons, const std::string& extra = "") {
    // chi = 1 at gamma = 1000 in B_x: B = 1 / (gamma hw)
    return "grid {\n lambda0 = 0.8 um\n nx = 8\n ny = 8\n nz = 8\n dx = 0.02 um\n dy = 0.04 um\n dz = 0.04 um\n t_end = 1 ps\n}\n"
           "numerics {\n seed = 3\n}\n"
           "species e {\n charge = -1\n mass = 1\n density = 1e-10\n uy = 999.9995\n spin = 0 0 1\n}\n" +
           std::string(photons ? "species ph {\n kind = photon\n}\n" : "") +
           "qed {\n species = e\n" + (photons ? " photons = ph\n" : "") + extra + "}\noutput {\n dir = qed_tmp_out\n}\n";
}
void set_bx(Simulation& s, double B0) {
    H h(s.fields->bx);
    for (int k = -s.dom.ng; k < s.dom.n[2] + s.dom.ng; ++k) for (int j = -s.dom.ng; j < s.dom.n[1] + s.dom.ng; ++j)
        for (int i = -s.dom.ng; i < s.dom.n[0] + s.dom.ng; ++i) h(i, j, k) = B0;
    h.put(s.fields->bx);
}
void q_balance(Comm& comm) {
    const double B0 = 1 / (1000 * qed_hw(LAM));
    for (int photons = 0; photons < 2; ++photons) {
        Config c = deck_config(mc_deck(photons), "qed_tmp.deck");
        Simulation s(c, comm);
        s.init();
        set_bx(s, B0);
        Species& e = s.species[size_t(idx(s, "e"))];
        double k0 = e.kinetic_energy();                  // (the field work on these 1e-10 n_c electrons is negligible)
        comm.allreduce_sum(&k0, 1);
        for (int n = 0; n < 60; ++n) s.step();
        double v[3] = {e.kinetic_energy(), 0, 0};
        if (photons) { Species& ph = s.species[size_t(idx(s, "ph"))]; v[1] = ph.kinetic_energy(); v[2] = double(ph.size()); }
        comm.allreduce_sum(v, 3);
        const double lost = k0 - v[0];
        double tally[2];                                 // E_rad of all emitted photons, E into the photon species (all ranks)
        qed_totals(s, tally);
        const double bal = photons ? (v[0] + v[1] - k0) / k0 : (v[0] + tally[0] - k0) / k0;
        check(std::string("qed Q8: energy balance electrons + photons ") + (photons ? "(photon species)" : "(tallies only)") + ", chi = 1, 60 steps",
              std::abs(bal) < 1e-3 && lost > 0.2 * k0 && (!photons || (v[2] > 0 && std::abs(v[1] / tally[1] - 1) < 1e-12)),
              fmt("balance %.1e", bal) + fmt(", radiated %.2f of the energy", lost / k0) + (photons ? fmt(", %.0f photons", v[2]) : std::string()));
    }
    // Q9: the same run with the particles in reverse order: identical final momenta (sorted) and spins
    std::vector<double> g[2];
    for (int rev = 0; rev < 2; ++rev) {
        Config c = deck_config(mc_deck(false, " spin_state = pure\n"), "qed_tmp.deck");
        Simulation s(c, comm);
        s.init();
        set_bx(s, B0);
        Species& e = s.species[size_t(idx(s, "e"))];
        if (rev) {
            std::vector<int> perm(e.size());
            for (size_t p = 0; p < perm.size(); ++p) perm[p] = int(perm.size() - 1 - p);
            e.permute(perm);
        }
        for (int n = 0; n < 30; ++n) s.step();
        const std::vector<double> ux = e.get(Species::UX), uy = e.get(Species::UY), s2 = e.get(e.slot("s2"));
        for (size_t p = 0; p < ux.size(); ++p) g[rev].push_back(ux[p] * ux[p] + uy[p] * uy[p] + 1e-3 * s2[p]);
        std::sort(g[rev].begin(), g[rev].end());
    }
    check("qed Q9: random numbers per particle id: the particles in reverse order give identical results", g[0] == g[1] && !g[0].empty(),
          fmt("%.0f particles", double(g[0].size())));
}

// ---------------------------------------------------------------- Q10 deck checks
void q_deck() {
    auto bad = [](const std::string& extra_species, const std::string& qed, const std::string& numerics) {
        const std::string d = "grid {\n lambda0 = 0.8 um\n nx = 8\n ny = 4\n nz = 4\n dx = 0.02 um\n dy = 0.04 um\n dz = 0.04 um\n t_end = 1 fs\n}\n"
                              "numerics {\n" + numerics + "}\nspecies e {\n charge = -1\n mass = 1\n density = 1e-6\n}\n" + extra_species + qed +
                              "output {\n dir = qed_tmp_out\n}\n";
        try { deck_config(d, "qed_tmp.deck"); } catch (const std::exception&) { return true; }
        return false;
    };
    const bool a = bad("species ph {\n kind = photon\n}\n", "qed {\n species = ph\n}\n", "");
    const bool b = bad("species ph {\n kind = photon\n spin = 0 0 1\n}\n", "", "");
    const bool c = bad("species p {\n charge = 1\n mass = 1836\n density = 1e-6\n}\n", "qed {\n species = p\n}\n", "");
    const bool d = bad("", "qed {\n species = e\n photons = e\n}\n", "");
    const bool e = bad("", "qed {\n species = e\n}\n", " solver = irip\n mover = implicit\n shape = 2\n");
    const bool f = !bad("species ph {\n kind = photon\n}\n", "qed {\n species = e\n photons = ph\n spin_state = pure\n}\n", "");
    check("qed Q10: deck checks (photon emitter, photon spin, ions, photons = electrons, implicit mover rejected; valid deck accepted)",
          a && b && c && d && e && f, "");
}
}  // namespace

// Q11 (decision 117): the frame changes of the boosted frame. Spin: the Wigner rotation of two perpendicular boosts (a particle
// along y with gamma2, the frame moving along x with gamma1) turns the rest-frame spin about z by cos theta = (gamma1 + gamma2) /
// (1 + gamma1 gamma2); a momentum along the boost axis leaves it unchanged; there and back is the identity, |s| kept.
// Polarization: a photon along y polarized along x, seen from a frame moving at beta along x: e1' = (1/gamma, beta, 0)
// (transverse to k' = omega (-gamma beta, 1, 0) and unit); along z it stays; there and back is the identity.
void q_boost() {
    double wig = 0, col = 0, rt = 0, nrm = 0, pol = 0, prt = 0;
    const double gs[3] = {1.5, 3, 20};
    for (double g1 : gs)
        for (double g2 : gs) {
            const double b1 = std::sqrt(1 - 1 / (g1 * g1)), u2 = std::sqrt(g2 * g2 - 1), ct = (g1 + g2) / (1 + g1 * g2);
            const double u[3] = {0, u2, 0};
            double sx[3] = {1, 0, 0}, sy[3] = {0, 1, 0}, sz[3] = {0, 0, 1};
            qed_spin_boost(u, sx, b1); qed_spin_boost(u, sy, b1); qed_spin_boost(u, sz, b1);
            wig = std::max({wig, std::abs(sx[0] - ct), std::abs(sy[1] - ct), std::abs(std::abs(sx[1]) - std::sqrt(1 - ct * ct)),
                            std::abs(sz[2] - 1), std::abs(sx[2]), std::abs(sy[2])});
            const double ua[3] = {u2, 0, 0};          // along the boost axis
            double s[3] = {0.3, -0.5, 0.7};
            qed_spin_boost(ua, s, b1);
            col = std::max({col, std::abs(s[0] - 0.3), std::abs(s[1] + 0.5), std::abs(s[2] - 0.7)});
            const double k[3] = {0, 1, 0};
            double e[3] = {1, 0, 0}, ez[3] = {0, 0, 1};
            qed_pol_boost(k, e, b1); qed_pol_boost(k, ez, b1);
            pol = std::max({pol, std::abs(e[0] - 1 / g1), std::abs(e[1] - b1), std::abs(e[2]), std::abs(ez[2] - 1)});
        }
    unsigned long long key = 12345;
    for (int t = 0; t < 1000; ++t) {
        auto r = [&]() { return 2 * ion_uniform(key += 7) - 1; };
        const double u[3] = {50 * r(), 50 * r(), 50 * r()}, beta = 0.999 * r();
        double s[3] = {r(), r(), r()};
        const double n0 = std::sqrt(qed_dot(s, s));
        if (n0 > 1) for (double& x : s) x /= n0;
        const double s0[3] = {s[0], s[1], s[2]}, gb = 1 / std::sqrt(1 - beta * beta), g = std::sqrt(1 + qed_dot(u, u));
        qed_spin_boost(u, s, beta);
        nrm = std::max(nrm, std::abs(std::sqrt(qed_dot(s, s)) - std::sqrt(qed_dot(s0, s0))));
        const double up[3] = {gb * (u[0] - beta * g), u[1], u[2]};
        qed_spin_boost(up, s, -beta);
        rt = std::max({rt, std::abs(s[0] - s0[0]), std::abs(s[1] - s0[1]), std::abs(s[2] - s0[2])});
        // a photon: k random, e1 a random unit vector transverse to it
        const double k[3] = {u[0], u[1], u[2]}, om = std::sqrt(qed_dot(k, k));
        const double a[3] = {r(), r(), r()};
        double e[3] = {a[1] * k[2] - a[2] * k[1], a[2] * k[0] - a[0] * k[2], a[0] * k[1] - a[1] * k[0]};
        const double en = std::sqrt(qed_dot(e, e));
        for (double& x : e) x /= en;
        const double e0[3] = {e[0], e[1], e[2]};
        qed_pol_boost(k, e, beta);
        const double kp[3] = {gb * (k[0] - beta * om), k[1], k[2]};
        prt = std::max({prt, std::abs(qed_dot(e, kp)) / std::sqrt(qed_dot(kp, kp)), std::abs(qed_dot(e, e) - 1)});
        qed_pol_boost(kp, e, -beta);
        prt = std::max({prt, std::abs(e[0] - e0[0]), std::abs(e[1] - e0[1]), std::abs(e[2] - e0[2])});
    }
    check("qed Q11: boosted frame: spin Wigner rotation = (g1 + g2)/(1 + g1 g2), along the boost axis unchanged, there and back, |s| "
          "kept; photon polarization axis (1/gamma, beta, 0), transverse and unit, there and back",
          wig < 1e-12 && col < 1e-12 && rt < 1e-10 && nrm < 1e-12 && pol < 1e-12 && prt < 1e-10,
          fmt("Wigner %.1e", wig) + fmt(", collinear %.1e", col) + fmt(", round trip %.1e", rt) + fmt(", |s| %.1e", nrm) +
              fmt(", polarization %.1e", pol) + fmt(" / %.1e", prt));
}

void test_qed(Comm& comm) {
    q_boost();
    q_tables();
    q_spectrum();
    q_relax();
    q_energy_loss();
    q_tbmt(comm);
    q_balance(comm);
    q_deck();
}
