// Binary collisions (decision 94, docs/specs/collisions.md, src/collisions.*): conservation per collision, Nanbu's mean
// deflection, isotropization against the NRL rate (Cart3D, 2D, RZ), electron-ion temperature relaxation against the NRL
// (Spitzer) rate with equal and unequal weights, slowing down and angular scattering of a relativistic beam against the
// Rutherford rates, the Maxwell-Juettner equilibrium, independence of the particle order, deck checks. Only the collisions
// act (no push): every test calls collisions_apply in a loop.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <vector>
#include "common.hpp"
#include "atomic.hpp"
#include "collision_kernels.hpp"
#include "deck.hpp"
#include "impact.hpp"
#include "sim.hpp"

namespace {
const double KRE = 2 * M_PI * 2.8179403262e-15 / 0.8e-6;     // k0 r_e at lambda0 = 0.8 um
const double UM = 0.8 / (2 * M_PI);                          // the code unit of length in um
Config cdeck(const std::string& text) {
    const char* path = "coll_tmp.deck";
    if (FILE* f = std::fopen(path, "w")) { std::fputs(text.c_str(), f); std::fclose(f); }
    Deck dk = Deck::read(path);
    return Config::from_deck(dk);
}
std::string num(double x) { char b[64]; std::snprintf(b, sizeof b, "%.10g", x); return b; }
// a periodic box of nx x ny x nz cells of size h (code units); geometry 0 Cart3D, 1 cartesian2d, 2 rz (nz x ny cells)
std::string grid(int g, int nx, int ny, int nz, double h) {
    if (g == 2)
        return "grid {\n geometry = rz\n modes = 1\n lambda0 = 0.8 um\n nz = " + std::to_string(nx) + "\n nr = " + std::to_string(ny) +
               "\n dz = " + num(h * UM) + " um\n dr = " + num(2 * h * UM) + " um\n t_end = 1 fs\n}\n";
    return "grid {\n lambda0 = 0.8 um\n" + std::string(g == 1 ? " geometry = cartesian2d\n" : "") + " nx = " + std::to_string(nx) +
           "\n ny = " + std::to_string(ny) + "\n" + (g == 1 ? "" : " nz = " + std::to_string(nz) + "\n") + " dx = " + num(h * UM) +
           " um\n dy = " + num(2 * h * UM) + " um\n" + (g == 1 ? " dz = 1 um\n" : " dz = " + num(2 * h * UM) + " um\n") + " t_end = 1 fs\n}\n";
}
std::string ppc(int g, int a, int b, int c) {
    if (g == 2) return " ppc_z = " + std::to_string(a) + "\n ppc_r = " + std::to_string(b) + "\n ppc_theta = " + std::to_string(c) + "\n";
    return " ppc_x = " + std::to_string(a) + "\n ppc_y = " + std::to_string(b) + "\n" + (g == 1 ? "" : " ppc_z = " + std::to_string(c) + "\n");
}
std::string block(const std::string& name, const std::string& a, const std::string& b, double lnL, const std::string& extra = "") {
    return "collisions " + name + " {\n species1 = " + a + "\n species2 = " + b + "\n coulomb_log = " + num(lnL) + "\n" + extra + "}\n";
}
int idx(Simulation& s, const char* name) {
    for (size_t i = 0; i < s.cfg.species.size(); ++i)
        if (s.cfg.species[i].name == name) return int(i);
    return -1;
}
// T_a = m <u_a^2 / gamma> (m c^2), weighted, over the ranks; T[3] = sum w
std::vector<double> temps(Simulation& s, int sp) {
    Species& S = s.species[size_t(sp)];
    const auto X = S.get(Species::UX), Y = S.get(Species::UY), Z = S.get(Species::UZ), W = S.get(Species::W);
    std::vector<double> t(4, 0.0);
    for (size_t p = 0; p < W.size(); ++p) {
        const double g = std::sqrt(1 + X[p] * X[p] + Y[p] * Y[p] + Z[p] * Z[p]);
        t[0] += W[p] * X[p] * X[p] / g; t[1] += W[p] * Y[p] * Y[p] / g; t[2] += W[p] * Z[p] * Z[p] / g; t[3] += W[p];
    }
    s.comm.allreduce_sum(t.data(), 4);
    for (int a = 0; a < 3; ++a) t[size_t(a)] *= S.m / t[3];
    return t;
}
// sum w m u (3), sum w m gamma over all species
std::vector<double> totals(Simulation& s) {
    std::vector<double> v(4, 0.0);
    for (Species& S : s.species) {
        const auto X = S.get(Species::UX), Y = S.get(Species::UY), Z = S.get(Species::UZ), W = S.get(Species::W);
        for (size_t p = 0; p < W.size(); ++p) {
            v[0] += W[p] * S.m * X[p]; v[1] += W[p] * S.m * Y[p]; v[2] += W[p] * S.m * Z[p];
            v[3] += W[p] * S.m * std::sqrt(1 + X[p] * X[p] + Y[p] * Y[p] + Z[p] * Z[p]);
        }
    }
    s.comm.allreduce_sum(v.data(), 4);
    return v;
}
void step(Simulation& s) { ++s.istep; collisions_apply(s); }
// RK4 for a system y' = f(y)
std::vector<double> rk4(std::vector<double> y, const std::function<std::vector<double>(const std::vector<double>&)>& f, double T, int n) {
    const double h = T / n;
    for (int i = 0; i < n; ++i) {
        auto k1 = f(y), t = y;
        for (size_t q = 0; q < y.size(); ++q) t[q] = y[q] + 0.5 * h * k1[q];
        auto k2 = f(t);
        for (size_t q = 0; q < y.size(); ++q) t[q] = y[q] + 0.5 * h * k2[q];
        auto k3 = f(t);
        for (size_t q = 0; q < y.size(); ++q) t[q] = y[q] + h * k3[q];
        auto k4 = f(t);
        for (size_t q = 0; q < y.size(); ++q) y[q] += h / 6 * (k1[q] + 2 * k2[q] + 2 * k3[q] + k4[q]);
    }
    return y;
}
// NRL formulary, temperature isotropy: dT_perp/dt = -nu (T_perp - T_par), dT_par/dt = 2 nu (T_perp - T_par),
// nu = 2 sqrt(pi) n r_e^2 c lnL / theta_par^(3/2) A^-2 [-3 + (A + 3) atan(sqrt A) / sqrt A], A = T_perp / T_par - 1
double nrl_iso(double n, double lnL, double tpar, double tperp) {
    const double A = tperp / tpar - 1;
    double f;
    if (std::fabs(A) < 1e-4) f = 4.0 / 15 - 8.0 / 35 * A;
    else if (A > 0) f = (-3 + (A + 3) * std::atan(std::sqrt(A)) / std::sqrt(A)) / (A * A);
    else f = (-3 + (A + 3) * std::atanh(std::sqrt(-A)) / std::sqrt(-A)) / (A * A);
    return KRE * n * lnL / (2 * std::sqrt(M_PI) * std::pow(tpar, 1.5)) * f;
}

// C1: one collision conserves the momentum and the energy of the pair to round-off (equal weights), any s, both angle laws
void conservation_unit() {
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> U(0, 1);
    CollParams P;
    P.kre = KRE; P.bq = 0.5 * 3.8615926796e-13 * 2 * M_PI / 0.8e-6; P.lt = std::pow(3 * KRE, -1.0 / 3);
    double dp = 0, de = 0;
    long n = 0;
    for (int t = 0; t < 20000; ++t) {
        const double m1 = 1, m2 = t % 3 == 0 ? 1.0 : (t % 3 == 1 ? 1836.15 : 4 * 1836.15);
        const double q1 = -1, q2 = t % 3 == 0 ? -1.0 : double(1 + t % 7);
        const double sc = std::pow(10.0, -3 + 5 * U(rng));          // |u| from 1e-3 to 1e2
        double u1[3], u2[3];
        for (int a = 0; a < 3; ++a) { u1[a] = sc * (2 * U(rng) - 1); u2[a] = sc * (2 * U(rng) - 1) / std::sqrt(m2); }
        P.lnL = t % 2 ? 10.0 : 0.0; P.angle = t % 4 < 2 ? 0 : 1;
        const double D = std::pow(10.0, 6 + 8 * U(rng));             // s from ~1e-6 to isotropic
        auto mom = [&](const double* a, const double* b, double* o) {
            const double g1 = std::sqrt(1 + a[0] * a[0] + a[1] * a[1] + a[2] * a[2]), g2 = std::sqrt(1 + b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
            for (int k = 0; k < 3; ++k) o[k] = m1 * a[k] + m2 * b[k];
            o[3] = m1 * g1 + m2 * g2;
        };
        double before[4], after[4];
        mom(u1, u2, before);
        double lnL;
        const double s = coll_pair(u1, u2, m1, m2, q1, q2, D, 1.0, 1.0, 1e-2, P, U(rng), U(rng), true, true, lnL);
        if (!(s > 0)) continue;
        mom(u1, u2, after);
        const double pn = std::sqrt(before[0] * before[0] + before[1] * before[1] + before[2] * before[2]) + before[3];
        for (int k = 0; k < 3; ++k) dp = std::max(dp, std::fabs(after[k] - before[k]) / pn);
        de = std::max(de, std::fabs(after[3] - before[3]) / before[3]);
        ++n;
    }
    check("C1 one collision conserves momentum and energy (20000 random pairs, e-e / e-p / e-He, |u| 1e-3 .. 1e2, s 1e-6 .. "
          "isotropic)", dp < 1e-13 && de < 1e-13, fmt("max |dP| / (|P| + E) %.2e, max |dE| / E %.2e", dp, de) + " (" + std::to_string(n) + " pairs)");
}

// C2: Nanbu's distribution has <cos chi> = e^-s (the definition of s); Perez's fits for A reproduce it to < 1e-3 (the
// fit of Smilei 5.1 is checked too)
void nanbu_mean() {
    double worst[2] = {0, 0};
    std::string info;
    for (int law = 0; law < 2; ++law)
        for (double s : {0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0}) {
            const int n = 400000;
            double m = 0;
            for (int i = 0; i < n; ++i) {
                double c, sn;
                coll_angle(s, (i + 0.5) / n, law, c, sn);              // midpoint rule in U: the exact mean to O(1/n^2)
                m += c;
            }
            m /= n;
            const double e = std::fabs(m - std::exp(-s));
            worst[law] = std::max(worst[law], e);
            if (law == 0 && (s == 0.2 || s == 2.0)) info += fmt("s = %g: %.5f", s, m) + fmt(" vs %.5f; ", std::exp(-s));
        }
    check("C2 Nanbu: <cos chi> = exp(-s) for s = 0.01 .. 4 (Perez's fits for A)", worst[0] < 1.5e-3,
          info + fmt("max |<cos> - e^-s| %.1e (Smilei 5.1 fit: %.1e)", worst[0], worst[1]));
}

// C3: isotropization of an electron plasma with T_perp = 2 T_par (theta_par = 1e-3, n = 0.25 n_c, lnL = 10) against the NRL
// rate (ODE of the bi-Maxwellian; it holds while the plasma stays bi-Maxwellian: compared at t = 6000, 1.3 e-folds). The
// binary collisions converge to it as dt -> 0 (a pair exchanges 1 - e^-s instead of s; s ~ 0.1 - 0.2 per step at dt = 100).
// Cart3D at dt = 100 and 25; 2D and RZ (cell volumes 2 pi r dr dz) at dt = 100 against Cart3D; RZ also in its radial halves.
// Returns the relative error of the decrease of T_perp - T_par at t = 6000 against NRL; din, dout: RZ halves (same measure).
double iso_run(int g, double h, double& din, double& dout) {
    const double n = 0.25, lnL = 10, tpar = 1e-3, tperp = 2e-3, tend = 6000;
    const int steps = int(tend / h + 0.5);
    std::string sp = "species e {\n charge = -1\n mass = 1\n mobile = false\n density = " + num(n) + "\n" +
                     ppc(g, g == 1 ? 32 : 16, g == 1 ? 32 : 8, g == 2 ? 32 : 8);
    if (g == 2) sp += " uth_z = " + num(std::sqrt(tpar)) + "\n uth_x = " + num(std::sqrt(tperp)) + "\n uth_y = " + num(std::sqrt(tperp)) + "\n}\n";
    else sp += " uth_x = " + num(std::sqrt(tpar)) + "\n uth_y = " + num(std::sqrt(tperp)) + "\n uth_z = " + num(std::sqrt(tperp)) + "\n}\n";
    const std::string text = grid(g, g == 1 ? 16 : 8, g == 1 ? 8 : (g == 2 ? 8 : 4), 4, h) + sp + block("ee", "e", "e", lnL) +
                             "output {\n dir = coll_tmp_out\n}\n";
    Comm comm;
    Simulation s(cdeck(text), comm);
    s.init();
    const int e = idx(s, "e");
    const auto t0 = temps(s, e);
    const double par0 = t0[0], perp0 = 0.5 * (t0[1] + t0[2]);
    double tp = par0, tq = perp0;
    for (int k = 1; k <= steps; ++k) step(s);
    const auto f = [&](const std::vector<double>& y) {
        const double nu = nrl_iso(n, lnL, y[0], y[1]);
        return std::vector<double>{2 * nu * (y[1] - y[0]), -nu * (y[1] - y[0])};
    };
    const auto y = rk4({tp, tq}, f, tend, 4000);
    const double dth = (y[1] - y[0]) / (perp0 - par0);
    const auto t = temps(s, e);
    const double d = (0.5 * (t[1] + t[2]) - t[0]) / (perp0 - par0);
    if (g == 2) {                                     // the radial halves (cells of very different volumes)
        Species& S = s.species[size_t(e)];
        const auto X = S.get(Species::UX), Y = S.get(Species::UY), Z = S.get(Species::UZ), W = S.get(Species::W);
        const auto yy = S.get(Species::Y), zz = S.get(Species::Z);
        double a[2][3] = {{0, 0, 0}, {0, 0, 0}};
        for (size_t p = 0; p < W.size(); ++p) {
            const int o = std::sqrt(yy[p] * yy[p] + zz[p] * zz[p]) < 8 * h ? 0 : 1;
            a[o][0] += W[p] * X[p] * X[p]; a[o][1] += W[p] * 0.5 * (Y[p] * Y[p] + Z[p] * Z[p]); a[o][2] += W[p];
        }
        din = (1 - (a[0][1] - a[0][0]) / a[0][2] / (perp0 - par0)) / (1 - dth) - 1;
        dout = (1 - (a[1][1] - a[1][0]) / a[1][2] / (perp0 - par0)) / (1 - dth) - 1;
    }
    return (1 - d) / (1 - dth) - 1;
}
void isotropization() {
    double din = 0, dout = 0;
    const double c100 = iso_run(0, 100, din, dout), c25 = iso_run(0, 25, din, dout);
    check("C3 isotropization vs NRL (Cart3D, decrease of T_perp - T_par after 1.3 e-folds): within the bi-Maxwellian theory (the deficit does not depend on dt)",
          std::fabs(c25) < 0.08 && std::fabs(c100) < 0.08,
          fmt("relative error at dt = 100: %.3f, at dt = 25: %.3f", c100, c25));
    const double d2 = iso_run(1, 100, din, dout), rz = iso_run(2, 100, din, dout);
    check("C3b the same in 2D and RZ (dt = 100) as in Cart3D", std::fabs(d2 - c100) < 0.04 && std::fabs(rz - c100) < 0.04,
          fmt("relative error vs NRL: 2D %.3f, RZ %.3f", d2, rz) + fmt(" (Cart3D %.3f)", c100));
    check("C3c RZ: the same at r < 4 dr and r > 4 dr (cells of different volume)", std::fabs(din - dout) < 0.05,
          fmt("relative error vs NRL: inner %.3f, outer %.3f", din, dout));
}

// C4: electron-ion temperature relaxation (e-e, i-i, e-i collisions, m_i = 4, Z = 1, theta_e = 1e-3, theta_i = 2e-4, n = 0.2 n_c,
// lnL = 10; lnL = 1000 within each species keeps both Maxwellian, as in Smilei's benchmark thermalisation_ei) against the NRL
// energy-exchange rate (Spitzer) over 6 e-folds, at dt = 100 and 25 (convergence), with equal weights and with ions of 4x the
// electron weight; the energy: exact with equal weights, on average with unequal ones. Returns the largest deviation of
// (T_e - T_i) / initial from NRL; rel: |energy change| / exchanged energy.
double relax_run(double h, int v, double& rel) {
    const double n = 0.2, lnL = 10, mi = 4, te = 1e-3, ti = 2e-4, tend = 12000;
    const int steps = int(tend / h + 0.5);
    const std::string text = grid(0, 8, 4, 4, h) +
        "species e {\n charge = -1\n mass = 1\n mobile = false\n density = " + num(n) + "\n" + ppc(0, 8, 8, 4) + " uth_x = " + num(std::sqrt(te)) +
        "\n uth_y = " + num(std::sqrt(te)) + "\n uth_z = " + num(std::sqrt(te)) + "\n}\n"
        "species i {\n charge = 1\n mass = " + num(mi) + "\n mobile = false\n density = " + num(n) + "\n" + (v ? ppc(0, 4, 4, 4) : ppc(0, 8, 8, 4)) +
        " uth_x = " + num(std::sqrt(ti / mi)) + "\n uth_y = " + num(std::sqrt(ti / mi)) + "\n uth_z = " + num(std::sqrt(ti / mi)) + "\n}\n" +
        block("ee", "e", "e", 1000) + block("ei", "e", "i", lnL) + block("ii", "i", "i", 1000) + "output {\n dir = coll_tmp_out\n}\n";
    Comm comm;
    Simulation s(cdeck(text), comm);
    s.init();
    const int e = idx(s, "e"), i = idx(s, "i");
    auto T = [&](int sp) { auto t = temps(s, sp); return (t[0] + t[1] + t[2]) / 3; };
    double ye = T(e), yi = T(i);
    const double d0 = ye - yi;
    const auto E0 = totals(s);
    double worst = 0;
    const int every = steps / 12;
    for (int k = 1; k <= steps; ++k) {
        step(s);
        const auto f = [&](const std::vector<double>& y) {
            const double r = 8 * std::sqrt(2 * M_PI) / 3 * KRE / (4 * M_PI) * n * lnL * std::sqrt(mi) / std::pow(y[1] + mi * y[0], 1.5);
            return std::vector<double>{-r * (y[0] - y[1]), r * (y[0] - y[1])};
        };
        const auto y = rk4({ye, yi}, f, h, 20);
        ye = y[0]; yi = y[1];
        if (k % every == 0) worst = std::max(worst, std::fabs((T(e) - T(i)) / d0 - (ye - yi) / d0));
    }
    const auto E1 = totals(s);
    const double wsum = temps(s, e)[3], exch = 1.5 * wsum * (d0 - (T(e) - T(i))) / 2;   // 3/2 n V (T_e0 - T_e), sum w m c^2 units
    rel = std::fabs(E1[3] - E0[3]) / exch;
    return worst;
}
void relaxation() {
    double r1, r2, r3;
    const double a = relax_run(100, 0, r1), b = relax_run(25, 0, r2), c = relax_run(25, 1, r3);
    check("C4 e-i relaxation vs NRL/Spitzer (m_i = 4, equal weights, 6 e-folds): converges as dt -> 0", b < 0.07 && b < a,
          fmt("max |deviation| of (T_e - T_i) / initial: dt = 100: %.4f, dt = 25: %.4f", a, b) + fmt("; |energy change| / exchanged %.1e", r2));
    check("C4b the same with ions of 4x the electron weight (dt = 25)", c < 0.07 && r3 < 0.05,
          fmt("max |deviation| %.4f; |energy change| / exchanged %.1e (exact only for equal weights)", c, r3));
    check("C4c equal weights conserve the energy exactly", r1 < 1e-10 && r2 < 1e-10, fmt("|energy change| / exchanged %.1e, %.1e", r1, r2));
}

// C5: a relativistic electron beam (gamma 3) slowing down on cold electrons: d gamma / dt = -n k0 r_e lnL / beta (the
// Rutherford/Coulomb-log stopping, d E / dx = -4 pi n r_e^2 m c^2 lnL / beta^2); beam weight 1e-4 of the background (unequal
// weights: the background stays cold); and angular scattering of a gamma = 2 beam on heavy ions (Z = 1, m = 1e9):
// <cos theta> = exp(-nu t), nu = n Z^2 k0 r_e lnL / (gamma^2 beta^3)
void beam() {
    {
        const double h = 4e-4, n = 2e7, lnL = 10, g0 = 3;
        const std::string text = grid(0, 3, 3, 3, h) +
            "species b {\n charge = -1\n mass = 1\n mobile = false\n density = " + num(n * 1e-4) + "\n" + ppc(0, 16, 16, 4) + " ux = " + num(std::sqrt(g0 * g0 - 1)) + "\n}\n"
            "species bg {\n charge = -1\n mass = 1\n mobile = false\n density = " + num(n) + "\n" + ppc(0, 16, 16, 4) + "}\n" +
            block("b-bg", "b", "bg", lnL) + "output {\n dir = coll_tmp_out\n}\n";
        Comm comm;
        Simulation s(cdeck(text), comm);
        s.init();
        const int b = idx(s, "b"), bg = idx(s, "bg");
        auto mean_gamma = [&](int sp) {
            Species& S = s.species[size_t(sp)];
            const auto X = S.get(Species::UX), Y = S.get(Species::UY), Z = S.get(Species::UZ);
            double m = 0;
            for (size_t p = 0; p < X.size(); ++p) m += std::sqrt(1 + X[p] * X[p] + Y[p] * Y[p] + Z[p] * Z[p]);
            return m / double(X.size());
        };
        double g = g0, worst = 0;
        const int steps = 500;
        for (int k = 1; k <= steps; ++k) {
            step(s);
            g = rk4({g}, [&](const std::vector<double>& y) { return std::vector<double>{-n * KRE * lnL / std::sqrt(1 - 1 / (y[0] * y[0]))}; }, h, 4)[0];
            if (k % 50 == 0) worst = std::max(worst, std::fabs(mean_gamma(b) - g) / (g0 - g));
        }
        check("C5 beam slowing down on cold electrons vs d gamma / dt = -n k0 r_e lnL / beta (gamma 3 -> ~2)", worst < 0.01,
              fmt("<gamma> %.5f vs %.5f", mean_gamma(b), g) + fmt(", max |error| / (gamma_0 - gamma) %.4f; background <gamma> - 1 = %.1e", worst, mean_gamma(bg) - 1));
    }
    {
        const double h = 0.1, ni = 6e5, lnL = 10, g0 = 2, nu = ni * KRE * lnL / (g0 * g0 * std::pow(std::sqrt(1 - 1 / (g0 * g0)), 3));
        const std::string text = grid(0, 3, 3, 3, h) +
            "species b {\n charge = -1\n mass = 1\n mobile = false\n density = 1\n" + ppc(0, 16, 16, 16) + " ux = " + num(std::sqrt(g0 * g0 - 1)) + "\n}\n"
            "species i {\n charge = 1\n mass = 1e9\n mobile = false\n density = " + num(ni) + "\n" + ppc(0, 4, 4, 4) + "}\n" +
            block("bi", "b", "i", lnL) + "output {\n dir = coll_tmp_out\n}\n";
        Comm comm;
        Simulation s(cdeck(text), comm);
        s.init();
        const int b = idx(s, "b");
        auto mean_cos = [&]() {
            Species& S = s.species[size_t(b)];
            const auto X = S.get(Species::UX), Y = S.get(Species::UY), Z = S.get(Species::UZ);
            double m = 0;
            for (size_t p = 0; p < X.size(); ++p) m += X[p] / std::sqrt(X[p] * X[p] + Y[p] * Y[p] + Z[p] * Z[p]);
            return m / double(X.size());
        };
        const int steps = 400;
        double worst = 0;
        for (int k = 1; k <= steps; ++k) {
            step(s);
            if (k % 50 == 0) worst = std::max(worst, std::fabs(mean_cos() - std::exp(-nu * h * k)));
        }
        check("C5b angular scattering of a gamma = 2 beam on heavy ions: <cos theta> = exp(-nu t)", worst < 0.01,
              fmt("<cos> %.4f vs %.4f", mean_cos(), std::exp(-nu * h * steps)) + fmt(" at nu t = %.2f; max |error| %.4f (32768 electrons)", nu * h * steps, worst));
    }
}

// Maxwell-Juettner: the cdf of gamma for theta, on a grid
double mj_cdf(double gam, double theta) {
    auto f = [&](double g) { return g * std::sqrt(g * g - 1) * std::exp(-(g - 1) / theta); };
    const int n = 4000;
    const double top = 1 + 60 * theta;
    double tot = 0, part = 0;
    for (int i = 0; i < n; ++i) {
        const double a = 1 + (top - 1) * i / n, b = 1 + (top - 1) * (i + 1) / n, v = 0.5 * (f(a) + f(b)) * (b - a);
        tot += v;
        if (b <= gam) part += v;
        else if (a < gam) part += v * (gam - a) / (b - a);
    }
    return part / tot;
}
double mj_mean_gamma(double theta) {
    auto f = [&](double g) { return g * std::sqrt(g * g - 1) * std::exp(-(g - 1) / theta); };
    double a = 0, b = 0;
    const double top = 1 + 60 * theta;
    for (int i = 0; i < 20000; ++i) {
        const double g = 1 + (top - 1) * (i + 0.5) / 20000;
        a += g * f(g); b += f(g);
    }
    return a / b;
}
double ks(std::vector<double> g, double theta) {
    std::sort(g.begin(), g.end());
    double d = 0;
    const double n = double(g.size());
    for (size_t i = 0; i < g.size(); i += 7) {
        const double F = mj_cdf(g[i], theta);
        d = std::max(d, std::max(std::fabs(F - i / n), std::fabs(F - (i + 1) / n)));
    }
    return d;
}

// C6: the relativistic equilibrium: a Maxwell-Juettner plasma (theta = 1) stays one, and a Gaussian in u relaxes to the
// Maxwell-Juettner distribution of the same energy (KS distance of the gamma distribution, 16384 electrons, sum s ~ 25)
void juttner() {
    for (int v = 0; v < 2; ++v) {
        const double h = 1.4e-4, n = 1.6e8, theta = 1;
        const std::string text = grid(0, 3, 3, 3, h) + "species e {\n charge = -1\n mass = 1\n mobile = false\n density = " + num(n) + "\n" + ppc(0, 16, 16, 8) +
            (v ? " uth_x = 1.2\n uth_y = 1.2\n uth_z = 1.2\n" : "") + "}\n" + block("ee", "e", "e", 10) + "output {\n dir = coll_tmp_out\n}\n";
        Comm comm;
        Simulation s(cdeck(text), comm);
        s.init();
        Species& S = s.species[size_t(idx(s, "e"))];
        if (v == 0) {                                 // Sobol's sampling of the Maxwell-Juettner momentum
            std::mt19937_64 rng(11);
            std::uniform_real_distribution<double> U(0, 1);
            std::vector<double> X(S.size()), Y(S.size()), Z(S.size());
            for (size_t p = 0; p < S.size(); ++p) {
                double u;
                for (;;) {
                    const double x1 = U(rng), x2 = U(rng), x3 = U(rng), x4 = U(rng);
                    u = -theta * std::log(x1 * x2 * x3);
                    const double eta = -theta * std::log(x1 * x2 * x3 * x4);
                    if (eta * eta - u * u > 1) break;
                }
                const double c = 2 * U(rng) - 1, ph = 2 * M_PI * U(rng), sn = std::sqrt(1 - c * c);
                X[p] = u * sn * std::cos(ph); Y[p] = u * sn * std::sin(ph); Z[p] = u * c;
            }
            S.set(Species::UX, X); S.set(Species::UY, Y); S.set(Species::UZ, Z);
        }
        auto gammas = [&]() {
            const auto X = S.get(Species::UX), Y = S.get(Species::UY), Z = S.get(Species::UZ);
            std::vector<double> g(X.size());
            for (size_t p = 0; p < X.size(); ++p) g[p] = std::sqrt(1 + X[p] * X[p] + Y[p] * Y[p] + Z[p] * Z[p]);
            return g;
        };
        auto g0 = gammas();
        double mg = 0;
        for (double x : g0) mg += x;
        mg /= double(g0.size());
        double th = theta;                            // the temperature of the same mean energy
        if (v) {
            double lo = 0.05, hi = 5;
            for (int it = 0; it < 60; ++it) { th = 0.5 * (lo + hi); (mj_mean_gamma(th) < mg ? lo : hi) = th; }
        }
        const double k0s = ks(g0, th);
        for (int k = 0; k < (v ? 1500 : 500); ++k) step(s);
        const double k1 = ks(gammas(), th), crit = 1.36 / std::sqrt(double(g0.size()));
        const CollisionTotals T = collisions_totals(s, 0);
        check(std::string("C6 Maxwell-Juettner ") + (v ? "reached from a Gaussian in u (theta from the energy)" : "kept (theta = 1)"),
              k1 < crit, fmt("KS distance of gamma before %.4f, after %.4f", k0s, k1) + fmt(" (95 %% level %.4f); theta %.4f", crit, th) +
                             fmt(", sum s per particle %.1f", T.s / double(g0.size()) * 2));
    }
}

// C7: results do not depend on the particle order (proxy for threads and ranks): the same collisions with the particles
// stored in reverse order give the same momenta per id, bit for bit; and one step conserves sum w m u, sum w m gamma
void order_and_totals() {
    const std::string text = grid(0, 4, 4, 3, 0.1) +
        "species e {\n charge = -1\n mass = 1\n mobile = false\n density = 200\n" + ppc(0, 4, 4, 5) + " uth_x = 0.05\n uth_y = 0.03\n uth_z = 0.03\n ux = 0.02\n}\n"
        "species i {\n charge = 1\n mass = 4\n mobile = false\n density = 200\n" + ppc(0, 4, 4, 5) + " uth_x = 0.01\n uth_y = 0.01\n uth_z = 0.01\n}\n" +
        block("ee", "e", "e", 0) + block("ei", "e", "i", 0) + "output {\n dir = coll_tmp_out\n}\n";
    std::map<long long, std::vector<double>> res[2];
    double dP = 0, dE = 0;
    for (int v = 0; v < 2; ++v) {
        Comm comm;
        Simulation s(cdeck(text), comm);
        s.init();
        if (v)
            for (Species& S : s.species) {
                std::vector<int> perm(S.size());
                for (size_t p = 0; p < S.size(); ++p) perm[p] = int(S.size() - 1 - p);
                S.permute(perm);
            }
        const auto A = totals(s);
        for (int k = 0; k < 3; ++k) step(s);
        const auto B = totals(s);
        if (!v) {
            dP = std::sqrt((B[0] - A[0]) * (B[0] - A[0]) + (B[1] - A[1]) * (B[1] - A[1]) + (B[2] - A[2]) * (B[2] - A[2])) / std::fabs(A[0]);
            dE = std::fabs(B[3] - A[3]) / A[3];
        }
        for (Species& S : s.species) {
            const auto ID = S.get_as<long long>(S.slot("id"));
            const auto X = S.get(Species::UX), Y = S.get(Species::UY), Z = S.get(Species::UZ);
            for (size_t p = 0; p < ID.size(); ++p) res[v][ID[p]] = {X[p], Y[p], Z[p]};
        }
    }
    bool same = res[0].size() == res[1].size() && res[0].size() > 100;
    for (auto& kv : res[0]) same = same && res[1].count(kv.first) && res[1][kv.first] == kv.second;
    check("C7 independent of the particle order (reversed storage: momenta per id bit-identical; automatic Coulomb log)", same,
          std::to_string(res[0].size()) + " particles");
    check("C7b three steps (e-e, e-i, equal weights in each block) conserve momentum and energy", dP < 1e-12 && dE < 1e-13,
          fmt("|dP| / |P_x| %.1e, |dE| / E %.1e", dP, dE));
}

// C9: the automatic Coulomb log of an electron beam on cold protons (every pair the same relative speed; the cold protons
// make the Debye length 0, so b_max is the ion-sphere radius a_i) against the Lee-More form evaluated in SI:
// lnL = max(2, 1/2 ln(1 + a_i^2 / b_min^2)), b_min = max(e^2 / (4 pi eps0 mu v^2), hbar / (2 mu v)) (quantum_bmin = h: h / (2 mu v))
void coulomb_log_auto() {
    const double e = 1.602176634e-19, eps0 = 8.8541878128e-12, hbar = 1.054571817e-34, me = 9.1093837015e-31, c = 299792458.0;
    const double mp = 1836.15267, N = 1.0;
    const double omega = 2 * M_PI * c / 0.8e-6, n = N * eps0 * me * omega * omega / (e * e), ai = std::cbrt(3 / (4 * M_PI * n));
    double worst = 0;
    std::string info;
    for (const double u : {0.01, 0.03}) {
        for (const int hq : {0, 1}) {
            const std::string text = grid(0, 3, 3, 3, 0.05) +
                "species e {\n charge = -1\n mass = 1\n mobile = false\n density = " + num(N) + "\n" + ppc(0, 2, 2, 2) + " ux = " + num(u) + "\n}\n"
                "species p {\n charge = 1\n mass = " + num(mp) + "\n mobile = false\n density = " + num(N) + "\n" + ppc(0, 2, 2, 2) + "}\n" +
                block("ep", "e", "p", 0, hq ? " quantum_bmin = h\n" : "") + "output {\n dir = coll_tmp_out\n}\n";
            Comm comm;
            Simulation s(cdeck(text), comm);
            s.init();
            step(s);
            const CollisionTotals T = collisions_totals(s, 0);
            const double v = u / std::sqrt(1 + u * u) * c, mu = me * mp / (1 + mp);
            const double bcl = e * e / (4 * M_PI * eps0 * mu * v * v), bqu = (hq ? 2 * M_PI : 1.0) * hbar / (2 * mu * v);
            const double ref = std::max(2.0, 0.5 * std::log(1 + ai * ai / std::pow(std::max(bcl, bqu), 2)));
            const double got = T.pairs > 0 ? T.lnL / T.pairs : 0;
            worst = std::max(worst, std::fabs(got - ref));
            info += fmt(" u %.2f", u) + (hq ? " h" : " hbar") + fmt(": %.4f vs %.4f;", got, ref);
        }
    }
    check("C9 automatic Coulomb log (beam on cold protons) vs the Lee-More form in SI, hbar / 2p* and Smilei's h / 2p*",
          worst < 3e-3, fmt("max |difference| %.1e;", worst) + info);
}

void deck_checks() {
    const std::string base = grid(0, 4, 4, 4, 0.1) + "species e {\n charge = -1\n mass = 1\n mobile = false\n density = 1\n}\n"
                             "output {\n dir = coll_tmp_out\n}\n";
    auto fails = [&](const std::string& extra) {
        try { cdeck(base + extra); } catch (const std::exception&) { return true; }
        return false;
    };
    const bool ok = fails(block("x", "e", "q", 0)) && fails(block("x", "e", "e", -1)) &&
                    fails(block("x", "e", "e", 0, " every = 0\n")) && fails(block("x", "e", "e", 0, " angle = isotropic\n")) &&
                    fails(block("x", "e", "e", 0, " quantum_bmin = planck\n")) &&
                    !fails(block("x", "e", "e", 0, " angle = smilei5\n every = 2\n low_t_cap = false\n quantum_bmin = h\n"));
    check("C8 deck checks (unknown species, coulomb_log < 0, every < 1, angle, quantum_bmin)", ok, "");
}
// I1: impact ionization of neutral carbon by a hot drifting electron beam (T = 1 keV, u_x = 0.2, n_e = 10 n_C), early times
// (<Z> ~ 0.05, the electrons keep their energy): <Z>(t) against the rate equations with R_k = n_e <sigma_k v> integrated over
// the electrons with the cross-section formula (not the tables); the charge is conserved exactly; and the RBEB cross-section
// of hydrogen at 100 eV against the measured 0.57e-16 cm^2 (Shah et al., J. Phys. B 20, 3501 (1987))
void impact() {
    const double ne = 1e3, h = 0.1, te = 1.0 / 511, ux = 0.2;
    const int steps = 40;
    const Element* C = find_element("C");
    const Element* Hy = find_element("H");
    {
        const double sig = impact_cross_section(1, Hy->ip_eV, KRE, 0, 100.0 / 510998.95) * 4 * M_PI * 2.8179403262e-15 / (KRE / 2.8179403262e-15) * 1e4;
        check("I0 RBEB cross-section of H at 100 eV vs the measured 0.57e-16 cm^2 (Shah et al. 1987)", std::fabs(sig / 0.57e-16 - 1) < 0.2,
              fmt("%.3e cm^2", sig));
    }
    const std::string text = grid(0, 4, 3, 3, h) +
        "species e {\n charge = -1\n mass = 1\n mobile = false\n density = " + num(ne) + "\n" + ppc(0, 16, 16, 16) + " ux = " + num(ux) +
        "\n uth_x = " + num(std::sqrt(te)) + "\n uth_y = " + num(std::sqrt(te)) + "\n uth_z = " + num(std::sqrt(te)) + "\n}\n"
        "species c {\n element = C\n charge_state = 0\n mobile = false\n density = " + num(ne / 10) + "\n" + ppc(0, 16, 16, 16) + "}\n" +
        block("ec", "e", "c", 5, " ionizing = true\n") + "output {\n dir = coll_tmp_out\n}\n";
    Comm comm;
    Simulation s(cdeck(text), comm);
    s.init();
    Species &E = s.species[size_t(idx(s, "e"))], &I = s.species[size_t(idx(s, "c"))];
    // R_k = n_e <sigma_k(E) v> over the electrons (the ions at rest)
    std::vector<double> R(6, 0.0);
    {
        const auto X = E.get(Species::UX), Y = E.get(Species::UY), Z = E.get(Species::UZ);
        for (int k = 0; k < 6; ++k) {
            double a = 0;
            for (size_t p = 0; p < X.size(); p += 4) {
                const double u2 = X[p] * X[p] + Y[p] * Y[p] + Z[p] * Z[p], g = std::sqrt(1 + u2);
                a += impact_cross_section(6, C->ip_eV, KRE, k, g - 1) * std::sqrt(u2) / g;
            }
            R[size_t(k)] = ne * a / double((X.size() + 3) / 4);
        }
    }
    auto charge = [&]() {
        const auto zs = I.get_as<int>(I.zslot);
        const auto W = I.get(Species::W), We = E.get(Species::W);
        double q = 0, w = 0, qe = 0;
        for (size_t p = 0; p < zs.size(); ++p) { q += W[p] * zs[p]; w += W[p]; }
        for (double x : We) qe -= x;
        return std::vector<double>{q / w, q + qe};
    };
    const double q0 = charge()[1];
    std::vector<double> N(7, 0.0);
    N[0] = 1;
    double worst = 0, zt = 0;
    for (int k = 1; k <= steps; ++k) {
        step(s);
        N = rk4(N, [&](const std::vector<double>& y) {
            std::vector<double> d(7, 0.0);
            for (int j = 0; j < 6; ++j) { d[size_t(j)] -= R[size_t(j)] * y[size_t(j)]; d[size_t(j) + 1] += R[size_t(j)] * y[size_t(j)]; }
            return d;
        }, h, 10);
        zt = 0;
        for (int j = 0; j <= 6; ++j) zt += j * N[size_t(j)];
        if (k % 10 == 0) worst = std::max(worst, std::fabs(charge()[0] / zt - 1));
    }
    const auto c1 = charge();
    const CollisionTotals T = collisions_totals(s, 0);
    check("I1 impact ionization of C by a 1 keV electron beam vs n_e <sigma v> (rate equations)", worst < 0.05,
          fmt("<Z> %.5f vs %.5f", c1[0], zt) + fmt("; max relative error %.3f, %g events", worst, T.events));
    check("I1b the charge is conserved (ions + new electrons)", std::fabs(c1[1] - q0) < 1e-10 * std::fabs(q0),
          fmt("sum w q before %.10e, relative change %.1e", q0, (c1[1] - q0) / q0) + " (" + std::to_string(E.size()) + " electrons now)");
    // deck checks of ionizing blocks
    const std::string base = grid(0, 4, 4, 4, 0.1) + "species e {\n charge = -1\n mass = 1\n density = 1\n}\n"
                             "species p {\n charge = 1\n mass = 1836\n density = 1\n}\nspecies c {\n element = C\n density = 1\n charge = 1\n}\n"
                             "output {\n dir = coll_tmp_out\n}\n";
    auto fails = [&](const std::string& extra) {
        try { cdeck(base + extra); } catch (const std::exception&) { return true; }
        return false;
    };
    const bool ok = fails(block("x", "e", "p", 0, " ionizing = true\n")) && fails(block("x", "e", "c", 0, " ionizing = true\n")) &&
                    fails(block("x", "p", "c", 0, " ionizing = true\n")) && fails(block("x", "e", "p", 0, " ionization_electrons = e\n"));
    check("I2 deck checks of ionizing blocks (no element, no charge_state, no electrons, ionization_electrons alone)", ok, "");
}
}  // namespace

void test_collisions(Comm&) {
    impact();
    conservation_unit();
    nanbu_mean();
    isotropization();
    relaxation();
    beam();
    juttner();
    order_and_totals();
    coulomb_log_auto();
    deck_checks();
}
