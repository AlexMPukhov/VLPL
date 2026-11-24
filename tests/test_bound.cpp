// O7, model = bound (decision 85, src/bloch.hpp): a gas of V-type Maxwell-Bloch atoms (e.g. Rb D1 + D2) in iRIP.
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "common.hpp"
#include "deck.hpp"
#include "fields.hpp"
#include "implicit_mover.hpp"
#include "medium.hpp"
#include "rz.hpp"
#include "sim.hpp"

namespace {
using cd = std::complex<double>;
const UnitFn no_units = [](const std::string&, double&) { return false; };
const double LAM0 = 800e-9, K0 = 2 * M_PI / LAM0, A0K0 = 5.29177210903e-11 * K0;

// the Rb lines in units of omega0 = 2 pi c / 800 nm (as medium_block)
MediumConfig rb(double na, int axis) {
    MediumConfig m;
    m.bound = true; m.lines = 2; m.axis = axis; m.density = na; m.a0k0 = A0K0;
    const double lam[2] = {794.979e-9, 780.241e-9}, f[2] = {0.342, 0.695};
    for (int j = 0; j < 2; ++j) {
        m.bw[j] = LAM0 / lam[j]; m.bf[j] = f[j];
        m.bd[j] = std::sqrt(f[j] * 27.211386245988 / (2 * 1.23984198e-6 / lam[j]));
    }
    return m;
}
Domain bdom(Comm& comm, int nx, int ny, double hx, double hy, bool px) {
    Domain d;
    d.N[0] = nx; d.N[1] = ny; d.N[2] = 1;
    d.h[0] = hx; d.h[1] = hy; d.h[2] = 1.0;
    d.flat = true; d.periodic[0] = px; d.ng = 3; d.solver = Solver::IRip;
    const int want[3] = {0, 1, 1};
    comm.decompose(d, want);
    return d;
}
Config deck_config(const std::string& text, const char* path) {
    if (FILE* f = std::fopen(path, "w")) { std::fputs(text.c_str(), f); std::fclose(f); }
    Deck dk = Deck::read(path);
    return Config::from_deck(dk);
}
}  // namespace

// B1. The node step alone: Rabi flopping of a resonant two-level atom (w = 1, Rabi frequency a E0 = 0.133, Gaussian envelope of
//     area pi / 2, pi, 3 pi / 2 in the rotating-wave sense) against the Schroedinger equation without the rotating-wave
//     approximation (RK4, dt / 50) in the same field; the field at the midpoints t_{n+1/2} (as the box closure's work field).
//     Second order in dt; the energy exchange sum_n E_{n+1/2} (P^{n+1} - P^n) = ef [sum w_j' p_j] exact.
static void bound_rabi_node() {
    const double tau = 20, t0 = 4 * tau, T = 2 * t0;
    auto field = [&](double t, double E0) { return E0 * std::exp(-std::pow((t - t0) / tau, 2)) * std::cos(t - t0); };
    auto reference = [&](double a, double E0) {                     // i dc/dt = H c, H = [[0, -a E], [-a E, 1]]
        const int n = int(std::round(T / (2 * M_PI / 3200)));
        const double h = T / n;
        cd c0 = 1, c1 = 0;
        auto rhs = [&](double t, cd x0, cd x1, cd& d0, cd& d1) {
            const double e = field(t, E0);
            d0 = cd(0, 1) * (a * e * x1);
            d1 = cd(0, 1) * (a * e * x0 - x1);
        };
        for (int q = 0; q < n; ++q) {
            const double t = q * h;
            cd k0a, k0b, k1a, k1b, k2a, k2b, k3a, k3b;
            rhs(t, c0, c1, k0a, k0b);
            rhs(t + h / 2, c0 + h / 2 * k0a, c1 + h / 2 * k0b, k1a, k1b);
            rhs(t + h / 2, c0 + h / 2 * k1a, c1 + h / 2 * k1b, k2a, k2b);
            rhs(t + h, c0 + h * k2a, c1 + h * k2b, k3a, k3b);
            c0 += h / 6 * (k0a + 2.0 * k1a + 2.0 * k2a + k3a);
            c1 += h / 6 * (k0b + 2.0 * k1b + 2.0 * k2b + k3b);
        }
        return std::norm(c1);
    };
    const double a = 0.133 / 1.0;
    const double w = 1, dd = a / 137.035999084;                     // d / (e a0) for the Rabi frequency a per unit field
    std::string info;
    bool ok = true;
    for (double area : {0.5, 1.0, 1.5}) {
        const double E0 = area * M_PI / (a * tau * std::sqrt(M_PI));
        const double ref = reference(a, E0);
        double err[3], eworst = 0;
        int q = 0;
        for (int N : {16, 32, 64}) {
            const double dt = 2 * M_PI / N;
            const int n = int(std::round(T / dt));
            const BlochC L = bloch_constants(1, &w, &dd, 1.0, dt);
            double s[BS_N] = {1, 0, 0, 0, 0, 0, 0, 0, 0}, work = 0, etot = 0;
            for (int k = 0; k < n; ++k) {
                const double e = field((k + 0.5) * dt, E0), p0 = bloch_P(L, s);
                bloch_rotate(L, s, e, s);
                work += e * (bloch_P(L, s) - p0);
                etot = std::max(etot, std::abs(bloch_energy(L, s)));
            }
            err[q++] = s[BS_P1] - ref;
            eworst = std::max(eworst, std::abs(bloch_energy(L, s) - work) / etot);
            ok = ok && eworst < 1e-12;
        }
        const double order = std::log2(std::abs(err[1] / err[2]));
        ok = ok && std::abs(err[2]) < 2e-3 && std::abs(err[2]) < std::abs(err[1]) && order > 1.6;
        info += fmt(" area %.1f pi:", area) + fmt(" ref %.5f,", ref) + fmt(" err %.1e", err[0]) + fmt(" / %.1e", err[1]) +
                fmt(" / %.1e", err[2]) + fmt(" (order %.2f);", order);
        if (area == 1.5) info += fmt(" energy exchange %.1e", eworst);
    }
    check("medium bound: node Rabi flopping vs Schroedinger without RWA (lambda/16, 32, 64), exact energy exchange", ok, info);
}

// B2. Weak field: a pulse along +x (carrier 0.92 omega0, i.e. 870 nm; envelope 12 lambda: its spectrum at D1 is 3e-5 of the peak,
//     so the undamped atoms do not ring at the lines) through 60 lambda of Rb vapour (n_a = 1e-4 n_c = 1.7e17 cm^-3, D1 + D2), 2D,
//     s-pol (dipole along z). The spectral phase of Ez behind the gas, divided by a vacuum run, against (n(omega) - 1) omega L of the
//     Lorentz susceptibility chi = n_a sum f_j / (w_j^2 - omega^2) over 0.90 .. 0.94 omega0 (chi = 4.6 .. 6.8e-4); lambda/20, 40.
//     (With an envelope of 6 lambda at 0.95 omega0 a third of the spectral amplitude sat at D1: the ringing of the lines leaked
//     into the band through the finite window, a 7 % offset, the same for the linear Lorentz medium.)
namespace {
struct Weak { double rms, maxrel, its, newton; };
Weak bound_weak(Comm& comm, int N) {
    const double lam = 2 * M_PI, dx = lam / N, w = 12 * lam, xb = 3.5 * w, xa = xb + 3.5 * w, L = 60 * lam, xp = xa + L + 2 * lam;
    const double wc = 0.92;
    const int nx = int(std::round((xp + 2 * lam) / dx));
    const Domain d = bdom(comm, nx, 4, dx, dx, false);
    const MediumConfig m0 = rb(1e-4, 2);
    std::vector<double> sig[2];
    Weak r{0, 0, 0, 0};
    for (int gas = 0; gas < 2; ++gas) {
        Fields F(d);
        Medium M;
        if (gas) {
            MediumConfig m = m0;
            char t[96];
            std::snprintf(t, sizeof t, "x >= %.17g && x < %.17g", xa, xa + L);
            m.profile = Expr(t, {"x", "y", "z"}, no_units);
            medium_build(M, d, {m}, 1);
            M.tol = 1e-11;
            F.medium = &M;
        }
        {
            H ez(F.ez), by(F.by);
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const double x = d.lo(0) + i * dx, g = 1e-6 * std::exp(-std::pow((x - xb) / w, 2)) * std::cos(wc * (x - xb));
                    ez(i, j, 0) = g; by(i, j, 0) = -g;
                }
            ez.put(F.ez); by.put(F.by);
        }
        comm.exchange(d, {&F.ez, &F.by});
        const int steps = int(std::round((xp - xb + 4 * w) / dx));
        const int ip = int(std::round(xp / dx)) - d.off[0];
        for (int n = 0; n < steps; ++n) {
            F.step(comm);
            double v = 0;
            if (ip >= 0 && ip < d.n[0] && d.off[1] == 0) v = H(F.ez)(ip, 0, 0);
            comm.allreduce_sum(&v, 1);
            sig[gas].push_back(v);
        }
        if (gas) { r.its = double(M.stats.iterations) / M.stats.solves; r.newton = double(M.stats.newton) / M.stats.newton_steps; }
    }
    double s2 = 0, ref2 = 0;
    int cnt = 0;
    for (double om = 0.90; om <= 0.94 + 1e-9; om += 0.0025) {
        cd a = 0, b = 0;
        for (size_t n = 0; n < sig[0].size(); ++n) {
            const cd e = std::exp(cd(0, om * (n + 1) * dx));
            a += sig[0][n] * e; b += sig[1][n] * e;
        }
        const double ph = std::arg(b / a);
        double chi = 0;
        for (int j = 0; j < m0.lines; ++j) chi += m0.density * m0.bf[j] / (m0.bw[j] * m0.bw[j] - om * om);
        const double ref = (std::sqrt(1 + chi) - 1) * om * L;
        s2 += (ph - ref) * (ph - ref); ref2 += ref * ref; ++cnt;
        r.maxrel = std::max(r.maxrel, std::abs(ph / ref - 1));
    }
    r.rms = std::sqrt(s2 / ref2);
    return r;
}
}  // namespace

static void bound_weak_field(Comm& comm) {
    const Weak a = bound_weak(comm, 20), b = bound_weak(comm, 40);
    check("medium bound: weak field in Rb vapour (D1 + D2, 1e-4 n_c, 60 lambda): spectral phase vs the Lorentz chi, 0.90 .. 0.94 omega0",
          b.rms < 5e-3 && b.rms < a.rms, fmt("rms of the phase lambda/20 %.2e", a.rms) + fmt(" (max %.2e)", a.maxrel) +
          fmt(", lambda/40 %.2e", b.rms) + fmt(" (max %.2e)", b.maxrel) + fmt("; GMRES %.1f per solve", b.its) +
          fmt(", outer iterations %.2f per step", b.newton));
}

// B3. Strong field (saturation): a resonant-region pulse (carrier omega0 = 800 nm, a0 = 1e-3: Rabi frequency of D2 0.33 omega0
//     >> the detunings 0.006, 0.025) through 6 lambda of dense Rb (1e-3 n_c), 2D, dipole along z: W + K_atoms constant to the
//     tolerances of the solves while the atoms are driven far from linear response (excited population up to ~0.5).
static void bound_energy_strong(Comm& comm, bool relax) {
    const double lam = 2 * M_PI, dx = lam / 24, w = 2 * lam, xb = 3.5 * w, xa = xb + 3 * w;
    const int nx = int(std::round((xa + 6 * lam + 4 * lam) / dx));
    const Domain d = bdom(comm, nx, 4, dx, dx, true);
    Fields F(d);
    Medium M;
    MediumConfig m = rb(1e-3, 2);
    if (relax) { m.bT1 = 30; m.bT2 = 30; }                          // ~5 periods: strong damping
    char t[96];
    std::snprintf(t, sizeof t, "x >= %.17g && x < %.17g", xa, xa + 6 * lam);
    m.profile = Expr(t, {"x", "y", "z"}, no_units);
    medium_build(M, d, {m}, 1);
    M.tol = 1e-12; M.btol = 1e-10;
    F.medium = &M;
    {
        H ez(F.ez), by(F.by);
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                const double x = d.lo(0) + i * dx, g = 1e-3 * std::exp(-std::pow((x - xb) / w, 2)) * std::cos(x - xb);
                ez(i, j, 0) = g; by(i, j, 0) = -g;
            }
        ez.put(F.ez); by.put(F.by);
    }
    comm.exchange(d, {&F.ez, &F.by});
    auto tot = [&](double* k) {
        double v[2] = {F.energy(), M.energy(d)};
        comm.allreduce_sum(v, 2);
        if (k) *k = v[1];
        return v[0] + v[1] + M.dissipated;
    };
    const double e0 = tot(nullptr);
    double dev = 0, kmax = 0, pmax = 0;
    const int steps = int(std::round((xa + 6 * lam - xb + 2 * w) / dx));
    for (int n = 0; n < steps; ++n) {
        F.step(comm);
        double k;
        dev = std::max(dev, std::abs(tot(&k) / e0 - 1));
        kmax = std::max(kmax, k / e0);
        const std::vector<double> p1 = M.bs[BS_P1].host(), p2 = M.bs[BS_P2].host();
        for (size_t q = 0; q < p1.size(); ++q) pmax = std::max(pmax, p1[q] + p2[q]);
    }
    pmax = comm.allreduce_max(pmax);
    const MediumStats& st = M.stats;
    check(std::string("medium bound: W + K_atoms") + (relax ? " + dissipated, with T1 = T2 = 30 / omega0," : "") +
              " in the saturated regime (Rb 1e-3 n_c, a0 1e-3 at 800 nm), 2D",
          dev < 1e-9 && st.unconverged == 0 && st.newton_unconverged == 0 && pmax > 0.1 && (!relax || M.dissipated > 0.01 * e0),
          fmt("max |E/E0 - 1| %.1e", dev) + (relax ? fmt(", dissipated %.3f E0", M.dissipated / e0) : std::string()) + fmt(", K_atoms up to %.2e E0", kmax) + fmt(", excited population up to %.3f", pmax) +
              fmt("; outer iterations %.2f per step", double(st.newton) / st.newton_steps) + fmt(" (max %.0f)", double(st.newton_max)) +
              fmt(", GMRES %.1f per solve", double(st.iterations) / st.solves));
}

// B4. The deck (atom = Rb, density in cm^-3, polarization) and the Gauss law with particles plus atoms (the charge -div P of the
//     atoms), 3D, dipole along y, explicit and implicit movers (W + K conserved with O6).
static void bound_deck_gauss(Comm& comm) {
    const std::string base =
        "grid {\n geometry = cartesian2d\n lambda0 = 0.8 um\n nx = 16\n ny = 4\n dx = 0.04 um\n dy = 0.04 um\n t_end = 1 fs\n}\n"
        "numerics {\n solver = irip\n}\n"
        "medium rb {\n model = bound\n atom = Rb\n density = 1.742e17 cm^-3\n polarization = y\n}\n"
        "output {\n dir = medium_tmp_out\n}\n";
    Config c = deck_config(base, "bound_tmp.deck");
    const MediumConfig& m = c.media.at(0);
    const bool parsed = m.bound && m.lines == 2 && m.axis == 1 && std::abs(m.density / 1.0005e-4 - 1) < 2e-3 &&
                        std::abs(m.bw[0] - 800 / 794.979) < 1e-9 && std::abs(m.bd[1] - 2.4394) < 1e-3;
    int rejected = 0;
    Deck dk = Deck::read("bound_tmp.deck");
    for (const char* o : {"medium rb.resonance=2", "medium rb.atom=Cs", "medium rb.polarization=r", "grid.geometry=rz"}) {
        Deck x = dk;
        x.override_from(o);
        try { Config::from_deck(x); } catch (const std::exception&) { ++rejected; }
    }
    check("medium bound: deck (atom = Rb, 1.742e17 cm^-3 = 1.0e-4 n_c at 0.8 um, D2 dipole 2.44 e a0), invalid keys rejected",
          parsed && rejected == 4, fmt("n_a %.4e n_c", m.density) + fmt(", d2 %.4f e a0", m.bd[1]) + fmt(", rejected %.0f of 4", rejected));
    for (int implicit = 0; implicit < 2; ++implicit) {
        const std::string t =
            std::string("grid {\n lambda0 = 0.8 um\n nx = 16\n ny = 8\n nz = 6\n dx = 0.04 um\n dy = 0.03 um\n dz = 0.04 um\n t_end = 3 fs\n}\n") +
            "numerics {\n solver = irip\n shape = 2\n" + (implicit ? " mover = implicit\n mover_tol = 1e-12\n mover_maxit = 200\n" : "") +
            " medium_tol = 1e-12\n medium_bound_tol = 1e-10\n}\n"
            "species e {\n charge = -1\n mass = 1\n density = 2\n xmax = 0.4 um\n ppc_x = 2\n ppc_y = 2\n ppc_z = 2\n"
            " uth_x = 0.2\n uth_y = 0.2\n uth_z = 0.2\n}\n"
            "species i {\n charge = 1\n mass = 1836\n density = 2\n xmax = 0.4 um\n ppc_x = 2\n ppc_y = 2\n ppc_z = 2\n mobile = false\n}\n"
            "medium rb {\n model = bound\n atom = Rb\n density = 0.05\n polarization = y\n profile = x > 0.2 um && x < 0.5 um\n}\n"
            "medium glass {\n model = lorentz\n density = 4\n resonance = 2\n profile = x > 0.3 um && y < 0.15 um\n}\n"
            "output {\n dir = medium_tmp_out\n}\n";
        Config cc = deck_config(t, "bound_tmp.deck");
        Simulation sim(cc, comm);
        sim.init();
        auto total = [&] {
            double w = sim.fields->energy(), k = 0;
            for (const Species& sp : sim.species) k += sp.kinetic_energy();
            comm.allreduce_sum(&w, 1);
            comm.allreduce_sum(&k, 1);
            return w + k + sim.medium_energy() + sim.medium->dissipated;
        };
        const double e0 = total();
        double g = 0, dev = 0, pb = 0;
        while (!sim.finished()) {
            sim.step();
            g = std::max(g, sim.gauss_drift());
            dev = std::max(dev, std::abs(total() / e0 - 1));
        }
        for (double v : sim.medium->bP.host()) pb = std::max(pb, std::abs(v));
        pb = comm.allreduce_max(pb);
        const MediumStats& ms = sim.medium->stats;
        std::string info = fmt("Gauss %.1e", g) + fmt(", max |P_atoms| %.1e", pb) +
                           fmt(", outer iterations %.2f per step", double(ms.newton) / ms.newton_steps);
        bool ok = g < 1e-12 && pb > 0 && ms.unconverged == 0 && ms.newton_unconverged == 0;
        if (implicit) {
            const MoverStats& st = implicit_mover_stats(sim);
            info += fmt(", max |E/E0 - 1| %.1e", dev) + fmt(", mover %.1f iterations per step", double(st.iterations) / st.steps);
            ok = ok && dev < 1e-9 && st.unconverged == 0;
        }
        check(std::string("medium bound: Gauss law with particles, atoms and a Lorentz layer, 3D, ") +
                  (implicit ? "implicit mover (W + K conserved)" : "explicit mover"), ok, info);
    }
    if (comm.rank == 0) std::remove("bound_tmp.deck");
}

// B5. Rabi flopping on the grid: a thin layer (n_a = 1e-10 n_c, no effect on the pulse) of resonant two-level atoms (w = 1, a =
//     0.133) hit by a pulse along +x of area pi / 2, pi, 3 pi / 2 (2D, s-pol, lambda/32). The excited population at an atom against
//     the Schroedinger equation (RK4) in the analytic vacuum field at its position, E(t) = g(x_a - t): the grid hands the atoms the
//     midpoint field of the step.
static void bound_rabi_grid(Comm& comm) {
    const double lam = 2 * M_PI, dx = lam / 32, tau = 20, xb = 4 * tau, xa = xb + 4 * tau, a = 0.133;
    const int nx = int(std::round((xa + 2 * lam) / dx));
    const Domain d = bdom(comm, nx, 4, dx, dx, false);
    std::string info;
    bool ok = true;
    for (double area : {0.5, 1.0, 1.5}) {
        const double E0 = area * M_PI / (a * tau * std::sqrt(M_PI));
        auto g = [&](double x) { return E0 * std::exp(-std::pow((x - xb) / tau, 2)) * std::cos(x - xb); };
        Fields F(d);
        Medium M;
        MediumConfig m;
        m.bound = true; m.lines = 1; m.bw[0] = 1; m.bf[0] = 1; m.bd[0] = a / 137.035999084; m.axis = 2; m.density = 1e-10;
        m.a0k0 = A0K0;
        char t[96];
        std::snprintf(t, sizeof t, "x >= %.17g && x < %.17g", xa, xa + lam);
        m.profile = Expr(t, {"x", "y", "z"}, no_units);
        m.supersample = 1;
        medium_build(M, d, {m}, 1);
        F.medium = &M;
        {
            H ez(F.ez), by(F.by);
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const double x = d.lo(0) + i * dx;
                    ez(i, j, 0) = g(x); by(i, j, 0) = -g(x);
                }
            ez.put(F.ez); by.put(F.by);
        }
        comm.exchange(d, {&F.ez, &F.by});
        const double T = xa + lam - xb + 4 * tau;
        const int steps = int(std::round(T / dx));
        for (int n = 0; n < steps; ++n) F.step(comm);
        // the atom at the Jz position x_i + dx / 2 of the global node ia (the first one inside the layer)
        const int ia = int(std::ceil(xa / dx - 0.5 - 1e-9));
        const double xn = (ia + 0.5) * dx;
        double p = -1;
        if (ia - d.off[0] >= 0 && ia - d.off[0] < d.n[0] && d.off[1] == 0) p = H(M.bs[BS_P1])(ia - d.off[0], 0, 0);
        p = comm.allreduce_max(p);
        const int nr = int(std::round(steps * dx / (2 * M_PI / 3200)));
        const double h = steps * dx / nr;
        cd c0 = 1, c1 = 0;
        auto rhs = [&](double tt, cd x0, cd x1, cd& d0, cd& d1) {
            const double e = g(xn - tt);
            d0 = cd(0, 1) * (a * e * x1);
            d1 = cd(0, 1) * (a * e * x0 - x1);
        };
        for (int q = 0; q < nr; ++q) {
            const double tt = q * h;
            cd k0a, k0b, k1a, k1b, k2a, k2b, k3a, k3b;
            rhs(tt, c0, c1, k0a, k0b);
            rhs(tt + h / 2, c0 + h / 2 * k0a, c1 + h / 2 * k0b, k1a, k1b);
            rhs(tt + h / 2, c0 + h / 2 * k1a, c1 + h / 2 * k1b, k2a, k2b);
            rhs(tt + h, c0 + h * k2a, c1 + h * k2b, k3a, k3b);
            c0 += h / 6 * (k0a + 2.0 * k1a + 2.0 * k2a + k3a);
            c1 += h / 6 * (k0b + 2.0 * k1b + 2.0 * k2b + k3b);
        }
        const double ref = std::norm(c1);
        const double its = double(M.stats.iterations) / M.stats.solves;
        ok = ok && std::abs(p - ref) < 2e-4 && its < 5 && M.stats.unconverged == 0;   // GMRES: Ex, Bx ~ 0 (the residual floor)
        info += fmt(" area %.1f pi:", area) + fmt(" %.6f", p) + fmt(" vs %.6f", ref) + fmt(" (GMRES %.1f per solve);", its);
    }
    check("medium bound: Rabi flopping on the grid (thin layer, lambda/32) vs Schroedinger in the vacuum field at the atom", ok, info);
}

// B6. Relaxation in the node step: a resonant two-level atom (a = 0.133) driven by a Gaussian pulse of area 3 pi / 2 with T1 = 60,
//     T2 = 40 (1 / omega0), against the density-matrix equations (RK4) with the same relaxation terms: p1 and Re rho_01 at the peak
//     of the pulse and at its end, lambda/16, 32, 64 (second order: Strang splitting of exact exponentials around the Cayley step).
static void bound_relax_node() {
    const double tau = 20, t0 = 4 * tau, a = 0.133, T1 = 60, T2 = 40, w = 1, dd = a / 137.035999084;
    const double E0 = 1.5 * M_PI / (a * tau * std::sqrt(M_PI));
    auto field = [&](double t) { return E0 * std::exp(-std::pow((t - t0) / tau, 2)) * std::cos(t - t0); };
    // reference at time tt: p1, Re c (c = rho_01)
    auto reference = [&](double tt, double& p1, double& rc) {
        const int n = int(std::round(tt / (2 * M_PI / 3200)));
        const double h = tt / n;
        double p0 = 1, q1 = 0;
        cd c = 0;
        auto rhs = [&](double t, double x0, double x1, cd y, double& d0, double& d1, cd& dy) {
            const double e = field(t);
            dy = cd(0, 1) * w * y - cd(0, 1) * (a * e * (x0 - x1)) - y * (0.5 / T1 + 1 / T2);
            d1 = -2 * a * e * y.imag() - x1 / T1;
            d0 = -d1;
        };
        for (int q = 0; q < n; ++q) {
            const double t = q * h;
            double a0, a1, b0, b1, c0, c1, e0, e1;
            cd ay, by, cy, ey;
            rhs(t, p0, q1, c, a0, a1, ay);
            rhs(t + h / 2, p0 + h / 2 * a0, q1 + h / 2 * a1, c + h / 2 * ay, b0, b1, by);
            rhs(t + h / 2, p0 + h / 2 * b0, q1 + h / 2 * b1, c + h / 2 * by, c0, c1, cy);
            rhs(t + h, p0 + h * c0, q1 + h * c1, c + h * cy, e0, e1, ey);
            p0 += h / 6 * (a0 + 2 * b0 + 2 * c0 + e0);
            q1 += h / 6 * (a1 + 2 * b1 + 2 * c1 + e1);
            c += h / 6 * (ay + 2.0 * by + 2.0 * cy + ey);
        }
        p1 = q1; rc = c.real();
    };
    double err[3], rp_end = 0;
    int q = 0;
    std::string info;
    for (int N : {16, 32, 64}) {
        const double dt = 2 * M_PI / N;
        // the reference at the sample times of the scheme (Re rho_01 oscillates with the carrier)
        double rp_mid, rc_mid, rc_end;
        reference(std::round(t0 / dt) * dt, rp_mid, rc_mid);
        reference(2 * std::round(t0 / dt) * dt, rp_end, rc_end);
        BlochC L = bloch_constants(1, &w, &dd, 1.0, dt);
        bloch_set_relaxation(L, T1, T2, dt);
        double s[BS_N] = {1, 0, 0, 0, 0, 0, 0, 0, 0}, emid = 0;
        const int nmid = int(std::round(t0 / dt)), n = 2 * nmid;
        for (int k = 0; k < n; ++k) {
            bloch_step(L, s, field((k + 0.5) * dt), s);
            if (k + 1 == nmid) emid = std::max(std::abs(s[BS_P1] - rp_mid), std::abs(s[BS_R01] - rc_mid));
        }
        err[q++] = std::max(emid, std::max(std::abs(s[BS_P1] - rp_end), std::abs(s[BS_R01] - rc_end)));
    }
    const double order = std::log2(err[1] / err[2]);
    info = fmt("p1 at the end %.5f", rp_end) + fmt(" (without relaxation 0.50017); max error %.1e", err[0]) + fmt(" / %.1e", err[1]) +
           fmt(" / %.1e", err[2]) + fmt(" (order %.2f)", order);
    check("medium bound: relaxation T1 = 60, T2 = 40: driven damped two-level atom vs the density-matrix equations (lambda/16, 32, 64)",
          err[2] < 1e-3 && order > 1.7, info);
}

// B7. Ionization (ADK per level): (a) the rate of hydrogen (U = 13.606 eV) against the Landau formula 4 omega_a (E_a / E)
//     exp(-2 E_a / 3E); (b) Rb (D1 + D2, ground 4.177 eV, excited 2.62 / 2.59 eV) in a pulse of a0 = 6e-4 at 800 nm (Gaussian
//     envelope tau = 40 / omega0): the neutral fraction (trace of rho) after the pulse against the density-matrix equations with the
//     ADK loss terms (RK4, its own SI ADK), lambda/16, 32, 64.
namespace {
double adk_si(double U, double E) {                                 // [1/s], E [V/m], Z after = 1
    const double UH = 13.605693122994, Ea = 5.14220674763e11, wa = 4.1341373335e16;
    const double ns = std::sqrt(UH / U), C2 = std::pow(2.0, 2 * ns) / (ns * std::tgamma(ns + 1) * std::tgamma(ns));
    const double kap = std::pow(U / UH, 1.5) * Ea, Ep = 2.0 / 3.0 * kap / (2 * ns - 1);
    E = std::min(std::abs(E), Ep);
    if (E <= 0) return 0;
    return wa * C2 * U / (2 * UH) * std::pow(2 * kap / E, 2 * ns - 1) * std::exp(-2.0 / 3.0 * kap / E);
}
}  // namespace
static void bound_ionize_node() {
    const double c = 299792458.0, w0 = c * K0, Eu = 9.1093837015e-31 * c * w0 / 1.602176634e-19;
    {   // (a) hydrogen
        const double one = 1, U[3] = {13.605693122994, 1, 1};
        BlochC L = bloch_constants(1, &one, &one, A0K0, 0.1);
        bloch_set_adk(L, U, K0);
        double worst = 0;
        for (double f : {0.02, 0.05, 0.1}) {
            const double Ea = 5.14220674763e11, E = f * Ea / Eu;
            const double ref = 4 * 4.1341373335e16 / f * std::exp(-2.0 / (3 * f)) / w0;
            worst = std::max(worst, std::abs(bloch_adk(L, 0, E) / ref - 1));
        }
        check("medium bound: ADK rate of hydrogen vs the Landau formula (E = 0.02, 0.05, 0.1 E_a)", worst < 1e-12, fmt("max rel. difference %.1e", worst));
    }
    // (b) Rb
    const MediumConfig m = rb(1, 2);
    const double U[3] = {4.177128, 4.177128 - 1.559591, 4.177128 - 1.589049};
    const double tau = 40, t0 = 4 * tau, E0 = 6e-4;
    auto field = [&](double t) { return E0 * std::exp(-std::pow((t - t0) / tau, 2)) * std::cos(t - t0); };
    const double a[2] = {m.bd[0] * 137.035999084, m.bd[1] * 137.035999084};
    // reference: i d rho / dt = [H, rho] with H = [[0, -a1 E, -a2 E], [-a1 E, w1, 0], [-a2 E, 0, w2]], plus -{G, rho} / 2,
    // G = diag(g0, g1, g2) the ADK rates at |E(t)|; returns the trace at time T
    auto reference = [&](double T) {
        typedef std::complex<double> C;
        const int n = int(std::round(T / (2 * M_PI / 1600)));
        const double h = T / n;
        C r[3][3] = {{1, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        auto rhs = [&](double t, const C (*x)[3], C (*o)[3]) {
            const double e = field(t);
            const double H[3][3] = {{0, -a[0] * e, -a[1] * e}, {-a[0] * e, m.bw[0], 0}, {-a[1] * e, 0, m.bw[1]}};
            const double g[3] = {adk_si(U[0], e * Eu) / w0, adk_si(U[1], e * Eu) / w0, adk_si(U[2], e * Eu) / w0};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    C s = 0;
                    for (int k = 0; k < 3; ++k) s += H[i][k] * x[k][j] - x[i][k] * H[k][j];
                    o[i][j] = C(0, -1) * s - 0.5 * (g[i] + g[j]) * x[i][j];
                }
        };
        C k1[3][3], k2[3][3], k3[3][3], k4[3][3], y[3][3];
        for (int q = 0; q < n; ++q) {
            const double t = q * h;
            rhs(t, r, k1);
            for (int i = 0; i < 9; ++i) y[i / 3][i % 3] = r[i / 3][i % 3] + h / 2 * k1[i / 3][i % 3];
            rhs(t + h / 2, y, k2);
            for (int i = 0; i < 9; ++i) y[i / 3][i % 3] = r[i / 3][i % 3] + h / 2 * k2[i / 3][i % 3];
            rhs(t + h / 2, y, k3);
            for (int i = 0; i < 9; ++i) y[i / 3][i % 3] = r[i / 3][i % 3] + h * k3[i / 3][i % 3];
            rhs(t + h, y, k4);
            for (int i = 0; i < 9; ++i)
                r[i / 3][i % 3] += h / 6 * (k1[i / 3][i % 3] + 2.0 * k2[i / 3][i % 3] + 2.0 * k3[i / 3][i % 3] + k4[i / 3][i % 3]);
        }
        return (r[0][0] + r[1][1] + r[2][2]).real();
    };
    double err[3], ref = 0;
    int q = 0;
    for (int N : {16, 32, 64}) {
        const double dt = 2 * M_PI / N;
        BlochC L = bloch_constants(2, m.bw, m.bd, A0K0, dt);
        bloch_set_adk(L, U, K0);
        const int n = int(std::round(2 * t0 / dt));
        double s[BS_N] = {1, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < n; ++k) bloch_step(L, s, field((k + 0.5) * dt), s);
        ref = reference(n * dt);
        err[q++] = s[BS_P0] + s[BS_P1] + s[BS_P2] - ref;
    }
    const double order = std::log2(std::abs(err[1] / err[2]));
    check("medium bound: Rb ionization (ADK per level, a0 6e-4 at 800 nm): neutral fraction vs the density-matrix equations",
          std::abs(err[2]) < 2e-3 && ref > 0.02 && ref < 0.98 && order > 1.5,
          fmt("neutral fraction %.4f;", ref) + fmt(" error %.1e", err[0]) + fmt(" / %.1e", err[1]) + fmt(" / %.1e", err[2]) +
              fmt(" at lambda/16, 32, 64 (order %.2f)", order));
}

// B8. Ionization on the grid with particles: an 800 nm pulse (a0 1.3e-3, 2D, s-pol) through Rb (1e-4 n_c) with ionization = adk,
//     electrons and ions (Rb+) as species, the implicit mover (O6) so that W + K_particles + K_atoms + dissipated is conserved: the Gauss
//     law to round-off with the pairs, the charge bookkeeping sum w_e + accumulated = ionized, the energy balance.
static void bound_ionize_grid(Comm& comm) {
    const std::string t =
        R"(grid {
 geometry = cartesian2d
 lambda0 = 0.8 um
 nx = 240
 ny = 8
 dx = 0.04 um
 dy = 0.04 um
 t_end = 16 fs
}
numerics {
 solver = irip
 shape = 2
 mover = implicit
 mover_tol = 1e-12
 mover_maxit = 200
 medium_tol = 1e-12
 medium_bound_tol = 1e-10
}
species electrons {
 charge = -1
 mass = 1
 density = 0
}
species rb_ions {
 charge = 1
 mass = 156700
 density = 0
}
medium rb {
 model = bound
 atom = Rb
 density = 1e-4
 polarization = z
 ionization = adk
 electrons = electrons
 ions = rb_ions
 ionization_ppc = 4
 profile = x > 6 um && x < 8 um
}
output {
 dir = medium_tmp_out
}
)";
    Config c = deck_config(t, "bound_tmp.deck");
    Simulation sim(c, comm);
    sim.init();
    {                                                                // the pulse in the periodic box (x_b = 3 um, envelope 1.2 um)
        const Domain& d = sim.dom;
        Fields& F = *sim.fields;
        const double xb = 3e-6 * K0, w = 1.2e-6 * K0;
        H ez(F.ez), by(F.by);
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                const double x = d.lo(0) + i * d.h[0], g = 1.3e-3 * std::exp(-std::pow((x - xb) / w, 2)) * std::cos(x - xb);
                ez(i, j, 0) = g; by(i, j, 0) = -g;
            }
        ez.put(F.ez); by.put(F.by);
        comm.exchange(d, {&F.ez, &F.by});
    }
    auto parts = [&](double* kin, double* we) {
        double v[2] = {0, 0};
        for (const Species& sp : sim.species) {
            v[0] += sp.kinetic_energy();
            if (sp.q < 0) for (double x : sp.get(Species::W)) v[1] += x;
        }
        comm.allreduce_sum(v, 2);
        *kin = v[0]; *we = v[1];
    };
    auto total = [&] {
        double kin, we, w = sim.fields->energy();
        parts(&kin, &we);
        comm.allreduce_sum(&w, 1);
        return w + kin + sim.medium_energy() + sim.medium->dissipated;
    };
    const double e0 = total();
    double g = 0, dev = 0;
    while (!sim.finished()) {
        sim.step();
        g = std::max(g, sim.gauss_drift());
        dev = std::max(dev, std::abs(total() / e0 - 1));
    }
    double kin, we;
    parts(&kin, &we);
    double acc = 0;
    {
        const std::vector<double> a = sim.medium->bacc.host();
        const Domain& d = sim.dom;
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) acc += a[sim.medium->bacc.idx(i, j, k)] * d.cell_volume();
        comm.allreduce_sum(&acc, 1);
    }
    const double ion = sim.medium->ionized, book = ion > 0 ? std::abs(we + acc - ion) / ion : 1;
    double atoms = 0;                                                // sum n_a dV
    {
        const std::vector<double> nb = sim.medium->bn.host();
        const Domain& d = sim.dom;
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) atoms += nb[sim.medium->bn.idx(i, j, k)] * d.cell_volume();
        comm.allreduce_sum(&atoms, 1);
    }
    const double frac = ion / atoms;
    check("medium bound: ionization on the grid with electron and ion species (Rb 1e-4 n_c, a0 1.3e-3, implicit mover)",
          g < 1e-12 && book < 1e-12 && frac > 0.05 && frac <= 1 && dev < 1e-9 && we > 0,
          fmt("ionized %.3f of the atoms", frac) + fmt(", emitted %.3e", we) + fmt(" + waiting %.2e", acc) + fmt(" = ionized to %.1e", book) +
              fmt("; Gauss %.1e", g) + fmt("; W + K + K_atoms + dissipated %.1e", dev) + fmt(" (dissipated %.2e E0)", sim.medium->dissipated / e0));
    if (comm.rank == 0) std::remove("bound_tmp.deck");
}

// B9. RZ (medium_rz_bound.cpp): (a) the energy identity with random fields in modes 0..2 (amplitude 2e-3: Rabi frequencies of the
//     D lines up to ~1 omega0), Rb atoms (5e-2 n_c, n_phi = 12) and random linear media on a stretched radial grid:
//     W + K + K_atoms + dissipated constant to the solve tolerances (the angular quadrature and the transposed deposit);
//     (b) Rabi flopping in a uniform x-polarised pulse (m = 1: E_r,1 = E/2, E_phi,1 = i E/2) at atoms on all angles and radii,
//     including the first half node (the axis rule E_phi,1(0) = i E_r,1(rh_0)), against the Schroedinger equation in the
//     analytic field at the atom; (c) a deck with ionization into species (m = 0, 1): Gauss law, charge bookkeeping.
static void bound_rz(Comm& comm) {
    {   // (a)
        const int M = 3;
        Domain d = rz_domain(comm, 10, 14, M, 0.4, 0.16, true, Solver::IRip);
        FieldsRZ F(d, M, rz_radial_grid(14, 0.16, 0.8, 1.15));
        std::mt19937_64 rng(23);
        std::normal_distribution<double> nd;
        std::uniform_real_distribution<double> ud;
        for (Field* f : {&F.ez, &F.er, &F.ep, &F.bz, &F.br, &F.bp}) {
            H h(*f);
            for (int k = 0; k < d.n[2]; ++k)
                for (int j = 0; j < d.n[1]; ++j)
                    for (int i = 0; i < d.n[0]; ++i) {
                        double v = 2e-3 * nd(rng);
                        if (j == 0 && ((f == &F.ez && k > 0) || f == &F.ep || f == &F.br)) v = 0;
                        h(i, j, k) = v;
                    }
            h.put(*f);
        }
        comm.exchange(d, {&F.ez, &F.er, &F.ep, &F.bz, &F.br, &F.bp});
        const RadialGrid& g = F.grid();
        Medium Md;
        MediumConfig m = rb(5e-2, 0);
        m.profile = Expr("r < 1.4", {"z", "r"}, no_units);
        MediumConfig gl;                                             // a Lorentz layer overlapping the atoms
        gl.density = 2; gl.w0 = 2;
        gl.profile = Expr("r < 0.8 && z > 1", {"z", "r"}, no_units);
        medium_build(Md, d, {m, gl}, 2, g.r.data(), g.rh.data());
        medium_bound_rz_setup(Md, d, g.view(), M, 12);
        Md.tol = 1e-13; Md.btol = 1e-11;
        Md.weights.resize(size_t(3 * d.n[1]));
        for (int j = 0; j < d.n[1]; ++j) {
            Md.weights[size_t(j)] = Md.weights[size_t(2 * d.n[1] + j)] = 2 * M_PI * g.A[size_t(j)] * d.h[0];
            Md.weights[size_t(d.n[1] + j)] = 2 * M_PI * g.B[size_t(j)] * d.h[0];
        }
        F.medium = &Md;
        auto tot = [&](double* kb) {
            double v[3] = {F.energy(), Md.energy(d, &Md.weights), medium_bound_rz_energy(Md, d, g.view())};
            comm.allreduce_sum(v, 3);
            if (kb) *kb = v[2];
            return v[0] + v[1] + v[2] + Md.dissipated;
        };
        const double e0 = tot(nullptr);
        double worst = 0, kmax = 0;
        for (int n = 0; n < 30; ++n) {
            F.step(comm);
            double kb;
            worst = std::max(worst, std::abs(tot(&kb) / e0 - 1));
            kmax = std::max(kmax, kb / e0);
        }
        const MediumStats& st = Md.stats;
        check("medium bound RZ: W + K + K_atoms, random fields in modes 0..2, Rb 5e-2 n_c on 12 angles plus a Lorentz layer, stretched grid",
              worst < 1e-10 && st.unconverged == 0 && st.newton_unconverged == 0 && kmax > 1e-5,
              fmt("max |E/E0 - 1| %.1e", worst) + fmt(", K_atoms up to %.1e E0", kmax) +
                  fmt("; outer iterations %.2f per step", double(st.newton) / st.newton_steps) + fmt(" (max %.0f)", double(st.newton_max)) +
                  fmt(", GMRES %.1f per solve", double(st.iterations) / st.solves));
    }
    {   // (b)
        const double lam = 2 * M_PI, dz = lam / 32, tau = 8, zb = 30, za = 60, a = 0.133;
        const int nz = int(std::round(100 / dz));
        Domain d = rz_domain(comm, nz, 360, 2, dz, 0.2, true, Solver::IRip);   // uniform dr, R = 72 > c t: the wall never reaches the atoms
        FieldsRZ F(d, 2, rz_radial_grid(360, 0.2));
        const RadialGrid& g = F.grid();
        std::string info;
        bool ok = true;
        for (double area : {0.5, 1.0, 1.5}) {
            const double E0 = area * M_PI / (a * tau * std::sqrt(M_PI));
            auto fld = [&](double z) { return E0 * std::exp(-std::pow((z - zb) / tau, 2)) * std::cos(z - zb); };
            for (Field* f : {&F.ez, &F.er, &F.ep, &F.bz, &F.br, &F.bp}) f->fill(0);
            {
                H er(F.er), ep(F.ep), br(F.br), bp(F.bp);                 // slot 1: Re of m = 1, slot 2: Im
                for (int j = 0; j < d.n[1]; ++j)
                    for (int i = 0; i < d.n[0]; ++i) {
                        const double z = d.lo(0) + i * dz, e = 0.5 * fld(z);
                        er(i, j, 1) = e; bp(i, j, 1) = e;                    // E_r,1 = E/2, B_phi,1 = B/2 (B_y = E_x)
                        if (j > 0) { ep(i, j, 2) = e; br(i, j, 2) = -e; }    // E_phi,1 = i E/2, B_r,1 = -i B/2
                    }
                er.put(F.er); ep.put(F.ep); br.put(F.br); bp.put(F.bp);
            }
            comm.exchange(d, {&F.ez, &F.er, &F.ep, &F.bz, &F.br, &F.bp});
            Medium Md;
            MediumConfig m;
            m.bound = true; m.lines = 1; m.bw[0] = 1; m.bf[0] = 1; m.bd[0] = a / 137.035999084; m.axis = 0; m.density = 1e-10;
            m.a0k0 = A0K0; m.supersample = 1;
            char t[96];
            std::snprintf(t, sizeof t, "z > %.17g && z < %.17g && r < 1", za, za + 0.5 * lam);
            m.profile = Expr(t, {"z", "r"}, no_units);
            medium_build(Md, d, {m}, 2, g.r.data(), g.rh.data());
            medium_bound_rz_setup(Md, d, g.view(), 2, 8);
            Md.weights.assign(size_t(3 * d.n[1]), 1.0);
            F.medium = &Md;
            const int steps = int(std::round((za + 0.5 * lam - zb + 4 * tau) / dz));
            for (int n = 0; n < steps; ++n) F.step(comm);
            // the atoms: the excited population against the reference in E(t) = g(z_atom - t)
            const std::vector<double> p1 = Md.ra[BS_P1].host(), nb = Md.bn.host();
            double worst = 0, ref_last = 0, wnode = 0, pnode = 0, waxis = 0;
            int count = 0;
            for (int i = 0; i < d.n[0]; ++i) {
                if (nb[Md.bn.idx(i, 0, 0)] == 0) continue;
                const double zn = d.lo(0) + (i + 0.5) * dz;
                const int nr2 = int(std::round(steps * dz / (2 * M_PI / 1600)));
                const double h = steps * dz / nr2;
                cd c0 = 1, c1 = 0;
                auto rhs = [&](double tt, cd x0, cd x1, cd& d0, cd& d1) {
                    const double e = fld(zn - tt);
                    d0 = cd(0, 1) * (a * e * x1);
                    d1 = cd(0, 1) * (a * e * x0 - x1);
                };
                for (int q = 0; q < nr2; ++q) {
                    const double tt = q * h;
                    cd k0a, k0b, k1a, k1b, k2a, k2b, k3a, k3b;
                    rhs(tt, c0, c1, k0a, k0b);
                    rhs(tt + h / 2, c0 + h / 2 * k0a, c1 + h / 2 * k0b, k1a, k1b);
                    rhs(tt + h / 2, c0 + h / 2 * k1a, c1 + h / 2 * k1b, k2a, k2b);
                    rhs(tt + h, c0 + h * k2a, c1 + h * k2b, k3a, k3b);
                    c0 += h / 6 * (k0a + 2.0 * k1a + 2.0 * k2a + k3a);
                    c1 += h / 6 * (k0b + 2.0 * k1b + 2.0 * k2b + k3b);
                }
                const double ref = std::norm(c1);
                {                                                   // the node step alone in the analytic midpoint field
                    const double one = 1, dd = a / 137.035999084;
                    const BlochC L = bloch_constants(1, &one, &dd, A0K0, dz);
                    double s[BS_N] = {1, 0, 0, 0, 0, 0, 0, 0, 0};
                    for (int n = 0; n < steps; ++n) bloch_step(L, s, fld(zn - (n + 0.5) * dz), s);
                    pnode = s[BS_P1];
                }
                for (int k = 0; k < Md.nphi; ++k)
                    for (int j = 0; j < d.n[1]; ++j) {
                        if (nb[Md.bn.idx(i, j, 0)] == 0) continue;
                        const size_t q = (size_t(k) * d.n[1] + j) * d.n[0] + i;
                        worst = std::max(worst, std::abs(p1[q] - ref));
                        wnode = std::max(wnode, std::abs(p1[q] - pnode));
                        if (j == 0) waxis = std::max(waxis, std::abs(p1[q] - pnode));
                        ++count;
                    }
                ref_last = ref;
            }
            double v[2] = {worst, double(count)};
            v[0] = comm.allreduce_max(v[0]);
            wnode = comm.allreduce_max(wnode); waxis = comm.allreduce_max(waxis);
            comm.allreduce_sum(&v[1], 1);
            ok = ok && wnode < 1e-5 && v[1] > 0;
            info += fmt(" area %.1f pi:", area) + fmt(" ref %.5f,", ref_last) + fmt(" max |p1 - ref| %.1e", v[0]) +
                    fmt(" (the node step alone %.1e;", std::abs(pnode - ref_last)) + fmt(" grid vs node step %.1e", wnode) + fmt(", first half node %.1e)", waxis) + fmt(" over %.0f atoms;", v[1]);
        }
        check("medium bound RZ: Rabi flopping in a uniform x-polarised pulse (m = 1) on all angles and radii vs Schroedinger", ok, info);
    }
    {   // (c)
        const std::string t = R"(grid {
 geometry = rz
 modes = 2
 lambda0 = 0.8 um
 nz = 200
 nr = 24
 dz = 0.025 um
 dr = 0.05 um
 boundary_z = open
 t_end = 14 fs
}
numerics {
 solver = irip
 shape = 2
 medium_tol = 1e-12
 medium_bound_tol = 1e-10
}
laser {
 a0 = 1.5e-3
 polarization = x
 duration = 4 fs
 delay = 5 fs
 waist = 0.8 um
}
species electrons {
 charge = -1
 mass = 1
 density = 0
}
species rb_ions {
 charge = 1
 mass = 156700
 density = 0
}
medium rb {
 model = bound
 atom = Rb
 density = 1e-3
 ionization = adk
 electrons = electrons
 ions = rb_ions
 ionization_ppc = 4
 n_phi = 8
 profile = z > 2 um && z < 3 um && r < 0.8 um
}
output {
 dir = medium_tmp_out
}
)";
        Config c = deck_config(t, "bound_tmp.deck");
        Simulation sim(c, comm);
        sim.init();
        double gd = 0;
        while (!sim.finished()) {
            sim.step();
            gd = std::max(gd, sim.gauss_drift());
        }
        double we = 0, acc = 0;
        for (const Species& sp : sim.species)
            if (sp.q < 0) for (double x : sp.get(Species::W)) we += x;
        {
            const Medium& Md = *sim.medium;
            const std::vector<double> a = Md.ra[RA_ACC].host(), nb = Md.bn.host();
            const RadialGrid& g = sim.rz->grid();
            const Domain& d = sim.dom;
            for (int k = 0; k < Md.nphi; ++k)
                for (int j = 0; j < d.n[1]; ++j)
                    for (int i = 0; i < d.n[0]; ++i) {
                        const size_t q = (size_t(k) * d.n[1] + j) * d.n[0] + i;
                        acc += a[q] * 2 * M_PI * g.B[size_t(j)] * d.h[0] / Md.nphi;
                    }
        }
        double v[2] = {we, acc};
        comm.allreduce_sum(v, 2);
        const double ion = sim.medium->ionized, book = ion > 0 ? std::abs(v[0] + v[1] - ion) / ion : 1;
        check("medium bound RZ: deck with ionization into species (Rb 1e-3 n_c, laser a0 1.5e-3, m = 0, 1): Gauss law, charge bookkeeping",
              gd < 1e-12 && book < 1e-12 && v[0] > 0 && sim.medium->stats.newton_unconverged == 0,
              fmt("Gauss %.1e", gd) + fmt(", ionized %.3e", ion) + fmt(" = emitted %.3e", v[0]) + fmt(" + waiting %.2e", v[1]) +
                  fmt(" to %.1e", book) + fmt("; outer iterations %.2f per step", double(sim.medium->stats.newton) / sim.medium->stats.newton_steps));
        if (comm.rank == 0) std::remove("bound_tmp.deck");
    }
}

// B10. Moving window with media (Simulation::shift_window_medium): the same run in a window (velocity c) and in a fixed box long
//     enough to hold it, compared cell by cell at the end (absolute index = window index + wx): the fields, the atoms' excited
//     population and the linear medium's polarization; the media (Rb atoms 1e-3 n_c saturated by a0 1e-3, a Lorentz glass layer)
//     enter through the right edge of the window. 2D (s-pol plane pulse) and RZ (an m = 1 Gaussian pulse; atoms on 8 angles).
//     The Gauss law in the window run (the iRIP entry cell with the media charge).
namespace {
struct WinRun { std::vector<double> ez, p1, pm; long wx; int nx, ny, np; double gauss; };
WinRun win_run(Comm& comm, bool rz, bool window) {
    const int nx0 = rz ? 320 : 400, extra = rz ? 300 : 300;
    std::string t;
    if (!rz)
        t = std::string("grid {\n geometry = cartesian2d\n lambda0 = 0.8 um\n nx = ") + std::to_string(window ? nx0 : nx0 + extra) +
            "\n ny = 4\n dx = 0.03333333333333 um\n dy = 0.03333333333333 um\n boundary_x = open\n t_end = 30 fs\n}\n"
            "numerics {\n solver = irip\n medium_tol = 1e-13\n medium_bound_tol = 1e-11\n}\n"
            "medium rb {\n model = bound\n atom = Rb\n density = 1e-3\n polarization = z\n profile = x > 8 um\n}\n"
            "medium glass {\n model = lorentz\n density = 4\n resonance = 3\n profile = x > 11 um && x < 12 um\n}\n";
    else
        t = std::string("grid {\n geometry = rz\n modes = 2\n lambda0 = 0.8 um\n nz = ") + std::to_string(window ? nx0 : nx0 + extra) +
            "\n nr = 40\n dz = 0.025 um\n dr = 0.05 um\n boundary_z = open\n t_end = 18 fs\n}\n"
            "numerics {\n solver = irip\n medium_tol = 1e-13\n medium_bound_tol = 1e-11\n}\n"
            "medium rb {\n model = bound\n atom = Rb\n density = 1e-3\n n_phi = 8\n profile = z > 6 um && r < 1.2 um\n}\n"
            "medium glass {\n model = lorentz\n density = 4\n resonance = 3\n profile = z > 7.5 um && z < 8 um && r < 1 um\n}\n";
    if (window) t += "window {\n start = 0 fs\n velocity = 1\n}\n";
    t += "output {\n dir = medium_tmp_out\n}\n";
    Config c = deck_config(t, "bound_tmp.deck");
    Simulation sim(c, comm);
    sim.init();
    const Domain& d = sim.dom;
    const double xb = (rz ? 3e-6 : 4e-6) * K0, w = 1e-6 * K0, wr = 1.2e-6 * K0;
    if (!rz) {
        Fields& F = *sim.fields;
        H ez(F.ez), by(F.by);
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                const double x = d.lo(0) + i * d.h[0], g = 1e-3 * std::exp(-std::pow((x - xb) / w, 2)) * std::cos(x - xb);
                ez(i, j, 0) = g; by(i, j, 0) = -g;
            }
        ez.put(F.ez); by.put(F.by);
        comm.exchange(d, {&F.ez, &F.by});
    } else {
        FieldsRZ& F = *sim.rz;
        const RadialGrid& g = F.grid();
        H er(F.er), ep(F.ep), br(F.br), bp(F.bp);
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                const double z = d.lo(0) + i * d.h[0], env = 0.5e-3 * std::exp(-std::pow((z - xb) / w, 2)) * std::cos(z - xb);
                const double fh = std::exp(-std::pow(g.rh[size_t(j)] / wr, 2)), fn = std::exp(-std::pow(g.r[size_t(j)] / wr, 2));
                er(i, j, 1) = env * fh; bp(i, j, 1) = env * fh;
                if (j > 0) { ep(i, j, 2) = env * fn; br(i, j, 2) = -env * fn; }
            }
        er.put(F.er); ep.put(F.ep); br.put(F.br); bp.put(F.bp);
        comm.exchange(d, {&F.ez, &F.er, &F.ep, &F.bz, &F.br, &F.bp});
    }
    if (!rz) { sim.compute_rho(); sim.fields->gauss_residual(sim.gauss0()); }   // the Gauss residual of the initial pulse
    else sim.rz_gauss_residual(sim.gauss0());
    double gd = 0;
    while (!sim.finished()) {
        sim.step();
        gd = std::max(gd, sim.gauss_drift());
    }
    WinRun r;
    r.wx = d.wx; r.nx = d.n[0]; r.ny = d.n[1]; r.gauss = gd;
    const Medium& M = *sim.medium;
    if (!rz) {
        r.ez = sim.fields->ez.host();
        r.p1 = M.bs[BS_P1].host();
        r.pm = M.P[2].host();
        r.np = 1;
    } else {
        r.ez = sim.rz->er.host();                                   // E_r, all slots
        r.p1 = M.ra[BS_P1].host();
        r.pm = M.P[1].host();
        r.np = M.nphi;
    }
    if (comm.rank == 0) std::remove("bound_tmp.deck");
    return r;
}
}  // namespace

static void bound_window(Comm& comm) {
    for (int rz = 0; rz < 2; ++rz) {
        const WinRun a = win_run(comm, rz, true), b = win_run(comm, rz, false);
        // compare the window box with the long box at the same absolute cells (Field layout: idx of a Field of each domain)
        const int ng = 3, gz = rz ? 3 : 0, S = rz ? 3 : 1;
        auto idx = [&](int nx, int ny, int i, int j, int k) {
            const int sx = nx + 2 * ng, sy = ny + 2 * ng;
            return (size_t(k + gz) * sy + (j + ng)) * sx + (i + ng);
        };
        double de = 0, me = 0, dp = 0, mp = 0, dm = 0, mm = 0, di = 0;   // di: the field 20 cells from the edges
        for (int k = 0; k < S; ++k)
            for (int j = 0; j < a.ny; ++j)
                for (int i = 0; i < a.nx; ++i) {
                    const size_t qa = rz ? idx(a.nx, a.ny, i, j, k) : idx(a.nx, a.ny, i, j, 0) , qb = rz ? idx(b.nx, b.ny, i + int(a.wx), j, k) : idx(b.nx, b.ny, i + int(a.wx), j, 0);
                    if (!rz && k > 0) continue;
                    de = std::max(de, std::abs(a.ez[qa] - b.ez[qb])); me = std::max(me, std::abs(b.ez[qb]));
                    if (i >= 20 && i < a.nx - 20) di = std::max(di, std::abs(a.ez[qa] - b.ez[qb]));
                    dm = std::max(dm, std::abs(a.pm[qa] - b.pm[qb])); mm = std::max(mm, std::abs(b.pm[qb]));
                }
        if (!rz) {
            for (int j = 0; j < a.ny; ++j)
                for (int i = 0; i < a.nx; ++i) {
                    const size_t qa = idx(a.nx, a.ny, i, j, 0), qb = idx(b.nx, b.ny, i + int(a.wx), j, 0);
                    dp = std::max(dp, std::abs(a.p1[qa] - b.p1[qb])); mp = std::max(mp, std::abs(b.p1[qb]));
                }
        } else {
            for (int k = 0; k < a.np; ++k)
                for (int j = 0; j < a.ny; ++j)
                    for (int i = 0; i < a.nx; ++i) {
                        const size_t qa = (size_t(k) * a.ny + j) * a.nx + i, qb = (size_t(k) * b.ny + j) * b.nx + i + size_t(a.wx);
                        dp = std::max(dp, std::abs(a.p1[qa] - b.p1[qb])); mp = std::max(mp, std::abs(b.p1[qb]));
                    }
        }
        const double re = de / me, ri = di / me, rp = dp / std::max(mp, 1e-300), rm = dm / std::max(mm, 1e-300);
        const double tol = rz ? 1e-6 : 1e-9;
        check(std::string("medium bound: moving window vs a long fixed box, ") + (rz ? "RZ (m = 1 pulse, atoms on 8 angles)" : "2D") +
                  ", Rb atoms and a Lorentz layer entering at the right edge",
              (rz ? ri : re) < tol && rp < tol && rm < tol && a.gauss < 1e-12 && a.wx > 100 && mp > 1e-3,
              fmt("%.0f shifts;", double(a.wx)) + fmt(" max |dE| %.1e", re) + fmt(" (interior %.1e)", ri) + fmt(", |dp1| %.1e", rp) + fmt(" (p1 up to %.2f)", mp) +
                  fmt(", |dP_glass| %.1e of the maximum", rm) + fmt("; Gauss in the window %.1e", a.gauss));
    }
}

void test_bound(Comm& comm) {
    if (comm.size == 1) bound_ionize_node();
    bound_ionize_grid(comm);
    bound_rz(comm);
    if (comm.size == 1) bound_window(comm);
    if (comm.size == 1) bound_relax_node();
    if (comm.size == 1) bound_rabi_node();
    bound_rabi_grid(comm);
    bound_deck_gauss(comm);
    bound_energy_strong(comm, false);
    bound_energy_strong(comm, true);
    if (comm.size > 1) return;
    bound_weak_field(comm);
}
