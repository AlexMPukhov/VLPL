// Self-field of a relativistic beam at the start (docs/specs/beam_self_field.md).
#include <cmath>
#include <cstdio>
#include "common.hpp"
#include "sim.hpp"

static Config beam_config(int nx, int ny, double u0, const char* profile, double xmin, double xmax) {
    Config c;
    c.N[0] = nx; c.N[1] = ny; c.N[2] = ny;
    c.h[0] = 0.1; c.h[1] = 0.25; c.h[2] = 0.25;
    c.periodic[0] = false;
    SpeciesConfig b;
    b.name = "beam"; b.q = -1; b.m = 1; b.density = 0.01;
    b.ppc[0] = 1; b.ppc[1] = 2; b.ppc[2] = 2;
    for (int a = 0; a < 3; ++a) { b.rmin[a] = -1e300; b.rmax[a] = 1e300; }
    b.rmin[0] = xmin; b.rmax[0] = xmax;
    b.u0[0] = u0;
    b.density_fn = Expr(profile, {"x", "y", "z"}, nullptr);
    b.self_field = true;
    c.species.push_back(b);
    return c;
}

// max |Gauss residual| over the checked cells and max |rho|
static void gauss_now(Simulation& sim, double& g, double& r) {
    sim.compute_rho();
    Field res(sim.dom);
    sim.fields->gauss_residual(res);
    H R(res), Q(sim.fields->rho);
    g = r = 0;
    for (int k = 0; k < sim.dom.n[2]; ++k)
        for (int j = 0; j < sim.dom.n[1]; ++j)
            for (int i = 0; i < sim.dom.n[0]; ++i) { g = std::max(g, std::abs(R(i, j, k))); r = std::max(r, std::abs(Q(i, j, k))); }
    g = sim.comm.allreduce_max(g); r = sim.comm.allreduce_max(r);
}

// B1. After the init the Gauss law holds to round-off for a non-neutral beam alone, moving (u = 20, 2000:
//     the k_perp = 0 mode is solved apart, else the residual grew as gamma^2, 6e-9 at u = 2000) and at
//     rest (the electrostatic Poisson initialisation); without self_field the residual is -rho.
static void beam_gauss(Comm& comm) {
    for (double u0 : {20.0, 2000.0, 0.0})
        for (int sf : {1, 0}) {
            if (u0 > 100 && !sf) continue;
            Config c = beam_config(32, 16, u0, "gauss(x - 1.6, 0.4) * gauss(y - 2, 0.6) * gauss(z - 2, 0.6)", -1e300, 1e300);
            c.species[0].self_field = sf;
            c.check();
            Simulation sim(c, comm);
            sim.init();
            double g, r;
            gauss_now(sim, g, r);
            if (!sf) {
                check(std::string("beam field, u = ") + (u0 > 0 ? "20" : "0") + ", no self_field: residual = -rho",
                      std::abs(g - r) < 1e-12 * r, fmt("max|res| %.3e, max|rho| %.3e", g, r));
                continue;
            }
            check(std::string("beam field, u = ") + (u0 > 100 ? "2000" : u0 > 0 ? "20" : "0 (Poisson)") + ": Gauss law at init",
                  g < 1e-12 * r && r > 0, fmt("max|res| / max|rho| = %.1e", g / r));
        }
}

// B2. A long round Gaussian beam (sigma = 1, gamma = 10, x in [2, 10.8]): in the middle slice the radial
//     field is Q(r) / (2 pi r) of a line charge minus the uniform compensation of the periodic y, z box
//     (the k_perp = 0 part of rho drives no transverse field, only the uniform sheet field Ex behind
//     the beam): lambda r / (2 A); Ex does not vary across the slice.
static void beam_radial(Comm& comm) {
    const double s = 1.0, yc = 8.0;
    Config c = beam_config(128, 64, std::sqrt(99.0), "(sqrt((y-8)^2 + (z-8)^2) < 4.5) * exp(-((y-8)^2 + (z-8)^2) / 2)", 2.0, 10.8);
    c.check();
    Simulation sim(c, comm);
    sim.init();
    double worst = 0;
    const int im = 64 - sim.dom.off[0], kc = 32 - sim.dom.off[2];
    bool mine = im >= 0 && im < sim.dom.n[0] && kc >= 0 && kc < sim.dom.n[2];
    H Ey(sim.fields->ey), Ex(sim.fields->ex);
    double exmax = 0, eymax = 0;
    const double lam = -c.species[0].density * 2 * M_PI * s * s, A = 16.0 * 16.0;
    for (int jg : {35, 36, 39, 40, 44}) {
        const int j = jg - sim.dom.off[1];
        if (!mine || j < 0 || j >= sim.dom.n[1]) continue;
        const double r = (jg + 0.5) * c.h[1] - yc;
        const double want = lam * (1 - std::exp(-r * r / (2 * s * s))) / (2 * M_PI * r) - lam * r / (2 * A);
        worst = std::max(worst, std::abs(Ey(im, j, kc) - want) / std::abs(want));
        eymax = std::max(eymax, std::abs(Ey(im, j, kc)));
        exmax = std::max(exmax, std::abs(Ex(im, j, kc) - Ex(im, 0, 0)));   // the k_perp = 0 sheet part is uniform
    }
    worst = comm.allreduce_max(worst);
    eymax = comm.allreduce_max(eymax);
    exmax = comm.allreduce_max(exmax);
    check("beam field: E_r of a long round beam = line charge", worst < 0.02 && eymax > 0 && exmax < 1e-3 * eymax,
          fmt("max rel err %.2e (r = 1.1 .. 3.1 sigma)", worst) + fmt(", transverse dEx / E_r %.1e in the middle", exmax / eymax));
}

// B3. The beam (gamma ~ 50, centre x = 3) in vacuum for 60 steps. Its near field (+-8 cells around the
//     beam) moves with it, one cell per step: with the steady init (decision 99) it changes by 1.4e-3 (beta = 0.9998 is
//     not one cell per step; the Gauss-law init: 1.0e-2 at h_x = 0.1, 1.6e-3 at 0.05, the O(h_x^2) offset from the discrete
//     steady state); the Gauss law stays at round-off. Without self_field the near field
//     has to build up from zero (change ~ 1). (The global field energy is no measure here: with
//     periodic y, z the long transverse wavelengths reach the x edges, gamma / k_perp ~ 64.)
static void beam_vacuum(Comm& comm) {
    std::vector<double> e0;
    for (int sf : {1, 0}) {
        Config c = beam_config(128, 32, 50.0, "(sqrt((y-4)^2 + (z-4)^2) < 2.4) * gauss(x - 3, 0.5) * gauss(sqrt((y-4)^2 + (z-4)^2), 0.6)",
                               -1e300, 1e300);
        c.species[0].self_field = sf;
        c.t_end = 60 * c.h[0];
        c.check();
        Simulation sim(c, comm);
        sim.init();
        if (sf) e0 = sim.fields->ey.host();               // the steady near field, reference for both runs
        const int i0 = 30;
        double m0 = 0, g = 0, r = 0;
        for (int k = 0; k < 32; ++k)
            for (int j = 0; j < 32; ++j)
                for (int i = i0 - 8; i <= i0 + 8; ++i) m0 = std::max(m0, std::abs(e0[sim.fields->ey.idx(i, j, k)]));
        while (!sim.finished()) { sim.step(); g = std::max(g, sim.gauss_drift()); }
        const std::vector<double> e = sim.fields->ey.host();
        const int n = int(sim.istep);
        double d = 0;
        for (int k = 0; k < 32; ++k)
            for (int j = 0; j < 32; ++j)
                for (int i = i0 - 8; i <= i0 + 8; ++i)
                    d = std::max(d, std::abs(e[sim.fields->ey.idx(i + n, j, k)] - e0[sim.fields->ey.idx(i, j, k)]));
        double gg;
        gauss_now(sim, gg, r);
        if (sf)
            check("beam field: gamma 50 in vacuum, near field steady, Gauss", d < 2e-2 * m0 && g < 1e-12 * r,
                  fmt("near-field change %.1e, max|dGauss|/max|rho| %.1e", d / m0, g / r));
        else
            check("beam field: without self_field the near field is far off", d > 0.2 * m0,
                  fmt("near-field change %.2f", d / m0));
    }
}

// B4. External injection: a beam (u = 100, sigma_x 3 cells, sigma_y 2.5 cells) loaded at t_start into a
//     moving window with a laser wake in a cold neutral plasma. No beam particle before t_start (the
//     window does not inject it), the beam present afterwards with a constant count, and the Gauss law
//     kept (its self-field adds to the fields present) up to the discrete field tail of the beam at the
//     right edge: 1e-12 here, 3e-10 for a beam of one cell (docs/specs/beam_self_field.md).
static void beam_delayed(Comm& comm) {
    Config c;
    c.N[0] = 96; c.N[1] = 32; c.N[2] = 4;
    c.h[0] = 0.25; c.h[1] = 0.6; c.h[2] = 1.0;
    c.periodic[0] = false;
    c.order = 2;
    c.laser.enabled = true;
    c.laser.a0 = 1; c.laser.tau = 2; c.laser.tc = 6; c.laser.w0 = 4; c.laser.yc = 0.5 * c.N[1] * c.h[1];
    c.window.enabled = true; c.window.start = 2 * c.laser.tc;
    c.t_end = c.window.start + 12;
    for (int s = 0; s < 2; ++s) {
        SpeciesConfig e;
        e.name = s ? "p" : "e"; e.q = s ? 1 : -1; e.m = s ? 1836 : 1; e.density = 0.02;
        e.ppc[0] = 2;
        for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
        c.species.push_back(e);
    }
    SpeciesConfig b;
    b.name = "beam"; b.q = -1; b.m = 1; b.density = 0.05; b.u0[0] = 100;
    b.ppc[0] = 2; b.ppc[1] = 2;
    for (int a = 0; a < 3; ++a) { b.rmin[a] = -1e300; b.rmax[a] = 1e300; }
    b.density_fn = Expr("gauss(x - 7, 0.75) * gauss(y - 9.6, 1.5) * (abs(x - 7) < 3.75) * (abs(y - 9.6) < 7.5)", {"x", "y", "z"}, nullptr);
    b.self_field = true;
    b.t_start = 16;
    c.species.push_back(b);
    c.check();
    Simulation sim(c, comm);
    sim.init();
    bool early = false;
    long n_at = -1, n_end = 0;
    double g = 0;
    while (!sim.finished()) {
        sim.step();
        g = std::max(g, sim.gauss_drift());
        const long nb = long(sim.species[2].size());
        if (sim.t < 16 - 0.5 * c.h[0] && nb > 0) early = true;
        if (n_at < 0 && nb > 0) n_at = nb;
        n_end = nb;
    }
    check("beam loaded at t_start in the window: Gauss law, count", !early && n_at > 0 && n_end == n_at && g < 1e-11,
          fmt("%.0f beam particles, max|dGauss| %.1e", double(n_end), g));
}

// B5. Start-up in a moving window at large gamma (decision 95): a beam alone (u = 1e4, sigma_x 6 cells, 12 cells ahead of
//     the right edge) in a window moving from t = 0. Its self-field has exactly zero fields ahead of the beam (every mode
//     with gamma h_x k_perp > 2 is the local slice solution), so the first shifts cut nothing: max|dGauss| over 12 steps
//     at round-off (before: 1.8e-4 of max rho, the zigzag tail of the boosted Poisson operator cut at the right edge,
//     bench/qed_pwfa 7e-5 .. 1.2e-4). And the long round beam at u = 1e4: E_r = line charge (as B2) to 1 %.
//     The steady init (decision 99, the default) is checked as well: its field ahead of the beam is not exactly zero (the
//     discrete steady state of RIP-CC has a tail ahead, measured 8.5e-11 of max|E|), and the first shifts cut it
//     (measured 2.7e-12 of max rho over 12 shifts). iRIP keeps the unclamped Gauss-law field (decision 99: the clamp is
//     for explicit RIP-CC only): its tail ahead reaches the right edge (1.6e-4 of max|E|), but the iRIP shifts keep the
//     Gauss law (2.6e-14 over 12 shifts).
static void beam_window(Comm& comm) {
  for (int st = 0; st < 3; ++st) {               // 0: RIP-CC, Gauss-law init; 1: RIP-CC, steady init; 2: iRIP
    Config c = beam_config(96, 32, 1e4, "(sqrt((y-4)^2 + (z-4)^2) < 2.4) * gauss(x - 5, 0.6) * gauss(sqrt((y-4)^2 + (z-4)^2), 0.6) * (abs(x - 5) < 2.4)",
                           -1e300, 1e300);
    c.window.enabled = true; c.window.start = 0;
    c.beam_steady = st == 1;                         // 0, 2: the Gauss-law field (clamped for RIP-CC, decision 95)
    if (st == 2) c.solver = Solver::IRip;
    c.t_end = 12 * c.h[0];
    c.check();
    Simulation sim(c, comm);
    sim.init();
    // the fields ahead of the beam (x > 5 + 2.4 + 2 h_x: beyond the deposit of its last particles) at the init
    double ahead = 0, emax = 0;
    {
        const std::vector<double> ey = sim.fields->ey.host(), ex = sim.fields->ex.host();
        for (int k = 0; k < sim.dom.n[2]; ++k)
            for (int j = 0; j < sim.dom.n[1]; ++j)
                for (int i = 0; i < sim.dom.n[0]; ++i) {
                    const size_t q = sim.fields->ey.idx(i, j, k);
                    const double v = std::max(std::abs(ey[q]), std::abs(ex[q]));
                    emax = std::max(emax, v);
                    if ((sim.dom.off[0] + i) * c.h[0] > 5 + 2.4 + 2 * c.h[0]) ahead = std::max(ahead, v);
                }
        ahead = comm.allreduce_max(ahead); emax = comm.allreduce_max(emax);
    }
    double g = 0, gg, r;
    while (!sim.finished()) { sim.step(); g = std::max(g, sim.gauss_drift()); }
    gauss_now(sim, gg, r);
    if (!st)
        check("beam field: u = 1e4 in a moving window from t = 0, no field ahead, Gauss", ahead == 0 && g < 1e-12 * r,
              fmt("max|E| ahead %.1e of max|E|", ahead / emax) + fmt(", max|dGauss|/max|rho| %.1e over 12 shifts", g / r));
    else if (st == 2)
        check("beam field: u = 1e4 in a moving window from t = 0, irip (unclamped Gauss-law field): Gauss", g < 1e-12 * r,
              fmt("max|E| ahead %.1e of max|E|", ahead / emax) + fmt(", max|dGauss|/max|rho| %.1e over 12 shifts", g / r));
    else
        check("beam field: u = 1e4 in a moving window from t = 0, steady init: small field ahead, Gauss",
              ahead < 1e-9 * emax && g < 1e-10 * r,
              fmt("max|E| ahead %.1e of max|E|", ahead / emax) + fmt(", max|dGauss|/max|rho| %.1e over 12 shifts", g / r));
  }
}

// B8. The self-field with the other solvers and periodic x (decision 98): the beam of B3 (gamma ~ 50) in vacuum,
//     RIP-CC with periodic x (3D DFT), iRIP open and periodic (RIP-CC's equilibrium: iRIP keeps the same Gauss law),
//     Yee periodic (its own Gauss law, B = beta x E at x_{i+1/2}; dt = h_x / 2 so the beam moves an integer number
//     of cells). Gauss law at the init to round-off (periodic: the residual is the uniform -<rho>: no neutralizing
//     background) and kept while the beam moves 60 cells; the near field moves with the beam and changes by
//     O(h_x^2) as in B3 (Yee: also its numerical dispersion, slower than the beam: Cherenkov emission starts).
static void beam_solvers(Comm& comm) {
    struct Case { const char* name; Solver s; bool px; double bound; };
    const Case cases[] = {{"rip_cc, periodic x", Solver::RipCC, true, 2e-2}, {"irip, open x", Solver::IRip, false, 2e-2},
                          {"irip, periodic x", Solver::IRip, true, 2e-2}, {"yee, periodic x", Solver::Yee, true, 5e-2}};
    for (const Case& cs : cases) {
        Config c = beam_config(128, 32, 50.0, "(sqrt((y-4)^2 + (z-4)^2) < 2.4) * gauss(x - 3, 0.5) * gauss(sqrt((y-4)^2 + (z-4)^2), 0.6)",
                               -1e300, 1e300);
        c.periodic[0] = cs.px;
        c.solver = cs.s;
        const int sub = cs.s == Solver::Yee ? 2 : 1;                     // steps per cell
        if (cs.s == Solver::Yee) c.cfl = 0.5 * std::sqrt(1 + 2 * (c.h[0] / c.h[1]) * (c.h[0] / c.h[1]));
        c.t_end = 1e9;
        c.check();
        Simulation sim(c, comm);
        sim.init();
        double g0, r;
        {   // the init: residual = -<rho> (periodic x) or 0
            sim.compute_rho();
            Field res(sim.dom);
            sim.fields->gauss_residual(res);
            H R(res), Q(sim.fields->rho);
            double mean = 0, cnt = 0;
            for (int k = 0; k < sim.dom.n[2]; ++k)
                for (int j = 0; j < sim.dom.n[1]; ++j)
                    for (int i = 0; i < sim.dom.n[0]; ++i) { mean += Q(i, j, k); cnt += 1; }
            { double b2[2] = {mean, cnt}; sim.comm.allreduce_sum(b2, 2); mean = cs.px ? b2[0] / b2[1] : 0; }
            g0 = r = 0;
            for (int k = 0; k < sim.dom.n[2]; ++k)
                for (int j = 0; j < sim.dom.n[1]; ++j)
                    for (int i = 0; i < sim.dom.n[0]; ++i) {
                        g0 = std::max(g0, std::abs(R(i, j, k) + mean));
                        r = std::max(r, std::abs(Q(i, j, k)));
                    }
            g0 = sim.comm.allreduce_max(g0); r = sim.comm.allreduce_max(r);
        }
        const std::vector<double> e0 = sim.fields->ey.host();
        const int i0 = 30, n = 60;
        double m0 = 0, g = 0;
        for (int k = 0; k < 32; ++k)
            for (int j = 0; j < 32; ++j)
                for (int i = i0 - 8; i <= i0 + 8; ++i) m0 = std::max(m0, std::abs(e0[sim.fields->ey.idx(i, j, k)]));
        for (int s = 0; s < sub * n; ++s) { sim.step(); g = std::max(g, sim.gauss_drift()); }
        const std::vector<double> e = sim.fields->ey.host();
        double d = 0;
        for (int k = 0; k < 32; ++k)
            for (int j = 0; j < 32; ++j)
                for (int i = i0 - 8; i <= i0 + 8; ++i)
                    d = std::max(d, std::abs(e[sim.fields->ey.idx(i + n, j, k)] - e0[sim.fields->ey.idx(i, j, k)]));
        m0 = comm.allreduce_max(m0); d = comm.allreduce_max(d);
        check(std::string("beam field, ") + cs.name + ": Gauss law, near field steady",
              g0 < 1e-12 * r && g < 1e-12 * r && d < cs.bound * m0,
              fmt("init %.1e, max|dGauss| %.1e of max|rho|", g0 / r, g / r) + fmt(", near-field change %.1e", d / m0));
    }
}

// normalised rms emittance along y (code units 1/k0) of species 0
static double eps_y(Simulation& sim) {
    const std::vector<double> y = sim.species[0].get(Species::Y), u = sim.species[0].get(Species::UY),
                              w = sim.species[0].get(Species::W);
    double s[3] = {0, 0, 0};
    for (size_t p = 0; p < y.size(); ++p) { s[0] += w[p]; s[1] += w[p] * y[p]; s[2] += w[p] * u[p]; }
    sim.comm.allreduce_sum(s, 3);
    const double my = s[1] / s[0], mu = s[2] / s[0];
    double m[3] = {0, 0, 0};
    for (size_t p = 0; p < y.size(); ++p) {
        const double a = y[p] - my, b = u[p] - mu;
        m[0] += w[p] * a * a; m[1] += w[p] * b * b; m[2] += w[p] * a * b;
    }
    sim.comm.allreduce_sum(m, 3);
    return std::sqrt(std::max(0.0, m[0] * m[1] - m[2] * m[2])) / s[0];
}

// B6. Emittance regression (bench/emittance/README.md, decision 98): Fig. 4 of Pukhov, JCP 418 (2020) scaled down. A
//     10 GeV beam (u = 19570), eps_n = 1 nm, sigma = 1 um (cut at 3 sigma), peak 0.761 n_c (lambda0 = 1 um: 2 nC), with its
//     self-field in a periodic box (h_x = 0.1 um, h_y = h_z = 0.15 um, shape 2), 15 um of vacuum. iRIP and RIP-CC keep
//     eps_n,y to 1 % (measured -2e-5, -3e-6; RIP-CC with the steady init of decision 99: with the Gauss-law init it jumped by
//     +6.35 in the first ~10 um, bench/emittance/README.md); Yee (c dt = 0.7 h_x, as in the paper) jumps (measured +19.8).
static void beam_emittance(Comm& comm) {
    const double k0 = 2 * M_PI;                                    // 1/um in code units (lambda0 = 1 um)
    const double sg = 1.0 * k0, L = 15.0 * k0;
    double e0[3] = {0, 0, 0}, e1[3] = {0, 0, 0};
    for (int s = 0; s < 3; ++s) {             // rip_cc, irip, yee
        Config c;
        c.N[0] = 96; c.N[1] = 48; c.N[2] = 48;
        c.h[0] = 0.1 * k0; c.h[1] = 0.15 * k0; c.h[2] = 0.15 * k0;
        c.order = 2;
        if (s == 1) c.solver = Solver::IRip;
        if (s == 2) { c.solver = Solver::Yee; c.cfl = 0.7 * c.h[0] * std::sqrt(1 / (c.h[0] * c.h[0]) + 2 / (c.h[1] * c.h[1])); }
        const double xc = 0.5 * c.N[0] * c.h[0], yc = 0.5 * c.N[1] * c.h[1];
        SpeciesConfig b;
        b.name = "beam"; b.q = -1; b.m = 1; b.density = 0.761;
        for (int a = 0; a < 3; ++a) { b.rmin[a] = -1e300; b.rmax[a] = 1e300; }
        b.u0[0] = 19570; b.uth[1] = b.uth[2] = 1e-3 * k0 / sg;        // eps_n = sigma sigma_u = 1 nm
        char prof[400];
        std::snprintf(prof, sizeof prof, "gauss(x - %.10g, %.10g) * gauss(sqrt((y - %.10g)^2 + (z - %.10g)^2), %.10g) * "
                      "(abs(x - %.10g) < %.10g) * (sqrt((y - %.10g)^2 + (z - %.10g)^2) < %.10g)",
                      xc, sg, yc, yc, sg, xc, 3 * sg, yc, yc, 3 * sg);
        b.density_fn = Expr(prof, {"x", "y", "z"}, nullptr);
        b.self_field = true;
        c.species.push_back(b);
        c.t_end = L;
        c.check();
        Simulation sim(c, comm);
        sim.init();
        e0[s] = eps_y(sim);
        while (!sim.finished()) sim.step();
        e1[s] = eps_y(sim);
    }
    const double grip = e1[0] / e0[0] - 1, girip = e1[1] / e0[1] - 1, gyee = e1[2] / e0[2] - 1;
    check("beam emittance B6 (Fig. 4 scaled): iRIP and RIP-CC (steady init) keep eps_n, Yee jumps",
          std::abs(girip) < 1e-2 && std::abs(grip) < 1e-2 && gyee > 1.0,
          fmt("eps_n,y %.4f nm; growth after 15 um: iRIP %+.2e", e0[0] / (2 * M_PI) * 1e3, girip) +
              fmt(", RIP-CC %+.2e, Yee %+.2e", grip, gyee));
}

// B7. The RIP-CC steady init (numerics.beam_init = steady, decision 99): a rigid beam (u = 1e6, momenta frozen) with its
//     self-field, one real RIP-CC step. Periodic x: the fields after the step equal the initial ones shifted by one cell
//     (the beam moved one cell) to round-off; in a window moving at c from t = 0 they are unchanged in the window, near the
//     beam (the open-x solve is a padded periodic one). The Gauss-law init (beam_init = gauss) with periodic x changes by
//     1.1e-3 (the O(h^2) mismatch of B3). In the window the Gauss-law init is stationary too at this gamma: decision 95
//     clamps every mode with gamma h_x k_perp > 2 to the local slice solution, which the RIP-CC step keeps. Gauss law of
//     the steady init at round-off.
static void beam_steady(Comm& comm) {
    if (comm.size > 1) return;                       // the shifted comparison reads across the whole box
    for (int win = 0; win < 2; ++win)
        for (int st = 1; st >= 0; --st) {
            Config c = beam_config(64, 32, 1e6, "(sqrt((y-4)^2 + (z-4)^2) < 2.4) * gauss(x - 3.2, 0.5) * gauss(sqrt((y-4)^2 + (z-4)^2), 0.6)",
                                   -1e300, 1e300);
            c.periodic[0] = !win;
            c.beam_steady = st;
            c.species[0].rigid_until = 1e9;
            if (win) { c.window.enabled = true; c.window.start = 0; }
            c.t_end = 1e9;
            c.check();
            Simulation sim(c, comm);
            sim.init();
            double g = 0, r = 0;
            {                                            // Gauss residual; periodic x: up to the uniform -<rho> (B8)
                sim.compute_rho();
                Field res(sim.dom);
                sim.fields->gauss_residual(res);
                H R(res), Q(sim.fields->rho);
                double mean = 0, cnt = 0;
                for (int k = 0; k < sim.dom.n[2]; ++k)
                    for (int j = 0; j < sim.dom.n[1]; ++j)
                        for (int i = 0; i < sim.dom.n[0]; ++i) { mean += Q(i, j, k); cnt += 1; }
                mean = win ? 0.0 : mean / cnt;
                for (int k = 0; k < sim.dom.n[2]; ++k)
                    for (int j = 0; j < sim.dom.n[1]; ++j)
                        for (int i = 0; i < sim.dom.n[0]; ++i) {
                            g = std::max(g, std::abs(R(i, j, k) + mean));
                            r = std::max(r, std::abs(Q(i, j, k)));
                        }
            }
            Field* Fp[6] = {&sim.fields->ex, &sim.fields->ey, &sim.fields->ez, &sim.fields->bx, &sim.fields->by, &sim.fields->bz};
            std::vector<double> f0[6];
            for (int a = 0; a < 6; ++a) f0[a] = Fp[a]->host();
            sim.step();
            double dmax = 0, fmax = 0;
            const int n = sim.dom.n[0], ny = sim.dom.n[1], nz = sim.dom.n[2];
            for (int a = 0; a < 6; ++a) {
                const std::vector<double> f1 = Fp[a]->host();
                for (int k = 0; k < nz; ++k)
                    for (int j = 0; j < ny; ++j)
                        for (int i = win ? 12 : 0; i < (win ? n - 12 : n); ++i) {
                            const int i1 = win ? i : (i + 1) % n;     // periodic: the beam moved one cell
                            dmax = std::max(dmax, std::abs(f1[Fp[a]->idx(i1, j, k)] - f0[a][Fp[a]->idx(i, j, k)]));
                            fmax = std::max(fmax, std::abs(f0[a][Fp[a]->idx(i, j, k)]));
                        }
            }
            const std::string name = std::string("beam field B7, ") + (st ? "steady" : "gauss") + " init, " + (win ? "window" : "periodic x");
            if (st)
                check(name + ": stationary under the RIP-CC step, Gauss law", dmax < 1e-12 * fmax && g < 1e-12 * r,
                      fmt("max change after one step %.1e of max|F|", dmax / fmax) + fmt(", Gauss %.1e of max|rho|", g / r));
            else if (!win)
                check(name + ": not stationary (reference)", dmax > 1e-4 * fmax, fmt("max change after one step %.1e of max|F|", dmax / fmax));
            else    // open x at u = 1e6: decision 95 clamps every mode to the local slice solution, which RIP-CC keeps too
                check(name + ": stationary too (the clamped modes of decision 95)", dmax < 1e-12 * fmax,
                      fmt("max change after one step %.1e of max|F|", dmax / fmax));
        }
}

// B9. The Gauss-law init (decisions 24, 95) against the steady init (decision 99) in a window moving at c from t = 0,
//     for u = 5 .. 1e6: max difference of the initial fields (the mismatch that the RIP-CC step carries with the beam) and
//     the change of each init in one step (both move with the beam: beta < 1 slips (1 - beta) cells per step). Measured:
//     difference 1.0e-2 (u = 5), 5.9e-3 (20), 8.8e-4 (60), 3.2e-12 (1e6) of max|F|: decision 95 makes the inits equal only
//     where every mode is clamped (gamma h_x k_perp > 2); below, the Gauss-law init is off by O(h^2) (B3).
static void beam_init_gamma(Comm& comm) {
    if (comm.size > 1) return;
    const double us[4] = {5, 20, 60, 1e6};
    for (double u : us) {
        std::vector<double> f[2][6], g1[2][6];
        double fmax = 0, chg[2] = {0, 0};
        for (int st = 0; st < 2; ++st) {
            Config c = beam_config(64, 32, u, "(sqrt((y-4)^2 + (z-4)^2) < 2.4) * gauss(x - 3.2, 0.5) * gauss(sqrt((y-4)^2 + (z-4)^2), 0.6)",
                                   -1e300, 1e300);
            c.beam_steady = st;
            c.species[0].rigid_until = 1e9;
            c.window.enabled = true; c.window.start = 0;
            c.t_end = 1e9;
            c.check();
            Simulation sim(c, comm);
            sim.init();
            Field* Fp[6] = {&sim.fields->ex, &sim.fields->ey, &sim.fields->ez, &sim.fields->bx, &sim.fields->by, &sim.fields->bz};
            for (int a = 0; a < 6; ++a) f[st][a] = Fp[a]->host();
            sim.step();
            for (int a = 0; a < 6; ++a) g1[st][a] = Fp[a]->host();
            const int n = sim.dom.n[0], ny = sim.dom.n[1], nz = sim.dom.n[2];
            for (int a = 0; a < 6; ++a)
                for (int k = 0; k < nz; ++k)
                    for (int j = 0; j < ny; ++j)
                        for (int i = 12; i < n - 12; ++i) {
                            const size_t q = Fp[a]->idx(i, j, k);
                            chg[st] = std::max(chg[st], std::abs(g1[st][a][q] - f[st][a][q]));
                            if (st) fmax = std::max(fmax, std::abs(f[1][a][q]));
                        }
        }
        double d = 0;
        for (int a = 0; a < 6; ++a)
            for (size_t q = 0; q < f[0][a].size(); ++q) d = std::max(d, std::abs(f[0][a][q] - f[1][a][q]));
        check(fmt("beam field B9, u = %g, window: Gauss-law vs steady init", u), u > 1e3 ? d < 1e-10 * fmax : d > 1e-4 * fmax,
              fmt("max|F_gauss - F_steady| %.2e of max|F|, one-step change gauss %.2e", d / fmax, chg[0] / fmax) + fmt(", steady %.2e", chg[1] / fmax));
    }
}

void test_beam(Comm& comm) {
    beam_gauss(comm);
    beam_radial(comm);
    beam_vacuum(comm);
    beam_delayed(comm);
    beam_window(comm);
    beam_solvers(comm);
    beam_emittance(comm);
    beam_steady(comm);
    beam_init_gamma(comm);
}
