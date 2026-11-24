// Moving window (docs/specs/moving_window.md).
#include <cmath>
#include <cstdio>
#include "common.hpp"
#include "sim.hpp"

// W1. A plane laser pulse (uniform in y, z) in vacuum, window at c: after the start the pulse stays at
//     the same window cells, exactly (the RIP-CC transport is an exact shift of one cell per step).
static void window_vacuum(Comm& comm) {
    Config c;
    c.N[0] = 160; c.N[1] = 4; c.N[2] = 4;
    c.h[0] = 0.25; c.h[1] = 0.5; c.h[2] = 0.5;
    c.periodic[0] = false;
    c.laser.enabled = true;
    c.laser.a0 = 1; c.laser.tau = 2; c.laser.tc = 8; c.laser.w0 = 1e9; c.laser.pol = 'z';
    c.window.enabled = true; c.window.start = 2 * c.laser.tc;
    c.t_end = c.window.start + 40;
    c.check();
    Simulation sim(c, comm);
    sim.init();
    std::vector<double> ez0, by0;
    double w0 = 0, wmin = 1e300, wmax = 0;
    while (!sim.finished()) {
        sim.step();
        if (sim.dom.wx == 1) { ez0 = sim.fields->ez.host(); by0 = sim.fields->by.host(); w0 = sim.fields->energy(); }
        if (sim.dom.wx >= 1) { wmin = std::min(wmin, sim.fields->energy()); wmax = std::max(wmax, sim.fields->energy()); }
    }
    const std::vector<double> ez1 = sim.fields->ez.host(), by1 = sim.fields->by.host();
    double e = 0, a = 0;
    for (size_t q = 0; q < ez0.size(); ++q) {
        e = std::max(e, std::max(std::abs(ez1[q] - ez0[q]), std::abs(by1[q] - by0[q])));
        a = std::max(a, std::abs(ez0[q]));
    }
    const long shifts = sim.dom.wx;
    check("window, vacuum: plane pulse frozen in the window", shifts > 100 && a > 0.5 && e < 1e-13 &&
              (wmax - wmin) < 1e-12 * w0,
          fmt("%.0f shifts, max|dEz, dBy| %.1e", double(shifts), e) + fmt(", dW/W %.1e", (wmax - wmin) / w0));
}

// W2. A laser pulse drives a wake in a cold neutral plasma that streams through the window: the Gauss
//     law stays at round-off (shapes 1, 2; global and tiled deposit); the ions, at rest, enter at the
//     right and leave at the left one x column per step, so their number grows by exactly one column
//     (the column kept beyond the left edge, decision 21).
static void window_wake(Comm& comm) {
    for (int order : {1, 2})
        for (Config::Deposit dep : {Config::Deposit::Global, Config::Deposit::Tiled}) {
            if (RIP_GPU && dep == Config::Deposit::Tiled) continue;
            Config c;
            c.N[0] = 96; c.N[1] = 32; c.N[2] = 4;
            c.h[0] = 0.25; c.h[1] = 0.6; c.h[2] = 1.0;
            c.periodic[0] = false;
            c.order = order;
            c.deposit = dep;
            c.tile[0] = 8; c.tile[1] = 8; c.tile[2] = 4;
            c.laser.enabled = true;
            c.laser.a0 = 1; c.laser.tau = 2; c.laser.tc = 6; c.laser.w0 = 4; c.laser.yc = 0.5 * c.N[1] * c.h[1];
            c.laser.pol = 'y';
            c.window.enabled = true; c.window.start = 2 * c.laser.tc;
            c.t_end = c.window.start + 12;
            for (int s = 0; s < 2; ++s) {
                SpeciesConfig e;
                e.name = s ? "p" : "e"; e.q = s ? 1 : -1; e.m = s ? 1836 : 1; e.density = 0.02;
                e.ppc[0] = 2;
                for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
                c.species.push_back(e);
            }
            c.check();
            Simulation sim(c, comm);
            sim.init();
            const double np0 = double(sim.species[1].size());
            double g = 0;
            while (!sim.finished()) { sim.step(); g = std::max(g, sim.gauss_drift()); }
            const double column = double(c.N[1]) * c.N[2] * 2;
            const double dnp = double(sim.species[1].size()) - np0;
            const std::string tag = std::string("window, wake, shape ") + std::to_string(order) +
                                    (dep == Config::Deposit::Tiled ? ", tiled" : ", global");
            check(tag + ": Gauss law, ion count", sim.dom.wx > 40 && g < 1e-12 && dnp == column,
                  fmt("max|dGauss| %.1e, ions +%.0f", g, dnp) + fmt(" (one column %.0f)", column));
        }
}

// W3. A plasma slab ahead of the box (absolute x in [L + 1, L + 3)) enters at the right edge: with no
//     fields it stays at rest, so the particles present are exactly the lattice points of the slab
//     inside [keep_lo, hi). Window velocities 1 and 0.5 (number of shifts floor((t - start) v / h)).
static void window_slab(Comm& comm) {
    for (double v : {1.0, 0.5}) {
        Config c;
        c.N[0] = 32; c.N[1] = 4; c.N[2] = 4;
        c.h[0] = 0.25; c.h[1] = 0.5; c.h[2] = 0.5;
        c.periodic[0] = false;
        c.window.enabled = true; c.window.start = 1.0; c.window.v = v;
        c.t_end = 1.0 + 30 * c.h[0] / v;                    // 30 shifts: the slab is inside then
        const double L = c.N[0] * c.h[0], x0 = L + 1.0, x1 = L + 3.0;
        for (int s = 0; s < 2; ++s) {
            SpeciesConfig e;
            e.name = s ? "p" : "e"; e.q = s ? 1 : -1; e.m = s ? 1836 : 1; e.density = 0.01;
            e.ppc[0] = 3;
            for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
            e.rmin[0] = x0; e.rmax[0] = x1;
            c.species.push_back(e);
        }
        c.check();
        Simulation sim(c, comm);
        sim.init();
        bool ok = sim.species[0].size() == 0;
        long worst = 0;
        double g = 0;
        while (!sim.finished()) {
            sim.step();
            g = std::max(g, sim.gauss_drift());
            const long want_wx = long(std::floor((sim.t - c.window.start) * v / c.h[0] + 1e-9));
            ok = ok && sim.dom.wx == std::max(0L, want_wx);
            long want = 0;                                  // lattice points of the slab in [keep_lo, hi)
            for (long gi = long(std::floor(x0 / c.h[0])) - 1; gi * c.h[0] < x1; ++gi)
                for (int a = 0; a < 3; ++a) {
                    const double x = (gi + (a + 0.5) / 3) * c.h[0];
                    if (x >= x0 && x < x1 && x >= sim.dom.keep_lo(0) && x < sim.dom.hi(0)) want += c.N[1] * c.N[2];
                }
            worst = std::max(worst, std::abs(long(sim.species[0].size()) - want));
        }
        const auto x = sim.species[0].get(Species::X);
        double xmin = 1e300;
        for (double q : x) xmin = std::min(xmin, q);
        check(std::string("window, slab enters at v = ") + (v == 1 ? "1" : "0.5") + ": exact count, shifts",
              ok && worst == 0 && !x.empty() && xmin >= x0 && g < 1e-12,
              fmt("%.0f shifts, count mismatch %.0f", double(sim.dom.wx), double(worst)) + fmt(", max|dGauss| %.1e", g));
    }
}

// W4. Hot plasma (electrons u_th = 0.05): the entry columns stay unpushed in the steps that end with a
//     shift, so no current reaches the last cell and the Gauss law stays at round-off: (a) a laser
//     wake in hot plasma that starts ahead of the box, (b) no laser, the box filled with hot plasma and
//     the window moving from t = 0. (Without the frozen columns: 2e-4 at u_th = 0.01, 9e-4 at 0.05.)
static void window_hot(Comm& comm) {
    for (int laser : {1, 0}) {
        Config c;
        c.N[0] = 96; c.N[1] = 32; c.N[2] = 4;
        c.h[0] = 0.25; c.h[1] = 0.6; c.h[2] = 1.0;
        c.periodic[0] = false;
        c.order = 2;
        c.laser.enabled = laser;
        c.laser.a0 = 1; c.laser.tau = 2; c.laser.tc = 6; c.laser.w0 = 4; c.laser.yc = 0.5 * c.N[1] * c.h[1];
        c.window.enabled = true; c.window.start = laser ? 2 * c.laser.tc : 0;
        c.t_end = c.window.start + 20;
        for (int s = 0; s < 2; ++s) {
            SpeciesConfig e;
            e.name = s ? "p" : "e"; e.q = s ? 1 : -1; e.m = s ? 1836 : 1; e.density = 0.02;
            e.ppc[0] = 2;
            for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; e.uth[a] = s ? 0.001 : 0.05; }
            if (laser) e.rmin[0] = c.N[0] * c.h[0] + 0.3;              // ahead of the box at the start
            c.species.push_back(e);
        }
        c.check();
        Simulation sim(c, comm);
        sim.init();
        double g = 0;
        while (!sim.finished()) { sim.step(); g = std::max(g, sim.gauss_drift()); }
        check(std::string("window, hot plasma, ") + (laser ? "laser wake, plasma ahead" : "box filled, start 0") +
                  ": Gauss law", sim.dom.wx > 60 && sim.species[0].size() > 0 && g < 1e-12,
              fmt("max|dGauss| %.1e, %.0f shifts", g, double(sim.dom.wx)));
    }
}

// W8. A non-neutral column entering (decision 91): electrons 2 x 2 x 2 per cell, warm (u_th = 0.05), protons 1 per cell, a
//     linear density ramp and a sharp front ahead of the box (as bench/lwfa_bubble: the two lattices put different
//     charges on the nodes where the density is not uniform). The charge of a column appears with no current and no field;
//     it is kept as a background (folded into the initial Gauss residual when the column leaves the frozen cells), so the
//     Gauss check stays at round-off, RIP-CC and iRIP. Before: the residual of the entering charge (rho of the column).
static void window_nonneutral(Comm& comm) {
    for (Solver sv : {Solver::RipCC, Solver::IRip}) {
        Config c;
        c.N[0] = 96; c.N[1] = 32; c.N[2] = 4;
        c.h[0] = 0.25; c.h[1] = 0.6; c.h[2] = 1.0;
        c.periodic[0] = false;
        c.order = 2;
        c.solver = sv;
        c.laser.enabled = true;
        c.laser.a0 = 1; c.laser.tau = 2; c.laser.tc = 6; c.laser.w0 = 4; c.laser.yc = 0.5 * c.N[1] * c.h[1];
        c.window.enabled = true; c.window.start = 2 * c.laser.tc;
        c.t_end = c.window.start + 20;
        const double x0 = c.N[0] * c.h[0] + 0.3;                     // ahead of the box at the start
        for (int s = 0; s < 2; ++s) {
            SpeciesConfig e;
            e.name = s ? "p" : "e"; e.q = s ? 1 : -1; e.m = s ? 1836 : 1; e.density = 0.02;
            for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; e.uth[a] = s ? 0.0 : 0.05; e.ppc[a] = s ? 1 : 2; }
            e.rmin[0] = x0;
            e.density_fn = Expr("x < " + std::to_string(x0 + 8) + " ? 0.2 + 0.8 * (x - " + std::to_string(x0) + ") / 8 : 1",
                                {"x", "y", "z"}, UnitFn());
            c.species.push_back(e);
        }
        c.check();
        Simulation sim(c, comm);
        sim.init();
        double g = 0, rmax = 0;
        while (!sim.finished()) {
            sim.step();
            g = std::max(g, sim.gauss_drift());
            const std::vector<double> r = sim.fields->rho.host();     // net charge (compute_rho of gauss_drift)
            for (int i = 0; i < sim.dom.n[0]; ++i) rmax = std::max(rmax, std::abs(r[sim.fields->rho.idx(i, 3, 1)]));
        }
        rmax = comm.allreduce_max(rmax);
        check(std::string("window, non-neutral warm column entering (") + (sv == Solver::IRip ? "iRIP" : "RIP-CC") + "): Gauss law",
              sim.dom.wx > 60 && rmax > 1e-6 && g < 1e-12,
              fmt("max|dGauss| %.1e, max|rho_net| %.1e", g, rmax) + fmt(", %.0f shifts", double(sim.dom.wx)));
    }
}

// W9. The current filter with the lab window (decision 107): W8's warm non-neutral column with a ramp entering, binomial filter
//     1 and 2 passes, 3D (4 z cells) and 2D, RIP-CC and iRIP, run until the plasma has left through the open left edge. The
//     entering plasma is frozen for 3 + passes cells and its charge folded into the background (decision 91, shared with RZ);
//     the Gauss check skips the first passes cells at the left edge (the closure offset of the leaving plasma, as RZ).
static void window_filter(Comm& comm) {
    for (int nz : {4, 1})
        for (Solver sv : {Solver::RipCC, Solver::IRip})
            for (int passes : {1, 2}) {
                Config c;
                c.flat = nz == 1;
                c.N[0] = 96; c.N[1] = 32; c.N[2] = nz;
                c.h[0] = 0.25; c.h[1] = 0.6; c.h[2] = 1.0;
                c.periodic[0] = false;
                c.order = 2;
                c.solver = sv;
                c.filter_passes = passes;
                c.laser.enabled = true;
                c.laser.a0 = 1; c.laser.tau = 2; c.laser.tc = 6; c.laser.w0 = 4; c.laser.yc = 0.5 * c.N[1] * c.h[1];
                c.window.enabled = true; c.window.start = 2 * c.laser.tc;
                c.t_end = c.window.start + 32;                                // 128 shifts: the plasma front leaves at the left
                const double x0 = c.N[0] * c.h[0] + 0.3;
                for (int s = 0; s < 2; ++s) {
                    SpeciesConfig e;
                    e.name = s ? "p" : "e"; e.q = s ? 1 : -1; e.m = s ? 1836 : 1; e.density = 0.02;
                    for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; e.uth[a] = s ? 0.0 : 0.05; e.ppc[a] = s ? 1 : 2; }
                    if (nz == 1) e.ppc[2] = 1;
                    e.rmin[0] = x0;
                    e.density_fn = Expr("x < " + std::to_string(x0 + 8) + " ? 0.2 + 0.8 * (x - " + std::to_string(x0) + ") / 8 : 1",
                                        {"x", "y", "z"}, UnitFn());
                    c.species.push_back(e);
                }
                c.check();
                Simulation sim(c, comm);
                sim.init();
                double g = 0;
                while (!sim.finished()) { sim.step(); g = std::max(g, sim.gauss_drift()); }
                check(std::string("window + current filter (") + (nz == 1 ? "2D" : "3D") + ", " + (sv == Solver::IRip ? "iRIP" : "RIP-CC") +
                          ", " + std::to_string(passes) + (passes == 1 ? " pass" : " passes") + "): Gauss law",
                      sim.dom.wx > 120 && g < 1e-12, fmt("max|dGauss| %.1e", g) + fmt(", %.0f shifts", double(sim.dom.wx)));
            }
}

// W5. Field::shift_x (the start of the array moves, a copy once the slack is used up) = the flat copy
//     f(p) = f(p + 1) of the whole array, over more shifts than the slack holds. The last element is
//     undefined after a shift (the caller's exchange fills it): set here as an exchange would.
static void window_shift_storage() {
    Domain d;
    d.n[0] = 5; d.n[1] = 3; d.n[2] = 2;
    Field f(d), tmp(d);
    const size_t n = f.size();
    std::vector<double> ref(n);
    for (size_t q = 0; q < n; ++q) ref[q] = 1.0 + double(q);
    f.upload(ref);
    const int shifts = 2 * (d.n[0] + 2 * d.ng) * (d.n[1] + 2 * d.ng) + 7;     // two copies
    double e = 0;
    for (int s = 0; s < shifts; ++s) {
        f.shift_x(tmp);
        for (size_t q = 0; q + 1 < n; ++q) ref[q] = ref[q + 1];
        ref[n - 1] = -1.0 - s;
        std::vector<double> h = f.host();
        for (size_t q = 0; q + 1 < n; ++q) e = std::max(e, std::abs(h[q] - ref[q]));
        h[n - 1] = ref[n - 1];
        f.upload(h);
        const FV v = f.view();                              // the view sees the same array
        Buf<double> one(1);
        double* o = one.data();
        par_for(1, [=] HD(long) { o[0] = v(d.n[0] - 1, d.n[1] - 1, d.n[2] - 1); });
        double x = 0;
        one.to_host(&x, 0, 1);
        e = std::max(e, std::abs(x - ref[f.idx(d.n[0] - 1, d.n[1] - 1, d.n[2] - 1)]));
    }
    check("window, field storage: shift_x = flat copy over two refills of the slack", e == 0,
          fmt("%.0f shifts, max difference %.1e", double(shifts), e));
}

// W6. Cold plasma loads in kernels (load.hpp): the same particles in the same order as the host loop
//     over the cells (k, j, i) and the particles (c, b, a), with bounds, a tilted slab, a density
//     function that is negative in half of the box (those particles are skipped) and a drift u0.
static void window_cold_load(Comm& comm) {
    Config c;
    c.N[0] = 12; c.N[1] = 8; c.N[2] = 6;
    c.h[0] = 0.25; c.h[1] = 0.5; c.h[2] = 0.5;
    SpeciesConfig e;
    e.name = "e"; e.density = 0.01;
    e.ppc[0] = 2; e.ppc[1] = 3; e.ppc[2] = 2;
    e.u0[0] = 0.1; e.u0[2] = -0.2;
    for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
    e.rmin[2] = 0.4; e.rmax[0] = 2.6;
    e.slab_x0 = 0.3; e.slab_y0 = 1.0; e.slab_thickness = 2.0; e.slab_angle = 0.3;
    e.density_fn = Expr("sin(2 * y) * (x < 1 ? x : 1)", {"x", "y", "z"}, [](const std::string&, double&) { return false; });
    c.species.push_back(e);
    c.check();
    Simulation sim(c, comm);
    sim.init();
    const Domain& d = sim.dom;
    const double w = e.density * d.cell_volume() / 12;
    std::vector<double> want[Species::NC];
    for (int k = 0; k < d.n[2]; ++k)
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i)
                for (int cc = 0; cc < 2; ++cc)
                    for (int b = 0; b < 3; ++b)
                        for (int a = 0; a < 2; ++a) {
                            const double r[3] = {(d.cell0(0) + i + (a + 0.5) / 2) * d.h[0],
                                                 (d.off[1] + j + (b + 0.5) / 3) * d.h[1],
                                                 (d.off[2] + k + (cc + 0.5) / 2) * d.h[2]};
                            const double sl = (r[0] - e.slab_x0) * std::cos(e.slab_angle) + (r[1] - e.slab_y0) * std::sin(e.slab_angle);
                            if (r[0] >= e.rmax[0] || r[2] < e.rmin[2] || sl < 0 || sl >= e.slab_thickness) continue;
                            const double f = std::sin(2 * r[1]) * (r[0] < 1 ? r[0] : 1);
                            if (!(f > 0)) continue;
                            const double v[Species::NC] = {r[0], r[1], r[2], e.u0[0], e.u0[1], e.u0[2], w * f};
                            for (int q = 0; q < Species::NC; ++q) want[q].push_back(v[q]);
                        }
    const Species& s = sim.species[0];
    bool ok = s.size() == want[0].size() && s.size() > 100;
    double ex = 0, ew = 0;                                  // positions, momenta exact; weights: device sin
    for (int q = 0; ok && q < Species::NC; ++q) {
        const std::vector<double> got = s.get(q);
        for (size_t p = 0; p < got.size(); ++p) {
            if (q == Species::W) ew = std::max(ew, std::abs(got[p] - want[q][p]) / w);
            else ex = std::max(ex, std::abs(got[p] - want[q][p]));
        }
    }
    check("window, cold plasma loaded in kernels = host loop (bounds, slab, density function)",
          ok && ex == 0 && ew < 1e-15, fmt("%.0f particles, max|dr, du| %.1e", double(s.size()), ex) + fmt(", max|dw|/w %.1e", ew));
}

void test_window(Comm& comm) {
    window_vacuum(comm);
    window_wake(comm);
    window_slab(comm);
    window_hot(comm);
    window_nonneutral(comm);
    window_filter(comm);
    window_shift_storage();
    window_cold_load(comm);
}
