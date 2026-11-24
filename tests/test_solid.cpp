// O4 solid-density targets (docs/specs/solid_target.md): the tilted slab with pre-plasma, the neutral target with mobile
// ions, the Cart3D current filter, and the omega_p dt check at the peak density. Shape 3 is also in tests/tests.cpp.
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include "common.hpp"
#include "filter.hpp"
#include "sim.hpp"
#include "xlines.hpp"

static SpeciesConfig species(const char* name, double q, double m, double n) {
    SpeciesConfig s;
    s.name = name; s.q = q; s.m = m; s.density = n;
    for (int a = 0; a < 3; ++a) { s.rmin[a] = -1e300; s.rmax[a] = 1e300; }
    return s;
}

static double sum_w(Comm& comm, const Species& s) {
    double v = 0;
    for (double w : s.get(Species::W)) v += w;
    comm.allreduce_sum(&v, 1);
    return v;
}

// S1. Loaded charge of a tilted slab with front and rear ramps: sum w = density * Ly Lz * integral p ds / cos(angle)
//     to round-off (exact sub-cell means), for scale lengths below and above the cell, cold (kernels) and hot (host loop)
//     loading; the point-sampled plain slab for comparison.
static void solid_charge(Comm& comm) {
    struct Case { double angle_deg, Lf, Ef, Lr, Er; };
    const Case cases[] = {{0, 0.07, 0.5, 0, 0}, {22.5, 0.07, 0.5, 0.3, 1.2}, {30, 0.6, 2.0, 0.05, 0.3}, {-17, 0.25, 1.0, 0, 0}};
    double worst = 0, worst_hot = 0;
    for (const Case& K : cases) {
        Config c;
        c.N[0] = 64; c.N[1] = 24; c.N[2] = 6;
        c.h[0] = 0.2; c.h[1] = 0.25; c.h[2] = 0.4;
        c.periodic[0] = false;
        c.t_end = c.h[0];
        SpeciesConfig e = species("e", -1, 1, 3.0);
        e.ppc[0] = 2; e.ppc[1] = 3;
        e.slab_angle = K.angle_deg * M_PI / 180; e.slab_x0 = 5.0; e.slab_y0 = 3.0; e.slab_thickness = 1.3;
        e.slab_ramp = K.Lf; e.slab_ramp_extent = K.Ef; e.slab_rear_ramp = K.Lr; e.slab_rear_extent = K.Er;
        SpeciesConfig hot = e;
        hot.name = "hot"; hot.uth[0] = hot.uth[1] = hot.uth[2] = 1e-3;
        c.species = {e, hot};
        c.check();
        Simulation sim(c, comm);
        sim.init();
        const double ip = K.Lf * -std::expm1(-K.Ef / K.Lf) + e.slab_thickness + (K.Lr > 0 ? K.Lr * -std::expm1(-K.Er / K.Lr) : 0.0);
        const double want = e.density * c.N[1] * c.h[1] * c.N[2] * c.h[2] * ip / std::cos(e.slab_angle);
        const double got = sum_w(comm, sim.species[0]), goth = sum_w(comm, sim.species[1]);
        worst = std::max(worst, std::abs(got / want - 1));
        worst_hot = std::max(worst_hot, std::abs(goth / got - 1));
    }
    check("O4 slab with ramps: loaded charge = analytic (4 tilts)", worst < 1e-12 && worst_hot < 1e-14,
          fmt("max rel err %.1e, hot / cold %.1e", worst, worst_hot));
    // the plain slab at 22.5 deg: point test (default) against the exact sub-cell means
    double err[2];
    for (int ex : {0, 1}) {
        Config c;
        c.N[0] = 64; c.N[1] = 24; c.N[2] = 6;
        c.h[0] = 0.2; c.h[1] = 0.25; c.h[2] = 0.4;
        c.periodic[0] = false;
        c.t_end = c.h[0];
        SpeciesConfig e = species("e", -1, 1, 3.0);
        e.ppc[0] = 2; e.ppc[1] = 3;
        e.slab_angle = 22.5 * M_PI / 180; e.slab_x0 = 5.0; e.slab_y0 = 3.0; e.slab_thickness = 1.3; e.slab_exact = ex;
        c.species = {e};
        c.check();
        Simulation sim(c, comm);
        sim.init();
        const double want = e.density * c.N[1] * c.h[1] * c.N[2] * c.h[2] * e.slab_thickness / std::cos(e.slab_angle);
        err[ex] = sum_w(comm, sim.species[0]) / want - 1;
    }
    check("O4 plain tilted slab: exact weights vs point test", std::abs(err[1]) < 1e-12,
          fmt("rel err exact %.1e, point test %.1e", err[1], err[0]));
}

// S2. A neutral tilted target with pre-plasma, hot electrons and mobile ions (m = 100, Z = 1) on the same lattice:
//     rho = 0 at t = 0; shape 3 with one filter pass, RIP-CC and iRIP, open x: Gauss law and div B at round-off over 60
//     steps while the electrons expand and pull the ions.
static void solid_target(Comm& comm) {
    for (Solver sv : {Solver::RipCC, Solver::IRip}) {
        Config c;
        c.N[0] = 64; c.N[1] = 16; c.N[2] = 6;
        c.h[0] = 0.2; c.h[1] = 0.4; c.h[2] = 0.4;
        c.periodic[0] = false;
        c.solver = sv;
        c.order = 3;
        c.filter_passes = 1;
        c.t_end = 60 * c.h[0];
        SpeciesConfig e = species("e", -1, 1, 20.0);
        e.ppc[0] = e.ppc[1] = 2;
        e.slab_angle = 22.5 * M_PI / 180; e.slab_x0 = 5.0; e.slab_y0 = 3.2; e.slab_thickness = 1.5;
        e.slab_ramp = 0.15; e.slab_ramp_extent = 0.75; e.slab_rear_ramp = 0.3; e.slab_rear_extent = 0.9;
        for (int a = 0; a < 3; ++a) e.uth[a] = 0.3;
        SpeciesConfig ion = e;
        ion.name = "ion"; ion.q = 1; ion.m = 100;
        for (int a = 0; a < 3; ++a) ion.uth[a] = 0;
        c.species = {e, ion};
        c.check();
        Simulation sim(c, comm);
        sim.init();
        double r0 = 0;
        {
            H rho(sim.fields->rho);
            for (int k = 0; k < sim.dom.n[2]; ++k)
                for (int j = 0; j < sim.dom.n[1]; ++j)
                    for (int i = 0; i < sim.dom.n[0]; ++i) r0 = std::max(r0, std::abs(rho(i, j, k)));
            r0 = comm.allreduce_max(r0);
        }
        double g = 0, db = 0;
        while (!sim.finished()) {
            sim.step();
            g = std::max(g, sim.gauss_drift());
            db = std::max(db, comm.allreduce_max(sim.fields->max_div_b()));
        }
        double um = 0;
        for (double u : sim.species[1].get(Species::UX)) um = std::max(um, std::abs(u));
        um = comm.allreduce_max(um);
        const std::string tag = std::string("O4 target, shape 3, filter, ") + (sv == Solver::IRip ? "iRIP" : "RIP-CC");
        check(tag + ": neutral, Gauss, div B", r0 < 1e-12 * e.density && g < 1e-11 && db < 1e-11 && um > 1e-4,
              fmt("rho(0) %.1e, max|dGauss| %.1e", r0, g) + fmt(", max|divB| %.1e, ion |ux| %.1e", db, um));
    }
}

// S4. The filter (filter.hpp) on its own, periodic and open x: rho of a wave along x, y, z gets cos^2(k h / 2) per pass;
//     a divergence-free J (a discrete curl) is unchanged; the total charge is kept with open x.
static void solid_filter(Comm& comm) {
    Domain d;
    d.N[0] = 16; d.N[1] = 12; d.N[2] = 10;
    d.h[0] = 0.3; d.h[1] = 0.5; d.h[2] = 0.45;
    d.ng = 3;
    const int want[3] = {1, 1, 1};
    comm.decompose(d, want);
    Field rho(d), w(d);
    double terr = 0;
    for (int a = 0; a < 3; ++a) {
        const double k = 2 * M_PI * 3 / (d.N[a] * d.h[a]);
        H r(rho);
        for (int kk = 0; kk < d.n[2]; ++kk)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const int g[3] = {d.off[0] + i, d.off[1] + j, d.off[2] + kk};
                    r(i, j, kk) = std::cos(k * (g[a] + (a == 0 ? 0.5 : 0.0)) * d.h[a]);
                }
        r.put(rho);
        filter_rho_cart(rho, w, d, comm, 2);
        const double f = std::pow(std::cos(0.5 * k * d.h[a]), 4);
        H o(rho);
        for (int kk = 0; kk < d.n[2]; ++kk)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const int g[3] = {d.off[0] + i, d.off[1] + j, d.off[2] + kk};
                    terr = std::max(terr, std::abs(o(i, j, kk) - f * std::cos(k * (g[a] + (a == 0 ? 0.5 : 0.0)) * d.h[a])));
                }
    }
    terr = comm.allreduce_max(terr);
    // a discrete curl: Jx = Dy Az - Dz Ay, Jy = Dz Ax - Dx Az, Jz = Dx Ay - Dy Ax with the differences of D (x forward,
    // y and z backward), so D J = 0 identically
    Fields f(d);
    double cerr = 0, jm = 0;
    {
        std::mt19937_64 rng(5);
        std::uniform_real_distribution<double> u(-1, 1);
        const int nx = d.N[0], ny = d.N[1], nz = d.N[2];
        std::vector<double> A[3];
        for (auto& v : A) { v.resize(size_t(nx) * ny * nz); for (double& x : v) x = u(rng); }
        auto at = [&](int c, int i, int j, int k) {
            return A[c][size_t(((k + nz) % nz) * ny + (j + ny) % ny) * nx + (i + nx) % nx];
        };
        auto Dx = [&](int c, int i, int j, int k) { return (at(c, i + 1, j, k) - at(c, i, j, k)) / d.h[0]; };
        auto Dy = [&](int c, int i, int j, int k) { return (at(c, i, j, k) - at(c, i, j - 1, k)) / d.h[1]; };
        auto Dz = [&](int c, int i, int j, int k) { return (at(c, i, j, k) - at(c, i, j, k - 1)) / d.h[2]; };
        H jx(f.jx), jy(f.jy), jz(f.jz);
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const int gi = d.off[0] + i, gj = d.off[1] + j, gk = d.off[2] + k;
                    jx(i, j, k) = Dy(2, gi, gj, gk) - Dz(1, gi, gj, gk);
                    jy(i, j, k) = Dz(0, gi, gj, gk) - Dx(2, gi, gj, gk);
                    jz(i, j, k) = Dx(1, gi, gj, gk) - Dy(0, gi, gj, gk);
                }
        // (D J)_c = (Jx(c + 1) - Jx(c)) / hx + ...: stored at the same index, Jx gets the forward difference Dx
        jx.put(f.jx); jy.put(f.jy); jz.put(f.jz);
        const std::vector<double> before[3] = {f.jx.host(), f.jy.host(), f.jz.host()};
        filter_currents_cart(f, w, d, comm, 2);
        const std::vector<double> after[3] = {f.jx.host(), f.jy.host(), f.jz.host()};
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < d.n[2]; ++k)
                for (int j = 0; j < d.n[1]; ++j)
                    for (int i = 0; i < d.n[0]; ++i) {
                        const size_t q = f.jx.idx(i, j, k);
                        cerr = std::max(cerr, std::abs(after[c][q] - before[c][q]));
                        jm = std::max(jm, std::abs(before[c][q]));
                    }
        cerr = comm.allreduce_max(cerr); jm = comm.allreduce_max(jm);
    }
    // open x: the total charge of the Gauss nodes is kept (Neumann)
    Domain e = d;
    e.periodic[0] = false;
    double qerr = 0;
    {
        Comm one;
        const int w1[3] = {1, 1, 1};
        one.decompose(e, w1);
        Field r(e), s(e);
        H h(r);
        std::mt19937_64 rng(9);
        std::uniform_real_distribution<double> u(0, 1);
        double q0 = 0;
        for (int k = 0; k < e.n[2]; ++k)
            for (int j = 0; j < e.n[1]; ++j)
                for (int i = 0; i < e.n[0]; ++i) { h(i, j, k) = u(rng); if (i < e.N[0] - 1) q0 += h(i, j, k); }
        h.put(r);
        filter_rho_cart(r, s, e, one, 3);
        H o(r);
        double q1 = 0;
        for (int k = 0; k < e.n[2]; ++k)
            for (int j = 0; j < e.n[1]; ++j)
                for (int i = 0; i < e.N[0] - 1; ++i) q1 += o(i, j, k);
        qerr = std::abs(q1 / q0 - 1);
    }
    check("O4 filter: rho waves cos^4(kh/2), curl J kept, charge kept", terr < 1e-14 && cerr < 1e-13 * jm && qerr < 1e-14,
          fmt("wave err %.1e, curl dJ/J %.1e", terr, cerr / jm) + fmt(", open-x charge %.1e", qerr));
}

// S4b. Grid heating of a cold-ion, hot-electron plasma with an unresolved Debye length (lambda_D = 0.17 dx), periodic:
//      the kinetic energy gained in 300 steps for shape 1, shape 3 and shape 3 with one filter pass (Gauss law checked).
static void solid_heating(Comm& comm) {
    double dk[3], gmax = 0;
    for (int run = 0; run < 3; ++run) {
        Config c;
        c.N[0] = 32; c.N[1] = 16; c.N[2] = 16;
        c.h[0] = 0.3; c.h[1] = 0.6; c.h[2] = 0.6;
        c.order = run == 0 ? 1 : 3;
        c.filter_passes = run == 2 ? 1 : 0;
        c.t_end = 300 * c.h[0];
        SpeciesConfig e = species("e", -1, 1, 1.0);
        e.ppc[0] = 2; e.ppc[1] = e.ppc[2] = 1;
        for (int a = 0; a < 3; ++a) e.uth[a] = 0.05;
        SpeciesConfig ion = species("ion", 1, 1836, 1.0);
        ion.ppc[0] = 2; ion.mobile = false;
        c.species = {e, ion};
        c.check();
        Simulation sim(c, comm);
        sim.init();
        double k0 = sim.species[0].kinetic_energy();
        comm.allreduce_sum(&k0, 1);
        while (!sim.finished()) sim.step();
        gmax = std::max(gmax, sim.gauss_drift());
        double k1 = sim.species[0].kinetic_energy();
        comm.allreduce_sum(&k1, 1);
        dk[run] = k1 / k0 - 1;
    }
    check("O4 grid heating 300 steps: shape 1 / 3 / 3 + filter", dk[2] < dk[0] && gmax < 1e-12,
          fmt("dK/K %.2e / ", dk[0], dk[1]) + fmt("%.2e / %.2e", dk[1], dk[2]) + fmt(", Gauss %.1e", gmax));
}

// S5. omega_p dt at the peak density: a density_function that rises to 50 x density is refused at the start (the deck
//     check sees `density` only), the same profile capped at 1 passes and reports omega_p dt.
static void solid_wpdt(Comm& comm) {
    bool refused = false;
    double wpdt = 0;
    for (int cap : {0, 1}) {
        Config c;
        c.N[0] = 32; c.N[1] = 8; c.N[2] = 8;
        c.h[0] = 0.3; c.h[1] = 0.6; c.h[2] = 0.6;
        c.periodic[0] = false;
        c.t_end = c.h[0];
        SpeciesConfig e = species("e", -1, 1, 1.0);
        e.density_fn = Expr(cap ? "x > 5 ? 1 : 0.5" : "x > 5 ? 50 : 0.5", {"x", "y", "z"}, nullptr);
        c.species = {e};
        c.check();
        Simulation sim(c, comm);
        try { sim.init(); wpdt = sim.peak_wpdt; }
        catch (const std::runtime_error& ex) { refused = !cap && std::string(ex.what()).find("peak") != std::string::npos; }
    }
    check("O4 omega_p dt checked at the peak density", refused && std::abs(wpdt - 0.3) < 1e-12,
          fmt("refused %.0f, omega_p dt %.3f with the capped profile", refused, wpdt));
}


// S6. The filter's change dJ = S J - J of a random J on a periodic grid, fed into an empty grid for one field step, then
//     J = 0 for 40 steps: a change that is a discrete gradient moves charge only and leaves B = 0 (a static E). The local
//     filter (numerics.filter = local, decision 64) changes one component at a time and radiates; the projection
//     (decision 69) is a gradient. B^2 / (E^2 + B^2) after 40 steps, against the raw J; RIP-CC and iRIP.
static void solid_filter_radiation(Comm& comm) {
    double bmax = 0, bloc = 1;
    for (int sv = 0; sv < 2; ++sv)
        for (int kind = 0; kind < 3; ++kind) {        // 0: the raw random J, 1: dJ of the local filter, 2: of the projection
            Config c;
            c.N[0] = 32; c.N[1] = 16; c.N[2] = 16;
            c.h[0] = 0.3; c.h[1] = 0.6; c.h[2] = 0.6;
            c.solver = sv == 0 ? Solver::RipCC : Solver::IRip;
            c.t_end = 41 * c.h[0];
            c.check();
            Simulation sim(c, comm);
            sim.init();
            Fields& f = *sim.fields;
            Field w(sim.dom);
            std::shared_ptr<FilterProjWork> W;
            std::mt19937_64 rng(11);
            std::uniform_real_distribution<double> u(-1, 1);
            Field* J[3] = {&f.jx, &f.jy, &f.jz};
            for (Field* x : J) {
                H h(*x);
                for (int k = 0; k < sim.dom.n[2]; ++k)
                    for (int j = 0; j < sim.dom.n[1]; ++j)
                        for (int i = 0; i < sim.dom.n[0]; ++i) h(i, j, k) = u(rng);
                h.put(*x);
            }
            if (kind > 0) {
                const std::vector<double> b[3] = {f.jx.host(), f.jy.host(), f.jz.host()};
                if (kind == 1) filter_currents_cart(f, w, sim.dom, comm, 1);
                else filter_currents_proj(f, w, W, sim.dom, comm, 1);
                for (int a = 0; a < 3; ++a) {
                    std::vector<double> v = J[a]->host();
                    for (size_t q = 0; q < v.size(); ++q) v[q] -= b[a][q];
                    J[a]->upload(v);
                }
            }
            f.step(comm);
            for (Field* x : J) x->upload(std::vector<double>(x->host().size(), 0.0));
            for (int n = 0; n < 40; ++n) f.step(comm);
            auto sq = [&](const Field& x) {
                double v = 0;
                const std::vector<double> h = x.host();
                for (int k = 0; k < sim.dom.n[2]; ++k)
                    for (int j = 0; j < sim.dom.n[1]; ++j)
                        for (int i = 0; i < sim.dom.n[0]; ++i) v += h[x.idx(i, j, k)] * h[x.idx(i, j, k)];
                comm.allreduce_sum(&v, 1);
                return v;
            };
            const double we = sq(f.ex) + sq(f.ey) + sq(f.ez), wb = sq(f.bx) + sq(f.by) + sq(f.bz), ratio = wb / (we + wb);
            if (kind == 1) bloc = std::min(bloc, ratio);
            if (kind == 2) bmax = std::max(bmax, ratio);
            if (comm.rank == 0)
                std::printf("  S6 %s, %s: after 40 steps B^2 / (E^2 + B^2) = %.3e\n", sv == 0 ? "rip_cc" : "irip",
                            kind == 0 ? "raw random J" : kind == 1 ? "dJ of the local filter" : "dJ of the projection", ratio);
        }
    check("O4 projection filter: its change radiates nothing (B = 0)", bmax < 1e-24 && bloc > 1e-3,
          fmt("projection %.1e, local %.1e", bmax, bloc));
}

// S7. The projection filter keeps the Gauss law exactly: D J' = T D J (T = T_z T_y T_x of filter_rho_cart) for a random J,
//     periodic and open x, one and two passes; a divergence-free J (a discrete curl) is unchanged.
static void solid_filter_projection(Comm& comm) {
    double err = 0, cerr = 0;
    for (int open = 0; open < 2; ++open)
        for (int passes = 1; passes <= 2; ++passes) {
            Domain d;
            d.N[0] = 24; d.N[1] = 12; d.N[2] = 8;
            d.h[0] = 0.3; d.h[1] = 0.5; d.h[2] = 0.45;
            d.periodic[0] = open == 0;
            d.ng = 3;
            const int want[3] = {0, 1, 1};
            comm.decompose(d, want);
            Fields f(d);
            Field w(d), rho(d), s(d);
            std::shared_ptr<FilterProjWork> W;
            std::mt19937_64 rng(13 + passes);
            std::uniform_real_distribution<double> u(-1, 1);
            Field* J[3] = {&f.jx, &f.jy, &f.jz};
            for (Field* x : J) {
                H h(*x);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i) h(i, j, k) = u(rng);
                h.put(*x);
            }
            // D J on the Gauss nodes (x: [0, N - 1) with open x), as a charge to filter
            auto div = [&](Field& out) {
                comm.exchange(d, {&f.jx, &f.jy, &f.jz});
                H jx(f.jx), jy(f.jy), jz(f.jz), o(out);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i) {
                            const bool g = d.periodic[0] || d.off[0] + i < d.N[0] - 1;
                            o(i, j, k) = g ? (jx(i + 1, j, k) - jx(i, j, k)) / d.h[0] + (jy(i, j, k) - jy(i, j - 1, k)) / d.h[1] +
                                                 (jz(i, j, k) - jz(i, j, k - 1)) / d.h[2]
                                           : 0.0;
                        }
                o.put(out);
            };
            div(rho);
            filter_rho_cart(rho, s, d, comm, passes);
            filter_currents_proj(f, w, W, d, comm, passes);
            div(s);
            H a(rho), b(s);
            double e = 0, m = 0;
            for (int k = 0; k < d.n[2]; ++k)
                for (int j = 0; j < d.n[1]; ++j)
                    for (int i = 0; i < d.n[0]; ++i) {
                        e = std::max(e, std::abs(a(i, j, k) - b(i, j, k)));
                        m = std::max(m, std::abs(a(i, j, k)));
                    }
            e = comm.allreduce_max(e); m = comm.allreduce_max(m);
            err = std::max(err, e / m);
            // a discrete curl (D J = 0) is unchanged
            if (open == 0 && passes == 1) {
                for (Field* x : J) x->fill(0);
                H jx(f.jx), jy(f.jy);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i) {     // J = (Dy A, -Dx A, 0) with A = cos(2 pi (x + y)/L)
                            const int gi = d.off[0] + i, gj = d.off[1] + j;
                            auto A = [&](int ii, int jj) { return std::cos(2 * M_PI * (double(ii) / d.N[0] + double(jj) / d.N[1])); };
                            jx(i, j, k) = (A(gi, gj) - A(gi, gj - 1)) / d.h[1];
                            jy(i, j, k) = -(A(gi + 1, gj) - A(gi, gj)) / d.h[0];
                        }
                jx.put(f.jx); jy.put(f.jy);
                const std::vector<double> b0 = f.jx.host(), b1 = f.jy.host();
                filter_currents_proj(f, w, W, d, comm, 1);
                const std::vector<double> a0 = f.jx.host(), a1 = f.jy.host();
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i) {
                            const size_t q = f.jx.idx(i, j, k);
                            cerr = std::max(cerr, std::max(std::abs(a0[q] - b0[q]), std::abs(a1[q] - b1[q])));
                        }
                cerr = comm.allreduce_max(cerr);
            }
        }
    check("O4 projection filter: D J' = T D J (periodic, open x), curl J kept", err < 1e-12 && cerr < 1e-12,
          fmt("%.1e, curl %.1e", err, cerr));
}

// S9. The binomial filter (numerics.filter = binomial, decision 91): (a) D J' = T D J (T of filter_rho_cart) for a random J,
//     periodic, open x, and open x, y, z (any boundary: no transform), one and two passes; (b) in a periodic box every
//     component of a plane wave cos(k . x) (at its own staggered position) is multiplied by prod_a cos^2(k_a h_a / 2) per pass,
//     the transfer function of the binomial (1, 2, 1) / 4, as the charge: J = v rho keeps its form (independent reference).
static void solid_filter_binomial(Comm& comm) {
    double err = 0, terr = 0;
    for (int open = 0; open < 3; ++open)
        for (int passes = 1; passes <= 2; ++passes) {
            Domain d;
            d.N[0] = 24; d.N[1] = 12; d.N[2] = 8;
            d.h[0] = 0.3; d.h[1] = 0.5; d.h[2] = 0.45;
            d.periodic[0] = open == 0; d.periodic[1] = d.periodic[2] = open < 2;
            d.ng = 3;
            const int want[3] = {0, 0, 0};
            comm.decompose(d, want);
            Fields f(d);
            Field w(d), rho(d), s(d);
            std::mt19937_64 rng(29 + passes + 4 * open);
            std::uniform_real_distribution<double> u(-1, 1);
            Field* J[3] = {&f.jx, &f.jy, &f.jz};
            for (Field* x : J) {
                H h(*x);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i) h(i, j, k) = u(rng);
                h.put(*x);
            }
            auto gx = [&](int a, int gi) { return d.periodic[a] || (gi >= 0 && gi < d.N[a] - (a == 0 ? 1 : 0)); };
            auto div = [&](Field& out) {
                comm.exchange(d, {&f.jx, &f.jy, &f.jz});
                H jx(f.jx), jy(f.jy), jz(f.jz), o(out);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i) {
                            const bool g = gx(0, d.off[0] + i) && gx(1, d.off[1] + j) && gx(2, d.off[2] + k);
                            o(i, j, k) = g ? (jx(i + 1, j, k) - jx(i, j, k)) / d.h[0] + (jy(i, j, k) - jy(i, j - 1, k)) / d.h[1] +
                                                 (jz(i, j, k) - jz(i, j, k - 1)) / d.h[2]
                                           : 0.0;
                        }
                o.put(out);
            };
            div(rho);
            filter_rho_cart(rho, s, d, comm, passes);
            filter_currents_binomial(f, w, d, comm, passes);
            div(s);
            H a(rho), b(s);
            double e = 0, m = 0;
            for (int k = 0; k < d.n[2]; ++k)
                for (int j = 0; j < d.n[1]; ++j)
                    for (int i = 0; i < d.n[0]; ++i) {
                        e = std::max(e, std::abs(a(i, j, k) - b(i, j, k)));
                        m = std::max(m, std::abs(a(i, j, k)));
                    }
            e = comm.allreduce_max(e); m = comm.allreduce_max(m);
            err = std::max(err, e / m);
            if (open != 0) continue;
            // (b) plane wave, k = 2 pi (3 / Nx, 2 / Ny, 1 / Nz); staggered positions: Jx at (i, j, k), Jy at (i + 1/2, j + 1/2, k),
            // Jz at (i + 1/2, j, k + 1/2) in cells (the charge nodes at (i + 1/2, j, k))
            const double kx = 2 * M_PI * 3 / d.N[0], ky = 2 * M_PI * 2 / d.N[1], kz = 2 * M_PI * 1 / d.N[2];
            const double off[3][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}};
            for (int c = 0; c < 3; ++c) {
                H h(*J[c]);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i)
                            h(i, j, k) = std::cos(kx * (d.off[0] + i + off[c][0]) + ky * (d.off[1] + j + off[c][1]) + kz * (d.off[2] + k + off[c][2]));
                h.put(*J[c]);
            }
            filter_currents_binomial(f, w, d, comm, passes);
            const double t = std::pow(std::cos(kx / 2) * std::cos(kx / 2) * std::cos(ky / 2) * std::cos(ky / 2) * std::cos(kz / 2) * std::cos(kz / 2), passes);
            for (int c = 0; c < 3; ++c) {
                H h(*J[c]);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i)
                            terr = std::max(terr, std::abs(h(i, j, k) - t * std::cos(kx * (d.off[0] + i + off[c][0]) + ky * (d.off[1] + j + off[c][1]) +
                                                                                     kz * (d.off[2] + k + off[c][2]))));
            }
            terr = comm.allreduce_max(terr);
        }
    check("binomial filter: D J' = T D J (periodic; open x; open x, y, z), every component of a plane wave x prod cos^2(k h / 2)",
          err < 1e-12 && terr < 1e-13, fmt("%.1e, transfer function %.1e", err, terr));
}

// S9b. The compensated and the x-only binomial filter (numerics.filter_compensate, filter_axes = x; bench/nci/mechanism): as
//      S9, (a) D J' = T' D J with T' of filter_rho_cart, (b) a plane wave in every component times the transfer function
//      T_x^N ((N + 1) - N T_x) (compensated; Lee and Vay) and (T_y T_z)^N (all axes) or 1 (x only), T_a = cos^2(k_a h_a / 2).
static void solid_filter_variants(Comm& comm) {
    // the maximally flat factor (numerics.filter_flat, decision 109): its real factors reproduce
    // P(S) = sum_{j<L} binom(N-1+j, j) S^j for N = 1 .. 16, L = 2 .. 9
    {
        double perr = 0;
        for (int N = 1; N <= 16; ++N)
            for (int L = 2; L <= 9; ++L) {
                Domain d;
                set_flat_filter(d, N, L);
                for (int q = 0; q <= 20; ++q) {
                    const double S = q / 20.0;
                    double P = 0, b = 1;
                    for (int j = 0; j < L; ++j) { P += b * std::pow(S, j); b = b * (N + j) / (j + 1); }
                    perr = std::max(perr, std::abs(flat_filter_response(d, N, S) - std::pow(1 - S, N) * P) / P);
                }
            }
        check("flat filter factors: T^N P(S) for N = 1 - 16, L = 2 - 9", perr < 1e-12, fmt("%.1e", perr));
    }
    for (int var = 0; var < 5; ++var) {                  // compensated; x only; compensated and x only; flat L = 4, 6 (x only)
    const bool comp = var == 0 || var == 2, xonly = var != 0;
    const int L = var == 3 ? 4 : var == 4 ? 6 : 0;
    double err = 0, terr = 0;
    for (int open = 0; open < 3; ++open)
        for (int passes = 1; passes <= 3; ++passes) {
            Domain d;
            d.N[0] = 24; d.N[1] = 12; d.N[2] = 8;
            d.h[0] = 0.3; d.h[1] = 0.5; d.h[2] = 0.45;
            d.periodic[0] = open == 0; d.periodic[1] = d.periodic[2] = open < 2;
            d.ng = 3;
            d.filt_comp = comp; d.filt_xonly = xonly;
            set_flat_filter(d, passes, L);
            const int want[3] = {0, 0, 0};
            comm.decompose(d, want);
            Fields f(d);
            Field w(d), rho(d), s(d);
            std::mt19937_64 rng(31 + passes + 4 * open + 16 * var);
            std::uniform_real_distribution<double> u(-1, 1);
            Field* J[3] = {&f.jx, &f.jy, &f.jz};
            for (Field* x : J) {
                H h(*x);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i) h(i, j, k) = u(rng);
                h.put(*x);
            }
            auto gx = [&](int a, int gi) { return d.periodic[a] || (gi >= 0 && gi < d.N[a] - (a == 0 ? 1 : 0)); };
            auto div = [&](Field& out) {
                comm.exchange(d, {&f.jx, &f.jy, &f.jz});
                H jx(f.jx), jy(f.jy), jz(f.jz), o(out);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i) {
                            const bool g = gx(0, d.off[0] + i) && gx(1, d.off[1] + j) && gx(2, d.off[2] + k);
                            o(i, j, k) = g ? (jx(i + 1, j, k) - jx(i, j, k)) / d.h[0] + (jy(i, j, k) - jy(i, j - 1, k)) / d.h[1] +
                                                 (jz(i, j, k) - jz(i, j, k - 1)) / d.h[2]
                                           : 0.0;
                        }
                o.put(out);
            };
            div(rho);
            filter_rho_cart(rho, s, d, comm, passes);
            filter_currents_binomial(f, w, d, comm, passes);
            div(s);
            H a(rho), b(s);
            double e = 0, m = 0;
            for (int k = 0; k < d.n[2]; ++k)
                for (int j = 0; j < d.n[1]; ++j)
                    for (int i = 0; i < d.n[0]; ++i) {
                        e = std::max(e, std::abs(a(i, j, k) - b(i, j, k)));
                        m = std::max(m, std::abs(a(i, j, k)));
                    }
            e = comm.allreduce_max(e); m = comm.allreduce_max(m);
            err = std::max(err, e / m);
            if (open != 0) continue;
            // (b) plane wave, k = 2 pi (3 / Nx, 2 / Ny, 1 / Nz); staggered positions: Jx at (i, j, k), Jy at (i + 1/2, j + 1/2, k),
            // Jz at (i + 1/2, j, k + 1/2) in cells (the charge nodes at (i + 1/2, j, k))
            const double kx = 2 * M_PI * 3 / d.N[0], ky = 2 * M_PI * 2 / d.N[1], kz = 2 * M_PI * 1 / d.N[2];
            const double off[3][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}};
            for (int c = 0; c < 3; ++c) {
                H h(*J[c]);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i)
                            h(i, j, k) = std::cos(kx * (d.off[0] + i + off[c][0]) + ky * (d.off[1] + j + off[c][1]) + kz * (d.off[2] + k + off[c][2]));
                h.put(*J[c]);
            }
            filter_currents_binomial(f, w, d, comm, passes);
            const double Tx = std::cos(kx / 2) * std::cos(kx / 2), Ty = std::cos(ky / 2) * std::cos(ky / 2), Tz = std::cos(kz / 2) * std::cos(kz / 2);
            const double N = passes;
            double P = 1;
            if (L > 0) {                                     // Herrmann: sum_{j<L} binom(N-1+j, j) (1 - Tx)^j
                P = 0; double b = 1;
                for (int j = 0; j < L; ++j) { P += b * std::pow(1 - Tx, j); b = b * (N + j) / (j + 1); }
            }
            const double t = std::pow(Tx, N) * (comp ? (N + 1) - N * Tx : 1.0) * P * (xonly ? 1.0 : std::pow(Ty * Tz, N));
            for (int c = 0; c < 3; ++c) {
                H h(*J[c]);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i)
                            terr = std::max(terr, std::abs(h(i, j, k) - t * std::cos(kx * (d.off[0] + i + off[c][0]) + ky * (d.off[1] + j + off[c][1]) +
                                                                                     kz * (d.off[2] + k + off[c][2]))));
            }
            terr = comm.allreduce_max(terr);
        }
    check(std::string("binomial filter, ") + (var == 0 ? "compensated" : var == 1 ? "x only" : var == 2 ? "compensated, x only" :
                                              var == 3 ? "flat L = 4, x only" : "flat L = 6, x only") +
              ", 1 - 3 passes: D J' = T' D J (periodic; open x; open x, y, z), plane wave x the transfer function",
          err < 1e-12 && terr < 1e-13, fmt("%.1e, transfer function %.1e", err, terr));
    }
}

// S10. Numerical Cherenkov test with the filter (decision 91; bench/nci/README.md "Current filter"): examples/streaming_plasma.deck
//      on 16 x 8 x 8 cells (e-p plasma, u = -10 along x, n = 0.01 n_c, hy = 3 c dt = 1.88 c/wp, 299 steps = 188/wp), one
//      filter pass. W_field / K_e at the end: the projection filter smooths rho but not the transverse current (J != v rho at
//      the grid scale): unstable (0.59); the binomial filter keeps J = v rho: below the unfiltered level (1.0e-10 vs 1.4e-9).
static void solid_nci_filter(Comm& comm) {
    double r[3];
    for (int run = 0; run < 3; ++run) {                       // none, projection, binomial
        Config c;
        c.N[0] = 16; c.N[1] = 8; c.N[2] = 8;
        c.h[0] = 6.3; c.h[1] = c.h[2] = 18.8;
        c.t_end = 299 * c.h[0];
        c.filter_passes = run ? 1 : 0;
        c.filter_binomial = run == 2;
        SpeciesConfig e = species("e", -1, 1, 0.01), p = species("p", 1, 1836, 0.01);
        for (SpeciesConfig* s : {&e, &p})
            for (int a = 0; a < 3; ++a) { s->ppc[a] = 2; s->u0[a] = a ? 0.0 : -10.0; }
        for (int a = 0; a < 3; ++a) e.uth[a] = 1e-4;
        c.species = {e, p};
        c.check();
        Simulation sim(c, comm);
        sim.init();
        while (!sim.finished()) sim.step();
        double v[2] = {sim.fields->energy(), sim.species[0].kinetic_energy()};
        comm.allreduce_sum(v, 2);
        r[run] = v[0] / v[1];
    }
    check("NCI, streaming plasma (gamma 10), W_field / K_e after 188/wp: projection filter unstable, binomial below no filter",
          r[1] > 1e-2 && r[2] < 1e-9 && r[2] < r[0] && r[0] < 1e-8,
          fmt("none %.1e, projection %.1e", r[0], r[1]) + fmt(", binomial %.1e", r[2]));
}

// S8. The DFT lines of src/xlines.hpp (iRIP and the projection filter) against the direct sum, both signs: radix 2 and the
//     mixed-radix Stockham path (prime factors 2, 3, 5, 7), strided lines.
static void solid_dft_lines() {
    double err = 0;
    for (int n : {6, 7, 12, 15, 49, 64, 120, 576}) {
        const std::vector<double> tw = xlines::twiddles(n);
        std::mt19937_64 rng(n);
        std::uniform_real_distribution<double> u(-1, 1);
        const long st = 3;
        std::vector<double> a(size_t(2 * n * st)), t(size_t(2 * n * st)), x(size_t(2 * n));
        for (int l = 0; l < n; ++l) { x[size_t(2 * l)] = u(rng); x[size_t(2 * l + 1)] = u(rng); }
        for (int sign : {-1, 1}) {
            for (int l = 0; l < n; ++l) { a[size_t(2 * l * st)] = x[size_t(2 * l)]; a[size_t(2 * l * st + 1)] = x[size_t(2 * l + 1)]; }
            xlines::dft_line(a.data(), st, n, tw.data(), sign, t.data(), st);
            for (int q = 0; q < n; ++q) {
                double sr = 0, si = 0;
                for (int l = 0; l < n; ++l) {
                    const double ph = sign * 2 * M_PI * double((long(q) * l) % n) / n;
                    sr += x[size_t(2 * l)] * std::cos(ph) - x[size_t(2 * l + 1)] * std::sin(ph);
                    si += x[size_t(2 * l)] * std::sin(ph) + x[size_t(2 * l + 1)] * std::cos(ph);
                }
                err = std::max(err, std::hypot(a[size_t(2 * q * st)] - sr, a[size_t(2 * q * st + 1)] - si) / std::sqrt(double(n)));
            }
        }
    }
    check("DFT lines (radix 2, mixed radix) against the direct sum", err < 1e-13, fmt("%.1e", err));
}

void test_solid(Comm& comm) {
    solid_charge(comm);
    solid_target(comm);
    solid_filter(comm);
    solid_heating(comm);
    solid_wpdt(comm);
    solid_filter_radiation(comm);
    solid_filter_projection(comm);
    solid_filter_binomial(comm);
    solid_filter_variants(comm);
    solid_nci_filter(comm);
    solid_dft_lines();
}
