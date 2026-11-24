// O6: the energy-conserving implicit mover with the iRIP box closure (numerics.mover = implicit,
// docs/specs/implicit_mover.md; numpy prototype research/o6_mover/).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include "common.hpp"
#include "deck.hpp"
#include "implicit_mover.hpp"
#include "rz.hpp"
#include "sim.hpp"

// M1. The work field of the box closure: W^{n+1} - W^n = -dt dV sum J . Et with random fields and currents, h_perp < c dt,
//     in 3D and 2D (irip_trial gives Et; the field step must not change in the trial).
static void mover_work_field(Comm& comm) {
    for (int flat = 0; flat < 2; ++flat) {
        Domain d;
        d.N[0] = 10; d.N[1] = 8; d.N[2] = flat ? 1 : 6;
        d.h[0] = 0.4; d.h[1] = 0.3; d.h[2] = flat ? 1.0 : 0.25;
        d.flat = flat == 1;
        d.ng = 3;
        d.solver = Solver::IRip;
        const int want[3] = {0, 1, 1};
        comm.decompose(d, want);
        Fields F(d);
        std::mt19937_64 rng(7 + flat);
        std::normal_distribution<double> nd;
        for (Field* f : {&F.ex, &F.ey, &F.ez, &F.bx, &F.by, &F.bz, &F.jx, &F.jy, &F.jz}) {
            H h(*f);
            for (int k = 0; k < d.n[2]; ++k)
                for (int j = 0; j < d.n[1]; ++j)
                    for (int i = 0; i < d.n[0]; ++i) h(i, j, k) = nd(rng);
            h.put(*f);
        }
        comm.exchange(d, {&F.ex, &F.ey, &F.ez, &F.bx, &F.by, &F.bz});
        Field et[3], bb[3];
        for (int c = 0; c < 3; ++c) { et[c].resize(d); bb[c].resize(d); }
        double w0 = F.energy();
        comm.allreduce_sum(&w0, 1);
        const std::vector<double> ex0 = F.ex.host();
        irip_trial(F, d, comm, {}, 0.0, et, bb);
        const bool same = F.ex.host() == ex0;
        double work = 0;
        {
            const Field* J[3] = {&F.jx, &F.jy, &F.jz};
            for (int c = 0; c < 3; ++c) {
                H a(*J[c]), b(et[c]);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i) work += a(i, j, k) * b(i, j, k);
            }
            comm.allreduce_sum(&work, 1);
            work *= -d.dt() * d.cell_volume();
        }
        F.step(comm);
        double w1 = F.energy();
        comm.allreduce_sum(&w1, 1);
        const double err = std::abs(w1 - w0 - work) / std::abs(work);
        check(std::string("mover: work field of the box closure, ") + (flat ? "2D" : "3D") + ", h_perp < c dt",
              err < 1e-12 && same, fmt("|dW - work| / |work| %.1e", err) + (same ? "" : ", the trial changed E"));
    }
}

static Config plasma_config(bool flat, double wpdt, double uth, int order) {
    Config c;
    c.flat = flat;
    c.N[0] = 16; c.N[1] = 12; c.N[2] = flat ? 1 : 8;
    c.h[0] = 0.5; c.h[1] = 0.25; c.h[2] = flat ? 1.0 : 0.3;
    c.solver = Solver::IRip;
    c.order = order;
    c.mover_implicit = true;
    c.mover_tol = 1e-12;
    c.mover_maxit = 200;
    const double wp = wpdt / c.h[0];
    SpeciesConfig e;
    e.name = "e"; e.q = -1; e.m = 1; e.density = wp * wp;
    e.ppc[0] = 2; e.ppc[1] = 2; e.ppc[2] = flat ? 1 : 2;
    for (int a = 0; a < 3; ++a) { e.uth[a] = uth; e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
    SpeciesConfig i = e;
    i.name = "i"; i.q = 1; i.m = 1836; i.mobile = false;
    for (int a = 0; a < 3; ++a) i.uth[a] = 0;
    c.species = {e, i};
    return c;
}

static double total_energy(Simulation& sim, Comm& comm, double* kin = nullptr) {
    double w = sim.fields->energy(), k = 0;
    for (const Species& s : sim.species) k += s.kinetic_energy();
    comm.allreduce_sum(&w, 1);
    comm.allreduce_sum(&k, 1);
    if (kin) *kin = k;
    return w + k;
}

// M2. Hot relativistic plasma beyond the explicit limit (omega_p dt = 3, u_th = 0.5, h_perp = h_x / 2: moves of several
//     transverse cells, Esirkepov segments): W + K conserved to the solve tolerance, Gauss law and div B at round-off;
//     3D and 2D, shapes 2 and 3.
static void mover_energy(Comm& comm) {
    for (int flat = 0; flat < 2; ++flat)
        for (int order : {2, 3}) {
            Config c = plasma_config(flat == 1, 3.0, 0.5, order);
            c.t_end = 20 * c.dt();
            c.check();
            Simulation sim(c, comm);
            sim.init();
            const double e0 = total_energy(sim, comm);
            double dev = 0, g = 0, db = 0;
            while (!sim.finished()) {
                sim.step();
                dev = std::max(dev, std::abs(total_energy(sim, comm) / e0 - 1));
                g = std::max(g, sim.gauss_drift());
                db = std::max(db, comm.allreduce_max(sim.fields->max_div_b()));
            }
            const MoverStats& st = implicit_mover_stats(sim);
            check(std::string("mover: hot plasma at omega_p dt = 3, ") + (flat ? "2D" : "3D") + ", shape " + std::to_string(order) +
                      ": W + K, Gauss law, div B",
                  dev < 1e-9 && g < 1e-12 && db < 1e-12 && st.unconverged == 0,
                  fmt("max |E/E0 - 1| %.1e", dev) + fmt(", Gauss %.1e", g) + fmt(", div B %.1e", db) +
                      fmt(", iterations %.1f per step", double(st.iterations) / st.steps));
        }
}

// M3. Cold Langmuir oscillation, k = 2 pi / L, omega_p dt = 0.5, 3, 10: the Crank-Nicolson frequency
//     cos(w dt) = (1 - a^2) / (1 + a^2), a = W dt / 2, W^2 = wp^2 sinc^(2 s)(k dx / 2) (the transposed gather), from the
//     recurrence e^{n+1} + e^{n-1} = 2 cos(w dt) e^n; the explicit push is unstable above omega_p dt = 2.
static void mover_langmuir(Comm& comm) {
    for (double wpdt : {0.5, 3.0, 10.0}) {
        Config c;
        c.flat = true;
        c.N[0] = 32; c.N[1] = 4; c.N[2] = 1;
        c.h[0] = 0.4; c.h[1] = 0.4; c.h[2] = 1.0;
        c.solver = Solver::IRip;
        c.order = 2;
        c.mover_implicit = true;
        c.mover_tol = 1e-12;
        c.mover_maxit = 200;
        const double wp = wpdt / c.h[0];
        c.t_end = 60 * c.dt();
        SpeciesConfig e;
        e.name = "e"; e.q = -1; e.m = 1; e.density = wp * wp; e.ppc[0] = 4;
        for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
        SpeciesConfig ion = e; ion.name = "i"; ion.q = 1; ion.m = 1836; ion.mobile = false;
        c.species = {e, ion};
        c.check();
        Simulation sim(c, comm);
        sim.init();
        const double k = 2 * M_PI / (c.N[0] * c.h[0]);
        {
            Species& el = sim.species[0];
            const std::vector<double> x = el.get(Species::X);
            std::vector<double> ux(x.size());
            for (size_t p = 0; p < x.size(); ++p) ux[p] = 1e-4 * std::sin(k * x[p]);
            el.set(Species::UX, ux);
        }
        std::vector<double> ex;
        while (!sim.finished()) {
            sim.step();
            H h(sim.fields->ex);
            double a = 0;
            for (int i = 0; i < sim.dom.n[0]; ++i) a += h(i, 0, 0) * std::sin(k * (sim.dom.off[0] + i) * c.h[0]);
            comm.allreduce_sum(&a, 1);
            ex.push_back(a);
        }
        double num = 0, den = 0;
        for (size_t n = 1; n + 1 < ex.size(); ++n) { num += (ex[n + 1] + ex[n - 1]) * ex[n]; den += 2 * ex[n] * ex[n]; }
        const double s = std::sin(0.5 * k * c.h[0]) / (0.5 * k * c.h[0]);
        const double a = 0.5 * wpdt * std::pow(s, c.order), cth = (1 - a * a) / (1 + a * a), cw = num / den;
        const double w = std::acos(std::max(-1.0, std::min(1.0, cw))), wt = std::acos(cth);
        check("mover: cold Langmuir oscillation at omega_p dt = " + fmt("%.1f", wpdt) + ", Crank-Nicolson frequency",
              std::abs(w / wt - 1) < 1e-4, fmt("w dt = %.6f", w) + fmt(", expected %.6f", wt));
    }
}

// M4. Grid heating at an unresolved Debye length (2D, 1 keV-like u_th = 0.044 at dx / lambda_D = 14, dy = 3.4 dx,
//     omega_p dt = 0.62, 300 steps, no filter): the implicit mover keeps W + K to 1e-8 (tolerance 1e-10) and K within a few
//     per cent of K0 (the initial exchange with the field fluctuations); the explicit push of the same deck heats.
static void mover_heating(Comm& comm) {
    double dK[2] = {0, 0}, dE[2] = {0, 0};
    for (int imp = 0; imp < 2; ++imp) {
        Config c;
        c.flat = true;
        c.N[0] = 32; c.N[1] = 12; c.N[2] = 1;
        c.h[0] = 0.0982; c.h[1] = 0.334; c.h[2] = 1.0;
        c.solver = Solver::IRip;
        c.order = 2;
        c.mover_implicit = imp == 1;
        c.mover_tol = 1e-10;
        c.t_end = 300 * c.dt();
        SpeciesConfig e;
        e.name = "e"; e.q = -1; e.m = 1; e.density = 40;
        e.ppc[0] = 4; e.ppc[1] = 4; e.ppc[2] = 1;
        for (int a = 0; a < 3; ++a) { e.uth[a] = 0.044; e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
        SpeciesConfig i = e;
        i.name = "i"; i.q = 1; i.m = 1836; i.mobile = false;
        for (int a = 0; a < 3; ++a) i.uth[a] = 0;
        c.species = {e, i};
        c.check();
        Simulation sim(c, comm);
        sim.init();
        double k0;
        const double e0 = total_energy(sim, comm, &k0);
        double k1 = k0, e1 = e0;
        while (!sim.finished()) sim.step();
        e1 = total_energy(sim, comm, &k1);
        dK[imp] = k1 / k0 - 1;
        dE[imp] = e1 / e0 - 1;
    }
    check("mover: no grid heating at dx / lambda_D = 14 (2D, 300 steps)",
          std::abs(dE[1]) < 1e-8 && std::abs(dK[1]) < 0.05 && dE[0] > 10 * std::abs(dE[1]),
          fmt("implicit: dE/E %.1e", dE[1]) + fmt(", dK/K %.3f", dK[1]) + fmt("; explicit: dE/E %.1e", dE[0]) +
              fmt(", dK/K %.3f", dK[0]));
}

// M5. What the first version rejects, and that it lifts the omega_p dt < 2 check.
static void mover_config() {
    Config c = plasma_config(false, 5.0, 0.0, 2);
    bool ok = true;
    try { c.check(); } catch (const std::exception&) { ok = false; }
    int rejected = 0;
    auto expect_reject = [&](Config x) { try { x.check(); } catch (const std::exception&) { ++rejected; } };
    { Config x = c; x.order = 1; expect_reject(x); }
    { Config x = c; x.o5 = true; expect_reject(x); }
    { Config x = c; x.solver = Solver::RipCC; expect_reject(x); }
    { Config x = c; x.filter_passes = 1; expect_reject(x); }
    { Config x = c; x.pml.on[1] = true; x.periodic[1] = false; x.pml.cells = 4; expect_reject(x); }
    { Config x = c; x.mover_implicit = false; expect_reject(x); }        // explicit at omega_p dt = 5: unstable
    check("mover: omega_p dt = 5 accepted, unsupported options rejected", ok && rejected == 6, fmt("rejected %.0f of 6", rejected));
}

// M6. RZ (decision 82): the work field of the RZ box step, W^{n+1} - W^n = -dt sum_m w_m dV J_m . Et_m, with random fields
//     and currents in 3 modes on a stretched radial grid (dz = 2.5 dr: beyond the explicit RZ limit), and Bbar = (B^n + B^{n+1})/2;
//     the trial must not change the fields. Axis values that are not degrees of freedom are zero (E_z of m >= 1, E_phi, B_r).
static void mover_rz_work_field(Comm& comm) {
    const int M = 3;
    Domain d = rz_domain(comm, 10, 14, M, 0.4, 0.16, true, Solver::IRip);
    FieldsRZ F(d, M, rz_radial_grid(14, 0.16, 0.8, 1.15));
    std::mt19937_64 rng(11);
    std::normal_distribution<double> nd;
    for (Field* f : {&F.ez, &F.er, &F.ep, &F.bz, &F.br, &F.bp, &F.jz, &F.jr, &F.jp}) {
        H h(*f);
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    double v = nd(rng);
                    if (j == 0 && ((f == &F.ez && k > 0) || f == &F.ep || f == &F.br)) v = 0;
                    h(i, j, k) = v;
                }
        h.put(*f);
    }
    comm.exchange(d, {&F.ez, &F.er, &F.ep, &F.bz, &F.br, &F.bp});
    Field et[3], bb[3];
    for (int c = 0; c < 3; ++c) { et[c].resize(d); bb[c].resize(d); }
    double w0 = F.energy();
    comm.allreduce_sum(&w0, 1);
    const std::vector<double> ez0 = F.ez.host(), er0 = F.er.host(), bz0 = F.bz.host();
    rz_irip_trial(F, comm, 0.0, et, bb);
    const bool same = F.ez.host() == ez0 && F.er.host() == er0;
    const RadialGrid& g = F.grid();
    double work = 0;
    {
        H jz(F.jz), jr(F.jr), jp(F.jp), tz(et[0]), tr(et[1]), tp(et[2]);
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i)
                    work += (k == 0 ? 1.0 : 2.0) * 2 * M_PI * d.h[0] *
                            (g.A[size_t(j)] * (jz(i, j, k) * tz(i, j, k) + jp(i, j, k) * tp(i, j, k)) + g.B[size_t(j)] * jr(i, j, k) * tr(i, j, k));
        comm.allreduce_sum(&work, 1);
        work *= -d.dt();
    }
    F.step(comm);
    double w1 = F.energy();
    comm.allreduce_sum(&w1, 1);
    const double err = std::abs(w1 - w0 - work) / std::abs(work);
    double berr = 0;
    {
        H b0(bb[0]), bz(F.bz);
        const std::vector<double>& z0 = bz0;
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i)
                    berr = std::max(berr, std::abs(b0(i, j, k) - 0.5 * (z0[F.bz.idx(i, j, k)] + bz(i, j, k))));
        berr = comm.allreduce_max(berr);
    }
    check("mover RZ: work field of the box step, 3 modes, stretched radial grid, dz = 2.5 dr",
          err < 1e-12 && berr < 1e-14 && same,
          fmt("|dW - work| / |work| %.1e", err) + fmt(", Bbar_z %.1e", berr) + (same ? "" : ", the trial changed the fields"));
}

// M7. RZ hot plasma column beyond the explicit limit (omega_p dt = 2.5, u_th = 0.2, m = 0, dz = 2 dr, many paths through the
//     axis; a uniform and a stretched radial grid): the uniform density loaded in the C1 map is q n on every node inside the
//     column, the axis node too (to the loading quadrature: 8 Gauss points per cell, 1e-3; 9e-2 with 2, research/o6_mover/
//     c1_loading.py); W + K conserved to the solve tolerance, the Gauss law at round-off. With m = 0, 1 (modes = 2, uniform grid)
//     rare steps stay unconverged (a path within ~0.01 dr of the axis, decision 82; which steps depends on the OpenMP summation
//     order): Gauss law at round-off, W + K to 1e-2 over the run, at most 8 of 30 steps unconverged (10 runs: 5e-5 ... 4.8e-3,
//     2 ... 5 steps). Unsupported RZ options rejected (m >= 2: decision 82).
static void mover_rz_plasma(Comm& comm) {
    const char* path = "mover_rz_tmp.deck";
    for (int run = 0; run < 3; ++run) {
        const int stretched = run == 1, m1 = run == 2;
        {
            std::ofstream f(path);
            f << "grid {\n geometry = rz\n modes = " << (m1 ? 2 : 1) << "\n lambda0 = 1 um\n nz = 16\n nr = 12\n dz = 0.2 um\n dr = 0.1 um\n t_end = 20 fs\n"
              << (stretched ? " r_uniform = 0.6 um\n stretch = 1.1\n" : "") << "}\n"
              << "numerics {\n solver = irip\n shape = 2\n mover = implicit\n mover_tol = 1e-12\n mover_maxit = 200\n}\n"
                 "species e {\n charge = -1\n mass = 1\n density = 4\n rmax = 0.6 um\n ppc_r = 8\n ppc_theta = 8\n"
                 " uth_z = 0.2\n uth_x = 0.2\n uth_y = 0.2\n}\n"
                 "species i {\n charge = 1\n mass = 1836\n density = 4\n rmax = 0.6 um\n ppc_r = 8\n ppc_theta = 8\n}\n"
                 "output {\n dir = mover_rz_tmp_out\n}\n";
        }
        Deck dk = Deck::read(path);
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        Field rho(sim.dom);
        rho.fill(0);
        rz_rho(sim.species[0], *sim.rz, rho, 2);
        comm.add_guards(sim.dom, {&rho});
        double lerr = 0;
        {
            const std::vector<double> h = rho.host();
            const RadialGrid& g = sim.rz->grid();
            for (int j = 0; g.r[size_t(j) + 2] < 0.6 * 2 * M_PI; ++j)       // nodes whose shape lies inside the column
                for (int i = 0; i < sim.dom.n[0]; ++i) lerr = std::max(lerr, std::abs(h[rho.idx(i, j, 0)] + 4) / 4);
            lerr = comm.allreduce_max(lerr);
        }
        double lost = 0;
        auto total = [&]() {                                      // with the energy of the electrons taken out at the wall
            double w = sim.rz->energy(), k = 0, l = 0;
            for (const Species& s : sim.species) { k += s.kinetic_energy(); l += s.lost_energy; }
            comm.allreduce_sum(&w, 1);
            comm.allreduce_sum(&k, 1);
            comm.allreduce_sum(&l, 1);
            lost = l;
            return w + k + l;
        };
        const double e0 = total();
        double dev = 0, gd = 0;
        while (!sim.finished()) {
            sim.step();
            dev = std::max(dev, std::abs(total() / e0 - 1));
            gd = std::max(gd, sim.gauss_drift());
        }
        const MoverStats& st = implicit_mover_stats(sim);
        const bool ok = m1 ? lerr < 2e-3 && dev < 1e-2 && gd < 1e-12 && st.unconverged <= 8
                           : lerr < 2e-3 && dev < 1e-9 && gd < 1e-12 && st.unconverged == 0;
        check(std::string("mover RZ: hot plasma column at omega_p dt = 2.5, ") + (m1 ? "m = 0, 1" : "m = 0") + ", " +
                  (stretched ? "stretched" : "uniform") + " radial grid: density in the C1 map, W + K (+ lost at the wall), Gauss law" +
                  (m1 ? " (rare unconverged steps)" : ""),
              ok,
              fmt("rho error %.1e", lerr) + fmt(", max |E/E0 - 1| %.1e", dev) + fmt(" (lost %.1e E0)", lost / e0) + fmt(", Gauss %.1e", gd) +
                  fmt(", iterations %.1f per step", double(st.iterations) / st.steps) + fmt(" (max %.0f)", double(st.max_iterations)) +
                  fmt(", unconverged %.0f", double(st.unconverged)) + fmt(" of %.0f", double(st.steps)));
    }
    Deck dk = Deck::read(path);
    int rejected = 0;
    for (const char* o : {"numerics.shape=1", "numerics.closure=o5", "numerics.filter_passes=1", "grid.modes=3"}) {
        Deck x = dk;
        x.override_from(o);
        try { Config::from_deck(x); } catch (const std::exception&) { ++rejected; }
    }
    check("mover RZ: shape 1, closure o5, current filter, modes = 3 rejected", rejected == 4, fmt("rejected %.0f of 4", rejected));
    if (comm.rank == 0) std::remove(path);
}

void test_mover(Comm& comm) {
    mover_work_field(comm);
    mover_rz_work_field(comm);
    mover_config();
    if (comm.size > 1) return;
    mover_energy(comm);
    mover_langmuir(comm);
    mover_heating(comm);
    mover_rz_plasma(comm);
}
