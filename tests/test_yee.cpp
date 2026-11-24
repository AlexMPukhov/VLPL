// Yee solver (docs/specs/yee.md).
#include <cmath>
#include <cstdio>
#include <random>
#include "common.hpp"
#include "sim.hpp"

static Domain yee_box(Comm& comm, int nx, int ny, int nz, double hx, double hy, double hz, double cfl) {
    Domain d;
    d.N[0] = nx; d.N[1] = ny; d.N[2] = nz;
    d.h[0] = hx; d.h[1] = hy; d.h[2] = hz;
    d.ng = 3;
    d.solver = Solver::Yee;
    d.tau = cfl / std::sqrt(1 / (hx * hx) + 1 / (hy * hy) + 1 / (hz * hz));
    const int want[3] = {0, 0, 0};
    comm.decompose(d, want);
    return d;
}

// Y1. Numerical dispersion, exactly: a standing wave E = sin(k.r) with B = 0 evolves as
//     E^n = cos(n theta) E^0, cos(theta) = 1 - dt^2 K^2 / 2, K_a = (2/h_a) sin(k_a h_a / 2) (the leapfrog
//     of the discrete curl curl), to round-off; along x (E_y) and oblique in x-y (E_z).
static void yee_dispersion(Comm& comm) {
    Domain d = yee_box(comm, 32, 24, 4, 0.4, 0.5, 0.6, 0.95);
    double worst = 0;
    for (int oblique = 0; oblique < 2; ++oblique) {
        Fields f(d);
        const double kx = 2 * M_PI * 3 / d.L(0), ky = oblique ? 2 * M_PI * 2 / d.L(1) : 0.0;
        Field& E = oblique ? f.ez : f.ey;
        H e(E);
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    // E_y at (i, j+1/2, k), E_z at (i, j, k+1/2): x = i hx, y = j hy (+ hy/2 for E_y)
                    const double x = (d.off[0] + i) * d.h[0], y = (d.off[1] + j + (oblique ? 0.0 : 0.5)) * d.h[1];
                    e(i, j, k) = std::sin(kx * x + ky * y);
                }
        e.put(E);
        comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
        const std::vector<double> e0 = E.host();
        const double Kx = 2 / d.h[0] * std::sin(kx * d.h[0] / 2), Ky = 2 / d.h[1] * std::sin(ky * d.h[1] / 2);
        const double dt = d.dt(), th = std::acos(1 - 0.5 * dt * dt * (Kx * Kx + Ky * Ky));
        const int steps = 400;
        for (int n = 0; n < steps; ++n) { f.zero_currents(); f.step(comm); }
        const std::vector<double> e1 = E.host();
        const double c = std::cos(steps * th);
        double err = 0;
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) err = std::max(err, std::abs(e1[E.idx(i, j, k)] - c * e0[E.idx(i, j, k)]));
        worst = std::max(worst, comm.allreduce_max(err));
    }
    check("yee: standing waves along x and oblique follow the discrete dispersion exactly", worst < 1e-11,
          fmt("max |E - cos(n theta) E0| %.1e after 400 steps", worst));
}

// Y2. Random vacuum fields: energy bounded over 2000 steps, div B (cell centres) unchanged.
static void yee_vacuum_random(Comm& comm) {
    Domain d = yee_box(comm, 16, 12, 10, 0.3, 0.5, 0.6, 0.95);
    Fields f(d);
    std::mt19937_64 rng(3 + comm.rank);
    std::normal_distribution<double> g;
    for (Field* F : {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz}) {
        H h(*F);
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) h(i, j, k) = g(rng);
        h.put(*F);
    }
    comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
    auto divb = [&]() {
        H bx(f.bx), by(f.by), bz(f.bz);
        std::vector<double> v;
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i)
                    v.push_back((bx(i + 1, j, k) - bx(i, j, k)) / d.h[0] + (by(i, j + 1, k) - by(i, j, k)) / d.h[1] +
                                (bz(i, j, k + 1) - bz(i, j, k)) / d.h[2]);
        return v;
    };
    const std::vector<double> b0 = divb();
    auto energy = [&]() { double e = f.energy(); comm.allreduce_sum(&e, 1); return e; };
    const double w0 = energy();
    double wmax = w0, wmin = w0, w1 = 0, w2 = 0;       // w1, w2: the largest W of the first and second 1000 steps
    for (int n = 0; n < 2000; ++n) {
        f.zero_currents(); f.step(comm);
        const double w = energy();
        wmax = std::max(wmax, w); wmin = std::min(wmin, w);
        (n < 1000 ? w1 : w2) = std::max(n < 1000 ? w1 : w2, w);
    }
    const std::vector<double> b1 = divb();
    double e = 0;
    for (size_t q = 0; q < b0.size(); ++q) e = std::max(e, std::abs(b1[q] - b0[q]));
    e = comm.allreduce_max(e);
    // E^2 + B^2 with B at integer levels is not the leapfrog's invariant (E^2 + B^{n-1/2} B^{n+1/2}): it
    // oscillates near the Courant limit, but must not grow
    check("yee: random vacuum fields, energy bounded, div B unchanged (2000 steps)", w2 <= w1 * 1.001 && wmax / w0 < 1.5 && e < 1e-11,
          fmt("W/W0 in [%.3f, ", wmin / w0) + fmt("%.3f]", wmax / w0) + fmt(", max W second / first half %.4f", w2 / w1) +
              fmt(", div B change %.1e", e));
}

// Y3. Plasma (both shapes, neutral with a drift across the periodic boundary, and a non-neutral slab):
//     Gauss law div E = rho at the nodes and div B = 0 kept to round-off (Esirkepov on the Yee mesh).
static void yee_plasma(Comm& comm) {
    for (int order : {1, 2})
        for (int neutral : {1, 0}) {
            Config c;
            c.N[0] = 12; c.N[1] = 10; c.N[2] = 8;
            c.h[0] = 0.3; c.h[1] = 0.5; c.h[2] = 0.6;
            c.solver = Solver::Yee; c.cfl = 0.9;
            c.t_end = 60 * c.dt();
            c.order = order;
            SpeciesConfig e;
            e.name = "e"; e.q = -1; e.m = 1; e.density = 0.5;
            e.ppc[0] = e.ppc[1] = e.ppc[2] = 2;
            for (int a = 0; a < 3; ++a) { e.uth[a] = 0.3; e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
            e.u0[0] = neutral ? -3.0 : 0.0;
            c.species.push_back(e);
            if (neutral) {
                SpeciesConfig p = e;
                p.name = "p"; p.q = 1; p.m = 10; p.uth[0] = p.uth[1] = p.uth[2] = 0.05;
                c.species.push_back(p);
            } else {
                c.species[0].rmin[0] = 0.5; c.species[0].rmax[0] = 2.0;
            }
            c.check();
            Simulation sim(c, comm);
            sim.init();
            double g = 0, db = 0;
            while (!sim.finished()) {
                sim.step();
                g = std::max(g, sim.gauss_drift());
                db = std::max(db, comm.allreduce_max(sim.fields->max_div_b()));
            }
            double wf = sim.fields->energy();
            comm.allreduce_sum(&wf, 1);
            const std::string tag = std::string("yee: plasma shape ") + std::to_string(order) +
                                    (neutral ? ", neutral, drift -3" : ", electron slab");
            check(tag + ": Gauss law and div B to round-off", g < 1e-12 && db < 1e-12 && wf > 0,
                  fmt("max|dGauss| %.1e", g) + fmt(", max|divB| %.1e", db));
        }
}

// Y4. Langmuir oscillation of cold electrons: sin(w dt/2) = wp dt/2 (the leapfrog), as for RIP-CC.
static void yee_langmuir(Comm& comm) {
    Config c;
    c.N[0] = 32; c.N[1] = 4; c.N[2] = 4;
    c.h[0] = 0.2; c.h[1] = 0.5; c.h[2] = 0.5;
    c.solver = Solver::Yee; c.cfl = 0.9;
    const double wp = 2.0;
    c.t_end = 6 * 2 * M_PI / wp;
    SpeciesConfig e;
    e.name = "e"; e.q = -1; e.m = 1; e.density = wp * wp; e.ppc[0] = 4;
    for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
    SpeciesConfig ion = e; ion.name = "i"; ion.q = 1; ion.m = 1e12;
    c.species = {e, ion};
    c.check();
    Simulation sim(c, comm);
    sim.init();
    const double k = 2 * M_PI / (c.N[0] * c.h[0]);
    {
        Species& el = sim.species[0];
        const std::vector<double> x = el.get(Species::X);
        std::vector<double> ux(x.size());
        for (size_t p = 0; p < x.size(); ++p) ux[p] = 1e-3 * std::sin(k * x[p]);
        el.set(Species::UX, ux);
    }
    std::vector<double> ex;
    while (!sim.finished()) {
        sim.step();
        double v = 0;
        if (sim.dom.off[0] <= 5 && 5 < sim.dom.off[0] + sim.dom.n[0] && sim.dom.off[1] == 0 && sim.dom.off[2] == 0)
            v = H(sim.fields->ex)(5 - sim.dom.off[0], 1, 1);
        comm.allreduce_sum(&v, 1);
        ex.push_back(v);
    }
    std::vector<double> zc;
    for (size_t n = 1; n < ex.size(); ++n)
        if (ex[n - 1] * ex[n] < 0) zc.push_back(double(n) - 1 + ex[n - 1] / (ex[n - 1] - ex[n]));
    const double dt = c.dt();
    if (zc.size() < 3) { check("yee: Langmuir oscillation frequency", false, "no oscillation"); return; }
    const double w = M_PI * double(zc.size() - 1) / ((zc.back() - zc.front()) * dt);
    const double wth = 2 / dt * std::asin(wp * dt / 2);
    check("yee: Langmuir oscillation frequency", std::abs(w / wth - 1) < 0.01, fmt("w = %.4f, expected %.4f", w, wth));
}

void test_yee(Comm& comm) {
    yee_dispersion(comm);
    yee_vacuum_random(comm);
    yee_plasma(comm);
    yee_langmuir(comm);
}
