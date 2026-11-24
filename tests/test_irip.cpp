// iRIP, box closure (docs/specs/implicit_rip.md): the implicit RIP-CC step without the Courant limit.
#include <cmath>
#include <cstdio>
#include <random>
#include "common.hpp"
#include "sim.hpp"

static Domain irip_box(Comm& comm, int nx, int ny, int nz, double hx, double hy, double hz, bool open_x = false) {
    Domain d;
    d.periodic[0] = !open_x;
    d.N[0] = nx; d.N[1] = ny; d.N[2] = nz;
    d.h[0] = hx; d.h[1] = hy; d.h[2] = hz;
    d.ng = 3;
    d.solver = Solver::IRip;
    const int want[3] = {0, 0, 0};
    comm.decompose(d, want);
    return d;
}

static void fill(Field& F, const Domain& d, double (*f)(const Domain&, int, int, int)) {
    H h(F);
    for (int k = 0; k < d.n[2]; ++k)
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i) h(i, j, k) = f(d, i, j, k);
    h.put(F);
}

// I1. Vacuum waves along x keep RIP-CC's exact transport: E_y = B_z = f(x), E_z = -B_y = g(x) move one
//     cell per step at c, with h_perp = h_x / 5 (beyond RIP-CC's Courant limit).
static void irip_x_wave(Comm& comm) {
    Domain d = irip_box(comm, 24, 6, 5, 0.5, 0.1, 0.1);
    Fields f(d);
    auto prof = [](int i, int, int c) { return std::exp(-0.1 * (i - 8.0) * (i - 8.0)) * std::sin(0.7 * i + c); };
    {
        H ey(f.ey), bz(f.bz), ez(f.ez), by(f.by);
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    ey(i, j, k) = bz(i, j, k) = prof(i, d.n[0], 0);
                    ez(i, j, k) = prof(i, d.n[0], 1); by(i, j, k) = -ez(i, j, k);
                }
        ey.put(f.ey); bz.put(f.bz); ez.put(f.ez); by.put(f.by);
    }
    comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
    const int steps = 10;
    for (int n = 0; n < steps; ++n) { f.zero_currents(); f.step(comm); }
    H ey(f.ey), bz(f.bz), ez(f.ez), by(f.by), ex(f.ex), bx(f.bx);
    double err = 0;
    for (int k = 0; k < d.n[2]; ++k)
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                const int s = (i - steps + d.n[0]) % d.n[0];
                err = std::max({err, std::abs(ey(i, j, k) - prof(s, d.n[0], 0)), std::abs(bz(i, j, k) - prof(s, d.n[0], 0)),
                                std::abs(ez(i, j, k) - prof(s, d.n[0], 1)), std::abs(by(i, j, k) + prof(s, d.n[0], 1)),
                                std::abs(ex(i, j, k)), std::abs(bx(i, j, k))});
            }
    check("irip: vacuum waves along x move one cell per step (h_perp = h_x / 5)", err < 1e-13,
          fmt("max error %.1e after 10 steps", err));
}

// I2. Numerical dispersion of an oblique Fourier mode: tan^2(w dt/2) = tan^2(kx h_x/2) + h_x^2 lambda/4,
//     lambda = 4/hy^2 sin^2(ky hy/2) + 4/hz^2 sin^2(kz hz/2) (research/irip_3d). A single Fourier
//     harmonic has the static part and +-w only, so the increments d^n = E^{n+1} - E^n obey
//     d^{n+1} + d^{n-1} = 2 cos(w dt) d^n exactly. Radix-2 (nz = 8) and direct (ny = 6) transforms.
static void irip_dispersion(Comm& comm) {
    Domain d = irip_box(comm, 10, 6, 8, 0.5, 0.12, 0.15);
    double worst = 0, wdt = 0, pred = 0;
    for (int c = 0; c < 2; ++c) {
        Fields f(d);
        static int mode;
        mode = c;
        auto init = [](const Domain& d, int i, int j, int k) {
            const double kx = 2 * M_PI * (mode ? 3 : 1) / d.N[0], ky = 2 * M_PI * 1 / d.N[1], kz = 2 * M_PI * (mode ? 3 : 2) / d.N[2];
            return std::cos(kx * i + ky * j + kz * k + 0.3);
        };
        fill(mode ? f.bx : f.ey, d, init);      // TE and TM content
        fill(mode ? f.ez : f.ex, d, init);
        comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
        const double kx = 2 * M_PI * (c ? 3 : 1) / (d.N[0] * d.h[0]), ky = 2 * M_PI / (d.N[1] * d.h[1]),
                     kz = 2 * M_PI * (c ? 3 : 2) / (d.N[2] * d.h[2]);
        const double sy = std::sin(ky * d.h[1] / 2), sz = std::sin(kz * d.h[2] / 2), tx = std::tan(kx * d.h[0] / 2);
        const double lam = 4 * sy * sy / (d.h[1] * d.h[1]) + 4 * sz * sz / (d.h[2] * d.h[2]);
        pred = 2 * std::atan(std::sqrt(tx * tx + d.h[0] * d.h[0] * lam / 4));
        std::vector<std::vector<double>> hist;
        for (int n = 0; n < 40; ++n) {
            std::vector<double> v = f.ey.host(), b = f.bx.host();
            v.insert(v.end(), b.begin(), b.end());
            hist.push_back(v);
            f.zero_currents(); f.step(comm);
        }
        // least squares for cos(w dt) over all points and steps, then the residual
        double num = 0, den = 0;
        for (size_t n = 1; n + 2 < hist.size(); ++n)
            for (size_t q = 0; q < hist[n].size(); ++q) {
                const double dm = hist[n][q] - hist[n - 1][q], d0 = hist[n + 1][q] - hist[n][q], dp = hist[n + 2][q] - hist[n + 1][q];
                num += (dp + dm) * 2 * d0; den += 4 * d0 * d0;
            }
        const double cw = num / den;
        double res = 0, scale = 0;
        for (size_t n = 1; n + 2 < hist.size(); ++n)
            for (size_t q = 0; q < hist[n].size(); ++q) {
                const double dm = hist[n][q] - hist[n - 1][q], d0 = hist[n + 1][q] - hist[n][q], dp = hist[n + 2][q] - hist[n + 1][q];
                res = std::max(res, std::abs(dp + dm - 2 * cw * d0)); scale = std::max(scale, std::abs(d0));
            }
        wdt = std::acos(cw);
        worst = std::max({worst, std::abs(wdt - pred), res / scale});
    }
    check("irip: oblique modes follow tan^2(w dt/2) = tan^2(kx dx/2) + dx^2 lambda/4 (dx/h_perp = 4)", worst < 1e-11,
          fmt("w dt %.6f", wdt) + fmt(" vs %.6f, ", pred) + fmt("max error %.1e", worst));
}

// I3. Random vacuum fields at h_x / h_perp = 10: E^2 + B^2 is the invariant of the box step (to the solver's
//     round-off); div B unchanged.
static void irip_vacuum_random(Comm& comm) {
    Domain d = irip_box(comm, 12, 16, 10, 1.0, 0.1, 0.1);
    Fields f(d);
    std::mt19937_64 rng(5);
    std::normal_distribution<double> g;
    for (Field* F : {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz}) {
        H h(*F);
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) h(i, j, k) = g(rng);
        h.put(*F);
    }
    comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
    const double b0 = f.max_div_b(), w0 = f.energy();
    double dev = 0, db = 0;
    for (int n = 0; n < 500; ++n) {
        f.zero_currents(); f.step(comm);
        dev = std::max(dev, std::abs(f.energy() / w0 - 1));
    }
    // max_div_b is of the RIP-CC div B; random B has div B != 0, which must not change
    db = std::abs(f.max_div_b() - b0) / b0;
    check("irip: random vacuum fields at dx/h_perp = 10, energy conserved, div B unchanged (500 steps)",
          dev < 1e-11 && db < 1e-11, fmt("max |W/W0 - 1| %.1e", dev) + fmt(", div B change %.1e", db));
}

// I4. Plasma beyond the Courant limit of RIP-CC (h_x = 0.5, h_perp = 0.2), both shapes, neutral with a
//     drift across the periodic boundary and a non-neutral slab: Gauss law and div B to round-off.
static void irip_plasma(Comm& comm) {
    for (int o5 : {0, 1})                                    // also with the O5 closure (decisions 72, 73)
    for (int order : {1, 2})
        for (int neutral : {1, 0}) {
            Config c;
            c.o5 = o5 == 1;
            c.N[0] = 12; c.N[1] = 10; c.N[2] = 8;
            c.h[0] = 0.5; c.h[1] = 0.2; c.h[2] = 0.25;
            c.solver = Solver::IRip;
            c.t_end = 40 * c.dt();
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
                c.species[0].rmin[0] = 1.0; c.species[0].rmax[0] = 3.0;
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
            const std::string tag = std::string(o5 ? "irip o5: plasma shape " : "irip: plasma shape ") + std::to_string(order) +
                                    (neutral ? ", neutral, drift -3" : ", electron slab");
            check(tag + ": Gauss law and div B to round-off", g < 1e-12 && db < 1e-12,
                  fmt("max|dGauss| %.1e", g) + fmt(", max|divB| %.1e", db));
        }
}

// I5. Langmuir oscillation of cold electrons at omega_p dt = 1.9, h_perp = h_x / 4: stable, and
//     sin(w dt/2) = wp dt/2 as for RIP-CC (the mode is uniform in y, z: the transverse solve is the identity).
static void irip_langmuir(Comm& comm) {
    Config c;
    c.N[0] = 32; c.N[1] = 6; c.N[2] = 6;      // 6 guard cells for moves of dt / h_perp = 4 cells
    c.h[0] = 0.4; c.h[1] = 0.1; c.h[2] = 0.1;
    c.solver = Solver::IRip;
    const double wp = 1.9 / c.h[0];
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
    double emax = 0;
    while (!sim.finished()) {
        sim.step();
        ex.push_back(H(sim.fields->ex)(5, 1, 1));
        emax = std::max(emax, std::abs(ex.back()));
    }
    // 2.5 steps per period: the leapfrog recurrence e^{n+1} + e^{n-1} = 2 cos(w dt) e^n, least squares,
    // against cos(w dt) = 1 - wp^2 dt^2 / 2 (to 1 %: at k = 2 pi / L the shape factor lowers wp^2 by ~0.5 %)
    double num = 0, den = 0;
    for (size_t n = 1; n + 1 < ex.size(); ++n) { num += (ex[n + 1] + ex[n - 1]) * ex[n]; den += 2 * ex[n] * ex[n]; }
    const double dt = c.dt(), cw = num / den, cth = 1 - 0.5 * wp * wp * dt * dt;
    check("irip: Langmuir oscillation at omega_p dt = 1.9, h_perp = h_x / 4", std::abs(cw - cth) < 0.01 * std::abs(1 - cth) && emax < 1e-1,
          fmt("cos(w dt) = %.5f, ", cw) + fmt("expected %.5f", cth) + fmt(", max |Ex| %.1e", emax));
}

// I6b. Particles crossing many cells per step: a relativistically hot pair plasma (u_th = 5) with
//      h_x / h_perp = 8, so particles move up to 8 transverse cells per step (the Esirkepov split into
//      one-cell segments, guards of 10 cells). Gauss law and div B to round-off, particles kept, and the
//      drift of field + kinetic energy (0.9 % in 30 steps, the push of this hot plasma) the same as on
//      a grid with h_perp = h_x, where no particle moves more than one cell: the reference.
static void irip_fast_particles(Comm& comm) {
    for (int order : {1, 2}) {
        double dev[2] = {0, 0}, g = 0, db = 0;
        bool kept = true, guards = true;
        for (int fine = 1; fine >= 0; --fine) {
            Config c;
            c.N[0] = 12; c.N[1] = 12; c.N[2] = 12;
            c.h[0] = 0.8; c.h[1] = c.h[2] = fine ? 0.1 : 0.8;
            c.solver = Solver::IRip;
            c.t_end = 30 * c.dt();
            c.order = order;
            SpeciesConfig e;
            e.name = "e"; e.q = -1; e.m = 1; e.density = 0.5;
            e.ppc[0] = e.ppc[1] = e.ppc[2] = 2;
            for (int a = 0; a < 3; ++a) { e.uth[a] = 5.0; e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
            e.u0[1] = 3.0;
            SpeciesConfig p = e;
            p.name = "p"; p.q = 1; p.u0[1] = -3.0;
            c.species = {e, p};
            c.check();
            Simulation sim(c, comm);
            sim.init();
            auto total = [&]() {
                double w = sim.fields->energy();
                for (const Species& s : sim.species) w += s.kinetic_energy();
                comm.allreduce_sum(&w, 1);
                return w;
            };
            const double w0 = total();
            long np0 = 0, np1 = 0;
            for (const Species& s : sim.species) np0 += long(s.size());
            while (!sim.finished()) {
                sim.step();
                if (fine) {
                    g = std::max(g, sim.gauss_drift());
                    db = std::max(db, comm.allreduce_max(sim.fields->max_div_b()));
                }
                dev[fine] = std::max(dev[fine], std::abs(total() / w0 - 1));
            }
            for (const Species& s : sim.species) np1 += long(s.size());
            kept = kept && np0 == np1;
            if (fine) guards = sim.dom.ng == 10;
        }
        check(std::string("irip: hot pair plasma crossing 8 cells per step, shape ") + std::to_string(order) +
                  ": Gauss law, div B, energy as with h_perp = h_x",
              g < 1e-12 && db < 1e-12 && kept && guards && std::abs(dev[1] - dev[0]) < 0.1 * dev[0],
              fmt("max|dGauss| %.1e", g) + fmt(", max|divB| %.1e", db) + fmt(", max |W/W0 - 1| %.2e", dev[1]) +
                  fmt(" (h_perp = h_x: %.2e)", dev[0]));
    }
}

// I7. Open x: (a) a wave along x (E_y = B_z = f(x)) leaves exactly, nothing is left after it has passed
//     the right edge; (b) random fields at h_x / h_perp = 10 never gain energy (the open edges only take
//     energy out). Most of it stays: random fields at k_perp dx >> 1 are nearly frozen (w dt -> pi) and
//     have static parts.
static void irip_open(Comm& comm) {
    Domain d = irip_box(comm, 24, 8, 6, 0.5, 0.05, 0.05, true);
    {
        Fields f(d);
        H ey(f.ey), bz(f.bz);
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) ey(i, j, k) = bz(i, j, k) = std::exp(-0.2 * (i - 12.0) * (i - 12.0));
        ey.put(f.ey); bz.put(f.bz);
        comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
        const double w0 = f.energy();
        for (int n = 0; n < 24; ++n) { f.zero_currents(); f.step(comm); }
        check("irip: open x, a wave along x leaves exactly", f.energy() / w0 < 1e-28, fmt("W/W0 left %.1e", f.energy() / w0));
    }
    Fields f(d);
    std::mt19937_64 rng(11);
    std::normal_distribution<double> g;
    for (Field* F : {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz}) {
        H h(*F);
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) h(i, j, k) = g(rng);
        h.put(*F);
    }
    comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
    const double w0 = f.energy();
    double w = w0, grow = 0;
    for (int n = 0; n < 400; ++n) {
        f.zero_currents(); f.step(comm);
        const double wn = f.energy();
        grow = std::max(grow, (wn - w) / w0);
        w = wn;
    }
    check("irip: open x, random fields at dx/h_perp = 10 never gain energy (400 steps)", grow < 1e-13 && w < 0.9 * w0,
          fmt("max step gain %.1e W0", grow) + fmt(", W/W0 at the end %.3f", w / w0));
}

// I8. The laser through the open x edge with h_y = 3 dx / 8 (beyond RIP-CC's limit): the focus test of
//     tests/test_laser.cpp (L1) with the same paraxial reference: widths at the entrance and the focus
//     and the fluence gain.
static void irip_laser_focus(Comm& comm) {
    Config c;
    const double dx = 2 * M_PI / 16;
    c.N[0] = 56; c.N[1] = 256; c.N[2] = 6;      // 6 >= the guard width 2 + ceil(dx / h_y) = 5
    c.h[0] = dx; c.h[1] = 0.375 * dx; c.h[2] = 10 * dx;
    c.periodic[0] = false;
    c.solver = Solver::IRip;
    LaserConfig& L = c.laser;
    L.enabled = true; L.a0 = 0.01; L.tau = 12; L.tc = 24; L.w0 = 6; L.w0z = 0; L.pol = 'z';
    L.yc = 0.5 * c.N[1] * c.h[1]; L.zc = 0.5 * c.N[2] * c.h[2];
    const double zr = 0.5 * L.w0 * L.w0;
    L.xf = zr;
    c.t_end = L.tc + 3 * L.tau + c.N[0] * dx;
    c.check();
    Simulation s(c, comm);
    s.init();
    const int i0 = 0, i1 = int(std::lround(L.xf / dx));
    std::vector<double> F0(size_t(c.N[1]), 0.0), F1(size_t(c.N[1]), 0.0);
    while (!s.finished()) {
        s.step();
        H ez(s.fields->ez);
        for (int j = 0; j < c.N[1]; ++j) {
            F0[size_t(j)] += ez(i0, j, 1) * ez(i0, j, 1);
            F1[size_t(j)] += ez(i1, j, 1) * ez(i1, j, 1);
        }
    }
    auto width = [&](const std::vector<double>& F, double& peak) {
        double m0 = 0, m2 = 0;
        peak = 0;
        for (int j = 0; j < c.N[1]; ++j) {
            const double y = j * c.h[1] - L.yc;
            m0 += F[size_t(j)]; m2 += F[size_t(j)] * y * y; peak = std::max(peak, F[size_t(j)]);
        }
        return 2 * std::sqrt(m2 / m0);
    };
    double p0, p1;
    const double w0m = width(F0, p0), w1m = width(F1, p1);
    const double w0a = L.w0 * std::sqrt(1 + L.xf * L.xf / (zr * zr)), w1a = L.w0 * std::sqrt(1 + std::pow((i1 * dx - L.xf) / zr, 2));
    const double e0 = std::abs(w0m / w0a - 1), e1 = std::abs(w1m / w1a - 1), er = std::abs(p1 / p0 / (w0a / w1a) - 1);
    check("irip: laser focus with h_y = 3 dx/8, widths and fluence gain against paraxial theory", e0 < 0.03 && e1 < 0.03 && er < 0.05,
          fmt("w(0) %.3f", w0m) + fmt(" (paraxial %.3f)", w0a) + fmt(", w(x_f) %.3f", w1m) + fmt(" (%.3f)", w1a) +
              fmt(", F(x_f)/F(0) %.3f", p1 / p0) + fmt(" (%.3f)", w0a / w1a));
}

// I9. CPML layers (decision 41). A Gaussian packet in the x-y plane (E = (Ex, Ey), Bz, uniform in z,
//     wavelength 2 pi = 16 dx, width w = 8 dx) runs at phi to x into a 16-cell y layer with h_y = dx/2,
//     centred 6 w from each layer (no initial overlap); the interior against the same packet in a tall
//     periodic box (512 cells, nothing comes back in time).
static Config irip_pml_box(int ny, bool pml, bool window = false, bool o5 = false) {
    Config c;
    const double dx = 2 * M_PI / 16;
    c.N[0] = window ? 128 : 16; c.N[1] = ny; c.N[2] = 4;      // x period 16 dx = 2 w, the same in both runs
    c.h[0] = dx; c.h[1] = 0.5 * dx; c.h[2] = 10 * dx;
    if (window) { c.periodic[0] = false; c.window.enabled = true; c.window.start = 0; }
    c.pml.on[1] = pml; c.periodic[1] = !pml; c.pml.cells = 16;
    c.solver = Solver::IRip;
    c.o5 = o5;
    c.t_end = 1e9;
    c.check();
    return c;
}
static void irip_packet(Simulation& s, double phi, double w, double c) {
    Fields& f = *s.fields;
    const Domain& d = s.dom;
    H ex(f.ex), ey(f.ey), bz(f.bz);
    const double kx = std::cos(phi), kt = std::sin(phi);
    auto g = [&](int i, double t) {
        const double X = i * d.h[0] - 0.5 * d.L(0), T = (t - c) * d.h[1];
        return std::exp(-(X * X + T * T) / (w * w)) * std::cos(kx * X + kt * T);
    };
    for (int k = 0; k < d.n[2]; ++k)
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                ex(i, j, k) = -kt * g(i, j);
                ey(i, j, k) = kx * g(i, j + 0.5);
                bz(i, j, k) = g(i, j + 0.5);
            }
    ex.put(f.ex); ey.put(f.ey); bz.put(f.bz);
    s.comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
}
static double irip_reflection(Comm& comm, double phi_deg, bool o5 = false) {
    const int nin = 192, L = 16, nr = 512, Lr = (nr - nin) / 2;
    const double phi = phi_deg * M_PI / 180, w = M_PI;
    Simulation a(irip_pml_box(nin + 2 * L, true, false, o5), comm), r(irip_pml_box(nr, false, false, o5), comm);
    a.init(); r.init();
    irip_packet(a, phi, w, L + 0.5 * nin);
    irip_packet(r, phi, w, Lr + 0.5 * nin);
    const double dx = a.dom.h[0], hy = a.dom.h[1];
    const int steps = int(1.1 * (0.5 * nin * hy + 4 * L * hy + 3 * w) / (dx * std::max(std::sin(phi), 0.25)));
    double amp = 0, err = 0;
    for (double v : a.fields->bz.host()) amp = std::max(amp, std::abs(v));
    for (int n = 0; n < steps; ++n) {
        a.step(); r.step();
        if (n % 10) continue;
        const Field* fa[6] = {&a.fields->ex, &a.fields->ey, &a.fields->ez, &a.fields->bx, &a.fields->by, &a.fields->bz};
        const Field* fr[6] = {&r.fields->ex, &r.fields->ey, &r.fields->ez, &r.fields->bx, &r.fields->by, &r.fields->bz};
        for (int q = 0; q < 6; ++q) {
            H ha(*fa[q]), hr(*fr[q]);
            for (int k = 0; k < a.dom.n[2]; ++k)
                for (int j = L; j < L + nin; ++j)
                    for (int i = 0; i < a.dom.n[0]; ++i) err = std::max(err, std::abs(ha(i, j, k) - hr(i, j + Lr - L, k)));
        }
    }
    return err / amp;
}
static void irip_pml_reflection(Comm& comm) {
    const double r90 = irip_reflection(comm, 90), r45 = irip_reflection(comm, 45);
    check("irip: packet into a y layer (16 cells, h_y = dx/2), reflection", r90 < 2e-5 && r45 < 1e-4,
          fmt("max interior error / amplitude: normal %.1e", r90) + fmt(", 45 deg %.1e", r45));
    // O5 (decision 75): the closure per mode of the stretched operator; the reference runs o5 in the tall periodic box
    const double o90 = irip_reflection(comm, 90, true), o45 = irip_reflection(comm, 45, true);
    // the closure acts on the instantaneous stretching b of the layer, not on the CPML memory: about 60 times the box closure's
    // reflection at 16 cells (1e-5 at 32 cells; decision 75: accepted, thicker layers recommended)
    check("irip, closure o5: packet into a y layer (16 cells, h_y = dx/2), reflection below 3e-4",
          o90 < 3e-4 && o45 < 3e-4, fmt("max interior error / amplitude: normal %.1e", o90) + fmt(", 45 deg %.1e", o45) +
          fmt(" (box closure %.1e", r90) + fmt(", %.1e)", r45));
}

// I10. Random fields with layers in y and z at dx/h_perp = 4, 300 steps: bounded and absorbed. The energy
//      is not monotone with CPML (psi stores energy; research/irip_3d/pml_identity.py), so the check is
//      a bound and the decay.
static void irip_pml_stability(Comm& comm) {
    for (int o5 = 0; o5 < 2; ++o5) {
    Config c;
    const double dx = 0.4;
    c.N[0] = 12; c.N[1] = 24; c.N[2] = 24;
    c.h[0] = dx; c.h[1] = 0.25 * dx; c.h[2] = 0.25 * dx;
    for (int a = 1; a < 3; ++a) { c.pml.on[a] = true; c.periodic[a] = false; }
    c.pml.cells = 6;
    c.solver = Solver::IRip;
    c.o5 = o5 == 1;
    c.t_end = 300 * dx;
    c.check();
    Simulation s(c, comm);
    s.init();
    std::mt19937_64 rng(9);
    std::normal_distribution<double> nd(0, 1);
    for (Field* f : {&s.fields->ex, &s.fields->ey, &s.fields->ez, &s.fields->bx, &s.fields->by, &s.fields->bz}) {
        H h(*f);
        for (int k = 0; k < s.dom.n[2]; ++k)
            for (int j = 0; j < s.dom.n[1]; ++j)
                for (int i = 0; i < s.dom.n[0]; ++i) h(i, j, k) = nd(rng);
        h.put(*f);
    }
    s.comm.exchange(s.dom, {&s.fields->ex, &s.fields->ey, &s.fields->ez, &s.fields->bx, &s.fields->by, &s.fields->bz});
    const double e0 = s.fields->energy();
    double emax = e0, e = e0, ehalf = e0;
    int n = 0;
    while (!s.finished()) { s.step(); e = s.fields->energy(); emax = std::max(emax, e); if (++n == 150) ehalf = e; }
    // o5: not energy conserving, a transient growth of the grid scale (<= 2.15 with the cap of decision 77, 7.2 without it at
    // h_perp = dx/4 in the periodic box); the
    // grid-scale noise that never reaches the layers stays (box closure: 0.70 of E_0 after 2000 steps, o5: 2.6), so o5 is
    // checked for a bound and no growth over the second half
    if (o5)
        check("irip, closure o5: random fields, layers in y and z, dx/h_perp = 4, 300 steps: bounded, not growing",
              emax < 8 * e0 && e < 1.02 * ehalf && std::isfinite(e),
              fmt("max E/E_0 %.3f", emax / e0) + fmt(", step 150 %.3f", ehalf / e0) + fmt(", end %.3f", e / e0));
    else
        check("irip: random fields, layers in y and z, dx/h_perp = 4, 300 steps: bounded and absorbed",
              emax < 1.5 * e0 && e < 0.9 * e0 && std::isfinite(e), fmt("max E/E_0 %.3f", emax / e0) + fmt(", end %.3f", e / e0));
    }
}

// I11. A hot plasma (u_th = 0.3, moves of 2.5 transverse cells) leaves through y and z layers: the Gauss
//      law and div B outside the layers at round-off, particles leave.
static void irip_pml_plasma(Comm& comm) {
    for (int o5 = 0; o5 < 2; ++o5) {
    Config c;
    c.N[0] = 12; c.N[1] = 32; c.N[2] = 32;
    c.h[0] = 0.5; c.h[1] = 0.2; c.h[2] = 0.2;
    for (int a = 1; a < 3; ++a) { c.pml.on[a] = true; c.periodic[a] = false; }
    c.pml.cells = 8;
    c.absorber_mode = 0;                           // keep: the test of the layers themselves (decision 111; P5 tests the removal)
    c.solver = Solver::IRip;
    c.o5 = o5 == 1;
    c.order = 2;
    c.t_end = 40 * c.h[0];
    for (int sp = 0; sp < 2; ++sp) {
        SpeciesConfig e;
        e.name = sp ? "p" : "e"; e.q = sp ? 1 : -1; e.m = sp ? 10 : 1; e.density = 0.5;
        e.ppc[0] = e.ppc[1] = e.ppc[2] = 1;
        for (int a = 0; a < 3; ++a) { e.uth[a] = sp ? 0.05 : 0.3; e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
        c.species.push_back(e);
    }
    c.check();
    Simulation sim(c, comm);
    sim.init();
    const long n0 = long(sim.species[0].size());
    double g = 0, db = 0;
    while (!sim.finished()) {
        sim.step();
        g = std::max(g, sim.gauss_drift());
        db = std::max(db, comm.allreduce_max(sim.fields->max_div_b()));
    }
    const double left = 1 - double(sim.species[0].size()) / double(n0);
    check(std::string(o5 ? "irip, closure o5" : "irip") + ": hot plasma through y and z layers: Gauss law and div B outside the layers, particles leave",
          g < 1e-12 && db < 1e-12 && left > 0.05, fmt("max|dGauss| %.1e", g) + fmt(", max|divB| %.1e", db) +
          fmt(", electrons gone %.2f", left));
    }
}

// I6. The scope: open x, pml and the beam self-field (RIP-CC's Gauss-law equilibrium, decision 98) are accepted; no
//     Courant limit.
static void irip_config() {
    Config c;
    c.N[0] = 8; c.N[1] = 24; c.N[2] = 24;
    c.h[0] = 1.0; c.h[1] = 0.1; c.h[2] = 0.1;
    c.solver = Solver::IRip;
    bool ok = true;
    try {
        c.check(); c.periodic[0] = false; c.check();
        c.pml.on[1] = true; c.periodic[1] = false; c.pml.cells = 4; c.check();
    } catch (const std::exception&) { ok = false; }
    SpeciesConfig b;
    b.name = "b"; b.q = -1; b.m = 1; b.density = 1e-3; b.self_field = true;
    c.species.push_back(b);
    bool accepted = true;
    try { c.check(); } catch (const std::exception&) { accepted = false; }
    check("irip: no Courant limit, open x, pml and self_field accepted", ok && accepted, "");
}

// I12. O5 in 3D (numerics.closure = o5, decisions 72, 73): the transverse factor per mode in the y, z Fourier basis.
//      (a) the vacuum dispersion of oblique (kx, ky, kz) modes, both polarisations, against tan^2(w dt/2) = tan^2(kx dx/2)
//          + dx^2 lam/4 (1 + 4/3 sin^2(kx dx/2)) g_perp, g_perp = 1 + (2/3) min(dx^2 lam/4, 1) + (hy^2 ly^2 + hz^2 lz^2)/(12 lam)
//          (the cap of decision 77 is active on the modes at h_perp = dx/4),
//          h_perp = dx/4 and 3.2 dx, also at 45 deg azimuth (ly = lz) where a local factor would overcorrect by 2;
//      (b) the corrector's Ex, Bx equal the solve's X, Y (periodic and open x); (c) random fields over 400 steps, h_perp = dx/10
//      .. 3.2 dx, stay below 2.15^2 = 4.62 E_0, the bound of the capped closure (research/o5/stab3d.py: ||G^n|| <= 2.15; without
//      the cap 7.2 at dx/4 and 17.6 at dx/10, decision 77).
static void irip_o5_3d(Comm& comm) {
    double worst = 0, wdt = 0, pred = 0;
    for (double ry : {0.25, 3.2})
        for (int c = 0; c < 3; ++c) {
            const double hx = 0.5, hp = ry * hx;
            Domain d = irip_box(comm, 10, 8, 8, hx, hp, hp);
            Fields f(d);
            f.o5 = true;
            static int mx, my, mz;
            mx = c == 1 ? 3 : 1; my = 1; mz = c == 0 ? 2 : 1;          // c = 1, 2: ky = kz, 45 deg azimuth
            auto init = [](const Domain& d, int i, int j, int k) {
                return std::cos(2 * M_PI * (double(mx) * i / d.N[0] + double(my) * j / d.N[1] + double(mz) * k / d.N[2]) + 0.3);
            };
            fill(c == 1 ? f.bx : f.ey, d, init);
            fill(c == 1 ? f.ez : f.ex, d, init);
            comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
            const double D = d.h[0], kx = 2 * M_PI * mx / (d.N[0] * D), ky = 2 * M_PI * my / (d.N[1] * hp),
                         kz = 2 * M_PI * mz / (d.N[2] * hp);
            const double sy = std::sin(ky * hp / 2), sz = std::sin(kz * hp / 2), sx = std::sin(kx * D / 2), tx = std::tan(kx * D / 2);
            const double ly = 4 * sy * sy / (hp * hp), lz = 4 * sz * sz / (hp * hp), lam = ly + lz;
            const double gp = 1 + 2.0 / 3 * std::min(D * D * lam / 4, 1.0) + hp * hp * (ly * ly + lz * lz) / (12 * lam);
            pred = 2 * std::atan(std::sqrt(tx * tx + D * D * lam / 4 * (1 + 4.0 / 3 * sx * sx) * gp));
            std::vector<std::vector<double>> hist;
            for (int n = 0; n < 40; ++n) {
                std::vector<double> v = f.ey.host(), b = f.bx.host(), e = f.ez.host();
                v.insert(v.end(), b.begin(), b.end());
                v.insert(v.end(), e.begin(), e.end());
                hist.push_back(v);
                f.zero_currents(); f.step(comm);
            }
            double num = 0, den = 0;
            for (size_t n = 1; n + 2 < hist.size(); ++n)
                for (size_t q = 0; q < hist[n].size(); ++q) {
                    const double dm = hist[n][q] - hist[n - 1][q], d0 = hist[n + 1][q] - hist[n][q], dp = hist[n + 2][q] - hist[n + 1][q];
                    num += (dp + dm) * 2 * d0; den += 4 * d0 * d0;
                }
            const double cw = num / den;
            double res = 0, scale = 0;
            for (size_t n = 1; n + 2 < hist.size(); ++n)
                for (size_t q = 0; q < hist[n].size(); ++q) {
                    const double dm = hist[n][q] - hist[n - 1][q], d0 = hist[n + 1][q] - hist[n][q], dp = hist[n + 2][q] - hist[n + 1][q];
                    res = std::max(res, std::abs(dp + dm - 2 * cw * d0)); scale = std::max(scale, std::abs(d0));
                }
            wdt = std::acos(cw);
            worst = std::max({worst, std::abs(wdt - pred), res / scale});
        }
    check("irip, closure o5, 3D: oblique modes (also 45 deg azimuth) = the corrected relation per mode (h_perp = dx/4, 3.2 dx)",
          worst < 1e-11, fmt("last w dt %.6f", wdt) + fmt(" vs %.6f, max error %.1e", pred, worst));
    double resid = 0, wmax = 0, wend = 0;
    for (int open = 0; open < 2; ++open)
        for (double ry : {0.1, 0.25, 1.0, 3.2}) {
            Domain d = irip_box(comm, 12, 8, 6, 0.5, ry * 0.5, ry * 0.4, open == 1);
            Fields f(d);
            f.o5 = true;
            std::mt19937_64 rng(91 + open + int(4 * ry));
            std::normal_distribution<double> nd;
            for (Field* F : {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz}) {
                H h(*F);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i) h(i, j, k) = (open && (i < 3 || i >= d.n[0] - 3)) ? 0.0 : nd(rng);
                h.put(*F);
            }
            comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
            const double w0 = f.energy();
            const int steps = open ? 5 : 400;
            for (int n = 0; n < steps; ++n) {
                f.zero_currents(); f.step(comm);
                resid = std::max(resid, irip_residual(f, d));
                if (!open) wmax = std::max(wmax, f.energy() / w0);
            }
            if (!open) wend = std::max(wend, f.energy() / w0);
        }
    check("irip, closure o5, 3D: corrector = solve (periodic and open x); random fields over 400 steps (h_perp = dx/10 .. 3.2 dx) "
          "below the bound 4.62 of the capped closure",
          resid < 1e-11 && wmax < 4.62 && wmax > 0.2, fmt("max|Ex - X| %.1e", resid) + fmt(", W / W0 max %.3f, end %.3f", wmax, wend));
}

void test_irip(Comm& comm) {
    if (comm.size > 1) return;          // one rank so far
    irip_x_wave(comm);
    irip_dispersion(comm);
    irip_o5_3d(comm);
    irip_vacuum_random(comm);
    irip_plasma(comm);
    irip_langmuir(comm);
    irip_fast_particles(comm);
    irip_open(comm);
    irip_laser_focus(comm);
    irip_pml_reflection(comm);
    irip_pml_stability(comm);
    irip_pml_plasma(comm);
    irip_config();
}
