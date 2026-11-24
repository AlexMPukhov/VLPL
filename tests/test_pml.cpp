// CPML layers in y and z (docs/specs/pml.md).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include "common.hpp"
#include "sim.hpp"

// A Gaussian wave packet in the x-y plane (axis = 1: E = (Ex, Ey), Bz; uniform in z) or in the x-z plane
// (axis = 2: E = (Ex, Ez), By; uniform in y), k = (cos phi, sin phi) along (x, axis), wavelength 2 pi,
// width w, centred at x = Lx / 2 and at c along the axis (in cells, from the box edge).
static void packet(Simulation& s, int axis, double phi, double w, double c, double xc = -1) {
    Fields& f = *s.fields;
    const Domain& d = s.dom;
    H ex(f.ex), ey(f.ey), ez(f.ez), by(f.by), bz(f.bz);
    const double kx = std::cos(phi), kt = std::sin(phi);
    auto g = [&](int i, double t) {                      // t: transverse coordinate in cells
        const double X = i * d.h[0] - (xc < 0 ? 0.5 * d.L(0) : xc), T = (t - c) * d.h[axis];
        return std::exp(-(X * X + T * T) / (w * w)) * std::cos(kx * X + kt * T);
    };
    for (int k = 0; k < d.n[2]; ++k)
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                const int jg = d.off[1] + j, kg = d.off[2] + k;
                if (axis == 1) {
                    ex(i, j, k) = -kt * g(i, jg);
                    ey(i, j, k) = kx * g(i, jg + 0.5);
                    bz(i, j, k) = g(i, jg + 0.5);
                } else {
                    ex(i, j, k) = -kt * g(i, kg);
                    ez(i, j, k) = kx * g(i, kg + 0.5);
                    by(i, j, k) = -g(i, kg + 0.5);
                }
            }
    ex.put(f.ex); ey.put(f.ey); ez.put(f.ez); by.put(f.by); bz.put(f.bz);
    s.comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
}

static Config vacuum(int axis, int nt, bool pml, int L, bool window = false) {
    Config c;
    const double dx = 2 * M_PI / 16;                     // lambda = 16 dx
    c.N[0] = window ? 256 : 64; c.h[0] = dx;         // 64: a and the reference share the x period
    if (window) { c.periodic[0] = false; c.window.enabled = true; c.window.start = 0; }
    const int other = 3 - axis;
    c.N[axis] = nt; c.h[axis] = 2 * dx;
    c.N[other] = 3; c.h[other] = 10 * dx;
    c.pml.on[axis] = pml; c.periodic[axis] = !pml;
    c.pml.cells = L;
    c.t_end = 1e9;
    c.check();
    return c;
}

// P1. A packet at angle phi (90 = normal) runs into the layer; the interior against a run in a box so
//     tall (periodic) that nothing comes back in time: max error / amplitude.
static double reflection(Comm& comm, int axis, double phi_deg, int L, bool window = false) {
    const int nin = 64, steps_max = 400;
    const double phi = phi_deg * M_PI / 180, w = 16 * 2 * M_PI / 16;
    const int steps = std::min(steps_max, int(1.1 * (0.5 * nin * 2 + 4 * L + 3 * 16) / std::max(std::sin(phi), 0.25)));
    const int Lr = steps / 4 + 8;    // reference margin (cells of 2 dx): a wave leaving it travels 2 Lr to come back
    Simulation a(vacuum(axis, nin + 2 * L, true, L, window), comm), r(vacuum(axis, nin + 2 * Lr, false, 0, window), comm);
    a.init(); r.init();
    packet(a, axis, phi, w, L + 0.5 * nin);
    packet(r, axis, phi, w, Lr + 0.5 * nin);
    double amp = 0, err = 0;
    for (double v : (axis == 1 ? a.fields->bz : a.fields->by).host()) amp = std::max(amp, std::abs(v));   // |B| = a
    for (int n = 0; n < steps; ++n) {
        a.step(); r.step();
        if (n % 20) continue;
        const Field* fa[6] = {&a.fields->ex, &a.fields->ey, &a.fields->ez, &a.fields->bx, &a.fields->by, &a.fields->bz};
        const Field* fr[6] = {&r.fields->ex, &r.fields->ey, &r.fields->ez, &r.fields->bx, &r.fields->by, &r.fields->bz};
        for (int q = 0; q < 6; ++q) {
            H ha(*fa[q]), hr(*fr[q]);
            for (int k = 0; k < a.dom.n[2]; ++k)
                for (int j = 0; j < a.dom.n[1]; ++j)
                    for (int i = 0; i < a.dom.n[0]; ++i) {
                        const int t = axis == 1 ? a.dom.off[1] + j : a.dom.off[2] + k;      // global, a's box
                        if (t < L || t >= L + nin) continue;
                        const int jr = axis == 1 ? j + Lr - L : j, kr = axis == 2 ? k + Lr - L : k;
                        err = std::max(err, std::abs(ha(i, j, k) - hr(i, jr, kr)));
                    }
        }
    }
    return err / amp;
}

static void pml_reflection(Comm& comm) {
    if (comm.size > 1) return;                           // the reference box has another decomposition
    const double r90 = reflection(comm, 1, 90, 16), r45 = reflection(comm, 1, 45, 16), z45 = reflection(comm, 2, 45, 16);
    check("pml: packet into a y layer (16 cells), reflection", r90 < 2e-5 && r45 < 1e-4,
          fmt("max interior error / amplitude: normal %.1e", r90) + fmt(", 45 deg %.1e", r45));
    check("pml: packet into a z layer (16 cells), reflection = y layer", z45 < 1e-4 && std::abs(z45 - r45) < 1e-3 * r45,
          fmt("45 deg %.1e", z45) + fmt(" (y: %.1e)", r45));
}

// P2. Random fields with layers in y and z: the energy never grows over 1000 steps.
static void pml_stability(Comm& comm) {
    Config c;
    const double dx = 2 * M_PI / 16;
    c.N[0] = 24; c.N[1] = 40; c.N[2] = 40;
    c.h[0] = dx; c.h[1] = 2 * dx; c.h[2] = 2 * dx;
    for (int a = 1; a < 3; ++a) { c.pml.on[a] = true; c.periodic[a] = false; }
    c.pml.cells = 12;
    c.t_end = 1000 * dx;
    c.check();
    Simulation s(c, comm);
    s.init();
    std::mt19937_64 rng(9 + comm.rank);
    std::normal_distribution<double> nd(0, 1);
    for (Field* f : {&s.fields->ex, &s.fields->ey, &s.fields->ez, &s.fields->bx, &s.fields->by, &s.fields->bz}) {
        H h(*f);
        for (int k = 0; k < s.dom.n[2]; ++k)
            for (int j = 0; j < s.dom.n[1]; ++j)
                for (int i = 0; i < s.dom.n[0]; ++i) h(i, j, k) = nd(rng);
        h.put(*f);
    }
    s.comm.exchange(s.dom, {&s.fields->ex, &s.fields->ey, &s.fields->ez, &s.fields->bx, &s.fields->by, &s.fields->bz});
    auto energy = [&]() { double e = s.fields->energy(); s.comm.allreduce_sum(&e, 1); return e; };
    const double e0 = energy();
    double rise = 0, last = e0;
    long n = 0;
    while (!s.finished()) {
        s.step();
        if (++n % 100 == 0) { const double e = energy(); rise = std::max(rise, e / last - 1); last = e; }
    }
    check("pml: random fields, layers in y and z, 1000 steps: energy never grows", rise <= 1e-12 && last < 0.7 * e0,
          fmt("largest rise per 100 steps %.1e", rise) + fmt(", E_end / E_0 %.2e", last / e0));
}

// P3. A hot plasma fills the box and leaves through the y and z edges behind the layers: the Gauss law
//     outside the layers stays at round-off, and particles leave.
static void pml_plasma(Comm& comm) {
    Config c;
    c.N[0] = 24; c.N[1] = 40; c.N[2] = 40;
    c.h[0] = 0.3; c.h[1] = 0.6; c.h[2] = 0.6;
    for (int a = 1; a < 3; ++a) { c.pml.on[a] = true; c.periodic[a] = false; }
    c.pml.cells = 8;
    c.order = 2;
    c.t_end = 60 * c.h[0];
    c.absorber_mode = 0;                           // keep: the test of the layers themselves (P5: the removal)
    for (int s = 0; s < 2; ++s) {
        SpeciesConfig e;
        e.name = s ? "p" : "e"; e.q = s ? 1 : -1; e.m = s ? 100 : 1; e.density = 0.05;
        e.ppc[0] = 2;
        for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; e.uth[a] = s ? 0.01 : 0.5; }
        c.species.push_back(e);
    }
    c.check();
    Simulation sim(c, comm);
    sim.init();
    double n0 = double(sim.species[0].size());
    comm.allreduce_sum(&n0, 1);
    double g = 0;
    while (!sim.finished()) { sim.step(); g = std::max(g, sim.gauss_drift()); }
    double n1 = double(sim.species[0].size());
    comm.allreduce_sum(&n1, 1);
    check("pml: hot plasma leaving through y and z, Gauss law outside the layers", g < 1e-12 && n1 < 0.95 * n0,
          fmt("max|dGauss| %.1e", g) + fmt(", electrons %.0f%% left", 100 * (1 - n1 / n0)));
}

// P4. Moving window with layers (v = c from t = 0) against the fixed box with layers, periodic in x and
//     wide enough not to wrap: the window shift is exact, so window cell i equals fixed cell i + wx,
//     inside the layers too, only if the psi arrays move with the fields. The packet starts at the upper
//     layer, so the layer works from the first step; the comparison stays 60 cells from the window edges,
//     which the edge differences (zero inflow; T- crosses the window at 2 cells per step) do not reach
//     in 60 steps.
static void pml_window(Comm& comm) {
    if (comm.size > 1) return;
    const int L = 16, nin = 80, steps = 60;
    Config cw = vacuum(1, nin + 2 * L, true, L, true), cf = vacuum(1, nin + 2 * L, true, L, false);
    cf.N[0] = 512;
    Simulation a(cw, comm), b(cf, comm);
    a.init(); b.init();
    const double w = 2 * M_PI, phi = 45 * M_PI / 180, yc = L + nin - 4;
    packet(a, 1, phi, w, yc);
    packet(b, 1, phi, w, yc, 0.5 * a.dom.L(0));       // the same lab position
    for (int n = 0; n < steps; ++n) { a.step(); b.step(); }
    double err = 0, amp = 0;
    const Field* fa[6] = {&a.fields->ex, &a.fields->ey, &a.fields->ez, &a.fields->bx, &a.fields->by, &a.fields->bz};
    const Field* fb[6] = {&b.fields->ex, &b.fields->ey, &b.fields->ez, &b.fields->bx, &b.fields->by, &b.fields->bz};
    const int nx = a.dom.n[0];
    for (int q = 0; q < 6; ++q) {
        H ha(*fa[q]), hb(*fb[q]);
        for (int k = 0; k < a.dom.n[2]; ++k)
            for (int j = 0; j < a.dom.n[1]; ++j)
                for (int i = 60; i < nx - 2 * steps - 8; ++i) {
                    const int ib = int(i + a.dom.wx);
                    err = std::max(err, std::abs(ha(i, j, k) - hb(ib, j, k)));
                    amp = std::max(amp, std::abs(hb(ib, j, k)));
                }
    }
    check("pml: moving window = fixed box shifted, layers included (psi moves with the fields)",
          a.dom.wx == steps && amp > 0.1 && err < 1e-14,
          fmt("%.0f shifts, max difference %.1e", double(a.dom.wx), err) + fmt(" (field up to %.2g)", amp));
}

// P5. grid.absorber_particles (decision 111; absorber_particles.cpp): the hot plasma of P3 leaves through the layers, 60 steps.
//     remove: after every step no particle is in a layer (y/h < L or > N - L, also z) and no current is inside the layers;
//       the charge balance per species closes, Q(t) + removed = Q(0) (x periodic: nothing else leaves); the Gauss residual
//       (div E - rho - its initial value) is printed next to the faces (<= 3 cells) and further inside: the interior stays at
//       round-off, at the faces the charge that vanished there shows (measured, not bounded: Alexander decides).
//     conserve (default, decision 115): none deeper than the removal planes, Gauss law outside the layers at round-off, the charge balance closes.
//     keep: particles in the layers, none counted, Gauss law outside the layers at round-off.
static void pml_remove(Comm& comm) {
    const char* names[3] = {"keep", "remove", "conserve"};
    for (int mode : {1, 2, 0}) {
        Config c;
        c.N[0] = 24; c.N[1] = 40; c.N[2] = 40;
        c.h[0] = 0.3; c.h[1] = 0.6; c.h[2] = 0.6;
        for (int a = 1; a < 3; ++a) { c.pml.on[a] = true; c.periodic[a] = false; }
        c.pml.cells = 8;
        c.order = 2;
        c.t_end = 60 * c.h[0];
        c.absorber_mode = mode;
        for (int s = 0; s < 2; ++s) {
            SpeciesConfig e;
            e.name = s ? "p" : "e"; e.q = s ? 1 : -1; e.m = s ? 100 : 1; e.density = 0.05;
            e.ppc[0] = 2;
            for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; e.uth[a] = s ? 0.01 : 0.5; }
            c.species.push_back(e);
        }
        c.check();
        Simulation sim(c, comm);
        sim.init();
        const int L = c.pml.cells, N1 = c.N[1], N2 = c.N[2];
        const double mg = 0.5 * (c.order - 1);
        const double lo = mode == 1 ? double(L) : L - 2 - mg, hi1 = mode == 1 ? double(N1 - L) : N1 - L + 1 + mg,
                     hi2 = mode == 1 ? double(N2 - L) : N2 - L + 1 + mg;
        auto charge = [&](size_t i) {
            double q = 0;
            for (double w : sim.species[i].get(Species::W)) q += sim.species[i].q * w;
            comm.allreduce_sum(&q, 1);
            return q;
        };
        const double q0[2] = {charge(0), charge(1)};
        double gin = 0, gface = 0, deep = 0, jlayer = 0, jmax = 0;
        Field res(sim.dom);
        while (!sim.finished()) {
            sim.step();
            for (auto& sp : sim.species) {
                const std::vector<double> y = sp.get(Species::Y), z = sp.get(Species::Z);
                for (size_t p = 0; p < y.size(); ++p) {
                    const double a = y[p] / c.h[1], b = z[p] / c.h[2];
                    if (a < lo || a > hi1 || b < lo || b > hi2) deep += 1;
                }
            }
            sim.compute_rho();
            sim.fields->gauss_residual(res);
            H R(res), G0(sim.gauss0()), Jx(sim.fields->jx), Jy(sim.fields->jy), Jz(sim.fields->jz);
            const Domain& d = sim.dom;
            for (int k = 0; k < d.n[2]; ++k)
                for (int j = 0; j < d.n[1]; ++j)
                    for (int i = 0; i < d.n[0]; ++i) {
                        const int gy = d.off[1] + j, gz = d.off[2] + k;
                        const double jm = std::max({std::abs(Jx(i, j, k)), std::abs(Jy(i, j, k)), std::abs(Jz(i, j, k))});
                        jmax = std::max(jmax, jm);
                        // a layer point of any component (J_x, J_z at nodes in y, J_y at half nodes; likewise in z)
                        const bool inl = gy < L || gy >= N1 - L || gz < L || gz >= N2 - L;
                        if (inl) {
                            const bool ny = gy < L || gy > N1 - L, hy = gy < L || gy >= N1 - L, nz = gz < L || gz > N2 - L, hz = gz < L || gz >= N2 - L;
                            double v = 0;
                            if (ny || nz) v = std::max(v, std::abs(Jx(i, j, k)));
                            if (hy || nz) v = std::max(v, std::abs(Jy(i, j, k)));
                            if (ny || hz) v = std::max(v, std::abs(Jz(i, j, k)));
                            jlayer = std::max(jlayer, v);
                        }
                        // the Gauss points checked outside the layers (pml.hpp check_lo / check_hi), by the distance to the nearest face
                        if (gy < L + 1 || gy >= N1 - L - 1 || gz < L + 1 || gz >= N2 - L - 1) continue;
                        const int dist = std::min({gy - L, N1 - L - gy, gz - L, N2 - L - gz});
                        const double e = std::abs(R(i, j, k) - G0(i, j, k));
                        if (dist <= 3) gface = std::max(gface, e); else gin = std::max(gin, e);
                    }
        }
        comm.allreduce_sum(&deep, 1);
        gin = comm.allreduce_max(gin); gface = comm.allreduce_max(gface); jlayer = comm.allreduce_max(jlayer); jmax = comm.allreduce_max(jmax);
        double bal = 0, nrem = 0;
        for (size_t i = 0; i < 2; ++i) {
            double v[2] = {sim.species[i].lost_count, sim.species[i].lost_charge};
            comm.allreduce_sum(v, 2);
            nrem += v[0];
            bal = std::max(bal, std::abs(charge(i) + v[1] - q0[i]) / std::abs(q0[i]));
        }
        std::printf("  P5 absorber_particles = %s: removed %.0f, particle-steps beyond the removal planes %.0f, |J| in the layers %.1e (max |J| "
                    "%.1e), Gauss residual: interior (> 3 cells from a face) %.1e, next to the faces %.1e, charge balance %.1e\n",
                    names[mode], nrem, deep, jlayer, jmax, gin, gface, bal);
        if (mode == 1)
            check("pml: absorber_particles = remove: none in the layers, no current there, charge balance, Gauss law in the interior",
                  deep == 0 && nrem > 0 && jlayer == 0 && bal < 1e-10 && gin < 1e-12,
                  fmt("Gauss interior %.1e", gin) + fmt(", at the faces %.1e (measured)", gface));
        else if (mode == 2)
            check("pml: absorber_particles = conserve: none deeper than the removal planes, Gauss law outside the layers, charge balance",
                  deep == 0 && nrem > 0 && bal < 1e-10 && gin < 1e-12 && gface < 1e-12, fmt("Gauss %.1e", std::max(gin, gface)));
        else
            check("pml: absorber_particles = keep: particles in the layers, none counted, Gauss law outside the layers",
                  deep > 0 && nrem == 0 && gin < 1e-12 && gface < 1e-12, fmt("Gauss %.1e", std::max(gin, gface)));
    }
}

void test_pml(Comm& comm) {
    check("pml: grid.absorber_particles defaults to conserve (decision 115)", Config{}.absorber_mode == ABSORBER_CONSERVE,
          fmt("absorber_mode %.0f",double(Config{}.absorber_mode)));
    pml_reflection(comm);
    pml_window(comm);
    pml_stability(comm);
    pml_plasma(comm);
    pml_remove(comm);
}
