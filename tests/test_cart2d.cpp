// 2D Cartesian geometry (docs/specs/cart2d.md, decision 66): the Cart3D code on one z cell with the z stride 0. Continuity of
// the 2D deposit, the Gauss law and div B with plasma (rip_cc, irip), the vacuum dispersion of both solvers, the Langmuir
// frequency, and the equivalence with a Cart3D run of 3 periodic z cells on z-uniform data.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include "common.hpp"
#include "deck.hpp"
#include "particles.hpp"
#include "sim.hpp"
#include "stagger.hpp"
#include "tfsf.hpp"

static Domain flat_box(Comm& comm, int nx, int ny, double hx, double hy, double hz, Solver sv = Solver::RipCC) {
    Domain d;
    d.N[0] = nx; d.N[1] = ny; d.N[2] = 1;
    d.h[0] = hx; d.h[1] = hy; d.h[2] = hz;
    d.flat = true;
    d.ng = 3;
    d.solver = sv;
    const int want[3] = {1, 1, 1};
    comm.decompose(d, want);
    return d;
}

static SpeciesConfig species(const char* name, double q, double m, double n) {
    SpeciesConfig s;
    s.name = name; s.q = q; s.m = m; s.density = n;
    for (int a = 0; a < 3; ++a) { s.rmin[a] = -1e300; s.rmax[a] = 1e300; }
    return s;
}

// C1. The discrete continuity equation on the plane for random moves (also over several cells, split), and J_z: the
//     total over the plane times h_x h_y h_z is sum q w v_z (the z factor of the 3D kernel summed over z).
static void cart2d_continuity(Comm& comm) {
    for (int split : {0, 1})
        for (int order : {1, 2, 3}) {
            Domain d = split ? flat_box(comm, 12, 48, 0.3, 0.05, 0.4) : flat_box(comm, 12, 10, 0.3, 0.5, 0.4);
            Fields f(d);
            Field r0(d), r1(d);
            Species sp;
            sp.q = -1; sp.m = 1;
            std::mt19937_64 rng(31 + order + 7 * split);
            std::uniform_real_distribution<double> u01(0, 1), uu(-3, 3);
            std::vector<double> h[Species::NC];
            for (int p = 0; p < 500; ++p) {
                const double y = split ? (16 + 16 * u01(rng)) * d.h[1] : (3 + 4 * u01(rng)) * d.h[1];
                const double v[Species::NC] = {(4 + 4 * u01(rng)) * d.h[0], y, u01(rng) * d.h[2], uu(rng), uu(rng), uu(rng), u01(rng)};
                for (int c = 0; c < Species::NC; ++c) h[c].push_back(v[c]);
            }
            sp.append(h);
            const std::vector<double> z0 = sp.get(Species::Z), y0 = sp.get(Species::Y);
            deposit_rho(sp, r0, d, order);
            f.zero_currents();
            push_and_deposit(sp, f, d, Pusher::Boris, order);          // E = B = 0: straight moves
            deposit_rho(sp, r1, d, order);
            const std::vector<double> z1 = sp.get(Species::Z), y1 = sp.get(Species::Y), w = sp.get(Species::W);
            H R0(r0), R1(r1), Jx(f.jx), Jy(f.jy), Jz(f.jz);
            constexpr int cx = current_shift<RipCC>(0), cy = current_shift<RipCC>(1);
            double err = 0, scale = 0, jz = 0, jzw = 0, cells = 0;
            for (int j = -3; j < d.n[1] + 3; ++j)
                for (int i = -3; i < d.n[0] + 3; ++i) {
                    jz += Jz(i, j, 0) * d.h[0] * d.h[1] * d.h[2];
                    if (i < 0 || i >= d.n[0] || j < 0 || j >= d.n[1]) continue;
                    const double dr = (R1(i, j, 0) - R0(i, j, 0)) / d.dt();
                    const double div = (Jx(i + cx, j, 0) - Jx(i + cx - 1, j, 0)) / d.h[0] + (Jy(i, j + cy, 0) - Jy(i, j + cy - 1, 0)) / d.h[1];
                    err = std::max(err, std::abs(dr + div));
                    scale = std::max(scale, std::abs(dr));
                }
            double jsc = 0;
            for (size_t p = 0; p < w.size(); ++p) {
                jzw += sp.q * w[p] * (z1[p] - z0[p]) / d.dt();
                jsc += std::abs(w[p] * (z1[p] - z0[p]) / d.dt());
                cells = std::max(cells, std::abs(y1[p] - y0[p]) / d.h[1]);
            }
            const std::string tag = std::string("cart2d continuity") + (split ? " (moves over several cells)" : "") + ", shape " +
                                    std::to_string(order);
            check(tag, err < 1e-13 * scale && (!split || cells > 4), fmt("max err %.1e (|drho/dt| ~ %.2g)", err, scale) +
                  fmt(", moves up to %.1f cells", cells));
            check(tag + ": sum J_z dV = sum q w v_z", std::abs(jz - jzw) < 1e-14 * jsc, fmt("%.1e relative", std::abs(jz - jzw) / jsc));
        }
}

// C2. Gauss law and div B at round-off with a hot, drifting plasma (neutral and an electron slab), rip_cc and irip.
static void cart2d_plasma(Comm& comm) {
    for (int sc = 0; sc < 3; ++sc)
        for (int order : {1, 2, 3})
            for (int neutral : {1, 0}) {
                const Solver sv = sc == 0 ? Solver::RipCC : Solver::IRip;
                Config c;
                c.flat = true; c.solver = sv;
                c.o5 = sc == 2;                                      // iRIP with the O5 closure (decision 72)
                c.N[0] = 12; c.N[1] = 10; c.N[2] = 1;
                c.h[0] = 0.3; c.h[1] = sv == Solver::IRip ? 0.1 : 0.5; c.h[2] = 0.6;
                c.t_end = 60 * c.h[0];
                c.order = order;
                SpeciesConfig e = species("e", -1, 1, 0.5);
                e.ppc[0] = e.ppc[1] = 2;
                for (int a = 0; a < 3; ++a) e.uth[a] = 0.3;
                if (neutral) {
                    e.u0[0] = -3.0; e.u0[2] = 1.0;
                    SpeciesConfig p = e;
                    p.name = "p"; p.q = 1; p.m = 10; p.u0[0] = p.u0[2] = 0; p.uth[0] = p.uth[1] = p.uth[2] = 0.05;
                    c.species = {e, p};
                } else {
                    e.rmin[0] = 0.5; e.rmax[0] = 2.0;
                    c.species = {e};
                }
                c.check();
                Simulation sim(c, comm);
                sim.init();
                double g = 0, db = 0;
                while (!sim.finished()) {
                    sim.step();
                    g = std::max(g, sim.gauss_drift());
                    db = std::max(db, sim.fields->max_div_b());
                }
                const std::string tag = std::string("cart2d ") + (sc == 2 ? "irip o5" : sv == Solver::IRip ? "irip" : "rip_cc") + " plasma shape " +
                                        std::to_string(order) + (neutral ? ", neutral, drift" : ", electron slab");
                check(tag + ": Gauss law and div B", g < 1e-12 && db < 1e-12 && sim.fields->energy() > 0,
                      fmt("max|dGauss| %.1e, max|divB| %.1e", g, db));
            }
}

// C3. Vacuum dispersion of a Fourier mode (kx, ky) on the plane: RIP-CC sin^2(w dt/2) = sin^2(kx dx/2) + dx^2 Lam/4
//     cos^2(kx dx/2), iRIP tan^2(w dt/2) = tan^2(kx dx/2) + dx^2 Lam/4, Lam = 4/dy^2 sin^2(ky dy/2) (no kz). A single
//     harmonic has the static part and +-w only: the increments obey d^{n+1} + d^{n-1} = 2 cos(w dt) d^n. ky = 0: one cell
//     per step (w dt = kx dx).
static void cart2d_dispersion(Comm& comm) {
    for (Solver sv : {Solver::RipCC, Solver::IRip}) {
        const bool ir = sv == Solver::IRip;
        Domain d = flat_box(comm, 10, ir ? 8 : 6, 0.5, ir ? 0.125 : 0.6, 0.4, sv);
        double worst = 0, wdt = 0, pred = 0;
        for (int c = 0; c < 3; ++c) {
            Fields f(d);
            const int mx = c == 1 ? 3 : 1, my = c == 2 ? 0 : 1;
            const double kx = 2 * M_PI * mx / (d.N[0] * d.h[0]), ky = 2 * M_PI * my / (d.N[1] * d.h[1]);
            for (Field* F : c == 1 ? std::vector<Field*>{&f.bx, &f.ez} : std::vector<Field*>{&f.ey, &f.ex, &f.by}) {
                H h(*F);
                for (int j = 0; j < d.n[1]; ++j)
                    for (int i = 0; i < d.n[0]; ++i) h(i, j, 0) = std::cos(kx * i * d.h[0] + ky * j * d.h[1] + 0.3);
                h.put(*F);
            }
            comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
            const double D = d.h[0], sy = std::sin(ky * d.h[1] / 2), lam = 4 * sy * sy / (d.h[1] * d.h[1]);
            if (ir) { const double tx = std::tan(kx * D / 2); pred = 2 * std::atan(std::sqrt(tx * tx + D * D * lam / 4)); }
            else {
                const double sx = std::sin(kx * D / 2), cx = std::cos(kx * D / 2);
                pred = 2 * std::asin(std::sqrt(sx * sx + D * D * lam / 4 * cx * cx));
            }
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
        check(std::string("cart2d ") + (ir ? "irip" : "rip_cc") + ": vacuum dispersion of (kx, ky) modes, along x one cell per step",
              worst < 1e-11, fmt("last w dt %.6f", wdt) + fmt(" vs %.6f, max error %.1e", pred, worst));
    }
}

// C4. Langmuir oscillation of cold electrons (immobile ions): sin(w dt/2) = wp dt/2.
static void cart2d_langmuir(Comm& comm) {
    for (int order : {1, 3}) {
        Config c;
        c.flat = true;
        c.N[0] = 32; c.N[1] = order == 3 ? 6 : 4; c.N[2] = 1;
        c.order = order;
        c.h[0] = 0.2; c.h[1] = 0.5; c.h[2] = 0.5;
        const double wp = 2.0;
        c.t_end = 6 * 2 * M_PI / wp;
        SpeciesConfig e = species("e", -1, 1, 4.0);
        e.ppc[0] = 4;
        SpeciesConfig ion = e; ion.name = "i"; ion.q = 1; ion.m = 1e12;
        c.species = {e, ion};
        c.check();
        Simulation sim(c, comm);
        sim.init();
        const double k = 2 * M_PI / (c.N[0] * c.h[0]);
        auto& el = sim.species[0];
        {
            const std::vector<double> x = el.get(Species::X);
            std::vector<double> ux(x.size());
            for (size_t p = 0; p < x.size(); ++p) ux[p] = 1e-3 * std::sin(k * x[p]);
            el.set(Species::UX, ux);
        }
        std::vector<double> ex;
        while (!sim.finished()) { sim.step(); ex.push_back(H(sim.fields->ex)(5, 1, 0)); }
        std::vector<double> zc;
        for (size_t n = 1; n < ex.size(); ++n)
            if (ex[n - 1] * ex[n] < 0) zc.push_back(n - 1 + ex[n - 1] / (ex[n - 1] - ex[n]));
        const std::string tag = "cart2d Langmuir frequency, shape " + std::to_string(order);
        if (zc.size() < 3) { check(tag, false, "no oscillation"); continue; }
        const double w = M_PI * (zc.size() - 1) / ((zc.back() - zc.front()) * c.h[0]);
        const double wth = 2 / c.h[0] * std::asin(wp * c.h[0] / 2);
        check(tag, std::abs(w / wth - 1) < 0.01, fmt("w = %.4f, expected %.4f", w, wth));
    }
}

// C5. Equivalence with Cart3D: a cold foil (electrons and protons, pre-plasma ramp) hit by a TFSF laser (the incident grid),
//     absorbing x, current filter; 3 periodic z cells with z-uniform data against the 2D run with the same dz. The 3D fields
//     of every z plane and the energies per z cell equal the 2D ones to round-off. (Not with tfsf_incident = modes: its
//     frequency lattice depends on the box diagonal, which has no z part in 2D: the two runs then differ by the source's
//     own sampling, ~5e-5 of a0, in 3D as well between boxes of other dz.) The laser-plasma run amplifies round-off (other
//     summation orders of the 2D deposit; between the z planes of the 3D run too: 1e-13 at 10 fs, 1e-9 at 20 fs at a0 = 3):
//     compared at 14 fs, fields to 1e-9 of the peak (measured 1.2e-11 rip_cc, 4e-11 irip), energies to 1e-11.
static const char* kFoil = R"(grid {
  lambda0 = 0.8 um
  nx = 240
  ny = 32
  nz = 3
  dx = 0.025 um
  dy = 0.05 um
  dz = 0.4 um
  boundary_x = absorbing
  x_absorber_width = 0.25 um
  t_end = 14 fs
}
numerics {
  shape = 2
  filter_passes = 1
}
laser {
  a0 = 3
  polarization = y
  duration = 3 fs
  delay = 5 fs
  waist = 0.6 um
  focus = 0.7 um
  tfsf_x = 3.8 um
  y_center = 0.8 um
}
species electrons {
  charge = -1
  mass = 1
  density = 8
  slab_x0 = 4.5 um
  slab_thickness = 0.6 um
  slab_ramp = 0.05 um
  slab_ramp_extent = 0.2 um
  ppc_x = 2
  ppc_y = 2
}
species protons {
  charge = 1
  mass = 1836
  density = 8
  slab_x0 = 4.5 um
  slab_thickness = 0.6 um
  slab_ramp = 0.05 um
  slab_ramp_extent = 0.2 um
  ppc_x = 2
  ppc_y = 2
}
output {
  dir = cart2d_test_out
  every = 100000
}
)";

static double kinetic(const Simulation& sim) {
    double k = 0;
    for (const Species& s : sim.species) {
        const std::vector<double> ux = s.get(Species::UX), uy = s.get(Species::UY), uz = s.get(Species::UZ), w = s.get(Species::W);
        for (size_t p = 0; p < w.size(); ++p) k += w[p] * s.m * (std::sqrt(1 + ux[p] * ux[p] + uy[p] * uy[p] + uz[p] * uz[p]) - 1);
    }
    return k;
}

static void cart2d_equivalence(Comm& comm) {
    const char* path = "cart2d_test_tmp.deck";
    { std::ofstream f(path); f << kFoil; }
    for (const char* sv : {"rip_cc", "irip"}) {
        std::vector<double> F[2][6];
        double wf[2] = {0, 0}, wk[2] = {0, 0}, gd[2] = {0, 0};
        Domain dm[2];
        for (int m = 0; m < 2; ++m) {
            Deck dk = Deck::read(path);
            dk.override_from(std::string("numerics.solver=") + sv);
            if (m) { dk.override_from("grid.geometry=cartesian2d"); dk.override_from("grid.nz=1"); }
            Config c = Config::from_deck(dk);
            Simulation sim(c, comm);
            sim.init();
            while (!sim.finished()) { sim.step(); gd[m] = std::max(gd[m], sim.gauss_drift()); }
            Field* fs[6] = {&sim.fields->ex, &sim.fields->ey, &sim.fields->ez, &sim.fields->bx, &sim.fields->by, &sim.fields->bz};
            for (int q = 0; q < 6; ++q) F[m][q] = fs[q]->host();
            wf[m] = sim.fields->energy() / sim.dom.N[2];
            wk[m] = kinetic(sim) / sim.dom.N[2];
            dm[m] = sim.dom;
        }
        double diff = 0, peak = 0;
        Field a(dm[0]), b(dm[1]);
        for (int q = 0; q < 6; ++q)
            for (int k = 0; k < dm[0].n[2]; ++k)
                for (int j = 0; j < dm[0].n[1]; ++j)
                    for (int i = 0; i < dm[0].n[0]; ++i) {
                        const double u = F[0][q][a.idx(i, j, k)], v = F[1][q][b.idx(i, j, 0)];
                        diff = std::max(diff, std::abs(u - v)); peak = std::max(peak, std::abs(u));
                    }
        diff = comm.allreduce_max(diff); peak = comm.allreduce_max(peak);
        comm.allreduce_sum(wf, 2); comm.allreduce_sum(wk, 2);
        check(std::string("cart2d = Cart3D with 3 periodic z cells (") + sv + ", foil, TFSF laser, absorbing x, filter)",
              diff < 1e-9 * peak && peak > 1 && std::abs(wf[1] / wf[0] - 1) < 1e-11 && std::abs(wk[1] / wk[0] - 1) < 1e-11 &&
                  gd[1] < 1e-12,
              fmt("fields %.1e of the peak", diff / peak) + fmt(", W_field %.1e, W_kin %.1e", wf[1] / wf[0] - 1, wk[1] / wk[0] - 1) +
                  fmt(", Gauss 2D %.1e", gd[1]));
    }
    std::remove(path);
}

// C6. The TFSF plane with exact plane waves in 2D (the TNSA benchmark's source): no field left of the plane, at 0 and 30 deg.
static void cart2d_tfsf_modes(Comm& comm) {
    const char* path = "cart2d_modes_tmp.deck";
    { std::ofstream f(path); f << kFoil; }
    for (const char* sv : {"rip_cc", "irip"})
        for (const char* ang : {"0deg", "30deg"}) {
            Deck dk = Deck::read(path);
            for (const char* o : {"grid.geometry=cartesian2d", "grid.nz=1", "laser.tfsf_incident=modes", "laser.tfsf_x=1um", "laser.duration=6fs", "laser.delay=12fs",
                                  "species electrons.density=0", "species protons.density=0", "grid.t_end=12fs"})
                dk.override_from(o);
            dk.override_from(std::string("numerics.solver=") + sv);
            dk.override_from(std::string("laser.angle=") + ang);
            Config c = Config::from_deck(dk);
            Simulation sim(c, comm);
            sim.init();
            double left = 0, peak = 0;
            int n = 0;
            const int s = sim.fields->tfsf->plane() - sim.dom.off[0];
            while (!sim.finished()) {
                sim.step();
                if (++n % 10) continue;
                for (Field* f : {&sim.fields->ex, &sim.fields->ey, &sim.fields->ez, &sim.fields->bx, &sim.fields->by, &sim.fields->bz}) {
                    H h(*f);
                    for (int j = 0; j < sim.dom.n[1]; ++j)
                        for (int i = 0; i < sim.dom.n[0]; ++i) {
                            const double v = std::abs(h(i, j, 0));
                            peak = std::max(peak, v);
                            if (i < s - 1) left = std::max(left, v);
                        }
                }
            }
            left = comm.allreduce_max(left); peak = comm.allreduce_max(peak);
            check(std::string("cart2d tfsf modes (") + sv + ", " + ang + "): no leakage", left < 1e-13 * peak && peak > 1,
                  fmt("left of the plane %.1e of the peak %.2f", left / peak, peak));
        }
    std::remove(path);
}

// C7. Moving window in 2D: a laser through the open left edge (the inflow hook), a hot plasma with a ramp entering at the right
//     edge as the window moves; the Gauss law at round-off, plasma entered.
static const char* kWin = R"(grid {
  geometry = cartesian2d
  lambda0 = 0.8 um
  nx = 256
  ny = 64
  dx = 0.05 um
  dy = 0.2 um
  boundary_x = open
  t_end = 80 fs
}
numerics {
  shape = 2
}
laser {
  a0 = 2
  polarization = y
  duration = 8 fs
  waist = 3 um
}
window {
  start = 32 fs
}
species electrons {
  charge = -1
  mass = 1
  density = 0.02
  xmin = 13 um
  uth_x = 0.02
  uth_y = 0.02
  uth_z = 0.02
  density_function = "x < 18 um ? (x - 13 um) / (5 um) : 1"
}
species protons {
  charge = 1
  mass = 1836
  density = 0.02
  xmin = 13 um
  density_function = "x < 18 um ? (x - 13 um) / (5 um) : 1"
}
output {
  dir = cart2d_window_out
  every = 100000
}
)";

static void cart2d_window(Comm& comm) {
    const char* path = "cart2d_window_tmp.deck";
    { std::ofstream f(path); f << kWin; }
    for (const char* sv : {"rip_cc", "irip"}) {
        Deck dk = Deck::read(path);
        dk.override_from(std::string("numerics.solver=") + sv);
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        double g = 0;
        while (!sim.finished()) { sim.step(); g = std::max(g, sim.gauss_drift()); }
        double n = double(sim.species[0].size());
        comm.allreduce_sum(&n, 1);
        check(std::string("cart2d moving window (") + sv + ", laser through the left edge, hot plasma entering): Gauss law",
              g < 1e-12 && n > 1000 && sim.dom.wx > 100, fmt("max|dGauss| %.1e, %.0f electrons", g, n) + fmt(", window %.0f cells", double(sim.dom.wx)));
    }
    std::remove(path);
}

// The filter choice (decisions 71, 91): numerics.filter = projection falls back to the local filter where the projection
// cannot run (here: CPML in y), with a warning; an explicit local filter stays; the plain foil keeps the projection. The
// default is the binomial filter (decision 91), with CPML too (it needs no transform).
static void cart2d_filter_fallback(Comm& comm) {
    const char* path = "cart2d_fallback_tmp.deck";
    { std::ofstream f(path); f << kFoil; }
    bool loc[5], bin[5];
    const char* ov[5] = {"numerics.filter=projection", "numerics.filter=projection", "numerics.filter=local", "", ""};
    for (int q = 0; q < 5; ++q) {
        Deck dk = Deck::read(path);
        dk.override_from("grid.geometry=cartesian2d"); dk.override_from("grid.nz=1");
        if (q == 1 || q == 4) { dk.override_from("grid.boundary_y=pml"); dk.override_from("pml.cells=12"); dk.override_from("laser.tfsf_incident=grid"); dk.override_from("laser.angle=0deg"); }
        if (*ov[q]) dk.override_from(ov[q]);
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        loc[q] = sim.cfg.filter_local;
        bin[q] = sim.cfg.filter_binomial;
    }
    std::remove(path);
    check("numerics.filter: the projection falls back to the local filter with CPML in y (and with ranks in y, z); default binomial",
          !loc[0] && loc[1] && loc[2] && !bin[0] && !bin[1] && !bin[2] && bin[3] && !loc[3] && bin[4] && !loc[4],
          std::string(loc[0] ? "foil local" : "foil projection") + (loc[1] ? ", pml local" : ", pml projection") +
              (bin[3] && bin[4] ? ", default binomial" : ", default not binomial"));
}

// C9. O5, iRIP's dispersion-corrected closure (numerics.closure = o5, decision 72):
//     (a) the vacuum dispersion of (kx, ky) modes, both polarisations, against tan^2(w dt/2) = tan^2(kx dx/2) + dx^2 Lam/4 g,
//         g = (1 + 4/3 sin^2(kx dx/2)) (1 + (2/3) min(q, 1) + dy^2 Lam/12), q = dx^2 Lam/4 (the cap of decision 77: some of
//         the modes at dy = dx/4 have q = 8), for dy = dx/4 and dy = 3.2 dx;
//     (b) the corrector's Ex, Bx equal the solve's X, Y (the solved operator is the step's), periodic and open x;
//     (c) random fields in a periodic box, 400 steps, dy = dx/10 .. 3.2 dx: the energy stays below 2.15^2 = 4.62 E_0, the
//         bound of the transient growth per mode of the capped closure (research/o5/stab_limit.py: ||G^n|| <= 2.15 for every
//         dy; without the cap 6.5 at dy = dx/5 and 12.6 at dx/10, decision 77);
//     the Gauss law and div B with plasma: C2 (cart2d_plasma) runs iRIP with both closures.
static void cart2d_o5(Comm& comm) {
    double worst = 0, wdt = 0, pred = 0, gain = 0;
    for (double ry : {0.25, 3.2})
        for (int c = 0; c < 3; ++c) {
            const double hx = 0.5, hy = ry * hx;
            Domain d = flat_box(comm, 10, 8, hx, hy, 0.4, Solver::IRip);
            Fields f(d);
            f.o5 = true;
            const int mx = c == 1 ? 3 : 1, my = c == 2 ? 0 : (c == 1 ? 2 : 1);
            const double kx = 2 * M_PI * mx / (d.N[0] * d.h[0]), ky = 2 * M_PI * my / (d.N[1] * d.h[1]);
            for (Field* F : c == 1 ? std::vector<Field*>{&f.bx, &f.ez} : std::vector<Field*>{&f.ey, &f.ex, &f.by}) {
                H h(*F);
                for (int j = 0; j < d.n[1]; ++j)
                    for (int i = 0; i < d.n[0]; ++i) h(i, j, 0) = std::cos(kx * i * d.h[0] + ky * j * d.h[1] + 0.3);
                h.put(*F);
            }
            comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
            const double D = d.h[0], sy = std::sin(ky * d.h[1] / 2), lam = 4 * sy * sy / (d.h[1] * d.h[1]);
            const double tx = std::tan(kx * D / 2), sx = std::sin(kx * D / 2);
            const double g = (1 + 4.0 / 3 * sx * sx) * (1 + 2.0 / 3 * std::min(D * D * lam / 4, 1.0) + hy * hy * lam / 12);
            pred = 2 * std::atan(std::sqrt(tx * tx + D * D * lam / 4 * g));
            const double plain = 2 * std::atan(std::sqrt(tx * tx + D * D * lam / 4)), exact = D * std::hypot(kx, ky);
            if (my > 0) gain = std::max(gain, std::abs(plain - exact) / std::abs(pred - exact));
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
    check("cart2d irip, closure o5: vacuum dispersion of (kx, ky) modes = the corrected relation (dy = dx/4, 3.2 dx)",
          worst < 1e-11, fmt("last w dt %.6f", wdt) + fmt(" vs %.6f, max error %.1e", pred, worst) +
              fmt("; these modes: error against c |k| %.0f times smaller than the box closure", gain));
    // (b), (c): random fields
    double resid = 0, wmax = 0, wend = 0;
    for (int open = 0; open < 2; ++open)
        for (double ry : {0.1, 0.25, 1.0, 3.2}) {
            Domain d;
            d.N[0] = 16; d.N[1] = 12; d.N[2] = 1;
            d.h[0] = 0.5; d.h[1] = ry * 0.5; d.h[2] = 0.4;
            d.flat = true; d.ng = 3; d.solver = Solver::IRip;
            d.periodic[0] = open == 0;
            const int want[3] = {1, 1, 1};
            comm.decompose(d, want);
            Fields f(d);
            f.o5 = true;
            std::mt19937_64 rng(71 + open + int(4 * ry));
            std::normal_distribution<double> nd;
            for (Field* F : {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz}) {
                H h(*F);
                for (int j = 0; j < d.n[1]; ++j)
                    for (int i = 0; i < d.n[0]; ++i) h(i, j, 0) = (open && (i < 4 || i >= d.n[0] - 4)) ? 0.0 : nd(rng);
                h.put(*F);
            }
            comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
            const double w0 = f.energy();
            double fmax = 1;
            const int steps = open ? 6 : 400;
            for (int n = 0; n < steps; ++n) {
                f.zero_currents(); f.step(comm);
                resid = std::max(resid, irip_residual(f, d) / fmax);
                if (!open) wmax = std::max(wmax, f.energy() / w0);
            }
            if (!open) wend = std::max(wend, f.energy() / w0);
        }
    check("cart2d irip, closure o5: corrector = solve (periodic and open x); random fields over 400 steps (dy = dx/10 .. 3.2 dx) "
          "below the bound 4.62 of the capped closure",
          resid < 1e-12 && wmax < 4.62 && wmax > 0.2, fmt("max|Ex - X| %.1e", resid) + fmt(", W / W0 max %.3f, end %.3f", wmax, wend));
}

void test_cart2d(Comm& comm) {
    cart2d_o5(comm);
    cart2d_filter_fallback(comm);
    cart2d_continuity(comm);
    cart2d_plasma(comm);
    cart2d_dispersion(comm);
    cart2d_langmuir(comm);
    cart2d_equivalence(comm);
    cart2d_tfsf_modes(comm);
    cart2d_window(comm);
}
