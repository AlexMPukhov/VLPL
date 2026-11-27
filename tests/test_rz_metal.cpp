// Embedded perfect conductors in RZ (block rings, docs/specs/rz_metal.md), the RZ self-field and the plane probe.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include "common.hpp"
#include "rz.hpp"
#include "sim.hpp"

namespace {
const double j01 = 2.404825557695773;

std::vector<Field*> all(FieldsRZ& f) { return {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp}; }

RingsConfig ring(double rin, double rout, double p, double g, double z0, double z1) {
    RingsConfig c;
    c.on = true; c.r_in = rin; c.r_out = rout; c.period = p; c.gap = g; c.z_start = z0; c.z_end = z1;
    return c;
}

// Initial fields: m = 0 Ez = J0(kp r) cos(kz z), m = 1 Re Bz = J1(kp' r) sin(kz z) (+ a second harmonic), z = (i - i0) dz
// on the columns [ia, ib] (all if ia > ib), 0 elsewhere.
void init_modes(FieldsRZ& f, double R, double kz, int i0, int ia, int ib) {
    const Domain& d = f.dom();
    const RadialGrid& g = f.grid();
    std::vector<double> ez = f.ez.host(), bz = f.bz.host();
    for (int i = 0; i < d.n[0]; ++i) {
        const int I = d.off[0] + i;
        if (ia <= ib && (I < ia || I > ib)) continue;
        const double z = (I - i0) * d.h[0];
        for (int j = 0; j < d.n[1]; ++j) {
            ez[f.ez.idx(i, j, 0)] = std::cyl_bessel_j(0, j01 / R * g.r[size_t(j)]) * (std::cos(kz * z) + 0.3 * std::cos(2 * kz * z));
            bz[f.bz.idx(i, j, 1)] = std::cyl_bessel_j(1, 1.84118 / R * g.rh[size_t(j)]) * (std::sin(kz * z) + 0.5 * std::sin(3 * kz * z));
        }
    }
    f.ez.upload(ez); f.bz.upload(bz);
}

// M1. A tube (gap = 0) from r_in = R to the wall of a larger grid reproduces the conducting wall at R, inside R to
//     round-off, after 300 steps of standing TM (m = 0) and TE (m = 1) fields; the fields in the metal stay 0.
void metal_tube(Comm& comm) {
    const int nz = 24, nr = 16, nr2 = 26;
    const double dr = 0.1, dz = 0.07, R = nr * dr, kz = 2 * M_PI / (nz * dz);
    Domain da = rz_domain(comm, nz, nr, 2, dz, dr, true, Solver::RipCC), db = rz_domain(comm, nz, nr2, 2, dz, dr, true, Solver::RipCC);
    FieldsRZ a(da, 2), b(db, 2);
    b.metal = rz_metal_setup(ring(R, 10.0, 1.0, 0.0, 0.0, nz * dz), nz, dz, true, b.grid());
    init_modes(a, R, kz, 0, 1, 0);
    init_modes(b, R, kz, 0, 1, 0);
    for (FieldsRZ* f : {&a, &b}) {                 // the metal region of b starts empty
        comm.exchange(f->dom(), all(*f));
    }
    {
        std::vector<double> h = b.ez.host(), k = b.bz.host();
        for (int i = 0; i < db.n[0]; ++i)
            for (int j = nr; j < nr2; ++j)
                for (int s = 0; s < 3; ++s) { h[b.ez.idx(i, j, s)] = 0; k[b.bz.idx(i, j, s)] = 0; }
        b.ez.upload(h); b.bz.upload(k);
        comm.exchange(db, all(b));
    }
    for (int n = 0; n < 300; ++n) { a.zero_currents(); a.step(comm); b.zero_currents(); b.step(comm); }
    double diff = 0, sc = 0, inside = 0;
    Field* fa[6] = {&a.ez, &a.er, &a.ep, &a.bz, &a.br, &a.bp};
    Field* fb[6] = {&b.ez, &b.er, &b.ep, &b.bz, &b.br, &b.bp};
    for (int c = 0; c < 6; ++c) {
        const std::vector<double> x = fa[c]->host(), y = fb[c]->host();
        for (int s = 0; s < 3; ++s)
            for (int i = 0; i < da.n[0]; ++i) {
                for (int j = 0; j < nr; ++j) {
                    diff = std::max(diff, std::abs(x[fa[c]->idx(i, j, s)] - y[fb[c]->idx(i, j, s)]));
                    sc = std::max(sc, std::abs(x[fa[c]->idx(i, j, s)]));
                }
                for (int j = nr + 1; j < nr2; ++j) inside = std::max(inside, std::abs(y[fb[c]->idx(i, j, s)]));
            }
    }
    diff = comm.allreduce_max(diff) / comm.allreduce_max(sc); inside = comm.allreduce_max(inside);
    check("rz metal M1: a tube r_in = R in a larger grid = the conducting wall at R (TM m = 0, TE m = 1, 300 steps)",
          diff < 1e-13 && inside == 0, fmt("max difference %.1e of max", diff) + fmt(", fields inside the metal %.1e", inside));
}

// M7. Jacket (block rings, jacket > 0): a solid tube r_in = R .. R2 (gap = 0) with a jacket R2 .. R3 is the same metal as
//     one tube R .. R3: the fields agree bit for bit after 300 steps (periodic z, TM m = 0 and TE m = 1 modes); with a
//     jacket thinner than a radial cell setup refuses.
void metal_jacket_union(Comm& comm) {
    const int nz = 24, nr = 30;
    const double dr = 0.1, dz = 0.07, R = 16 * dr, kz = 2 * M_PI / (nz * dz);
    Domain d = rz_domain(comm, nz, nr, 2, dz, dr, true, Solver::RipCC);
    FieldsRZ a(d, 2), b(d, 2);
    a.metal = rz_metal_setup(ring(R, R + 6 * dr, 1.0, 0.0, 0.0, nz * dz), nz, dz, true, a.grid());
    RingsConfig cj = ring(R, R + 3 * dr, 1.0, 0.0, 0.0, nz * dz);
    cj.jacket = 3 * dr;
    b.metal = rz_metal_setup(cj, nz, dz, true, b.grid());
    for (FieldsRZ* f : {&a, &b}) { init_modes(*f, R, kz, 0, 1, 0); comm.exchange(f->dom(), all(*f)); }
    for (int n = 0; n < 300; ++n) { a.zero_currents(); a.step(comm); b.zero_currents(); b.step(comm); }
    double diff = 0;
    Field* fa[6] = {&a.ez, &a.er, &a.ep, &a.bz, &a.br, &a.bp};
    Field* fb[6] = {&b.ez, &b.er, &b.ep, &b.bz, &b.br, &b.bp};
    for (int c = 0; c < 6; ++c) {
        const std::vector<double> x = fa[c]->host(), y = fb[c]->host();
        for (size_t q = 0; q < x.size(); ++q) diff = std::max(diff, std::abs(x[q] - y[q]));
    }
    diff = comm.allreduce_max(diff);
    bool thin = false;
    RingsConfig ct = cj;
    ct.jacket = 0.3 * dr;
    try { rz_metal_setup(ct, nz, dz, true, b.grid()); } catch (const std::exception& e) { thin = std::string(e.what()).find("jacket") != std::string::npos; }
    check("rz metal M7: tube R..R2 + jacket R2..R3 = one tube R..R3 bit for bit (300 steps, TM m = 0, TE m = 1); thin jacket refused",
          diff == 0 && thin, fmt("max difference %.1e", diff));
}

// Frequency w dt from the increments of a sampled history (the recurrence of the guided-mode tests in test_rz.cpp)
double recurrence(const std::vector<std::vector<double>>& hist) {
    double num = 0, den = 0;
    for (size_t n = 1; n + 2 < hist.size(); ++n)
        for (size_t q = 0; q < hist[n].size(); ++q) {
            const double dm = hist[n][q] - hist[n - 1][q], d0 = hist[n + 1][q] - hist[n][q], dp = hist[n + 2][q] - hist[n + 1][q];
            num += (dp + dm) * d0; den += 2 * d0 * d0;
        }
    return std::acos(num / den);
}

// M2. Cavity closed by discs (r_in = 0 to the wall, 3 cells thick, periodic z): faces at cells ia and ib = ia + l of the
//     next disc. By the mirror symmetry of the RIP transport the cavity field equals that of a periodic box of length 2 l
//     (no metal) with the even (TM: Ez ~ cos) and odd (TE: Bz ~ sin) continuation: to round-off over the cavity
//     columns, faces included, after 400 steps. The TM_011 frequency against the continuum sqrt((j01/R)^2 + (pi/l)^2)
//     (the RIP-CC radial error, as the guided TM01 of test_rz.cpp).
void metal_cavity(Comm& comm) {
    const int l = 20, t = 3, nzc = l + t, nzp = 2 * l, nr = 16;
    const double dr = 0.1, dz = 0.07, R = nr * dr, kz = M_PI / (l * dz);
    Domain dc = rz_domain(comm, nzc, nr, 2, dz, dr, true, Solver::RipCC), dp = rz_domain(comm, nzp, nr, 2, dz, dr, true, Solver::RipCC);
    FieldsRZ c(dc, 2), p(dp, 2);
    // the disc occupies cells 1 .. 1 + t; the cavity runs from the face 1 + t to the face 1 + nzc (= 1 of the next period)
    c.metal = rz_metal_setup(ring(0.0, 10.0, nzc * dz, (nzc - t) * dz, 1 * dz, (1 + t) * dz + 1e-9), nzc, dz, true, c.grid());
    const int f0 = 1 + t;                          // left face of the cavity (the disc's right face)
    init_modes(p, R, kz, 0, 1, 0);
    // cavity: z measured from the face f0; columns f0 .. nzc - 1 and 0, 1 (the next disc's left face at 1 + nzc = 1)
    {
        const RadialGrid& g = c.grid();
        std::vector<double> ez = c.ez.host(), bz = c.bz.host();
        for (int i = 0; i < dc.n[0]; ++i) {
            const int I = dc.off[0] + i;
            const int s = I >= f0 ? I - f0 : I + nzc - f0;    // cells from the left face
            if (s < 0 || s > l) continue;
            const double z = s * dz;
            for (int j = 0; j < nr; ++j) {
                ez[c.ez.idx(i, j, 0)] = std::cyl_bessel_j(0, j01 / R * g.r[size_t(j)]) * (std::cos(kz * z) + 0.3 * std::cos(2 * kz * z));
                bz[c.bz.idx(i, j, 1)] = std::cyl_bessel_j(1, 1.84118 / R * g.rh[size_t(j)]) * (std::sin(kz * z) + 0.5 * std::sin(3 * kz * z));
            }
        }
        c.ez.upload(ez); c.bz.upload(bz);
    }
    comm.exchange(dc, all(c)); comm.exchange(dp, all(p));
    std::vector<std::vector<double>> hist;
    const double w0 = c.energy();
    for (int n = 0; n < 400; ++n) {
        c.zero_currents(); c.step(comm); p.zero_currents(); p.step(comm);
        if (n < 120) hist.push_back(c.ez.host());
    }
    double diff = 0, sc = 0;
    Field* fc[6] = {&c.ez, &c.er, &c.ep, &c.bz, &c.br, &c.bp};
    Field* fp[6] = {&p.ez, &p.er, &p.ep, &p.bz, &p.br, &p.bp};
    if (comm.size == 1)
        for (int q = 0; q < 6; ++q) {
            const std::vector<double> x = fc[q]->host(), y = fp[q]->host();
            for (int s = 0; s < 3; ++s)
                for (int k = 0; k <= l; ++k) {
                    const int ic = (f0 + k) % nzc;
                    for (int j = 0; j < nr; ++j) {
                        diff = std::max(diff, std::abs(x[fc[q]->idx(ic, j, s)] - y[fp[q]->idx(k, j, s)]));
                        sc = std::max(sc, std::abs(y[fp[q]->idx(k, j, s)]));
                    }
                }
        }
    diff /= std::max(sc, 1e-300);
    const double w1 = c.energy();
    // TM_011 and TM_012 are both in the m = 0 data; fit the dominant one from a pure TM_011 run
    FieldsRZ c1(dc, 1 + 1);
    c1.metal = c.metal;
    {
        const RadialGrid& g = c1.grid();
        std::vector<double> ez = c1.ez.host();
        for (int i = 0; i < dc.n[0]; ++i) {
            const int I = dc.off[0] + i, s = I >= f0 ? I - f0 : I + nzc - f0;
            if (s < 0 || s > l) continue;
            for (int j = 0; j < nr; ++j) ez[c1.ez.idx(i, j, 0)] = std::cyl_bessel_j(0, j01 / R * g.r[size_t(j)]) * std::cos(kz * s * dz);
        }
        c1.ez.upload(ez);
    }
    comm.exchange(dc, all(c1));
    hist.clear();
    for (int n = 0; n < 80; ++n) { hist.push_back(c1.ez.host()); c1.zero_currents(); c1.step(comm); }
    const double w = recurrence(hist) / dz, wth = std::sqrt(std::pow(j01 / R, 2) + kz * kz);
    check("rz metal M2: cavity closed by discs = mirror-symmetric periodic box (TM, TE; faces incl.); TM011 frequency",
          (comm.size > 1 || diff < 1e-12) && std::abs(w / wth - 1) < 5e-3 && std::abs(w1 / w0 - 1) < 0.05,
          fmt("max difference %.1e of max", diff) + fmt(", TM011 w/w_continuum - 1 = %.1e", w / wth - 1) +
              fmt(", W(400)/W(0) = %.4f", w1 / w0));
}

// M3. Deck path: hot electrons and ions in a periodic tube with rings (modes 2, RIP-CC, shape 1 and 2): the Gauss law at
//     every node off the metal surface and div B stay at round-off while the rings absorb particles; nothing deep in metal.
void metal_plasma(Comm& comm) {
    const char* path = "rz_metal_test_tmp.deck";
    {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n modes = 2\n lambda0 = 1 um\n nz = 50\n nr = 24\n dz = 0.04 um\n dr = 0.06 um\n"
             " t_end = 30 fs\n}\nnumerics {\n solver = rip_cc\n pusher = boris\n}\n"
             "rings {\n r_in = 0.48 um\n r_out = 0.9 um\n period = 1 um\n gap = 0.5 um\n z_start = 0.3 um\n z_end = 1.9 um\n}\n"
             "species e {\n charge = -1\n mass = 1\n density = 0.02\n rmax = 1.3 um\n ppc_r = 2\n ppc_theta = 6\n"
             " uth_z = 0.15\n uth_x = 0.15\n uth_y = 0.15\n uz = 0.1\n}\n"
             "species i {\n charge = 1\n mass = 100\n density = 0.02\n rmax = 1.3 um\n ppc_r = 2\n ppc_theta = 6\n"
             " uth_z = 0.01\n uth_x = 0.01\n uth_y = 0.01\n}\n"
             "output {\n dir = rz_metal_test_out\n every = 1000\n}\n";
    }
    Deck dk = Deck::read(path);
    // both rings.particles modes (decision 122): absorb, then pass (the default: the plasma crosses the rings; more particles
    // survive, only the outer wall takes them)
    double nabs = 0;
    for (int run = 0; run < 8; ++run) {
        const int jk = run / 2;
        const bool absorb = run % 2 == 0;
        const int order = 1 + jk % 2;
        const std::string tag = std::string(jk < 2 ? "" : " + jacket") + (absorb ? ", particles = absorb" : ", particles = pass");
        dk.override_from(std::string("numerics.shape=") + std::to_string(order));
        dk.override_from(absorb ? "rings.particles=absorb" : "rings.particles=pass");
        if (run == 4) { dk.override_from("rings.r_out=0.72um"); dk.override_from("rings.jacket=0.24um"); }
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        double n0 = double(sim.species[0].size()) + double(sim.species[1].size());
        n0 = comm.allreduce_max(n0);
        while (!sim.finished()) sim.step();
        double wf, g, rmax, db;
        sim.rz_diagnostics(wf, g, rmax, db);
        double n1 = double(sim.species[0].size()) + double(sim.species[1].size()), lost = sim.species[0].lost_energy;
        n1 = comm.allreduce_max(n1);
        // field inside the interior of the metal
        const MetalView mv = sim.rz->metal->host_view();
        const std::vector<double> ez = sim.rz->ez.host(), er = sim.rz->er.host();
        double inside = 0;                         // points whose four neighbours are metal too
        for (int i = 0; i < sim.dom.n[0]; ++i) {
            const long I = sim.dom.cell0(0) + i;
            for (int j = 1; j + 1 < sim.dom.n[1]; ++j)
                for (int s = 0; s < 3; ++s) {
                    if (mv.M(I, j) && mv.M(I - 1, j) && mv.M(I + 1, j) && mv.M(I, j - 1) && mv.M(I, j + 1))
                        inside = std::max(inside, std::abs(ez[sim.rz->ez.idx(i, j, s)]));
                    if (mv.MH(I, j) && mv.MH(I - 1, j) && mv.MH(I + 1, j) && mv.MH(I, j - 1) && mv.MH(I, j + 1))
                        inside = std::max(inside, std::abs(er[sim.rz->er.idx(i, j, s)]));
                }
        }
        inside = comm.allreduce_max(inside);
        check("rz metal M3: plasma with rings" + tag + " (shape " + std::to_string(order) + "): Gauss law and div B off the surface at round-off",
              g < 1e-12 * std::max(rmax, 1e-30) + 1e-14 && db < 1e-12 && (absorb ? n1 < n0 : n1 > nabs && n1 <= n0) && inside == 0,
              fmt("Gauss %.1e", g) + fmt(" (max rho %.1e)", rmax) + fmt(", div B %.1e", db) + fmt(", particles %.0f", n0) +
                  fmt(" -> %.0f", n1) + fmt(", E_part_out(e) %.2e", lost));
        if (absorb) nabs = n1;
    }
    bool rejected = false;
    dk.override_from("rings.jacket=0um");
    dk.override_from("numerics.solver=irip");
    std::string why;                               // decision 124: in irip the rings are a Drude medium named rings
    try {
        const Config ci = Config::from_deck(dk);
        rejected = !(ci.media.size() == 1 && ci.media[0].name == "rings" && !ci.rings.on);
    } catch (const std::exception& e) { rejected = true; why = e.what(); }
    check("rz metal: rings with iRIP become a Drude medium (decision 124)", !rejected, why);
    if (comm.rank == 0) std::remove(path);
}

// M4. Self-field in RZ: a proton beam at gamma = 10 (Gaussian, open z, moving window off): the Gauss law right after the
//     init at round-off (without self_field it is -rho); B = beta x E; then 200 steps in vacuum: the Gauss drift stays at
//     round-off and the field energy constant to 5e-3 (measured -1.9e-3: the discrete steady state of RIP-CC differs slightly from the Gauss-law field, a weak transient); W_kin unchanged to 1e-9.
void self_field(Comm& comm) {
    const char* path = "rz_self_test_tmp.deck";
    {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n modes = 2\n lambda0 = 1 um\n nz = 200\n nr = 32\n dz = 0.04 um\n dr = 0.06 um\n"
             " boundary_z = open\n boundary_r = wall\n t_end = 10 fs\n}\nnumerics {\n solver = rip_cc\n}\n"
             "species b {\n charge = 1\n mass = 1836.15\n density = 0.001\n uz = 9.95\n self_field = true\n ppc_z = 2\n ppc_r = 2\n ppc_theta = 6\n"
             " density_function = gauss(z - 2.5 um, 0.3 um) * gauss(r, 0.4 um)\n rmax = 1.4 um\n}\n"
             "output {\n dir = rz_self_test_out\n every = 1000\n}\n";
    }
    Deck dk = Deck::read(path);
    Config c = Config::from_deck(dk);
    Simulation sim(c, comm);
    sim.init();
    Field res(sim.dom);
    sim.rz_gauss_residual(res);
    const std::vector<double> h = res.host();
    double g0 = 0;
    for (int i = 0; i < sim.dom.n[0]; ++i)
        for (int j = 0; j < sim.dom.n[1]; ++j)
            for (int s = 0; s < 3; ++s)
                if (sim.dom.off[0] + i < sim.dom.N[0] - 1) g0 = std::max(g0, std::abs(h[res.idx(i, j, s)]));
    g0 = comm.allreduce_max(g0);
    double wf0, gd, rmax, db;
    sim.rz_diagnostics(wf0, gd, rmax, db);
    wf0 = sim.rz->energy();
    comm.allreduce_sum(&wf0, 1);
    double k0 = sim.species[0].kinetic_energy();
    comm.allreduce_sum(&k0, 1);
    while (!sim.finished()) sim.step();
    double wf1 = sim.rz->energy(), k1 = sim.species[0].kinetic_energy();
    comm.allreduce_sum(&wf1, 1); comm.allreduce_sum(&k1, 1);
    double wdum;
    sim.rz_diagnostics(wdum, gd, rmax, db);
    check("rz self_field M4: Gauss law at the init, steady field of a gamma = 10 beam, Gauss drift",
          g0 < 1e-12 * rmax && gd < 1e-12 * rmax && std::abs(wf1 / wf0 - 1) < 5e-3 && std::abs(k1 / k0 - 1) < 1e-6,
          fmt("init residual %.1e", g0 / rmax) + fmt(" of max rho, drift %.1e", gd / rmax) + fmt(", W(T)/W(0) - 1 = %.1e", wf1 / wf0 - 1) +
              fmt(", W_kin change %.1e", k1 / k0 - 1));
    if (comm.rank == 0) std::remove(path);
}
// M5. Start-up in a moving window at large gamma (decision 95): an electron beam alone at u = 1e4 in RZ, 12 dz behind the
//     right edge, the window moving from t = 0: max|dGauss| over 12 shifts at round-off (before: the zigzag field tail
//     of the boosted Poisson operator per radial eigenmode reached the right edge and was cut by the first shift). iRZ keeps
//     the unclamped field (decision 99, explicit RIP-CC only) and the Gauss law too: 1.9e-14.
void self_field_window(Comm& comm, const char* solver) {
    const char* path = "rz_self_window_tmp.deck";
    {
        std::ofstream f(path);
        f << R"(grid {
 geometry = rz
 modes = 2
 lambda0 = 1 um
 nz = 96
 nr = 16
 dz = 0.25 um
 dr = 0.5 um
 boundary_z = open
 boundary_r = wall
 t_end = 10 fs
}
numerics {
 solver = )" << solver << R"(
}
window {
 start = 0 fs
 velocity = 1
}
species b {
 charge = -1
 mass = 1
 density = 0.16
 uz = 10000
 self_field = true
 ppc_z = 2
 ppc_r = 2
 ppc_theta = 6
 density_function = gauss(z - 14 um, 2 um) * gauss(r, 1 um) * (abs(z - 14 um) < 6 um)
 rmax = 3 um
}
output {
 dir = rz_self_window_out
 every = 1000
}
)";
    }
    Deck dk = Deck::read(path);
    Config c = Config::from_deck(dk);
    c.t_end = 12 * c.h[0];
    Simulation sim(c, comm);
    sim.init();
    double g = 0, wf, gd, rmax, db;
    while (!sim.finished()) {
        sim.step();
        sim.rz_diagnostics(wf, gd, rmax, db);
        g = std::max(g, gd);
    }
    check(std::string("rz self_field M5 (") + solver + "): u = 1e4 beam in a moving window from t = 0, Gauss law over 12 shifts", g < 1e-12 * rmax,
          fmt("max|dGauss| %.1e of max rho", g / rmax));
}

// M6. External injection in RZ (decision 98): a gamma = 50 electron beam with its self-field loaded at t_start into a
//     moving window with a cold neutral plasma entering at the right edge, RIP-CC and iRIP. No beam particle before
//     t_start, a constant count afterwards, and the Gauss drift at round-off (the solve adds to the fields present) up to
//     the discrete field tail ahead of the beam at the right edge (docs/specs/beam_self_field.md): RIP-CC 3.7e-12 of max
//     rho with the beam 9 um from the edge (5.6e-10 at 5 um), iRIP 1e-13.
void injected(Comm& comm, const char* solver) {
    const char* path = "rz_inject_test_tmp.deck";
    {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n modes = 2\n lambda0 = 1 um\n nz = 240\n nr = 32\n dz = 0.05 um\n dr = 0.08 um\n"
             " boundary_z = open\n boundary_r = wall\n t_end = 40 fs\n}\nnumerics {\n solver = " << solver << "\n shape = 2\n}\n"
             "window {\n start = 0 fs\n velocity = 1\n}\n"
             "species e {\n charge = -1\n mass = 1\n density = 0.01\n zmin = 12.5 um\n ppc_z = 1\n ppc_r = 1\n ppc_theta = 6\n}\n"
             "species p {\n charge = 1\n mass = 1836.15\n density = 0.01\n zmin = 12.5 um\n ppc_z = 1\n ppc_r = 1\n ppc_theta = 6\n}\n"
             "species b {\n charge = -1\n mass = 1\n density = 0.02\n uz = 50\n self_field = true\n t_start = 20 fs\n"
             " ppc_z = 2\n ppc_r = 2\n ppc_theta = 6\n"
             " density_function = gauss(z - 9 um, 0.3 um) * gauss(r, 0.4 um)\n zmin = 7.8 um\n zmax = 10.2 um\n rmax = 1.6 um\n}\n"
             "output {\n dir = rz_inject_test_out\n every = 10000\n}\n";
    }
    Deck dk = Deck::read(path);
    Config c = Config::from_deck(dk);
    Simulation sim(c, comm);
    sim.init();
    size_t ib = 0;                                       // the beam (the Simulation orders the species itself)
    while (sim.cfg.species[ib].name != "b") ++ib;
    const double ts = sim.cfg.species[ib].t_start;
    bool early = false;
    long n_at = -1, n_end = 0;
    double gmax = 0, rmax = 0;
    while (!sim.finished()) {
        sim.step();
        double wf, gd, rm, db;
        sim.rz_diagnostics(wf, gd, rm, db);
        gmax = std::max(gmax, gd); rmax = std::max(rmax, rm);
        double nb = double(sim.species[ib].size());
        comm.allreduce_sum(&nb, 1);
        if (sim.t < ts - 0.5 * c.h[0] && nb > 0) early = true;
        if (n_at < 0 && nb > 0) n_at = long(nb);
        n_end = long(nb);
    }
    check(std::string("rz self_field M6 (") + solver + "): beam loaded at t_start in the window, Gauss drift, count",
          !early && n_at > 0 && n_end == n_at && gmax < 1e-11 * rmax,
          fmt("%.0f beam particles", double(n_end)) + fmt(", max|dGauss| %.1e of max rho", gmax / rmax));
    if (comm.rank == 0) std::remove(path);
}
}  // namespace

void test_rz_metal(Comm& comm) {
    if (comm.size > 1) return;
    metal_tube(comm);
    metal_jacket_union(comm);
    metal_cavity(comm);
    metal_plasma(comm);
    self_field(comm);
    self_field_window(comm, "rip_cc");
    self_field_window(comm, "irip");
    injected(comm, "rip_cc");
    injected(comm, "irip");
}
