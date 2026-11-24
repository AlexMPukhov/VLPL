// RZ on a stretched radial grid (docs/specs/rz_modes.md step 6a): fields in vacuum.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include "filter.hpp"
#include "common.hpp"
#include "rz.hpp"
#include "sim.hpp"
#include "openpmd.hpp"
#ifdef RIP_HDF5
#include "h5.hpp"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
const double j01 = 2.404825557695773, j11p = 1.841183781340659;   // first zero of J0, of J1'

// Random fields in modes 0..M-1 (the non-dynamic axis values zeroed)
void randomize(FieldsRZ& f, Comm& comm, unsigned seed) {
    const Domain& d = f.dom();
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g;
    for (Field* F : {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp}) {
        std::vector<double> h = F->host();
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    double v = g(rng);
                    if (j == 0 && (F == &f.ep || F == &f.br || (F == &f.ez && k > 0))) v = 0;
                    h[F->idx(i, j, k)] = v;
                }
        F->upload(h);
    }
    comm.exchange(d, {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp});
}

// Largest change of a divergence (the last row touches the wall) relative to its initial maximum
double div_change(const Field& a, const std::vector<double>& b0, const Domain& d) {
    const std::vector<double> b = a.host();
    double e = 0, s = 0;
    for (int k = 0; k < d.n[2]; ++k)
        for (int j = 0; j < d.n[1] - 1; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                e = std::max(e, std::abs(b[a.idx(i, j, k)] - b0[a.idx(i, j, k)]));
                s = std::max(s, std::abs(b0[a.idx(i, j, k)]));
            }
    return e / s;
}

// R13. Random fields in modes 0..2 on a stretched grid (8 cells of 0.05, then 12 % per cell up to 0.4):
//      iRIP at dz = 0.4 (8 times the finest cell) conserves the energy of the grid's weights to
//      round-off, RIP-CC (dz from its stability limit) stays bounded; div E and div B per mode unchanged.
void rz_grid_random(Comm& comm) {
    for (int sv = 0; sv < 2; ++sv) {
        RadialGrid g = rz_radial_grid(40, 0.05, 0.4, 1.12, 0.4);
        const double dz = sv ? 0.9 * 2 / std::sqrt(rz_lambda_max(g, 3)) : 0.4;
        Domain d = rz_domain(comm, 16, 40, 3, dz, 0.05, true, sv ? Solver::RipCC : Solver::IRip);
        FieldsRZ f(d, 3, std::move(g));
        randomize(f, comm, 7);
        Field de(d), db(d);
        f.div_e(de); f.div_b(db);
        const std::vector<double> e0 = de.host(), b0 = db.host();
        const double w0 = f.energy();
        double dev = 0, wmax = 0;
        for (int n = 0; n < 300; ++n) {
            f.zero_currents(); f.step(comm);
            dev = std::max(dev, std::abs(f.energy() / w0 - 1)); wmax = std::max(wmax, f.energy() / w0);
        }
        f.div_e(de); f.div_b(db);
        const double ge = div_change(de, e0, d), gb = div_change(db, b0, d);
        const bool ok = (sv ? wmax < 1.02 : dev < 1e-11) && ge < 1e-12 && gb < 1e-12;
        check(std::string("rz stretched grid (") + (sv ? "rip_cc" : "irip, dz = 8 dr_min") +
                  "): random fields in modes 0..2, energy " + (sv ? "bounded" : "conserved") + ", div E and div B unchanged",
              ok, (sv ? fmt("max W/W0 %.3f", wmax) : fmt("max |W/W0 - 1| %.1e", dev)) + fmt(", div E change %.1e", ge) +
                      fmt(", div B %.1e", gb) + fmt(", dr %.2f .. ", 0.05) + fmt("%.2f", f.grid().dn[39]));
    }
}

// Standing TM01 / TE11 mode of the conducting cylinder r < R on the grid g (iRIP, periodic z): w dt
// from the recurrence of the increments (as tests/test_rz.cpp), relative to the iRIP law with the
// continuum k_perp: tan^2(w dt/2) = tan^2(kz dz/2) + dz^2 kp^2 / 4.
double guided_err(Comm& comm, RadialGrid g, double dz, bool te) {
    const int M = te ? 2 : 1, nz = 32, nk = 2, nr = g.nr;
    const double R = g.R(), kp = (te ? j11p : j01) / R, kz = 2 * M_PI * nk / (nz * dz);
    Domain d = rz_domain(comm, nz, nr, M, dz, g.dr, true, Solver::IRip);
    const std::vector<double> rn = g.r, rh = g.rh;
    FieldsRZ f(d, M, std::move(g));
    Field& F = te ? f.bz : f.ez;
    std::vector<double> h = F.host();
    for (int j = 0; j < nr; ++j)
        for (int i = 0; i < nz; ++i) {
            const double r = te ? rh[size_t(j)] : rn[size_t(j)], z = (d.off[0] + i) * dz;
            h[F.idx(i, j, te ? 1 : 0)] = (te ? std::cyl_bessel_j(1, kp * r) : std::cyl_bessel_j(0, kp * r)) * std::cos(kz * z);
        }
    F.upload(h);
    comm.exchange(d, {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp});
    std::vector<std::vector<double>> hist;
    for (int n = 0; n < 60; ++n) { hist.push_back(F.host()); f.zero_currents(); f.step(comm); }
    double num = 0, den = 0;
    for (size_t n = 1; n + 2 < hist.size(); ++n)
        for (size_t q = 0; q < hist[n].size(); ++q) {
            const double dm = hist[n][q] - hist[n - 1][q], d0 = hist[n + 1][q] - hist[n][q], dp = hist[n + 2][q] - hist[n + 1][q];
            num += (dp + dm) * d0; den += 2 * d0 * d0;
        }
    const double w = std::acos(num / den);
    const double tx = std::tan(kz * dz / 2), wth = 2 * std::atan(std::sqrt(tx * tx + dz * dz * kp * kp / 4));
    return std::abs(w / wth - 1);
}

// R14. TM01 and TE11 on stretched grids (uniform to R/3, then cells growing by s per cell): the frequency
//      error is the radial discretisation, second order: refining (2 nr, dr / 2, s -> sqrt(s)) reduces it by
//      more than 3.
void rz_grid_guided(Comm& comm) {
    for (int te = 0; te < 2; ++te) {
        const double e0 = guided_err(comm, rz_radial_grid(16, 0.1, 0.6, 1.15, 1e300), 0.4, te);
        const double e1 = guided_err(comm, rz_radial_grid(32, 0.05, 0.6, std::sqrt(1.15), 1e300), 0.4, te);
        check(std::string("rz stretched grid: ") + (te ? "TE11" : "TM01") + " guided mode, frequency second order in the cells",
              e1 < 2e-3 && e0 / e1 > 3, fmt("rel. error nr = 16: %.2e", e0) + fmt(", nr = 32: %.2e", e1));
    }
}

// The energy inside r < rin (the grid's weights) of the fields of f
double energy_inside(const FieldsRZ& f, double rin) {
    const Domain& d = f.dom();
    const RadialGrid& g = f.grid();
    double w = 0;
    for (const Field* F : {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp}) {
        const bool half = F == &f.er || F == &f.bz || F == &f.bp;
        const std::vector<double> h = F->host();
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1] && (half ? g.rh : g.r)[size_t(j)] < rin; ++j)
                for (int i = 0; i < d.n[0]; ++i)
                    w += (k ? 2.0 : 1.0) * (half ? g.B : g.A)[size_t(j)] * h[F->idx(i, j, k)] * h[F->idx(i, j, k)];
    }
    return w;
}

// A ring wave e^{-(r - 1.5)^2 / 0.5} cos(4 (r - 1.5)) (radial wavenumber 4, in Ez of m = 0 and Bz of m = 1,
// uniform in z) starts in the fine region r < 3 and runs out through the grid into a 32-cell absorber
// (iRIP, dz = 0.2 = 4 dr_min). After the time to cross the cylinder twice: the energy that came back into
// r < 3, relative to the initial one.
double ring_back(Comm& comm, RadialGrid g) {
    const int nr = g.nr, M = 2, cells = 32;
    const double dz = 0.2, rin = 3.0;
    Domain d = rz_domain(comm, 4, nr, M, dz, g.dr, true, Solver::IRip);
    FieldsRZ f(d, M, std::move(g));
    const RadialGrid& G = f.grid();
    auto ring = [&](Field& F, int slot, bool half) {
        std::vector<double> h = F.host();
        for (int j = 0; j < nr; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                const double x = (half ? G.rh : G.r)[size_t(j)] - 1.5;
                h[F.idx(i, j, slot)] = std::exp(-x * x / 0.5) * std::cos(4 * x);
            }
        F.upload(h);
    };
    ring(f.ez, 0, false); ring(f.bz, 1, true);
    comm.exchange(d, {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp});
    const double w0 = energy_inside(f, rin);
    const double a = (G.R() - G.r[size_t(nr - cells)]) / dz, kappa = 1.5 * std::log(1e4) / a;
    const int steps = int(2.2 * G.R() / dz);
    for (int n = 0; n < steps; ++n) {
        f.zero_currents(); f.step(comm);
        rz_absorb(f, cells, kappa);
        comm.exchange(d, {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp});
    }
    return energy_inside(f, rin) / w0;
}

// R15. No reflection from a stretched grid (the Gate B radiation test, docs/specs/rz_modes.md step 6):
//      the ring wave (k_r = 4) leaves through 0.05 -> 0.2 (k dr_max = 0.8, 4 % per cell) as through the
//      uniform fine grid of the same radius: the energy returning into r < 3 is the absorber's floor. The
//      same grid stretched to 0.6 (k dr_max = 2.4, beyond the cutoff 2 / dr) throws the wave back.
void rz_grid_reflection(Comm& comm) {
    const double uni = ring_back(comm, rz_radial_grid(280, 0.05));                          // R = 14
    RadialGrid gs = rz_radial_grid(110, 0.05, 3.0, 1.04, 0.2);
    const double Rs = gs.R();
    const double str = ring_back(comm, std::move(gs));
    const double cut = ring_back(comm, rz_radial_grid(170, 0.05, 3.0, 1.04, 0.6));   // k dr = 2 at r = 15, absorber from 26
    check("rz stretched grid: an outgoing wave (k dr_max = 0.8, 4 % per cell) leaves as on the uniform grid; "
          "beyond the cutoff (k dr_max = 2.4) it is reflected",
          str < 3e-4 && str < 3 * uni + 1e-5 && cut > 1e-2,
          fmt("energy back in r < 3: uniform %.1e", uni) + fmt(", stretched %.1e", str) + fmt(" (R %.1f)", Rs) +
              fmt(", beyond the cutoff %.1e", cut));
}

// The laser deck of R16 on the given radial grid; returns the field energy and the largest Gauss change
double laser_energy(Comm& comm, const std::string& grid, double& gauss) {
    const char* path = "rz_test_grid_tmp.deck";
    {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n modes = 2\n lambda0 = 0.8 um\n nz = 128\n dz = 0.1 um\n" << grid
          << " boundary_z = open\n t_end = 27 fs\n}\n"
             "numerics {\n solver = irip\n}\nlaser {\n a0 = 1\n polarization = x\n duration = 5 fs\n waist = 1.5 um\n}\n"
             "output {\n dir = rz_test_grid_out\n}\n";
    }
    Config c = Config::from_deck(Deck::read(path));
    if (comm.rank == 0) std::remove(path);
    Simulation sim(c, comm);
    sim.init();
    gauss = 0;
    while (!sim.finished()) {
        sim.step();
        double w, g, r, b;
        sim.rz_diagnostics(w, g, r, b);
        gauss = std::max(gauss, g);
    }
    double w, g, r, b;
    sim.rz_diagnostics(w, g, r, b);
    return comm.allreduce_max(w);
}

// R16. The laser (waist 1.5 um, 5 fs) entering through the open left edge on a stretched grid (0.025 um
//      up to 0.5 um, 5 % per cell up to 0.15 um: k0 dr <= 1 everywhere) carries the energy of the uniform
//      fine grid of the same radius (3.6 um, conducting wall: an absorber of a fixed cell count would be 6x
//      thicker on the stretched grid) to 1e-3, the Gauss law at round-off.
void rz_grid_laser(Comm& comm) {
    double gu, gs;
    const double wu = laser_energy(comm, " nr = 144\n dr = 0.025 um\n", gu);
    const double ws = laser_energy(comm, " nr = 60\n dr = 0.025 um\n r_uniform = 0.5 um\n stretch = 1.05\n dr_max = 0.15 um\n", gs);
    check("rz stretched grid: the laser enters with the energy of the uniform fine grid, Gauss law at round-off",
          std::abs(ws / wu - 1) < 1e-3 && gu < 1e-13 && gs < 1e-13,
          fmt("W_stretched / W_uniform - 1 = %.1e", ws / wu - 1) + fmt(", Gauss %.1e", std::max(gu, gs)) + " (nr 60 against 144)");
}

// The Config of a deck: the lines of the grid block (after geometry and lambda0) and the other blocks.
Config grid_deck(const std::string& grid, const std::string& rest) {
    const char* path = "rz_test_grid_tmp.deck";
    {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n lambda0 = 0.8 um\n" << grid << "}\n" << rest << "output {\n dir = rz_test_grid_out\n}\n";
    }
    Config c = Config::from_deck(Deck::read(path));
    std::remove(path);                             // one rank (test_rz_grid)
    return c;
}

// R17. A hot plasma column (electrons u_th = 0.1, protons) reaching into the stretched cells (0.025 um up to
//      0.2 um, then 8 % per cell up to 0.1 um, wall at 1.5 um, electrons partly absorbed by it), modes 0..2:
//      the Gauss law per mode at round-off with iRIP (shape 1 and 2) and RIP-CC (shape 2): the charge
//      conservation of the deposition in the index coordinate (decision 52).
void rz_grid_plasma(Comm& comm) {
    const char* runs[3][2] = {{"irip", "1"}, {"irip", "2"}, {"rip_cc", "2"}};
    for (auto& rn : runs) {
        const bool ex = std::string(rn[0]) == "rip_cc";
        const std::string grid = std::string(" modes = 3\n nz = 16\n nr = 24\n dz = ") + (ex ? "0.004 um" : "0.05 um") +
                                 "\n dr = 0.025 um\n r_uniform = 0.2 um\n stretch = 1.08\n dr_max = 0.1 um\n t_end = " +
                                 (ex ? "3 fs" : "8 fs") + "\n";
        const std::string rest = std::string("numerics {\n solver = ") + rn[0] + "\n shape = " + rn[1] + "\n}\n"
            "species e {\n charge = -1\n mass = 1\n density = 0.02\n rmax = 1.2 um\n ppc_r = 2\n ppc_theta = 6\n"
            " uth_z = 0.1\n uth_x = 0.1\n uth_y = 0.1\n}\n"
            "species p {\n charge = 1\n mass = 1836\n density = 0.02\n rmax = 1.2 um\n ppc_r = 2\n ppc_theta = 6\n}\n";
        Simulation sim(grid_deck(grid, rest), comm);
        sim.init();
        double g = 0, b = 0, w, gg, r, bb;
        const double n0 = double(sim.species[0].size());
        while (!sim.finished()) {
            sim.step();
            sim.rz_diagnostics(w, gg, r, bb);
            g = std::max(g, gg); b = std::max(b, bb);
        }
        const double n1 = double(sim.species[0].size());
        check(std::string("rz stretched grid: hot plasma (") + rn[0] + ", shape " + rn[1] +
                  "), modes 0..2, Gauss law and div B at round-off with wall absorption",
              g < 1e-13 && b < 1e-13 && n1 < n0, fmt("Gauss %.1e", g) + fmt(", div B %.1e", b) + fmt(", electrons %.0f", n0) + fmt(" -> %.0f", n1));
    }
}

// R18. Loading on the stretched grid: rho = q n of a uniform column (shape 1) at every node inside it, the
//      axis node and the stretch (8 % per cell) included: the Ruyten weights with the Gauss-point loading
//      (decision 53) are exact on any grid.
void rz_grid_loading(Comm& comm) {
    const std::string grid = " modes = 1\n nz = 4\n nr = 24\n dz = 0.05 um\n dr = 0.025 um\n r_uniform = 0.2 um\n"
                             " stretch = 1.08\n dr_max = 0.1 um\n t_end = 1 fs\n";
    // the loading alone: no transverse shape (decision 79 smooths rho as it smooths the current)
    const std::string rest = "numerics {\n solver = irip\n rz_transverse_shape = none\n}\nspecies e {\n charge = -1\n mass = 1\n density = 0.01\n"
                             " rmax = 1.2 um\n ppc_r = 4\n ppc_theta = 4\n}\n";
    Simulation sim(grid_deck(grid, rest), comm);
    sim.init();
    Field rho(sim.dom);
    sim.rz_rho_all(rho);
    const std::vector<double> h = rho.host();
    const RadialGrid& g = sim.rz->grid();
    const double rmax = 1.2e-6 * 2 * M_PI / 0.8e-6;
    double eu = 0, es = 0;
    for (int j = 0; j < sim.dom.n[1] && g.r[size_t(j + 1)] < rmax; ++j)          // the axis node included
        for (int i = 0; i < sim.dom.n[0]; ++i) {
            const double e = std::abs(h[rho.idx(i, j, 0)] / -0.01 - 1);
            (j < g.ju ? eu : es) = std::max(j < g.ju ? eu : es, e);
        }
    check("rz stretched grid: loading rho = q n on every node, the axis and the stretch (Ruyten weights)",
          eu < 1e-12 && es < 1e-12, fmt("uniform part %.1e", eu) + fmt(", stretched part %.1e", es));
}

// R19. A laser wake (a0 = 1.5, waist 1.5 um, n = 0.01 n_c) on the stretched grid (0.025 um up to 0.5 um, 5 %
//      per cell up to 0.15 um) against the uniform fine grid of the same radius: E_z on the axis after the
//      pulse has run 7 um into the plasma, the L2 difference relative to the uniform run below 2 %.
std::vector<double> wake_axis(Comm& comm, const std::string& rgrid) {
    const std::string grid = " modes = 2\n nz = 160\n dz = 0.05 um\n" + rgrid + " boundary_z = open\n t_end = 40 fs\n";
    const std::string rest = "numerics {\n solver = irip\n shape = 2\n}\nlaser {\n a0 = 1.5\n polarization = x\n duration = 6 fs\n"
                             " waist = 1.5 um\n}\nspecies e {\n charge = -1\n mass = 1\n density = 0.01\n zmin = 2 um\n ppc_r = 2\n"
                             " ppc_theta = 4\n}\n";
    Simulation sim(grid_deck(grid, rest), comm);
    sim.init();
    while (!sim.finished()) sim.step();
    const std::vector<double> h = sim.rz->ez.host();
    std::vector<double> v;
    for (int i = 0; i < sim.dom.n[0]; ++i) v.push_back(h[sim.rz->ez.idx(i, 0, 0)]);
    return v;
}
void rz_grid_wake(Comm& comm) {
    const std::vector<double> u = wake_axis(comm, " nr = 144\n dr = 0.025 um\n");
    const std::vector<double> s = wake_axis(comm, " nr = 60\n dr = 0.025 um\n r_uniform = 0.5 um\n stretch = 1.05\n dr_max = 0.15 um\n");
    double num = 0, den = 0, pk = 0;
    for (size_t i = 0; i < u.size(); ++i) { num += (s[i] - u[i]) * (s[i] - u[i]); den += u[i] * u[i]; pk = std::max(pk, std::abs(u[i])); }
    const double l2 = std::sqrt(num / den);
    check("rz stretched grid: laser wake on the axis as on the uniform fine grid (60 against 144 cells)",
          l2 < 2e-2 && pk > 1e-3, fmt("L2 difference of E_z on the axis %.1e", l2) + fmt(", peak %.2e", pk));
}

// R21. A static background (species mobile = false, as B7's ions): a laser wake in RZ with the moving window
//      and electrons, the ions immobile against the same ions mobile with the mass 1e12 m_e: the field
//      energy equal to 1e-10, the immobile ions unmoved (their z only shifts out with the window), the Gauss
//      law at round-off.
double static_run(Comm& comm, bool mobile, double& gauss, bool& unmoved) {
    const std::string grid = " modes = 2\n nz = 64\n nr = 16\n dz = 0.1 um\n dr = 0.05 um\n boundary_z = open\n t_end = 30 fs\n";
    const std::string rest = std::string("numerics {\n solver = irip\n}\nlaser {\n a0 = 1\n polarization = x\n duration = 5 fs\n"
        " waist = 0.4 um\n}\nwindow {\n start = 20 fs\n}\nspecies e {\n charge = -1\n mass = 1\n density = 0.02\n zmin = 6.4 um\n"
        " rmax = 0.5 um\n ppc_r = 2\n ppc_theta = 4\n}\nspecies p {\n charge = 1\n mass = ") + (mobile ? "1e12" : "1") +
        "\n density = 0.02\n zmin = 6.4 um\n rmax = 0.5 um\n ppc_r = 2\n ppc_theta = 4\n mobile = " + (mobile ? "true" : "false") + "\n}\n";
    Simulation sim(grid_deck(grid, rest), comm);
    sim.init();
    gauss = 0;
    double w = 0, g, r, b;
    while (!sim.finished()) {
        sim.step();
        sim.rz_diagnostics(w, g, r, b);
        gauss = std::max(gauss, g);
    }
    // the ions sit on the loading lattice: r at the Gauss points of each cell (ppc_r = 2)
    const std::vector<double> y = sim.species[1].get(Species::Y), zz = sim.species[1].get(Species::Z);
    unmoved = !y.empty();
    const double dr = sim.dom.h[1];
    for (size_t q = 0; q < y.size(); ++q) {
        const double rr = std::sqrt(y[q] * y[q] + zz[q] * zz[q]) / dr, f = rr - std::floor(rr);
        unmoved = unmoved && std::abs(std::abs(f - 0.5) - 0.5 / std::sqrt(3.0)) < 1e-9;
    }
    return w;
}
void rz_static_background(Comm& comm) {
    double gm, gs;
    bool um, us;
    const double wm = static_run(comm, true, gm, um), ws = static_run(comm, false, gs, us);
    check("rz species mobile = false: a static background as heavy ions (window, laser wake), Gauss law at round-off",
          std::abs(ws / wm - 1) < 1e-10 && us && gs < 1e-13, fmt("W(immobile) / W(mass 1e12) - 1 = %.1e", ws / wm - 1) +
              fmt(", Gauss %.1e", gs) + (us ? ", ions unmoved" : ", ions MOVED"));
}

// R22. The density-conserving maps of the deposition (decision 53): the share of every node of a continuous uniform
//      density, by a fine midpoint quadrature over the cells with the maps of the grid's coefficients (independent of
//      the Gauss-Legendre construction in src/rz_ruyten.cpp), equals its volume A_j, shapes 1 and 2, the axis node
//      included, on a uniform and a stretched grid; the plain splines give 4/3 and 13/8 on the axis node.
double node_share(const RadialGrid& g, int shape, int j, bool mapped) {
    auto S = [shape](double x) {
        const double a = std::abs(x);
        if (shape == 1) return a < 1 ? 1 - a : 0.0;
        return a < 0.5 ? 0.75 - a * a : (a < 1.5 ? 0.5 * (1.5 - a) * (1.5 - a) : 0.0);
    };
    const int NQ = 20000;
    double tot = 0;
    for (int c = std::max(0, j - 2); c <= j + 1; ++c)
        for (int q = 0; q < NQ; ++q) {
            const double u = (q + 0.5) / NQ, r = g.r[size_t(c)] + u * g.dn[size_t(c)];
            double xi = c + u;
            if (mapped) xi = shape == 1 ? c + u + g.bt[size_t(c)] * u * (1 - u)
                                        : (c == 0 ? std::pow(u, g.bt2[0]) : c + u + g.bt2[size_t(c)] * u * (1 - u));
            tot += (S(xi - j) + (j > 0 ? S(xi + j) : 0.0)) * r * g.dn[size_t(c)] / NQ;
        }
    return tot;
}
void rz_density_conserving(Comm&) {
    double e1 = 0, e2 = 0, ea = 0, p1 = 0, p2 = 0;
    for (int stretched = 0; stretched < 2; ++stretched) {
        const RadialGrid g = stretched ? rz_radial_grid(60, 0.05, 0.3, 1.06, 0.4) : rz_radial_grid(60, 0.05);
        for (int j = 0; j < 50; ++j) {
            e1 = std::max(e1, std::abs(node_share(g, 1, j, true) / g.A[size_t(j)] - 1));
            e2 = std::max(e2, std::abs(node_share(g, 2, j, true) / g.A[size_t(j)] - 1));
            if (!stretched) ea = std::max(ea, std::abs(g.bt[size_t(j)] * 2 * (2 * j + 1) - 1));
        }
        if (!stretched) { p1 = node_share(g, 1, 0, false) / g.A[0]; p2 = node_share(g, 2, 0, false) / g.A[0]; }
    }
    check("rz density-conserving deposition maps: every node receives its volume's share of a uniform density "
          "(shapes 1, 2; uniform and stretched grid)",
          e1 < 1e-6 && e2 < 1e-6 && ea < 1e-10 && std::abs(p1 - 4.0 / 3) < 1e-6 && std::abs(p2 - 13.0 / 8) < 1e-6,
          fmt("max relative error shape 1 %.1e", e1) + fmt(", shape 2 %.1e", e2) + fmt("; plain splines on the axis: %.4f", p1) +
              fmt(", %.4f", p2) + fmt("; shape 1 bt_j = 1/(2(2j+1)) to %.0e", ea));
}

// R23. The current filter (numerics.filter_passes = 2, decision 54): a hot plasma column (modes 0..2, partly absorbed
//      by the wall) keeps the Gauss law per mode at round-off with the filtered current (the filtered rho of the
//      diagnostics), iRIP and RIP-CC, shapes 1 and 2; also with open z edges: the plasma drifting out through the
//      left edge (a plain binomial along z reached 2e-2 there: the charge of the leaving particles vanishes in the
//      guard cell), and a laser wake with the moving window (RIP-CC: nothing reaches ahead of the pulse, so the
//      entering plasma is neutral and at rest, which the right edge closure needs; iRIP couples the front to the
//      pulse at 1e-7 relative) and plasma leaving from 41 fs (the first filter_passes cells not checked, decision 54). The loading of a uniform column is unchanged by the filter
//      (rho = q n on every node, the axis node too: the radial smoothing leaves a uniform density alone).
//      The default filter is the projection (decision 70); the window case also with the local filter. The projection
//      is an elliptic solve along z: its change of J (the Coulomb field of the filter's charge rearrangement) reaches a
//      few cells ahead of the pulse, so while the pulse meets the plasma at the front edge the entering cells are not
//      quite empty: 1.2e-13 for ~80 steps, then it leaves with the window (local filter 2e-15).
void rz_filter(Comm& comm) {
    double gmax = 0, bmax = 0, gopen = 0, gwin = 0, gwinl = 0;
    for (int run = 3; run < 6; ++run) {
        const bool win = run >= 4, local = run == 5;
        const std::string grid = std::string(" modes = 2\n nz = ") + (win ? "256\n dz = 0.025 um" : "32\n dz = 0.1 um") +
                                 "\n nr = 16\n dr = 0.05 um\n boundary_z = open\n t_end = " + (win ? "55 fs" : "15 fs") + "\n";
        const std::string plasma = std::string(" density = 0.02\n rmax = 0.5 um\n ppc_r = 2\n ppc_theta = 4\n") +
                                   (win ? " zmin = 6.4 um\n" : " uz = -0.5\n uth_x = 0.05\n");
        const std::string rest = std::string("numerics {\n solver = ") + (win ? "rip_cc" : "irip") + "\n filter_passes = 2\n" +
            (local ? " filter = local\n" : "") + "}\n" +
            (win ? "laser {\n a0 = 1\n polarization = x\n duration = 5 fs\n waist = 0.4 um\n}\nwindow {\n start = 20 fs\n}\n" : "") +
            "species e {\n charge = -1\n mass = 1\n" + plasma + "}\nspecies p {\n charge = 1\n mass = 1836\n" + plasma + "}\n";
        Simulation sim(grid_deck(grid, rest), comm);
        sim.init();
        double w, g, r, b;
        size_t nmax = sim.species[0].size();
        while (!sim.finished()) {
            sim.step();
            sim.rz_diagnostics(w, g, r, b);
            double& gg = local ? gwinl : win ? gwin : gopen;
            gg = std::max(gg, g);
            nmax = std::max(nmax, sim.species[0].size());
        }
        if (!win && 2 * sim.species[0].size() > nmax) gopen = 1;       // at least half of the electrons left
    }
    for (int run = 0; run < 3; ++run) {
        const bool ex = run == 2;
        const std::string grid = std::string(" modes = 3\n nz = 16\n nr = 24\n dz = ") + (ex ? "0.004 um" : "0.05 um") +
                                 "\n dr = 0.025 um\n t_end = " + (ex ? "3 fs" : "8 fs") + "\n";
        const std::string rest = std::string("numerics {\n solver = ") + (ex ? "rip_cc" : "irip") + "\n shape = " + (run == 1 ? "2" : "1") +
            "\n filter_passes = 2\n}\nspecies e {\n charge = -1\n mass = 1\n density = 0.02\n rmax = 0.5 um\n ppc_r = 2\n ppc_theta = 6\n"
            " uth_z = 0.1\n uth_x = 0.1\n uth_y = 0.1\n}\n"
            "species p {\n charge = 1\n mass = 1836\n density = 0.02\n rmax = 0.5 um\n ppc_r = 2\n ppc_theta = 6\n}\n";
        Simulation sim(grid_deck(grid, rest), comm);
        sim.init();
        double w, g, r, b;
        while (!sim.finished()) { sim.step(); sim.rz_diagnostics(w, g, r, b); gmax = std::max(gmax, g); bmax = std::max(bmax, b); }
    }
    // the loading of a uniform column with and without the filter
    double lerr = 0;
    for (int fp = 0; fp < 2; ++fp) {
        const std::string grid = " modes = 1\n nz = 8\n nr = 24\n dz = 0.05 um\n dr = 0.025 um\n t_end = 1 fs\n";
        const std::string rest = std::string("numerics {\n solver = irip\n rz_transverse_shape = none\n filter_passes = ") + (fp ? "2" : "0") +
            "\n}\nspecies e {\n charge = -1\n mass = 1\n density = 0.01\n ppc_r = 2\n ppc_theta = 4\n}\n";
        Simulation sim(grid_deck(grid, rest), comm);
        sim.init();
        Field rho(sim.dom);
        sim.rz_rho_all(rho);
        const std::vector<double> h = rho.host();
        for (int j = 0; j < 20; ++j)
            for (int i = 0; i < sim.dom.n[0]; ++i) lerr = std::max(lerr, std::abs(h[rho.idx(i, j, 0)] / -0.01 - 1));
    }
    check("rz current filter (2 passes): Gauss law and div B per mode at round-off (iRIP, RIP-CC, shapes 1, 2; plasma "
          "leaving an open edge; moving window), a uniform density unchanged on every node",
          gmax < 1e-13 && bmax < 1e-13 && gopen < 1e-13 && gwin < 1e-12 && gwinl < 1e-13 && lerr < 1e-12,
          fmt("Gauss %.1e", gmax) + fmt(", open edge %.1e", gopen) + fmt(", window %.1e (local %.1e)", gwin, gwinl) +
              fmt(", div B %.1e", bmax) +
              fmt(", rho error %.1e", lerr));
}

#ifdef RIP_HDF5
// R20. Output and checkpoints on a stretched grid (step 6c): the thetaMode records carry the node radii
//      (rippic_r_nodes); the restart of the hot plasma of R17 (iRIP, shape 2) from a checkpoint equals the
//      uninterrupted run in every field slot and particle, exactly on one thread; a checkpoint of another
//      radial grid is refused.
void rz_grid_output(Comm& comm) {
#ifdef _OPENMP
    const int nthr = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    auto grid = [](const char* stretch) {
        return std::string(" modes = 3\n nz = 16\n nr = 24\n dz = 0.05 um\n dr = 0.025 um\n r_uniform = 0.2 um\n stretch = ") +
               stretch + "\n dr_max = 0.1 um\n t_end = 8 fs\n";
    };
    const std::string rest = "numerics {\n solver = irip\n shape = 2\n}\n"
        "species e {\n charge = -1\n mass = 1\n density = 0.02\n rmax = 1.2 um\n ppc_r = 2\n ppc_theta = 6\n"
        " uth_z = 0.1\n uth_x = 0.1\n uth_y = 0.1\n}\n"
        "species p {\n charge = 1\n mass = 1836\n density = 0.02\n rmax = 1.2 um\n ppc_r = 2\n ppc_theta = 6\n}\n";
    const Config c = grid_deck(grid("1.08"), rest);
    Simulation a(c, comm);
    a.init();
    while (!a.finished()) a.step();
    std::string chk;
    long nchk = 0;
    bool nodes = false;
    {
        Simulation b(c, comm);
        b.init();
        while (b.istep < a.istep / 2) b.step();
        chk = write_checkpoint(b);
        nchk = b.istep;
        write_openpmd(b);
        h5::Id f = h5::open_file("rz_test_grid_out/openpmd/" + std::string("data_") + std::string(6 - std::to_string(nchk).size(), '0') +
                                 std::to_string(nchk) + ".h5");
        h5::Id e = h5::open_group(f, "data/" + std::to_string(nchk) + "/meshes/E");
        const std::vector<double> rn = h5::attr_vec<double>(e, "rippic_r_nodes");
        const RadialGrid& g = b.rz->grid();
        nodes = rn == std::vector<double>(g.r.begin(), g.r.begin() + g.nr + 1);
        for (int q = 0; q < 7; ++q) b.step();
    }
    Simulation r(c, comm);
    r.init(chk);
    while (!r.finished()) r.step();
    double worst = 0;
    bool same = a.istep == r.istep && r.restart_step == nchk;
    auto cmp = [&](const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size()) { same = false; return; }
        double m = 0, d = 0;
        for (size_t q = 0; q < x.size(); ++q) { m = std::max(m, std::abs(x[q])); d = std::max(d, std::abs(x[q] - y[q])); }
        if (d > 0) worst = std::max(worst, d / m);
    };
    for (Field FieldsRZ::*f : {&FieldsRZ::ez, &FieldsRZ::er, &FieldsRZ::ep, &FieldsRZ::bz, &FieldsRZ::br, &FieldsRZ::bp})
        cmp(((*a.rz).*f).host(), ((*r.rz).*f).host());
    for (size_t s = 0; s < a.species.size(); ++s)
        for (int sl = 0; sl < Species::NC; ++sl) cmp(a.species[s].get(sl), r.species[s].get(sl));
    bool refused = false;
    try {
        Simulation o(grid_deck(grid("1.1"), rest), comm);
        o.init(chk);
    } catch (const std::exception&) { refused = true; }
#ifdef _OPENMP
    omp_set_num_threads(nthr);
#endif
    check("rz stretched grid: output carries the node radii, restart = uninterrupted run, another grid refused",
          nodes && same && (!RIP_GPU ? worst == 0 : worst < 1e-9) && refused,
          fmt("checkpoint at step %.0f", double(nchk)) + fmt(", max relative difference %.1e", worst) +
              (nodes ? "" : ", node radii WRONG") + (refused ? "" : ", another grid ACCEPTED"));
}
#endif
// R24. Radiation of the current filter's change (as S6 of the Cart3D filter, decision 69): a current, fed into an empty
//      periodic-z grid with the wall for one field step, then 40 steps without current; B^2 / (E^2 + B^2) of the fields
//      (plain sums over the components and slots). Kinds: the raw random J; dJ = S J - J of one pass of the local filter
//      (decision 54) and of the projection (decision 70); the plain gradient (dz, G_m) psi and (dz, mu_z^2 G_m) psi of a
//      random psi on the rho nodes (a gradient moves charge only). RIP-CC / iRIP: raw 0.34 / 0.33, local dJ 0.17 / 0.15,
//      plain gradient 1.9e-2, mu_z^2 gradient 5e-32 / 7e-32.
static void rz_filter_radiation(Comm& comm) {
    const int nz = 32, nr = 16, M = 3;
    const char* name[5] = {"raw random J", "dJ of the local filter", "plain gradient (dz, G_m) psi", "gradient (dz, mu_z^2 G_m) psi",
                           "dJ of the projection filter"};
    double r[2][5] = {};
    for (int sv = 0; sv < 2; ++sv)
        for (int kind = 0; kind < 5; ++kind) {
            const double dz = 0.2, dr = 0.6;
            Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, sv == 0 ? Solver::RipCC : Solver::IRip);
            FieldsRZ f(d, M);
            const RadialGrid& g = f.grid();
            std::mt19937_64 rng(17);
            std::uniform_real_distribution<double> u(-1, 1);
            Field* J[3] = {&f.jz, &f.jr, &f.jp};
            auto slot = [](int m, int c) { return m == 0 ? 0 : 2 * m - 1 + c; };
            if (kind <= 1 || kind == 4) {
                for (Field* x : J) {
                    std::vector<double> h = x->host();
                    for (int k = 0; k < 2 * M - 1; ++k)
                        for (int j = 0; j < nr; ++j)
                            for (int i = 0; i < nz; ++i) h[x->idx(i, j, k)] = (k >= 1 && j == 0) ? 0.0 : u(rng);
                    x->upload(h);
                }
                if (kind == 1 || kind == 4) {
                    const std::vector<double> b[3] = {f.jz.host(), f.jr.host(), f.jp.host()};
                    rz_filter_currents(f, comm, 1, kind == 1);
                    for (int a = 0; a < 3; ++a) {
                        std::vector<double> v = J[a]->host();
                        for (size_t q = 0; q < v.size(); ++q) v[q] -= b[a][q];
                        J[a]->upload(v);
                    }
                }
            } else {
                // psi_m on the rho nodes (i + 1/2, r_j), 0 beyond the wall and (m >= 1) on the axis
                std::vector<double> psi(size_t(nz) * (nr + 1) * (2 * M - 1), 0.0);
                auto P = [&](int i, int j, int k) -> double& { return psi[(size_t(k) * (nr + 1) + j) * nz + size_t((i + nz) % nz)]; };
                for (int k = 0; k < 2 * M - 1; ++k)
                    for (int j = 0; j < nr; ++j)
                        for (int i = 0; i < nz; ++i) P(i, j, k) = (k >= 1 && j == 0) ? 0.0 : u(rng);
                std::vector<double> hz = f.jz.host(), hr = f.jr.host(), hp = f.jp.host();
                const bool mu = kind == 3;
                for (int m = 0; m < M; ++m)
                    for (int c = 0; c < (m ? 2 : 1); ++c) {
                        const int k = slot(m, c), kc = slot(m, 1 - c);
                        for (int j = 0; j < nr; ++j)
                            for (int i = 0; i < nz; ++i) {
                                hz[f.jz.idx(i, j, k)] = (P(i, j, k) - P(i - 1, j, k)) / dz;   // J_z at (i, r_j)
                                auto gr = [&](int ii) { return (P(ii, j + 1, k) - P(ii, j, k)) / g.dn[size_t(j)]; };
                                // (i m / r) psi: Re <- -m/r Im, Im <- m/r Re
                                auto gp = [&](int ii) { return j == 0 ? 0.0 : (c == 0 ? -1.0 : 1.0) * m / g.r[size_t(j)] * P(ii, j, kc); };
                                hr[f.jr.idx(i, j, k)] = mu ? 0.25 * (gr(i - 1) + 2 * gr(i) + gr(i + 1)) : gr(i);
                                hp[f.jp.idx(i, j, k)] = mu ? 0.25 * (gp(i - 1) + 2 * gp(i) + gp(i + 1)) : gp(i);
                            }
                    }
                f.jz.upload(hz); f.jr.upload(hr); f.jp.upload(hp);
            }
            comm.exchange(d, {&f.jz, &f.jr, &f.jp});
            f.step(comm);
            f.zero_currents();
            for (int n = 0; n < 40; ++n) { f.zero_currents(); f.step(comm); }
            auto sq = [&](const Field& x) {
                double v = 0;
                const std::vector<double> h = x.host();
                for (int k = 0; k < 2 * M - 1; ++k)
                    for (int j = 0; j < nr; ++j)
                        for (int i = 0; i < nz; ++i) v += h[x.idx(i, j, k)] * h[x.idx(i, j, k)];
                return v;
            };
            const double we = sq(f.ez) + sq(f.er) + sq(f.ep), wb = sq(f.bz) + sq(f.br) + sq(f.bp);
            r[sv][kind] = wb / (we + wb);
            std::printf("  R24 %s, %s: after 40 steps B^2 / (E^2 + B^2) = %.3e\n", sv == 0 ? "rip_cc" : "irip", name[kind], r[sv][kind]);
        }
    // the gradient of the RZ Gauss law (dz Ez + mu_z D_m E_perp = rho) is free of curl for both solvers, the plain one is
    // not; the local filter changes J_z alone in its z step and radiates, the projection's change is that gradient
    check("rz projection filter: its change radiates nothing (B = 0), the local filter's and a plain gradient do",
          std::max({r[0][3], r[1][3], r[0][4], r[1][4]}) < 1e-24 && std::min(r[0][2], r[1][2]) > 1e-3 && std::min(r[0][1], r[1][1]) > 1e-2,
          fmt("projection %.1e, mu_z^2 gradient %.1e", std::max(r[0][4], r[1][4]), std::max(r[0][3], r[1][3])) +
              fmt(", local %.1e, plain gradient %.1e", std::min(r[0][1], r[1][1]), std::min(r[0][2], r[1][2])));
}

// R25. The projection filter keeps the continuity of the filtered charge: D J' = T D J (T of rz_filter_rho) for a random J
//      in modes 0..2, periodic and open z, uniform and stretched radial grid, one and two passes (open z: on the Gauss cells
//      [0, N - 1)); D of the RZ deposit: (J_z,i+1 - J_z,i) / dz + D_m J_perp.
//      Also the binomial filter (decision 91) and the local one: the same identity; and the binomial on J_z = cos(k z) of m = 0,
//      uniform in r (uniform grid): J_z' = cos^2(k dz / 2) J_z per pass on every node (the radial T leaves a uniform column
//      alone), the transfer function of the binomial (1, 2, 1) / 4. Kinds 3, 4: the compensated binomial (filter_compensate),
//      and compensated along z only (filter_axes = x; bench/nci/mechanism): the same identity, transfer T^N ((N + 1) - N T).
static void rz_filter_projection_gauss(Comm& comm) {
    double err = 0, errb = 0, terr = 0;
    for (int kind = 0; kind < 7; ++kind)    // projection, binomial, local, compensated, compensated z only, flat L = 4 / 6 z only
    for (int open = 0; open < 2; ++open)
        for (int st = 0; st < 2; ++st)
            for (int passes = 1; passes <= 2; ++passes) {
                const int nz = 24, nr = 14, M = 3;
                const double dz = 0.3, dr = 0.4;
                Domain d = rz_domain(comm, nz, nr, M, dz, dr, open == 0, Solver::IRip);
                d.filt_comp = kind == 3 || kind == 4; d.filt_xonly = kind >= 4;
                set_flat_filter(d, passes, kind == 5 ? 4 : kind == 6 ? 6 : 0);
                FieldsRZ f(d, M, st ? rz_radial_grid(nr, dr, 6 * dr, 1.08) : rz_radial_grid(nr, dr));
                const RadialGrid& g = f.grid();
                std::mt19937_64 rng(31 + 4 * open + 2 * st + passes + 64 * kind);
                std::uniform_real_distribution<double> u(-1, 1);
                Field* J[3] = {&f.jz, &f.jr, &f.jp};
                for (Field* x : J) {
                    std::vector<double> h = x->host();
                    for (int k = 0; k < 2 * M - 1; ++k)
                        for (int j = 0; j < nr; ++j)
                            for (int i = 0; i < nz; ++i) h[x->idx(i, j, k)] = (k >= 1 && j == 0) ? 0.0 : u(rng);
                    x->upload(h);
                }
                auto D = [&](Field& out) {
                    comm.exchange(d, {&f.jz, &f.jr, &f.jp});
                    const std::vector<double> hz = f.jz.host(), hr = f.jr.host(), hp = f.jp.host();
                    std::vector<double> o = out.host();
                    for (int m = 0; m < M; ++m)
                        for (int c = 0; c < (m ? 2 : 1); ++c) {
                            const int k = m == 0 ? 0 : 2 * m - 1 + c, kc = m == 0 ? 0 : 2 * m - c;
                            for (int j = 0; j < nr; ++j)
                                for (int i = 0; i < nz; ++i) {
                                    double v = 0;
                                    const bool gauss = open == 0 || d.off[0] + i < d.N[0] - 1;
                                    if (gauss && !(m >= 1 && j == 0)) {
                                        v = (hz[f.jz.idx(i + 1, j, k)] - hz[f.jz.idx(i, j, k)]) / dz;
                                        if (j == 0) v += g.rh[0] * hr[f.jr.idx(i, 0, k)] / g.A[0];
                                        else {
                                            v += (g.rh[size_t(j)] * hr[f.jr.idx(i, j, k)] - g.rh[size_t(j - 1)] * hr[f.jr.idx(i, j - 1, k)]) / g.A[size_t(j)];
                                            v += (c == 0 ? -1.0 : 1.0) * m / g.r[size_t(j)] * hp[f.jp.idx(i, j, kc)];
                                        }
                                    }
                                    o[out.idx(i, j, k)] = v;
                                }
                        }
                    out.upload(o);
                };
                Field w0(d), w1(d);
                D(w0);
                rz_filter_rho(f, comm, w0, passes);                  // T D J
                rz_filter_currents(f, comm, passes, kind == 2, kind == 1 || kind >= 3);
                D(w1);                                               // D J'
                const std::vector<double> a = w0.host(), b = w1.host();
                double e = 0, sc = 0;
                for (int k = 0; k < 2 * M - 1; ++k)
                    for (int j = 0; j < nr; ++j)
                        for (int i = 0; i < nz; ++i) {
                            e = std::max(e, std::abs(a[w0.idx(i, j, k)] - b[w0.idx(i, j, k)]));
                            sc = std::max(sc, std::abs(a[w0.idx(i, j, k)]));
                        }
                (kind == 0 ? err : errb) = std::max(kind == 0 ? err : errb, e / sc);
                if ((kind != 1 && kind < 3) || open || st) continue;
                // the binomial's transfer function on J_z = cos(k z), m = 0, uniform in r (k = 2 pi 5 / nz)
                for (Field* x : J) x->fill(0);
                const double kq = 2 * M_PI * 5 / nz;
                std::vector<double> h = f.jz.host();
                for (int j = 0; j < nr; ++j)
                    for (int i = 0; i < nz; ++i) h[f.jz.idx(i, j, 0)] = std::cos(kq * (d.off[0] + i));
                f.jz.upload(h);
                rz_filter_currents(f, comm, passes, false, true);
                h = f.jz.host();
                const double T = std::cos(kq / 2) * std::cos(kq / 2);
                double P = 1;
                if (kind >= 5) {                                     // Herrmann: sum_{j<L} binom(N-1+j, j) (1 - T)^j
                    P = 0; double b = 1;
                    for (int j = 0; j < (kind == 5 ? 4 : 6); ++j) { P += b * std::pow(1 - T, j); b = b * (passes + j) / (j + 1); }
                }
                const double t = std::pow(T, passes) * (kind == 3 || kind == 4 ? (passes + 1) - passes * T : 1.0) * P;
                for (int j = 0; j < nr - 1 - passes; ++j)               // the last nodes feel the wall (J = 0 beyond it)
                    for (int i = 0; i < nz; ++i) terr = std::max(terr, std::abs(h[f.jz.idx(i, j, 0)] - t * std::cos(kq * (d.off[0] + i))));
            }
    check("rz projection, binomial and local filter: D J' = T D J (periodic and open z, uniform and stretched r, 1 and 2 "
          "passes; binomial also compensated, z only, flat L = 4, 6); binomial: J_z of a uniform column x the transfer function",
          err < 1e-12 && errb < 1e-12 && terr < 1e-13, fmt("%.1e, binomial/local %.1e", err, errb) + fmt(", transfer %.1e", terr));
}

// R27. Warm plasma entering a moving window with the current filter (2 passes; decision 91): electrons u_th = 0.05,
//      2 per cell in z, ions 1 per cell in z (a non-neutral entering column), a ramp, a laser wake (R23's window case).
//      The filter's closure at the right edge changes at every shift; the entering plasma is now frozen for 3 + passes
//      cells and its charge kept as a background, so the Gauss law per mode stays at round-off: iRIP with the projection,
//      RIP-CC with the binomial filter. Before: 1e-7 .. 1e-6 of the field (B7 hot: 3e-4 of max rho in every mode).
static void rz_window_warm_filter(Comm& comm) {
    double gw[2] = {0, 0}, rmax = 0;
    for (int run = 0; run < 2; ++run) {
        const std::string grid = " modes = 2\n nz = 256\n dz = 0.025 um\n nr = 16\n dr = 0.05 um\n boundary_z = open\n t_end = 45 fs\n";
        auto plasma = [](int ppcz, double uth) {
            return std::string(" density = 0.02\n rmax = 0.5 um\n ppc_z = ") + std::to_string(ppcz) +
                   "\n ppc_r = 2\n ppc_theta = 6\n zmin = 6.4 um\n density_function = z < 7.4 um ? 0.3 + 0.7 * (z - 6.4 um) / (1 um) : 1\n" +
                   (uth > 0 ? " uth_z = 0.05\n uth_x = 0.05\n uth_y = 0.05\n" : "");
        };
        const std::string rest = std::string("numerics {\n solver = ") + (run ? "rip_cc" : "irip") + "\n filter_passes = 2\n filter = " +
            (run ? "binomial" : "projection") + "\n}\n" +
            "laser {\n a0 = 1\n polarization = x\n duration = 5 fs\n waist = 0.4 um\n}\nwindow {\n start = 20 fs\n}\n" +
            "species e {\n charge = -1\n mass = 1\n" + plasma(2, 0.05) + "}\nspecies p {\n charge = 1\n mass = 1836\n" + plasma(1, 0) + "}\n";
        Simulation sim(grid_deck(grid, rest), comm);
        sim.init();
        double w, g, r, b;
        while (!sim.finished()) {
            sim.step();
            sim.rz_diagnostics(w, g, r, b);
            gw[run] = std::max(gw[run], g);
            rmax = std::max(rmax, r);
        }
        if (sim.species[0].size() == 0 || sim.dom.wx < 200) gw[run] = 1;
    }
    check("rz window, warm non-neutral plasma entering with the filter (2 passes): Gauss law per mode at round-off",
          gw[0] < 1e-13 && gw[1] < 1e-13 && rmax > 1e-6,
          fmt("iRIP + projection %.1e, RIP-CC + binomial %.1e", gw[0], gw[1]) + fmt(" (max|rho| %.1e)", rmax));
}

}  // namespace

void test_rz_grid(Comm& comm) {
    if (comm.size > 1) return;
    rz_grid_random(comm);
    rz_grid_guided(comm);
    rz_grid_reflection(comm);
    rz_grid_laser(comm);
    rz_grid_plasma(comm);
    rz_grid_loading(comm);
    rz_grid_wake(comm);
    rz_static_background(comm);
    rz_density_conserving(comm);
    rz_filter(comm);
    rz_filter_radiation(comm);
    rz_filter_projection_gauss(comm);
    rz_window_warm_filter(comm);
#ifdef RIP_HDF5
    rz_grid_output(comm);
#endif
}
