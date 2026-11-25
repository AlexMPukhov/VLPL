// Dielectric blocks in the RIP-CC RZ step (rz_dielectric.hpp, docs/specs/rz_dielectric.md, decision 100).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include "common.hpp"
#include "rz.hpp"
#include "sim.hpp"

namespace {
const double j01 = 2.404825557695773;

std::vector<Field*> all(FieldsRZ& f) { return {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp}; }

DielectricConfig block(double rin, double rout, double z0, double z1, double eps) {
    DielectricConfig c;
    c.r_in = rin; c.r_out = rout; c.z_start = z0; c.z_end = z1; c.eps = eps;
    return c;
}

// m = 0 Ez = J0(kp r) (cos(kz z) + 0.3 cos(2 kz z)), m = 1 Re Bz, Im Er: a mixture (z = I dz, I the global column)
void init_mix(FieldsRZ& f, double R, double kz) {
    const Domain& d = f.dom();
    const RadialGrid& g = f.grid();
    H ez(f.ez), bz(f.bz), er(f.er);
    for (int i = 0; i < d.n[0]; ++i) {
        const double z = (d.off[0] + i) * d.h[0];
        for (int j = 0; j < d.n[1]; ++j) {
            ez(i, j, 0) = std::cyl_bessel_j(0, j01 / R * g.r[size_t(j)]) * (std::cos(kz * z) + 0.3 * std::cos(2 * kz * z));
            bz(i, j, 1) = std::cyl_bessel_j(1, 1.84118 / R * g.rh[size_t(j)]) * (std::sin(kz * z) + 0.5 * std::sin(3 * kz * z));
            er(i, j, 2) = 0.4 * std::sin(M_PI * g.rh[size_t(j)] / R) * std::cos(2 * kz * z + 0.3);
        }
    }
    ez.put(f.ez); bz.put(f.bz); er.put(f.er);
}

double max_abs_diff(const Field& a, const Field& b, double* scale) {
    const std::vector<double> x = a.host(), y = b.host();
    double m = 0, s = 0;
    for (size_t q = 0; q < x.size(); ++q) { m = std::max(m, std::abs(x[q] - y[q])); s = std::max(s, std::abs(y[q])); }
    if (scale) *scale = std::max(*scale, s);
    return m;
}

// D1. Fields only, periodic z, modes 2, two blocks with partial cells (eps 1.7 and 3, one crossing the periodic seam):
//     the Gauss law div D is unchanged to round-off after 400 steps (D = (1 + chi) E at the staggered points), the energy
//     W = (E.D + B^2) / 2 stays bounded (as without dielectric), and the fields stay finite (chi up to 2, no limit).
void diel_gauss_energy(Comm& comm) {
    const int nz = 40, nr = 22, M = 2;
    const double dr = 0.1, dz = 0.07, R = nr * dr, kz = 2 * M_PI / (nz * dz);
    Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, Solver::RipCC);
    FieldsRZ f(d, M), v(d, M);
    f.diel = std::make_shared<DielectricRZ>(std::vector<DielectricConfig>{block(0.37, 1.13, 0.33, 1.47, 1.7),
                                                                         block(0.0, 2.6, 2.31, 3.17, 3.0)}, d, f.grid());
    for (FieldsRZ* x : {&f, &v}) { init_mix(*x, R, kz); comm.exchange(d, all(*x)); }
    Field g0(d), g1(d);
    f.div_e(g0);
    const double w0 = f.energy(), v0 = v.energy();
    double wmin = 1e300, wmax = 0, vmin = 1e300, vmax = 0;
    for (int n = 0; n < 400; ++n) {
        f.zero_currents(); f.step(comm); v.zero_currents(); v.step(comm);
        double w = f.energy(), u = v.energy();
        comm.allreduce_sum(&w, 1); comm.allreduce_sum(&u, 1);
        wmin = std::min(wmin, w); wmax = std::max(wmax, w); vmin = std::min(vmin, u); vmax = std::max(vmax, u);
    }
    double W0 = w0, V0 = v0;
    comm.allreduce_sum(&W0, 1); comm.allreduce_sum(&V0, 1);
    f.div_e(g1);
    double sc = 0;
    const double dg = comm.allreduce_max(max_abs_diff(g1, g0, &sc));
    sc = comm.allreduce_max(sc);
    const double spread = (wmax - wmin) / W0, vspread = (vmax - vmin) / V0;
    check("rz dielectric D1: div D unchanged (400 steps, modes 2, eps 1.7 / 3 with partial cells); energy (E.D + B^2)/2 bounded",
          dg < 1e-12 * sc && spread < 0.05 && std::isfinite(wmax),
          fmt("max change of div D %.1e of max", dg / sc) + fmt(", energy spread %.2e", spread) +
              fmt(" (vacuum %.2e)", vspread));
}

// frequency w dt from the increments of a sampled history (as test_rz_metal.cpp)
double recurrence(const std::vector<std::vector<double>>& hist) {
    double num = 0, den = 0;
    for (size_t n = 1; n + 2 < hist.size(); ++n)
        for (size_t q = 0; q < hist[n].size(); ++q) {
            const double dm = hist[n][q] - hist[n - 1][q], d0 = hist[n + 1][q] - hist[n][q], dp = hist[n + 2][q] - hist[n + 1][q];
            num += (dp + dm) * d0; den += 2 * d0 * d0;
        }
    return std::acos(num / den);
}

// TM01 standing wave in a tube of radius R filled with eps (the block reaches beyond the wall): w dt
double tm01(Comm& comm, double eps, int nz, int nr, double dr, double dz, int lz) {
    const double R = nr * dr, kz = 2 * M_PI * lz / (nz * dz);
    Domain d = rz_domain(comm, nz, nr, 1, dz, dr, true, Solver::RipCC);
    FieldsRZ f(d, 1);
    if (eps > 1) f.diel = std::make_shared<DielectricRZ>(std::vector<DielectricConfig>{block(0.0, 2 * R, -1.0, nz * dz + 1, eps)}, d, f.grid());
    {
        H ez(f.ez), er(f.er);
        const RadialGrid& g = f.grid();
        const double kp = j01 / R;
        for (int i = 0; i < d.n[0]; ++i) {
            const double z = (d.off[0] + i) * dz;
            for (int j = 0; j < nr; ++j) {
                ez(i, j, 0) = std::cyl_bessel_j(0, kp * g.r[size_t(j)]) * std::cos(kz * z);
                er(i, j, 0) = kz / kp * std::cyl_bessel_j(1, kp * g.rh[size_t(j)]) * std::sin(kz * z);   // div E = 0
            }
        }
        ez.put(f.ez); er.put(f.er);
    }
    comm.exchange(d, all(f));
    std::vector<std::vector<double>> hist;
    const int steps = int(4 * 2 * M_PI * std::sqrt(eps) / (std::sqrt(kz * kz + std::pow(j01 / R, 2)) * dz)) + 10;
    for (int n = 0; n < steps; ++n) { hist.push_back(f.ez.host()); f.zero_currents(); f.step(comm); }
    return recurrence(hist) / dz;
}

// D2 (handover test 1). TM01 in a dielectric-filled tube, periodic z: e = w sqrt(eps) / w(vacuum, same grid) - 1 (0 in the
//    continuum: w = c sqrt(kz^2 + (j01/R)^2) / sqrt(eps)) against the dispersion relation of the scheme, the eigenvalues of
//    the von Neumann amplification matrix of the RIP-CC step with the Crank-Nicolson polarization current
//    (research/rz_dielectric/vonneumann.py, tm01_prediction(); K = (j01/R) dz): the second-order error of the scheme.
void diel_tm01(Comm& comm) {
    const int nz = 64, nr = 32;
    const double dr = 0.05, dz = 0.04;
    const double pred[2][3] = {{5.93e-05, 3.63e-04, 4.90e-04}, {1.15e-03, 7.09e-03, 9.62e-03}};
    const double eps[3] = {1.1, 2.25, 4.0};
    std::string info;
    double worst = 0;
    int a = 0;
    for (int lz : {1, 4}) {
        const double w1 = tm01(comm, 1.0, nz, nr, dr, dz, lz);
        const double kz = 2 * M_PI * lz / (nz * dz), wc = std::sqrt(kz * kz + std::pow(j01 / (nr * dr), 2));
        info += fmt("kz dz %.3f: vacuum w/w_c - 1 = %.1e;", kz * dz, w1 / wc - 1);
        for (int b = 0; b < 3; ++b) {
            const double w = tm01(comm, eps[b], nz, nr, dr, dz, lz), e = w * std::sqrt(eps[b]) / w1 - 1;
            worst = std::max(worst, std::abs(e - pred[a][b]) / pred[a][b]);
            info += fmt(" eps %.2f: %.3e", eps[b], e) + fmt(" (scheme %.3e)", pred[a][b]);
        }
        info += ";  ";
        ++a;
    }
    check("rz dielectric D2: TM01 in a filled tube, w sqrt(eps) / w(vacuum) - 1 = the scheme's dispersion (eps 1.1, 2.25, 4; kz dz 0.1, 0.4)",
          worst < 0.02, info + fmt("worst relative deviation %.1e", worst));
}

// D3. A block with eps = 1 changes nothing: fields bit for bit equal to the run without the block (300 steps, modes 2)
void diel_identity(Comm& comm) {
    const int nz = 24, nr = 16, M = 2;
    const double dr = 0.1, dz = 0.07, R = nr * dr, kz = 2 * M_PI / (nz * dz);
    Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, Solver::RipCC);
    FieldsRZ a(d, M), b(d, M);
    b.diel = std::make_shared<DielectricRZ>(std::vector<DielectricConfig>{block(0.3, 1.2, 0.2, 1.1, 1.0)}, d, b.grid());
    for (FieldsRZ* x : {&a, &b}) { init_mix(*x, R, kz); comm.exchange(d, all(*x)); }
    for (int n = 0; n < 300; ++n) { a.zero_currents(); a.step(comm); b.zero_currents(); b.step(comm); }
    double diff = 0;
    Field* fa[6] = {&a.ez, &a.er, &a.ep, &a.bz, &a.br, &a.bp};
    Field* fb[6] = {&b.ez, &b.er, &b.ep, &b.bz, &b.br, &b.bp};
    for (int c = 0; c < 6; ++c) diff = std::max(diff, max_abs_diff(*fa[c], *fb[c], nullptr));
    diff = comm.allreduce_max(diff);
    check("rz dielectric D3: a block with eps = 1 leaves the fields bit for bit unchanged (300 steps)", diff == 0, fmt("max difference %.1e", diff));
}

// D4. Deck path: hot plasma in a periodic tube with dielectric blocks (eps 1.5 over the ring region, a collar outside, eps 2.2,
//     and a rod, eps 1.3; modes 2, shape 1 and 2), the Gauss law div D = rho off the metal surface and div B at round-off, with
//     (a) rings (the gaps filled), (b) no metal, (c) the block metal: four ridges depending on phi with z faces (the lines on
//     the angles).
void diel_plasma(Comm& comm, int variant) {
    const char* path = "rz_diel_test_tmp.deck";
    const char* metal[3] = {"rings {\n r_in = 0.48 um\n r_out = 0.9 um\n period = 1 um\n gap = 0.5 um\n z_start = 0.3 um\n z_end = 1.9 um\n}\n",
                            "",
                            "metal {\n function = r > 0.48 um && r < 0.9 um && abs(phi - pi/2 * floor((phi + pi/4) / (pi/2))) < 0.5 && "
                            "(z - 1 um * floor(z / (1 um))) < 0.5 um\n}\n"};
    const char* name[3] = {"rings", "no metal", "metal ridges (phi)"};
    if (comm.rank == 0) {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n modes = 2\n lambda0 = 1 um\n nz = 50\n nr = 24\n dz = 0.04 um\n dr = 0.06 um\n"
             " t_end = 30 fs\n}\nnumerics {\n solver = rip_cc\n pusher = boris\n}\n"
          << metal[variant] <<
             "dielectric gaps {\n r_in = 0.48 um\n r_out = 0.9 um\n z_start = 0.3 um\n z_end = 1.9 um\n eps = 1.5\n}\n"
             "dielectric collar {\n r_in = 0.9 um\n r_out = 1.05 um\n z_start = 0.1 um\n z_end = 1.7 um\n eps = 2.2\n}\n"
             "dielectric rod {\n r_in = 0.1 um\n r_out = 0.31 um\n z_start = 1.13 um\n z_end = 1.62 um\n eps = 1.3\n}\n"
             "species e {\n charge = -1\n mass = 1\n density = 0.02\n rmax = 1.3 um\n ppc_r = 2\n ppc_theta = 6\n"
             " uth_z = 0.15\n uth_x = 0.15\n uth_y = 0.15\n uz = 0.1\n}\n"
             "species i {\n charge = 1\n mass = 100\n density = 0.02\n rmax = 1.3 um\n ppc_r = 2\n ppc_theta = 6\n"
             " uth_z = 0.01\n uth_x = 0.01\n uth_y = 0.01\n}\n"
             "output {\n dir = rz_diel_test_out\n every = 1000\n}\n";
    }
    comm.barrier();
    Deck dk = Deck::read(path);
    for (int order = 1; order <= 2; ++order) {
        dk.override_from(std::string("numerics.shape=") + std::to_string(order));
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        double wf0, g0, r0, b0;
        sim.rz_diagnostics(wf0, g0, r0, b0);
        while (!sim.finished()) sim.step();
        double wf, g, rmax, db;
        sim.rz_diagnostics(wf, g, rmax, db);
        check(std::string("rz dielectric D4: plasma, ") + name[variant] + ", dielectric over the rings region, collar, rod (shape " +
                  std::to_string(order) + "): div D - rho and div B off the metal at round-off",
              g < 1e-12 * std::max(rmax, 1e-30) + 1e-14 && db < 1e-12 && std::isfinite(wf),
              fmt("Gauss %.1e", g) + fmt(" (max rho %.1e)", rmax) + fmt(", div B %.1e", db) + fmt(", W %.3e", wf));
    }
    if (variant == 0) {
        bool rejected = false;
        dk.override_from("numerics.solver=irip");
        try { Config::from_deck(dk); } catch (const std::exception& e) { rejected = std::string(e.what()).find("rip_cc") != std::string::npos; }
        check("rz dielectric: dielectric blocks with iRIP rejected", rejected, "");
    }
    comm.barrier();
    if (comm.rank == 0) std::remove(path);
}
// D5. Moving window (a small DWA: metal wall 20 um, eps = 3.8 lining from 12 um, a rigid bunch at gamma = 1000 with its
//     self-field entering the lining): the Gauss law div D = rho at round-off over 3000 steps (the entering column and the
//     front layer keep it, decision 100), and the field energy in the window settles (no growth at the moving front edge:
//     without the front layer the grid-scale branch of eps = 3.8 grows there).
void diel_window(Comm& comm) {
    const char* path = "rz_diel_win_tmp.deck";
    if (comm.rank == 0) {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n modes = 1\n lambda0 = 100 um\n nz = 600\n nr = 40\n dz = 0.4 um\n dr = 0.5 um\n"
             " boundary_z = open\n t_end = 4 ps\n}\nnumerics {\n solver = rip_cc\n pusher = boris\n}\n"
             "window {\n start = 0 fs\n velocity = 1\n}\n"
             "dielectric layer {\n r_in = 12 um\n r_out = 30 um\n z_start = 0.2 mm\n z_end = 10 mm\n eps = 3.8\n}\n"
             "species bunch {\n charge = 1\n mass = 18361526.7\n density = 1\n density_function = gauss(z - 0.12 mm, 8 um) * gauss(r, 3 um)\n"
             " zmin = 0.09 mm\n zmax = 0.15 mm\n rmax = 10 um\n uz = 999.9995\n self_field = true\n ppc_z = 1\n ppc_r = 2\n ppc_theta = 1\n}\n"
             "output {\n dir = rz_diel_win_out\n every = 100000\n}\n";
    }
    comm.barrier();
    Deck dk = Deck::read(path);
    Config c = Config::from_deck(dk);
    Simulation sim(c, comm);
    sim.init();
    double gmax = 0, rmax = 0, w1 = 0, w2 = 0;
    long n = 0;
    while (!sim.finished()) {
        sim.step();
        ++n;
        if (n % 250 == 0) {
            double wf, g, rm, db;
            sim.rz_diagnostics(wf, g, rm, db);
            comm.allreduce_sum(&wf, 1);
            gmax = std::max(gmax, g); rmax = std::max(rmax, rm);
            if (n == 1500) w1 = wf;
            w2 = wf;
        }
    }
    check("rz dielectric D5: moving window, eps 3.8 lining, bunch: div D - rho at round-off, no growth at the front edge (3000 steps)",
          gmax < 1e-11 * rmax && std::isfinite(w2) && w2 < 1.5 * w1 && sim.front_layer_warnings == 0,
          fmt("max Gauss %.1e of max rho", gmax / rmax) + fmt(", W(3000) / W(1500) = %.4f", w2 / w1) +
              fmt(", front-layer warnings %.0f", double(sim.front_layer_warnings)));
    comm.barrier();
    if (comm.rank == 0) std::remove(path);
}
// D6. The warning of decision 100 (Alexander, 2026-11-13): with dielectric blocks in a moving window, charged particles in the
//     32-cell front absorbing layer are reported. (a) a bunch loaded inside the layer: warned at the start; (b) a window at
//     v = 0.9 c that a bunch at c catches up: warned once when it enters (then rate-limited, at most every 2000 steps).
void diel_window_warning(Comm& comm) {
    const char* path = "rz_diel_warn_tmp.deck";
    auto run = [&](const std::string& vel, double z0, double t_ps) {   // z0 in um
        if (comm.rank == 0) {
            std::ofstream f(path);
            f << "grid {\n geometry = rz\n modes = 1\n lambda0 = 100 um\n nz = 600\n nr = 40\n dz = 0.4 um\n dr = 0.5 um\n"
                 " boundary_z = open\n t_end = " << t_ps << " ps\n}\nnumerics {\n solver = rip_cc\n pusher = boris\n}\n"
                 "window {\n start = 0 fs\n velocity = " << vel << "\n}\n"
                 "dielectric layer {\n r_in = 12 um\n r_out = 30 um\n z_start = 0.02 mm\n z_end = 10 mm\n eps = 2.1\n}\n"
                 "species bunch {\n charge = 1\n mass = 18361526.7\n density = 1\n density_function = gauss(z - " << z0 << " um, 2 um) * gauss(r, 3 um)\n"
                 " zmin = " << z0 - 4 << " um\n zmax = " << z0 + 4 << " um\n rmax = 10 um\n uz = 999.9995\n ppc_z = 1\n ppc_r = 2\n ppc_theta = 1\n}\n"
                 "output {\n dir = rz_diel_warn_out\n every = 100000\n}\n";
        }
        comm.barrier();
        Deck dk = Deck::read(path);
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        const long w0 = sim.front_layer_warnings;
        while (!sim.finished()) sim.step();
        comm.barrier();
        if (comm.rank == 0) std::remove(path);
        return std::make_pair(w0, sim.front_layer_warnings);
    };
    const auto a = run("1", 234, 0.1);           // the layer: z > 0.24 mm - 32 x 0.4 um = 0.2272 mm
    const auto b = run("0.9", 210, 1.0);          // 0.214 mm: 33 cells behind the layer, 0.1 cell per step closer
    check("rz dielectric D6: charged particles in the front layer of a moving window are warned (start; entry, rate-limited)",
          a.first >= 1 && b.first == 0 && b.second == 1,
          fmt("bunch in the layer: %.0f warning(s) at the start", double(a.first)) +
              fmt("; window at 0.9 c: %.0f at the start, ", double(b.first)) + fmt("%.0f after 750 steps", double(b.second)));
}

// eps as a deck expression of (x, y, z, r, phi) (code units, no unit names); phi: the caller sets c.phi
DielectricConfig fblock(const std::string& text) {
    DielectricConfig c;
    c.fn = Expr(text, {"x", "y", "z", "r", "phi"}, [](const std::string&, double&) { return false; });
    c.r_in = 0; c.r_out = 1e300; c.z_start = -1e300; c.z_end = 1e300;
    return c;
}

// D7. eps from a function that is the rectangles of D1 (indicator functions) = the rectangle blocks: chi of every point and
//     the fields after 300 steps (modes 2, periodic, one block across the seam) equal to round-off. The function is sampled
//     8 x 8 everywhere, the rectangle exactly outside its cut cells and at the same sub-points inside them.
void diel_function_rect(Comm& comm) {
    const int nz = 40, nr = 22, M = 2;
    const double dr = 0.1, dz = 0.07, R = nr * dr, kz = 2 * M_PI / (nz * dz), Lz = nz * dz;
    Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, Solver::RipCC);
    FieldsRZ a(d, M), b(d, M);
    a.diel = std::make_shared<DielectricRZ>(std::vector<DielectricConfig>{block(0.37, 1.13, 0.33, 1.47, 1.7),
                                                                         block(0.0, 2.6, 2.31, 3.17, 3.0)}, d, a.grid());
    // the second block crosses the seam z = Lz: its periodic image [2.31 - Lz, 3.17 - Lz) inside [0, Lz)
    char t[400];
    std::snprintf(t, sizeof t, "1 + 0.7 * (r >= 0.37 && r < 1.13 && z >= 0.33 && z < 1.47) + 2 * (r < 2.6 && (z >= 2.31 || z < %.17g))", 3.17 - Lz);
    b.diel = std::make_shared<DielectricRZ>(std::vector<DielectricConfig>{fblock(t)}, d, b.grid());
    double dchi = 0;
    for (int c = 0; c < 3; ++c) {
        const std::vector<double> x = a.diel->chi_host(c), y = b.diel->chi_host(c);
        for (size_t q = 0; q < x.size(); ++q) dchi = std::max(dchi, std::abs(x[q] - y[q]));
    }
    for (FieldsRZ* f : {&a, &b}) { init_mix(*f, R, kz); comm.exchange(d, all(*f)); }
    for (int n = 0; n < 300; ++n) { a.zero_currents(); a.step(comm); b.zero_currents(); b.step(comm); }
    double diff = 0, sc = 0;
    Field* fa[6] = {&a.ez, &a.er, &a.ep, &a.bz, &a.br, &a.bp};
    Field* fb[6] = {&b.ez, &b.er, &b.ep, &b.bz, &b.br, &b.bp};
    for (int c = 0; c < 6; ++c) diff = std::max(diff, max_abs_diff(*fa[c], *fb[c], &sc));
    diff = comm.allreduce_max(diff) / comm.allreduce_max(sc);
    dchi = comm.allreduce_max(dchi);
    check("rz dielectric D7: eps from a function (the indicator of the rectangles) = the rectangle blocks (chi, fields after 300 steps)",
          dchi < 1e-14 && diff < 1e-12, fmt("max chi difference %.1e", dchi) + fmt(", fields %.1e of max", diff));
}

// Frequency w dt of the dominant line of s_n: maximum of the Hann-windowed DTFT (scan, then golden section)
double peak(const std::vector<double>& s, double wlo, double whi) {
    const size_t N = s.size();
    auto amp = [&](double w) {
        double re = 0, im = 0;
        for (size_t n = 0; n < N; ++n) {
            const double h = s[n] * (0.5 - 0.5 * std::cos(2 * M_PI * n / (N - 1)));
            re += h * std::cos(w * double(n)); im -= h * std::sin(w * double(n));
        }
        return std::hypot(re, im);
    };
    double best = wlo, bv = -1;
    const int K = 2000;
    for (int q = 0; q <= K; ++q) {
        const double w = wlo + (whi - wlo) * q / K, v = amp(w);
        if (v > bv) { bv = v; best = w; }
    }
    double lo = best - (whi - wlo) / K, hi = best + (whi - wlo) / K;
    const double g = (std::sqrt(5.0) - 1) / 2;
    for (int it = 0; it < 80; ++it) {
        const double x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
        if (amp(x1) > amp(x2)) hi = x2; else lo = x1;
    }
    return 0.5 * (lo + hi);
}

// TM01 of a tube R = nr dr filled with the graded eps(r) = 1 + 1.5 (1 - (r/R)^2) (a function block), periodic z (k_z = 2 pi / Lz):
// w from the DTFT peak of E_z projected on its start
double grin_tm01(Comm& comm, int nz, int nr, double dr, double dz) {
    const double R = nr * dr, kz = 2 * M_PI / (nz * dz);
    Domain d = rz_domain(comm, nz, nr, 1, dz, dr, true, Solver::RipCC);
    FieldsRZ f(d, 1);
    char t[200];
    std::snprintf(t, sizeof t, "1 + 1.5 * (1 - (r / %.17g)^2)", R);
    f.diel = std::make_shared<DielectricRZ>(std::vector<DielectricConfig>{fblock(t)}, d, f.grid());
    {
        H ez(f.ez), er(f.er);
        const RadialGrid& g = f.grid();
        const double kp = j01 / R;
        for (int i = 0; i < d.n[0]; ++i) {
            const double z = (d.off[0] + i) * dz;
            for (int j = 0; j < nr; ++j) {
                ez(i, j, 0) = std::cyl_bessel_j(0, kp * g.r[size_t(j)]) * std::cos(kz * z);
                er(i, j, 0) = kz / kp * std::cyl_bessel_j(1, kp * g.rh[size_t(j)]) * std::sin(kz * z);
            }
        }
        ez.put(f.ez); er.put(f.er);
    }
    comm.exchange(d, all(f));
    const std::vector<double> h0 = f.ez.host();
    std::vector<double> s;
    const int steps = int(60 * 2 * M_PI / (2.0 * dz));
    for (int n = 0; n < steps; ++n) {
        const std::vector<double> x = f.ez.host();
        double v = 0;
        for (int i = 0; i < d.n[0]; ++i)
            for (int j = 0; j < nr; ++j) v += (j + 0.5) * x[f.ez.idx(i, j, 0)] * h0[f.ez.idx(i, j, 0)];
        comm.allreduce_sum(&v, 1);
        s.push_back(v);
        f.zero_currents(); f.step(comm);
    }
    return peak(s, 1.0 * dz, 4.0 * dz) / dz;
}

// D8. Graded-index tube (eps from a function of r): TM01 against the continuum (research/rz_dielectric/grin_tm01.py, the radial
//     ODE shot with RK4: w = 2.155441729447 at k_z = 2 pi / 2.56, R = 1.6): second-order convergence (dr, dz halved).
void diel_grin(Comm& comm) {
    const double wref = 2.155441729447;
    const double w1 = grin_tm01(comm, 64, 32, 0.05, 0.04), w2 = grin_tm01(comm, 128, 64, 0.025, 0.02);
    const double e1 = w1 / wref - 1, e2 = w2 / wref - 1;
    check("rz dielectric D8: graded eps(r) = 1 + 1.5 (1 - r^2/R^2) as a function: TM01 against the radial ODE, second order",
          std::abs(e2) < 5e-4 && std::abs(e1 / e2) > 3 && std::abs(e1 / e2) < 5.5,
          fmt("w/w_ref - 1 = %.2e (dr 0.05)", e1) + fmt(", %.2e (dr 0.025)", e2) + fmt(", ratio %.2f", e1 / e2));
}

// D9. Moving window: the lining of D5 as a function of (z, r) (sampled only in the columns a shift brings in) = the rectangle
//     block: the diagnostics (energies, Gauss law) of 1500 steps equal to round-off.
void diel_function_window(Comm& comm) {
    const char* path = "rz_diel_fwin_tmp.deck";
    auto run = [&](const std::string& lining) {
        if (comm.rank == 0) {
            std::ofstream f(path);
            f << "grid {\n geometry = rz\n modes = 1\n lambda0 = 100 um\n nz = 600\n nr = 40\n dz = 0.4 um\n dr = 0.5 um\n"
                 " boundary_z = open\n t_end = 2 ps\n}\nnumerics {\n solver = rip_cc\n pusher = boris\n}\n"
                 "window {\n start = 0 fs\n velocity = 1\n}\n" << lining <<
                 "species bunch {\n charge = 1\n mass = 18361526.7\n density = 1\n density_function = gauss(z - 0.12 mm, 8 um) * gauss(r, 3 um)\n"
                 " zmin = 0.09 mm\n zmax = 0.15 mm\n rmax = 10 um\n uz = 999.9995\n self_field = true\n ppc_z = 1\n ppc_r = 2\n ppc_theta = 1\n}\n"
                 "output {\n dir = rz_diel_fwin_out\n every = 100000\n}\n";
        }
        comm.barrier();
        Deck dk = Deck::read(path);
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        std::vector<double> out;
        long n = 0;
        while (!sim.finished()) {
            sim.step();
            if (++n % 250 == 0) {
                double wf, g, rm, db;
                sim.rz_diagnostics(wf, g, rm, db);
                comm.allreduce_sum(&wf, 1);
                out.insert(out.end(), {wf, g / rm});
            }
        }
        comm.barrier();
        if (comm.rank == 0) std::remove(path);
        return out;
    };
    const auto a = run("dielectric layer {\n r_in = 12 um\n r_out = 30 um\n z_start = 0.2 mm\n z_end = 10 mm\n eps = 3.8\n}\n");
    const auto b = run("dielectric layer {\n eps = 1 + 2.8 * (r >= 12 um && r < 30 um && z >= 0.2 mm && z < 10 mm)\n}\n");
    double dw = 0, gm = 0;
    for (size_t q = 0; q + 1 < a.size(); q += 2) { dw = std::max(dw, std::abs(a[q] / b[q] - 1)); gm = std::max(gm, b[q + 1]); }
    check("rz dielectric D9: moving window, the lining as a function of (z, r) (filled column by column) = the rectangle block",
          a.size() == b.size() && dw < 1e-12 && gm < 1e-11, fmt("max relative energy difference %.1e", dw) + fmt(", Gauss %.1e of max rho", gm));
}
// D10. chi per angle = chi per (r, z) for an axisymmetric eps written in x, y (the graded rod of D8 as 1 + 1.5 (1 - (x^2 + y^2)
//      / R^2): eps per angle, E_z divided, the lines solved, the energy and div D taken on the 3 angles of 2 modes) against the
//      same eps as a function of r (per mode): fields after 300 steps, the energy, W of mode 1 and div D to round-off.
void diel_phi_axisym(Comm& comm) {
    const int nz = 40, nr = 22, M = 2;
    const double dr = 0.1, dz = 0.07, R = nr * dr, kz = 2 * M_PI / (nz * dz);
    Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, Solver::RipCC);
    FieldsRZ a(d, M), b(d, M);
    char t[200];
    std::snprintf(t, sizeof t, "1 + 1.5 * (1 - (r / %.17g)^2)", R);
    a.diel = std::make_shared<DielectricRZ>(std::vector<DielectricConfig>{fblock(t)}, d, a.grid());
    std::snprintf(t, sizeof t, "1 + 1.5 * (1 - (x^2 + y^2) / %.17g)", R * R);
    DielectricConfig c = fblock(t);
    c.phi = true;
    b.diel = std::make_shared<DielectricRZ>(std::vector<DielectricConfig>{c}, d, b.grid());
    b.diel->set_angles(d, b.grid(), M, false);
    for (FieldsRZ* f : {&a, &b}) { init_mix(*f, R, kz); comm.exchange(d, all(*f)); }
    for (int n = 0; n < 300; ++n) { a.zero_currents(); a.step(comm); b.zero_currents(); b.step(comm); }
    double diff = 0, sc = 0;
    Field* fa[6] = {&a.ez, &a.er, &a.ep, &a.bz, &a.br, &a.bp};
    Field* fb[6] = {&b.ez, &b.er, &b.ep, &b.bz, &b.br, &b.bp};
    for (int q = 0; q < 6; ++q) diff = std::max(diff, max_abs_diff(*fa[q], *fb[q], &sc));
    diff = comm.allreduce_max(diff) / comm.allreduce_max(sc);
    double wa = a.energy(), wb = b.energy(), ma = a.energy_mode(1), mb = b.energy_mode(1);
    comm.allreduce_sum(&wa, 1); comm.allreduce_sum(&wb, 1); comm.allreduce_sum(&ma, 1); comm.allreduce_sum(&mb, 1);
    Field ga(d), gb(d);
    a.div_e(ga); b.div_e(gb);
    double gs = 0;
    const double dg = comm.allreduce_max(max_abs_diff(ga, gb, &gs)) / comm.allreduce_max(gs);
    check("rz dielectric D10: eps per angle (written in x, y) = the same eps per (r, z): fields, W, W_1, div D (300 steps, 2 modes)",
          b.diel->angular() && diff < 1e-12 && std::abs(wb / wa - 1) < 1e-12 && std::abs(mb / ma - 1) < 1e-12 && dg < 1e-12,
          fmt("fields %.1e of max", diff) + fmt(", W %.1e", wb / wa - 1) + fmt(", W_1 %.1e", mb / ma - 1) + fmt(", div D %.1e", dg));
}

// lowest TM cutoff (k_z = 0) of the pipe R = 1 (nr = 40) half filled with eps = 2.1 (x < 0), M modes: w from the DTFT peak of
// E_z projected on its start, sin(w dt / 2) = dt k / 2
double half_pipe(Comm& comm, int M) {
    const int nr = 40;
    const double dr = 1.0 / nr;
    const RadialGrid rg = rz_radial_grid(nr, dr);
    const double dz = 0.95 * 2 / std::sqrt(rz_lambda_max(rg, M));
    Domain d = rz_domain(comm, 4, nr, M, dz, dr, true, Solver::RipCC);
    FieldsRZ f(d, M);
    DielectricConfig c = fblock("x < 0 ? 2.1 : 1");
    c.phi = true;
    f.diel = std::make_shared<DielectricRZ>(std::vector<DielectricConfig>{c}, d, f.grid());
    f.diel->set_angles(d, f.grid(), M, false);
    {
        H ez(f.ez);
        for (int i = 0; i < d.n[0]; ++i)
            for (int j = 0; j < nr; ++j) ez(i, j, 0) = std::cyl_bessel_j(0, j01 * f.grid().r[size_t(j)]);
        ez.put(f.ez);
    }
    comm.exchange(d, all(f));
    const std::vector<double> h0 = f.ez.host();
    std::vector<double> s;
    const int steps = int(40 * 2 * M_PI / (1.88 * dz));
    for (int n = 0; n < steps; ++n) {
        const std::vector<double> x = f.ez.host();
        double v = 0;
        for (int j = 0; j < nr; ++j)
            for (int q = 0; q < 2 * M - 1; ++q) v += (q ? 2 : 1) * (j + 0.5) * x[f.ez.idx(0, j, q)] * h0[f.ez.idx(0, j, q)];
        s.push_back(comm.allreduce_max(v));
        f.zero_currents(); f.step(comm);
    }
    const double w = peak(s, 1.5 * dz, 2.4 * dz);
    return 2 / dz * std::sin(w / 2);
}

// D11. A round pipe half filled with a dielectric slab (eps 2.1 for x < 0, R = 1, nr = 40; eps depends on phi): the lowest TM
//      cutoff against the continuum of research/rz_dielectric/xsection_eps.py (polar finite differences, Richardson over three
//      grids, second order: 1.88013382), for 8 and 16 modes.
void diel_half_pipe(Comm& comm) {
    const double ref = 1.88013382;
    const double k8 = half_pipe(comm, 8), k16 = half_pipe(comm, 16);
    const double e8 = k8 / ref - 1, e16 = k16 / ref - 1;
    check("rz dielectric D11: pipe half filled with eps 2.1 (eps depending on phi): lowest TM cutoff against polar finite differences",
          std::abs(e16) < 5e-3 && std::abs(e16) <= std::abs(e8) * 1.05,
          fmt("8 modes %.6f", k8) + fmt(" (%+.2e)", e8) + fmt(", 16 modes %.6f", k16) + fmt(" (%+.2e); reference 1.88013382", e16));
}
}  // namespace

void test_rz_dielectric(Comm& comm) {
    diel_window(comm);
    diel_window_warning(comm);
    diel_identity(comm);
    diel_gauss_energy(comm);
    diel_tm01(comm);
    for (int v = 0; v < 3; ++v) diel_plasma(comm, v);
    diel_function_rect(comm);
    diel_grin(comm);
    diel_function_window(comm);
    diel_phi_axisym(comm);
    if (comm.size == 1) diel_half_pipe(comm);
}
