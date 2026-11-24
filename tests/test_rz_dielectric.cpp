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

// D4. Deck path: hot plasma in a periodic tube with rings, the gaps filled (eps 1.5), a collar outside (eps 2.2) and a block
//     crossing a ring face (modes 2, shape 1 and 2): the Gauss law div D = rho off the metal surface and div B at round-off.
void diel_plasma(Comm& comm) {
    const char* path = "rz_diel_test_tmp.deck";
    if (comm.rank == 0) {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n modes = 2\n lambda0 = 1 um\n nz = 50\n nr = 24\n dz = 0.04 um\n dr = 0.06 um\n"
             " t_end = 30 fs\n}\nnumerics {\n solver = rip_cc\n pusher = boris\n}\n"
             "rings {\n r_in = 0.48 um\n r_out = 0.9 um\n period = 1 um\n gap = 0.5 um\n z_start = 0.3 um\n z_end = 1.9 um\n}\n"
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
        check("rz dielectric D4: plasma, rings, filled gaps, collar, rod (shape " + std::to_string(order) +
                  "): div D - rho and div B off the metal at round-off",
              g < 1e-12 * std::max(rmax, 1e-30) + 1e-14 && db < 1e-12 && std::isfinite(wf),
              fmt("Gauss %.1e", g) + fmt(" (max rho %.1e)", rmax) + fmt(", div B %.1e", db) + fmt(", W %.3e", wf));
    }
    bool rejected = false;
    dk.override_from("numerics.solver=irip");
    try { Config::from_deck(dk); } catch (const std::exception& e) { rejected = std::string(e.what()).find("rip_cc") != std::string::npos; }
    check("rz dielectric: dielectric blocks with iRIP rejected", rejected, "");
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

}  // namespace

void test_rz_dielectric(Comm& comm) {
    diel_window(comm);
    diel_window_warning(comm);
    diel_identity(comm);
    diel_gauss_energy(comm);
    diel_tm01(comm);
    diel_plasma(comm);
}
