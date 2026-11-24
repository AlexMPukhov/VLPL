// Lorentz-boosted frame (docs/specs/boosted_frame.md, decision 87): the boosted run, back-transformed to the lab by
// lab_diagnostics, against the lab run on the same lab grid and against the analytic paraxial beam.
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include "common.hpp"
#include "fsutil.hpp"
#include "ionization.hpp"
#include "sim.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

Config deck_config(const std::string& text, const std::string& path) {
    { std::ofstream f(path); f << text; }
    return Config::from_deck(Deck::read(path));
}

struct Snap {                                    // a lab_fields snapshot (boost_diag.cpp)
    std::map<std::string, std::string> meta;
    long nx = 0, n1 = 0, n2 = 0;
    double x0 = 0, dx = 0;
    std::vector<double> v;                       // [c][k][j][i]
    double at(int c, long i, long j, long k) const { return v[size_t(((c * n2 + k) * n1 + j) * nx + i)]; }
};
Snap read_snap(const std::string& dir, int k) {
    Snap S;
    char b[64];
    std::snprintf(b, sizeof b, "/lab_fields/snap_%03d", k);
    std::ifstream m(dir + b + ".txt");
    for (std::string line; std::getline(m, line);) {
        if (line.empty() || line[0] == '#') continue;
        const size_t p = line.find(' ');
        if (p != std::string::npos) S.meta[line.substr(0, p)] = line.substr(p + 1);
    }
    if (S.meta.empty()) return S;
    S.nx = std::stol(S.meta["nx"]); S.n1 = std::stol(S.meta["n1"]); S.n2 = std::stol(S.meta["n2"]);
    S.x0 = std::stod(S.meta["x0"]); S.dx = std::stod(S.meta["dx"]);
    S.v.resize(size_t(6 * S.nx * S.n1 * S.n2));
    FILE* f = std::fopen((dir + b + ".bin").c_str(), "rb");
    if (f) { if (std::fread(S.v.data(), sizeof(double), S.v.size(), f) != S.v.size()) S.v.clear(); std::fclose(f); }
    return S;
}

// cubic Lagrange interpolation of the samples g(i) (spacing h, first at x0) at x
double interp(const std::vector<double>& g, double x0, double h, double x) {
    const double u = (x - x0) / h, fi = std::floor(u), w = u - fi;
    const long i = long(fi);
    if (i < 1 || i + 2 >= long(g.size())) return 0;
    return -w * (w - 1) * (w - 2) / 6 * g[size_t(i - 1)] + (w + 1) * (w - 1) * (w - 2) / 2 * g[size_t(i)] -
           (w + 1) * w * (w - 2) / 2 * g[size_t(i + 1)] + (w + 1) * w * (w - 1) / 6 * g[size_t(i + 2)];
}

std::string num(double v) { char b[64]; std::snprintf(b, sizeof b, "%.15g", v); return b; }

double rel(double num, double den) { return den > 0 ? std::sqrt(num / den) : 1e300; }

// B1. A vacuum laser (a0 = 0.01, 4 cycles, waist 5 um, focus 20 um): the boosted run (gamma = 2), back-transformed
//     to the lab time T = 1300 dx, against the lab run at T (interpolated on the planes) and the analytic paraxial beam.
//     2D Cartesian with RIP-CC, RZ (2 modes) with iRIP; window at c in both. The transverse box holds the beam at the lab
//     events of the boosted entrance plane (lab x from X0 = -83 um to 0, where it is up to 7.3 um wide; boost.cpp).
void boost_vacuum(Comm& comm) {
    for (int rz = 0; rz < 2; ++rz) {
        std::ostringstream base;
        base.precision(15);
        const double lam = 0.8, dx = lam / 16, c = 299792458.0;
        base << "grid {\n lambda0 = 0.8 um\n";
        if (rz) base << " geometry = rz\n modes = 2\n nz = 400\n nr = 80\n dz = " << dx << " um\n dr = 0.25 um\n boundary_z = open\n";
        else base << " geometry = cartesian2d\n nx = 400\n ny = 160\n dx = " << dx << " um\n dy = 0.25 um\n boundary_x = open\n";
        const double T_fs = 1300 * dx * 1e-6 / c * 1e15, tau_fs = 4 * lam * 1e-6 / c * 1e15;
        base << " t_end = " << T_fs << " fs\n}\n";
        base << "numerics {\n solver = " << (rz ? "irip" : "rip_cc") << "\n}\n";
        base << "laser {\n a0 = 0.01\n polarization = " << (rz ? "x" : "y") << "\n duration = " << tau_fs << " fs\n delay = "
             << 2 * tau_fs << " fs\n waist = 5 um\n focus = 20 um\n" << (rz ? "" : " y_center = 20 um\n") << "}\n";
        base << "window {\n start = " << 4.01 * tau_fs << " fs\n}\n";
        const std::string dir = rz ? "test_out_boost_rz" : "test_out_boost_2d";
        base << "output {\n dir = " << dir << "\n every = 100000\n}\n";
        // the lab run to T
        Config cl = deck_config(base.str(), "boost_test_tmp.deck");
        Simulation lab(cl, comm);
        lab.init();
        while (!lab.finished()) lab.step();
        const double T = lab.t;
        // the boosted run
        const std::string bdeck = base.str() + "boost {\n gamma = 2\n}\nlab_diagnostics {\n t_first = " + num(T_fs) + " fs\n}\n";
        Config cb = deck_config(bdeck, "boost_test_tmp.deck");
        if (comm.rank == 0) make_dirs(dir);
        comm.barrier();
        Simulation bs(cb, comm);
        bs.init();
        while (!bs.finished()) bs.step();
        bs.pipeline.finish(bs);
        comm.barrier();
        const Domain& d = lab.dom;
        const int N1 = d.N[1];
        const double h = d.h[0], xw = d.wx * h;           // lab node i at x = (wx + i) h
        // the lab field on the global x nodes of every transverse row (Ey, or Er slot 1 = Re m 1)
        std::vector<double> row(size_t(d.N[0]) * N1, 0.0);
        {
            const Field& F = rz ? lab.rz->er : lab.fields->ey;
            const std::vector<double> hv = F.host();
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i)
                    row[size_t(d.off[1] + j) * d.N[0] + d.off[0] + i] = hv[F.idx(i, j, rz ? 1 : 0)];
            comm.allreduce_sum(row.data(), int(row.size()));
        }
        Snap S = read_snap(dir, 0);
        double e_bl = 0, e_ba = 0, e_la = 0, nrm = 0, amax = 0;
        bool ok = !S.v.empty() && S.meta["complete"] == "1";
        if (ok) {
            LaserConfig L = cl.laser;                        // the lab beam
            for (long j = 0; j < S.n1; ++j) {
                const double y = (j + 0.5) * d.h[1];          // Ey at j + 1/2; RZ Er at r_{j+1/2}
                std::vector<double> g(row.begin() + long(j) * d.N[0], row.begin() + long(j + 1) * d.N[0]);
                for (long i = 0; i < S.nx; ++i) {
                    const double x = S.x0 + i * S.dx;
                    if (x < xw + 2 * h || x > xw + (d.N[0] - 3) * h) continue;
                    const double a = (rz ? 0.5 : 1.0) * L.field_at(T, x, y, 0.0);   // RZ: E_x = a -> Er of m = 1 is a / 2
                    const double b = S.at(1, i, j, rz ? 1 : 0), l = interp(g, xw, h, x);
                    e_bl += (b - l) * (b - l); e_ba += (b - a) * (b - a); e_la += (l - a) * (l - a); nrm += a * a;
                    amax = std::max(amax, std::abs(a));
                }
            }
        }
        const double rbl = rel(e_bl, nrm), rba = rel(e_ba, nrm), rla = rel(e_la, nrm);
        check(std::string("boost, vacuum laser, ") + (rz ? "RZ iRIP" : "2D RIP-CC") + ": lab snapshot = lab run, analytic beam",
              ok && amax > (rz ? 0.002 : 0.005) && rbl < 3e-2 && rba < 6e-2,
              fmt("boosted vs lab run %.2e, vs paraxial %.2e", rbl, rba) + fmt(" (lab run vs paraxial %.2e), steps", rla) +
                  fmt(" %.0f boosted / %.0f lab", double(bs.istep), double(lab.istep)));
    }
}

// B2. A laser wake (a0 = 1, 0.005 n_c, iRIP; 2D and RZ with 2 modes): the on-axis E_x (E_z) of the boosted run (gamma = 2)
//     back-transformed to the lab time T against the lab run at T, the peak of the wake and of the laser; the Gauss law of
//     the boosted run while the plasma streams through the window at -0.866 c (injected 3 cells behind the right edge).
//     lambda/32: the plasma streams into the laser, so a particle crosses a laser cycle in 32/(1 + beta) steps; at lambda/16
//     (8.6 steps) the boosted wake was 15 % (2D) and 21 % (RZ) too strong, at lambda/32 1.4 %, 1.1 % (decision 87).
void boost_wake(Comm& comm, bool c2) {
    std::ostringstream base;
    base.precision(15);
    const int dv = std::getenv("BOOST_DZDIV") ? std::atoi(std::getenv("BOOST_DZDIV")) : 32;   // cells per wavelength (16: the drive 15-21 % high)
    const double lam = 0.8, dz = lam / dv, c = 299792458.0;
    const double T_fs = 2600.0 * dv / 16 * dz * 1e-6 / c * 1e15;
    // c2: the same wake in 2D Cartesian (a slab beam)
    if (c2) base << "grid {\n lambda0 = 0.8 um\n geometry = cartesian2d\n nx = " << 30 * dv << "\n ny = 100\n dx = " << dz << " um\n dy = 0.4 um\n"
                 << " boundary_x = open\n t_end = " << T_fs << " fs\n}\n";
    else base << "grid {\n lambda0 = 0.8 um\n geometry = rz\n modes = 2\n nz = " << 30 * dv << "\n nr = 50\n dz = " << dz << " um\n dr = 0.4 um\n"
              << " boundary_z = open\n t_end = " << T_fs << " fs\n}\n";
    base << "numerics {\n solver = " << (std::getenv("BOOST_SOLVER") ? std::getenv("BOOST_SOLVER") : "irip") << "\n}\n";
    base << "laser {\n a0 = 1\n polarization = y\n duration = 12 fs\n delay = 24 fs\n waist = 4 um\n focus = 15 um\n" << (c2 ? " y_center = 20 um\n" : "") << "}\n";
    base << "window {\n start = 79 fs\n}\n";
    for (int s = 0; s < 2; ++s)
        base << "species " << (s ? "p" : "e") << " {\n charge = " << (s ? 1 : -1) << "\n mass = " << (s ? 1836 : 1)
             << "\n density = 0.005\n " << (c2 ? "xmin" : "zmin") << " = 30 um\n"
             << (std::getenv("BOOST_PPC") ? std::getenv("BOOST_PPC") : c2 ? " ppc_x = 1\n ppc_y = 2\n" : " ppc_r = 1\n ppc_theta = 6\n") << "}\n";
    base << "output {\n dir = test_out_boost_wake\n every = 100000\n}\n";
    Config cl = deck_config(base.str(), "boost_test_tmp.deck");
    Simulation lab(cl, comm);
    lab.init();
    while (!lab.finished()) lab.step();
    const std::string gb = std::getenv("BOOST_G") ? std::getenv("BOOST_G") : "2";
    Config cb = deck_config(base.str() + "boost {\n gamma = " + gb + "\n}\nlab_diagnostics {\n t_first = " + num(T_fs) + " fs\n}\n",
                            "boost_test_tmp.deck");
    if (comm.rank == 0) make_dirs("test_out_boost_wake");
    comm.barrier();
    Simulation bs(cb, comm);
    bs.init();
    double g = 0;
    while (!bs.finished()) {
        bs.step();
        if (bs.istep % 10 == 0) g = std::max(g, bs.gauss_drift());
    }
    bs.pipeline.finish(bs);
    comm.barrier();
    const Domain& d = lab.dom;
    const double h = d.h[0], xw = d.wx * h;
    std::vector<double> ax(size_t(d.N[0]), 0.0);          // E_z (m = 0) on the axis, lab run
    {
        const Field& F = c2 ? lab.fields->ex : lab.rz->ez;
        const int jc = c2 ? d.N[1] / 2 - d.off[1] : 0;
        const std::vector<double> hv = F.host();
        if (jc >= 0 && jc < d.n[1])
            for (int i = 0; i < d.n[0]; ++i) ax[size_t(d.off[0] + i)] = hv[F.idx(i, jc, 0)];
        comm.allreduce_sum(ax.data(), int(ax.size()));
    }
    Snap S = read_snap("test_out_boost_wake", 0);
    double e = 0, n = 0, emax = 0, bmax = 0;
    bool ok = !S.v.empty() && S.meta["complete"] == "1";
    for (long i = 0; ok && i < S.nx; ++i) {
        const double x = S.x0 + i * S.dx;
        if (x < xw + 4 * h || x > xw + (d.N[0] - 4) * h) continue;
        const double b = S.at(0, i, c2 ? S.n1 / 2 : 0, 0), l = interp(ax, xw, h, x);
        if (std::getenv("BOOST_DEBUG")) {
            static FILE* df = std::fopen("boost_wake_debug.txt", "w");
            std::fprintf(df, "%.10g %.10g %.10g\n", x, l, b);
        }
        e += (b - l) * (b - l); n += l * l;
        emax = std::max(emax, std::abs(l)); bmax = std::max(bmax, std::abs(b));
    }
    const double r = rel(e, n);
    // the laser: the largest transverse E (Ey of 2D, Ephi of RZ for polarization y: every slot), lab run and snapshot
    double llab = 0, lbst = 0;
    {
        const Field& F = c2 ? lab.fields->ey : lab.rz->ep;
        const std::vector<double> hv = F.host();
        for (int k = 0; k < (c2 ? 1 : d.n[2]); ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) llab = std::max(llab, std::abs(hv[F.idx(i, j, k)]));
        llab = comm.allreduce_max(llab);
        for (long k = 0; ok && k < S.n2; ++k)
            for (long j = 0; j < S.n1; ++j)
                for (long i = 0; i < S.nx; ++i) lbst = std::max(lbst, std::abs(S.at(c2 ? 1 : 2, i, j, k)));
    }
    const double dp = bmax / emax - 1, dl = lbst / llab - 1;
    check(std::string("boost, wake (a0 = 1, 0.005 n_c, gamma 2, lambda/32), ") + (c2 ? "2D" : "RZ") +
              ": on-axis Ez and laser = lab run; Gauss law",
          ok && emax > 1e-3 && std::abs(dp) < 0.03 && std::abs(dl) < 0.05 && r < (c2 ? 0.08 : 0.15) && g < 1e-12,
          fmt("peak Ez %+.1e, laser %+.1e", dp, dl) + fmt(", rel. L2 %.2e (axis noise), max|dGauss| %.1e", r, g) +
              fmt(", steps %.0f boosted / %.0f lab", double(bs.istep), double(lab.istep)));
}

// B3. Gauss law at round-off in the boosted frame with a laser and hot plasma streaming through the window
//     (2D, RIP-CC and iRIP; 3D RIP-CC), and the lab-frame beam of a ballistic beam: charge, mean energy, centre and length
//     at the lab time against the lab deck's beam moved to that time.
void boost_gauss_beam(Comm& comm) {
    const char* geo[3] = {"2D RIP-CC, cold", "2D iRIP, hot", "3D RIP-CC, cold"};
    for (int k = 0; k < 3; ++k) {
        std::ostringstream o;
        o << "grid {\n lambda0 = 0.8 um\n" << (k < 2 ? " geometry = cartesian2d\n" : "") << " nx = 128\n ny = 20\n"
          << (k < 2 ? "" : " nz = 8\n dz = 0.5 um\n") << " dx = 0.05 um\n dy = 0.5 um\n boundary_x = open\n t_end = 60 fs\n}\n"
          << "numerics {\n solver = " << (k == 1 ? "irip" : "rip_cc") << "\n shape = 2\n}\n"
          << "laser {\n a0 = 1\n polarization = y\n duration = 3 fs\n delay = 6 fs\n waist = 2 um\n y_center = 5 um\n"
          << (k < 2 ? "" : " z_center = 2 um\n") << "}\n"
          << "window {\n start = 12.1 fs\n}\nboost {\n gamma = 3\n}\n";
        for (int s = 0; s < 2; ++s)
            o << "species " << (s ? "p" : "e") << " {\n charge = " << (s ? 1 : -1) << "\n mass = " << (s ? 100 : 1)
              << "\n density = 0.02\n xmin = 4 um\n ppc_x = 2\n ppc_y = 2\n" << (k == 1 ? " uth_x = 0.02\n uth_y = 0.02\n uth_z = 0.02\n" : "") << "}\n";
        o << "output {\n dir = test_out_boost_gauss\n every = 100000\n}\n";
        Config c = deck_config(o.str(), "boost_test_tmp.deck");
        Simulation s(c, comm);
        s.init();
        double g = 0, n0 = 0;
        while (!s.finished()) {
            s.step();
            if (s.istep % 5 == 0) g = std::max(g, s.gauss_drift());
        }
        n0 = double(s.species[0].size());
        comm.allreduce_sum(&n0, 1);
        check(std::string("boost, Gauss law with plasma streaming through the window, ") + geo[k],
              g < 1e-12 && n0 > 1000, fmt("max|dGauss| %.1e, %.0f electrons", g, n0));
    }
    // the ballistic beam: RZ vacuum, lab gamma 20 (cold), z in [20, 30] at t = 0, r < 5 (code units); density 1e-12 n_c
    // (its own field negligible)
    const double k0 = 2 * M_PI / 0.8e-6, um = 1e6 / k0;
    std::ostringstream o;
    o.precision(15);
    o << "grid {\n lambda0 = 0.8 um\n geometry = rz\n modes = 1\n nz = 128\n nr = 16\n dz = " << 0.4 * um << " um\n dr = "
      << 0.5 * um << " um\n boundary_z = open\n t_end = " << 300 / (299792458.0 * k0) * 1e15 << " fs\n}\n"
      << "numerics {\n solver = irip\n}\nwindow {\n start = 0 fs\n}\nboost {\n gamma = 2\n}\n"
      << "lab_diagnostics {\n t_first = " << 250 / (299792458.0 * k0) * 1e15 << " fs\n species = b\n}\n"
      << "species b {\n charge = -1\n mass = 1\n density = 1e-12\n uz = " << std::sqrt(399.0) << "\n zmin = " << 20 * um
      << " um\n zmax = " << 30 * um << " um\n rmax = " << 5 * um << " um\n ppc_z = 4\n ppc_r = 2\n ppc_theta = 4\n}\n"
      << "output {\n dir = test_out_boost_beam\n every = 100000\n}\n";
    Config c = deck_config(o.str(), "boost_test_tmp.deck");
    if (comm.rank == 0) make_dirs("test_out_boost_beam");
    comm.barrier();
    Simulation s(c, comm);
    s.init();
    while (!s.finished()) s.step();
    s.pipeline.finish(s);
    comm.barrier();
    // lab_beams.txt: snapshot t_lab species complete N Q E_mean dE/E E_max eps_y eps_z x_mean sx sy sz
    std::ifstream f("test_out_boost_beam/lab_beams.txt");
    std::vector<double> v;
    std::string comp;
    for (std::string line; std::getline(f, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        double a; std::string sp; int cpl;
        is >> a >> a >> sp >> cpl;
        comp = std::to_string(cpl);
        for (double x; is >> x;) v.push_back(x);
    }
    bool ok = v.size() == 11 && comp == "1";
    const double v_b = std::sqrt(399.0) / 20.0, xm = 25 + v_b * 250, sx = 10 / std::sqrt(12.0);
    const double e_si = 1.602176634e-19, eps0 = 8.8541878128e-12, me = 9.1093837015e-31, w0 = 299792458.0 * k0;
    const double qpc = e_si * eps0 * me * w0 * w0 / (e_si * e_si) / (k0 * k0 * k0) * 1e12;
    const double Q = 1e-12 * M_PI * 25 * 10 * qpc, E = (20 - 1) * 0.51099895;
    double dq = 1, de = 1, dxm = 1, dsx = 1;
    if (ok) { dq = v[1] / Q - 1; de = v[2] / E - 1; dxm = v[7] / um - xm; dsx = v[8] / um / sx - 1; }
    check("boost, ballistic beam in the lab frame (gamma 20, boost 2): charge, energy, centre, length",
          ok && std::abs(dq) < 0.03 && std::abs(de) < 1e-9 && std::abs(dxm) < 0.05 && std::abs(dsx) < 0.03,
          fmt("Q %+.2e, E_mean %+.1e", dq, de) + fmt(", x_mean %+.3f (1/k0), sigma_x %+.2e", dxm, dsx));
}

// lab_diagnostics.select_uz, select_r (decision 97): the recording cut of the lab particle snapshots (FBPIC's select). The
// ballistic beam of boost_gauss_beam (lab gamma 20) plus a slow beam (u_z = 2) through the same window: with select_uz = 10
// and select_r = 3 (1/k0) exactly the records of the unfiltered file that pass the cut (the same count; the slow beam none).
void boost_lab_select(Comm& comm) {
    const double k0 = 2 * M_PI / 0.8e-6, um = 1e6 / k0;
    auto deck = [&](bool cut) {
        std::ostringstream o;
        o.precision(15);
        o << "grid {\n lambda0 = 0.8 um\n geometry = rz\n modes = 1\n nz = 128\n nr = 16\n dz = " << 0.4 * um << " um\n dr = "
          << 0.5 * um << " um\n boundary_z = open\n t_end = " << 300 / (299792458.0 * k0) * 1e15 << " fs\n}\n"
          << "numerics {\n solver = irip\n}\nwindow {\n start = 0 fs\n}\nboost {\n gamma = 2\n}\n"
          << "lab_diagnostics {\n t_first = " << 250 / (299792458.0 * k0) * 1e15 << " fs\n species = b s\n"
          << (cut ? " select_uz = 10\n select_r = " + std::to_string(3 * um) + " um\n" : "") << "}\n";
        for (int k = 0; k < 2; ++k)
            o << "species " << (k ? "s" : "b") << " {\n charge = -1\n mass = 1\n density = 1e-12\n uz = " << (k ? 2.0 : std::sqrt(399.0))
              << "\n zmin = " << 20 * um << " um\n zmax = " << 30 * um << " um\n rmax = " << 5 * um << " um\n ppc_z = 4\n ppc_r = 2\n"
              << " ppc_theta = 4\n}\n";
        o << "output {\n dir = test_out_boost_select" << (cut ? "_cut" : "") << "\n every = 100000\n}\n";
        return o.str();
    };
    auto read = [](const std::string& base) {
        std::vector<double> v;
        FILE* f = std::fopen((base + ".bin").c_str(), "rb");
        if (!f) return v;
        double x;
        while (std::fread(&x, sizeof x, 1, f) == 1) v.push_back(x);
        std::fclose(f);
        return v;
    };
    for (int cut = 0; cut < 2; ++cut) {
        Config c = deck_config(deck(cut), "boost_test_tmp.deck");
        if (comm.rank == 0) make_dirs(std::string("test_out_boost_select") + (cut ? "_cut" : "") + "/lab_particles");
        comm.barrier();
        Simulation s(c, comm);
        s.init();
        while (!s.finished()) s.step();
        s.pipeline.finish(s);
        comm.barrier();
    }
    if (comm.rank != 0) return;
    const std::vector<double> b0 = read("test_out_boost_select/lab_particles/b_000"), s0 = read("test_out_boost_select/lab_particles/s_000");
    const std::vector<double> b1 = read("test_out_boost_select_cut/lab_particles/b_000");
    const std::vector<double> s1 = read("test_out_boost_select_cut/lab_particles/s_000");
    size_t pass = 0;
    for (size_t p = 0; p + 7 <= b0.size(); p += 7)
        if (b0[p + 3] >= 10 && std::abs(b0[p + 1]) < 3 && std::abs(b0[p + 2]) < 3) ++pass;
    check("boost, lab particle snapshots with select_uz = 10, select_r = 3: the records of the unfiltered file that pass",
          b0.size() / 7 > 1000 && s0.size() / 7 > 1000 && pass > 100 && pass < b0.size() / 7 && b1.size() / 7 == pass && s1.empty(),
          fmt("unfiltered %.0f fast + %.0f slow records", double(b0.size() / 7), double(s0.size() / 7)) +
              fmt("; with the cut %.0f fast (expected %.0f), ", double(b1.size() / 7), double(pass)) +
              fmt("%.0f slow", double(s1.size() / 7)));
}

// B4. Field ionization in the boosted frame (decision 90). (a) The rest-frame field of ionization.hpp against the Lorentz
//     transformation E_rest = gamma (E + v x B) - gamma^2/(gamma + 1) v (v . E), and the lab plane wave seen by an ion at rest
//     in the lab from the boosted frame. (b) A laser (8 fs, lambda/32) through a nitrogen slab (lab x in [60, 66] um,
//     1e-4 n_c atoms, mobile; a0 = 0.0105 and 0.03): the charge-state fractions after the pulse in the boosted run (gamma = 2)
//     against the lab run on the same lab grid, 2D (plane wave, periodic y) and RZ (waist 10 um, ions within r < 2 um;
//     2 modes), iRIP.
//     The slab is chosen so that every ion is still in the boosted box at the end: an ion at lab x leaves the box at the lab
//     time x / beta + 2 delay, after the pulse (which has passed x at x + 2 delay), so t_end in [b + 2 delay, a / beta + 2 delay).
//     The boosted lattice holds gamma^2 (1 + beta) = 7.5 times fewer points of the slab: ppc_x 8 times the lab's.
//     Statistical error: sigma_k^2 = f_k (1 - f_k) (1/N_lab + 1/N_boost) (independent random numbers and lattices).
void boost_ionization(Comm& comm) {
    {
        double worst = 0;
        unsigned long long key = 12345;
        auto rnd = [&] { key = ion_mix(key); return 2 * ion_uniform(key) - 1; };
        for (int t = 0; t < 1000; ++t) {
            double E[3], B[3], u[3];
            for (int a = 0; a < 3; ++a) { E[a] = rnd(); B[a] = rnd(); u[a] = 3 * rnd(); }
            double g;
            const double r = ion_rest_field(E, B, u[0], u[1], u[2], g);
            const double v[3] = {u[0] / g, u[1] / g, u[2] / g}, ve = v[0] * E[0] + v[1] * E[1] + v[2] * E[2];
            const double vxb[3] = {v[1] * B[2] - v[2] * B[1], v[2] * B[0] - v[0] * B[2], v[0] * B[1] - v[1] * B[0]};
            double s2 = 0;
            for (int a = 0; a < 3; ++a) { const double e = g * (E[a] + vxb[a]) - g * g / (g + 1) * v[a] * ve; s2 += e * e; }
            worst = std::max(worst, std::abs(r / std::sqrt(s2) - 1));
        }
        // the lab plane wave E_y = B_z = 1 in the frame boosted by gamma = 2: E'_y = B'_z = gamma (1 - beta); the ion moves at -beta
        const double gb = 2, bb = std::sqrt(1 - 1 / (gb * gb)), f = gb * (1 - bb);
        const double E[3] = {0, f, 0}, B[3] = {0, 0, f};
        double g;
        const double r = ion_rest_field(E, B, -gb * bb, 0, 0, g);
        check("boost, ionization: the field in the ion's rest frame = Lorentz transformation; lab plane wave seen by a lab ion",
              worst < 1e-12 && std::abs(r - 1) < 1e-14 && std::abs(g - gb) < 1e-14,
              fmt("1000 random (E, B, u): max rel. difference %.1e", worst) + fmt("; plane wave |E_rest| - 1 = %.1e", r - 1));
    }
    // a0 = 0.0105: N0 / N1+ about 1/2 each (2D, RZ); a0 = 0.03: N0 -> N2+ / N3+, up to three levels per ion (2D)
    const double ea = std::getenv("BOOST_ION_A0") ? std::atof(std::getenv("BOOST_ION_A0")) : 0.0;
    const double cases[3][2] = {{0, 0.0105}, {1, 0.0105}, {0, 0.03}};
    for (const auto& cs : cases) {
        const int rz = int(cs[0]);
        const double a0 = ea > 0 ? ea : cs[1];
        std::vector<double> frac[2], neff(2), wsum(2);
        double gmax = 0, ecur = 0, eion = 0, events[2] = {0, 0}, steps[2] = {0, 0}, ne0 = 0, nemis = 0;
        for (int b = 0; b < 2; ++b) {
            std::ostringstream o;
            o << "grid {\n lambda0 = 0.8 um\n";
            if (rz) o << " geometry = rz\n modes = 2\n nz = 2800\n nr = 72\n dz = 0.025 um\n dr = 0.4 um\n boundary_z = open\n";
            else o << " geometry = cartesian2d\n nx = 2800\n ny = 8\n dx = 0.025 um\n dy = 0.1 um\n boundary_x = open\n";
            o << " t_end = 257.5 fs\n}\nnumerics {\n solver = irip\n seed = " << (b ? 11 : 7) << "\n}\n";
            o << "laser {\n a0 = " << a0 << "\n polarization = " << (rz ? "x" : "y") << "\n duration = 8 fs\n delay = 16 fs\n"
              << (rz ? " waist = 10 um\n focus = 63 um\n" : " waist = 1000 um\n") << "}\n";
            const int px = b ? 32 : 4;
            o << "species nitrogen {\n element = N\n ionization = adk\n ionization_electrons = electrons\n density = 1e-4\n"
              << (rz ? " zmin = 60 um\n zmax = 66 um\n rmax = 2 um\n ppc_z = " : " xmin = 60 um\n xmax = 66 um\n ppc_x = ") << px
              << (rz ? "\n ppc_r = 2\n ppc_theta = 8\n" : "\n ppc_y = 4\n") << "}\n";
            // the electron species (density = 0) starts empty, also in RZ, and each event adds one macro-electron
            o << "species electrons {\n charge = -1\n mass = 1\n density = 0\n" << (rz ? " ppc_theta = 8\n" : "") << "}\n";
            o << "output {\n dir = test_out_boost_ion\n every = 100000\n}\n";
            if (b) o << "boost {\n gamma = 2\n}\n";
            Config c = deck_config(o.str(), "boost_test_tmp.deck");
            Simulation s(c, comm);
            s.init();
            int ie = -1;
            for (size_t i = 0; i < s.cfg.species.size(); ++i)
                if (s.cfg.species[i].name == "electrons") ie = int(i);
            double nel = double(s.species[size_t(ie)].size());
            comm.allreduce_sum(&nel, 1);
            ne0 += nel;
            while (!s.finished()) {
                s.step();
                if (b && s.istep % 10 == 0) gmax = std::max(gmax, s.gauss_drift());
            }
            int is = -1;
            for (size_t i = 0; i < s.cfg.species.size(); ++i)
                if (s.cfg.species[i].name == "nitrogen") is = int(i);
            Species& S = s.species[size_t(is)];
            const std::vector<int> z = S.get_as<int>(S.zslot);
            const std::vector<double> w = S.get_as<double>(Species::W);
            std::vector<double> h(9, 0.0);                  // weights of N0 .. N7+, sum of w^2
            for (size_t p = 0; p < z.size(); ++p) { h[size_t(z[p])] += w[p]; h[8] += w[p] * w[p]; }
            comm.allreduce_sum(h.data(), 9);
            double W = 0;
            for (int k = 0; k < 8; ++k) W += h[size_t(k)];
            for (int k = 0; k < 8; ++k) frac[b].push_back(h[size_t(k)] / W);
            neff[size_t(b)] = W * W / h[8]; wsum[size_t(b)] = W;
            const IonizationTotals T = ionization_totals(s);
            events[b] = double(T.events); steps[b] = double(s.istep);
            nel = double(s.species[size_t(ie)].size());
            comm.allreduce_sum(&nel, 1);
            nemis += std::abs(nel - events[b]);
            if (b) { ecur = T.e_cur; eion = T.e_ion; }
        }
        double zmax = 0, zbar[2] = {0, 0};
        int nstates = 0;                                    // states above 1 % (the case must not be trivial)
        std::string fr;
        for (int k = 0; k < 8; ++k) {
            zbar[0] += k * frac[0][size_t(k)]; zbar[1] += k * frac[1][size_t(k)];
            const double f = 0.5 * (frac[0][size_t(k)] + frac[1][size_t(k)]);
            if (!(f > 0)) continue;
            if (f > 0.01) ++nstates;
            const double sg = std::sqrt(f * (1 - f) * (1 / neff[0] + 1 / neff[1]));
            const double zk = (frac[1][size_t(k)] - frac[0][size_t(k)]) / sg;
            zmax = std::max(zmax, std::abs(zk));
            if (f > 1e-3)
                fr += (k ? fmt(" N%.0f+", k) : std::string(" N0")) + fmt(" %.4f", frac[0][size_t(k)]) + fmt("/%.4f", frac[1][size_t(k)]) +
                      fmt(" (%+.1f sigma)", zk);
        }
        const double dw = wsum[1] / wsum[0] - 1, dc = eion > 0 ? ecur / eion - 1 : 1;
        check(std::string("boost, field ionization (N, a0 = ") + fmt("%.4g", a0) + ", gamma 2), " + (rz ? "RZ" : "2D") +
                  ": charge states lab = boosted within the statistics; Gauss law, E_cur = E_ion",
              zmax < 4 && nstates >= 2 && std::abs(dw) < 0.02 && gmax < 1e-12 && std::abs(dc) < 1e-10 && ne0 == 0 && nemis == 0,
              std::string("lab/boosted:") + fr + fmt("; max |dev| %.2f sigma", zmax) + fmt(", <Z> %.4f", zbar[0]) +
                  fmt("/%.4f", zbar[1]) + fmt(", N_eff %.0f", neff[0]) + fmt("/%.0f", neff[1]) + fmt(", weight %+.1e", dw) +
                  fmt(", events %.0f", events[0]) + fmt("/%.0f", events[1]) + fmt(", max|dGauss| %.1e", gmax) +
                  fmt(", E_cur/E_ion - 1 %.1e", dc) + fmt(", steps %.0f", steps[0]) + fmt("/%.0f", steps[1]) +
                  fmt("; electrons at the start %.0f, electrons - events %.0f", ne0, nemis));
    }
}

// B5. The summary's warning that the transverse box cuts the laser beam at its lab entrance events (x in [X0, 0]): a 2 um
//     waist focused 63 um downstream in a 40 um box (gamma 2, X0 = -62 um) is ~16 um wide there: warning; a plane wave
//     (waist 1000 um over a periodic 0.8 um y) is wider than the box everywhere, the deck's choice: no warning; the same
//     beam focused at 10 um fits at its entrance: no warning.
void boost_beam_warning() {
    auto deck = [](int ny, double dy, double waist, double focus) {
        std::ostringstream o;
        o << "grid {\n lambda0 = 0.8 um\n geometry = cartesian2d\n nx = 2800\n ny = " << ny << "\n dx = 0.025 um\n dy = " << dy
          << " um\n boundary_x = open\n t_end = 257.5 fs\n}\nnumerics {\n solver = irip\n}\n"
          << "laser {\n a0 = 0.01\n duration = 8 fs\n delay = 16 fs\n waist = " << waist << " um\n focus = " << focus << " um\n}\n"
          << "boost {\n gamma = 2\n}\noutput {\n dir = test_out_boost_ion\n}\n";
        Config c = deck_config(o.str(), "boost_test_tmp.deck");
        double w, d, half;
        const bool cut = boost_beam_cut(c, c.laser, w, d, half);
        const double um = 0.8 / (2 * M_PI);
        return std::make_pair(cut, fmt("%.3g", w * um) + fmt("/%.3g um", half * um));
    };
    const auto narrow = deck(160, 0.25, 2, 63), plane = deck(8, 0.1, 1000, 0), near = deck(160, 0.25, 5, 10);
    check("boost, laser beam cut at its lab entrance events: warning only when the box holds the beam at its focus",
          narrow.first && !plane.first && !near.first,
          "width/half box: narrow far focus " + narrow.second + (narrow.first ? " warns" : " no warning") + ", plane wave " +
              plane.second + (plane.first ? " warns" : " no warning") + ", near focus " + near.second +
              (near.first ? " warns" : " no warning"));
}

// B6. Stability of the streaming plasma (bench/nci/boost_stability, decision 104).
//  (a) The plasma of a boosted LWFA at gamma 10 (lab n0 = 0.004 n_c, dx_lab = lambda/32 -> dx' = 0.62 um, h_perp = 0.4 um), warm
//      (u_th = 1e-2), streaming through a periodic box (2D 64 x 32, RZ 64 x 16 with 2 modes) for 100 / omega_p: without the filter
//      the field energy grows to more than 10 times the level with one binomial pass and shape 2, which stays < 1e-5 (the
//      noise level; bench numbers: 1.2e-3 against 5e-6 in 2D at 100 / omega_p).
//  (b) A boosted 2D run (gamma 10, iRIP, the window) with one filter pass and warm plasma that streams in at the right edge and
//      out at the left (171 steps; the plasma crosses the 192 cells in ~96): the Gauss law at round-off outside the first
//      filter_passes cells.
//  (c) The summary's warning (boost_nci_risk) for filter_passes = 0 at gamma 10, for shape 1 at gamma 3 and with two passes at
//      gamma 10; not for shape 2 with one pass (a short box), nor for shape 2 without a filter at gamma 2.
double stream_noise(Comm& comm, bool rz, int passes) {
    std::ostringstream o;
    o << "grid {\n lambda0 = 1 um\n" << (rz ? " geometry = rz\n modes = 2\n nz = 64\n nr = 16\n dz = 0.6234 um\n dr = 0.4 um\n"
                                           " boundary_z = periodic\n boundary_r = wall\n"
                                         : " geometry = cartesian2d\n nx = 64\n ny = 32\n dx = 0.6234 um\n dy = 0.4 um\n"
                                           " boundary_x = periodic\n")
      << " t_end = 850 fs\n}\nnumerics {\n solver = irip\n shape = 2\n filter_passes = " << passes << "\n}\n";
    for (int q = 0; q < 2; ++q)
        o << "species " << (q ? "p" : "e") << " {\n charge = " << (q ? 1 : -1) << "\n mass = " << (q ? 1836 : 1)
          << "\n density = 0.04\n" << (rz ? " uz" : " ux") << " = -9.949874\n"
          << (rz ? " ppc_z = 2\n ppc_r = 2\n ppc_theta = 8\n" : " ppc_x = 2\n ppc_y = 2\n")
          << (q ? "" : " uth_x = 1e-2\n uth_y = 1e-2\n uth_z = 1e-2\n") << "}\n";
    o << "output {\n dir = test_out_boost_nci\n every = 100000\n}\n";
    Config c = deck_config(o.str(), "boost_test_tmp.deck");
    Simulation s(c, comm);
    s.init();
    double k0[1] = {s.species[0].kinetic_energy()};
    comm.allreduce_sum(k0, 1);
    while (!s.finished()) s.step();
    double w[1] = {rz ? s.rz->energy() : s.fields->energy()};
    comm.allreduce_sum(w, 1);
    return w[0] / k0[0];
}

void boost_stability(Comm& comm) {
    for (int rz = 0; rz < 2; ++rz) {
        const double w0 = stream_noise(comm, rz, 0), w1 = stream_noise(comm, rz, 1);
        check(std::string("boost, streaming plasma at gamma 10 (warm, periodic, 100/wp): unstable without filter, stable with one pass, ") +
                  (rz ? "RZ" : "2D"),
              w0 > 10 * w1 && w1 < 1e-5, fmt("W_field / K_e: no filter %.1e", w0) + fmt(", one pass %.1e", w1));
    }
    {
        std::ostringstream o;
        o << "grid {\n lambda0 = 1 um\n geometry = cartesian2d\n nx = 192\n ny = 16\n dx = 0.03125 um\n dy = 0.4 um\n"
             " boundary_x = open\n t_end = 7 ps\n}\nnumerics {\n solver = irip\n shape = 2\n filter_passes = 1\n}\n"
             "window {\n start = 0 fs\n}\nboost {\n gamma = 10\n}\n";
        for (int q = 0; q < 2; ++q)
            o << "species " << (q ? "p" : "e") << " {\n charge = " << (q ? 1 : -1) << "\n mass = " << (q ? 1836 : 1)
              << "\n density = 0.004\n ppc_x = 2\n ppc_y = 2\n uth_x = 1e-2\n uth_y = 1e-2\n uth_z = 1e-2\n}\n";
        o << "output {\n dir = test_out_boost_nci\n every = 100000\n}\n";
        Config c = deck_config(o.str(), "boost_test_tmp.deck");
        const bool warn = !boost_nci_risk(c).empty();
        Simulation s(c, comm);
        s.init();
        double g = 0, left = 0;
        long n = 0;
        while (!s.finished()) {
            s.step();
            ++n;
            if (s.istep % 5 == 0) g = std::max(g, s.gauss_drift());
        }
        for (const Species& sp : s.species) left += double(sp.size());
        comm.allreduce_sum(&left, 1);
        check("boost, current filter with the window (gamma 10, 2D iRIP, warm plasma streaming out at the left edge): Gauss law",
              g < 1e-12 && !warn && n > 120, fmt("max|dGauss| %.1e", g) + fmt(" over %.0f steps", double(n)) +
                  fmt(", %.0f particles", left) + (warn ? ", warns" : ""));
    }
    auto risk = [](double gamma, int shape, int passes) {
        std::ostringstream o;
        o << "grid {\n lambda0 = 1 um\n geometry = cartesian2d\n nx = 64\n ny = 16\n dx = 0.03125 um\n dy = 0.4 um\n"
             " boundary_x = open\n t_end = 10 fs\n}\nnumerics {\n solver = irip\n shape = " << shape << "\n filter_passes = " << passes
          << "\n}\nwindow {\n start = 0 fs\n}\nboost {\n gamma = " << gamma << "\n}\nspecies e {\n charge = -1\n mass = 1\n"
             " density = 0.004\n}\nspecies p {\n charge = 1\n mass = 1836\n density = 0.004\n}\noutput {\n dir = test_out_boost_nci\n}\n";
        return !boost_nci_risk(deck_config(o.str(), "boost_test_tmp.deck")).empty();
    };
    const bool a = risk(10, 2, 0), b = risk(3, 1, 0), d = risk(10, 2, 1), e = risk(2, 2, 0), f = risk(10, 1, 2);
    check("boost, NCI warning: filter 0 at gamma 10, shape 1 at gamma 3 and shape 1 with two passes warn; shape 2 + one pass, "
          "shape 2 at gamma 2 do not", a && b && f && !d && !e,
          std::string(a ? "" : "g10 f0 silent ") + (b ? "" : "g3 s1 silent ") + (f ? "" : "g10 s1 f2 silent ") + (d ? "g10 s2 f1 warns " : "") +
              (e ? "g2 s2 f0 warns" : "") + "ok");
}


// B7. The boosted injection must be neutral (decision 108; Alexander: "absolutely neutral ... a precondition"): BoostInject creates
//     the plasma with no creation current, so any net charge it creates stays at the injection plane. (a) Config::check refuses a
//     boosted plasma that is not neutral at every lattice point, naming the mismatch: the rounded electron density of a He / N
//     gas deck (3.11893e-4 vs 3.118936e-4; the message gives the neutral density), different ramps, different ppc, electrons alone;
//     the exact deck passes; boost_net_charge = (n_p - n_e) / (n_p + n_e) for the rounded density. (b) B8 below: neutral decks keep
//     the Gauss law at round-off.
std::string neutral_deck(double ne, double np, const char* fe, const char* fp, double tend_ps, const char* ppe = "ppc_x = 2\n ppc_y = 2",
                         bool ions = true) {
    std::ostringstream o;
    o.precision(12);
    o << "grid {\n lambda0 = 1 um\n geometry = cartesian2d\n nx = 192\n ny = 16\n dx = 0.03125 um\n dy = 0.4 um\n"
         " boundary_x = open\n t_end = " << tend_ps << " ps\n}\nnumerics {\n solver = irip\n shape = 2\n filter_passes = 1\n}\n"
         "window {\n start = 0 fs\n}\nboost {\n gamma = 10\n}\n";
    for (int q = 0; q < (ions ? 2 : 1); ++q)
        o << "species " << (q ? "p" : "e") << " {\n charge = " << (q ? 1 : -1) << "\n mass = " << (q ? 1836 : 1)
          << "\n density = " << (q ? np : ne) << "\n " << (q ? "ppc_x = 2\n ppc_y = 2" : ppe) << "\n density_function = " << (q ? fp : fe) << "\n}\n";
    o << "output {\n dir = test_out_boost_nci\n every = 100000\n}\n";
    return o.str();
}

void boost_neutral(Comm& comm) {
    const char* ramp = "x < 20 um ? x / (20 um) : 1";
    auto refused = [&](const std::string& deck, std::string* msg = nullptr) {
        try { deck_config(deck, "boost_test_tmp.deck"); } catch (const std::exception& e) { if (msg) *msg = e.what(); return true; }
        return false;
    };
    std::string m;
    const bool exact = !refused(neutral_deck(3.118936e-4, 3.118936e-4, ramp, ramp, 0.1));
    const bool round = refused(neutral_deck(3.11893e-4, 3.118936e-4, ramp, ramp, 0.1), &m);
    const bool named = m.find("3.118936e-04") != std::string::npos || m.find("0.0003118936") != std::string::npos;
    const bool ramps = refused(neutral_deck(3.118936e-4, 3.118936e-4, ramp, "x < 10 um ? x / (10 um) : 1", 0.1));
    const bool ppc = refused(neutral_deck(3.118936e-4, 3.118936e-4, ramp, ramp, 0.1, "ppc_x = 1\n ppc_y = 2"));
    const bool alone = refused(neutral_deck(3.118936e-4, 3.118936e-4, ramp, ramp, 0.1, "ppc_x = 2\n ppc_y = 2", false));
    Config c = deck_config(neutral_deck(3.118936e-4, 3.118936e-4, ramp, ramp, 0.1), "boost_test_tmp.deck");
    c.species[0].density = 3.11893e-4;
    const double q1 = boost_net_charge(c), want1 = (3.118936e-4 - 3.11893e-4) / (3.118936e-4 + 3.11893e-4);
    check("boost, neutral injection: exact deck accepted; rounded density (message names 3.118936e-4), other ramp, other ppc, "
          "electrons alone refused; boost_net_charge of the rounded density",
          exact && round && named && ramps && ppc && alone && std::abs(q1 / want1 - 1) < 1e-6,
          std::string(exact ? "" : "exact refused; ") + (round ? "" : "rounded accepted; ") + (named ? "" : "neutral density not named; ") +
              (ramps ? "" : "ramps accepted; ") + (ppc ? "" : "ppc accepted; ") + (alone ? "" : "electrons alone accepted; ") +
              fmt("rounded %.4e", q1) + fmt(" (want %.4e)", want1));
}

// B8. Neutral boosted injection keeps the Gauss law at round-off: 2D, 3D and RZ, filter_passes 0, 1, 2, warm electrons and the
//     He / N ions of a gas-mixture deck in partly ionized states (He charge_state 1 and N charge_state 3, both ionization = adk;
//     no laser, so they stay), electrons n_e = n_He + 3 n_N, a 20 um lab ramp, the window: the plasma enters at the injection
//     plane and leaves through the open left edge (2D: 192 cells crossed in ~96 of 171 steps).
void boost_neutral_gauss(Comm& comm) {
    for (int geo = 0; geo < 3; ++geo)
        for (int passes = 0; passes <= 2; ++passes) {
            std::ostringstream o;
            o.precision(12);
            o << "grid {\n lambda0 = 1 um\n";
            if (geo == 0) o << " geometry = cartesian2d\n nx = 192\n ny = 16\n dx = 0.03125 um\n dy = 0.4 um\n boundary_x = open\n t_end = 7 ps\n";
            if (geo == 1) o << " nx = 96\n ny = 8\n nz = 8\n dx = 0.03125 um\n dy = 0.4 um\n dz = 0.4 um\n boundary_x = open\n t_end = 3.5 ps\n";
            if (geo == 2) o << " geometry = rz\n modes = 2\n nz = 192\n nr = 8\n dz = 0.03125 um\n dr = 0.4 um\n boundary_z = open\n t_end = 7 ps\n";
            o << "}\nnumerics {\n solver = irip\n shape = 2\n filter_passes = " << passes << "\n}\nwindow {\n start = 0 fs\n}\nboost {\n gamma = 10\n}\n";
            const char* x = geo == 2 ? "z" : "x";
            const std::string prof = std::string(" density_function = ") + x + " < 20 um ? " + x + " / (20 um) : 1\n";
            const std::string ppc = geo == 2 ? " ppc_z = 2\n ppc_r = 2\n ppc_theta = 4\n" : geo == 1 ? " ppc_x = 2\n ppc_y = 1\n ppc_z = 1\n"
                                                                                                : " ppc_x = 2\n ppc_y = 2\n";
            o << "species e {\n charge = -1\n mass = 1\n density = " << 0.003 + 3 * 0.0004 << "\n uth_x = 1e-2\n uth_y = 1e-2\n uth_z = 1e-2\n"
              << ppc << prof << "}\n";
            o << "species He {\n element = He\n charge_state = 1\n ionization = adk\n ionization_electrons = e\n density = 0.003\n" << ppc << prof << "}\n";
            o << "species N {\n element = N\n charge_state = 3\n ionization = adk\n ionization_electrons = e\n density = 0.0004\n" << ppc << prof << "}\n";
            o << "output {\n dir = test_out_boost_nci\n every = 100000\n}\n";
            Config c = deck_config(o.str(), "boost_test_tmp.deck");
            Simulation sim(c, comm);
            sim.init();
            double g = 0, n = 0, in = 0;
            while (!sim.finished()) {
                sim.step();
                if (sim.istep % 5 == 0) g = std::max(g, sim.gauss_drift());
            }
            for (const Species& sp : sim.species) in += double(sp.size());
            comm.allreduce_sum(&in, 1);
            n = double(sim.istep);
            check(std::string("boost, neutral injection of e, He+ and N3+ with a ramp (") + (geo == 0 ? "2D" : geo == 1 ? "3D" : "RZ") +
                      ", filter " + std::to_string(passes) + "): Gauss law at round-off",
                  g < 1e-12 && in > 0, fmt("max|dGauss| %.1e", g) + fmt(" over %.0f steps", n) + fmt(", %.0f particles", in));
        }
}

#ifdef RIP_HDF5
// B9. The lab snapshots across checkpoint restarts (decision 116): an RZ LWFA in a boosted frame (gamma 2, 2 modes, RIP-CC,
//     rz_deposit = chord, shape 2, filter 2; lab fields at 3 lab times, lab particles of the electrons, lab_beams.txt). The
//     uninterrupted run against the same run stopped after a checkpoint (the stopped run goes on 7 steps, writing files as a
//     run killed later would) and restarted once, and twice: every lab file byte for byte. The runs use one OpenMP thread:
//     the RZ deposit adds with atomics, so with several threads even two uninterrupted runs differ at round-off (1e-15 of the
//     fields); on the GPU the same holds, the files are compared to 1e-8 there. Then a restart from a checkpoint whose lab
//     times differ (no usable state): the snapshots that had begun are written with complete 0, the others as usual.
// Largest difference of two lab files relative to the largest value of the first: float64 records (.bin) or the numbers of a text
// file (floor 1e-6 as lcompare in tests/mpi.sh); 1e300 for another size or other words.
double file_diff(const std::string& a, const std::string& b, bool bin) {
    if (bin) {
        if (a.size() != b.size() || a.size() % 8) return 1e300;
        std::vector<double> u(a.size() / 8), v(b.size() / 8);
        std::memcpy(u.data(), a.data(), a.size());
        std::memcpy(v.data(), b.data(), b.size());
        double m = 0, d = 0;
        for (size_t i = 0; i < u.size(); ++i) { m = std::max(m, std::abs(u[i])); d = std::max(d, std::abs(u[i] - v[i])); }
        return d / std::max(m, 1e-300);
    }
    std::istringstream sa(a), sb(b);
    std::string x, y;
    double d = 0;
    while (sa >> x) {
        if (!(sb >> y)) return 1e300;
        char* ea; char* eb;
        const double p = std::strtod(x.c_str(), &ea), q = std::strtod(y.c_str(), &eb);
        if (*ea || *eb) { if (x != y) return 1e300; continue; }
        d = std::max(d, std::abs(p - q) / std::max(std::abs(p), 1e-6));
    }
    return sb >> y ? 1e300 : d;
}
std::string file_text(const std::string& p) {
    std::string t;
    if (FILE* f = std::fopen(p.c_str(), "rb")) {
        char b[65536];
        for (size_t n; (n = std::fread(b, 1, sizeof b, f)) > 0;) t.append(b, n);
        std::fclose(f);
    }
    return t;
}
std::string labrst_deck(const std::string& dir, double t_first_fs) {
    std::ostringstream o;
    o.precision(15);
    o << "grid {\n geometry = rz\n modes = 2\n lambda0 = 0.8 um\n nz = 128\n nr = 24\n dz = 0.05 um\n dr = 0.3 um\n boundary_z = open\n"
         " t_end = 95 fs\n}\nboost {\n gamma = 2\n}\n"
      << "lab_diagnostics {\n t_first = " << t_first_fs << " fs\n t_every = 15 fs\n count = 3\n species = electrons\n gamma_min = 1\n"
         " particles = true\n}\n"
         "numerics {\n solver = rip_cc\n rz_deposit = chord\n shape = 2\n filter_passes = 2\n pusher = higuera_cary\n}\n"
         "laser {\n a0 = 2.5\n polarization = y\n duration = 10 fs\n waist = 4 um\n}\nwindow {\n start = 40 fs\n velocity = 1\n}\n";
    for (int q = 0; q < 2; ++q)
        o << "species " << (q ? "protons" : "electrons") << " {\n charge = " << (q ? 1 : -1) << "\n mass = " << (q ? 1836 : 1)
          << "\n density = 0.01\n zmin = 4 um\n ppc_r = 2\n ppc_theta = 4\n}\n";
    o << "output {\n dir = " << dir << "\n every = 100000\n}\n";
    return o.str();
}
void boost_lab_restart(Comm& comm) {
    const std::vector<std::string> files = {"lab_beams.txt", "lab_fields/snap_000.bin", "lab_fields/snap_000.txt", "lab_fields/snap_001.bin",
        "lab_fields/snap_001.txt", "lab_fields/snap_002.bin", "lab_fields/snap_002.txt", "lab_particles/electrons_000.bin",
        "lab_particles/electrons_000.txt", "lab_particles/electrons_001.bin", "lab_particles/electrons_001.txt",
        "lab_particles/electrons_002.bin", "lab_particles/electrons_002.txt"};
    auto run = [&](const std::string& dir, double t_first, const std::string& from, long stop) {   // stop >= 0: checkpoint there
        Config c = deck_config(labrst_deck(dir, t_first), "boost_test_tmp.deck");
        if (comm.rank == 0 && from.empty()) {
            make_dirs(dir);
            for (const std::string& f : files) std::remove((dir + "/" + f).c_str());   // no files of an earlier test run
            for (const std::string& f : list_dir(dir + "/checkpoints"))               // nor its checkpoints (checkpoint.keep
                std::remove((dir + "/checkpoints/" + f).c_str());                    // would delete the new ones as older)
        }
        comm.barrier();
        Simulation s(c, comm);
        s.init(from);
        std::string path;
        int open = 0;                                // snapshots with planes in the checkpoint this run started from
        if (!from.empty()) {
            OpState o;
            read_op_state(s, "lab_fields", {"xl_0", "xl_1", "xl_2"}, o);
            for (const auto& a : o.arrays) open += !a.second.empty();
        }
        while (!s.finished()) {
            s.step();
            if (s.istep == stop) path = write_checkpoint(s);
            if (stop >= 0 && s.istep == stop + 7) break;       // stopped (killed) after some more steps
        }
        if (stop < 0) s.pipeline.finish(s);
        comm.barrier();
        return std::make_pair(path, std::make_pair(open, s.istep));
    };
#ifdef _OPENMP
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);                          // deterministic sums (the RZ deposit adds with atomics)
#endif
    const auto a = run("test_out_labrst_a", 50, "", -1);
    // checkpoints while snapshots are open: the first lab time (50 fs) is reached in the box from ~ 0.5 n on, the last (80 fs)
    // from ~ 0.6 n; at 9/16 n the first is open and the last not begun, at 3/4 n all three are open
    const long n = a.second.second, n1 = 9 * n / 16, n2 = 3 * n / 4;
    const auto c1 = run("test_out_labrst_c", 50, "", n1);                      // restarted once
    const auto c2 = run("test_out_labrst_c", 50, c1.first, -1);
    const auto b1 = run("test_out_labrst_b", 50, "", n1);                      // restarted twice
    const auto b2 = run("test_out_labrst_b", 50, b1.first, n2);
    const auto b3 = run("test_out_labrst_b", 50, b2.first, -1);
    const auto f1 = run("test_out_labrst_f", 50, "", n1);                      // restarted with other lab times
    const auto f2 = run("test_out_labrst_f", 50.001, f1.first, -1);
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    if (comm.rank != 0) return;
    int same_c = 0, same_b = 0;
    size_t bytes = 0;
    double worst = 0;
    std::string differ;
    for (const std::string& f : files) {
        const std::string ta = file_text("test_out_labrst_a/" + f), tc = file_text("test_out_labrst_c/" + f),
                          tb = file_text("test_out_labrst_b/" + f);
        const bool bin = f.size() > 4 && f.compare(f.size() - 4, 4, ".bin") == 0;
        bytes += ta.size();
        if (ta.empty()) { differ += " missing:" + f; worst = 1e300; continue; }
        if (ta == tc) ++same_c; else { differ += " c:" + f; worst = std::max(worst, file_diff(ta, tc, bin)); }
        if (ta == tb) ++same_b; else { differ += " b:" + f; worst = std::max(worst, file_diff(ta, tb, bin)); }
    }
    const bool exact = !RIP_GPU;
    check("boost, lab snapshots across restarts (RZ RIP-CC, chord, filter 2): restarted once and twice = the uninterrupted run, "
          "byte for byte (lab fields, particles, lab_beams.txt; GPU: to 1e-8)",
          (exact ? same_c == int(files.size()) && same_b == int(files.size()) : worst < 1e-8) && c2.second.first > 0 &&
              b2.second.first > 0 && b3.second.first > 0,
          fmt("%.0f steps, checkpoints at ", double(n)) + fmt("%.0f and %.0f", double(n1), double(n2)) +
              fmt(" with %.0f, %.0f", c2.second.first, b2.second.first) + fmt(" / %.0f open field snapshots; ", b3.second.first) +
              fmt("%.0f + %.0f of ", same_c, same_b) + fmt("%.0f files identical (%.0f bytes)", double(files.size()), double(bytes)) +
              (worst > 0 ? fmt(", max difference %.1e", worst) : std::string()) + differ);
    // other lab times: the snapshots open at the checkpoint complete 0, the ones not begun complete 1
    int c0 = 0, c1n = 0;
    for (int k = 0; k < 3; ++k) {
        const std::string m = file_text("test_out_labrst_f/lab_fields/snap_00" + std::to_string(k) + ".txt");
        const std::string p = file_text("test_out_labrst_f/lab_particles/electrons_00" + std::to_string(k) + ".txt");
        c0 += (m.find("complete 0") != std::string::npos) + (p.find("complete 0") != std::string::npos);
        c1n += (m.find("complete 1") != std::string::npos) + (p.find("complete 1") != std::string::npos);
    }
    const std::string bt = file_text("test_out_labrst_f/lab_beams.txt");
    check("boost, restart without usable lab state (other lab times): snapshots begun before the checkpoint written with complete 0",
          f2.second.second > n1 && f2.second.first > 0 && c0 >= 2 && c1n >= 1 && c0 + c1n == 6 &&
              bt.find(" electrons 0 ") != std::string::npos,
          fmt("%.0f files complete 0, %.0f complete 1", c0, c1n));
}
#endif

// B7b. Lattice groups (decision 114; Alexander: "each pre-ionized ion species brings its own electrons with the same ppc and
//      matching weight"): the injection is neutral at every particle position when every lattice group (species sharing ppc,
//      bounds and slab) is neutral on its own; groups may differ in ppc and may overlap. RZ, gamma 10, a 20 um lab ramp, iRIP,
//      shape 2, filter 2, modes 2. Layouts: (zones) an inner zone r < 1.6 um with e + He2+ at ppc 2 x 2 x 8 and an outer zone
//      1.6 .. 3.2 um with e_o + He2+_o at ppc 1 x 1 x 6 (the layout of a wide gas-mixture deck); (overlap) He2+ with
//      its electrons at ppc 2 x 2 x 8 and N3+ (ionization = adk, no laser) with its electrons at ppc 1 x 1 x 6 in the same cells.
//      (a) Config::check accepts both, and refuses a zone whose electron density is rounded (3.11893e-4 for 3.118936e-4), with a
//      message naming that group; (b) the runs keep the Gauss law at round-off everywhere, the zone boundary included, while the
//      plasma enters and crosses the box; every species is present.
std::string group_deck(int layout, double ne_outer) {
    std::ostringstream o;
    o.precision(12);
    o << "grid {\n lambda0 = 1 um\n geometry = rz\n modes = 2\n nz = 192\n nr = 8\n dz = 0.03125 um\n dr = 0.4 um\n boundary_z = open\n"
         " t_end = 7 ps\n}\nnumerics {\n solver = irip\n shape = 2\n filter_passes = 2\n}\nwindow {\n start = 0 fs\n}\nboost {\n gamma = 10\n}\n";
    const std::string prof = " density_function = z < 20 um ? z / (20 um) : 1\n";
    const std::string fine = " ppc_z = 2\n ppc_r = 2\n ppc_theta = 8\n", coarse = " ppc_z = 1\n ppc_r = 1\n ppc_theta = 6\n";
    const double nhe = 1.42706e-3, nn = 8.8272e-5;
    if (layout == 0) {
        for (int z = 0; z < 2; ++z) {
            const std::string sfx = z ? "_o" : "", bnd = z ? " rmin = 1.6 um\n" : " rmax = 1.6 um\n", ppc = z ? coarse : fine;
            o << "species e" << sfx << " {\n charge = -1\n mass = 1\n density = " << (z ? ne_outer : 2 * nhe) << "\n" << bnd << ppc << prof << "}\n";
            o << "species He" << sfx << " {\n element = He\n charge_state = 2\n density = " << nhe << "\n" << bnd << ppc << prof << "}\n";
        }
    } else {
        o << "species eHe {\n charge = -1\n mass = 1\n density = " << 2 * nhe << "\n" << fine << prof << "}\n";
        o << "species He {\n element = He\n charge_state = 2\n density = " << nhe << "\n" << fine << prof << "}\n";
        o << "species eN {\n charge = -1\n mass = 1\n density = " << 3 * nn << "\n" << coarse << prof << "}\n";
        o << "species N {\n element = N\n charge_state = 3\n ionization = adk\n ionization_electrons = eN\n density = " << nn << "\n"
          << coarse << prof << "}\n";
    }
    o << "output {\n dir = test_out_boost_nci\n every = 100000\n}\n";
    return o.str();
}

void boost_groups(Comm& comm) {
    auto refused = [&](const std::string& deck, std::string* msg = nullptr) {
        try { deck_config(deck, "boost_test_tmp.deck"); } catch (const std::exception& e) { if (msg) *msg = e.what(); return true; }
        return false;
    };
    std::string m;
    const bool zones = !refused(group_deck(0, 2 * 1.42706e-3)), overlap = !refused(group_deck(1, 0));
    const bool round = refused(group_deck(0, 2 * 1.42706e-3 * (1 - 2e-6)), &m);
    const bool named = m.find("lattice group of") != std::string::npos && m.find(" e_o (") != std::string::npos &&
                       m.find(" He_o (") != std::string::npos && m.find(" e (") == std::string::npos;
    check("boost, lattice groups: two radial zones with different ppc and two overlapping groups (He2+, N3+ with their own electrons) "
          "accepted; a non-neutral zone refused, message names its group",
          zones && overlap && round && named,
          std::string(zones ? "" : "zones refused; ") + (overlap ? "" : "overlap refused; ") + (round ? "" : "rounded zone accepted; ") +
              (named ? "" : "group not named: " + m.substr(0, 300)));
    for (int layout = 0; layout < 2; ++layout) {
        Config c = deck_config(group_deck(layout, 2 * 1.42706e-3), "boost_test_tmp.deck");
        Simulation sim(c, comm);
        sim.init();
        double g = 0;
        while (!sim.finished()) {
            sim.step();
            if (sim.istep % 5 == 0) g = std::max(g, sim.gauss_drift());
        }
        double nmin = 1e300;
        for (const Species& sp : sim.species) {
            double n = double(sp.size());
            comm.allreduce_sum(&n, 1);
            nmin = std::min(nmin, n);
        }
        check(std::string("boost, lattice groups (") + (layout ? "overlapping He2+ / N3+ groups, ppc 2x2x8 and 1x1x6" :
                                                                 "two radial zones, ppc 2x2x8 and 1x1x6") +
                  ", RZ, filter 2): Gauss law at round-off, zone boundary included",
              g < 1e-12 && nmin > 0, fmt("max|dGauss| %.1e", g) + fmt(" over %.0f steps", double(sim.istep)) + fmt(", fewest particles %.0f", nmin));
    }
}

// B10 (decision 118): the loading checks of a boosted beam. A 1.5 um beam moving against the boost (the QED decks' beam, lab
// gamma 1957, ppc_x 2, dx 0.04 um) is s^2 shorter in boosted cells (s = gamma (1 + beta)): n = 1.5 um ppc_x / (s^2 dx) planes,
// 0.765 at gamma 5 (an error with the ppc_x needed), 5.385 at gamma 2 (a warning; the loaded charge off by more than 1 %),
// with ppc_x 28 at gamma 2 75.4 planes (silent: the charge within 1 % of the deck's).
void boost_beam_checks(Comm& comm) {
    auto deck = [](double g, int ppc) {
        std::ostringstream o;
        o << "grid {\n lambda0 = 0.8 um\n nx = 64\n ny = 8\n nz = 8\n dx = 0.04 um\n dy = 0.16 um\n dz = 0.16 um\n boundary_x = open\n"
          << " t_end = 20 fs\n}\nnumerics {\n solver = irip\n}\nboost {\n gamma = " << g << "\n}\n"
          << "species e {\n charge = -1\n mass = 1\n density = 1e-4\n ux = -1957\n xmin = 1 um\n xmax = 2.5 um\n ymin = 0.32 um\n"
          << " ymax = 0.96 um\n zmin = 0.32 um\n zmax = 0.96 um\n ppc_x = " << ppc << "\n}\n"
          << "output {\n dir = test_out_boost_load\n every = 100000\n}\n";
        return o.str();
    };
    auto planes = [](double g, int ppc) {               // n = L a ppc / dx', a = 1 / (g (1 - b v)), dx' = g (1 + b) dx
        const double b = std::sqrt(1 - 1 / (g * g)), v = -1957 / std::sqrt(1 + 1957.0 * 1957.0);
        return 1.5 * ppc / (g * (1 - b * v) * g * (1 + b) * 0.04);
    };
    // gamma 5: an error naming the planes and the ppc_x needed
    std::string msg;
    try { deck_config(deck(5, 2), "boost_test_tmp.deck"); } catch (const std::exception& e) { msg = e.what(); }
    const double n5 = planes(5, 2);
    char want[64];
    std::snprintf(want, sizeof want, "holds %.3g planes", n5);
    const int p1 = int(std::ceil(2 / n5)), p20 = int(std::ceil(40 / n5));
    const bool err = msg.find(want) != std::string::npos && msg.find("ppc_x >= " + std::to_string(p1)) != std::string::npos &&
                     msg.find("(" + std::to_string(p20) + " for 20") != std::string::npos;
    // gamma 2, ppc_x 2: a warning, and the loaded charge off; ppc_x 28: silent, the charge right
    double n2[2] = {0, 0}, rel[2] = {0, 0};
    bool warn[2] = {false, false};
    const int pp[2] = {2, 28};
    for (int k = 0; k < 2; ++k) {
        Config c = deck_config(deck(2, pp[k]), "boost_test_tmp.deck");
        n2[k] = boost_beam_planes(c, c.species[0]);
        warn[k] = !boost_beam_warning(c).empty();
        Simulation s(c, comm);
        s.init();
        while (!s.loaded[0] && !s.finished()) s.step();
        double q = 0;
        const std::vector<double> w = s.species[0].get(Species::W);
        for (double x : w) q += x;
        comm.allreduce_sum(&q, 1);
        // the deck's charge: the transverse box is cell-aligned (4 x 4 cells), so n 1.5 x 0.64 x 0.64 um^3
        const double k0 = 2 * M_PI / 0.8e-6, um = 1e-6 * k0;
        rel[k] = q / (1e-4 * 1.5 * 0.64 * 0.64 * um * um * um) - 1;
        const double rc = boost_beam_charge_check(s, 0);
        rel[k] = std::abs(rc - rel[k]) < 1e-9 ? rel[k] : 1e9;   // the check's deck charge = the analytic one
    }
    check("boost B10: boosted-beam loading checks: error below 1 lattice plane (gamma 5), warning below 20 and the loaded charge "
          "off by > 1 % (gamma 2, ppc_x 2), silent and right with ppc_x 28",
          err && std::abs(n2[0] / planes(2, 2) - 1) < 1e-12 && std::abs(n2[1] / planes(2, 28) - 1) < 1e-12 && warn[0] && !warn[1] &&
              std::abs(rel[0]) > 0.01 && std::abs(rel[0]) < 1 && std::abs(rel[1]) < 0.01,
          fmt("planes gamma 5 %.4f (analytic %.4f, error names ppc_x >= ", n5, planes(5, 2)) + std::to_string(p1) +
              fmt("); gamma 2 %.3f (analytic %.3f)", n2[0], planes(2, 2)) + fmt(", loaded charge %+.1f %%", 100 * rel[0]) +
              fmt("; ppc_x 28: %.1f planes, %+.2f %%", n2[1], 100 * rel[1]));
}

}  // namespace

void test_boost(Comm& comm) {
    boost_beam_checks(comm);
#ifdef RIP_HDF5
    if (std::getenv("BOOST_RESTART_ONLY")) { boost_lab_restart(comm); return; }   // development: B9 alone
#endif
    boost_beam_warning();
    if (std::getenv("BOOST_NCI_ONLY")) { boost_stability(comm); return; }   // development: B6 alone
    if (std::getenv("BOOST_NEUTRAL_ONLY")) { boost_neutral(comm); boost_groups(comm); boost_neutral_gauss(comm); return; }   // development: B7, B7b, B8
    if (std::getenv("BOOST_ION_ONLY")) { boost_ionization(comm); return; }   // development: B4 (and B5) alone
    boost_vacuum(comm);
    boost_wake(comm, true);
    boost_wake(comm, false);
    boost_gauss_beam(comm);
    boost_lab_select(comm);
#ifdef RIP_HDF5
    boost_lab_restart(comm);
#else
    std::printf("boost, lab snapshots across restarts: skipped (built without HDF5)\n");
#endif
    boost_ionization(comm);
    boost_stability(comm);
    boost_neutral(comm);
    boost_groups(comm);
    boost_neutral_gauss(comm);
}
