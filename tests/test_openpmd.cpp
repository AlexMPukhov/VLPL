// openPMD output and checkpoint/restart (docs/specs/openpmd.md). Needs a build with HDF5.
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <string>
#include "common.hpp"
#include "sim.hpp"
#include "deck.hpp"
#ifdef RIP_HDF5
#include <hdf5.h>

namespace {
// Plain HDF5 reads (not the writer's helpers): whole datasets and attributes as double.
std::vector<double> read_ds(hid_t f, const std::string& path, std::vector<hsize_t>& dims) {
    const hid_t ds = H5Dopen2(f, path.c_str(), H5P_DEFAULT);
    if (ds < 0) return {};
    const hid_t sp = H5Dget_space(ds);
    dims.assign(size_t(H5Sget_simple_extent_ndims(sp)), 0);
    H5Sget_simple_extent_dims(sp, dims.data(), nullptr);
    std::vector<double> v(size_t(H5Sget_simple_extent_npoints(sp)));
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Sclose(sp); H5Dclose(ds);
    return v;
}
std::vector<double> read_attr(hid_t f, const std::string& obj, const std::string& name) {
    const hid_t a = H5Aopen_by_name(f, obj.c_str(), name.c_str(), H5P_DEFAULT, H5P_DEFAULT);
    if (a < 0) return {};
    const hid_t sp = H5Aget_space(a);
    std::vector<double> v(size_t(H5Sget_simple_extent_npoints(sp)));
    H5Aread(a, H5T_NATIVE_DOUBLE, v.data());
    H5Sclose(sp); H5Aclose(a);
    return v;
}
double rel(double a, double b) { return std::abs(a - b) / std::max(std::abs(b), 1e-300); }

// The deck of both tests: moving window, laser, CPML in y and z, a hot plasma with an up-ramp entering
// at the right edge, two beams loaded later (before and after the checkpoint), beam diagnostics.
Config deck(const std::string& dir) {
    Config c;
    const double dx = 2 * M_PI / 16;
    c.N[0] = 48; c.N[1] = 24; c.N[2] = 24;
    c.h[0] = dx; c.h[1] = 2.5 * dx; c.h[2] = 2.5 * dx;
    c.periodic[0] = false;
    for (int a = 1; a < 3; ++a) { c.pml.on[a] = true; c.periodic[a] = false; }
    c.pml.cells = 5;
    c.order = 2;
    c.laser.enabled = true; c.laser.a0 = 2; c.laser.tau = 4; c.laser.tc = 8; c.laser.w0 = 5; c.laser.w0z = 5;
    c.laser.yc = 0.5 * c.N[1] * c.h[1]; c.laser.zc = 0.5 * c.N[2] * c.h[2]; c.laser.pol = 'z';
    c.window.enabled = true; c.window.start = 16;
    c.t_end = 46;
    for (int s = 0; s < 4; ++s) {
        SpeciesConfig e;
        const char* names[4] = {"electrons", "protons", "beam1", "beam2"};
        e.name = names[s]; e.q = s == 1 ? 1 : -1; e.m = s == 1 ? 1836 : 1;
        for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
        if (s < 2) {
            e.density = 0.02; e.rmin[0] = 20;                     // ahead of the box until the window moves
            for (int a = 0; a < 3; ++a) e.uth[a] = s ? 0.001 : 0.05;
            e.density_fn = Expr("x < 24 ? (x - 20) / 4 : 1", {"x", "y", "z"}, UnitFn());
        } else {
            e.density = 0.002; e.u0[0] = 10; e.t_start = s == 2 ? 20 : 32;
            e.rmin[0] = s == 2 ? 8 : 26; e.rmax[0] = e.rmin[0] + 2;      // inside the window at t_start
            for (int a = 1; a < 3; ++a) { e.rmin[a] = 0.4 * c.N[a] * c.h[a]; e.rmax[a] = 0.6 * c.N[a] * c.h[a]; }
            e.ppc[0] = 2;
        }
        c.species.push_back(e);
    }
    BeamDiagConfig b;
    b.name = "b1"; b.species = "beam1"; b.every = 10; b.gamma_min = 2; b.bins = 20; b.e_max_mev = 20;
    c.diags.push_back(b);
    // bit-exact on the CPU for any thread count: the coloured tiles sorted every step (no atomics)
    if (!RIP_GPU) { c.deposit = Config::Deposit::Tiled; c.sort_every = 1; }
    c.dir = dir;
    c.check();
    return c;
}

std::string text(const std::string& path) {
    std::string s;
    if (FILE* f = std::fopen(path.c_str(), "r")) {
        for (int ch; (ch = std::fgetc(f)) != EOF;) s.push_back(char(ch));
        std::fclose(f);
    }
    return s;
}

// Largest difference of the state of two simulations (fields, CPML memories, particles), relative
// to the largest value of each array; exact = 0. Particle counts must agree.
double state_diff(Simulation& a, Simulation& b, bool& same_counts) {
    double worst = 0;
    auto cmp = [&](const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size()) { same_counts = false; return; }
        double m = 0, d = 0;
        for (size_t q = 0; q < x.size(); ++q) { m = std::max(m, std::abs(x[q])); d = std::max(d, std::abs(x[q] - y[q])); }
        if (d > 0) worst = std::max(worst, d / m);
    };
    Fields &A = *a.fields, &B = *b.fields;
    const Field* fa[6] = {&A.ex, &A.ey, &A.ez, &A.bx, &A.by, &A.bz};
    const Field* fb[6] = {&B.ex, &B.ey, &B.ez, &B.bx, &B.by, &B.bz};
    for (int q = 0; q < 6; ++q) cmp(fa[q]->host(), fb[q]->host());
    const std::vector<Field*> pa = a.pml()->psi_fields(), pb = b.pml()->psi_fields();
    if (pa.size() != pb.size()) same_counts = false;
    else
        for (size_t i = 0; i < pa.size(); ++i) {     // owned cells and the transport's i = -1 (guards in y, z unused)
            const std::vector<double> x = pa[i]->host(), y = pb[i]->host();
            std::vector<double> u, v;
            for (int k = 0; k < a.dom.n[2]; ++k)
                for (int j = 0; j < a.dom.n[1]; ++j)
                    for (int ii = -1; ii < a.dom.n[0]; ++ii) { u.push_back(x[pa[i]->idx(ii, j, k)]); v.push_back(y[pb[i]->idx(ii, j, k)]); }
            cmp(u, v);
        }
    for (size_t s = 0; s < a.species.size(); ++s)
        for (int sl = 0; sl < Species::NC; ++sl) cmp(a.species[s].get(sl), b.species[s].get(sl));
    return worst;
}
// O-AX (2026-11-21, Alexander: "m = 1 Ez and Jz must be physically correct everywhere in the code and in the openPMD saves"): an RZ run with
// a warm plasma on the axis (2 modes, 12 steps); in the code J_z and E_z of mode 1 are zero on the axis node; in the output E_z, J_z, rho of
// mode 1 are zero on the axis and the transverse components carry the regular axis values of a uniform transverse vector:
// E_phi1(0) = i E_r1(dr/2), J_phi1(0) = i J_r1(dr/2), B_r1(0) = -i B_phi1(dr/2); in the thetaMode coefficients (C = 2 Re F, S = -2 Im F)
// C_t(0) = S_r(dr/2), S_t(0) = -C_r(dr/2), and for B: C_r(0) = -S_t(dr/2), S_r(0) = C_t(dr/2).
static void openpmd_rz_axis(Comm& comm) {
    if (comm.size > 1) return;
    const char* path = "rz_axis_out_tmp.deck";
    {
        std::FILE* f = std::fopen(path, "w");
        std::fprintf(f, "grid {\n geometry = rz\n modes = 2\n lambda0 = 0.8 um\n nz = 32\n nr = 16\n dz = 0.05 um\n dr = 0.1 um\n boundary_z = periodic\n t_end = 2 fs\n}\n"
                        "numerics {\n solver = irip\n shape = 2\n filter_passes = 2\n}\n"
                        "species e {\n charge = -1\n mass = 1\n density = 0.05\n ppc_z = 2\n ppc_r = 2\n ppc_theta = 8\n uth_x = 0.05\n uth_y = 0.05\n uth_z = 0.05\n}\n"
                        "species i {\n charge = 1\n mass = 1836\n density = 0.05\n ppc_z = 2\n ppc_r = 2\n ppc_theta = 8\n}\n"
                        "output {\n dir = rz_axis_out_tmp\n openpmd_fields = E B J rho\n openpmd_precision = double\n}\n");
        std::fclose(f);
    }
    Config c = Config::from_deck(Deck::read(path));
    Simulation s(c, comm);
    s.init();
    while (!s.finished()) s.step();
    // in the code
    const std::vector<double> jz = s.rz->jz.host(), ez = s.rz->ez.host();
    double cz = 0, cmax = 0;
    for (int i = 0; i < s.dom.n[0]; ++i) {
        for (int k = 1; k <= 2; ++k) cz = std::max({cz, std::abs(jz[s.rz->jz.idx(i, 0, k)]), std::abs(ez[s.rz->ez.idx(i, 0, k)])});
        for (int k = 1; k <= 2; ++k) cmax = std::max(cmax, std::abs(jz[s.rz->jz.idx(i, 1, k)]));
    }
    write_openpmd(s);
    char fn[160];
    std::snprintf(fn, sizeof fn, "%s/openpmd/data_%06ld.h5", c.dir.c_str(), s.istep);
    const hid_t f = H5Fopen(fn, H5F_ACC_RDONLY, H5P_DEFAULT);
    const std::string it = "/data/" + std::to_string(s.istep) + "/meshes/";
    std::vector<hsize_t> dm;
    auto rd = [&](const std::string& p) { return read_ds(f, it + p, dm); };
    const std::vector<double> Er = rd("E/r"), Et = rd("E/t"), Ez = rd("E/z"), Br = rd("B/r"), Bt = rd("B/t"), Jr = rd("J/r"), Jt = rd("J/t"),
                              Jz = rd("J/z"), rho = rd("rho");
    H5Fclose(f);
    const size_t nr = dm[1], nz = dm[2];
    auto at = [&](const std::vector<double>& v, size_t k, size_t j, size_t i) { return v[(k * nr + j) * nz + i]; };
    double zero = 0, reg = 0, scale = 0;
    for (size_t i = 0; i < nz; ++i) {
        for (size_t k = 1; k <= 2; ++k) zero = std::max({zero, std::abs(at(Ez, k, 0, i)), std::abs(at(Jz, k, 0, i)), std::abs(at(rho, k, 0, i))});
        reg = std::max({reg, std::abs(at(Et, 1, 0, i) - at(Er, 2, 0, i)), std::abs(at(Et, 2, 0, i) + at(Er, 1, 0, i)),
                        std::abs(at(Jt, 1, 0, i) - at(Jr, 2, 0, i)), std::abs(at(Jt, 2, 0, i) + at(Jr, 1, 0, i)),
                        std::abs(at(Br, 1, 0, i) + at(Bt, 2, 0, i)), std::abs(at(Br, 2, 0, i) - at(Bt, 1, 0, i))});
        scale = std::max({scale, std::abs(at(Er, 1, 0, i)), std::abs(at(Jr, 1, 0, i)), std::abs(at(Bt, 1, 0, i))});
    }
    std::printf("  O-AX: code: max |J_z1|, |E_z1| on the axis %.1e (J_z1 at node 1 %.2e); output: m = 1 E_z, J_z, rho on the axis %.1e, "
                "transverse axis values against the regular rule %.1e (scale %.2e)\n", cz, cmax, zero, reg, scale);
    check("openpmd RZ: m = 1 E_z and J_z zero on the axis in the code (J_z ~ r)", cz == 0 && cmax > 0, fmt("%.1e", cz));
    check("openpmd RZ: m = 1 axis values in the output: E_z, J_z, rho zero, E_phi, J_phi, B_r the regular values", zero == 0 && reg < 1e-12 * std::max(scale, 1e-300) + 1e-300 && scale > 0,
          fmt("zero %.1e", zero) + fmt(", regular %.1e", reg));
}


}  // namespace

// O1. Output of a short run (fields E B J rho rho_<species>, particles of one species), read back:
//     the values of this rank's block against the arrays of the run, the unitSI against the SI constants,
//     the staggering positions and the window offset; the per-species densities add up to rho.
static void openpmd_output(Comm& comm) {
    Config c = deck("test_out_openpmd");
    c.t_end = 22;                                                  // the window has moved, beam1 loaded
    c.openpmd.fields = {"E", "B", "J", "rho", "rho_electrons", "rho_protons", "rho_beam1", "rho_beam2"};
    c.openpmd.species = {"beam1"};
    c.openpmd.single = false;
    Simulation s(c, comm);
    s.init();
    while (!s.finished()) s.step();
    write_openpmd(s);
    char fn[96];
    std::snprintf(fn, sizeof fn, "%s/openpmd/data_%06ld.h5", c.dir.c_str(), s.istep);
    const hid_t f = H5Fopen(fn, H5F_ACC_RDONLY, H5P_DEFAULT);
    const std::string it = "/data/" + std::to_string(s.istep) + "/";
    const Domain& d = s.dom;
    // SI units, written out again from the constants (CODATA 2018)
    const double k0 = 2 * M_PI / c.lambda0, w0 = 299792458.0 * k0, me = 9.1093837015e-31, e = 1.602176634e-19,
                 eps0 = 8.8541878128e-12, nc = eps0 * me * w0 * w0 / (e * e);
    double err = 0, uerr = 0;
    auto block = [&](const std::string& name, const Field& fl, double& err) {     // this rank's owned block
        std::vector<hsize_t> dims;
        const std::vector<double> v = read_ds(f, it + "meshes/" + name, dims);
        if (dims.size() != 3 || dims[0] != hsize_t(d.N[2]) || dims[2] != hsize_t(d.N[0])) { err = 1; return; }
        const std::vector<double> h = fl.host();
        double m = 0, dd = 0;
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const double x = v[(size_t(d.off[2] + k) * d.N[1] + size_t(d.off[1] + j)) * d.N[0] + size_t(d.off[0] + i)];
                    m = std::max(m, std::abs(h[fl.idx(i, j, k)]));
                    dd = std::max(dd, std::abs(x - h[fl.idx(i, j, k)]));
                }
        err = std::max(err, m > 0 ? dd / m : (dd > 0 ? 1.0 : 0.0));
    };
    Fields& F = *s.fields;
    block("E/x", F.ex, err); block("E/y", F.ey, err); block("E/z", F.ez, err);
    block("B/x", F.bx, err); block("B/y", F.by, err); block("B/z", F.bz, err);
    block("J/x", F.jx, err); block("J/y", F.jy, err); block("J/z", F.jz, err);
    double rerr = 0;                                               // rho deposited again: summation order
    s.compute_rho();
    block("rho", F.rho, rerr);
    uerr = std::max(uerr, rel(read_attr(f, it + "meshes/E/x", "unitSI").at(0), me * 299792458.0 * w0 / e));
    uerr = std::max(uerr, rel(read_attr(f, it + "meshes/B/z", "unitSI").at(0), me * w0 / e));
    uerr = std::max(uerr, rel(read_attr(f, it + "meshes/J/y", "unitSI").at(0), e * nc * 299792458.0));
    uerr = std::max(uerr, rel(read_attr(f, it + "meshes/rho", "unitSI").at(0), e * nc));
    uerr = std::max(uerr, rel(read_attr(f, it + "meshes/E", "gridUnitSI").at(0), 1 / k0));
    uerr = std::max(uerr, rel(read_attr(f, "/data/" + std::to_string(s.istep), "timeUnitSI").at(0), 1 / w0));
    // staggering (z, y, x order) and the absolute x of index 0
    const std::vector<double> pey = read_attr(f, it + "meshes/E/y", "position"), pbx = read_attr(f, it + "meshes/B/x", "position"),
                              pjz = read_attr(f, it + "meshes/J/z", "position"), pr = read_attr(f, it + "meshes/rho", "position"),
                              go = read_attr(f, it + "meshes/B", "gridGlobalOffset");
    const bool stag = pey == std::vector<double>{0, 0.5, 0} && pbx == std::vector<double>{0.5, 0.5, 0} &&
                      pjz == std::vector<double>{0.5, 0, 0.5} && pr == std::vector<double>{0, 0, 0.5} && go.size() == 3 &&
                      go[2] == double(d.wx) * d.h[0];
    // rho = sum of the species densities
    std::vector<hsize_t> dims;
    const std::vector<double> rho = read_ds(f, it + "meshes/rho", dims);
    std::vector<double> sum(rho.size(), 0.0);
    for (const char* n : {"rho_electrons", "rho_protons", "rho_beam1", "rho_beam2"}) {
        const std::vector<double> v = read_ds(f, it + "meshes/" + n, dims);
        for (size_t q = 0; q < v.size() && q < sum.size(); ++q) sum[q] += v[q];
    }
    double rs = 0, rm = 0;
    for (size_t q = 0; q < rho.size(); ++q) { rm = std::max(rm, std::abs(rho[q])); rs = std::max(rs, std::abs(rho[q] - sum[q])); }
    // particles of beam1: count, this rank's segment (particlePatches) and the units
    const std::string pb = it + "particles/beam1/";
    const std::vector<double> px = read_ds(f, pb + "position/x", dims), wt = read_ds(f, pb + "weighting", dims),
                              uy = read_ds(f, pb + "momentum/y", dims), np = read_ds(f, pb + "particlePatches/numParticles", dims),
                              no = read_ds(f, pb + "particlePatches/numParticlesOffset", dims);
    const Species& bm = s.species[2];
    double ntot = double(bm.size());
    comm.allreduce_sum(&ntot, 1);
    double perr = 0;
    bool pok = double(px.size()) == ntot && ntot > 0 && np.size() == size_t(comm.size) && np[size_t(comm.rank)] == double(bm.size());
    if (pok) {
        const std::vector<double> x = bm.get(Species::X), w = bm.get(Species::W), u = bm.get(Species::UY);
        const size_t o = size_t(no[size_t(comm.rank)]);
        for (size_t q = 0; q < x.size(); ++q)
            perr = std::max({perr, rel(px[o + q], x[q]), rel(wt[o + q], w[q] * nc / (k0 * k0 * k0)), std::abs(uy[o + q] - u[q])});
    }
    uerr = std::max(uerr, rel(read_attr(f, pb + "momentum/x", "unitSI").at(0), me * 299792458.0));
    uerr = std::max(uerr, rel(read_attr(f, pb + "charge", "value").at(0) * read_attr(f, pb + "charge", "unitSI").at(0), -e));
    H5Fclose(f);
    check("openpmd: output read back = the arrays of the run, units, staggering",
          err == 0 && rerr < 1e-14 && uerr < 1e-15 && stag && rs <= 1e-14 * rm && pok && perr == 0,
          fmt("E B J %.1e", err) + fmt(", rho %.1e", rerr) + fmt(", units %.1e", uerr) + fmt(", rho - sum rho_s %.1e", rm > 0 ? rs / rm : 0.0) +
              fmt(", %.0f beam particles", ntot) + fmt(" (max error %.1e)", perr) + (stag ? "" : ", staggering WRONG"));
}

// O2. Restart: an uninterrupted run against one that writes a checkpoint, runs on 7 steps (as if it
//     stopped later) and is restarted from it: fields, CPML memories and all particles, also of the
//     beam loaded after the checkpoint, and the beam diagnostics file. Exact on the CPU (tiled deposit);
//     on the GPU the atomic deposit sums in another order (round-off).
static void openpmd_restart(Comm& comm) {
    Simulation a(deck("test_out_restart_a"), comm);
    a.init();
    while (!a.finished()) a.step();
    const Config cb = deck("test_out_restart_b");
    std::string path;
    long nchk = 0;
    {
        Simulation b(cb, comm);
        b.init();
        while (b.istep < 60) b.step();
        path = write_checkpoint(b);
        nchk = b.istep;
        for (int q = 0; q < 7; ++q) b.step();
    }
    Simulation r(cb, comm);
    r.init(path);
    const bool at = r.istep == nchk && r.restart_step == nchk && r.dom.wx > 0 && r.loaded[2] && !r.loaded[3];
    while (!r.finished()) r.step();
    bool counts = true;
    const double diff = state_diff(a, r, counts);
    double nb2 = double(r.species[3].size());
    comm.allreduce_sum(&nb2, 1);
    bool files = true;
    if (comm.rank == 0) {
        const std::string fa = text("test_out_restart_a/beam_b1.txt"), fb = text("test_out_restart_b/beam_b1.txt");
        files = !fa.empty() && fa == fb;
        files = files && text("test_out_restart_a/spectrum_b1.txt") == text("test_out_restart_b/spectrum_b1.txt");
    }
    const bool exact = !RIP_GPU;
    check("openpmd: restart = uninterrupted run (window, pml, laser, plasma, delayed beams)",
          at && counts && nb2 > 0 && files && a.istep == r.istep && a.dom.wx == r.dom.wx && (exact ? diff == 0 : diff < 1e-9),
          fmt("checkpoint at step %.0f", double(nchk)) + fmt(", window shifts %.0f", double(r.dom.wx)) +
              fmt(", max relative difference %.1e", diff) + (files ? ", beam files equal" : ", beam files DIFFER") +
              (at ? "" : ", restored state WRONG"));
}

void test_openpmd(Comm& comm) {
    openpmd_rz_axis(comm);
    openpmd_output(comm);
    openpmd_restart(comm);
}
#else
void test_openpmd(Comm&) { std::printf("openpmd: skipped (built without HDF5)\n"); }
#endif
