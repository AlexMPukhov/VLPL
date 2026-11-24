// Structure runs in RZ (decision 105; Cart3D: decision 102, docs/specs/structure3d.md): external quadrupoles in the RZ push,
// beams from openPMD files in RZ, the wake probe off the axis (all modes summed) and beam slices. References: thick-lens
// transfer matrices, the moments of the written beam computed here, the Cart3D import of the same file, the integral of a
// backward plane wave.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include "common.hpp"
#include "external.hpp"
#include "rz.hpp"
#include "sim.hpp"
#ifdef RIP_HDF5
#include "h5.hpp"
#endif

namespace {
// R1. Cold electrons through a FODO of four hard-edged quadrupoles (axis off r = 0, edges between the grid points) in the RZ
//     push against thick-lens matrices with K = G / u_z (paraxial): G > 0 focuses electrons in x (B_x = -G y, B_y = -G x).
//     Positions and angles (x 1000) at the exit below 1e-4 of the amplitude and falling with dz; both pushers.
struct Pt { double y, yp; };
Pt matrix(Pt p, double K, double L) {
    if (K > 0) { const double w = std::sqrt(K); return {p.y * std::cos(w * L) + p.yp * std::sin(w * L) / w, -p.y * w * std::sin(w * L) + p.yp * std::cos(w * L)}; }
    if (K < 0) { const double w = std::sqrt(-K); return {p.y * std::cosh(w * L) + p.yp * std::sinh(w * L) / w, p.y * w * std::sinh(w * L) + p.yp * std::cosh(w * L)}; }
    return {p.y + p.yp * L, p.yp};
}
void fodo(Comm& comm) {
    const double uz = 2000, ell = 1000, G = 5.78e-3, z0 = 50.7, gap = 137.3, xc = 0.4, yc = -0.3;
    std::vector<Quad> Q;
    for (int q = 0; q < 4; ++q) {
        const double a = z0 + q * (ell + gap);
        Q.push_back(Quad{a, a + ell, q % 2 ? -G : G, xc, yc});     // Quad::yc, zc: the particle axes (z, x, y) -> x_c, y_c
    }
    const double zend = z0 + 4 * ell + 3 * gap + 20;
    const double x0[3] = {0.15, -0.08, 0.03}, xp0[3] = {0.0, 1e-4, -2e-4};
    double err[2][2] = {{0, 0}, {0, 0}}, scale = 0;
    for (int pu = 0; pu < 2; ++pu)
        for (int r = 0; r < 2; ++r) {
            const double h = r ? 2.0 : 4.0;
            Domain d = rz_domain(comm, int(zend / h) + 20, 8, 2, h, 0.2, false, Solver::RipCC);
            FieldsRZ f(d, 2);
            Species sp;
            sp.q = -1; sp.m = 1;
            std::vector<double> hv[Species::NC];
            for (int p = 0; p < 3; ++p) {
                const double v[Species::NC] = {2.0, xc + x0[p], yc - x0[p], uz, xp0[p] * uz, -xp0[p] * uz, 1e-30};
                for (int c = 0; c < Species::NC; ++c) hv[c].push_back(v[c]);
            }
            sp.append(hv);
            set_quadrupoles(Q);
            const int steps = int((zend - 2.0) / h);
            for (int n = 0; n < steps; ++n) rz_push_deposit(sp, f, pu ? Pusher::HigueraCary : Pusher::Boris, false, 1);
            set_quadrupoles({});
            const std::vector<double> Z = sp.get(Species::X), X = sp.get(Species::Y), Y = sp.get(Species::Z), UZ = sp.get(Species::UX),
                                      UX = sp.get(Species::UY), UY = sp.get(Species::UZ);
            for (int p = 0; p < 3; ++p) {
                Pt a{x0[p], xp0[p]}, b{-x0[p], -xp0[p]};
                a = matrix(a, 0, z0 - 2.0); b = matrix(b, 0, z0 - 2.0);
                for (int q = 0; q < 4; ++q) {
                    const double K = (q % 2 ? -G : G) / uz;
                    a = matrix(a, K, ell); b = matrix(b, -K, ell);
                    if (q < 3) { a = matrix(a, 0, gap); b = matrix(b, 0, gap); }
                }
                a = matrix(a, 0, 20); b = matrix(b, 0, 20);
                const double px = X[p] - xc + (zend - Z[p]) * UX[p] / UZ[p], ppx = UX[p] / UZ[p];
                const double py = Y[p] - yc + (zend - Z[p]) * UY[p] / UZ[p], ppy = UY[p] / UZ[p];
                err[pu][r] = std::max({err[pu][r], std::abs(px - a.y), std::abs(py - b.y), 1000 * std::abs(ppx - a.yp),
                                       1000 * std::abs(ppy - b.yp)});
                scale = std::max({scale, std::abs(a.y), std::abs(b.y)});
            }
        }
    bool ok = true;
    for (int pu = 0; pu < 2; ++pu) ok = ok && err[pu][0] / scale < 1e-4 && err[pu][1] < err[pu][0];
    check("rz structure R1: quadrupole FODO in the RZ push = thick-lens matrices (axis off r = 0, edges between grid points)", ok,
          fmt("boris %.2e (dz = 4), ", err[0][0] / scale) + fmt("%.2e (dz = 2); ", err[0][1] / scale) +
              fmt("higuera_cary %.2e, ", err[1][0] / scale) + fmt("%.2e", err[1][1] / scale));
}

void write_deck(const std::string& path, const std::string& text) { std::ofstream f(path); f << text; }

#ifdef RIP_HDF5
// R2. A beam written here as openPMD (unitSI 1e-6 positions, a constant positionOffset, macroWeighted momenta, constant
//     charge / mass records) read by an RZ species with file = ...: (a) the moments of the written beam (computed here in
//     SI) with the centroid at z_center and the offsets x_offset, y_offset; (b) the particles equal those of the Cart3D
//     import of the same file (axes (x, y, z) -> (z, x, y), transverse origin y_center, z_center -> the axis); (c) the
//     beam_slices file of the RZ run: the sum of the slice charges = Q, the charge-weighted <x>, <y> = the centroid.
void beam_file(Comm& comm) {
    const double QE = 1.602176634e-19, ME = 9.1093837015e-31, CL = 299792458.0;
    const std::string path = "test_rz_structure_beam.h5";
    const size_t n = 4000;
    std::mt19937_64 rng(11);
    std::normal_distribution<double> nd(0, 1);
    std::uniform_real_distribution<double> ud(0.5, 1.5);
    std::vector<double> px(n), py(n), pz(n), ux(n), uy(n), uz(n), w(n);
    for (size_t k = 0; k < n; ++k) {
        px[k] = 6 * nd(rng) + 3; py[k] = 5 * nd(rng) - 2; pz[k] = 100 * nd(rng);      // um
        uz[k] = 2000 * (1 + 0.01 * pz[k] / 100 + 1e-3 * nd(rng));
        ux[k] = 0.5 * nd(rng) + 0.002 * px[k]; uy[k] = 0.4 * nd(rng) - 0.001 * py[k];
        w[k] = 1e6 * ud(rng);
    }
    if (comm.rank == 0) {
        h5::Id f = h5::create_file(path);
        h5::attr(f, "openPMD", "1.1.0"); h5::attr(f, "basePath", "/data/%T/"); h5::attr(f, "particlesPath", "particles/");
        h5::attr(f, "meshesPath", "meshes/");
        const std::string g = "/data/3/particles/beam/";
        auto ds = [&](const std::string& p, const std::vector<double>& v, double unit) {
            h5::Id grp = h5::group(f, p.substr(0, p.rfind('/')));
            h5::Id d = h5::dataset<double>(grp, p.substr(p.rfind('/') + 1), {hsize_t(n)});
            h5::write<double>(d, v.data(), {0}, {hsize_t(n)});
            h5::attr<double>(d, "unitSI", unit);
        };
        auto cst = [&](const std::string& p, double v, double unit) {
            h5::Id gr = h5::group(f, p);
            h5::attr<double>(gr, "value", v); h5::attr<double>(gr, "unitSI", unit);
            h5::attr<double>(gr, "shape", std::vector<double>{double(n)});
        };
        ds(g + "position/x", px, 1e-6); ds(g + "position/y", py, 1e-6); ds(g + "position/z", pz, 1e-6);
        cst(g + "positionOffset/x", 0, 1); cst(g + "positionOffset/y", 0, 1); cst(g + "positionOffset/z", 2e-3, 1);
        std::vector<double> mx(n), my(n), mz(n);
        for (size_t k = 0; k < n; ++k) { mx[k] = ux[k] * w[k]; my[k] = uy[k] * w[k]; mz[k] = uz[k] * w[k]; }
        ds(g + "momentum/x", mx, ME * CL); ds(g + "momentum/y", my, ME * CL); ds(g + "momentum/z", mz, ME * CL);
        { h5::Id m = h5::open_group(f, g + "momentum"); h5::attr<double>(m, "macroWeighted", 1.0); h5::attr<double>(m, "weightingPower", 1.0); }
        ds(g + "weighting", w, 1);
        cst(g + "charge", -QE, 1); cst(g + "mass", ME, 1);
    }
    comm.barrier();
    // the reference moments (SI, um) of the written beam
    double W = 0, m1[6] = {0, 0, 0, 0, 0, 0}, g1 = 0;
    for (size_t k = 0; k < n; ++k) {
        W += w[k];
        const double v[6] = {px[k], py[k], pz[k], ux[k], uy[k], uz[k]};
        for (int a = 0; a < 6; ++a) m1[a] += w[k] * v[a];
        g1 += w[k] * std::sqrt(1 + ux[k] * ux[k] + uy[k] * uy[k] + uz[k] * uz[k]);
    }
    for (double& x : m1) x /= W;
    g1 /= W;
    double s[2][3] = {{0, 0, 0}, {0, 0, 0}}, sz2 = 0;            // per plane: W dr^2, W du^2, W dr du
    for (size_t k = 0; k < n; ++k) {
        const double dr[2] = {px[k] - m1[0], py[k] - m1[1]}, du[2] = {ux[k] - m1[3], uy[k] - m1[4]};
        for (int a = 0; a < 2; ++a) { s[a][0] += w[k] * dr[a] * dr[a]; s[a][1] += w[k] * du[a] * du[a]; s[a][2] += w[k] * dr[a] * du[a]; }
        sz2 += w[k] * (pz[k] - m1[2]) * (pz[k] - m1[2]);
    }
    double epsref[2];
    for (int a = 0; a < 2; ++a) epsref[a] = std::sqrt(s[a][0] / W * s[a][1] / W - s[a][2] * s[a][2] / (W * W));   // um
    const double Qref = W * QE, sigz = std::sqrt(sz2 / W);

    const char* rzdeck = "test_rz_structure_beam.deck";
    const char* cdeck = "test_rz_structure_beam3d.deck";
    const double xoff = 5, yoff = -4;                               // um
    if (comm.rank == 0) {
        write_deck(rzdeck, "grid {\n geometry = rz\n modes = 2\n lambda0 = 100 um\n nz = 500\n nr = 25\n dz = 2 um\n dr = 4 um\n"
                           " boundary_z = open\n t_end = 6.6 fs\n}\nnumerics {\n solver = rip_cc\n}\n"
                           "species beam {\n file = test_rz_structure_beam.h5\n z_center = 500 um\n x_offset = 5 um\n y_offset = -4 um\n}\n"
                           "beam_slices b {\n species = beam\n dx = 10 um\n every = 1\n}\n"
                           "output {\n dir = test_rz_structure_out\n every = 100000\n}\n");
        write_deck(cdeck, "grid {\n lambda0 = 100 um\n nx = 500\n ny = 32\n nz = 32\n dx = 2 um\n dy = 4 um\n dz = 4 um\n"
                          " boundary_x = open\n t_end = 6.6 fs\n}\nnumerics {\n solver = rip_cc\n}\n"
                          "species beam {\n file = test_rz_structure_beam.h5\n x_center = 500 um\n y_center = 64 um\n z_center = 64 um\n"
                          " y_offset = 5 um\n z_offset = -4 um\n}\n"
                          "output {\n dir = test_rz_structure_out3\n every = 100000\n}\n");
    }
    comm.barrier();
    // (a) moments of the RZ import
    Config c = Config::from_deck(Deck::read(rzdeck));
    Simulation sim(c, comm);
    sim.init();
    const double k0 = 2 * M_PI / c.lambda0, w0 = CL * k0, nc = 8.8541878128e-12 * ME * w0 * w0 / (QE * QE);
    const Species& sp = sim.species[0];
    std::vector<double> P[Species::NC];
    for (int q = 0; q < Species::NC; ++q) P[q] = sp.get(q);
    const size_t np = P[0].size();
    double v[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, double(np)};         // W, W z, W x, W y, W ux, W uy, W gamma
    for (size_t p = 0; p < np; ++p) {
        const double wp = P[Species::W][p];
        v[0] += wp; v[1] += wp * P[Species::X][p]; v[2] += wp * P[Species::Y][p]; v[3] += wp * P[Species::Z][p];
        v[4] += wp * P[Species::UY][p]; v[5] += wp * P[Species::UZ][p];
        v[6] += wp * std::sqrt(1 + P[Species::UX][p] * P[Species::UX][p] + P[Species::UY][p] * P[Species::UY][p] +
                               P[Species::UZ][p] * P[Species::UZ][p]);
    }
    comm.allreduce_sum(v, 10);
    const double zm = v[1] / v[0], xm = v[2] / v[0], ym = v[3] / v[0], uxm = v[4] / v[0], uym = v[5] / v[0];
    double u[7] = {0, 0, 0, 0, 0, 0, 0};                            // x: W dr^2, du^2, dr du; y: the same; W dz^2
    for (size_t p = 0; p < np; ++p) {
        const double wp = P[Species::W][p];
        const double dr[2] = {P[Species::Y][p] - xm, P[Species::Z][p] - ym}, du[2] = {P[Species::UY][p] - uxm, P[Species::UZ][p] - uym};
        for (int a = 0; a < 2; ++a) { u[3 * a] += wp * dr[a] * dr[a]; u[3 * a + 1] += wp * du[a] * du[a]; u[3 * a + 2] += wp * dr[a] * du[a]; }
        u[6] += wp * (P[Species::X][p] - zm) * (P[Species::X][p] - zm);
    }
    comm.allreduce_sum(u, 7);
    const double um = 1e6 / k0;                                      // code length -> um
    double e[8];
    e[0] = v[0] * nc / (k0 * k0 * k0) * QE / Qref - 1;
    e[1] = (zm * um - 500) / 500;
    e[2] = (xm * um - xoff) / m1[0] - 1;
    e[3] = (ym * um - yoff) / m1[1] - 1;
    for (int a = 0; a < 2; ++a)
        e[4 + a] = std::sqrt(u[3 * a] / v[0] * u[3 * a + 1] / v[0] - u[3 * a + 2] * u[3 * a + 2] / (v[0] * v[0])) * um / epsref[a] - 1;
    e[6] = std::sqrt(u[6] / v[0]) * um / sigz - 1;
    e[7] = v[6] / v[0] / g1 - 1;
    double worst = v[9] == double(n) ? 0.0 : 1.0;
    for (double x : e) worst = std::max(worst, std::abs(x));
    check("rz structure R2a: openPMD import in RZ = moments of the written beam (Q, centroid at z_center + offsets, eps_n x y, "
          "sigma_z, gamma)", worst < 1e-10, fmt("max rel. error %.1e", worst) + fmt(", %.0f particles", v[9]));

    // (b) the Cart3D import of the same file, particle by particle (the gather to rank 0 keeps the file order per rank)
    {
        Config c3 = Config::from_deck(Deck::read(cdeck));
        Simulation s3(c3, comm);
        s3.init();
        std::vector<double> Q[Species::NC];
        for (int q = 0; q < Species::NC; ++q) Q[q] = s3.species[0].get(q);
        // compare as sorted lists of (z, x, y, uz, ux, uy, w) tuples, gathered over the ranks
        auto tuples = [&](const std::vector<double>* A, double yc, double zc) {
            std::vector<std::vector<double>> t;
            for (size_t p = 0; p < A[0].size(); ++p)
                t.push_back({A[Species::X][p], A[Species::Y][p] - yc, A[Species::Z][p] - zc, A[Species::UX][p], A[Species::UY][p],
                             A[Species::UZ][p], A[Species::W][p]});
            std::sort(t.begin(), t.end());
            return t;
        };
        const double yc = 64e-6 * k0, zc = 64e-6 * k0;
        const auto a = tuples(P, 0, 0), b = tuples(Q, yc, zc);
        double d = a.size() == b.size() ? 0.0 : 1.0, cnt[2] = {double(a.size()), double(b.size())};
        for (size_t p = 0; p < a.size() && p < b.size(); ++p)
            for (int q = 0; q < 7; ++q) d = std::max(d, std::abs(a[p][size_t(q)] - b[p][size_t(q)]) / (std::abs(b[p][size_t(q)]) + 1e-3));
        comm.allreduce_sum(cnt, 2);
        const double dmax = comm.allreduce_max(d);
        check("rz structure R2b: RZ import = Cart3D import of the same file (axes (x, y, z) -> (z, x, y))", dmax < 1e-12 && cnt[0] == cnt[1],
              fmt("max rel. difference %.1e", dmax) + fmt(", particles %.0f", cnt[0]) + fmt(" / %.0f", cnt[1]));
    }

    // (c) beam_slices after one step
    sim.step();
    double sq = 0, sx = 0, sy = 0;
    if (comm.rank == 0) {
        std::ifstream in("test_rz_structure_out/slices_b.txt");
        std::string line;
        std::vector<std::string> block;
        while (std::getline(in, line)) {
            if (line.rfind("# step", 0) == 0) block.clear();
            else if (!line.empty() && line[0] != '#') block.push_back(line);
        }
        for (const std::string& l : block) {
            std::istringstream is(l);
            double r[11];
            for (double& x : r) is >> x;
            sq += r[1]; sx += r[1] * r[3]; sy += r[1] * r[4];
        }
    }
    comm.allreduce_sum(&sq, 1); comm.allreduce_sum(&sx, 1); comm.allreduce_sum(&sy, 1);
    const double eq = std::abs(sq * 1e-12 / Qref - 1), ex = std::abs(sx / sq - xoff - m1[0]) / 6, ey = std::abs(sy / sq - yoff - m1[1]) / 5;
    check("rz structure R2c: beam_slices in RZ: sum of the slice charges = Q, <x>, <y> = the centroid", eq < 1e-7 && ex < 1e-5 && ey < 1e-5,
          fmt("Q %.1e", eq) + fmt(", <x> %.1e", ex) + fmt(", <y> %.1e (of sigma)", ey));
    if (comm.rank == 0) { std::remove(path.c_str()); std::remove(rzdeck); std::remove(cdeck); }
}
#endif

// R3. The RZ wake probe off the axis (x, y) = (0.3, 0.2) against a backward plane wave E_x = -c B_y = g(z + t) (mode 1,
//     uniform in r) in a window moving with c: F_x = 2 g(xi + 2t), and the accumulated potential is the sum over the steps
//     of 2 g(xi + 2 t_n) dt (computed here) -> int g du; F_z, F_y stay 0. The radial edge is far: the wave stays uniform
//     near the axis.
void wake_probe(Comm& comm) {
    const char* deck = "test_rz_structure_wake.deck";
    const double lam = 1e-6, k0 = 2 * M_PI / lam, L = 1e6 / k0;   // code length in um
    if (comm.rank == 0) {
        std::ostringstream o;
        o.precision(17);
        o << "grid {\n geometry = rz\n modes = 2\n lambda0 = 1 um\n nz = 200\n nr = 110\n dz = " << 0.5 * L << " um\n dr = " << L
          << " um\n boundary_z = open\n boundary_r = absorbing\n r_absorber_cells = 4\n t_end = " << 60 / (2 * M_PI * 299792458.0 / lam) * 1e15 << " fs\n}\n"
          << "numerics {\n solver = rip_cc\n}\nwindow {\n start = 0 fs\n velocity = 1\n}\n"
          << "wake s8 {\n x = " << 0.3 * L << " um\n y = " << 0.2 * L << " um\n every = 1000000\n}\n"
          << "output {\n dir = test_rz_structure_out\n every = 1000000\n}\n";
        write_deck(deck, o.str());
    }
    comm.barrier();
    Config c = Config::from_deck(Deck::read(deck));
    Simulation sim(c, comm);
    sim.init();
    const double xa = 70, s = 3, dz = c.h[0];
    auto g = [&](double z) { return std::exp(-(z - xa) * (z - xa) / (s * s)); };
    FieldsRZ& f = *sim.rz;
    // E_r = g cos phi, E_phi = -g sin phi, B_r = -g sin phi, B_phi = -g cos phi; F = F_0 + 2 Re sum F_m e^{i m phi}
    std::vector<double> er = f.er.host(), ep = f.ep.host(), br = f.br.host(), bp = f.bp.host();
    const Domain& d = sim.dom;
    for (int j = -d.ng; j < d.n[1] + d.ng; ++j)
        for (int i = -d.ng; i < d.n[0] + d.ng; ++i) {
            const double gz = g((d.off[0] + i) * dz);
            er[f.er.idx(i, j, 1)] = 0.5 * gz; ep[f.ep.idx(i, j, 2)] = 0.5 * gz;
            br[f.br.idx(i, j, 2)] = 0.5 * gz; bp[f.bp.idx(i, j, 1)] = -0.5 * gz;
        }
    f.er.upload(er); f.ep.upload(ep); f.br.upload(br); f.bp.upload(bp);
    long nst = 0;
    while (!sim.finished()) { sim.step(); ++nst; }
    sim.pipeline.finish(sim);
    double e = 0, peak = 0, other = 0;
    if (comm.rank == 0) {
        std::ifstream in("test_rz_structure_out/wake_s8.txt");
        std::string line;
        std::vector<std::string> block;
        while (std::getline(in, line)) {
            if (line.rfind("# step", 0) == 0) block.clear();
            else if (!line.empty() && line[0] != '#') block.push_back(line);
        }
        const double E0 = 9.1093837015e-31 * 299792458.0 * 299792458.0 * k0 / 1.602176634e-19;
        for (const std::string& l : block) {
            std::istringstream is(l);
            double v[7];
            for (double& x : v) is >> x;
            const double xi = v[0] * 1e-6 * k0;
            double ref = 0;
            for (long m = 1; m <= nst; ++m) ref += 2 * g(xi + 2 * m * dz) * dz;
            e = std::max(e, std::abs(v[5] / (E0 / k0) - ref));
            other = std::max({other, std::abs(v[4]), std::abs(v[6])});
            peak = std::max(peak, ref);
        }
        other /= E0 / k0;
    }
    comm.allreduce_sum(&e, 1); comm.allreduce_sum(&peak, 1); comm.allreduce_sum(&other, 1);
    const double integral = std::sqrt(M_PI) * s;
    check("rz structure R3: wake probe off the axis (all modes) = sum of 2 g(xi + 2 t) dt, backward plane wave",
          e < 1e-6 * peak && other < 1e-6 * peak && std::abs(peak / integral - 1) < 1e-4,
          fmt("max error %.1e", e / peak) + fmt(", V_z, V_y %.1e", other / peak) + fmt(", peak %.8f", peak) + fmt(" vs int g = %.8f", integral));
    if (comm.rank == 0) std::remove(deck);
}

// R4. Deck path: the RZ keys (quadrupole z_start / x_center / y_center, quadrupole_lattice z_offset, wake x / y) are read
//     in the RZ axes, side_flux is rejected in RZ.
void deck_keys(Comm& comm) {
    const char* deck = "test_rz_structure_keys.deck";
    const char* lat = "test_rz_structure_lattice.txt";
    if (comm.rank == 0) {
        write_deck(lat, "# z_start length gradient\n0.01 0.02 100\n0.04 0.02 -100\n");
        write_deck(deck, "grid {\n geometry = rz\n modes = 2\n lambda0 = 100 um\n nz = 20\n nr = 10\n dz = 10 um\n dr = 20 um\n"
                         " boundary_z = open\n t_end = 10 fs\n}\nnumerics {\n solver = rip_cc\n}\nwindow {\n start = 0 fs\n velocity = 1\n}\n"
                         "quadrupole Q1 {\n z_start = 1 mm\n length = 2 mm\n gradient = 50\n x_center = 30 um\n y_center = -20 um\n}\n"
                         "quadrupole_lattice {\n file = test_rz_structure_lattice.txt\n z_offset = 5 mm\n}\n"
                         "wake w {\n x = 40 um\n y = -10 um\n}\n"
                         "output {\n dir = test_rz_structure_out\n every = 100000\n}\n");
    }
    comm.barrier();
    Deck dk = Deck::read(deck);
    Config c = Config::from_deck(dk);
    const double k0 = 2 * M_PI / c.lambda0, um = 1e-6 * k0;
    bool ok = c.quads.size() == 3 && c.wakes.size() == 1;
    double e = 0;
    if (ok) {
        const double gunit = 9.1093837015e-31 * 299792458.0 * k0 / 1.602176634e-19 * k0;
        const double want[3][5] = {{1000, 3000, 50, 30, -20}, {15000, 35000, 100, 0, 0}, {45000, 65000, -100, 0, 0}};
        for (int q = 0; q < 3; ++q) {
            const Quad& Q = c.quads[size_t(q)];
            const double got[5] = {Q.x0 / um, Q.x1 / um, Q.g * gunit, Q.yc / um, Q.zc / um};
            for (int a = 0; a < 5; ++a) e = std::max(e, std::abs(got[a] - want[q][a]) / (std::abs(want[q][a]) + 1));
        }
        e = std::max({e, std::abs(c.wakes[0].y / um - 40) / 40, std::abs(c.wakes[0].z / um + 10) / 10});
    }
    bool rejected = false;
    if (comm.rank == 0) {
        std::ofstream f(deck, std::ios::app);
        f << "side_flux {\n every = 10\n}\n";
    }
    comm.barrier();
    try { Config::from_deck(Deck::read(deck)); } catch (const std::exception& ex) { rejected = std::string(ex.what()).find("Cart3D") != std::string::npos; }
    check("rz structure R4: deck keys in the RZ axes (quadrupole, quadrupole_lattice, wake); side_flux rejected in RZ",
          ok && e < 1e-12 && rejected, fmt("max rel. error %.1e", e));
    if (comm.rank == 0) { std::remove(deck); std::remove(lat); }
}
}  // namespace

void test_rz_structure(Comm& comm) {
    fodo(comm);
#ifdef RIP_HDF5
    beam_file(comm);
#endif
    wake_probe(comm);
    deck_keys(comm);
}
