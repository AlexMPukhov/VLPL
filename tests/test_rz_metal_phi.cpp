// Embedded conductors that depend on the azimuth in RZ (block metal, docs/specs/rz_metal_phi.md, decision 103).
// References: the rings block (axisymmetric mask), research/rz_metal_phi/xsection.py (the collocation operator as dense
// matrices, the continuum by aligned polar finite differences, the analytic square guide).
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include "common.hpp"
#include "rz.hpp"
#include "sim.hpp"

// research/rz_metal_phi/xsection.py, discrete_fd: the three lowest k_c of the staggered collocation operator (nr = 40, R = 1)
// for M = 4, 8, 16; the continuum: square analytic, ridged pipe aligned polar finite differences (nr = 80 / Richardson).
const double SQ_TM[3][3] = {{3.58772277, 5.41481672, 5.54648754}, {3.59669231, 5.623694, 5.67406352}, {3.60710669, 5.69093536, 5.69758219}};
const double SQ_TE[3][3] = {{2.24623268, 2.41615436, 3.50532204}, {2.31369259, 2.35765834, 3.29100023}, {2.40286347, 2.41147374, 3.32596607}};
const double RD_TM[3][3] = {{2.59036682, 3.78604208, 4.14205695}, {2.91023143, 4.1529047, 4.51641266}, {3.18494658, 4.64196057, 4.71616513}};
const double RD_TE[3][3] = {{1.61618325, 1.79860118, 2.44415851}, {1.57984812, 1.66158338, 2.29099389}, {1.59706633, 1.64568786, 2.0866625}};
const double RD_TM_CONT = 3.367652, RD_TE_CONT = 1.66611;

namespace {
std::vector<Field*> all(FieldsRZ& f) { return {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp}; }

UnitFn no_units() { return [](const std::string&, double&) { return false; }; }

std::shared_ptr<MetalPhi> make_metal(const std::string& text, const Domain& d, FieldsRZ& f, bool spectral = false) {
    MetalPhiConfig c;
    c.on = true; c.text = text; c.spectral = spectral;
    c.fn = Expr(text, {"x", "y", "z", "r", "phi"}, no_units());
    return std::make_shared<MetalPhi>(c, d, f.modes(), f.grid().r);   // the caller: f.set_metal_phi(...)
}

// sets the slots of field F at (i, j) from the values g(phi_k) on the 2M - 1 collocation angles
void set_from_angles(std::vector<double>& h, const Field& F, int i, int j, int M, const std::function<double(double)>& g) {
    const int np = 2 * M - 1;
    for (int m = 0; m < M; ++m) {
        double a = 0, b = 0;
        for (int k = 0; k < np; ++k) {
            const double p = 2 * M_PI * k / np, v = g(p);
            a += v * std::cos(m * p) / np;
            b -= v * std::sin(m * p) / np;
        }
        if (m == 0) h[F.idx(i, j, 0)] = a;
        else { h[F.idx(i, j, 2 * m - 1)] = a; h[F.idx(i, j, 2 * m)] = b; }
    }
}

// P1. An axisymmetric mask equals the rings block: a periodic stack of rings (faces in z, r surfaces) and the field
//     of modes 0..2 after 300 steps, metal function vs rings, to round-off: with phi_derivative = spectral against the
//     rings block as it is, with difference (m_eff) against rings with the same m_eff (an empty metal mask sets it).
void axisymmetric(Comm& comm, bool spectral) {
    const int nz = 40, nr = 16, M = 3;
    const double dr = 0.1, dz = 0.05;
    Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, Solver::RipCC);
    FieldsRZ a(d, M), b(d, M);
    RingsConfig rc;
    rc.on = true; rc.r_in = 0.6; rc.r_out = 1.2; rc.period = 1.0; rc.gap = 0.5; rc.z_start = 0.2; rc.z_end = 2.0;
    a.metal = rz_metal_setup(rc, nz, dz, true, a.grid());
    b.set_metal_phi(make_metal("(z - floor(z)) >= 0.175 && (z - floor(z)) <= 0.725 && r >= 0.55 && r <= 1.25", d, b, spectral));
    if (!spectral) a.set_metal_phi(make_metal("0", d, a));     // m_eff for the rings run, no metal from the mask
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(-1, 1);
    for (Field* F : all(a)) {                          // random fields in all modes, the same in both runs
        std::vector<double> h = F->host();
        for (int i = 0; i < d.n[0]; ++i)
            for (int j = 0; j < nr; ++j)
                for (int s = 0; s < 2 * M - 1; ++s) h[F->idx(i, j, s)] = (j == 0 && s > 0) ? 0 : U(rng);
        F->upload(h);
    }
    std::vector<Field*> fa = all(a), fb = all(b);
    for (int c = 0; c < 6; ++c) fb[size_t(c)]->upload(fa[size_t(c)]->host());
    comm.exchange(d, fa); comm.exchange(d, fb);
    for (int n = 0; n < 300; ++n) { a.zero_currents(); a.step(comm); b.zero_currents(); b.step(comm); }
    double diff = 0, sc = 0;
    for (int c = 0; c < 6; ++c) {
        const std::vector<double> x = fa[size_t(c)]->host(), y = fb[size_t(c)]->host();
        for (int s = 0; s < 2 * M - 1; ++s)
            for (int i = 0; i < d.n[0]; ++i)
                for (int j = 0; j < nr; ++j) {
                    diff = std::max(diff, std::abs(x[fa[size_t(c)]->idx(i, j, s)] - y[fb[size_t(c)]->idx(i, j, s)]));
                    sc = std::max(sc, std::abs(x[fa[size_t(c)]->idx(i, j, s)]));
                }
    }
    diff = comm.allreduce_max(diff) / comm.allreduce_max(sc);
    check(std::string("rz metal_phi P1: an axisymmetric metal function = the rings block (stack, modes 0..2, random fields, 300 steps, ") +
              (spectral ? "spectral)" : "difference)"),
          diff < 1e-12, fmt("max difference %.1e of max", diff));
}

// Frequency w dt of the dominant line of s_n: maximum of the Hann-windowed DTFT (scan, then golden section)
double peak_frequency(const std::vector<double>& s, double wlo, double whi) {
    const size_t N = s.size();
    auto amp = [&](double w) {
        std::complex<double> z = 0;
        for (size_t n = 0; n < N; ++n) z += s[n] * (0.5 - 0.5 * std::cos(2 * M_PI * n / (N - 1))) * std::polar(1.0, -w * double(n));
        return std::abs(z);
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

// The cutoff k_c of the cross-section mode excited by g (TM: E_z at the nodes, TE: B_z at the half nodes), z uniform
// (periodic, 4 cells), from the dominant frequency with sin(w dt / 2) = dt k_c / 2 (RIP-CC at k_z = 0).
double cutoff(Comm& comm, const std::string& metal, int M, int nr, double R, bool tm,
              const std::function<double(double, double)>& g, double periods, double kguess, double kappa = 0,
              double* dz_out = nullptr) {
    const double kmax = 3 * kguess;
    const double dr = R / nr;
    RadialGrid rg = rz_radial_grid(nr, dr);
    rg.set_meff(M, true); rg.set_rod(M, kappa);                        // the grid as Config::radial_grid makes it
    const double dz = 0.95 * 2 / std::sqrt(rz_lambda_max(rg, M));     // the RIP-CC limit dz^2 lambda_max / 4 < 1
    Domain d = rz_domain(comm, 4, nr, M, dz, dr, true, Solver::RipCC);
    if (dz_out) *dz_out = dz;
    FieldsRZ f(d, M, std::move(rg));
    f.set_metal_phi(make_metal(metal, d, f));
    Field& F = tm ? f.ez : f.bz;
    std::vector<double> h = F.host();
    for (int i = 0; i < d.n[0]; ++i)
        for (int j = 0; j < nr; ++j) {
            const double r = tm ? f.grid().r[size_t(j)] : f.grid().rh[size_t(j)];
            set_from_angles(h, F, i, j, M, [&](double p) { return g(r * std::cos(p), r * std::sin(p)); });
        }
    F.upload(h);
    comm.exchange(d, all(f));
    std::vector<double> s;
    const std::vector<double> h0 = F.host();
    const int steps = int(periods * 2 * M_PI / (kguess * dz));
    for (int n = 0; n < steps; ++n) {
        f.zero_currents(); f.step(comm);
        const std::vector<double> x = F.host();
        double v = 0;
        for (int j = 0; j < nr; ++j)
            for (int q = 0; q < 2 * M - 1; ++q) v += (q ? 2 : 1) * (j + 0.5) * x[F.idx(0, j, q)] * h0[F.idx(0, j, q)];
        s.push_back(comm.allreduce_max(v));
    }
    const double w = peak_frequency(s, 0.2 * dz, 2 * std::asin(std::min(1.0, kmax * dz / 2)));
    return 2 / dz * std::sin(w / 2);
}

// P2. Square guide |x|, |y| < b = 0.6 in a pipe R = 1 (nr = 40): the lowest TM (TM11, modes 0, 4, 8, ..) and TE (TE10,
//     modes 1, 3, 5, ..) against the discrete collocation operator (xsection.py, dense eigenvalues) and against the
//     continuum (analytic): the code = the operator to the frequency fit; staircase error vs M.
// P3. Pipe R = 1 with four ridges r > 0.6, |phi - n pi/2| < pi/16: lowest TM (C4-symmetric excitation) and TE (dipole).
void cutoffs(Comm& comm) {
    const char* sq = "abs(x) > 0.6 || abs(y) > 0.6 || r > 1";
    const char* rd = "(r > 0.6 && abs(phi - pi/2 * floor((phi + pi/4) / (pi/2) )) < pi/16) || r > 1";
    const double b = 0.6, a = 2 * b;
    auto tm11 = [&](double x, double y) { return std::abs(x) < b && std::abs(y) < b ? std::cos(M_PI * x / a) * std::cos(M_PI * y / a) : 0.0; };
    auto te10 = [&](double x, double y) { return std::abs(x) < b && std::abs(y) < b ? std::sin(M_PI * x / a) : 0.0; };
    auto j0 = [&](double x, double y) { const double r = std::hypot(x, y); return r < 0.6 ? std::cyl_bessel_j(0, 2.4048 * r / 0.6) : 0.0; };
    auto dip = [&](double x, double y) { return std::hypot(x, y) < 0.6 ? x : 0.0; };
    // xsection.py: discrete (nr = 40, R = 1) for M = 4, 8, 16; continuum: square analytic, ridges Richardson (polar FD)
    struct Case { const char* name; const char* metal; bool tm; std::function<double(double, double)> g; const double (*disc)[3]; double cont; };
    const Case cs[] = {
        {"square TM11", sq, true, tm11, SQ_TM, std::sqrt(2.0) * M_PI / a},
        {"square TE10", sq, false, te10, SQ_TE, M_PI / a},
        {"ridges TM", rd, true, j0, RD_TM, RD_TM_CONT},
        {"ridges TE", rd, false, dip, RD_TE, RD_TE_CONT},
    };
    const int Ms[3] = {4, 8, 16};
    for (const Case& c : cs) {
        double emax = 0;
        std::string info;
        double ec[3];
        for (int q = 0; q < 3; ++q) {
            const double k = cutoff(comm, c.metal, Ms[q], 40, 1.0, c.tm, c.g, 40, c.cont);
            double ref = c.disc[q][0];                   // the nearest of the three (pairs split by the odd angle count)
            for (int e = 1; e < 3; ++e) if (std::abs(k - c.disc[q][e]) < std::abs(k - ref)) ref = c.disc[q][e];
            emax = std::max(emax, std::abs(k / ref - 1));
            ec[q] = k / c.cont - 1;
            info += fmt(" M=%.0f:", Ms[q]) + fmt(" %.5f", k) + fmt(" (op %.1e,", k / ref - 1) + fmt(" cont %+.1e)", ec[q]);
        }
        check(std::string("rz metal_phi P2/P3: ") + c.name + " cutoff = the discrete operator (numpy) to 1e-5; staircase error < 10 % at M = 16",
              emax < 1e-5 && std::abs(ec[2]) < 0.1, info);
    }
}

// A1. Axis cut (numerics.rz_axis_cut = kappa): with 32 modes on nr = 40 lambda_max dr^2 stays O(1) (without: ~ m_eff^2), and
//     the square-guide TM11 cutoff (modes 0, 4, 8, .. 15) with kappa = 1 equals the run without the cut, at a larger dz.
void axis_cut(Comm& comm) {
    RadialGrid g = rz_radial_grid(40, 0.025);
    g.set_meff(32, true);
    const double l0 = rz_lambda_max(g, 32) * 0.025 * 0.025;
    g.set_rod(32, 1.0);
    const double l1 = rz_lambda_max(g, 32) * 0.025 * 0.025;
    const double b = 0.6, a = 2 * b;
    auto tm11 = [&](double x, double y) { return std::abs(x) < b && std::abs(y) < b ? std::cos(M_PI * x / a) * std::cos(M_PI * y / a) : 0.0; };
    double dz0, dz1;
    const char* sq = "abs(x) > 0.6 || abs(y) > 0.6 || r > 1";
    const double k0 = cutoff(comm, sq, 16, 40, 1.0, true, tm11, 40, 3.7, 0.0, &dz0);
    const double k1 = cutoff(comm, sq, 16, 40, 1.0, true, tm11, 40, 3.7, 1.0, &dz1);
    check("rz metal_phi A1: axis cut, lambda_max bounded with 32 modes; square TM11 (16 modes) unchanged at a larger dz",
          l1 < 8 && l0 > 30 && std::abs(k1 / k0 - 1) < 2e-3,
          fmt("lambda_max dr^2: %.1f without, ", l0) + fmt("%.2f with kappa = 1; ", l1) + fmt("TM11 %.6f -> ", k0) + fmt("%.6f", k1) +
              fmt(" (dz %.4f -> ", dz0) + fmt("%.4f)", dz1));
}

// P4. Stability and energy: a square corrugated pipe (C4, z faces) inside the round grid, random fields in modes 0..6, 3000
//     RIP-CC steps: the field energy stays bounded (no growth from the per-angle rules or the m <-> m +- 4 coupling),
//     the Gauss residual and div B away from the metal unchanged, the fields in the metal interior zero.
void stability(Comm& comm) {
    const int nz = 32, nr = 24, M = 7;
    const double dr = 0.05;
    const RadialGrid rg = rz_radial_grid(nr, dr);
    const double dz = 0.95 * 2 / std::sqrt(rz_lambda_max(rg, M));
    Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, Solver::RipCC);
    FieldsRZ f(d, M);
    // metal outside the square max(|x|, |y|) = 0.95, teeth to 0.87 over 2/3 of a period nz dz / 2: corners of the
    // R = 1.2 grid, no radius is metal at all angles
    const std::string sq = "max(abs(x), abs(y))";
    const double p = nz * dz / 2;
    char text[600];
    std::snprintf(text, sizeof text, "%s > 0.95 || (%s > 0.87 && (z - %.17g * floor(z / %.17g)) < %.17g)", sq.c_str(), sq.c_str(), p, p,
                  2 * p / 3);
    const std::string t = text;
    f.set_metal_phi(make_metal(t, d, f));
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> U(-1, 1);
    for (Field* F : all(f)) {
        std::vector<double> h = F->host();
        for (int i = 0; i < d.n[0]; ++i)
            for (int j = 0; j < nr; ++j)
                for (int s = 0; s < 2 * M - 1; ++s) h[F->idx(i, j, s)] = (j == 0 && s > 0) ? 0 : U(rng);
        F->upload(h);
    }
    comm.exchange(d, all(f));
    f.zero_currents(); f.step(comm); f.zero_currents(); f.step(comm);    // the metal points zeroed by the first steps
    Field ga, gb, da, db;
    for (Field* F : {&ga, &gb, &da, &db}) F->resize(d);
    f.div_e(ga); f.div_b(da);
    const double w0 = f.energy();
    double wmax = w0, wmin = w0;
    for (int n = 0; n < 3000; ++n) {
        f.zero_currents(); f.step(comm);
        const double w = f.energy();
        wmax = std::max(wmax, w); wmin = std::min(wmin, w);
    }
    f.div_e(gb); f.div_b(db);
    const MetalPhiView pv = f.metal_phi->host_view();
    const std::vector<double> A = ga.host(), B = gb.host(), C = da.host(), D = db.host();
    double dg = 0, ddb = 0, sc = 0;
    for (int i = 0; i < d.n[0]; ++i)
        for (int j = 0; j < nr; ++j)
            for (int s = 0; s < 2 * M - 1; ++s) {
                const long I = d.cell0(0) + i;
                sc = std::max(sc, std::abs(A[ga.idx(i, j, s)]));
                if (!(j == 0 && s > 0) && !pv.gauss_skip(I, j)) dg = std::max(dg, std::abs(A[ga.idx(i, j, s)] - B[gb.idx(i, j, s)]));
                if (!pv.divb_skip(I, j)) ddb = std::max(ddb, std::abs(C[da.idx(i, j, s)] - D[db.idx(i, j, s)]));
            }
    dg = comm.allreduce_max(dg) / comm.allreduce_max(sc); ddb = comm.allreduce_max(ddb) / comm.allreduce_max(sc);
    // fields at interior metal points (all angles metal and both z neighbours) are zero in every mode
    double inside = 0;
    const std::vector<double> ez = f.ez.host(), bp = f.bp.host();
    for (int i = 0; i < d.n[0]; ++i)
        for (int j = 1; j < nr; ++j) {
            const long I = d.cell0(0) + i;
            bool allm = true;
            for (int k = 0; k < pv.np; ++k) allm = allm && pv.rEz(I, j, k) == 1 && pv.rBp(I, j, k) == 1 && pv.Mn(I - 1, j, k) && pv.Mn(I + 1, j, k);
            if (allm)
                for (int s = 0; s < 2 * M - 1; ++s) inside = std::max({inside, std::abs(ez[f.ez.idx(i, j, s)]), std::abs(bp[f.bp.idx(i, j, s)])});
        }
    check("rz metal_phi P4: square corrugated pipe, random fields (modes 0..6), 3000 steps: energy bounded, Gauss / div B off the metal",
          wmax < 1.02 * w0 && wmin > 0.9 * w0 && dg < 1e-11 && ddb < 1e-11 && inside == 0,
          fmt("W/W0 in [%.4f, ", wmin / w0) + fmt("%.4f]", wmax / w0) + fmt(", dGauss %.1e", dg) + fmt(", d divB %.1e", ddb) +
              fmt(", f in metal %.0e", inside));
}

// P5. Deck path: hot plasma in a pipe with four corrugated ridges (modes 0..4, RIP-CC, shapes 1 and 2): Gauss law and div B
//     at round-off off the metal surface, particles absorbed by the metal; iRIP rejected.
void plasma(Comm& comm) {
    const char* path = "rz_metal_phi_test_tmp.deck";
    {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n modes = 5\n lambda0 = 1 um\n nz = 40\n nr = 24\n dz = 0.012 um\n dr = 0.06 um\n"
             " t_end = 6 fs\n}\nnumerics {\n solver = rip_cc\n pusher = boris\n}\n"
             "metal {\n function = r > 0.8 um && abs(phi - pi/2 * floor((phi + pi/4) / (pi/2))) < 0.3 && (z - 0.24 um * floor(z / (0.24 um))) < 0.12 um\n}\n"
             "species e {\n charge = -1\n mass = 1\n density = 0.02\n rmax = 1.3 um\n ppc_r = 2\n ppc_theta = 18\n"
             " uth_z = 0.15\n uth_x = 0.15\n uth_y = 0.15\n uz = 0.1\n}\n"
             "species i {\n charge = 1\n mass = 100\n density = 0.02\n rmax = 1.3 um\n ppc_r = 2\n ppc_theta = 18\n"
             " uth_z = 0.01\n uth_x = 0.01\n uth_y = 0.01\n}\n"
             "output {\n dir = rz_metal_phi_test_out\n every = 100000\n}\n";
    }
    Deck dk = Deck::read(path);
    for (int order = 1; order <= 2; ++order) {
        dk.override_from(std::string("numerics.shape=") + std::to_string(order));
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        double n0 = double(sim.species[0].size()) + double(sim.species[1].size());
    
        double gmax = 0, rmax = 0, dbm = 0;
        while (!sim.finished()) {
            sim.step();
            double wf, g, rm, db;
            sim.rz_diagnostics(wf, g, rm, db);
            gmax = std::max(gmax, g); rmax = std::max(rmax, rm); dbm = std::max(dbm, db);
        }
        double n1 = double(sim.species[0].size()) + double(sim.species[1].size());

        check("rz metal_phi P5: plasma with corrugated ridges (shape " + std::to_string(order) + "): Gauss law, div B, absorption",
              gmax < 1e-12 * std::max(rmax, 1e-30) + 1e-14 && dbm < 1e-12 && n1 < n0,
              fmt("Gauss %.1e", gmax) + fmt(" (max rho %.1e)", rmax) + fmt(", div B %.1e", dbm) + fmt(", particles %.0f", n0) +
                  fmt(" -> %.0f", n1));
    }
    bool rejected = false;
    dk.override_from("numerics.solver=irip");
    try { Config c = Config::from_deck(dk); c.check_rz(); } catch (const std::exception& e) { rejected = std::string(e.what()).find("rip_cc") != std::string::npos; }
    check("rz metal_phi: the metal block with iRIP rejected", rejected, "");
    if (comm.rank == 0) std::remove(path);
}
}  // namespace

void test_rz_metal_phi(Comm& comm) {
    if (comm.size > 1) return;
    axisymmetric(comm, true);
    axisymmetric(comm, false);
    cutoffs(comm);
    axis_cut(comm);
    stability(comm);
    plasma(comm);
}
