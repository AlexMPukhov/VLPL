// The lab-fixed laser antenna of a boosted run (laser.injection = antenna, src/boost_antenna.cpp, decision 89): a beam focused far
// downstream, which the boosted entrance plane cuts (it injects at lab x in [X0, 0], where the beam is wide), against the lab
// run and the paraxial beam; the Gauss law with the antenna's moving charge (2D, 3D RIP-CC, RZ iRIP, with streaming plasma).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include "common.hpp"
#include "fsutil.hpp"
#include "sim.hpp"

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

double interp(const std::vector<double>& g, double x0, double h, double x) {   // cubic Lagrange
    const double u = (x - x0) / h, fi = std::floor(u), w = u - fi;
    const long i = long(fi);
    if (i < 1 || i + 2 >= long(g.size())) return 0;
    return -w * (w - 1) * (w - 2) / 6 * g[size_t(i - 1)] + (w + 1) * (w - 1) * (w - 2) / 2 * g[size_t(i)] -
           (w + 1) * w * (w - 2) / 2 * g[size_t(i + 1)] + (w + 1) * w * (w - 1) / 6 * g[size_t(i + 2)];
}

std::string num(double v) { char b[64]; std::snprintf(b, sizeof b, "%.15g", v); return b; }
double rel(double num, double den) { return den > 0 ? std::sqrt(num / den) : 1e300; }

double gauss_now(Simulation& s) {
    if (!s.cfg.rz) return s.gauss_drift();
    double w, g, r, b;
    s.rz_diagnostics(w, g, r, b);
    return g;
}

// A1. Vacuum laser (a0 = 0.01, 4 cycles, waist 3 um, focus 70 um: z_R = 35 um; at the lab plane x = 0 the beam is 6.7 um wide,
//     at the boosted entrance plane's first lab event X0 = -83 um 13.3 um): the lab run to T (the pulse at the focus), the boosted
//     run (gamma 2) with the entrance plane and with the antenna, back-transformed to T. 2D RIP-CC (box +-17.5 um, y periodic) and
//     RZ iRIP with 2 modes (wall at 17.5 um): the box holds the beam at x = 0 but cuts it at X0.
void antenna_vacuum(Comm& comm) {
    for (int rz = 0; rz < 2; ++rz) {
        std::ostringstream base;
        base.precision(15);
        const double lam = 0.8, dx = lam / 16, c = 299792458.0;
        base << "grid {\n lambda0 = 0.8 um\n";
        if (rz) base << " geometry = rz\n modes = 2\n nz = 400\n nr = 70\n dz = " << dx << " um\n dr = 0.25 um\n boundary_z = open\n";
        else base << " geometry = cartesian2d\n nx = 400\n ny = 140\n dx = " << dx << " um\n dy = 0.25 um\n boundary_x = open\n";
        const double T_fs = 1528 * dx * 1e-6 / c * 1e15, tau_fs = 4 * lam * 1e-6 / c * 1e15;
        base << " t_end = " << T_fs << " fs\n}\n";
        base << "numerics {\n solver = " << (rz ? "irip" : "rip_cc") << "\n}\n";
        const std::string laser = std::string("laser {\n a0 = 0.01\n polarization = ") + (rz ? "x" : "y") + "\n duration = " +
                                  num(tau_fs) + " fs\n delay = " + num(2 * tau_fs) + " fs\n waist = 3 um\n focus = 70 um\n" +
                                  (rz ? "" : " y_center = 17.5 um\n");
        base << "window {\n start = " << 4.01 * tau_fs << " fs\n}\n";
        const std::string dir = rz ? "test_out_antenna_rz" : "test_out_antenna_2d";
        base << "output {\n dir = " << dir << "\n every = 100000\n}\n";
        Config cl = deck_config(base.str() + laser + "}\n", "antenna_test_tmp.deck");
        Simulation lab(cl, comm);
        lab.init();
        while (!lab.finished()) lab.step();
        const double T = lab.t;
        const Domain& d = lab.dom;
        const int N1 = d.N[1];
        const double h = d.h[0], xw = d.wx * h;
        std::vector<double> row(size_t(d.N[0]) * N1, 0.0);
        {
            const Field& F = rz ? lab.rz->er : lab.fields->ey;
            const std::vector<double> hv = F.host();
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i)
                    row[size_t(d.off[1] + j) * d.N[0] + d.off[0] + i] = hv[F.idx(i, j, rz ? 1 : 0)];
            comm.allreduce_sum(row.data(), int(row.size()));
        }
        double r[2][2] = {{1, 1}, {1, 1}}, rla = 1, gmax = 0, amax = 0;
        long steps = 0;
        bool ok = true;
        for (int ant = 0; ant < 2; ++ant) {
            const std::string bdeck = base.str() + laser + (ant ? " injection = antenna\n" : "") + "}\nboost {\n gamma = 2\n}\n" +
                                      "lab_diagnostics {\n t_first = " + num(T_fs) + " fs\n}\n";
            Config cb = deck_config(bdeck, "antenna_test_tmp.deck");
            if (comm.rank == 0) make_dirs(dir);
            comm.barrier();
            Simulation bs(cb, comm);
            bs.init();
            ok = ok && (ant == 0 || bs.antenna);
            while (!bs.finished()) {
                bs.step();
                if (ant && bs.istep % 10 == 0) gmax = std::max(gmax, gauss_now(bs));
            }
            bs.pipeline.finish(bs);
            comm.barrier();
            steps = bs.istep;
            Snap S = read_snap(dir, 0);
            double e_bl = 0, e_ba = 0, e_la = 0, nrm = 0, sbl = 0, sll = 0, sbb = 0;
            ok = ok && !S.v.empty() && S.meta["complete"] == "1";
            if (S.v.empty()) continue;
            const LaserConfig L = cl.laser;
            for (long j = 0; j < S.n1; ++j) {
                const double y = (j + 0.5) * d.h[1];
                std::vector<double> g(row.begin() + long(j) * d.N[0], row.begin() + long(j + 1) * d.N[0]);
                for (long i = 0; i < S.nx; ++i) {
                    const double x = S.x0 + i * S.dx;
                    if (x < xw + 2 * h || x > xw + (d.N[0] - 3) * h) continue;
                    const double a = (rz ? 0.5 : 1.0) * L.field_at(T, x, y, 0.0);
                    const double b = S.at(1, i, j, rz ? 1 : 0), l = interp(g, xw, h, x);
                    e_bl += (b - l) * (b - l); e_ba += (b - a) * (b - a); e_la += (l - a) * (l - a); nrm += a * a;
                    sbl += b * l; sll += l * l; sbb += b * b;
                    amax = std::max(amax, std::abs(a));
                }
            }
            if (getenv("VLPL_TEST_VERBOSE") && comm.rank == 0)
                std::printf("  %s %s: amplitude fit <b l>/<l l> = %.4f, correlation %.5f\n", rz ? "RZ" : "2D", ant ? "antenna" : "plane",
                            sbl / sll, sbl / std::sqrt(sll * sbb));
            r[ant][0] = rel(e_bl, nrm); r[ant][1] = rel(e_ba, nrm); rla = rel(e_la, nrm);
        }
        const std::string geo = rz ? "RZ iRIP" : "2D RIP-CC";
        check("boost antenna, beam focused 70 um downstream, " + geo + ": lab snapshot = lab run, paraxial; Gauss law",
              ok && amax > (rz ? 0.004 : 0.008) && r[1][0] < 1.5e-2 && r[1][1] < rla + 1e-2 && r[0][0] > 2 * r[1][0] && gmax < 1e-15,
              fmt("antenna vs lab run %.2e, vs paraxial %.2e", r[1][0], r[1][1]) +
                  fmt("; entrance plane %.2e, %.2e", r[0][0], r[0][1]) + fmt("; lab run vs paraxial %.2e", rla) +
                  fmt("; max|dGauss| %.1e", gmax) + fmt(", %.0f boosted steps", double(steps)));
    }
}

// A2. Gauss law at round-off with the antenna (laser off the axis, focused, a0 = 1) and a cold plasma streaming through the
//     window at -beta c (gamma 3): 2D RIP-CC, 2D iRIP, 3D RIP-CC (the antenna's charge has y and z divergence), RZ iRIP 2 modes.
void antenna_gauss(Comm& comm) {
    const char* geo[4] = {"2D RIP-CC", "2D iRIP", "3D RIP-CC", "RZ iRIP 2 modes"};
    for (int k = 0; k < 4; ++k) {
        std::ostringstream o;
        o << "grid {\n lambda0 = 0.8 um\n";
        if (k == 3) o << " geometry = rz\n modes = 2\n nz = 128\n nr = 24\n dz = 0.05 um\n dr = 0.5 um\n boundary_z = open\n";
        else o << (k < 2 ? " geometry = cartesian2d\n" : "") << " nx = 128\n ny = 20\n" << (k < 2 ? "" : " nz = 8\n dz = 0.5 um\n")
               << " dx = 0.05 um\n dy = 0.5 um\n boundary_x = open\n";
        o << " t_end = 60 fs\n}\nnumerics {\n solver = " << (k == 0 || k == 2 ? "rip_cc" : "irip") << "\n shape = 2\n}\n"
          << "laser {\n a0 = 1\n polarization = " << (k == 3 ? "x" : "y") << "\n duration = 3 fs\n delay = 6 fs\n waist = 2 um\n"
          << " focus = 3 um\n injection = antenna\n" << (k == 3 ? " y_center = 1 um\n" : " y_center = 5 um\n")
          << (k == 2 ? " z_center = 2 um\n" : "") << "}\n"
          << "window {\n start = 12.1 fs\n}\nboost {\n gamma = 3\n}\n";
        for (int s = 0; s < 2; ++s)
            o << "species " << (s ? "p" : "e") << " {\n charge = " << (s ? 1 : -1) << "\n mass = " << (s ? 100 : 1)
              << "\n density = 0.02\n " << (k == 3 ? "zmin" : "xmin") << " = 4 um\n" << (k == 3 ? " ppc_z = 2\n ppc_r = 2\n"
                                                                                                   : " ppc_x = 2\n ppc_y = 2\n") << "}\n";
        o << "output {\n dir = test_out_antenna_gauss\n every = 100000\n}\n";
        Config c = deck_config(o.str(), "antenna_test_tmp.deck");
        Simulation s(c, comm);
        s.init();
        double g = 0, wmax = 0, n0 = 0;
        while (!s.finished()) {
            s.step();
            if (s.istep % 5 == 0) {
                g = std::max(g, gauss_now(s));
                double w = c.rz ? s.rz->energy() : s.fields->energy();
                comm.allreduce_sum(&w, 1);
                wmax = std::max(wmax, w);
            }
        }
        n0 = double(s.species[0].size());
        comm.allreduce_sum(&n0, 1);
        check(std::string("boost antenna, Gauss law with the moving sheet and streaming plasma, ") + geo[k],
              s.antenna && g < 1e-12 && n0 > 1000 && wmax > 0, fmt("max|dGauss| %.1e, %.0f electrons", g, n0) + fmt(", W_max %.3g", wmax));
    }
}


// Near field of the lab-fixed antenna (bench/antenna_nearfield/README.md). A long Gaussian pulse in vacuum (2D), emitted by the
// antenna of a boosted run and by the entrance plane x = 0 of the lab run on the same lab grid; lab snapshots at count lab times
// (by default the first one with the pulse centre on the antenna). Per snapshot and lab x on the axis: the carrier amplitude
// A = sqrt(E^2 + (E' / k_eff)^2) (E' the centred difference over the snapshot's plane spacing dx_s, k_eff = sin(k dx_s) / dx_s),
// boosted against lab. Distance from the antenna in boosted steps since emission: N = x / (gamma (1 + beta) c dt'), i.e.
// x / ((gamma (1 + beta))^2 h_lab) for c dt' = h'.
struct NFParams {
    double gamma = 5, ncell = 20, tau = 40, dfac = 2, L = 48, dy = 0.4, Ly = 40, w = 8, focus = 0, tfirst = -1, tevery = -1;
    int count = 4, shape = 2, filter = 0;
    std::string solver = "irip", dir = "nf_scan_out";
};
struct NFPoint { int k; double x_um, N, el, eb, al, ab; };
struct NFResult {
    std::vector<NFPoint> p;
    std::vector<double> t_fs;
    double ystep = 0, tmis = 0, dxs = 0;     // lab um per boosted step; lab-time mismatch (cells); snapshot plane spacing / h
    long steps = 0;
    bool complete = true;
};

NFResult nf_run(Comm& comm, const NFParams& P) {
    NFResult R;
    const double lam = 0.8, c = 299792458.0, k0 = 2 * M_PI / (lam * 1e-6);
    const double h = lam / P.ncell, hc = 2 * M_PI / P.ncell;        // lab cell (um; code units = lab dt)
    const long nx = long(std::lround(P.L / h)), ny = long(std::lround(P.Ly / P.dy));
    auto fs_of = [&](double tc) { return tc / (c * k0) * 1e15; };
    auto code_of = [&](double fsv) { return fsv * 1e-15 * c * k0; };
    const double delay = P.dfac * P.tau;
    std::vector<double> T(static_cast<size_t>(P.count), 0.0);    // multiples of the lab step
    for (int k = 0; k < P.count; ++k) {
        const double tf = P.tfirst >= 0 ? P.tfirst + k * (P.tevery > 0 ? P.tevery : 0) : delay + k * 0.25 * P.L * 1e-6 / c * 1e15;
        T[size_t(k)] = std::round(code_of(tf) / hc) * hc;
        R.t_fs.push_back(fs_of(T[size_t(k)]));
    }
    const double tend = fs_of(T.back()), wstart = std::max(tend + 10, 2 * delay + 1);   // window.start >= 2 laser.delay
    const double beta = std::sqrt(1 - 1 / (P.gamma * P.gamma)), D = P.gamma * (1 + beta);
    R.ystep = D * D * h;
    std::ostringstream base;
    base.precision(15);
    base << "grid {\n lambda0 = 0.8 um\n geometry = cartesian2d\n nx = " << nx << "\n ny = " << ny << "\n dx = " << h
         << " um\n dy = " << P.dy << " um\n boundary_x = open\n t_end = " << tend << " fs\n}\n";
    base << "numerics {\n solver = " << P.solver << "\n shape = " << P.shape << "\n filter_passes = " << P.filter << "\n}\n";
    const long jc = ny / 2;
    base << "laser {\n a0 = 0.01\n polarization = y\n duration = " << P.tau << " fs\n delay = " << delay << " fs\n waist = " << P.w
         << " um\n focus = " << P.focus << " um\n y_center = " << (jc + 0.5) * P.dy << " um\n";
    const std::string tail = "window {\n start = " + num(wstart) + " fs\n}\noutput {\n dir = " + P.dir + "\n every = 1000000\n}\n";
    Config cl = deck_config(base.str() + "}\n" + tail, "nf_scan_tmp.deck");      // lab run: E_y at the snapshot times
    Simulation lab(cl, comm);
    lab.init();
    const Domain& d = lab.dom;
    std::vector<std::vector<double>> labE(static_cast<size_t>(P.count));
    size_t k = 0;
    while (!lab.finished() && k < T.size()) {
        lab.step();
        while (k < T.size() && lab.t > T[k] - 0.5 * hc) {
            R.tmis = std::max(R.tmis, std::abs(lab.t - T[k]) / hc);
            std::vector<double> g(size_t(d.N[0]) * d.N[1], 0.0);
            const std::vector<double> hv = lab.fields->ey.host();
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) g[size_t(d.off[1] + j) * d.N[0] + d.off[0] + i] = hv[lab.fields->ey.idx(i, j, 0)];
            comm.allreduce_sum(g.data(), int(g.size()));
            labE[k++] = g;
        }
    }
    const double xw = d.wx * d.h[0];
    Config cb = deck_config(base.str() + " injection = antenna\n}\n" + tail + "boost {\n gamma = " + num(P.gamma) +
                                "\n}\nlab_diagnostics {\n t_first = " + num(fs_of(T[0])) + " fs\n t_every = " +
                                num(P.count > 1 ? fs_of(T[1] - T[0]) : 1.0) + " fs\n count = " + std::to_string(P.count) + "\n}\n",
                            "nf_scan_tmp.deck");
    if (comm.rank == 0) make_dirs(P.dir);
    comm.barrier();
    Simulation bs(cb, comm);
    bs.init();
    while (!bs.finished()) bs.step();
    bs.pipeline.finish(bs);
    comm.barrier();
    R.steps = bs.istep;
    for (int q = 0; q < P.count; ++q) {
        Snap S = read_snap(P.dir, q);
        if (S.v.empty() || S.meta["complete"] != "1") { R.complete = false; continue; }
        R.dxs = S.dx / hc;
        const double ke = std::sin(S.dx) / S.dx;
        std::vector<double> g(labE[size_t(q)].begin() + jc * d.N[0], labE[size_t(q)].begin() + (jc + 1) * d.N[0]);
        std::vector<double> xs, el, eb;
        for (long i = 0; i < S.nx; ++i) {
            const double x = S.x0 + i * S.dx;
            if (x < xw + 2 * hc || x > xw + (d.N[0] - 3) * hc) continue;
            xs.push_back(x); eb.push_back(S.at(1, i, jc, 0)); el.push_back(interp(g, xw, hc, x));
        }
        for (size_t i = 1; i + 1 < xs.size(); ++i) {
            const double dl = (el[i + 1] - el[i - 1]) / (2 * S.dx * ke), db = (eb[i + 1] - eb[i - 1]) / (2 * S.dx * ke);
            const double xu = xs[i] / k0 * 1e6;
            R.p.push_back({q, xu, xu / R.ystep, el[i], eb[i], std::sqrt(el[i] * el[i] + dl * dl), std::sqrt(eb[i] * eb[i] + db * db)});
        }
    }
    return R;
}

// A_boost / A_lab averaged over the points with N in [lo, hi) and A_lab > 0.3 of its snapshot maximum (all snapshots); the
// largest deviation |ratio - 1| of the single points in that range in dmax
double nf_ratio(const NFResult& R, double lo, double hi, double* dmax = nullptr, long* cnt = nullptr) {
    std::map<int, double> mx;
    for (const NFPoint& p : R.p) mx[p.k] = std::max(mx[p.k], p.al);
    double s = 0, dm = 0;
    long n = 0;
    for (const NFPoint& p : R.p)
        if (p.N >= lo && p.N < hi && p.al > 0.3 * mx[p.k]) { s += p.ab / p.al; ++n; dm = std::max(dm, std::abs(p.ab / p.al - 1)); }
    if (dmax) *dmax = dm;
    if (cnt) *cnt = n;
    return n ? s / double(n) : 0;
}

double nf_env(const char* k, double v) { const char* s = getenv(k); return s ? std::atof(s) : v; }
std::string nf_envs(const char* k, const char* v) { const char* s = getenv(k); return s ? s : v; }

// The scan (research, not part of the suite): VLPL_NF_SCAN=1 build/tests test_boost_antenna, parameters NF_* (bench/antenna_nearfield)
void antenna_nearfield_scan(Comm& comm) {
    NFParams P;
    P.gamma = nf_env("NF_GAMMA", P.gamma); P.ncell = nf_env("NF_N", P.ncell); P.tau = nf_env("NF_TAU", P.tau);
    P.dfac = nf_env("NF_DELAYF", P.dfac); P.L = nf_env("NF_L", P.L); P.dy = nf_env("NF_DY", P.dy); P.Ly = nf_env("NF_LY", P.Ly);
    P.w = nf_env("NF_W", P.w); P.focus = nf_env("NF_FOCUS", P.focus); P.tfirst = nf_env("NF_TFIRST", P.tfirst);
    P.tevery = nf_env("NF_TEVERY", P.tevery); P.count = int(nf_env("NF_COUNT", P.count)); P.shape = int(nf_env("NF_SHAPE", P.shape));
    P.filter = int(nf_env("NF_FILTER", P.filter)); P.solver = nf_envs("NF_SOLVER", P.solver.c_str());
    const std::string out = nf_envs("NF_OUT", "nf_scan.txt");
    const NFResult R = nf_run(comm, P);
    if (comm.rank != 0) return;
    FILE* f = std::fopen(out.c_str(), "w");
    std::fprintf(f, "# NF scan: gamma %g N %g (h %.5g um) solver %s shape %d filter %d tau %g fs delay %g fs w %g um focus %g um dy %g um\n",
                 P.gamma, P.ncell, 0.8 / P.ncell, P.solver.c_str(), P.shape, P.filter, P.tau, P.dfac * P.tau, P.w, P.focus, P.dy);
    std::fprintf(f, "# lab length per boosted step (gamma (1+beta))^2 h = %.6g um; boosted steps %ld; lab step mismatch %.2e cells; "
                    "snapshot planes %.6f h; complete %d\n", R.ystep, R.steps, R.tmis, R.dxs, int(R.complete));
    for (size_t q = 0; q < R.t_fs.size(); ++q) std::fprintf(f, "# snapshot %zu t_lab %.6g fs\n", q, R.t_fs[q]);
    std::fprintf(f, "# k x_um N=x/ystep E_lab E_boost A_lab A_boost\n");
    for (const NFPoint& p : R.p) std::fprintf(f, "%d %.6f %.6f %.6e %.6e %.6e %.6e\n", p.k, p.x_um, p.N, p.el, p.eb, p.al, p.ab);
    std::fclose(f);
    std::printf("NF scan written to %s (gamma %g, N %g, %ld boosted steps)\n", out.c_str(), P.gamma, P.ncell, R.steps);
}

// A3. Near field of the antenna (bench/antenna_nearfield/README.md): gamma 5, lab lambda/20, iRIP, 2D vacuum, a long pulse (40 fs)
//     emitted by the antenna, 4 lab snapshots against the lab run on the same lab grid. The emitted field reaches the lab field
//     within 1 % only one boosted step after its emission: from N = x / (gamma (1 + beta) c dt') = 1.1 on, i.e. lab distance
//     d_1% = 1.1 gamma (1 + beta) c dt' (printed by the boost summary). Checked: every point with N >= 1.1 within 1 % (the
//     scan: 0.990 at N = 1.0 - 1.1, then 0.994 - 1.005); the deficit inside the zone as measured (0.6 - 0.7 at N < 0.25, < 0.97 at
//     N = 0.5 - 0.75: a change of the antenna that removes it shows up here and in the docs); the same curve for two snapshots
//     40 fs apart (a fixed distance from the antenna, not a start-up in time).
void antenna_nearfield(Comm& comm) {
    NFParams P;
    P.L = 24; P.tau = 20; P.Ly = 16; P.dy = 0.8; P.tfirst = 40; P.tevery = 6.67; P.dir = "test_out_antenna_nf";   // pulse centre at 0, 2, 4, 6 um
    const NFResult R = nf_run(comm, P);
    double dfar = 0, d1 = 0, d2 = 0;
    long nfar = 0, n1 = 0;
    const double far = nf_ratio(R, 1.1, 1e9, &dfar, &nfar), z0 = nf_ratio(R, 0, 0.25, nullptr, &n1), z1 = nf_ratio(R, 0.5, 0.75);
    NFResult R0 = R, R1 = R;                                  // snapshots 0 and 1 apart
    R0.p.clear(); R1.p.clear();
    for (const NFPoint& p : R.p) (p.k == 0 ? R0 : R1).p.push_back(p);
    for (double lo = 0.25; lo < 1.0; lo += 0.25) {
        long a = 0, b = 0;
        const double r0 = nf_ratio(R0, lo, lo + 0.25, nullptr, &a), r1 = nf_ratio(R1, lo, lo + 0.25, nullptr, &b);
        if (a > 3 && b > 3) d2 = std::max(d2, std::abs(r0 - r1)); else d1 += 1;
    }
    check("boost antenna, near field (gamma 5, lambda/20, iRIP): the field within 1 % of the lab run from 1.1 boosted steps after "
          "emission on; deficit inside, fixed in distance",
          R.complete && nfar > 100 && n1 > 3 && dfar < 0.01 && std::abs(far - 1) < 0.005 && z0 > 0.55 && z0 < 0.75 && z1 < 0.97 &&
              d1 < 2 && d2 < 0.02,
          fmt("N >= 1.1: mean %.4f, max |dev| %.4f", far, dfar) + fmt("; N < 0.25: %.3f, 0.5 - 0.75: %.3f", z0, z1) +
              fmt("; snapshots 0 vs 1 within %.3f", d2) + fmt(" (%.0f boosted steps)", double(R.steps)));
}

}  // namespace

void test_boost_antenna(Comm& comm) {
    if (getenv("VLPL_NF_SCAN")) { antenna_nearfield_scan(comm); return; }
    antenna_gauss(comm);
    antenna_vacuum(comm);
    antenna_nearfield(comm);
}
