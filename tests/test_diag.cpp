// Beam diagnostics (docs/specs/beam_diagnostics.md).
#include <cmath>
#include <cstdio>
#include <random>
#include "common.hpp"
#include "beam_diag.hpp"
#include "sim.hpp"

static Species make_species(const std::vector<double> h[Species::NC], double q, double m) {
    Species s;
    s.q = q; s.m = m; s.name = "b";
    s.append(h);
    return s;
}

// D1. Random particles: count, weight, mean and rms of gamma, mean x, rms sizes, emittances and the
//     maximum against a direct host computation (two passes, sequential), gamma_min selects.
static void diag_moments(Comm& comm) {
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> u01(0, 1);
    std::normal_distribution<double> nd(0, 1);
    std::vector<double> h[Species::NC];
    for (int p = 0; p < 5000; ++p) {
        const double v[Species::NC] = {3 + nd(rng), 2 + 0.3 * nd(rng), -1 + 0.2 * nd(rng), 50 + 40 * u01(rng),
                                       0.5 * nd(rng) + 0.2 * nd(rng), 0.4 * nd(rng), 0.1 + u01(rng)};
        for (int c = 0; c < Species::NC; ++c) h[c].push_back(v[c]);
        h[Species::UY].back() += 0.3 * (h[Species::Y].back() - 2);             // a correlation y - uy
    }
    const Species s = make_species(h, -1, 1);
    const double gmin = 70;
    const BeamMoments m = beam_moments(s, gmin, comm);
    // reference
    double W = 0, cnt = 0, sg = 0, sx = 0, sy = 0, sz = 0, suy = 0, suz = 0, gmax = 0;
    auto gam = [&](size_t p) {
        return std::sqrt(1.0 + h[3][p] * h[3][p] + h[4][p] * h[4][p] + h[5][p] * h[5][p]);
    };
    for (size_t p = 0; p < h[0].size(); ++p) {
        const double g = gam(p), w = h[6][p];
        if (g < gmin) continue;
        cnt += 1; W += w; sg += w * g; sx += w * h[0][p]; sy += w * h[1][p]; sz += w * h[2][p];
        suy += w * h[4][p]; suz += w * h[5][p]; gmax = std::max(gmax, g);
    }
    const double gm = sg / W, xm = sx / W, ym = sy / W, zm = sz / W, uym = suy / W, uzm = suz / W;
    double vg = 0, vx = 0, vy = 0, vz = 0, vuy = 0, vuz = 0, cy = 0, cz = 0;
    for (size_t p = 0; p < h[0].size(); ++p) {
        const double g = gam(p), w = h[6][p];
        if (g < gmin) continue;
        vg += w * (g - gm) * (g - gm); vx += w * (h[0][p] - xm) * (h[0][p] - xm);
        vy += w * (h[1][p] - ym) * (h[1][p] - ym); vz += w * (h[2][p] - zm) * (h[2][p] - zm);
        vuy += w * (h[4][p] - uym) * (h[4][p] - uym); vuz += w * (h[5][p] - uzm) * (h[5][p] - uzm);
        cy += w * (h[1][p] - ym) * (h[4][p] - uym); cz += w * (h[2][p] - zm) * (h[5][p] - uzm);
    }
    vg /= W; vx /= W; vy /= W; vz /= W; vuy /= W; vuz /= W; cy /= W; cz /= W;
    const double ref[10] = {double(cnt), double(W), double(gm), double(std::sqrt(vg)), double(gmax), double(xm),
                            double(std::sqrt(vx)), double(std::sqrt(vy)), double(std::sqrt(vy * vuy - cy * cy)),
                            double(std::sqrt(vz * vuz - cz * cz))};
    const double got[10] = {m.count, m.wsum, m.gmean, m.gsig, m.gmax, m.xmean, m.sx, m.sy, m.eps_y, m.eps_z};
    double worst = 0;
    for (int i = 0; i < 10; ++i) worst = std::max(worst, std::abs(got[i] - ref[i]) / std::abs(ref[i]));
    check("beam diagnostics: moments above gamma_min = direct sums", worst < 1e-12 && cnt > 1000 && cnt < 4500,
          fmt("%.0f of 5000 selected, max rel. difference %.1e", double(cnt), worst));
}

// D2. Particles on an ellipse y = a cos(phi), uy = b sin(phi), N equally spaced phi: eps_n,y = a b / 2.
static void diag_emittance(Comm& comm) {
    const int N = 360;
    const double a = 0.7, b = 0.03;
    std::vector<double> h[Species::NC];
    for (int k = 0; k < N; ++k) {
        const double ph = 2 * M_PI * k / N;
        const double v[Species::NC] = {1, a * std::cos(ph), 0, 100, b * std::sin(ph), 0, 1};
        for (int c = 0; c < Species::NC; ++c) h[c].push_back(v[c]);
    }
    const BeamMoments m = beam_moments(make_species(h, -1, 1), 1, comm);
    const double want = a * b / 2;
    check("beam diagnostics: emittance of an ellipse = a b / 2", std::abs(m.eps_y - want) < 1e-13 * want && m.eps_z == 0,
          fmt("eps_y %.15g, exact %.15g", m.eps_y, want));
}

// D3. Weights placed at the bin centres of the kinetic energy: the spectrum is exact, its sum is the
//     weight of the particles inside [0, e_max), gamma_min cuts the lower bins.
static void diag_spectrum(Comm& comm) {
    const int bins = 20;
    const double mass = 1, ekmax = 100;                    // units m_e c^2
    std::vector<double> h[Species::NC];
    std::vector<double> want(bins, 0.0);
    for (int b = 0; b < bins; ++b)
        for (int k = 0; k <= b % 3; ++k) {
            const double ek = (b + 0.5) * ekmax / bins, g = 1 + ek / mass, u = std::sqrt(g * g - 1);
            const double v[Species::NC] = {0, 0, 0, u, 0, 0, 0.5 + b};
            for (int c = 0; c < Species::NC; ++c) h[c].push_back(v[c]);
            if (g >= 30) want[size_t(b)] += 0.5 + b;
        }
    const double v[Species::NC] = {0, 0, 0, 400, 0, 0, 7};      // beyond e_max: in no bin
    for (int c = 0; c < Species::NC; ++c) h[c].push_back(v[c]);
    const std::vector<double> got = beam_spectrum(make_species(h, -1, mass), 30, bins, ekmax, comm);
    double err = 0;
    for (int b = 0; b < bins; ++b) err = std::max(err, std::abs(got[size_t(b)] - want[size_t(b)]));
    check("beam diagnostics: spectrum exact at bin centres, gamma_min, e_max", err < 1e-12 && got.size() == size_t(bins),
          fmt("max error %.1e", err));
}

// D4. A run with a diagnostic block that writes the particles: the files of all ranks read back give
//     exactly the particles of the species above gamma_min (values and count), and beam_*.txt exists.
static void diag_particles(Comm& comm) {
    Config c;
    c.N[0] = 16; c.N[1] = 8; c.N[2] = 8;
    c.h[0] = 0.3; c.h[1] = 0.5; c.h[2] = 0.5;
    c.t_end = 4 * c.h[0];
    c.dir = "test_out_diag";
    SpeciesConfig e;
    e.name = "e"; e.q = -1; e.m = 1; e.density = 1e-3;
    e.ppc[0] = 2;
    for (int a = 0; a < 3; ++a) { e.uth[a] = 3; e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
    c.species.push_back(e);
    BeamDiagConfig b;
    b.name = "hot"; b.species = "e"; b.every = 4; b.gamma_min = 4; b.particles = true;
    c.diags.push_back(b);
    c.check();
    Simulation sim(c, comm);
    sim.init();
    while (!sim.finished()) sim.step();
    // the species above gamma_min now, in index order (global over ranks via counts)
    const std::vector<double> x = sim.species[0].get(Species::X), ux = sim.species[0].get(Species::UX),
                              uy = sim.species[0].get(Species::UY), uz = sim.species[0].get(Species::UZ);
    std::vector<double> mine;
    for (size_t p = 0; p < x.size(); ++p)
        if (std::sqrt(1 + ux[p] * ux[p] + uy[p] * uy[p] + uz[p] * uz[p]) >= b.gamma_min) mine.push_back(x[p]);
    char fn[128];
    std::snprintf(fn, sizeof fn, "%s/particles/hot_%06ld_r%d.bin", c.dir.c_str(), sim.istep, comm.rank);
    std::vector<double> got;
    if (FILE* f = std::fopen(fn, "rb")) {
        double v[7];
        while (std::fread(v, sizeof(double), 7, f) == 7) got.push_back(v[0]);
        std::fclose(f);
    }
    double bad = got.size() == mine.size() ? 0 : 1;
    for (size_t q = 0; q < got.size() && q < mine.size(); ++q) bad += got[q] != mine[q];
    double n = double(mine.size());
    comm.allreduce_sum(&bad, 1);
    comm.allreduce_sum(&n, 1);
    FILE* bf = std::fopen((c.dir + "/beam_hot.txt").c_str(), "r");
    const bool beam = comm.rank != 0 || bf != nullptr;
    if (bf) std::fclose(bf);
    check("beam diagnostics: particle files = selected particles, in index order", bad == 0 && n > 0 && beam,
          fmt("%.0f particles above gamma_min, %.0f mismatches", n, bad));
}

void test_diag(Comm& comm) {
    diag_moments(comm);
    diag_emittance(comm);
    diag_spectrum(comm);
    diag_particles(comm);
}
