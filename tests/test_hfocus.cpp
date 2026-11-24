// CHF diagnostic (docs/specs/harmonic_focus.md, src/hfocus.*): the band-pass and the focus maps.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include "common.hpp"
#include "sim.hpp"

namespace {
// the maps of rank 0 (the whole region on one rank): [quantities][nodes]
std::vector<double> read_maps(const std::string& dir) {
    std::vector<double> v;
    std::ifstream b(dir + "/harmonic_focus/r0.bin", std::ios::binary);
    for (double x; b.read(reinterpret_cast<char*>(&x), sizeof x);) v.push_back(x);
    return v;
}

Config run_deck(Comm& comm, const std::string& text, const std::function<void(Simulation&)>& after_step) {
    const char* path = "hfocus_test_tmp.deck";
    { std::ofstream f(path); f << text; }
    Config c = Config::from_deck(Deck::read(path));
    Simulation sim(c, comm);
    sim.init();
    while (!sim.finished()) {
        sim.step();
        after_step(sim);
    }
    return c;
}

// The band-pass of three field components at one point, run on the host from the coefficients (direct form II
// transposed), with the peak of |E_band|^2, its time and the fluence.
struct Replay {
    HFFilter f;
    std::vector<double> s;
    double peak = 0, tpeak = 0, fluence = 0;
    explicit Replay(const HFFilter& fl) : f(fl), s(size_t(6 * fl.ns), 0.0) {}
    double one(int c, double x) {
        for (int k = 0; k < f.ns; ++k) {
            double& s1 = s[size_t(2 * (c * f.ns + k))];
            double& s2 = s[size_t(2 * (c * f.ns + k) + 1)];
            const double y = f.b0[k] * x + s1;
            s1 = f.b1[k] * x - f.a1[k] * y + s2;
            s2 = f.b2[k] * x - f.a2[k] * y;
            x = y;
        }
        return x;
    }
    void add(const double e[3], double t, double dt) {
        double I = 0;
        for (int c = 0; c < 3; ++c) { const double y = one(c, e[c]); I += y * y; }
        if (I > peak) { peak = I; tpeak = t; }
        fluence += I * dt;
    }
};
}  // namespace

void test_hfocus(Comm& comm) {
    // H1: the cascade of the coefficients on a sine of frequency w: the steady amplitude (least squares over the last
    // 8 periods) against the analytic bilinear Butterworth |H| = [1 + (K_lo / K)^2n]^-1/2 [1 + (K / K_hi)^2n]^-1/2,
    // K = tan(w dt / 2); also hfocus_gain.
    {
        const double dt = 2 * M_PI / 64, nlo = 4.5, nhi = 25;
        const int order = 8;
        const HFFilter f = hfocus_filter(nlo, nhi, order, dt);
        double err = 0, errg = 0;
        for (double w : {0.5, 1.0, 3.0, 4.5, 6.0, 10.0, 20.0, 25.0, 28.0}) {
            Replay r(f);
            const long n = 40000, nfit = long(std::lround(8 * 2 * M_PI / (w * dt)));
            double sc = 0, cc = 0, ss = 0, ys = 0, yc = 0;
            for (long k = 0; k < n; ++k) {
                const double t = k * dt, y = r.one(0, std::sin(w * t));
                if (k >= n - nfit) {
                    const double a = std::sin(w * t), b = std::cos(w * t);
                    ss += a * a; cc += b * b; sc += a * b; ys += y * a; yc += y * b;
                }
            }
            const double det = ss * cc - sc * sc, A = (ys * cc - yc * sc) / det, B = (yc * ss - ys * sc) / det;
            const double K = std::tan(0.5 * w * dt), Kl = std::tan(0.5 * nlo * dt), Kh = std::tan(0.5 * nhi * dt);
            const double g = 1 / std::sqrt((1 + std::pow(Kl / K, 2 * order)) * (1 + std::pow(K / Kh, 2 * order)));
            err = std::max(err, std::abs(std::hypot(A, B) - g));
            errg = std::max(errg, std::abs(hfocus_gain(f, w, dt) - g));
        }
        std::printf("  H1 band-pass 4.5 .. 25, 8 poles per edge, dt = 2 pi / 64: steady amplitude - Butterworth %.1e, "
                    "hfocus_gain - Butterworth %.1e\n", err, errg);
        check("H1 band-pass = bilinear Butterworth", err < 1e-9 && errg < 1e-13, fmt("%.1e %.1e", err, errg));
    }
    // H2: 2D, a plane pulse (waist 1e3 um, periodic y) through the box, band 0.5 .. 1.5 around the laser.
    // (a) the maps at nodes against the host replay of the grid's field (Ex, Ey and Ez averaged to the node): exact;
    // (b) the fluence at every node against a0^2 int g^2 sin^2 dt = a0^2 (tau / 2) sqrt(pi / (4 ln 2)) (the filter's
    // |H|^2 over the spectrum of the 6 fs pulse: 3e-3 low), the peak against a0^2 (the filter's phase: 4 % low).
    {
        const std::string deck = R"(
grid {
  geometry = cartesian2d
  lambda0 = 0.8 um
  nx = 160
  ny = 4
  dx = 0.025 um
  dy = 0.1 um
  boundary_x = open
  t_end = 40 fs
}
laser {
  a0 = 0.5
  polarization = y
  duration = 6 fs
  waist = 1000 um
}
harmonic_focus {
  n_min = 0.5
  n_max = 1.5
  xmin = 0.5 um
  xmax = 3.5 um
}
output { dir = hfocus_test_out
  every = 100000 }
)";
        const int gi[3] = {20, 80, 140}, gj[2] = {0, 2};
        std::vector<Replay> rep;
        const Config c = run_deck(comm, deck, [&](Simulation& s) {
            if (rep.empty()) rep.assign(6, Replay(hfocus_filter(0.5, 1.5, 8, s.dom.dt())));
            H ex(s.fields->ex), ey(s.fields->ey), ez(s.fields->ez);
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 2; ++b) {
                    const int i = gi[a], j = gj[b];
                    const double e[3] = {ex(i, j, 0), 0.5 * (ey(i, j - 1, 0) + ey(i, j, 0)), ez(i, j, 0)};
                    rep[size_t(2 * a + b)].add(e, s.t, s.dom.dt());
                }
        });
        const double a0 = 0.5, tau = c.laser.tau;
        const double flu = a0 * a0 * 0.5 * tau * std::sqrt(M_PI / (4 * std::log(2.0)));
        if (comm.rank == 0) {
            const std::vector<double> m = read_maps(c.dir);
            const size_t n = m.size() / 3;
            double ef = 0, ep = 0, er = 0;
            for (size_t q = 0; q < n; ++q) {
                ef = std::max(ef, std::abs(m[2 * n + q] / flu - 1));
                ep = std::max(ep, std::abs(m[q] / (a0 * a0) - 1));
            }
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 2; ++b) {
                    const Replay& r = rep[size_t(2 * a + b)];
                    const size_t q = size_t(gj[b] * 121 + gi[a] - 20);
                    er = std::max({er, std::abs(m[q] - r.peak) / r.peak, std::abs(m[n + q] - r.tpeak) / r.tpeak,
                                   std::abs(m[2 * n + q] - r.fluence) / r.fluence});
                }
            std::printf("  H2 2D plane pulse, band 0.5 .. 1.5: %zu nodes, maps - host replay %.1e; fluence / analytic - 1 <= %.2e, "
                        "peak / a0^2 - 1 <= %.2e\n", n, er, ef, ep);
            check("H2 maps = host replay (2D)", n == 121 * 4 && er < 1e-12, fmt("%.1e", er));
            check("H2 fluence and peak of a plane pulse", ef < 4e-3 && ep < 5e-2, fmt("%.2e %.2e", ef, ep));
        }
    }
    // H3: RZ, a Gaussian beam (2 modes, polarised along x) through its focus: the maps at phi = 0 and pi/2 against
    // the host replay of the Cartesian field from rz_gather_point (E_r interpolated linearly to the node), on the axis
    // and off it. Exact up to round-off: the rotation by phi commutes with the filter and leaves |E|^2 unchanged.
    {
        const std::string deck = R"(
grid {
  geometry = rz
  modes = 2
  lambda0 = 0.8 um
  nz = 240
  nr = 48
  dz = 0.025 um
  dr = 0.05 um
  boundary_z = open
  t_end = 40 fs
}
laser {
  a0 = 1
  polarization = x
  duration = 6 fs
  waist = 1 um
  focus = 3 um
}
harmonic_focus {
  n_min = 0.5
  n_max = 1.5
  zmin = 1 um
  zmax = 5 um
  rmax = 1.5 um
}
output { dir = hfocus_test_rz
  every = 100000 }
)";
        // points: (z node, r node, phi index 0 / 1)
        const int pz[5] = {120, 120, 120, 80, 200}, pr[5] = {0, 10, 10, 20, 5}, pa[5] = {0, 0, 1, 0, 1};
        std::vector<Replay> rep;
        int nzr = 0, nrr = 0;
        const Config c = run_deck(comm, deck, [&](Simulation& s) {
            if (rep.empty()) rep.assign(5, Replay(hfocus_filter(0.5, 1.5, 8, s.dom.dt())));
            for (int p = 0; p < 5; ++p) {
                const double z = pz[p] * s.dom.h[0], r = s.rz->grid().r[size_t(pr[p])];
                double E[3], B[3];
                rz_gather_point(*s.rz, z, pa[p] ? 0.0 : r, pa[p] ? r : 0.0, E, B, 1);
                rep[size_t(p)].add(E, s.t, s.dom.dt());
            }
        });
        nzr = 161; nrr = 31;                       // nodes z 40 .. 200, r 0 .. 30
        if (comm.rank == 0) {
            const std::vector<double> m = read_maps(c.dir);
            const size_t n = m.size() / 6;
            double er = 0;
            for (int p = 0; p < 5; ++p) {
                const Replay& r = rep[size_t(p)];
                const size_t q = size_t(pr[p] * nzr + pz[p] - 40), o = size_t(3 * pa[p]) * n;
                er = std::max({er, std::abs(m[o + q] - r.peak) / r.peak, std::abs(m[o + 2 * n + q] - r.fluence) / r.fluence});
            }
            std::printf("  H3 RZ Gaussian beam, band 0.5 .. 1.5: %zu nodes, maps (phi = 0, pi/2, axis) - host replay %.1e\n", n, er);
            check("H3 maps = host replay (RZ)", n == size_t(nzr * nrr) && er < 1e-10, fmt("%.1e", er));
        }
    }
}
