// Dielectrics in the RIP-CC step of Cart3D and 2D (dielectric3d.hpp, docs/specs/dielectric3d.md, decision 100). References: the
// Gauss law div D = rho and the energy (E.D + B^2) / 2 of the scheme, the Fresnel coefficients of a plane interface (normal and
// oblique incidence, both polarizations), the moving window of decision 100 (front layer, entry from div D, the warning).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include "common.hpp"
#include "dielectric3d.hpp"
#include "fields.hpp"
#include "sim.hpp"

namespace {
Domain box(Comm& comm, int nx, int ny, int nz, double hx, double hy, double hz, bool flat) {
    Domain d;
    d.N[0] = nx; d.N[1] = ny; d.N[2] = flat ? 1 : nz;
    d.h[0] = hx; d.h[1] = hy; d.h[2] = hz;
    d.flat = flat;
    d.ng = 3;
    const int want[3] = {1, 1, 1};
    comm.decompose(d, want);
    return d;
}
std::vector<Field*> six(Fields& f) { return {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz}; }
Dielectric3Config dblock(const std::string& eps) {
    Dielectric3Config c;
    c.fn = Expr(eps, {"x", "y", "z"}, [](const std::string&, double&) { return false; });
    return c;
}
double max_abs_diff(const Field& a, const Field& b, double* scale) {
    const std::vector<double> x = a.host(), y = b.host();
    double m = 0, s = 0;
    for (size_t q = 0; q < x.size(); ++q) { m = std::max(m, std::abs(x[q] - y[q])); s = std::max(s, std::abs(y[q])); }
    if (scale) *scale = std::max(*scale, s);
    return m;
}

// E1. Fields only, periodic box, random fields, a sphere (eps 2.5) and a tilted slab (eps 3.8) with cut cells, 400 steps: div D
//     (gauss_residual with rho = 0) unchanged to round-off, the energy (E.D + B^2) / 2 bounded; 3D and 2D.
void gauss_energy(Comm& comm, bool flat) {
    const int nx = 32, ny = 16, nz = 16;
    const double hx = 0.1, hy = 0.15, hz = 0.15;
    Domain d = box(comm, nx, ny, nz, hx, hy, hz, flat);
    Fields f(d);
    Dielectric3 diel({dblock("(x - 1.6)^2 + (y - 1.2)^2 + (z - 1.2)^2 < 0.81 ? 2.5 : 1"), dblock("abs(x + 0.4 * y - 0.6) < 0.35 ? 3.8 : 1")}, d);
    f.diel = &diel;
    std::mt19937 rng(9);
    std::uniform_real_distribution<double> U(-1, 1);
    for (Field* F : six(f)) {     // smooth random fields: 4 periodic harmonics of random amplitude and phase per component
        std::vector<double> h = F->host();
        double a[4], ph[4];
        int kx[4], ky[4], kz[4];
        for (int m = 0; m < 4; ++m) {
            a[m] = U(rng); ph[m] = M_PI * U(rng);
            kx[m] = 1 + int(2.99 * (0.5 + 0.5 * U(rng))); ky[m] = int(2.99 * (0.5 + 0.5 * U(rng))); kz[m] = flat ? 0 : int(2.99 * (0.5 + 0.5 * U(rng)));
        }
        for (int i = -d.ng; i < d.n[0] + d.ng; ++i)
            for (int j = -d.ng; j < d.n[1] + d.ng; ++j)
                for (int k = flat ? 0 : -d.ng; k < (flat ? 1 : d.n[2] + d.ng); ++k) {
                    double v = 0;
                    for (int m = 0; m < 4; ++m)
                        v += a[m] * std::sin(2 * M_PI * (kx[m] * double(i) / nx + ky[m] * double(j) / ny + kz[m] * double(k) / nz) + ph[m]);
                    h[F->idx(i, j, k)] = v;
                }
        F->upload(h);
    }
    comm.exchange(d, six(f));
    f.zero_currents();
    Field g0(d), g1(d);
    f.rho.fill(0);
    f.gauss_residual(g0);
    double w0 = f.energy(), wmin = 1e300, wmax = 0;
    comm.allreduce_sum(&w0, 1);
    for (int n = 0; n < 400; ++n) {
        f.zero_currents(); f.step(comm);
        double w = f.energy();
        comm.allreduce_sum(&w, 1);
        wmin = std::min(wmin, w); wmax = std::max(wmax, w);
    }
    f.gauss_residual(g1);
    double sc = 0;
    const double dg = comm.allreduce_max(max_abs_diff(g1, g0, &sc)) / comm.allreduce_max(sc);
    check(std::string("dielectric3d E1 (") + (flat ? "2D" : "3D") + "): random fields, sphere eps 2.5 + tilted slab eps 3.8, 400 steps: div D unchanged, "
              "energy (E.D + B^2)/2 bounded",
          dg < 1e-12 && wmax < 1.02 * w0 && wmin > 0.8 * w0 && std::isfinite(wmax),
          fmt("max change of div D %.1e of max", dg) + fmt(", W/W0 in [%.4f, ", wmin / w0) + fmt("%.4f]", wmax / w0));
}

// E2. Normal incidence on a dielectric half-space (eps 4, n = 2, from x = 40 on, 1D in a 3D / 2D box): a unipolar Gaussian pulse
//     (width w = 1 and 2: 20 and 40 cells) of E_y (B_z = E_y) or E_z (B_y = -E_z); after 40 time units the reflected pulse has the
//     amplitude r = (1 - n) / (1 + n) = -1/3 (Fresnel; the x transport of the vacuum side is dispersionless: exact) and the
//     transmitted one t = 2 / (1 + n) = 2/3, less the spreading by the grid dispersion inside the dielectric (the peak loss
//     falls steeply with the width: 4.1e-3 at w = 1, 1.5e-4 at w = 2).
void fresnel_run(Comm& comm, bool flat, int pol, double w, double& r, double& t) {
    const int nx = 1600;
    const double h = 0.05, x0 = 40.0;
    Domain d = box(comm, nx, 4, 4, h, 1.0, 1.0, flat);
    Fields f(d);
    Dielectric3 diel({dblock("x >= 40 ? 4 : 1")}, d);
    f.diel = &diel;
    {
        std::vector<double> e = (pol ? f.ez : f.ey).host(), b = (pol ? f.by : f.bz).host();
        Field& E = pol ? f.ez : f.ey;
        for (int i = 0; i < d.n[0]; ++i) {
            const double x = (d.off[0] + i) * h, v = std::exp(-std::pow((x - 20.0) / w, 2));
            for (int j = 0; j < d.n[1]; ++j)
                for (int k = 0; k < d.n[2]; ++k) { e[E.idx(i, j, k)] = v; b[E.idx(i, j, k)] = pol ? -v : v; }
        }
        E.upload(e);
        (pol ? f.by : f.bz).upload(b);
    }
    comm.exchange(d, six(f));
    for (int n = 0; n < 800; ++n) { f.zero_currents(); f.step(comm); }
    const std::vector<double> e = (pol ? f.ez : f.ey).host();
    const Field& E = pol ? f.ez : f.ey;
    double rmin = 0, tmax = 0;
    for (int i = 0; i < d.n[0]; ++i) {
        const double x = (d.off[0] + i) * h, v = e[E.idx(i, 0, 0)];
        if (x < x0 - 10) rmin = std::min(rmin, v);
        else if (x > x0) tmax = std::max(tmax, v);
    }
    r = -comm.allreduce_max(-rmin);
    t = comm.allreduce_max(tmax);
}
void fresnel_normal(Comm& comm, bool flat, int pol) {
    double r1, t1, r2, t2;
    fresnel_run(comm, flat, pol, 1.0, r1, t1);
    fresnel_run(comm, flat, pol, 2.0, r2, t2);
    const double er = std::max(std::abs(r1 / (-1.0 / 3) - 1), std::abs(r2 / (-1.0 / 3) - 1));
    const double e1 = t1 / (2.0 / 3) - 1, e2 = t2 / (2.0 / 3) - 1;
    check(std::string("dielectric3d E2 (") + (flat ? "2D" : "3D") + (pol ? ", E_z" : ", E_y") +
              "): normal incidence on eps 4: reflected -1/3 exactly, transmitted 2/3 to the grid dispersion (falls with the pulse width)",
          er < 1e-12 && std::abs(e2) < 1.5e-3 && e1 / e2 > 3,
          fmt("r - 1/3: %.1e;", er) + fmt(" t = %.5f", t1) + fmt(" (%+.1e, w = 1),", e1) + fmt(" %.5f", t2) + fmt(" (%+.1e, w = 2)", e2));
}

// E3. Oblique incidence (2D, 30 degrees, wavelength 1 = 20-25 cells, envelope width 4): the reflected amplitude against Fresnel for
//     eps 4: p (B_z, in-plane E with the normal E_x across the interface) r_p = 0.28286 (B), s (E_z) r_s = -0.38198 (E). The p
//     wave starts with the discrete Gauss law (E_x from the transverse divergence, summed along x).
void fresnel_oblique(Comm& comm, bool p) {
    const double k = 2 * M_PI, th = M_PI / 6, kx = k * std::cos(th), ky = k * std::sin(th), Ly = 2 * M_PI / ky;
    const int nx = 1000, ny = 40;
    const double hx = 0.04, hy = Ly / ny, x0 = 20.0, x1 = 10.0, w = 4.0;
    Domain d = box(comm, nx, ny, 1, hx, hy, 1.0, true);
    Fields f(d);
    Dielectric3 diel({dblock("x >= 20 ? 4 : 1")}, d);
    f.diel = &diel;
    const double n = 2, c = std::cos(th), s2 = std::sin(th) * std::sin(th), sq = std::sqrt(n * n - s2);
    const double ref = p ? (n * n * c - sq) / (n * n * c + sq) : (c - sq) / (c + sq);
    auto env = [&](double x) { return std::exp(-std::pow((x - x1) / w, 2)); };
    if (p) {      // B_z and E_y at (i, j+1/2), E_x at (i, j): E = (-ky/k, kx/k) B_z
        std::vector<double> bz = f.bz.host(), ey = f.ey.host(), ex = f.ex.host();
        for (int i = 0; i < d.n[0]; ++i)
            for (int j = 0; j < d.n[1]; ++j) {
                const double x = (d.off[0] + i) * hx, y = (d.off[1] + j + 0.5) * hy, b = env(x) * std::cos(kx * x + ky * y);
                bz[f.bz.idx(i, j, 0)] = b; ey[f.ey.idx(i, j, 0)] = kx / k * b;
            }
        // E_x from the discrete Gauss law (Ex(i+1) - Ex(i)) / hx = -(mu_x Ey(j) - mu_x Ey(j-1)) / hy, from the left edge (one rank)
        for (int j = 0; j < d.n[1]; ++j) {
            double e = 0;
            for (int i = 0; i < d.n[0]; ++i) {
                ex[f.ex.idx(i, j, 0)] = e;
                if (i + 1 < d.n[0]) {
                    auto Ey = [&](int ii, int jj) { jj = (jj + d.n[1]) % d.n[1]; return ey[f.ey.idx(ii, jj, 0)]; };
                    e -= hx * (0.5 * (Ey(i, j) + Ey(i + 1, j)) - 0.5 * (Ey(i, j - 1) + Ey(i + 1, j - 1))) / hy;
                }
            }
        }
        f.bz.upload(bz); f.ey.upload(ey); f.ex.upload(ex);
    } else {      // E_z at (i, j), B_x, B_y: B = k x E / k: B_x = ky/k E_z, B_y = -kx/k E_z (B_x at (i, j+1/2))
        std::vector<double> ez = f.ez.host(), by = f.by.host(), bx = f.bx.host();
        for (int i = 0; i < d.n[0]; ++i)
            for (int j = 0; j < d.n[1]; ++j) {
                const double x = (d.off[0] + i) * hx, y = (d.off[1] + j) * hy, e = env(x) * std::cos(kx * x + ky * y);
                ez[f.ez.idx(i, j, 0)] = e; by[f.by.idx(i, j, 0)] = -kx / k * e;
                const double yh = (d.off[1] + j + 0.5) * hy;
                bx[f.bx.idx(i, j, 0)] = ky / k * env(x) * std::cos(kx * x + ky * yh);
            }
        f.ez.upload(ez); f.by.upload(by); f.bx.upload(bx);
    }
    comm.exchange(d, six(f));
    const Field& F = p ? f.bz : f.ez;
    auto amp = [&](double xa, double xb) {
        const std::vector<double> v = F.host();
        double m = 0;
        for (int i = 0; i < d.n[0]; ++i) {
            const double x = (d.off[0] + i) * hx;
            if (x < xa || x > xb) continue;
            for (int j = 0; j < d.n[1]; ++j) m = std::max(m, std::abs(v[F.idx(i, j, 0)]));
        }
        return comm.allreduce_max(m);
    };
    const double a0 = amp(0, x0 - 2);
    const int steps = int(2 * (x0 - x1) / c / hx);
    for (int q = 0; q < steps; ++q) { f.zero_currents(); f.step(comm); }
    const double a1 = amp(0, x0 - 6);
    const double e = a1 / a0 / std::abs(ref) - 1;
    check(std::string("dielectric3d E3 (2D, 30 deg, ") + (p ? "p" : "s") + "): reflected amplitude against Fresnel for eps 4",
          std::abs(e) < 0.02, fmt("|r| = %.5f", a1 / a0) + fmt(" (Fresnel %.5f,", std::abs(ref)) + fmt(" %+.1e)", e));
}

// E4. Deck path, 2D moving window: a rigid bunch at gamma 1000 between two dielectric slabs (eps 3.8, |y| > 12 um, from 0.2 mm on):
//     the Gauss law div D - rho at round-off and no growth at the front edge (the front layer and the entry of decision 100); a
//     bunch loaded inside the front layer is warned (decision 100).
void window_deck(Comm& comm) {
    const char* path = "diel3_win_tmp.deck";
    auto run = [&](double xb, double t_ps, double& gmax, double& w1, double& w2) {
        if (comm.rank == 0) {
            std::ofstream f(path);
            f << "grid {\n lambda0 = 100 um\n geometry = cartesian2d\n nx = 600\n ny = 80\n dx = 0.4 um\n dy = 0.5 um\n"
                 " boundary_x = open\n t_end = " << t_ps << " ps\n}\nnumerics {\n solver = rip_cc\n}\n"
                 "window {\n start = 0 fs\n velocity = 1\n}\n"
                 "dielectric lining {\n eps = x > 0.2 mm && abs(y - 20 um) > 12 um ? 3.8 : 1\n}\n"
                 "species bunch {\n charge = 1\n mass = 18361526.7\n density = 1\n density_function = gauss(x - " << xb << " um, 8 um) * gauss(y - 20 um, 3 um)\n"
                 " xmin = " << xb - 30 << " um\n xmax = " << xb + 30 << " um\n ymin = 10 um\n ymax = 30 um\n ux = 999.9995\n rigid_until = 1 ns\n ppc_x = 1\n ppc_y = 2\n}\n"
                 "output {\n dir = diel3_win_out\n every = 100000\n}\n";
        }
        comm.barrier();
        Deck dk = Deck::read(path);
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        const long w0 = sim.front_layer_warnings;
        gmax = 0; w1 = w2 = 0;
        long n = 0;
        while (!sim.finished()) {
            sim.step();
            if (++n % 250 == 0) {
                double wf = sim.fields->energy();
                comm.allreduce_sum(&wf, 1);
                gmax = std::max(gmax, sim.gauss_drift());
                if (n == 1500) w1 = wf;
                w2 = wf;
            }
        }
        comm.barrier();
        if (comm.rank == 0) std::remove(path);
        return std::make_pair(w0, sim.front_layer_warnings);
    };
    double g, w1, w2, g2, a, b;
    const auto r = run(120, 4.0, g, w1, w2);
    const auto q = run(234, 0.1, g2, a, b);
    check("dielectric3d E4 (2D window, eps 3.8 lining, rigid bunch): div D - rho at round-off, no growth at the front edge, the "
          "front-layer warning",
          g < 1e-11 && std::isfinite(w2) && w2 < 1.5 * w1 && r.second == 0 && q.first >= 1,
          fmt("Gauss %.1e", g) + fmt(", W(3000) / W(1500) = %.4f", w2 / w1) + fmt(", warnings %.0f", double(r.second)) +
              fmt("; bunch in the layer: %.0f warning(s)", double(q.first)));
}
}  // namespace

void test_dielectric3d(Comm& comm) {
    gauss_energy(comm, false);
    gauss_energy(comm, true);
    if (comm.size == 1) {
        for (int pol = 0; pol < 2; ++pol) { fresnel_normal(comm, false, pol); fresnel_normal(comm, true, pol); }
        fresnel_oblique(comm, true);
        fresnel_oblique(comm, false);
    }
    window_deck(comm);
}
