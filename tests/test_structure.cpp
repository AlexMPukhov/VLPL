// Structure runs in Cart3D (docs/specs/structure3d.md, decision 102): the embedded conductor (block metal), external
// quadrupoles, beams from openPMD files and the wake diagnostics. References: the discrete and continuum mode spectra of
// rectangular and round pipes, the exact reflection of the dispersionless x transport, the Gauss law, thick-lens transfer
// matrices, moments computed here from the written file, Poynting energy balance, the integral of a backward plane wave.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include "common.hpp"
#include "metal3d.hpp"
#include "external.hpp"
#include "particles.hpp"
#include "sim.hpp"
#ifdef RIP_HDF5
#include "h5.hpp"
#endif

namespace {
Expr metal_expr(const std::string& t) {
    const UnitFn u = [](const std::string&, double&) { return false; };
    return Expr(t, {"x", "y", "z"}, u);
}

Domain box3(Comm& comm, int nx, int ny, int nz, double hx, double hy, double hz, bool px = true) {
    Domain d;
    d.N[0] = nx; d.N[1] = ny; d.N[2] = nz;
    d.h[0] = hx; d.h[1] = hy; d.h[2] = hz;
    d.periodic[0] = px;
    d.ng = 3;
    const int want[3] = {1, 1, 1};
    comm.decompose(d, want);
    return d;
}

std::vector<Field*> six(Fields& f) { return {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz}; }

// w dt of a single (kx, mode) harmonic from the increments: d^{n+1} + d^{n-1} = 2 cos(w dt) d^n (least squares over all
// field values; the static part drops out); res: the relative residual of that recurrence
double recurrence_wdt(Fields& f, Comm& comm, int steps, double& res) {
    std::vector<std::vector<double>> hist;
    for (int n = 0; n < steps; ++n) {
        std::vector<double> v;
        for (Field* F : six(f)) { std::vector<double> h = F->host(); v.insert(v.end(), h.begin(), h.end()); }
        hist.push_back(v);
        f.zero_currents(); f.step(comm);
    }
    double num = 0, den = 0;
    for (size_t n = 1; n + 2 < hist.size(); ++n)
        for (size_t q = 0; q < hist[n].size(); ++q) {
            const double dm = hist[n][q] - hist[n - 1][q], d0 = hist[n + 1][q] - hist[n][q], dp = hist[n + 2][q] - hist[n + 1][q];
            num += (dp + dm) * 2 * d0; den += 4 * d0 * d0;
        }
    const double cw = num / den;
    double r = 0, sc = 0;
    for (size_t n = 1; n + 2 < hist.size(); ++n)
        for (size_t q = 0; q < hist[n].size(); ++q) {
            const double dm = hist[n][q] - hist[n - 1][q], d0 = hist[n + 1][q] - hist[n][q], dp = hist[n + 2][q] - hist[n + 1][q];
            r = std::max(r, std::abs(dp + dm - 2 * cw * d0)); sc = std::max(sc, std::abs(d0));
        }
    res = r / sc;
    return std::acos(cw);
}

// S1. Rectangular pipe a x b (walls on node planes), uniform along x: TM_21 (Ex) and TE_11 (Bx) with kx = 0 and
//     kx = 2 pi / L against the RIP-CC relation sin^2(w dt/2) = sin^2(kx dx/2) + dx^2 Lam/4 cos^2(kx dx/2) with the discrete
//     transverse eigenvalue Lam = (2/hy)^2 sin^2(m pi hy / 2a) + (2/hz)^2 sin^2(n pi hz / 2b): exact (round-off). The
//     continuum cutoff (m pi / a)^2 + (n pi / b)^2 is approached to second order (printed).
void rect_pipe(Comm& comm) {
    double worst = 0, wres = 0, last = 0, lastp = 0;
    for (int te = 0; te < 2; ++te)
        for (int kxm = 0; kxm < 2; ++kxm) {
            const int ny = 16, nz = 14, ja = 2, ma = 11, ka = 3, mb = 8;   // walls at j = 2, 13 and k = 3, 11
            Domain d = box3(comm, 8, ny, nz, 0.25, 0.4, 0.35);
            const double y0 = ja * d.h[1], y1 = (ja + ma) * d.h[1], z0 = ka * d.h[2], z1 = (ka + mb) * d.h[2];
            char t[256];
            std::snprintf(t, sizeof t, "(y > %.17g && y < %.17g && z > %.17g && z < %.17g) ? -1 : 1", y0 + 0.01, y1 - 0.01, z0 + 0.01,
                          z1 - 0.01);
            Metal3Config mc; mc.on = true; mc.fn = metal_expr(t);
            Metal3 M(d, mc);
            Fields f(d);
            f.metal = &M;
            const int m = te ? 1 : 2, n = 1;
            const double a = ma * d.h[1], b = mb * d.h[2], kx = kxm ? 2 * M_PI / d.L(0) : 0.0;
            H Ex(f.ex), Bx(f.bx);
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < d.n[0]; ++i) {
                        const double cx = std::cos(kx * i * d.h[0] + 0.4);
                        if (!te && j > ja && j < ja + ma && k > ka && k < ka + mb)
                            Ex(i, j, k) = std::sin(m * M_PI * (j - ja) / ma) * std::sin(n * M_PI * (k - ka) / mb) * cx;
                        if (te && j >= ja && j < ja + ma && k >= ka && k < ka + mb)
                            Bx(i, j, k) = std::cos(m * M_PI * (j + 0.5 - ja) / ma) * std::cos(n * M_PI * (k + 0.5 - ka) / mb) * cx;
                    }
            Ex.put(f.ex); Bx.put(f.bx);
            comm.exchange(d, {&f.ex, &f.bx});
            double res;
            const double wdt = recurrence_wdt(f, comm, 40, res);
            const double sy = std::sin(m * M_PI * d.h[1] / (2 * a)), sz = std::sin(n * M_PI * d.h[2] / (2 * b));
            const double lam = 4 * sy * sy / (d.h[1] * d.h[1]) + 4 * sz * sz / (d.h[2] * d.h[2]);
            const double D = d.h[0], sx = std::sin(kx * D / 2), cx = std::cos(kx * D / 2);
            const double pred = 2 * std::asin(std::sqrt(sx * sx + D * D * lam / 4 * cx * cx));
            worst = std::max(worst, std::abs(wdt - pred));
            wres = std::max(wres, res);
            if (te && kxm == 0) {               // the continuum cutoff
                last = 2 / D * std::sin(wdt / 2);
                lastp = std::sqrt(std::pow(m * M_PI / a, 2) + std::pow(n * M_PI / b, 2));
            }
        }
    check("metal S1: rectangular pipe TM21 / TE11 (kx = 0, 2 pi / L) = discrete spectrum", worst < 1e-11 && wres < 1e-9,
          fmt("max |w dt - pred| %.1e, residual %.1e", worst, wres) + fmt(", TE11 cutoff %.5f vs continuum %.5f", last, lastp));
}

// frequency (w dt) of the dominant oscillation of a(n): the peak of the Hann-windowed spectrum near w0, refined
double peak_wdt(const std::vector<double>& a, double w0) {
    const size_t N = a.size();
    auto amp = [&](double w) {
        std::complex<double> s = 0;
        for (size_t n = 0; n < N; ++n) s += a[n] * (0.5 - 0.5 * std::cos(2 * M_PI * n / (N - 1))) * std::exp(std::complex<double>(0, -w * n));
        return std::abs(s);
    };
    double best = w0, bv = 0;
    for (int q = -400; q <= 400; ++q) {
        const double w = w0 * (1 + 0.0005 * q);
        const double v = amp(w);
        if (v > bv) { bv = v; best = w; }
    }
    double lo = best * 0.999, hi = best * 1.001;   // golden-section refinement
    for (int it = 0; it < 60; ++it) {
        const double m1 = lo + 0.382 * (hi - lo), m2 = lo + 0.618 * (hi - lo);
        if (amp(m1) > amp(m2)) hi = m2; else lo = m1;
    }
    return 0.5 * (lo + hi);
}

// S2. Round pipe of radius R (staircase), kx = 0: the TM01 and TE11 cutoffs against j01 / R and j'11 / R. The staircase
//     error is first order in h / R; checked at R / h = 8, 16, 32 (both must fall; TM01 below 1.2 %, TE11 below 2.5 % at 32).
void round_pipe(Comm& comm) {
    const double j01 = 2.404825557695773, j11p = 1.841183781340659;
    std::string info;
    bool ok = true;
    double prev[2] = {1, 1};
    for (int R8 : {8, 16, 32}) {
        double err[2];
        for (int te = 0; te < 2; ++te) {
            const int n = 2 * R8 + 8;
            Domain d = box3(comm, 4, n, n, 0.5 / R8, 1.0 / R8, 1.0 / R8);   // R = 1
            const double c = 0.5 * n * d.h[1], R = 1.0;
            char t[160];
            std::snprintf(t, sizeof t, "(y - %.17g)^2 + (z - %.17g)^2 >= %.17g ? 1 : -1", c, c, R * R);
            Metal3Config mc; mc.on = true; mc.fn = metal_expr(t);
            Metal3 M(d, mc);
            Fields f(d);
            f.metal = &M;
            H Ex(f.ex), Bx(f.bx);
            for (int k = 0; k < n; ++k)
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < d.n[0]; ++i) {
                        if (!te) {
                            const double y = j * d.h[1] - c, z = k * d.h[2] - c, r = std::hypot(y, z);
                            if (r < R) Ex(i, j, k) = std::cyl_bessel_j(0, j01 * r / R);
                        } else {
                            const double y = (j + 0.5) * d.h[1] - c, z = (k + 0.5) * d.h[2] - c, r = std::hypot(y, z);
                            if (r < R) Bx(i, j, k) = std::cyl_bessel_j(1, j11p * r / R) * (r > 0 ? y / r : 0.0);
                        }
                    }
            Ex.put(f.ex); Bx.put(f.bx);
            comm.exchange(d, {&f.ex, &f.bx});
            const std::vector<double> f0 = te ? f.bx.host() : f.ex.host();
            const double kc = te ? j11p / R : j01 / R, D = d.h[0];
            const double w0 = 2 * std::asin(D * kc / 2);
            const int steps = int(30 * 2 * M_PI / w0);
            std::vector<double> a;
            for (int s = 0; s < steps; ++s) {
                const std::vector<double> v = te ? f.bx.host() : f.ex.host();
                double p = 0;
                for (size_t q = 0; q < v.size(); ++q) p += v[q] * f0[q];
                a.push_back(p);
                f.zero_currents(); f.step(comm);
            }
            const double wdt = peak_wdt(a, w0);
            const double kd = 2 / D * std::sin(wdt / 2);   // the transverse wavenumber of the scheme's mode
            err[te] = kd / kc - 1;
            info += fmt(" R/h %.0f:", R8) + fmt(te ? " TE11 %+.2e" : " TM01 %+.2e", err[te]);
        }
        if (R8 > 8) ok = ok && std::abs(err[0]) < std::abs(prev[0]) && std::abs(err[1]) < std::abs(prev[1]);
        prev[0] = err[0]; prev[1] = err[1];
        if (R8 == 32) ok = ok && std::abs(err[0]) < 0.012 && std::abs(err[1]) < 0.025;
    }
    check("metal S2: round pipe TM01 / TE11 cutoffs against j01 / R, j'11 / R", ok, info);
}

// S3. Normal incidence on the x faces of a metal slab (ky = kz = 0): the dispersionless transport reflects exactly, the
//     reflected pulse is the mirror image about the face column with E reversed (round-off); nothing enters the slab.
void x_face(Comm& comm) {
    Domain d = box3(comm, 300, 4, 4, 0.5, 0.6, 0.6);
    const int I1 = 140, I2 = 180;                   // the slab columns [I1, I2]
    char t[128];
    std::snprintf(t, sizeof t, "x > %.17g && x < %.17g ? 1 : -1", (I1 - 0.1) * d.h[0], (I2 + 0.1) * d.h[0]);
    Metal3Config mc; mc.on = true; mc.fn = metal_expr(t);
    Metal3 M(d, mc);
    Fields f(d);
    f.metal = &M;
    const double xa = 60 * d.h[0], xb = 250 * d.h[0], s = 4 * d.h[0];   // both fully reflected after 120 steps
    auto g = [&](double x, double x0) { return std::exp(-(x - x0) * (x - x0) / (s * s)); };
    H Ey(f.ey), Bz(f.bz), Ez(f.ez), By(f.by);
    for (int k = 0; k < 4; ++k)
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                const double x = i * d.h[0];
                Ey(i, j, k) = g(x, xa) + g(x, xb);          // A: T+_y toward the left face; B: T-_y toward the right face
                Bz(i, j, k) = g(x, xa) - g(x, xb);
                Ez(i, j, k) = 0.5 * g(x, xa) + 0.5 * g(x, xb);   // A: T-_z (moves +x); B: T+_z (moves -x)
                By(i, j, k) = -0.5 * g(x, xa) + 0.5 * g(x, xb);
            }
    Ey.put(f.ey); Bz.put(f.bz); Ez.put(f.ez); By.put(f.by);
    comm.exchange(d, six(f));
    const double w0 = f.energy();
    const int steps = 120;
    for (int n = 0; n < steps; ++n) { f.zero_currents(); f.step(comm); }
    // after 120 steps: A reflected at I1 (mirror about x1, moving left), B at I2 (moving right)
    const double x1 = I1 * d.h[0], x2 = I2 * d.h[0], T = steps * d.dt();
    H Ey1(f.ey), Bz1(f.bz), Ez1(f.ez), By1(f.by);
    double e = 0, inside = 0;
    for (int i = 0; i < d.n[0]; ++i) {
        const double x = i * d.h[0];
        const double ra = g(2 * x1 - x, xa + T), rb = g(2 * x2 - x, xb - T);   // mirrored pulses
        const double ey = -ra - rb, bz = ra - rb, ez = -0.5 * ra - 0.5 * rb, by = -0.5 * ra + 0.5 * rb;
        if (i >= I1 && i <= I2) {
            inside = std::max({inside, std::abs(Ey1(i, 1, 1)), std::abs(Ez1(i, 1, 1)),
                               i > I1 && i < I2 ? std::abs(Bz1(i, 1, 1)) + std::abs(By1(i, 1, 1)) : 0.0});
            continue;
        }
        e = std::max({e, std::abs(Ey1(i, 1, 1) - ey), std::abs(Bz1(i, 1, 1) - bz), std::abs(Ez1(i, 1, 1) - ez), std::abs(By1(i, 1, 1) - by)});
    }
    const double w1 = f.energy();
    check("metal S3: x faces reflect exactly (both polarizations, left / right face)", e < 1e-13 && inside < 1e-13 &&
          std::abs(w1 / w0 - 1) < 1e-13, fmt("max error %.1e, inside the slab %.1e", e, inside) + fmt(", energy %.1e", w1 / w0 - 1));
}

SpeciesConfig hot(const char* name, double q, double m, double n, double uth) {
    SpeciesConfig s;
    s.name = name; s.q = q; s.m = m; s.density = n;
    for (int a = 0; a < 3; ++a) { s.rmin[a] = -1e300; s.rmax[a] = 1e300; s.ppc[a] = 1; s.uth[a] = uth; }
    s.ppc[0] = 2;
    return s;
}

// S4. Plasma in a corrugated square tube (teeth along x): the Gauss law and div B at round-off at every checked point,
//     particles absorbed by the metal (also at the start), stable. Hot electrons in a periodic box (with and without CPML
//     layers in y, z), cold plasma in the moving window (the metal enters with the window). (A hot plasma that fills a
//     window from t = 0 breaks the frozen-entry bookkeeping of decision 91 also without metal: 0.12 at u_th = 0.2.)
void gauss_plasma(Comm& comm) {
    for (int run = 0; run < 3; ++run) {
        const bool pml = run == 1, win = run == 2;
        Config c;
        c.N[0] = 48; c.N[1] = 28; c.N[2] = 28;
        c.h[0] = 0.2; c.h[1] = 0.4; c.h[2] = 0.4;
        c.periodic[0] = !win;
        if (pml || win) { for (int a = 1; a < 3; ++a) { c.pml.on[a] = true; c.periodic[a] = false; } c.pml.cells = 6; }
        if (win) { c.window.enabled = true; c.window.start = 0; }
        c.t_end = 6;
        // tube |y - 5.6|, |z - 5.6| < 3 (teeth reaching 2 every 2.4, 1.2 long), walls up to 4.6
        c.metal.on = true;
        c.metal.fn = metal_expr("(max(abs(y - 5.6), abs(z - 5.6)) > (x - 2.4 * floor(x / 2.4) < 1.2 ? 2.0 : 3.0)) && "
                                "max(abs(y - 5.6), abs(z - 5.6)) < 4.6 ? 1 : -1");
        c.species = {hot("e", -1, 1, 0.3, win ? 0.0 : 0.2), hot("i", 1, 1836, 0.3, 0.0)};
        if (win) c.species[0].u0[1] = 0.1;          // a cold drift along y into the walls
        for (auto& s : c.species) { s.rmin[1] = 1.5; s.rmax[1] = 9.7; }
        c.check(); c.check_structure();
        Simulation sim(c, comm);
        sim.init();
        double g = 0, b = 0, lost = 0;
        while (!sim.finished()) {
            sim.step();
            g = std::max(g, sim.gauss_drift());
            b = std::max(b, comm.allreduce_max(sim.fields->max_div_b()));
        }
        for (auto& s : sim.species) lost += s.lost_energy;
        comm.allreduce_sum(&lost, 1);
        double w = sim.fields->energy();
        comm.allreduce_sum(&w, 1);
        check(std::string("metal S4: Gauss law and div B, plasma in a corrugated tube") + (win ? ", window, CPML" : pml ? ", CPML" : ""),
              g < 1e-12 && b < 1e-12 && lost > 0 && std::isfinite(w) && (!win || sim.dom.wx > 20),
              fmt("max|dGauss| %.1e, max|divB| %.1e", g, b) + fmt(", absorbed W_kin %.3g, shifts %.0f", lost, double(sim.dom.wx)));
    }
}

// S5. Cold electrons through a FODO of four hard-edged quadrupoles (edges between the grid points) against thick-lens
//     transfer matrices with K = G / u_x (paraxial): positions and angles (x 1000) at the exit below 1e-4 of the amplitude
//     and falling with h (not strictly second order: the fractions of the edge steps differ between the resolutions).
struct Pt { double y, yp; };
Pt matrix(Pt p, double K, double L) {
    if (K > 0) { const double w = std::sqrt(K); return {p.y * std::cos(w * L) + p.yp * std::sin(w * L) / w, -p.y * w * std::sin(w * L) + p.yp * std::cos(w * L)}; }
    if (K < 0) { const double w = std::sqrt(-K); return {p.y * std::cosh(w * L) + p.yp * std::sinh(w * L) / w, p.y * w * std::sinh(w * L) + p.yp * std::cosh(w * L)}; }
    return {p.y + p.yp * L, p.yp};
}
void fodo(Comm& comm) {
    const double ux = 2000, ell = 1000, G = 5.78e-3, x0 = 50.7, gap = 137.3, yc = 8, zc = 8;
    std::vector<Quad> Q;
    for (int q = 0; q < 4; ++q) {
        const double a = x0 + q * (ell + gap);
        Q.push_back(Quad{a, a + ell, q % 2 ? -G : G, yc, zc});
    }
    const double xend = x0 + 4 * ell + 3 * gap + 20;
    const double y0[3] = {0.15, -0.08, 0.03}, yp0[3] = {0.0, 1e-4, -2e-4};   // paraxial: angles <~ 2e-3 at the exit
    double err[2] = {0, 0}, scale = 0;
    for (int r = 0; r < 2; ++r) {
        const double h = r ? 2.0 : 4.0;
        Domain d = box3(comm, int(xend / h) + 20, 8, 8, h, 2, 2, false);
        Fields f(d);
        Species sp;
        sp.q = -1; sp.m = 1;
        std::vector<double> hv[Species::NC];
        for (int p = 0; p < 3; ++p) {
            const double uy = yp0[p] * ux, uz = -yp0[p] * ux;
            const double v[Species::NC] = {2.0, yc + y0[p], zc - y0[p], ux, uy, uz, 1e-30};
            for (int c = 0; c < Species::NC; ++c) hv[c].push_back(v[c]);
        }
        sp.append(hv);
        set_quadrupoles(Q);
        const int steps = int((xend - 2.0) / h);
        for (int n = 0; n < steps; ++n) push_only(sp, f, d, Pusher::Boris, 1);
        set_quadrupoles({});
        const std::vector<double> X = sp.get(Species::X), Y = sp.get(Species::Y), Z = sp.get(Species::Z), UX = sp.get(Species::UX),
                                  UY = sp.get(Species::UY), UZ = sp.get(Species::UZ);
        for (int p = 0; p < 3; ++p) {
            Pt a{y0[p], yp0[p]}, b{-y0[p], -yp0[p]};
            a = matrix(a, 0, x0 - 2.0); b = matrix(b, 0, x0 - 2.0);
            for (int q = 0; q < 4; ++q) {
                const double K = (q % 2 ? -G : G) / ux;
                a = matrix(a, K, ell); b = matrix(b, -K, ell);
                if (q < 3) { a = matrix(a, 0, gap); b = matrix(b, 0, gap); }
            }
            const double py = Y[p] - yc + (xend - X[p]) * UY[p] / UX[p], ppy = UY[p] / UX[p];
            const double pz = Z[p] - zc + (xend - X[p]) * UZ[p] / UX[p], ppz = UZ[p] / UX[p];
            a = matrix(a, 0, 20); b = matrix(b, 0, 20);
            err[r] = std::max({err[r], std::abs(py - a.y), std::abs(pz - b.y), 1000 * std::abs(ppy - a.yp), 1000 * std::abs(ppz - b.yp)});
            scale = std::max({scale, std::abs(a.y), std::abs(b.y)});
        }
    }
    check("quadrupole S5: FODO tracking = thick-lens matrices (edges between grid points)", err[0] / scale < 1e-4 && err[1] < err[0],
          fmt("rel. error %.2e (h = 4), ", err[0] / scale) + fmt("%.2e (h = 2)", err[1] / scale));
}

#ifdef RIP_HDF5
// S6. A beam written here as openPMD (unitSI 1e-6 positions, a constant positionOffset, macroWeighted momenta, constant
//     charge / mass records) read by a species with file = ...: the loaded particles have the moments of the written ones
//     (computed here in SI), with the centroid at x_center; resampling keeps the charge and the moments within noise.
void beam_file(Comm& comm) {
    const double QE = 1.602176634e-19, ME = 9.1093837015e-31, CL = 299792458.0;
    const std::string path = "test_structure_beam.h5";
    const size_t n = 4000;
    std::mt19937_64 rng(5);
    std::normal_distribution<double> nd(0, 1);
    std::uniform_real_distribution<double> ud(0.5, 1.5);
    std::vector<double> px(n), py(n), pz(n), ux(n), uy(n), uz(n), w(n);
    for (size_t k = 0; k < n; ++k) {
        px[k] = 8 * nd(rng) + 3; py[k] = 6 * nd(rng) - 2; pz[k] = 100 * nd(rng);      // um
        uz[k] = 2000 * (1 + 0.01 * pz[k] / 100 + 1e-3 * nd(rng));
        ux[k] = 0.5 * nd(rng) + 0.002 * px[k]; uy[k] = 0.4 * nd(rng);
        w[k] = 1e6 * ud(rng);
    }
    if (comm.rank == 0) {
        h5::Id f = h5::create_file(path);
        h5::attr(f, "openPMD", "1.1.0"); h5::attr(f, "basePath", "/data/%T/"); h5::attr(f, "particlesPath", "particles/");
        h5::attr(f, "meshesPath", "meshes/");
        const std::string g = "/data/7/particles/beam/";
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
        cst(g + "positionOffset/x", 0, 1); cst(g + "positionOffset/y", 0, 1); cst(g + "positionOffset/z", 1e-3, 1);
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
    double sx2 = 0, sux2 = 0, sxux = 0, sz2 = 0;
    for (size_t k = 0; k < n; ++k) {
        sx2 += w[k] * (px[k] - m1[0]) * (px[k] - m1[0]); sux2 += w[k] * (ux[k] - m1[3]) * (ux[k] - m1[3]);
        sxux += w[k] * (px[k] - m1[0]) * (ux[k] - m1[3]); sz2 += w[k] * (pz[k] - m1[2]) * (pz[k] - m1[2]);
    }
    const double Qref = W * QE, epsx = std::sqrt(sx2 / W * sux2 / W - sxux * sxux / (W * W)), sigz = std::sqrt(sz2 / W);   // um
    double worst = 0, rs = 0;
    for (int resample = 0; resample < 2; ++resample) {
        Config c;
        c.lambda0 = 1e-4;                           // k0 = 2 pi / 100 um
        const double k0 = 2 * M_PI / c.lambda0;
        c.N[0] = 1000; c.N[1] = 60; c.N[2] = 60;
        c.h[0] = 0.05; c.h[1] = 0.1; c.h[2] = 0.1;
        c.periodic[0] = false;
        SpeciesConfig s;
        s.name = "beam"; s.q = -1; s.m = 1; s.density = 0;
        for (int a = 0; a < 3; ++a) { s.rmin[a] = -1e300; s.rmax[a] = 1e300; }
        s.file.path = path;
        s.file.x_center = 25.0; s.file.y_center = 3.0; s.file.z_center = 3.0;
        if (resample) s.file.macroparticles = 40000;
        c.species = {s};
        c.check();
        Simulation sim(c, comm);
        sim.init();
        const Species& sp = sim.species[0];
        const std::vector<double> X = sp.get(Species::X), Y = sp.get(Species::Y), UY = sp.get(Species::UY), UX = sp.get(Species::UX),
                                  UZ = sp.get(Species::UZ), Wt = sp.get(Species::W);
        double v[8] = {0, 0, 0, 0, 0, 0, 0, double(X.size())};   // W, W x, W y, W uy, W gamma, -, -, count
        for (size_t p = 0; p < X.size(); ++p) {
            v[0] += Wt[p]; v[1] += Wt[p] * X[p]; v[2] += Wt[p] * Y[p]; v[3] += Wt[p] * UY[p];
            v[4] += Wt[p] * std::sqrt(1 + UX[p] * UX[p] + UY[p] * UY[p] + UZ[p] * UZ[p]);
        }
        comm.allreduce_sum(v, 8);
        const double xm = v[1] / v[0], ym = v[2] / v[0], uym = v[3] / v[0];
        double u[4] = {0, 0, 0, 0};                 // W dy^2, W duy^2, W dy duy, W dx^2
        for (size_t p = 0; p < X.size(); ++p) {
            const double dy = Y[p] - ym, du = UY[p] - uym, dx = X[p] - xm;
            u[0] += Wt[p] * dy * dy; u[1] += Wt[p] * du * du; u[2] += Wt[p] * dy * du; u[3] += Wt[p] * dx * dx;
        }
        comm.allreduce_sum(u, 4);
        const double w0 = CL * k0, nc = 8.8541878128e-12 * ME * w0 * w0 / (QE * QE);
        const double Q = v[0] * nc / (k0 * k0 * k0) * QE;
        const double eps = std::sqrt(u[0] / v[0] * u[1] / v[0] - u[2] * u[2] / (v[0] * v[0])) / k0 * 1e6;
        const double sig = std::sqrt(u[3] / v[0]) / k0 * 1e6, ycen = (ym - 3.0) / k0 * 1e6;
        const double e[6] = {Q / Qref - 1, (xm - 25.0) / 25.0, ycen / m1[0] - 1, eps / epsx - 1, sig / sigz - 1, v[4] / v[0] / g1 - 1};
        double mx = 0;
        if (!resample) {
            for (double x : e) mx = std::max(mx, std::abs(x));
            if (v[7] != double(n)) mx = 1;
        } else mx = std::max({std::abs(e[0]) * 1e8, std::abs(e[1]), std::abs(e[2]), std::abs(e[3]), std::abs(e[4]), std::abs(e[5])});
        (resample ? rs : worst) = mx;
    }
    if (comm.rank == 0) std::remove(path.c_str());
    check("beam file S6: openPMD import = moments of the written beam (and resampled)", worst < 1e-10 && rs < 3e-2,
          fmt("max rel. error %.1e, resampled %.1e", worst, rs));
}
#endif

// S7. A TE field (Bx blob, uniform in x) radiating into CPML layers in y and z: the field energy inside the side_flux box
//     plus the energy that left through its faces stays the initial energy (Poynting balance, second order in h).
void side_flux(Comm& comm) {
    Config c;
    c.N[0] = 4; c.N[1] = 80; c.N[2] = 80;
    c.h[0] = 0.25; c.h[1] = 0.5; c.h[2] = 0.5;
    for (int a = 1; a < 3; ++a) { c.pml.on[a] = true; c.periodic[a] = false; }
    c.pml.cells = 12;
    c.t_end = 40;
    c.dir = "test_structure_out";
    c.check();
    Simulation sim(c, comm);
    sim.init();
    H B(sim.fields->bx);
    const double cy = 0.5 * c.N[1] * c.h[1], s = 2.0;
    for (int k = 0; k < sim.dom.n[2]; ++k)
        for (int j = 0; j < sim.dom.n[1]; ++j)
            for (int i = 0; i < sim.dom.n[0]; ++i) {
                const double y = (sim.dom.off[1] + j + 0.5) * c.h[1] - cy, z = (sim.dom.off[2] + k + 0.5) * c.h[2] - cy;
                B(i, j, k) = std::exp(-(y * y + z * z) / (2 * s * s));
            }
    B.put(sim.fields->bx);
    comm.exchange(sim.dom, six(*sim.fields));
    SideFluxConfig sf; sf.on = true; sf.every = 1;
    std::unique_ptr<Operator> op = make_side_flux_operator(sf);
    op->setup(sim);
    double W0 = sim.fields->energy();
    comm.allreduce_sum(&W0, 1);
    while (!sim.finished()) { sim.step(); op->apply(sim); }
    op.reset();                                     // closes the file
    double v[11] = {0};
    if (comm.rank == 0) {
        std::ifstream in(c.dir + "/side_flux.txt");
        std::string line, last;
        while (std::getline(in, line)) if (!line.empty() && line[0] != '#') last = line;
        std::istringstream is(last);
        for (double& x : v) is >> x;
    }
    comm.allreduce_sum(v, 11);
    const double k0 = 2 * M_PI / c.lambda0, E0 = 9.1093837015e-31 * 299792458.0 * 299792458.0 * k0 / 1.602176634e-19;
    const double wu = 8.8541878128e-12 * E0 * E0 / (k0 * k0 * k0);
    const double out = v[6] + v[7] + v[8] + v[9], Wi = v[10];
    const double bal = (Wi + out) / (W0 * wu) - 1;
    check("side_flux S7: energy inside + energy out through the faces = initial energy", std::abs(bal) < 5e-3 && out / (W0 * wu) > 0.9,
          fmt("balance %.2e, fraction out %.3f", bal, out / (W0 * wu)));
}

// S8. The wake operator against a backward plane wave Ey = -Bz = g(x + t) in a window moving with c: F_y = 2 g(xi + 2t),
//     and the accumulated potential is the sum over the steps of 2 g(xi + 2 t_n) dt (computed here), -> int g du.
void wake_probe(Comm& comm) {
    Config c;
    c.N[0] = 200; c.N[1] = 4; c.N[2] = 4;
    c.h[0] = 0.5; c.h[1] = 0.8; c.h[2] = 0.8;
    c.periodic[0] = false;
    c.window.enabled = true; c.window.start = 0;
    c.t_end = 60;
    c.dir = "test_structure_out";
    WakeConfig w; w.name = "s8"; w.every = 1000000;
    c.wakes = {w};
    c.check();
    Simulation sim(c, comm);
    sim.init();
    const double xa = 70, s = 3;
    auto g = [&](double u) { return std::exp(-(u - xa) * (u - xa) / (s * s)); };
    H Ey(sim.fields->ey), Bz(sim.fields->bz);
    for (int k = 0; k < sim.dom.n[2]; ++k)
        for (int j = 0; j < sim.dom.n[1]; ++j)
            for (int i = 0; i < sim.dom.n[0]; ++i) {
                const double x = (sim.dom.off[0] + i) * c.h[0];
                Ey(i, j, k) = g(x); Bz(i, j, k) = -g(x);
            }
    Ey.put(sim.fields->ey); Bz.put(sim.fields->bz);
    comm.exchange(sim.dom, six(*sim.fields));
    long n = 0;
    while (!sim.finished()) { sim.step(); ++n; }
    sim.pipeline.finish(sim);
    double e = 0, peak = 0;
    if (comm.rank == 0) {
        std::ifstream in(c.dir + "/wake_s8.txt");
        std::string line;
        std::vector<std::string> block;
        while (std::getline(in, line)) {
            if (line.rfind("# step", 0) == 0) block.clear();
            else if (!line.empty() && line[0] != '#') block.push_back(line);
        }
        const double k0 = 2 * M_PI / c.lambda0, E0 = 9.1093837015e-31 * 299792458.0 * 299792458.0 * k0 / 1.602176634e-19;
        for (size_t q = 0; q < block.size(); ++q) {
            std::istringstream is(block[q]);
            double v[7];
            for (double& x : v) is >> x;
            const double xi = v[0] * 1e-6 * k0;   // code units
            double ref = 0;
            for (long m = 1; m <= n; ++m) ref += 2 * g(xi + 2 * m * c.h[0]) * c.h[0];
            e = std::max(e, std::abs(v[5] / (E0 / k0) - ref));
            peak = std::max(peak, ref);
        }
    }
    comm.allreduce_sum(&e, 1); comm.allreduce_sum(&peak, 1);
    const double integral = std::sqrt(M_PI) * s;
    check("wake S8: probe potential = sum of 2 g(xi + 2 t) dt, backward plane wave", e < 1e-6 * peak && std::abs(peak / integral - 1) < 1e-4,
          fmt("max error %.1e, peak %.8f", e / peak, peak) + fmt(" vs int g = %.8f", integral));
}
}  // namespace

void test_structure(Comm& comm) {
    rect_pipe(comm);
    round_pipe(comm);
    x_face(comm);
    gauss_plasma(comm);
    fodo(comm);
#ifdef RIP_HDF5
    beam_file(comm);
#endif
    side_flux(comm);
    wake_probe(comm);
}
