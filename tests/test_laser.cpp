// Laser modes (docs/specs/laser_modes.md): focus away from the entrance plane, Laguerre-Gauss modes,
// several lasers with a phase (circular polarisation).
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include "common.hpp"
#include "sim.hpp"

// L1. A beam focused in y only (uniform in z), W = 6 (about one wavelength), focus at x_f = z_R = 18:
//     the width from the time-integrated fluence F(y) at the entrance and at the focus, w = 2 sqrt(<y^2>),
//     against w(x) = W sqrt(1 + (x - x_f)^2 / z_R^2), and the ratio of the peak fluences w(0) / w(x_f)
//     (|E|^2 ~ W / w when focused along one axis).
static void laser_focus(Comm& comm) {
    if (comm.size > 1) return;
    Config c;
    const double dx = 2 * M_PI / 16;
    c.N[0] = 56; c.N[1] = 64; c.N[2] = 3;
    c.h[0] = dx; c.h[1] = 1.5 * dx; c.h[2] = 10 * dx;
    c.periodic[0] = false;
    LaserConfig& L = c.laser;
    L.enabled = true; L.a0 = 0.01; L.tau = 12; L.tc = 24; L.w0 = 6; L.w0z = 0; L.pol = 'z';
    L.yc = 0.5 * c.N[1] * c.h[1]; L.zc = 0.5 * c.N[2] * c.h[2];
    const double zr = 0.5 * L.w0 * L.w0;
    L.xf = zr;
    c.t_end = L.tc + 3 * L.tau + c.N[0] * dx;
    c.check();
    Simulation s(c, comm);
    s.init();
    const int i0 = 0, i1 = int(std::lround(L.xf / dx));
    std::vector<double> F0(size_t(c.N[1]), 0.0), F1(size_t(c.N[1]), 0.0);
    while (!s.finished()) {
        s.step();
        H ez(s.fields->ez);
        for (int j = 0; j < c.N[1]; ++j) {
            F0[size_t(j)] += ez(i0, j, 1) * ez(i0, j, 1);
            F1[size_t(j)] += ez(i1, j, 1) * ez(i1, j, 1);
        }
    }
    auto width = [&](const std::vector<double>& F, double& peak) {
        double m0 = 0, m2 = 0;
        peak = 0;
        for (int j = 0; j < c.N[1]; ++j) {
            const double y = j * c.h[1] - L.yc;
            m0 += F[size_t(j)]; m2 += F[size_t(j)] * y * y; peak = std::max(peak, F[size_t(j)]);
        }
        return 2 * std::sqrt(m2 / m0);
    };
    double p0, p1;
    const double w0m = width(F0, p0), w1m = width(F1, p1);
    const double w0a = L.w0 * std::sqrt(1 + L.xf * L.xf / (zr * zr)), w1a = L.w0 * std::sqrt(1 + std::pow((i1 * dx - L.xf) / zr, 2));
    const double e0 = std::abs(w0m / w0a - 1), e1 = std::abs(w1m / w1a - 1), er = std::abs(p1 / p0 / (w0a / w1a) - 1);   // |E|^2 ~ W/w (1D)
    check("laser: focus at x_f = z_R, width at the entrance and the focus, fluence gain", e0 < 0.03 && e1 < 0.03 && er < 0.05,
          fmt("w(0) %.3f", w0m) + fmt(" (paraxial %.3f)", w0a) + fmt(", w(x_f) %.3f", w1m) + fmt(" (%.3f)", w1a) +
              fmt(", F(x_f)/F(0) %.3f", p1 / p0) + fmt(" (%.3f)", w0a / w1a));
}

// L2. LaserConfig::field of Laguerre-Gauss modes (p, l) = (0, 0), (1, 1), (2, -1) away from the focus
//     against the q-parameter form: E = Re[ a0 (W/w) (sqrt(2) r / w)^|l| L_p^|l|(2 r^2/w^2)
//     exp(-i r^2 / (2 q)) ... ] written with complex numbers (1/q = 1/R - 2i/w^2), the Gouy phase as the
//     argument of q, and L_p^|l| from its explicit polynomials.
static void laser_lg(Comm&) {
    std::mt19937_64 rng(4);
    std::uniform_real_distribution<double> u(-1, 1);
    double worst = 0;
    const int modes[3][2] = {{0, 0}, {1, 1}, {2, -1}};
    for (auto& md : modes) {
        LaserConfig L;
        L.enabled = true; L.a0 = 1.3; L.tau = 9; L.tc = 30; L.w0 = 5; L.w0z = 5; L.yc = 1; L.zc = -2;
        L.xf = 17; L.p = md[0]; L.l = md[1]; L.phase = 0.4;
        const double W = L.w0, zr = 0.5 * W * W, X = -L.xf;
        const std::complex<double> q(X, zr), I(0, 1);
        for (int n = 0; n < 200; ++n) {
            const double t = L.tc + 8 * u(rng), y = L.yc + 12 * u(rng), z = L.zc + 12 * u(rng);
            const double dy = y - L.yc, dz = z - L.zc, r2 = dy * dy + dz * dz;
            const double w = W * std::abs(q) / zr;                        // w = W sqrt(1 + X^2 / zR^2)
            const std::complex<double> iq = 1.0 / q;                     // 1/R - i 2/w^2 (k = 1)
            const double invR = iq.real(), al = std::abs(L.l), x = 2 * r2 / (w * w);
            const double lp = L.p == 0 ? 1 : (L.p == 1 ? 1 + al - x : 0.5 * (x * x - 2 * (al + 2) * x + (al + 1) * (al + 2)));
            // Gouy phase (2p + |l| + 1) atan(X / zR) = N (pi/2 - arg q), for the phase t - x - r^2/(2R) + psi
            const double gouy = (2 * L.p + al + 1) * (M_PI / 2 - std::arg(q));
            const double te = t - L.tc - 0.5 * r2 * invR;
            const std::complex<double> e = (W / w) * std::pow(std::sqrt(x), al) * lp * std::exp(-r2 / (w * w)) *
                std::exp(-1.3862943611198906 * te * te / (L.tau * L.tau)) *
                std::exp(I * (te + gouy + L.l * std::atan2(dz, dy) + L.phase));
            const double want = L.a0 * e.imag();                         // sin of the phase
            worst = std::max(worst, std::abs(L.field(t, y, z) - want));
        }
    }
    check("laser: Laguerre-Gauss (p, l) = (0,0), (1,1), (2,-1) away from the focus = q-parameter form", worst < 1e-12,
          fmt("max difference %.1e", worst));
}

// L3. Two lasers, polarised along y (phase 0) and z (phase pi/2), plane (uniform), the same envelope:
//     circular polarisation, so Ey^2 + Ez^2 in a cell shows no oscillation at twice the frequency at the
//     pulse peak (within the change of the envelope).
static void laser_circular(Comm& comm) {
    if (comm.size > 1) return;
    Config c;
    const double dx = 2 * M_PI / 16;
    c.N[0] = 32; c.N[1] = 3; c.N[2] = 3;
    c.h[0] = dx; c.h[1] = 10 * dx; c.h[2] = 10 * dx;
    c.periodic[0] = false;
    LaserConfig L;
    L.enabled = true; L.a0 = 0.01; L.tau = 200; L.tc = 400; L.w0 = 1e9; L.w0z = 0; L.pol = 'y';
    c.laser = L;
    L.pol = 'z'; L.phase = M_PI / 2;
    c.extra_lasers.push_back(L);
    const int i = 10;
    c.t_end = L.tc + i * dx + 2 * M_PI;                 // the peak reaches cell i at tc + x_i
    c.check();
    Simulation s(c, comm);
    s.init();
    double mn = 1e300, mx = 0, ref = 0;
    while (!s.finished()) {
        s.step();
        if (s.t < L.tc + i * dx - 2 * M_PI) continue;   // the last two periods around the peak
        H ey(s.fields->ey), ez(s.fields->ez);
        const double e2 = ey(i, 1, 1) * ey(i, 1, 1) + ez(i, 1, 1) * ez(i, 1, 1);
        mn = std::min(mn, e2); mx = std::max(mx, e2); ref = L.a0 * L.a0;
    }
    const double ripple = (mx - mn) / ref;
    // the envelope alone changes Ey^2 + Ez^2 by 4 ln2 (2 pi)^2 / tau^2 = 0.3 % over the window; linear
    // polarisation would swing by 100 % at 2 omega
    check("laser: two crossed lasers, phase pi/2: circular, |E_perp|^2 without 2 omega ripple", ripple < 1e-2 && mx > 0.95 * ref,
          fmt("(max - min) / a0^2 %.1e", ripple) + fmt(", max %.3f a0^2", mx / ref));
}

void test_laser(Comm& comm) {
    laser_focus(comm);
    laser_lg(comm);
    laser_circular(comm);
}
