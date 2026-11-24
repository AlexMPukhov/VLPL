// O7: linear-response media without particles in iRIP (docs/specs/linear_medium.md, decision 84; prototype
// research/linear_medium/).
#include "env.hpp"
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include "common.hpp"
#include "fields.hpp"
#include "medium.hpp"
#include "sim.hpp"

namespace {
using cd = std::complex<double>;
const UnitFn no_units = [](const std::string&, double&) { return false; };

Domain mdom(Comm& comm, int nx, int ny, int nz, double hx, double hy, double hz, bool flat, bool px = true) {
    Domain d;
    d.N[0] = nx; d.N[1] = ny; d.N[2] = flat ? 1 : nz;
    d.h[0] = hx; d.h[1] = hy; d.h[2] = flat ? 1.0 : hz;
    d.flat = flat;
    d.periodic[0] = px;
    d.ng = 3;
    d.solver = Solver::IRip;
    const int want[3] = {0, 1, 1};
    comm.decompose(d, want);
    return d;
}

double total(Comm& comm, const Domain& d, const Fields& F, const Medium& M, double* w = nullptr, double* k = nullptr) {
    double v[2] = {F.energy(), M.energy(d)};
    comm.allreduce_sum(v, 2);
    if (w) *w = v[0];
    if (k) *k = v[1];
    return v[0] + v[1] + M.dissipated;
}

// field energy of the owned cells with x in [x0, x1) (absolute code units)
double energy_x(Comm& comm, const Domain& d, const Fields& F, double x0, double x1) {
    double s = 0;
    for (const Field* f : {&F.ex, &F.ey, &F.ez, &F.bx, &F.by, &F.bz}) {
        const std::vector<double> h = f->host();
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const double x = d.lo(0) + i * d.h[0];
                    if (x >= x0 && x < x1) s += h[f->idx(i, j, k)] * h[f->idx(i, j, k)];
                }
    }
    s *= 0.5 * d.cell_volume();
    comm.allreduce_sum(&s, 1);
    return s;
}

// a slab x0 <= x < x1 (sub-cell volume fraction) of a Drude / Lorentz medium
MediumConfig mslab(double x0, double x1, double wp2, double nu = 0, double w0 = 0) {
    MediumConfig m;
    m.density = wp2; m.nu = nu; m.w0 = w0;
    char t[128];
    std::snprintf(t, sizeof t, "x >= %.17g && x < %.17g", x0, x1);
    m.profile = Expr(t, {"x", "y", "z"}, no_units);
    return m;
}
}  // namespace

// L1. The exact energy identity: random fields and random media (wp dt <= 30, collisions, Lorentz), h_y < c dt, periodic,
//     3D and 2D: W + K + dissipated constant to the tolerance of the linear solve.
static void medium_energy(Comm& comm) {
    struct Case { double wpdt, w0dt, nudt; };
    const Case cases[] = {{3, 0, 0}, {10, 0, 0}, {10, 0, 1}, {2, 4, 0.3}, {30, 0, 0}};
    for (int flat = 1; flat >= 0; --flat) {
        const Domain d = flat ? mdom(comm, 24, 20, 1, 1.0, 0.4, 1.0, true) : mdom(comm, 12, 10, 8, 1.0, 0.4, 0.5, false);
        for (const Case& cs : cases) {
            Fields F(d);
            Medium M;
            M.resize(d);
            M.tol = 1e-13;
            std::mt19937_64 rng(3 + flat);
            std::normal_distribution<double> nd;
            std::uniform_real_distribution<double> ud;
            for (Field* f : {&F.ex, &F.ey, &F.ez, &F.bx, &F.by, &F.bz}) {
                H h(*f);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i) h(i, j, k) = nd(rng);
                h.put(*f);
            }
            const double D = d.dt();
            for (int c = 0; c < 3; ++c) {
                H w(M.wp2[c]), o(M.w02[c]), n(M.nu[c]);
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i)
                            if (ud(rng) < 0.5) {
                                w(i, j, k) = cs.wpdt * cs.wpdt / (D * D) * ud(rng);
                                o(i, j, k) = cs.w0dt * cs.w0dt / (D * D);
                                n(i, j, k) = cs.nudt / D;
                            }
                w.put(M.wp2[c]); o.put(M.w02[c]); n.put(M.nu[c]);
            }
            comm.exchange(d, {&F.ex, &F.ey, &F.ez, &F.bx, &F.by, &F.bz});
            F.medium = &M;
            const double e0 = total(comm, d, F, M);
            double worst = 0;
            for (int n = 0; n < 30; ++n) {
                F.step(comm);
                worst = std::max(worst, std::abs(total(comm, d, F, M) / e0 - 1));
            }
            char lab[160];
            std::snprintf(lab, sizeof lab, "medium: W + K + dissipated, random fields and media, %s, wp dt <= %g, w0 dt = %g, nu dt = %g",
                          flat ? "2D" : "3D", cs.wpdt, cs.w0dt, cs.nudt);
            check(lab, worst < 1e-10 && M.stats.unconverged == 0,
                  fmt("max |E/E0 - 1| %.1e", worst) + fmt(", dissipated %.3f E0", M.dissipated / e0) +
                      fmt(", GMRES %.1f iterations per step", double(M.stats.iterations) / M.stats.solves));
        }
    }
}

// L2. Normal incidence on a lossless overdense Drude slab (the transport along x is exact at c dt = dx): |r| = 1 for any
//     omega_p dt, here 0.5, 3 and 20; the pulse Ey = Bz moves along +x and comes back whole.
static void medium_mirror_1d(Comm& comm) {
    const double lam = 2 * M_PI, dx = lam / 20, w = 2 * lam;
    const double x0 = 3.5 * w, xs = 7 * w, L = xs + 3 * lam + 2 * w;
    const int nx = int(std::round(L / dx));
    const Domain d = mdom(comm, nx, 4, 1, dx, dx, 1.0, true);
    for (double wpdt : {0.5, 3.0, 20.0}) {
        Fields F(d);
        Medium M;
        const double wp = wpdt / dx;
        medium_build(M, d, {mslab(xs, xs + 3 * lam, wp * wp)}, 1);
        M.tol = 1e-12;
        {
            H ey(F.ey), bz(F.bz);
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const double x = d.lo(0) + i * dx, g = std::exp(-std::pow((x - x0) / w, 2)) * std::cos(x - x0);
                    ey(i, j, 0) = g; bz(i, j, 0) = g;
                }
            ey.put(F.ey); bz.put(F.bz);
        }
        comm.exchange(d, {&F.ey, &F.bz});
        F.medium = &M;
        const double w0 = total(comm, d, F, M);
        const int steps = int(std::round(2 * (xs - x0) / dx));
        for (int n = 0; n < steps; ++n) F.step(comm);
        const double R = energy_x(comm, d, F, 0, xs) / w0;
        double wf, k;
        const double e1 = total(comm, d, F, M, &wf, &k);
        check("medium: |r| = 1 at normal incidence on a lossless overdense slab, omega_p dt = " + fmt("%g", wpdt),
              std::abs(R - 1) < 1e-8 && std::abs(e1 / w0 - 1) < 1e-10,
              fmt("|R - 1| %.1e", std::abs(R - 1)) + fmt(", |dE|/E %.1e", std::abs(e1 / w0 - 1)) +
                  fmt(", GMRES %.1f per step", double(M.stats.iterations) / M.stats.solves));
    }
}

namespace {
cd fresnel_s(cd eps, double cth) {
    const double s2 = 1 - cth * cth;
    cd q = std::sqrt(eps - s2);
    if (q.imag() < 0) q = -q;
    return (cth - q) / (cth + q);
}
struct ObliqueResult { double R, A, kxt, dE; double its; };
// An s-polarised pulse (Ez, Bx, By; envelope exp(-((x - x0)/w)^2) along x, a plane wave along y with ky = k0 sin(theta), periodic
// in y over one wavelength of ky) hits the half space x >= xs (open x, the far end never reached in time). After the reflected
// pulse is back at x0: R = W(x < xs) / W0, A = dissipated / W0 and, from the field in the medium, the mean kx of the transmitted
// pulse (the phase gradient of its ky Fourier coefficient weighted by its power).
ObliqueResult oblique(Comm& comm, int N, double th, const MediumConfig& med, int ydiv = 1) {
    const double lam = 2 * M_PI, dx = lam / N, w = 2 * lam, x0 = 3.5 * w, xs = x0 + 3.5 * w;
    const double ky = std::sin(th), kx = std::cos(th), Ly = lam / ky;
    const int ny = ydiv * int(std::round(Ly / dx));
    const double dy = Ly / ny, L = xs + 10 * lam;
    const Domain d = mdom(comm, int(std::round(L / dx)), ny, 1, dx, dy, 1.0, true, false);
    Fields F(d);
    Medium M;
    MediumConfig m = med;
    char t[96];
    std::snprintf(t, sizeof t, "x >= %.17g", xs);
    m.profile = Expr(t, {"x", "y", "z"}, no_units);
    medium_build(M, d, {m}, 1);
    M.tol = 1e-10;
    {
        // a forward wave per Fourier component along x: Ez = Re sum_q a_q e^{i (k_q x + ky y)}, By = -(k_q / k) Ez_q,
        // Bx = (ky / k) Ez_q (k = |(k_q, ky)|); with one angle for all components (By = -cos(theta) Ez) a pulse of 2 lambda starts
        // with ~8e-4 of its energy moving backwards, whatever the resolution
        const int N0 = d.N[0];
        std::vector<cd> a(static_cast<size_t>(N0));
        for (int i = 0; i < N0; ++i) {
            const double x = i * dx;
            a[size_t(i)] = std::exp(-std::pow((x - x0) / w, 2)) * std::exp(cd(0, kx * (x - x0))) / double(N0);
        }
        std::vector<cd> A(static_cast<size_t>(N0));
        for (int q = 0; q < N0; ++q)                                         // the DFT (positive k_q only below N0 / 2)
            for (int i = 0; i < N0; ++i) A[size_t(q)] += a[size_t(i)] * std::exp(cd(0, -2 * M_PI * double(q) * i / N0));
        H ez(F.ez), bx(F.bx), by(F.by);
        for (int i = 0; i < d.n[0]; ++i) {
            cd e = 0, b = 0, c = 0;
            const int ig = i + d.off[0];
            for (int q = 0; q < N0 / 2; ++q) {
                const double kq = 2 * M_PI * q / (N0 * dx), k = std::hypot(kq, ky);
                const cd v = A[size_t(q)] * std::exp(cd(0, kq * ig * dx));
                e += v; b += -kq / k * v; c += ky / k * v;
            }
            for (int j = 0; j < d.n[1]; ++j) {
                const cd py = std::exp(cd(0, ky * j * dy)), ph = std::exp(cd(0, ky * (j + 0.5) * dy));   // Ez, By at (i, j); Bx at j + 1/2
                ez(i, j, 0) = (e * py).real(); by(i, j, 0) = (b * py).real(); bx(i, j, 0) = (c * ph).real();
            }
        }
        ez.put(F.ez); bx.put(F.bx); by.put(F.by);
    }
    comm.exchange(d, {&F.ez, &F.bx, &F.by});
    F.medium = &M;
    const double w0 = total(comm, d, F, M);
    const int steps = int(std::round(2 * (xs - x0) / kx / dx));
    for (int n = 0; n < steps; ++n) F.step(comm);
    ObliqueResult r;
    r.R = energy_x(comm, d, F, 0, xs) / w0;
    r.A = M.dissipated / w0;
    r.dE = std::abs(total(comm, d, F, M) / w0 - 1);
    r.its = double(M.stats.iterations) / M.stats.solves;
    // the transmitted pulse: c(x) = sum_y Ez e^{-i ky y}, mean kx = sum Im(c* dc/dx) / sum |c|^2 over x > xs + lambda
    const std::vector<double> ez = F.ez.host();
    std::vector<cd> c(static_cast<size_t>(d.n[0]));
    for (int i = 0; i < d.n[0]; ++i)
        for (int j = 0; j < d.n[1]; ++j) c[size_t(i)] += ez[F.ez.idx(i, j, 0)] * std::exp(cd(0, -ky * j * dy));
    double num = 0, den = 0;
    for (int i = 0; i + 1 < d.n[0]; ++i) {
        const double x = d.lo(0) + (i + 0.5) * dx;
        if (x < xs + lam) continue;
        const cd a = c[size_t(i)], b = c[size_t(i + 1)], mid = 0.5 * (a + b);
        num += std::arg(b / a) / dx * std::norm(mid);
        den += std::norm(mid);
    }
    double v[2] = {num, den};
    comm.allreduce_sum(v, 2);
    r.kxt = v[1] > 0 ? v[0] / v[1] : 0;
    return r;
}
// The references over the spectrum of the pulse (|A(kx)|^2 ~ exp(-(kx - kx0)^2 w^2 / 2), ky fixed): R, and the power-weighted
// mean kx of the transmitted field, sqrt(eps(w) w^2 - ky^2) with the weight |t A|^2 dkx/dkx_t (the density in kx_t).
void oblique_ref(double th, std::complex<double> (*eps)(double), double& R, double& kxt) {
    const double w = 2 * 2 * M_PI, ky = std::sin(th), kx0 = std::cos(th);
    double sr = 0, sw = 0, sk = 0, skw = 0;
    for (int q = -4000; q <= 4000; ++q) {
        const double kx = kx0 + q * 1e-3 * 8 / w;
        if (kx <= 0) continue;
        const double a2 = std::exp(-0.5 * std::pow((kx - kx0) * w, 2));
        const double om = std::hypot(kx, ky), cth = kx / om;
        const cd e = eps(om), r = fresnel_s(e, cth);
        sr += a2 * std::norm(r); sw += a2;
        const cd kt = std::sqrt(e * om * om - ky * ky);
        if (std::abs(e.imag()) < 1e-14 && kt.real() > 0) {
            const double t2 = std::norm(1.0 + r), dk = kt.real() / (kx * e.real() + 0.0);   // dkx_t/dkx = eps kx / kx_t
            const double wt = a2 * t2 / (1 / dk);
            sk += wt * kt.real(); skw += wt;
        }
    }
    R = sr / sw;
    kxt = skw > 0 ? sk / skw : 0;
}
std::complex<double> eps_glass(double w) { return 1.0 + 10.0 / (9.0 - w * w); }              // Lorentz: wp2 10, w0 3: eps(1) = 2.25
std::complex<double> eps_lossy(double w) { return 1.0 - 9.0 / (w * cd(w, 1.0)); }            // Drude 9 n_c, nu = omega0
}  // namespace

// L3. Oblique-incidence Fresnel reflection (s-pol, grid-aligned surface, 30 degrees): Lorentz glass (eps(omega0) = 2.25) and a
//     collisional 9 n_c plasma (nu = omega0) against the spectrally weighted Fresnel formula at lambda/dx = 10, 20, 40: second
//     order. L4. Snell: the transmitted kx in the glass against sqrt(eps w^2 - ky^2), i.e. the refraction angle atan(ky / kx).
static void medium_fresnel(Comm& comm) {
    if (vlpl_getenv("VLPL_MEDIUM_DIAG")) {               // the floor of the Fresnel error: vacuum (initial backward content), dy / 2
        MediumConfig g; g.density = 10; g.w0 = 3;
        MediumConfig v; v.density = 1e-300;
        for (int N : {10, 20, 40}) {
            const ObliqueResult a = oblique(comm, N, M_PI / 6, v), b = oblique(comm, N, M_PI / 6, g, 2);
            if (comm.rank == 0) std::printf("diag l/%d: R_vacuum %.3e, glass dy = dx/2: R %.6f, kx_t %.6f\n", N, a.R, b.R, b.kxt);
        }
        return;
    }
    const double th = M_PI / 6;
    const int Ns[3] = {10, 20, 40};
    const int nN = vlpl_getenv("VLPL_MEDIUM_FAST") ? 2 : 3;     // VLPL_MEDIUM_FAST: lambda/10, 20 only (slow FP64 GPUs), bounds at lambda/20
    {
        double Rr, kr;
        oblique_ref(th, eps_glass, Rr, kr);
        const double thr = std::atan(std::sin(th) / kr) * 180 / M_PI;
        MediumConfig g; g.density = 10; g.w0 = 3;
        double eR[3], eT[3];
        std::string info;
        for (int q = 0; q < nN; ++q) {
            const ObliqueResult r = oblique(comm, Ns[q], th, g);
            const double tht = std::atan(std::sin(th) / r.kxt) * 180 / M_PI;
            eR[q] = std::abs(r.R - Rr); eT[q] = std::abs(tht - thr);
            char b[200];
            std::snprintf(b, sizeof b, "%sl/%d: R %.5f, theta_t %.3f deg, |dE|/E %.0e, GMRES %.1f", q ? "; " : "", Ns[q], r.R, tht, r.dE, r.its);
            info += b;
        }
        bool ok = true;
        for (int q = 1; q < nN; ++q) ok = ok && eR[q] < eR[q - 1] / 3 && eT[q] < eT[q - 1] / 3;
        check("medium: Fresnel R, glass, s-pol, 30 deg, second order (R_ref " + fmt("%.5f", Rr) + ")", ok && eR[nN - 1] < (nN > 2 ? 2e-3 : 5e-3),
              info + fmt("; errors of R %.1e", eR[0]) + fmt(" %.1e", eR[1]) + (nN > 2 ? fmt(" %.1e", eR[2]) : ""));
        check("medium: Snell refraction angle in the glass, second order (theta_t,ref " + fmt("%.3f deg)", thr), ok && eT[nN - 1] < (nN > 2 ? 0.05 : 0.1),
              fmt("errors %.2e", eT[0]) + fmt(" %.2e", eT[1]) + (nN > 2 ? fmt(" %.2e deg", eT[2]) : " deg"));
    }
    {
        double Rr, kr;
        oblique_ref(th, eps_lossy, Rr, kr);
        MediumConfig p; p.density = 9; p.nu = 1;
        double eA[3];
        std::string info;
        for (int q = 0; q < nN; ++q) {
            const ObliqueResult r = oblique(comm, Ns[q], th, p);
            eA[q] = std::abs(r.A - (1 - Rr));
            char b[200];
            std::snprintf(b, sizeof b, "%sl/%d: A %.5f, R %.5f, |dE|/E %.0e, GMRES %.1f", q ? "; " : "", Ns[q], r.A, r.R, r.dE, r.its);
            info += b;
        }
        bool ok = true;
        for (int q = 1; q < nN; ++q) ok = ok && eA[q] < eA[q - 1] / 3;
        check("medium: absorption of a collisional 9 n_c plasma, s-pol, 30 deg, second order (A_ref " + fmt("%.5f", 1 - Rr) + ")",
              ok && eA[nN - 1] < (nN > 2 ? 1e-2 : 3e-2), info + fmt("; errors %.1e", eA[0]) + fmt(" %.1e", eA[1]) + (nN > 2 ? fmt(" %.1e", eA[2]) : ""));
    }
}

void test_medium(Comm& comm) {
    medium_energy(comm);
    medium_mirror_1d(comm);
    medium_fresnel(comm);
}
