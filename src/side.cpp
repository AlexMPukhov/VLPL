#include "side.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "side_kernels.hpp"

using namespace side;

namespace {
using cd = std::complex<double>;
constexpr int NC = int(sizeof(SideTfsf::Comp) / sizeof(double));   // doubles per component
static_assert(NC == NCD, "side_kernels.hpp: the component layout");

// One step of the solver (src/fields.cpp; iRIP src/irip.cpp) for the Bloch mode of wave vector k: the stored values of
// cell (i, j, k) are in[c] e^{i k . (i D, j hy, k hz)}, c = Ex, Ey, Ez, Bx, By, Bz at t_n and the fluid current
// Jx, Jy, Jz at t_n - D/2 (background plasma wp2, SideTfsf::fluid_current; with wp2 = 0 it stays and drives nothing
// in a mode with lambda != 1). The predictor of RIP-CC is returned in exs, bxs.
// O5 (d.o5, iRIP): the closure multiplies the box means Ex*, Bx* by o5_factor (Cx g_perp of the mode).
void mode_step(const Domain& d, const double k[3], double wp2, const cd in[9], cd out[9], bool irip, cd* exs = nullptr,
               cd* bxs = nullptr) {
    const double D = d.h[0], hy = d.h[1], hz = d.h[2];
    const double cf = irip ? o5_factor(d, k) : 1.0;
    const cd I(0, 1), a = std::exp(I * (k[0] * D));
    const cd dyp = (std::exp(I * (k[1] * hy)) - 1.0) / hy, dym = (1.0 - std::exp(-I * (k[1] * hy))) / hy;
    const cd dzp = (std::exp(I * (k[2] * hz)) - 1.0) / hz, dzm = (1.0 - std::exp(-I * (k[2] * hz))) / hz;
    const cd m = 0.5 * (1.0 + a);
    const cd jx = in[6] + D * wp2 * in[0], jy = in[7] + D * wp2 * m * in[1], jz = in[8] + D * wp2 * m * in[2];
    out[6] = jx; out[7] = jy; out[8] = jz;
    auto pass = [&](cd Exs, cd Bxs, cd o[6]) {
        const cd psiy = dzm * m * Bxs - jy, psiz = -dym * m * Bxs - jz, phiy = -dzp * m * Exs, phiz = dyp * m * Exs;
        const cd tpy = (in[1] + in[5] + D * (psiy + phiz)) / a, tmy = a * (in[1] - in[5]) + D * (psiy - phiz);
        const cd tpz = a * (in[2] + in[4]) + D * (psiz + phiy), tmz = (in[2] - in[4] + D * (psiz - phiy)) / a;
        o[1] = 0.5 * (tpy + tmy); o[5] = 0.5 * (tpy - tmy);
        o[2] = 0.5 * (tpz + tmz); o[4] = 0.5 * (tpz - tmz);
        const cd bz = 0.5 * (in[5] + o[5]), by = 0.5 * (in[4] + o[4]), ez = 0.5 * (in[2] + o[2]), ey = 0.5 * (in[1] + o[1]);
        o[0] = in[0] + D * (dym * bz - dzm * by - jx);
        o[3] = in[3] - D * (dyp * ez - dzp * ey);
    };
    const cd pE = in[0] + 0.5 * D * (dym * in[5] - dzm * in[4] - jx), pB = in[3] - 0.5 * D * (dyp * in[2] - dzp * in[1]);
    if (exs) { *exs = pE; *bxs = pB; }
    if (!irip) { pass(pE, pB, out); return; }
    // iRIP: the step is affine in X = Ex^{n+1}, Y = Bx^{n+1} through the box means (Ex + X)/2, (Bx + Y)/2 (o5: closed)
    auto passc = [&](cd E, cd B, cd o[6]) { pass(cf * E, cf * B, o); };
    cd r0[6], rx[6], ry[6];
    passc(0.5 * in[0], 0.5 * in[3], r0);
    passc(0.5 * (in[0] + 1.0), 0.5 * in[3], rx);
    passc(0.5 * in[0], 0.5 * (in[3] + 1.0), ry);
    const cd a11 = 1.0 - (rx[0] - r0[0]), a12 = -(ry[0] - r0[0]), a21 = -(rx[3] - r0[3]), a22 = 1.0 - (ry[3] - r0[3]);
    const cd det = a11 * a22 - a12 * a21;
    const cd X = (a22 * r0[0] - a12 * r0[3]) / det, Y = (a11 * r0[3] - a21 * r0[0]) / det;
    passc(0.5 * (in[0] + X), 0.5 * (in[3] + Y), out);
}

// The null space of the R x 9 matrix A (R = 9, or 10 with a constraint row) by Gauss-Jordan with full pivoting: its
// basis in N (at most 2 vectors), the dimension returned (0 if it is not 1 or 2).
int nullsp(cd A[10][9], int R, cd N[2][9]) {
    int col[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    double scale = 0;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < 9; ++c) scale = std::max(scale, std::abs(A[r][c]));
    int rank = 0;
    for (; rank < std::min(R, 9); ++rank) {
        int pr = rank, pc = rank;
        double best = 0;
        for (int r = rank; r < R; ++r)
            for (int c = rank; c < 9; ++c)
                if (std::abs(A[r][c]) > best) { best = std::abs(A[r][c]); pr = r; pc = c; }
        if (best < 1e-9 * scale) break;
        for (int c = 0; c < 9; ++c) std::swap(A[rank][c], A[pr][c]);
        for (int r = 0; r < R; ++r) std::swap(A[r][rank], A[r][pc]);
        std::swap(col[rank], col[pc]);
        const cd p = A[rank][rank];
        for (int c = 0; c < 9; ++c) A[rank][c] /= p;
        for (int r = 0; r < R; ++r) {
            if (r == rank) continue;
            const cd f = A[r][rank];
            for (int c = 0; c < 9; ++c) A[r][c] -= f * A[rank][c];
        }
    }
    const int nul = 9 - rank;
    if (nul < 1 || nul > 2) return 0;
    for (int n = 0; n < nul; ++n) {
        cd x[9] = {};
        x[rank + n] = 1;                               // free variable (permuted column rank + n)
        for (int r = 0; r < rank; ++r) x[r] = -A[r][rank + n];
        for (int c = 0; c < 9; ++c) N[n][col[c]] = x[c];
    }
    return nul;
}

// The eigenvectors of the one-step matrix for lambda = e^{-i w D} at the wave vector k: cls -1 the (degenerate) pair,
// 0 the TE mode (Ex = 0), 1 the TM mode (Bx = 0) as the null space with that constraint row.
int modes(const Domain& d, const double k[3], double w, double wp2, bool irip, int cls, cd N[2][9]) {
    const cd lam = std::exp(cd(0, -w * d.h[0]));
    cd A[10][9] = {};
    for (int b = 0; b < 9; ++b) {
        cd in[9] = {}, out[9];
        in[b] = 1;
        mode_step(d, k, wp2, in, out, irip);
        for (int r = 0; r < 9; ++r) A[r][b] = out[r] - (r == b ? lam : 0.0);
    }
    if (cls >= 0) A[9][cls == 0 ? 0 : 3] = 1;
    return nullsp(A, cls >= 0 ? 10 : 9, N);
}
Geo geo(const Domain& d, const SideConfig& s, const Buf<double>& cp, const Buf<double>& tb, const Buf<double>& tx,
        const Buf<double>& ft) {
    Geo g;
    for (int a = 0; a < 3; ++a) { g.lo[a] = s.lo[a]; g.hi[a] = s.hi[a]; g.f[a] = s.faces[a]; }
    g.o1 = d.off[1]; g.o2 = d.off[2];
    g.nx = d.n[0]; g.ny = d.n[1]; g.nz = d.n[2]; g.S = 3 + g.ny + g.nz;
    g.D = d.h[0]; g.hy = d.h[1]; g.hz = d.h[2];
    g.cp = cp.data(); g.tb = tb.data(); g.tx = tx.data(); g.ft = ft.data();
    g.nc = int(cp.size() / NCD);
    return g;
}
struct Nine { FV v[9]; };                             // the field (and fluid current) views, by value into a kernel
}  // namespace


// The discrete frequency of wave vector k (code units, c = 1) in a plasma of wp2 (side.hpp): RIP-CC TE (cls 0, also
// vacuum) or TM (cls 1); iRIP both.
double SideTfsf::omega(const Domain& d, const double k[3], bool irip, double wp2, int cls) {
    const double D = d.h[0], q = 0.25 * D * D, sy = std::sin(0.5 * k[1] * d.h[1]), sz = std::sin(0.5 * k[2] * d.h[2]);
    const double lam = 4 * sy * sy / (d.h[1] * d.h[1]) + 4 * sz * sz / (d.h[2] * d.h[2]);
    if (irip) {                                        // o5: the transverse term times the closure factor (decision 76)
        const double t = std::tan(0.5 * k[0] * D), p = q * wp2;
        return 2 / D * std::atan(std::sqrt((t * t + q * lam * o5_factor(d, k) + p) / (1 - p)));
    }
    const double s = std::sin(0.5 * k[0] * D), c = std::cos(0.5 * k[0] * D);
    const double pl = cls == 1 ? wp2 * (1 - q * lam) : wp2;
    return 2 / D * std::asin(std::min(1.0, std::sqrt(s * s + q * c * c * (lam + pl))));
}

// One exact discrete vacuum plane wave of the solver of d (the TFSF x plane, src/tfsf_modes.cpp): the eigenvector of the
// one-step matrix for wave vector k (2D eigenspace, no fluid current) whose E at the physical points (Ey at j + 1/2, Ez
// at k + 1/2) is closest to the wanted complex vector E (least squares), with the phase reference k . (r - r0) - w (t -
// tc). mism: the largest relative mismatch so far.
SideTfsf::Comp plane_mode(const Domain& d, const double k[3], double w, const cd E[3], const double r0[3], double tc,
                          double& mism) {
    const bool irip = d.solver == Solver::IRip;
    cd U[2][9];
    if (modes(d, k, w, 0.0, irip, -1, U) != 2)
        throw std::runtime_error("plane_mode: no two-dimensional eigenspace of the discrete plane wave");
    const cd sh[3] = {1.0, std::exp(cd(0, -0.5 * k[1] * d.h[1])), std::exp(cd(0, -0.5 * k[2] * d.h[2]))};
    cd M11 = 0, M12 = 0, M22 = 0, r1 = 0, r2 = 0;
    for (int q = 0; q < 3; ++q) {
        const cd w1 = U[0][q] * sh[q], w2 = U[1][q] * sh[q], t = E[q];
        M11 += std::conj(w1) * w1; M12 += std::conj(w1) * w2; M22 += std::conj(w2) * w2;
        r1 += std::conj(w1) * t; r2 += std::conj(w2) * t;
    }
    const cd det = M11 * M22 - M12 * std::conj(M12);
    const cd al1 = (M22 * r1 - M12 * r2) / det, al2 = (M11 * r2 - std::conj(M12) * r1) / det;
    cd v[9];
    for (int q = 0; q < 9; ++q) v[q] = al1 * U[0][q] + al2 * U[1][q];
    double res = 0, ref = 0;
    for (int q = 0; q < 3; ++q) { res += std::norm(v[q] * sh[q] - E[q]); ref += std::norm(E[q]); }
    if (ref > 0) mism = std::max(mism, std::sqrt(res / ref));
    const cd ph = std::exp(cd(0, -(k[0] * r0[0] + k[1] * r0[1] + k[2] * r0[2]) + w * tc));
    for (cd& q : v) q *= ph;
    cd out[9], pE, pB;
    mode_step(d, k, 0.0, v, out, irip, &pE, &pB);
    SideTfsf::Comp p{};
    for (int a = 0; a < 3; ++a) p.k[a] = k[a];
    p.w = w;
    for (int q = 0; q < 6; ++q) { p.v[2 * q] = v[q].real(); p.v[2 * q + 1] = v[q].imag(); }
    p.pr[0] = pE.real(); p.pr[1] = pE.imag(); p.pr[2] = pB.real(); p.pr[3] = pB.imag();
    const cd mu = 0.5 * (1.0 + std::exp(cd(0, -w * d.h[0])));
    p.mu[0] = mu.real(); p.mu[1] = mu.imag();
    if (irip && d.o5) {                                // the closed mean as Ex*, Bx* (kind 3, TfsfModes)
        const cd e = o5_factor(d, k) * mu * v[0], b = o5_factor(d, k) * mu * v[3];
        p.pr[0] = e.real(); p.pr[1] = e.imag(); p.pr[2] = b.real(); p.pr[3] = b.imag();
    }
    return p;
}

double o5_factor(const Domain& d, const double k[3], bool cx) {
    if (!d.o5 || d.solver != Solver::IRip) return 1.0;
    const double D = d.h[0], hy = d.h[1], hz = d.h[2];
    const double sx = std::sin(0.5 * k[0] * D), sy = std::sin(0.5 * k[1] * hy), sz = std::sin(0.5 * k[2] * hz);
    const double ly = 4 * sy * sy / (hy * hy), lz = 4 * sz * sz / (hz * hz), lam = ly + lz;
    const double g = lam > 0 ? 1 + 2.0 / 3 * std::min(D * D * lam / 4, 1.0) + (hy * hy * ly * ly + hz * hz * lz * lz) / (12 * lam)
                             : 1.0;   // irip.cpp o5_gperp (with the cap of decision 77)
    return (cx ? 1 + 4.0 / 3 * sx * sx : 1.0) * g;
}

double plane_omega(const Domain& d, const double k[3]) { return SideTfsf::omega(d, k, d.solver == Solver::IRip); }

// The plane wave: a continuous wave (k = k0 n, w from the dispersion; in a plasma w = omega0, |k| from it) or a pulse, a
// Fourier sum of components of the envelope at r0 (open x: every component along n; periodic x: the common k_x on the
// lattice, k_perp along (n_y, n_z)). RIP-CC in a plasma: a TE and a TM wave per component.
std::vector<SideTfsf::Wave> SideTfsf::plane_waves(const Domain& d, const SideConfig& c, double t_span) {
    const bool irip = d.solver == Solver::IRip;
    const double wp2 = c.density;
    const double ca = std::cos(c.angle), sa = std::sin(c.angle), ce = std::cos(c.elevation), se = std::sin(c.elevation);
    double n[3] = {ce * ca, ce * sa, se};
    const bool px = d.periodic[0];
    if (px) {                                          // k_x of a box periodic along x: a multiple of 2 pi / Lx (snapped)
        const double Lx = d.N[0] * d.h[0], m = n[0] * Lx / (2 * M_PI), up = std::hypot(n[1], n[2]);
        if (std::abs(m - std::round(m)) > 1e-6)
            throw std::runtime_error("side: periodic x needs cos(elevation) cos(angle) nx dx = m lambda0 (now m = " +
                                     std::to_string(m) + "), or open x");
        n[0] = 2 * M_PI * std::round(m) / Lx;
        if (up > 0) { const double f = std::sqrt(std::max(0.0, 1 - n[0] * n[0])) / up; n[1] *= f; n[2] *= f; }
    }
    // polarisation of a component of wave vector k: es = z x k / |z x k| (in the x-y plane for elevation 0, 'y'),
    // ep = k x es / |k| (z for elevation 0, 'z')
    auto pol_of = [&](Wave& x) {
        const double kk = std::sqrt(x.k[0] * x.k[0] + x.k[1] * x.k[1] + x.k[2] * x.k[2]);
        const double u[3] = {x.k[0] / kk, x.k[1] / kk, x.k[2] / kk};
        double es[3] = {-u[1], u[0], 0};
        const double nes = std::hypot(es[0], es[1]);
        if (nes < 1e-9) { es[0] = 0; es[1] = 1; } else { es[0] /= nes; es[1] /= nes; }
        const double ep[3] = {u[1] * es[2] - u[2] * es[1], u[2] * es[0] - u[0] * es[2], u[0] * es[1] - u[1] * es[0]};
        for (int a = 0; a < 3; ++a) x.e[a] = c.pol == 'y' ? es[a] : ep[a];
    };
    std::vector<Wave> ws;
    auto add = [&](const double k[3], double w, cd amp, int cls) {
        Wave x{{k[0], k[1], k[2]}, w, {amp.real(), amp.imag()}, {0, 0, 0}, cls};
        pol_of(x);
        // the reference: the phase k . (r - r0) - w (t - tc)
        const cd a = amp * std::exp(cd(0, -(k[0] * c.r0[0] + k[1] * c.r0[1] + k[2] * c.r0[2]) + w * c.tc));
        x.amp[0] = a.real(); x.amp[1] = a.imag();
        ws.push_back(x);
    };
    if (c.tau <= 0 && wp2 == 0) {
        add(n, omega(d, n, irip), c.a0 * std::exp(cd(0, c.phase)), -1);
        return ws;
    }
    // envelope g(t) = exp(-al t^2) = sum over nu of (dnu / 2 pi) G(nu) e^{-i nu t}, G = sqrt(pi / al) e^{-nu^2 / (4 al)}:
    // periodic in time with T = 2 pi / dnu, longer than everything the run sees (continuous wave: nu = 0 alone)
    const double al = c.tau > 0 ? 4 * std::log(2.0) / (c.tau * c.tau) : 0.0;
    const double numax = 12 * std::sqrt(al), T = 2 * (t_span + (al > 0 ? 12 / std::sqrt(2 * al) : 0.0)), dnu = 2 * M_PI / T;
    if (numax >= 0.9) throw std::runtime_error("side.duration: too short for a pulse of carrier frequency omega0");
    const int M = al > 0 ? int(std::ceil(numax / dnu)) : 0;
    if (2 * M + 1 > 20000) throw std::runtime_error("side: more than 20000 spectral components (run too long for the pulse)");
    const double up = std::hypot(n[1], n[2]);
    if (px && up < 1e-9) throw std::runtime_error("side: a pulse along x in a box periodic along x never crosses a face");
    const double dir[3] = {px ? 0.0 : n[0], px ? n[1] / up : n[1], px ? n[2] / up : n[2]};
    double smax = 1e300;
    for (int a = 0; a < 3; ++a)
        if (std::abs(dir[a]) > 1e-12) smax = std::min(smax, M_PI / (std::abs(dir[a]) * d.h[a]));
    smax *= 1 - 1e-12;
    // RIP-CC in a plasma: TE and TM apart (degenerate only for k_perp = 0)
    const bool two = split(d, c) && std::hypot(dir[1], dir[2]) > 1e-12;
    for (int m = -M; m <= M; ++m) {
        const double nu = m * dnu, w = 1 + nu;
        const double rel = std::exp(-nu * nu * (al > 0 ? 1 / (4 * al) : 0.0));   // relative to the central component
        const cd amp = c.a0 * std::exp(cd(0, c.phase)) * (al > 0 ? (dnu / (2 * M_PI)) * std::sqrt(M_PI / al) : 1.0) * rel;
        auto kof = [&](double s, double k[3]) { for (int a = 0; a < 3; ++a) k[a] = s * dir[a]; if (px) k[0] = n[0]; };
        for (int cls = two ? 0 : -1; cls <= (two ? 1 : -1); ++cls) {
            const int cw = cls < 0 ? 0 : cls;
            double k[3], lo = 0, hi = smax;
            kof(hi, k);
            if (omega(d, k, irip, wp2, cw) < w) {
                if (rel > 1e-6) throw std::runtime_error("side: the pulse spectrum reaches beyond the grid cutoff");
                break;
            }
            kof(lo, k);
            if (omega(d, k, irip, wp2, cw) > w) {      // evanescent: below the cutoff of the family (the plasma, periodic x)
                if (rel > 1e-6)
                    throw std::runtime_error(wp2 > 0 ? "side: the spectrum reaches below the plasma cutoff (background_density "
                                                       "or a longer pulse)"
                                                     : "side: the pulse spectrum reaches below omega = k_x (a longer pulse)");
                break;
            }
            for (int it = 0; it < 200; ++it) {
                const double s = 0.5 * (lo + hi);
                kof(s, k);
                (omega(d, k, irip, wp2, cw) < w ? lo : hi) = s;
            }
            kof(0.5 * (lo + hi), k);
            add(k, w, amp, cls);
        }
        if (two && !ws.empty() && ws.back().cls == 0) ws.pop_back();   // a TE wave whose TM partner was dropped
    }
    return ws;
}

SideTfsf::SideTfsf(const Domain& d, const SideConfig& c, double t_span) : d_(d), c_(c) {
    const bool irip = d.solver == Solver::IRip;
    const double wp2 = c.density;
    const std::vector<Wave> ws = c.wave == 1 ? luminal_waves(d, c, groups_) : plane_waves(d, c, t_span);
    // per component: the discrete eigenvector with E closest to amp e (least squares in the 2D eigenspace, or over the
    // TE and TM modes of a pair at their own wave vectors)
    for (size_t n = 0; n < ws.size(); ++n) {
        const Wave& x = ws[n];
        const bool pair = x.cls == 0;
        if (x.cls == 1 || (pair && (n + 1 >= ws.size() || ws[n + 1].cls != 1)))
            throw std::runtime_error("side: a TE / TM pair of the provider is incomplete");
        const Wave* xs[2] = {&x, pair ? &ws[n + 1] : &x};
        const cd amp(x.amp[0], x.amp[1]);
        cd U[2][9];
        if (pair) {
            for (int p = 0; p < 2; ++p) {
                cd N[2][9];
                if (modes(d, xs[p]->k, x.w, wp2, irip, p, N) != 1)
                    throw std::runtime_error("side: no single TE / TM mode of the discrete plane wave in the plasma");
                for (int q = 0; q < 9; ++q) U[p][q] = N[0][q];
            }
            // the provider's amplitudes hold the phase of the reference point (k . r0): the pair fits amp e there, so
            // the TM vector carries e^{i (k_TM - k_TE) . r0} = amp_TE / amp_TM
            const cd ra = amp / cd(xs[1]->amp[0], xs[1]->amp[1]);
            for (int q = 0; q < 9; ++q) U[1][q] *= ra;
        } else if (modes(d, x.k, x.w, wp2, irip, -1, U) != 2) {
            throw std::runtime_error("side: no two-dimensional eigenspace of the discrete plane wave");
        }
        // E at the physical points (Ey at j + 1/2, Ez at k + 1/2), the phase of the stored index taken off
        cd sh[2][3];
        for (int p = 0; p < 2; ++p) {
            sh[p][0] = 1.0;
            sh[p][1] = std::exp(cd(0, -0.5 * xs[p]->k[1] * d.h[1]));
            sh[p][2] = std::exp(cd(0, -0.5 * xs[p]->k[2] * d.h[2]));
        }
        cd M11 = 0, M12 = 0, M22 = 0, r1 = 0, r2 = 0;
        for (int q = 0; q < 3; ++q) {
            const cd w1 = U[0][q] * sh[0][q], w2 = U[1][q] * sh[1][q], t = amp * x.e[q];
            M11 += std::conj(w1) * w1; M12 += std::conj(w1) * w2; M22 += std::conj(w2) * w2;
            r1 += std::conj(w1) * t; r2 += std::conj(w2) * t;
        }
        const cd det = M11 * M22 - M12 * std::conj(M12);
        const cd al1 = (M22 * r1 - M12 * r2) / det, al2 = (M11 * r2 - std::conj(M12) * r1) / det;
        cd v[2][9];
        for (int q = 0; q < 9; ++q) {
            if (pair) { v[0][q] = al1 * U[0][q]; v[1][q] = al2 * U[1][q]; }
            else v[0][q] = al1 * U[0][q] + al2 * U[1][q];
        }
        double res = 0, ref = 0;
        for (int q = 0; q < 3; ++q) {
            const cd e = v[0][q] * sh[0][q] + (pair ? v[1][q] * sh[1][q] : 0.0);
            res += std::norm(e - amp * x.e[q]); ref += std::norm(amp * x.e[q]);
        }
        if (ref > 0) mism_ = std::max(mism_, std::sqrt(res / ref));
        const cd lam = std::exp(cd(0, -x.w * d.h[0])), mu = 0.5 * (1.0 + lam);
        for (int p = 0; p < (pair ? 2 : 1); ++p) {
            cd pE, pB, out[9];
            mode_step(d, xs[p]->k, wp2, v[p], out, irip, &pE, &pB);
            double r = 0, nv = 0;
            for (int q = 0; q < 9; ++q) { r += std::norm(out[q] - lam * v[p][q]); nv += std::norm(v[p][q]); }
            if (nv > 0) res_ = std::max(res_, std::sqrt(r / nv));
            Comp o{};
            for (int a = 0; a < 3; ++a) o.k[a] = xs[p]->k[a];
            o.w = x.w;
            for (int q = 0; q < 6; ++q) { o.v[2 * q] = v[p][q].real(); o.v[2 * q + 1] = v[p][q].imag(); }
            for (int q = 0; q < 3; ++q) { o.j[2 * q] = v[p][6 + q].real(); o.j[2 * q + 1] = v[p][6 + q].imag(); }
            o.pr[0] = pE.real(); o.pr[1] = pE.imag(); o.pr[2] = pB.real(); o.pr[3] = pB.imag();
            o.mu[0] = mu.real(); o.mu[1] = mu.imag();
            if (irip && d.o5) {                        // the closed mean as Ex*, Bx* (kind 3)
                const double cf = o5_factor(d, xs[p]->k);
                const cd e = cf * mu * v[p][0], b = cf * mu * v[p][3];
                o.pr[0] = e.real(); o.pr[1] = e.imag(); o.pr[2] = b.real(); o.pr[3] = b.imag();
            }
            comps_.push_back(o);
        }
        if (pair) ++n;
    }
    if (res_ > 1e-10) throw std::runtime_error("side: a discrete mode off by " + std::to_string(res_) + " (one-step matrix)");
    std::vector<double> h(comps_.size() * NC);
    for (size_t m = 0; m < comps_.size(); ++m) std::memcpy(&h[m * NC], &comps_[m], sizeof(Comp));
    dev_.upload(h);
    // the phase e^{i k . r} as a product of tables along x (plane waves; local i, without the window offset), y and z
    // (absolute index), after e^{i kx D}, e^{i ky hy}, e^{i kz hz}: no sin, cos in the kernels
    const int nx = d.n[0], ny = d.n[1], nz = d.n[2], S = 3 + ny + nz;
    std::vector<double> tb(2 * size_t(S) * comps_.size());
    std::vector<double> tx(c.wave == 1 ? 2 : 2 * size_t(nx) * comps_.size());
    for (size_t m = 0; m < comps_.size(); ++m) {
        const Comp& p = comps_[m];
        double* t = &tb[2 * size_t(S) * m];
        auto put = [](double* t, int q, double ph) { t[2 * q] = std::cos(ph); t[2 * q + 1] = std::sin(ph); };
        put(t, 0, p.k[0] * d.h[0]); put(t, 1, p.k[1] * d.h[1]); put(t, 2, p.k[2] * d.h[2]);
        for (int j = 0; j < ny; ++j) put(t, 3 + j, p.k[1] * double(d.off[1] + j) * d.h[1]);
        for (int k = 0; k < nz; ++k) put(t, 3 + ny + k, p.k[2] * double(d.off[2] + k) * d.h[2]);
        if (c.wave != 1)
            for (int i = 0; i < nx; ++i) put(&tx[2 * size_t(nx) * m], i, p.k[0] * i * d.h[0]);
    }
    tab_.upload(tb);
    tx_.upload(tx);
    ft_.resize(2 * comps_.size());
    // the (j, k) columns whose differences may cross a face: half node lo - 1, node lo, half node hi along each face axis
    std::vector<int> jk;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            const long jg = d.off[1] + j, kg = d.off[2] + k;
            const bool ny_ = c.faces[1] && (jg == c.lo[1] - 1 || jg == c.lo[1] || jg == c.hi[1]);
            const bool nz_ = c.faces[2] && (kg == c.lo[2] - 1 || kg == c.lo[2] || kg == c.hi[2]);
            if (ny_ || nz_) { jk.push_back(j); jk.push_back(k); }
        }
    jk_.resize(std::max<size_t>(jk.size(), 2));
    if (!jk.empty()) jk_.upload(jk);
    nb_ = int(jk.size() / 2);
    if (fluid()) {                                     // the fluid fractions of the local (j, k) columns
        std::vector<double> F(3 * size_t(ny) * nz);
        for (int q = 0; q < 3; ++q)
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j) F[(size_t(q) * nz + k) * ny + j] = fluid_fraction(q, d.off[1] + j, d.off[2] + k);
        fl_.upload(F);
    }
    if (c.wave == 1) {
        kappa_ = comps_.empty() ? 0 : comps_[0].w - comps_[0].k[0];
        for (size_t g = 0; g + 1 < groups_.size(); ++g) gx_.push_back(comps_[size_t(groups_[g])].k[0]);
        build_table();
    }
}

// Plane waves: e^{i (kx D cell0(0) - w t)} per component. Flying focus: e^{-i kappa t}; the table holds the phase of the
// window frame, so the window must have moved one cell per step from t = 0 (x0 = t).
void SideTfsf::begin(double t) {
    t_ = t;
    const double x0 = double(d_.cell0(0) - d_.off[0]) * d_.h[0];
    if (c_.wave == 1) {
        if (std::abs(x0 - t) > 1e-6 * d_.h[0])
            throw std::runtime_error("side flying focus: the window must move at c from t = 0 (shifted " + std::to_string(x0) +
                                     ", t = " + std::to_string(t) + ")");
        ftk_[0] = std::cos(kappa_ * t); ftk_[1] = -std::sin(kappa_ * t);
        return;
    }
    std::vector<double> f(2 * comps_.size());
    const double xa = double(d_.cell0(0)) * d_.h[0];
    for (size_t m = 0; m < comps_.size(); ++m) {
        const double ph = comps_[m].k[0] * xa - comps_[m].w * t;
        f[2 * m] = std::cos(ph); f[2 * m + 1] = std::sin(ph);
    }
    ft_.from_host(f.data(), 0, f.size());
}

// Predictor (RIP-CC, t_n): Ex* at node (j, k), Bx* at half (j, k); a neighbour across a face enters with its incident value.
void SideTfsf::fix_pred(const FV Exs, const FV Bxs) const {
    const Geo g = geo(d_, c_, dev_, tab_, tx_, ft_);
    const int* jk = jk_.data();
    const double iy = 1 / g.hy, iz = 1 / g.hz;
    const double* H = H_.data();
    const bool tab = c_.wave == 1;
    const Cx fk{ftk_[0], ftk_[1]};
    par_for(long(nb_) * g.nx, [=] HD(long q) {
        const long b = q / g.nx;
        const int i = int(q % g.nx), j = jk[2 * b], k = jk[2 * b + 1];
        Cx dE{0, 0}, dB{0, 0};
        if (tab) {
            const double* h = H + 16 * (b * g.nx + i);
            dE = fk * Cx{h[0], h[1]}; dB = fk * Cx{h[2], h[3]};
        } else {
            const Wts w = weights(g, g.o1 + j, g.o2 + k);
            for (int m = 0; m < g.nc; ++m) {
                Cx aE, aB;
                bracket_pc(g.cp + m * NCD, g.ey(m), g.ez(m), w, iy, iz, aE, aB);
                const Cx P = g.phase(m, i, j, k);
                dE += P * aE; dB += P * aB;
            }
        }
        Exs(i, j, k) += 0.5 * g.D * dE.r;
        Bxs(i, j, k) += 0.5 * g.D * dB.r;
    });
}

void SideTfsf::fix_transport(const FV nEy, const FV nEz, const FV nBy, const FV nBz, int kind) const {
    if (c_.wave == 1 && kind != (d_.solver == Solver::IRip ? (d_.o5 ? 3 : 2) : 0))
        throw std::runtime_error("side flying focus: the table was built for the other solver");
    const Geo g = geo(d_, c_, dev_, tab_, tx_, ft_);
    const int* jk = jk_.data();
    const double iy = 1 / g.hy, iz = 1 / g.hz;
    const double* H = H_.data();
    const bool tab = c_.wave == 1;
    const Cx fk{ftk_[0], ftk_[1]};
    par_for(long(nb_) * g.nx, [=] HD(long q) {
        const long b = q / g.nx;
        const int i = int(q % g.nx), j = jk[2 * b], k = jk[2 * b + 1];
        Cx d[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        if (tab) {
            const double* h = H + 16 * (b * g.nx + i) + 4;
            for (int r = 0; r < 4; ++r) d[r] = fk * Cx{h[2 * r], h[2 * r + 1]};
        } else {
            const Wts w = weights(g, g.o1 + j, g.o2 + k);
            for (int m = 0; m < g.nc; ++m) {
                Cx t[4];
                bracket_t(g.cp + m * NCD, g.ex(m), g.ey(m), g.ez(m), w, iy, iz, kind, t);
                const Cx P = g.phase(m, i, j, k);
                for (int r = 0; r < 4; ++r) d[r] += P * t[r];
            }
        }
        const double h = 0.5 * g.D;
        nEy(i, j, k) += h * d[0].r; nBz(i, j, k) += h * d[1].r; nEz(i, j, k) += h * d[2].r; nBy(i, j, k) += h * d[3].r;
    });
}

// Corrector: as the predictor with the mean of t_n and t_n+1 and the full step D.
void SideTfsf::fix_corr(const FV Ex, const FV Bx) const {
    const Geo g = geo(d_, c_, dev_, tab_, tx_, ft_);
    const int* jk = jk_.data();
    const double iy = 1 / g.hy, iz = 1 / g.hz;
    const double* H = H_.data();
    const bool tab = c_.wave == 1;
    const Cx fk{ftk_[0], ftk_[1]};
    par_for(long(nb_) * g.nx, [=] HD(long q) {
        const long b = q / g.nx;
        const int i = int(q % g.nx), j = jk[2 * b], k = jk[2 * b + 1];
        Cx dE{0, 0}, dB{0, 0};
        if (tab) {
            const double* h = H + 16 * (b * g.nx + i) + 12;
            dE = fk * Cx{h[0], h[1]}; dB = fk * Cx{h[2], h[3]};
        } else {
            const Wts w = weights(g, g.o1 + j, g.o2 + k);
            for (int m = 0; m < g.nc; ++m) {
                Cx aE, aB;
                bracket_pc(g.cp + m * NCD, g.ey(m), g.ez(m), w, iy, iz, aE, aB);
                const Cx P = g.phase(m, i, j, k) * MU(g.cp + m * NCD);
                dE += P * aE; dB += P * aB;
            }
        }
        Ex(i, j, k) += g.D * dE.r;
        Bx(i, j, k) += g.D * dB.r;
    });
}

void SideTfsf::add_incident(const FV f[6], double t, bool masked, const FV* jf) {
    begin(t);
    if (c_.wave == 1) { fill_luminal(f, t, masked, jf); return; }
    const Geo g = geo(d_, c_, dev_, tab_, tx_, ft_);
    Nine s;
    for (int q = 0; q < 6; ++q) s.v[q] = f[q];
    for (int q = 0; q < 3; ++q) s.v[6 + q] = jf ? jf[q] : f[q];
    const int nq = jf ? 9 : 6, ny = d_.n[1];
    const long nyz = long(ny) * d_.n[2];
    const double* F = fl_.data();
    for_owned(d_, [=] HD(int i, int j, int k) {
        const long jg = g.o1 + j, kg = g.o2 + k;
        for (int q = 0; q < nq; ++q) {
            if (masked && g.T(q, jg, kg) == 0) continue;
            const double w = q < 6 ? 1.0 : F[(q - 6) * nyz + long(k) * ny + j];
            if (w == 0) continue;
            double v = 0;
            for (int m = 0; m < g.nc; ++m) v += (g.phase(m, i, j, k) * V9(g.cp + m * NCD, q)).r;
            s.v[q](i, j, k) += w * v;
        }
    });
}

void SideTfsf::o5_means(const FV ex, const FV bx, bool closed) {
    if (c_.wave == 1) { means_luminal(ex, bx, closed); return; }
    const Geo g = geo(d_, c_, dev_, tab_, tx_, ft_);
    for_owned(d_, [=] HD(int i, int j, int k) {
        const long jg = g.o1 + j, kg = g.o2 + k;
        const bool te = g.T(0, jg, kg) != 0, tb = g.T(3, jg, kg) != 0;
        if (!te && !tb) return;
        Cx a{0, 0}, b{0, 0};
        for (int m = 0; m < g.nc; ++m) {
            const double* c = g.cp + m * NCD;
            const Cx P = g.phase(m, i, j, k);
            a += P * (closed ? PR(c, 0) : V(c, 0) * MU(c));
            b += P * (closed ? PR(c, 1) : V(c, 3) * MU(c));
        }
        if (te) ex(i, j, k) += a.r;
        if (tb) bx(i, j, k) += b.r;
    });
}

double SideTfsf::incident(int q, long i, long j, long k, double t) const {
    double v = 0;
    for (const Comp& c : comps_) {
        const double p = c.k[0] * double(i) * d_.h[0] + c.k[1] * double(j) * d_.h[1] + c.k[2] * double(k) * d_.h[2] - c.w * t;
        const double* a = q < 6 ? &c.v[2 * q] : &c.j[2 * (q - 6)];
        v += a[0] * std::cos(p) - a[1] * std::sin(p);
    }
    return v;
}

bool SideTfsf::total(int q, long j, long k) const {
    Geo g;
    for (int a = 0; a < 3; ++a) { g.lo[a] = c_.lo[a]; g.hi[a] = c_.hi[a]; g.f[a] = c_.faces[a]; }
    return g.T(q, j, k) > 0;
}

namespace {
// The cumulative of the B-spline of order q (0: top hat, 1: linear, 2: quadratic) of unit cell, int_{-inf}^u S_q.
double spline_cum(int q, double u) {
    if (q == 0) return std::min(1.0, std::max(0.0, u + 0.5));
    if (q == 1) return u <= -1 ? 0.0 : u <= 0 ? 0.5 * (1 + u) * (1 + u) : u <= 1 ? 1 - 0.5 * (1 - u) * (1 - u) : 1.0;
    if (u <= -1.5) return 0.0;
    if (u <= -0.5) return (u + 1.5) * (u + 1.5) * (u + 1.5) / 6;
    if (u <= 0.5) return 0.5 + 0.75 * u - u * u * u / 3;
    if (u <= 1.5) return 1 - (1.5 - u) * (1.5 - u) * (1.5 - u) / 6;
    return 1.0;
}
}  // namespace

// The fluid fraction of current component c (Jx, Jy, Jz) at the global (j, k): 1 minus the coverage of the point by
// the particles of a uniform plasma in the kinetic region (2 cells inside the faces: [lo + 2, hi - 2] h). The deposit
// (Esirkepov) weights a point with the shape of order p across the motion and of order p - 1 along it (Jy at the half
// node in y, Jz in z), so the deposited current of that plasma plus the fluid is the uniform current. 0 in the CPML.
double SideTfsf::fluid_fraction(int c, long j, long k) const {
    if (!fluid()) return 0;
    const double y = double(j) + (c == 1 ? 0.5 : 0.0), z = double(k) + (c == 2 ? 0.5 : 0.0), u[3] = {0, y, z};
    double cover = c_.fluid_all ? 0.0 : 1.0;
    for (int a = 1; a < 3; ++a) {
        const int L = c_.absorb[a];
        if (L > 0 && (u[a] < L || u[a] > d_.N[a] - 1 - L)) return 0;
        if (!c_.faces[a] || c_.fluid_all) continue;
        const int q = c == a ? c_.order - 1 : c_.order;
        const double lo = c_.lo[a] + 2, hi = c_.hi[a] - 2;
        cover *= spline_cum(q, u[a] - lo) - spline_cum(q, u[a] - hi);
    }
    return 1 - cover;
}

// One step of the cold fluid, j^{n+1/2} = j^{n-1/2} + D wp^2 F (Ex, mux Ey, mux Ez)^n, added to the deposited j.
void SideTfsf::fluid_current(const FV E[3], const FV Jf[3], const FV J[3]) const {
    const FV Ex = E[0], Ey = E[1], Ez = E[2], Fx = Jf[0], Fy = Jf[1], Fz = Jf[2], Jx = J[0], Jy = J[1], Jz = J[2];
    const double* F = fl_.data();
    const double c = d_.h[0] * c_.density;
    const int ny = d_.n[1];
    const long nyz = long(ny) * d_.n[2];
    for_owned(d_, [=] HD(int i, int j, int k) {
        const long p = long(k) * ny + j;
        Fx(i, j, k) += c * F[p] * Ex(i, j, k);
        Fy(i, j, k) += c * F[nyz + p] * 0.5 * (Ey(i, j, k) + Ey(i + 1, j, k));
        Fz(i, j, k) += c * F[2 * nyz + p] * 0.5 * (Ez(i, j, k) + Ez(i + 1, j, k));
        Jx(i, j, k) += Fx(i, j, k); Jy(i, j, k) += Fy(i, j, k); Jz(i, j, k) += Fz(i, j, k);
    });
}
