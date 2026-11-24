// Line systems along x after a transform in y, z (src/irip.cpp, the iRIP box solve; src/filter.cpp, the projection
// filter): the DFT lines, the Thomas algorithm and the systems of all transverse modes with x split over the ranks
// along x (the partition method). Moved from src/irip.cpp; the mode coefficients are a parameter.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include "comm.hpp"
#include "grid.hpp"
#include "portable.hpp"

namespace xlines {

// One complex line a[q * stride] (interleaved re, im), q < n, transformed in place:
// a_q <- sum_l a_l e^{sign 2 pi i q l / n}. Twiddles tw: cos, sin of -2 pi k / n, k < n (twiddles()). Radix 2 in place
// for powers of two; else mixed radix (Stockham autosort, the smallest prime factor first, O(n sum p)), ping-pong between
// a and the scratch line t (stride tstride).
HD inline void dft_line(double* a, long stride, int n, const double* tw, int sign, double* t, long tstride) {
    if ((n & (n - 1)) == 0) {
        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                const long p = 2 * i * stride, q = 2 * j * stride;
                const double r = a[p], m = a[p + 1];
                a[p] = a[q]; a[p + 1] = a[q + 1]; a[q] = r; a[q + 1] = m;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            const int step = n / len;
            for (int i = 0; i < n; i += len)
                for (int k = 0; k < len / 2; ++k) {
                    const double wr = tw[2 * (k * step)], wi = sign < 0 ? tw[2 * (k * step) + 1] : -tw[2 * (k * step) + 1];
                    const long p = 2 * (i + k) * stride, q = 2 * (i + k + len / 2) * stride;
                    const double vr = a[q] * wr - a[q + 1] * wi, vi = a[q] * wi + a[q + 1] * wr;
                    const double ur = a[p], ui = a[p + 1];
                    a[p] = ur + vr; a[p + 1] = ui + vi;
                    a[q] = ur - vr; a[q + 1] = ui - vi;
                }
        }
        return;
    }
    // stage with factor p (l p m = n, l: the factors done; Van Loan, Computational Frameworks for the FFT, alg. 1.7.2):
    // y[(j + r l) m + k] = sum_q x[j p m + q m + k] w^{q (j + r l)}, w = e^{sign 2 pi i / (p l)} = tw[(n / (p l)) .]
    double* x = a;
    double* y = t;
    long xs = stride, ys = tstride;
    int l = 1, m = n;
    while (m > 1) {
        int p = 2;
        while (m % p) ++p;
        m /= p;
        const long step = n / (long(p) * l);
        for (int j = 0; j < l; ++j)
            for (int k = 0; k < m; ++k)
                for (int r = 0; r < p; ++r) {
                    double sr = 0, si = 0;
                    for (int q = 0; q < p; ++q) {
                        const long src = 2 * (long(j) * p * m + long(q) * m + k) * xs;
                        const long e = (long(q) * (j + long(r) * l) * step) % n;
                        const double wr = tw[2 * e], wi = sign < 0 ? tw[2 * e + 1] : -tw[2 * e + 1];
                        sr += x[src] * wr - x[src + 1] * wi;
                        si += x[src] * wi + x[src + 1] * wr;
                    }
                    const long dst = 2 * ((j + long(r) * l) * m + k) * ys;
                    y[dst] = sr; y[dst + 1] = si;
                }
        double* sw = x; x = y; y = sw;
        const long ss = xs; xs = ys; ys = ss;
        l *= p;
    }
    if (x != a)
        for (int q = 0; q < n; ++q) { a[2 * q * stride] = x[2 * q * xs]; a[2 * q * stride + 1] = x[2 * q * xs + 1]; }
}

// Thomas algorithm for b x_i + s (x_{i-1} + x_{i+1}) = r_i, i < n, with the first and last diagonal
// entries b0, bn (the cyclic reduction below); complex right-hand side in place (stride 2 * stride),
// g: scratch of n reals.
HD inline void thomas(double* x, long stride, int n, double b, double b0, double bn, double s, double* g, int re) {
    double bet = b0;
    x[2 * 0 * stride + re] /= bet;
    for (int i = 1; i < n; ++i) {
        g[i] = s / bet;
        bet = (i == n - 1 ? bn : b) - s * g[i];
        x[2 * i * stride + re] = (x[2 * i * stride + re] - s * x[2 * (i - 1) * stride + re]) / bet;
    }
    for (int i = n - 2; i >= 0; --i) x[2 * i * stride + re] -= g[i + 1] * x[2 * (i + 1) * stride + re];
}

// One complex line a[q * stride] (interleaved re, im), q < n, multiplied by the real n x n matrix M
// (row-major), through the scratch line t (same stride).
HD inline void dense_line(double* a, long stride, int n, const double* M, double* t) {
    for (int q = 0; q < n; ++q) {
        double sr = 0, si = 0;
        for (int l = 0; l < n; ++l) { sr += M[long(q) * n + l] * a[2 * l * stride]; si += M[long(q) * n + l] * a[2 * l * stride + 1]; }
        t[2 * q * stride] = sr; t[2 * q * stride + 1] = si;
    }
    for (int q = 0; q < n; ++q) { a[2 * q * stride] = t[2 * q * stride]; a[2 * q * stride + 1] = t[2 * q * stride + 1]; }
}

inline std::vector<double> twiddles(int n) {
    std::vector<double> t(static_cast<size_t>(2 * std::max(n, 1)), 0.0);
    for (int k = 0; k < n; ++k) { t[size_t(2 * k)] = std::cos(-2 * M_PI * k / n); t[size_t(2 * k + 1)] = std::sin(-2 * M_PI * k / n); }
    return t;
}

// The transforms of the spectrum w (nx x ny x nz complex) along y (lines over i, k) and z (lines over
// i, j): the DFT (sign), or with a matrix M the dense eigenbasis of a pml axis.
inline void transform_y(double* w, double* t, int nx, int ny, int nz, const double* M, const double* ty, int sign) {
    const long nxy = long(nx) * ny;
    par_for(long(nx) * nz, [=] HD(long L) {
        const int i = int(L % nx), k = int(L / nx);
        const long o = 2 * (long(k) * nxy + i);
        if (M) dense_line(w + o, nx, ny, M, t + o);
        else dft_line(w + o, nx, ny, ty, sign, t + o, nx);
    });
}
inline void transform_z(double* w, double* t, int nx, int ny, int nz, const double* M, const double* tz, int sign) {
    const long nxy = long(nx) * ny;
    par_for(nxy, [=] HD(long L) {
        if (M) dense_line(w + 2 * L, nxy, nz, M, t + 2 * L);
        else dft_line(w + 2 * L, nxy, nz, tz, sign, t + 2 * L, nxy);
    });
}

// The system of one transverse mode along x: b x_i + s (x_{i-1} + x_{i+1}) = r_i over the global x nodes; cyclic
// (periodic x, Sherman-Morrison), or with the diagonals b0, bn in the first and last global rows (zeros outside).
// s = 0 with b = 1 leaves the line as it is (iRIP's mode k = 0, where the system is the identity).
struct LineCoef { double b, s, b0, bn; bool cyc; };

// Spectrum and scratch of the line solves: w (complex, nx per mode, mode M = qy + ny qz), t (transform scratch), g, z
// (Thomas and the cyclic correction), the twiddles, and for x split over ranks yw, ex, ga, red (partition method).
struct LineWork {
    Buf<double> w, t, g, z;
    Buf<double> twy, twz;
    Buf<double> yw, ex, ga, red;
    std::vector<double> exh, gah;
    void resize(const Domain& d) {
        const size_t n = size_t(d.owned());
        w.resize(2 * n); t.resize(2 * n); g.resize(n); z.resize(2 * n);
        const std::vector<double> ty = twiddles(d.n[1]), tz = twiddles(d.n[2]);
        twy.resize(ty.size()); twy.from_host(ty.data(), 0, ty.size());
        twz.resize(tz.size()); twz.from_host(tz.data(), 0, tz.size());
    }
};

// The systems of all modes, the lines of W.w in place. coef(M) -> LineCoef (HD). One rank along x: Thomas per mode
// (cyclic: Sherman-Morrison). x split over P = comm.dims[0] ranks (y, z not split): the partition method. The global
// cyclic matrix is A' + (gam e_0 + s e_N-1)(e_0 + s/gam e_N-1)^T (Sherman-Morrison, A' tridiagonal with modified first
// and last diagonal); A' x = r per rank: x = y + R_{p-1} u + L_{p+1} v with the local solutions T y = r, T u = -s e_0,
// T v = -s e_n-1 and the neighbours' edge values R_{p-1} (last of rank p-1), L_{p+1} (first of rank p+1). These
// interface values solve a block-tridiagonal system of 2 x 2 blocks, unknowns I_q = (R_{q-1}, L_q), q = 1 .. P-1, from
// the edge values of all ranks (allgather, 10 per mode and rank); every rank solves it for all modes. Three right-hand
// sides: Re r, Im r and w (the Sherman-Morrison vector). Non-cyclic modes: b0, bn on the first and last rank, no
// Sherman-Morrison term.
template <class Coef>
void solve_lines(LineWork& W, const Domain& d, Comm& comm, Coef coef) {
    const int nx = d.n[0], ny = d.n[1], nz = d.n[2], P = comm.dims[0], pr = comm.coords[0];
    const long nm = long(ny) * nz;
    double* w = W.w.data();
    double* g = W.g.data();
    double* zc = W.z.data();
    if (P == 1) {
        par_for(nm, [=] HD(long M) {
            const LineCoef c = coef(M);
            const double b = c.b, s = c.s;
            double* x = w + 2 * M * nx;
            double* gl = g + M * nx;
            double* zl = zc + 2 * M * nx;
            if (s == 0 && b == 1) return;                        // the identity
            if (!c.cyc) {
                for (int re = 0; re < 2; ++re) thomas(x, 1, nx, b, c.b0, c.bn, s, gl, re);
                return;
            }
            const double gam = -b, b0 = b - gam, bn = b - s * s / gam;
            for (int re = 0; re < 2; ++re) thomas(x, 1, nx, b, b0, bn, s, gl, re);
            for (int i = 0; i < nx; ++i) { zl[2 * i] = 0; zl[2 * i + 1] = 0; }
            zl[0] = gam; zl[2 * (nx - 1)] = s;
            thomas(zl, 1, nx, b, b0, bn, s, gl, 0);
            const double den = 1 + zl[0] + s * zl[2 * (nx - 1)] / gam;
            for (int re = 0; re < 2; ++re) {
                const double f = (x[re] + s * x[2 * (nx - 1) + re] / gam) / den;
                for (int i = 0; i < nx; ++i) x[2 * i + re] -= f * zl[2 * i];
            }
        });
        return;
    }
    if (W.ex.size() != size_t(10 * nm)) {
        W.yw.resize(size_t(2 * nm * nx)); W.ex.resize(size_t(10 * nm));
        W.ga.resize(size_t(10 * nm * P)); W.red.resize(size_t(10 * nm * (P - 1)));
        W.exh.resize(size_t(10 * nm)); W.gah.resize(size_t(10 * nm * P));
    }
    double* yw = W.yw.data();
    double* ex = W.ex.data();
    // A. local solves: y in w (Re, Im), u, v in z (Re, Im slots), A'^-1 w in yw; the edge values
    par_for(nm, [=] HD(long M) {
        const LineCoef c = coef(M);
        const double b = c.b, s = c.s;
        double* e = ex + 10 * M;
        for (int q = 0; q < 10; ++q) e[q] = 0;
        if (s == 0 && b == 1) return;                        // the identity on every rank
        const bool first = c.cyc && pr == 0, last = c.cyc && pr == P - 1;
        const double gam = -b;
        const double b0 = first ? b - gam : (!c.cyc && pr == 0 ? c.b0 : b);
        const double bn = last ? b - s * s / gam : (!c.cyc && pr == P - 1 ? c.bn : b);
        double* x = w + 2 * M * nx;
        double* gl = g + M * nx;
        double* zl = zc + 2 * M * nx;
        double* wl = yw + 2 * M * nx;
        for (int re = 0; re < 2; ++re) thomas(x, 1, nx, b, b0, bn, s, gl, re);
        for (int i = 0; i < 2 * nx; ++i) { zl[i] = 0; wl[i] = 0; }
        zl[0] = -s; zl[2 * (nx - 1) + 1] = -s;
        if (first) wl[0] = gam;
        if (last) wl[2 * (nx - 1)] += s;
        for (int re = 0; re < 2; ++re) thomas(zl, 1, nx, b, b0, bn, s, gl, re);
        thomas(wl, 1, nx, b, b0, bn, s, gl, 0);
        const int l = 2 * (nx - 1);
        e[0] = zl[0]; e[1] = zl[l]; e[2] = zl[1]; e[3] = zl[l + 1];          // u_0, u_n-1, v_0, v_n-1
        e[4] = x[0]; e[5] = x[l]; e[6] = x[1]; e[7] = x[l + 1]; e[8] = wl[0]; e[9] = wl[l];
    });
    // B. the edge values of all ranks
    W.ex.to_host(W.exh.data(), 0, W.exh.size());
    comm.allgather(W.exh.data(), int(10 * nm), W.gah.data());
    W.ga.from_host(W.gah.data(), 0, W.gah.size());
    const double* ga = W.ga.data();
    double* red = W.red.data();
    // C. per mode the reduced system (block Thomas), the global edge values for Sherman-Morrison, and
    //    the local solution
    par_for(nm, [=] HD(long M) {
        const LineCoef c = coef(M);
        const double b = c.b, s = c.s;
        if (s == 0 && b == 1) return;
        const double gam = -b;
        auto G = [&](int q, int k) { return ga[(long(q) * nm + M) * 10 + k]; };
        double* sc = red + M * 10 * (P - 1);                 // block q - 1: D' (4), rhs' (3 x 2)
        for (int q = 1; q < P; ++q) {
            double* B = sc + 10 * (q - 1);
            B[0] = 1; B[1] = -G(q - 1, 3); B[2] = -G(q, 0); B[3] = 1;
            for (int k = 0; k < 3; ++k) { B[4 + 2 * k] = G(q - 1, 5 + 2 * k); B[5 + 2 * k] = G(q, 4 + 2 * k); }
            if (q > 1) {
                // M = Lo_q inv(D'_{q-1}), Lo_q = [[-u_n-1^{q-1}, 0], [0, 0]]; Up_{q-1} = [[0, 0], [0, -v_0^{q-1}]]
                const double* A = B - 10;
                const double det = A[0] * A[3] - A[1] * A[2], l = -G(q - 1, 1);
                const double m0 = l * A[3] / det, m1 = -l * A[1] / det;      // row 0 of M; row 1 is 0
                B[1] -= m1 * (-G(q - 1, 2));
                for (int k = 0; k < 3; ++k) B[4 + 2 * k] -= m0 * A[4 + 2 * k] + m1 * A[5 + 2 * k];
            }
        }
        for (int q = P - 1; q >= 1; --q) {                   // back: I_q into the rhs' slots
            double* B = sc + 10 * (q - 1);
            const double det = B[0] * B[3] - B[1] * B[2];
            for (int k = 0; k < 3; ++k) {
                double r0 = B[4 + 2 * k], r1 = B[5 + 2 * k];
                if (q < P - 1) r1 += G(q, 2) * sc[10 * q + 5 + 2 * k];          // - Up_q I_{q+1}
                B[4 + 2 * k] = (B[3] * r0 - B[1] * r1) / det;
                B[5 + 2 * k] = (B[0] * r1 - B[2] * r0) / det;
            }
        }
        auto I = [&](int q, int k, int cc) { return sc[10 * (q - 1) + 4 + 2 * cc + k]; };   // k 0: R_{q-1}, 1: L_q
        double Rl[3], Lr[3], f[2];
        for (int k = 0; k < 3; ++k) {
            Rl[k] = pr > 0 ? I(pr, 0, k) : 0.0;
            Lr[k] = pr < P - 1 ? I(pr + 1, 1, k) : 0.0;
        }
        double x0[3], xl[3];                                 // global first and last values of A'^-1 (.)
        for (int k = 0; k < 3; ++k) {
            x0[k] = G(0, 4 + 2 * k) + G(0, 2) * I(1, 1, k);
            xl[k] = G(P - 1, 5 + 2 * k) + G(P - 1, 1) * I(P - 1, 0, k);
        }
        const double den = 1 + x0[2] + s * xl[2] / gam;
        for (int k = 0; k < 2; ++k) f[k] = c.cyc ? (x0[k] + s * xl[k] / gam) / den : 0.0;
        double* x = w + 2 * M * nx;
        const double* zl = zc + 2 * M * nx;
        const double* wl = yw + 2 * M * nx;
        for (int i = 0; i < nx; ++i) {
            const double u = zl[2 * i], v = zl[2 * i + 1];
            const double xw = wl[2 * i] + Rl[2] * u + Lr[2] * v;
            for (int k = 0; k < 2; ++k) x[2 * i + k] += Rl[k] * u + Lr[k] * v - f[k] * xw;
        }
    });
}

}  // namespace xlines
