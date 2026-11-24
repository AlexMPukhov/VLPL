// Density-conserving (Ruyten-type) radial shapes for RZ (decision 53): the coefficients of the maps r -> xi in
// which the particle shapes are the plain linear (shape 1) or quadratic (shape 2) splines. With the plain
// splines a uniform density deposits 4/3 (shape 1) or 13/8 (shape 2) of its value onto the axis node, whose
// volume is that of the field solver (decision 42); the maps move the near-axis charge outward so that every
// node receives its finite-volume share 2 pi A_j dz (Ruyten, JCP 105, 224, 1993; FBPIC's default).
//   shape 1, cell j: xi = j + u + bt_j u (1 - u), u = (r - r_j) / dn_j; bt_j from one equation per cell
//     (uniform grid: bt_j = 1 / (2 (2j + 1))).
//   shape 2: axis cell xi = u^p (bt2_0 = p), cells j >= 1 as shape 1 with bt2_j, from the coupled equations of
//     all nodes (Newton, banded); started from the first-moment rule (a spline reproduces linear functions, so bt
//     shifts the first moment by bt (r_j dn / 6 + dn^2 / 12) and the excess to carry outward is dr^2 / 12:
//     bt2_j = (dr^2 / 12) / (r_j dn_j / 6 + dn_j^2 / 12), uniform 1 / (2j + 1), the limit far from the axis).
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "rz.hpp"

namespace {
double s2(double x) {
    const double a = std::fabs(x);
    return a < 0.5 ? 0.75 - a * a : (a < 1.5 ? 0.5 * (1.5 - a) * (1.5 - a) : 0.0);
}

// xi of the position u in cell c for the shape-2 coefficients b
double xi2(const std::vector<double>& b, int c, double u) {
    return c == 0 ? std::pow(u, b[0]) : c + u + b[size_t(c)] * u * (1 - u);
}

// u in cell c with xi(c, u) = x (monotone; bisection)
template <class X> double u_of(const X& xi, int c, double x) {
    double lo = 0, hi = 1;
    for (int it = 0; it < 80; ++it) {
        const double m = 0.5 * (lo + hi);
        (xi(c, m) < x ? lo : hi) = m;
    }
    return 0.5 * (lo + hi);
}

// The share of node j (folded across the axis, m = 0) from a uniform density, per 2 pi dz: the integral of
// S2(xi - j) + S2(xi + j) r dr over the cells that reach it, in pieces between the kinks of S2 (16-point
// Gauss-Legendre on each); xi(c, u) the map of cell c (values in [c, c + 1]), cells 0 .. nc - 1.
template <class X> double share(const RadialGrid& g, const X& xi, int nc, int j) {
    static const double gx[8] = {0.0950125098376374, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
                                 0.7554044083550030, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499};
    static const double gw[8] = {0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
                                 0.1246289712555339, 0.0951585116824928, 0.0622535239024133, 0.0271524594117541};
    double tot = 0;
    for (int c = j - 2; c <= j + 1; ++c) {
        if (c < 0 || c >= nc) continue;
        // breakpoints in u: the kinks of S2 at half-integer xi inside the cell
        std::vector<double> br{0.0};
        for (double x : {c + 0.5}) {
            const double u = u_of(xi, c, x);
            if (u > 0 && u < 1) br.push_back(u);
        }
        br.push_back(1.0);
        const double r0 = g.r[size_t(c)], h = g.dn[size_t(c)];
        for (size_t p = 0; p + 1 < br.size(); ++p) {
            const double a = br[p], e = br[p + 1], m = 0.5 * (a + e), hw = 0.5 * (e - a);
            for (int q = 0; q < 16; ++q) {
                const double u = m + (q < 8 ? -gx[q] : gx[q - 8]) * hw, w = gw[q < 8 ? q : q - 8] * hw;
                const double x = xi(c, u), W = s2(x - j) + (j > 0 ? s2(x + j) : 0.0);
                tot += w * W * (r0 + u * h) * h;
            }
        }
    }
    return tot;
}

// Solve A x = y in place for a banded A (lower bandwidth kl, upper ku), stored by rows as the columns
// i - kl .. i + ku + kl (B(i, c) = A[i * W + c - i + kl], W = 2 kl + ku + 1: room for the fill of the partial
// pivoting within the band): O(n (kl + ku)^2) time, O(n (kl + ku)) memory.
void solve_banded(std::vector<double>& A, std::vector<double>& y, int n, int kl, int ku) {
    const int W = 2 * kl + ku + 1, w = ku + kl;
    auto B = [&](int i, int c) -> double& { return A[size_t(i) * W + size_t(c - i + kl)]; };
    for (int k = 0; k < n; ++k) {
        const int last = std::min(n - 1, k + kl), cend = std::min(n - 1, k + w);
        int p = k;
        for (int i = k + 1; i <= last; ++i)
            if (std::fabs(B(i, k)) > std::fabs(B(p, k))) p = i;
        if (p != k) {
            for (int c = k; c <= cend; ++c) std::swap(B(k, c), B(p, c));
            std::swap(y[size_t(k)], y[size_t(p)]);
        }
        for (int i = k + 1; i <= last; ++i) {
            const double f = B(i, k) / B(k, k);
            if (f == 0) continue;
            for (int c = k; c <= cend; ++c) B(i, c) -= f * B(k, c);
            y[size_t(i)] -= f * y[size_t(k)];
        }
    }
    for (int k = n - 1; k >= 0; --k) {
        for (int c = k + 1; c <= std::min(n - 1, k + w); ++c) y[size_t(k)] -= B(k, c) * y[size_t(c)];
        y[size_t(k)] /= B(k, k);
    }
}
}  // namespace

void rz_ruyten_coefficients(RadialGrid& g) {
    const size_t n = g.r.size();
    // shape 1: node j + 1 receives, from a uniform density in cell j, its finite-volume share
    // (r_{j+1}^2 - rh_j^2) / 2 with the weight u + bt u (1 - u): r_j dn (1/2 + bt/6) + dn^2 (1/3 + bt/12) = share
    g.bt.assign(n, 0.0);
    for (size_t j = 0; j + 1 < n; ++j) {
        const double r0 = g.r[j], h = g.dn[j], share = 0.5 * (g.r[j + 1] * g.r[j + 1] - g.rh[j] * g.rh[j]);
        g.bt[j] = (share - r0 * h / 2 - h * h / 3) / (r0 * h / 6 + h * h / 12);
        if (!(std::fabs(g.bt[j]) <= 1)) throw std::runtime_error("rz: radial cells grow too fast for the shape-1 weights");
    }
    // shape 2: the first-moment rule as the start, then Newton on all cells for the nodes 0 .. K - 1 (the last
    // cells beyond the wall's guards keep the rule); node j depends on the cells j - 2 .. j + 1: a banded Jacobian
    std::vector<double> b(n, 0.0);
    const double E = g.dr * g.dr / 12;
    for (size_t j = 1; j < n; ++j) b[j] = E / (g.r[j] * g.dn[j] / 6 + g.dn[j] * g.dn[j] / 12);
    b[0] = 1.0 / 3;
    const int K = int(n) - 3;
    const int kl = 2, ku = 1, W = 2 * kl + ku + 1;
    std::vector<double> R(static_cast<size_t>(K)), J(static_cast<size_t>(K) * W, 0.0);
    const auto x2 = [&](int c, double u) { return xi2(b, c, u); };
    const auto share2 = [&](const RadialGrid& gg, const std::vector<double>&, int j) { return share(gg, x2, int(n), j); };
    double res = 0;
    for (int it = 0; it < 30; ++it) {
        for (int j = 0; j < K; ++j) R[size_t(j)] = share2(g, b, j) - g.A[size_t(j)];
        res = 0;
        for (int j = 0; j < K; ++j) res = std::max(res, std::fabs(R[size_t(j)]) / g.A[size_t(j)]);
        if (res < 1e-14) break;
        std::fill(J.begin(), J.end(), 0.0);
        for (int c = 0; c < K; ++c) {                         // the Jacobian by differences: the nodes c - 1 .. c + 2
            const double d = 1e-7, keep = b[size_t(c)];
            b[size_t(c)] = keep + d;
            for (int j = std::max(0, c - 1); j <= std::min(K - 1, c + 2); ++j)
                J[size_t(j) * W + size_t(c - j + kl)] = (share2(g, b, j) - g.A[size_t(j)] - R[size_t(j)]) / d;
            b[size_t(c)] = keep;
        }
        for (double& v : R) v = -v;
        solve_banded(J, R, K, kl, ku);
        for (int c = 0; c < K; ++c) b[size_t(c)] += R[size_t(c)];
    }
    if (!(res < 1e-12) || !(b[0] > 0 && b[0] <= 1))
        throw std::runtime_error("rz: the shape-2 axis weights did not converge (decision 53)");
    for (size_t j = 1; j < n; ++j)
        if (!(std::fabs(b[j]) <= 1)) throw std::runtime_error("rz: radial cells grow too fast for the shape-2 weights");
    g.bt2 = b;
}

// The C1 shape-2 map of the implicit mover (rz.hpp, rz_xi shape 4; decision 82; research/o6_mover/c1map.py): node slopes
// dxi/dr s_0 = s0 / dr, s_1 = s1 2 / (dn_0 + dn_1) and s_j = 2 / (dn_{j-1} + dn_j), the bumps ca_j from the shares of the nodes 0 .. K - 1 (Newton,
// banded as above); started from the first-moment rule (the bump u^2 (1 - u)^2 has the mean 1/30 against 1/6 of
// u (1 - u): ca_j = 5 bt2_j rule) and, in the cells 0 and 1, from the uniform-grid fits of c1scan.py.
void rz_c1_coefficients(RadialGrid& g, double s0, double s1) {
    const size_t n = g.r.size();
    std::vector<double> s(n), cl(n), cr(n), a(n, 0.0);
    for (size_t j = 0; j < n; ++j) s[j] = j == 0 ? s0 / g.dr : (j == 1 ? s1 : 1.0) * 2 / (g.dn[j - 1] + g.dn[j]);
    for (size_t j = 0; j < n; ++j) {
        cl[j] = s[j] * g.dn[j] - 1;
        cr[j] = (j + 1 < n ? s[j + 1] : 1 / g.dn[j]) * g.dn[j] - 1;
    }
    const double E = g.dr * g.dr / 12;
    for (size_t j = 1; j < n; ++j) a[j] = 5 * E / (g.r[j] * g.dn[j] / 6 + g.dn[j] * g.dn[j] / 12);
    a[0] = 1.87 - 2.1 * (s0 - 3) - 2.2 * (1 - s1);
    a[1] = 1.44 + 2.2 * (1 - s1);
    const auto xi = [&](int c, double u) {
        const size_t k = size_t(c);
        return c + u + cl[k] * u * (1 - u) * (1 - u) - cr[k] * u * u * (1 - u) + a[k] * u * u * (1 - u) * (1 - u);
    };
    const int K = int(n) - 3;
    const int kl = 2, ku = 1, W = 2 * kl + ku + 1;
    std::vector<double> R(static_cast<size_t>(K)), J(static_cast<size_t>(K) * W, 0.0);
    double res = 0;
    for (int it = 0; it < 40; ++it) {
        for (int j = 0; j < K; ++j) R[size_t(j)] = share(g, xi, int(n), j) - g.A[size_t(j)];
        res = 0;
        for (int j = 0; j < K; ++j) res = std::max(res, std::fabs(R[size_t(j)]) / g.A[size_t(j)]);
        if (res < 1e-14) break;
        std::fill(J.begin(), J.end(), 0.0);
        for (int c = 0; c < K; ++c) {
            const double d = 1e-7, keep = a[size_t(c)];
            a[size_t(c)] = keep + d;
            for (int j = std::max(0, c - 1); j <= std::min(K - 1, c + 2); ++j)
                J[size_t(j) * W + size_t(c - j + kl)] = (share(g, xi, int(n), j) - g.A[size_t(j)] - R[size_t(j)]) / d;
            a[size_t(c)] = keep;
        }
        for (double& v : R) v = -v;
        solve_banded(J, R, K, kl, ku);
        for (int c = 0; c < K; ++c) a[size_t(c)] += R[size_t(c)];
    }
    if (!(res < 1e-12)) throw std::runtime_error("rz: the C1 radial map did not converge (numerics.mover = implicit, decision 82)");
    for (size_t j = 0; j + 1 < n; ++j)                         // monotone: dxi/du > 0 in every cell
        for (int q = 0; q <= 64; ++q) {
            const double u = q / 64.0;
            const double d = 1 + cl[j] * (1 - u) * (1 - 3 * u) - cr[j] * u * (2 - 3 * u) + a[j] * 2 * u * (1 - u) * (1 - 2 * u);
            if (!(d > 0))
                throw std::runtime_error("rz: the C1 radial map is not monotone in cell " + std::to_string(j) +
                                         " (the radial cells grow too fast; decision 82)");
        }
    g.ca = a; g.cl = cl; g.cr = cr; g.c1_s0 = s0; g.c1_s1 = s1;
    g.upload();
}
