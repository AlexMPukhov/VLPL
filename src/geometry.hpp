// Geometry (decision 10, docs/specs/architecture.md section 2): everything that ties particles to the
// grid. Particles are Cartesian in every geometry (axis 0 = transport axis); a geometry provides the
// field views of the kernels, the gather of E^n, B^n, the Esirkepov current deposit and the charge
// deposit, each generated from the staggering descriptor St (stagger.hpp) and the shape order O.
// Present: Cart3D. RZModes (azimuthal modes) follows in Phase 2 with the same interface.
#pragma once
#include <stdexcept>
#include <type_traits>
#include "fields.hpp"
#include "shape.hpp"
#include "stagger.hpp"

namespace cart3d {

struct FieldViews { FV ex, ey, ez, bx, by, bz, jx, jy, jz; };

// ---------------------------------------------------------------- gather
// The weights are computed once per axis and stagger offset (0 or 1/2 cell: at most 6 shape
// evaluations, RipCC 5), and the cells are walked with pointer strides. The same arithmetic in the same
// order as one shape evaluation per component and axis, so the result is bit-identical.
template <int O> struct GatherWeights {
    int i0[3][2];
    double w[3][2][Shape<O>::W];
};
constexpr int half_index(double o) { return o == 0 ? 0 : 1; }
constexpr bool half_or_zero(double o) { return o == 0 || o == 0.5; }

template <int O, int hx, int hy, int hz>
HD inline double gather_w(const FV& f, const GatherWeights<O>& g) {
    using S = Shape<O>;
    const double* p = &f(g.i0[0][hx], g.i0[1][hy], g.i0[2][hz]);
    const long sx = f.sx, sxy = long(f.sx) * f.sy;
    double v = 0;
    for (int c = 0; c < S::W; ++c)
            for (int b = 0; b < S::W; ++b) {
            const double wyz = g.w[1][hy][b] * g.w[2][hz][c];
            const double* q = p + b * sx + c * sxy;
                    for (int a = 0; a < S::W; ++a) v += g.w[0][hx][a] * wyz * q[a];
        }
    return v;
}
template <class St, int O, int C, bool IsE>
HD inline double gather_c(const FV& f, const GatherWeights<O>& g) {
    constexpr Loc l = IsE ? St::E[C] : St::B[C];
    static_assert(half_or_zero(l.o[0]) && half_or_zero(l.o[1]) && half_or_zero(l.o[2]), "stagger offsets 0 or 1/2");
    return gather_w<O, half_index(l.o[0]), half_index(l.o[1]), half_index(l.o[2])>(f, g);
}

// E^n, B^n at r, at the positions declared by the staggering descriptor St.
template <class St, int O>
HD inline void fields_at(const FieldViews& f, const double r[3], const double ih[3], double E[3], double B[3]) {
    GatherWeights<O> g;
    for (int a = 0; a < 3; ++a) {
        Shape<O>::eval(r[a] * ih[a] - 0.0, g.i0[a][0], g.w[a][0]);
        Shape<O>::eval(r[a] * ih[a] - 0.5, g.i0[a][1], g.w[a][1]);
    }
    E[0] = gather_c<St, O, 0, true>(f.ex, g);
    E[1] = gather_c<St, O, 1, true>(f.ey, g);
    E[2] = gather_c<St, O, 2, true>(f.ez, g);
    B[0] = gather_c<St, O, 0, false>(f.bx, g);
    B[1] = gather_c<St, O, 1, false>(f.by, g);
    B[2] = gather_c<St, O, 2, false>(f.bz, g);
}

// ---------------------------------------------------------------- Esirkepov deposition
// Charge grid: node (c, j, k) at the rho offset of St (RIP-CC: ((c+1/2) hx, j hy, k hz)). J_a between
// nodes c and c+1 along a is stored at J_a index c + current_shift<St>(a) (RIP-CC: Jx at c+1, Jy, Jz at c).
// The additions go through add(p, v): AddTo<false> plain when the caller guarantees exclusive access
// (coloured tiles), AddTo<true> atomic; the box deposit adds fixed-point integers (decision 26).
// Zero contributions are skipped only where an add is expensive (skip_zero: atomics, fixed point); a
// plain add of 0 is cheaper than the branch.
template <bool A> struct AddTo {
    static constexpr bool skip_zero = A;
    HD void operator()(double* p, double v) const {
        if (A) atomic_add(p, v);
        else *p += v;
    }
};

template <class St, int O, class Add>
HD inline void esirkepov(const FieldViews& f, const double r0[3], const double r1[3], const double ih[3],
                         const double coef[3], const int off[3], const Add& add) {
    using S = Shape<O>;
    constexpr int L = S::W + 1;
    constexpr double sh0 = St::rho.o[0], sh1 = St::rho.o[1], sh2 = St::rho.o[2];
    constexpr int cx = current_shift<St>(0), cy = current_shift<St>(1), cz = current_shift<St>(2);
    const double sh[3] = {sh0, sh1, sh2};
    double S0[3][L], DS[3][L];
    int s[3];
    for (int a = 0; a < 3; ++a) {
        int i0, i1;
        double w0[S::W], w1[S::W];
        S::eval(r0[a] * ih[a] - sh[a], i0, w0);
        S::eval(r1[a] * ih[a] - sh[a], i1, w1);
        s[a] = i0 < i1 ? i0 : i1;
        // contract: a segment moves at most one cell per axis (esirkepov_segments splits longer moves), so
        // i0 - s and i1 - s are 0 or 1: selects at fixed indices instead of stores at a computed index
        // keep S0, S1 in registers (the same values)
        const bool d0 = i0 > s[a], d1 = i1 > s[a];
        for (int m = 0; m < L; ++m) {
            const double lo0 = m < S::W ? w0[m] : 0.0, hi0 = m >= 1 ? w0[m - 1] : 0.0;
            const double lo1 = m < S::W ? w1[m] : 0.0, hi1 = m >= 1 ? w1[m - 1] : 0.0;
            S0[a][m] = d0 ? hi0 : lo0;
            DS[a][m] = (d1 ? hi1 : lo1) - S0[a][m];
        }
        s[a] -= off[a];                             // write into an array whose index 0 is cell off[a]
    }
    constexpr double third = 1.0 / 3.0;
    for (int l = 0; l < L; ++l)                     // Jx: cumulative along x
        for (int m = 0; m < L; ++m) {
            const double wyz = S0[1][m] * S0[2][l] + 0.5 * DS[1][m] * S0[2][l] + 0.5 * S0[1][m] * DS[2][l] +
                               third * DS[1][m] * DS[2][l];
            if (wyz == 0.0) continue;
            double acc = 0;
            double* p = &f.jx(s[0] + cx, s[1] + m, s[2] + l);
            for (int n = 0; n < L - 1; ++n) {
                acc -= coef[0] * DS[0][n] * wyz;
                if (!Add::skip_zero || acc != 0.0) add(p + n, acc);
            }
        }
    for (int l = 0; l < L; ++l)                     // Jy: cumulative along y
        for (int n = 0; n < L; ++n) {
            const double wxz = S0[0][n] * S0[2][l] + 0.5 * DS[0][n] * S0[2][l] + 0.5 * S0[0][n] * DS[2][l] +
                               third * DS[0][n] * DS[2][l];
            if (wxz == 0.0) continue;
            double acc = 0;
            double* p = &f.jy(s[0] + n, s[1] + cy, s[2] + l);
            const long sy = f.jy.sx;
            for (int m = 0; m < L - 1; ++m) {
                acc -= coef[1] * DS[1][m] * wxz;
                if (!Add::skip_zero || acc != 0.0) add(p + m * sy, acc);
            }
        }
    for (int m = 0; m < L; ++m)                     // Jz: cumulative along z
        for (int n = 0; n < L; ++n) {
            const double wxy = S0[0][n] * S0[1][m] + 0.5 * DS[0][n] * S0[1][m] + 0.5 * S0[0][n] * DS[1][m] +
                               third * DS[0][n] * DS[1][m];
            if (wxy == 0.0) continue;
            double acc = 0;
            double* p = &f.jz(s[0] + n, s[1] + m, s[2] + cz);
            const long sz = long(f.jz.sx) * f.jz.sy;
            for (int l = 0; l < L - 1; ++l) {
                acc -= coef[2] * DS[2][l] * wxy;
                if (!Add::skip_zero || acc != 0.0) add(p + l * sz, acc);
            }
        }
}


// Charge density of one particle at local position r on the charge grid (rho offset of St).
// Moves of more than one cell along an axis (an implicit solver allows c dt > h, decision 28) are split
// into n equal straight segments of at most one cell each, deposited one after the other: each segment
// conserves charge exactly, so their sum does. n = 1 (every explicit RIP-CC step: dt <= h) is the plain
// esirkepov call, bit-identical.
template <class St, int O, class Add>
HD inline void esirkepov_segments(const FieldViews& f, const double r0[3], const double r1[3], const double ih[3],
                                  const double coef[3], const int off[3], const Add& add) {
    double m = 0;
    for (int a = 0; a < 3; ++a) {
        const double c = fabs(r1[a] - r0[a]) * ih[a];
        m = c > m ? c : m;
    }
    if (m <= 1.0) { esirkepov<St, O>(f, r0, r1, ih, coef, off, add); return; }
    const int n = int(ceil(m * (1.0 + 1e-12)));        // segments strictly shorter than a cell despite rounding
    double a0[3] = {r0[0], r0[1], r0[2]};
    for (int k = 1; k <= n; ++k) {
        double a1[3];
        for (int a = 0; a < 3; ++a) a1[a] = k == n ? r1[a] : r0[a] + (r1[a] - r0[a]) * (double(k) / n);
        esirkepov<St, O>(f, a0, a1, ih, coef, off, add);
        for (int a = 0; a < 3; ++a) a0[a] = a1[a];
    }
}

template <class St, int O>
HD inline void rho_one(const FV& R, const double r[3], const double ih[3], double qw) {
    using S = Shape<O>;
    constexpr double sh0 = St::rho.o[0], sh1 = St::rho.o[1], sh2 = St::rho.o[2];
    int i0, j0, k0;
    double wx[S::W], wy[S::W], wz[S::W];
    S::eval(r[0] * ih[0] - sh0, i0, wx);
    S::eval(r[1] * ih[1] - sh1, j0, wy);
    S::eval(r[2] * ih[2] - sh2, k0, wz);
    for (int c = 0; c < S::W; ++c)
        for (int b = 0; b < S::W; ++b)
            for (int a = 0; a < S::W; ++a) atomic_add(&R(i0 + a, j0 + b, k0 + c), qw * wx[a] * wy[b] * wz[c]);
}

}  // namespace cart3d

// 3D Cartesian grid (x, y, z); r is the particle position relative to the local box (x - lo).
struct Cart3D {
    static constexpr int grid_dims = 3;
    using FieldViews = cart3d::FieldViews;
    static FieldViews views(const Fields& f) {
        return FieldViews{f.ex.view(), f.ey.view(), f.ez.view(), f.bx.view(), f.by.view(), f.bz.view(),
                          f.jx.view(), f.jy.view(), f.jz.view()};
    }
    template <class St, int O>
    HD static void gather(const FieldViews& f, const double r[3], const double ih[3], double E[3], double B[3]) {
        cart3d::fields_at<St, O>(f, r, ih, E, B);
    }
    // Esirkepov current of the move r0 -> r1; coef[a] = q w / (dt h_b h_c). A = false: plain additions
    // (the caller guarantees that no other thread writes the same cells, docs/specs/tile_deposit.md).
    template <class St, int O, bool A = true>
    HD static void deposit(const FieldViews& f, const double r0[3], const double r1[3], const double ih[3],
                           const double coef[3]) {
        const int off[3] = {0, 0, 0};
        cart3d::esirkepov_segments<St, O>(f, r0, r1, ih, coef, off, cart3d::AddTo<A>{});
    }
    // The same into arrays (f.jx, f.jy, f.jz) whose index 0 is the cell off[a] of the local box, through
    // add(p, v) (cart3d::AddTo<A>, or the fixed-point adder of the box deposit).
    template <class St, int O, class Add>
    HD static void deposit_shifted(const FieldViews& f, const double r0[3], const double r1[3], const double ih[3],
                                   const double coef[3], const int off[3], const Add& add) {
        cart3d::esirkepov_segments<St, O>(f, r0, r1, ih, coef, off, add);
    }
    // charge density q w / dV of one particle
    template <class St, int O>
    HD static void deposit_rho(const FV& rho, const double r[3], const double ih[3], double qw) {
        cart3d::rho_one<St, O>(rho, r, ih, qw);
    }
};

// ---------------------------------------------------------------- Cart2D (docs/specs/cart2d.md)
// grid.geometry = cartesian2d: the Cart3D fields on one z cell with the z stride 0 (Domain::flat), so a field is one
// (x, y) plane and f(i, j, k) does not depend on k. The particles keep z and uz (z periodic over the slab h_z, only the
// displacement matters); their shape has no extent in z: the z factor of the Cart3D kernels is 1, and the Esirkepov
// current J_z = coef_z (dz / h_z) W_xy is the sum over z of Cart3D's J_z of the same move (the first moment of the
// z shape moves by dz / h_z). Equal to the Cart3D kernels on z-uniform data up to the order of the additions.
namespace cart2d {
using cart3d::FieldViews;

template <int O, int hx, int hy>
HD inline double gather_w(const FV& f, const cart3d::GatherWeights<O>& g) {
    using S = Shape<O>;
    const double* p = &f(g.i0[0][hx], g.i0[1][hy], 0);
    const long sx = f.sx;
    double v = 0;
    for (int b = 0; b < S::W; ++b) {
        const double wy = g.w[1][hy][b];
        const double* q = p + b * sx;
        for (int a = 0; a < S::W; ++a) v += g.w[0][hx][a] * wy * q[a];
    }
    return v;
}
template <class St, int O, int C, bool IsE>
HD inline double gather_c(const FV& f, const cart3d::GatherWeights<O>& g) {
    constexpr Loc l = IsE ? St::E[C] : St::B[C];
    return cart2d::gather_w<O, cart3d::half_index(l.o[0]), cart3d::half_index(l.o[1])>(f, g);
}
template <class St, int O>
HD inline void fields_at(const FieldViews& f, const double r[3], const double ih[3], double E[3], double B[3]) {
    cart3d::GatherWeights<O> g;
    for (int a = 0; a < 2; ++a) {
        Shape<O>::eval(r[a] * ih[a] - 0.0, g.i0[a][0], g.w[a][0]);
        Shape<O>::eval(r[a] * ih[a] - 0.5, g.i0[a][1], g.w[a][1]);
    }
    E[0] = cart2d::gather_c<St, O, 0, true>(f.ex, g);
    E[1] = cart2d::gather_c<St, O, 1, true>(f.ey, g);
    E[2] = cart2d::gather_c<St, O, 2, true>(f.ez, g);
    B[0] = cart2d::gather_c<St, O, 0, false>(f.bx, g);
    B[1] = cart2d::gather_c<St, O, 1, false>(f.by, g);
    B[2] = cart2d::gather_c<St, O, 2, false>(f.bz, g);
}

// Esirkepov in (x, y), J_z from the z displacement; off[2] is ignored (one plane).
template <class St, int O, class Add>
HD inline void esirkepov(const FieldViews& f, const double r0[3], const double r1[3], const double ih[3],
                         const double coef[3], const int off[3], const Add& add) {
    using S = Shape<O>;
    constexpr int L = S::W + 1;
    constexpr double sh0 = St::rho.o[0], sh1 = St::rho.o[1];
    constexpr int cx = current_shift<St>(0), cy = current_shift<St>(1);
    const double sh[2] = {sh0, sh1};
    double S0[2][L], DS[2][L];
    int s[2];
    for (int a = 0; a < 2; ++a) {
        int i0, i1;
        double w0[S::W], w1[S::W];
        S::eval(r0[a] * ih[a] - sh[a], i0, w0);
        S::eval(r1[a] * ih[a] - sh[a], i1, w1);
        s[a] = i0 < i1 ? i0 : i1;
        const bool d0 = i0 > s[a], d1 = i1 > s[a];
        for (int m = 0; m < L; ++m) {
            const double lo0 = m < S::W ? w0[m] : 0.0, hi0 = m >= 1 ? w0[m - 1] : 0.0;
            const double lo1 = m < S::W ? w1[m] : 0.0, hi1 = m >= 1 ? w1[m - 1] : 0.0;
            S0[a][m] = d0 ? hi0 : lo0;
            DS[a][m] = (d1 ? hi1 : lo1) - S0[a][m];
        }
        s[a] -= off[a];
    }
    constexpr double third = 1.0 / 3.0;
    for (int m = 0; m < L; ++m) {                   // Jx: cumulative along x
        const double wy = S0[1][m] + 0.5 * DS[1][m];
        if (wy == 0.0) continue;
        double acc = 0;
        double* p = &f.jx(s[0] + cx, s[1] + m, 0);
        for (int n = 0; n < L - 1; ++n) {
            acc -= coef[0] * DS[0][n] * wy;
            if (!Add::skip_zero || acc != 0.0) add(p + n, acc);
        }
    }
    const long sy = f.jy.sx;
    for (int n = 0; n < L; ++n) {                   // Jy: cumulative along y
        const double wx = S0[0][n] + 0.5 * DS[0][n];
        if (wx == 0.0) continue;
        double acc = 0;
        double* p = &f.jy(s[0] + n, s[1] + cy, 0);
        for (int m = 0; m < L - 1; ++m) {
            acc -= coef[1] * DS[1][m] * wx;
            if (!Add::skip_zero || acc != 0.0) add(p + m * sy, acc);
        }
    }
    const double cz = coef[2] * (r1[2] - r0[2]) * ih[2];
    if (cz == 0.0 && Add::skip_zero) return;
    for (int m = 0; m < L; ++m) {                   // Jz = q w v_z W_xy
        double* p = &f.jz(s[0], s[1] + m, 0);
        for (int n = 0; n < L; ++n) {
            const double v = cz * (S0[0][n] * S0[1][m] + 0.5 * DS[0][n] * S0[1][m] + 0.5 * S0[0][n] * DS[1][m] +
                                   third * DS[0][n] * DS[1][m]);
            if (!Add::skip_zero || v != 0.0) add(p + n, v);
        }
    }
}

// Moves of more than one cell along x or y are split as in Cart3D (cart3d::esirkepov_segments); z does not count.
template <class St, int O, class Add>
HD inline void esirkepov_segments(const FieldViews& f, const double r0[3], const double r1[3], const double ih[3],
                                  const double coef[3], const int off[3], const Add& add) {
    double m = 0;
    for (int a = 0; a < 2; ++a) {
        const double c = fabs(r1[a] - r0[a]) * ih[a];
        m = c > m ? c : m;
    }
    if (m <= 1.0) { cart2d::esirkepov<St, O>(f, r0, r1, ih, coef, off, add); return; }
    const int n = int(ceil(m * (1.0 + 1e-12)));
    double a0[3] = {r0[0], r0[1], r0[2]};
    for (int k = 1; k <= n; ++k) {
        double a1[3];
        for (int a = 0; a < 3; ++a) a1[a] = k == n ? r1[a] : r0[a] + (r1[a] - r0[a]) * (double(k) / n);
        cart2d::esirkepov<St, O>(f, a0, a1, ih, coef, off, add);
        for (int a = 0; a < 3; ++a) a0[a] = a1[a];
    }
}

template <class St, int O>
HD inline void rho_one(const FV& R, const double r[3], const double ih[3], double qw) {
    using S = Shape<O>;
    constexpr double sh0 = St::rho.o[0], sh1 = St::rho.o[1];
    int i0, j0;
    double wx[S::W], wy[S::W];
    S::eval(r[0] * ih[0] - sh0, i0, wx);
    S::eval(r[1] * ih[1] - sh1, j0, wy);
    for (int b = 0; b < S::W; ++b)
        for (int a = 0; a < S::W; ++a) atomic_add(&R(i0 + a, j0 + b, 0), qw * wx[a] * wy[b]);
}
}  // namespace cart2d

// 2D Cartesian (x, y) on the flat Cart3D grid: the same interface as Cart3D.
struct Cart2D {
    static constexpr int grid_dims = 2;
    using FieldViews = cart3d::FieldViews;
    static FieldViews views(const Fields& f) { return Cart3D::views(f); }
    template <class St, int O>
    HD static void gather(const FieldViews& f, const double r[3], const double ih[3], double E[3], double B[3]) {
        cart2d::fields_at<St, O>(f, r, ih, E, B);
    }
    template <class St, int O, bool A = true>
    HD static void deposit(const FieldViews& f, const double r0[3], const double r1[3], const double ih[3],
                           const double coef[3]) {
        const int off[3] = {0, 0, 0};
        cart2d::esirkepov_segments<St, O>(f, r0, r1, ih, coef, off, cart3d::AddTo<A>{});
    }
    template <class St, int O, class Add>
    HD static void deposit_shifted(const FieldViews& f, const double r0[3], const double r1[3], const double ih[3],
                                   const double coef[3], const int off[3], const Add& add) {
        cart2d::esirkepov_segments<St, O>(f, r0, r1, ih, coef, off, add);
    }
    template <class St, int O>
    HD static void deposit_rho(const FV& rho, const double r[3], const double ih[3], double qw) {
        cart2d::rho_one<St, O>(rho, r, ih, qw);
    }
};

// Calls f(G{}, St{}, std::integral_constant<int, O>{}) with the geometry (Cart2D on a flat domain, else Cart3D), the
// staggering descriptor of the run's solver (Domain::solver) and the shape order. Cart2D: RIP-CC's descriptor only
// (rip_cc, irip; docs/specs/cart2d.md).
template <class F> auto with_stagger(const Domain& d, int order, F f) {
    using O1 = std::integral_constant<int, 1>;
    using O2 = std::integral_constant<int, 2>;
    using O3 = std::integral_constant<int, 3>;
    if (order < 1 || order > 3) throw std::runtime_error("shape order must be 1, 2 or 3");
    if (d.flat) {
        if (d.solver == Solver::Yee) throw std::runtime_error("grid.geometry = cartesian2d: numerics.solver = rip_cc or irip");
        return order == 1 ? f(Cart2D{}, RipCC{}, O1{}) : (order == 2 ? f(Cart2D{}, RipCC{}, O2{}) : f(Cart2D{}, RipCC{}, O3{}));
    }
    if (d.solver == Solver::Yee)
        return order == 1 ? f(Cart3D{}, Yee{}, O1{}) : (order == 2 ? f(Cart3D{}, Yee{}, O2{}) : f(Cart3D{}, Yee{}, O3{}));
    return order == 1 ? f(Cart3D{}, RipCC{}, O1{}) : (order == 2 ? f(Cart3D{}, RipCC{}, O2{}) : f(Cart3D{}, RipCC{}, O3{}));
}
