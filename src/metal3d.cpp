// Embedded perfect conductors in Cart3D (docs/specs/metal3d.md, src/metal3d.hpp).
#include "metal3d.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "species.hpp"

Metal3::Metal3(const Domain& d, const Metal3Config& c) {
    if (c.fn.empty()) throw std::runtime_error("metal: the block needs function = <expression of x, y, z> (> 0 inside the metal)");
    if (c.fn.max_stack() > Expr::MAX_STACK) throw std::runtime_error("metal.function: expression too deeply nested");
    text_ = c.fn.text();
    ncode_ = int(c.fn.program().size());
    code_.resize(size_t(ncode_));
    code_.from_host(c.fn.program().data(), 0, size_t(ncode_));
    mask.resize(d);
    fill(d, -d.gw(0), d.n[0] + d.gw(0));
}

// global node index of local index i along axis a, wrapped on a periodic axis
HD inline double node_pos(long g, long N, bool per, double h) {
    if (per) g = ((g % N) + N) % N;
    return double(g) * h;
}

void Metal3::fill(const Domain& d, int i0, int i1) {
    if (i1 <= i0) return;
    const int gy = d.gw(1), gz = d.gw(2), ey = d.ext(1), ez = d.ext(2), nx = i1 - i0;
    const long c0 = d.cell0(0), c1 = d.cell0(1), c2 = d.cell0(2);
    const long N0 = d.N[0], N1 = d.N[1], N2 = d.N[2];
    const bool p0 = d.periodic[0], p1 = d.periodic[1], p2 = d.periodic[2];
    const double h0 = d.h[0], h1 = d.h[1], h2 = d.h[2];
    const FV m = mask.view();
    const Expr::Ins* code = code_.data();
    const int nc = ncode_;
    par_for(long(nx) * ey * ez, [=] HD(long q) {
        const int i = i0 + int(q % nx), j = int((q / nx) % ey) - gy, k = int(q / (long(nx) * ey)) - gz;
        const double v[3] = {node_pos(c0 + i, N0, p0, h0), node_pos(c1 + j, N1, p1, h1), node_pos(c2 + k, N2, p2, h2)};
        m(i, j, k) = expr_eval(code, nc, v) > 0 ? 1.0 : 0.0;
    });
}

void Metal3::shift(const Domain& d, Field& tmp) {
    mask.shift_x(tmp);
    fill(d, d.n[0] - 1, d.n[0] + d.gw(0));       // the last owned column and the right guards: new absolute columns
}

long Metal3::owned_nodes(const Domain& d) const {
    const Metal3View mv = view();
    const int nx = d.n[0], ny = d.n[1];
    return long(par_sum(d.owned(), [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        return mv.M(i, j, k) ? 1.0 : 0.0;
    }));
}

long Metal3::thin_edges(const Domain& d) const {
    const Metal3View mv = view();
    const int nx = d.n[0], ny = d.n[1];
    return long(par_sum(d.owned(), [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        double s = 0;
        if (mv.ey(i, j, k) && !mv.ey(i - 1, j, k) && !mv.ey(i + 1, j, k)) s += 1;
        if (mv.ez(i, j, k) && !mv.ez(i - 1, j, k) && !mv.ez(i + 1, j, k)) s += 1;
        return s;
    }));
}

std::string Metal3::describe() const { return "  metal: function = " + text_ + "\n"; }

// ------------------------------------------------------------------------------------------------ the rules of the step
void metal_fix_pred(const Domain& d, const Metal3View& mv, const FV Exs, const FV Bxs, int halo) {
    for_owned_halo(d, halo, [=] HD(int i, int j, int k) {
        if (!mv.M(i, j, k)) return;
        if (mv.ex_zero(i, j, k)) Exs(i, j, k) = 0;
        if (mv.bx(i, j, k)) Bxs(i, j, k) = 0;
    });
}

void metal_fix_transport(const Domain& d, const Metal3View& mv, const FV nEy, const FV nEz, const FV nBy, const FV nBz) {
    for_owned(d, [=] HD(int i, int j, int k) {
        if (!mv.M(i, j, k)) return;
        if (mv.ey(i, j, k)) {                     // T+_y = Ey + Bz arrives from i - 1, T-_y = Ey - Bz from i + 1
            const bool L = mv.ey(i - 1, j, k), R = mv.ey(i + 1, j, k);
            const double tp = nEy(i, j, k) + nBz(i, j, k), tm = nEy(i, j, k) - nBz(i, j, k);
            nEy(i, j, k) = 0;
            nBz(i, j, k) = L == R ? 0.0 : (!L ? tp : -tm);
        }
        if (mv.ez(i, j, k)) {                     // T+_z = Ez + By arrives from i + 1, T-_z = Ez - By from i - 1
            const bool L = mv.ez(i - 1, j, k), R = mv.ez(i + 1, j, k);
            const double tp = nEz(i, j, k) + nBy(i, j, k), tm = nEz(i, j, k) - nBy(i, j, k);
            nEz(i, j, k) = 0;
            nBy(i, j, k) = L == R ? 0.0 : (!L ? -tm : tp);
        }
    });
}

void metal_fix_corr(const Domain& d, const Metal3View& mv, const FV Ex, const FV Bx) {
    for_owned(d, [=] HD(int i, int j, int k) {
        if (!mv.M(i, j, k)) return;
        if (mv.ex_zero(i, j, k)) Ex(i, j, k) = 0;
        if (mv.bx(i, j, k)) Bx(i, j, k) = 0;
    });
}

void metal_zero_fields(const Domain& d, const Metal3View& mv, const FV Ex, const FV Ey, const FV Ez, const FV Bx, const FV By,
                       const FV Bz) {
    // the owned cells and all guards but the outermost layer (the rules read one neighbour)
    const int w = d.ng - 1;
    for_owned_halo(d, w, [=] HD(int i, int j, int k) {
        if (!mv.M(i, j, k)) return;
        if (mv.ex_zero(i, j, k)) Ex(i, j, k) = 0;
        if (mv.bx(i, j, k)) Bx(i, j, k) = 0;
        if (mv.ey(i, j, k)) { Ey(i, j, k) = 0; Bz(i, j, k) = 0; }
        if (mv.ez(i, j, k)) { Ez(i, j, k) = 0; By(i, j, k) = 0; }
    });
}

// A particle is inside when every node of the box [floor(r - s), ceil(r + s)] (s = (order + 1) / 2 cells, per axis) is
// metal: its charge (rho at i + 1/2, shape <= order) and current then touch only skipped Gauss points.
double metal_absorb(Species& sp, const Domain& d, const Metal3View& mv, int order) {
    const long n = long(sp.size());
    if (n == 0) return 0;
    double* X = sp.ptr<double>(Species::X);
    double* Y = sp.ptr<double>(Species::Y);
    double* Z = sp.ptr<double>(Species::Z);
    double* UX = sp.ptr<double>(Species::UX);
    double* UY = sp.ptr<double>(Species::UY);
    double* UZ = sp.ptr<double>(Species::UZ);
    double* W = sp.ptr<double>(Species::W);
    const double lo0 = d.lo(0), lo1 = d.lo(1), lo2 = d.lo(2), i0 = 1 / d.h[0], i1 = 1 / d.h[1], i2 = 1 / d.h[2];
    const double s = 0.5 * (order + 1), m = sp.m;
    const bool flat = d.flat;
    const int g = d.ng - 1, nx = d.n[0], ny = d.n[1], nz = d.n[2];
    return par_sum(n, [=] HD(long p) {
        if (W[p] == 0) return 0.0;
        const double r[3] = {(X[p] - lo0) * i0, (Y[p] - lo1) * i1, (Z[p] - lo2) * i2};
        int a[3], b[3];
        const int lim[3] = {nx, ny, nz};
        for (int q = 0; q < 3; ++q) {
            a[q] = int(floor(r[q] - s)); b[q] = int(ceil(r[q] + s));
            if (a[q] < -g || b[q] >= lim[q] + g) return 0.0;   // beyond the guards: decided by its owner
        }
        if (flat) { a[2] = 0; b[2] = 0; }
        for (int k = a[2]; k <= b[2]; ++k)
            for (int j = a[1]; j <= b[1]; ++j)
                for (int i = a[0]; i <= b[0]; ++i)
                    if (!mv.M(i, j, k)) return 0.0;
        const double gm = sqrt(1 + UX[p] * UX[p] + UY[p] * UY[p] + UZ[p] * UZ[p]);
        const double e = W[p] * m * (gm - 1);
        W[p] = 0; UX[p] = 0; UY[p] = 0; UZ[p] = 0;
        return e;
    });
}
