// O6: the energy-conserving implicit mover with the iRIP box closure (implicit_mover.hpp, docs/specs/implicit_mover.md).
#include "env.hpp"
#include "implicit_mover.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "geometry.hpp"
#include "rz.hpp"
#include "sim.hpp"

namespace {

// ---------------------------------------------------------------- the work of Et along a path (transpose of Esirkepov)
// S_a = sum over the cells of the deposit's a-current of (its Esirkepov weight with coef = 1) x Et_a: the same loops and
// the same arithmetic of the weights as cart3d::esirkepov / cart2d::esirkepov (geometry.hpp). The work of Et on the
// current of a particle of charge q w is q w h_a S_a per axis.
template <int O>
HD inline void work3(const cart3d::FieldViews& et, const double r0[3], const double r1[3], const double ih[3], double S[3]) {
    using Sh = Shape<O>;
    constexpr int L = Sh::W + 1;
    constexpr int cx = current_shift<RipCC>(0), cy = current_shift<RipCC>(1), cz = current_shift<RipCC>(2);
    const double sh[3] = {RipCC::rho.o[0], RipCC::rho.o[1], RipCC::rho.o[2]};
    double S0[3][L], DS[3][L];
    int s[3];
    for (int a = 0; a < 3; ++a) {
        int i0, i1;
        double w0[Sh::W], w1[Sh::W];
        Sh::eval(r0[a] * ih[a] - sh[a], i0, w0);
        Sh::eval(r1[a] * ih[a] - sh[a], i1, w1);
        s[a] = i0 < i1 ? i0 : i1;
        const bool d0 = i0 > s[a], d1 = i1 > s[a];
        for (int m = 0; m < L; ++m) {
            const double lo0 = m < Sh::W ? w0[m] : 0.0, hi0 = m >= 1 ? w0[m - 1] : 0.0;
            const double lo1 = m < Sh::W ? w1[m] : 0.0, hi1 = m >= 1 ? w1[m - 1] : 0.0;
            S0[a][m] = d0 ? hi0 : lo0;
            DS[a][m] = (d1 ? hi1 : lo1) - S0[a][m];
        }
    }
    constexpr double third = 1.0 / 3.0;
    for (int l = 0; l < L; ++l)
        for (int m = 0; m < L; ++m) {
            const double wyz = S0[1][m] * S0[2][l] + 0.5 * DS[1][m] * S0[2][l] + 0.5 * S0[1][m] * DS[2][l] + third * DS[1][m] * DS[2][l];
            if (wyz == 0.0) continue;
            double acc = 0;
            const double* p = &et.jx(s[0] + cx, s[1] + m, s[2] + l);
            for (int n = 0; n < L - 1; ++n) { acc -= DS[0][n] * wyz; S[0] += acc * p[n]; }
        }
    const long sy = et.jy.sx;
    for (int l = 0; l < L; ++l)
        for (int n = 0; n < L; ++n) {
            const double wxz = S0[0][n] * S0[2][l] + 0.5 * DS[0][n] * S0[2][l] + 0.5 * S0[0][n] * DS[2][l] + third * DS[0][n] * DS[2][l];
            if (wxz == 0.0) continue;
            double acc = 0;
            const double* p = &et.jy(s[0] + n, s[1] + cy, s[2] + l);
            for (int m = 0; m < L - 1; ++m) { acc -= DS[1][m] * wxz; S[1] += acc * p[m * sy]; }
        }
    const long sz = long(et.jz.sx) * et.jz.sy;
    for (int m = 0; m < L; ++m)
        for (int n = 0; n < L; ++n) {
            const double wxy = S0[0][n] * S0[1][m] + 0.5 * DS[0][n] * S0[1][m] + 0.5 * S0[0][n] * DS[1][m] + third * DS[0][n] * DS[1][m];
            if (wxy == 0.0) continue;
            double acc = 0;
            const double* p = &et.jz(s[0] + n, s[1] + m, s[2] + cz);
            for (int l = 0; l < L - 1; ++l) { acc -= DS[2][l] * wxy; S[2] += acc * p[l * sz]; }
        }
}

template <int O>
HD inline void work2(const cart3d::FieldViews& et, const double r0[3], const double r1[3], const double ih[3], double S[3]) {
    using Sh = Shape<O>;
    constexpr int L = Sh::W + 1;
    constexpr int cx = current_shift<RipCC>(0), cy = current_shift<RipCC>(1);
    const double sh[2] = {RipCC::rho.o[0], RipCC::rho.o[1]};
    double S0[2][L], DS[2][L];
    int s[2];
    for (int a = 0; a < 2; ++a) {
        int i0, i1;
        double w0[Sh::W], w1[Sh::W];
        Sh::eval(r0[a] * ih[a] - sh[a], i0, w0);
        Sh::eval(r1[a] * ih[a] - sh[a], i1, w1);
        s[a] = i0 < i1 ? i0 : i1;
        const bool d0 = i0 > s[a], d1 = i1 > s[a];
        for (int m = 0; m < L; ++m) {
            const double lo0 = m < Sh::W ? w0[m] : 0.0, hi0 = m >= 1 ? w0[m - 1] : 0.0;
            const double lo1 = m < Sh::W ? w1[m] : 0.0, hi1 = m >= 1 ? w1[m - 1] : 0.0;
            S0[a][m] = d0 ? hi0 : lo0;
            DS[a][m] = (d1 ? hi1 : lo1) - S0[a][m];
        }
    }
    constexpr double third = 1.0 / 3.0;
    for (int m = 0; m < L; ++m) {
        const double wy = S0[1][m] + 0.5 * DS[1][m];
        if (wy == 0.0) continue;
        double acc = 0;
        const double* p = &et.jx(s[0] + cx, s[1] + m, 0);
        for (int n = 0; n < L - 1; ++n) { acc -= DS[0][n] * wy; S[0] += acc * p[n]; }
    }
    const long sy = et.jy.sx;
    for (int n = 0; n < L; ++n) {
        const double wx = S0[0][n] + 0.5 * DS[0][n];
        if (wx == 0.0) continue;
        double acc = 0;
        const double* p = &et.jy(s[0] + n, s[1] + cy, 0);
        for (int m = 0; m < L - 1; ++m) { acc -= DS[1][m] * wx; S[1] += acc * p[m * sy]; }
    }
    const double cz = (r1[2] - r0[2]) * ih[2];
    for (int m = 0; m < L; ++m) {
        const double* p = &et.jz(s[0], s[1] + m, 0);
        for (int n = 0; n < L; ++n)
            S[2] += cz * (S0[0][n] * S0[1][m] + 0.5 * DS[0][n] * S0[1][m] + 0.5 * S0[0][n] * DS[1][m] + third * DS[0][n] * DS[1][m]) * p[n];
    }
}

// The path in n equal straight segments, n = ceil(c dt / h_min) fixed for the run (PassC::nseg): |vbar| < c, so every
// segment moves less than one cell (the contract of the Esirkepov kernels). A count that follows the displacement, as
// esirkepov_segments does, makes the current a discontinuous function of the iterate where a move crosses one cell, and
// the nonlinear solve stalls there (residual 1e-3 with a hot plasma at h_perp = c dt / 2).
template <class G, int O>
HD inline void work_path(const cart3d::FieldViews& et, const double r0[3], const double r1[3], const double ih[3], int n,
                         double S[3]) {
    S[0] = S[1] = S[2] = 0;
    double a0[3] = {r0[0], r0[1], r0[2]};
    for (int k = 1; k <= n; ++k) {
        double a1[3];
        for (int a = 0; a < 3; ++a) a1[a] = k == n ? r1[a] : r0[a] + (r1[a] - r0[a]) * (double(k) / n);
        if (G::grid_dims == 3) work3<O>(et, a0, a1, ih, S); else work2<O>(et, a0, a1, ih, S);
        for (int a = 0; a < 3; ++a) a0[a] = a1[a];
    }
}
template <class G, int O>
HD inline void deposit_path(const cart3d::FieldViews& f, const double r0[3], const double r1[3], const double ih[3], int n,
                            const double coef[3]) {
    const int off[3] = {0, 0, 0};
    double a0[3] = {r0[0], r0[1], r0[2]};
    for (int k = 1; k <= n; ++k) {
        double a1[3];
        for (int a = 0; a < 3; ++a) a1[a] = k == n ? r1[a] : r0[a] + (r1[a] - r0[a]) * (double(k) / n);
        if (G::grid_dims == 3) cart3d::esirkepov<RipCC, O>(f, a0, a1, ih, coef, off, cart3d::AddTo<true>{});
        else cart2d::esirkepov<RipCC, O>(f, a0, a1, ih, coef, off, cart3d::AddTo<true>{});
        for (int a = 0; a < 3; ++a) a0[a] = a1[a];
    }
}

// Et at r with the plain shape (only for a displacement that vanishes along an axis: the force there does no work).
template <class G, int O>
HD inline void et_plain(const cart3d::FieldViews& et, const double r[3], const double ih[3], double E[3]) {
    cart3d::GatherWeights<O> g;
    for (int a = 0; a < 3; ++a) {
        Shape<O>::eval(r[a] * ih[a] - 0.0, g.i0[a][0], g.w[a][0]);
        Shape<O>::eval(r[a] * ih[a] - 0.5, g.i0[a][1], g.w[a][1]);
    }
    if (G::grid_dims == 3) {                       // Jx (0, 0, 0), Jy (1/2, 1/2, 0), Jz (1/2, 0, 1/2)
        E[0] = cart3d::gather_w<O, 0, 0, 0>(et.jx, g);
        E[1] = cart3d::gather_w<O, 1, 1, 0>(et.jy, g);
        E[2] = cart3d::gather_w<O, 1, 0, 1>(et.jz, g);
    } else {
        E[0] = cart2d::gather_w<O, 0, 0>(et.jx, g);
        E[1] = cart2d::gather_w<O, 1, 1>(et.jy, g);
        E[2] = cart2d::gather_w<O, 1, 0>(et.jz, g);
    }
}

struct PassC {
    double dt, qmdt, qw0, ih[3], h[3], lo[3];
    int nseg;                                   // segments per path (work_path)
};
struct PV { double *x, *y, *z, *ux, *uy, *uz, *w, *u1x, *u1y, *u1z; };

// The end of the path and its stored global position: r1 = fl(r0 + dt vbar + lo) - lo, as move_one (decision 58), so the
// current ends where the next step starts.
HD inline void path_end(const double r0[3], const double vb[3], const PassC& c, double xg[3], double r1[3]) {
    for (int a = 0; a < 3; ++a) { xg[a] = (r0[a] + c.dt * vb[a]) + c.lo[a]; r1[a] = xg[a] - c.lo[a]; }
}
HD inline void vbar(const double u0[3], const double u1[3], double vb[3]) {
    const double g = sqrt(1.0 + u0[0] * u0[0] + u0[1] * u0[1] + u0[2] * u0[2]) + sqrt(1.0 + u1[0] * u1[0] + u1[1] * u1[1] + u1[2] * u1[2]);
    for (int a = 0; a < 3; ++a) vb[a] = (u0[a] + u1[a]) / g;
}

// mode 0: update u1 from Et along the current path, then deposit the new path into FD; mode 1: commit (x, u <- path, u1)
template <class G, int O>
void pass_impl(Species& sp, const PassC c, const PV P, const cart3d::FieldViews ET, const cart3d::FieldViews FB,
               const cart3d::FieldViews FD, int mode) {
    par_for(long(sp.size()), [=] HD(long p) {
        const double r0[3] = {P.x[p] - c.lo[0], P.y[p] - c.lo[1], P.z[p] - c.lo[2]};
        const double u0[3] = {P.ux[p], P.uy[p], P.uz[p]};
        double u1[3] = {P.u1x[p], P.u1y[p], P.u1z[p]}, vb[3], xg[3], r1[3];
        if (mode == 0) {
            vbar(u0, u1, vb);
            path_end(r0, vb, c, xg, r1);
            double S[3], Eb[3];
            work_path<G, O>(ET, r0, r1, c.ih, c.nseg, S);
            const double rm[3] = {0.5 * (r0[0] + r1[0]), 0.5 * (r0[1] + r1[1]), 0.5 * (r0[2] + r1[2])};
            bool small = false;
            for (int a = 0; a < 3; ++a) {
                const double dl = r1[a] - r0[a];
                if (fabs(dl) * c.ih[a] > 1e-12) Eb[a] = c.h[a] * S[a] / dl;
                else small = true;
            }
            if (small) {
                double Ep[3];
                et_plain<G, O>(ET, rm, c.ih, Ep);
                for (int a = 0; a < 3; ++a)
                    if (!(fabs(r1[a] - r0[a]) * c.ih[a] > 1e-12)) Eb[a] = Ep[a];
            }
            double Ed[3], B[3];
            G::template gather<RipCC, O>(FB, rm, c.ih, Ed, B);
            const double vxb[3] = {vb[1] * B[2] - vb[2] * B[1], vb[2] * B[0] - vb[0] * B[2], vb[0] * B[1] - vb[1] * B[0]};
            for (int a = 0; a < 3; ++a) u1[a] = u0[a] + c.qmdt * (Eb[a] + vxb[a]);
            P.u1x[p] = u1[0]; P.u1y[p] = u1[1]; P.u1z[p] = u1[2];
        }
        vbar(u0, u1, vb);
        path_end(r0, vb, c, xg, r1);
        if (mode == 0) {
            const double qw = c.qw0 * P.w[p];
            const double coef[3] = {qw * c.ih[1] * c.ih[2], qw * c.ih[0] * c.ih[2], qw * c.ih[0] * c.ih[1]};
            deposit_path<G, O>(FD, r0, r1, c.ih, c.nseg, coef);
        } else {
            P.x[p] = xg[0]; P.y[p] = xg[1]; P.z[p] = xg[2];
            P.ux[p] = u1[0]; P.uy[p] = u1[1]; P.uz[p] = u1[2];
        }
    });
}

template <class G, int O>
void wp2_impl(const Species& sp, const Domain& d, const FV R) {
    const double ih0 = 1.0 / d.h[0], ih1 = 1.0 / d.h[1], ih2 = 1.0 / d.h[2], iv = 1.0 / d.cell_volume();
    const double lo0 = d.lo(0), lo1 = d.lo(1), lo2 = d.lo(2), f = sp.q * sp.q / sp.m * iv;
    Species& s = const_cast<Species&>(sp);
    const double *X = s.ptr<double>(Species::X), *Y = s.ptr<double>(Species::Y), *Z = s.ptr<double>(Species::Z);
    const double *UX = s.ptr<double>(Species::UX), *UY = s.ptr<double>(Species::UY), *UZ = s.ptr<double>(Species::UZ);
    const double* W = s.ptr<double>(Species::W);
    par_for(long(sp.size()), [=] HD(long p) {
        const double ih[3] = {ih0, ih1, ih2};
        const double r[3] = {X[p] - lo0, Y[p] - lo1, Z[p] - lo2};
        const double g = sqrt(1.0 + UX[p] * UX[p] + UY[p] * UY[p] + UZ[p] * UZ[p]);
        G::template deposit_rho<RipCC, O>(R, r, ih, f * W[p] / g);
    });
}

template <class F> void dispatch(const Domain& d, int order, F f) {
    using O2 = std::integral_constant<int, 2>;
    using O3 = std::integral_constant<int, 3>;
    if (order != 2 && order != 3) throw std::runtime_error("numerics.mover = implicit: shape 2 or 3");
    if (d.flat) { if (order == 2) f(Cart2D{}, O2{}); else f(Cart2D{}, O3{}); }
    else { if (order == 2) f(Cart3D{}, O2{}); else f(Cart3D{}, O3{}); }
}

// ---------------------------------------------------------------- flat vectors of the owned currents (3 components)
void pack(const Domain& d, Field* const* f, double* v) {
    const long N = d.owned();
    const int nx = d.n[0], ny = d.n[1];
    for (int c = 0; c < 3; ++c) {
        const FV F = f[c]->view();
        double* w = v + c * N;
        par_for(N, [=] HD(long q) {
            const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
            w[q] = F(i, j, k);
        });
    }
}
void unpack(const Domain& d, const double* v, Field* const* f) {
    const long N = d.owned();
    const int nx = d.n[0], ny = d.n[1];
    for (int c = 0; c < 3; ++c) {
        const FV F = f[c]->view();
        const double* w = v + c * N;
        par_for(N, [=] HD(long q) {
            const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
            F(i, j, k) = w[q];
        });
    }
}
double dot(Comm& comm, long n, const double* a, const double* b) {
    double s = par_sum(n, [=] HD(long q) { return a[q] * b[q]; });
    comm.allreduce_sum(&s, 1);
    return s;
}

}  // namespace

struct ImplicitMover {
    Field et[3], bb[3], jd[3], pw, pinv[3];
    Buf<double> x, f, jv, xp, fp;                 // iterate, residual, deposit (flat), previous iterate and residual
    std::vector<Buf<double>> dX, dF;              // Anderson differences
    std::vector<Buf<double>> u1;                  // per species: 3 n
    MoverStats stats;
};

const MoverStats& implicit_mover_stats(const Simulation& s) {
    static MoverStats none;
    return s.mover ? s.mover->stats : none;
}

static void anderson_update(ImplicitMover& M, Comm& comm, long N3, int m_and, int it);
static void implicit_mover_step_rz(Simulation& s);

void implicit_mover_step(Simulation& s) {
    if (s.rz) { implicit_mover_step_rz(s); return; }
    const Domain& d = s.dom;
    Comm& comm = s.comm;
    Fields& F = *s.fields;
    if (!s.mover) {
        auto M = std::make_shared<ImplicitMover>();
        for (int c = 0; c < 3; ++c) { M->et[c].resize(d); M->bb[c].resize(d); M->jd[c].resize(d); M->pinv[c].resize(d); }
        M->pw.resize(d);
        const size_t n3 = size_t(3 * d.owned());
        M->x.resize(n3); M->f.resize(n3); M->jv.resize(n3); M->xp.resize(n3); M->fp.resize(n3);
        s.mover = M;
    }
    ImplicitMover& M = *s.mover;
    const int m_and = s.cfg.mover_anderson;
    const long N3 = 3 * d.owned();
    const double D = d.dt();
    // the lumped cold response at x^n: P = 1 + D^2/4 wp2 at the current positions (Jx between charge nodes i-1, i; Jy, Jz
    // between j, j+1 and k, k+1)
    M.pw.fill(0);
    for (size_t i = 0; i < s.species.size(); ++i)
        if (s.cfg.species[i].mobile)
            dispatch(d, s.cfg.order, [&](auto geo, auto o) { wp2_impl<decltype(geo), decltype(o)::value>(s.species[i], d, M.pw.view()); });
    comm.add_guards(d, {&M.pw});
    comm.exchange(d, {&M.pw});
    {
        const FV W = M.pw.view(), P0 = M.pinv[0].view(), P1 = M.pinv[1].view(), P2 = M.pinv[2].view();
        const double q = 0.25 * D * D;
        for_owned(d, [=] HD(int i, int j, int k) {
            P0(i, j, k) = 1.0 / (1.0 + q * 0.5 * (W(i - 1, j, k) + W(i, j, k)));
            P1(i, j, k) = 1.0 / (1.0 + q * 0.5 * (W(i, j, k) + W(i, j + 1, k)));
            P2(i, j, k) = 1.0 / (1.0 + q * 0.5 * (W(i, j, k) + W(i, j, k + 1)));
        });
    }
    // per species the iterate u1, started at u^n
    int nseg = 1;                                  // c dt / h_min cells at most per step: fixed segments (work_path)
    for (int a = 0; a < (d.flat ? 2 : 3); ++a) nseg = std::max(nseg, int(std::ceil(D / d.h[a] - 1e-12)));
    M.u1.resize(s.species.size());
    std::vector<PV> pv(s.species.size());
    std::vector<PassC> pc(s.species.size());
    for (size_t i = 0; i < s.species.size(); ++i) {
        Species& sp = s.species[i];
        const size_t n = sp.size();
        if (M.u1[i].size() < 3 * n || M.u1[i].size() == 0) M.u1[i].resize(3 * n + 3);
        double* U = M.u1[i].data();
        PV v{sp.ptr<double>(Species::X), sp.ptr<double>(Species::Y), sp.ptr<double>(Species::Z), sp.ptr<double>(Species::UX),
             sp.ptr<double>(Species::UY), sp.ptr<double>(Species::UZ), sp.ptr<double>(Species::W), U, U + n, U + 2 * n};
        const double *ux = v.ux, *uy = v.uy, *uz = v.uz;
        par_for(long(n), [=] HD(long p) { U[p] = ux[p]; U[n + p] = uy[p]; U[2 * n + p] = uz[p]; });
        pv[i] = v;
        pc[i] = PassC{D, sp.q * D / sp.m, sp.q / D, {1.0 / d.h[0], 1.0 / d.h[1], 1.0 / d.h[2]}, {d.h[0], d.h[1], d.h[2]},
                      {d.lo(0), d.lo(1), d.lo(2)}, nseg};
    }
    const BoundaryHooks none;
    Field* const Fj[3] = {&F.jx, &F.jy, &F.jz};
    Field* const Jd[3] = {&M.jd[0], &M.jd[1], &M.jd[2]};
    const cart3d::FieldViews ET{M.et[0].view(), M.et[1].view(), M.et[2].view(), M.bb[0].view(), M.bb[1].view(), M.bb[2].view(),
                                M.et[0].view(), M.et[1].view(), M.et[2].view()};
    const cart3d::FieldViews FB{M.bb[0].view(), M.bb[1].view(), M.bb[2].view(), M.bb[0].view(), M.bb[1].view(), M.bb[2].view(),
                                M.bb[0].view(), M.bb[1].view(), M.bb[2].view()};
    const cart3d::FieldViews FD{M.jd[0].view(), M.jd[1].view(), M.jd[2].view(), M.jd[0].view(), M.jd[1].view(), M.jd[2].view(),
                                M.jd[0].view(), M.jd[1].view(), M.jd[2].view()};
    M.dX.clear(); M.dF.clear();
    int it = 0;
    double res = 0;
    for (;;) {
        irip_trial(F, d, comm, it == 0 ? s.hooks : none, s.t, M.et, M.bb);   // the hooks (laser, xflux) once per step
        for (int c = 0; c < 3; ++c) M.jd[c].fill(0);
        for (size_t i = 0; i < s.species.size(); ++i)
            if (s.cfg.species[i].mobile)
                dispatch(d, s.cfg.order, [&](auto geo, auto o) {
                    pass_impl<decltype(geo), decltype(o)::value>(s.species[i], pc[i], pv[i], ET, FB, FD, 0);
                });
        comm.add_guards(d, {&M.jd[0], &M.jd[1], &M.jd[2]});
        pack(d, Fj, M.x.data());
        pack(d, Jd, M.jv.data());
        {
            double* fr = M.f.data();
            const double *xv = M.x.data(), *jv = M.jv.data();
            Field* pi = M.pinv;
            const long N = d.owned();
            const int nx = d.n[0], ny = d.n[1];
            for (int c = 0; c < 3; ++c) {
                const FV P = pi[c].view();
                par_for(N, [=] HD(long q) {
                    const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
                    fr[c * N + q] = P(i, j, k) * (jv[c * N + q] - xv[c * N + q]);
                });
            }
        }
        const double ff = dot(comm, N3, M.f.data(), M.f.data()), jj = dot(comm, N3, M.jv.data(), M.jv.data());
        res = jj > 0 ? std::sqrt(ff / jj) : (ff > 0 ? 1e300 : 0.0);
        if (vlpl_getenv("VLPL_MOVER_DEBUG") && comm.rank == 0) std::fprintf(stderr, "mover step %ld it %d res %.3e\n", s.istep, it, res);
        ++it;
        if (res < s.cfg.mover_tol || it >= s.cfg.mover_maxit) break;
        anderson_update(M, comm, N3, m_and, it);
        unpack(d, M.x.data(), Fj);
    }
    // commit: the field step with the deposit of the final paths, then the particles
    unpack(d, M.jv.data(), Fj);
    F.step(comm, none, s.t);
    for (size_t i = 0; i < s.species.size(); ++i)
        if (s.cfg.species[i].mobile)
            dispatch(d, s.cfg.order, [&](auto geo, auto o) {
                pass_impl<decltype(geo), decltype(o)::value>(s.species[i], pc[i], pv[i], ET, FB, FD, 1);
            });
    MoverStats& st = M.stats;
    ++st.steps; st.iterations += it;
    if (it > st.max_iterations) st.max_iterations = it;
    if (res >= s.cfg.mover_tol) ++st.unconverged;
    st.last_residual = res;
}

// Anderson (type II, depth m_and) on x -> x + f: M.x updated in place from M.x, M.f (it: the iteration count, >= 1)
static void anderson_update(ImplicitMover& M, Comm& comm, long N3, int m_and, int it) {
    {
        double* x = M.x.data();
        const double* f = M.f.data();
        if (it > 1 && m_and > 0) {
            if (int(M.dX.size()) == m_and) {       // drop the oldest: rotate
                Buf<double> a = std::move(M.dX.front()), b = std::move(M.dF.front());
                M.dX.erase(M.dX.begin()); M.dF.erase(M.dF.begin());
                M.dX.push_back(std::move(a)); M.dF.push_back(std::move(b));
            } else {
                M.dX.emplace_back(size_t(N3)); M.dF.emplace_back(size_t(N3));
            }
            double* dx = M.dX.back().data();
            double* df = M.dF.back().data();
            const double *xp = M.xp.data(), *fp = M.fp.data();
            par_for(N3, [=] HD(long q) { dx[q] = x[q] - xp[q]; df[q] = f[q] - fp[q]; });
        }
        {
            double *xp = M.xp.data(), *fp = M.fp.data();
            par_for(N3, [=] HD(long q) { xp[q] = x[q]; fp[q] = f[q]; });
        }
        const int k = int(M.dF.size());
        std::vector<double> gam(size_t(k), 0.0);
        if (k > 0) {
            std::vector<double> A(static_cast<size_t>(k * k), 0.0), b(static_cast<size_t>(k), 0.0);
            for (int a = 0; a < k; ++a) {
                b[size_t(a)] = dot(comm, N3, M.dF[size_t(a)].data(), f);
                for (int c = 0; c <= a; ++c)
                    A[size_t(a * k + c)] = A[size_t(c * k + a)] = dot(comm, N3, M.dF[size_t(a)].data(), M.dF[size_t(c)].data());
            }
            double tr = 0;
            for (int a = 0; a < k; ++a) tr += A[size_t(a * k + a)];
            for (int a = 0; a < k; ++a) A[size_t(a * k + a)] += 1e-14 * tr;      // regularised normal equations
            for (int c = 0; c < k; ++c)                                          // Gauss elimination (SPD, small)
                for (int r = c + 1; r < k; ++r) {
                    const double m = A[size_t(r * k + c)] / A[size_t(c * k + c)];
                    for (int q = c; q < k; ++q) A[size_t(r * k + q)] -= m * A[size_t(c * k + q)];
                    b[size_t(r)] -= m * b[size_t(c)];
                }
            for (int r = k - 1; r >= 0; --r) {
                double v = b[size_t(r)];
                for (int q = r + 1; q < k; ++q) v -= A[size_t(r * k + q)] * gam[size_t(q)];
                gam[size_t(r)] = v / A[size_t(r * k + r)];
            }
        }
        par_for(N3, [=] HD(long q) { x[q] += f[q]; });
        for (int a = 0; a < k; ++a) {
            const double g = gam[size_t(a)];
            const double *dx = M.dX[size_t(a)].data(), *df = M.dF[size_t(a)].data();
            par_for(N3, [=] HD(long q) { x[q] -= g * (dx[q] + df[q]); });
        }
    }
}

// RZ (decision 82): the same solve with the RZ trial step (rz_irip_trial), the particle pass of src/rz_particles.cpp (the C1
// radial map, the force per path segment from the exact work) and the currents J_z, J_r, J_phi of all modes as the iterate.
// Preconditioner: P = 1 + D^2/4 wp2 of the m = 0 density at the J points (the same factor for every mode).
// Segments: n = ceil(c dt max(1/dz, dxi/dr)): every segment moves less than one cell in z and in xi.
static void implicit_mover_step_rz(Simulation& s) {
    const Domain& d = s.dom;
    Comm& comm = s.comm;
    FieldsRZ& F = *s.rz;
    if (!s.mover) {
        auto M = std::make_shared<ImplicitMover>();
        for (int c = 0; c < 3; ++c) { M->et[c].resize(d); M->bb[c].resize(d); M->jd[c].resize(d); M->pinv[c].resize(d); }
        M->pw.resize(d);
        const size_t n3 = size_t(3 * d.owned());
        M->x.resize(n3); M->f.resize(n3); M->jv.resize(n3); M->xp.resize(n3); M->fp.resize(n3);
        s.mover = M;
    }
    ImplicitMover& M = *s.mover;
    const long N3 = 3 * d.owned();
    const double D = d.dt();
    M.pw.fill(0);
    for (size_t i = 0; i < s.species.size(); ++i)
        if (s.cfg.species[i].mobile) rz_wp2(s.species[i], F, M.pw);
    comm.add_guards(d, {&M.pw});
    comm.exchange(d, {&M.pw});
    {
        const FV W = M.pw.view(), P0 = M.pinv[0].view(), P1 = M.pinv[1].view(), P2 = M.pinv[2].view();
        const double q = 0.25 * D * D;
        const int nr = d.n[1];
        for_owned(d, [=] HD(int i, int j, int k) {
            P0(i, j, k) = 1.0 / (1.0 + q * 0.5 * (W(i - 1, j, 0) + W(i, j, 0)));                    // J_z at (i, r_j)
            P1(i, j, k) = 1.0 / (1.0 + q * 0.5 * (W(i, j, 0) + (j + 1 < nr ? W(i, j + 1, 0) : 0.0)));   // J_r at (i+1/2, r_{j+1/2})
            P2(i, j, k) = 1.0 / (1.0 + q * W(i, j, 0));                                         // J_phi at (i+1/2, r_j)
        });
    }
    const RadialGrid& g = F.grid();
    double smax = 0;                                // the largest dxi/dr of the C1 map
    for (size_t j = 0; j + 1 < g.r.size(); ++j)
        for (int q = 0; q <= 32; ++q) {
            const double u = q / 32.0;
            smax = std::max(smax, (1 + g.cl[j] * (1 - u) * (1 - 3 * u) - g.cr[j] * u * (2 - 3 * u) + g.ca[j] * 2 * u * (1 - u) * (1 - 2 * u)) / g.dn[j]);
        }
    const int nseg = std::max(1, int(std::ceil(std::max(D / d.h[0], D * smax) - 1e-12)));
    M.u1.resize(s.species.size());
    for (size_t i = 0; i < s.species.size(); ++i) {
        Species& sp = s.species[i];
        const size_t n = sp.size();
        if (M.u1[i].size() < 3 * n || M.u1[i].size() == 0) M.u1[i].resize(3 * n + 3);
        double* U = M.u1[i].data();
        const double *ux = sp.ptr<double>(Species::UX), *uy = sp.ptr<double>(Species::UY), *uz = sp.ptr<double>(Species::UZ);
        par_for(long(n), [=] HD(long p) { U[p] = ux[p]; U[n + p] = uy[p]; U[2 * n + p] = uz[p]; });
    }
    Field* const Fj[3] = {&F.jz, &F.jr, &F.jp};
    Field* const Jd[3] = {&M.jd[0], &M.jd[1], &M.jd[2]};
    M.dX.clear(); M.dF.clear();
    int it = 0;
    double res = 0;
    for (;;) {
        rz_irip_trial(F, comm, s.t, M.et, M.bb);
        for (int c = 0; c < 3; ++c) M.jd[c].fill(0);
        for (size_t i = 0; i < s.species.size(); ++i)
            if (s.cfg.species[i].mobile) rz_implicit_pass(s.species[i], F, M.et, M.bb, M.jd, M.u1[i].data(), nseg, 0);
        comm.add_guards(d, {&M.jd[0], &M.jd[1], &M.jd[2]});
        pack(d, Fj, M.x.data());
        pack(d, Jd, M.jv.data());
        {
            double* fr = M.f.data();
            const double *xv = M.x.data(), *jv = M.jv.data();
            const long N = d.owned();
            const int nz = d.n[0], nr = d.n[1];
            for (int c = 0; c < 3; ++c) {
                const FV P = M.pinv[c].view();
                par_for(N, [=] HD(long q) {
                    const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
                    fr[c * N + q] = P(i, j, k) * (jv[c * N + q] - xv[c * N + q]);
                });
            }
        }
        const double ff = dot(comm, N3, M.f.data(), M.f.data()), jj = dot(comm, N3, M.jv.data(), M.jv.data());
        res = jj > 0 ? std::sqrt(ff / jj) : (ff > 0 ? 1e300 : 0.0);
        if (vlpl_getenv("VLPL_MOVER_DEBUG") && comm.rank == 0) std::fprintf(stderr, "mover step %ld it %d res %.3e\n", s.istep, it, res);
        ++it;
        if (res < s.cfg.mover_tol || it >= s.cfg.mover_maxit) break;
        anderson_update(M, comm, N3, s.cfg.mover_anderson, it);
        unpack(d, M.x.data(), Fj);
    }
    unpack(d, M.jv.data(), Fj);                     // commit: the field step with the deposit of the final paths
    F.step(comm, s.t);
    for (size_t i = 0; i < s.species.size(); ++i)
        if (s.cfg.species[i].mobile) rz_implicit_pass(s.species[i], F, M.et, M.bb, M.jd, M.u1[i].data(), nseg, 1);
    MoverStats& st = M.stats;
    ++st.steps; st.iterations += it;
    if (it > st.max_iterations) st.max_iterations = it;
    if (res >= s.cfg.mover_tol) ++st.unconverged;
    st.last_residual = res;
}
