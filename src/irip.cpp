// Implicit RIP, box closure (docs/specs/implicit_rip.md, decision 17, decision 38): RIP-CC with the
// predictor Ex*, Bx* of the transverse sources replaced by the time average mu_t Ex = (Ex^n + Ex^{n+1})/2,
// mu_t Bx. Both movers use the same longitudinal fields, so the RIP-CC Gauss law and div B hold to
// round-off for any value of them; the box average makes the step energy conserving and unconditionally
// stable (no h_perp >= c dt limit), with zero dispersion along x (c dt = h_x stays).
//
// The step is linear in X = Ex^{n+1}, Y = Bx^{n+1} (research/irip_3d/identity.py, to 1e-15):
//   (I - D^2/4 mu_x^2 Lap_perp) X = R_E,   the same for Y with R_B, decoupled,
// R_E, R_B: Ex, Bx of the RIP-CC step with X = Y = 0 in the averages; mu_x^2 = [1, 2, 1]/4 along x.
// One step:
//   1. pass 1: transport + corrector with Ex* = Ex^n/2, Bx* = Bx^n/2 -> R_E, R_B (fields untouched);
//   2. solve: 2D DFT in y, z per x slice (periodic: exact eigenbasis of the 5-point Lap_perp,
//      lambda = 4/hy^2 sin^2(pi qy/ny) + 4/hz^2 sin^2(pi qz/nz)), then per mode a cyclic tridiagonal
//      system along x (diagonal 1 + kappa^2/2, off-diagonals kappa^2/4, kappa^2 = D^2 lambda/4), back;
//   3. pass 2: transport + corrector with Ex* = (Ex^n + X)/2, Bx* = (Bx^n + Y)/2: the new fields. The
//      corrector's Ex, Bx (equal to X, Y up to the solver's round-off) are kept, which keeps the Gauss law.
// O5 (numerics.closure = o5, decision 72, docs/specs/overdense.md "O5 prototype"): the averages get the closure
//   Ex* <- Cx C_perp Ex*,  Cx = 1 - (1/3) h_x^2 delta_x^2,  C_perp per transverse mode (lambda = lambda_y + lambda_z)
//   g_perp = 1 + (2/3) min(q, 1) + (h_y^2 lambda_y^2 + h_z^2 lambda_z^2) / (12 lambda),  q = D^2 lambda/4 (the cap: decision 77),
// Bx* likewise, which cancels the leading oblique dispersion of the time step and of the transverse grid per axis (any
// closure shared by both movers keeps the Gauss law; a local C_perp cannot cancel the per-axis error h_a^2 lambda_a^2/12
// in 3D, decision 73). The factor is applied in the y, z Fourier basis of the solve (in 2D too since decision 77; without the
// cap it was the stencil 1 - (D^2/6 + h_y^2/12) delta_y^2). The system per mode becomes X + kappa (2 + S)(5 - S)/12 X = R
// with S x_i = x_{i-1} + x_{i+1}, kappa = D^2/4 lambda g_perp: the pentadiagonal factors into -(kappa/12)(S - r1)(S - r2),
// r = (3 +- sqrt(49 + 48/kappa)) / 2, two tridiagonal line solves (cyclic, or zeros outside: both factors are polynomials in
// the same truncated S). Not energy conserving: transient growth <= 1.4 (h_perp = 3.2 dx) .. 2 (h_perp = dx) at the grid
// scale. With CPML (decision 75) the factor acts per mode of the eigenbasis of the stretched transverse operator of the kind
// (the basis of the solve; lambda_y, lambda_z its eigenvalues), so L C_perp stays diagonal per mode. The TFSF plane (decision
// 75): the incident grid runs o5 too; its Ex*, Bx* get the closure, and the Cx stencil of the main grid across the plane
// gets the incident value of the far cell (Tfsf::o5_fix). The exact plane waves (side tube, tfsf_incident = modes; decision
// 76) carry the closure in their one-step matrix; the tube corrects the closed Ex*, Bx* by [in] c I - C(chi_in I).
// MPI: ranks along x only (y, z planes whole on every rank for the DFT); the systems along x are solved
// by the partition method (xlines::solve_lines, src/xlines.hpp). x periodic or open (laser, window: the hooks and guards of
// RIP-CC); y, z periodic so far (PML follows).
#include "fields.hpp"
#include <cmath>
#include "irip.hpp"
#include "medium.hpp"
#include "pml.hpp"
#include "side.hpp"
#include "tfsf.hpp"
#include "xlines.hpp"
#include <stdexcept>

namespace {
HD inline double mux(const FV& f, int i, int j, int k) { return 0.5 * (f(i, j, k) + f(i + 1, j, k)); }

}  // namespace

struct IripWork : xlines::LineWork {   // spectrum, scratch, twiddles and partition-method buffers (src/xlines.hpp)
    Field rE, rB;                    // R_E, R_B, then X, Y
    // CPML in y, z (decision 41): a pml axis uses the dense eigenbasis V (inverse Vi) of its transverse
    // operator instead of the DFT; lam: the eigenvalues of -L per axis, for Ex (kind 0) and Bx (kind 1)
    bool dense[3] = {false, false, false};
    Buf<double> V[2][3], Vi[2][3], lam[2][3];
    Field sE, sB;                    // O5 with the side tube: the correction of the closed Ex*, Bx* per step (decision 76)
};

namespace {
// Transport + corrector of RIP-CC (fields.cpp) with given longitudinal averages Exs, Bxs (guards valid):
// the transverse fields into the scratch nEy..nBz (exchanged), Ex, Bx of the corrector into oEx, oBx
// (owned cells; may be the fields themselves).
void box_pass(const Domain& d, Comm& comm, const FV Ex, const FV Ey, const FV Ez, const FV Bx, const FV By, const FV Bz,
              const FV Jx, const FV Jy, const FV Jz, const FV Exs, const FV Bxs, Field& ney, Field& nez, Field& nby,
              Field& nbz, const FV oEx, const FV oBx, Tfsf* tf = nullptr, int kind = 0, const SideTfsf* sd = nullptr) {
    const double D = d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    const FV nEy = ney.view(), nEz = nez.view(), nBy = nby.view(), nBz = nbz.view();
    for_owned(d, [=] HD(int i, int j, int k) {
        auto psiy = [&](int a) { return (mux(Bxs, a, j, k) - mux(Bxs, a, j, k - 1)) * iz - Jy(a, j, k); };
        auto psiz = [&](int a) { return -(mux(Bxs, a, j, k) - mux(Bxs, a, j - 1, k)) * iy - Jz(a, j, k); };
        auto phiy = [&](int a) { return -(mux(Exs, a, j, k + 1) - mux(Exs, a, j, k)) * iz; };
        auto phiz = [&](int a) { return (mux(Exs, a, j + 1, k) - mux(Exs, a, j, k)) * iy; };
        const double tpy = Ey(i - 1, j, k) + Bz(i - 1, j, k) + D * (psiy(i - 1) + phiz(i - 1));
        const double tmy = Ey(i + 1, j, k) - Bz(i + 1, j, k) + D * (psiy(i) - phiz(i));
        const double tpz = Ez(i + 1, j, k) + By(i + 1, j, k) + D * (psiz(i) + phiy(i));
        const double tmz = Ez(i - 1, j, k) - By(i - 1, j, k) + D * (psiz(i - 1) - phiy(i - 1));
        nEy(i, j, k) = 0.5 * (tpy + tmy);  nBz(i, j, k) = 0.5 * (tpy - tmy);
        nEz(i, j, k) = 0.5 * (tpz + tmz);  nBy(i, j, k) = 0.5 * (tpz - tmz);
    });
    if (tf) tf->correct(comm, ney, nez, nby, nbz, kind);      // the incident T across the TFSF plane (kind: tfsf.hpp)
    if (sd) sd->fix_transport(nEy, nEz, nBy, nBz, kind == 2 && d.o5 ? 3 : kind);   // the side tube (side.hpp; o5: kind 3)
    comm.exchange(d, {&ney, &nez, &nby, &nbz});
    for_owned(d, [=] HD(int i, int j, int k) {
        const double bz0 = 0.5 * (Bz(i, j, k) + nBz(i, j, k)), bzm = 0.5 * (Bz(i, j - 1, k) + nBz(i, j - 1, k));
        const double by0 = 0.5 * (By(i, j, k) + nBy(i, j, k)), bym = 0.5 * (By(i, j, k - 1) + nBy(i, j, k - 1));
        const double ez0 = 0.5 * (Ez(i, j, k) + nEz(i, j, k)), ezp = 0.5 * (Ez(i, j + 1, k) + nEz(i, j + 1, k));
        const double ey0 = 0.5 * (Ey(i, j, k) + nEy(i, j, k)), eyp = 0.5 * (Ey(i, j, k + 1) + nEy(i, j, k + 1));
        oEx(i, j, k) = Ex(i, j, k) + D * ((bz0 - bzm) * iy - (by0 - bym) * iz - Jx(i, j, k));
        oBx(i, j, k) = Bx(i, j, k) - D * ((ezp - ez0) * iy - (eyp - ey0) * iz);
    });
    if (sd) sd->fix_corr(oEx, oBx);
}

// Mode coefficients of the system along x: b x_i + s (x_{i-1} + x_{i+1}) = r_i, cyclic for periodic x, zeros outside
// for open x. ly, lz: the eigenvalues of -L per axis (with CPML), else the 5-point formula of the periodic DFT.
struct IripCoef {
    int ny, nz;
    double D, hy, hz;
    const double *ly, *lz;
    bool cyc;
    HD xlines::LineCoef operator()(long M) const {
        const int qy = int(M % ny), qz = int(M / ny);
        double lam;
        if (ly) lam = ly[qy] + lz[qz];
        else {
            const double sy = sin(M_PI * qy / ny), sz = sin(M_PI * qz / nz);
            lam = 4 * sy * sy / (hy * hy) + 4 * sz * sz / (hz * hz);
        }
        const double k2 = D * D * lam / 4, b = 1 + 0.5 * k2;
        return {b, 0.25 * k2, b, b, cyc};
    }
};

// O5: the transverse factor g_perp of the closure per mode (eigenvalues ly, lz of -L per axis). The time-step part D^2 lam/6 =
// (2/3) q, q = D^2 lam/4, is capped at q = 1 (k_perp c dt <= 2; decision 77): beyond, the modes are not resolved in time and
// the uncapped term gives transient growth ~ (dx/dy)^2 at the grid scale for dy << dx.
HD inline double o5_gtime(double D, double lam) { return 2.0 / 3 * fmin(D * D * lam / 4, 1.0); }
HD inline double o5_gperp(double D, double hy, double hz, double ly, double lz) {
    const double lam = ly + lz;
    return lam > 0 ? 1 + o5_gtime(D, lam) + (hy * hy * ly * ly + hz * hz * lz * lz) / (12 * lam) : 1.0;
}

// O5: the two tridiagonal factors of the pentadiagonal system per mode, first -(kappa/12)(S - r2) y = r, then (S - r1) x = y.
// lys, lzs: the eigenvalues of -L per axis (with CPML), else the 5-point formula of the periodic DFT.
struct O5Coef {
    int ny, nz;
    double D, hy, hz;
    const double *lys, *lzs;
    bool cyc;
    int which;                       // 0: the factor with r2 (and the scale), 1: the factor with r1
    HD xlines::LineCoef operator()(long M) const {
        const int qy = int(M % ny), qz = int(M / ny);
        double ly, lz;
        if (lys) { ly = lys[qy]; lz = lzs[qz]; }
        else {
            const double sy = sin(M_PI * qy / ny), sz = sin(M_PI * qz / nz);
            ly = 4 * sy * sy / (hy * hy); lz = 4 * sz * sz / (hz * hz);
        }
        const double kap = D * D / 4 * (ly + lz) * o5_gperp(D, hy, hz, ly, lz);
        if (kap == 0) return {1, 0, 1, 1, false};                // the mode lambda = 0: the identity
        const double q = sqrt(49 + 48 / kap);
        if (which == 0) {
            const double b = kap * (3 - q) / 24, s = -kap / 12;
            return {b, s, b, b, cyc};
        }
        const double b = -(3 + q) / 2;
        return {b, 1, b, b, cyc};
    }
};

// O5: Ex* <- Cx C_perp Ex* on the owned cells (guards valid on entry; the guards of an open x edge stay 0), then exchanged.
// C_perp per mode in the transverse basis of the solve of this kind (0: Ex*, 1: Bx*; the DFT, or on a pml axis the eigenbasis of
// the stretched operator), also in 2D (the capped factor, decision 77, is no stencil)
void o5_closure(IripWork& W, const Domain& d, Comm& comm, Field& f, int kind, bool cx = true) {
    const double D = d.h[0];
    const bool dy = W.dense[1], dz = W.dense[2];
    {
        const int nx = d.n[0], ny = d.n[1], nz = d.n[2];
        const double hy = d.h[1], hz = d.h[2];
        const double norm = 1.0 / ((dy ? 1.0 : double(ny)) * (dz ? 1.0 : double(nz)));
        const double* ly = dy || dz ? W.lam[kind][1].data() : nullptr;
        const double* lz = dy || dz ? W.lam[kind][2].data() : nullptr;
        const FV F = f.view();
        double* w = W.w.data();
        double* t = W.t.data();
        const double* ty = W.twy.data();
        const double* tz = W.twz.data();
        for_owned(d, [=] HD(int i, int j, int k) {
            const long q = 2 * ((long(k) * ny + j) * nx + i);
            w[q] = cx ? F(i, j, k) - (F(i + 1, j, k) - 2 * F(i, j, k) + F(i - 1, j, k)) / 3 : F(i, j, k); w[q + 1] = 0;
        });
        xlines::transform_y(w, t, nx, ny, nz, dy ? W.Vi[kind][1].data() : nullptr, ty, -1);
        xlines::transform_z(w, t, nx, ny, nz, dz ? W.Vi[kind][2].data() : nullptr, tz, -1);
        for_owned(d, [=] HD(int i, int j, int k) {           // j, k: the mode (qy, qz)
            const long q = 2 * ((long(k) * ny + j) * nx + i);
            double a, b;
            if (ly) { a = ly[j]; b = lz[k]; }
            else {
                const double sy = sin(M_PI * j / ny), sz = sin(M_PI * k / nz);
                a = 4 * sy * sy / (hy * hy); b = 4 * sz * sz / (hz * hz);
            }
            const double c = norm * o5_gperp(D, hy, hz, a, b);
            w[q] *= c; w[q + 1] *= c;
        });
        xlines::transform_z(w, t, nx, ny, nz, dz ? W.V[kind][2].data() : nullptr, tz, +1);
        xlines::transform_y(w, t, nx, ny, nz, dy ? W.V[kind][1].data() : nullptr, ty, +1);
        for_owned(d, [=] HD(int i, int j, int k) { F(i, j, k) = w[2 * ((long(k) * ny + j) * nx + i)]; });
        comm.exchange(d, {&f});
    }
}

// (I - D^2/4 mu_x^2 Lap_perp) X = R in place on the owned cells of r (periodic box; x may be split
// over ranks, y and z not). O5: (I - D^2/4 mu_x^2 Cx Lap_perp C_perp) X = R.
void solve(IripWork& W, const Domain& d, Comm& comm, Field& r, int kind, bool o5) {
    const int nx = d.n[0], ny = d.n[1], nz = d.n[2];
    const double D = d.h[0], hy = d.h[1], hz = d.h[2];
    const FV R = r.view();
    double* w = W.w.data();
    double* t = W.t.data();
    const double* ty = W.twy.data();
    const double* tz = W.twz.data();
    for_owned(d, [=] HD(int i, int j, int k) {
        const long q = 2 * ((long(k) * ny + j) * nx + i);
        w[q] = R(i, j, k); w[q + 1] = 0;
    });
    // forward transforms along y (lines over i, k) and z (lines over i, j): the DFT, or on a pml axis
    // the dense eigenbasis (real matrices on the complex lines)
    const bool dy = W.dense[1], dz = W.dense[2];
    const double* ly = dy || dz ? W.lam[kind][1].data() : nullptr;
    const double* lz = dy || dz ? W.lam[kind][2].data() : nullptr;
    xlines::transform_y(w, t, nx, ny, nz, dy ? W.Vi[kind][1].data() : nullptr, ty, -1);
    xlines::transform_z(w, t, nx, ny, nz, dz ? W.Vi[kind][2].data() : nullptr, tz, -1);
    // per mode (qy, qz) the system along x, b x + s (x_- + x_+) = r: cyclic (periodic x, Sherman-Morrison)
    // or with zeros outside (open x: the guards of Ex*, Bx* are 0 at an open edge)
    if (o5) {
        xlines::solve_lines(W, d, comm, O5Coef{ny, nz, D, hy, hz, ly, lz, d.periodic[0], 0});
        xlines::solve_lines(W, d, comm, O5Coef{ny, nz, D, hy, hz, ly, lz, d.periodic[0], 1});
    } else xlines::solve_lines(W, d, comm, IripCoef{ny, nz, D, hy, hz, ly, lz, d.periodic[0]});
    // back along z and y, the real part / (ny nz) (the DFT axes only)
    xlines::transform_z(w, t, nx, ny, nz, dz ? W.V[kind][2].data() : nullptr, tz, +1);
    xlines::transform_y(w, t, nx, ny, nz, dy ? W.V[kind][1].data() : nullptr, ty, +1);
    const double norm = 1.0 / ((dy ? 1.0 : double(ny)) * (dz ? 1.0 : double(nz)));
    for_owned(d, [=] HD(int i, int j, int k) { R(i, j, k) = w[2 * ((long(k) * ny + j) * nx + i)] * norm; });
}
}  // namespace

void irip_o5_closure(Fields& F, const Domain& d, Comm& comm, Field& f, int kind, bool cx) {
    if (!F.irip) throw std::runtime_error("irip_o5_closure: before the first iRIP step");
    o5_closure(*F.irip, d, comm, f, kind, cx);
}

void irip_step(Fields& F, const Domain& d, Comm& comm, const BoundaryHooks& hooks, double t) {
    irip_step_impl(F, d, comm, hooks, t, nullptr, nullptr);
}

void irip_trial(Fields& F, const Domain& d, Comm& comm, const BoundaryHooks& hooks, double t, Field* et, Field* bb) {
    if (F.pml || F.tfsf || F.side || F.o5) throw std::runtime_error("irip_trial: box closure without pml, TFSF or side tube");
    irip_step_impl(F, d, comm, hooks, t, et, bb);
}

// The work field of the box closure (O6, docs/specs/implicit_mover.md): with Exs = mu_t Ex, Bxs = mu_t Bx of the step and
// the fields E^n, B^n, at the positions of Jx, Jy, Jz,
//   Et_x = mu_t Ex,
//   Et_y = [(Ey + Bz)(i) + (Ey - Bz)(i+1)] / 2 + D/2 Psi_y - D^2/4 delta_y delta_x mu_t Ex   at (i+1/2, j+1/2, k),
//   Et_z = [(Ez - By)(i) + (Ez + By)(i+1)] / 2 + D/2 Psi_z - D^2/4 delta_z delta_x mu_t Ex   at (i+1/2, j, k+1/2),
// the midpoints of the characteristics plus the corner term of Ex; then W^{n+1} - W^n = -D dV sum J . Et exactly
// (research/o6_mover/work3d.py: 5e-15; the Psi contain -J_y, -J_z, the current's own source).
static void irip_work_field(const Domain& d, const FV Ey, const FV Ez, const FV By, const FV Bz, const FV Jy, const FV Jz,
                            const FV Exs, const FV Bxs, const FV Ex_, const FV Ey_, const FV Ez_) {
    const double D = d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    for_owned(d, [=] HD(int i, int j, int k) {
        auto dxm = [&](int b, int c) { return (Exs(i + 1, b, c) - Exs(i, b, c)) / D; };
        const double psiy = (mux(Bxs, i, j, k) - mux(Bxs, i, j, k - 1)) * iz - Jy(i, j, k);
        const double psiz = -(mux(Bxs, i, j, k) - mux(Bxs, i, j - 1, k)) * iy - Jz(i, j, k);
        Ex_(i, j, k) = Exs(i, j, k);
        Ey_(i, j, k) = 0.5 * (Ey(i, j, k) + Bz(i, j, k) + Ey(i + 1, j, k) - Bz(i + 1, j, k)) + 0.5 * D * psiy -
                       0.25 * D * D * (dxm(j + 1, k) - dxm(j, k)) * iy;
        Ez_(i, j, k) = 0.5 * (Ez(i, j, k) - By(i, j, k) + Ez(i + 1, j, k) + By(i + 1, j, k)) + 0.5 * D * psiz -
                       0.25 * D * D * (dxm(j, k + 1) - dxm(j, k)) * iz;
    });
}

static void box_means(const Domain& d, const FV Ex, const FV Bx, const FV X, const FV Y, const FV Exs, const FV Bxs) {
    for_owned(d, [=] HD(int i, int j, int k) {
        Exs(i, j, k) = 0.5 * (Ex(i, j, k) + X(i, j, k));
        Bxs(i, j, k) = 0.5 * (Bx(i, j, k) + Y(i, j, k));
    });
}
static void mean_b(const Domain& d, const FV Bx, const FV By, const FV Bz, const FV Y, const FV nBy, const FV nBz, const FV b0,
                   const FV b1, const FV b2) {
    for_owned(d, [=] HD(int i, int j, int k) {
        b0(i, j, k) = 0.5 * (Bx(i, j, k) + Y(i, j, k));
        b1(i, j, k) = 0.5 * (By(i, j, k) + nBy(i, j, k));
        b2(i, j, k) = 0.5 * (Bz(i, j, k) + nBz(i, j, k));
    });
}

// et, bb null: the step (commit). Else the trial of O6: the same step without changing F; et[3] get the work field, bb[3]
// (B^n + B^{n+1}) / 2, both with valid guards.
void irip_step_impl(Fields& F, const Domain& d, Comm& comm, const BoundaryHooks& hooks, double t, Field* et, Field* bb) {
    Pml* P = F.pml;
    if (!F.irip) {
        bool ok = comm.dims[1] == 1 && comm.dims[2] == 1 && d.n[0] >= 3;
        for (int a = 1; a < 3; ++a) ok = ok && (d.periodic[a] || (P && P->cfg.on[a]));
        if (!ok) throw std::runtime_error("iRIP: y and z periodic or pml, ranks along x only, nx >= 3 per rank");
        if (F.o5 && (F.side || F.tfsf) && !d.o5)
            throw std::runtime_error("numerics.closure = o5 with a TFSF source: the domain must carry o5 (Simulation)");
        auto W = std::make_shared<IripWork>();
        W->rE.resize(d); W->rB.resize(d);
        if (F.o5 && F.side) { W->sE.resize(d); W->sB.resize(d); }
        W->resize(d);
        if (P && (P->cfg.on[1] || P->cfg.on[2]))           // eigenbases of the pml axes, DFT eigenvalues else
            for (int a = 1; a < 3; ++a)
                for (int kind = 0; kind < 2; ++kind) {
                    std::vector<double> V, Vi, lam;
                    if (P->cfg.on[a]) {
                        irip_axis_basis(d, *P, a, kind, V, Vi, lam);
                        W->dense[a] = true;
                        W->V[kind][a].upload(V); W->Vi[kind][a].upload(Vi);
                    } else {
                        lam.resize(size_t(d.n[a]));
                        for (int q = 0; q < d.n[a]; ++q) {
                            const double sq = std::sin(M_PI * q / d.n[a]);
                            lam[size_t(q)] = 4 * sq * sq / (d.h[a] * d.h[a]);
                        }
                    }
                    W->lam[kind][a].upload(lam);
                }
        F.irip = W;
    }
    IripWork& W = *F.irip;
    if (F.tfsf) F.tfsf->begin(comm, t);              // the incident grid of the TFSF source to t_n+1 (tfsf.hpp)
    if (F.side) F.side->begin(t);                    // the side tube at t_n (side.hpp)
    // O5 with the side tube (decision 76): the closed Ex*, Bx* must be C(F) + [in] c I - C(chi_in I) (I: the incident means,
    // c I its closure per plane wave, C the closure of the grid), so that a field F = chi_in I (+ scattered S) gives c I in the
    // tube and C(S) outside, whatever C does near the CPML layers. The correction s = [in] c I - C(chi_in I), once per step.
    if (F.o5 && F.side) {
        W.sE.fill(0); W.sB.fill(0);
        F.side->o5_means(W.sE.view(), W.sB.view(), false);
        comm.exchange(d, {&W.sE, &W.sB});
        o5_closure(W, d, comm, W.sE, 0); o5_closure(W, d, comm, W.sB, 1);
        const FV sE = W.sE.view(), sB = W.sB.view();
        for_owned(d, [=] HD(int i, int j, int k) { sE(i, j, k) = -sE(i, j, k); sB(i, j, k) = -sB(i, j, k); });
        F.side->o5_means(sE, sB, true);
    }
    comm.exchange(d, {&F.jx, &F.jy, &F.jz});           // sources at i-1/2 need the left guard
    const FV Ex = F.ex.view(), Ey = F.ey.view(), Ez = F.ez.view(), Bx = F.bx.view(), By = F.by.view(), Bz = F.bz.view();
    const FV Jx = F.jx.view(), Jy = F.jy.view(), Jz = F.jz.view();
    const FV Exs = F.exs_.view(), Bxs = F.bxs_.view(), RE = W.rE.view(), RB = W.rB.view();
    // 1. right-hand sides: the averages with Ex^{n+1} = Bx^{n+1} = 0
    for_owned(d, [=] HD(int i, int j, int k) { Exs(i, j, k) = 0.5 * Ex(i, j, k); Bxs(i, j, k) = 0.5 * Bx(i, j, k); });
    comm.exchange(d, {&F.exs_, &F.bxs_});
    if (F.o5) { o5_closure(W, d, comm, F.exs_, 0); o5_closure(W, d, comm, F.bxs_, 1); }
    if (F.o5 && F.tfsf) F.tfsf->o5_fix(comm, F.exs_, F.bxs_);    // the Cx stencil across the TFSF plane
    if (F.o5 && F.side) {
        const FV a = F.exs_.view(), b = F.bxs_.view(), sE = W.sE.view(), sB = W.sB.view();
        for_owned(d, [=] HD(int i, int j, int k) { a(i, j, k) += sE(i, j, k); b(i, j, k) += sB(i, j, k); });
        comm.exchange(d, {&F.exs_, &F.bxs_});
    }
    // inbound T at an open x edge (laser): the x guards of the transverse fields, used by both passes
    for (auto& h : hooks.inflow) h(F, t);
    if (F.medium) {                                  // O7 (medium.hpp): the step with a linear medium, GMRES on Ex^{n+1}, Bx^{n+1}
        if (F.o5 || F.tfsf || F.side) throw std::runtime_error("medium: closure = box, no TFSF plane or side tube (decision 84)");
        Medium& M = *F.medium;
        const FV T0 = M.jt[0].view(), T1 = M.jt[1].view(), T2 = M.jt[2].view();
        // the iterate's means, the medium current and the total current (exchanged): the inputs of a pass
        auto prepare = [&](const Field& X, const Field& Y) {
            box_means(d, Ex, Bx, X.view(), Y.view(), Exs, Bxs);
            comm.exchange(d, {&F.exs_, &F.bxs_});
            medium_current_cart(M, d, Ey, Ez, By, Bz, Jx, Jy, Jz, Exs, Bxs);
            comm.exchange(d, {&M.jt[0], &M.jt[1], &M.jt[2]});
        };
        MediumOps ops;
        ops.apply = [&](Field& X, Field& Y, Field& gx, Field& gy) {
            prepare(X, Y);
            if (P) irip_pml_pass(d, comm, *P, Ex, Ey, Ez, Bx, By, Bz, T0, T1, T2, Exs, Bxs, F.ney_, F.nez_, F.nby_, F.nbz_, gx.view(), gy.view(), false);
            else box_pass(d, comm, Ex, Ey, Ez, Bx, By, Bz, T0, T1, T2, Exs, Bxs, F.ney_, F.nez_, F.nby_, F.nbz_, gx.view(), gy.view());
        };
        ops.precond = [&](Field& f, int kind) { solve(W, d, comm, f, kind, false); };
        {                                            // the floor of the solve's residual scale: the norm of the field state
            const int nx = d.n[0], ny = d.n[1];
            double s = par_sum(d.owned(), [=] HD(long q) {
                const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
                return Ex(i, j, k) * Ex(i, j, k) + Ey(i, j, k) * Ey(i, j, k) + Ez(i, j, k) * Ez(i, j, k) + Bx(i, j, k) * Bx(i, j, k) +
                       By(i, j, k) * By(i, j, k) + Bz(i, j, k) * Bz(i, j, k);
            });
            comm.allreduce_sum(&s, 1);
            M.fnorm = std::sqrt(s);
        }
        if (M.bound) medium_bound_solve(M, d, comm, t, [&] { medium_solve(M, d, comm, ops, F.ex, F.bx, t); },
                                        [&] { prepare(M.sx, M.sy); });
        else medium_solve(M, d, comm, ops, F.ex, F.bx, t);
        W.rE.swap(M.sx); W.rB.swap(M.sy);            // the solution (irip_residual compares the corrector with it)
        prepare(W.rE, W.rB);
        if (et) {                                    // O6 trial: Et with the medium current's own source, the mean B
            box_pass(d, comm, Ex, Ey, Ez, Bx, By, Bz, T0, T1, T2, Exs, Bxs, F.ney_, F.nez_, F.nby_, F.nbz_, M.gx.view(), M.gy.view());
            irip_work_field(d, Ey, Ez, By, Bz, T1, T2, Exs, Bxs, et[0].view(), et[1].view(), et[2].view());
            mean_b(d, Bx, By, Bz, M.gy.view(), F.nby_.view(), F.nbz_.view(), bb[0].view(), bb[1].view(), bb[2].view());
            comm.exchange(d, {&et[0], &et[1], &et[2], &bb[0], &bb[1], &bb[2]});
            return;
        }
        if (P) irip_pml_pass(d, comm, *P, Ex, Ey, Ez, Bx, By, Bz, T0, T1, T2, Exs, Bxs, F.ney_, F.nez_, F.nby_, F.nbz_, Ex, Bx, true);
        else box_pass(d, comm, Ex, Ey, Ez, Bx, By, Bz, T0, T1, T2, Exs, Bxs, F.ney_, F.nez_, F.nby_, F.nbz_, Ex, Bx);
        F.ey.swap(F.ney_); F.ez.swap(F.nez_); F.by.swap(F.nby_); F.bz.swap(F.nbz_);
        comm.exchange(d, {&F.ex, &F.bx});
        medium_commit(M, d, comm);
        if (M.bound) medium_bound_commit(M, d, comm);
        return;
    }
    // with CPML layers psi is only read here (committed in the final pass)
    if (P) irip_pml_pass(d, comm, *P, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, Exs, Bxs, F.ney_, F.nez_, F.nby_, F.nbz_, RE, RB, false, F.tfsf, 2, F.side);
    else box_pass(d, comm, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, Exs, Bxs, F.ney_, F.nez_, F.nby_, F.nbz_, RE, RB, F.tfsf, 2, F.side);
    // 2. Ex^{n+1}, Bx^{n+1}
    solve(W, d, comm, W.rE, 0, F.o5);
    solve(W, d, comm, W.rB, 1, F.o5);
    // 3. the step with the box averages; the corrector writes Ex, Bx in place (each cell reads only itself)
    for_owned(d, [=] HD(int i, int j, int k) {
        Exs(i, j, k) = 0.5 * (Ex(i, j, k) + RE(i, j, k));
        Bxs(i, j, k) = 0.5 * (Bx(i, j, k) + RB(i, j, k));
    });
    comm.exchange(d, {&F.exs_, &F.bxs_});
    if (F.o5) { o5_closure(W, d, comm, F.exs_, 0); o5_closure(W, d, comm, F.bxs_, 1); }
    if (F.o5 && F.tfsf) F.tfsf->o5_fix(comm, F.exs_, F.bxs_);    // the Cx stencil across the TFSF plane
    if (F.o5 && F.side) {
        const FV a = F.exs_.view(), b = F.bxs_.view(), sE = W.sE.view(), sB = W.sB.view();
        for_owned(d, [=] HD(int i, int j, int k) { a(i, j, k) += sE(i, j, k); b(i, j, k) += sB(i, j, k); });
        comm.exchange(d, {&F.exs_, &F.bxs_});
    }
    if (et) {                                        // O6 trial: the second pass into scratch, then Et and the mean B
        box_pass(d, comm, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, Exs, Bxs, F.ney_, F.nez_, F.nby_, F.nbz_, RE, RB, nullptr, 2, nullptr);
        irip_work_field(d, Ey, Ez, By, Bz, Jy, Jz, Exs, Bxs, et[0].view(), et[1].view(), et[2].view());
        const FV b0 = bb[0].view(), b1 = bb[1].view(), b2 = bb[2].view(), nBy = F.nby_.view(), nBz = F.nbz_.view();
        for_owned(d, [=] HD(int i, int j, int k) {
            b0(i, j, k) = 0.5 * (Bx(i, j, k) + RB(i, j, k));
            b1(i, j, k) = 0.5 * (By(i, j, k) + nBy(i, j, k));
            b2(i, j, k) = 0.5 * (Bz(i, j, k) + nBz(i, j, k));
        });
        comm.exchange(d, {&et[0], &et[1], &et[2], &bb[0], &bb[1], &bb[2]});
        return;
    }
    if (P) irip_pml_pass(d, comm, *P, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, Exs, Bxs, F.ney_, F.nez_, F.nby_, F.nbz_, Ex, Bx, true, F.tfsf, 2, F.side);
    else box_pass(d, comm, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, Exs, Bxs, F.ney_, F.nez_, F.nby_, F.nbz_, Ex, Bx, F.tfsf, 2, F.side);
    F.ey.swap(F.ney_); F.ez.swap(F.nez_); F.by.swap(F.nby_); F.bz.swap(F.nbz_);
    comm.exchange(d, {&F.ex, &F.bx});
}

double irip_residual(const Fields& f, const Domain& d) {
    if (!f.irip) return 0;
    const std::vector<double> Ex = f.ex.host(), Bx = f.bx.host(), X = f.irip->rE.host(), Y = f.irip->rB.host();
    double r = 0;
    for (int k = 0; k < d.n[2]; ++k)
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i)
                r = std::max(r, std::max(std::abs(Ex[f.ex.idx(i, j, k)] - X[f.ex.idx(i, j, k)]), std::abs(Bx[f.ex.idx(i, j, k)] - Y[f.ex.idx(i, j, k)])));
    return r;
}
