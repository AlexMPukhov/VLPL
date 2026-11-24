// iRIP with CPML layers in y and z (docs/specs/implicit_rip.md, decision 41): the pass of the box step
// with the layers, and the eigenbasis of the transverse operator along a pml axis.
//
// With CPML every transverse derivative g becomes g + psi_new, psi_new = b psi + a g, a = b - 1 inside a
// layer: linear in the current values with the factor b, fixed per location. So the solve operator
// stays separable (research/irip_3d/pml_identity.py, to 1e-14):
//   L_E = (Cn Dm Ch Dp)_y + (...)_z   on Ex (nodes),   L_B = (Ch Dp Cn Dm)_y + (...)_z   on Bx (half nodes),
// Dp: node -> half node, Dm: half -> node, zero beyond the walls, Cn, Ch = b at the nodes and half nodes.
// Each 1D operator is C T with T symmetric tridiagonal: its eigenbasis V = C^(1/2) Q, V^-1 = Q^T C^(-1/2)
// with S = C^(1/2) T C^(1/2) = Q Lambda Q^T, computed once (QL with implicit shifts). b >= 1e-4 for iRIP
// (Pml's floor) keeps C^(+-1/2) within 1e2.
#include "irip.hpp"
#include <cmath>
#include <stdexcept>
#include "fields.hpp"
#include "pml.hpp"
#include "side.hpp"
#include "tfsf.hpp"
#include "tridiag_eigen.hpp"

namespace {
// CPML derivative g + psi_new; psi committed only in the final pass
struct CpI {
    const double *a, *b;
    int off;
    bool commit;
    HD double operator()(double g, const FV& P, int jl, int i, int j, int k) const {
        if (!a) return g;
        const double aa = a[off + jl];
        if (aa == 0.0) return g;
        const double psi = b[off + jl] * P(i, j, k) + aa * g;
        if (commit) P(i, j, k) = psi;
        return g + psi;
    }
};
HD inline double mux(const FV& f, int i, int j, int k) { return 0.5 * (f(i, j, k) + f(i + 1, j, k)); }

}  // namespace

void irip_axis_basis(const Domain& d, const Pml& P, int axis, int kind, std::vector<double>& V, std::vector<double>& Vi,
                     std::vector<double>& lam) {
    const int n = d.N[axis];
    const double h2 = d.h[axis] * d.h[axis];
    const std::vector<double> bn = P.b_n[axis].host(), bh = P.b_h[axis].host();
    // operator C T: E (nodes) C = Cn, T from Ch; B (half nodes) C = Ch, T from Cn (see the header)
    std::vector<double> C(static_cast<size_t>(n)), dg(static_cast<size_t>(n)), of(static_cast<size_t>(n > 1 ? n - 1 : 0));
    for (int j = 0; j < n; ++j) {
        if (kind == 0) {
            C[size_t(j)] = bn[size_t(j)];
            dg[size_t(j)] = -(bh[size_t(j)] + (j > 0 ? bh[size_t(j - 1)] : 0.0)) / h2;
            if (j + 1 < n) of[size_t(j)] = bh[size_t(j)] / h2;
        } else {
            C[size_t(j)] = bh[size_t(j)];
            dg[size_t(j)] = -(bn[size_t(j)] + (j + 1 < n ? bn[size_t(j + 1)] : 0.0)) / h2;
            if (j + 1 < n) of[size_t(j)] = bn[size_t(j + 1)] / h2;
        }
    }
    // S = C^(1/2) T C^(1/2)
    for (int j = 0; j < n; ++j) dg[size_t(j)] *= C[size_t(j)];
    for (int j = 0; j + 1 < n; ++j) of[size_t(j)] *= std::sqrt(C[size_t(j)] * C[size_t(j + 1)]);
    std::vector<double> Q;
    tridiag_eigen(n, dg, of, Q);
    lam.resize(size_t(n));
    V.resize(size_t(n) * n); Vi.resize(size_t(n) * n);
    for (int q = 0; q < n; ++q) lam[size_t(q)] = std::max(-dg[size_t(q)], 0.0);   // eigenvalues of -L >= 0
    for (int j = 0; j < n; ++j)
        for (int q = 0; q < n; ++q) {
            const double c = std::sqrt(C[size_t(j)]);
            V[size_t(j) * n + q] = c * Q[size_t(j) * n + q];          // V[j][q]: column q = mode q
            Vi[size_t(q) * n + j] = Q[size_t(j) * n + q] / c;         // Vi[q][j]
        }
}

void irip_pml_pass(const Domain& d, Comm& comm, Pml& P, const FV Ex, const FV Ey, const FV Ez, const FV Bx, const FV By,
                   const FV Bz, const FV Jx, const FV Jy, const FV Jz, const FV Exs, const FV Bxs, Field& ney, Field& nez,
                   Field& nby, Field& nbz, const FV oEx, const FV oBx, bool commit, Tfsf* tf, int kind, const SideTfsf* sd) {
    const double D = d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    const bool oy = P.cfg.on[1], oz = P.cfg.on[2];
    const CpI yn{oy ? P.a_n[1].data() : nullptr, oy ? P.b_n[1].data() : nullptr, d.off[1], commit};
    const CpI yh{oy ? P.a_h[1].data() : nullptr, oy ? P.b_h[1].data() : nullptr, d.off[1], commit};
    const CpI zn{oz ? P.a_n[2].data() : nullptr, oz ? P.b_n[2].data() : nullptr, d.off[2], commit};
    const CpI zh{oz ? P.a_h[2].data() : nullptr, oz ? P.b_h[2].data() : nullptr, d.off[2], commit};
    // transport sources at (a + 1/2), a = -1 .. n-1, once per location (each with its psi)
    const FV SPY = P.s_py.view(), SPZ = P.s_pz.view(), SFY = P.s_fy.view(), SFZ = P.s_fz.view();
    const FV TBY = P.t_bx_y.view(), TEY = P.t_ex_y.view(), TBZ = P.t_bx_z.view(), TEZ = P.t_ex_z.view();
    const int nx1 = d.n[0] + 1, ny = d.n[1];
    par_for(long(nx1) * d.n[1] * d.n[2], [=] HD(long q) {
        const int a = int(q % nx1) - 1, j = int((q / nx1) % ny), k = int(q / (long(nx1) * ny));
        const double gbxz = zn((mux(Bxs, a, j, k) - mux(Bxs, a, j, k - 1)) * iz, TBZ, k, a, j, k);
        const double gbxy = yn((mux(Bxs, a, j, k) - mux(Bxs, a, j - 1, k)) * iy, TBY, j, a, j, k);
        const double gexz = zh((mux(Exs, a, j, k + 1) - mux(Exs, a, j, k)) * iz, TEZ, k, a, j, k);
        const double gexy = yh((mux(Exs, a, j + 1, k) - mux(Exs, a, j, k)) * iy, TEY, j, a, j, k);
        SPY(a, j, k) = gbxz - Jy(a, j, k);
        SPZ(a, j, k) = -gbxy - Jz(a, j, k);
        SFY(a, j, k) = -gexz;
        SFZ(a, j, k) = gexy;
    });
    const FV nEy = ney.view(), nEz = nez.view(), nBy = nby.view(), nBz = nbz.view();
    for_owned(d, [=] HD(int i, int j, int k) {
        const double tpy = Ey(i - 1, j, k) + Bz(i - 1, j, k) + D * (SPY(i - 1, j, k) + SFZ(i - 1, j, k));
        const double tmy = Ey(i + 1, j, k) - Bz(i + 1, j, k) + D * (SPY(i, j, k) - SFZ(i, j, k));
        const double tpz = Ez(i + 1, j, k) + By(i + 1, j, k) + D * (SPZ(i, j, k) + SFY(i, j, k));
        const double tmz = Ez(i - 1, j, k) - By(i - 1, j, k) + D * (SPZ(i - 1, j, k) - SFY(i - 1, j, k));
        nEy(i, j, k) = 0.5 * (tpy + tmy);  nBz(i, j, k) = 0.5 * (tpy - tmy);
        nEz(i, j, k) = 0.5 * (tpz + tmz);  nBy(i, j, k) = 0.5 * (tpz - tmz);
    });
    if (tf) tf->correct(comm, ney, nez, nby, nbz, kind, commit);   // the TFSF plane, CPML memories as the pass
    if (sd) sd->fix_transport(nEy, nEz, nBy, nBz, kind == 2 && d.o5 ? 3 : kind);   // the side tube, both passes (o5: kind 3)
    comm.exchange(d, {&ney, &nez, &nby, &nbz});
    const FV CBZ = P.c_bz_y.view(), CEZ = P.c_ez_y.view(), CBY = P.c_by_z.view(), CEY = P.c_ey_z.view();
    for_owned(d, [=] HD(int i, int j, int k) {
        const double bz0 = 0.5 * (Bz(i, j, k) + nBz(i, j, k)), bzm = 0.5 * (Bz(i, j - 1, k) + nBz(i, j - 1, k));
        const double by0 = 0.5 * (By(i, j, k) + nBy(i, j, k)), bym = 0.5 * (By(i, j, k - 1) + nBy(i, j, k - 1));
        const double ez0 = 0.5 * (Ez(i, j, k) + nEz(i, j, k)), ezp = 0.5 * (Ez(i, j + 1, k) + nEz(i, j + 1, k));
        const double ey0 = 0.5 * (Ey(i, j, k) + nEy(i, j, k)), eyp = 0.5 * (Ey(i, j, k + 1) + nEy(i, j, k + 1));
        const double gbz = yn((bz0 - bzm) * iy, CBZ, j, i, j, k), gby = zn((by0 - bym) * iz, CBY, k, i, j, k);
        const double gez = yh((ezp - ez0) * iy, CEZ, j, i, j, k), gey = zh((eyp - ey0) * iz, CEY, k, i, j, k);
        oEx(i, j, k) = Ex(i, j, k) + D * (gbz - gby - Jx(i, j, k));
        oBx(i, j, k) = Bx(i, j, k) - D * (gez - gey);
    });
    if (sd) sd->fix_corr(oEx, oBx);
}
