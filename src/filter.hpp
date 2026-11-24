// Current filter of Cart3D that keeps the Gauss law exact (numerics.filter_passes, docs/specs/solid_target.md,
// decision 64; the RZ version is src/rz_filter.cpp, decision 54). RIP-CC and iRIP staggering (charge nodes at
// (i + 1/2, j, k), Jx between nodes c - 1 and c stored at c, Jy, Jz between j and j + 1 stored at j).
#pragma once
#include "comm.hpp"
#include "fields.hpp"
#include "grid.hpp"

// Per pass three steps a = x, y, z: w = D J at the Gauss nodes (0 elsewhere), J_a += (h_a / 4) (w_+ - w_-) on the faces
// between two Gauss nodes. Then D S_a = T_a D with T_a the binomial (1, 2, 1) / 4 along a over the Gauss nodes
// (Neumann at the ends): the continuity of the filtered current is that of the filtered charge T rho, exactly.
// J must be reduced (add_guards); its guards are zero again afterwards. w: scratch of the layout of J.
void filter_currents_cart(Fields& f, Field& w, const Domain& d, Comm& comm, int passes);
// The binomial filter (numerics.filter = binomial, decision 91): per pass and axis a, J_a <- S_a J_a (the step above
// with w = d_a J_a only) and J_b <- T_a J_b for b != a (the binomial along a over the Gauss range of a, Neumann at its
// ends). In the interior every component gets T_x T_y T_z, as the charge: a streaming plasma keeps J = v rho at the grid
// scale (the projection and the local filter smooth only the longitudinal current, and a relativistic drift is then
// unstable). D J' = T D J exactly on any boundary and decomposition. J reduced; guards zero afterwards.
void filter_currents_binomial(Fields& f, Field& w, const Domain& d, Comm& comm, int passes);
// rho <- T_z T_y T_x rho, passes times (the charge the filtered current carries); rho reduced, guards zero afterwards.
void filter_rho_cart(Field& rho, Field& w, const Domain& d, Comm& comm, int passes);

// numerics.filter_flat = L (decision 109): fills Domain::filt_nflat, filt_fk, filt_fa, filt_fb with the real factors of
// P(S) = sum_{j<L} binom(N-1+j, j) S^j (N = passes; roots by Durand-Kerner, conjugate pairs combined). L = 0: none.
void set_flat_filter(Domain& d, int passes, int L);
// The response of N passes and the flat factor at S = sin^2(k h / 2), from the factors (tests).
double flat_filter_response(const Domain& d, int passes, double S);
// The flat factor on the fields X[0..nx) after the passes: step(cf) is one step of all of them with coefficient cf
// ((1 - cf S) along x, z in RZ). A pair (1 - a S + b S^2): x0, x1 = T x0, x2 = T x1, result (1 - a + b) x0 + (a - 2 b) x1
// + b x2 (S x0 = x0 - x1, S^2 x0 = x0 - 2 x1 + x2). Linear in the steps, so the Gauss law stays exact.
template <class Step> void flat_filter_steps(const Domain& d, Field* const* X, int nx, Step step) {
    static Field s0[3], s1[3];
    for (int s = 0; s < d.filt_nflat; ++s) {
        if (d.filt_fk[s] == 1) { step(d.filt_fa[s]); continue; }
        const double a = d.filt_fa[s], b = d.filt_fb[s];
        for (int i = 0; i < nx; ++i) s0[i].assign(*X[i]);
        step(1.0);
        for (int i = 0; i < nx; ++i) s1[i].assign(*X[i]);
        step(1.0);
        for (int i = 0; i < nx; ++i) X[i]->lincomb(1 - a + b, s0[i], a - 2 * b, s1[i], b);
    }
}

// The projection filter (numerics.filter = projection; docs/specs/solid_target.md, decision 69; the default until 91): per pass
// w = D J, r = (T - 1) w with T = T_z T_y T_x over the Gauss nodes, L psi = r with L = D G = dxx + mu_x^2 (dyy + dzz)
// over the Gauss nodes (Neumann at an open x edge), J <- J + G psi, G = (dx, mu_x^2 dy, mu_x^2 dz) the gradient of
// RIP-CC's Gauss law. Then D J' = T D J exactly, as with filter_currents_cart, but the change G psi only moves charge and
// radiates nothing in RIP-CC and iRIP (test S6), while the local filter changes one component at a time (J_a alone),
// which adds a grid-scale transverse current that radiates. y and z periodic, ranks along
// x only (the DFT in y, z and the line solves of src/xlines.hpp). W: its work arrays, made on first use.
struct FilterProjWork;
void filter_currents_proj(Fields& f, Field& w, std::shared_ptr<FilterProjWork>& W, const Domain& d, Comm& comm, int passes);
