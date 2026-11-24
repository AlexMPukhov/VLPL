// Self-field of a beam moving along z in RZ (species.self_field, docs/specs/rz_metal.md "Self-field"). The RZ Gauss law
// at the rho node (i+1/2, r_j) is dz Ez + mu_z D_m E_perp = rho (FieldsRZ::div_e). With Ez(i) = -(phi_{i+1/2} -
// phi_{i-1/2}) / (gamma^2 dz) and E_perp(i) = -G_m phibar(i), phibar(i) = (phi_{i-1/2} + phi_{i+1/2}) / 2, it reads
// (dzz / gamma^2 + mu_z^2 D_m G_m) phi = -rho. Per radial eigenmode of the TM operator (D_m G_m -> -lam, rz_tm_basis):
//   s (phi_{I-1} + phi_{I+1}) + b phi_I = -dz^2 rho_I,  s = 1/gamma^2 - dz^2 lam / 4,  b = -2/gamma^2 - dz^2 lam / 2
// (|b| > 2|s|), phi = 0 just beyond the Gauss cells [0, N - 1) of the open z axis. B = beta z x E. Host code, setup only.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "rz.hpp"

void rz_add_beam_field(FieldsRZ& f, const Field& rho, double beta, Comm& comm) {
    const Domain& d = f.dom();
    if (d.periodic[0]) throw std::runtime_error("self_field in geometry = rz needs an open z axis (boundary_z = open)");
    const int N = d.N[0], nz = d.n[0], nr = d.n[1], M = f.modes(), S = 2 * M - 1, o = d.off[0], NG = N - 1;
    const RadialGrid& g = f.grid();
    const double dz = d.h[0], gi2 = 1 - beta * beta;
    auto at = [&](long I, int k, int j) { return (size_t(I) * size_t(S) + size_t(k)) * size_t(nr) + size_t(j); };
    std::vector<double> R(size_t(N) * size_t(S) * size_t(nr), 0.0), P(R.size(), 0.0), Q(R.size(), 0.0);   // Q: phi / gamma_eff^2
    {
        const std::vector<double> h = rho.host();
        for (int k = 0; k < S; ++k)
            for (int j = 0; j < nr; ++j)
                for (int i = 0; i < nz; ++i) R[at(o + i, k, j)] = h[rho.idx(i, j, k)];
    }
    comm.allreduce_sum(R.data(), int(R.size()));
    std::vector<double> V, Vi, lam, X(size_t(nr) * size_t(NG > 0 ? NG : 1)), cp(size_t(NG > 0 ? NG : 1));
    std::vector<double> G(static_cast<size_t>(nr));                 // 1 / gamma_eff^2 per radial eigenmode
    for (int m = 0; m < M; ++m) {
        rz_tm_basis(g, m, V, Vi, lam);
        double lmax = 0;
        for (double l : lam) lmax = std::max(lmax, l);
        for (int c = 0; c < (m ? 2 : 1); ++c) {
            const int k = m == 0 ? 0 : 2 * m - 1 + c;
            for (int I = 0; I < NG; ++I)
                for (int a = 0; a < nr; ++a) {
                    double s = 0;
                    for (int j = 0; j < nr; ++j) s += Vi[size_t(a) * nr + j] * R[at(I, k, j)];
                    X[size_t(a) * NG + I] = -dz * dz * s;
                }
            for (int a = 0; a < nr; ++a) {
                double* x = &X[size_t(a) * NG];
                const double l = lam[size_t(a)];
                // the dropped axis node; with numerics.rz_axis_cut also the rod nodes of the mode (their eigenvalues are 0 up to
                // the round-off of the eigensolver: a near-singular line would amplify it)
                if (m >= 1 && l <= 1e-12 * lmax) { for (int I = 0; I < NG; ++I) x[I] = 0; continue; }
                // 1 / gamma^2 clamped at dz^2 lam / 4 (s >= 0), as in Cartesian (src/beam_field.cpp, decision 95): beyond it the
                // pancake is thinner than dz / 2 and s < 0 has undamped zigzags, a field tail ahead of the beam
                // explicit RIP-CC only (decision 99): iRZ keeps 1 / gamma^2, the clamp made its Fig. 4 beam jump 1 -> 12 nm
                const double ge = d.solver == Solver::RipCC ? std::max(gi2, 0.25 * dz * dz * l) : gi2;
                G[size_t(a)] = ge;
                const double s = ge - 0.25 * dz * dz * l, b = -2 * ge - 0.5 * dz * dz * l;
                // Thomas, Dirichlet at both ends
                double den = b;
                cp[0] = s / den; x[0] /= den;
                for (int I = 1; I < NG; ++I) {
                    den = b - s * cp[size_t(I - 1)];
                    cp[size_t(I)] = s / den;
                    x[I] = (x[I] - s * x[I - 1]) / den;
                }
                for (int I = NG - 2; I >= 0; --I) x[I] -= cp[size_t(I)] * x[I + 1];
            }
            for (int I = 0; I < NG; ++I)
                for (int j = 0; j < nr; ++j) {
                    double s = 0;
                    for (int a = 0; a < nr; ++a) s += V[size_t(j) * nr + a] * X[size_t(a) * NG + I];
                    P[at(I, k, j)] = s;
                    double q = 0;
                    for (int a = 0; a < nr; ++a) q += V[size_t(j) * nr + a] * G[size_t(a)] * X[size_t(a) * NG + I];
                    Q[at(I, k, j)] = q;
                }
        }
    }
    // phi at the rho node I + 1/2 (0 outside the Gauss cells), phibar at the integer column I; node value 0 beyond the
    // wall and, m >= 1, on the axis (as the operators)
    auto phi = [&](long I, int k, int j) { return I < 0 || I >= NG || j >= nr ? 0.0 : P[at(I, k, j)]; };
    auto gphi = [&](long I, int k, int j) { return I < 0 || I >= NG || j >= nr ? 0.0 : Q[at(I, k, j)]; };
    auto pb = [&](long I, int k, int j, int m) { return m >= 1 && j == 0 ? 0.0 : 0.5 * (phi(I - 1, k, j) + phi(I, k, j)); };
    Field* F[6] = {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp};
    std::vector<double> h[6];
    for (int c = 0; c < 6; ++c) h[c] = F[c]->host();
    for (int m = 0; m < M; ++m) {
        const int k0 = m == 0 ? 0 : 2 * m - 1;
        for (int i = 0; i < nz; ++i) {
            const long I = o + i;
            for (int j = 0; j < nr; ++j) {
                double ez[2], er[2], ep[2] = {0, 0};
                for (int c = 0; c < (m ? 2 : 1); ++c) {
                    const int k = k0 + c;
                    ez[c] = m >= 1 && j == 0 ? 0.0 : -(gphi(I, k, j) - gphi(I - 1, k, j)) / dz;
                    er[c] = -(pb(I, k, j + 1, m) - pb(I, k, j, m)) / g.dn[size_t(j)];
                }
                if (m >= 1 && j > 0) {                     // -(i m / r) phibar
                    const double s = g.mf(m) / g.r[size_t(j)];
                    ep[0] = s * pb(I, k0 + 1, j, m); ep[1] = -s * pb(I, k0, j, m);
                }
                for (int c = 0; c < (m ? 2 : 1); ++c) {
                    const size_t q = f.ez.idx(i, j, k0 + c);
                    h[0][q] += ez[c]; h[1][q] += er[c]; h[2][q] += ep[c];
                    h[5][q] += beta * er[c]; h[4][q] -= beta * ep[c];
                }
            }
        }
    }
    for (int c = 0; c < 6; ++c) F[c]->upload(h[c]);
    comm.exchange(d, {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp});
}
