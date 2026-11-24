// iRIP (box closure) in RZ per azimuthal mode (docs/specs/rz_modes.md, decision 43). As in Cart3D
// (src/irip.cpp): pass 1 with Ez*, Bz* = Ez/2, Bz/2 gives R_E, R_B; (I - D^2/4 mu_z^2 L_m) X = R per mode
// and kind; pass 2 with the box averages. L_m is the radial operator of the mode, TM on Ez (nodes):
// psi_x(phi_phi(.), phi_r(.)), TE on Bz (half nodes): phi_x(psi_phi(.), psi_r(.)) (research/irip_rz), real
// and tridiagonal. For m >= 1 the axis node of Ez is not a degree of freedom (Ez = 0): its row and column
// are dropped. L = Dg S Dg^-1 (diagonal similarity, S symmetric tridiagonal), S = Q Lambda Q^T once per
// mode; per step the radial transform, the system along z per radial mode (cyclic for periodic z,
// zeros outside for open z), back. Ranks along z: the partition method (solve_partitioned).
// O5 (numerics.closure = o5, decision 74, docs/specs/overdense.md "O5 in RZ"): Ez*, Bz* <- Cz C_perp Ez*, Bz* in both
// passes, Cz = 1 - (1/3) dz^2 delta_z^2, C_perp per radial eigenmode a of the kind (TM basis for Ez*, TE for Bz*):
//   g_a = 1 + (2/3) min(D^2 lam_a / 4, 1) + h_a^2 lam_a / 12 (the cap of the time-step part: decision 77),   h_a^2 = sum_j w_j v_ja^2 h_j^2 / sum_j w_j v_ja^2
// (w, h: A_j, dd_j at the nodes for TM, B_j, dn_j at the half nodes for TE; uniform grid: h_a = dr). The system along z
// per radial mode becomes pentadiagonal and factors into two tridiagonal line solves as in Cart3D (src/irip.cpp, O5Coef).
// With the TFSF plane (decision 78) the incident grid runs o5 too, and the Cz stencil of the main grid across the plane gets
// the incident value of the far cell (RzTfsf::o5_fix), as in Cart3D (decision 75).
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include "rz.hpp"
#include "medium.hpp"
#include "tridiag_eigen.hpp"

struct RZWork {
    Field X, Y;                               // R_E, R_B, then Ez^{n+1}, Bz^{n+1}
    Field T;                                  // radial transform scratch
    Buf<double> V[2], Vi[2], lam[2];          // per kind: M blocks of Nr x Nr (V, Vi) and Nr (lam)
    Buf<double> gf[2];                        // O5: per kind the closure factor g_a (M blocks of Nr)
    Buf<double> g;                            // Thomas scratch, nz per line
    // z split over P ranks (partitioned solve, as solve_partitioned in src/irip.cpp): per line the local
    // solutions u, v, A'^-1 w (3 nz), the edge values (ex: this rank's, ga: all ranks'), the reduced systems
    Buf<double> uvw, ex, ga, red;
    std::vector<double> exh, gah;
};

namespace {
using C = std::complex<double>;
// The radial operators on host vectors of mode m (the formulas of src/rz_fields.cpp)
struct HostOps {
    int n, m;
    const RadialGrid& g;
    double r(int j) const { return g.r[size_t(j)]; }
    double rh(int j) const { return g.rh[size_t(j)]; }
    C at(const std::vector<C>& v, int j) const { return j >= 0 && j < n ? v[size_t(j)] : C(0); }
    std::vector<C> psi_x(const std::vector<C>& Bp, const std::vector<C>& Br) const {
        std::vector<C> o(static_cast<size_t>(n));
        o[0] = m == 0 ? rh(0) * Bp[0] / g.A[0] : C(0);
        for (int j = 1; j < n; ++j)
            o[size_t(j)] = (rh(j) * Bp[size_t(j)] - rh(j - 1) * Bp[size_t(j - 1)]) / g.A[size_t(j)] - C(0, g.mf(m) / r(j)) * Br[size_t(j)];
        return o;
    }
    std::vector<C> phi_x(const std::vector<C>& Ep, const std::vector<C>& Er) const {
        std::vector<C> o(static_cast<size_t>(n));
        for (int j = 0; j < n; ++j)
            o[size_t(j)] = -((r(j + 1) * at(Ep, j + 1) - r(j) * Ep[size_t(j)]) / g.B[size_t(j)]) + C(0, g.mf(m) / rh(j)) * Er[size_t(j)];
        return o;
    }
    std::vector<C> psi_r(const std::vector<C>& Bz) const {
        std::vector<C> o(static_cast<size_t>(n));
        for (int j = 0; j < n; ++j) o[size_t(j)] = C(0, g.mf(m) / rh(j)) * Bz[size_t(j)];
        return o;
    }
    std::vector<C> psi_t(const std::vector<C>& Bz) const {
        std::vector<C> o(static_cast<size_t>(n));
        o[0] = m == 1 ? -Bz[0] / rh(0) : C(0);
        for (int j = 1; j < n; ++j) o[size_t(j)] = -(Bz[size_t(j)] - Bz[size_t(j - 1)]) / g.dd[size_t(j)];
        return o;
    }
    std::vector<C> phi_r(const std::vector<C>& Ez) const {
        std::vector<C> o(static_cast<size_t>(n));
        o[0] = m == 1 ? C(0, -g.mf(1) / r(1)) * at(Ez, 1) : C(0);
        for (int j = 1; j < n; ++j) o[size_t(j)] = C(0, -g.mf(m) / r(j)) * Ez[size_t(j)];
        return o;
    }
    std::vector<C> phi_t(const std::vector<C>& Ez) const {
        std::vector<C> o(static_cast<size_t>(n));
        for (int j = 0; j < n; ++j) o[size_t(j)] = (at(Ez, j + 1) - Ez[size_t(j)]) / g.dn[size_t(j)];
        return o;
    }
};

// Eigenbasis of the operator of kind (0: TM on Ez, 1: TE on Bz) for mode m: V, Vi (n x n row-major), lam
// = eigenvalues of -L (>= 0).
// h2 (optional): the mode-weighted squared spacing h_a^2 of the O5 closure.
void basis(const RadialGrid& g, int m, int kind, std::vector<double>& V, std::vector<double>& Vi, std::vector<double>& lam,
           std::vector<double>* h2 = nullptr) {
    const int n = g.nr;
    const HostOps op{n, m, g};
    std::vector<double> L(size_t(n) * n, 0.0);
    const int j0 = kind == 0 && m >= 1 ? 1 : 0;                           // TM, m >= 1: no axis DOF
    for (int c = j0; c < n; ++c) {
        std::vector<C> e(size_t(n), C(0));
        e[size_t(c)] = 1;
        const std::vector<C> o = kind == 0 ? op.psi_x(op.phi_t(e), op.phi_r(e)) : op.phi_x(op.psi_t(e), op.psi_r(e));
        for (int r = j0; r < n; ++r) {
            if (std::abs(o[size_t(r)].imag()) > 1e-9 * (1 + std::abs(o[size_t(r)].real())))
                throw std::runtime_error("rz iRIP: radial operator not real");
            L[size_t(r) * n + c] = o[size_t(r)].real();
        }
    }
    if (g.jc(m) >= 0) {                                                  // numerics.rz_axis_cut: the rod of mode m
        const int lim = kind == 0 ? g.jc(m) : g.jc(m) - 1;              // TM: nodes j <= jc; TE: half nodes j < jc
        for (int q = 0; q <= lim && q < n; ++q)
            for (int p = 0; p < n; ++p) { L[size_t(q) * n + p] = 0; L[size_t(p) * n + q] = 0; }
    }
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            if (std::abs(r - c) > 1 && L[size_t(r) * n + c] != 0.0) throw std::runtime_error("rz iRIP: radial operator not tridiagonal");
    // diagonal similarity: L = Dg S Dg^-1 with S_{j,j+1} = S_{j+1,j} = sqrt(L_{j,j+1} L_{j+1,j})
    std::vector<double> dg(size_t(n), 1.0), d(static_cast<size_t>(n)), e(size_t(n > 1 ? n - 1 : 0), 0.0);
    for (int j = 0; j + 1 < n; ++j) {
        const double up = L[size_t(j) * n + j + 1], lo = L[size_t(j + 1) * n + j];
        if (up * lo < 0) throw std::runtime_error("rz iRIP: radial operator not symmetrisable");
        if (up != 0 && lo != 0) { dg[size_t(j + 1)] = dg[size_t(j)] * std::sqrt(lo / up); e[size_t(j)] = std::sqrt(up * lo); }
        else dg[size_t(j + 1)] = 1.0;                                     // decoupled (the dropped axis node)
    }
    for (int j = 0; j < n; ++j) d[size_t(j)] = L[size_t(j) * n + j];
    std::vector<double> Q;
    tridiag_eigen(n, d, e, Q);
    V.resize(size_t(n) * n); Vi.resize(size_t(n) * n); lam.resize(size_t(n));
    for (int q = 0; q < n; ++q) lam[size_t(q)] = std::max(-d[size_t(q)], 0.0);
    for (int j = 0; j < n; ++j)
        for (int q = 0; q < n; ++q) {
            V[size_t(j) * n + q] = dg[size_t(j)] * Q[size_t(j) * n + q];
            Vi[size_t(q) * n + j] = Q[size_t(j) * n + q] / dg[size_t(j)];
        }
    if (h2) {
        h2->assign(size_t(n), 0.0);
        for (int q = 0; q < n; ++q) {
            double num = 0, den = 0;
            for (int j = j0; j < n; ++j) {
                const double v = V[size_t(j) * n + q], w = kind == 0 ? g.A[size_t(j)] : g.B[size_t(j)];
                const double h = kind == 0 ? g.dd[size_t(j)] : g.dn[size_t(j)];
                num += w * v * v * h * h; den += w * v * v;
            }
            (*h2)[size_t(q)] = den > 0 ? num / den : 0.0;
        }
    }
}

// The line along z of radial mode index mi = m nr + a: b x_i + s (x_{i-1} + x_{i+1}) = r_i. Box: kappa^2 = D^2 lam / 4,
// b = 1 + kappa^2 / 2, s = kappa^2 / 4. O5 (gf): -(kap/12)(S - r1)(S - r2), kap = D^2/4 lam g_a, r = (3 +- sqrt(49 + 48/kap))/2,
// pass 0 the factor -(kap/12)(S - r2), pass 1 (S - r1) (src/irip.cpp O5Coef). False: the identity (lam = 0).
HD inline bool line_coef(double D, const double* lam, const double* gf, long mi, int pass, double& b, double& s) {
    if (!gf) {
        const double k2 = D * D * lam[mi] / 4;
        b = 1 + 0.5 * k2; s = 0.25 * k2;
        return s != 0;
    }
    const double kap = D * D / 4 * lam[mi] * gf[mi];
    if (kap == 0) { b = 1; s = 0; return false; }
    const double q = sqrt(49 + 48 / kap);
    if (pass == 0) { b = kap * (3 - q) / 24; s = -kap / 12; }
    else { b = -(3 + q) / 2; s = 1; }
    return true;
}

// Thomas for b x_i + s (x_{i-1} + x_{i+1}) = r_i, first / last diagonal b0 / bn, stride st; g: scratch.
HD inline void thomas_line(double* x, long st, int n, double b, double b0, double bn, double s, double* g) {
    double bet = b0;
    x[0] /= bet;
    for (int i = 1; i < n; ++i) {
        g[i] = s / bet;
        bet = (i == n - 1 ? bn : b) - s * g[i];
        x[i * st] = (x[i * st] - s * x[(i - 1) * st]) / bet;
    }
    for (int i = n - 2; i >= 0; --i) x[i * st] -= g[i + 1] * x[(i + 1) * st];
}

// The systems along z of all lines (radial mode a, slot k) with z split over P = comm.dims[0] ranks: the
// partition method of solve_partitioned (src/irip.cpp, there for complex lines), here for real lines.
// Per rank T y = r, T u = -s e_0, T v = -s e_n-1 and, periodic z, T y_w = w (the Sherman-Morrison vector
// gam e_0 + s e_N-1 of the global cyclic matrix: e_0 on the first rank, e_N-1 on the last); the interface
// values I_q = (R_{q-1}, L_q) solve a block-tridiagonal system of 2 x 2 blocks from the edge values of
// all ranks (allgather, 8 per line and rank), solved for all lines on every rank, two right-hand sides
// (r, w).
void solve_partitioned(RZWork& W, const Domain& d, Comm& comm, const FV T, int kind, const double* gf, int pass) {
    const int nz = d.n[0], nr = d.n[1], S = d.n[2], P = comm.dims[0], pr = comm.coords[0];
    const long L = long(nr) * S;
    const double D = d.h[0];
    if (W.ex.size() != size_t(8 * L)) {
        W.uvw.resize(size_t(3 * L * nz)); W.ex.resize(size_t(8 * L));
        W.ga.resize(size_t(8 * L * P)); W.red.resize(size_t(8 * L * (P - 1)));
        W.exh.resize(size_t(8 * L)); W.gah.resize(size_t(8 * L * P));
    }
    const double* lam = W.lam[kind].data();
    double* g = W.g.data();
    double* uvw = W.uvw.data();
    double* ex = W.ex.data();
    const bool cyc = d.periodic[0];
    const bool first = cyc && pr == 0, last = cyc && pr == P - 1;
    // A. local solves: y in place, u, v, y_w in uvw; the edge values
    par_for(L, [=] HD(long q) {
        const int a = int(q % nr), k = int(q / nr), m = (k + 1) / 2;
        double b, s;
        const bool line = line_coef(D, lam, gf, long(m) * nr + a, pass, b, s);
        double* e = ex + 8 * q;
        for (int c = 0; c < 8; ++c) e[c] = 0;
        if (!line) return;                                            // the identity on every rank
        const double gam = -b, b0 = first ? b - gam : b, bn = last ? b - s * s / gam : b;
        double* x = &T(0, a, k);
        double* gl = g + 2 * q * nz;
        double* u = uvw + 3 * q * nz;
        double* v = u + nz;
        double* w = v + nz;
        thomas_line(x, 1, nz, b, b0, bn, s, gl);
        for (int i = 0; i < nz; ++i) { u[i] = 0; v[i] = 0; w[i] = 0; }
        u[0] = -s; v[nz - 1] = -s;
        if (first) w[0] = gam;
        if (last) w[nz - 1] += s;
        thomas_line(u, 1, nz, b, b0, bn, s, gl);
        thomas_line(v, 1, nz, b, b0, bn, s, gl);
        thomas_line(w, 1, nz, b, b0, bn, s, gl);
        e[0] = u[0]; e[1] = u[nz - 1]; e[2] = v[0]; e[3] = v[nz - 1];
        e[4] = x[0]; e[5] = x[nz - 1]; e[6] = w[0]; e[7] = w[nz - 1];
    });
    // B. the edge values of all ranks
    W.ex.to_host(W.exh.data(), 0, W.exh.size());
    comm.allgather(W.exh.data(), int(8 * L), W.gah.data());
    W.ga.from_host(W.gah.data(), 0, W.gah.size());
    const double* ga = W.ga.data();
    double* red = W.red.data();
    // C. per line the reduced system (block Thomas), the global edge values for Sherman-Morrison, and the
    //    local solution
    par_for(L, [=] HD(long q) {
        const int a = int(q % nr), k = int(q / nr), m = (k + 1) / 2;
        double b, s;
        if (!line_coef(D, lam, gf, long(m) * nr + a, pass, b, s)) return;
        const double gam = -b;
        auto G = [&](int p, int c) { return ga[(long(p) * L + q) * 8 + c]; };
        double* sc = red + q * 8 * (P - 1);                           // block p - 1: D' (4), rhs' (2 x 2)
        for (int p = 1; p < P; ++p) {
            double* B = sc + 8 * (p - 1);
            B[0] = 1; B[1] = -G(p - 1, 3); B[2] = -G(p, 0); B[3] = 1;
            for (int c = 0; c < 2; ++c) { B[4 + 2 * c] = G(p - 1, 5 + 2 * c); B[5 + 2 * c] = G(p, 4 + 2 * c); }
            if (p > 1) {
                const double* A = B - 8;
                const double det = A[0] * A[3] - A[1] * A[2], l = -G(p - 1, 1);
                const double m0 = l * A[3] / det, m1 = -l * A[1] / det;
                B[1] -= m1 * (-G(p - 1, 2));
                for (int c = 0; c < 2; ++c) B[4 + 2 * c] -= m0 * A[4 + 2 * c] + m1 * A[5 + 2 * c];
            }
        }
        for (int p = P - 1; p >= 1; --p) {                            // back: I_p into the rhs' slots
            double* B = sc + 8 * (p - 1);
            const double det = B[0] * B[3] - B[1] * B[2];
            for (int c = 0; c < 2; ++c) {
                double r0 = B[4 + 2 * c], r1 = B[5 + 2 * c];
                if (p < P - 1) r1 += G(p, 2) * sc[8 * p + 5 + 2 * c];
                B[4 + 2 * c] = (B[3] * r0 - B[1] * r1) / det;
                B[5 + 2 * c] = (B[0] * r1 - B[2] * r0) / det;
            }
        }
        auto I = [&](int p, int kk, int c) { return sc[8 * (p - 1) + 4 + 2 * c + kk]; };   // kk 0: R_{p-1}, 1: L_p
        double Rl[2], Lr[2], x0[2], xl[2];
        for (int c = 0; c < 2; ++c) {
            Rl[c] = pr > 0 ? I(pr, 0, c) : 0.0;
            Lr[c] = pr < P - 1 ? I(pr + 1, 1, c) : 0.0;
            x0[c] = G(0, 4 + 2 * c) + G(0, 2) * I(1, 1, c);
            xl[c] = G(P - 1, 5 + 2 * c) + G(P - 1, 1) * I(P - 1, 0, c);
        }
        const double f = cyc ? (x0[0] + s * xl[0] / gam) / (1 + x0[1] + s * xl[1] / gam) : 0.0;
        double* x = &T(0, a, k);
        const double* u = uvw + 3 * q * nz;
        const double* v = u + nz;
        const double* w = v + nz;
        for (int i = 0; i < nz; ++i) {
            const double yw = w[i] + Rl[1] * u[i] + Lr[1] * v[i];
            x[i] += Rl[0] * u[i] + Lr[0] * v[i] - f * yw;
        }
    });
}

// (I - D^2/4 mu_z^2 L_m) X = R in place in F for all modes, kind 0 (E) or 1 (B); O5 (o5):
// (I - D^2/4 mu_z^2 Cz L_m C_perp) X = R, two line solves per radial mode.
void solve(RZWork& W, const Domain& d, Comm& comm, Field& F, int kind, bool o5) {
    const int nz = d.n[0], nr = d.n[1], S = d.n[2];
    const double D = d.h[0];
    const bool cyc = d.periodic[0];
    const FV R = F.view(), T = W.T.view();
    const double* V = W.V[kind].data();
    const double* Vi = W.Vi[kind].data();
    const double* lam = W.lam[kind].data();
    const double* gf = o5 ? W.gf[kind].data() : nullptr;
    double* g = W.g.data();
    const long nn = long(nr) * nr;
    // radial transform of every column (i, slot)
    par_for(long(nz) * S, [=] HD(long q) {
        const int i = int(q % nz), k = int(q / nz), m = (k + 1) / 2;
        const double* A = Vi + m * nn;
        for (int a = 0; a < nr; ++a) {
            double s = 0;
            for (int j = 0; j < nr; ++j) s += A[long(a) * nr + j] * R(i, j, k);
            T(i, a, k) = s;
        }
    });
    // per (radial mode a, slot k) the system along z (stride 1 in the Field layout)
    for (int pass = 0; pass < (o5 ? 2 : 1); ++pass)
    if (comm.dims[0] > 1) solve_partitioned(W, d, comm, T, kind, gf, pass);
    else par_for(long(nr) * S, [=] HD(long q) {
        const int a = int(q % nr), k = int(q / nr), m = (k + 1) / 2;
        double b, s;
        if (!line_coef(D, lam, gf, long(m) * nr + a, pass, b, s)) return;
        double* x = &T(0, a, k);
        double* gl = g + 2 * q * nz;                                  // two blocks of nz per line
        if (!cyc) { thomas_line(x, 1, nz, b, b, b, s, gl); return; }
        // cyclic (Sherman-Morrison): A' = A - u v^T corrections at the corners; the correction vector in
        // the second half of the scratch line
        const double gam = -b, b0 = b - gam, bn = b - s * s / gam;
        double* z = gl + nz;                                          // scratch: second nz block
        thomas_line(x, 1, nz, b, b0, bn, s, gl);
        for (int i = 0; i < nz; ++i) z[i] = 0;
        z[0] = gam; z[nz - 1] = s;
        thomas_line(z, 1, nz, b, b0, bn, s, gl);
        const double f = (x[0] + s * x[nz - 1] / gam) / (1 + z[0] + s * z[nz - 1] / gam);
        for (int i = 0; i < nz; ++i) x[i] -= f * z[i];
    });
    // back
    par_for(long(nz) * S, [=] HD(long q) {
        const int i = int(q % nz), k = int(q / nz), m = (k + 1) / 2;
        const double* A = V + m * nn;
        for (int j = 0; j < nr; ++j) {
            double s = 0;
            for (int a = 0; a < nr; ++a) s += A[long(j) * nr + a] * T(i, a, k);
            R(i, j, k) = s;
        }
    });
}
// O5: F <- Cz C_perp F on the owned cells (guards valid on entry), then exchanged. Per column (i, slot) the radial
// transform of Cz F, times g_a, back.
// cz = false: C_perp only (the incident grid of the TFSF plane, whose Cz is applied across the plane by RzTfsf)
void o5_closure(RZWork& W, const Domain& d, Comm& comm, Field& f, int kind, bool cz = true) {
    const int nz = d.n[0], nr = d.n[1], S = d.n[2];
    const FV F = f.view(), T = W.T.view();
    const double* V = W.V[kind].data();
    const double* Vi = W.Vi[kind].data();
    const double* gf = W.gf[kind].data();
    const long nn = long(nr) * nr;
    par_for(long(nz) * S, [=] HD(long q) {
        const int i = int(q % nz), k = int(q / nz), m = (k + 1) / 2;
        const double* A = Vi + m * nn;
        for (int a = 0; a < nr; ++a) {
            double s = 0;
            for (int j = 0; j < nr; ++j)
                s += A[long(a) * nr + j] * (cz ? F(i, j, k) - (F(i + 1, j, k) - 2 * F(i, j, k) + F(i - 1, j, k)) / 3 : F(i, j, k));
            T(i, a, k) = gf[long(m) * nr + a] * s;
        }
    });
    par_for(long(nz) * S, [=] HD(long q) {
        const int i = int(q % nz), k = int(q / nz), m = (k + 1) / 2;
        const double* A = V + m * nn;
        for (int j = 0; j < nr; ++j) {
            double s = 0;
            for (int a = 0; a < nr; ++a) s += A[long(j) * nr + a] * T(i, a, k);
            F(i, j, k) = s;
        }
    });
    comm.exchange(d, {&f});
}
}  // namespace

void rz_o5_closure(FieldsRZ& f, Comm& comm, Field& x, int kind, bool cz) {
    if (!f.o5 || !f.irip_) throw std::runtime_error("rz_o5_closure: an iRIP grid with closure = o5, after its first step");
    o5_closure(*f.irip_, f.dom(), comm, x, kind, cz);
}

void rz_basis(const RadialGrid& g, int m, int kind, std::vector<double>& V, std::vector<double>& Vi, std::vector<double>& lam,
              std::vector<double>& h2) {
    basis(g, m, kind, V, Vi, lam, &h2);
}

void rz_tm_basis(const RadialGrid& g, int m, std::vector<double>& V, std::vector<double>& Vi, std::vector<double>& lam) {
    basis(g, m, 0, V, Vi, lam);
}

double rz_lambda_max_tm(const RadialGrid& g, int m) {
    std::vector<double> V, Vi, lam;
    basis(g, m, 0, V, Vi, lam);
    double lm = 0;
    for (double l : lam) lm = std::max(lm, l);
    return lm;
}

double rz_lambda_max(const RadialGrid& g, int modes) {
    double lm = 0;
    std::vector<double> V, Vi, lam;
    for (int m = 0; m < modes; ++m)
        for (int kind = 0; kind < 2; ++kind) {
            basis(g, m, kind, V, Vi, lam);
            for (double l : lam) lm = std::max(lm, l);
        }
    return lm;
}

namespace {
// the work arrays and radial bases of the iRIP grid f (once)
void ensure_work(FieldsRZ& f, std::shared_ptr<RZWork>& irip) {
    const Domain& d = f.dom();
    const int M = f.modes();
    if (!irip) {
        if (d.n[0] < 3) throw std::runtime_error("rz iRIP: at least 3 cells along z per rank");
        auto W = std::make_shared<RZWork>();
        W->X.resize(d); W->Y.resize(d); W->T.resize(d);
        W->g.resize(size_t(2) * d.n[0] * d.n[1] * d.n[2]);
        for (int kind = 0; kind < 2; ++kind) {
            std::vector<double> V, Vi, lam, gf, vb, vib, lb, h2;
            const double D = d.h[0];
            for (int m = 0; m < M; ++m) {
                basis(f.grid(), m, kind, vb, vib, lb, &h2);
                V.insert(V.end(), vb.begin(), vb.end()); Vi.insert(Vi.end(), vib.begin(), vib.end());
                lam.insert(lam.end(), lb.begin(), lb.end());
                for (size_t a = 0; a < lb.size(); ++a) gf.push_back(1 + 2.0 / 3 * std::min(D * D * lb[a] / 4, 1.0) + h2[a] * lb[a] / 12);   // cap: decision 77
            }
            W->V[kind].upload(V); W->Vi[kind].upload(Vi); W->lam[kind].upload(lam);
            if (f.o5) W->gf[kind].upload(gf);
        }
        irip = W;
    }
}

struct Cr { double re, im; };
HD inline Cr operator+(Cr a, Cr b) { return {a.re + b.re, a.im + b.im}; }
HD inline Cr operator-(Cr a, Cr b) { return {a.re - b.re, a.im - b.im}; }
HD inline Cr operator*(double s, Cr a) { return {s * a.re, s * a.im}; }
HD inline Cr imr(double s, Cr a) { return {-s * a.im, s * a.re}; }                 // i s a
HD inline int slt(int m, int c) { return m == 0 ? 0 : 2 * m - 1 + c; }
HD inline Cr gt(const FV& F, int i, int j, int m) { return {F(i, j, slt(m, 0)), m ? F(i, j, slt(m, 1)) : 0.0}; }
HD inline void pt(const FV& F, int i, int j, int m, Cr v) { F(i, j, slt(m, 0)) = v.re; if (m) F(i, j, slt(m, 1)) = v.im; }
HD inline Cr mzr(const FV& F, int i, int j, int m) { return 0.5 * (gt(F, i, j, m) + gt(F, i + 1, j, m)); }
}  // namespace

static void rz_box_means(long cells, int nz, int nr, const FV Ez, const FV Bz, const FV X, const FV Y, const FV Exs, const FV Bxs) {
    par_for(cells, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
        Exs(i, j, k) = 0.5 * (Ez(i, j, k) + X(i, j, k)); Bxs(i, j, k) = 0.5 * (Bz(i, j, k) + Y(i, j, k));
    });
}

void rz_irip_step(FieldsRZ& f, Comm& comm, double t) {
    const Domain& d = f.dom();
    ensure_work(f, f.irip_);
    RZWork& W = *f.irip_;
    const int nz = d.n[0], nr = d.n[1], S = d.n[2];
    const FV Ez = f.ez.view(), Bz = f.bz.view(), Exs = f.exs_.view(), Bxs = f.bxs_.view(), X = W.X.view(), Y = W.Y.view();
    if (f.medium) {                                  // O7 (medium.hpp): GMRES on Ez^{n+1}, Bz^{n+1} of all modes
        if (f.o5 || f.tfsf) throw std::runtime_error("medium: closure = box, no TFSF plane (decision 84)");
        Medium& Md = *f.medium;
        const long cells = long(nz) * nr * S;
        auto prepare = [&](const Field& Xf, const Field& Yf) {
            rz_box_means(cells, nz, nr, Ez, Bz, Xf.view(), Yf.view(), Exs, Bxs);
            comm.exchange(d, {&f.exs_, &f.bxs_});
            medium_current_rz(Md, d, f.grid().view(), f.modes(), f.er.view(), f.ep.view(), f.br.view(), f.bp.view(), f.jz.view(),
                              f.jr.view(), f.jp.view(), Exs, Bxs);
            comm.exchange(d, {&Md.jt[0], &Md.jt[1], &Md.jt[2]});
        };
        auto pass_jt = [&](Field& oz, Field& ob) {   // the pass with the total current (particles + medium)
            f.jz.swap(Md.jt[0]); f.jr.swap(Md.jt[1]); f.jp.swap(Md.jt[2]);
            f.pass(comm, false, oz, ob);
            f.jz.swap(Md.jt[0]); f.jr.swap(Md.jt[1]); f.jp.swap(Md.jt[2]);
        };
        MediumOps ops;
        ops.apply = [&](Field& Xf, Field& Yf, Field& gx, Field& gy) { prepare(Xf, Yf); pass_jt(gx, gy); };
        ops.precond = [&](Field& r, int kind) { solve(W, d, comm, r, kind, false); };
        {                                            // the floor of the solve's residual scale: the norm of the field state (all slots)
            const FV a = f.ez.view(), b = f.er.view(), c = f.ep.view(), e = f.bz.view(), g = f.br.view(), h = f.bp.view();
            const int nx = d.n[0], ny = d.n[1];
            double s = par_sum(d.owned(), [=] HD(long q) {
                const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
                return a(i, j, k) * a(i, j, k) + b(i, j, k) * b(i, j, k) + c(i, j, k) * c(i, j, k) + e(i, j, k) * e(i, j, k) +
                       g(i, j, k) * g(i, j, k) + h(i, j, k) * h(i, j, k);
            });
            comm.allreduce_sum(&s, 1);
            Md.fnorm = std::sqrt(s);
        }
        if (Md.bound) medium_bound_rz_solve(Md, d, comm, t, [&] { medium_solve(Md, d, comm, ops, f.ez, f.bz, t); },
                                            [&] { prepare(Md.sx, Md.sy); });
        else medium_solve(Md, d, comm, ops, f.ez, f.bz, t);
        W.X.swap(Md.sx); W.Y.swap(Md.sy);
        prepare(W.X, W.Y);
        pass_jt(f.ez, f.bz);
        f.er.swap(f.ner_); f.ep.swap(f.nep_); f.br.swap(f.nbr_); f.bp.swap(f.nbp_);
        comm.exchange(d, {&f.ez, &f.bz});
        medium_commit(Md, d, comm, &Md.weights);
        if (Md.bound) medium_bound_rz_commit(Md, d, comm, f.grid().view());
        return;
    }
    const long cells = long(nz) * nr * S;
    // 1. right-hand sides: the averages with Ez^{n+1} = Bz^{n+1} = 0
    par_for(cells, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
        Exs(i, j, k) = 0.5 * Ez(i, j, k); Bxs(i, j, k) = 0.5 * Bz(i, j, k);
    });
    comm.exchange(d, {&f.exs_, &f.bxs_});
    if (f.o5) {
        o5_closure(W, d, comm, f.exs_, 0); o5_closure(W, d, comm, f.bxs_, 1);
        if (f.tfsf) f.tfsf->o5_fix(comm, f.exs_, f.bxs_);   // the Cz stencil across the TFSF plane (decision 78)
    }
    f.pass(comm, false, W.X, W.Y);
    // 2. Ez^{n+1}, Bz^{n+1}
    solve(W, d, comm, W.X, 0, f.o5);
    solve(W, d, comm, W.Y, 1, f.o5);
    // 3. the step with the box averages (the corrector writes Ez, Bz in place)
    par_for(cells, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
        Exs(i, j, k) = 0.5 * (Ez(i, j, k) + X(i, j, k)); Bxs(i, j, k) = 0.5 * (Bz(i, j, k) + Y(i, j, k));
    });
    comm.exchange(d, {&f.exs_, &f.bxs_});
    if (f.o5) {
        o5_closure(W, d, comm, f.exs_, 0); o5_closure(W, d, comm, f.bxs_, 1);
        if (f.tfsf) f.tfsf->o5_fix(comm, f.exs_, f.bxs_);   // the Cz stencil across the TFSF plane (decision 78)
    }
    f.pass(comm, false, f.ez, f.bz);
    f.er.swap(f.ner_); f.ep.swap(f.nep_); f.br.swap(f.nbr_); f.bp.swap(f.nbp_);
    comm.exchange(d, {&f.ez, &f.bz});
}

// O6 (decision 82): the box step without committing it, and its work field. As in Cart3D (irip_trial, src/irip.cpp) with
// (x, y, z) -> (z, r, phi) per mode: the transport of T+- = E_r +- B_phi (along +-z) and U-+ = E_phi -+ B_r gives
//   Et_z   = Ez* (the box mean of pass 2)                                                         at (i, r_j)
//   Et_r   = [(E_r + B_phi)(a) + (E_r - B_phi)(a+1)] / 2 + D/2 Psi_r(a) - D/4 d_r [Ez*(a+1) - Ez*(a)]   at (a+1/2, r_{j+1/2})
//   Et_phi = [(E_phi - B_r)(a) + (E_phi + B_r)(a+1)] / 2 + D/2 Psi_phi(a) + D/4 G_phi [Ez*(a+1) - Ez*(a)] at (a+1/2, r_j)
// with Psi_r = (i m / rh) mu_z Bz* - J_r, Psi_phi = -d_r mu_z Bz* - J_phi (the sources of pass 2) and G_phi = -(i m / r) (the
// operator of Phi_r; axis rules as there). Old fields at t_n. Checked: W^{n+1} - W^n = -D dV sum J . Et (test M6).
void rz_irip_trial(FieldsRZ& f, Comm& comm, double t, Field* et, Field* bb) {
    const Domain& d = f.dom();
    if (f.o5 || f.tfsf || f.medium) throw std::runtime_error("rz_irip_trial: closure = box, no TFSF plane, no medium (decision 84)");
    comm.exchange(d, {&f.jz, &f.jr, &f.jp});
    if (f.axis_regular) rz_axis_regular_current(f);    // as FieldsRZ::step (decision 112)
    for (auto& h : f.inflow) h(f, t);
    ensure_work(f, f.irip_);
    RZWork& W = *f.irip_;
    const int nz = d.n[0], nr = d.n[1], S = d.n[2], M = f.modes();
    const double D = d.h[0];
    const FV Ez = f.ez.view(), Bz = f.bz.view(), Exs = f.exs_.view(), Bxs = f.bxs_.view(), X = W.X.view(), Y = W.Y.view();
    const long cells = long(nz) * nr * S;
    par_for(cells, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
        Exs(i, j, k) = 0.5 * Ez(i, j, k); Bxs(i, j, k) = 0.5 * Bz(i, j, k);
    });
    comm.exchange(d, {&f.exs_, &f.bxs_});
    f.pass(comm, false, W.X, W.Y);
    solve(W, d, comm, W.X, 0, false);
    solve(W, d, comm, W.Y, 1, false);
    par_for(cells, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
        Exs(i, j, k) = 0.5 * (Ez(i, j, k) + X(i, j, k)); Bxs(i, j, k) = 0.5 * (Bz(i, j, k) + Y(i, j, k));
    });
    comm.exchange(d, {&f.exs_, &f.bxs_});
    f.pass(comm, false, W.X, W.Y);                   // Ez^{n+1}, Bz^{n+1} into X, Y; the transverse fields into ner_ ..
    const RadialView g = f.grid().view();
    const FV Er = f.er.view(), Ep = f.ep.view(), Br = f.br.view(), Bp = f.bp.view(), Jr = f.jr.view(), Jp = f.jp.view();
    const FV nBr = f.nbr_.view(), nBp = f.nbp_.view();
    const FV Tz = et[0].view(), Tr = et[1].view(), Tp = et[2].view(), Hz = bb[0].view(), Hr = bb[1].view(), Hp = bb[2].view();
    par_for(long(nz) * nr * M, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), m = int(q / (long(nz) * nr));
        pt(Tz, i, j, m, gt(Exs, i, j, m));
        const Cr tr = 0.5 * (gt(Er, i, j, m) + gt(Bp, i, j, m) + gt(Er, i + 1, j, m) - gt(Bp, i + 1, j, m));
        const Cr psr = imr(g.mf(m) / g.rh[j], mzr(Bxs, i, j, m)) - gt(Jr, i, j, m);
        const Cr dex = (1 / g.dn[j]) * ((gt(Exs, i + 1, j + 1, m) - gt(Exs, i, j + 1, m)) - (gt(Exs, i + 1, j, m) - gt(Exs, i, j, m)));
        pt(Tr, i, j, m, tr + (0.5 * D) * psr - (0.25 * D) * dex);
        Cr ep{0, 0};
        if (j > 0) {
            const Cr tp = 0.5 * (gt(Ep, i, j, m) - gt(Br, i, j, m) + gt(Ep, i + 1, j, m) + gt(Br, i + 1, j, m));
            const Cr pst = (-1 / g.dd[j]) * (mzr(Bxs, i, j, m) - mzr(Bxs, i, j - 1, m)) - gt(Jp, i, j, m);
            const Cr gph = imr(-g.mf(m) / g.r[j], gt(Exs, i + 1, j, m) - gt(Exs, i, j, m));
            ep = tp + (0.5 * D) * pst + (0.25 * D) * gph;
        }
        pt(Tp, i, j, m, ep);                          // E_phi, B_r on the axis are not dynamic (no J_phi there)
        pt(Hz, i, j, m, 0.5 * (gt(Bz, i, j, m) + gt(Y, i, j, m)));
        pt(Hr, i, j, m, 0.5 * (gt(Br, i, j, m) + gt(nBr, i, j, m)));
        pt(Hp, i, j, m, 0.5 * (gt(Bp, i, j, m) + gt(nBp, i, j, m)));
    });
    comm.exchange(d, {&et[0], &et[1], &et[2], &bb[0], &bb[1], &bb[2]});
    if (f.axis_regular) rz_axis_regular_gather_e(f, et[1], et[2]);   // the gather of Et: adjoint of the projected current (decision 112)
}
