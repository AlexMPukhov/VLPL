// O7, model = bound: the node step of a V-type Maxwell-Bloch atom (medium.hpp, docs/specs/linear_medium.md "Bound electrons",
// decision 85). Ground state 0, excited states 1, 2 (e.g. Rb 5S, 5P_1/2, 5P_3/2); scalar dipole along one axis; the carrier is
// resolved (no rotating-wave approximation); no relaxation, no ionization (yet). Code units: E in m c omega0 / e, time 1/omega0,
// densities in n_c. Hamiltonian per atom (units hbar omega0)
//     H(E) = H0' - E X',   H0' = diag(0, w1', w2'),   X'_0j = X'_j0 = a_j',
// a_j = d_j / (e a0 alpha) the Rabi frequency per unit field, polarization per atom P = sum_j cP_j' Re rho_0j, cP_j = 2 (d_j/(e a0))
// a0 k0. The medium polarization is n_a P, its current u = n_a (P^{n+1} - P^n) / dt.
//
// The step: rho^{n+1} = U rho^n U^+, U = (1 + i K)^-1 (1 - i K), K = (dt/2) H(Et), Et the work field of the box closure (decision
// 81). U commutes with H(Et), so tr(rho H(Et)) is conserved by the step and
//     ef n_a [tr(rho H0')]^{n+1} - ef n_a [tr(rho H0')]^n = Et n_a (P^{n+1} - P^n) = dt Et u,    ef = cP_j' / (2 a_j'),
// exactly, if ef is the same for all lines: the energy the field loses is the atoms' energy K_b = ef n_a sum_j w_j' p_j.
// Pre-warped frequencies w' = (2/dt) tan(w dt/2): the free precession is exact (the resonance of the scheme sits at w). With the
// field at the midpoint of the step, the Cayley step drives a resonant line with a_j' cos x_j (x_j = w_j dt/2) and its weak-field
// susceptibility (P^n against that field) is a_j' cP_j' cos x_j / (a_j cP_j) of the exact one at the line. Hence
//     a_j' = a_j / cos x_j (the Rabi frequency exact),  cP_j' = cP_j cos(xbar) / cos x_j (chi at the line exact to
//     O(x (x_j - xbar)): 2e-4 for Rb D1, D2 at lambda/30),  ef = alpha a0 k0 cos(xbar),  xbar the mean x_j;
// both second order elsewhere (the static limit of chi: x / tan x).
#pragma once
#include <math.h>
#include "portable.hpp"

struct BlochC {
    int n = 0;                  // excited states (1 or 2)
    double k[2] = {0, 0};       // (dt/2) a_j' (the coupling of K per unit field)
    double t[2] = {0, 0};       // tan(w_j dt/2) (the diagonal of K)
    double cP[2] = {0, 0};      // cP_j'
    double wq[2] = {0, 0};      // w_j' (the energy of the levels)
    double ef = 0;              // alpha a0 k0 cos(xbar)
    bool loss = false;          // relaxation on (T1 > 0 or T2 > 0)
    double fp = 1, fc = 1, f12 = 1;   // over dt/2: populations of 1, 2 (exp(-h/T1)), c_0j (exp(-h/(2 T1) - h/T2)), c_12
    bool ion = false;           // ADK ionization of the levels 0, 1, 2 (rates in 1 / omega0, E in code units)
    double lp[3] = {0, 0, 0}, pw[3] = {0, 0, 0}, xe[3] = {0, 0, 0}, ep[3] = {0, 0, 0};   // w = exp(lp - pw ln E + xe / E), E <= ep
    double hdt = 0;             // dt / 2
};

// state per node: p0, p1, p2, Re c01, Im c01, Re c02, Im c02, Re c12, Im c12 (c_kl = rho_kl)
enum { BS_P0, BS_P1, BS_P2, BS_R01, BS_I01, BS_R02, BS_I02, BS_R12, BS_I12, BS_N };

HD inline double bloch_P(const BlochC& L, const double* s) { return L.cP[0] * s[BS_R01] + (L.n > 1 ? L.cP[1] * s[BS_R02] : 0.0); }
HD inline double bloch_energy(const BlochC& L, const double* s) { return L.ef * (L.wq[0] * s[BS_P1] + (L.n > 1 ? L.wq[1] * s[BS_P2] : 0.0)); }

namespace bloch_detail {
struct Cx { double r, i; };
HD inline Cx cmul(Cx a, Cx b) { return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r}; }
HD inline Cx cadd(Cx a, Cx b) { return {a.r + b.r, a.i + b.i}; }
HD inline Cx csub(Cx a, Cx b) { return {a.r - b.r, a.i - b.i}; }
HD inline Cx cconj(Cx a) { return {a.r, -a.i}; }
HD inline Cx cscale(double s, Cx a) { return {s * a.r, s * a.i}; }
}  // namespace bloch_detail

// rho -> U rho U^+ with K = (dt/2) H(E): s in, o out (o may be s)
HD inline void bloch_rotate(const BlochC& L, const double* s, double E, double* o) {
    using namespace bloch_detail;
    const double k01 = -L.k[0] * E, k02 = L.n > 1 ? -L.k[1] * E : 0.0, k11 = L.t[0], k22 = L.n > 1 ? L.t[1] : 0.0;
    // A = 1 + i K = [[1, i k01, i k02], [i k01, 1 + i k11, 0], [i k02, 0, 1 + i k22]] (symmetric); U = 2 A^-1 - 1
    const Cx A00{1, 0}, A01{0, k01}, A02{0, k02}, A11{1, k11}, A22{1, k22};
    const Cx C00 = cmul(A11, A22), C01 = cscale(-1, cmul(A01, A22)), C02 = cscale(-1, cmul(A02, A11));
    const Cx C11 = csub(cmul(A00, A22), cmul(A02, A02)), C12 = cmul(A01, A02), C22 = csub(cmul(A00, A11), cmul(A01, A01));
    const Cx det = cadd(cadd(cmul(A00, C00), cmul(A01, C01)), cmul(A02, C02));
    const double dn = det.r * det.r + det.i * det.i;
    const Cx inv{det.r / dn, -det.i / dn};
    const Cx one{1, 0};
    const Cx U00 = csub(cscale(2, cmul(C00, inv)), one), U01 = cscale(2, cmul(C01, inv)), U02 = cscale(2, cmul(C02, inv));
    const Cx U11 = csub(cscale(2, cmul(C11, inv)), one), U12 = cscale(2, cmul(C12, inv)), U22 = csub(cscale(2, cmul(C22, inv)), one);
    const Cx U[3][3] = {{U00, U01, U02}, {U01, U11, U12}, {U02, U12, U22}};
    const Cx c01{s[BS_R01], s[BS_I01]}, c02{s[BS_R02], s[BS_I02]}, c12{s[BS_R12], s[BS_I12]};
    const Cx R[3][3] = {{{s[BS_P0], 0}, c01, c02}, {cconj(c01), {s[BS_P1], 0}, c12}, {cconj(c02), cconj(c12), {s[BS_P2], 0}}};
    Cx T[3][3];                                                   // T = U rho
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) T[i][j] = cadd(cadd(cmul(U[i][0], R[0][j]), cmul(U[i][1], R[1][j])), cmul(U[i][2], R[2][j]));
    auto el = [&](int i, int j) {                                 // (T U^+)_ij, (U^+)_kj = conj(U_jk)
        return cadd(cadd(cmul(T[i][0], cconj(U[j][0])), cmul(T[i][1], cconj(U[j][1]))), cmul(T[i][2], cconj(U[j][2])));
    };
    const Cx r00 = el(0, 0), r11 = el(1, 1), r22 = el(2, 2), r01 = el(0, 1), r02 = el(0, 2), r12 = el(1, 2);
    o[BS_P0] = r00.r; o[BS_P1] = r11.r; o[BS_P2] = r22.r;
    o[BS_R01] = r01.r; o[BS_I01] = r01.i; o[BS_R02] = r02.r; o[BS_I02] = r02.i; o[BS_R12] = r12.r; o[BS_I12] = r12.i;
}

// the ADK rate of level k (1 / omega0) in the field |E| (code units); constant beyond the maximum of the ADK expression
HD inline double bloch_adk(const BlochC& L, int k, double E) {
    E = fabs(E);
    if (E < L.ep[k] * 1e-3) return 0.0;                          // exp(-(2/3) kappa / E) underflows anyway
    if (E > L.ep[k]) E = L.ep[k];
    return exp(L.lp[k] - L.pw[k] * log(E) + L.xe[k] / E);
}
// relaxation over dt/2 (exact exponentials; T1: excited -> ground, T2: pure dephasing of the coherences), then the ionization of
// the levels with the rates g (population k: exp(-g_k h), coherence kl: exp(-(g_k + g_l) h / 2); the trace is the neutral fraction)
HD inline void bloch_ionize(const BlochC& L, double* s, const double* g) {
        const double h = L.hdt, e0 = exp(-0.5 * g[0] * h), e1 = exp(-0.5 * g[1] * h), e2 = exp(-0.5 * g[2] * h);
        s[BS_P0] *= e0 * e0; s[BS_P1] *= e1 * e1; s[BS_P2] *= e2 * e2;
        s[BS_R01] *= e0 * e1; s[BS_I01] *= e0 * e1; s[BS_R02] *= e0 * e2; s[BS_I02] *= e0 * e2; s[BS_R12] *= e1 * e2; s[BS_I12] *= e1 * e2;
}
HD inline void bloch_t1t2(const BlochC& L, double* s) {
    const double d1 = s[BS_P1] * (1 - L.fp), d2 = s[BS_P2] * (1 - L.fp);
    s[BS_P1] -= d1; s[BS_P2] -= d2; s[BS_P0] += d1 + d2;
    s[BS_R01] *= L.fc; s[BS_I01] *= L.fc; s[BS_R02] *= L.fc; s[BS_I02] *= L.fc; s[BS_R12] *= L.f12; s[BS_I12] *= L.f12;
}
// the losses over dt/2: first = true ionization then relaxation, false the reverse (a symmetric splitting)
HD inline void bloch_relax(const BlochC& L, double* s, const double* g, bool first) {
    if (first) { if (g) bloch_ionize(L, s, g); if (L.loss) bloch_t1t2(L, s); }
    else { if (L.loss) bloch_t1t2(L, s); if (g) bloch_ionize(L, s, g); }
}
// one step: Strang splitting of the relaxation around the Cayley step (second order); without relaxation the Cayley step alone
HD inline void bloch_step(const BlochC& L, const double* s, double E, double* o) {
    if (!L.loss && !L.ion) { bloch_rotate(L, s, E, o); return; }
    double r[BS_N], g[3] = {0, 0, 0};
    if (L.ion) for (int k = 0; k <= L.n; ++k) g[k] = bloch_adk(L, k, E);
    for (int q = 0; q < BS_N; ++q) r[q] = s[q];
    bloch_relax(L, r, L.ion ? g : nullptr, true);
    bloch_rotate(L, r, E, o);
    bloch_relax(L, o, L.ion ? g : nullptr, false);
}
// T1, T2 in 1/omega0 (0: none)
inline void bloch_set_relaxation(BlochC& L, double T1, double T2, double dt) {
    const double h = 0.5 * dt, a = T1 > 0 ? h / T1 : 0.0, b = T2 > 0 ? h / T2 : 0.0;
    L.loss = a > 0 || b > 0;
    L.fp = exp(-a); L.fc = exp(-0.5 * a - b); L.f12 = exp(-a - b);
}

// ADK (Ammosov, Delone, Krainov 1986; s-states, l* = n* - 1, m = 0; the form of GEM-PIC-RZ src/ionization.hpp) for the levels
// 0 .. n with the ionization energies U_k (eV), the ion charge 1 after the ionization; omega0 = c k0 (k0 in 1/m):
//   w = omega_a C^2 (U / 2 U_H) [2 kappa / E]^(2 n* - 1) exp(-(2/3) kappa / E),  kappa = (U / U_H)^(3/2) E_a,  n* = sqrt(U_H / U),
//   C^2 = 2^(2 n*) / (n* Gamma(n* + 1) Gamma(n*)); kept constant beyond its maximum at E = (2/3) kappa / (2 n* - 1)
inline void bloch_set_adk(BlochC& L, const double* U_eV, double k0) {
    const double UH = 13.605693122994, Ea = 5.14220674763e11, wa = 4.1341373335e16, c = 299792458.0;
    const double w0 = c * k0, Eu = 9.1093837015e-31 * c * w0 / 1.602176634e-19;   // m c omega0 / e [V/m]
    L.ion = true;
    for (int k = 0; k <= L.n; ++k) {
        const double ns = sqrt(UH / U_eV[k]), C2 = pow(2.0, 2 * ns) / (ns * tgamma(ns + 1) * tgamma(ns));
        const double kap = pow(U_eV[k] / UH, 1.5) * Ea / Eu;   // code units
        L.pw[k] = 2 * ns - 1;
        L.lp[k] = log(wa / w0 * C2 * U_eV[k] / (2 * UH)) + L.pw[k] * log(2 * kap);
        L.xe[k] = -2.0 / 3.0 * kap;
        L.ep[k] = 2.0 / 3.0 * kap / L.pw[k];
    }
}

// the constants of the lines for the time step dt: w_j (omega0 units), d_j (e a0), a0k0 = a0 k0
inline BlochC bloch_constants(int n, const double* w, const double* d, double a0k0, double dt) {
    const double alpha_inv = 137.035999084;
    BlochC L;
    L.n = n;
    L.hdt = 0.5 * dt;
    double cbar = 0;                                              // cos of the mean x_j
    for (int j = 0; j < n; ++j) cbar += 0.5 * w[j] * dt / n;
    cbar = cos(cbar);
    L.ef = cbar * a0k0 / alpha_inv;
    for (int j = 0; j < n; ++j) {
        const double x = 0.5 * w[j] * dt, c = cos(x);
        L.t[j] = tan(x);
        L.wq[j] = 2 * L.t[j] / dt;
        L.k[j] = 0.5 * dt * d[j] * alpha_inv / c;               // a_j' = a_j / cos x_j
        L.cP[j] = 2 * d[j] * a0k0 * cbar / c;                    // cP_j' = cP_j cos(xbar) / cos x_j
    }
    return L;
}
