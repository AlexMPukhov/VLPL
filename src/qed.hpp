// Phase 4, spin-resolved strong-field QED (docs/specs/qed.md, decision 93): T-BMT spin precession of every species with
// spin, and the Monte-Carlo emission of photons (nonlinear inverse Compton scattering, LCFA) resolved in the electron
// spin and the photon polarization, with the spin update after an emission and the non-radiative spin change.
//
// Units: c = 1, times in 1/omega0, fields in m c omega0 / e, u = gamma v, energies in m_e c^2. hw = hbar omega0 / (m c^2).
//
// Rate (per unit time, per du, u = eps_gamma / eps_e, summed over the final spin and the photon polarization):
//   d2W/(du dt) = C0 (w + <g, S>),   C0 = alpha / (sqrt(3) pi hw gamma)     (= alpha m^2 c^4 / (sqrt3 pi hbar eps_e))
//   w  = (u^2 - 2u + 2) / (1 - u) K23(y) - IK13(y),   g = -u K13(y) e2,   y = 2u / (3 (1 - u) chi),   IK13(y) = int_y^inf K13
//   pf = -u/(1-u) K13 e2 + (2 K23 - IK13) S + u^2/(1-u) (K23 - IK13) <S, ev> ev           (final spin)
//   pxi = ( u/(1-u) K13 <S, e1>,  ((2u - u^2)/(1-u) K23 - u IK13) <S, ev>,  K23 - u/(1-u) K13 <S, e2> )   (Stokes vector)
// in the basis ev = u/|u|, e1 = the transverse part of the Lorentz force q (E + beta x B) (normalized), e2 = ev x e1.
// Mixed state (|S| <= 1, default): S <- pf / p0, xi = pxi / p0, p0 = w + <g, S>. Pure state: S <- +-pf/|pf| with
// P(+) = (p0 + |pf|) / (2 p0) (the same for xi). No emission in a (sub)step a = C0 dt:
//   S <- [S (1 - a W) - a G] / (1 - a (W + <G, S>)),   W = int w du,  G = int g du = -Gs e2,  Gs = int u K13 du.
// The emitted photon goes along ev with eps_gamma = u eps_e; the lepton keeps its direction, gamma <- (1 - u) gamma.
// Sampling: u = r^3 (r uniform), emission if r' < 3 r^2 C0 (w + <g, S>) dt (Algorithm 1 of the QV3D draft); the step
// is split into n sub-steps so that C0 M dt_s <= 1 and C0 (W + Gs) dt_s <= substep_prob (M = max_r 3 r^2 (w + u K13)).
//
// T-BMT (anomaly a): dS/dt = (q/m) S x [ (a + 1/gamma) B - a gamma/(gamma + 1) (beta.B) beta - (a + 1/(gamma + 1)) beta x E ],
// done as an exact rotation per step with E^n, B^n and u^n = (u^{n-1/2} + u^{n+1/2}) / 2.
#pragma once
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "ionization.hpp"          // ion_mix, ion_uniform (counter-based random numbers)
#include "portable.hpp"

// species block: spin = sx sy sz (|S| <= 1), spin_anomaly = a (default 1.15965218e-3, the electron's)
struct SpinConfig {
    bool on = false;
    double s[3] = {0, 0, 0};
    double anomaly = 1.15965218128e-3;
};

// block qed
struct QedConfig {
    bool on = false;
    enum Model { None = 0, MonteCarlo = 1, Deterministic = 2 } model = MonteCarlo;
    std::vector<std::string> species;          // emitting species (|q| = 1, m = 1)
    std::string photons;                       // a species with kind = photon, or empty: tallies only
    double photon_min = 0;                     // photons into that species above this energy (m c^2)
    bool pure = false;                         // spin_state = pure | mixed (default)
    bool precession = true;                    // T-BMT for every species with spin
    bool spin_flip = true;                     // false: the emission leaves the spin unchanged (radiative spin effects off)
    double substep_prob = 0.1;
    int bins = 200;                            // spectra: log bins in u from u_min to 1, in eps_gamma from e_min to e_max
    double u_min = 1e-8, e_min_mev = 1e-3, e_max_mev = 1e7;
    long spectrum_every = 0;                   // dir/qed_spectrum_<step>.txt every n steps (0: only at the end)
    // Breit-Wheeler pairs (decision 96): the photons of `photons` decay into an electron of pair_electrons and a positron of
    // pair_positrons (both empty: no pairs)
    std::string pair_e, pair_p;
    bool pair_selection = true;                // the photon Stokes vector changes without a decay (dichroism of the vacuum)
    bool pairs() const { return !pair_e.empty(); }
};

// Kernel view of the tables (qed.cpp builds them). Bessel functions of y: ln K13, ln K23, ln IK13 at ln y = ly0 + k dly
// (k = 0 .. ny - 1). Integrals over u at ln chi = lc0 + k dlc: ln W, ln Gs, ln M, ln P (P = int u w du: the power).
struct QedTab {
    const double* fy = nullptr;
    int ny = 0;
    double ly0 = 0, dly = 1;
    const double* fc = nullptr;
    int nc = 0;
    double lc0 = 0, dlc = 1;
    // Breit-Wheeler integrals at ln chi_gamma = lb0 + k dlb: ln T0 + 8/(3 chi), ln T3 + 8/(3 chi), ln Mb + 8/(3 chi)
    const double* fb = nullptr;
    int nb = 0;
    double lb0 = 0, dlb = 1;
};

// quadratic interpolation in a table of ln f at t = (x - x0) / dx (nearest-centred three points); outside: the end slope
HD inline double qed_interp(const double* f, int stride, int n, double t) {
    if (t <= 0) return f[0] + t * (f[stride] - f[0]);
    if (t >= n - 1) return f[(n - 1) * stride] + (t - (n - 1)) * (f[(n - 1) * stride] - f[(n - 2) * stride]);
    int j = int(t + 0.5);
    j = j < 1 ? 1 : (j > n - 2 ? n - 2 : j);
    const double s = t - j, a = f[(j - 1) * stride], b = f[j * stride], c = f[(j + 1) * stride];
    return b + 0.5 * s * (c - a) + 0.5 * s * s * (c - 2 * b + a);
}
// K13(y), K23(y), IK13(y); zero beyond the table (y > ~700: below 1e-300)
HD inline void qed_bessel(const QedTab& T, double y, double& k13, double& k23, double& ik13) {
    const double t = (log(y) - T.ly0) / T.dly;
    if (t > T.ny - 1) { k13 = k23 = ik13 = 0; return; }
    k13 = exp(qed_interp(T.fy, 3, T.ny, t));
    k23 = exp(qed_interp(T.fy + 1, 3, T.ny, t));
    ik13 = exp(qed_interp(T.fy + 2, 3, T.ny, t));
}
// W, Gs, M, P at chi
HD inline void qed_integrals(const QedTab& T, double chi, double& W, double& Gs, double& M, double& P) {
    const double t = (log(chi) - T.lc0) / T.dlc;
    W = exp(qed_interp(T.fc, 4, T.nc, t));
    Gs = exp(qed_interp(T.fc + 1, 4, T.nc, t));
    M = exp(qed_interp(T.fc + 2, 4, T.nc, t));
    P = exp(qed_interp(T.fc + 3, 4, T.nc, t));
}

// Breit-Wheeler T0 = int_0^1 B0 dx, T3 = int_0^1 K23 dx, Mb = 1.05 max_x (B0 + K23) at chi_gamma, each times exp(8/(3 chi))
// (the exponential factor of chi << 1 taken out: smooth in ln chi). Below the table (chi < 0.01, rate ~ e^-267): false.
constexpr double QED_BW_CHI_MIN = 0.01;
HD inline bool qed_bw_integrals(const QedTab& T, double chi, double& T0, double& T3, double& Mb) {
    const double t = (log(chi) - T.lb0) / T.dlb;
    if (!(t >= 0)) return false;
    T0 = exp(qed_interp(T.fb, 3, T.nb, t));
    T3 = exp(qed_interp(T.fb + 1, 3, T.nb, t));
    Mb = exp(qed_interp(T.fb + 2, 3, T.nb, t));
    return true;
}

// chi of a particle with momentum u (gamma g) in E, B: chi = gamma hw sqrt((E + beta x B)^2 - (E.beta)^2)
HD inline double qed_chi(const double E[3], const double B[3], const double u[3], double g, double hw) {
    const double b[3] = {u[0] / g, u[1] / g, u[2] / g};
    const double F[3] = {E[0] + b[1] * B[2] - b[2] * B[1], E[1] + b[2] * B[0] - b[0] * B[2], E[2] + b[0] * B[1] - b[1] * B[0]};
    const double eb = E[0] * b[0] + E[1] * b[1] + E[2] * b[2];
    const double x = F[0] * F[0] + F[1] * F[1] + F[2] * F[2] - eb * eb;
    return x > 0 ? g * sqrt(x) * hw : 0.0;
}
// basis (ev, e1, e2) of a particle of charge sign q; false if the transverse force vanishes
HD inline bool qed_basis(const double E[3], const double B[3], const double u[3], double g, double q, double ev[3], double e1[3],
                         double e2[3]) {
    const double un = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    if (!(un > 0)) return false;
    ev[0] = u[0] / un; ev[1] = u[1] / un; ev[2] = u[2] / un;
    const double b[3] = {u[0] / g, u[1] / g, u[2] / g};
    double F[3] = {q * (E[0] + b[1] * B[2] - b[2] * B[1]), q * (E[1] + b[2] * B[0] - b[0] * B[2]), q * (E[2] + b[0] * B[1] - b[1] * B[0])};
    const double fv = F[0] * ev[0] + F[1] * ev[1] + F[2] * ev[2];
    F[0] -= fv * ev[0]; F[1] -= fv * ev[1]; F[2] -= fv * ev[2];
    const double fn = sqrt(F[0] * F[0] + F[1] * F[1] + F[2] * F[2]);
    if (!(fn > 0)) return false;
    e1[0] = F[0] / fn; e1[1] = F[1] / fn; e1[2] = F[2] / fn;
    e2[0] = ev[1] * e1[2] - ev[2] * e1[1]; e2[1] = ev[2] * e1[0] - ev[0] * e1[2]; e2[2] = ev[0] * e1[1] - ev[1] * e1[0];
    return true;
}

// T-BMT: S rotated exactly over dt (qm = q / m, a = anomaly) with E, B and the momentum u at the middle of the step
HD inline void qed_tbmt(double qm, double a, const double E[3], const double B[3], const double u[3], double dt, double S[3]) {
    const double g = sqrt(1.0 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    const double b[3] = {u[0] / g, u[1] / g, u[2] / g};
    const double bB = b[0] * B[0] + b[1] * B[1] + b[2] * B[2];
    const double bxE[3] = {b[1] * E[2] - b[2] * E[1], b[2] * E[0] - b[0] * E[2], b[0] * E[1] - b[1] * E[0]};
    const double cB = a + 1.0 / g, cv = a * g / (g + 1.0) * bB, cE = a + 1.0 / (g + 1.0);
    // dS/dt = w x S with w = -(q/m) Omega
    const double w[3] = {-qm * (cB * B[0] - cv * b[0] - cE * bxE[0]), -qm * (cB * B[1] - cv * b[1] - cE * bxE[1]),
                         -qm * (cB * B[2] - cv * b[2] - cE * bxE[2])};
    const double wn = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    if (!(wn > 0)) return;
    const double n[3] = {w[0] / wn, w[1] / wn, w[2] / wn}, th = wn * dt, c = cos(th), s = sin(th);
    const double nS = n[0] * S[0] + n[1] * S[1] + n[2] * S[2];
    const double nxS[3] = {n[1] * S[2] - n[2] * S[1], n[2] * S[0] - n[0] * S[2], n[0] * S[1] - n[1] * S[0]};
    for (int k = 0; k < 3; ++k) S[k] = S[k] * c + nxS[k] * s + n[k] * nS * (1 - c);
}

// Constants of one emission kernel
struct QedK {
    QedTab T;
    double hw = 0, c0f = 0;                    // C0 = c0f / gamma, c0f = alpha / (sqrt3 pi hw)
    double dt = 0, pmax = 0.1, q = -1;         // step, substep_prob, charge sign
    int model = 1;                             // QedConfig::Model
    bool pure = false, flip = true, spin = false;
    double ph_min = 0;                         // photon threshold (m c^2)
    bool photons = false;                      // a photon species: the photons above ph_min go into it
    double lu0 = 0, dlu = 1, le0 = 0, dle = 1; // spectra: bin = (ln u - lu0) / dlu, (ln eps - le0) / dle
    int bins = 0;
    // boosted frame (decision 117): photon energies of the tallies, the spectra and the threshold ph_min in the lab,
    // eps_lab = bg eps (1 + bb n_0) (n the photon direction, axis 0 the boost axis); 1, 0: the simulation frame
    double bg = 1, bb = 0;
};
// Outputs of one kernel: tallies (atomic), the spectra (bins x 5: sum w, sum w eps, sum w xi_1..3), photons
struct QedOut {
    double* tot = nullptr;                     // [0] sum w, [1] sum w eps, [2..4] sum w xi, [5] events with P > 1, [6] sum w eps to photons
    double* hu = nullptr;                      // spectrum in u
    double* he = nullptr;                      // spectrum in eps_gamma
    double *X = nullptr, *Y = nullptr, *Z = nullptr, *UX = nullptr, *UY = nullptr, *UZ = nullptr, *W = nullptr;
    double *XI1 = nullptr, *XI2 = nullptr, *XI3 = nullptr, *P0 = nullptr, *P1 = nullptr, *P2 = nullptr;   // Stokes, e1
    long long* ID = nullptr;                   // ids of the photons (with pairs: their random-number streams)
};

HD inline double qed_dot(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
// Boosted frame (decision 117): a frame moving at beta along axis 0 (x; RZ: z) with respect to the frame of u, s.
// Rest-frame spin s of a particle of momentum u (the rest frame reached by the pure boost from this frame) -> the rest-frame
// spin of the same particle reached by the pure boost from the moving frame (the Wigner rotation included): through the
// spin 4-vector a = (u.s, s + (u.s) u / (gamma + 1)), boosted, and s' = a' - a'^0 u' / (gamma' + 1). |s| is kept.
HD inline void qed_spin_boost(const double u[3], double s[3], double beta) {
    const double gb = 1 / sqrt((1 - beta) * (1 + beta));
    const double g = sqrt(1 + qed_dot(u, u)), us = qed_dot(u, s);
    const double a0 = us, ax = s[0] + us * u[0] / (g + 1);
    const double b0 = gb * (a0 - beta * ax), bx = gb * (ax - beta * a0);
    const double ux = gb * (u[0] - beta * g), gp = gb * (g - beta * u[0]);
    const double up[3] = {ux, u[1], u[2]}, ap[3] = {bx, s[1] + us * u[1] / (g + 1), s[2] + us * u[2] / (g + 1)};
    for (int k = 0; k < 3; ++k) s[k] = ap[k] - b0 * up[k] / (gp + 1);
}
// Photon of momentum k (|k| = its energy): its polarization axis e1 (unit, transverse) -> in the moving frame, as the
// polarization 4-vector (0, e1) with the gauge fixed to be transverse there: e1' = eps' - eps'^0 k' / |k'|. Unit and
// transverse; the basis (e1, k x e1) goes over into (e1', k' x e1'), so the Stokes vector in it is unchanged.
HD inline void qed_pol_boost(const double k[3], double e1[3], double beta) {
    const double gb = 1 / sqrt((1 - beta) * (1 + beta));
    const double om = sqrt(qed_dot(k, k));
    const double kx = gb * (k[0] - beta * om), omp = gb * (om - beta * k[0]);
    const double e0 = -gb * beta * e1[0], ex = gb * e1[0];
    const double kp[3] = {kx, k[1], k[2]};
    const double ep[3] = {ex - e0 * kp[0] / omp, e1[1] - e0 * kp[1] / omp, e1[2] - e0 * kp[2] / omp};
    const double n = sqrt(qed_dot(ep, ep));
    for (int c = 0; c < 3; ++c) e1[c] = n > 0 ? ep[c] / n : 0.0;
}
// the state from p (|p| <= p0): mixed: p / p0; pure: +-p / |p| with P(+) = (p0 + |p|) / (2 p0)
HD inline void qed_select(bool pure, const double p[3], double p0, double r, double out[3]) {
    if (!pure) { for (int k = 0; k < 3; ++k) out[k] = p[k] / p0; return; }
    const double pn = sqrt(qed_dot(p, p));
    if (!(pn > 0)) { out[0] = out[1] = out[2] = 0; return; }
    const double sg = r < 0.5 * (p0 + pn) / p0 ? 1.0 : -1.0;
    for (int k = 0; k < 3; ++k) out[k] = sg * p[k] / pn;
}

// One particle over one step: emission (Monte-Carlo or deterministic) with E, B held, u (u^{n+1/2}) and S updated in place.
// Count = true: only counts the photons above the threshold (no tallies); the same random numbers give the same events
// in the writing pass, whose photons go to O.X[first ..] (O.X null: none). Returns the photons above the threshold.
// chi_out: chi at the start of the step.
template <bool Count>
HD inline int qed_emit(const QedK& K, const QedOut& O, const double E[3], const double B[3], double u[3], double S[3], double w,
                       const double x[3], unsigned long long key, long first, double& chi_out) {
    double g = sqrt(1.0 + qed_dot(u, u));
    double chi = qed_chi(E, B, u, g, K.hw);
    chi_out = chi;
    if (!(chi > 0) || K.model == 0) return 0;
    double W, Gs, M, P;
    qed_integrals(K.T, chi, W, Gs, M, P);
    double C0 = K.c0f / g;
    if (K.model == 2) {                        // deterministic: d gamma / dt = -gamma C0 P (= P_alpha chi^2 g(chi)), direction kept
        const int n = int(ceil(C0 * P * K.dt / 0.01)) > 1 ? int(ceil(C0 * P * K.dt / 0.01)) : 1;
        const double h = K.dt / n;
        double lost = 0;
        for (int k = 0; k < n; ++k) {
            const double g1 = g - g * C0 * P * h;
            const double gn = g1 > 1 ? g1 : 1.0;
            const double un = sqrt(qed_dot(u, u)), s = un > 0 ? sqrt(gn * gn - 1) / un : 0.0;
            const double ux0 = u[0];
            u[0] *= s; u[1] *= s; u[2] *= s;
            lost += K.bg * ((g - gn) + K.bb * (ux0 - u[0]));   // the lab energy of the loss (bg = 1, bb = 0: this frame)
            g = gn;
            if (!(g > 1)) break;
            chi = qed_chi(E, B, u, g, K.hw);
            if (!(chi > 0)) break;
            qed_integrals(K.T, chi, W, Gs, M, P);
            C0 = K.c0f / g;
        }
        if (!Count) atomic_add(O.tot + 1, w * lost);
        return 0;
    }
    double ev[3], e1[3], e2[3];
    if (!qed_basis(E, B, u, g, K.q, ev, e1, e2)) return 0;
    double a1 = C0 * M * K.dt, a2 = C0 * (W + Gs) * K.dt / K.pmax;
    const double am = a1 > a2 ? a1 : a2;
    const int ns = am > 1 ? int(ceil(am)) : 1;
    const double h = K.dt / ns;
    int made = 0;
    for (int k = 0; k < ns; ++k) {
        const unsigned long long kk = key + 4ULL * (unsigned long long)k;
        const double r = ion_uniform(kk), uu = r * r * r;
        const double y = 2 * uu / (3 * (1 - uu) * chi);
        double k13, k23, ik13;
        qed_bessel(K.T, y, k13, k23, ik13);
        const double om = 1 - uu;
        const double wr = (uu * uu - 2 * uu + 2) / om * k23 - ik13;
        const double S1 = qed_dot(S, e1), S2 = qed_dot(S, e2), Sv = qed_dot(S, ev);
        const double p0 = wr - uu * k13 * S2;            // w + <g, S>
        const double pe = 3 * r * r * h * C0 * p0;
        if (ion_uniform(kk + 1) < pe) {                  // emission
            if (pe > 1 && !Count) atomic_add(O.tot + 5, 1.0);
            const double xi_p[3] = {uu / om * k13 * S1, ((2 * uu - uu * uu) / om * k23 - uu * ik13) * Sv, k23 - uu / om * k13 * S2};
            double xi[3];
            qed_select(K.pure, xi_p, p0, ion_uniform(kk + 3), xi);
            if (K.spin && K.flip) {
                const double cs = 2 * k23 - ik13, cv = uu * uu / om * (k23 - ik13) * Sv, ce = -uu / om * k13;
                const double pf[3] = {ce * e2[0] + cs * S[0] + cv * ev[0], ce * e2[1] + cs * S[1] + cv * ev[1], ce * e2[2] + cs * S[2] + cv * ev[2]};
                qed_select(K.pure, pf, p0, ion_uniform(kk + 2), S);
            }
            const double g1 = om * g, gn = g1 > 1 ? g1 : 1.0, eph = g - gn;
            const double sc = sqrt(gn * gn - 1) / sqrt(qed_dot(u, u));
            u[0] *= sc; u[1] *= sc; u[2] *= sc;
            const double elab = K.bg * eph * (1 + K.bb * ev[0]);   // the photon's lab energy (= eph without a boost)
            const bool keep = K.photons && elab >= K.ph_min;
            if (Count) { made += keep; }
            else {
                atomic_add(O.tot, w); atomic_add(O.tot + 1, w * elab);
                atomic_add(O.tot + 2, w * xi[0]); atomic_add(O.tot + 3, w * xi[1]); atomic_add(O.tot + 4, w * xi[2]);
                const double vals[5] = {w, w * elab, w * xi[0], w * xi[1], w * xi[2]};
                const int bu = int(floor((log(uu) - K.lu0) / K.dlu)), be = int(floor((log(elab) - K.le0) / K.dle));
                if (bu >= 0 && bu < K.bins) for (int c = 0; c < 5; ++c) atomic_add(O.hu + 5 * bu + c, vals[c]);
                if (be >= 0 && be < K.bins) for (int c = 0; c < 5; ++c) atomic_add(O.he + 5 * be + c, vals[c]);
                if (keep) {
                    const long q = first + made++;
                    O.X[q] = x[0]; O.Y[q] = x[1]; O.Z[q] = x[2];
                    O.UX[q] = eph * ev[0]; O.UY[q] = eph * ev[1]; O.UZ[q] = eph * ev[2];
                    O.W[q] = w;
                    O.XI1[q] = xi[0]; O.XI2[q] = xi[1]; O.XI3[q] = xi[2];
                    O.P0[q] = e1[0]; O.P1[q] = e1[1]; O.P2[q] = e1[2];
                    if (O.ID) { const unsigned long long id = ion_mix(kk + 7); O.ID[q] = (long long)(id == 0 ? 1 : id); }
                    atomic_add(O.tot + 6, w * elab);
                }
            }
            g = gn;
            if (!(g > 1)) break;
            chi = qed_chi(E, B, u, g, K.hw);
            if (!(chi > 0)) break;
            qed_integrals(K.T, chi, W, Gs, M, P);
            C0 = K.c0f / g;
            if (!qed_basis(E, B, u, g, K.q, ev, e1, e2)) break;
        } else if (K.spin && K.flip) {                   // no emission: the non-radiative change
            const double a = C0 * h, pn0 = 1 - a * (W - Gs * S2);
            const double p[3] = {S[0] * (1 - a * W) + a * Gs * e2[0], S[1] * (1 - a * W) + a * Gs * e2[1], S[2] * (1 - a * W) + a * Gs * e2[2]};
            qed_select(K.pure, p, pn0, ion_uniform(kk + 2), S);
        }
    }
    return made;
}

// ---------------------------------------------------------------- Breit-Wheeler pairs (decision 96, docs/specs/qed.md)
// Photon of energy om (m c^2) along n, chi_gamma = om hw |E_perp + n x B|; x = eps_+ / om, x_- = 1 - x, rho = 2/(3 chi x x_-).
// Basis: n, e1 = the transverse part of E + n x B (the force on the positron), e2 = n x e1; Stokes xi in (e1, e2).
//   d2W/(dx dt) = Cg (B0 - xi3 K23(rho)),  B0 = (x^2 + x_-^2)/(x x_-) K23 + IK13,  Cg = alpha / (sqrt3 pi hw om)
// Pair spins (their means in a mixed state: S_+- = b_+- / p0, p0 = B0 - xi3 K23), as Seipt & King, PRA 102, 052805 (2020) and
// Chen et al., PRD 105, 116013 (2022); e2 = -B'/|B'| (B' the rest-frame magnetic field): the positron preferably along +B', the
// electron along -B' (the lower magnetic energy):
//   b_+ = -(K13/x - xi3 K13/x_-) e2 - xi1 K13/x_- e1 + xi2 ((x - x_-)/(x x_-) K23 + IK13/x) n
//   b_- = (K13/x_- - xi3 K13/x) e2 + xi1 K13/x e1 + xi2 ((x_- - x)/(x x_-) K23 + IK13/x_-) n
// (|b_+-| <= p0 for all x, chi and |xi| <= 1.) No decay in dt (a = Cg dt): xi <- [xi (1 - a T0) + a T3 (0,0,1)] / (1 - a (T0 -
// xi3 T3)). Threshold om > 2; the pair shares the energy exactly (eps_+- >= 1), both along n (|p| = sqrt(eps^2 - 1): the field
// takes the momentum m^2 c / (2 eps) per lepton, as in the LCFA).
struct BwK {
    QedTab T;
    double hw = 0, c0f = 0, dt = 0, pmax = 0.1;
    bool pure = false, sel = true, spin_e = false, spin_p = false;
    int bins = 0;                              // spectra: x in bins linear bins, eps_+ in log bins (le0, dle as the photons')
    double le0 = 0, dle = 1;
    double bg = 1, bb = 0;                     // boosted frame: the photon energy and the lepton spectra in the lab (decision 117)
};
struct BwOut {
    // [0] sum w, [1] sum w om, [2] P > 1 events, [3] sum w <S_+, e2>, [4] sum w <S_+, n>, [5] sum w <S_-, -e2>, [6] sum w <S_-, n>,
    // [7] sum w xi3 (of the decayed photons, in the decay basis)
    double* tot = nullptr;
    double* hx = nullptr;                      // bins x 4: sum w, sum w xi3, sum w <S_+, e2>, sum w <S_+, n>  (x = eps_+ / om)
    double* he = nullptr;                      // bins x 2: sum w, sum w eps (positrons and electrons, by their energy)
    struct L { double *X, *Y, *Z, *UX, *UY, *UZ, *W, *S0, *S1, *S2; long long* ID; } e{}, p{};
};
// Stokes vector from the basis (pe, n x pe) to (e1, n x e1)
HD inline void qed_stokes_rotate(const double n[3], const double pe[3], const double e1[3], double xi[3]) {
    const double c = qed_dot(pe, e1);
    const double npe[3] = {n[1] * pe[2] - n[2] * pe[1], n[2] * pe[0] - n[0] * pe[2], n[0] * pe[1] - n[1] * pe[0]};
    const double s = qed_dot(npe, e1), r = c * c + s * s;
    if (!(r > 0)) return;
    const double c2 = (c * c - s * s) / r, s2 = 2 * c * s / r;
    const double x1 = xi[0], x3 = xi[2];
    xi[2] = c2 * x3 + s2 * x1;
    xi[0] = -s2 * x3 + c2 * x1;
}
// One photon over one step. xi, pe (its Stokes vector and basis axis) are updated in place unless Count; returns 1 if it decays
// (the pair goes to index q of O.e, O.p unless Count).
template <bool Count>
HD inline int qed_pair(const BwK& K, const BwOut& O, const double E[3], const double B[3], const double u[3], double xi[3],
                       double pe[3], double w, const double x[3], unsigned long long key, long q) {
    const double om = sqrt(qed_dot(u, u));
    if (!(om > 2)) return 0;
    const double chi = qed_chi(E, B, u, om, K.hw);
    double T0, T3, Mb;
    if (!(chi >= QED_BW_CHI_MIN) || !qed_bw_integrals(K.T, chi, T0, T3, Mb)) return 0;
    double n[3], e1[3], e2[3];
    if (!qed_basis(E, B, u, om, 1.0, n, e1, e2)) return 0;
    double X[3] = {xi[0], xi[1], xi[2]};
    qed_stokes_rotate(n, pe, e1, X);
    const double ef = exp(-8.0 / (3 * chi)), C = K.c0f / om * ef;      // the tables carry exp(+8/(3 chi))
    const double a1 = C * Mb * K.dt, a2 = C * (T0 + T3) * K.dt / K.pmax;
    const double am = a1 > a2 ? a1 : a2;
    const int ns = am > 1 ? int(ceil(am)) : 1;
    const double h = K.dt / ns;
    int dec = 0;
    for (int k = 0; k < ns && !dec; ++k) {
        const unsigned long long kk = key + 8ULL * (unsigned long long)k;
        const double xp = ion_uniform(kk), xm = 1 - xp;
        if (!(xp > 0 && xm > 0)) continue;
        const double rho = 2 / (3 * chi * xp * xm);
        double k13, k23, ik13;
        qed_bessel(K.T, rho, k13, k23, ik13);
        k13 /= ef; k23 /= ef; ik13 /= ef;                               // (C carries ef)
        const double p0 = (xp * xp + xm * xm) / (xp * xm) * k23 + ik13 - X[2] * k23;
        const double pd = h * C * p0;
        if (ion_uniform(kk + 1) < pd) {
            dec = 1;
            if (Count) break;
            if (pd > 1) atomic_add(O.tot + 2, 1.0);
            const double cp2 = -(k13 / xp - X[2] * k13 / xm), cp1 = -X[0] * k13 / xm, cpn = X[1] * ((xp - xm) / (xp * xm) * k23 + ik13 / xp);
            const double cm2 = k13 / xm - X[2] * k13 / xp, cm1 = X[0] * k13 / xp, cmn = X[1] * ((xm - xp) / (xp * xm) * k23 + ik13 / xm);
            const double bp[3] = {cp2 * e2[0] + cp1 * e1[0] + cpn * n[0], cp2 * e2[1] + cp1 * e1[1] + cpn * n[1], cp2 * e2[2] + cp1 * e1[2] + cpn * n[2]};
            const double bm[3] = {cm2 * e2[0] + cm1 * e1[0] + cmn * n[0], cm2 * e2[1] + cm1 * e1[1] + cmn * n[1], cm2 * e2[2] + cm1 * e1[2] + cmn * n[2]};
            double Sp[3], Sm[3];
            qed_select(K.pure, bp, p0, ion_uniform(kk + 2), Sp);
            qed_select(K.pure, bm, p0, ion_uniform(kk + 3), Sm);
            double ep = xp * om, em = om - ep;                          // eps_+- >= 1 (om > 2)
            if (ep < 1) { ep = 1; em = om - 1; }
            if (em < 1) { em = 1; ep = om - 1; }
            const double pp = sqrt(ep * ep - 1), pm = sqrt(em * em - 1);
            const BwOut::L* L[2] = {&O.p, &O.e};
            const double pl[2] = {pp, pm};
            const double* Sv[2] = {Sp, Sm};
            const unsigned long long id0 = ion_mix(kk + 5), id1 = ion_mix(kk + 6);
            for (int c = 0; c < 2; ++c) {
                const BwOut::L& o = *L[c];
                o.X[q] = x[0]; o.Y[q] = x[1]; o.Z[q] = x[2];
                o.UX[q] = pl[c] * n[0]; o.UY[q] = pl[c] * n[1]; o.UZ[q] = pl[c] * n[2];
                o.W[q] = w;
                if (o.S0) { o.S0[q] = Sv[c][0]; o.S1[q] = Sv[c][1]; o.S2[q] = Sv[c][2]; }
                const unsigned long long id = c ? id1 : id0;
                if (o.ID) o.ID[q] = (long long)(id == 0 ? 1 : id);
            }
            const double sp2 = qed_dot(Sp, e2), spn = qed_dot(Sp, n), sm2 = -qed_dot(Sm, e2), smn = qed_dot(Sm, n);
            atomic_add(O.tot, w); atomic_add(O.tot + 1, w * K.bg * om * (1 + K.bb * n[0]));   // the lab energy of the photon
            atomic_add(O.tot + 3, w * sp2); atomic_add(O.tot + 4, w * spn); atomic_add(O.tot + 5, w * sm2); atomic_add(O.tot + 6, w * smn);
            atomic_add(O.tot + 7, w * X[2]);
            int bx = int(xp * K.bins);
            bx = bx < 0 ? 0 : (bx >= K.bins ? K.bins - 1 : bx);
            atomic_add(O.hx + 4 * bx, w); atomic_add(O.hx + 4 * bx + 1, w * X[2]);
            atomic_add(O.hx + 4 * bx + 2, w * sp2); atomic_add(O.hx + 4 * bx + 3, w * spn);
            const double ee[2] = {K.bg * (ep + K.bb * pp * n[0]), K.bg * (em + K.bb * pm * n[0])};   // lab energies
            for (int c = 0; c < 2; ++c) {
                const int be = int(floor((log(ee[c]) - K.le0) / K.dle));
                if (be >= 0 && be < K.bins) { atomic_add(O.he + 2 * be, w); atomic_add(O.he + 2 * be + 1, w * ee[c]); }
            }
        } else if (K.sel) {                                             // no decay: the selection (dichroism) of xi
            const double a = C * h, pn0 = 1 - a * (T0 - T3 * X[2]);
            const double p[3] = {X[0] * (1 - a * T0), X[1] * (1 - a * T0), X[2] * (1 - a * T0) + a * T3};
            qed_select(K.pure, p, pn0, ion_uniform(kk + 4), X);
        }
    }
    if (!Count && !dec) {
        xi[0] = X[0]; xi[1] = X[1]; xi[2] = X[2];
        pe[0] = e1[0]; pe[1] = e1[1]; pe[2] = e1[2];
    }
    return dec;
}

class Simulation;
class Operator;
class Species;
class Deck;
struct Config;
// the qed block and the spin / kind keys of a species block (sim.cpp calls it from Config::from_deck)
QedConfig qed_from_deck(const Deck& d, double lambda0);
SpinConfig spin_from_deck(const Deck& d, const std::string& block);
void qed_check(const Config& c);
void qed_summary(FILE* f, const Config& c);
// true if the species needs the push / move split (spin or emitting; decision 15)
bool qed_split(const Config& c, size_t species);
bool qed_active(const Config& c);              // any species with spin or emission
// Registers s0 s1 s2 (spin), id, chi, and the photon attributes xi1 xi2 xi3 (Stokes), pe0 pe1 pe2 (polarization axis e1) on
// species i (before its load)
void qed_register(Simulation& s, size_t i);
// Builds the tables, adds the operators qed [qed], move_deposit [move_deposit] and photon_move [move_deposit];
// Simulation::split marks the species pushed by qed_push.
void qed_setup(Simulation& s);
// the stage MoveDeposit core operator (sim.cpp, rz_sim.cpp): the move and deposit of the split species (Pusher::None), then
// the rest of the push stage (ionization currents, guards, filters)
std::unique_ptr<Operator> make_move_deposit_operator(bool rz);
// stage Push of a split species: gather and push only, storing E, B and u^{n-1/2} (qed.cpp)
void qed_push(Simulation& s, size_t species);
// dir/qed.txt at the cadence of diag.txt (collective) and the spectra
void qed_diagnostics(Simulation& s, bool root);
void qed_write_spectra(Simulation& s, const std::string& path);   // collective
// out[0]: energy of all photons emitted since the start, out[1]: of those put into the photon species (sum over the emitting
// species and the ranks; collective)
void qed_totals(Simulation& s, double out[2]);
// Breit-Wheeler: out[0] = sum w of the pairs made since the start, out[1] = the energy of their photons (collective); and the
// pair spectra dir/qed_pairs.txt (collective)
void qed_pair_totals(Simulation& s, double out[2]);
// events with probability > 1 per sub-step since the start (the P>1 columns of qed.txt: emission of all species + pairs;
// collective)
double qed_events_over_one(Simulation& s);
void qed_write_pair_spectra(Simulation& s, const std::string& path);

// Host-side tables and helpers (qed_tables.cpp): exposed for the tests
struct QedTables {
    std::vector<double> fy, fc, fb;
    QedTab view_host() const;                  // pointers into fy, fc, fb (host)
    double ly0, dly, lc0, dlc, lb0, dlb;
    int ny, nc, nb;
};
const QedTables& qed_tables();                 // built once (host)
// the Bessel functions by the integral representation K_nu(y) = int_0^inf exp(-y cosh t) cosh(nu t) dt (independent of the
// tables' interpolation; reference values are in the tests)
void qed_bessel_exact(double y, double& k13, double& k23, double& ik13);
double qed_hw(double lambda0);                 // hbar omega0 / (m c^2) for the wavelength lambda0 [m]
constexpr double QED_ALPHA = 7.2973525693e-3;
