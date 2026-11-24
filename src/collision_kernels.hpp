// Binary Coulomb collisions: the device functions (docs/specs/collisions.md, decision 94). Relativistic Monte Carlo of
// Perez et al., Phys. Plasmas 19, 083104 (2012), the method of Smilei, with Nanbu's scattering-angle distribution.
// Units: u = p / (m c), m in m_e, q in e, densities in n_c, lengths 1/k0, times 1/omega0.
#pragma once
#include <cmath>
#include "ionization.hpp"   // ion_mix, ion_uniform (splitmix64)
#include "portable.hpp"

struct CollParams {
    double kre = 0;          // k0 r_e
    double bq = 0;           // hbar k0 / (2 m_e c): the quantum minimum impact parameter hbar / (2 p*) is bq / p* (p* in m_e c)
    double lt = 0;           // (3 k0 r_e)^(-1/3): the low-temperature limit of s (Perez eq. 22); 0: no limit (low_t_cap = false)
    double lnL = 0;          // > 0: fixed Coulomb logarithm; <= 0: automatic (from the Debye length of the cell)
    int angle = 0;           // 0: Nanbu's distribution (Perez's fits), 1: the fit of Smilei 5.1
};

// cos(chi) and sin(chi) of the deflection in the centre-of-mass frame for the parameter s, U uniform in (0, 1).
// angle 0: Nanbu, Phys. Rev. E 55, 4642 (1997), cos chi = ln(e^-A + 2 U sinh A) / A with coth A - 1/A = e^-s, A from the
// fits of Perez et al. 2012: s < 0.1: cos chi = 1 + s ln U; s < 3: 1/A = polynomial; s < 6: A = 3 e^-s; s >= 6: isotropic.
// angle 1: Smilei 5.1 (src/Collisions/Collisions.cpp): sin^2(chi/2) = a U / sqrt(1 - U + a^2 U), a = 0.37 s - 0.005 s^2
// - 0.0064 s^3 for s < 4, isotropic beyond.
HD inline void coll_angle(double s, double U, int angle, double& c, double& sn) {
    if (angle == 1) {
        if (s < 4) {
            const double s2 = s * s, a = 0.37 * s - 0.005 * s2 - 0.0064 * s2 * s;
            const double h = a * U / sqrt((1 - U) + a * a * U);
            c = 1 - 2 * h;
            sn = 2 * sqrt(h * (1 - h));
        } else {
            c = 2 * U - 1;
            sn = sqrt(1 - c * c);
        }
        return;
    }
    if (s < 0.1) {
        c = 1 + s * log(U);
        if (c < -1) c = -1;
    } else if (s < 6) {
        double A;
        if (s < 3) A = 1 / (0.0056958 + s * (0.9560202 + s * (-0.508139 + s * (0.47913906 + s * (-0.12788975 + s * 0.02389567)))));
        else A = 3 * exp(-s);
        c = log(exp(-A) + 2 * U * sinh(A)) / A;
        if (c > 1) c = 1;
        if (c < -1) c = -1;
    } else c = 2 * U - 1;
    sn = sqrt((1 - c) * (1 + c));
}

// One binary collision (Perez et al. 2012, section II). u1, u2 are replaced by the momenta after the collision when upd1,
// upd2 (the weight rule of Nanbu and Yonemura: the macro-particle of the larger weight is deflected with the probability
// w_small / w_large). D: dt_eff N max(w) / V and the correction for repeated partners (the density of the collision partners
// times the time); n1_23, n2_23 the densities of the two groups to the power 2/3 (the low-temperature cap); debye2 the
// squared Debye length (automatic Coulomb log). U1: the angle chi, U2: phi. Returns s (0: no collision); lnL the log used.
// The energies E* in the centre-of-mass frame are kept, so sum w m u and sum w m gamma are conserved to round-off when both
// particles are updated.
HD inline double coll_pair(double u1[3], double u2[3], double m1, double m2, double q1, double q2, double D, double n1_23,
                           double n2_23, double debye2, const CollParams& P, double U1, double U2, bool upd1, bool upd2,
                           double& lnL) {
    lnL = 0;
    const double qq = q1 * q2;
    if (qq == 0) return 0;
    const double g1 = sqrt(1 + u1[0] * u1[0] + u1[1] * u1[1] + u1[2] * u1[2]);
    const double g2 = sqrt(1 + u2[0] * u2[0] + u2[1] * u2[1] + u2[2] * u2[2]);
    const double E1 = m1 * g1, E2 = m2 * g2, Et = E1 + E2;
    const double p1[3] = {m1 * u1[0], m1 * u1[1], m1 * u1[2]}, p2[3] = {m2 * u2[0], m2 * u2[1], m2 * u2[2]};
    const double vc[3] = {(p1[0] + p2[0]) / Et, (p1[1] + p2[1]) / Et, (p1[2] + p2[2]) / Et};
    const double vc2 = vc[0] * vc[0] + vc[1] * vc[1] + vc[2] * vc[2];
    const double gc = 1 / sqrt(1 - vc2), t1 = gc * gc / (gc + 1);            // t1 = (gc - 1) / vc^2
    const double vp1 = vc[0] * p1[0] + vc[1] * p1[1] + vc[2] * p1[2], vp2 = vc[0] * p2[0] + vc[1] * p2[1] + vc[2] * p2[2];
    const double E1s = gc * (E1 - vp1), E2s = gc * (E2 - vp2);               // energies in the centre-of-mass frame
    const double t2 = t1 * vp1 - gc * E1;
    const double px = p1[0] + t2 * vc[0], py = p1[1] + t2 * vc[1], pz = p1[2] + t2 * vc[2];   // p1* (= -p2*)
    const double ps2 = px * px + py * py + pz * pz, ps = sqrt(ps2);
    if (!(ps > 0)) return 0;
    const double X = E1s * E2s / ps2 + 1;
    if (P.lnL > 0) lnL = P.lnL;
    else {
        const double b0 = fabs(qq) * P.kre * gc / Et * X, bqm = P.bq / ps, bmin = b0 > bqm ? b0 : bqm;
        lnL = 0.5 * log(1 + debye2 / (bmin * bmin));
        if (lnL < 2) lnL = 2;
    }
    double s = P.kre * lnL * qq * qq * gc * ps * X * X / (m1 * m2 * Et * g1 * g2);
    const double vrel = ps * Et / (gc * E1s * E2s);                          // |v1* - v2*|
    const double a = m1 * n1_23, b = m2 * n2_23;
    const double smax = P.lt * (m1 + m2) * vrel / (a > b ? a : b);
    if (P.lt > 0 && s > smax) s = smax;
    s *= D;
    double c, sn;
    coll_angle(s, U1, P.angle, c, sn);
    const double phi = 2 * M_PI * U2, sc = sn * cos(phi), ss = sn * sin(phi);
    const double pp = sqrt(px * px + py * py);
    double nx, ny, nz;
    if (pp > 1e-10 * ps) {
        nx = (px * pz * sc - py * ps * ss) / pp + px * c;
        ny = (py * pz * sc + px * ps * ss) / pp + py * c;
        nz = -pp * sc + pz * c;
    } else {
        nx = ps * sc; ny = ps * ss; nz = ps * c;
    }
    const double vcp = vc[0] * nx + vc[1] * ny + vc[2] * nz;
    if (upd1) {
        const double t = t1 * vcp + gc * E1s;
        u1[0] = (nx + vc[0] * t) / m1; u1[1] = (ny + vc[1] * t) / m1; u1[2] = (nz + vc[2] * t) / m1;
    }
    if (upd2) {
        const double t = -t1 * vcp + gc * E2s;
        u2[0] = (-nx + vc[0] * t) / m2; u2[1] = (-ny + vc[1] * t) / m2; u2[2] = (-nz + vc[2] * t) / m2;
    }
    return s;
}

// a[0 .. n) sorted by key[a[q]] (heap sort: no recursion, no scratch; the order of equal keys is not defined)
HD inline void coll_heap_sort(int* a, long n, const unsigned long long* key) {
    auto sift = [&](long r, long end) {
        while (2 * r + 1 < end) {
            long c = 2 * r + 1;
            if (c + 1 < end && key[a[c]] < key[a[c + 1]]) ++c;
            if (key[a[r]] < key[a[c]]) { const int t = a[r]; a[r] = a[c]; a[c] = t; r = c; }
            else return;
        }
    };
    for (long r = n / 2 - 1; r >= 0; --r) sift(r, n);
    for (long end = n - 1; end > 0; --end) {
        const int t = a[0]; a[0] = a[end]; a[end] = t;
        sift(0, end);
    }
}
