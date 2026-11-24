// Tilted slab with pre-plasma (O4, docs/specs/solid_target.md, decision 64): the profile p(s) along the slab normal,
// s = (x - x0) cos + (y - y0) sin,
//   exp(s / Lf) on [-Ef, 0),  1 on [0, T),  exp(-(s - T) / Lr) on [T, T + Er),  0 elsewhere,
// and the factor of a macro-particle: the point test of the plain slab (p at the particle, 0 or 1), or with `exact` the
// exact mean of p over the particle's sub-cell (closed forms, so the loaded charge is the integral of the profile to
// round-off for any tilt). Plain values and HD functions: the same arithmetic in the host and the device loaders.
#pragma once
#include "portable.hpp"

struct SlabProfile {
    double x0 = 0, y0 = 0, c = 1, s = 0;    // reference point and the normal (cos, sin) in the x-y plane
    double T = 0;                           // plateau thickness (0: no slab)
    double Lf = 0, Ef = 0, Lr = 0, Er = 0;  // front / rear scale lengths and extents (0: no ramp)
    int exact = 0;                          // 1: sub-cell means (always with a ramp)
};

namespace slab {

// a b + c d rounded as the host loop does (no fused multiply-add in the device code)
HD inline double dot2(double a, double b, double c, double d) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(__dmul_rn(a, b), __dmul_rn(c, d));
#else
    return a * b + c * d;
#endif
}

// phi1(z) = (e^z - 1) / z, phi2(z) = (e^z (z - 1) + 1) / z^2, for z <= 0 (series near 0)
HD inline void phis(double z, double& p1, double& p2) {
    if (z > -0.1) {
        double t = 1, f1 = 0, f2 = 0;                // t = z^k / k!
        for (int k = 0; k < 12; ++k) { f1 += t / (k + 1); f2 += t / (k + 2); t *= z / (k + 1); }
        p1 = f1; p2 = f2;
        return;
    }
    const double em = expm1(z);
    p1 = em / z;
    p2 = (z * (em + 1) - em) / (z * z);
}

// int_{u0}^{u1} (x - t)^k e^{b (t - ta)} dt for k = 0 (m = 0) or k = 1 (m = 1), u0 < u1, written from the end of the
// piece where the exponent is largest, so no exponential exceeds 1
HD inline double piece(double x, double u0, double u1, double ta, double b, int m) {
    const double D = u1 - u0;
    if (b == 0) return m ? D * (x - 0.5 * (u0 + u1)) : D;
    double p1, p2;
    if (b < 0) {                                     // e^{b (t - ta)} = e^{b (u0 - ta)} e^{b tau}, tau = t - u0
        phis(b * D, p1, p2);
        const double e = exp(b * (u0 - ta));
        return m ? e * ((x - u0) * D * p1 - D * D * p2) : e * D * p1;
    }
    phis(-b * D, p1, p2);                            // tau = u1 - t
    const double e = exp(b * (u1 - ta));
    return m ? e * ((x - u1) * D * p1 + D * D * p2) : e * D * p1;
}

// F_m(x) = int_{ref}^{x} (x - t)^m p(t) dt (m = 0: first, m = 1: second antiderivative of p from ref)
HD inline double antider(const SlabProfile& P, double ref, double x, int m) {
    const double lo = x < ref ? x : ref, hi = x < ref ? ref : x;
    // pieces [a, b) with p = e^{k (t - ta)}
    const double A[3] = {-P.Ef, 0.0, P.T}, B[3] = {0.0, P.T, P.T + P.Er};
    const double K[3] = {P.Lf > 0 ? 1 / P.Lf : 0.0, 0.0, P.Lr > 0 ? -1 / P.Lr : 0.0}, TA[3] = {0.0, 0.0, P.T};
    double v = 0;
    for (int q = 0; q < 3; ++q) {
        const double u0 = A[q] > lo ? A[q] : lo, u1 = B[q] < hi ? B[q] : hi;
        if (u1 > u0) v += piece(x, u0, u1, TA[q], K[q], m);
    }
    return x < ref ? -v : v;                         // int_ref^x = -int_x^ref
}

// The profile factor of a particle at r whose sub-cell has the sizes hs[0], hs[1] (x, y).
HD inline double factor(const SlabProfile& P, const double r[3], const double hs[2]) {
    const double sm = dot2(r[0] - P.x0, P.c, r[1] - P.y0, P.s);
    if (!P.exact) return sm >= 0 && sm < P.T ? 1.0 : 0.0;
    if (sm >= P.T + P.Er + hs[0] + hs[1] || sm < -P.Ef - hs[0] - hs[1]) return 0.0;   // the sub-cell lies outside
    const double ca = P.c * 0.5 * hs[0], sb = P.s * 0.5 * hs[1];
    double v;
    if (fabs(P.c * P.s) < 1e-6) {                    // aligned: one-dimensional along the larger component
        const double e = fabs(ca) > fabs(sb) ? ca : sb;
        v = (antider(P, sm, sm + e, 0) - antider(P, sm, sm - e, 0)) / (2 * e);
    } else {
        v = (antider(P, sm, sm + ca + sb, 1) - antider(P, sm, sm - ca + sb, 1) - antider(P, sm, sm + ca - sb, 1) +
             antider(P, sm, sm - ca - sb, 1)) / (4 * ca * sb);
    }
    return v > 0 ? v : 0.0;                          // round-off at the edges
}

}  // namespace slab
