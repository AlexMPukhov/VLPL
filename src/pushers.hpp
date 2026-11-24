// The momentum pushers (Boris, Higuera-Cary) as inline device functions, for every translation unit with
// particle kernels (the CUDA build has no relocatable device code: device functions are not called
// across files).
#pragma once
#include <cmath>
#include "particles.hpp"

template <Pusher PU> HD inline void push_momentum_t(double qmdt, const double E[3], const double B[3], double u[3]) {
    const double h = 0.5 * qmdt;
    double um[3] = {u[0] + h * E[0], u[1] + h * E[1], u[2] + h * E[2]};
    const double b[3] = {h * B[0], h * B[1], h * B[2]};
    const double um2 = um[0] * um[0] + um[1] * um[1] + um[2] * um[2];
    double g;
    if (PU == Pusher::Boris) {
        g = sqrt(1.0 + um2);
    } else {                                        // Higuera & Cary, Phys. Plasmas 24, 052104 (2017)
        const double b2 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
        const double ub = um[0] * b[0] + um[1] * b[1] + um[2] * b[2];
        const double sig = 1.0 + um2 - b2;
        g = sqrt(0.5 * (sig + sqrt(sig * sig + 4.0 * (b2 + ub * ub))));
    }
    const double t[3] = {b[0] / g, b[1] / g, b[2] / g};
    const double t2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
    const double s = 2.0 / (1.0 + t2);
    const double up[3] = {um[0] + (um[1] * t[2] - um[2] * t[1]),          // u' = u- + u- x t
                          um[1] + (um[2] * t[0] - um[0] * t[2]),
                          um[2] + (um[0] * t[1] - um[1] * t[0])};
    const double upl[3] = {um[0] + s * (up[1] * t[2] - up[2] * t[1]),     // u+ = u- + s u' x t
                           um[1] + s * (up[2] * t[0] - up[0] * t[2]),
                           um[2] + s * (up[0] * t[1] - up[1] * t[0])};
    u[0] = upl[0] + h * E[0]; u[1] = upl[1] + h * E[1]; u[2] = upl[2] + h * E[2];
}

// u <- u + q/m dt (E + v x B) by the chosen pusher
HD inline void push_momentum_inline(Pusher p, double qmdt, const double E[3], const double B[3], double u[3]) {
    if (p == Pusher::Boris) push_momentum_t<Pusher::Boris>(qmdt, E, B, u);
    else if (p == Pusher::HigueraCary) push_momentum_t<Pusher::HigueraCary>(qmdt, E, B, u);   // None: unchanged
}
