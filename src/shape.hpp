// Particle shape functions (B-splines of order 1, 2 and 3) on a uniform grid axis.
#pragma once
#include "portable.hpp"

template <int O> struct Shape;
template <> struct Shape<1> {                       // linear: nodes floor(u), floor(u)+1
    static constexpr int W = 2;
    HD static inline void eval(double u, int& i0, double* w) {
        const double fl = floor(u), f = u - fl;
        i0 = int(fl); w[0] = 1.0 - f; w[1] = f;
    }
};
template <> struct Shape<2> {                       // quadratic: nearest node and its two neighbours
    static constexpr int W = 3;
    HD static inline void eval(double u, int& i0, double* w) {
        const double c = floor(u + 0.5), f = u - c;
        i0 = int(c) - 1;
        w[0] = 0.5 * (0.5 - f) * (0.5 - f); w[1] = 0.75 - f * f; w[2] = 0.5 * (0.5 + f) * (0.5 + f);
    }
};
template <> struct Shape<3> {                       // cubic: nodes floor(u)-1 .. floor(u)+2 (docs/specs/solid_target.md)
    static constexpr int W = 4;
    HD static inline void eval(double u, int& i0, double* w) {
        const double fl = floor(u), f = u - fl, g = 1.0 - f;
        i0 = int(fl) - 1;
        w[0] = g * g * g / 6.0; w[1] = 2.0 / 3.0 - f * f * (1.0 - 0.5 * f);
        w[2] = 2.0 / 3.0 - g * g * (1.0 - 0.5 * g); w[3] = f * f * f / 6.0;
    }
};

