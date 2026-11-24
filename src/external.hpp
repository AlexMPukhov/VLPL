// External magnetic quadrupoles acting on the particles only (blocks `quadrupole [<name>]`, docs/specs/structure3d.md,
// decision 102). Lab-fixed, hard-edged along x (the propagation axis), axis at (y_c, z_c):
//   B_y = -G (z - z_c),  B_z = -G (y - y_c)       (G > 0 focuses electrons moving along +x in y, defocuses them in z)
// i.e. the GEM-PIC-RZ / elegant convention B_x = -G y, B_y = -G x with (x, y, z) -> (y, z, x).
// The kick of step n uses the field at x^n, but a particle crosses [x^n - v_x dt / 2, x^n + v_x dt / 2] during the kick
// interval [t_n - dt/2, t_n + dt/2]: G is averaged over that interval (the overlap of the interval with each magnet),
// so an edge inside a step is exact to first order and the focusing strength of a magnet does not depend on where its
// edges fall between the particle positions (GEM-PIC-RZ PR #38).
// The magnets live in one process-wide device table (set_quadrupoles); the push kernels read it through PushParams.
#pragma once
#include <vector>
#include "portable.hpp"

struct Quad {
    double x0 = 0, x1 = 0;           // lab x range [x0, x1) (code units)
    double g = 0;                    // gradient (code: B per code length; > 0 focuses electrons in y)
    double yc = 0, zc = 0;           // axis (code units)
};

struct QuadView {
    const Quad* q = nullptr;
    int n = 0;
    // adds the averaged quadrupole field at (x, y, z) to B; dx = v_x dt, the x travelled during the kick interval
    HD void add(double x, double y, double z, double dx, double B[3]) const {
        const double a = x - 0.5 * fabs(dx), b = x + 0.5 * fabs(dx), L = b - a;
        for (int m = 0; m < n; ++m) {
            double f;
            if (L > 1e-12 * (fabs(x) + 1)) {
                const double lo = a > q[m].x0 ? a : q[m].x0, hi = b < q[m].x1 ? b : q[m].x1;
                f = hi > lo ? (hi - lo) / L : 0.0;
            } else f = (x >= q[m].x0 && x < q[m].x1) ? 1.0 : 0.0;
            if (f == 0.0) continue;
            const double G = f * q[m].g;
            B[1] -= G * (z - q[m].zc);
            B[2] -= G * (y - q[m].yc);
        }
    }
};

// Replaces the table (empty: none). Host code; kernels launched afterwards see it.
void set_quadrupoles(const std::vector<Quad>& q);
QuadView quadrupoles();              // the current table (n = 0: none)
