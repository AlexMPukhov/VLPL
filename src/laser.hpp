// Laser injected through the open left x boundary (inbound transport vector, decision 5). A boundary
// hook of the field step (BoundaryHooks in fields.hpp), deck blocks `laser` and `laser <name>` (summed;
// docs/specs/laser_modes.md, decision 32).
// At the entrance plane x = 0, for a pulse focused at x = x_f (X = -x_f, distance from the focus),
// paraxial, separable in y and z (waists w0, w0z; w0z = 0: uniform in z), code units (k = omega = 1):
//   z_R = W^2 / 2,  w = W sqrt(1 + (X / z_R)^2),  1/R = X / (X^2 + z_R^2),  Gouy psi = atan(X / z_R)
//   E_pol = a0 prod_a sqrt(W/w) exp(-r_a^2 / w^2) g(t') sin(t' + psi/2 per axis + phase),
//   t' = t - tc - sum_a r_a^2 / (2 R),   g = exp(-2 ln2 t'^2 / tau^2)
// Laguerre-Gauss (round beam, w0z = w0): the factor (sqrt(u))^|l| L_p^|l|(u), u = 2 r^2 / w^2, the
// phase l theta and the Gouy phase (2p + |l| + 1) psi. With x_f = 0, p = l = 0, phase = 0 this is the
// Gaussian at its focus, computed by the original expression (bit-identical).
#pragma once
#include <memory>
#include <vector>
#include "comm.hpp"
#include "fields.hpp"

struct LaserConfig {
    bool enabled = false;
    double a0 = 1, tau = 10, tc = 20, w0 = 10, w0z = 0, yc = 0, zc = 0;   // code units; w0z = 0: uniform in z
    char pol = 'z';                                                       // 'y' (Ey, Bz) or 'z' (Ez, By)
    double xf = 0;                     // focus position along x (the entrance plane is x = 0)
    int p = 0, l = 0;                  // Laguerre-Gauss indices (radial, azimuthal)
    double phase = 0;                  // added to the carrier phase (e.g. pi/2 for circular polarisation)
    double tfsf_x = -1, angle = 0;     // O1 (docs/specs/tfsf.md): TFSF plane (code units; < 0: entrance plane x = 0), angle about z
                                       // from x towards y (rad)
    bool tfsf_modes = false;           // laser.tfsf_incident = modes: exact discrete plane waves at the plane (tfsf_modes.hpp)
    bool cut = false;                  // envelope with compact support (TFSF lasers, docs/specs/tfsf.md): the Gaussian up to
                                       // 2 sigma (sigma = tau / (2 sqrt(ln 2)) of the field), a C2 spline step to 0 at 3 sigma
    // the cut-off factor of the envelope at the time te from the peak (1 up to 2 sigma, 0 from 3 sigma)
    HD double taper(double te) const {
        const double u = fabs(te) / (0.6005612043932249 * tau);
        if (u <= 2) return 1;
        if (u >= 3) return 0;
        const double v = u - 2;
        return 1 - v * v * v * (10 - v * (15 - 6 * v));
    }
    // boosted frame (boost.hpp, decision 87): the lab-frame beam above seen at the boosted entrance plane x' = 0 (inbound
    // E + B = gamma (1 - beta) (E + B)_lab at the lab event of (x', t')); bg = 1: the lab run (bit-identical)
    double bg = 1, bb = 0, bX0 = 0, bT0 = 0;
    // laser.injection = antenna (boosted runs; boost_antenna.cpp, decision 89): a current sheet fixed at the lab plane x = 0
    // (moving at -beta c in the boosted frame) instead of the boosted entrance plane; a lab run injects at x = 0 as before
    bool antenna = false;
    HD double field(double t, double y, double z) const {
        if (bg > 1) return field_boosted(t, 0.0, y, z);
        return field_at(t, 0.0, y, z);
    }
    HD double field_boosted(double tp, double xp, double y, double z) const {
        return bg * (1 - bb) * field_at(bg * (tp + bb * xp) + bT0, bg * (xp + bb * tp) + bX0, y, z);
    }
    // the paraxial beam at the plane x (lab frame): focus distance X = x - x_f, retarded time t - x
    HD double field_at(double t, double x, double y, double z) const {
        const double dt = t - x - tc;
        if (x == xf && p == 0 && l == 0 && phase == 0) {
            double r2 = (y - yc) * (y - yc) / (w0 * w0);
            if (w0z > 0) r2 += (z - zc) * (z - zc) / (w0z * w0z);
            const double f = a0 * exp(-1.3862943611198906 * dt * dt / (tau * tau) - r2) * sin(dt);     // 2 ln 2
            return cut ? f * taper(dt) : f;
        }
        const double X = x - xf, dy = y - yc, dz = z - zc;
        double amp = 1, r2 = 0, curv = 0, gouy = 0, wy = w0;
        {
            const double zr = 0.5 * w0 * w0, s = 1 + X * X / (zr * zr);
            wy = w0 * sqrt(s);
            amp *= sqrt(1 / sqrt(s));
            r2 += dy * dy / (wy * wy);
            curv += 0.5 * dy * dy * X / (X * X + zr * zr);
            gouy += 0.5 * atan(X / zr);
        }
        if (w0z > 0) {
            const double zr = 0.5 * w0z * w0z, s = 1 + X * X / (zr * zr), wz = w0z * sqrt(s);
            amp *= sqrt(1 / sqrt(s));
            r2 += dz * dz / (wz * wz);
            curv += 0.5 * dz * dz * X / (X * X + zr * zr);
            gouy += 0.5 * atan(X / zr);
        }
        double lg = 1, ang = 0;
        if (p != 0 || l != 0) {                       // round beam: w0z = w0 (checked in Config::check)
            const int al = l < 0 ? -l : l;
            const double u = 2 * (dy * dy + dz * dz) / (wy * wy);
            double lm = 1, lk = 1 + al - u;           // generalised Laguerre L_p^al(u) by the recurrence
            if (p == 0) lk = 1;
            for (int k = 1; k < p; ++k) {
                const double ln = ((2 * k + 1 + al - u) * lk - (k + al) * lm) / (k + 1);
                lm = lk; lk = ln;
            }
            lg = pow(sqrt(u), al) * lk;
            ang = l * atan2(dz, dy);
            gouy *= 2 * p + al + 1;                   // (2p + |l| + 1) atan(X / z_R)
        }
        const double te = dt - curv;
        const double f = a0 * amp * lg * exp(-1.3862943611198906 * te * te / (tau * tau) - r2) * sin(te + gouy + ang + phase);
        return cut ? f * taper(te) : f;
    }
};

class FileLaser;
// Adds the inflow hook on the ranks at the open left x boundary for all enabled lasers, built-in and
// from files (laser_file.hpp), summed.
void add_laser_inflow(BoundaryHooks& hooks, const std::vector<LaserConfig>& lasers,
                      const std::vector<std::shared_ptr<FileLaser>>& files, const Domain& d, const Comm& comm);
