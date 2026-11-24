// Laser through the open left z edge in RZ (docs/specs/rz_modes.md, step 3): the Cartesian LaserConfig
// (laser.hpp: focus, Laguerre-Gauss, phase) evaluated at the azimuthal quadrature points of each radial
// node and projected on the modes, F_m = (1/2 pi) int F e^{-i m phi} dphi. As in Cart3D (laser.cpp) the
// guard layer i = -1 at t_n holds the wave that enters cell 0 at t_{n+1}.
#include <cmath>
#include "laser.hpp"
#include "laser_file.hpp"
#include "rz.hpp"

namespace {
HD inline int slot(int m, int c) { return m == 0 ? 0 : 2 * m - 1 + c; }

void rz_laser_zero(FieldsRZ& f) {
    const Domain& d = f.dom();
    const int nr = d.n[1], S = d.n[2];
    const FV Er = f.er.view(), Bp = f.bp.view(), Ep = f.ep.view(), Br = f.br.view();
    par_for(long(nr) * S, [=] HD(long q) {
        const int j = int(q % nr), k = int(q / nr);
        Er(-1, j, k) = 0; Bp(-1, j, k) = 0; Ep(-1, j, k) = 0; Br(-1, j, k) = 0;
    });
}

// Quadrature with np points: exact for the modes m < M of a field with azimuthal content below np - M.
void rz_laser_add(FieldsRZ& f, const LaserConfig L, double t) {
    const Domain& d = f.dom();
    const int nr = d.n[1], M = f.modes(), np = 4 * (M + 8);
    const double tl = t + d.dt();
    const FV Er = f.er.view(), Bp = f.bp.view(), Ep = f.ep.view(), Br = f.br.view();
    const RadialView g = f.grid().view();
    par_for(long(nr) * M, [=] HD(long q) {
        const int j = int(q % nr), m = int(q / nr);
        const double rh = g.rh[j], rn = g.r[j];
        double er[2] = {0, 0}, ep[2] = {0, 0};
        for (int k = 0; k < np; ++k) {
            const double ph = 2 * M_PI * k / np, c = cos(ph), s = sin(ph), cm = cos(m * ph), sm = sin(m * ph);
            const double ah = L.field(tl, rh * c, rh * s), an = L.field(tl, rn * c, rn * s);
            // pol 'y': E along x of the particles, 'z': along y
            const double vr = L.pol == 'y' ? ah * c : ah * s, vp = L.pol == 'y' ? -an * s : an * c;
            er[0] += vr * cm; er[1] -= vr * sm;
            ep[0] += vp * cm; ep[1] -= vp * sm;
        }
        for (int c = 0; c < (m ? 2 : 1); ++c) {
            const double a = er[c] / np, b = ep[c] / np;
            Er(-1, j, slot(m, c)) += a; Bp(-1, j, slot(m, c)) += a;      // E_r + B_phi
            Ep(-1, j, slot(m, c)) += b; Br(-1, j, slot(m, c)) += -b;     // E_phi - B_r
        }
    });
}
// a LASY laser (laser_file.hpp): E_x, E_y of the particles at the quadrature points, E_r at rh_j, E_phi at r_j
void rz_laser_add_file(FieldsRZ& f, const RzFileLaserView L, double t) {
    const Domain& d = f.dom();
    const int nr = d.n[1], M = f.modes(), np = 4 * (M + 8);
    const double tl = t + d.dt();
    const FV Er = f.er.view(), Bp = f.bp.view(), Ep = f.ep.view(), Br = f.br.view();
    const RadialView g = f.grid().view();
    par_for(long(nr) * M, [=] HD(long q) {
        const int j = int(q % nr), m = int(q / nr);
        const double rh = g.rh[j], rn = g.r[j];
        double er[2] = {0, 0}, ep[2] = {0, 0};
        for (int k = 0; k < np; ++k) {
            const double ph = 2 * M_PI * k / np, c = cos(ph), s = sin(ph), cm = cos(m * ph), sm = sin(m * ph);
            double xh, yh, xn, yn;
            L.field(tl, rh, c, s, xh, yh);
            L.field(tl, rn, c, s, xn, yn);
            const double vr = xh * c + yh * s, vp = -xn * s + yn * c;
            er[0] += vr * cm; er[1] -= vr * sm;
            ep[0] += vp * cm; ep[1] -= vp * sm;
        }
        for (int c = 0; c < (m ? 2 : 1); ++c) {
            const double a = er[c] / np, b = ep[c] / np;
            Er(-1, j, slot(m, c)) += a; Bp(-1, j, slot(m, c)) += a;
            Ep(-1, j, slot(m, c)) += b; Br(-1, j, slot(m, c)) += -b;
        }
    });
}
}  // namespace

void rz_add_laser_inflow(FieldsRZ& f, const std::vector<LaserConfig>& lasers, const Comm& comm,
                         const std::vector<std::shared_ptr<RzFileLaser>>& files) {
    std::vector<LaserConfig> on;
    for (const LaserConfig& L : lasers)
        if (L.enabled && !L.antenna) on.push_back(L);   // antenna lasers: boost_antenna.cpp
    if ((on.empty() && files.empty()) || comm.nbr[0][0] >= 0) return;
    f.inflow.push_back([on, files](FieldsRZ& F, double t) {
        if (F.dom().wx != 0) return;            // after the first window shift the inbound T stays 0
        rz_laser_zero(F);
        for (const LaserConfig& L : on) rz_laser_add(F, L, t);
        for (const auto& L : files) rz_laser_add_file(F, L->view(), t);
    });
}
