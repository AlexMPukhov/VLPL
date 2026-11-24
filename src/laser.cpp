#include "laser.hpp"
#include "laser_file.hpp"

// The left guard layer i = -1 at time t_n holds the wave that enters cell 0 at t_{n+1}:
// T(0)^{n+1} = T(-1)^n, so E_pol(-1) = a(t_n + dt) with the B of a wave moving along +x.
static void laser_zero(Fields& f, const Domain& dom) {
    const int g = dom.ng, gz = dom.gw(2), ey_ = dom.n[1] + 2 * g, ez_ = dom.ext(2);
    const FV Ey = f.ey.view(), Bz = f.bz.view(), Ez = f.ez.view(), By = f.by.view();
    par_for(long(ey_) * ez_, [=] HD(long q) {
        const int j = int(q % ey_) - g, k = int(q / ey_) - gz;
        Ey(-1, j, k) = 0; Bz(-1, j, k) = 0; Ez(-1, j, k) = 0; By(-1, j, k) = 0;
    });
}

static void laser_add(Fields& f, const LaserConfig L, const Domain& dom, double t) {
    const double tl = t + dom.dt(), hy = dom.h[1], hz = dom.h[2];
    const int g = dom.ng, gz = dom.gw(2), ey_ = dom.n[1] + 2 * g, ez_ = dom.ext(2), o1 = dom.off[1], o2 = dom.off[2];
    const FV Ey = f.ey.view(), Bz = f.bz.view(), Ez = f.ez.view(), By = f.by.view();
    par_for(long(ey_) * ez_, [=] HD(long q) {
        const int j = int(q % ey_) - g, k = int(q / ey_) - gz;
        const double y0 = (o1 + j) * hy, z0 = (o2 + k) * hz;
        if (L.pol == 'y') {                       // Ey, Bz at (j+1/2, k): T+_y = Ey + Bz
            const double a = L.field(tl, y0 + 0.5 * hy, z0);
            Ey(-1, j, k) += a; Bz(-1, j, k) += a;
        } else {                                  // Ez, By at (j, k+1/2): T-_z = Ez - By
            const double a = L.field(tl, y0, z0 + 0.5 * hz);
            Ez(-1, j, k) += a; By(-1, j, k) += -a;
        }
    });
}

// A file laser: both transverse components, E_y with B_z (T+_y) and E_z with -B_y (T-_z).
static void laser_add_file(Fields& f, const FileLaserView L, const Domain& dom, double t) {
    const double tl = t + dom.dt(), hy = dom.h[1], hz = dom.h[2];
    const int g = dom.ng, gz = dom.gw(2), ey_ = dom.n[1] + 2 * g, ez_ = dom.ext(2), o1 = dom.off[1], o2 = dom.off[2];
    const FV Ey = f.ey.view(), Bz = f.bz.view(), Ez = f.ez.view(), By = f.by.view();
    par_for(long(ey_) * ez_, [=] HD(long q) {
        const int j = int(q % ey_) - g, k = int(q / ey_) - gz;
        const double y0 = (o1 + j) * hy, z0 = (o2 + k) * hz;
        const double ay = L.field(0, tl, y0 + 0.5 * hy, z0), az = L.field(1, tl, y0, z0 + 0.5 * hz);
        Ey(-1, j, k) += ay; Bz(-1, j, k) += ay;
        Ez(-1, j, k) += az; By(-1, j, k) += -az;
    });
}

void add_laser_inflow(BoundaryHooks& hooks, const std::vector<LaserConfig>& lasers,
                      const std::vector<std::shared_ptr<FileLaser>>& files, const Domain& d, const Comm& comm) {
    std::vector<LaserConfig> on;
    for (const LaserConfig& L : lasers)
        if (L.enabled && L.tfsf_x < 0 && !L.antenna) on.push_back(L);   // TFSF lasers: their plane (tfsf.cpp); antenna: boost_antenna.cpp
    if ((on.empty() && files.empty()) || comm.nbr[0][0] >= 0) return;
    const Domain* dp = &d;                  // the simulation's domain: the hook sees the window shift
    hooks.inflow.push_back([on, files, dp](Fields& f, double t) {
        if (dp->wx != 0) return;            // after the first shift the inbound T stays 0
        laser_zero(f, *dp);
        for (const LaserConfig& L : on) laser_add(f, L, *dp, t);
        for (const auto& F : files) laser_add_file(f, F->view(), *dp, t);
    });
}
