// The lab-fixed laser antenna of a boosted run (laser.injection = antenna; docs/specs/boosted_frame.md "Antenna", decision 89).
//
// Lab frame: a current sheet at the entrance plane x = x_a = 0, K = -2 E_perp,laser(t, x_a, y, z) (the tangential B of the
// forward beam fixes K: B_z(x_a+) = -K_y / 2; paraxial B_z = E_y), radiating the deck's paraxial beam forwards (and its
// mirror image backwards, out of the box). Its surface charge sigma = -div_perp Pi, Pi = int K dt: a polarization sheet
// P = Pi delta(x - x_a). Boosted frame: the sheet moves at -beta c, x'_a(t') = (x_a - X0) / gamma - beta t'; with
// Pi~(t') = Pi(t_lab(t')) the transformed sources are rho' = sigma~ delta(x' - x'_a), J'_perp = dPi~/dt' delta,
// J'_x = -beta rho' (the convection of the sheet's charge), dPi~/dt' = K(t_lab) / gamma.
//
// Discretisation (charge-conserving, exact discrete continuity with the stencil of the Gauss law of RIP-CC / iRIP):
//   Pi^{n+1} = Pi^n + (dt / gamma) K(t_lab(t'_{n+1/2}))         (transverse plane, state; restarts re-sum it)
//   sigma^n  = -div_perp Pi^n                                   (at the rho points: Cart (j, k), RZ r_j per mode)
//   w^n_i    = quadratic spline of x'_a(t_n) at the rho nodes x_{i+1/2}, C^n(i) = sum_{i' < i} w^n_{i'}
//   rho^n    = sigma^n w^n_i / h_x
//   J_perp   = (Pi^{n+1} - Pi^n) / dt  (w^n_i + w^{n+1}_i) / (2 h_x)        (at i + 1/2, where J_y, J_z / J_r, J_phi live)
//   J_x(i)   = -(sigma^n + sigma^{n+1}) / 2  (C^{n+1}(i) - C^n(i)) / dt    (at the integer nodes, as the Esirkepov split)
// rho^{n+1} - rho^n = [(sigma^{n+1} - sigma^n) wbar + sigmabar (w^{n+1} - w^n)] / h_x = -dt div J exactly, so the solver
// keeps dx Ex + mu_x div_perp E = rho_particles + rho_antenna at round-off. compute_rho / rz_rho_all add rho^n.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "boost.hpp"
#include "laser.hpp"
#include "laser_file.hpp"
#include "rz.hpp"
#include "sim.hpp"

struct BoostAntenna {
    std::vector<LaserConfig> lasers;
    std::vector<FileLaserView> cfiles;              // LASY lasers (Cart3D, 2D): the field at the lab plane x = 0
    std::vector<RzFileLaserView> rfiles;            // LASY lasers (RZ)
    double g = 1, b = 0, X0 = 0, T0 = 0, xa = 0;    // the boost; the antenna's lab position (the entrance plane x = 0)
    bool rz = false, flat = false, per1 = false, per2 = false;
    int n1 = 0, n2 = 0, nk = 1, o1 = 0, o2 = 0, N1 = 0, N2 = 0;   // Cart: local transverse cells, rows of the Pi arrays along z
    double hy = 1, hz = 1;
    int M = 0, S = 0;                                // RZ: modes, slots
    size_t np = 0;                                   // points per component
    Buf<double> pa, pb;                              // Pi^n, Pi^{n+1}: two components (y, z; RZ r at rh_j, phi at r_j)
    long n = 0;                                      // the step of pa
    double sw = 1, c2 = 0;                           // the stretch of the spline (cells), the pre-filter of K (ant_w, ant_field)
    double xp(double tp) const { return (xa - X0) / g - b * tp; }                // x'_a(t')
    double tlab(double tp) const { return g * (tp + b * xp(tp)) + T0; }        // the lab time on the antenna's world line
};

namespace {

// The sheet's weight on node i at the node coordinate xi: the quadratic B-spline stretched to s = (1 + beta) c dt / h_x cells,
// normalised to sum 1 at every step. RIP moves the forward wave exactly c dt per step while the sheet moves -beta c dt: a forward
// characteristic meets the sheet's nodes s cells apart, and the stretched spline is a partition of unity at that spacing, so
// every characteristic receives the same total weight 1/s (a unit-width spline sampled every 1.87 cells at gamma 2 modulated the
// emitted beam at the grid scale by ~20 %: 7 % error in the first test). C(i) = sum_{i' < i} w_{i'} (0 left of the support,
// 1 right of it up to round-off).
HD inline double ant_b2(double u) {
    const double d = fabs(u);
    return d < 0.5 ? 0.75 - d * d : d < 1.5 ? 0.5 * (1.5 - d) * (1.5 - d) : 0.0;
}
HD inline double ant_w(double xi, double s, long i) {
    const long kl = long(ceil(xi - 1.5 * s)), kh = long(floor(xi + 1.5 * s));
    if (i < kl || i > kh) return 0;
    double nrm = 0;
    for (long k = kl; k <= kh; ++k) nrm += ant_b2((k - xi) / s);
    return ant_b2((i - xi) / s) / nrm;
}
HD inline double ant_c(double xi, double s, long i) {
    const long kl = long(ceil(xi - 1.5 * s)), kh = long(floor(xi + 1.5 * s));
    if (i <= kl) return 0;
    if (i > kh) return 1;
    double nrm = 0, lo = 0, hi = 0;
    for (long k = kl; k <= kh; ++k) {
        const double w = ant_b2((k - xi) / s);
        nrm += w;
        if (k < i) lo += w; else hi += w;
    }
    return i <= long(floor(xi + 0.5)) ? lo / nrm : 1 - hi / nrm;
}

// The lab field of laser L at its antenna point, pre-compensated for the time smoothing of the sheet: a characteristic sums the
// samples K_{n+1/2} with the stretched spline's weights (variance var in steps), K + (var / 2) K'' to leading order; the samples
// K - c2 (K_+ - 2 K + K_-) (c2 = var / 2, K_+- one boosted step later / earlier on the world line) cancel it to fourth order.
HD inline double ant_field(const LaserConfig& L, const double tl[3], double c2, double x, double y, double z) {
    double v[3];
    for (int q = 0; q < 3; ++q) v[q] = tl[q] > 0 ? L.field_at(tl[q], x, y, z) : 0.0;
    return v[1] - c2 * (v[2] - 2 * v[1] + v[0]);
}

// Pi^{n+1} = Pi^n + f K(tl): Cart3D / 2D, the points j = -1 .. n1 - 1, k = -1 .. n2 - 1 (flat: k = 0) at global indices wrapped
// at periodic edges (a guard holds what the neighbour's owned point holds; outside an open edge 0, as the current there)
void advance_cart(BoostAntenna& A, const double tl3[3], double f) {
    const double tl[3] = {tl3[0], tl3[1], tl3[2]}, c2 = A.c2;
    const int n1p = A.n1 + 1, nk = A.nk, o1 = A.o1, o2 = A.o2, N1 = A.N1, N2 = A.N2;
    const bool flat = A.flat, per1 = A.per1, per2 = A.per2;
    const double hy = A.hy, hz = A.hz, xa = A.xa;
    const size_t np = A.np;
    const double* P = A.pa.data();
    double* Q = A.pb.data();
    par_for(long(2 * np), [=] HD(long q) { Q[q] = P[q]; });
    if (!(tl[2] > 0)) return;                           // the lab run injects from t = 0 on
    for (const LaserConfig& L0 : A.lasers) {
        const LaserConfig L = L0;
        par_for(long(n1p) * nk, [=] HD(long q) {
            const int j = int(q % n1p) - 1, k = flat ? 0 : int(q / n1p) - 1;
            long jg = o1 + j, kg = o2 + k;
            if (jg < 0) { if (!per1) return; jg += N1; }
            if (kg < 0) { if (!per2) return; kg += N2; }
            const double y0 = jg * hy, z0 = kg * hz;
            if (L.pol == 'y') Q[q] += -2 * f * ant_field(L, tl, c2, xa, y0 + 0.5 * hy, z0);
            else Q[np + q] += -2 * f * ant_field(L, tl, c2, xa, y0, z0 + 0.5 * hz);
        });
    }
    for (const FileLaserView& F0 : A.cfiles) {         // LASY: the field of the file at the lab plane x = 0 (laser_file.hpp)
        const FileLaserView F = F0;
        par_for(long(n1p) * nk, [=] HD(long q) {
            const int j = int(q % n1p) - 1, k = flat ? 0 : int(q / n1p) - 1;
            long jg = o1 + j, kg = o2 + k;
            if (jg < 0) { if (!per1) return; jg += N1; }
            if (kg < 0) { if (!per2) return; kg += N2; }
            const double y0 = jg * hy, z0 = kg * hz;
            double vy[3], vz[3];
            for (int p = 0; p < 3; ++p) {
                vy[p] = tl[p] > 0 ? F.field(0, tl[p], y0 + 0.5 * hy, z0) : 0.0;
                vz[p] = tl[p] > 0 ? F.field(1, tl[p], y0, z0 + 0.5 * hz) : 0.0;
            }
            Q[q] += -2 * f * (vy[1] - c2 * (vy[2] - 2 * vy[1] + vy[0]));
            Q[np + q] += -2 * f * (vz[1] - c2 * (vz[2] - 2 * vz[1] + vz[0]));
        });
    }
}

// E_r at rh and E_phi at rn of a LASY laser, pre-compensated as ant_field
HD inline void ant_file_rz(const RzFileLaserView& F, const double tl[3], double c2, double rh, double rn, double c, double s,
                           double& vr, double& vp) {
    double a[3], b[3];
    for (int p = 0; p < 3; ++p) {
        double xh = 0, yh = 0, xn = 0, yn = 0;
        if (tl[p] > 0) { F.field(tl[p], rh, c, s, xh, yh); F.field(tl[p], rn, c, s, xn, yn); }
        a[p] = xh * c + yh * s; b[p] = -xn * s + yn * c;
    }
    vr = a[1] - c2 * (a[2] - 2 * a[1] + a[0]);
    vp = b[1] - c2 * (b[2] - 2 * b[1] + b[0]);
}

// RZ: per radial node j and mode m the projection F_m = (1/2 pi) int F e^{-i m phi} dphi (as rz_laser.cpp), E_r at rh_j, E_phi at r_j
void advance_rz(BoostAntenna& A, const RadialView g, const double tl3[3], double f) {
    const double tl[3] = {tl3[0], tl3[1], tl3[2]}, c2 = A.c2;
    const int nr = A.n1, M = A.M, npq = 4 * (M + 8);
    const double xa = A.xa;
    const size_t np = A.np;
    const double* P = A.pa.data();
    double* Q = A.pb.data();
    par_for(long(2 * np), [=] HD(long q) { Q[q] = P[q]; });
    if (!(tl[2] > 0)) return;
    for (const LaserConfig& L0 : A.lasers) {
        const LaserConfig L = L0;
        par_for(long(nr) * M, [=] HD(long q) {
            const int j = int(q % nr), m = int(q / nr);
            const double rh = g.rh[j], rn = g.r[j];
            double er[2] = {0, 0}, ep[2] = {0, 0};
            for (int k = 0; k < npq; ++k) {
                const double ph = 2 * M_PI * k / npq, c = cos(ph), s = sin(ph), cm = cos(m * ph), sm = sin(m * ph);
                const double ah = ant_field(L, tl, c2, xa, rh * c, rh * s), an = ant_field(L, tl, c2, xa, rn * c, rn * s);
                const double vr = L.pol == 'y' ? ah * c : ah * s, vp = L.pol == 'y' ? -an * s : an * c;
                er[0] += vr * cm; er[1] -= vr * sm;
                ep[0] += vp * cm; ep[1] -= vp * sm;
            }
            for (int c = 0; c < (m ? 2 : 1); ++c) {
                const int sl = m == 0 ? 0 : 2 * m - 1 + c;
                Q[size_t(sl) * nr + j] += -2 * f * er[c] / npq;
                Q[np + size_t(sl) * nr + j] += -2 * f * ep[c] / npq;
            }
        });
    }
    for (const RzFileLaserView& F0 : A.rfiles) {
        const RzFileLaserView F = F0;
        par_for(long(nr) * M, [=] HD(long q) {
            const int j = int(q % nr), m = int(q / nr);
            const double rh = g.rh[j], rn = g.r[j];
            double er[2] = {0, 0}, ep[2] = {0, 0};
            for (int k = 0; k < npq; ++k) {
                const double ph = 2 * M_PI * k / npq, c = cos(ph), s = sin(ph), cm = cos(m * ph), sm = sin(m * ph);
                double vr, vp;
                ant_file_rz(F, tl, c2, rh, rn, c, s, vr, vp);
                er[0] += vr * cm; er[1] -= vr * sm;
                ep[0] += vp * cm; ep[1] -= vp * sm;
            }
            for (int c = 0; c < (m ? 2 : 1); ++c) {
                const int sl = m == 0 ? 0 : 2 * m - 1 + c;
                Q[size_t(sl) * nr + j] += -2 * f * er[c] / npq;
                Q[np + size_t(sl) * nr + j] += -2 * f * ep[c] / npq;
            }
        });
    }
}

void advance(BoostAntenna& A, const Simulation& s) {
    const double dt = s.dom.dt(), tm = (double(A.n) + 0.5) * dt;
    const double tl[3] = {A.tlab(tm - dt), A.tlab(tm), A.tlab(tm + dt)};
    if (A.rz) advance_rz(A, s.rz->grid().view(), tl, dt / A.g);
    else advance_cart(A, tl, dt / A.g);
}

void swap_step(BoostAntenna& A) { A.pa.swap(A.pb); ++A.n; }

// Pi^n at the step of the simulation (a restart re-sums it from step 0)
void sync(BoostAntenna& A, const Simulation& s) {
    if (A.n > s.istep) { A.pa.zero(); A.n = 0; }
    while (A.n < s.istep) { advance(A, s); swap_step(A); }
}

// the local x range [i0, i1] touched by the sheet at the node coordinates xa, xb (half nodes: xi = x' / h - 1/2), or i0 > i1
void x_range(const Domain& d, double sw, double xia, double xib, int& i0, int& i1) {
    const long c0 = d.cell0(0), r = long(std::ceil(1.5 * sw)) + 2;
    const long lo = long(std::floor(std::min(xia, xib))) - r, hi = long(std::floor(std::max(xia, xib))) + r + 1;
    i0 = int(std::max<long>(lo - c0, 0));
    i1 = int(std::min<long>(hi - c0, d.n[0] - 1));
}

void deposit_cart(BoostAntenna& A, Fields& F, const Domain& d, double xia, double xib, int i0, int i1) {
    const int n1p = A.n1 + 1, ny = d.n[1], nz = d.n[2], ni = i1 - i0 + 1;
    const bool flat = A.flat;
    const double dt = d.dt(), ihx = 1.0 / d.h[0], iy = 1.0 / A.hy, iz = 1.0 / A.hz;
    const long c0 = d.cell0(0);
    const double sw = A.sw;
    const size_t np = A.np;
    const double *P = A.pa.data(), *Q = A.pb.data();
    const FV Jx = F.jx.view(), Jy = F.jy.view(), Jz = F.jz.view();
    par_for(long(ni) * ny * nz, [=] HD(long q) {
        const int i = i0 + int(q % ni), j = int((q / ni) % ny), k = int(q / (long(ni) * ny));
        const long ig = c0 + i;
        const long p = flat ? j + 1 : long(k + 1) * n1p + (j + 1), pj = p - 1, pk = flat ? p : p - n1p;
        const double sa = -((P[p] - P[pj]) * iy + (P[np + p] - P[np + pk]) * iz);
        const double sb = -((Q[p] - Q[pj]) * iy + (Q[np + p] - Q[np + pk]) * iz);
        const double wm = 0.5 * (ant_w(xia, sw, ig) + ant_w(xib, sw, ig)) * ihx;
        Jy(i, j, k) += (Q[p] - P[p]) / dt * wm;
        Jz(i, j, k) += (Q[np + p] - P[np + p]) / dt * wm;
        Jx(i, j, k) += -0.5 * (sa + sb) * (ant_c(xib, sw, ig) - ant_c(xia, sw, ig)) / dt;
    });
}

// RZ per mode: div_n of the RZ continuity (medium_add_rho_rz): (1/A_j)(rh_j Pr_j - rh_{j-1} Pr_{j-1}) + (i m / r_j) Pphi_j; the
// axis is a Gauss point of m = 0 only (rh_0 Pr_0 / A_0)
HD inline void ant_div_rz(const double* P, size_t np, int nr, const RadialView& g, int j, int m, double dv[2]) {
    const int s0 = m == 0 ? 0 : 2 * m - 1;
    for (int c = 0; c < (m ? 2 : 1); ++c) {
        const size_t q = size_t(s0 + c) * nr + j;
        dv[c] = j == 0 ? g.rh[0] * P[q] / g.A[0] : (g.rh[j] * P[q] - g.rh[j - 1] * P[q - 1]) / g.A[j];
    }
    if (m && j > 0) {                                // + i (m / r_j) Pphi: re -= s im, im += s re
        const double s = m / g.r[j], re = P[np + size_t(s0) * nr + j], im = P[np + size_t(s0 + 1) * nr + j];
        dv[0] -= s * im; dv[1] += s * re;
    }
}

void deposit_rz(BoostAntenna& A, FieldsRZ& F, const Domain& d, double xia, double xib, int i0, int i1) {
    const int nr = A.n1, M = A.M, ni = i1 - i0 + 1;
    const double dt = d.dt(), ihx = 1.0 / d.h[0];
    const long c0 = d.cell0(0);
    const double sw = A.sw;
    const size_t np = A.np;
    const double *P = A.pa.data(), *Q = A.pb.data();
    const RadialView g = F.grid().view();
    const FV Jz = F.jz.view(), Jr = F.jr.view(), Jp = F.jp.view();
    par_for(long(ni) * nr * M, [=] HD(long q) {
        const int i = i0 + int(q % ni), j = int((q / ni) % nr), m = int(q / (long(ni) * nr));
        const long ig = c0 + i;
        const double wm = 0.5 * (ant_w(xia, sw, ig) + ant_w(xib, sw, ig)) * ihx, dC = (ant_c(xib, sw, ig) - ant_c(xia, sw, ig)) / dt;
        double da[2] = {0, 0}, db[2] = {0, 0};
        ant_div_rz(P, np, nr, g, j, m, da);
        ant_div_rz(Q, np, nr, g, j, m, db);
        for (int c = 0; c < (m ? 2 : 1); ++c) {
            const int sl = m == 0 ? 0 : 2 * m - 1 + c;
            const size_t pq = size_t(sl) * nr + j;
            Jr(i, j, sl) += (Q[pq] - P[pq]) / dt * wm;
            Jp(i, j, sl) += (Q[np + pq] - P[np + pq]) / dt * wm;
            if (m == 0 || j > 0) Jz(i, j, sl) += 0.5 * (da[c] + db[c]) * dC;    // -sigmabar dC / dt, sigma = -div
        }
    });
}

}  // namespace

std::shared_ptr<BoostAntenna> make_boost_antenna(const Simulation& s) {
    const Config& c = s.cfg;
    if (!c.boost.on) return nullptr;
    auto A = std::make_shared<BoostAntenna>();
    for (const LaserConfig& L : c.all_lasers())
        if (L.enabled && L.antenna) A->lasers.push_back(L);
    for (const auto& F : s.file_lasers) A->cfiles.push_back(F->view());     // boosted runs: every LASY laser via the antenna
    for (const auto& F : s.rz_file_lasers) A->rfiles.push_back(F->view());
    if (A->lasers.empty() && A->cfiles.empty() && A->rfiles.empty()) return nullptr;
    const BoostConfig& B = c.boost;
    const Domain& d = s.dom;
    A->g = B.gamma; A->b = B.beta; A->X0 = B.X0; A->T0 = B.T0; A->xa = 0;
    A->rz = c.rz;
    // the spline stretched to the spacing (1 + beta) c dt / h_x at which a forward characteristic meets the nodes; the variance of
    // a characteristic's weights in steps: 1/4 (the spline) + (beta / (2 (1 + beta)))^2 (the two-step average of J_perp)
    A->sw = (1 + B.beta) * d.dt() / d.h[0];
    {
        const double o = 0.5 * B.beta / (1 + B.beta);
        A->c2 = 0.5 * (0.25 + o * o);
    }
    if (c.rz) {
        A->n1 = d.n[1]; A->M = c.modes; A->S = 2 * c.modes - 1;
        A->np = size_t(A->S) * A->n1;
    } else {
        A->flat = d.flat; A->n1 = d.n[1]; A->n2 = d.n[2]; A->o1 = d.off[1]; A->o2 = d.off[2]; A->N1 = d.N[1]; A->N2 = d.N[2];
        A->per1 = d.periodic[1]; A->per2 = d.periodic[2]; A->hy = d.h[1]; A->hz = d.h[2];
        A->nk = d.flat ? 1 : d.n[2] + 1;
        A->np = size_t(A->n1 + 1) * A->nk;
    }
    A->pa.resize(2 * A->np); A->pb.resize(2 * A->np);
    // the antenna must be inside the box when it starts to emit (lab t = 0): x'_a = gamma beta t_ref / (gamma (1 + beta))
    const double t0 = B.gamma * (1 - B.beta) * (-B.X0), x0 = A->xp(t0), edge = (c.N[0] - 6) * c.h[0];
    if (x0 > edge) {
        char m[400];
        std::snprintf(m, sizeof m, "laser.injection = antenna: the lab plane x = 0 is at x' = %.4g (1/k0) when the laser starts (lab "
                                   "t = 0), beyond the box (%.4g, 6 cells before its end): a shorter window.start (lab) or a longer box",
                      x0, edge);
        throw std::runtime_error(m);
    }
    return A;
}

namespace {
class AntennaOp : public Operator {
public:
    const char* name() const override { return "boost_antenna"; }
    Stage stage() const override { return Stage::Push; }      // after the deposit and its guard sums and filter
    void apply(Simulation& s) override {
        BoostAntenna& A = *s.antenna;
        sync(A, s);
        advance(A, s);
        const Domain& d = s.dom;
        const double dt = d.dt(), h = d.h[0];
        const double xia = A.xp(double(s.istep) * dt) / h - 0.5, xib = A.xp(double(s.istep + 1) * dt) / h - 0.5;
        int i0, i1;
        x_range(d, A.sw, xia, xib, i0, i1);
        if (i0 <= i1) {
            if (A.rz) deposit_rz(A, *s.rz, d, xia, xib, i0, i1);
            else deposit_cart(A, *s.fields, d, xia, xib, i0, i1);
        }
        swap_step(A);
    }
};

void add_rho_cart(const BoostAntenna& A, const Domain& d, Field& rho, double xi, int i0, int i1) {
    const int n1p = A.n1 + 1, ny = d.n[1], nz = d.n[2], ni = i1 - i0 + 1;
    const bool flat = A.flat;
    const double ihx = 1.0 / d.h[0], iy = 1.0 / A.hy, iz = 1.0 / A.hz;
    const long c0 = d.cell0(0);
    const double sw = A.sw;
    const size_t np = A.np;
    const double* P = A.pa.data();
    const FV R = rho.view();
    par_for(long(ni) * ny * nz, [=] HD(long q) {
        const int i = i0 + int(q % ni), j = int((q / ni) % ny), k = int(q / (long(ni) * ny));
        const long p = flat ? j + 1 : long(k + 1) * n1p + (j + 1), pj = p - 1, pk = flat ? p : p - n1p;
        R(i, j, k) += -((P[p] - P[pj]) * iy + (P[np + p] - P[np + pk]) * iz) * ant_w(xi, sw, c0 + i) * ihx;
    });
}

void add_rho_rz(const BoostAntenna& A, const RadialView g, const Domain& d, Field& rho, double xi, int i0, int i1) {
    const int nr = A.n1, M = A.M, ni = i1 - i0 + 1;
    const double ihx = 1.0 / d.h[0];
    const long c0 = d.cell0(0);
    const double sw = A.sw;
    const size_t np = A.np;
    const double* P = A.pa.data();
    const FV R = rho.view();
    par_for(long(ni) * nr * M, [=] HD(long q) {
        const int i = i0 + int(q % ni), j = int((q / ni) % nr), m = int(q / (long(ni) * nr));
        if (m && j == 0) return;
        double dv[2] = {0, 0};
        ant_div_rz(P, np, nr, g, j, m, dv);
        const double w = ant_w(xi, sw, c0 + i) * ihx;
        for (int c = 0; c < (m ? 2 : 1); ++c) R(i, j, m == 0 ? 0 : 2 * m - 1 + c) -= dv[c] * w;
    });
}
}  // namespace

std::unique_ptr<Operator> make_boost_antenna_operator() { return std::make_unique<AntennaOp>(); }

void boost_antenna_add_rho(Simulation& s, Field& rho) {
    if (!s.antenna) return;
    BoostAntenna& A = *s.antenna;
    sync(A, s);
    const Domain& d = s.dom;
    const double xi = A.xp(double(s.istep) * d.dt()) / d.h[0] - 0.5;
    int i0, i1;
    x_range(d, A.sw, xi, xi, i0, i1);
    if (i0 > i1) return;
    if (A.rz) add_rho_rz(A, s.rz->grid().view(), d, rho, xi, i0, i1);
    else add_rho_cart(A, d, rho, xi, i0, i1);
}

// the lab position of the antenna at the lab time t (for the summary): x' at the start of the emission
double boost_antenna_start(const Config& c) {
    const BoostConfig& B = c.boost;
    const double t0 = B.gamma * (1 - B.beta) * (-B.X0);
    return (0 - B.X0) / B.gamma - B.beta * t0;
}
