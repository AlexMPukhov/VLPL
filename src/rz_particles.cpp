// Particles in RZ with azimuthal modes (docs/specs/rz_modes.md, decisions 42, 44, 50): gather, push and
// the charge-conserving deposition per mode, shape 1 or 2. Particles are Cartesian (z, x, y), the cylinder
// axis at x = y = 0; z is relative to the global box (the rank's lo is subtracted here).
// Deposition (research/rz_modes/deposit.py, continuity per mode at 1e-16): W = S_z S_r on the rho grid
// (z: i + 1/2, r: r_j), e = e^{-i m phi}; Delta(W e) = (Delta W) e_bar + W_bar Delta e. Delta W by the 2D
// Esirkepov rule in (z, r) with cylindrical volumes (fluxes through z faces and r surfaces, cumulative from
// the lower end of the window), times e_bar; W_bar Delta e gives J_phi at the rho node. Moves of more than
// one cell in z or r (iRIP) are split into straight sub-segments, each conserving charge.
// Shape 2 reaches below the axis: the stencil is laid out on the line r in (-inf, inf) and folded onto
// r >= 0 at charge level (decision 50): a node -j onto +j with (-1)^m (rho, J_z, J_phi), a radial flux at
// -(j + 1/2) onto +(j + 1/2) with -(-1)^m, then divided by the target's volume or area. The folded fluxes
// balance the folded charges node by node, so the continuity per mode stays exact (the m = 0 axis node too).
// The deposition uses the density-conserving maps of src/rz_ruyten.cpp (decision 53: rz_xi(.., shape), the
// splines in the mapped xi; the gather stays in the plain xi, as FBPIC), so that a uniform density gives
// rho = q n on every node including the axis (the plain splines: 4/3, 13/8 of it on the axis node).
// numerics.rz_gather = matched (the default, decision 80): the gather uses the same maps (the shapes of the deposit), so that a particle does not
// see a different shape of its own charge than it deposits (noise heating study, 2026-11-06).
// On a stretched radial grid (decision 52) the shapes, the fold and the fluxes live in the index coordinate
// xi (rz_xi): node j at xi = j; the charge of a node is divided by its volume 2 pi A_j dz, a radial flux by
// its face 2 pi rh_j dz. The continuity is a statement about charges and fluxes in xi: exact on any grid.
#include <cmath>
#include "external.hpp"
#include "pushers.hpp"
#include "rz.hpp"
#include "species.hpp"

namespace {
struct Cx { double re, im; };
HD inline int slot(int m, int c) { return m == 0 ? 0 : 2 * m - 1 + c; }

struct RZK {                       // kernel constants
    double dz, lo, dt;
    int nr, M, ng;
    RadialView g;
    int gs = 0;                    // the radial map of the gather: 0 plain xi, O the deposit's (numerics.rz_gather = matched)
    bool ge = false;               // rz_gather = energy: E_r with the derivative of the deposit's radial shape (below)
    int xm = 0;                    // the radial map of the deposit and rho of m = 0 (rz_xi shape): O, or 4 (the C1 map, f.c1_map)
    int x1 = 0;                    // the same for the modes m >= 1: 0 the plain xi (decision 106), or xm (numerics.rz_axis_map = all, the C1 map)
    int gs1 = 0;                   // the radial map of the gather of the modes m >= 1: 0, or gs (numerics.rz_axis_map = all)
    bool chord = false;            // numerics.rz_deposit = chord: the explicit deposit with the splits of seg_currents (below)
};
HD inline double vol(const RZK& k, int j) { return 2 * M_PI * k.g.A[j] * k.dz; }

// Shape-O weights of a position s in node units (node n at s = n): nodes i0 .. i0 + O.
template <int O> HD inline void weights(double s, int& i0, double w[O + 1]) {
    if (O == 1) {
        i0 = int(floor(s));
        const double f = s - i0;
        w[0] = 1 - f; w[1] = f;
    } else if (O == 0) {                                     // nearest node
        i0 = int(floor(s + 0.5));
        w[0] = 1;
    } else {
        const int c = int(floor(s + 0.5));
        const double f = s - c;
        i0 = c - 1;
        w[0] = 0.5 * (0.5 - f) * (0.5 - f); w[1] = 0.75 - f * f; w[O] = 0.5 * (0.5 + f) * (0.5 + f);
    }
}

// Mode value of F at column i, radial index j (ro = 0: node r_j, ro = 1/2: half node r_{j+1/2}), mirrored
// below the axis with parity par: node -j -> j, half node -1 - j -> j.
HD inline Cx mode_at(const FV& F, int i, int j, int m, double par, double ro) {
    double p = 1;
    if (j < 0) { j = ro > 0 ? -1 - j : -j; p = par; }
    return {p * F(i, j, slot(m, 0)), m ? p * F(i, j, slot(m, 1)) : 0.0};
}

// Interpolated mode values of one component: shape O in z (integer positions) and r (offset ro = 0 or
// 1/2); kind: 0 scalar (Ez, Bz), 1 r or phi component (parity (-1)^(m+1) below the axis). axis_m1: for the
// node quantities Ephi, Br the m = 1 axis value from the partner half-node component G (Er, Bphi): a
// uniform transverse vector has V_phi1 = i V_r1, so Ephi_1(0) = i Er_1(dr/2), Br_1(0) = -i Bphi_1(dr/2)
// (sign s).
// rz_gather = energy (2026-11-06, decision 80): the radial weights of E_r at the half nodes h that make the work of the
// gathered E_r equal to that of the deposited J_r (J_r,h = q dC_h / (2 pi rh_h dz dt), C_h the charge above the face h in the
// deposit's map xi, the field energy weight of E_r B_h = rh_h dn_h): g_h = dn_h xi'(r) S_{O-1}(xi(r) - h - 1/2). With the
// plain gather a particle next to the axis feels a different shape of its own charge than it deposits: a dense plasma
// there (k_p dr > ~1.5) is unstable (bench/noise_heating, "The axis instability").
HD inline double dxi_dr(const RadialView& g, double r, int O) {
    const double h = 1e-6 * g.dr, a = r > h ? r - h : 0.0, b = r + h;
    return (rz_xi(g, b, O) - rz_xi(g, a, O)) / (b - a);
}
template <int O>
HD inline Cx interp(const FV& F, const RZK& k, double zl, double r, int m, double ro, int kind, const FV* G, double s, bool er = false) {
    int iz, jr;
    double wz[O + 1], wr[O + 1];
    weights<O>(zl / k.dz, iz, wz);
    if (er && k.ge) {                                            // E_r, energy-consistent: O - 1 in xi at the half nodes
        const int mp = m == 0 || k.gs1 ? O : 0;                    // the deposit's map of this mode (m >= 1: plain, decision 106)
        const double x = rz_xi(k.g, r, mp), dx = dxi_dr(k.g, r, mp);
        double w0[O];
        weights<O - 1>(x - 0.5, jr, w0);
        for (int b = 0; b <= O; ++b) {
            const int h = jr + b;
            wr[b] = b < O ? w0[b] * dx * k.g.dn[h >= 0 ? h : -1 - h] : 0.0;
        }
    } else weights<O>(rz_xi(k.g, r, m == 0 ? k.gs : k.gs1) - ro, jr, wr);
    const double par = (m % 2 == 0 ? 1.0 : -1.0) * (kind == 1 ? -1.0 : 1.0);
    Cx v{0, 0};
    for (int a = 0; a <= O; ++a)
        for (int b = 0; b <= O; ++b) {
            const double w = wz[a] * wr[b];
            Cx c;
            if (G && m == 1 && jr + b == 0) {                         // axis value of Ephi_1 or Br_1
                const Cx g = mode_at(*G, iz + a, 0, 1, 1.0, 0.5);
                c = {-s * g.im, s * g.re};                            // i s g
            } else c = mode_at(F, iz + a, jr + b, m, par, ro);
            v.re += w * c.re; v.im += w * c.im;
        }
    return v;
}

// F = F_0 + 2 Re sum_m F_m e^{i m phi}
template <class Fn> HD inline double reconstruct(int M, double phi, Fn f) {
    double v = 0;
    for (int m = 0; m < M; ++m) {
        const Cx c = f(m);
        v += m == 0 ? c.re : 2 * (c.re * cos(m * phi) - c.im * sin(m * phi));
    }
    return v;
}

struct RZViews { FV ez, er, ep, bz, br, bp, jz, jr, jp; };

// E, B (Cartesian, particle axes (z, x, y)) at (zl, x, y)
template <int O> HD inline void gather_rz(const RZViews& F, const RZK& k, double zl, double x, double y, double E[3], double B[3]) {
    const double r = sqrt(x * x + y * y), phi = r > 0 ? atan2(y, x) : 0.0;
    const double ez = reconstruct(k.M, phi, [&](int m) { return interp<O>(F.ez, k, zl, r, m, 0.0, 0, nullptr, 0); });
    const double er = reconstruct(k.M, phi, [&](int m) { return interp<O>(F.er, k, zl, r, m, 0.5, 1, nullptr, 0, true); });
    const double ep = reconstruct(k.M, phi, [&](int m) { return interp<O>(F.ep, k, zl, r, m, 0.0, 1, &F.er, 1.0); });
    const double bz = reconstruct(k.M, phi, [&](int m) { return interp<O>(F.bz, k, zl, r, m, 0.5, 0, nullptr, 0); });
    const double br = reconstruct(k.M, phi, [&](int m) { return interp<O>(F.br, k, zl, r, m, 0.0, 1, &F.bp, -1.0); });
    const double bp = reconstruct(k.M, phi, [&](int m) { return interp<O>(F.bp, k, zl, r, m, 0.5, 1, nullptr, 0); });
    const double c = cos(phi), s = sin(phi);
    E[0] = ez; E[1] = er * c - ep * s; E[2] = er * s + ep * c;
    B[0] = bz; B[1] = br * c - bp * s; B[2] = br * s + bp * c;
}

// Folding onto r >= 0 (header): node j and half node j (r_{j+1/2}); the factor of mode m.
HD inline int fold_node(int j, int m, double& f) { f = j < 0 && (m & 1) ? -1.0 : 1.0; return j < 0 ? -j : j; }
HD inline int fold_half(int j, int m, double& f) { f = j < 0 ? ((m & 1) ? 1.0 : -1.0) : 1.0; return j < 0 ? -1 - j : j; }

// Deposit of one straight sub-segment (moves < 1 cell in z and r), charge q w. Window of N = O + 2 nodes
// per axis from the lower ends ib, jb of both supports.
template <int O> HD inline void deposit_seg(const RZViews& F, const RZK& k, const double p0[3], const double p1[3], double qw) {
    constexpr int N = O + 2;
    const double r0 = sqrt(p0[1] * p0[1] + p0[2] * p0[2]), r1 = sqrt(p1[1] * p1[1] + p1[2] * p1[2]);
    const double f0 = r0 > 0 ? atan2(p0[2], p0[1]) : 0.0, f1 = r1 > 0 ? atan2(p1[2], p1[1]) : 0.0;
    int iz0, iz1, j0, j1;
    double a0[O + 1], a1[O + 1], b0[O + 1], b1[O + 1];
    weights<O>(p0[0] / k.dz - 0.5, iz0, a0); weights<O>(p1[0] / k.dz - 0.5, iz1, a1);
    const int ib = iz0 < iz1 ? iz0 : iz1;
    // radial weights per map: set 0 the density-conserving map of m = 0, set 1 the map of the modes m >= 1 (the plain xi,
    // decision 106: the density-conserving map is a statement about m = 0; in the modes m >= 1 it distorts the dipole of a
    // charge next to the axis and a uniform transverse current). Each mode conserves charge with its own weights.
    const int ns = k.M > 1 && k.x1 != k.xm ? 2 : 1;
    int jbs[2];
    bool out[2] = {false, false};                                // beyond the wall's guards (absorbed there), per map: each set of
                                                                 // modes stops where its own rho stops (rho_impl)
    double Sr0[2][N], Sr1[2][N];
    for (int s = 0; s < ns; ++s) {
        const int mp = s == 0 ? k.xm : k.x1;
        weights<O>(rz_xi(k.g, r0, mp), j0, b0); weights<O>(rz_xi(k.g, r1, mp), j1, b1);
        jbs[s] = j0 < j1 ? j0 : j1;
        out[s] = jbs[s] + N > k.nr + k.ng;
        for (int q = 0; q < N; ++q) Sr0[s][q] = Sr1[s][q] = 0;
        for (int q = 0; q <= O; ++q) { Sr0[s][j0 - jbs[s] + q] = b0[q]; Sr1[s][j1 - jbs[s] + q] = b1[q]; }
    }
    if (out[0] && (ns == 1 || out[1])) return;
    double S0z[N], S1z[N];
    for (int q = 0; q < N; ++q) S0z[q] = S1z[q] = 0;
    for (int q = 0; q <= O; ++q) { S0z[iz0 - ib + q] = a0[q]; S1z[iz1 - ib + q] = a1[q]; }
    double dphi = f1 - f0;                                        // for J_phi of m = 0
    if (dphi > M_PI) dphi -= 2 * M_PI;
    if (dphi < -M_PI) dphi += 2 * M_PI;
    for (int m = 0; m < k.M; ++m) {
        const int s = m > 0 && ns > 1 ? 1 : 0, jb = jbs[s];
        if (out[s]) continue;
        const double *S0r = Sr0[s], *S1r = Sr1[s];
        const Cx e0{cos(m * f0), -sin(m * f0)}, e1{cos(m * f1), -sin(m * f1)};
        const Cx eb{0.5 * (e0.re + e1.re), 0.5 * (e0.im + e1.im)}, de{e1.re - e0.re, e1.im - e0.im};
        for (int b = 0; b < N; ++b) {                              // J_z: fluxes through the z faces
            double fs;
            const int j = fold_node(jb + b, m, fs);
            const double A = vol(k, j) / k.dz;
            double fz = 0;
            for (int a = 0; a < N; ++a) {
                fz -= 0.5 * (S1z[a] - S0z[a]) * (S0r[b] + S1r[b]);
                if (fz == 0.0) continue;
                const double c = fs * qw * fz / (A * k.dt);
                atomic_add(&F.jz(ib + a + 1, j, slot(m, 0)), c * eb.re);
                if (m) atomic_add(&F.jz(ib + a + 1, j, slot(m, 1)), c * eb.im);
            }
        }
        for (int a = 0; a < N; ++a) {                              // J_r: fluxes through the r surfaces
            double fr = 0;
            for (int b = 0; b < N; ++b) {
                fr -= 0.5 * (S0z[a] + S1z[a]) * (S1r[b] - S0r[b]);
                if (fr == 0.0) continue;
                double fs;
                const int j = fold_half(jb + b, m, fs);
                const double c = fs * qw * fr / (2 * M_PI * k.g.rh[j] * k.dz * k.dt);
                atomic_add(&F.jr(ib + a, j, slot(m, 0)), c * eb.re);
                if (m) atomic_add(&F.jr(ib + a, j, slot(m, 1)), c * eb.im);
            }
        }
        for (int a = 0; a < N; ++a)                                // J_phi from the phase change
            for (int b = 0; b < N; ++b) {
                double fs;
                const int j = fold_node(jb + b, m, fs);
                const double wbar = 0.5 * (S0z[a] * S0r[b] + S1z[a] * S1r[b]);
                if (wbar == 0.0 || j == 0) continue;
                const double c = fs * qw * wbar * k.g.r[j] / (k.dt * vol(k, j));
                if (m == 0) atomic_add(&F.jp(ib + a, j, 0), c * dphi);
                else {                                                // i r W_bar (e1 - e0) / m
                    atomic_add(&F.jp(ib + a, j, slot(m, 0)), -c * de.im / k.g.mf(m));
                    atomic_add(&F.jp(ib + a, j, slot(m, 1)), c * de.re / k.g.mf(m));
                }
            }
    }
}

// numerics.rz_deposit = chord (2026-11-13): deposit_seg with the splits of seg_currents (defined with the implicit mover below):
// m = 0 J_phi from the angular momentum L = x0 y1 - y0 x1 of the move instead of r_j dphi, m >= 1 J_r with the chord phase e_c
// and J_phi from the bracket. Charge-conserving per mode as deposit_seg; the two agree far from the axis. deposit_seg gives a
// segment that passes the axis at the impact parameter b << dr an m = 0 J_phi of ~ pi r_j^2 / dt (its dphi is ~ pi) instead of
// ~ L / dt (found as axis noise of the wake electrons in a boosted RZ LWFA run, 2026-11-13).
template <int O> HD inline void deposit_seg_chord(const RZViews& F, const RZK& k, const double p0[3], const double p1[3], double qw);
template <int O> HD inline void deposit_any(const RZViews& F, const RZK& k, const double p0[3], const double p1[3], double qw) {
    if (k.chord) deposit_seg_chord<O>(F, k, p0, p1, qw);
    else deposit_seg<O>(F, k, p0, p1, qw);
}

template <int O> void push_deposit_impl(Species& sp, FieldsRZ& f, Pusher pusher, bool deposit) {
    const Domain& d = f.dom();
    if (f.axis_regular && f.smooth_a <= 0) rz_axis_gather_copies(f);   // decision 112: the gather's E (fresh for this push)
    const RZK k{d.h[0], d.lo(0), d.dt(), d.n[1], f.modes(), d.ng, f.grid().view(), f.gather_matched ? O : 0, f.gather_energy, f.c1_map ? 4 : O,
                f.c1_map ? 4 : (f.map_all ? O : 0), (f.map_all || f.c1_map) && f.gather_matched ? O : 0,
                f.deposit_chord};
    const bool s = f.smooth_a > 0 || f.axis_regular;   // the gather copies of rz_smooth_e: the transverse shape (decision 79), the
                                                       // adjoint first-cell projection (decision 112); B as it is
    const RZViews F{(s ? f.sez_ : f.ez).view(), (s ? f.ser_ : f.er).view(), (s ? f.sep_ : f.ep).view(), f.bz.view(), f.br.view(),
                    f.bp.view(), f.jz.view(), f.jr.view(), f.jp.view()};
    double *X = sp.ptr<double>(Species::X), *Y = sp.ptr<double>(Species::Y), *Z = sp.ptr<double>(Species::Z);
    double *UX = sp.ptr<double>(Species::UX), *UY = sp.ptr<double>(Species::UY), *UZ = sp.ptr<double>(Species::UZ);
    const double* W = sp.ptr<double>(Species::W);
    const int* ZS = sp.zslot >= 0 ? sp.ptr<int>(sp.zslot) : nullptr;   // charge state per particle (ionization.hpp)
    const double qmdt0 = sp.q * k.dt / sp.m, q0 = sp.q, xf = d.xfreeze;
    const QuadView qv = quadrupoles();              // external quadrupoles (external.hpp): lab z = X, axes (z, x, y)
    par_for(long(sp.size()), [=] HD(long p) {
        if (X[p] >= xf) return;                                   // entry columns of the moving window
        const double qmdt = ZS ? qmdt0 * ZS[p] : qmdt0, q = ZS ? q0 * ZS[p] : q0;
        // particle axes (z, x, y): X holds z, Y holds x, Z holds y
        const double p0[3] = {X[p] - k.lo, Y[p], Z[p]};
        double E[3], B[3], u[3] = {UX[p], UY[p], UZ[p]};
        if (pusher != Pusher::None) {                             // None: pushed before by rz_push_store (decision 15)
            gather_rz<O>(F, k, p0[0], p0[1], p0[2], E, B);
            if (qv.n) qv.add(X[p], p0[1], p0[2], u[0] * k.dt / sqrt(1.0 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), B);
            push_momentum_inline(pusher, qmdt, E, B, u);
        }
        const double ig = k.dt / sqrt(1.0 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        double p1[3] = {p0[0] + u[0] * ig, p0[1] + u[1] * ig, p0[2] + u[2] * ig};
        UX[p] = u[0]; UY[p] = u[1]; UZ[p] = u[2];
        X[p] = p1[0] + k.lo; Y[p] = p1[1]; Z[p] = p1[2];
        p1[0] = X[p] - k.lo;                                      // the stored end point (as move_one, decision 58)
        if (!deposit) return;
        // sub-segments of at most one cell in z and in r: the chord bounds the change of r, and the cells
        // grow outward, so the cell at the chord's closest approach to the axis bounds the change of xi
        const double cx = p1[1] - p0[1], cy = p1[2] - p0[2], cl2 = cx * cx + cy * cy;
        double ta = cl2 > 0 ? -(p0[1] * cx + p0[2] * cy) / cl2 : 0.0;
        ta = ta < 0 ? 0.0 : (ta > 1 ? 1.0 : ta);
        const double ra = sqrt((p0[1] + ta * cx) * (p0[1] + ta * cx) + (p0[2] + ta * cy) * (p0[2] + ta * cy));
        // (the density-conserving map stretches xi by at most 1 + |bt| there; in the axis cell of shape 2 a move
        // of less than one cell changes xi by less than one, as its map lies above the one of cell 1)
        int ja = int(rz_xi(k.g, ra));
        ja = ja < k.nr + 4 ? ja : k.nr + 4;
        const double ha = ra < k.g.ju * k.g.dr ? k.g.dr : k.g.dn[ja];
        const double dzc = fabs(p1[0] - p0[0]) / k.dz, dtr = sqrt(cl2) / ha * (1 + fabs(O == 1 ? k.g.bt[ja] : (ja > 0 ? k.g.bt2[ja] : 0.0)));
        const double mc = dzc > dtr ? dzc : dtr;
        const int n = mc <= 1.0 ? 1 : int(ceil(mc * (1.0 + 1e-12)));
        double a0[3] = {p0[0], p0[1], p0[2]};
        for (int s = 1; s <= n; ++s) {
            double a1[3];
            for (int c = 0; c < 3; ++c) a1[c] = s == n ? p1[c] : p0[c] + (p1[c] - p0[c]) * (double(s) / n);
            deposit_any<O>(F, k, a0, a1, q * W[p]);
            for (int c = 0; c < 3; ++c) a0[c] = a1[c];
        }
    });
}

template <int O> void rho_impl(Species& sp, FieldsRZ& f, Field& rho, double x0 = 0, double x1 = 0, bool range = false) {
    const Domain& d = f.dom();
    const RZK k{d.h[0], d.lo(0), d.dt(), d.n[1], f.modes(), d.ng, f.grid().view(), f.gather_matched ? O : 0, f.gather_energy, f.c1_map ? 4 : O,
                f.c1_map ? 4 : (f.map_all ? O : 0), (f.map_all || f.c1_map) && f.gather_matched ? O : 0};
    const FV R = rho.view();
    const double *X = sp.ptr<double>(Species::X), *Y = sp.ptr<double>(Species::Y), *Z = sp.ptr<double>(Species::Z);
    const double* W = sp.ptr<double>(Species::W);
    const int* ZS = sp.zslot >= 0 ? sp.ptr<int>(sp.zslot) : nullptr;
    const double q0 = sp.q;
    par_for(long(sp.size()), [=] HD(long p) {
        if (range && (X[p] < x0 || X[p] >= x1)) return;
        const double q = ZS ? q0 * ZS[p] : q0;
        const double zl = X[p] - k.lo, r = sqrt(Y[p] * Y[p] + Z[p] * Z[p]), phi = r > 0 ? atan2(Z[p], Y[p]) : 0.0;
        int i, j;
        double wz[O + 1], wr[O + 1];
        int j1 = 0;
        double wr1[O + 1];                                        // the modes m >= 1 in their own map (deposit_seg)
        const bool two = k.M > 1 && k.x1 != k.xm;
        weights<O>(zl / k.dz - 0.5, i, wz); weights<O>(rz_xi(k.g, r, k.xm), j, wr);
        if (two) weights<O>(rz_xi(k.g, r, k.x1), j1, wr1);
        const bool o0 = j + O + 1 > k.nr + k.ng, o1 = two ? j1 + O + 1 > k.nr + k.ng : o0;   // beyond the wall's guards, per map
        if (o0 && o1) return;
        for (int a = 0; a <= O; ++a)
            for (int b = 0; b <= O; ++b)
                for (int m = 0; m < k.M; ++m) {
                    double fs;
                    const bool s1 = two && m > 0;
                    if (s1 ? o1 : o0) continue;
                    const int jt = fold_node((s1 ? j1 : j) + b, m, fs);
                    if (m && jt == 0) continue;                  // rho_m ~ r^m: no m >= 1 value on the axis node
                    const double c = fs * q * W[p] * wz[a] * (s1 ? wr1[b] : wr[b]) / vol(k, jt);
                    atomic_add(&R(i + a, jt, slot(m, 0)), c * cos(m * phi));
                    if (m) atomic_add(&R(i + a, jt, slot(m, 1)), -c * sin(m * phi));
                }
    });
}

template <int O> void gather_point_impl(FieldsRZ& f, double z, double x, double y, double E[3], double B[3]) {
    const Domain& d = f.dom();
    const RZK k{d.h[0], d.lo(0), d.dt(), d.n[1], f.modes(), d.ng, f.grid().view(), f.gather_matched ? O : 0, f.gather_energy, f.c1_map ? 4 : O,
                f.c1_map ? 4 : (f.map_all ? O : 0), (f.map_all || f.c1_map) && f.gather_matched ? O : 0};
    const RZViews F{f.ez.view(), f.er.view(), f.ep.view(), f.bz.view(), f.br.view(), f.bp.view(), f.jz.view(), f.jr.view(), f.jp.view()};
    Buf<double> out;
    out.resize(6);
    double* o = out.data();
    const double zl = z - k.lo;
    par_for(1, [=] HD(long) {
        double e[3], b[3];
        gather_rz<O>(F, k, zl, x, y, e, b);
        for (int c = 0; c < 3; ++c) { o[c] = e[c]; o[3 + c] = b[c]; }
    });
    const std::vector<double> v = out.host();
    for (int c = 0; c < 3; ++c) { E[c] = v[size_t(c)]; B[c] = v[size_t(3 + c)]; }
}

// ---------------------------------------------------------------- O6: the implicit mover in RZ (decision 82)
struct RZET { FV z, r, p; };                                     // the work field at the J_z, J_r, J_phi points
HD inline double edot(Cx e, const FV& F, int i, int j, int m) {
    return e.re * F(i, j, slot(m, 0)) + (m ? e.im * F(i, j, slot(m, 1)) : 0.0);
}

// The current of one straight segment in the implicit mover (decision 82), per mode, as contributions emit(c, i, j, m, v): component
// c (0 J_z at (i, r_j), 1 J_r at (i+1/2, r_{j+1/2}), 2 J_phi at (i+1/2, r_j)), folded index j, complex v (charge units, fs included;
// the current is q w v / (dt x the face or volume): 2 pi A_j, 2 pi rh_j dz, 2 pi A_j dz). Charge-conserving per mode as
// deposit_seg, with two splits that are smooth in the end points (the nonlinear solve needs that):
//   m >= 1: rho_k = S_z,k (x) P_k with the folded radial profile P_k = S_r,k e_k (e_k = e^{-i m phi_k}; smooth for odd m: the
//     folded weights vanish on the axis), Delta(S_z P) = dS_z (x) Pbar + Sbar_z (x) dP, and dP = (S_r1 - S_r0) e_c +
//     [S_r1 (e1 - e_c) - S_r0 (e0 - e_c)]: J_z from dS_z Pbar, J_r from the first part (a radial flux, it closes), J_phi from the
//     bracket (i r_j / m). e_c = ((z0 + z1) / (r0 + r1))^m, z = x - i y: smooth (Lipschitz) in the end points. deposit_seg's
//     (Delta W) e_bar + W_bar Delta e has the cross terms S_r1 e0, S_r0 e1, which jump when an end passes next to the axis
//     (its phase turns by pi over a short distance). Far from the axis the two agree to second order in the step.
//   m = 0: J_phi at node j from wbar L / r_j, L = x0 y1 - y0 x1 (the angular momentum of the move times dt: v_phi = L / r at
//     the node radius) instead of r_j dphi: m = 0 has no continuity constraint on J_phi, and r_j dphi jumps by 2 pi r_j wbar when a
//     segment passes over the axis; L is a polynomial in the positions.
template <int O, class E> HD inline bool seg_currents(const RZK& k, const double p0[3], const double p1[3], E&& emit) {
    constexpr int N = O + 2;
    const double r0 = sqrt(p0[1] * p0[1] + p0[2] * p0[2]), r1 = sqrt(p1[1] * p1[1] + p1[2] * p1[2]);
    int iz0, iz1, j0, j1;
    double a0[O + 1], a1[O + 1], b0[O + 1], b1[O + 1];
    weights<O>(p0[0] / k.dz - 0.5, iz0, a0); weights<O>(p1[0] / k.dz - 0.5, iz1, a1);
    const int ib = iz0 < iz1 ? iz0 : iz1;
    // radial weights per map as in deposit_seg: set 0 the map of m = 0 (k.xm), set 1 the map of the modes m >= 1 (k.x1, the plain
    // xi by default, decision 106); each mode then conserves charge with the weights of rho_impl
    const int ns = k.M > 1 && k.x1 != k.xm ? 2 : 1;
    int jbs[2] = {0, 0};
    bool out[2] = {false, false};
    double S0z[N], S1z[N], Sr0[2][N], Sr1[2][N];
    for (int s = 0; s < ns; ++s) {
        const int mp = s == 0 ? k.xm : k.x1;
        weights<O>(rz_xi(k.g, r0, mp), j0, b0); weights<O>(rz_xi(k.g, r1, mp), j1, b1);
        jbs[s] = j0 < j1 ? j0 : j1;
        out[s] = jbs[s] + N > k.nr + k.ng;                        // beyond the guards of the wall (absorbed there)
        for (int q = 0; q < N; ++q) Sr0[s][q] = Sr1[s][q] = 0;
        for (int q = 0; q <= O; ++q) { Sr0[s][j0 - jbs[s] + q] = b0[q]; Sr1[s][j1 - jbs[s] + q] = b1[q]; }
    }
    if (out[0] && (ns == 1 || out[1])) return false;
    for (int q = 0; q < N; ++q) S0z[q] = S1z[q] = 0;
    for (int q = 0; q <= O; ++q) { S0z[iz0 - ib + q] = a0[q]; S1z[iz1 - ib + q] = a1[q]; }
    const double rs = r0 + r1, L = p0[1] * p1[2] - p0[2] * p1[1];
    const Cx w0 = r0 > 0 ? Cx{p0[1] / r0, -p0[2] / r0} : Cx{1, 0}, w1 = r1 > 0 ? Cx{p1[1] / r1, -p1[2] / r1} : Cx{1, 0};
    const Cx wc = rs > 0 ? Cx{(p0[1] + p1[1]) / rs, -(p0[2] + p1[2]) / rs} : Cx{1, 0};
    Cx e0{1, 0}, e1{1, 0}, ec{1, 0};
    for (int m = 0; m < k.M; ++m) {
        const int st = m > 0 && ns == 2 ? 1 : 0;
        if (out[st]) continue;                                    // this set beyond the guards (absorbed there)
        const int jb = jbs[st];
        const double *S0r = Sr0[st], *S1r = Sr1[st];
        if (m > 0) {                                              // e^{-i m phi} by powers
            e0 = {e0.re * w0.re - e0.im * w0.im, e0.re * w0.im + e0.im * w0.re};
            e1 = {e1.re * w1.re - e1.im * w1.im, e1.re * w1.im + e1.im * w1.re};
            ec = {ec.re * wc.re - ec.im * wc.im, ec.re * wc.im + ec.im * wc.re};
        }
        for (int b = 0; b < N; ++b) {                              // J_z: -cum_a dS_z times Pbar
            double fs;
            const int j = fold_node(jb + b, m, fs);
            const Cx P{0.5 * (S0r[b] * e0.re + S1r[b] * e1.re), 0.5 * (S0r[b] * e0.im + S1r[b] * e1.im)};
            double fz = 0;
            for (int a = 0; a < N; ++a) {
                fz -= S1z[a] - S0z[a];
                if (fz == 0.0) continue;
                emit(0, ib + a + 1, j, m, Cx{fs * fz * P.re, fs * fz * P.im});
            }
        }
        for (int a = 0; a < N; ++a) {                              // J_r: -cum_b (S_r1 - S_r0) Sbar_z, phase e_c
            double fr = 0;
            for (int b = 0; b < N; ++b) {
                fr -= 0.5 * (S0z[a] + S1z[a]) * (S1r[b] - S0r[b]);
                if (fr == 0.0) continue;
                double fs;
                const int j = fold_half(jb + b, m, fs);
                emit(1, ib + a, j, m, Cx{fs * fr * ec.re, fs * fr * ec.im});
            }
        }
        for (int a = 0; a < N; ++a)                                // J_phi
            for (int b = 0; b < N; ++b) {
                double fs;
                const int j = fold_node(jb + b, m, fs);
                if (j == 0) continue;                                 // no J_phi on the axis node
                if (m == 0) {
                    const double wbar = 0.5 * (S0z[a] * S0r[b] + S1z[a] * S1r[b]);
                    if (wbar != 0.0) emit(2, ib + a, j, 0, Cx{fs * wbar * L / k.g.r[j], 0.0});
                } else {
                    const double sz = 0.5 * (S0z[a] + S1z[a]);
                    const Cx dq{sz * (S1r[b] * (e1.re - ec.re) - S0r[b] * (e0.re - ec.re)), sz * (S1r[b] * (e1.im - ec.im) - S0r[b] * (e0.im - ec.im))};
                    if (dq.re == 0.0 && dq.im == 0.0) continue;
                    const double c = fs * k.g.r[j] / k.g.mf(m);         // i r_j dq / m_eff
                    emit(2, ib + a, j, m, Cx{-c * dq.im, c * dq.re});
                }
            }
    }
    return true;
}

// numerics.rz_deposit = chord: the currents of seg_currents into the current fields (scales as deposit_seg: q w / (dt x the face
// 2 pi A_j or 2 pi rh_j dz, or the volume 2 pi A_j dz))
template <int O> HD inline void deposit_seg_chord(const RZViews& F, const RZK& k, const double p0[3], const double p1[3], double qw) {
    seg_currents<O>(k, p0, p1, [&](int c, int i, int j, int m, Cx v) {
        const FV& G = c == 0 ? F.jz : (c == 1 ? F.jr : F.jp);
        const double s = qw / (k.dt * (c == 0 ? vol(k, j) / k.dz : (c == 1 ? 2 * M_PI * k.g.rh[j] * k.dz : vol(k, j))));
        atomic_add(&G(i, j, slot(m, 0)), s * v.re);
        if (m) atomic_add(&G(i, j, slot(m, 1)), s * v.im);
    });
}

// The work per unit charge (q w = 1) of Et on the current of seg_currents: dt dV w_m J . Et (w_0 = 1, w_m = 2), the volume or face
// of the current cancels: J_z dz, J_r dn_j, J_phi 1, times the real dot product of the (Re, Im) slots.
template <int O> HD inline double work_seg(const RZET& T, const RZK& k, const double p0[3], const double p1[3]) {
    double W = 0;
    seg_currents<O>(k, p0, p1, [&](int c, int i, int j, int m, Cx v) {
        const FV& F = c == 0 ? T.z : (c == 1 ? T.r : T.p);
        const double g = c == 0 ? k.dz : (c == 1 ? k.g.dn[j] : 1.0);
        W += (m ? 2.0 : 1.0) * g * edot(v, F, i, j, m);
    });
    return W;
}

// The deposit of seg_currents, charge q w
template <int O> HD inline void deposit_imp(const RZViews& F, const RZK& k, const double p0[3], const double p1[3], double qw) {
    seg_currents<O>(k, p0, p1, [&](int c, int i, int j, int m, Cx v) {
        const FV& J = c == 0 ? F.jz : (c == 1 ? F.jr : F.jp);
        const double s = qw / (k.dt * (c == 0 ? vol(k, j) / k.dz : (c == 1 ? 2 * M_PI * k.g.rh[j] * k.dz : vol(k, j))));
        atomic_add(&J(i, j, slot(m, 0)), s * v.re);
        if (m) atomic_add(&J(i, j, slot(m, 1)), s * v.im);
    });
}

// The limit of work_seg / |dp| for a vanishing step dp at p, per direction (Cartesian, unit charge): the force whose work is
// work_seg to first order. z: the linear spline in z at the E_z nodes times S_O in xi (the cumulative derivative of the
// Esirkepov flux); r: S_O in z at the charge nodes times S_{O-1} in xi at the half nodes, xi'(r) dn_h; phi: S_O S_O at the
// J_phi points, m = 0 (r / r_j) Et_phi (dL = r ds), m >= 1 (r_j / r) Re(Et_m e^{i m phi}) (the folded weights of odd m vanish on the axis).
template <int O> HD inline void fref(const RZET& T, const RZK& k, const double p[3], double f[3]) {
    const double r = sqrt(p[1] * p[1] + p[2] * p[2]), phi = r > 0 ? atan2(p[2], p[1]) : 0.0;
    int iz, ic, jn, jh;
    double wz1[2], wzc[O + 1], wrn[O + 1], wrh[O];
    weights<1>(p[0] / k.dz, iz, wz1);
    weights<O>(p[0] / k.dz - 0.5, ic, wzc);
    const double x = rz_xi(k.g, r, k.xm), dx = k.xm == 4 ? rz_dxi_c1(k.g, r) : dxi_dr(k.g, r, k.xm);
    weights<O>(x, jn, wrn);
    weights<O - 1>(x - 0.5, jh, wrh);
    f[0] = f[1] = f[2] = 0;
    if (jn + O + 2 > k.nr + k.ng) return;                        // at the wall (the deposit is cut there)
    double Lz = 0, Lr = 0, Lp = 0;
    for (int m = 0; m < k.M; ++m) {
        const double wm = m ? 2.0 : 1.0, cm = cos(m * phi), sm = sin(m * phi);
        const Cx e{cm, -sm};
        for (int b = 0; b <= O; ++b) {
            double fs;
            const int j = fold_node(jn + b, m, fs);
            for (int a = 0; a < 2; ++a) Lz += wz1[a] * wrn[b] * fs * wm * edot(e, T.z, iz + a, j, m);
            if (j == 0) continue;
            for (int a = 0; a <= O; ++a) {
                const double w = wzc[a] * wrn[b] * fs;
                if (m == 0) Lp += w * r / k.g.r[j] * T.p(ic + a, j, 0);
                else if (r > 0) Lp += w * k.g.r[j] / r * wm * (cm * T.p(ic + a, j, slot(m, 0)) - sm * T.p(ic + a, j, slot(m, 1)));
            }
        }
        for (int b = 0; b < O; ++b) {
            double fs;
            const int h = fold_half(jh + b, m, fs);
            for (int a = 0; a <= O; ++a) Lr += wzc[a] * wrh[b] * dx * k.g.dn[h] * fs * wm * edot(e, T.r, ic + a, h, m);
        }
    }
    const double c = cos(phi), s = sin(phi);
    f[0] = Lz; f[1] = Lr * c - Lp * s; f[2] = Lr * s + Lp * c;
}

HD inline void cn_end(const double r0[3], const double u0[3], const double u1[3], double dt, double lo, double vb[3], double r1[3],
                      double& zg) {
    const double g = sqrt(1.0 + u0[0] * u0[0] + u0[1] * u0[1] + u0[2] * u0[2]) + sqrt(1.0 + u1[0] * u1[0] + u1[1] * u1[1] + u1[2] * u1[2]);
    for (int a = 0; a < 3; ++a) vb[a] = (u0[a] + u1[a]) / g;
    zg = (r0[0] + dt * vb[0]) + lo;                               // the stored position (decision 58)
    r1[0] = zg - lo; r1[1] = r0[1] + dt * vb[1]; r1[2] = r0[2] + dt * vb[2];
}

// The Crank-Nicolson map of one particle with the fields fixed: G(u1) = u0 + q/m dt (Ebar + vbar x Bbar) along the path of u1 (per
// segment f_s = f_ref + (W_s - f_ref . ds) ds / |ds|^2, so f_s . ds = W_s exactly and f_s stays bounded for a short segment;
// Ebar = the mean of f_s; Bbar at the path midpoint)
template <int O>
HD inline void cn_map(const RZET& T, const RZViews& FB, const RZK& k, const double r0[3], const double u0[3], const double u1[3],
                      int nseg, double qmdt, double dt, double lo, double out[3]) {
    double vb[3], r1[3], zg;
    cn_end(r0, u0, u1, dt, lo, vb, r1, zg);
    double F[3] = {0, 0, 0}, a0[3] = {r0[0], r0[1], r0[2]};
    for (int s = 1; s <= nseg; ++s) {
        double a1[3], pm[3], fr[3], dl[3];
        for (int c = 0; c < 3; ++c) a1[c] = s == nseg ? r1[c] : r0[c] + (r1[c] - r0[c]) * (double(s) / nseg);
        for (int c = 0; c < 3; ++c) { pm[c] = 0.5 * (a0[c] + a1[c]); dl[c] = a1[c] - a0[c]; }
        const double Ws = work_seg<O>(T, k, a0, a1);
        fref<O>(T, k, pm, fr);
        const double l2 = dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2];
        const double cc = l2 > 0 ? (Ws - (fr[0] * dl[0] + fr[1] * dl[1] + fr[2] * dl[2])) / l2 : 0.0;
        for (int c = 0; c < 3; ++c) { F[c] += (fr[c] + cc * dl[c]) / nseg; a0[c] = a1[c]; }
    }
    double Ed[3], B[3];
    gather_rz<O>(FB, k, 0.5 * (r0[0] + r1[0]), 0.5 * (r0[1] + r1[1]), 0.5 * (r0[2] + r1[2]), Ed, B);
    const double vxb[3] = {vb[1] * B[2] - vb[2] * B[1], vb[2] * B[0] - vb[0] * B[2], vb[0] * B[1] - vb[1] * B[0]};
    for (int c = 0; c < 3; ++c) out[c] = u0[c] + qmdt * (F[c] + vxb[c]);
}

// mode 0: one update of u1, then the deposit of the new path into jd; mode 1: commit (x, u <- the path end, u1).
// The update is the Picard step u1 <- G(u1), except for a path that comes within 2 dr of the axis: there the radial force is the
// field gradient times xi'^2 (~11 for the C1 map), the Picard map of the particle stops contracting (|dG/du1| > 1: a period-2
// oscillation that the global Anderson on J does not see), and one Newton step u1 <- u1 + (1 - dG/du1)^-1 (G(u1) - u1) with a
// difference Jacobian (3 more evaluations of G) is taken instead. The fixed point, and so the step, is the same.
template <int O> void implicit_pass_impl(Species& sp, FieldsRZ& f, Field* et, Field* bb, Field* jd, double* u1p, int nseg, int mode) {
    const Domain& d = f.dom();
    const RZK k{d.h[0], d.lo(0), d.dt(), d.n[1], f.modes(), d.ng, f.grid().view(), 0, false, f.c1_map ? 4 : O, f.c1_map ? 4 : (f.map_all ? O : 0), 0};
    const RZET T{et[0].view(), et[1].view(), et[2].view()};
    const RZViews FB{et[0].view(), et[1].view(), et[2].view(), bb[0].view(), bb[1].view(), bb[2].view(), jd[0].view(), jd[1].view(),
                     jd[2].view()};
    double *X = sp.ptr<double>(Species::X), *Y = sp.ptr<double>(Species::Y), *Z = sp.ptr<double>(Species::Z);
    double *UX = sp.ptr<double>(Species::UX), *UY = sp.ptr<double>(Species::UY), *UZ = sp.ptr<double>(Species::UZ);
    const double* W = sp.ptr<double>(Species::W);
    const long n = long(sp.size());
    double *U1x = u1p, *U1y = u1p + n, *U1z = u1p + 2 * n;
    const double qmdt = sp.q * k.dt / sp.m, q = sp.q, dt = k.dt, lo = k.lo;
    par_for(n, [=] HD(long p) {
        const double r0[3] = {X[p] - lo, Y[p], Z[p]}, u0[3] = {UX[p], UY[p], UZ[p]};
        double u1[3] = {U1x[p], U1y[p], U1z[p]}, vb[3], r1[3], zg;
        if (mode == 0) {
            cn_end(r0, u0, u1, dt, lo, vb, r1, zg);
            const double cx = r1[1] - r0[1], cy = r1[2] - r0[2], cl2 = cx * cx + cy * cy;   // closest approach of the path to the axis
            double ta = cl2 > 0 ? -(r0[1] * cx + r0[2] * cy) / cl2 : 0.0;
            ta = ta < 0 ? 0.0 : (ta > 1 ? 1.0 : ta);
            const double ax = r0[1] + ta * cx, ay = r0[2] + ta * cy;
            double g0[3];
            cn_map<O>(T, FB, k, r0, u0, u1, nseg, qmdt, dt, lo, g0);
            bool newton = ax * ax + ay * ay < 4 * k.g.dr * k.g.dr;
            if (newton) {
                double A[3][3];
                for (int c = 0; c < 3; ++c) {
                    const double h = 1e-7 * (1 + fabs(u1[c]));
                    double up[3] = {u1[0], u1[1], u1[2]}, gc[3];
                    up[c] += h;
                    cn_map<O>(T, FB, k, r0, u0, up, nseg, qmdt, dt, lo, gc);
                    for (int a = 0; a < 3; ++a) A[a][c] = (a == c ? 1.0 : 0.0) - (gc[a] - g0[a]) / h;
                }
                const double b[3] = {g0[0] - u1[0], g0[1] - u1[1], g0[2] - u1[2]};
                const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                                   A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
                if (fabs(det) > 1e-12) {                                  // Cramer
                    for (int c = 0; c < 3; ++c) {
                        double M[3][3];
                        for (int a = 0; a < 3; ++a)
                            for (int e = 0; e < 3; ++e) M[a][e] = e == c ? b[a] : A[a][e];
                        const double dc = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                                          M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
                        g0[c] = u1[c] + dc / det;
                    }
                }
            }
            for (int c = 0; c < 3; ++c) u1[c] = g0[c];
            U1x[p] = u1[0]; U1y[p] = u1[1]; U1z[p] = u1[2];
        }
        cn_end(r0, u0, u1, dt, lo, vb, r1, zg);
        if (mode == 0) {
            double a0[3] = {r0[0], r0[1], r0[2]};
            for (int s = 1; s <= nseg; ++s) {
                double a1[3];
                for (int c = 0; c < 3; ++c) a1[c] = s == nseg ? r1[c] : r0[c] + (r1[c] - r0[c]) * (double(s) / nseg);
                deposit_imp<O>(FB, k, a0, a1, q * W[p]);
                for (int c = 0; c < 3; ++c) a0[c] = a1[c];
            }
        } else {
            X[p] = zg; Y[p] = r1[1]; Z[p] = r1[2];
            UX[p] = u1[0]; UY[p] = u1[1]; UZ[p] = u1[2];
        }
    });
}

template <int O> void wp2_rz_impl(Species& sp, FieldsRZ& f, Field& Wf) {
    const Domain& d = f.dom();
    const RZK k{d.h[0], d.lo(0), d.dt(), d.n[1], f.modes(), d.ng, f.grid().view(), 0, false, f.c1_map ? 4 : O, f.c1_map ? 4 : (f.map_all ? O : 0), 0};
    const FV R = Wf.view();
    const double *X = sp.ptr<double>(Species::X), *Y = sp.ptr<double>(Species::Y), *Z = sp.ptr<double>(Species::Z);
    const double *UX = sp.ptr<double>(Species::UX), *UY = sp.ptr<double>(Species::UY), *UZ = sp.ptr<double>(Species::UZ);
    const double* W = sp.ptr<double>(Species::W);
    const double fq = sp.q * sp.q / sp.m;
    par_for(long(sp.size()), [=] HD(long p) {
        const double zl = X[p] - k.lo, r = sqrt(Y[p] * Y[p] + Z[p] * Z[p]);
        int i, j;
        double wz[O + 1], wr[O + 1];
        weights<O>(zl / k.dz - 0.5, i, wz); weights<O>(rz_xi(k.g, r, k.xm), j, wr);
        if (j + O + 1 > k.nr + k.ng) return;
        const double c0 = fq * W[p] / sqrt(1.0 + UX[p] * UX[p] + UY[p] * UY[p] + UZ[p] * UZ[p]);
        for (int a = 0; a <= O; ++a)
            for (int b = 0; b <= O; ++b) {
                double fs;
                const int jt = fold_node(j + b, 0, fs);
                atomic_add(&R(i + a, jt, 0), fs * c0 * wz[a] * wr[b] / vol(k, jt));
            }
    });
}
}  // namespace

void rz_push_deposit(Species& sp, FieldsRZ& f, Pusher pusher, bool deposit, int order) {
    if (order == 2) push_deposit_impl<2>(sp, f, pusher, deposit);
    else push_deposit_impl<1>(sp, f, pusher, deposit);
}

void rz_rho(Species& sp, FieldsRZ& f, Field& rho, int order) {
    if (order == 2) rho_impl<2>(sp, f, rho);
    else rho_impl<1>(sp, f, rho);
}

void rz_rho_range(Species& sp, FieldsRZ& f, Field& rho, int order, double z0, double z1) {
    if (order == 2) rho_impl<2>(sp, f, rho, z0, z1, true);
    else rho_impl<1>(sp, f, rho, z0, z1, true);
}

void rz_gather_point(FieldsRZ& f, double z, double x, double y, double E[3], double B[3], int order) {
    if (order == 2) gather_point_impl<2>(f, z, x, y, E, B);
    else gather_point_impl<1>(f, z, x, y, E, B);
}

// The wake probe (wake.hpp, decision 105): F/q of a test charge moving with c along +z at (x, y) and the local z nodes
// i = 0 .. nz - 1, all modes summed at its phi, linear in r (plain shape-1 map, no smoothing): out[3 i + c] =
// (E_z, E_x - c B_y, E_y + c B_x). Every component sits at the z nodes, so along z these are the node values.
void rz_wake_line(FieldsRZ& f, double x, double y, int nz, double* out) {
    const Domain& d = f.dom();
    const RZK k{d.h[0], d.lo(0), d.dt(), d.n[1], f.modes(), d.ng, f.grid().view(), 0, false, 1};
    const RZViews F{f.ez.view(), f.er.view(), f.ep.view(), f.bz.view(), f.br.view(), f.bp.view(), f.jz.view(), f.jr.view(), f.jp.view()};
    par_for(nz, [=] HD(long q) {
        double e[3], b[3];
        gather_rz<1>(F, k, double(q) * k.dz, x, y, e, b);
        out[3 * q] = e[0];
        out[3 * q + 1] = e[1] - b[2];
        out[3 * q + 2] = e[2] + b[1];
    });
}

// Field ionization (ionization.hpp): E^n at the particles (the unsmoothed field), and the current of the displacements of the
// new electrons (straight segments shorter than half a cell: one deposit_seg each, charge-conserving per mode).
namespace {
template <int O> void field_at_impl(Species& sp, FieldsRZ& f, double* E, double* Bo) {
    const Domain& d = f.dom();
    const RZK k{d.h[0], d.lo(0), d.dt(), d.n[1], f.modes(), d.ng, f.grid().view(), f.gather_matched ? O : 0, f.gather_energy, f.c1_map ? 4 : O,
                f.c1_map ? 4 : (f.map_all ? O : 0), (f.map_all || f.c1_map) && f.gather_matched ? O : 0};
    const RZViews F{f.ez.view(), f.er.view(), f.ep.view(), f.bz.view(), f.br.view(), f.bp.view(), f.jz.view(), f.jr.view(), f.jp.view()};
    const double *X = sp.ptr<double>(Species::X), *Y = sp.ptr<double>(Species::Y), *Z = sp.ptr<double>(Species::Z);
    const double xf = d.xfreeze;
    par_for(long(sp.size()), [=] HD(long p) {
        double e[3] = {0, 0, 0}, b[3] = {0, 0, 0};
        if (X[p] < xf) gather_rz<O>(F, k, X[p] - k.lo, Y[p], Z[p], e, b);
        E[3 * p] = e[0]; E[3 * p + 1] = e[1]; E[3 * p + 2] = e[2];
        if (Bo) { Bo[3 * p] = b[0]; Bo[3 * p + 1] = b[1]; Bo[3 * p + 2] = b[2]; }
    });
}
// gather and push only, E, B, u^{n-1/2} into EB (9 per particle; the views of push_deposit_impl, incl. the smoothed E)
template <int O> void push_store_impl(Species& sp, FieldsRZ& f, Pusher pusher, double* EB) {
    const Domain& d = f.dom();
    const RZK k{d.h[0], d.lo(0), d.dt(), d.n[1], f.modes(), d.ng, f.grid().view(), f.gather_matched ? O : 0, f.gather_energy, f.c1_map ? 4 : O,
                f.c1_map ? 4 : (f.map_all ? O : 0), (f.map_all || f.c1_map) && f.gather_matched ? O : 0};
    const bool s = f.smooth_a > 0;
    const RZViews F{(s ? f.sez_ : f.ez).view(), (s ? f.ser_ : f.er).view(), (s ? f.sep_ : f.ep).view(), f.bz.view(), f.br.view(),
                    f.bp.view(), f.jz.view(), f.jr.view(), f.jp.view()};
    const double *X = sp.ptr<double>(Species::X), *Y = sp.ptr<double>(Species::Y), *Z = sp.ptr<double>(Species::Z);
    double *UX = sp.ptr<double>(Species::UX), *UY = sp.ptr<double>(Species::UY), *UZ = sp.ptr<double>(Species::UZ);
    const int* ZS = sp.zslot >= 0 ? sp.ptr<int>(sp.zslot) : nullptr;
    const double qmdt0 = sp.q * k.dt / sp.m, xf = d.xfreeze;
    const QuadView qv = quadrupoles();
    par_for(long(sp.size()), [=] HD(long p) {
        double* o = EB + 9 * p;
        if (X[p] >= xf) { for (int c = 0; c < 9; ++c) o[c] = 0; return; }
        double E[3], B[3], u[3] = {UX[p], UY[p], UZ[p]};
        gather_rz<O>(F, k, X[p] - k.lo, Y[p], Z[p], E, B);
        if (qv.n) qv.add(X[p], Y[p], Z[p], u[0] * k.dt / sqrt(1.0 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), B);
        for (int c = 0; c < 3; ++c) { o[c] = E[c]; o[3 + c] = B[c]; o[6 + c] = u[c]; }
        push_momentum_inline(pusher, ZS ? qmdt0 * ZS[p] : qmdt0, E, B, u);
        UX[p] = u[0]; UY[p] = u[1]; UZ[p] = u[2];
    });
}
template <int O> void jumps_impl(FieldsRZ& f, const double* J0, const double* J1, const double* QW, long n) {
    const Domain& d = f.dom();
    const RZK k{d.h[0], d.lo(0), d.dt(), d.n[1], f.modes(), d.ng, f.grid().view(), f.gather_matched ? O : 0, f.gather_energy, f.c1_map ? 4 : O,
                f.c1_map ? 4 : (f.map_all ? O : 0), (f.map_all || f.c1_map) && f.gather_matched ? O : 0,
                f.deposit_chord};
    const RZViews F{f.ez.view(), f.er.view(), f.ep.view(), f.bz.view(), f.br.view(), f.bp.view(), f.jz.view(), f.jr.view(), f.jp.view()};
    par_for(n, [=] HD(long q) {
        const double a0[3] = {J0[3 * q] - k.lo, J0[3 * q + 1], J0[3 * q + 2]}, a1[3] = {J1[3 * q] - k.lo, J1[3 * q + 1], J1[3 * q + 2]};
        deposit_any<O>(F, k, a0, a1, QW[q]);
    });
}
}  // namespace

void rz_field_at(Species& sp, FieldsRZ& f, double* E, int order, double* B) {
    if (order == 2) field_at_impl<2>(sp, f, E, B);
    else field_at_impl<1>(sp, f, E, B);
}

void rz_push_store(Species& sp, FieldsRZ& f, Pusher pusher, int order, double* EB) {
    if (order == 2) push_store_impl<2>(sp, f, pusher, EB);
    else push_store_impl<1>(sp, f, pusher, EB);
}

void rz_jumps(FieldsRZ& f,const double* J0, const double* J1, const double* QW, long n, int order) {
    if (order == 2) jumps_impl<2>(f, J0, J1, QW, n);
    else jumps_impl<1>(f, J0, J1, QW, n);
}

void rz_implicit_pass(Species& sp, FieldsRZ& f, Field* et, Field* bb, Field* jd, double* u1, int nseg, int mode) {
    implicit_pass_impl<2>(sp, f, et, bb, jd, u1, nseg, mode);
}

void rz_wp2(Species& sp, FieldsRZ& f, Field& W) { wp2_rz_impl<2>(sp, f, W); }
