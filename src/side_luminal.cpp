// Side injection, the flying focus (docs/specs/side_injection.md): a luminal Bessel pulse as a sum of exact discrete
// plane waves. Every component has w - kx = kappa = omega0 (1 - cos theta), so the pattern moves at c; frequencies
// w = 1 + nu (a Gaussian envelope exp(-zeta^2 / L^2) on the axis, sampled in nu), azimuths phi of k_perp on N_phi
// points (the quadrature of the cone, exact for the Bessel orders up to ~N_phi in the box), |k_perp| from the discrete
// dispersion at each phi. In the window moving at c from t = 0 the incident field is a fixed pattern times e^{-i kappa t}:
// the corrections at the face columns are tabulated once (build_table), the initial field is a sum over azimuths per
// transverse cell and then over frequencies (fill_luminal).
#include "side.hpp"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include "side_kernels.hpp"

using namespace side;

std::vector<SideTfsf::Wave> SideTfsf::luminal_waves(const Domain& d, const SideConfig& c, std::vector<int>& groups) {
    using cd = std::complex<double>;
    const bool irip = d.solver == Solver::IRip;
    const double wp2 = c.density;
    const bool two = split(d, c);                      // RIP-CC in a plasma: a TE and a TM wave per component
    const double th = c.cone, kap = 1 - std::cos(th), L = c.len, k0x = std::cos(th);
    const double yc = c.r0[1], zc = c.r0[2];
    double Rc = 0;                                     // the farthest transverse corner of the box from the axis
    for (int p = 0; p < 4; ++p)
        Rc = std::max(Rc, std::hypot((p & 1 ? d.N[1] * d.h[1] : 0.0) - yc, (p & 2 ? d.N[2] * d.h[2] : 0.0) - zc));
    // the envelope exp(-zeta^2 / L^2) = sum over nu of (dnu / 2 pi) G(nu) e^{i nu zeta}, G = sqrt(pi) L e^{-nu^2 L^2 / 4},
    // to e^{-36} of the centre; periodic in zeta with 2 pi / dnu, twice the window, the pulse and the ring lead
    const double numax = 12 / L, Pz = 2 * (d.N[0] * d.h[0] + 12 * L + Rc * std::tan(th)), dnu = 2 * M_PI / Pz;
    if (k0x - numax <= 0.02)
        throw std::runtime_error("side flying focus: length_1e too short for the cone (components with k_x <= 0)");
    const int M = int(std::ceil(numax / dnu));
    const double kpm = std::sqrt(2 * (k0x + numax) * kap + kap * kap);   // the largest k_perp (continuum)
    const int Nphi = std::max(8, 2 * int(std::ceil((1.25 * kpm * Rc + 12) / 2)));
    if (long(2 * M + 1) * Nphi > 50000)
        throw std::runtime_error("side flying focus: " + std::to_string(long(2 * M + 1) * Nphi) +
                                 " components (> 50000): a smaller box or a longer pulse");
    const double p[3] = {0, c.pol == 'y' ? 1.0 : 0.0, c.pol == 'z' ? 1.0 : 0.0};   // E on the axis
    std::vector<Wave> ws;
    groups.clear();
    cd S = 0;                                          // the envelope at the centre on the axis, before scaling
    for (int m = -M; m <= M; ++m) {
        const double nu = m * dnu, kx = k0x + nu, w = kx + kap;
        const double G = (dnu / (2 * M_PI)) * std::sqrt(M_PI) * L * std::exp(-0.25 * nu * nu * L * L) / Nphi;
        {                                              // below the plasma cutoff of the cone (k_perp^2 < 0): only in the far tail
            const double k[3] = {kx, 0, 0};
            if (omega(d, k, irip, wp2, 0) > w) {
                if (std::exp(-0.25 * nu * nu * L * L) > 1e-6)
                    throw std::runtime_error("side flying focus: no conical wave at omega = " + std::to_string(w) +
                                             " (k_perp^2 < 0: background_density too high for this cone angle and pulse)");
                continue;
            }
        }
        groups.push_back(int(ws.size()));
        for (int l = 0; l < Nphi; ++l) {
            const double ph = 2 * M_PI * (l + 0.5) / Nphi, u[2] = {std::cos(ph), std::sin(ph)};
            double smax = 1e300;
            for (int a = 0; a < 2; ++a)
                if (std::abs(u[a]) > 1e-12) smax = std::min(smax, M_PI / (std::abs(u[a]) * d.h[1 + a]));
            smax *= 1 - 1e-12;
            // |k_perp| at this azimuth from the discrete dispersion (RIP-CC in a plasma: the TE and the TM wave)
            Wave x[2];
            for (int p2 = 0; p2 < (two ? 2 : 1); ++p2) {
                const int cw = two ? p2 : 0;
                double k[3] = {kx, smax * u[0], smax * u[1]}, lo = 0, hi = smax;
                if (omega(d, k, irip, wp2, cw) < w) throw std::runtime_error("side flying focus: the cone reaches beyond the grid cutoff");
                for (int it = 0; it < 200; ++it) {
                    const double s = 0.5 * (lo + hi);
                    k[1] = s * u[0]; k[2] = s * u[1];
                    (omega(d, k, irip, wp2, cw) < w ? lo : hi) = s;
                }
                const double s = 0.5 * (lo + hi);
                k[1] = s * u[0]; k[2] = s * u[1];
                x[p2] = Wave{{k[0], k[1], k[2]}, w, {0, 0}, {0, 0, 0}, two ? p2 : -1};
            }
            // E along p - (p . k^) k^ (k of the first wave), the amplitude at the reference point
            const double* k = x[0].k;
            const double kk = std::sqrt(k[0] * k[0] + k[1] * k[1] + k[2] * k[2]), kh[3] = {k[0] / kk, k[1] / kk, k[2] / kk};
            const double pk = p[0] * kh[0] + p[1] * kh[1] + p[2] * kh[2];
            const double e[3] = {p[0] - pk * kh[0], p[1] - pk * kh[1], p[2] - pk * kh[2]};
            S += G * (e[0] * p[0] + e[1] * p[1] + e[2] * p[2]);
            for (int p2 = 0; p2 < (two ? 2 : 1); ++p2) {
                const double* q = x[p2].k;
                // the carrier-envelope phase and the reference: k . (r - (zeta_c, y_c, z_c)) at t = 0
                const cd a = G * std::exp(cd(0, c.phase - (q[0] * c.r0[0] + q[1] * yc + q[2] * zc)));
                x[p2].amp[0] = a.real(); x[p2].amp[1] = a.imag();
                for (int b = 0; b < 3; ++b) x[p2].e[b] = e[b];
                ws.push_back(x[p2]);
            }
        }
    }
    groups.push_back(int(ws.size()));
    const double f = c.a0 / std::abs(S);               // the peak of the envelope on the axis is a0
    for (Wave& x : ws) { x.amp[0] *= f; x.amp[1] *= f; }
    return ws;
}

namespace {
Geo geo_l(const Domain& d, const SideConfig& s, const Buf<double>& cp, const Buf<double>& tb) {
    Geo g;
    for (int a = 0; a < 3; ++a) { g.lo[a] = s.lo[a]; g.hi[a] = s.hi[a]; g.f[a] = s.faces[a]; }
    g.o1 = d.off[1]; g.o2 = d.off[2];
    g.nx = d.n[0]; g.ny = d.n[1]; g.nz = d.n[2]; g.S = 3 + g.ny + g.nz;
    g.D = d.h[0]; g.hy = d.h[1]; g.hz = d.h[2];
    g.cp = cp.data(); g.tb = tb.data(); g.tx = nullptr; g.ft = nullptr;
    g.nc = int(cp.size() / NCD);
    return g;
}
// e^{i kx_g (off + i) D} per frequency group g and local i (the window frame: the shift is in e^{-i kappa t})
std::vector<double> xtab(const Domain& d, const std::vector<double>& gx) {
    const int nx = d.n[0];
    std::vector<double> X(2 * gx.size() * size_t(nx));
    for (size_t g = 0; g < gx.size(); ++g)
        for (int i = 0; i < nx; ++i) {
            const double ph = gx[g] * double(d.off[0] + i) * d.h[0];
            X[2 * (g * nx + i)] = std::cos(ph); X[2 * (g * nx + i) + 1] = std::sin(ph);
        }
    return X;
}
struct Nine { FV v[9]; };
}  // namespace

// The corrections at the face columns: per frequency group the sum over the azimuths at each column (Q), then the sum
// over the groups with e^{i kx (off + i) D} at each local x (H: 8 complex per column and x, see side.hpp).
void SideTfsf::build_table() {
    const Geo g = geo_l(d_, c_, dev_, tab_);
    const int nb = nb_, nx = d_.n[0], G = int(gx_.size()), kind = d_.solver == Solver::IRip ? (d_.o5 ? 3 : 2) : 0;
    const double iy = 1 / g.hy, iz = 1 / g.hz;
    Buf<int> gs;
    gs.upload(groups_);
    Buf<double> Q(size_t(16) * G * std::max(nb, 1)), X;
    X.upload(xtab(d_, gx_));
    H_.resize(size_t(16) * std::max(nb, 1) * nx);
    const int* jk = jk_.data();
    const int* gp = gs.data();
    double* Qp = Q.data();
    par_for(long(nb) * G, [=] HD(long q) {
        const int b = int(q % nb), gr = int(q / nb), j = jk[2 * b], k = jk[2 * b + 1];
        const Wts w = weights(g, g.o1 + j, g.o2 + k);
        Cx s[8] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
        for (int m = gp[gr]; m < gp[gr + 1]; ++m) {
            const double* c = g.cp + m * NCD;
            const Cx P = g.perp(m, j, k);
            Cx aE, aB, t[4];
            bracket_pc(c, g.ey(m), g.ez(m), w, iy, iz, aE, aB);
            bracket_t(c, g.ex(m), g.ey(m), g.ez(m), w, iy, iz, kind, t);
            s[0] += P * aE; s[1] += P * aB;
            for (int r = 0; r < 4; ++r) s[2 + r] += P * t[r];
            const Cx Pm = P * MU(c);
            s[6] += Pm * aE; s[7] += Pm * aB;
        }
        double* o = Qp + 16 * (long(gr) * nb + b);
        for (int r = 0; r < 8; ++r) { o[2 * r] = s[r].r; o[2 * r + 1] = s[r].i; }
    });
    const double* Xp = X.data();
    double* Hp = H_.data();
    par_for(long(nb) * nx, [=] HD(long q) {
        const int i = int(q % nx), b = int(q / nx);
        Cx s[8] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
        for (int gr = 0; gr < G; ++gr) {
            const Cx x{Xp[2 * (long(gr) * nx + i)], Xp[2 * (long(gr) * nx + i) + 1]};
            const double* o = Qp + 16 * (long(gr) * nb + b);
            for (int r = 0; r < 8; ++r) s[r] += x * Cx{o[2 * r], o[2 * r + 1]};
        }
        double* h = Hp + 16 * q;
        for (int r = 0; r < 8; ++r) { h[2 * r] = s[r].r; h[2 * r + 1] = s[r].i; }
    });
}
// The incident field at t (the window at x0 = t): per chunk of frequency groups the sum over the azimuths at every
// transverse cell, then the sum over the groups along x. jf: the fluid currents (background plasma), F j^{t - D/2}.
void SideTfsf::fill_luminal(const FV f[6], double t, bool masked, const FV* jf) {
    const Geo g = geo_l(d_, c_, dev_, tab_);
    const int nx = d_.n[0], ny = d_.n[1], nz = d_.n[2], G = int(gx_.size()), CH = 8, nq = jf ? 9 : 6;
    const long nyz = long(ny) * nz;
    Buf<int> gs;
    gs.upload(groups_);
    Buf<double> X, P(size_t(18) * CH * nyz);
    X.upload(xtab(d_, gx_));
    const int* gp = gs.data();
    const double* Xp = X.data();
    double* Pp = P.data();
    const double* F = fl_.data();
    const Cx fk{std::cos(kappa_ * t), -std::sin(kappa_ * t)};
    Nine s;
    for (int q = 0; q < 6; ++q) s.v[q] = f[q];
    for (int q = 0; q < 3; ++q) s.v[6 + q] = jf ? jf[q] : f[q];
    for (int g0 = 0; g0 < G; g0 += CH) {
        const int n = std::min(CH, G - g0);
        par_for(long(n) * nyz, [=] HD(long q) {
            const int gl = int(q / nyz), j = int(q % nyz % ny), k = int(q % nyz / ny);
            Cx a[9] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
            for (int m = gp[g0 + gl]; m < gp[g0 + gl + 1]; ++m) {
                const Cx p = g.perp(m, j, k);
                for (int c = 0; c < nq; ++c) a[c] += p * V9(g.cp + m * NCD, c);
            }
            double* o = Pp + 18 * q;
            for (int c = 0; c < nq; ++c) { o[2 * c] = a[c].r; o[2 * c + 1] = a[c].i; }
        });
        for_owned(d_, [=] HD(int i, int j, int k) {
            const long jg = g.o1 + j, kg = g.o2 + k, jk = long(k) * ny + j;
            for (int c = 0; c < nq; ++c) {
                if (masked && g.T(c, jg, kg) == 0) continue;
                const double w = c < 6 ? 1.0 : F[(c - 6) * nyz + jk];
                if (w == 0) continue;
                double v = 0;
                for (int gl = 0; gl < n; ++gl) {
                    const Cx x{Xp[2 * (long(g0 + gl) * nx + i)], Xp[2 * (long(g0 + gl) * nx + i) + 1]};
                    const double* o = Pp + 18 * (long(gl) * nyz + jk);
                    v += (fk * x * Cx{o[2 * c], o[2 * c + 1]}).r;
                }
                s.v[c](i, j, k) += w * v;
            }
        });
    }
}

// O5: the incident means of Ex, Bx at the time of begin() on the owned tube points (plain or closed), as fill_luminal.
void SideTfsf::means_luminal(const FV ex, const FV bx, bool closed) {
    const Geo g = geo_l(d_, c_, dev_, tab_);
    const int nx = d_.n[0], ny = d_.n[1], nz = d_.n[2], G = int(gx_.size()), CH = 8;
    const long nyz = long(ny) * nz;
    Buf<int> gs;
    gs.upload(groups_);
    Buf<double> X, P(size_t(4) * CH * nyz);
    X.upload(xtab(d_, gx_));
    const int* gp = gs.data();
    const double* Xp = X.data();
    double* Pp = P.data();
    const Cx fk{std::cos(kappa_ * t_), -std::sin(kappa_ * t_)};
    for (int g0 = 0; g0 < G; g0 += CH) {
        const int n = std::min(CH, G - g0);
        par_for(long(n) * nyz, [=] HD(long q) {
            const int gl = int(q / nyz), j = int(q % nyz % ny), k = int(q % nyz / ny);
            Cx a{0, 0}, b{0, 0};
            for (int m = gp[g0 + gl]; m < gp[g0 + gl + 1]; ++m) {
                const double* c = g.cp + m * NCD;
                const Cx p = g.perp(m, j, k);
                a += p * (closed ? PR(c, 0) : V(c, 0) * MU(c));
                b += p * (closed ? PR(c, 1) : V(c, 3) * MU(c));
            }
            double* o = Pp + 4 * q;
            o[0] = a.r; o[1] = a.i; o[2] = b.r; o[3] = b.i;
        });
        for_owned(d_, [=] HD(int i, int j, int k) {
            const long jg = g.o1 + j, kg = g.o2 + k, jk = long(k) * ny + j;
            const bool te = g.T(0, jg, kg) != 0, tb = g.T(3, jg, kg) != 0;
            if (!te && !tb) return;
            double va = 0, vb = 0;
            for (int gl = 0; gl < n; ++gl) {
                const Cx x = fk * Cx{Xp[2 * (long(g0 + gl) * nx + i)], Xp[2 * (long(g0 + gl) * nx + i) + 1]};
                const double* o = Pp + 4 * (long(gl) * nyz + jk);
                va += (x * Cx{o[0], o[1]}).r; vb += (x * Cx{o[2], o[3]}).r;
            }
            if (te) ex(i, j, k) += va;
            if (tb) bx(i, j, k) += vb;
        });
    }
}
