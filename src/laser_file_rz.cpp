// LASY laser in RZ (laser_file.hpp RzFileLaser; docs/specs/laser_file.md "RZ"): the envelope as azimuthal modes at the
// entrance plane z = 0, exact vacuum propagation from file_x per mode (Hankel transform of order |k| and an FFT in u; the
// quadrature of the forward transform from GEM-PIC-RZ src/laser_file.cpp, exact for an interpolant regular at the axis).
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "laser_file.hpp"
#include "laser_file_io.hpp"

using lasy::cd;
using lasy::fmt;

namespace {
// trapezoidal weights on [0, (n-1) h] with Gregory's corrections at the left end up to the 4th differences (the integrands
// r f(r) and kappa F(kappa) are odd at 0, where the plain trapezoid has an O(h^2) error)
double gregory_left(int p, int n) {
    double w = (p == 0 || p == n - 1) ? 0.5 : 1.0;
    if (n < 10) return w;
    const double a1 = 1.0 / 12, a2 = -1.0 / 24, a3 = 19.0 / 720, a4 = -3.0 / 160;
    const double c[5] = {-a1 + a2 - a3 + a4, a1 - 2 * a2 + 3 * a3 - 4 * a4, a2 - 3 * a3 + 6 * a4, a3 - 4 * a4, a4};
    if (p < 5) w += c[p];
    return w;
}

// Weights W_j of F(kappa) = int f(r) J_nu(kappa r) r dr = sum_j W_j f_j on the nodes r_j = j dr, exact for the interpolant
// f = r^nu g(r^2) with g linear in u = r^2 between the nodes (regular at the axis; g(0) extrapolated from nodes 1, 2 for nu > 0).
//   int r^{nu+1} J_nu(k r) dr = r^{nu+1} J_{nu+1} / k,  int r^{nu+3} J_nu dr = r^{nu+3} J_{nu+1} / k - 2 r^{nu+2} J_{nu+2} / k^2
std::vector<double> hankel_weights(int nu, double kap, double dr, int nr) {
    std::vector<double> W(size_t(nr), 0.0);
    auto rr = [&](int j) { return j * dr; };
    auto B1 = [&](double r) {
        if (kap == 0.0) return nu == 0 ? 0.5 * r * r : 0.0;
        return std::pow(r, nu + 1) * std::cyl_bessel_j(nu + 1.0, kap * r) / kap;
    };
    auto B3 = [&](double r) {
        if (kap == 0.0) return nu == 0 ? 0.25 * r * r * r * r : 0.0;
        return std::pow(r, nu + 3) * std::cyl_bessel_j(nu + 1.0, kap * r) / kap -
               2.0 * std::pow(r, nu + 2) * std::cyl_bessel_j(nu + 2.0, kap * r) / (kap * kap);
    };
    auto addG = [&](int j, double c) {
        const double r = rr(j);
        if (r > 0.0 || nu == 0) { W[size_t(j)] += c / std::pow(r, nu); return; }
        if (nr < 3) { W[1] += c / std::pow(rr(1), nu); return; }
        const double ua = rr(1) * rr(1), ub = rr(2) * rr(2), x = ua / (ub - ua);
        W[1] += c * (1.0 + x) / std::pow(rr(1), nu);
        W[2] -= c * x / std::pow(rr(2), nu);
    };
    double b1a = B1(0), b3a = B3(0);
    for (int j = 0; j < nr - 1; ++j) {
        const double b1b = B1(rr(j + 1)), b3b = B3(rr(j + 1)), I1 = b1b - b1a;
        const double ua = rr(j) * rr(j), du = rr(j + 1) * rr(j + 1) - ua, Iu = (b3b - b3a) - ua * I1;
        addG(j, I1 - Iu / du);
        addG(j + 1, Iu / du);
        b1a = b1b; b3a = b3b;
    }
    return W;
}

// a cartesian file (u, Y = LASY x, Z = LASY y) as modes on polar nodes around its origin: a_k = (1/n) sum_q e(phi_q) e^{i k phi_q}
lasy::Modes polar_modes(lasy::Grid& g, int K, const std::string& where, std::vector<std::string>& info) {
    if (g.uy || g.uz) throw std::runtime_error(where + "geometry = rz: a cartesian file needs both transverse axes");
    lasy::Modes M;
    const double dr = std::min(g.dy, g.dz);
    const double R = std::min({g.y0 + (g.ny - 1) * g.dy, -g.y0, g.z0 + (g.nz - 1) * g.dz, -g.z0});
    if (!(R > 2 * dr)) throw std::runtime_error(where + "geometry = rz: the cartesian grid must contain its origin (the axis) well inside");
    M.nu = g.nu; M.u0 = g.u0; M.du = g.du; M.r0 = 0; M.dr = dr; M.K = K;
    M.nr = int(std::floor(R / dr + 1e-9)) + 1;
    const int nth = std::max(32, 4 * K + 4);
    M.a.assign(size_t(2 * K + 1) * M.nu * M.nr, cd(0));
    std::vector<cd> e(static_cast<size_t>(nth));
    for (int l = 0; l < M.nu; ++l)
        for (int j = 0; j < M.nr; ++j) {
            for (int q = 0; q < nth; ++q) {
                const double ph = 2 * M_PI * q / nth, fy = (j * dr * std::cos(ph) - g.y0) / g.dy, fz = (j * dr * std::sin(ph) - g.z0) / g.dz;
                const int iy = std::min(std::max(int(std::floor(fy)), 0), g.ny - 2), iz = std::min(std::max(int(std::floor(fz)), 0), g.nz - 2);
                const double wy = std::min(1.0, std::max(0.0, fy - iy)), wz = std::min(1.0, std::max(0.0, fz - iz));
                e[size_t(q)] = (1 - wz) * ((1 - wy) * g.at(l, iy, iz) + wy * g.at(l, iy + 1, iz)) +
                               wz * ((1 - wy) * g.at(l, iy, iz + 1) + wy * g.at(l, iy + 1, iz + 1));
            }
            for (int k = -K; k <= K; ++k) {
                cd s = 0;
                for (int q = 0; q < nth; ++q) s += e[size_t(q)] * std::polar(1.0, k * 2 * M_PI * q / nth);
                M.at(k, l, j) = s / double(nth);
            }
        }
    info.push_back(fmt("cartesian file resampled on polar nodes around its origin: %.0f radii (dr %.4g), %.0f angles", M.nr, dr, nth));
    return M;
}

// the modes on the nodes j dr from r = 0 (cell-centred files with r0 > 0: a_0 constant and a_k ~ r inside r0)
lasy::Modes axis_nodes(const lasy::Modes& m) {
    if (m.r0 == 0) return m;
    lasy::Modes o = m;
    o.r0 = 0;
    o.nr = int(std::floor((m.r0 + (m.nr - 1) * m.dr) / m.dr + 1e-9)) + 1;
    o.a.assign(size_t(2 * m.K + 1) * m.nu * o.nr, cd(0));
    for (int k = -m.K; k <= m.K; ++k)
        for (int l = 0; l < m.nu; ++l)
            for (int j = 0; j < o.nr; ++j) {
                const double r = j * m.dr, y = (r - m.r0) / m.dr;
                if (y < 0) { o.at(k, l, j) = m.at(k, l, 0) * (k ? r / m.r0 : 1.0); continue; }
                const int a = std::min(int(y), m.nr - 2);
                const double w = std::min(1.0, y - a);
                o.at(k, l, j) = (1 - w) * m.at(k, l, a) + w * m.at(k, l, a + 1);
            }
    return o;
}

// Exact propagation over L along z at fixed retarded time: every (omega = omega_L + Omega, kappa) of mode k gets
// e^{i (k_z - omega) L}, k_z = (omega^2 - kappa^2)^{1/2}; evanescent and backward parts dropped. Output on the nodes j dr up to
// rout and on the padded time window (twice the file's).
lasy::Modes propagate_modes(const lasy::Modes& m, double omegaL, double L, double rout, std::vector<std::string>& info) {
    const int K = m.K, NK = 2 * K + 1, nr = m.nr, nu = m.nu;
    const int nt2 = int(lasy::pow2(size_t(2 * nu))), off = (nt2 - nu) / 2;
    const double dOm = 2 * M_PI / (nt2 * m.du);
    lasy::Modes o;
    o.K = K; o.nu = nt2; o.du = m.du; o.u0 = m.u0 - off * m.du; o.r0 = 0; o.dr = m.dr;
    o.nr = int(std::ceil(rout / m.dr)) + 2;
    o.a.assign(size_t(NK) * nt2 * o.nr, cd(0));
    // FFT in u: F[(k * nr + j) * nt2 + q] = sum_l a_l e^{+i Omega_q (u_l - u_start)}
    std::vector<cd> F(size_t(NK) * nr * nt2, cd(0));
    for (int k = 0; k < NK; ++k)
        for (int j = 0; j < nr; ++j) {
            cd* s = &F[(size_t(k) * nr + j) * nt2];
            for (int l = 0; l < nu; ++l) s[off + l] = m.at(k - K, l, j);
            lasy::fft(s, size_t(nt2), 1, +1);
        }
    std::vector<double> pw(size_t(nt2), 0.0);
    for (int k = 0; k < NK; ++k)
        for (int j = 0; j < nr; ++j)
            for (int q = 0; q < nt2; ++q) pw[size_t(q)] += std::norm(F[(size_t(k) * nr + j) * nt2 + q]) * std::max(j * m.dr, 0.25 * m.dr);
    const double pmax = *std::max_element(pw.begin(), pw.end());
    int qlo = nt2, qhi = -nt2;
    for (int q = 0; q < nt2; ++q) {
        const int qs = q < nt2 / 2 ? q : q - nt2;
        if (pw[size_t(q)] > 1e-12 * pmax && omegaL + qs * dOm > 0) { qlo = std::min(qlo, qs); qhi = std::max(qhi, qs); }
    }
    if (qhi < qlo) throw std::runtime_error("laser file: no propagating frequency in the spectrum");
    const int nq = qhi - qlo + 1;
    const double Rf = (nr - 1) * m.dr, Rn = std::max(Rf, rout), dk = 2 * M_PI / (4 * (Rf + Rn));   // images 2 pi / dk away: 4 (Rf + Rn) keeps them off the radii used
    const double wmax = omegaL + qhi * dOm, kmax = std::min(M_PI / m.dr, wmax);
    const int nk = std::max(2, int(std::floor(kmax / dk)) + 1);
    // G = forward Hankel transform with the inverse weights kappa dk folded in
    std::vector<cd> G(size_t(NK) * nq * nk, cd(0));
    for (int k = 0; k < NK; ++k) {
        const int order = std::abs(k - K);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int p = 0; p < nk; ++p) {
            const double kap = p * dk;
            const std::vector<double> W = hankel_weights(order, kap, m.dr, nr);
            for (int q = 0; q < nq; ++q) {
                const int qi = ((qlo + q) % nt2 + nt2) % nt2;
                cd s = 0;
                for (int j = 0; j < nr; ++j) s += F[(size_t(k) * nr + j) * nt2 + qi] * W[size_t(j)];
                G[(size_t(k) * nq + q) * nk + p] = s * kap * dk * gregory_left(p, nk);
            }
        }
    }
    std::vector<cd> ph(size_t(nq) * nk, cd(0));
    double all = 0, kept = 0;
    for (int q = 0; q < nq; ++q) {
        const double w = omegaL + (qlo + q) * dOm;
        for (int p = 0; p < nk; ++p) {
            const double kz2 = w * w - p * dk * p * dk;
            if (kz2 > 0) ph[size_t(q) * nk + p] = std::polar(1.0, (std::sqrt(kz2) - w) * L);
        }
    }
    for (int k = 0; k < NK; ++k)
        for (int q = 0; q < nq; ++q)
            for (int p = 0; p < nk; ++p) {
                const double e = std::norm(G[(size_t(k) * nq + q) * nk + p]) / std::max(p * dk, 0.25 * dk);
                all += e;
                if (std::abs(ph[size_t(q) * nk + p]) > 0) kept += e;
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < o.nr; ++i) {
        std::vector<cd> H(static_cast<size_t>(nq));
        std::vector<double> J(static_cast<size_t>(nk));
        for (int k = 0; k < NK; ++k) {
            const int order = std::abs(k - K);
            for (int p = 0; p < nk; ++p) J[size_t(p)] = std::cyl_bessel_j(double(order), p * dk * i * m.dr);
            for (int q = 0; q < nq; ++q) {
                cd s = 0;
                const cd* g = &G[(size_t(k) * nq + q) * nk];
                const cd* e = &ph[size_t(q) * nk];
                for (int p = 0; p < nk; ++p) s += g[p] * e[p] * J[size_t(p)];
                H[size_t(q)] = s;
            }
            for (int l = 0; l < nt2; ++l) {
                const double x = l * m.du;
                const cd step = std::polar(1.0, -dOm * x);
                cd e = std::polar(1.0, -qlo * dOm * x), s = 0;
                for (int q = 0; q < nq; ++q) { s += H[size_t(q)] * e; e *= step; }
                o.at(k - K, l, i) = s / double(nt2);
            }
        }
    }
    info.push_back(fmt("propagated over %.6g (1/k0) to the entrance plane, exactly per mode (FFT %.0f, %.0f frequencies, ", L, nt2, nq) +
                   fmt("%.0f radial wavenumbers), %.2g of the energy evanescent or backward, dropped", nk, all > 0 ? 1 - kept / all : 0.0));
    return o;
}
}  // namespace

RzFileLaser::RzFileLaser(const LaserFileConfig& c, double lambda0, int modes, double r_need) {
    double omegaL;
    cd px, py;
    const int Ksim = modes - 1;
    lasy::Modes file;
    lasy::Grid g = lasy::read_file(c, lambda0, omegaL, px, py, info, &file);
    if (!g.e.empty()) file = polar_modes(g, Ksim, c.path + ": ", info);
    file = axis_nodes(file);
    // keep the modes the simulation has
    lasy::Modes m;
    m.K = std::min(file.K, Ksim); m.nu = file.nu; m.nr = file.nr; m.u0 = file.u0; m.du = file.du; m.dr = file.dr;
    m.a.assign(size_t(2 * m.K + 1) * m.nu * m.nr, cd(0));
    double lost = 0, tot = 0;
    for (int k = -file.K; k <= file.K; ++k)
        for (int l = 0; l < m.nu; ++l)
            for (int j = 0; j < m.nr; ++j) {
                const double e = std::norm(file.at(k, l, j)) * std::max(j * m.dr, 0.25 * m.dr);
                tot += e;
                if (std::abs(k) <= m.K) m.at(k, l, j) = file.at(k, l, j);
                else lost += e;
            }
    if (lost > 0) info.push_back(fmt("modes |k| > %.0f dropped: %.2g of the energy", m.K, tot > 0 ? lost / tot : 0.0));
    // peak |env| over angles, intensity centroid in time
    const int nth = 4 * m.K + 8;
    auto peak_of = [&](const lasy::Modes& M) {
        double pk = 0;
        for (int l = 0; l < M.nu; ++l)
            for (int j = 0; j < M.nr; ++j)
                for (int q = 0; q < (M.K ? nth : 1); ++q) {
                    cd v = 0;
                    for (int k = -M.K; k <= M.K; ++k) v += M.at(k, l, j) * std::polar(1.0, -k * 2 * M_PI * q / nth);
                    pk = std::max(pk, std::abs(v));
                }
        return pk;
    };
    peak = peak_of(m);
    if (!(peak > 0)) throw std::runtime_error(c.path + ": the envelope is zero");
    double w = 0, wt = 0;
    for (int k = -m.K; k <= m.K; ++k)
        for (int l = 0; l < m.nu; ++l)
            for (int j = 0; j < m.nr; ++j) {
                const double a = std::norm(m.at(k, l, j)) * std::max(j * m.dr, 0.25 * m.dr);
                w += a; wt += a * (m.u0 + l * m.du);
            }
    const double tc = c.t_center < 1e299 ? c.t_center : wt / w;
    const double first = m.u0, last = m.u0 + (m.nu - 1) * m.du;
    const double delay = c.delay >= 0 ? c.delay : tc - first;
    if (c.a0 > 0) {
        const double s = c.a0 * omegaL / peak;
        for (cd& v : m.a) v *= s;
        peak = c.a0 * omegaL;
    }
    info.push_back(fmt("peak |E| %.4g (a0 %.4g), t_center %.6g in the file's time", peak, peak / omegaL, tc) +
                   fmt(", at the entrance plane at t = %.6g (1/omega0)", delay));
    if (c.file_x != 0) m = propagate_modes(m, omegaL, -c.file_x, std::max(r_need, (m.nr - 1) * m.dr), info);
    double edge = 0;
    for (int k = -m.K; k <= m.K; ++k)
        for (int l = 0; l < m.nu; ++l)
            for (int j = 0; j < m.nr; ++j)
                if (l == 0 || l == m.nu - 1 || j == m.nr - 1) edge = std::max(edge, std::abs(m.at(k, l, j)));
    this->edge = edge / peak;
    t_first = delay + m.u0 - tc;
    t_last = delay + m.u0 + (m.nu - 1) * m.du - tc;
    (void)first; (void)last;
    // the view's layout: [((k + K) * nr + j) * nu + l]
    std::vector<double> h(2 * size_t(2 * m.K + 1) * m.nr * m.nu);
    for (int k = -m.K; k <= m.K; ++k)
        for (int j = 0; j < m.nr; ++j)
            for (int l = 0; l < m.nu; ++l) {
                const size_t q = 2 * ((size_t(k + m.K) * m.nr + j) * m.nu + l);
                h[q] = m.at(k, l, j).real(); h[q + 1] = m.at(k, l, j).imag();
            }
    env_.resize(h.size());
    env_.from_host(h.data(), 0, h.size());
    host_ = std::move(h);
    RzFileLaserView& v = v_;
    v.nu = m.nu; v.nr = m.nr; v.K = m.K; v.u0 = m.u0; v.du = m.du; v.dr = m.dr;
    v.omega = omegaL; v.u_shift = tc - delay;
    v.pxr = px.real(); v.pxi = px.imag(); v.pyr = py.real(); v.pyi = py.imag();
    if (m.nr > 1 && (m.nr - 1) * m.dr < r_need && this->edge > 1e-3)
        info.push_back(fmt("the file ends at r = %.4g (1/k0) inside the radial grid (%.4g): the field is 0 beyond", (m.nr - 1) * m.dr, r_need));
    info.push_back(fmt("enters z = 0 from t = %.6g to %.6g (1/omega0); envelope at the grid edge %.2g of the peak", t_first, t_last,
                       this->edge));
}
