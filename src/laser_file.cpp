#include "laser_file.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include "h5.hpp"
#include "laser_file_io.hpp"

using lasy::cd;
using lasy::fft;
using lasy::fmt;
using lasy::Grid;
using lasy::pow2;
namespace {
const double c_si = 299792458.0, e_si = 1.602176634e-19, me_si = 9.1093837015e-31;
}  // namespace

std::string lasy::fmt(const char* f, double a, double b, double c) {
    char s[200];
    std::snprintf(s, sizeof s, f, a, b, c);
    return s;
}

// in-place FFT of a power-of-two length: a_q <- sum_l a_l exp(sign 2 pi i q l / N)
void lasy::fft(cd* a, size_t n, long stride, int sign) {
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i * stride], a[j * stride]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2 * M_PI / double(len);
        for (size_t i = 0; i < n; i += len)
            for (size_t k = 0; k < len / 2; ++k) {
                const cd w = std::polar(1.0, ang * double(k));
                const cd u = a[(i + k) * stride], v = a[(i + k + len / 2) * stride] * w;
                a[(i + k) * stride] = u + v;
                a[(i + k + len / 2) * stride] = u - v;
            }
    }
}
size_t lasy::pow2(size_t n) { size_t p = 1; while (p < n) p <<= 1; return p; }

#ifndef RIP_HDF5
FileLaser::FileLaser(const LaserFileConfig& c, double, double, double, double, double) {
    throw std::runtime_error("laser " + c.name + ": a laser from a file needs a build with HDF5 (make HDF5=1)");
}
Grid lasy::read_file(const LaserFileConfig& c, double, double&, cd&, cd&, std::vector<std::string>&, Modes*) {
    throw std::runtime_error("laser " + c.name + ": a laser from a file needs a build with HDF5 (make HDF5=1)");
}
#else
Grid lasy::read_file(const LaserFileConfig& c, double lambda0, double& omegaL, cd& px, cd& py, std::vector<std::string>& info,
                     Modes* modes) {
    const double k0 = 2 * M_PI / lambda0, w0 = c_si * k0, E0 = me_si * c_si * w0 / e_si;
    h5::Id f = h5::open_file(c.path);
    const std::string where = c.path + ": ";
    if (!h5::has_attr(f, "openPMD")) throw std::runtime_error(where + "not an openPMD file");
    h5::Id data = h5::open_group(f, "data");
    long it = c.iteration;
    if (it < 0)
        for (const std::string& n : h5::children(data)) it = std::max(it, std::atol(n.c_str()));
    if (it < 0 || !h5::exists(data, std::to_string(it))) throw std::runtime_error(where + "no iteration " + std::to_string(it));
    std::string mp = h5::has_attr(f, "meshesPath") ? h5::attr_string(f, "meshesPath") : "meshes/";
    if (mp.empty() || mp.back() != '/') mp += '/';
    const std::string path = std::to_string(it) + "/" + mp + c.mesh;
    if (!h5::exists(data, path)) throw std::runtime_error(where + "no mesh " + c.mesh + " in iteration " + std::to_string(it));
    h5::Id ds = h5::open_dataset(data, path);
    const std::string geo = h5::attr_string(ds, "geometry");
    const bool cart = geo == "cartesian";
    if (!cart && geo != "thetaMode") throw std::runtime_error(where + "geometry " + geo + " (cartesian or thetaMode)");
    std::vector<std::string> lab = h5::attr_strings(ds, "axisLabels");
    std::vector<double> sp = h5::attr_vec<double>(ds, "gridSpacing"), off = h5::attr_vec<double>(ds, "gridGlobalOffset");
    std::vector<double> pos = h5::has_attr(ds, "position") ? h5::attr_vec<double>(ds, "position") : std::vector<double>(lab.size(), 0.0);
    if (h5::has_attr(ds, "dataOrder") && h5::attr_string(ds, "dataOrder") == "F") {
        std::reverse(lab.begin(), lab.end()); std::reverse(sp.begin(), sp.end());
        std::reverse(off.begin(), off.end()); std::reverse(pos.begin(), pos.end());
    }
    const size_t na = cart ? 3 : 2;
    if (lab.size() != na || sp.size() != na || off.size() != na || pos.size() != na)
        throw std::runtime_error(where + "expected " + std::to_string(na) + " axes");
    auto axis = [&](const char* n) {
        for (size_t k = 0; k < lab.size(); ++k)
            if (lab[k] == n) return int(k);
        throw std::runtime_error(where + "no axis " + n + (std::string(n) == "t" ? " (the envelope must be given in time)" : ""));
    };
    const double gu = h5::has_attr(ds, "gridUnitSI") ? h5::attr_num<double>(ds, "gridUnitSI") : 1.0;
    const double usi = h5::has_attr(ds, "unitSI") ? h5::attr_num<double>(ds, "unitSI") : 1.0;
    if (!h5::has_attr(ds, "angularFrequency")) throw std::runtime_error(where + "no attribute angularFrequency");
    omegaL = h5::attr_num<double>(ds, "angularFrequency") / w0;
    px = 1; py = 0;
    if (h5::has_attr(ds, "polarization")) {
        const std::vector<double> p = h5::attr_reim(ds, "polarization");     // real or complex
        if (p.size() != 4) throw std::runtime_error(where + "polarization must have 2 entries");
        px = cd(p[0], p[1]); py = cd(p[2], p[3]);
    }
    const double pn = std::sqrt(std::norm(px) + std::norm(py));
    if (!(pn > 0)) throw std::runtime_error(where + "polarization is zero");
    px /= pn; py /= pn;
    const std::string field = h5::has_attr(ds, "envelopeField") ? h5::attr_string(ds, "envelopeField") : "normalized_vector_potential";
    const bool vecpot = field == "normalized_vector_potential";
    if (!vecpot && field != "electric_field") throw std::runtime_error(where + "envelopeField " + field);
    const std::vector<hsize_t> dims = h5::dims(ds);
    if (dims.size() != 3) throw std::runtime_error(where + "the envelope must have 3 dimensions");
    std::vector<double> raw(2 * size_t(dims[0]) * dims[1] * dims[2]);
    h5::read_complex(ds, raw.data(), {0, 0, 0}, dims);
    auto R = [&](size_t q) { return cd(raw[2 * q], raw[2 * q + 1]); };
    const int at = axis("t");
    Grid g;
    const size_t dt_ax = cart ? size_t(at) : size_t(at) + 1;    // dimension index of t
    g.nu = int(dims[dt_ax]);
    g.du = sp[size_t(at)] * gu * w0;
    g.u0 = (off[size_t(at)] + pos[size_t(at)] * sp[size_t(at)]) * gu * w0;
    if (g.nu < 4 || !(g.du > 0)) throw std::runtime_error(where + "the time axis needs >= 4 points");
    if (cart) {
        const int ax = axis("x"), ay = axis("y");                 // LASY x -> our y, LASY y -> our z
        const size_t st[3] = {size_t(dims[1] * dims[2]), size_t(dims[2]), 1};
        g.ny = int(dims[size_t(ax)]); g.nz = int(dims[size_t(ay)]);
        g.dy = sp[size_t(ax)] * gu * k0; g.dz = sp[size_t(ay)] * gu * k0;
        g.y0 = (off[size_t(ax)] + pos[size_t(ax)] * sp[size_t(ax)]) * gu * k0;
        g.z0 = (off[size_t(ay)] + pos[size_t(ay)] * sp[size_t(ay)]) * gu * k0;
        g.uy = g.ny == 1; g.uz = g.nz == 1;
        g.e.resize(size_t(g.nu) * g.ny * g.nz);
        for (int l = 0; l < g.nu; ++l)
            for (int j = 0; j < g.ny; ++j)
                for (int k = 0; k < g.nz; ++k) g.at(l, j, k) = R(l * st[at] + j * st[ax] + k * st[ay]);
        info.push_back(fmt("cartesian file: %.0f x %.0f x %.0f (t, x, y)", g.nu, g.ny, g.nz));
    } else {
        // thetaMode [comp][t][r]: env = C_0 + sum_m C_m cos(m theta) + S_m sin(m theta), resampled on the
        // (Y, Z) plane with the radial spacing (theta from LASY's x towards y, i.e. from our y towards z)
        const int ar = axis("r");
        const int nc = int(dims[0]), nr = int(dims[1 + size_t(ar)]), K = (nc - 1) / 2;
        if (nc % 2 != 1) throw std::runtime_error(where + "thetaMode: an odd number of components (2 m + 1)");
        const double dr = sp[size_t(ar)] * gu * k0, r0 = (off[size_t(ar)] + pos[size_t(ar)] * sp[size_t(ar)]) * gu * k0;
        const size_t st[2] = {size_t(dims[2]), 1};
        auto C = [&](int comp, int l, int j) { return R(size_t(comp) * dims[1] * dims[2] + l * st[at] + j * st[ar]); };
        if (modes) {   // RZ: keep the modes, a_0 = C_0, a_{+-m} = (C_m +- i S_m) / 2 (env = sum a_k e^{-i k theta})
            Modes& M = *modes;
            M.nu = g.nu; M.nr = nr; M.K = K; M.u0 = g.u0; M.du = g.du; M.r0 = r0; M.dr = dr;
            M.a.assign(size_t(2 * K + 1) * M.nu * nr, cd(0));
            const double s = usi / (vecpot ? 1.0 : E0);
            for (int l = 0; l < M.nu; ++l)
                for (int j = 0; j < nr; ++j) {
                    M.at(0, l, j) = s * C(0, l, j);
                    for (int m = 1; m <= K; ++m) {
                        const cd Cm = s * C(2 * m - 1, l, j), Sm = s * C(2 * m, l, j);
                        M.at(m, l, j) = 0.5 * (Cm + cd(0, 1) * Sm);
                        M.at(-m, l, j) = 0.5 * (Cm - cd(0, 1) * Sm);
                    }
                }
            if (vecpot) {
                std::vector<cd> col(static_cast<size_t>(M.nu));
                for (int k = -K; k <= K; ++k)
                    for (int j = 0; j < nr; ++j) {
                        for (int l = 0; l < M.nu; ++l) col[size_t(l)] = M.at(k, l, j);
                        for (int l = 0; l < M.nu; ++l) {
                            const int la = std::max(0, l - 1), lb = std::min(M.nu - 1, l + 1);
                            M.at(k, l, j) = cd(0, omegaL) * col[size_t(l)] - (col[size_t(lb)] - col[size_t(la)]) / ((lb - la) * M.du);
                        }
                    }
            }
            info.push_back(fmt("thetaMode file: %.0f modes, %.0f x %.0f (t, r), kept as azimuthal modes", K + 1, M.nu, nr));
            info.push_back(fmt("omega_L %.6g omega0, ", omegaL) + "polarization (" + fmt("%.3g%+.3gi, ", px.real(), px.imag()) +
                           fmt("%.3g%+.3gi)", py.real(), py.imag()) + ", " + field);
            return Grid{};
        }
        g.ny = g.nz = 2 * nr - 1;
        g.dy = g.dz = dr; g.y0 = g.z0 = -(nr - 1) * dr;
        g.e.assign(size_t(g.nu) * g.ny * g.nz, cd(0));
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j) {
                const double Y = g.y0 + j * dr, Z = g.z0 + k * dr, r = std::hypot(Y, Z), th = std::atan2(Z, Y);
                double fr = (r - r0) / dr;
                if (fr > nr - 1) continue;
                fr = std::max(fr, 0.0);
                const int a = std::min(int(fr), nr - 2);
                const double w = fr - a;
                for (int l = 0; l < g.nu; ++l) {
                    auto lin = [&](int comp) { return (1 - w) * C(comp, l, a) + w * C(comp, l, a + 1); };
                    cd v = lin(0);
                    for (int m = 1; m <= K; ++m) v += lin(2 * m - 1) * std::cos(m * th) + lin(2 * m) * std::sin(m * th);
                    g.at(l, j, k) = v;
                }
            }
        info.push_back(fmt("thetaMode file: %.0f modes, %.0f x %.0f (t, r), resampled on the (y, z) plane", K + 1, g.nu, nr));
    }
    // units: E [V/m] -> m c omega0 / e; a vector potential a: E = i omega_L a - da/dt
    for (cd& v : g.e) v *= usi / (vecpot ? 1.0 : E0);
    if (vecpot) {
        std::vector<cd> col(static_cast<size_t>(g.nu));
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j) {
                for (int l = 0; l < g.nu; ++l) col[size_t(l)] = g.at(l, j, k);
                for (int l = 0; l < g.nu; ++l) {
                    const int la = std::max(0, l - 1), lb = std::min(g.nu - 1, l + 1);
                    g.at(l, j, k) = cd(0, omegaL) * col[size_t(l)] - (col[size_t(lb)] - col[size_t(la)]) / ((lb - la) * g.du);
                }
            }
    }
    info.push_back(fmt("omega_L %.6g omega0, ", omegaL) + "polarization (" + fmt("%.3g%+.3gi, ", px.real(), px.imag()) +
                   fmt("%.3g%+.3gi)", py.real(), py.imag()) + ", " + field);
    return g;
}

namespace {
// Exact propagation over the distance L along x in vacuum at fixed retarded time: every plane wave
// (omega = omega_L - Omega, k_y, k_z) gets e^{i (k_x - omega) L}, k_x = (omega^2 - k_perp^2)^{1/2};
// evanescent and backward parts are dropped. The grid is padded to powers of two (twice the time
// window, twice the transverse extent) and keeps the padded size.
void propagate(Grid& g, double omegaL, double L, std::vector<std::string>& info) {
    const int NU = int(pow2(size_t(2 * g.nu))), NY = g.uy ? 1 : int(pow2(size_t(2 * g.ny))), NZ = g.uz ? 1 : int(pow2(size_t(2 * g.nz)));
    Grid p = g;
    p.nu = NU; p.ny = NY; p.nz = NZ;
    p.u0 = g.u0 - (NU - g.nu) / 2 * g.du;
    if (!g.uy) p.y0 = g.y0 - (NY - g.ny) / 2 * g.dy;
    if (!g.uz) p.z0 = g.z0 - (NZ - g.nz) / 2 * g.dz;
    p.e.assign(size_t(NU) * NY * NZ, cd(0));
    const int su = (NU - g.nu) / 2, sy = (NY - g.ny) / 2, sz = (NZ - g.nz) / 2;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int l = 0; l < g.nu; ++l) p.at(l + su, j + sy, k + sz) = g.at(l, j, k);
    // forward transform along every axis: a_q = sum_l a_l e^{-2 pi i q l / N}
    auto transform = [&](int sign) {
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j) fft(&p.at(0, j, k), size_t(NU), 1, sign);
        if (NY > 1)
            for (int k = 0; k < NZ; ++k)
                for (int l = 0; l < NU; ++l) fft(&p.at(l, 0, k), size_t(NY), NU, sign);
        if (NZ > 1)
            for (int j = 0; j < NY; ++j)
                for (int l = 0; l < NU; ++l) fft(&p.at(l, j, 0), size_t(NZ), long(NU) * NY, sign);
    };
    transform(-1);
    // env(u) = sum_q F_q e^{+i Omega_q (u - u0)}: the carrier e^{-i omega_L u} makes omega = omega_L - Omega;
    // env(y) = sum F e^{+i k_y (y - y0)} as a plane wave e^{i k_y y}
    auto freq = [](int q, int N, double h) { return 2 * M_PI * (q < N / 2 ? q : q - N) / (N * h); };
    double kept = 0, all = 0;
    for (int k = 0; k < NZ; ++k)
        for (int j = 0; j < NY; ++j)
            for (int l = 0; l < NU; ++l) {
                const double om = omegaL - freq(l, NU, p.du), ky = NY > 1 ? freq(j, NY, p.dy) : 0, kz = NZ > 1 ? freq(k, NZ, p.dz) : 0;
                const double kx2 = om * om - ky * ky - kz * kz;
                cd& v = p.at(l, j, k);
                all += std::norm(v);
                if (om <= 0 || kx2 <= 0) { v = 0; continue; }
                kept += std::norm(v);
                v *= std::polar(1.0, (std::sqrt(kx2) - om) * L);
            }
    transform(+1);
    const double n = double(NU) * NY * NZ;
    for (cd& v : p.e) v /= n;
    g = std::move(p);
    info.push_back(fmt("propagated over %.6g (1/k0) to the entrance plane, exactly (grid %.0f x %.0f", L, NU, NY) +
                   fmt(" x %.0f), %.2g of the energy evanescent or backward, dropped", NZ, all > 0 ? 1 - kept / all : 0.0));
}
}  // namespace

FileLaser::FileLaser(const LaserFileConfig& c, double lambda0, double y_lo, double y_hi, double z_lo, double z_hi) {
    double omegaL;
    cd px, py;
    Grid g = lasy::read_file(c, lambda0, omegaL, px, py, info);
    if (c.flat && !g.uz)
        throw std::runtime_error(c.path + ": geometry = cartesian2d needs a slab file (cartesian, LASY y axis of length 1)");
    // peak, intensity centroid in time
    double peak = 0, w = 0, wt = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int l = 0; l < g.nu; ++l) {
                const double a = std::norm(g.at(l, j, k));
                peak = std::max(peak, std::sqrt(a)); w += a; wt += a * (g.u0 + l * g.du);
            }
    if (!(peak > 0)) throw std::runtime_error(c.path + ": the envelope is zero");
    const double tc = c.t_center < 1e299 ? c.t_center : wt / w;
    const double first = g.u0, last = g.u0 + (g.nu - 1) * g.du;
    const double delay = c.delay >= 0 ? c.delay : tc - first;
    if (c.a0 > 0) {
        const double s = c.a0 * omegaL / peak;
        for (cd& v : g.e) v *= s;
        peak = c.a0 * omegaL;
    }
    info.push_back(fmt("peak |E| %.4g (a0 %.4g), t_center %.6g in the file's time", peak, peak / omegaL, tc) +
                   fmt(", at the entrance plane at t = %.6g (1/omega0)", delay));
    if (c.file_x != 0) propagate(g, omegaL, -c.file_x, info);
    // the edge of the grid at x = 0 (wrap-around of the FFT, or a pulse cut by the file)
    double edge = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int l = 0; l < g.nu; ++l)
                if (l == 0 || l == g.nu - 1 || (!g.uy && (j == 0 || j == g.ny - 1)) || (!g.uz && (k == 0 || k == g.nz - 1)))
                    edge = std::max(edge, std::abs(g.at(l, j, k)));
    this->edge = edge / peak;
    t_first = delay + first - tc;
    t_last = delay + last - tc;
    // only the part of the grid this rank's cells need (one node more on each side)
    auto range = [](double lo, double hi, double x0, double h, int n, bool uni, int& a, int& b) {
        if (uni) { a = 0; b = 1; return; }
        a = std::max(0, int(std::floor((lo - x0) / h)) - 1);
        b = std::min(n, int(std::ceil((hi - x0) / h)) + 2);
        if (b <= a) { a = 0; b = std::min(n, 2); }
    };
    int ja, jb, ka, kb;
    range(y_lo - c.yc, y_hi - c.yc, g.y0, g.dy, g.ny, g.uy, ja, jb);
    range(z_lo - c.zc, z_hi - c.zc, g.z0, g.dz, g.nz, g.uz, ka, kb);
    std::vector<double> h(2 * size_t(g.nu) * (jb - ja) * (kb - ka));
    for (int k = ka; k < kb; ++k)
        for (int j = ja; j < jb; ++j)
            for (int l = 0; l < g.nu; ++l) {
                const size_t q = 2 * ((size_t(k - ka) * (jb - ja) + (j - ja)) * g.nu + l);
                h[q] = g.at(l, j, k).real(); h[q + 1] = g.at(l, j, k).imag();
            }
    env_.resize(h.size());
    env_.from_host(h.data(), 0, h.size());
    host_ = std::move(h);
    FileLaserView& v = v_;
    v.nu = g.nu; v.ny = jb - ja; v.nz = kb - ka;
    v.u0 = g.u0; v.du = g.du;
    v.y0 = c.yc + g.y0 + ja * g.dy; v.dy = g.dy; v.z0 = c.zc + g.z0 + ka * g.dz; v.dz = g.dz;
    v.uniform_y = g.uy; v.uniform_z = g.uz;
    v.omega = omegaL; v.u_shift = tc - delay;
    v.pyr = px.real(); v.pyi = px.imag(); v.pzr = py.real(); v.pzi = py.imag();
    info.push_back(fmt("enters x = 0 from t = %.6g to %.6g (1/omega0); envelope at the grid edge %.2g of the peak",
                       t_first, t_last, this->edge));
}
#endif
