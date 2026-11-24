#include "tfsf_modes.hpp"
#include <cmath>
#include <complex>
#include <map>
#include <stdexcept>
#include <string>

namespace {
using cd = std::complex<double>;

// the outputs of one evaluation: quantity q goes to f[q] at local x index c[q], set (add = 0) or added (add = 1)
struct Outs { FV f[16]; int c[16]; int add; };

// P[q][a][kk] = sum_b C[q][a][b] tz[b][kk] (complex pairs), q < nq, kk < nzg
void modes_pass_z(const double* C, const double* TZ, double* P, int nq, int na, int nb, int nzg) {
    par_for(long(nq) * na * nzg, [=] HD(long x) {
        const int kk = int(x % nzg), a = int((x / nzg) % na), q = int(x / (long(nzg) * na));
        double re = 0, im = 0;
        const double* c = C + 2 * (long(q) * na + a) * nb;
        for (int b = 0; b < nb; ++b) {
            const double cr = c[2 * b], ci = c[2 * b + 1], tr = TZ[2 * (long(b) * nzg + kk)], ti = TZ[2 * (long(b) * nzg + kk) + 1];
            re += cr * tr - ci * ti; im += cr * ti + ci * tr;
        }
        P[2 * x] = re; P[2 * x + 1] = im;
    });
}

// quantity q at (c[q], jj - 1, kk - gz): Re sum_a ty[a][jj] P[q][a][kk] (gz = 1, 0 on a flat 2D domain: one plane)
void modes_pass_y(const double* P, const double* TY, int nq, int na, int nyg, int nzg, int gz, const Outs o) {
    par_for(long(nq) * nyg * nzg, [=] HD(long x) {
        const int kk = int(x % nzg), jj = int((x / nzg) % nyg), q = int(x / (long(nzg) * nyg));
        double v = 0;
        for (int a = 0; a < na; ++a) {
            const long p = 2 * ((long(q) * na + a) * nzg + kk), t = 2 * (long(a) * nyg + jj);
            v += TY[t] * P[p] - TY[t + 1] * P[p + 1];
        }
        double& F = o.f[q](o.c[q], jj - 1, kk - gz);
        F = o.add ? F + v : v;
    });
}
}  // namespace

TfsfModes::TfsfModes(const Domain& d, const std::vector<LaserConfig>& lasers, int is, double t_span)
    : d_(d), ds_(d), is_(is) {
    irip_ = d.solver == Solver::IRip;
    o5_ = irip_ && d.o5;
    ds_.n[0] = 2;
    for (Field& f : f_) f.resize(ds_);
    const double D = d.h[0], Ly = d.N[1] * d.h[1], Lz = d.N[2] * d.h[2], dky = 2 * M_PI / Ly, dkz = 2 * M_PI / Lz;
    const double xp = is * D, sqpi = std::sqrt(M_PI);
    std::map<long, int> ia, ib;                        // lattice index of k_y, k_z -> group
    std::vector<long> my_of, mz_of;
    auto group = [](std::map<long, int>& g, std::vector<long>& of, long m) {
        auto it = g.find(m);
        if (it != g.end()) return it->second;
        const int n = int(of.size());
        g[m] = n; of.push_back(m);
        return n;
    };
    for (const LaserConfig& L : lasers) {
        if (!L.enabled || L.tfsf_x < 0) continue;
        if (L.p != 0 || L.l != 0) throw std::runtime_error("laser.tfsf_incident = modes: Gaussian beams only (no mode_p, mode_l)");
        peak_ = std::max(peak_, L.a0);
        const double ca = std::cos(L.angle), sa = std::sin(L.angle), f = L.xf;
        const double rf[3] = {xp + f * ca, L.yc + f * sa, L.zc}, tf = L.tc + f;
        const double al = 2 * std::log(2.0) / (L.tau * L.tau);       // field envelope exp(-al t^2) (LaserConfig)
        // |nu| up to the 1e-12 point of the spectrum; the sum is periodic in time with T, longer than the run sees
        const double numax = 10.6 * std::sqrt(al), T = 2 * (t_span + 12 / std::sqrt(2 * al)), dnu = 2 * M_PI / T;
        const int M = int(std::ceil(numax / dnu));
        const cd amp0 = cd(0, 1) * L.a0 * std::exp(cd(0, -L.phase));
        const double qy = L.w0 > 0 ? 10.6 / L.w0 : 0, qz = L.w0z > 0 ? 10.6 / L.w0z : 0;
        for (int m = -M; m <= M; ++m) {
            const double nu = m * dnu, w = 1 + nu, rt = std::exp(-nu * nu / (4 * al));
            if (w < 0.05) {                            // near zero frequency: must carry nothing
                if (rt > 1e-6) throw std::runtime_error("laser.duration: too short for tfsf_incident = modes (spectrum reaches omega = 0)");
                continue;
            }
            const double at = dnu / (2 * M_PI) * std::sqrt(M_PI / al) * rt;
            long y0, y1, z0 = 0, z1 = 0;
            if (L.w0 > 0) { y0 = long(std::floor((w * sa - qy) / dky)) - 2; y1 = long(std::ceil((w * sa + qy) / dky)) + 2; }
            else y0 = y1 = std::lround(sa / dky);                     // the plane wave of one k_y = sin a (periodic y)
            if (L.w0z > 0) { z0 = long(std::floor(-qz / dkz)) - 2; z1 = -z0; }
            for (long my = y0; my <= y1; ++my)
                for (long mz = z0; mz <= z1; ++mz) {
                    const double ky = my * dky, kz = mz * dkz, k2 = w * w - ky * ky - kz * kz;
                    if (k2 <= 0) continue;
                    const double kxc = std::sqrt(k2), qp = -kxc * sa + ky * ca;
                    const double sy = L.w0 > 0 ? std::exp(-0.25 * qp * qp * L.w0 * L.w0) : 1.0;
                    const double sz = L.w0z > 0 ? std::exp(-0.25 * kz * kz * L.w0z * L.w0z) : 1.0;
                    const double rel = rt * sy * sz;
                    if (rel < 1e-12) continue;
                    const double J = std::abs(ca + ky / kxc * sa);
                    const cd A = amp0 * at * (L.w0 > 0 ? sqpi * L.w0 * J / Ly * sy : 1.0) * (L.w0z > 0 ? sqpi * L.w0z / Lz * sz : 1.0);
                    // k_x of the discrete dispersion (increasing in k_x for a stable step)
                    double k[3] = {0, ky, kz};
                    if (plane_omega(d, k) >= w) {             // evanescent along x at this k_perp
                        if (rel > 1e-6) throw std::runtime_error("laser.tfsf_incident = modes: the beam spectrum reaches the "
                                                                 "discrete cutoff (finer dy, dz or a wider waist)");
                        continue;
                    }
                    double lo = 0, hi = M_PI / D * (1 - 1e-12);
                    k[0] = hi;
                    if (plane_omega(d, k) <= w) {
                        if (rel > 1e-6) throw std::runtime_error("laser.tfsf_incident = modes: the pulse spectrum reaches beyond the grid");
                        continue;
                    }
                    for (int it = 0; it < 200 && hi - lo > 1e-15 * hi; ++it) {
                        k[0] = 0.5 * (lo + hi);
                        (plane_omega(d, k) < w ? lo : hi) = k[0];
                    }
                    k[0] = 0.5 * (lo + hi);
                    // polarisation 'y': E along z x k (in the x-y plane), 'z': along k x (z x k) (as side.cpp)
                    const double kk = std::sqrt(k[0] * k[0] + ky * ky + kz * kz), u[3] = {k[0] / kk, ky / kk, kz / kk};
                    double es[3] = {-u[1], u[0], 0};
                    const double nes = std::hypot(es[0], es[1]);
                    es[0] /= nes; es[1] /= nes;
                    const double ep[3] = {u[1] * es[2] - u[2] * es[1], u[2] * es[0] - u[0] * es[2], u[0] * es[1] - u[1] * es[0]};
                    const double* e = L.pol == 'y' ? es : ep;
                    const cd E[3] = {A * e[0], A * e[1], A * e[2]};
                    comps_.push_back(plane_mode(d, k, w, E, rf, tf, mism_));
                    ga_.push_back(group(ia, my_of, my));
                    gb_.push_back(group(ib, mz_of, mz));
                    if (comps_.size() > 5000000) throw std::runtime_error("laser.tfsf_incident = modes: more than 5e6 plane waves");
                }
        }
    }
    na_ = int(my_of.size()); nb_ = int(mz_of.size());
    if (comps_.empty()) throw std::runtime_error("laser.tfsf_incident = modes: no plane wave");
    // e^{i ky hy j}, e^{i kz hz k} at the absolute transverse indices of the slab (one guard on each side)
    const int gz = d.flat ? 0 : 1, nyg = d.n[1] + 2, nzg = d.n[2] + 2 * gz;
    std::vector<double> ty(2 * size_t(na_) * nyg), tz(2 * size_t(nb_) * nzg);
    for (int a = 0; a < na_; ++a)
        for (int j = 0; j < nyg; ++j) {
            const double ph = my_of[a] * dky * double(d.off[1] + j - 1) * d.h[1];
            ty[2 * (size_t(a) * nyg + j)] = std::cos(ph); ty[2 * (size_t(a) * nyg + j) + 1] = std::sin(ph);
        }
    for (int b = 0; b < nb_; ++b)
        for (int k = 0; k < nzg; ++k) {
            const double ph = mz_of[b] * dkz * double(d.off[2] + k - gz) * d.h[2];
            tz[2 * (size_t(b) * nzg + k)] = std::cos(ph); tz[2 * (size_t(b) * nzg + k) + 1] = std::sin(ph);
        }
    ty_.upload(ty); tz_.upload(tz);
    c_.assign(2 * size_t(16) * na_ * nb_, 0.0);
    cd_.resize(c_.size());
    p_.resize(2 * size_t(16) * na_ * nzg);
}

// nq quantities at time t: quantity q of the global x index ig[q], selector sel[q] (0..5: Ex .. Bz of the mode, 6, 7: Ex*,
// Bx* of the solver, tfsf_modes.hpp), into o
void TfsfModes::evaluate(double t, int nq, const long* ig, const int* sel, const void* outs) {
    std::fill(c_.begin(), c_.begin() + 2 * size_t(nq) * na_ * nb_, 0.0);
    const long nab = long(na_) * nb_;
    for (size_t m = 0; m < comps_.size(); ++m) {
        const SideTfsf::Comp& p = comps_[m];
        const cd et = std::exp(cd(0, -p.w * t));
        auto V = [&](int q) { return cd(p.v[2 * q], p.v[2 * q + 1]); };
        const cd mu(p.mu[0], p.mu[1]);
        const long g = long(ga_[m]) * nb_ + gb_[m];
        for (int q = 0; q < nq; ++q) {
            const int s = sel[q];
            // 6, 7: Ex*, Bx* of the solver (iRIP the mean; o5: the closed mean, in pr); 8, 9 (o5): g_perp times the mean
            const cd val = s < 6 ? V(s)
                           : s >= 8 ? o5_factor(d_, p.k, false) * mu * V(s == 8 ? 0 : 3)
                           : (irip_ && !o5_) ? mu * V(s == 6 ? 0 : 3) : (s == 6 ? cd(p.pr[0], p.pr[1]) : cd(p.pr[2], p.pr[3]));
            const cd x = val * et * std::exp(cd(0, p.k[0] * d_.h[0] * double(ig[q])));
            double* o = &c_[2 * (q * nab + g)];
            o[0] += x.real(); o[1] += x.imag();
        }
    }
    cd_.from_host(c_.data(), 0, 2 * size_t(nq) * nab);
    const int gz = d_.flat ? 0 : 1, nyg = d_.n[1] + 2, nzg = d_.n[2] + 2 * gz;
    modes_pass_z(cd_.data(), tz_.data(), p_.data(), nq, na_, nb_, nzg);
    modes_pass_y(p_.data(), ty_.data(), nq, na_, nyg, nzg, gz, *static_cast<const Outs*>(outs));
}

void TfsfModes::begin(double t) {
    long ig[16];
    int sel[16];
    Outs o{};
    const int nf = o5_ ? 8 : 6;
    const int fs[8] = {1, 2, 4, 5, 6, 7, 8, 9};        // Ey, Ez, By, Bz, Ex*, Bx* (o5: and g_perp of the means)
    for (int c = 0; c < 2; ++c)
        for (int q = 0; q < nf; ++q) {
            ig[c * nf + q] = is_ - 1 + c; sel[c * nf + q] = fs[q];
            o.f[c * nf + q] = f_[q].view(); o.c[c * nf + q] = c;
        }
    o.add = 0;
    evaluate(t, 2 * nf, ig, sel, &o);
}

// t = 0: the incident field added to the owned total-field cells (x index >= is) of the main grid, slice by slice
void TfsfModes::add_incident(const FV f[6], double t) {
    const int s = is_ - d_.off[0];
    for (int i = s < 0 ? 0 : s; i < d_.n[0]; ++i) {
        long ig[6];
        int sel[6];
        Outs o{};
        for (int q = 0; q < 6; ++q) { ig[q] = d_.off[0] + i; sel[q] = q; o.f[q] = f[q]; o.c[q] = i; }
        o.add = 1;
        evaluate(t, 6, ig, sel, &o);
    }
}

double TfsfModes::plane_max(double t) {
    begin(t);
    double m = 0;
    for (int q = 0; q < 4; ++q) {
        const std::vector<double> h = f_[q].host();
        for (double v : h) m = std::max(m, std::abs(v));
    }
    return m;
}
