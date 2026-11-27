// RZ fields per azimuthal mode (docs/specs/rz_modes.md): RIP-CC and iRIP steps in vacuum.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include "common.hpp"
#include "rz.hpp"
#include "species.hpp"
#include "particles.hpp"
#include "sim.hpp"
#include "laser.hpp"
#include "openpmd.hpp"
#ifdef RIP_HDF5
#include "h5.hpp"
#endif
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
const double j01 = 2.404825557695773, j11p = 1.841183781340659;   // first zero of J0, of J1'

// Standing guided mode in a conducting cylinder of radius R = nr dr, periodic z with kz = 2 pi nk / Lz:
// TM01 (m = 0, Ez = J0(kp r) cos(kz z)) or TE11 (m = 1, Bz = J1(kp r) cos(kz z)). Returns w dt from the
// recurrence of the increments d^n = F^{n+1} - F^n: d^{n+1} + d^{n-1} = 2 cos(w dt) d^n (least squares),
// and the relative residual of that recurrence (a single mode: small).
double guided(Comm& comm, int nr, double dz, double dr, Solver sv, bool te, double& resid) {
    const int M = te ? 2 : 1, nz = 32, nk = 2;
    Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, sv);
    FieldsRZ f(d, M);
    const double R = nr * dr, kp = (te ? j11p : j01) / R, kz = 2 * M_PI * nk / (nz * dz);
    Field& F = te ? f.bz : f.ez;
    std::vector<double> h = F.host();
    for (int j = 0; j < nr; ++j)
        for (int i = 0; i < nz; ++i) {
            const double r = (j + (te ? 0.5 : 0.0)) * dr, z = (d.off[0] + i) * dz;
            const double v = (te ? std::cyl_bessel_j(1, kp * r) : std::cyl_bessel_j(0, kp * r)) * std::cos(kz * z);
            h[F.idx(i, j, te ? 1 : 0)] = v;                              // TE: Re of mode 1 (slot 1)
        }
    F.upload(h);
    comm.exchange(d, {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp});
    std::vector<std::vector<double>> hist;
    for (int n = 0; n < 60; ++n) { hist.push_back(F.host()); f.zero_currents(); f.step(comm); }
    double num = 0, den = 0;
    for (size_t n = 1; n + 2 < hist.size(); ++n)
        for (size_t q = 0; q < hist[n].size(); ++q) {
            const double dm = hist[n][q] - hist[n - 1][q], d0 = hist[n + 1][q] - hist[n][q], dp = hist[n + 2][q] - hist[n + 1][q];
            num += (dp + dm) * d0; den += 2 * d0 * d0;
        }
    const double c = num / den;
    double res = 0, sc = 0;
    for (size_t n = 1; n + 2 < hist.size(); ++n)
        for (size_t q = 0; q < hist[n].size(); ++q) {
            const double dm = hist[n][q] - hist[n - 1][q], d0 = hist[n + 1][q] - hist[n][q], dp = hist[n + 2][q] - hist[n + 1][q];
            res = std::max(res, std::abs(dp + dm - 2 * c * d0)); sc = std::max(sc, std::abs(d0));
        }
    resid = res / sc;
    return std::acos(c);
}

// R1. iRIP: TM01 and TE11 at dz/dr = 2 against the continuum k_perp through the iRIP law
//     tan^2(w dt/2) = tan^2(kz dz/2) + dz^2 kp^2/4; the error is the radial discretisation, at least
//     second order: it falls by more than 3 when dr halves (TM01: 11, TE11: 5).
void rz_guided_irip(Comm& comm) {
    for (int te = 0; te < 2; ++te) {
        double e[2], res = 0;
        for (int r = 0; r < 2; ++r) {
            const int nr = r ? 32 : 16;
            const double R = 3.0, dr = R / nr, dz = 2 * dr, kz = 2 * M_PI * 2 / (32 * dz);
            const double kp = (te ? j11p : j01) / R;
            const double tx = std::tan(kz * dz / 2), wth = 2 * std::atan(std::sqrt(tx * tx + dz * dz * kp * kp / 4));
            double rs;
            const double w = guided(comm, nr, dz, dr, Solver::IRip, te, rs);
            e[r] = std::abs(w / wth - 1); res = std::max(res, rs);
        }
        check(std::string("rz irip: ") + (te ? "TE11 (m = 1)" : "TM01 (m = 0)") +
                  " guided mode, frequency against the Bessel root, second order in dr",
              e[1] < 3e-4 && e[0] / e[1] > 3 && res < 0.05,
              fmt("rel. error nr = 16: %.2e", e[0]) + fmt(", nr = 32: %.2e", e[1]) + fmt(", recurrence residual %.1e", res));
    }
}

// Random fields in modes 0, 1, 2 (the non-dynamic axis values zeroed): energy, div E and div B per mode.
void randomize(FieldsRZ& f, Comm& comm, unsigned seed) {
    const Domain& d = f.dom();
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g;
    for (Field* F : {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp}) {
        std::vector<double> h = F->host();
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    double v = g(rng);
                    if (j == 0 && (F == &f.ep || F == &f.br || (F == &f.ez && k > 0))) v = 0;
                    h[F->idx(i, j, k)] = v;
                }
        F->upload(h);
    }
    comm.exchange(d, {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp});
}
double max_change(const Field& a, const std::vector<double>& b0, const Domain& d) {
    const std::vector<double> b = a.host();
    double e = 0;
    for (int k = 0; k < d.n[2]; ++k)
        for (int j = 0; j < d.n[1] - 1; ++j)                               // the last half node touches the wall
            for (int i = 0; i < d.n[0]; ++i) e = std::max(e, std::abs(b[a.idx(i, j, k)] - b0[a.idx(i, j, k)]));
    return e;
}

// R2. iRIP, random fields, dz/dr = 4 (far beyond the explicit limit), 300 steps: E^2 + B^2 (area-weighted,
//     modes |F_0|^2 + 2 |F_m|^2) conserved; div E and div B per mode unchanged.
void rz_irip_random(Comm& comm) {
    Domain d = rz_domain(comm, 16, 20, 3, 0.4, 0.1, true, Solver::IRip);
    FieldsRZ f(d, 3);
    randomize(f, comm, 21);
    Field de(d), db(d);
    f.div_e(de); f.div_b(db);
    const std::vector<double> e0 = de.host(), b0 = db.host();
    const double w0 = f.energy();
    double dev = 0;
    for (int n = 0; n < 300; ++n) { f.zero_currents(); f.step(comm); dev = std::max(dev, std::abs(f.energy() / w0 - 1)); }
    f.div_e(de); f.div_b(db);
    double ge = 0, gb = 0, se = 0, sb = 0;
    for (double v : e0) se = std::max(se, std::abs(v));
    for (double v : b0) sb = std::max(sb, std::abs(v));
    ge = max_change(de, e0, d) / se; gb = max_change(db, b0, d) / sb;
    check("rz irip: random fields in modes 0..2 at dz/dr = 4, energy conserved, div E and div B per mode unchanged",
          dev < 1e-11 && ge < 1e-12 && gb < 1e-12, fmt("max |W/W0 - 1| %.1e", dev) + fmt(", div E change %.1e", ge) + fmt(", div B %.1e", gb));
}

// The increments recurrence of guided(): w dt from d^{n+1} + d^{n-1} = 2 cos(w dt) d^n over a history of one field,
// and the relative residual.
double recurrence(const std::vector<std::vector<double>>& hist, double& resid) {
    double num = 0, den = 0;
    for (size_t n = 1; n + 2 < hist.size(); ++n)
        for (size_t q = 0; q < hist[n].size(); ++q) {
            const double dm = hist[n][q] - hist[n - 1][q], d0 = hist[n + 1][q] - hist[n][q], dp = hist[n + 2][q] - hist[n + 1][q];
            num += (dp + dm) * d0; den += 2 * d0 * d0;
        }
    const double c = num / den;
    double res = 0, sc = 0;
    for (size_t n = 1; n + 2 < hist.size(); ++n)
        for (size_t q = 0; q < hist[n].size(); ++q) {
            const double dm = hist[n][q] - hist[n - 1][q], d0 = hist[n + 1][q] - hist[n][q], dp = hist[n + 2][q] - hist[n + 1][q];
            res = std::max(res, std::abs(dp + dm - 2 * c * d0)); sc = std::max(sc, std::abs(d0));
        }
    resid = res / sc;
    return std::acos(c);
}

// R-O5. iRIP with closure o5 (decision 74): (a) a discrete radial eigenmode a of L_m times cos(kz z) (uniform and
//     stretched radial grid; TM m = 0 on Ez, TE m = 1 on Bz) has the frequency of the corrected relation per mode,
//     tan^2(w dt/2) = tan^2(kz dz/2) + dz^2 lam_a / 4 (1 + 4/3 sin^2(kz dz/2)) g_a, g_a = 1 + (2/3) min(dz^2 lam_a / 4, 1)
//     + h_a^2 lam_a / 12 (the cap: decision 77);
//     (b) the Bessel guided modes TM01, TE11 at dz = 2 dr against the continuum c^2 (kz^2 + kp^2): o5 against the box
//     closure; (c) random fields in modes 0..2 at dz/dr = 4 and 10 (also on the stretched grid) over 300 steps below
//     2.15^2 = 4.62 E_0 (the bound of the capped closure in Cartesian geometry, decision 77), div E and div B per mode unchanged.
void rz_o5(Comm& comm) {
    double worst = 0, wl = 0, pl = 0;
    for (int st = 0; st < 2; ++st)
        for (int te = 0; te < 2; ++te)
            for (int a : {2, 6}) {
                const int nz = 32, nr = 24, M = te ? 2 : 1, m = te ? 1 : 0, nk = 3;
                const double dr = 0.1, dz = 0.25;
                RadialGrid g = st ? rz_radial_grid(nr, dr, 8 * dr, 1.08) : rz_radial_grid(nr, dr);
                Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, Solver::IRip);
                FieldsRZ f(d, M, std::move(g));
                f.o5 = true;
                std::vector<double> V, Vi, lam, h2;
                rz_basis(f.grid(), m, te, V, Vi, lam, h2);
                std::vector<int> ord(static_cast<size_t>(nr));
                for (int q = 0; q < nr; ++q) ord[size_t(q)] = q;
                std::sort(ord.begin(), ord.end(), [&](int x, int y) { return lam[size_t(x)] < lam[size_t(y)]; });
                const int q = ord[size_t(a + (te ? 0 : 0))];
                Field& F = te ? f.bz : f.ez;
                std::vector<double> h = F.host();
                const double kz = 2 * M_PI * nk / (nz * dz);
                for (int j = 0; j < nr; ++j)
                    for (int i = 0; i < nz; ++i) h[F.idx(i, j, te ? 1 : 0)] = V[size_t(j) * nr + q] * std::cos(kz * (d.off[0] + i) * dz + 0.3);
                F.upload(h);
                comm.exchange(d, {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp});
                std::vector<std::vector<double>> hist;
                for (int n = 0; n < 40; ++n) { hist.push_back(F.host()); f.zero_currents(); f.step(comm); }
                double res;
                const double w = recurrence(hist, res);
                const double L = lam[size_t(q)], ga = 1 + 2.0 / 3 * std::min(dz * dz * L / 4, 1.0) + h2[size_t(q)] * L / 12;
                const double sx = std::sin(kz * dz / 2), tx = std::tan(kz * dz / 2);
                const double pred = 2 * std::atan(std::sqrt(tx * tx + dz * dz * L / 4 * (1 + 4.0 / 3 * sx * sx) * ga));
                worst = std::max({worst, std::abs(w - pred), res});
                wl = w; pl = pred;
            }
    check("rz irip, closure o5: discrete radial eigenmodes (uniform and stretched grid, TM m = 0, TE m = 1) = the corrected "
          "relation per mode", worst < 1e-11, fmt("last w dt %.6f", wl) + fmt(" vs %.6f, max error %.1e", pl, worst));
    // (b) the Bessel modes against the continuum
    double eb[2][2], eo[2][2];
    for (int te = 0; te < 2; ++te)
        for (int r = 0; r < 2; ++r) {
            const int nr = r ? 32 : 16, nz = 32, M = te ? 2 : 1, nk = 2;
            const double R = 3.0, dr = R / nr, dz = 2 * dr, kz = 2 * M_PI * nk / (nz * dz), kp = (te ? j11p : j01) / R;
            const double wex = dz * std::sqrt(kz * kz + kp * kp);
            for (int o5 = 0; o5 < 2; ++o5) {
                Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, Solver::IRip);
                FieldsRZ f(d, M);
                f.o5 = o5 == 1;
                Field& F = te ? f.bz : f.ez;
                std::vector<double> h = F.host();
                for (int j = 0; j < nr; ++j)
                    for (int i = 0; i < nz; ++i) {
                        const double rr = (j + (te ? 0.5 : 0.0)) * dr, z = (d.off[0] + i) * dz;
                        h[F.idx(i, j, te ? 1 : 0)] = (te ? std::cyl_bessel_j(1, kp * rr) : std::cyl_bessel_j(0, kp * rr)) * std::cos(kz * z);
                    }
                F.upload(h);
                comm.exchange(d, {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp});
                std::vector<std::vector<double>> hist;
                for (int n = 0; n < 60; ++n) { hist.push_back(F.host()); f.zero_currents(); f.step(comm); }
                double res;
                const double e = std::abs(recurrence(hist, res) / wex - 1);
                (o5 ? eo : eb)[te][r] = e;
            }
        }
    for (int te = 0; te < 2; ++te)
        check(std::string("rz irip, closure o5: ") + (te ? "TE11" : "TM01") + " Bessel mode at dz = 2 dr, frequency against the continuum: "
                  "o5 more than 10 times closer than the box closure",
              eo[te][0] * 10 < eb[te][0] && eo[te][1] * 10 < eb[te][1],
              fmt("rel. error nr = 16 / 32: box %.2e", eb[te][0]) + fmt(" / %.2e", eb[te][1]) + fmt(", o5 %.2e", eo[te][0]) +
                  fmt(" / %.2e", eo[te][1]));
    // (c) random fields
    double wmax = 0, wend = 0, ge = 0, gb = 0;
    for (int st = 0; st < 4; ++st) {
        const double dr = st < 2 ? 0.1 : 0.04;
        RadialGrid g = st % 2 ? rz_radial_grid(20, dr, 8 * dr, 1.1) : rz_radial_grid(20, dr);
        Domain d = rz_domain(comm, 16, 20, 3, 0.4, dr, true, Solver::IRip);
        FieldsRZ f(d, 3, std::move(g));
        f.o5 = true;
        randomize(f, comm, 23 + st);
        Field de(d), db(d);
        f.div_e(de); f.div_b(db);
        const std::vector<double> e0 = de.host(), b0 = db.host();
        const double w0 = f.energy();
        for (int n = 0; n < 300; ++n) { f.zero_currents(); f.step(comm); wmax = std::max(wmax, f.energy() / w0); }
        wend = std::max(wend, f.energy() / w0);
        f.div_e(de); f.div_b(db);
        double se = 0, sb = 0;
        for (double v : e0) se = std::max(se, std::abs(v));
        for (double v : b0) sb = std::max(sb, std::abs(v));
        ge = std::max(ge, max_change(de, e0, d) / se); gb = std::max(gb, max_change(db, b0, d) / sb);
    }
    check("rz irip, closure o5: random fields in modes 0..2 at dz/dr = 4, 10 (uniform and stretched), over 300 steps below "
          "4.62, div E and div B per mode unchanged",
          wmax < 4.62 && wmax > 0.2 && ge < 1e-12 && gb < 1e-12,
          fmt("W / W0 max %.3f", wmax) + fmt(", end %.3f", wend) + fmt(", div E change %.1e", ge) + fmt(", div B %.1e", gb));
}

// R3. Explicit RIP-CC on a stable grid (dz = dr/4, modes 0..1): random fields bounded, div E and div B per
//     mode unchanged; TM01 converges to the continuum frequency w^2 = kz^2 + kp^2.
void rz_ripcc(Comm& comm) {
    Domain d = rz_domain(comm, 16, 20, 2, 0.025, 0.1, true, Solver::RipCC);
    FieldsRZ f(d, 2);
    randomize(f, comm, 5);
    Field de(d), db(d);
    f.div_e(de); f.div_b(db);
    const std::vector<double> e0 = de.host(), b0 = db.host();
    const double w0 = f.energy();
    double wmax = w0;
    for (int n = 0; n < 400; ++n) { f.zero_currents(); f.step(comm); wmax = std::max(wmax, f.energy()); }
    f.div_e(de); f.div_b(db);
    double se = 0, sb = 0;
    for (double v : e0) se = std::max(se, std::abs(v));
    for (double v : b0) sb = std::max(sb, std::abs(v));
    const double ge = max_change(de, e0, d) / se, gb = max_change(db, b0, d) / sb;
    double rs;
    const int nr = 32;
    const double R = 3.0, dr = R / nr, dz = 0.25 * dr, kz = 2 * M_PI * 2 / (32 * dz), kp = j01 / R;
    const double w = guided(comm, nr, dz, dr, Solver::RipCC, false, rs) / dz, wth = std::sqrt(kz * kz + kp * kp);
    check("rz rip_cc: random fields bounded, div E and div B per mode unchanged; TM01 frequency",
          wmax < 1.5 * w0 && ge < 1e-12 && gb < 1e-12 && std::abs(w / wth - 1) < 5e-3,
          fmt("max W/W0 %.3f", wmax / w0) + fmt(", div E change %.1e", ge) + fmt(", div B %.1e", gb) +
              fmt(", TM01 w/w_continuum - 1 = %.1e", w / wth - 1));
}

// Particles: n per cell of an annulus r < rmax, z uniform, thermal momenta uth, drift uz0 (Cartesian, X = z).
void load(Species& sp, const Domain& d, double rmax, int ppc, double density, double uth, double uz0, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0, 1);
    std::normal_distribution<double> G;
    std::vector<double> h[Species::NC];
    const double vol = M_PI * rmax * rmax * d.L(0);
    const long n = long(ppc * vol / (d.h[0] * d.h[1] * d.h[1] * 2 * M_PI * 0.5 * rmax / d.h[1]));   // ~ppc per cell
    for (long p = 0; p < n; ++p) {
        const double r = rmax * std::sqrt(U(rng)), ph = 2 * M_PI * U(rng);
        h[Species::X].push_back(d.L(0) * U(rng)); h[Species::Y].push_back(r * std::cos(ph)); h[Species::Z].push_back(r * std::sin(ph));
        h[Species::UX].push_back(uz0 + uth * G(rng)); h[Species::UY].push_back(uth * G(rng)); h[Species::UZ].push_back(uth * G(rng));
        h[Species::W].push_back(density * vol / double(n));
    }
    sp.append(h);
}
// quiet start: 2 x 2 particles per (z, r) cell, 8 phi each (rotated per r row), weights n 2 pi r dr dz / 32
void quiet(Species& sp, const Domain& d, int nrp, double density) {
    std::vector<double> h[Species::NC];
    const double dz = d.h[0], dr = d.h[1];
    for (int i = 0; i < 2 * d.N[0]; ++i)
        for (int j = 0; j < 2 * nrp; ++j)                       // two Gauss points per radial cell (decision 53)
            for (int a = 0; a < 8; ++a) {
                const double x = j % 2 ? 0.5 + 0.5 / std::sqrt(3.0) : 0.5 - 0.5 / std::sqrt(3.0);
                const double z = (i + 0.5) * dz / 2, r = (j / 2 + x) * dr, ph = 2 * M_PI * (a + 0.37 * j) / 8;
                h[Species::X].push_back(z); h[Species::Y].push_back(r * std::cos(ph)); h[Species::Z].push_back(r * std::sin(ph));
                h[Species::UX].push_back(0); h[Species::UY].push_back(0); h[Species::UZ].push_back(0);
                h[Species::W].push_back(density * 2 * M_PI * r * (dr / 2) * (dz / 2) / 8);
            }
    sp.append(h);
}
void wrap_z(Species& sp, const Domain& d) {
    std::vector<double> z = sp.get(Species::X);
    for (double& v : z) v = std::fmod(std::fmod(v, d.L(0)) + d.L(0), d.L(0));
    sp.set(Species::X, z);
}

// R4. Gather: a uniform transverse field E_x = E0, B_y = B0 (only m = 1: E_r1 = E0/2, E_phi1 = i E0/2,
//     B_r1 = -i B0/2, B_phi1 = B0/2) is returned exactly at random points, also next to the axis.
void rz_gather_uniform(Comm& comm) {
    Domain d = rz_domain(comm, 8, 10, 2, 0.3, 0.2, true, Solver::RipCC);
    FieldsRZ f(d, 2);
    const double E0 = 1.3, B0 = -0.7;
    // E = E0 x_hat: E_r = E0 cos(phi), E_phi = -E0 sin(phi) -> mode 1: E_r1 = E0/2, E_phi1 = i E0/2
    // B = B0 y_hat: B_r = B0 sin(phi), B_phi = B0 cos(phi) -> B_r1 = -i B0/2, B_phi1 = B0/2
    auto set = [&](Field& F, double re, double im) {
        std::vector<double> h = F.host();
        for (int j = 0; j < d.n[1]; ++j) for (int i = -d.ng; i < d.n[0] + d.ng; ++i) { h[F.idx(i, j, 1)] = re; h[F.idx(i, j, 2)] = im; }
        F.upload(h);
    };
    set(f.er, E0 / 2, 0); set(f.ep, 0, E0 / 2); set(f.br, 0, -B0 / 2); set(f.bp, B0 / 2, 0);
    Species sp;
    sp.q = 0; sp.m = 1;                                   // no force: only the gather is exercised through E, B
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> U(0, 1);
    double err = 0;
    for (int order = 1; order <= 2; ++order)
        for (int t = 0; t < 200; ++t) {
            const double r = (t < 100 ? 0.25 : 1.4) * U(rng), ph = 2 * M_PI * U(rng), z = 2.0 * U(rng) + 0.3;
            double E[3], B[3];
            rz_gather_point(f, z, r * std::cos(ph), r * std::sin(ph), E, B, order);
            err = std::max({err, std::abs(E[0]), std::abs(E[1] - E0), std::abs(E[2]), std::abs(B[0]), std::abs(B[1]), std::abs(B[2] - B0)});
        }
    check("rz: gather of a uniform transverse E and B (mode 1), also next to the axis, shapes 1 and 2", err < 1e-13,
          fmt("max error %.1e", err));
    // rz_gather = matched returns it exactly too; energy does not next to the axis (its E_r weights sum to dr xi'(r), the
    // Ruyten map has xi' != 1/dr there: shape 2 E_x/E0 - 1 = +1.46 at r = 0.05 dr, -0.62 at 0.8 dr, -0.05 at 2 dr)
    double errm = 0, erre = 0;
    for (int g = 1; g <= 2; ++g) {
        f.gather_matched = true; f.gather_energy = g == 2;
        for (int order = 1; order <= 2; ++order)
            for (int t = 0; t < 200; ++t) {
                const double r = (t < 100 ? 0.25 : 1.4) * U(rng), ph = 2 * M_PI * U(rng), z = 2.0 * U(rng) + 0.3;
                double E[3], B[3];
                rz_gather_point(f, z, r * std::cos(ph), r * std::sin(ph), E, B, order);
                double& e = g == 1 ? errm : erre;
                e = std::max({e, std::abs(E[0]), std::abs(E[1] - E0), std::abs(E[2]), std::abs(B[0]), std::abs(B[1]), std::abs(B[2] - B0)});
            }
    }
    f.gather_matched = f.gather_energy = false;
    std::printf("  R4 uniform transverse field: rz_gather = matched max error %.1e, energy %.2f (next to the axis)\n", errm, erre / E0);
    check("rz: gather of a uniform transverse E and B with rz_gather = matched (exact)", errm < 1e-13, fmt("max error %.1e", errm));
}

// R5. A hot plasma (electrons u_th = 0.3 drifting, protons) in a cylinder, modes 0..2: the Gauss law per mode
//     (div E - rho, rho from the same shapes) and div B unchanged to round-off, explicit RIP-CC (stable grid)
//     and iRIP (dz/dr = 2.5, moves of up to a few radial cells).
void rz_plasma_gauss(Comm& comm) {
    for (int order = 1; order <= 2; ++order)
    for (int variant = 0; variant < 4; ++variant) {                  // rip_cc, irip, irip with closure o5, rip_cc with rz_deposit = chord
        const int implicit = variant == 1 || variant == 2;
        const double dr = 0.2, dz = implicit ? 0.5 : 0.05;
        Domain d = rz_domain(comm, 16, implicit ? 48 : 24, 3, dz, dr, true, implicit ? Solver::IRip : Solver::RipCC);
        FieldsRZ f(d, 3);
        f.o5 = variant == 2;
        f.deposit_chord = variant == 3;
        Species e, ion;
        e.q = -1; e.m = 1; ion.q = 1; ion.m = 10;
        load(e, d, 14 * dr, 4, 0.3, 0.3, 0.5, 1);
        load(ion, d, 14 * dr, 4, 0.3, 0.05, 0.0, 2);
        Field rho(d), de(d), db(d);
        auto residual = [&]() {
            rho.fill(0);
            rz_rho(e, f, rho, order); rz_rho(ion, f, rho, order);
            comm.add_guards(d, {&rho});
            f.div_e(de);
            std::vector<double> a = de.host(), b = rho.host();
            for (size_t q = 0; q < a.size(); ++q) a[q] -= b[q];
            return a;
        };
        const std::vector<double> g0 = residual();
        f.div_b(db);
        const std::vector<double> b0 = db.host();
        double sc = 0;
        for (double v : g0) sc = std::max(sc, std::abs(v));
        double gerr = 0, berr = 0;
        for (int n = 0; n < (implicit ? 8 : 20); ++n) {                // iRIP: moves of up to 2.5 dr; off the wall
            f.zero_currents();
            rz_push_deposit(e, f, Pusher::Boris, true, order); rz_push_deposit(ion, f, Pusher::Boris, true, order);
            comm.add_guards(d, {&f.jz, &f.jr, &f.jp});
            f.step(comm);
            wrap_z(e, d); wrap_z(ion, d);
            const std::vector<double> g = residual();
            f.div_b(db);
            const std::vector<double> b = db.host();
            for (int k = 0; k < d.n[2]; ++k)
                for (int j = (k > 0 ? 1 : 0); j < 18; ++j)              // off the wall; the axis is a Gauss point for m = 0 only
                    for (int i = 0; i < d.n[0]; ++i) {
                        const size_t q = de.idx(i, j, k);
                        gerr = std::max(gerr, std::abs(g[q] - g0[q]) / sc);
                        berr = std::max(berr, std::abs(b[q] - b0[q]));
                    }
        }
        double axis = 0;                                                // rho_m ~ r^m: no m >= 1 charge on the axis node
        const std::vector<double> rh = rho.host();
        for (int k = 1; k < d.n[2]; ++k)
            for (int i = 0; i < d.n[0]; ++i) axis = std::max(axis, std::abs(rh[rho.idx(i, 0, k)]));
        check(std::string("rz shape ") + std::to_string(order) + (variant == 3 ? ", rip_cc, rz_deposit = chord" : variant == 2 ? ", irip o5 (dz/dr = 2.5)" : implicit ? ", irip (dz/dr = 2.5)" : ", rip_cc") +
                  ": hot plasma, modes 0..2, Gauss law and div B per mode to round-off, rho_m>=1 = 0 on the axis",
              gerr < 1e-12 && berr < 1e-12 && axis == 0,
              fmt("Gauss change / max|residual| %.1e", gerr) + fmt(", div B change %.1e", berr) + fmt(", axis rho_m>=1 %.1e", axis));
    }
}

// Electrostatic oscillation of a cold plasma filling the conducting cylinder (quiet start). The
// perturbation u = -eps grad phi, phi = J0(kp r) sin(k z), kp = j01 / R, is curl free with phi = 0 at the
// wall, so no TM mode of the plasma-filled guide (w^2 = wp^2 + k^2 + kp^2) is excited (a perturbation
// uniform in r would excite them: E_z must vanish at the wall). The signal is the projection of E_z on
// cos(k z) J0(kp r); the leapfrog law cos(w dt) = 1 - wp^2 dt^2 / 2 (the Gauss-point loading gives rho = q n
// exactly on every node, R7).
double langmuir(Comm& comm, Solver sv, int nr, double dr, double& cth) {
    const double dz = 0.1, wp = 1.5;
    Domain d = rz_domain(comm, 32, nr, 1, dz, dr, true, sv);
    FieldsRZ f(d, 1);
    Species e, ion;
    e.q = -1; e.m = 1; ion.q = 1; ion.m = 1e12;
    quiet(e, d, nr, wp * wp); quiet(ion, d, nr, wp * wp);
    const double k = 2 * M_PI / d.L(0), kp = j01 / (nr * dr);
    {
        const std::vector<double> z = e.get(Species::X), x = e.get(Species::Y), y = e.get(Species::Z);
        std::vector<double> uz(z.size()), ux(z.size()), uy(z.size());
        for (size_t p = 0; p < z.size(); ++p) {
            const double r = std::hypot(x[p], y[p]), ur = 1e-3 * kp * std::cyl_bessel_j(1, kp * r) * std::sin(k * z[p]);
            uz[p] = -1e-3 * k * std::cyl_bessel_j(0, kp * r) * std::cos(k * z[p]);
            ux[p] = r > 0 ? ur * x[p] / r : 0.0; uy[p] = r > 0 ? ur * y[p] / r : 0.0;
        }
        e.set(Species::UX, uz); e.set(Species::UY, ux); e.set(Species::UZ, uy);
    }
    std::vector<double> sig;
    for (int n = 0; n < 120; ++n) {
        f.zero_currents();
        rz_push_deposit(e, f, Pusher::Boris); rz_push_deposit(ion, f, Pusher::Boris);
        comm.add_guards(d, {&f.jz, &f.jr, &f.jp});
        f.step(comm);
        wrap_z(e, d); wrap_z(ion, d);
        const std::vector<double> h = f.ez.host();
        double s = 0;
        for (int j = 0; j < nr; ++j)
            for (int i = 0; i < d.n[0]; ++i)
                s += h[f.ez.idx(i, j, 0)] * std::cos(k * (d.off[0] + i) * dz) * std::cyl_bessel_j(0, kp * j * dr) * (j ? j : 0.125);
        sig.push_back(s);
    }
    double num = 0, den = 0;
    for (size_t n = 1; n + 1 < sig.size(); ++n) { num += (sig[n + 1] + sig[n - 1]) * sig[n]; den += 2 * sig[n] * sig[n]; }
    cth = 1 - 0.5 * wp * wp * dz * dz;
    return num / den;
}

// R6. Langmuir oscillation in the cylinder, iRIP (dz < dr) and explicit RIP-CC at kp dr = 0.2: w within 1 % of
//     the leapfrog's, as the Cartesian test. The shapes lower w by O((k dz)^2) (k dz = 0.2: 3.3e-3, the floor)
//     and O((kp dr)^2): -7.7e-3, -3.3e-3, -1.7e-3 above the floor at kp dr = 0.3, 0.2, 0.15 with the Ruyten
//     deposition (decision 53); iRIP at kp dr = 0.1 must be closer to the leapfrog by more than 40 %.
void rz_langmuir(Comm& comm) {
    double ti, tf, te;
    const double ci = langmuir(comm, Solver::IRip, 12, 0.2, ti), cf = langmuir(comm, Solver::IRip, 24, 0.1, tf);
    const double ce = langmuir(comm, Solver::RipCC, 12, 0.2, te);
    const double ei = std::acos(ci) / std::acos(ti) - 1, ef = std::acos(cf) / std::acos(tf) - 1, ee = std::acos(ce) / std::acos(te) - 1;
    check("rz: Langmuir oscillation (m = 0) in the cylinder, iRIP and RIP-CC, converging in dr",
          std::abs(ei) < 0.01 && std::abs(ee) < 0.01 && std::abs(ef) < 0.6 * std::abs(ei),
          fmt("w / w_leapfrog - 1: irip %.1e", ei) + fmt(" (kp dr = 0.1: %.1e)", ef) + fmt(", rip_cc %.1e", ee));
}

// R7. The deck path (grid.geometry = rz, step 2b): a hot plasma column in the conducting cylinder,
// iRIP at dz = 2 dr, 2 modes. The loaded charge density of mode 0 is q n at the nodes inside the column
// (Gauss-point loading, Ruyten weights: exact on every node, decision 53); after 60 steps,
// with electrons absorbed by the wall, the Gauss law per mode is unchanged to round-off. An explicit
// RIP-CC deck beyond dz^2 (lambda_max + wp^2)/4 = 1 is rejected.
void rz_deck(Comm& comm) {
    const char* path = "rz_test_tmp.deck";
    {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n modes = 2\n lambda0 = 1 um\n nz = 16\n nr = 16\n dz = 0.2 um\n dr = 0.1 um\n"
             " t_end = 40 fs\n}\nnumerics {\n solver = irip\n pusher = boris\n}\n"
             "species e {\n charge = -1\n mass = 1\n density = 0.01\n rmax = 1.2 um\n ppc_r = 2\n ppc_theta = 8\n"
             " uth_z = 0.1\n uth_x = 0.1\n uth_y = 0.1\n uz = 0.05\n}\n"
             "species i {\n charge = 1\n mass = 1836\n density = 0.01\n rmax = 1.2 um\n ppc_r = 2\n ppc_theta = 8\n}\n"
             "output {\n dir = rz_test_tmp_out\n}\n";
    }
    Deck dk = Deck::read(path);
    for (int order = 1; order <= 2; ++order) {
        if (order == 2) dk.override_from("numerics.shape=2");
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        Field rho(sim.dom);
        rho.fill(0);
        rz_rho(sim.species[0], *sim.rz, rho, order);
        comm.add_guards(sim.dom, {&rho});
        const std::vector<double> h = rho.host();
        double lerr = 0;
        for (int j = 0; j < 11; ++j)                              // the axis node included (decision 53)
            for (int i = 0; i < sim.dom.n[0]; ++i) lerr = std::max(lerr, std::abs(h[rho.idx(i, j, 0)] + 0.01) / 0.01);
        lerr = comm.allreduce_max(lerr);
        const double n0 = double(sim.species[0].size());
        while (!sim.finished()) sim.step();
        const double g = sim.gauss_drift(), n1 = double(sim.species[0].size());
        // shape 1: the Gauss-point loading with the Ruyten weights deposits q n exactly, the axis node too
        // (decision 53); shape 2: to the quadrature of the 2 Gauss points across the spline's kinks (without the
        // density-conserving map 13/8 on the axis node), and the
        // Gauss law with absorption (the wall at (nr + 1/2) dr for shape 2, decision 50)
        check("rz deck (geometry = rz, iRIP, 2 modes, shape " + std::to_string(order) + "): " +
                  (order == 1 ? "loading rho = q n exactly" : "loading rho = q n to 5e-3") + " (the axis node too), Gauss law with wall absorption",
              (order == 2 ? lerr < 5e-3 : lerr < 1e-12) && g < 1e-12 && n1 < n0,
              fmt("rho error %.1e", lerr) + fmt(", Gauss drift %.1e", g) + fmt(", electrons %.0f", n0) + fmt(" -> %.0f", n1));
    }
    dk.override_from("numerics.shape=1");
    bool rejected = false;
    dk.override_from("numerics.solver=rip_cc");
    try { Config::from_deck(dk); } catch (const std::exception& e) { rejected = std::string(e.what()).find("lambda_max") != std::string::npos; }
    check("rz deck: explicit RIP-CC beyond dz^2 (lambda_max + wp^2) / 4 = 1 rejected", rejected, "");
    if (comm.rank == 0) std::remove(path);
}

// R8. Laser in RZ (step 3): a Gaussian pulse polarised along x enters through the open left z edge
// (iRIP, 2 modes, vacuum), propagates and leaves through the right edge. Checks against independent
// references: the field energy inside equals the analytic (pi w0^2 / 2)(tau / 2) sqrt(pi / (4 ln 2))
// (a0 = 1, focus at the entrance) and is conserved while the pulse is inside; the on-axis fluence at
// z = L/2 relative to z = 2 dz follows the paraxial law (1 + (z / z_R)^2)^-1 of the round beam (to 1e-3
// here; the scheme's diffraction carries O((k dz)^2) and O((k_perp dr)^2) errors: +2.4e-2 at dz = lambda/8,
// -3e-3 at dz = lambda/32 and dr = w/20, converging to the paraxial law; spec);
// E_y = 0 on the axis; nothing remains after the exit; Gauss law and div B at round-off.
void rz_laser(Comm& comm) {
    const char* path = "rz_test_laser_tmp.deck";
    {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n modes = 2\n lambda0 = 0.8 um\n nz = 320\n nr = 40\n dz = 0.05 um\n dr = 0.25 um\n"
             " boundary_z = open\n t_end = 100 fs\n}\nnumerics {\n solver = irip\n}\n"
             "laser {\n a0 = 1\n polarization = x\n duration = 8 fs\n waist = 2.5 um\n}\n"
             "output {\n dir = rz_test_tmp_out\n}\n";
    }
    Config c = Config::from_deck(Deck::read(path));
    if (comm.rank == 0) std::remove(path);
    Simulation sim(c, comm);
    sim.init();
    const LaserConfig& L = c.laser;
    const double w0 = L.w0, zr = 0.5 * w0 * w0, dz = sim.dom.h[0], dt = sim.dom.dt();
    const double wth = 0.5 * M_PI * w0 * w0 * 0.5 * L.tau * std::sqrt(M_PI / (4 * std::log(2.0)));
    const double z1 = 2 * dz, z2 = 0.5 * sim.dom.L(0);
    double f1 = 0, f2 = 0, eymax = 0, exmax = 0, wmax = 0, wmin = 1e300, wend = 0;
    bool inside = false;
    while (!sim.finished()) {
        sim.step();
        double E[3], B[3];
        rz_gather_point(*sim.rz, z1, 0, 0, E, B);
        f1 += E[1] * E[1] * dt;
        rz_gather_point(*sim.rz, z2, 0, 0, E, B);
        f2 += E[1] * E[1] * dt;
        exmax = std::max(exmax, std::abs(E[1])); eymax = std::max(eymax, std::abs(E[2]));
        const double w = sim.rz->energy();
        // the pulse is inside from t = 2 delay + (its tail) until its head reaches the right edge
        inside = sim.t > 2 * L.tc + 2 * dt && sim.t < sim.dom.L(0) - 2 * L.tau;
        if (inside) { wmax = std::max(wmax, w); wmin = std::min(wmin, w); }
        wend = w;
    }
    const double g = sim.gauss_drift();
    double wr, gr, rr, br;
    sim.rz_diagnostics(wr, gr, rr, br);
    const double eW = wmax / wth - 1, cons = (wmax - wmin) / wmax;
    const double fl = (f2 / f1) / ((1 + z1 * z1 / (zr * zr)) / (1 + z2 * z2 / (zr * zr))) - 1;
    check("rz laser (open z, iRIP): energy = analytic, conserved inside, paraxial on-axis fluence, leaves",
          std::abs(eW) < 2e-3 && cons < 1e-5 && std::abs(fl) < 1e-2 && eymax < 1e-12 * exmax && wend < 1e-5 * wmax &&
              g < 1e-12 && br < 1e-12,
          fmt("W/W_th - 1 = %.1e", eW) + fmt(", change inside %.1e", cons) + fmt(", fluence ratio / paraxial - 1 = %.1e", fl) +
              fmt(", left %.1e", wend / wmax) + fmt(", Gauss %.1e", g) + fmt(", div B %.1e", br));
}

// R9. Moving window in RZ (step 3): a laser drives a wake in a plasma that enters at the right edge as
// the window moves at c (hot electrons, cold protons), explicit RIP-CC and iRIP (dz = dr / 2 and 2 dr: the
// entry cell from the constraints). The Gauss law per mode and div B stay at round-off, and the number
// of protons is exactly the number loaded into the entering columns: shifts x radial cells x ppc (hot
// electrons may reach the wall of the narrow iRIP case). Without the iRIP entry rule the Gauss law would
// break at 1.7e-4 here.
void rz_window(Comm& comm) {
    for (int run = 0; run < 3; ++run) {                       // rip_cc, irip, irip with shape 2
        const int implicit = run > 0, order = run == 2 ? 2 : 1;
        const char* path = "rz_test_window_tmp.deck";
        {
            std::ofstream f(path);
            f << "grid {\n geometry = rz\n modes = 2\n lambda0 = 0.8 um\n nz = 64\n nr = 16\n dz = 0.1 um\n"
              << (implicit ? " dr = 0.05 um\n" : " dr = 0.2 um\n")
              << " boundary_z = open\n t_end = 40 fs\n}\nnumerics {\n solver = " << (implicit ? "irip" : "rip_cc") << "\n shape = " << order << "\n}\n"
                 "laser {\n a0 = 1\n polarization = x\n duration = 5 fs\n waist = " << (implicit ? "0.4 um" : "1.5 um") << "\n}\n"
                 "window {\n start = 20 fs\n}\n"
                 "species e {\n charge = -1\n mass = 1\n density = 0.02\n zmin = 6.4 um\n rmax = " << (implicit ? "0.5 um" : "2 um")
              << "\n ppc_r = 2\n ppc_theta = 4\n uth_z = 0.02\n uth_x = 0.02\n uth_y = 0.02\n}\n"
                 "species p {\n charge = 1\n mass = 1836\n density = 0.02\n zmin = 6.4 um\n rmax = " << (implicit ? "0.5 um" : "2 um")
              << "\n ppc_r = 2\n ppc_theta = 4\n}\noutput {\n dir = rz_test_tmp_out\n}\n";
        }
        Config c = Config::from_deck(Deck::read(path));
        if (comm.rank == 0) std::remove(path);
        Simulation sim(c, comm);
        sim.init();
        double g = 0, b = 0;
        while (!sim.finished()) {
            sim.step();
            double w, gg, r, bb;
            sim.rz_diagnostics(w, gg, r, bb);
            g = std::max(g, gg); b = std::max(b, bb);
        }
        const long expect = sim.dom.wx * 10 * 2 * 4;                    // rmax / dr = 10 radial cells, ppc 1 x 2 x 4
        const long np = long(sim.species[1].size()), ne = long(sim.species[0].size());
        check(std::string("rz moving window (") + (implicit ? "irip, dz = 2 dr" : "rip_cc, dz = dr / 2") + (order == 2 ? ", shape 2" : "") +
                  "): laser wake, plasma entering, Gauss law and div B at round-off, exact entering count",
              g < 1e-12 && b < 1e-12 && np == expect && ne <= expect && sim.dom.wx > 50,
              fmt("Gauss %.1e", g) + fmt(", div B %.1e", b) + fmt(", protons %.0f", double(np)) + fmt(" (expected %.0f)", double(expect)));
    }
}

#ifdef RIP_HDF5
// R12. Checkpoint/restart in RZ: an uninterrupted run against one that writes a checkpoint, runs on 7
//      steps (as if it stopped later) and is restarted from it. The moving window, the laser, plasma
//      entering, the radial absorber and shape 2, for rip_cc and irip: every field slot, the initial Gauss
//      residual and every particle equal, exactly on one CPU thread (the RZ deposit adds atomically: with
//      several threads or on the GPU it sums in another order, round-off). The initial Gauss residual is compared
//      relative to max |rho| of the electrons: every window shift folds the charge of the column leaving the frozen
//      cells into it (decision 91), and for this neutral plasma that charge is round-off of rho itself, so relative to
//      its own maximum the GPU's summation order would give O(1).
void rz_restart(Comm& comm) {
#ifdef _OPENMP
    const int nthr = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    for (int implicit = 0; implicit < 2; ++implicit) {
        const char* path = "rz_test_restart_tmp.deck";
        {
            std::ofstream f(path);
            f << "grid {\n geometry = rz\n modes = 3\n lambda0 = 0.8 um\n nz = 64\n nr = 24\n dz = 0.1 um\n dr = "
              << (implicit ? "0.05 um" : "0.3 um") << "\n boundary_z = open\n boundary_r = absorbing\n r_absorber_cells = 8\n"
                 " t_end = 36 fs\n}\nnumerics {\n solver = " << (implicit ? "irip" : "rip_cc") << "\n shape = 2\n}\n"
                 "laser {\n a0 = 1.5\n polarization = x\n duration = 5 fs\n waist = " << (implicit ? "0.4 um" : "1.5 um") << "\n}\n"
                 "window {\n start = 20 fs\n}\n"
                 "species e {\n charge = -1\n mass = 1\n density = 0.02\n zmin = 6.4 um\n ppc_r = 2\n ppc_theta = 4\n"
                 " uth_z = 0.02\n uth_x = 0.02\n uth_y = 0.02\n}\n"
                 "species p {\n charge = 1\n mass = 1836\n density = 0.02\n zmin = 6.4 um\n ppc_r = 2\n ppc_theta = 4\n}\n"
                 "output {\n dir = rz_test_restart_out\n}\n";
        }
        const Config c = Config::from_deck(Deck::read(path));
        if (comm.rank == 0) std::remove(path);
        Simulation a(c, comm);
        a.init();
        while (!a.finished()) a.step();
        std::string chk;
        long nchk = 0;
        {
            Simulation b(c, comm);
            b.init();
            const long k = long(0.75 * double(a.istep));               // after the window started
            while (b.istep < k) b.step();
            chk = write_checkpoint(b);
            nchk = b.istep;
            for (int q = 0; q < 7; ++q) b.step();
        }
        Simulation r(c, comm);
        r.init(chk);
        const bool at = r.istep == nchk && r.restart_step == nchk && r.dom.wx > 0;
        while (!r.finished()) r.step();
        double worst = 0;
        bool same = a.istep == r.istep && a.dom.wx == r.dom.wx;
        auto cmp = [&](const std::vector<double>& x, const std::vector<double>& y, double scale = 0) {
            if (x.size() != y.size()) { same = false; return; }
            double m = scale, d = 0;
            for (size_t q = 0; q < x.size(); ++q) { m = std::max(m, std::abs(x[q])); d = std::max(d, std::abs(x[q] - y[q])); }
            if (d > 0) worst = std::max(worst, d / m);
        };
        double rho_e = 0;                                              // the scale of the Gauss residual
        {
            Field rho(a.dom);
            rho.fill(0);
            rz_rho(a.species[0], *a.rz, rho, a.cfg.order);
            comm.add_guards(a.dom, {&rho});
            for (double v : rho.host()) rho_e = std::max(rho_e, std::abs(v));
            rho_e = comm.allreduce_max(rho_e);
        }
        const Field* fa[7] = {&a.rz->ez, &a.rz->er, &a.rz->ep, &a.rz->bz, &a.rz->br, &a.rz->bp, &a.gauss0()};
        const Field* fb[7] = {&r.rz->ez, &r.rz->er, &r.rz->ep, &r.rz->bz, &r.rz->br, &r.rz->bp, &r.gauss0()};
        for (int q = 0; q < 7; ++q) cmp(fa[q]->host(), fb[q]->host(), q == 6 ? rho_e : 0.0);
        for (size_t s = 0; s < a.species.size(); ++s)
            for (int sl = 0; sl < Species::NC; ++sl) cmp(a.species[s].get(sl), r.species[s].get(sl));
        double ne = double(r.species[0].size());
        comm.allreduce_sum(&ne, 1);
        check(std::string("rz restart (") + (implicit ? "irip" : "rip_cc") + ", shape 2, window, laser, plasma, absorber) = uninterrupted run",
              at && same && ne > 0 && (!RIP_GPU ? worst == 0 : worst < 1e-9),
              fmt("checkpoint at step %.0f", double(nchk)) + fmt(", window shifts %.0f", double(r.dom.wx)) +
                  fmt(", %.0f electrons", ne) + fmt(", max relative difference %.1e", worst) + (at ? "" : ", restored state WRONG"));
    }
#ifdef _OPENMP
    omp_set_num_threads(nthr);
#endif
}

// R10. openPMD thetaMode output (step 3b). A uniform transverse field E = E0 x, B = B0 y (mode 1:
// E_r1 = E0/2, E_phi1 = i E0/2, B_r1 = -i B0/2, B_phi1 = B0/2) is written and read back; the openPMD
// definition F = F_0 + sum_m [C_m cos(m theta) + S_m sin(m theta)] of the file's coefficients must give
// E_x = E0, E_y = 0, B_y = B0 at every node and angle (an independent check of the convention: only the
// physics of the field and the standard's formula). Also the mesh attributes and the particle axes
// (openPMD x, y, z = the particles' Y, Z, X).
void rz_openpmd(Comm& comm) {
    const char* path = "rz_test_opmd_tmp.deck";
    {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n modes = 2\n lambda0 = 1 um\n nz = 8\n nr = 6\n dz = 0.1 um\n dr = 0.2 um\n t_end = 1 fs\n}\n"
             "numerics {\n solver = irip\n}\n"
             "species e {\n charge = -1\n mass = 1\n density = 0.01\n rmax = 0.6 um\n ppc_theta = 3\n uth_x = 0.1\n}\n"
             "output {\n dir = rz_test_opmd_out\n openpmd_every = 1\n openpmd_fields = E B rho\n openpmd_species = all\n"
             " openpmd_precision = double\n}\n";
    }
    Config c = Config::from_deck(Deck::read(path));
    if (comm.rank == 0) std::remove(path);
    Simulation sim(c, comm);
    sim.init();
    const double E0 = 1.3, B0 = -0.7;
    auto set = [&](Field& F, double re, double im) {
        std::vector<double> h = F.host();
        std::fill(h.begin(), h.end(), 0.0);
        for (int j = 0; j < sim.dom.n[1]; ++j)
            for (int i = 0; i < sim.dom.n[0]; ++i) { h[F.idx(i, j, 1)] = re; h[F.idx(i, j, 2)] = im; }
        F.upload(h);
    };
    set(sim.rz->er, E0 / 2, 0); set(sim.rz->ep, 0, E0 / 2); set(sim.rz->ez, 0, 0);
    set(sim.rz->br, 0, -B0 / 2); set(sim.rz->bp, B0 / 2, 0); set(sim.rz->bz, 0, 0);
    write_openpmd(sim);
    const int nz = sim.dom.N[0], nr = sim.dom.N[1];
    double err = 0;
    bool attrs = true;
    {
        h5::Id f = h5::open_file("rz_test_opmd_out/openpmd/data_000000.h5");
        h5::Id m = h5::open_group(f, "data/0/meshes");
        auto read = [&](const char* rec, const char* comp, std::vector<double>& v, double& unit) {
            h5::Id g = h5::open_group(m, rec);
            h5::Id ds = h5::open_dataset(g, comp);
            const std::vector<hsize_t> dm = h5::dims(ds);
            attrs = attrs && dm.size() == 3 && dm[0] == 3 && int(dm[1]) == nr && int(dm[2]) == nz;
            v.resize(size_t(3 * nr * nz));
            h5::read<double>(ds, v.data(), {0, 0, 0}, {3, hsize_t(nr), hsize_t(nz)});
            unit = h5::attr_num<double>(ds, "unitSI");
            attrs = attrs && h5::attr_string(g, "geometry") == "thetaMode" && h5::attr_strings(g, "axisLabels")[0] == "r";
        };
        std::vector<double> er, et, br, bt;
        double ue, ub;
        read("E", "r", er, ue); read("E", "t", et, ue); read("B", "r", br, ub); read("B", "t", bt, ub);
        // values in code units (unitSI: their SI factor, here m c omega0 / e and m omega0 / e)
        attrs = attrs && ue > 1e11 && ub > 1e3;
        for (int j = 0; j < nr; ++j)
            for (int i = 0; i < nz; ++i)
                for (int q = 0; q < 7; ++q) {
                    const double th = 0.9 * q, cs = std::cos(th), sn = std::sin(th);
                    auto val = [&](const std::vector<double>& v) {
                        auto at = [&](int k) { return v[size_t((k * nr + j) * nz + i)]; };
                        return at(0) + at(1) * cs + at(2) * sn;
                    };
                    const double Er = val(er), Et = val(et), Br = val(br), Bt = val(bt);
                    const double Ex = Er * cs - Et * sn, Ey = Er * sn + Et * cs, Bx = Br * cs - Bt * sn, By = Br * sn + Bt * cs;
                    err = std::max({err, std::abs(Ex - E0), std::abs(Ey), std::abs(Bx), std::abs(By - B0)});
                }
        // particles: openPMD x, y, z against the species' Y, Z, X
        h5::Id pg = h5::open_group(f, "data/0/particles/e/position");
        const size_t np = sim.species[0].size();
        std::vector<double> px(np), pz(np);
        h5::read<double>(h5::open_dataset(pg, "x"), px.data(), {0}, {np});
        h5::read<double>(h5::open_dataset(pg, "z"), pz.data(), {0}, {np});
        const std::vector<double> Y = sim.species[0].get(Species::Y), X = sim.species[0].get(Species::X);
        for (size_t p = 0; p < np; ++p) err = std::max({err, std::abs(px[p] - Y[p]), std::abs(pz[p] - X[p])});
    }
    check("rz openPMD thetaMode output: a uniform transverse E, B reconstructed by the standard's formula, particle axes",
          attrs && err < 1e-12, fmt("max error %.1e", err));
}
#endif

// R11. Radial absorbing layer (step 4a): a ring pulse (m = 0 E_z and m = 1 B_z, uniform in z) runs out
// radially, partly through the axis first. After the time to cross the cylinder twice, the energy left
// inside r < R - layer relative to the initial one: with the conducting wall it is O(1) (reflected),
// with the 32-cell layer below 3e-4 (the residual is the reflection from the layer's gradient of the
// long radial wavelengths of the ring; ~cells^-3: 9e-4, 2.7e-4, 9e-5 for 16, 24, 32 cells at dz = dr;
// independent of R0). iRIP and explicit RIP-CC.
double absorber_left(Comm& comm, Solver sv, double dz, int cells) {
    const int nr = 80, M = 2;
    const double dr = 0.2;
    Domain d = rz_domain(comm, 4, nr, M, dz, dr, true, sv);
    FieldsRZ f(d, M);
    auto ring = [&](Field& F, int slot, double half) {
        std::vector<double> h = F.host();
        for (int j = 0; j < nr; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                const double r = (j + half) * dr;
                h[F.idx(i, j, slot)] = std::exp(-(r - 3) * (r - 3) / 0.36);
            }
        F.upload(h);
    };
    ring(f.ez, 0, 0.0); ring(f.bz, 1, 0.5);
    const int j0 = nr - (cells > 0 ? cells : 32);
    auto inner = [&]() {
        double w = 0;
        for (Field* F : {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp}) {
            const bool half = F == &f.er || F == &f.bz || F == &f.bp;
            const std::vector<double> h = F->host();
            for (int k = 0; k < d.n[2]; ++k)
                for (int j = 0; j < j0; ++j)
                    for (int i = 0; i < d.n[0]; ++i) {
                        const double v = half ? (j + 0.5) : (j ? j : 0.125);
                        w += (k ? 2.0 : 1.0) * v * h[F->idx(i, j, k)] * h[F->idx(i, j, k)];
                    }
        }
        return w;
    };
    const double w0 = inner();
    const int steps = int(2.2 * nr * dr / dz);
    const double kappa = cells > 0 ? 1.5 * std::log(1e4) / (cells * dr / dz) : 0.0;
    for (int n = 0; n < steps; ++n) {
        f.zero_currents();
        f.step(comm);
        if (cells > 0) { rz_absorb(f, cells, kappa); comm.exchange(d, {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp}); }
    }
    return inner() / w0;
}

void rz_absorber(Comm& comm) {
    const double wall = absorber_left(comm, Solver::IRip, 0.4, 0);
    const double ai = absorber_left(comm, Solver::IRip, 0.4, 32), ae = absorber_left(comm, Solver::RipCC, 0.05, 32);
    check("rz radial absorbing layer: ring pulse in modes 0, 1 leaves (iRIP and RIP-CC), the wall reflects it",
          ai < 3e-4 && ae < 3e-4 && wall > 0.1, fmt("energy left inside: irip %.1e", ai) + fmt(", rip_cc %.1e", ae) + fmt(", wall %.2f", wall));
}

// R-B7. Many azimuthal modes, plasma with a small thermal seed (bench/fbpic_nm_scan/rip_rz.deck without the laser).
// iRIP, Nm = 10, p_theta = 40, B7's grid
// (dz = c dt = 0.05 um, dr = 8 dz) and density (2.3e-3 n_c), u_th = 0.01: every mode stays at its thermal level
// (W_m at step 300 within a factor 3 of step 150, after the relaxation to the thermal level;
// with rz_transverse_shape = gaussian, decision 79, the smoothed noise starts lower and reaches its plateau, 2-3 times the level of
// step 50, by step 150, flat to step 850; without it 1.5-1.9 times; the top mode no
// different from the others) and the Gauss law per
// mode at round-off. A cold plasma at rest is no test: its current is exactly zero.
// R-AX. The axis instability of a dense plasma (decision 80, bench/noise_heating "The axis instability"): a uniform 40 n_c
// plasma at 1 keV with fixed ions, no laser, dz = 0.0125 um, dr = 0.0425 um (k_p dr = 2.1), 4 modes, periodic z. Physically
// nothing happens; 2D Cartesian on the same cells keeps W_kin within 1 % over 29 fs. With the plain gather (the radial
// shape of the gather differs from the deposit's next to the axis) W_kin grows 8-fold by 15 fs from the axis; with
// numerics.rz_gather = matched (the deposit's radial map) or energy (E_r with the derivative of the deposit's radial shape)
// it stays within 30 % to 15 fs (1.18, 1.08-1.12 by the thread count) (both grow later: 2.1 and 2.4 times at 30 fs, plain 18; bench/noise_heating).
double rz_axis_heating(Comm& comm, const char* gather) {
    if (comm.size > 1) return 0;
    const char* path = "rz_test_axis.deck";
    {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n modes = 4\n lambda0 = 0.8 um\n nz = 32\n nr = 40\n dz = 0.0125 um\n dr = 0.0425 um\n"
             " boundary_z = periodic\n t_end = 15 fs\n}\nnumerics {\n solver = rip_cc\n shape = 2\n filter_passes = 1\n rz_gather = "
          << gather << "\n}\n"
             "species e {\n charge = -1\n mass = 1\n density = 40\n rmax = 1.2 um\n ppc_z = 1\n ppc_r = 2\n ppc_theta = 10\n"
             " uth_z = 0.044\n uth_x = 0.044\n uth_y = 0.044\n}\n"
             "species i {\n charge = 1\n mass = 1836\n mobile = false\n density = 40\n rmax = 1.2 um\n ppc_z = 1\n ppc_r = 2\n"
             " ppc_theta = 10\n}\noutput {\n dir = rz_test_tmp_out\n}\n";
    }
    Config c = Config::from_deck(Deck::read(path));
    Simulation sim(c, comm);
    sim.init();
    const double fs = 0.8e-6 / 2.99792458e8 / (2 * M_PI) * 1e15;   // 1 / omega0 in fs
    double k5 = 0, kmax = 0, gmax = 0;
    std::vector<double> w, g;
    while (!sim.finished()) {
        sim.step();
        double kin = 0;
        for (auto& sp : sim.species) kin += sp.kinetic_energy();
        const double t = sim.t * fs;
        if (k5 == 0 && t >= 5) k5 = kin;
        if (k5 > 0) kmax = std::max(kmax, kin / k5);
        if (sim.istep % 40 == 0) { sim.rz_mode_diagnostics(w, g); for (double x : g) gmax = std::max(gmax, x); }
    }
    std::printf("  R-AX dense plasma next to the axis, rz_gather = %s: max W_kin(t) / W_kin(5 fs) %.3f, max dGauss %.1e\n", gather, kmax, gmax);
    if (gmax > 1e-12) return 1e9;
    return kmax;
}

void rz_axis_instability(Comm& comm) {
    if (comm.size > 1) return;
    const double pl = rz_axis_heating(comm, "plain"), ma = rz_axis_heating(comm, "matched"), en = rz_axis_heating(comm, "energy");
    check("rz dense plasma next to the axis: the plain gather heats it (axis instability), rz_gather = matched, energy keep it",
          pl > 2 && ma < 1.3 && en < 1.3, fmt("W_kin(15 fs) / W_kin(5 fs): plain %.2f", pl) + fmt(", matched %.3f", ma) +
          fmt(", energy %.3f", en));
}

void rz_thermal_modes(Comm& comm) {
    const char* path = "rz_test_thermal.deck";
    {
        std::ofstream f(path);
        f << "grid {\n geometry = rz\n modes = 10\n lambda0 = 0.8 um\n nz = 64\n nr = 12\n dz = 0.05 um\n dr = 0.4 um\n"
             " t_end = 50 fs\n}\nnumerics {\n solver = irip\n}\n"
             "species e {\n charge = -1\n mass = 1\n density = 2.2963e-3\n rmax = 4.4 um\n ppc_z = 2\n ppc_r = 2\n"
             " ppc_theta = 40\n uth_z = 0.01\n uth_x = 0.01\n uth_y = 0.01\n}\n"
             "species i {\n charge = 1\n mass = 1\n mobile = false\n density = 2.2963e-3\n rmax = 4.4 um\n ppc_z = 2\n"
             " ppc_r = 2\n ppc_theta = 40\n}\noutput {\n dir = rz_test_tmp_out\n}\n";
    }
    Config c = Config::from_deck(Deck::read(path));
    Simulation sim(c, comm);
    sim.init();
    std::vector<double> w50, w, g;
    double grow = 0, gmax = 0;
    for (int n = 1; n <= 300; ++n) {
        sim.step();
        if (n == 150) sim.rz_mode_diagnostics(w50, g);
        if (n > 150 && n % 25 == 0) {
            sim.rz_mode_diagnostics(w, g);
            for (int m = 1; m < c.modes; ++m) grow = std::max(grow, w[size_t(m)] / w50[size_t(m)]);
            for (double x : g) gmax = std::max(gmax, x);
        }
    }
    check("rz many modes (B7, Nm = 10, iRIP): thermal plasma, no mode grows, Gauss law per mode at round-off",
          grow < 3 && gmax < 1e-13 && w50[size_t(c.modes - 1)] > 0,
          fmt("max W_m(t) / W_m(150 steps) %.2f", grow) + fmt(", W_9 / W_1 %.2f", w50[size_t(c.modes - 1)] / w50[1]) +
              fmt(", max dGauss %.1e", gmax));
}
// R13. Axis crossing (numerics.rz_deposit, 2026-11-13). One particle moves 0.2 dr along x past the axis at the impact parameter
//     b (shape 2, one explicit step). The m = 0 azimuthal moment of its current, dt sum r_j J_phi0 dV, must equal q w L
//     (L = x0 y1 - y0 x1, the angular momentum of the move): rz_deposit = phase (deposit_seg, r_j dphi with dphi ~ pi near the
//     axis) gives ~ pi r_j^2 W for b << dr, chord (L / r_j) the exact value up to the folded weights of the axis cell.
void rz_axis_crossing(Comm& comm) {
    const double dr = 0.2, dz = 0.05;
    for (double b : {2.0, 0.2, 0.01}) {
        double mom[2];
        double L = 0;
        for (int ch = 0; ch < 2; ++ch) {
            Domain d = rz_domain(comm, 8, 12, 2, dz, dr, true, Solver::RipCC);
            FieldsRZ f(d, 2);
            f.deposit_chord = ch == 1;
            Species e;
            e.q = -1; e.m = 1;
            const double dx = 0.2 * dr, v = dx / d.dt(), u = v / std::sqrt(1 - v * v);
            std::vector<double> h[Species::NC];
            h[Species::X].push_back(4 * dz); h[Species::Y].push_back(-0.5 * dx); h[Species::Z].push_back(b * dr);
            h[Species::UX].push_back(0); h[Species::UY].push_back(u); h[Species::UZ].push_back(0); h[Species::W].push_back(1.0);
            e.append(h);
            L = (-0.5 * dx) * (b * dr) - (b * dr) * (0.5 * dx);
            f.zero_currents();
            rz_push_deposit(e, f, Pusher::None, true, 2);
            comm.add_guards(d, {&f.jz, &f.jr, &f.jp});
            const std::vector<double> jp = f.jp.host();
            const RadialGrid& g = f.grid();
            double s = 0;
            for (int j = 1; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) s += g.r[size_t(j)] * jp[f.jp.idx(i, j, 0)] * 2 * M_PI * g.A[size_t(j)] * dz * d.dt();
            mom[ch] = s;
        }
        const double ref = -L;                                        // q w L, q = -1
        std::printf("  R13 axis crossing, b = %.2f dr, move 0.2 dr: dt sum r J_phi0 dV / (q w L): phase %.3g, chord %.3g\n", b,
                    mom[0] / ref, mom[1] / ref);
        // far from the axis chord is exact (L / r_j over the stencil), phase within second order (1.06 at b = 2 dr);
        // near it the error of chord stays below 0.02 dr^2 |q w| (the folded weights of the axis cell: 0.016 at b = 0.2 dr,
        // 0.001 at 0.01 dr), that of phase is 30 to 1000 times larger (0.54, 1.38 dr^2)
        if (b >= 1) check(fmt("rz axis crossing, b = %.2f dr: chord gives the angular momentum exactly, phase within 10 %%", b),
                          std::abs(mom[0] / ref - 1) < 0.1 && std::abs(mom[1] / ref - 1) < 1e-12, fmt("%.3g", mom[0] / ref) + fmt(" %.3g", mom[1] / ref));
        else check(fmt("rz axis crossing, b = %.2f dr: chord keeps the m = 0 J_phi near the angular momentum (< 0.02 dr^2), phase 30x worse", b),
                   std::abs(mom[1] - ref) < 0.02 * dr * dr && std::abs(mom[0] - ref) > 30 * std::abs(mom[1] - ref),
                   fmt("chord %.2e", (mom[1] - ref) / (dr * dr)) + fmt(", phase %.2e dr^2", (mom[0] - ref) / (dr * dr)));
    }
}
}  // namespace

namespace {
// The radial transform of iRIP (src/rz_irip.cpp; on the GPU a tiled kernel with 64 x 64 outputs per block and j in tiles of
// 16) against the direct sum, on shapes that are no multiples of the tiles. Every output sums j = 0 .. nr - 1 in order, so
// only the rounding of the multiply-adds (fused on the GPU) may differ from the host sum.
void rz_radial_transform_check(Comm& comm) {
    double err = 0;
    const int shapes[3][3] = {{70, 37, 2}, {130, 129, 3}, {5, 17, 1}};          // nz, nr, modes
    for (const auto& sh : shapes) {
        const int nz = sh[0], nr = sh[1], M = sh[2];
        Domain d = rz_domain(comm, nz, nr, M, 0.1, 0.1, true, Solver::IRip);
        const int S = d.n[2];
        std::mt19937_64 gen(7);
        std::uniform_real_distribution<double> u(-1, 1);
        std::vector<double> A(size_t(M) * nr * nr);
        for (double& a : A) a = u(gen);
        Field in(d), out(d);
        std::vector<double> h = in.host();
        for (int k = 0; k < S; ++k)
            for (int j = 0; j < nr; ++j)
                for (int i = 0; i < nz; ++i) h[in.idx(i, j, k)] = u(gen);
        in.upload(h);
        rz_radial_transform(A, in, out, nz, nr, S);
        const std::vector<double> o = out.host();
        for (int k = 0; k < S; ++k)
            for (int a = 0; a < nr; ++a)
                for (int i = 0; i < nz; ++i) {
                    double s = 0, sa = 0;
                    for (int j = 0; j < nr; ++j) {
                        const double t = A[(size_t((k + 1) / 2) * nr + a) * nr + j] * h[in.idx(i, j, k)];
                        s += t; sa += std::abs(t);
                    }
                    err = std::max(err, std::abs(o[out.idx(i, a, k)] - s) / sa);
                }
    }
    check("rz irip: radial transform (tiled GPU kernel) = direct sum, nz x nr x slots 70 x 37 x 3, 130 x 129 x 5, 5 x 17 x 1",
          err < 1e-14, [&] { char b[64]; std::snprintf(b, sizeof b, "max |out - sum| / sum |terms| = %.2e", err); return std::string(b); }());
}
}  // namespace

void test_rz(Comm& comm) {
    if (comm.size > 1) return;                     // one rank so far (iRIP)
    rz_guided_irip(comm);
    rz_irip_random(comm);
    rz_o5(comm);
    rz_radial_transform_check(comm);
    rz_ripcc(comm);
    rz_gather_uniform(comm);
    rz_plasma_gauss(comm);
    rz_axis_crossing(comm);
    rz_langmuir(comm);
    rz_deck(comm);
    rz_laser(comm);
    rz_window(comm);
    rz_absorber(comm);
    rz_thermal_modes(comm);
    rz_axis_instability(comm);
#ifdef RIP_HDF5
    rz_openpmd(comm);
    rz_restart(comm);
#endif
}
