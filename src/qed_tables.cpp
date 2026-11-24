// Tables of the spin-resolved emission rate (qed.hpp, docs/specs/qed.md): the Bessel functions of y and the integrals over
// u at each chi, built once on the host at the start (about 0.3 s).
#include <algorithm>
#include <cmath>
#include "qed.hpp"

namespace {
constexpr double Y_LO = 1e-14, Y_HI = 720.0;      // y beyond Y_HI: K < 1e-313 (zero)
constexpr int NY = 4096;
constexpr double CHI_LO = 1e-6, CHI_HI = 1e4;
constexpr int NC = 801;
constexpr int NU = 8000;                         // Simpson intervals per part of the u integral (even)
constexpr double CHB_HI = 1e4;                   // Breit-Wheeler: chi_gamma from QED_BW_CHI_MIN
constexpr int NB = 701;
}  // namespace

// K_nu(y) = int_0^inf exp(-y cosh t) cosh(nu t) dt, IK13(y) = int_y^inf K13 = int_0^inf exp(-y cosh t) cosh(t/3) / cosh t dt;
// the trapezoidal rule on the whole line converges exponentially (analytic integrand); h resolves the width 1/sqrt(y)
void qed_bessel_exact(double y, double& k13, double& k23, double& ik13) {
    const double h = std::min(0.02, 0.3 / std::sqrt(y));
    const double tmax = y < 745 ? std::acosh(745.0 / y) : 0.0;
    double a = 0, b = 0, c = 0;
    const int n = int(tmax / h) + 1;
    for (int i = 0; i <= n; ++i) {
        const double t = i * h, ch = std::cosh(t), e = std::exp(-y * ch), f = i == 0 ? 0.5 : 1.0;
        a += f * e * std::cosh(t / 3);
        b += f * e * std::cosh(2 * t / 3);
        c += f * e * std::cosh(t / 3) / ch;
    }
    k13 = a * h; k23 = b * h; ik13 = c * h;
}

double qed_hw(double lambda0) {
    const double hbar_c = 197.3269804e-15, mc2 = 0.51099895000;   // hbar c [MeV m], m c^2 [MeV]
    return 2 * M_PI * hbar_c / lambda0 / mc2;
}

QedTab QedTables::view_host() const {
    QedTab T;
    T.fy = fy.data(); T.ny = ny; T.ly0 = ly0; T.dly = dly;
    T.fc = fc.data(); T.nc = nc; T.lc0 = lc0; T.dlc = dlc;
    T.fb = fb.data(); T.nb = nb; T.lb0 = lb0; T.dlb = dlb;
    return T;
}

static QedTables build() {
    QedTables Q;
    Q.ny = NY; Q.ly0 = std::log(Y_LO); Q.dly = (std::log(Y_HI) - Q.ly0) / (NY - 1);
    Q.fy.resize(3 * size_t(NY));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int k = 0; k < NY; ++k) {
        double a, b, c;
        qed_bessel_exact(std::exp(Q.ly0 + k * Q.dly), a, b, c);
        Q.fy[3 * size_t(k)] = std::log(a); Q.fy[3 * size_t(k) + 1] = std::log(b); Q.fy[3 * size_t(k) + 2] = std::log(c);
    }
    Q.nc = NC; Q.lc0 = std::log(CHI_LO); Q.dlc = (std::log(CHI_HI) - Q.lc0) / (NC - 1);
    Q.fc.resize(4 * size_t(NC));
    const QedTab T = Q.view_host();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int k = 0; k < NC; ++k) {
        const double chi = std::exp(Q.lc0 + k * Q.dlc);
        double W = 0, G = 0, M = 0, P = 0;
        // f(u) = (w, u K13); part 1: u = e^t, t in [ln(1e-30 chi), ln 1/2]; part 2: u = 1 - e^v, v in [-60, ln 1/2]
        for (int part = 0; part < 2; ++part) {
            const double lo = part == 0 ? std::log(1e-30 * chi) : -60.0, hi = std::log(0.5), hh = (hi - lo) / NU;
            for (int i = 0; i <= NU; ++i) {
                const double t = lo + i * hh, e = std::exp(t), u = part == 0 ? e : 1 - e, om = part == 0 ? 1 - e : e;
                const double y = 2 * u / (3 * om * chi);
                double k13, k23, ik13;
                qed_bessel(T, y, k13, k23, ik13);
                const double w = (u * u - 2 * u + 2) / om * k23 - ik13, g = u * k13;
                const double sw = (i == 0 || i == NU ? 1.0 : (i % 2 ? 4.0 : 2.0)) * hh / 3 * e;   // du = e dt
                W += sw * w; G += sw * g; P += sw * u * w;
                M = std::max(M, 3 * std::cbrt(u * u) * (w + g));
            }
        }
        Q.fc[4 * size_t(k)] = std::log(W); Q.fc[4 * size_t(k) + 1] = std::log(G);
        Q.fc[4 * size_t(k) + 2] = std::log(1.05 * M); Q.fc[4 * size_t(k) + 3] = std::log(P);
    }
    // Breit-Wheeler: T0 = int_0^1 B0 dx, T3 = int_0^1 K23 dx (symmetric in x <-> 1 - x: twice the half [x_lo, 1/2] in ln x;
    // below x_lo rho > 720), Mb = 1.05 max (B0 + K23); all times exp(8/(3 chi))
    Q.nb = NB; Q.lb0 = std::log(QED_BW_CHI_MIN); Q.dlb = (std::log(CHB_HI) - Q.lb0) / (NB - 1);
    Q.fb.resize(3 * size_t(NB));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int k = 0; k < NB; ++k) {
        const double chi = std::exp(Q.lb0 + k * Q.dlb), ex = 8.0 / (3 * chi);
        const double xlo = std::min(0.45, std::max(1e-30, 0.9 * 2 / (3 * Y_HI * chi)));
        const double lo = std::log(xlo), hi = std::log(0.5), hh = (hi - lo) / NU;
        double T0 = 0, T3 = 0, M = 0;
        for (int i = 0; i <= NU; ++i) {
            const double x = std::exp(lo + i * hh), xm = 1 - x, rho = 2 / (3 * chi * x * xm);
            double k13, k23, ik13;
            qed_bessel(T, rho, k13, k23, ik13);
            const double s = std::exp(ex);                     // chi >= 0.01: at most e^267
            const double b0 = ((x * x + xm * xm) / (x * xm) * k23 + ik13) * s, b3 = k23 * s;
            const double sw = (i == 0 || i == NU ? 1.0 : (i % 2 ? 4.0 : 2.0)) * hh / 3 * x;   // dx = x dt
            T0 += 2 * sw * b0; T3 += 2 * sw * b3;
            M = std::max(M, b0 + b3);
        }
        Q.fb[3 * size_t(k)] = std::log(T0); Q.fb[3 * size_t(k) + 1] = std::log(T3); Q.fb[3 * size_t(k) + 2] = std::log(1.05 * M);
    }
    return Q;
}

const QedTables& qed_tables() {
    static const QedTables Q = build();
    return Q;
}
