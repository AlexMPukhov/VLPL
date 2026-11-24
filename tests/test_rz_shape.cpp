// The transverse particle shape of RZ (numerics.rz_transverse_shape = gaussian, decision 79, src/rz_smooth.cpp): S =
// exp(a^2/2 Lap_perp) per mode against the heat kernel of the plane. For h_m = (x + i y)^m (harmonic, degree m),
//   exp(t Lap) [h_m exp(-r^2 / w^2)] = (w^2 / (w^2 + 4 t))^(1 + m) h_m exp(-r^2 / (w^2 + 4 t)),  t = a^2 / 2,
// so a scalar of mode m with the profile r^m exp(-r^2/w^2) and the vector field E_x = exp(-r^2/w^2) (mode 1: E_r1 = g/2,
// E_phi1 = i g/2) have analytic images. Scalars converge as dr^2; the vector field as dr in the first two cells next to the
// axis (the axis term of the radial operators, 4.7e-3 -> 1.7e-3 of the peak at w = 8 dr -> 16 dr), as dr^2 beyond. And S is self-adjoint in the field energy's inner product (the deposit and the
// gather exchange the same energy).
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include "common.hpp"
#include "rz.hpp"

namespace {
HD inline int slot(int m, int c) { return m == 0 ? 0 : 2 * m - 1 + c; }
}  // namespace

void test_rz_shape(Comm& comm) {
    if (comm.size > 1) return;                       // host copies of the whole grid (MPI: tests/mpi.sh)
    for (int q = 1; q <= 2; ++q) {                   // dr and dr / 2: the error is second order
    const int nz = 4, nr = 96 * q, M = 4;
    const double dz = 0.05, dr = 0.05 / q, w = 0.4, a = 0.1;
    Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, Solver::RipCC);
    FieldsRZ f(d, M);
    f.smooth_a = a;
    const RadialGrid& g = f.grid();
    const double W2 = w * w + 2 * a * a;
    auto amp = [&](int m) { return std::pow(w * w / W2, 1 + m); };
    // scalars of mode m = 0, 1, 2, 3 at the nodes (E_z), profile r^m exp(-r^2/w^2) in the Re slot
    for (int m = 0; m < M; ++m) {
        std::vector<double> h(f.ez.size(), 0.0);
        for (int i = 0; i < nz; ++i)
            for (int j = 0; j < nr; ++j) h[f.ez.idx(i, j, slot(m, 0))] = std::pow(g.r[size_t(j)], m) * std::exp(-g.r[size_t(j)] * g.r[size_t(j)] / (w * w));
        f.ez.upload(h);
        rz_smooth_apply(f, f.ez, 0, nullptr);
        const std::vector<double> o = f.ez.host();
        double err = 0, pk = 0;
        for (int i = 0; i < nz; ++i)
            for (int j = 0; j < nr / 2; ++j) {
                const double r = g.r[size_t(j)], ref = amp(m) * std::pow(r, m) * std::exp(-r * r / W2);
                err = std::max(err, std::abs(o[f.ez.idx(i, j, slot(m, 0))] - ref));
                pk = std::max(pk, std::abs(ref));
            }
        std::printf("  S1 scalar, mode %d, r^m exp(-r^2/w^2), w = 8 dr0, a = 2 dr0, dr = dr0 / %d: max |S X - heat kernel| / peak %.2e (peak drops to %.3f)\n",
                    m, q, err / pk, amp(m) * std::pow(W2 / (w * w), m / 2.0));
        check("S1 transverse shape: scalar mode " + std::to_string(m) + " against the heat kernel, dr / " + std::to_string(q), err < 3e-3 * pk / (q * q), fmt("%.1e", err / pk));
    }
    // the vector field E_x = exp(-r^2/w^2): mode 1, E_r1 = g(rh)/2 (Re), E_phi1 = i g(r)/2 (Im)
    {
        std::vector<double> hr(f.er.size(), 0.0), hp(f.ep.size(), 0.0);
        auto G = [&](double r, double ww) { return std::exp(-r * r / ww); };
        for (int i = 0; i < nz; ++i)
            for (int j = 0; j < nr; ++j) {
                hr[f.er.idx(i, j, slot(1, 0))] = 0.5 * G(g.rh[size_t(j)], w * w);
                if (j > 0) hp[f.ep.idx(i, j, slot(1, 1))] = 0.5 * G(g.r[size_t(j)], w * w);
            }
        f.er.upload(hr); f.ep.upload(hp);
        rz_smooth_apply(f, f.er, 1, &f.ep);
        const std::vector<double> orr = f.er.host(), op = f.ep.host();
        double er = 0, ep = 0, other = 0;
        for (int i = 0; i < nz; ++i)
            for (int j = 0; j < nr / 2; ++j) {
                er = std::max(er, std::abs(orr[f.er.idx(i, j, slot(1, 0))] - 0.5 * amp(0) * G(g.rh[size_t(j)], W2)));
                if (j > 0) ep = std::max(ep, std::abs(op[f.ep.idx(i, j, slot(1, 1))] - 0.5 * amp(0) * G(g.r[size_t(j)], W2)));
                other = std::max({other, std::abs(orr[f.er.idx(i, j, slot(1, 1))]), std::abs(op[f.ep.idx(i, j, slot(1, 0))])});
            }
        std::printf("  S2 vector E_x = exp(-r^2/w^2) (mode 1), dr = dr0 / %d: max |S E - heat kernel| / peak: E_r %.2e, E_phi %.2e, wrong slots %.1e\n",
                    q, er / 0.5, ep / 0.5, other);
        check("S2 transverse shape: the vector field E_x against the heat kernel, dr / " + std::to_string(q), er < 6e-3 / q && ep < 2.4e-3 / q && other < 1e-14,
              fmt("%.1e %.1e", er / 0.5, ep / 0.5));
    }
    // self-adjoint: <S u, v> = <u, S v> with the energy weights (E_r: B_j, E_phi, E_z: A_j; modes m >= 1 twice), random u, v
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> U(-1, 1);
        auto rnd = [&](Field& F, bool node) {
            std::vector<double> h(F.size(), 0.0);
            for (int k = 0; k < 2 * M - 1; ++k)
                for (int i = 0; i < nz; ++i)
                    for (int j = 0; j < nr; ++j)
                        if (!(node && j == 0 && k > 0)) h[F.idx(i, j, k)] = U(rng);
            F.upload(h);
        };
        auto dot = [&](const Field& A, const Field& B, const std::vector<double>& wt) {
            const std::vector<double> x = A.host(), y = B.host();
            double s = 0;
            for (int k = 0; k < 2 * M - 1; ++k)
                for (int i = 0; i < nz; ++i)
                    for (int j = 0; j < nr; ++j) s += (k ? 2.0 : 1.0) * wt[size_t(j)] * x[A.idx(i, j, k)] * y[B.idx(i, j, k)];
            return s;
        };
        Field ur(d), up(d), vr(d), vp(d), us(d), vs(d);
        rnd(ur, false); rnd(up, true); rnd(vr, false); rnd(vp, true); rnd(us, true); rnd(vs, true);
        {   // V_phi on the axis is no DOF: 0 for all modes
            for (Field* F : {&up, &vp}) {
                std::vector<double> h = F->host();
                for (int k = 0; k < 2 * M - 1; ++k) for (int i = 0; i < nz; ++i) h[F->idx(i, 0, k)] = 0;
                F->upload(h);
            }
        }
        Field sur(d), sup(d), svr(d), svp(d), sus(d), svs(d);
        sur.upload(ur.host()); sup.upload(up.host()); svr.upload(vr.host()); svp.upload(vp.host()); sus.upload(us.host()); svs.upload(vs.host());
        rz_smooth_apply(f, sur, 1, &sup); rz_smooth_apply(f, svr, 1, &svp);
        rz_smooth_apply(f, sus, 0, nullptr); rz_smooth_apply(f, svs, 0, nullptr);
        const std::vector<double> A(g.A.begin(), g.A.begin() + nr), B(g.B.begin(), g.B.begin() + nr);
        const double l = dot(sur, vr, B) + dot(sup, vp, A), r = dot(ur, svr, B) + dot(up, svp, A);
        const double ls = dot(sus, vs, A), rs = dot(us, svs, A);
        const double n = std::sqrt((dot(ur, ur, B) + dot(up, up, A)) * (dot(vr, vr, B) + dot(vp, vp, A)));
        const double ns = std::sqrt(dot(us, us, A) * dot(vs, vs, A));
        std::printf("  S3 self-adjoint in the energy inner product (random fields, %d modes): vector %.1e, scalar %.1e\n", M,
                    std::abs(l - r) / n, std::abs(ls - rs) / ns);
        check("S3 transverse shape: self-adjoint (deposit and gather fit)", std::abs(l - r) < 1e-12 * n && std::abs(ls - rs) < 1e-12 * ns,
              fmt("%.1e %.1e", std::abs(l - r) / n, std::abs(ls - rs) / ns));
    }
    }
}
