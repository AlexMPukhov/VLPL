#include "load.hpp"
#include <stdexcept>
#if RIP_GPU
#include <cub/cub.cuh>
#endif

namespace {

// Particle p of the cell (gi, gj, gk) (a fastest): its position, and its density factor, 0 if it is
// not loaded. code: the density function (n = 0: none).
HD inline double cold_particle(const ColdLoad& L, const Expr::Ins* code, int n, long gi, long gj, long gk, int p,
                               double r[3]) {
    const int a = p % L.ppc[0], b = (p / L.ppc[0]) % L.ppc[1], c = p / (L.ppc[0] * L.ppc[1]);
    r[0] = (gi + (a + 0.5) / L.ppc[0]) * L.h[0];
    r[1] = (gj + (b + 0.5) / L.ppc[1]) * L.h[1];
    r[2] = (gk + (c + 0.5) / L.ppc[2]) * L.h[2];
    if (!(r[0] >= L.qlo && r[0] < L.qhi)) return 0;
    const double q0 = r[0];
    r[0] = L.ax * q0 + L.bx;                  // the deck's coordinate (boost: lab x; plain: q0 itself)
    bool in = true;
    for (int q = 0; q < 3; ++q) in = in && r[q] >= L.rmin[q] && r[q] < L.rmax[q];
    double pf = in ? 1.0 : 0.0;
    if (in && L.slab.T > 0) {                // the point test, or the sub-cell mean of the profile (slab.hpp)
        const double hs[2] = {L.ax * L.h[0] / L.ppc[0], L.h[1] / L.ppc[1]};
        pf = slab::factor(L.slab, r, hs);
        if (!(pf > 0)) pf = 0;
    }
    if (pf > 0 && n > 0) {
        const double f = expr_eval(code, n, r);
        pf = f > 0 ? pf * f : 0;              // also NaN
    }
    r[0] = q0 + L.sx;                         // the stored position
    return pf;
}

struct LoadScratch { Buf<int> cnt, off; Buf<double> fmax; Buf<Expr::Ins> code; Buf<unsigned char> tmp; };
LoadScratch& scratch() {                      // kept between calls (never freed)
    static LoadScratch* w = new LoadScratch;
    return *w;
}
template <class T> void ensure(Buf<T>& b, size_t n) { if (b.size() < n) b.resize(n); }

}  // namespace

double load_cold(Species& s, const ColdLoad& L, const Expr& fn, const long g0[3], const int n[3]) {
    const long nc = long(n[0]) * n[1] * n[2];
    if (nc <= 0) return 0;
    if (fn.max_stack() > Expr::MAX_STACK) throw std::invalid_argument("expression too deeply nested: " + fn.text());
    LoadScratch& W = scratch();
    ensure(W.cnt, size_t(nc)); ensure(W.off, size_t(nc) + 1); ensure(W.fmax, size_t(nc));
    const int ncode = int(fn.program().size());
    ensure(W.code, size_t(ncode));
    W.code.from_host(fn.program().data(), 0, size_t(ncode));
    const Expr::Ins* code = W.code.data();
    int* C = W.cnt.data(); int* O = W.off.data(); double* FM = W.fmax.data();
    const int n0 = n[0], n1 = n[1], ppc = L.ppc[0] * L.ppc[1] * L.ppc[2];
    const long gi0 = g0[0], gj0 = g0[1], gk0 = g0[2];
    // 1. loaded particles and the largest density factor per cell (cells in the order k, j, i)
    par_for(nc, [=] HD(long q) {
        const long gi = gi0 + q % n0, gj = gj0 + (q / n0) % n1, gk = gk0 + q / (long(n0) * n1);
        int m = 0;
        double fm = 0, r[3];
        for (int p = 0; p < ppc; ++p) {
            const double f = cold_particle(L, code, ncode, gi, gj, gk, p, r);
            if (f > 0) { ++m; fm = f > fm ? f : fm; }
        }
        C[q] = m; FM[q] = fm;
    });
    // 2. first particle of each cell, O[nc] = total
#if RIP_GPU
    size_t bytes = 0;
    gpu_check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, C, O, int(nc)), "cub scan (size)");
    ensure(W.tmp, bytes);
    gpu_check(cub::DeviceScan::ExclusiveSum(W.tmp.data(), bytes, C, O, int(nc)), "cub scan");
    par_for(1, [=] HD(long) { O[nc] = O[nc - 1] + C[nc - 1]; });
#else
    for (long q = 0, t = 0; q <= nc; ++q) { O[q] = int(t); if (q < nc) t += C[q]; }
#endif
    int total = 0;
    W.off.to_host(&total, size_t(nc), 1);
    const double fmax = ncode > 0 || L.slab.exact ? par_max_abs(nc, [=] HD(long q) { return FM[q]; }) : (total > 0 ? 1.0 : 0.0);
    if (total == 0) return fmax;
    // 3. the particles, from index first on in the order of the host loop
    const long first = long(s.extend(size_t(total)));
    double* X = s.ptr<double>(Species::X); double* Y = s.ptr<double>(Species::Y); double* Z = s.ptr<double>(Species::Z);
    double* UX = s.ptr<double>(Species::UX); double* UY = s.ptr<double>(Species::UY); double* UZ = s.ptr<double>(Species::UZ);
    double* Wt = s.ptr<double>(Species::W);
    par_for(nc, [=] HD(long q) {
        if (C[q] == 0) return;
        const long gi = gi0 + q % n0, gj = gj0 + (q / n0) % n1, gk = gk0 + q / (long(n0) * n1);
        long o = first + O[q];
        double r[3];
        for (int p = 0; p < ppc; ++p) {
            const double f = cold_particle(L, code, ncode, gi, gj, gk, p, r);
            if (!(f > 0)) continue;
            X[o] = r[0]; Y[o] = r[1]; Z[o] = r[2];
            UX[o] = L.u0[0]; UY[o] = L.u0[1]; UZ[o] = L.u0[2];
            Wt[o] = L.w * f;
            ++o;
        }
    });
    return fmax;
}
