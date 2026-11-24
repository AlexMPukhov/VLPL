#include "pml.hpp"
#include "metal3d.hpp"
#include "side.hpp"
#include "tfsf.hpp"
#include <cmath>
#include "fields.hpp"

Pml::Pml(const Domain& d, const PmlConfig& c, double dt, double bmin) : cfg(c) {
    for (int a = 1; a < 3; ++a) {
        if (!c.on[a]) continue;
        const int N = d.N[a], L = c.cells;
        const double h = d.h[a], Ly = L * h;
        const double smax = (c.order + 1) * std::log(1 / c.R0) / (2 * Ly);       // c = 1
        std::vector<double> an(static_cast<size_t>(N)), bn(static_cast<size_t>(N)), ah(static_cast<size_t>(N)), bh(static_cast<size_t>(N));
        for (int g = 0; g < N; ++g)
            for (int half = 0; half < 2; ++half) {
                const double y = (g + 0.5 * half) * h;
                const double dep = std::max(std::max(Ly - y, y - (N - L) * h), 0.0) / Ly;
                const double s = smax * std::pow(dep, c.order), b = std::max(std::exp(-s * dt), bmin);
                (half ? bh : bn)[size_t(g)] = b;
                (half ? ah : an)[size_t(g)] = s > 0 ? b - 1 : 0.0;
            }
        for (auto* p : {&a_n[a], &b_n[a], &a_h[a], &b_h[a]}) p->resize(size_t(N));
        a_n[a].from_host(an.data(), 0, an.size()); b_n[a].from_host(bn.data(), 0, bn.size());
        a_h[a].from_host(ah.data(), 0, ah.size()); b_h[a].from_host(bh.data(), 0, bh.size());
    }
    // psi only on ranks whose cells reach into a layer (zero-initialised)
    auto touches = [&](int a) { return c.on[a] && (d.off[a] < c.cells + 1 || d.off[a] + d.n[a] > d.N[a] - c.cells - 1); };
    if (touches(1))
        for (Field* f : {&p_bz_y, &p_ez_y, &t_bx_y, &t_ex_y, &c_bz_y, &c_ez_y}) f->resize(d);
    if (touches(2))
        for (Field* f : {&p_by_z, &p_ey_z, &t_bx_z, &t_ex_z, &c_by_z, &c_ey_z}) f->resize(d);
    if (c.on[1] || c.on[2])
        for (Field* f : {&s_py, &s_pz, &s_fy, &s_fz}) f->resize(d);
}

std::vector<Field*> Pml::psi_fields() {
    std::vector<Field*> v;
    for (Field* f : {&p_bz_y, &p_ez_y, &p_by_z, &p_ey_z, &t_bx_y, &t_ex_y, &t_bx_z, &t_ex_z, &c_bz_y, &c_ez_y,
                     &c_by_z, &c_ey_z})
        if (f->n(0) > 0) v.push_back(f);
    return v;
}

namespace {
// CPML derivative: g + psi with psi <- b psi + a g, only where a != 0 (inside a layer)
struct Cp {
    const double *a, *b;
    int off;
    HD double operator()(double g, const FV& P, int jl, int i, int j, int k) const {
        if (!a) return g;
        const double aa = a[off + jl];
        if (aa == 0.0) return g;
        double& psi = P(i, j, k);
        psi = b[off + jl] * psi + aa * g;
        return g + psi;
    }
};
HD inline double mux(const FV& f, int i, int j, int k) { return 0.5 * (f(i, j, k) + f(i + 1, j, k)); }
}  // namespace

void rip_cc_step_pml(Fields& F, Pml& P, Comm& comm, const BoundaryHooks& hooks, double t) {
    const Domain& d = F.d_;
    const double D = d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    comm.exchange(d, {&F.jx, &F.jy, &F.jz});
    const FV Ex = F.ex.view(), Ey = F.ey.view(), Ez = F.ez.view(), Bx = F.bx.view(), By = F.by.view(), Bz = F.bz.view();
    const FV Jx = F.jx.view(), Jy = F.jy.view(), Jz = F.jz.view();
    const FV Exs = F.exs_.view(), Bxs = F.bxs_.view();
    const bool oy = P.cfg.on[1], oz = P.cfg.on[2];
    const Cp yn{oy ? P.a_n[1].data() : nullptr, oy ? P.b_n[1].data() : nullptr, d.off[1]};
    const Cp yh{oy ? P.a_h[1].data() : nullptr, oy ? P.b_h[1].data() : nullptr, d.off[1]};
    const Cp zn{oz ? P.a_n[2].data() : nullptr, oz ? P.b_n[2].data() : nullptr, d.off[2]};
    const Cp zh{oz ? P.a_h[2].data() : nullptr, oz ? P.b_h[2].data() : nullptr, d.off[2]};

    // 1. predictor (Ex at node j, k; Bx at half j, k)
    const FV PBZ = P.p_bz_y.view(), PEZ = P.p_ez_y.view(), PBY = P.p_by_z.view(), PEY = P.p_ey_z.view();
    for_owned(d, [=] HD(int i, int j, int k) {
        const double gbz = yn((Bz(i, j, k) - Bz(i, j - 1, k)) * iy, PBZ, j, i, j, k);
        const double gby = zn((By(i, j, k) - By(i, j, k - 1)) * iz, PBY, k, i, j, k);
        const double gez = yh((Ez(i, j + 1, k) - Ez(i, j, k)) * iy, PEZ, j, i, j, k);
        const double gey = zh((Ey(i, j, k + 1) - Ey(i, j, k)) * iz, PEY, k, i, j, k);
        Exs(i, j, k) = Ex(i, j, k) + 0.5 * D * (gbz - gby - Jx(i, j, k));
        Bxs(i, j, k) = Bx(i, j, k) - 0.5 * D * (gez - gey);
    });
    if (F.metal) metal_fix_pred(d, F.metal->view(), Exs, Bxs, 0);   // embedded metal (metal3d.hpp)
    if (F.side) F.side->fix_pred(Exs, Bxs);        // the side tube (faces outside the layers)
    comm.exchange(d, {&F.exs_, &F.bxs_});

    for (auto& h : hooks.inflow) h(F, t);

    // 2a. the transport sources at (a + 1/2), a = -1 .. n-1, once per location (each updates its psi)
    const FV SPY = P.s_py.view(), SPZ = P.s_pz.view(), SFY = P.s_fy.view(), SFZ = P.s_fz.view();
    const FV TBY = P.t_bx_y.view(), TEY = P.t_ex_y.view(), TBZ = P.t_bx_z.view(), TEZ = P.t_ex_z.view();
    const int nx1 = d.n[0] + 1, ny = d.n[1];
    par_for(long(nx1) * d.n[1] * d.n[2], [=] HD(long q) {
        const int a = int(q % nx1) - 1, j = int((q / nx1) % ny), k = int(q / (long(nx1) * ny));
        const double gbxz = zn((mux(Bxs, a, j, k) - mux(Bxs, a, j, k - 1)) * iz, TBZ, k, a, j, k);
        const double gbxy = yn((mux(Bxs, a, j, k) - mux(Bxs, a, j - 1, k)) * iy, TBY, j, a, j, k);
        const double gexz = zh((mux(Exs, a, j, k + 1) - mux(Exs, a, j, k)) * iz, TEZ, k, a, j, k);
        const double gexy = yh((mux(Exs, a, j + 1, k) - mux(Exs, a, j, k)) * iy, TEY, j, a, j, k);
        SPY(a, j, k) = gbxz - Jy(a, j, k);
        SPZ(a, j, k) = -gbxy - Jz(a, j, k);
        SFY(a, j, k) = -gexz;
        SFZ(a, j, k) = gexy;
    });

    // 2b. transport of T+- along x
    const FV nEy = F.ney_.view(), nEz = F.nez_.view(), nBy = F.nby_.view(), nBz = F.nbz_.view();
    for_owned(d, [=] HD(int i, int j, int k) {
        const double tpy = Ey(i - 1, j, k) + Bz(i - 1, j, k) + D * (SPY(i - 1, j, k) + SFZ(i - 1, j, k));
        const double tmy = Ey(i + 1, j, k) - Bz(i + 1, j, k) + D * (SPY(i, j, k) - SFZ(i, j, k));
        const double tpz = Ez(i + 1, j, k) + By(i + 1, j, k) + D * (SPZ(i, j, k) + SFY(i, j, k));
        const double tmz = Ez(i - 1, j, k) - By(i - 1, j, k) + D * (SPZ(i - 1, j, k) - SFY(i - 1, j, k));
        nEy(i, j, k) = 0.5 * (tpy + tmy);  nBz(i, j, k) = 0.5 * (tpy - tmy);
        nEz(i, j, k) = 0.5 * (tpz + tmz);  nBy(i, j, k) = 0.5 * (tpz - tmz);
    });
    if (F.tfsf) F.tfsf->correct(comm, F.ney_, F.nez_, F.nby_, F.nbz_, 0);   // the incident T across the TFSF plane
    if (F.side) F.side->fix_transport(nEy, nEz, nBy, nBz, 0);
    if (F.metal) metal_fix_transport(d, F.metal->view(), nEy, nEz, nBy, nBz);
    comm.exchange(d, {&F.ney_, &F.nez_, &F.nby_, &F.nbz_});

    // 3. corrector with the time-centred transverse fields
    const FV CBZ = P.c_bz_y.view(), CEZ = P.c_ez_y.view(), CBY = P.c_by_z.view(), CEY = P.c_ey_z.view();
    for_owned(d, [=] HD(int i, int j, int k) {
        const double bz0 = 0.5 * (Bz(i, j, k) + nBz(i, j, k)), bzm = 0.5 * (Bz(i, j - 1, k) + nBz(i, j - 1, k));
        const double by0 = 0.5 * (By(i, j, k) + nBy(i, j, k)), bym = 0.5 * (By(i, j, k - 1) + nBy(i, j, k - 1));
        const double ez0 = 0.5 * (Ez(i, j, k) + nEz(i, j, k)), ezp = 0.5 * (Ez(i, j + 1, k) + nEz(i, j + 1, k));
        const double ey0 = 0.5 * (Ey(i, j, k) + nEy(i, j, k)), eyp = 0.5 * (Ey(i, j, k + 1) + nEy(i, j, k + 1));
        const double gbz = yn((bz0 - bzm) * iy, CBZ, j, i, j, k), gby = zn((by0 - bym) * iz, CBY, k, i, j, k);
        const double gez = yh((ezp - ez0) * iy, CEZ, j, i, j, k), gey = zh((eyp - ey0) * iz, CEY, k, i, j, k);
        Ex(i, j, k) += D * (gbz - gby - Jx(i, j, k));
        Bx(i, j, k) -= D * (gez - gey);
    });
    if (F.side) F.side->fix_corr(Ex, Bx);
    if (F.metal) metal_fix_corr(d, F.metal->view(), Ex, Bx);
    F.ey.swap(F.ney_); F.ez.swap(F.nez_); F.by.swap(F.nby_); F.bz.swap(F.nbz_);
    comm.exchange(d, {&F.ex, &F.bx});
}
