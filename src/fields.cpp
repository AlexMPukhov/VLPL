#include "fields.hpp"
#include "prof.hpp"
#include "metal3d.hpp"
#include "dielectric3d.hpp"
#include "side.hpp"
#include "tfsf.hpp"
#include <algorithm>
#include <cmath>

Fields::Fields(const Domain& d) : d_(d) {
    for (Field* f : {&ex, &ey, &ez, &bx, &by, &bz, &jx, &jy, &jz, &rho, &exs_, &bxs_, &ney_, &nez_, &nby_, &nbz_})
        f->resize(d);
}

void Fields::zero_currents() { jx.fill(0); jy.fill(0); jz.fill(0); }

HD inline double mux(const FV& f, int i, int j, int k) { return 0.5 * (f(i, j, k) + f(i + 1, j, k)); }

// The damping of the x layers (O3), on the whole array: the factor depends only on the global x index, so the guard
// copies get the same factor as their originals and no exchange is needed.
void fields_absorb_x(const Domain& d, int L, double kappa, const FV A, const FV B, const FV C, const FV D, const FV E, const FV F) {
    const int g = d.ng, gz = d.gw(2), sx = d.n[0] + 2 * g, sy = d.n[1] + 2 * g, sz = d.ext(2), N = d.N[0];
    const long c0 = d.cell0(0) - d.wx;
    par_for(long(sx) * sy * sz, [=] HD(long q) {
        const int i = int(q % sx) - g, j = int((q / sx) % sy) - g, k = int(q / (long(sx) * sy)) - gz;
        const long gi = c0 + i;
        double s = 0;
        if (gi < L) s = double(L - gi) / L;
        else if (gi > N - 1 - L) s = double(gi - (N - 1 - L)) / L;
        if (s <= 0) return;
        const double f = exp(-kappa * s * s);
        A(i, j, k) *= f; B(i, j, k) *= f; C(i, j, k) *= f; D(i, j, k) *= f; E(i, j, k) *= f; F(i, j, k) *= f;
    });
}

void Fields::step(Comm& comm, const BoundaryHooks& hooks, double t) {
    step_core(comm, hooks, t);
    if (xabs.cells > 0) fields_absorb_x(d_, xabs.cells, xabs.kappa, ex.view(), ey.view(), ez.view(), bx.view(), by.view(), bz.view());
    if (dfront > 0) dielectric3_absorb_front(*this, d_, dfront, dfront_kappa);   // dielectric + window: the front layer (decision 100)
}

void Fields::step_core(Comm& comm, const BoundaryHooks& hooks, double t) {
    if (d_.solver == Solver::Yee) { yee_step(*this, d_, comm); return; }   // periodic boxes: no hooks
    if (side) side->begin(t);                      // the side TFSF tube at t_n (side.hpp)
    if (d_.solver == Solver::IRip) { irip_step(*this, d_, comm, hooks, t); return; }
    if (tfsf) tfsf->begin(comm, t);                // the incident grid of the TFSF source to t_n+1 (or its plane waves at t_n)
    if (pml) { rip_cc_step_pml(*this, *pml, comm, hooks, t); return; }
    const double D = d_.h[0], iy = 1.0 / d_.h[1], iz = 1.0 / d_.h[2];
    if (!j_ready) { PROF_SCOPE("rip_cc.xchg J"); comm.exchange(d_, {&jx, &jy, &jz}); }   // sources at i-1/2 need the left guard
    j_ready = false;                               // done during the push (PushDeposit::apply_overlap)
    const FV Ex = ex.view(), Ey = ey.view(), Ez = ez.view(), Bx = bx.view(), By = by.view(), Bz = bz.view();
    const FV Jx = jx.view(), Jy = jy.view(), Jz = jz.view();
    const FV Exs = exs_.view(), Bxs = bxs_.view();
    const bool de = diel != nullptr;                // dielectric: (1 + chi_x) d_t E_x = curl B - J_x (decision 100)
    const Diel3View dv = de ? diel->view() : Diel3View{};

    // 1. predictor for the longitudinal fields at n+1/2. The transport reads it one cell beyond the owned cells; it is
    // computed there too, from the guards of the fields and currents (copies of the neighbours' cells, 0 at open
    // edges), instead of being exchanged: the same values as the exchange gives (the same formula on the same
    // numbers), three messages per step fewer. With a side tube its face corrections need the exchange.
    auto pred = [=] HD(int i, int j, int k) {
        const double sx = (Bz(i, j, k) - Bz(i, j - 1, k)) * iy - (By(i, j, k) - By(i, j, k - 1)) * iz - Jx(i, j, k);
        Exs(i, j, k) = Ex(i, j, k) + (de ? 0.5 * D / (1 + dv.cx(i, j, k)) : 0.5 * D) * sx;
        Bxs(i, j, k) = Bx(i, j, k) - 0.5 * D * ((Ez(i, j + 1, k) - Ez(i, j, k)) * iy -
                                                (Ey(i, j, k + 1) - Ey(i, j, k)) * iz);
    };
    if (!side) {
        for_owned_halo(d_, 1, pred);
        if (metal) metal_fix_pred(d_, metal->view(), Exs, Bxs, 1);   // embedded metal (metal3d.hpp)
    } else {
        for_owned(d_, pred);
        if (metal) metal_fix_pred(d_, metal->view(), Exs, Bxs, 0);
        side->fix_pred(Exs, Bxs);                 // differences across the faces of the side tube
        PROF_SCOPE("rip_cc.xchg pred");
        comm.exchange(d_, {&exs_, &bxs_});
    }

    for (auto& h : hooks.inflow) h(*this, t);

    // 2. transport of T+- along x with the sources at (i+1/2, n+1/2)
    const FV nEy = ney_.view(), nEz = nez_.view(), nBy = nby_.view(), nBz = nbz_.view();
    for_owned(d_, [=] HD(int i, int j, int k) {
        auto psiy = [&](int a) { return (mux(Bxs, a, j, k) - mux(Bxs, a, j, k - 1)) * iz - Jy(a, j, k); };
        auto psiz = [&](int a) { return -(mux(Bxs, a, j, k) - mux(Bxs, a, j - 1, k)) * iy - Jz(a, j, k); };
        auto phiy = [&](int a) { return -(mux(Exs, a, j, k + 1) - mux(Exs, a, j, k)) * iz; };
        auto phiz = [&](int a) { return (mux(Exs, a, j + 1, k) - mux(Exs, a, j, k)) * iy; };
        const double tpy = Ey(i - 1, j, k) + Bz(i - 1, j, k) + D * (psiy(i - 1) + phiz(i - 1));
        const double tmy = Ey(i + 1, j, k) - Bz(i + 1, j, k) + D * (psiy(i) - phiz(i));
        const double tpz = Ez(i + 1, j, k) + By(i + 1, j, k) + D * (psiz(i) + phiy(i));
        const double tmz = Ez(i - 1, j, k) - By(i - 1, j, k) + D * (psiz(i - 1) - phiy(i - 1));
        nEy(i, j, k) = 0.5 * (tpy + tmy);  nBz(i, j, k) = 0.5 * (tpy - tmy);
        nEz(i, j, k) = 0.5 * (tpz + tmz);  nBy(i, j, k) = 0.5 * (tpz - tmz);
    });
    if (tfsf) tfsf->correct(comm, ney_, nez_, nby_, nbz_, 0);   // the incident T across the TFSF plane
    if (side) side->fix_transport(nEy, nEz, nBy, nBz, 0);
    if (metal) metal_fix_transport(d_, metal->view(), nEy, nEz, nBy, nBz);
    { PROF_SCOPE("rip_cc.xchg T"); comm.exchange(d_, {&ney_, &nez_, &nby_, &nbz_}); }
    if (de) {                                      // the polarization current of E_y, E_z: lines along x (decision 100)
        diel->correct(comm, d_, *this, ney_, nez_, nby_, nbz_);
        comm.exchange(d_, {&ney_, &nez_, &nby_, &nbz_});
    }

    // 3. corrector with the time-centred transverse fields (this is what conserves charge)
    for_owned(d_, [=] HD(int i, int j, int k) {
        const double bz0 = 0.5 * (Bz(i, j, k) + nBz(i, j, k)), bzm = 0.5 * (Bz(i, j - 1, k) + nBz(i, j - 1, k));
        const double by0 = 0.5 * (By(i, j, k) + nBy(i, j, k)), bym = 0.5 * (By(i, j, k - 1) + nBy(i, j, k - 1));
        const double ez0 = 0.5 * (Ez(i, j, k) + nEz(i, j, k)), ezp = 0.5 * (Ez(i, j + 1, k) + nEz(i, j + 1, k));
        const double ey0 = 0.5 * (Ey(i, j, k) + nEy(i, j, k)), eyp = 0.5 * (Ey(i, j, k + 1) + nEy(i, j, k + 1));
        const double sx = (bz0 - bzm) * iy - (by0 - bym) * iz - Jx(i, j, k);
        Ex(i, j, k) += (de ? D / (1 + dv.cx(i, j, k)) : D) * sx;
        Bx(i, j, k) -= D * ((ezp - ez0) * iy - (eyp - ey0) * iz);
    });
    if (side) side->fix_corr(Ex, Bx);
    if (metal) metal_fix_corr(d_, metal->view(), Ex, Bx);
    ey.swap(ney_); ez.swap(nez_); by.swap(nby_); bz.swap(nbz_);   // the new arrays carry valid guards
    { PROF_SCOPE("rip_cc.xchg ExBx"); comm.exchange(d_, {&ex, &bx}); }
}

double Fields::energy() const {
    const FV a = ex.view(), b = ey.view(), c = ez.view(), e = bx.view(), f = by.view(), g = bz.view();
    const int nx = d_.n[0], ny = d_.n[1];
    const bool de = diel != nullptr;
    const Diel3View dv = de ? diel->view() : Diel3View{};
    const double s = par_sum(d_.owned(), [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        auto sq = [&](const FV& v) { return v(i, j, k) * v(i, j, k); };
        const double w = sq(a) + sq(b) + sq(c) + sq(e) + sq(f) + sq(g);
        if (!de) return w;
        // the dielectric: + E.P, P = chi E at E_x and at the J_y, J_z points (mu_x E there)
        const double my = 0.5 * (b(i, j, k) + b(i + 1, j, k)), mz = 0.5 * (c(i, j, k) + c(i + 1, j, k));
        return w + dv.cx(i, j, k) * sq(a) + dv.cy(i, j, k) * my * my + dv.cz(i, j, k) * mz * mz;
    });
    return 0.5 * s * d_.cell_volume();
}

// Local index range [lo, hi) along y, z where the Gauss law and div B are checked: all owned cells, or
// outside the CPML layers and one cell more (pml.hpp).
void Fields::check_range(int lo[3], int hi[3]) const {
    for (int a = 0; a < 3; ++a) {
        lo[a] = 0; hi[a] = d_.n[a];
        if (a > 0 && pml) {
            lo[a] = std::max(0, pml->check_lo(a) - d_.off[a]);
            hi[a] = std::min(d_.n[a], pml->check_hi(a, d_.N[a]) - d_.off[a]);
        }
        if (a > 0 && side && side->config().faces[a]) {   // inside the side tube, one cell away from its faces
            // with a fluid shell: inside the kinetic region (2 cells in) away from the fluid fraction at its edge, whose
            // charge rho does not hold
            const int m = side->fluid() && !side->config().fluid_all ? 5 : 1;
            lo[a] = std::max(lo[a], side->config().lo[a] + m - d_.off[a]);
            hi[a] = std::min(hi[a], side->config().hi[a] - m - d_.off[a]);
        }
        if (a == 0 && xabs.cells > 0) {        // the x layers and one cell more (O3)
            lo[0] = std::max(0, xabs.cells + 1 - d_.off[0]);
            hi[0] = std::min(d_.n[0], d_.N[0] - xabs.cells - 1 - d_.off[0]);
        }
        if (a == 0 && dfront > 0) hi[0] = std::min(hi[0], d_.N[0] - dfront - 1 - d_.off[0]);   // the dielectric front layer
    }
}

// At (i+1/2, j+1/2, k+1/2). The last cell at an open right x edge is skipped: its point needs the
// fields at i = n, beyond the edge, where the guard layer is zero, not the solution. With a TFSF source the
// point at the plane (between the scattered and the total cell) is skipped too (tfsf_point).
double Fields::max_div_b() const {
    if (d_.solver == Solver::Yee) return yee_max_div_b(*this, d_);
    const double ix = 1.0 / d_.h[0], iy = 1.0 / d_.h[1], iz = 1.0 / d_.h[2];
    const FV Bx = bx.view(), By = by.view(), Bz = bz.view();
    const int nx = d_.n[0], ny = d_.n[1];
    const int iend = !d_.periodic[0] && d_.off[0] + d_.n[0] == d_.N[0] ? nx - 1 : nx;
    const int sp = tfsf_point();
    int lo[3], hi[3];
    check_range(lo, hi);
    const bool me = metal != nullptr;
    const Metal3View mv = me ? metal->view() : Metal3View{};
    return par_max_abs(d_.owned(), [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        if (i >= iend || i == sp || i < lo[0] || i >= hi[0] || j < lo[1] || j >= hi[1] || k < lo[2] || k >= hi[2]) return 0.0;
        if (me && mv.divb_skip(i, j, k)) return 0.0;   // the surface current of the metal (metal3d.hpp)
        return (Bx(i + 1, j, k) - Bx(i, j, k)) * ix +
               0.5 * (By(i, j + 1, k) + By(i + 1, j + 1, k) - By(i, j, k) - By(i + 1, j, k)) * iy +
               0.5 * (Bz(i, j, k + 1) + Bz(i + 1, j, k + 1) - Bz(i, j, k) - Bz(i + 1, j, k)) * iz;
    });
}

// At (i+1/2, j, k), the position of rho. Set to 0 in the last cell at an open right x edge: its point
// needs Ex, Ey, Ez at i = n, beyond the edge (zero guards), as in max_div_b.
void Fields::gauss_residual(Field& out) const {
    if (d_.solver == Solver::Yee) { yee_gauss_residual(*this, d_, out); return; }
    const double ix = 1.0 / d_.h[0], iy = 1.0 / d_.h[1], iz = 1.0 / d_.h[2];
    const FV Ex = ex.view(), Ey = ey.view(), Ez = ez.view(), R = rho.view(), O = out.view();
    const int iend = !d_.periodic[0] && d_.off[0] + d_.n[0] == d_.N[0] ? d_.n[0] - 1 : d_.n[0];
    int lo[3], hi[3];
    check_range(lo, hi);
    const bool me = metal != nullptr;
    const Metal3View mv = me ? metal->view() : Metal3View{};
    const bool de = diel != nullptr;
    const Diel3View dv = de ? diel->view() : Diel3View{};
    for_owned(d_, [=] HD(int i, int j, int k) {
        if (i >= iend || i < lo[0] || i >= hi[0] || j < lo[1] || j >= hi[1] || k < lo[2] || k >= hi[2]) { O(i, j, k) = 0; return; }
        if (me && mv.gauss_skip(i, j, k)) { O(i, j, k) = 0; return; }   // the surface charge of the metal (metal3d.hpp)
        if (de) {                                 // dielectric: div D, D = (1 + chi) E at E_x and at the J_y, J_z points
            auto Dy = [&](int jj) { return 0.5 * (1 + dv.cy(i, jj, k)) * (Ey(i, jj, k) + Ey(i + 1, jj, k)); };
            auto Dz = [&](int kk) { return 0.5 * (1 + dv.cz(i, j, kk)) * (Ez(i, j, kk) + Ez(i + 1, j, kk)); };
            O(i, j, k) = ((1 + dv.cx(i + 1, j, k)) * Ex(i + 1, j, k) - (1 + dv.cx(i, j, k)) * Ex(i, j, k)) * ix +
                         (Dy(j) - Dy(j - 1)) * iy + (Dz(k) - Dz(k - 1)) * iz - R(i, j, k);
            return;
        }
        O(i, j, k) = (Ex(i + 1, j, k) - Ex(i, j, k)) * ix +
                     0.5 * (Ey(i, j, k) + Ey(i + 1, j, k) - Ey(i, j - 1, k) - Ey(i + 1, j - 1, k)) * iy +
                     0.5 * (Ez(i, j, k) + Ez(i + 1, j, k) - Ez(i, j, k - 1) - Ez(i + 1, j, k - 1)) * iz - R(i, j, k);
    });
}

// The local cell i whose Gauss point (i + 1/2) lies on the TFSF plane: the main grid holds the scattered field left
// of it and the total field right of it, so E and B jump there by the incident field (a surface charge of the
// bookkeeping, not of the physics: the total field satisfies the Gauss law). Skipped by the diagnostics; -2^30 without.
int Fields::tfsf_point() const { return tfsf ? tfsf->plane() - d_.off[0] - 1 : -(1 << 30); }
