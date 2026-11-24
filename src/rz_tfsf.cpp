// O1 in RZ (docs/specs/tfsf.md, "RZ"): the TFSF plane between the z cells i_s - 1 (scattered) and i_s (total). The
// incident field is the discrete solution of an auxiliary vacuum FieldsRZ on the same domain and radial grid: the laser
// enters it through its open left z edge as in rz_laser.cpp, it has the radial absorber and the right z layer of the main
// grid (the same scheme wherever the total field lives) but no left layer (the laser enters there). Every step the
// incident grid of t_n is kept and the grid is advanced; the transport of the main grid gets the incident components
// across the plane, per mode, as tfsf_correct_kernel of Cart3D with (E_r, B_phi) and (E_phi, B_r) for (Ey, Bz), (Ez, By).
// O5 (decision 78, as Cart3D decision 75): the incident grid runs the closure too. Its box means get C_perp (pe_, pb_, per z
// cell, so the same in the main grid on either side of the plane) and Cz (xs_, bs_, used by the corrections); the main
// grid's Cz across the plane mixes the scattered cell s - 1 with the total cell s, and o5_fix adds the incident value of the
// far cell there, so the scheme is the plain o5 scheme for (total right, scattered + incident left).
#include <cmath>
#include <stdexcept>
#include <utility>
#include "laser.hpp"
#include "rz.hpp"

namespace {
struct Cx { double re, im; };
HD inline Cx operator+(Cx a, Cx b) { return {a.re + b.re, a.im + b.im}; }
HD inline Cx operator-(Cx a, Cx b) { return {a.re - b.re, a.im - b.im}; }
HD inline Cx operator*(double s, Cx a) { return {s * a.re, s * a.im}; }
HD inline Cx imul(double s, Cx a) { return {-s * a.im, s * a.re}; }          // i s a
HD inline int slot(int m, int c) { return m == 0 ? 0 : 2 * m - 1 + c; }
HD inline Cx get(const FV& F, int i, int j, int m) { return {F(i, j, slot(m, 0)), m ? F(i, j, slot(m, 1)) : 0.0}; }
HD inline void add(const FV& F, int i, int j, int m, Cx v) { F(i, j, slot(m, 0)) += v.re; if (m) F(i, j, slot(m, 1)) += v.im; }
}  // namespace

// dst = src over the whole array, guards included
void rz_tfsf_copy(const Domain& d, const FV src, const FV dst) {
    const int g = d.ng, sx = d.n[0] + 2 * g, sy = d.n[1] + 2 * g, sz = d.n[2] + 2 * g;
    par_for(long(sx) * sy * sz, [=] HD(long q) {
        const int i = int(q % sx) - g, j = int((q / sx) % sy) - g, k = int(q / (long(sx) * sy)) - g;
        dst(i, j, k) = src(i, j, k);
    });
}

// XS = (Ez0 + Ez1) / 2, BS = (Bz0 + Bz1) / 2 on the owned cells (iRIP: the box mean of the incident field)
void rz_tfsf_means(const Domain& d, const FV Ez0, const FV Bz0, const FV Ez1, const FV Bz1, const FV XS, const FV BS) {
    const int nz = d.n[0], nr = d.n[1];
    par_for(d.owned(), [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
        XS(i, j, k) = 0.5 * (Ez0(i, j, k) + Ez1(i, j, k)); BS(i, j, k) = 0.5 * (Bz0(i, j, k) + Bz1(i, j, k));
    });
}

// O5: X = Cz P on the owned cells (P C_perp-closed, guards valid)
void rz_tfsf_cz(const Domain& d, const FV P, const FV X) {
    const int nz = d.n[0], nr = d.n[1];
    par_for(d.owned(), [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
        X(i, j, k) = P(i, j, k) - (P(i + 1, j, k) - 2 * P(i, j, k) + P(i - 1, j, k)) / 3;
    });
}

// O5: the Cz stencil across the plane (local cells s - 1 scattered, s total): the scattered cell took -P(s)/3 of the total
// field, it gets back the incident part; the total cell took -P(s - 1)/3 of the scattered field, it gets the incident part
void rz_tfsf_o5_fix_kernel(const Domain& d, int s, const FV PE, const FV PB, const FV Exs, const FV Bxs) {
    const int nz = d.n[0], nr = d.n[1];
    const bool tf = s >= 0 && s < nz, sf = s - 1 >= 0 && s - 1 < nz;
    if (!tf && !sf) return;
    par_for(long(nr) * d.n[2], [=] HD(long q) {
        const int j = int(q % nr), k = int(q / nr);
        if (sf) { Exs(s - 1, j, k) += PE(s, j, k) / 3; Bxs(s - 1, j, k) += PB(s, j, k) / 3; }
        if (tf) { Exs(s, j, k) -= PE(s - 1, j, k) / 3; Bxs(s, j, k) -= PB(s - 1, j, k) / 3; }
    });
}

// The corrections at the plane (local cells s - 1, s) into the transported fields nEr .. nBp of the main grid, from the
// incident (Er, Ep, Br, Bp) of t_n and its longitudinal averages XS, BS (Ez*, Bz*). The sources at a + 1/2 of FieldsRZ::pass
// average the cells a, a + 1 (mu_z): the total cell s takes T+ of s - 1 and the sources at s - 1/2, so it gets the incident
// T+ of s - 1 with half its XS, BS; the scattered cell s - 1 takes T- of s and the same sources: minus the incident T- of s
// with half its XS, BS. Ephi, Br on the axis are no DOFs (not corrected).
void rz_tfsf_correct_kernel(const Domain& d, const RadialView g, int s, const FV Er, const FV Ep, const FV Br, const FV Bp,
                            const FV XS, const FV BS, const FV nEr, const FV nEp, const FV nBr, const FV nBp, int M) {
    const int nz = d.n[0], nr = d.n[1];
    const double D = d.h[0];
    const bool tf = s >= 0 && s < nz, sf = s - 1 >= 0 && s - 1 < nz;
    if (!tf && !sf) return;
    par_for(long(nr) * M, [=] HD(long q) {
        const int j = int(q % nr), m = int(q / nr);
        const double rh = g.rh[j], rn = g.r[j];
        auto h = [&](const FV& F, int a, int jj) { return 0.5 * get(F, a, jj, m); };     // the half of one cell in mu_z
        auto PSr = [&](int a) { return imul(g.mf(m) / rh, h(BS, a, j)); };
        auto PHt = [&](int a) { return (1 / g.dn[j]) * (h(XS, a, j + 1) - h(XS, a, j)); };
        auto PSt = [&](int a) {
            if (j > 0) return (-1 / g.dd[j]) * (h(BS, a, j) - h(BS, a, j - 1));
            return m == 1 ? (-1 / g.rh[0]) * h(BS, a, 0) : Cx{0, 0};
        };
        auto PHr = [&](int a) {
            if (j > 0) return imul(-g.mf(m) / rn, h(XS, a, j));
            return m == 1 ? imul(-g.mf(1) / g.r[1], h(XS, a, 1)) : Cx{0, 0};
        };
        if (tf) {
            const int a = s - 1;
            const Cx dp = get(Er, a, j, m) + get(Bp, a, j, m) + D * (PSr(a) + PHt(a));     // T+ of the (Er, Bp) pair into s
            const Cx dm = get(Ep, a, j, m) - get(Br, a, j, m) + D * (PSt(a) - PHr(a));     // (Ep - Br), along +z, into s
            add(nEr, s, j, m, 0.5 * dp); add(nBp, s, j, m, 0.5 * dp);
            if (j > 0) { add(nEp, s, j, m, 0.5 * dm); add(nBr, s, j, m, -0.5 * dm); }
        }
        if (sf) {
            const int b = s, c = s - 1;
            const Cx dm = Cx{0, 0} - (get(Er, b, j, m) - get(Bp, b, j, m) + D * (PSr(b) - PHt(b)));   // T- into s - 1
            const Cx dp = Cx{0, 0} - (get(Ep, b, j, m) + get(Br, b, j, m) + D * (PSt(b) + PHr(b)));   // (Ep + Br), along -z
            add(nEr, c, j, m, 0.5 * dm); add(nBp, c, j, m, -0.5 * dm);
            if (j > 0) { add(nEp, c, j, m, 0.5 * dp); add(nBr, c, j, m, 0.5 * dp); }
        }
    });
}

// The z layers (grid.boundary_z = absorbing): every component times exp(-kappa s^2), s the depth into the lo cells at the
// left or the hi cells at the right end (global z index; guards get the factor of their originals), as fields_absorb_x.
void rz_absorb_z_kernel(const Domain& d, int lo, int hi, double kappa, const FV A, const FV B, const FV C, const FV D, const FV E,
                        const FV F) {
    const int g = d.ng, sx = d.n[0] + 2 * g, sy = d.n[1] + 2 * g, sz = d.n[2] + 2 * g, N = d.N[0];
    const long c0 = d.off[0];
    par_for(long(sx) * sy * sz, [=] HD(long q) {
        const int i = int(q % sx) - g, j = int((q / sx) % sy) - g, k = int(q / (long(sx) * sy)) - g;
        const long gi = c0 + i;
        double s = 0;
        if (gi < lo) s = double(lo - gi) / lo;
        else if (hi > 0 && gi > N - 1 - hi) s = double(gi - (N - 1 - hi)) / hi;
        if (s <= 0) return;
        const double f = exp(-kappa * s * s);
        A(i, j, k) *= f; B(i, j, k) *= f; C(i, j, k) *= f; D(i, j, k) *= f; E(i, j, k) *= f; F(i, j, k) *= f;
    });
}

void rz_absorb_z(FieldsRZ& f) {
    if (f.zabs_lo <= 0 && f.zabs_hi <= 0) return;
    rz_absorb_z_kernel(f.dom(), f.zabs_lo, f.zabs_hi, f.zabs_kappa, f.ez.view(), f.er.view(), f.ep.view(), f.bz.view(),
                       f.br.view(), f.bp.view());
}

// ---------------------------------------------------------------- RzTfsf
RzTfsf::RzTfsf(const Domain& d, Comm& comm, int modes, RadialGrid g, const std::vector<LaserConfig>& lasers,
               int zabs_hi, double zkappa, int rabs_cells, double rkappa)
    : d_(d), rabs_(rabs_cells), rkappa_(rkappa) {
    double zs = -1;
    std::vector<LaserConfig> on;
    for (const LaserConfig& L : lasers) {
        if (!L.enabled || L.tfsf_x < 0) continue;
        if (zs >= 0 && std::abs(L.tfsf_x - zs) > 1e-12 * (1 + zs)) throw std::runtime_error("laser.tfsf_x: one plane for all lasers");
        zs = L.tfsf_x;
        on.push_back(L);
    }
    if (zs < 0) throw std::runtime_error("RzTfsf: no laser with tfsf_x");
    is_ = int(std::lround(zs / d.h[0]));
    if (is_ < 1 || is_ > d.N[0] - 1) throw std::runtime_error("laser.tfsf_x: the plane must lie inside the box (1 .. nz - 1 cells)");
    inc_ = std::make_unique<FieldsRZ>(d, modes, std::move(g));
    inc_->zabs_hi = zabs_hi; inc_->zabs_kappa = zkappa;     // the right layer only: the laser enters at the left edge
    rz_add_laser_inflow(*inc_, on, comm);
    for (Field* f : {&er0_, &ep0_, &br0_, &bp0_, &ez0_, &bz0_, &xs_, &bs_}) f->resize(d);
}

void RzTfsf::begin(Comm& comm, double t) {
    FieldsRZ& g = *inc_;
    const Field* src[6] = {&g.er, &g.ep, &g.br, &g.bp, &g.ez, &g.bz};
    Field* dst[6] = {&er0_, &ep0_, &br0_, &bp0_, &ez0_, &bz0_};
    for (int q = 0; q < 6; ++q) rz_tfsf_copy(d_, src[q]->view(), dst[q]->view());
    g.step(comm, t);
    if (d_.solver == Solver::IRip && o5_) {        // o5: the closed means (C_perp in pe_, pb_; then Cz), before the damping
        rz_tfsf_means(d_, ez0_.view(), bz0_.view(), g.ez.view(), g.bz.view(), pe_.view(), pb_.view());
        comm.exchange(d_, {&pe_, &pb_});
        rz_o5_closure(g, comm, pe_, 0, false);
        rz_o5_closure(g, comm, pb_, 1, false);
        rz_tfsf_cz(d_, pe_.view(), xs_.view());
        rz_tfsf_cz(d_, pb_.view(), bs_.view());
        comm.exchange(d_, {&xs_, &bs_});
    } else if (d_.solver == Solver::IRip) {        // the box means of t_n and t_n+1, before the radial damping (the main grid's
        rz_tfsf_means(d_, ez0_.view(), bz0_.view(), g.ez.view(), g.bz.view(), xs_.view(), bs_.view());   // solve sees them so)
        comm.exchange(d_, {&xs_, &bs_});
    }
    if (rabs_ > 0) {                               // the radial absorber of the main grid (RZFieldStep), then the exchange
        rz_absorb(g, rabs_, rkappa_);
        comm.exchange(d_, {&g.ez, &g.er, &g.ep, &g.bz, &g.br, &g.bp});
    }
}

void RzTfsf::set_o5() {
    if (d_.solver != Solver::IRip) throw std::runtime_error("RzTfsf: closure = o5 with solver = irip only");
    o5_ = true;
    inc_->o5 = true;
    pe_.resize(d_); pb_.resize(d_);
}

void RzTfsf::o5_fix(Comm& comm, Field& exs, Field& bxs) {
    rz_tfsf_o5_fix_kernel(d_, is_ - d_.off[0], pe_.view(), pb_.view(), exs.view(), bxs.view());
    comm.exchange(d_, {&exs, &bxs});
}

void RzTfsf::correct(Comm& comm, FieldsRZ& f, int kind) {
    const Domain& d = d_;
    // RIP-CC: the predictor of the incident grid's own step from t_n; iRIP: the box mean of t_n and t_n+1 (begin)
    (void)comm;
    const Field* X = kind == 0 ? &inc_->exs_ : &xs_;
    const Field* B = kind == 0 ? &inc_->bxs_ : &bs_;
    rz_tfsf_correct_kernel(d, f.grid().view(), is_ - d.off[0], er0_.view(), ep0_.view(), br0_.view(), bp0_.view(), X->view(),
                           B->view(), f.ner_.view(), f.nep_.view(), f.nbr_.view(), f.nbp_.view(), f.modes());
}
