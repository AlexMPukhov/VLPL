// RZ fields per azimuthal mode and the explicit RIP-CC step (docs/specs/rz_modes.md). The operators and
// axis rules are those of research/rz_modes/gauss_loop.py (Gauss law per mode at round-off there):
//   psi_x(Bphi, Br) = (1/r) d_r (r Bphi) - (i m/r) Br     at nodes (axis: m = 0: rh_0 Bphi_{1/2} / A_0, else 0)
//   phi_x(Ephi, Er) = -[(1/r) d_r (r Ephi) - (i m/r) Er]  at half nodes
//   psi_r(Bz) = (i m/r) Bz (half), psi_phi(Bz) = -d_r Bz (node; axis: m = 1: -Bz_{1/2} / rh_0, else 0)
//   phi_r(Ez) = -(i m/r) Ez (node; axis: m = 1: -i Ez_1 / r_1, else 0), phi_phi(Ez) = d_r Ez (half)
// in the finite-volume form of the radial grid (rz.hpp, RadialGrid): divergences with the areas A_j, B_j,
// gradients with the spacings dn_j, dd_j (uniform: A_j = r_j dr = 4 / dr at the axis, ...).
// Corrector: Ez += D (psi_x(<Bphi>, <Br>) - Jz), Bz += D phi_x(<Ephi>, <Er>); transport as RIP-CC with
// T+ = Er + Bphi, Ephi - Br moving +z. Axis: m >= 1: Ez = 0; Ephi, Br = 0 on the axis (not dynamic).
#include "rz.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {
struct Cx { double re, im; };
HD inline Cx operator+(Cx a, Cx b) { return {a.re + b.re, a.im + b.im}; }
HD inline Cx operator-(Cx a, Cx b) { return {a.re - b.re, a.im - b.im}; }
HD inline Cx operator*(double s, Cx a) { return {s * a.re, s * a.im}; }
HD inline Cx imul(double s, Cx a) { return {-s * a.im, s * a.re}; }          // i s a
HD inline int slot(int m, int c) { return m == 0 ? 0 : 2 * m - 1 + c; }
HD inline Cx get(const FV& F, int i, int j, int m) { return {F(i, j, slot(m, 0)), m ? F(i, j, slot(m, 1)) : 0.0}; }
HD inline void put(const FV& F, int i, int j, int m, Cx v) { F(i, j, slot(m, 0)) = v.re; if (m) F(i, j, slot(m, 1)) = v.im; }
HD inline Cx mz(const FV& F, int i, int j, int m) { return 0.5 * (get(F, i, j, m) + get(F, i + 1, j, m)); }   // mu_z at i+1/2

// The radial operators at (column i of the given arrays), mode m, on the radial grid g.
struct Ops {
    RadialView g;
    // (1/r) d_r (r Bp) + s (i m / r) Br at node j (s = -1: psi_x; div E with Er, Ep: s = +1)
    HD Cx div_n(const FV& Bp, const FV& Br, int i, int j, int m, double s) const {
        if (j == 0) return m == 0 ? Cx{g.rh[0] * Bp(i, 0, 0) / g.A[0], 0.0} : Cx{0, 0};
        return (1 / g.A[j]) * (g.rh[j] * get(Bp, i, j, m) - g.rh[j - 1] * get(Bp, i, j - 1, m)) +
               imul(s * g.mf(m) / g.r[j], get(Br, i, j, m));
    }
    // (1/r) d_r (r Ep) + s (i m / r) Er at half node j
    HD Cx div_h(const FV& Ep, const FV& Er, int i, int j, int m, double s) const {
        return (1 / g.B[j]) * (g.r[j + 1] * get(Ep, i, j + 1, m) - g.r[j] * get(Ep, i, j, m)) + imul(s * g.mf(m) / g.rh[j], get(Er, i, j, m));
    }
    HD Cx psi_x(const FV& Bp, const FV& Br, int i, int j, int m) const { return div_n(Bp, Br, i, j, m, -1); }
    HD Cx phi_x(const FV& Ep, const FV& Er, int i, int j, int m) const { return -1.0 * div_h(Ep, Er, i, j, m, -1); }
};
}  // namespace

RadialView RadialGrid::view() const {
    const double* p = dev.data();
    const size_t n = size_t(nr) + 5;
    RadialView v{p, p + n, p + 2 * n, p + 3 * n, p + 4 * n, p + 5 * n, p + 6 * n, p + 7 * n, nr, ju, dr, p + 8 * n, p + 9 * n, p + 10 * n};
    v.me = meff.empty() ? nullptr : dme.data();
    v.rod = rodj.empty() ? nullptr : drod.data();
    return v;
}

void RadialGrid::set_meff(int modes, bool on) {
    meff.clear();
    if (on) {
        const double dp = 2 * M_PI / (2 * modes - 1);
        for (int m = 0; m < modes; ++m) meff.push_back(2 / dp * std::sin(m * dp / 2));
    }
    dme.upload(meff);
}

void RadialGrid::upload() {
    const size_t n = size_t(nr) + 5;
    for (std::vector<double>* v : {&ca, &cl, &cr})
        if (v->size() != n) v->assign(n, 0.0);
    std::vector<double> all;
    for (const std::vector<double>* v : {&r, &rh, &A, &B, &dn, &dd, &bt, &bt2, &ca, &cl, &cr}) all.insert(all.end(), v->begin(), v->end());
    dev.upload(all);
}

RadialGrid rz_radial_grid(int nr, double dr, double r_uniform, double stretch, double dr_max) {
    if (nr < 1 || !(dr > 0)) throw std::runtime_error("rz: nr >= 1 and dr > 0");
    if (!(stretch >= 1) || !(dr_max >= dr)) throw std::runtime_error("rz: grid.stretch >= 1 and grid.dr_max >= dr");
    RadialGrid g;
    g.nr = nr; g.dr = dr;
    const size_t n = size_t(nr) + 5;
    const int ju = stretch == 1 || r_uniform >= (nr + 5) * dr ? nr + 5 : std::max(4, int(std::floor(r_uniform / dr + 0.5)));
    g.ju = ju;
    g.r.assign(n, 0.0); g.rh.assign(n, 0.0); g.A.assign(n, 0.0); g.B.assign(n, 0.0); g.dn.assign(n, 0.0); g.dd.assign(n, 0.0);
    double h = dr;
    for (int j = 0; j + 1 < int(n); ++j) {
        if (j < nr) h = j < ju ? dr : std::min(dr * std::pow(stretch, j - ju + 1), dr_max);   // beyond: the last
        g.r[size_t(j + 1)] = j < ju ? (j + 1) * dr : g.r[size_t(j)] + h;      // uniform part exactly j dr
        if (j < nr && h != dr) g.uniform = false;
    }
    for (int j = 0; j + 1 < int(n); ++j) {
        g.dn[size_t(j)] = g.r[size_t(j + 1)] - g.r[size_t(j)];
        g.rh[size_t(j)] = j < ju ? (j + 0.5) * dr : 0.5 * (g.r[size_t(j)] + g.r[size_t(j + 1)]);
    }
    g.dn[n - 1] = g.dn[n - 2]; g.rh[n - 1] = g.r[n - 1] + 0.5 * g.dn[n - 1];
    for (size_t j = 0; j < n; ++j) {
        g.A[j] = j == 0 ? 0.5 * g.rh[0] * g.rh[0] : 0.5 * (g.rh[j] * g.rh[j] - g.rh[j - 1] * g.rh[j - 1]);
        g.B[j] = g.rh[j] * g.dn[j];
        g.dd[j] = j == 0 ? dr : g.A[j] / g.r[j];
    }
    rz_ruyten_coefficients(g);
    g.upload();
    return g;
}

Domain rz_domain(Comm& comm, int nz, int nr, int modes, double dz, double dr, bool periodic_z, Solver s, int ranks_z) {
    if (modes < 1) throw std::runtime_error("rz: modes must be >= 1");
    Domain d;
    d.N[0] = nz; d.N[1] = nr; d.N[2] = 2 * modes - 1;
    d.h[0] = dz; d.h[1] = dr; d.h[2] = 1;
    d.periodic[0] = periodic_z; d.periodic[1] = false; d.periodic[2] = false;
    d.slots = true;                                   // the mode slots: no guard exchange (grid.hpp)
    d.solver = s;
    d.ng = 3;
    const int want[3] = {ranks_z, 1, 1};
    comm.decompose(d, want);
    return d;
}

FieldsRZ::FieldsRZ(const Domain& d, int modes) : FieldsRZ(d, modes, rz_radial_grid(d.N[1], d.h[1])) {}

FieldsRZ::FieldsRZ(const Domain& d, int modes, RadialGrid grid) : d_(d), M_(modes), g_(std::move(grid)) {
    if (g_.nr != d.N[1]) throw std::runtime_error("rz: the radial grid has another nr");
    for (Field* f : {&ez, &er, &ep, &bz, &br, &bp, &jz, &jr, &jp, &exs_, &bxs_, &ner_, &nep_, &nbr_, &nbp_}) f->resize(d);
}

void FieldsRZ::zero_currents() { jz.fill(0); jr.fill(0); jp.fill(0); }

void FieldsRZ::use_c1_map(double s0, double s1) {
    if (g_.c1_s0 != s0 || g_.c1_s1 != s1) rz_c1_coefficients(g_, s0, s1);
    c1_map = true;
}

// One pass: (predictor: Ez*, Bz* from the fields, else the given exs_, bxs_ with valid guards), the
// transport into ner_.. (exchanged), the corrector into oEz, oBz (may be ez, bz).
void FieldsRZ::pass(Comm& comm, bool predictor, Field& oEz, Field& oBz) {
    const Domain& d = d_;
    const double D = d.h[0];
    const int nz = d.n[0], nr = d.n[1], M = M_;
    const RadialView g = g_.view();
    const Ops op{g};
    const FV Ez = ez.view(), Er = er.view(), Ep = ep.view(), Bz = bz.view(), Br = br.view(), Bp = bp.view();
    const FV Jz = jz.view(), Jr = jr.view(), Jp = jp.view(), Exs = exs_.view(), Bxs = bxs_.view();
    const long cols = long(nz) * nr * M;
    const MetalView mv = metal ? metal->view() : MetalView{};    // block rings (rz_metal.hpp): n = 0 none
    const long c0 = d.cell0(0);
    if (diel) diel->prepare(d, g_);
    const DielView dv = diel ? diel->view() : DielView{};       // blocks dielectric (rz_dielectric.hpp): on = false none
    if (predictor) {
        par_for(cols, [=] HD(long q) {
            const int i = int(q % nz), j = int((q / nz) % nr), m = int(q / (long(nz) * nr));
            // (1 + chi) d_t E_z = curl B - J_z (chi per angle: divided on the angles after this kernel)
            const double fe = dv.on && !dv.np ? 0.5 * D / (1 + dv.z(i, j)) : 0.5 * D;
            Cx e = get(Ez, i, j, m) + fe * (op.psi_x(Bp, Br, i, j, m) - get(Jz, i, j, m));
            if (m >= 1 && j == 0) e = {0, 0};
            Cx b = get(Bz, i, j, m) + (0.5 * D) * op.phi_x(Ep, Er, i, j, m);
            if (mv.n) {                                         // metal: E_z* = 0 where E_z is, B_z* = 0 in and on the metal
                if (mv.node_at(c0 + i, j) == 1) e = {0, 0};
                if (mv.half_at(c0 + i, j)) b = {0, 0};
            }
            put(Exs, i, j, m, e);
            put(Bxs, i, j, m, b);
        });
        if (dv.np) rz_diel_ez_angles(*this, exs_, ez, exs_);                // chi per angle: the E_z increment / (1 + chi_z)
        if (metal_phi) rz_metal_phi_apply(*metal_phi, *this, 0, exs_, bxs_);   // block metal: the rules per angle
        if (g_.jc(M_ - 1) >= 0) rz_axis_cut_apply(*this, 0, exs_, bxs_);          // numerics.rz_axis_cut
        comm.exchange(d, {&exs_, &bxs_});
    }
    // transport sources at a + 1/2 (a = i - 1, i): Psi_r, Phi_phi at half nodes, Psi_phi, Phi_r at nodes
    const FV nEr = ner_.view(), nEp = nep_.view(), nBr = nbr_.view(), nBp = nbp_.view();
    par_for(cols, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), m = int(q / (long(nz) * nr));
        const double rh = g.rh[j], rn = g.r[j];
        auto PSr = [&](int a) { return imul(g.mf(m) / rh, mz(Bxs, a, j, m)) - get(Jr, a, j, m); };
        auto PHt = [&](int a) { return (1 / g.dn[j]) * (mz(Exs, a, j + 1, m) - mz(Exs, a, j, m)); };
        auto PSt = [&](int a) {
            Cx v;
            if (j > 0) v = (-1 / g.dd[j]) * (mz(Bxs, a, j, m) - mz(Bxs, a, j - 1, m));
            else v = m == 1 ? (-1 / g.rh[0]) * mz(Bxs, a, 0, m) : Cx{0, 0};
            return v - get(Jp, a, j, m);
        };
        auto PHr = [&](int a) {
            if (j > 0) return imul(-g.mf(m) / rn, mz(Exs, a, j, m));
            return m == 1 ? imul(-g.mf(1) / g.r[1], mz(Exs, a, 1, m)) : Cx{0, 0};
        };
        const Cx tpy = get(Er, i - 1, j, m) + get(Bp, i - 1, j, m) + D * (PSr(i - 1) + PHt(i - 1));
        const Cx tmy = get(Er, i + 1, j, m) - get(Bp, i + 1, j, m) + D * (PSr(i) - PHt(i));
        const Cx tpz = get(Ep, i + 1, j, m) + get(Br, i + 1, j, m) + D * (PSt(i) + PHr(i));
        const Cx tmz = get(Ep, i - 1, j, m) - get(Br, i - 1, j, m) + D * (PSt(i - 1) - PHr(i - 1));
        put(nEr, i, j, m, 0.5 * (tpy + tmy)); put(nBp, i, j, m, 0.5 * (tpy - tmy));
        const bool axis = j == 0;                                   // Ephi, Br on the axis: not dynamic
        if (!mv.n) {
            put(nEp, i, j, m, axis ? Cx{0, 0} : 0.5 * (tpz + tmz)); put(nBr, i, j, m, axis ? Cx{0, 0} : 0.5 * (tpz - tmz));
            return;
        }
        // metal: zero inside; at a z face E_r = E_phi = 0 and B_phi, B_r carry the characteristic from the vacuum side
        Cx ep = axis ? Cx{0, 0} : 0.5 * (tpz + tmz), br = axis ? Cx{0, 0} : 0.5 * (tpz - tmz);
        const int h = mv.half_at(c0 + i, j), nd = mv.node_at(c0 + i, j);
        if (h) { put(nEr, i, j, m, Cx{0, 0}); put(nBp, i, j, m, h == 1 ? Cx{0, 0} : (h == 2 ? tpy : -1.0 * tmy)); }
        if (nd) { ep = {0, 0}; br = nd == 1 || axis ? Cx{0, 0} : (nd == 2 ? -1.0 * tmz : tpz); }
        put(nEp, i, j, m, ep); put(nBr, i, j, m, br);
    });
    if (metal_phi) rz_metal_phi_apply(*metal_phi, *this, 1, ner_, nep_, &nbr_, &nbp_);
    if (g_.jc(M_ - 1) >= 0) rz_axis_cut_apply(*this, 1, ner_, nep_, &nbr_, &nbp_);
    if (tfsf) tfsf->correct(comm, *this, predictor ? 0 : 2);   // the incident T across the TFSF plane (rz_tfsf.cpp)
    comm.exchange(d, {&ner_, &nep_, &nbr_, &nbp_});
    if (diel) {                                               // the polarization current of E_r, E_phi: lines along z
        diel->correct(comm, *this);
        comm.exchange(d, {&ner_, &nep_, &nbr_, &nbp_});
    }
    // corrector with the time-centred transverse fields (the averages through two views)
    const FV oE = oEz.view(), oB = oBz.view();
    par_for(cols, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), m = int(q / (long(nz) * nr));
        auto avg = [&](const FV& A, const FV& B, int jj) { return 0.5 * (get(A, i, jj, m) + get(B, i, jj, m)); };
        Cx px;
        if (j == 0) px = m == 0 ? Cx{g.rh[0] * avg(Bp, nBp, 0).re / g.A[0], 0.0} : Cx{0, 0};
        else px = (1 / g.A[j]) * (g.rh[j] * avg(Bp, nBp, j) - g.rh[j - 1] * avg(Bp, nBp, j - 1)) - imul(g.mf(m) / g.r[j], avg(Br, nBr, j));
        const Cx fx = (-1 / g.B[j]) * (g.r[j + 1] * avg(Ep, nEp, j + 1) - g.r[j] * avg(Ep, nEp, j)) + imul(g.mf(m) / g.rh[j], avg(Er, nEr, j));
        Cx e = get(Ez, i, j, m) + (dv.on && !dv.np ? D / (1 + dv.z(i, j)) : D) * (px - get(Jz, i, j, m));
        if (m >= 1 && j == 0) e = {0, 0};
        Cx b = get(Bz, i, j, m) + D * fx;
        if (mv.n) {                                             // metal: E_z = 0 where tangential or inside, B_z = 0 in and on it
            if (mv.node_at(c0 + i, j) == 1) e = {0, 0};
            if (mv.half_at(c0 + i, j)) b = {0, 0};
        }
        put(dv.np ? Exs : oE, i, j, m, e);                      // chi per angle: the vacuum value into the scratch E_z*
        put(oB, i, j, m, b);
    });
    if (dv.np) rz_diel_ez_angles(*this, oEz, ez, exs_);         // E_z = E_z^n + increment / (1 + chi_z) on the angles
    if (metal_phi) rz_metal_phi_apply(*metal_phi, *this, 2, oEz, oBz);
    if (g_.jc(M_ - 1) >= 0) rz_axis_cut_apply(*this, 2, oEz, oBz);
}

void FieldsRZ::step(Comm& comm, double t) {
    comm.exchange(d_, {&jz, &jr, &jp});
    rz_axis_zero_jz(*this);                         // J_z of m >= 1 zero on the axis (J_z ~ r^m)
    if (axis_regular) rz_axis_regular_current(*this);   // numerics.rz_axis_current = regular (decision 112)
    if (metal_phi) metal_phi->ensure(d_);           // the mask of the columns entering with a moving window
    for (auto& h : inflow) h(*this, t);
    if (tfsf) tfsf->begin(comm, t);                // the incident grid of the TFSF plane to t_n+1
    if (d_.solver == Solver::IRip) {
        if (metal) throw std::runtime_error("rz: rings (embedded metal) need numerics.solver = rip_cc");
        if (metal_phi) throw std::runtime_error("rz: the metal block needs numerics.solver = rip_cc");
        if (diel) throw std::runtime_error("rz: dielectric blocks need numerics.solver = rip_cc (iRIP: the medium block)");
        rz_irip_step(*this, comm, t);
    }
    else {
        if (d_.solver != Solver::RipCC) throw std::runtime_error("rz: solver rip_cc or irip");
        pass(comm, true, ez, bz);
        er.swap(ner_); ep.swap(nep_); br.swap(nbr_); bp.swap(nbp_);
        comm.exchange(d_, {&ez, &bz});
    }
    rz_absorb_z(*this);                            // the z layers (the guards get the factor of their originals)
}

double FieldsRZ::energy() const { return energy(0, d_.n[0] - 1); }

double FieldsRZ::energy(int i0, int i1) const {
    const FV a = ez.view(), b = er.view(), c = ep.view(), e = bz.view(), f = br.view(), g = bp.view();
    const int nz = d_.n[0], nr = d_.n[1], S = d_.n[2];
    const double dz = d_.h[0];
    const RadialView rg = g_.view();
    if (diel) diel->prepare(d_, g_);
    const DielView dv = diel ? diel->view() : DielView{};
    const double s = par_sum(long(nz) * nr * S, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
        if (i < i0 || i > i1) return 0.0;
        const double wm = k == 0 ? 1.0 : 2.0;                             // |F_0|^2 + 2 sum |F_m|^2
        const double vn = 2 * M_PI * rg.A[j], vh = 2 * M_PI * rg.B[j];
        auto sq = [&](const FV& v) { return v(i, j, k) * v(i, j, k); };
        const double w = wm * (vn * (sq(a) + sq(c) + sq(f)) + vh * (sq(b) + sq(e) + sq(g)));
        if (!dv.on || dv.np) return w;
        // the dielectric: + E.P with P = chi E at the E_z nodes and the J_r, J_phi points a = i + 1/2 (mu_z E there)
        const double mr = 0.5 * (b(i, j, k) + b(i + 1, j, k)), mp = 0.5 * (c(i, j, k) + c(i + 1, j, k));
        return w + wm * (vn * (dv.z(i, j) * sq(a) + dv.p(i, j) * mp * mp) + vh * dv.r(i, j) * mr * mr);
    });
    return 0.5 * s * dz + (dv.np ? rz_diel_pol_energy(*this, i0, i1) : 0.0);   // chi per angle: E.P on the angles
}

double FieldsRZ::energy_mode(int m) const {
    const FV a = ez.view(), b = er.view(), c = ep.view(), e = bz.view(), f = br.view(), g = bp.view();
    const int nz = d_.n[0], nr = d_.n[1], k0 = m == 0 ? 0 : 2 * m - 1, ns = m == 0 ? 1 : 2;
    const RadialView rg = g_.view();
    if (diel) diel->prepare(d_, g_);
    const DielView dv = diel ? diel->view() : DielView{};
    const double s = par_sum(long(nz) * nr * ns, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = k0 + int(q / (long(nz) * nr));
        auto sq = [&](const FV& v) { return v(i, j, k) * v(i, j, k); };
        const double w = (m ? 2.0 : 1.0) * 2 * M_PI * (rg.A[j] * (sq(a) + sq(c) + sq(f)) + rg.B[j] * (sq(b) + sq(e) + sq(g)));
        if (!dv.on || dv.np) return w;
        const double mr = 0.5 * (b(i, j, k) + b(i + 1, j, k)), mp = 0.5 * (c(i, j, k) + c(i + 1, j, k));
        return w + (m ? 2.0 : 1.0) * 2 * M_PI * (rg.A[j] * (dv.z(i, j) * sq(a) + dv.p(i, j) * mp * mp) + rg.B[j] * dv.r(i, j) * mr * mr);
    });
    return 0.5 * s * d_.h[0] + (dv.np ? rz_diel_pol_energy_mode(*this, m) : 0.0);   // chi per angle: E_m . P_m
}

void FieldsRZ::div_e(Field& out) const {
    const double dz = d_.h[0];
    const int nz = d_.n[0], nr = d_.n[1], M = M_;
    const FV Ez = ez.view(), Er = er.view(), Ep = ep.view(), O = out.view();
    const Ops op{g_.view()};
    if (diel) diel->prepare(d_, g_);
    const DielView dv = diel ? diel->view() : DielView{};
    if (dv.np) { rz_diel_div_d(*this, out); return; }   // chi per angle: D on the angles, back to the modes
    const RadialView g = g_.view();
    par_for(long(nz) * nr * M, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), m = int(q / (long(nz) * nr));
        auto dt = [&](int ii) { return op.div_n(Er, Ep, ii, j, m, 1); };   // transverse divergence at column ii
        Cx v;
        if (!dv.on) v = (1 / dz) * (get(Ez, i + 1, j, m) - get(Ez, i, j, m)) + 0.5 * (dt(i) + dt(i + 1));
        else {                                    // div D: D = (1 + chi) E at the E_z nodes and at the points (i+1/2) of mu_z E_perp
            auto Dr = [&](int jj) { return (0.5 * (1 + dv.r(i, jj))) * (get(Er, i, jj, m) + get(Er, i + 1, jj, m)); };
            const Cx Dp = (0.5 * (1 + dv.p(i, j))) * (get(Ep, i, j, m) + get(Ep, i + 1, j, m));
            Cx t;
            if (j == 0) t = m == 0 ? Cx{g.rh[0] * Dr(0).re / g.A[0], 0.0} : Cx{0, 0};
            else t = (1 / g.A[j]) * (g.rh[j] * Dr(j) - g.rh[j - 1] * Dr(j - 1)) + imul(g.mf(m) / g.r[j], Dp);
            v = (1 / dz) * ((1 + dv.z(i + 1, j)) * get(Ez, i + 1, j, m) - (1 + dv.z(i, j)) * get(Ez, i, j, m)) + t;
        }
        if (m >= 1 && j == 0) v = {0, 0};                                 // no Gauss point on the axis
        put(O, i, j, m, v);
    });
}

void FieldsRZ::div_b(Field& out) const {
    const double dz = d_.h[0];
    const int nz = d_.n[0], nr = d_.n[1], M = M_;
    const FV Bz = bz.view(), Br = br.view(), Bp = bp.view(), O = out.view();
    const Ops op{g_.view()};
    par_for(long(nz) * nr * M, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), m = int(q / (long(nz) * nr));
        auto dt = [&](int ii) { return op.div_h(Br, Bp, ii, j, m, 1); };
        put(O, i, j, m, (1 / dz) * (get(Bz, i + 1, j, m) - get(Bz, i, j, m)) + 0.5 * (dt(i) + dt(i + 1)));
    });
}

// Entry cell of the moving window with iRIP (decision 40 per mode): the last cell m + 1 = n - 1 enters
// with zero transverse fields; Ez, Bz there are set so that the Gauss residual and div B of cell m keep
// their initial values (the entering plasma is neutral): Ez(m+1) = Ez(m) + dz (G0(m) - dt_E(m) / 2),
// Bz(m+1) = Bz(m) - dz dt_B(m) / 2, dt the transverse divergence of column m (as div_e, div_b).
void rz_window_entry(FieldsRZ& f, const Field& gauss0, const Field* rhom) {
    const Domain& d = f.dom();
    const double dz = d.h[0];
    const int nr = d.n[1], M = f.modes(), mi = d.n[0] - 2;
    const FV Ez = f.ez.view(), Er = f.er.view(), Ep = f.ep.view(), Bz = f.bz.view(), Br = f.br.view(), Bp = f.bp.view();
    const FV G0 = gauss0.view(), RM = rhom ? rhom->view() : gauss0.view();   // O7: the medium charge per mode at the Gauss point
    const bool med = rhom != nullptr;
    const Ops op{f.grid().view()};
    par_for(long(nr) * M, [=] HD(long q) {
        const int j = int(q % nr), m = int(q / nr);
        const Cx te = op.div_n(Er, Ep, mi, j, m, 1);
        if (!(m >= 1 && j == 0)) put(Ez, mi + 1, j, m, get(Ez, mi, j, m) + dz * (get(G0, mi, j, m) + (med ? get(RM, mi, j, m) : Cx{0, 0}) - 0.5 * te));
        const Cx tb = op.div_h(Br, Bp, mi, j, m, 1);
        put(Bz, mi + 1, j, m, get(Bz, mi, j, m) - dz * 0.5 * tb);
    });
}

void rz_absorb(FieldsRZ& f, int cells, double kappa) {
    const Domain& d = f.dom();
    const int nz = d.n[0], nr = d.n[1], S = d.n[2], j0 = nr - cells;
    const double L = double(cells);
    const FV Ez = f.ez.view(), Er = f.er.view(), Ep = f.ep.view(), Bz = f.bz.view(), Br = f.br.view(), Bp = f.bp.view();
    // the layer only: columns i, rows j0 - 1 .. nr - 1 (the half node of row j0 - 1 is at its edge)
    const int rows = cells + 1;
    par_for(long(nz) * rows * S, [=] HD(long q) {
        const int i = int(q % nz), j = j0 - 1 + int((q / nz) % rows), k = int(q / (long(nz) * rows));
        if (j < 0) return;
        const double sn = (j - j0) / L, sh = (j + 0.5 - j0) / L;
        const double fn = sn > 0 ? exp(-kappa * sn * sn) : 1.0, fh = sh > 0 ? exp(-kappa * sh * sh) : 1.0;
        Ez(i, j, k) *= fn; Ep(i, j, k) *= fn; Br(i, j, k) *= fn;     // nodes r_j
        Er(i, j, k) *= fh; Bp(i, j, k) *= fh; Bz(i, j, k) *= fh;     // half nodes r_{j+1/2}
    });
}

void FieldsRZ::set_metal_phi(std::shared_ptr<MetalPhi> mp) {
    metal_phi = std::move(mp);
    g_.set_meff(M_, metal_phi && !metal_phi->cfg.spectral);
}

void RadialGrid::set_rod(int modes, double kappa) {
    rodj.clear();
    if (kappa > 0) {
        rodj.assign(size_t(modes), -1);
        for (int m = 2; m < modes; ++m) {
            const double rc = mf(m) * dr / kappa;
            for (int j = 0; j <= nr && r[size_t(j)] < rc; ++j) rodj[size_t(m)] = j;
        }
    }
    drod.upload(rodj);
}

// numerics.rz_axis_cut (RadialGrid::set_rod): mode m held at zero on its rod, nodes j <= jc(m) (E_z, E_phi, B_r) and half
// nodes j < jc(m) (E_r, B_z, B_phi). Stages as rz_metal_phi_apply: 0, 2 a = Ez, b = Bz; 1 a = Er, b = Ephi, c = Br, e = Bphi.
void rz_axis_cut_apply(FieldsRZ& f, int stage, Field& fa, Field& fb, Field* fc, Field* fe) {
    const Domain& d = f.dom();
    const int nz = d.n[0], nr = d.n[1], M = f.modes();
    const RadialView g = f.grid().view();
    const FV A = fa.view(), B = fb.view(), C = fc ? fc->view() : fa.view(), E = fe ? fe->view() : fa.view();
    const bool tr = stage == 1;
    par_for(long(nz) * nr * M, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), m = int(q / (long(nz) * nr));
        const int jc = g.jc(m);
        if (jc < 0 || j > jc) return;
        for (int c = 0; c < (m ? 2 : 1); ++c) {
            const int s = m == 0 ? 0 : 2 * m - 1 + c;
            if (!tr) { A(i, j, s) = 0; if (j < jc) B(i, j, s) = 0; }                    // Ez node, Bz half
            else { if (j < jc) { A(i, j, s) = 0; E(i, j, s) = 0; } B(i, j, s) = 0; C(i, j, s) = 0; }   // Er, Bphi half; Ephi, Br node
        }
    });
}
