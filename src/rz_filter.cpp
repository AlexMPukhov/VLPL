// Current filter for RZ (numerics.filter_passes, decision 54; FBPIC filters rho and J by default). Per pass, two
// charge-conserving steps of the form S = 1 + a G D on the current and T = 1 + a D G on the charge, D the divergence
// of the Gauss law restricted to its points (D S = T D identically: the continuity per mode stays exact, the Gauss law
// at round-off, on any boundary and any decomposition; a uniform density is unchanged):
// - along z: D = the full divergence of the current (z difference of J_z plus the transverse divergence), G_z its
//   z part: J_z,i += (dz / 4) (w_i - w_{i-1}), w = D J on the Gauss cells (0 outside), on the columns inside
//   (an open edge column stays); rho: the binomial (1, 2, 1) / 4 over the Gauss cells with a Neumann closure. A plain
//   binomial of all components would mix in the guard cell at an open edge, where the charge of the leaving particles
//   vanishes without a current (Gauss defect 4e-3 in rz_lwfa).
// - per mode m in r, with the divergence of the grid D_m (V_r at half nodes, V_phi at nodes -> nodes:
//   (rh_j V_r,j - rh_{j-1} V_r,j-1) / A_j + (i m / r_j) V_phi,j) and its adjoint gradient G_m (X at nodes ->
//   ((X_{j+1} - X_j) / dn_j, (i m / r_j) X_j)): rho and J_z by T = 1 + a_m D_m G_m, (J_r, J_phi) by
//   S = 1 + a_m G_m D_m, slice by slice in z (it commutes with the z difference). D_m G_m is the
//   scalar operator of the field solver (TM), its spectrum in [0, lambda_max,m]: a_m = 1 / lambda_max,m keeps T, S in
//   [0, 1] (no axis excess; uniform grid, m = 0 away from the axis: T = (1, 2, 1)/4). Axis: m = 0 by the axis volume
//   A_0; m >= 1 the axis node is no degree of freedom (unchanged, 0 in the operators).
// The z step changes J_z alone and the r step is no gradient either: the change of J radiates (test R24: 0.15-0.17 of
// the field energy of the change in B). Kept as numerics.filter = local.
// Projection (numerics.filter = projection, decision 70; the default until decision 91): per pass w = D J, r = (T - 1) w with the T of the
// charge (T_r T_z, as rz_filter_rho), L psi = r and J <- J + G psi, G = (dz, mu_z^2 G_m) the gradient of the RZ Gauss law
// (dz Ez + mu_z D_m E_perp = rho) and L = D G = dzz + mu_z^2 D_m G_m, mu_z the (1, 2, 1) / 4 average along z. Then
// D J' = T D J exactly, and G psi only moves charge: fed alone into the field solver it leaves B = 0 (R24, 1e-31).
// L psi = r: per mode the TM eigenbasis of D_m G_m (eigenvalues -lambda, iRIP's radial transform), per radial eigenmode
// s (psi_i-1 + psi_i+1) + b psi_i = dz^2 r_i, s = 1 - dz^2 lambda / 4, b = -2 - dz^2 lambda / 2 (|b| > 2 |s|), solved by
// src/xlines.hpp (cyclic for periodic z, the partition method for ranks along z). Open z: the Gauss cells [0, N - 1),
// psi reflected at both ends (psi_-1 = psi_0, psi_N-1 = psi_N-2; J_z on the faces 0 and N - 1 unchanged, as before).
#include <stdexcept>
#include "rz.hpp"
#include "filter.hpp"
#include "xlines.hpp"

namespace {
struct Cx { double re, im; };
HD inline int slot(int m, int c) { return m == 0 ? 0 : 2 * m - 1 + c; }
HD inline Cx get(const FV& F, int i, int j, int m) { return {F(i, j, slot(m, 0)), m ? F(i, j, slot(m, 1)) : 0.0}; }
HD inline void put(const FV& F, int i, int j, int m, Cx v) { F(i, j, slot(m, 0)) = v.re; if (m) F(i, j, slot(m, 1)) = v.im; }

// node value X_j of mode m as the operators see it: 0 beyond the wall and, m >= 1, on the axis
HD inline Cx xn(const FV& X, int i, int j, int m, int nr) {
    if (j >= nr || (m >= 1 && j == 0)) return {0, 0};
    return get(X, i, j, m);
}

// (D_m V)_j at node j (V_r half nodes, V_phi nodes); 0 beyond the wall and, m >= 1, on the axis
HD inline Cx div_m(const FV& Vr, const FV& Vp, const RadialView& g, int i, int j, int m, int nr) {
    if (j >= nr || (m >= 1 && j == 0)) return {0, 0};
    const Cx a = get(Vr, i, j, m);
    if (j == 0) return {g.rh[0] * a.re / g.A[0], 0.0};
    const Cx b = get(Vr, i, j - 1, m), p = get(Vp, i, j, m);
    const double s = g.mf(m) / g.r[j];
    return {(g.rh[j] * a.re - g.rh[j - 1] * b.re) / g.A[j] - s * p.im, (g.rh[j] * a.im - g.rh[j - 1] * b.im) / g.A[j] + s * p.re};
}

// O = T X (nodes), X with valid values in the owned columns
void filter_scalar_r(const Domain& d, const RadialView g, const FV X, const FV O, int M, const double* am) {
    const int nz = d.n[0], nr = d.n[1];
    par_for(long(nz) * M, [=] HD(long q) {
        const int i = int(q % nz), m = int(q / nz);
        const double a = am[m];
        for (int j = 0; j < nr; ++j) {
            if (m >= 1 && j == 0) { put(O, i, 0, m, get(X, i, 0, m)); continue; }
            const Cx x = xn(X, i, j, m, nr), xp = xn(X, i, j + 1, m, nr);
            // G X at the half nodes j, j - 1 and the node j; then D
            const Cx gr = {(xp.re - x.re) / g.dn[j], (xp.im - x.im) / g.dn[j]};
            Cx lx;
            if (j == 0) lx = {g.rh[0] * gr.re / g.A[0], 0.0};
            else {
                const Cx xm = xn(X, i, j - 1, m, nr);
                const Cx gm = {(x.re - xm.re) / g.dn[j - 1], (x.im - xm.im) / g.dn[j - 1]};
                const double s = g.mf(m) * g.mf(m) / (g.r[j] * g.r[j]);     // (i m / r)(i m / r) X = -m^2 / r^2 X
                lx = {(g.rh[j] * gr.re - g.rh[j - 1] * gm.re) / g.A[j] - s * x.re, (g.rh[j] * gr.im - g.rh[j - 1] * gm.im) / g.A[j] - s * x.im};
            }
            put(O, i, j, m, {x.re + a * lx.re, x.im + a * lx.im});
        }
    });
}

// (Or, Op) = S (Vr, Vp)
void filter_vector_r(const Domain& d, const RadialView g, const FV Vr, const FV Vp, const FV Or, const FV Op, int M, const double* am) {
    const int nz = d.n[0], nr = d.n[1];
    par_for(long(nz) * M, [=] HD(long q) {
        const int i = int(q % nz), m = int(q / nz);
        const double a = am[m];
        for (int j = 0; j < nr; ++j) {
            const Cx w = div_m(Vr, Vp, g, i, j, m, nr), wp = div_m(Vr, Vp, g, i, j + 1, m, nr);
            const Cx vr = get(Vr, i, j, m);
            put(Or, i, j, m, {vr.re + a * (wp.re - w.re) / g.dn[j], vr.im + a * (wp.im - w.im) / g.dn[j]});
            const Cx vp = get(Vp, i, j, m);
            if (j == 0) { put(Op, i, 0, m, vp); continue; }
            const double s = g.mf(m) / g.r[j];                                 // (G W)_phi = (i m / r) W
            put(Op, i, j, m, {vp.re - a * s * w.im, vp.im + a * s * w.re});
        }
    });
}

// The Gauss cells along z in global indices: [0, N) periodic, [0, N - 1) open (the last cell has no Gauss point)
struct ZCells {
    int off, end;
    bool periodic;
    HD bool cell(int i) const { return periodic || (off + i >= 0 && off + i < end); }
    HD bool column(int i) const { return periodic || (off + i >= 1 && off + i < end); }   // both neighbour cells inside
};
ZCells zcells(const Domain& d) { return {d.off[0], d.periodic[0] ? d.N[0] : d.N[0] - 1, d.periodic[0]}; }

// W = D J on the cells -1 .. nz - 1 (guards of J valid), 0 off the Gauss points: J_z,i on the face below cell i,
// J_r, J_phi at the cell (as rho; the deposit), so (D J)_i = (J_z,i+1 - J_z,i) / dz + D_m J_perp,i
void divergence_z(const Domain& d, const RadialView g, const FV Jz, const FV Jr, const FV Jp, const FV W, int M, ZCells zc) {
    const int nz = d.n[0], nr = d.n[1];
    const double dz = d.h[0];
    par_for(long(nz + 1) * nr * M, [=] HD(long q) {
        const int i = int(q % (nz + 1)) - 1, j = int((q / (nz + 1)) % nr), m = int(q / (long(nz + 1) * nr));
        Cx w = {0, 0};
        if (zc.cell(i) && !(m >= 1 && j == 0)) {
            const Cx a = get(Jz, i + 1, j, m), b = get(Jz, i, j, m);
            const Cx t = div_m(Jr, Jp, g, i, j, m, nr);                  // J_r, J_phi at the cell, as rho
            w = {(a.re - b.re) / dz + t.re, (a.im - b.im) / dz + t.im};
        }
        put(W, i, j, m, w);
    });
}

// W = d_z J_z alone on the cells -1 .. nz - 1 (the binomial filter's z step), 0 off the Gauss points
void divergence_zz(const Domain& d, const FV Jz, const FV W, int M, ZCells zc) {
    const int nz = d.n[0], nr = d.n[1];
    const double dz = d.h[0];
    par_for(long(nz + 1) * nr * M, [=] HD(long q) {
        const int i = int(q % (nz + 1)) - 1, j = int((q / (nz + 1)) % nr), m = int(q / (long(nz + 1) * nr));
        Cx w = {0, 0};
        if (zc.cell(i) && !(m >= 1 && j == 0)) {
            const Cx a = get(Jz, i + 1, j, m), b = get(Jz, i, j, m);
            w = {(a.re - b.re) / dz, (a.im - b.im) / dz};
        }
        put(W, i, j, m, w);
    });
}

// J_z += (dz / 4) G_z W on the inner columns
void current_z(const Domain& d, const FV Jz, const FV W, int M, ZCells zc, double cf = 1.0) {
    const int nz = d.n[0], nr = d.n[1];
    const double c = 0.25 * cf * d.h[0];
    par_for(long(nz) * nr * M, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), m = int(q / (long(nz) * nr));
        if (!zc.column(i)) return;
        const Cx a = get(W, i, j, m), b = get(W, i - 1, j, m), x = get(Jz, i, j, m);
        put(Jz, i, j, m, {x.re + c * (a.re - b.re), x.im + c * (a.im - b.im)});
    });
}

// O = the binomial of X over the Gauss cells, Neumann at their ends (guards valid); other cells copied
void charge_z(const Domain& d, const FV X, const FV O, ZCells zc, double cf = 1.0) {
    const double q = 0.25 * cf;
    for_owned(d, [=] HD(int i, int j, int k) {
        const double x = X(i, j, k);
        double o = x;
        if (zc.cell(i)) {
            if (zc.cell(i + 1)) o += q * (X(i + 1, j, k) - x);
            if (zc.cell(i - 1)) o += q * (X(i - 1, j, k) - x);
        }
        O(i, j, k) = o;
    });
}

// dst = src on the owned cells; the guards of dst keep their values. (A swap with the scratch brought the scratch's
// stale guards, transverse fields of the last transport, into the current: at an open z edge no exchange
// overwrites them and the field step reads J_z there; with the laser's tail at the front edge of a moving window
// that was a Gauss defect growing in m = 1 in the last Gauss cell, 3e-11 in B7.)
void copy_owned(const Domain& d, const FV src, const FV dst) {
    for_owned(d, [=] HD(int i, int j, int k) { dst(i, j, k) = src(i, j, k); });
}

// The projection's line coefficients of radial eigenmode a of mode m, line q = a + nr m (lam stored at [m nr + a] = [q])
struct RzProjCoef {
    const double* lam;
    double dz;
    bool cyc;
    HD xlines::LineCoef operator()(long q) const {
        const double x = dz * dz * lam[q];
        if (x == 0) return {1, 0, 1, 1, false};                  // the dropped axis node of m >= 1: identity (r = 0)
        const double s = 1 - 0.25 * x, b = -2 - 0.5 * x;
        if (cyc) return {b, s, b, b, true};
        return {b, s, b + s, s != 0 ? -s : 1.0, false};          // Neumann at cell 0; psi_N-1 = psi_N-2 (s = 0: decoupled)
    }
};

std::vector<double> mode_alpha(const FieldsRZ& f) {
    std::vector<double> a(static_cast<size_t>(f.modes()));
    for (int m = 0; m < f.modes(); ++m) a[size_t(m)] = 1.0 / rz_lambda_max_tm(f.grid(), m);
    return a;
}
}  // namespace

// The projection's arrays: div (D J), tdiv (T D J), psi; the lines (xlines::LineWork, layout n = {nz, nr, M}: line
// q = a + nr m, complex = the Re, Im slots of mode m); the TM basis per mode.
struct RzProjWork : xlines::LineWork {
    Field div, tdiv, psi;
    Buf<double> V, Vi, lam;
    Domain ld;
};

namespace {
void filter_projection(FieldsRZ& f, Comm& comm, int passes, const double* am) {
    const Domain& d = f.dom();
    const int nz = d.n[0], nr = d.n[1], M = f.modes();
    if (!f.fproj_) {
        if (nz < 3) throw std::runtime_error("numerics.filter = projection (RZ): at least 3 cells along z per rank");
        auto W = std::make_shared<RzProjWork>();
        W->div.resize(d); W->tdiv.resize(d); W->psi.resize(d);
        W->ld = d; W->ld.n[2] = M;
        W->resize(W->ld);
        std::vector<double> V, Vi, lam, vb, vib, lb;
        for (int m = 0; m < M; ++m) {
            rz_tm_basis(f.grid(), m, vb, vib, lb);
            V.insert(V.end(), vb.begin(), vb.end()); Vi.insert(Vi.end(), vib.begin(), vib.end());
            lam.insert(lam.end(), lb.begin(), lb.end());
        }
        W->V.upload(V); W->Vi.upload(Vi); W->lam.upload(lam);
        f.fproj_ = W;
    }
    RzProjWork& W = *f.fproj_;
    const RadialView g = f.grid().view();
    const ZCells zc = zcells(d);
    const double dz = d.h[0];
    const long nn = long(nr) * nr;
    const double* V = W.V.data();
    const double* Vi = W.Vi.data();
    double* lw = W.w.data();
    const FV Dv = W.div.view(), Tv = W.tdiv.view(), Ps = W.psi.view();
    const RzProjCoef coef{W.lam.data(), dz, d.periodic[0]};
    for (int p = 0; p < passes; ++p) {
        comm.exchange(d, {&f.jz, &f.jr, &f.jp});
        const FV Jz = f.jz.view(), Jr = f.jr.view(), Jp = f.jp.view();
        divergence_z(d, g, Jz, Jr, Jp, Dv, M, zc);
        comm.exchange(d, {&W.div});
        charge_z(d, Dv, Ps, zc);                                 // psi as scratch: T_z w
        filter_scalar_r(d, g, Ps, Tv, M, am);                    // T_r T_z w
        // radial transform of dz^2 (T - 1) w into the lines
        par_for(long(nz) * M, [=] HD(long q) {
            const int i = int(q % nz), m = int(q / nz);
            const double* A = Vi + m * nn;
            const int k0 = slot(m, 0), k1 = m ? slot(m, 1) : 0;
            for (int a = 0; a < nr; ++a) {
                double sr = 0, si = 0;
                for (int j = 0; j < nr; ++j) {
                    sr += A[long(a) * nr + j] * (Tv(i, j, k0) - Dv(i, j, k0));
                    if (m) si += A[long(a) * nr + j] * (Tv(i, j, k1) - Dv(i, j, k1));
                }
                double* x = lw + 2 * ((long(m) * nr + a) * nz + i);
                x[0] = dz * dz * sr; x[1] = dz * dz * si;
            }
        });
        xlines::solve_lines(W, W.ld, comm, coef);
        // back: psi on the owned cells, then its guards
        par_for(long(nz) * M, [=] HD(long q) {
            const int i = int(q % nz), m = int(q / nz);
            const double* A = V + m * nn;
            for (int j = 0; j < nr; ++j) {
                double sr = 0, si = 0;
                for (int a = 0; a < nr; ++a) {
                    const double* x = lw + 2 * ((long(m) * nr + a) * nz + i);
                    sr += A[long(j) * nr + a] * x[0]; si += A[long(j) * nr + a] * x[1];
                }
                put(Ps, i, j, m, {sr, si});
            }
        });
        comm.exchange(d, {&W.psi});
        // J += G psi: J_z on the inner columns, (J_r, J_phi) += mu_z^2 G_m psi on the Gauss cells (psi reflected at an open end)
        par_for(long(nz) * nr * M, [=] HD(long q) {
            const int i = int(q % nz), j = int((q / nz) % nr), m = int(q / (long(nz) * nr));
            if (zc.column(i)) {
                const Cx a = xn(Ps, i, j, m, nr), b = xn(Ps, i - 1, j, m, nr), x = get(Jz, i, j, m);
                put(Jz, i, j, m, {x.re + (a.re - b.re) / dz, x.im + (a.im - b.im) / dz});
            }
            if (!zc.cell(i)) return;
            const int ii[3] = {zc.cell(i - 1) ? i - 1 : i, i, zc.cell(i + 1) ? i + 1 : i};
            const double wz[3] = {0.25, 0.5, 0.25};
            const double s = j > 0 ? g.mf(m) / g.r[j] : 0.0;               // (i m / r) psi
            Cx gr = {0, 0}, gp = {0, 0};
            for (int c = 0; c < 3; ++c) {
                const Cx x0 = xn(Ps, ii[c], j, m, nr), x1 = xn(Ps, ii[c], j + 1, m, nr);
                gr.re += wz[c] * (x1.re - x0.re) / g.dn[j]; gr.im += wz[c] * (x1.im - x0.im) / g.dn[j];
                gp.re -= wz[c] * s * x0.im; gp.im += wz[c] * s * x0.re;
            }
            const Cx xr = get(Jr, i, j, m);
            put(Jr, i, j, m, {xr.re + gr.re, xr.im + gr.im});
            if (j > 0) { const Cx xp = get(Jp, i, j, m); put(Jp, i, j, m, {xp.re + gp.re, xp.im + gp.im}); }
        });
    }
}
}  // namespace

// The scratch of the local filter: the transport arrays ner_, nep_, nbr_ of the field step (free between steps).
void rz_filter_currents(FieldsRZ& f, Comm& comm, int passes, bool local, bool binomial) {
    if (passes <= 0) return;
    const Domain& d = f.dom();
    if (f.filter_alpha_.size() != size_t(f.modes())) f.filter_alpha_.upload(mode_alpha(f));
    const Buf<double>& ab = f.filter_alpha_;
    if (!local && !binomial) { filter_projection(f, comm, passes, ab.data()); return; }
    // the compensator (numerics.filter_compensate, binomial only): one more z step with cf = -passes after the passes
    const int last = binomial && !local && d.filt_comp ? passes : passes - 1;
    for (int p = 0; p <= last; ++p) {
        const bool comp = p == passes;
        const double cf = comp ? -double(passes) : 1.0;
        comm.exchange(d, {&f.jz, &f.jr, &f.jp});
        if (binomial && !local) {                                        // S_z on J_z (w = d_z J_z), T_z on J_r, J_phi
            divergence_zz(d, f.jz.view(), f.ner_.view(), f.modes(), zcells(d));
            current_z(d, f.jz.view(), f.ner_.view(), f.modes(), zcells(d), cf);
            charge_z(d, f.jr.view(), f.ner_.view(), zcells(d), cf); copy_owned(d, f.ner_.view(), f.jr.view());
            charge_z(d, f.jp.view(), f.ner_.view(), zcells(d), cf); copy_owned(d, f.ner_.view(), f.jp.view());
        } else {
            divergence_z(d, f.grid().view(), f.jz.view(), f.jr.view(), f.jp.view(), f.ner_.view(), f.modes(), zcells(d));
            current_z(d, f.jz.view(), f.ner_.view(), f.modes(), zcells(d));
        }
        if (comp || (binomial && !local && d.filt_xonly)) continue;     // numerics.filter_axes = x: no radial step
        filter_scalar_r(d, f.grid().view(), f.jz.view(), f.ner_.view(), f.modes(), ab.data()); copy_owned(d, f.ner_.view(), f.jz.view());
        filter_vector_r(d, f.grid().view(), f.jr.view(), f.jp.view(), f.ner_.view(), f.nep_.view(), f.modes(), ab.data());
        copy_owned(d, f.ner_.view(), f.jr.view()); copy_owned(d, f.nep_.view(), f.jp.view());
    }
    if (binomial && !local && d.filt_nflat > 0) {                       // numerics.filter_flat (decision 109)
        Field* X[3] = {&f.jz, &f.jr, &f.jp};
        flat_filter_steps(d, X, 3, [&](double cf) {
            comm.exchange(d, {&f.jz, &f.jr, &f.jp});
            divergence_zz(d, f.jz.view(), f.ner_.view(), f.modes(), zcells(d));
            current_z(d, f.jz.view(), f.ner_.view(), f.modes(), zcells(d), cf);
            charge_z(d, f.jr.view(), f.ner_.view(), zcells(d), cf); copy_owned(d, f.ner_.view(), f.jr.view());
            charge_z(d, f.jp.view(), f.ner_.view(), zcells(d), cf); copy_owned(d, f.ner_.view(), f.jp.view());
        });
    }
}

void rz_filter_rho(FieldsRZ& f, Comm& comm, Field& rho, int passes) {
    if (passes <= 0) return;
    const Domain& d = f.dom();
    if (f.filter_alpha_.size() != size_t(f.modes())) f.filter_alpha_.upload(mode_alpha(f));
    const Buf<double>& ab = f.filter_alpha_;
    const int last = d.filt_comp ? passes : passes - 1;                  // the compensator: one more z step, cf = -passes
    for (int p = 0; p <= last; ++p) {
        const bool comp = p == passes;
        comm.exchange(d, {&rho});
        charge_z(d, rho.view(), f.nbr_.view(), zcells(d), comp ? -double(passes) : 1.0); copy_owned(d, f.nbr_.view(), rho.view());
        if (comp || d.filt_xonly) continue;
        filter_scalar_r(d, f.grid().view(), rho.view(), f.nbr_.view(), f.modes(), ab.data()); copy_owned(d, f.nbr_.view(), rho.view());
    }
    if (d.filt_nflat > 0) {                                              // numerics.filter_flat (decision 109)
        Field* X[1] = {&rho};
        flat_filter_steps(d, X, 1, [&](double cf) {
            comm.exchange(d, {&rho});
            charge_z(d, rho.view(), f.nbr_.view(), zcells(d), cf); copy_owned(d, f.nbr_.view(), rho.view());
        });
    }
}
