// Transverse particle shape for RZ (numerics.rz_transverse_shape = gaussian, the default; noise heating study 2026-11-06,
// decision 79, docs/specs/rz_modes.md "Transverse particle shape"). With M azimuthal modes a macro-particle is a truncated
// Fourier series in phi; ppc_theta particles sample 2M - 1 azimuthal components, and the noise heating grows with 2M - 1
// (bench/noise_heating). A Gaussian of width a in the transverse plane, the same for every mode, damps the noise at large
// k_perp (near the axis the high modes have k_perp ~ m / r) and leaves smooth fields (laser, uniform fields: k_perp ~ 0)
// alone. It acts on the grid, per mode, as S = exp(a^2 / 2 Lap_perp) built from the field solver's own radial operators:
//   scalars at the nodes (J_z, E_z, rho): X <- f(L_TM) X,  f(lam) = exp(-lam a^2 / 2), L_TM = -D_m G_m (iRIP's TM basis);
//   vectors (V_r at half nodes, V_phi at nodes: J_perp, E_perp), split into gradient and curl parts:
//     V <- V + G_m k(L_TM) D_m V - C_m k(L_TE) C_m^T V,  k(lam) = (1 - f(lam)) / lam  (k(0) = a^2 / 2),
//   G_m X = ((X_j+1 - X_j) / dn_j, (i m / r_j) X_j), D_m its negative adjoint (the Gauss law's divergence),
//   C_m Y = ((i m / rh_j) Y_j, -(Y_j - Y_j-1) / dd_j) (curl of Y z^, Y at half nodes), C_m^T V = (r_j+1 V_phi,j+1 -
//   r_j V_phi,j) / B_j - (i m / rh_j) V_r,j, L_TE = C_m^T C_m (iRIP's TE basis).
// D_m C_m = 0 and C_m^T G_m = 0 on the grid, so D_m S V = f(L_TM) D_m V: the current S J deposits the charge f(L_TM) rho,
// the continuity per mode stays exact (rho of the diagnostics gets f(L_TM) too). S is self-adjoint in the field energy's
// inner product, so the gather of S E with the deposit of J and the deposit of S J with the gather of E exchange the same
// energy (deposit and gather fit each other). B is gathered as it is (no work). V_phi on the axis is no DOF (unchanged).
#include <cmath>
#include <stdexcept>
#include "rz.hpp"

struct RzSmoothWork {
    Buf<double> V[2], Vi[2], f[2], k[2];      // per kind (0 TM nodes, 1 TE half nodes): M blocks of Nr x Nr, Nr
    Field W, Y, T;                            // D V (nodes), C^T V (half nodes), transform scratch
};

namespace {
struct Cx { double re, im; };
HD inline int slot(int m, int c) { return m == 0 ? 0 : 2 * m - 1 + c; }
HD inline Cx get(const FV& F, int i, int j, int m) { return {F(i, j, slot(m, 0)), m ? F(i, j, slot(m, 1)) : 0.0}; }
HD inline void put(const FV& F, int i, int j, int m, Cx v) { F(i, j, slot(m, 0)) = v.re; if (m) F(i, j, slot(m, 1)) = v.im; }
HD inline Cx imul(double s, Cx a) { return {-s * a.im, s * a.re}; }   // i s a

// W = D_m V (nodes), Y = C_m^T V (half nodes), on the owned columns
void div_curl(const Domain& d, const RadialView g, const FV Vr, const FV Vp, const FV W, const FV Y, int M) {
    const int nz = d.n[0], nr = d.n[1];
    par_for(long(nz) * M, [=] HD(long q) {
        const int i = int(q % nz), m = int(q / nz);
        for (int j = 0; j < nr; ++j) {
            const Cx vr = get(Vr, i, j, m);
            Cx w;
            if (j == 0) w = m == 0 ? Cx{g.rh[0] * vr.re / g.A[0], 0.0} : Cx{0, 0};
            else {
                const Cx b = get(Vr, i, j - 1, m), p = get(Vp, i, j, m), ip = imul(g.mf(m) / g.r[j], p);
                w = {(g.rh[j] * vr.re - g.rh[j - 1] * b.re) / g.A[j] + ip.re, (g.rh[j] * vr.im - g.rh[j - 1] * b.im) / g.A[j] + ip.im};
            }
            put(W, i, j, m, w);
            const Cx p1 = j + 1 < nr ? get(Vp, i, j + 1, m) : Cx{0, 0}, p0 = j > 0 ? get(Vp, i, j, m) : Cx{0, 0};
            const Cx ir = imul(g.mf(m) / g.rh[j], vr);
            put(Y, i, j, m, {(g.r[j + 1] * p1.re - g.r[j] * p0.re) / g.B[j] - ir.re, (g.r[j + 1] * p1.im - g.r[j] * p0.im) / g.B[j] - ir.im});
        }
    });
}

// V += G_m W - C_m Y on the owned columns (W at the nodes, 0 on the axis for m >= 1 and beyond the wall; Y at half nodes)
void add_grad_curl(const Domain& d, const RadialView g, const FV W, const FV Y, const FV Vr, const FV Vp, int M) {
    const int nz = d.n[0], nr = d.n[1];
    par_for(long(nz) * M, [=] HD(long q) {
        const int i = int(q % nz), m = int(q / nz);
        auto wn = [&](int j) { return j >= nr || (m >= 1 && j == 0) ? Cx{0, 0} : get(W, i, j, m); };
        for (int j = 0; j < nr; ++j) {
            const Cx w0 = wn(j), w1 = wn(j + 1), y0 = get(Y, i, j, m), cy = imul(g.mf(m) / g.rh[j], y0);
            Cx vr = get(Vr, i, j, m);
            vr.re += (w1.re - w0.re) / g.dn[j] - cy.re; vr.im += (w1.im - w0.im) / g.dn[j] - cy.im;
            put(Vr, i, j, m, vr);
            if (j == 0) continue;                                  // V_phi on the axis: no DOF
            const Cx gp = imul(g.mf(m) / g.r[j], w0), ym = get(Y, i, j - 1, m);
            Cx vp = get(Vp, i, j, m);
            vp.re += gp.re + (y0.re - ym.re) / g.dd[j]; vp.im += gp.im + (y0.im - ym.im) / g.dd[j];
            put(Vp, i, j, m, vp);
        }
    });
}

// X <- V diag(c) Vi X per column and slot (the basis of the kind, real; both slots of a mode alike)
void apply_fun(const Domain& d, const FV X, const FV T, const double* V, const double* Vi, const double* c) {
    const int nz = d.n[0], nr = d.n[1], S = d.n[2];
    const long nn = long(nr) * nr;
    par_for(long(nz) * S, [=] HD(long q) {
        const int i = int(q % nz), k = int(q / nz), m = (k + 1) / 2;
        const double* A = Vi + m * nn;
        for (int a = 0; a < nr; ++a) {
            double s = 0;
            for (int j = 0; j < nr; ++j) s += A[long(a) * nr + j] * X(i, j, k);
            T(i, a, k) = c[long(m) * nr + a] * s;
        }
    });
    par_for(long(nz) * S, [=] HD(long q) {
        const int i = int(q % nz), k = int(q / nz), m = (k + 1) / 2;
        const double* A = V + m * nn;
        for (int j = 0; j < nr; ++j) {
            double s = 0;
            for (int a = 0; a < nr; ++a) s += A[long(j) * nr + a] * T(i, a, k);
            X(i, j, k) = s;
        }
    });
}

void copy_all(const Domain& d, const FV src, const FV dst) {
    const int g = d.ng, sx = d.n[0] + 2 * g, sy = d.n[1] + 2 * g, sz = d.n[2] + 2 * g;
    par_for(long(sx) * sy * sz, [=] HD(long q) {
        const int i = int(q % sx) - g, j = int((q / sx) % sy) - g, k = int(q / (long(sx) * sy)) - g;
        dst(i, j, k) = src(i, j, k);
    });
}

RzSmoothWork& work(FieldsRZ& f) {
    if (!f.smooth_) {
        const Domain& d = f.dom();
        auto W = std::make_shared<RzSmoothWork>();
        W->W.resize(d); W->Y.resize(d); W->T.resize(d);
        const double h = 0.5 * f.smooth_a * f.smooth_a;
        for (int kind = 0; kind < 2; ++kind) {
            std::vector<double> V, Vi, fv, kv, vb, vib, lb;
            for (int m = 0; m < f.modes(); ++m) {
                std::vector<double> h2;
                rz_basis(f.grid(), m, kind, vb, vib, lb, h2);
                V.insert(V.end(), vb.begin(), vb.end()); Vi.insert(Vi.end(), vib.begin(), vib.end());
                for (double l : lb) {
                    fv.push_back(std::exp(-h * l));
                    kv.push_back(l * h > 1e-8 ? -std::expm1(-h * l) / l : h);
                }
            }
            W->V[kind].upload(V); W->Vi[kind].upload(Vi); W->f[kind].upload(fv); W->k[kind].upload(kv);
        }
        f.smooth_ = W;
    }
    return *f.smooth_;
}

// (Vr, Vp) <- S (Vr, Vp) on the owned columns (guards of the inputs unused)
void smooth_vector(FieldsRZ& f, Field& Vr, Field& Vp) {
    RzSmoothWork& W = work(f);
    const Domain& d = f.dom();
    div_curl(d, f.grid().view(), Vr.view(), Vp.view(), W.W.view(), W.Y.view(), f.modes());
    apply_fun(d, W.W.view(), W.T.view(), W.V[0].data(), W.Vi[0].data(), W.k[0].data());
    apply_fun(d, W.Y.view(), W.T.view(), W.V[1].data(), W.Vi[1].data(), W.k[1].data());
    add_grad_curl(d, f.grid().view(), W.W.view(), W.Y.view(), Vr.view(), Vp.view(), f.modes());
}

void smooth_scalar(FieldsRZ& f, Field& X) {
    RzSmoothWork& W = work(f);
    apply_fun(f.dom(), X.view(), W.T.view(), W.V[0].data(), W.Vi[0].data(), W.f[0].data());
}
}  // namespace

void rz_smooth_currents(FieldsRZ& f, Comm& comm) {
    if (f.smooth_a <= 0) return;
    smooth_scalar(f, f.jz);
    smooth_vector(f, f.jr, f.jp);
    comm.exchange(f.dom(), {&f.jz, &f.jr, &f.jp});
}

void rz_smooth_e(FieldsRZ& f, Comm& comm) {
    if (f.smooth_a <= 0) return;
    const Domain& d = f.dom();
    if (!f.se_alloc_) { f.sez_.resize(d); f.ser_.resize(d); f.sep_.resize(d); f.se_alloc_ = true; }
    copy_all(d, f.ez.view(), f.sez_.view()); copy_all(d, f.er.view(), f.ser_.view()); copy_all(d, f.ep.view(), f.sep_.view());
    if (f.smooth_a > 0) {
        smooth_scalar(f, f.sez_);
        smooth_vector(f, f.ser_, f.sep_);
        comm.exchange(d, {&f.sez_, &f.ser_, &f.sep_});
    }
    if (f.axis_regular) rz_axis_regular_gather_e(f, f.ser_, f.sep_);   // decision 112: the adjoint of the first-cell projection
}

void rz_smooth_rho(FieldsRZ& f, Comm& comm, Field& rho) {
    if (f.smooth_a <= 0) return;
    smooth_scalar(f, rho);
    comm.exchange(f.dom(), {&rho});
}

void rz_smooth_apply(FieldsRZ& f, Field& X, int kind_or_vector, Field* Vp) {
    if (kind_or_vector == 0) smooth_scalar(f, X);
    else smooth_vector(f, X, *Vp);
}
