// O7 in RZ (medium.hpp, decision 84): the medium current per azimuthal mode from the work field of the RZ box step
// (src/rz_irip.cpp rz_irip_trial, decision 82) without the medium's own source, and the medium charge -div P per mode.
// The coefficients are axisymmetric (wp2(z, r)), so every mode and both slots (Re, Im) of m >= 1 respond alike.
#include "medium.hpp"
#include "rz.hpp"

namespace {
struct Cr { double re, im; };
HD inline Cr operator+(Cr a, Cr b) { return {a.re + b.re, a.im + b.im}; }
HD inline Cr operator-(Cr a, Cr b) { return {a.re - b.re, a.im - b.im}; }
HD inline Cr operator*(double s, Cr a) { return {s * a.re, s * a.im}; }
HD inline Cr imr(double s, Cr a) { return {-s * a.im, s * a.re}; }                 // i s a
HD inline int slt(int m, int c) { return m == 0 ? 0 : 2 * m - 1 + c; }
HD inline Cr gt(const FV& F, int i, int j, int m) { return {F(i, j, slt(m, 0)), m ? F(i, j, slt(m, 1)) : 0.0}; }
HD inline Cr mzr(const FV& F, int i, int j, int m) { return 0.5 * (gt(F, i, j, m) + gt(F, i + 1, j, m)); }
}  // namespace

// Et0 per mode (rz_irip_trial with the external currents only):
//   Et_z = Ez*,  Et_r = [(E_r + B_phi)(i) + (E_r - B_phi)(i+1)] / 2 + D/2 Psi_r - D/4 d_r [Ez*(i+1) - Ez*(i)],
//   Et_phi = [(E_phi - B_r)(i) + (E_phi + B_r)(i+1)] / 2 + D/2 Psi_phi + D/4 G_phi [Ez*(i+1) - Ez*(i)]  (0 on the axis)
void medium_current_rz(Medium& Md, const Domain& d, const RadialView g, int modes, const FV Er, const FV Ep, const FV Br,
                       const FV Bp, const FV Jz, const FV Jr, const FV Jp, const FV Exs, const FV Bxs) {
    const double D = d.h[0];
    const int nz = d.n[0], nr = d.n[1];
    const FV W0 = Md.wp2[0].view(), W1 = Md.wp2[1].view(), W2 = Md.wp2[2].view();
    const FV O0 = Md.w02[0].view(), O1 = Md.w02[1].view(), O2 = Md.w02[2].view();
    const FV N0 = Md.nu[0].view(), N1 = Md.nu[1].view(), N2 = Md.nu[2].view();
    const FV J0 = Md.J[0].view(), J1 = Md.J[1].view(), J2 = Md.J[2].view();
    const FV P0 = Md.P[0].view(), P1 = Md.P[1].view(), P2 = Md.P[2].view();
    const FV U0 = Md.u[0].view(), U1 = Md.u[1].view(), U2 = Md.u[2].view();
    const FV T0 = Md.jt[0].view(), T1 = Md.jt[1].view(), T2 = Md.jt[2].view();
    const bool bnd = Md.bound;
    const FV LR = bnd ? Md.bLr.view() : W0, LP = bnd ? Md.bLp.view() : W0, ER = bnd ? Md.bEr.view() : W0, EP = bnd ? Md.bEp.view() : W0;
    par_for(long(nz) * nr * modes, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), m = int(q / (long(nz) * nr));
        const Cr ez = gt(Exs, i, j, m);
        const Cr tr = 0.5 * (gt(Er, i, j, m) + gt(Bp, i, j, m) + gt(Er, i + 1, j, m) - gt(Bp, i + 1, j, m));
        const Cr psr = imr(g.mf(m) / g.rh[j], mzr(Bxs, i, j, m)) - gt(Jr, i, j, m);
        const Cr dex = (1 / g.dn[j]) * ((gt(Exs, i + 1, j + 1, m) - gt(Exs, i, j + 1, m)) - (gt(Exs, i + 1, j, m) - gt(Exs, i, j, m)));
        const Cr er = tr + (0.5 * D) * psr - (0.25 * D) * dex;
        Cr ep{0, 0};
        if (j > 0) {
            const Cr tp = 0.5 * (gt(Ep, i, j, m) - gt(Br, i, j, m) + gt(Ep, i + 1, j, m) + gt(Br, i + 1, j, m));
            const Cr pst = (-1 / g.dd[j]) * (mzr(Bxs, i, j, m) - mzr(Bxs, i, j - 1, m)) - gt(Jp, i, j, m);
            const Cr gph = imr(-g.mf(m) / g.r[j], gt(Exs, i + 1, j, m) - gt(Exs, i, j, m));
            ep = tp + (0.5 * D) * pst + (0.25 * D) * gph;
        }
        for (int c = 0; c < (m ? 2 : 1); ++c) {
            const int k = slt(m, c);
            auto cur = [&](const FV& W, const FV& O, const FV& N, const FV& J, const FV& P, double E, double kap) {
                const double w = W(i, j, k), o = O(i, j, k);
                return w == 0.0 ? 0.0 : (2 * J(i, j, k) + D * w * E - D * o * P(i, j, k)) / (2 + D * N(i, j, k) + 0.5 * D * D * o + 0.5 * kap * D * D * w);
            };
            const double uz = cur(W0, O0, N0, J0, P0, c ? ez.im : ez.re, 0.0);
            // model = bound: the atoms' own source with the lagged atom current (medium_rz_bound.cpp), then the linear media; the Et
            // seen by the atoms is that minus the linear media's own source
            const double erc = (c ? er.im : er.re) - (bnd ? 0.5 * D * LR(i, j, k) : 0.0);
            const double epc = (c ? ep.im : ep.re) - (bnd ? 0.5 * D * LP(i, j, k) : 0.0);
            const double ur = cur(W1, O1, N1, J1, P1, erc, 1.0);
            const double up = cur(W2, O2, N2, J2, P2, epc, 1.0);
            if (bnd) { ER(i, j, k) = erc - 0.5 * D * ur; EP(i, j, k) = epc - 0.5 * D * up; }
            U0(i, j, k) = uz; U1(i, j, k) = ur; U2(i, j, k) = up;
            T0(i, j, k) = Jz(i, j, k) + uz; T1(i, j, k) = Jr(i, j, k) + ur; T2(i, j, k) = Jp(i, j, k) + up;
        }
    });
    if (bnd) medium_bound_rz_current(Md, d, g, modes);   // the atoms: u_b at the angles, deposited per mode into jt
}

// rho_med = -[(Pz(i+1) - Pz(i)) / dz + div_n(Pr, Pphi)] at the Gauss point (i+1/2, r_j) per mode (the continuity of the RZ step:
// div_n = (1/A_j)(rh_j Pr_j - rh_{j-1} Pr_{j-1}) + (i m / r_j) Pphi_j; the axis: m = 0 rh_0 Pr_0 / A_0, m >= 1 no Gauss point)
void medium_add_rho_rz(const Medium& Md, const Domain& d, const RadialView g, int modes, Field& rho) {
    const double dz = d.h[0];
    const int nz = d.n[0], nr = d.n[1];
    const FV Pz = Md.P[0].view(), Pr = Md.P[1].view(), Pp = Md.P[2].view(), R = rho.view();
    par_for(long(nz) * nr * modes, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), m = int(q / (long(nz) * nr));
        if (m >= 1 && j == 0) return;
        Cr dv = (1 / dz) * (gt(Pz, i + 1, j, m) - gt(Pz, i, j, m));
        if (j == 0) dv = dv + Cr{g.rh[0] * Pr(i, 0, 0) / g.A[0], 0.0};
        else dv = dv + (1 / g.A[j]) * (g.rh[j] * gt(Pr, i, j, m) - g.rh[j - 1] * gt(Pr, i, j - 1, m)) + imr(g.mf(m) / g.r[j], gt(Pp, i, j, m));
        R(i, j, slt(m, 0)) -= dv.re;
        if (m) R(i, j, slt(m, 1)) -= dv.im;
    });
}
