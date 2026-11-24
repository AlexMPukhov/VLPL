// O7, model = bound in RZ (medium.hpp, decision 85, docs/specs/linear_medium.md "Bound electrons"): the atoms of a gas with the
// dipole along x on nphi azimuthal samples at the Jr positions (z_{i+1/2}, rh_j), j = 0 .. nr - 1.
//
// Field at an atom: Et_x(phi) = Et_r(rh_j, phi) cos(phi) - Et_phi(rh_j, phi) sin(phi), Et_c(phi) = a_0 + 2 sum_m Re(a_m e^{i m phi})
// (the slots (Re, Im) of mode m), Et_phi interpolated linearly between the nodes r_j and r_{j+1}, (1 - th_j) Et_phi(r_j) +
// th_j Et_phi(r_{j+1}); on the axis (j = 0) Et_phi,1(0) = i Et_r,1(rh_0) (regularity of E_x + i E_y), the other modes 0.
// The atom current u_x goes back to the modes with the transpose of that map in the energy inner product of the RZ step
// (weights 2 pi B_j dz for Jr, 2 pi A_j dz for Jphi, factor 2 for m >= 1; an atom 2 pi B_j dz / nphi; the angular quadrature
// (1/nphi) sum f g = f_0 g_0 + 2 sum Re(f_m g_m^*) is exact for nphi >= 2 M + 1):
//   u_r(j, m) = P_m[u_x(j) cos] + [j = 0, m = 1] (1 - th_0) (-i) P_1[-u_x(0) sin],
//   u_phi(j, m) = -(1 / A_j) (B_j (1 - th_j) P_m[u_x(j) sin] + B_{j-1} th_{j-1} P_m[u_x(j-1) sin])   (j >= 1),
// P_m[f] = (1/nphi) sum_k f(phi_k) e^{-i m phi_k}. So sum over the atoms of V Et_x u_x = the field's work with (u_r, u_phi):
// the energy exchange stays exact. The atoms' own source (-dt/2 u in Et_r, Et_phi) couples neighbouring nodes, so it is not
// eliminated locally as in Cart3D: the iterate uses the atom current of the previous outer iteration (bLr, bLp), and the outer
// iteration (medium_bound_rz_solve) runs until Et* and that lag are consistent.
#include "env.hpp"
#include "medium.hpp"
#include "rz.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {
HD inline int slt(int m, int c) { return m == 0 ? 0 : 2 * m - 1 + c; }
struct RA {                                       // the raw atom arrays
    double* p[RA_N];
};
RA ra_view(const Medium& M) {
    RA a;
    for (int q = 0; q < RA_N; ++q) a.p[q] = const_cast<double*>(M.ra[size_t(q)].data());
    return a;
}
// the radial interpolation weight th_j = (rh_j - r_j) / (r_{j+1} - r_j)
HD inline double theta(const RadialView& g, int j) { return (g.rh[j] - g.r[j]) / (g.r[j + 1] - g.r[j]); }
}  // namespace

void medium_bound_rz_setup(Medium& M, const Domain& d, const RadialView g, int modes, int nphi) {
    (void)g;
    if (nphi < 2 * modes - 1) throw std::runtime_error("medium (bound, rz): n_phi >= 2 m_max + 1");
    M.nphi = nphi;
    const size_t na = size_t(d.n[0]) * d.n[1] * nphi;
    M.ra.clear();
    for (int q = 0; q < RA_N; ++q) { M.ra.emplace_back(na); M.ra.back().zero(); }
    double* p0 = M.ra[BS_P0].data();
    par_for(long(na), [=] HD(long q) { p0[q] = 1.0; });
    for (Field* f : {&M.bLr, &M.bLp, &M.bEr, &M.bEp, &M.bUr, &M.bUp, &M.bPr, &M.bPp}) { f->resize(d); f->fill(0); }
}

// after medium_current_rz (bEr, bEp: Et per mode with the lagged atom current; jt: J + linear media): the atoms' linearized
// current at the angles from Et_x, then u_r, u_phi per mode into bUr, bUp and jt
void medium_bound_rz_current(Medium& M, const Domain& d, const RadialView g, int modes) {
    const int nz = d.n[0], nr = d.n[1], np = M.nphi;
    const RA a = ra_view(M);
    const FV N = M.bn.view(), ER = M.bEr.view(), EP = M.bEp.view();
    const double dph = 2 * M_PI / np;
    par_for(long(nz) * nr * np, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
        if (N(i, j, 0) == 0.0) { a.p[RA_E][q] = 0; a.p[RA_U][q] = 0; return; }
        const double ph = k * dph, cp = cos(ph), sp = sin(ph), th = theta(g, j);
        double er = ER(i, j, 0), ef = 0;
        if (j > 0) ef = (1 - th) * EP(i, j, 0) + th * EP(i, j + 1, 0);
        else ef = th * EP(i, 1, 0);
        for (int m = 1; m < modes; ++m) {
            const double cm = cos(m * ph), sm = sin(m * ph);
            const double rr = ER(i, j, slt(m, 0)), ri = ER(i, j, slt(m, 1));
            er += 2 * (rr * cm - ri * sm);
            double pr = th * EP(i, j + 1, slt(m, 0)), pi = th * EP(i, j + 1, slt(m, 1));
            if (j > 0) { pr += (1 - th) * EP(i, j, slt(m, 0)); pi += (1 - th) * EP(i, j, slt(m, 1)); }
            else if (m == 1) { pr += (1 - th) * (-ri); pi += (1 - th) * rr; }     // Et_phi,1(0) = i Et_r,1(rh_0)
            ef += 2 * (pr * cm - pi * sm);
        }
        const double ex = er * cp - ef * sp;
        a.p[RA_E][q] = ex;
        a.p[RA_U][q] = a.p[RA_U0][q] + a.p[RA_SG][q] * (ex - a.p[RA_ES][q]);
    });
    // the deposit per mode (transpose of the map above)
    const FV UR = M.bUr.view(), UP = M.bUp.view(), T1 = M.jt[1].view(), T2 = M.jt[2].view();
    par_for(long(nz) * nr * modes, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), m = int(q / (long(nz) * nr));
        double rc = 0, rs = 0, sc0 = 0, ss0 = 0, sc1 = 0, ss1 = 0, xc = 0, xs = 0;   // P_m[u cos], P_m[u sin] at j, j - 1; axis term
        for (int k = 0; k < np; ++k) {
            const double ph = k * dph, cp = cos(ph), sp = sin(ph), cm = cos(m * ph), sm = sin(m * ph);
            const long a0 = (long(k) * nr + j) * nz + i;
            const double u = a.p[RA_U][a0];
            rc += u * cp * cm; rs -= u * cp * sm;                  // e^{-i m phi} = cos - i sin
            sc0 += u * sp * cm; ss0 -= u * sp * sm;
            if (j > 0) {
                const double u1 = a.p[RA_U][a0 - nz];
                sc1 += u1 * sp * cm; ss1 -= u1 * sp * sm;
            }
            if (j == 0 && m == 1) { xc += -u * sp * cm; xs -= -u * sp * sm; }
        }
        const double f = 1.0 / np;
        double ur_re = f * rc, ur_im = f * rs;
        if (j == 0 && m == 1) {                                    // (1 - th_0) (-i) P_1[-u sin]
            const double th0 = theta(g, 0);
            ur_re += (1 - th0) * f * xs; ur_im += -(1 - th0) * f * xc;
        }
        double up_re = 0, up_im = 0;
        if (j > 0) {
            const double wj = g.B[j] * (1 - theta(g, j)), wl = g.B[j - 1] * theta(g, j - 1);
            up_re = -(wj * f * sc0 + wl * f * sc1) / g.A[j];
            up_im = -(wj * f * ss0 + wl * f * ss1) / g.A[j];
        }
        for (int c = 0; c < (m ? 2 : 1); ++c) {
            const int s = slt(m, c);
            const double ur = c ? ur_im : ur_re, up = c ? up_im : up_re;
            UR(i, j, s) = ur; UP(i, j, s) = up;
            T1(i, j, s) += ur; T2(i, j, s) += up;
        }
    });
}

namespace {
void rz_linearize(Medium& M, const Domain& d) {
    const BlochC L = M.bc;
    const RA a = ra_view(M);
    const FV N = M.bn.view();
    const int nz = d.n[0], nr = d.n[1];
    const double D = d.dt();
    par_for(long(nz) * nr * M.nphi, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr);
        const double n = N(i, j, 0);
        if (n == 0.0) return;
        double s[BS_N], o[BS_N];
        for (int r = 0; r < BS_N; ++r) s[r] = a.p[r][q];
        const double p0 = bloch_P(L, s), e = a.p[RA_ES][q], de = 1e-6 * (fabs(e) + 1e-3);
        bloch_step(L, s, e, o);
        a.p[RA_U0][q] = n * (bloch_P(L, o) - p0) / D;
        bloch_step(L, s, e + de, o);
        const double up = bloch_P(L, o);
        bloch_step(L, s, e - de, o);
        a.p[RA_SG][q] = n * (up - bloch_P(L, o)) / (2 * de * D);
    });
}
}  // namespace

void medium_bound_rz_solve(Medium& M, const Domain& d, Comm& comm, double t, const std::function<void()>& solve,
                           const std::function<void()>& prepare) {
    const long na = long(d.n[0]) * d.n[1] * M.nphi;
    double* E = M.ra[RA_E].data();
    double* Es = M.ra[RA_ES].data();
    if (M.bt != t) {                                // a new step: linearize at the last step's Et (the lag is the last u_b)
        par_for(na, [=] HD(long q) { Es[q] = E[q]; });
        M.bt = t;
    }
    const long nl = long(M.bUr.size());
    int it = 0;
    bool conv = false;
    for (; it < M.bmaxit; ++it) {
        rz_linearize(M, d);
        solve();
        prepare();                                  // E, bUr, bUp of the solution with this linearization and lag
        double v[3] = {par_max_abs(na, [=] HD(long q) { return E[q] - Es[q]; }), par_max_abs(na, [=] HD(long q) { return E[q]; }), 0};
        // the lag's error in Et is dt/2 |u_b - lag| (owned cells: the guards of the lag hold exchanged values after a restart)
        const FV UR = M.bUr.view(), LR = M.bLr.view(), UP = M.bUp.view(), LP = M.bLp.view();
        const int nz = d.n[0], nr = d.n[1];
        (void)nl;
        v[2] = 0.5 * d.dt() * par_max_abs(d.owned(), [=] HD(long q) {
            const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
            return fmax(fabs(UR(i, j, k) - LR(i, j, k)), fabs(UP(i, j, k) - LP(i, j, k)));
        });
        v[0] = comm.allreduce_max(v[0]); v[1] = comm.allreduce_max(v[1]); v[2] = comm.allreduce_max(v[2]);
        if (vlpl_getenv("VLPL_MEDIUM_DEBUG") && comm.rank == 0)
            std::fprintf(stderr, "bound rz newton %d: |dEt| %.3e |Et| %.3e lag %.3e\n", it, v[0], v[1], v[2]);
        if ((std::max(v[0], v[2]) <= M.btol * std::max(v[1], 1e-300)) || v[1] == 0) { conv = true; ++it; break; }
        par_for(na, [=] HD(long q) { Es[q] = E[q]; });
        M.bLr.swap(M.bUr); M.bLp.swap(M.bUp);       // the new lag (bUr, bUp are recomputed by the next prepare)
    }
    MediumStats& st = M.stats;
    ++st.newton_steps; st.newton += it;
    st.newton_max = std::max<long>(st.newton_max, it);
    if (!conv) ++st.newton_unconverged;
}

void medium_bound_rz_commit(Medium& M, const Domain& d, Comm& comm, const RadialView g) {
    const BlochC L = M.bc;
    const RA a = ra_view(M);
    const FV N = M.bn.view();
    const int nz = d.n[0], nr = d.n[1], np = M.nphi;
    const double D = d.dt(), dz = d.h[0];
    const bool ion = L.ion;
    double v[2] = {0, 0};
    double* Dn = M.ra[RA_U0].data();                 // scratch: the ionized density of this step (u* is recomputed next step)
    v[0] = par_sum(long(nz) * nr * np, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr);
        const double n = N(i, j, 0);
        if (n == 0.0) { Dn[q] = 0; return 0.0; }
        double s[BS_N];
        for (int r = 0; r < BS_N; ++r) s[r] = a.p[r][q];
        const double k0 = bloch_energy(L, s), tr0 = s[BS_P0] + s[BS_P1] + s[BS_P2];
        bloch_step(L, s, a.p[RA_E][q], s);
        for (int r = 0; r < BS_N; ++r) a.p[r][q] = s[r];
        const double V = 2 * M_PI * g.B[j] * dz / np;
        const double dn = ion ? n * (tr0 - s[BS_P0] - s[BS_P1] - s[BS_P2]) : 0.0;
        a.p[RA_ACC][q] += dn;
        Dn[q] = dn * V;
        return L.loss || ion ? (D * a.p[RA_E][q] * a.p[RA_U][q] - n * (bloch_energy(L, s) - k0)) * V : 0.0;
    });
    // the atoms' polarization per mode (Gauss law): bP += dt u_b
    const FV UR = M.bUr.view(), UP = M.bUp.view(), PR = M.bPr.view(), PP = M.bPp.view();
    for_owned(d, [=] HD(int i, int j, int k) { PR(i, j, k) += D * UR(i, j, k); PP(i, j, k) += D * UP(i, j, k); });
    if (L.loss || ion) { comm.allreduce_sum(&v[0], 1); M.dissipated += v[0]; }
    if (ion) {
        v[1] = par_sum(long(nz) * nr * np, [=] HD(long q) { return Dn[q]; });
        comm.allreduce_sum(&v[1], 1);
        M.ionized += v[1];
    }
    // the lag for the next step: this step's atom current
    M.bLr.swap(M.bUr); M.bLp.swap(M.bUp);
}

double medium_bound_rz_energy(const Medium& M, const Domain& d, const RadialView g) {
    const BlochC L = M.bc;
    const RA a = ra_view(M);
    const FV N = M.bn.view();
    const int nz = d.n[0], nr = d.n[1], np = M.nphi;
    const double dz = d.h[0];
    return par_sum(long(nz) * nr * np, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr);
        const double n = N(i, j, 0);
        if (n == 0.0) return 0.0;
        double s[BS_N] = {0};
        s[BS_P1] = a.p[BS_P1][q]; s[BS_P2] = a.p[BS_P2][q];
        return n * bloch_energy(L, s) * 2 * M_PI * g.B[j] * dz / np;
    });
}

// -div bP per mode with the stencil of medium_add_rho_rz (Pz = 0)
void medium_bound_rz_add_rho(const Medium& M, const Domain& d, const RadialView g, int modes, Field& rho) {
    const int nz = d.n[0], nr = d.n[1];
    const FV Pr = M.bPr.view(), Pp = M.bPp.view(), R = rho.view();
    par_for(long(nz) * nr * modes, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), m = int(q / (long(nz) * nr));
        if (m >= 1 && j == 0) return;
        for (int c = 0; c < (m ? 2 : 1); ++c) {
            const int s = slt(m, c);
            double dv;
            if (j == 0) dv = g.rh[0] * Pr(i, 0, 0) / g.A[0];
            else {
                dv = (g.rh[j] * Pr(i, j, s) - g.rh[j - 1] * Pr(i, j - 1, s)) / g.A[j];
                // (i m / r_j) Pphi: Re part -m/r Im, Im part m/r Re
                const double pre = Pp(i, j, slt(m, 0)), pim = m ? Pp(i, j, slt(m, 1)) : 0.0;
                dv += c ? g.mf(m) / g.r[j] * pre : -g.mf(m) / g.r[j] * pim;
            }
            R(i, j, s) -= dv;
        }
    });
}
