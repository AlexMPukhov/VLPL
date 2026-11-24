// numerics.rz_axis_current = regular (decision 112; Alexander 2026-11-20: "this is the most plausible error. fix it"; nci/rzm1):
// the m = 1 transverse current in the first radial cell keeps only its regular part.
// For m = 1 the transverse current has two parts near the axis: J_+ = J_r - i J_phi ~ r^0 (a uniform J_x: F_r1 = J_x/2, F_phi1 = i J_x/2,
// VLPL's F = F_0 + 2 Re sum F_m e^{i m phi}) and J_- = J_r + i J_phi ~ r^2. The deposit next to the axis fills J_- in the first cell
// (J_r at the half node r_{1/2}, J_phi at the node r_1) with noise of the size of J_+ (bunch-column run, step 900: |J_-|/|J_+| = 12.6 at
// r = dr, the 3D grid 0.10), and that part drives the TE axis bound state (lambda dr^2 = 6.365, the line at 0.48 pi): 99.8 % of its drive
// power comes from the first cell, and with J_- = 0 there the drive power falls to 1e-3 (nci/rzm1/drive.py).
// Charge: the m = 1 continuity at node 1 constrains only -(rh_0 / A_1) J_r,0 + (i m_f / r_1) J_phi,1 (node 0 carries no m = 1 charge),
// so the shift J_r,0 += d, J_phi,1 += -i k d with k = rh_0 r_1 / (m_f A_1) is divergence free: rho, the continuity and the Gauss law are
// unchanged to round-off. With d = -D / (1 + k), D = J_r,0 + i J_phi,1, the first cell's J_- is zero. On the uniform grid k = 1/2 and the
// total transverse current moment sum (B_j Re F_r,j + A_j Im F_phi,j) is unchanged (B_0 = rh_0 r_1). A uniform transverse current has
// D = 0 (unchanged); a smooth one D = O(dr^2) of J_+ (the laser). Applied in FieldsRZ::step after the exchange of the currents, so every
// deposit path (phase, chord, the implicit mover, ionization) and every backend sees it; columns are independent (MPI along z).
// Modes m >= 2 (regularity r^{m-1}, r^{m+1}) are not changed.
#include <cmath>
#include "rz.hpp"

void rz_axis_regular_current(FieldsRZ& f) {
    if (f.modes() < 2) return;
    const Domain& d = f.dom();
    const RadialGrid& g = f.grid();
    if (g.nr < 2) return;
    const double k = g.rh[0] * g.r[1] / (g.mf(1) * g.A[1]);
    const double c = 1.0 / (1.0 + k);
    const FV Jr = f.jr.view(), Jp = f.jp.view();
    const int ng = d.ng, cols = d.n[0] + 2 * ng;
    par_for(long(cols), [=] HD(long q) {
        const int i = int(q) - ng;
        // slots of mode 1: 1 (Re), 2 (Im)
        const double rr = Jr(i, 0, 1), ri = Jr(i, 0, 2), pr = Jp(i, 1, 1), pi = Jp(i, 1, 2);
        const double Dr = rr - pi, Di = ri + pr;               // D = J_r,0 + i J_phi,1
        const double dr_ = -c * Dr, di = -c * Di;              // d = -D / (1 + k)
        Jr(i, 0, 1) = rr + dr_; Jr(i, 0, 2) = ri + di;
        Jp(i, 1, 1) = pr + k * di; Jp(i, 1, 2) = pi - k * dr_;   // J_phi,1 += -i k d = k (Im d - i Re d)
    });
}

// The gather's E (decision 112, energy): with J' = P J in the first cell the field work sum J' . E equals the particles' work only if the
// gather uses E' = P^T E (P^T the adjoint in the energy weights B_0 for E_r,0 and A_1 for E_phi,1): s = c (B_0 E_r,0 + i k A_1 E_phi,1),
// E_r,0 -= s / B_0, E_phi,1 += i s / A_1 (mode 1, every column including the guards). A uniform transverse field has s = 0 (B_0 = k A_1 on
// the uniform grid): its gather stays exact. Applied to the gather copies (rz_smooth_e), not to the fields.
void rz_axis_regular_gather_e(FieldsRZ& f, Field& er, Field& ep) {
    if (f.modes() < 2) return;
    const Domain& d = f.dom();
    const RadialGrid& g = f.grid();
    if (g.nr < 2) return;
    const double k = g.rh[0] * g.r[1] / (g.mf(1) * g.A[1]);
    const double c = 1.0 / (1.0 + k), B0 = g.B[0], A1 = g.A[1];
    const FV Er = er.view(), Ep = ep.view();
    const int ng = d.ng, cols = d.n[0] + 2 * ng;
    par_for(long(cols), [=] HD(long q) {
        const int i = int(q) - ng;
        const double a = Er(i, 0, 1), b = Er(i, 0, 2), p = Ep(i, 1, 1), qq = Ep(i, 1, 2);
        const double sr = c * (B0 * a - k * A1 * qq), si = c * (B0 * b + k * A1 * p);
        Er(i, 0, 1) = a - sr / B0; Er(i, 0, 2) = b - si / B0;
        Ep(i, 1, 1) = p - si / A1; Ep(i, 1, 2) = qq + sr / A1;
    });
}

// The gather copies without the transverse shape (smooth_a = 0): E copied into sez_, ser_, sep_ (every cell including the guards) and the
// adjoint applied; called by the push itself, so the copies are never stale (with the transverse shape rz_smooth_e builds them).
void rz_axis_gather_copies(FieldsRZ& f) {
    const Domain& d = f.dom();
    if (!f.se_alloc_) { f.sez_.resize(d); f.ser_.resize(d); f.sep_.resize(d); f.se_alloc_ = true; }
    const int g = d.ng, sx = d.n[0] + 2 * g, sy = d.n[1] + 2 * g, sz = d.n[2];
    const FV a0 = f.ez.view(), a1 = f.er.view(), a2 = f.ep.view(), b0 = f.sez_.view(), b1 = f.ser_.view(), b2 = f.sep_.view();
    par_for(long(sx) * sy * sz, [=] HD(long q) {
        const int i = int(q % sx) - g, j = int((q / sx) % sy) - g, k = int(q / (long(sx) * sy));
        b0(i, j, k) = a0(i, j, k); b1(i, j, k) = a1(i, j, k); b2(i, j, k) = a2(i, j, k);
    });
    rz_axis_regular_gather_e(f, f.ser_, f.sep_);
}

// J_z of the modes m >= 1 on the axis node is zero (J_z ~ r^m; Alexander 2026-11-21: "m = 1 Ez and Jz must be physically correct
// everywhere"). The deposit leaves the folded flux of a particle next to the axis there; the field step never used it (E_z of m >= 1 is
// held at zero on the axis), but the filter spreads it to the first node (odd m: the mirror value) and the output showed it. Every
// column including the guards; called after the deposit (rz_push_tail, before and after the filter), in FieldsRZ::step and rz_irip_trial.
void rz_axis_zero_jz(FieldsRZ& f) {
    const Domain& d = f.dom();
    const int S = d.n[2];
    if (S < 2) return;
    const FV Jz = f.jz.view();
    const int ng = d.ng, cols = d.n[0] + 2 * ng;
    par_for(long(cols) * (S - 1), [=] HD(long q) {
        const int i = int(q % cols) - ng, k = 1 + int(q / cols);
        Jz(i, 0, k) = 0;
    });
}
