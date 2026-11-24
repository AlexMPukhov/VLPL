// Quasi-3D RZ with azimuthal modes (docs/specs/rz_modes.md, decision 43): fields per mode and the
// RIP-CC / iRIP field steps. F(z, r, phi) = F_0 + 2 Re sum_{m=1}^{M-1} F_m e^{i m phi}.
// Storage: the Cart3D Field with axes (z, r, slot): slot 0 = m = 0 (real), slots 2m - 1, 2m = Re, Im of
// mode m >= 1. The Domain of the fields has N = {Nz, Nr, 2M - 1}, h = {dz, dr, 1}; r and the slots are
// not periodic and not split (ranks along z only), so their guards are zero: the wall beyond r_Nr.
// Staggering (RIP-CC with x -> z, y -> r, z -> phi): Ez, Jz, Ephi, Br at (i, r_j); Er, Bphi, Bz at
// (i, r_{j+1/2}); rho, J_phi at (i+1/2, r_j); J_r at (i+1/2, r_{j+1/2}). The radii r_j come from the
// RadialGrid (uniform: r_j = j dr; stretched: docs/specs/rz_modes.md step 6), r_{j+1/2} the midpoints.
#pragma once
#include <functional>
#include <memory>
#include <vector>
#include "comm.hpp"
#include "grid.hpp"
#include "rz_metal.hpp"
#include "rz_metal_phi.hpp"
#include "rz_dielectric.hpp"

struct RZWork;
class RzTfsf;
class Species;
struct LaserConfig;

// The radial grid (docs/specs/rz_modes.md step 6): node radii r_j, half nodes rh_j = (r_j + r_{j+1}) / 2,
// the node areas A_j = (rh_j^2 - rh_{j-1}^2) / 2 (A_0 = rh_0^2 / 2), the half-node weights B_j = rh_j dn_j,
// the gradient spacings dn_j = r_{j+1} - r_j (node -> half) and dd_j = A_j / r_j (half -> node): the
// discrete curl pair is adjoint in the weights 2 pi A_j dz, 2 pi B_j dz. Arrays j = 0 .. nr + 4 (beyond the
// wall the last spacing continues: the guards of the particle shapes); on the device as one buffer, read by
// the kernels through the plain view. Particles use the index coordinate xi(r) (rz_xi).
struct RadialView {
    const double *r, *rh, *A, *B, *dn, *dd, *bt, *bt2;
    int nr, ju;                        // cells; cells of dr next to the axis (xi = r / dr there)
    double dr;
    const double *ca, *cl, *cr;        // the C1 map (rz_xi shape 4): bump, left and right end-slope terms per cell
    const double* me = nullptr;        // block metal, phi_derivative = difference: m_eff per mode (null: m), decision 103
    HD double mf(int m) const { return me ? me[m] : double(m); }   // the phi derivative of mode m is i mf(m)
    const int* rod = nullptr;          // numerics.rz_axis_cut: per mode the last node of its axis rod (null: none)
    HD int jc(int m) const { return rod ? rod[m] : -1; }
};
struct RadialGrid {
    int nr = 0, ju = 0;
    double dr = 0;                     // the spacing next to the axis
    bool uniform = true;
    std::vector<double> r, rh, A, B, dn, dd;
    std::vector<double> bt, bt2;       // the Ruyten coefficients of cell j for shape 1, 2 (rz_xi, decision 53)
    std::vector<double> ca, cl, cr;    // the C1 shape-2 map (rz_c1_coefficients; zeros until computed)
    double c1_s0 = 0, c1_s1 = 0;       // its node slopes dxi/dr * dr at the axis and at node 1 (0: not computed)
    Buf<double> dev;
    RadialView view() const;
    void upload();                     // the arrays into dev (after a change of the coefficients)
    double R() const { return r[size_t(nr)]; }          // the wall
    // block metal (decision 103): the staggered angular difference over dphi = 2 pi / (2M - 1) replaces i m by i m_eff,
    // m_eff = (2 / dphi) sin(m dphi / 2) (on = false: m); every RZ operator reads it through mf()
    std::vector<double> meff;
    Buf<double> dme;
    double mf(int m) const { return meff.empty() ? double(m) : meff[size_t(m)]; }
    void set_meff(int modes, bool on);
    // numerics.rz_axis_cut = kappa > 0: mode m >= 2 is held at zero on the nodes r_j < m_eff dr / kappa and the half nodes
    // inside (a perfectly conducting rod for that mode, the polar filter of global models): m_eff / r <= kappa / dr where
    // the mode lives, so the RIP-CC limit no longer falls with the number of modes. rodj[m] = the last rod node (-1: none).
    std::vector<int> rodj;
    Buf<int> drod;
    int jc(int m) const { return rodj.empty() ? -1 : rodj[size_t(m)]; }
    void set_rod(int modes, double kappa);
};
// nr cells: dr for r < r_uniform (at least 4 cells), then dn_j = min(dr stretch^k, dr_max), k = 1, 2, ..
// (stretch = 1 or r_uniform >= nr dr: uniform).
// The index coordinate of radius r: xi = j + (r - r_j) / dn_j for r_j <= r < r_{j+1} (r / dr next to the
// axis; beyond the arrays the last spacing). Node j at xi = j, half node j at j + 1/2.
// shape (the deposition, decision 53): 1, 2: the density-conserving maps of src/rz_ruyten.cpp, in which the
// linear (quadratic) spline deposits a uniform density onto every node with its volume 2 pi A_j dz (the plain
// splines put 4/3, 13/8 of it on the axis node): cell j: xi = j + u + bt_j u (1 - u), u = (r - r_j) / dn_j;
// shape 2 in the axis cell xi = u^p, p = bt2_0. 0: the plain coordinate (the gather).
// shape 4: the C1 shape-2 map of the implicit mover (decision 82, rz_c1_coefficients): cell j,
//   xi = j + u + cl_j (u^3 - 2u^2 + u) + cr_j (u^3 - u^2) + ca_j u^2 (1 - u)^2,
// a cubic Hermite with the node slopes dxi/dr = s_0 / dr at the axis, s_1 2 / (dn_0 + dn_1) at node 1 and 2 / (dn_{j-1} + dn_j) beyond
// (cl_j = s_j dn_j - 1, cr_j = s_{j+1} dn_j - 1), plus a bump that fixes the node shares: dxi/dr is continuous (the
// energy-consistent radial force of the implicit mover is proportional to it; with the slope jumps of shape 2 at the
// cell edges its nonlinear solve stalls, research/o6_mover/README.md).
HD inline int rz_cell(const RadialView& g, double r, double& u) {
    const int last = g.nr + 4;
    if (r < g.ju * g.dr) { const double x = r / g.dr; const int j = int(floor(x)); u = x - j; return j; }
    if (r >= g.r[last]) { u = (r - g.r[last]) / g.dn[last]; return -1; }      // beyond the arrays: linear
    int lo = g.ju < last ? g.ju : last - 1, hi = last;                         // r[lo] <= r < r[hi]
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (g.r[mid] <= r) lo = mid; else hi = mid;
    }
    u = (r - g.r[lo]) / g.dn[lo];
    return lo;
}
HD inline double rz_xi(const RadialView& g, double r, int shape = 0) {
    double u;
    const int j = rz_cell(g, r, u);
    if (j < 0) return g.nr + 4 + u;
    if (shape == 1) return j + u + g.bt[j] * u * (1 - u);
    if (shape == 2) return j == 0 ? pow(u, g.bt2[0]) : j + u + g.bt2[j] * u * (1 - u);
    if (shape == 4) return j + u + g.cl[j] * u * (1 - u) * (1 - u) - g.cr[j] * u * u * (1 - u) + g.ca[j] * u * u * (1 - u) * (1 - u);
    return j + u;
}
// dxi/dr of the C1 map (shape 4)
HD inline double rz_dxi_c1(const RadialView& g, double r) {
    double u;
    const int j = rz_cell(g, r, u);
    if (j < 0) return 1 / g.dn[g.nr + 4];
    return (1 + g.cl[j] * (1 - u) * (1 - 3 * u) - g.cr[j] * u * (2 - 3 * u) + g.ca[j] * 2 * u * (1 - u) * (1 - 2 * u)) / g.dn[j];
}
RadialGrid rz_radial_grid(int nr, double dr, double r_uniform = 1e300, double stretch = 1, double dr_max = 1e300);
// bt, bt2 of the grid (src/rz_ruyten.cpp)
void rz_ruyten_coefficients(RadialGrid& g);
// ca, cl, cr of the C1 map with the node slopes s0 at the axis and s1 at node 1 (in cells per cell; the default: dxi/dr from
// 0.20 to 3.5 on a uniform grid, research/o6_mover/c1scan.py), uploaded;
// throws if the map does not converge or is not monotone (src/rz_ruyten.cpp)
void rz_c1_coefficients(RadialGrid& g, double s0 = 3.5, double s1 = 0.4);

class FieldsRZ {
public:
    FieldsRZ(const Domain& d, int modes);                     // uniform radial grid of d.h[1]
    FieldsRZ(const Domain& d, int modes, RadialGrid grid);
    const Domain& dom() const { return d_; }
    const RadialGrid& grid() const { return g_; }
    int modes() const { return M_; }
    Field ez, er, ep, bz, br, bp;      // E_z, E_r, E_phi, B_z, B_r, B_phi
    Field jz, jr, jp;                  // currents at n + 1/2
    void zero_currents();
    // One step n -> n+1 at t = t_n: explicit RIP-CC (predictor) or iRIP (box closure) per d.solver. The
    // inflow hooks (laser at an open left z edge) run first and set the left guard layer i = -1, read by
    // the transport of both closures (iRIP: the same guard in both passes).
    void step(Comm& comm, double t = 0);
    std::vector<std::function<void(FieldsRZ&, double)>> inflow;
    std::shared_ptr<MetalRZ> metal;    // block rings: embedded conductors (rz_metal.hpp; RIP-CC only; null: none)
    std::shared_ptr<MetalPhi> metal_phi;   // block metal: conductors depending on phi (rz_metal_phi.hpp; RIP-CC only)
    // sets metal_phi and the angular derivative of the solver (difference: m_eff on the grid, decision 103)
    void set_metal_phi(std::shared_ptr<MetalPhi> mp);
    std::shared_ptr<DielectricRZ> diel;   // blocks dielectric: eps(r, z) (rz_dielectric.hpp, decision 100; RIP-CC only; null: none)
    RzTfsf* tfsf = nullptr;            // O1: the TFSF plane (rz_tfsf.cpp), its corrections in pass()
    bool gather_matched = false;       // numerics.rz_gather = matched: the gather in the deposit's radial map
    bool gather_energy = false;        // numerics.rz_gather = energy: matched, E_r energy-consistent (decision 80)
    bool map_all = false;              // numerics.rz_axis_map = all: the density-conserving radial map for every mode (before decision 106); default m = 0 only
    bool axis_regular = true;          // numerics.rz_axis_current = regular: only the regular m = 1 transverse current in the first cell (rz_axis_current.cpp)
    bool deposit_chord = false;        // numerics.rz_deposit = chord: the explicit deposit with the axis-smooth splits (rz_particles.cpp)
    double smooth_a = 0;               // numerics.rz_transverse_shape = gaussian: its width a (0: none; rz_smooth.cpp)
    std::shared_ptr<struct RzSmoothWork> smooth_;
    Field sez_, ser_, sep_;            // S E for the gather (rz_smooth_e)
    bool se_alloc_ = false;
    bool o5 = false;                   // numerics.closure = o5: iRIP's dispersion-corrected closure per radial mode (decision 74)
    class Medium* medium = nullptr;    // O7: linear media (medium.hpp, decision 84); null: none
    bool c1_map = false;               // numerics.mover = implicit: the deposit and rho in the C1 radial map (rz_xi shape 4, decision 82)
    void use_c1_map(double s0 = 3.5, double s1 = 0.4);   // computes the C1 map on the grid and sets c1_map
    // grid.boundary_z = absorbing: damping layers of zabs_lo, zabs_hi cells at the z ends, applied at the end of step()
    int zabs_lo = 0, zabs_hi = 0;
    double zabs_kappa = 0;
    // Diagnostics (owned cells, local; reduce over ranks in the caller)
    double energy() const;            // (1/2) int (E^2 + B^2) dV, modes: |F_0|^2 + 2 sum |F_m|^2
    double energy(int i0, int i1) const;  // the same over the owned z cells with the local index in [i0, i1]
    double energy_mode(int m) const;  // the part of mode m (the sum over m is energy())
    // Gauss residual per mode at the rho nodes (i+1/2, r_j) without rho: d_z Ez + mu_z [(1/r) d_r (r Er)
    // + (i m / r) Ephi]; into out (same layout as the fields). div B per mode at (i+1/2, r_{j+1/2}).
    void div_e(Field& out) const;
    void div_b(Field& out) const;
    // One pass of the step (public: it holds the kernels, CUDA): with predictor, Ez*, Bz* from the fields,
    // else the given exs_, bxs_ (guards valid); transport into the scratch (exchanged), the corrector
    // into oEz, oBz (may be ez, bz).
    void pass(Comm& comm, bool predictor, Field& oEz, Field& oBz);
    Field exs_, bxs_, ner_, nep_, nbr_, nbp_;
    Buf<double> filter_alpha_;         // rz_filter_*: a_m per mode (set on first use)
    std::shared_ptr<struct RzProjWork> fproj_;   // the projection filter's work arrays (rz_filter.cpp)
private:
    const Domain& d_;
    int M_;
    RadialGrid g_;
    std::shared_ptr<RZWork> irip_;
    friend void rz_irip_step(FieldsRZ& f, Comm& comm, double t);
    friend void rz_irip_trial(FieldsRZ& f, Comm& comm, double t, Field* et, Field* bb);
    friend void rz_o5_closure(FieldsRZ& f, Comm& comm, Field& x, int kind, bool cz);
};
void rz_irip_step(FieldsRZ& f, Comm& comm, double t = 0);
// numerics.rz_axis_current = regular (src/rz_axis_current.cpp, decision 112): the r^2 part J_r,0 + i J_phi,1 of mode 1 in the first cell
// removed by a divergence-free shift of J_r,0 and J_phi,1 (rho and the Gauss law unchanged), every z column including the guards
void rz_axis_regular_current(FieldsRZ& f);
// its adjoint on the gather copies of E (E_r,0, E_phi,1 of mode 1): the work of the deposit and of the gather agree (decision 112)
void rz_axis_regular_gather_e(FieldsRZ& f, Field& er, Field& ep);
// the gather copies sez_, ser_, sep_ = E with that adjoint (without the transverse shape; the push calls it)
void rz_axis_gather_copies(FieldsRZ& f);
// J_z of the modes m >= 1 on the axis node set to zero (J_z ~ r^m), every column including the guards (src/rz_axis_current.cpp)
void rz_axis_zero_jz(FieldsRZ& f);
// O6 (implicit mover, decision 82): the iRIP box step with the currents in f.jz, f.jr, f.jp, without changing the fields:
// the work field Et (et[0] at the J_z points (i, r_j), et[1] at J_r (i+1/2, r_{j+1/2}), et[2] at J_phi (i+1/2, r_j); per mode
// as the fields) with W^{n+1} - W^n = -dt sum_m w_m sum dV J_m . Et_m exactly (w_0 = 1, w_m = 2, the real dot product of the
// (Re, Im) slots), and bb = (B^n + B^{n+1}) / 2 (bb[0] B_z, bb[1] B_r, bb[2] B_phi); guards exchanged. The inflow hooks run
// (they set the guard layer: idempotent). closure = box only, no TFSF plane.
void rz_irip_trial(FieldsRZ& f, Comm& comm, double t, Field* et, Field* bb);
// The implicit mover's particle pass in RZ (src/rz_particles.cpp, decision 82): mode 0 updates u1 (3 n per species: u1x, u1y,
// u1z blocks) with the work of et along the Crank-Nicolson path (nseg equal segments) and bb at the midpoint, then deposits the
// path into jd (jd[0] J_z, jd[1] J_r, jd[2] J_phi; add_guards in the caller); mode 1 commits x, u <- the path end, u1.
void rz_implicit_pass(Species& sp, FieldsRZ& f, Field* et, Field* bb, Field* jd, double* u1, int nseg, int mode);
// sum q^2 w / (m gamma) of the species per node volume (m = 0 slot, C1 map), added into W (add_guards in the caller): the
// lumped cold response of the preconditioner
void rz_wp2(Species& sp, FieldsRZ& f, Field& W);
// O5: x <- Cz C_perp x (cz = false: C_perp x) with f's closure factors of the kind (0 TM: Ez*, 1 TE: Bz*), exchanged; f is an
// iRIP grid with closure = o5 after its first step
void rz_o5_closure(FieldsRZ& f, Comm& comm, Field& x, int kind, bool cz);
// Transverse particle shape (numerics.rz_transverse_shape = gaussian, decision 79, src/rz_smooth.cpp): S = exp(a^2/2 Lap_perp)
// per mode on the deposited currents, on E for the gather (into sez_, ser_, sep_) and on rho of the diagnostics; no-ops when
// f.smooth_a = 0. rz_smooth_apply (tests): kind 0 X <- f(L_TM) X; 1 (X, *Vp) <- S (X, *Vp) as (V_r, V_phi), owned cells only.
void rz_smooth_currents(FieldsRZ& f, Comm& comm);
void rz_smooth_e(FieldsRZ& f, Comm& comm);
void rz_smooth_rho(FieldsRZ& f, Comm& comm, Field& rho);
void rz_smooth_apply(FieldsRZ& f, Field& X, int kind_or_vector, Field* Vp);
// Radial absorbing layer (grid.boundary_r = absorbing, docs/specs/rz_modes.md, decision 49): after the
// field step every component of every mode is multiplied by exp(-kappa s^2), s = (r - r0) / (R - r0) >= 0
// the depth into the outer `cells` radial cells in the index coordinate, at its own position (nodes j or
// half nodes j + 1/2). Not a PML: the damping per step is small and graded, so the reflection from its
// gradient is small, and a wave that crosses the layer twice loses ~exp(-2 kappa a / 3), a = (R - r0) / dt
// steps per crossing (r0 = r_{nr - cells}).
void rz_absorb(FieldsRZ& f, int cells, double kappa);
// numerics.rz_axis_cut: zeros of each mode on its axis rod after a pass (RadialGrid::set_rod; src/rz_fields.cpp)
void rz_axis_cut_apply(FieldsRZ& f, int stage, Field& a, Field& b, Field* c = nullptr, Field* e = nullptr);
// Self-field of a beam moving along z with beta (species.self_field in RZ, docs/specs/rz_metal.md): the discrete form of
// (dzz / gamma^2 + mu_z^2 D_m G_m) phi = -rho per mode (phi at the rho nodes, 0 just beyond the Gauss cells of the open
// z ends and at the wall), Ez = -dz phi / gamma^2, E_perp = -G_m mu_z phi, B_phi = beta E_r, B_r = -beta E_phi: the RZ
// Gauss law holds to round-off. Added to the fields present (owned cells, guards exchanged). Metal is ignored (start
// the beam before the rings). Host code at the setup; every rank solves the global problem (src/rz_beam_field.cpp).
void rz_add_beam_field(FieldsRZ& f, const Field& rho, double beta, Comm& comm);
// z absorbing layers (grid.boundary_z = absorbing, docs/specs/x_absorber.md "RZ"): every component times exp(-kappa s^2),
// s the depth into the zabs_lo cells at the left or the zabs_hi cells at the right end (as the Cart3D x layers)
void rz_absorb_z(FieldsRZ& f);

// O1 in RZ (src/rz_tfsf.cpp, docs/specs/tfsf.md "RZ"): the TFSF plane between the z cells i_s - 1 (scattered) and i_s
// (total), normal incidence. The incident field is an auxiliary vacuum FieldsRZ on the same domain and radial grid with
// the laser through its open left z edge, the radial absorber and the right z layer of the main grid (not the left one).
class RzTfsf {
public:
    // lasers with tfsf_x >= 0 (one plane z_s = tfsf_x); zabs_hi, zkappa: the right z layer; rabs, rkappa: the radial absorber
    RzTfsf(const Domain& d, Comm& comm, int modes, RadialGrid g, const std::vector<LaserConfig>& lasers, int zabs_hi,
           double zkappa, int rabs, double rkappa);
    int plane() const { return is_; }            // global index i_s
    void begin(Comm& comm, double t);            // keeps the incident field of t_n, advances the incident grid to t_n+1
    // adds the incident transport components across the plane to f's transported fields (ner_ .. nbp_): kind 0 RIP-CC
    // (the incident predictor of t_n), 2 iRIP (the box mean of t_n and t_n+1, both passes)
    void correct(Comm& comm, FieldsRZ& f, int kind);
    // O5 (decision 78): the incident grid runs the closure; its means get C_perp (pe_, pb_) and Cz (xs_, bs_)
    void set_o5();
    // O5: the Cz stencil of the main grid's closed Ez*, Bz* across the plane gets the incident value of the far cell
    void o5_fix(Comm& comm, Field& exs, Field& bxs);
    FieldsRZ& incident() { return *inc_; }
private:
    const Domain& d_;
    std::unique_ptr<FieldsRZ> inc_;
    Field er0_, ep0_, br0_, bp0_, ez0_, bz0_, xs_, bs_, pe_, pb_;
    bool o5_ = false;
    int is_ = 0, rabs_ = 0;
    double rkappa_ = 0;
};
// Moving window with iRIP: Ez, Bz of the entering last cell from the Gauss law and div B of the cell
// before it (gauss0: the initial residual, shifted with the fields). On the rank at the right edge.
void rz_window_entry(FieldsRZ& f, const Field& gauss0, const Field* rhom = nullptr);   // rhom: the media charge per mode (O7)
struct LaserConfig;
// Laser through the open left z edge (src/rz_laser.cpp): each enabled LaserConfig, evaluated in the
// transverse plane (x, y) = (r cos phi, r sin phi) of the particles (pol 'y': E_x, 'z': E_y), is projected
// on the modes m < M by an azimuthal quadrature and set as the inbound T in the guard layer i = -1 at
// t + dt: E_r + B_phi and E_phi - B_r, the wave moving along +z (B_phi = E_r, B_r = -E_phi). Only on the
// rank at the global left edge.
// files: LASY lasers (laser_file.hpp RzFileLaser, lab runs), summed with the built-in ones.
class RzFileLaser;
void rz_add_laser_inflow(FieldsRZ& f, const std::vector<LaserConfig>& lasers, const Comm& comm,
                         const std::vector<std::shared_ptr<RzFileLaser>>& files = {});
// Largest eigenvalue of -L_m over the modes m < modes and both kinds (TM, TE) on the radial grid: the
// explicit RIP-CC step is stable for dz^2 (lambda_max + wp^2) / 4 < 1 (as Cart3D's dx^2 (4/dy^2 + 4/dz^2) / 4).
double rz_lambda_max(const RadialGrid& g, int modes);
// The largest eigenvalue of -L_m of the TM kind (the scalar operator on Ez nodes) for mode m.
double rz_lambda_max_tm(const RadialGrid& g, int m);
// Current filter (numerics.filter_passes, src/rz_filter.cpp, decision 54): per pass the binomial (1, 2, 1)/4 along
// z of every component and, per mode m, the radial smoothing T = 1 + a_m D_m G_m of rho and J_z, S = 1 + a_m G_m D_m
// of (J_r, J_phi) (D_m the divergence of the grid, G_m = -D_m^+ its adjoint gradient, a_m = 1 / lambda_max,m):
// D_m S = T D_m, so the filtered current and charge keep the continuity per mode exactly; a uniform density is
// left unchanged. The guards along z must be valid (exchanged).
// local = false (numerics.filter = projection, decision 70): per pass r = (T - 1) D J, L psi = r, J <- J + G psi with
// G = (dz, mu_z^2 G_m) the gradient of the RZ Gauss law and L = D G = dzz + mu_z^2 D_m G_m (per mode the TM eigenbasis of
// D_m G_m, rz_tm_basis, and a tridiagonal line along z per radial eigenmode): D J' = T D J as with the local filter,
// but the change only moves charge and radiates nothing (test R24). local = true: the filter above (decision 54).
// binomial = true (numerics.filter = binomial, decision 91): the z step as S_z on J_z (w = d_z J_z only) and the binomial
// over the Gauss cells on J_r, J_phi; the r step as local. Every component then gets T_z, J_z also T_r, as rho.
void rz_filter_currents(FieldsRZ& f, Comm& comm, int passes, bool local = false, bool binomial = false);
// The eigenbasis of the TM radial operator D_m G_m of mode m (iRIP's kind 0): V, Vi (nr x nr, row-major), lam = the
// eigenvalues of -D_m G_m (> 0; 0 for the dropped axis node of m >= 1).
void rz_tm_basis(const RadialGrid& g, int m, std::vector<double>& V, std::vector<double>& Vi, std::vector<double>& lam);
// The eigenbasis of kind (0: TM on Ez, 1: TE on Bz) of mode m and the O5 spacing h_a^2 per radial mode (src/rz_irip.cpp)
void rz_basis(const RadialGrid& g, int m, int kind, std::vector<double>& V, std::vector<double>& Vi, std::vector<double>& lam,
              std::vector<double>& h2);
void rz_filter_rho(FieldsRZ& f, Comm& comm, Field& rho, int passes);

class Species;
enum class Pusher;
// Particles (src/rz_particles.cpp, Cartesian (z, x, y) in X, Y, Z): gather from the modes, push, move and
// (deposit) the charge-conserving current per mode into f.jz, f.jr, f.jp (add_guards in the caller);
// order: the shape, 1 or 2 (decision 50).
void rz_push_deposit(Species& sp, FieldsRZ& f, Pusher pusher, bool deposit = true, int order = 1);
// Charge density per mode at the rho nodes (i+1/2, r_j), added into rho (add_guards in the caller).
void rz_rho(Species& sp, FieldsRZ& f, Field& rho, int order = 1);
// only the particles with z0 <= z < z1 (the window's entering columns, decision 91)
void rz_rho_range(Species& sp, FieldsRZ& f, Field& rho, int order, double z0, double z1);
// E, B (Cartesian, axes (z, x, y)) at one point (tests, diagnostics)
void rz_gather_point(FieldsRZ& f, double z, double x, double y, double E[3], double B[3], int order = 1);
// the wake probe line of F/q (decision 105, src/wake.cpp): out (device, 3 nz) at (x, y) and the local z nodes 0 .. nz - 1
void rz_wake_line(FieldsRZ& f, double x, double y, int nz, double* out);
// Field ionization (ionization.hpp): E^n (Cartesian (z, x, y)) at every particle of sp into E (3 per particle); the current of
// the moves J0 -> J1 (global positions (z, x, y), 3 per move, each shorter than half a cell), charge QW each.
void rz_field_at(Species& sp, FieldsRZ& f, double* E, int order, double* B = nullptr);   // E (and B if given) at the particles
// gather and push only (spin / QED: the move follows with Pusher::None, decision 15); EB: E, B, u^{n-1/2} per particle
void rz_push_store(Species& sp, FieldsRZ& f, Pusher pusher, int order, double* EB);
void rz_jumps(FieldsRZ& f,const double* J0, const double* J1, const double* QW, long n, int order);

// Domain of the RZ fields for Nz x Nr cells and M modes (decomposed along z);
// ranks_z: ranks along z (0: all ranks).
Domain rz_domain(Comm& comm, int nz, int nr, int modes, double dz, double dr, bool periodic_z, Solver s, int ranks_z = 0);
