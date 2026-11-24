# Overdense targets, high harmonics (HHG / CHF) and TNSA: physics constraints (track O)

Status: planning spec (Alexander + cloud session, 2026-10-28). Roadmap track O. Numbers from the per-mode
analysis of the box closure (`research/irip_rz/`, `docs/specs/implicit_rip.md`) and the iRIP paper (in preparation).

## Idea
The dispersionless axis x is chosen along the direction of the most demanding radiation: for harmonics
from an overdense surface (relativistic oscillating mirror, relativistic spikes; Baeva, Gordienko, Pukhov,
PRE 74, 046404 (2006)) the specular direction. Harmonics along x are transported exactly (no dispersion for
any order resolved by the grid, k_x dx < pi). Coherent harmonic focusing (CHF; Gordienko et al., PRL 94,
103903 (2005)) concentrates them in a spot ~ lambda0/(2 N sin alpha); the box iRIP allows a radial or
transverse grid refined there, independent of dt, and stretched outward.

## Geometries
- **Normal incidence, RZ (or 3D):** incident laser along +x, reflected harmonics along -x, both exact. The
  reflected field leaves through the left boundary as the outbound transport components: the boundary absorbs
  it exactly at normal incidence and records it (O2: time series of T- at x = 0 -> harmonic spectrum and
  transverse profile per harmonic). Linear polarisation: fields in odd m, surface deformation in even m;
  expect m = 0..5 (check convergence in the number of modes). Circular polarisation at normal incidence gives
  (almost) no harmonics from a flat surface.
- **Oblique incidence, 3D:** the target normal is tilted so that the specular direction is x. The incident
  laser comes in at the angle 2 theta from x and has the small error -(k0 dx)^2 sin^2(2 theta)/6. It cannot
  be injected through the x = 0 plane exactly: TFSF source (O1). Waves leaving the x boundaries at an angle
  need CPML along x (O3); the harmonics along x do not.

## Constraints
1. **Plasma frequency:** omega_p dt < 2 with c dt = dx: dx < lambda0 / (pi sqrt(n/n_c))
   (lambda0/31 at 100 n_c, lambda0/63 at 400 n_c). Resolving the skin layer and the surface oscillation
   usually needs this anyway. Check at the peak density at start (O4).
2. **RZ near the axis:** explicit rip_cc needs h_r >~ m c dt (largest eigenvalue of -L_m ~ 4(m^2+1)/h_r^2):
   h_r >= 1.10, 1.26, 2.13, 3.08, 4.06, 6.04, 8.03 c dt for m = 0, 1, 2, 3, 4, 6, 8. With several modes RZ needs irip.
3. **Focused harmonics:** box-iRIP phase error of harmonic N at angle alpha to x:
   d omega / omega ~ -(N k0 dx)^2 sin^2(alpha) / 6, set by dx = c dt, not by h_perp. Examples:

   | dx | N | alpha = 10 deg | 20 deg | 30 deg |
   |---|---|---|---|---|
   | lambda0/128 | 10 | -1.2e-3 | -4.3e-3 | -8.5e-3 |
   | lambda0/128 | 20 | -4.2e-3 | -1.6e-2 | -3.1e-2 |
   | lambda0/256 | 20 | -1.2e-3 | -4.3e-3 | -8.5e-3 |
   | lambda0/256 | 40 | -4.2e-3 | -1.6e-2 | -3.1e-2 |

   The phase over a focusing distance L is N k0 L |d omega/omega| (N = 20, 20 deg, lambda0/256, L = 5 lambda0:
   2.7 rad). rip_cc at h_perp = dx (2D) is about 4 times more accurate. Choose dx from this table for the
   highest harmonic that must focus correctly.
4. **Grid heating at solid density:** Debye length unresolved; use shapes 2-3, the binomial filter (keeps the
   Gauss law), enough particles per cell; later the implicit energy-conserving mover (O6).

## O5: dispersion-corrected closure (option of irip)
Theorem 1 of the paper (Gauss law for any closure shared by both transport directions) allows other closures
without losing charge conservation. Paraxial correction:
  Ex_bar = mu_x mu_t (1 - dx^2/3 delta_x^2) Ex,   Bx_bar likewise,
which multiplies the transverse eigenvalue by 1 + (4/3) sin^2(kx dx/2) and removes the leading error:
3-10 times smaller for N k0 dx <~ 1 up to 45 deg (N = 10, lambda0/128, 20 deg: -4.3e-3 -> -5.3e-4). Price:
no exact energy conservation; per-mode transient growth <= 1.53 (tested kappa <= 100, all kx); the system
along x per transverse mode becomes pentadiagonal. A closure that is both dispersion-corrected and energy
conserving is an open research item (cloud session).

### O5 prototype, 2026-11-04 (research/o5/)
- **Where the error is (CHF).** `research/o5/dispersion.py`: over the 3 um from the paraboloid to the focus, order 7 at
  angle 0.4 has 0.30 rad (RIP-CC) and 0.69 rad (iRIP) at dz = lambda0/128, dr = lambda0/40; the paraxial factor above brings
  iRIP to 0.26 rad, but the rest is mostly the transverse grid (the floor with exact time stepping: 0.21 rad). The
  angle-dependent error is a spherical aberration of the harmonic focus (bench/chf/README.md: iRIP 17 % below RIP-CC).
- **Full closure.** Expanding tan^2 a = tan^2 b + q g (a = w D/2, b = kx D/2, q = D^2 Lam/4) against a^2 = b^2 + k_perp^2 D^2/4
  gives g = 1 + (4/3) sin^2 b + (2/3) q + (leading transverse-grid error, h^2 Lam / 12 per transverse axis). So
    Ex_bar = mu_x mu_t Cx C_perp Ex,  Cx = 1 - (hx^2/3) delta_x^2,  C_perp = 1 - (D^2/6 + h_perp^2/12) L_perp,
  Bx_bar likewise (L_perp the transverse Laplacian on the node of Ex, of Bx). Order 7 at 0.4: 0.30 / 0.69 -> 0.027 rad
  (dz = lambda0/128, dr = lambda0/40), 0.002 rad at lambda0/256, lambda0/80 (RIP-CC 0.075).
- **Checked on the scheme itself.** `research/o5/closure2d.py`: the numpy copy of the iRIP step (research/irip_3d/
  identity.py) in 2D with the modified closure, per Fourier mode the 6 x 6 amplification matrix. Its phase velocity equals
  the dispersion relation with g to all printed digits for plain iRIP and the full closure (e.g. order 7 at 0.4,
  dy = 3.2 dx: -3.83e-3 iRIP, -1.45e-3 paraxial, -1.10e-4 full; order 15 at 0.4: -1.62e-2, -7.4e-3, -2.1e-3).
- **Stability.** `research/o5/stab2d.py` (33 x 33 modes over the Brillouin zone, 400 steps): spectral radius 1 to 1e-12 for
  every mode and dy / dx = 3.2, 1, 0.5, 0.2 (neutral, as plain iRIP). Not energy conserving: the largest transient growth of
  ||G^n|| (plain E^2 + B^2) is 1.20 / 1.41 / 1.49 / 1.52 for the paraxial closure (the 1.53 above) and 1.40 / 2.05 / 3.02 /
  6.48 for the full one, always at the grid scale (ky dy = pi, kx dx ~ 2.4 .. 3): the D^2/6 L_perp term grows like
  (D / dy)^2 there. For dy >~ dx (CHF: dr = 3.2 dz) it is 1.4 - 2; for dy << dx (many modes near the axis) the transverse
  term should be limited, e.g. applied only to resolved k_perp (spectrally in iRIP's eigenbasis) or capped.
- **Open before C++:** (1) 3D: a common factor on L_perp = L_y + L_z cannot cancel the per-axis error h^2 Lam_a^2/12 exactly
  (cross term L_y L_z at 45 deg azimuth: the h^2 part overcorrects by up to 2); the D^2/6 part is isotropic. Either the
  per-mode factor in the eigenbasis (iRIP already transforms; the explicit pass then needs it too) or only D^2/6 in 3D.
  (2) RZ: the radial operator L_m per mode; its discretisation error is not the sin^2 form near the axis, to be measured
  with the guided-mode test. (3) The Gauss law: Theorem 1 covers any closure shared by both movers; to be confirmed with
  particles once implemented (R5/R23-type tests). (4) The line along x becomes pentadiagonal (Cx).

### O5 in the code, 2D (2026-11-04, decision 72, `numerics.closure = o5`, src/irip.cpp)
- The closure Cx C_perp is applied to Ex*, Bx* in both passes (two local kernels and two exchanges per field and pass).
- The pentadiagonal system per mode, X + kappa (2 + S)(5 - S)/12 X = R (S x_i = x_{i-1} + x_{i+1}, kappa = D^2/4 Lam g_perp),
  factors exactly into -(kappa/12)(S - r1)(S - r2), r = (3 +- sqrt(49 + 48/kappa)) / 2: two tridiagonal line solves with the
  existing solver (cyclic for periodic x; zeros outside for open x, where both factors are polynomials in the same truncated
  S as the step's operator; the partition method for ranks along x). |r2| > 2, so both factors are diagonally dominant.
- Tests (tests/test_cart2d.cpp C9, C2; tests/mpi.sh): the vacuum dispersion of (kx, ky) modes equals the corrected relation
  to 9e-14 (dy = dx/4 and 3.2 dx, both polarisations; on these modes the error against c |k| is 9 times smaller than with the
  box closure); the corrector's Ex, Bx equal the solve's X, Y to 1.4e-13 (periodic and open x); the Gauss law and div B with
  plasma at round-off (1e-14, shapes 1-3); random fields over 400 steps stay bounded but not conserved: the energy reaches
  3.4 times its start and ends at 2.7 (dy = dx/4 .. 3.2 dx), the grid-scale transient growth of the analysis; 8 ranks along x
  = 1 rank to 1e-12 (the 2D foil with open x and the laser at 30 deg through the left edge).
- The limiter of the transverse term for dy << dx (decision 77), 3D, RZ, CPML, the TFSF plane and the side tube: below.

### O5 in 3D (2026-11-04, decision 73)
- A local transverse factor cannot cancel the per-axis grid error h_a^2 lambda_a^2/12 (with h_y = h_z it is exact along y or z
  and overcorrects by 2 at 45 deg azimuth), and the closure must be one operator shared by Ex and Bx (Gauss law). Phase error
  over the CHF path (order 7 at 0.4 rad, dx = lambda0/128, dy = dz = lambda0/40), azimuth 0 / 22 / 45 deg: box closure
  +0.69 / +0.64 / +0.59 rad; the 2D factor as a stencil +0.020 / -0.033 / -0.086 (as anisotropic as the box closure);
  per mode +0.020 / +0.018 / +0.015 (order 15: azimuthal spread 1.1 rad -> 0.18 rad).
- So in 3D the factor is applied per mode in the y, z Fourier basis of iRIP's solve:
    g_perp = 1 + D^2 lambda/6 + (h_y^2 lambda_y^2 + h_z^2 lambda_z^2) / (12 lambda),  lambda = lambda_y + lambda_z,
  the same g_perp in the line systems (O5Coef); Cx stays the stencil. In 2D g_perp is the stencil of decision 72 (unchanged).
  Cost: per closure two forward and two backward transverse transforms, four closures per step (about three times the transform
  work of a box-closure step; particles unchanged).
- Stability (`research/o5/stab3d.py`, the 6 x 6 amplification matrix per mode from src/irip.cpp's step, 13^3 modes over the
  Brillouin zone, 400 steps): spectral radius 1 to 1e-12 for every mode (neutral); transient growth of ||G^n|| 1.53 / 2.44 /
  3.93 / 7.24 for h_perp = 3.2 / 1 / 0.5 / 0.25 dx (box closure: 1), at the grid-scale corner ky dy = kz dz = pi. The matrix
  reproduces the relation to 1e-12.

### O5 in RZ (2026-11-04, decision 74)
- The radial operator L_m of iRIP's solve (src/rz_irip.cpp: TM on Ez at the nodes, TE on Bz at the half nodes, per azimuthal
  mode m) has a dense real eigenbasis; the closure Ez*, Bz* <- Cz C_perp Ez*, Bz* applies C_perp per radial eigenmode a:
    g_a = 1 + D^2 lambda_a/6 + h_a^2 lambda_a/12,   h_a^2 = sum_j w_j v_ja^2 h_j^2 / sum_j w_j v_ja^2,
  w_j, h_j = A_j, dd_j at the nodes (TM) and B_j, dn_j at the half nodes (TE): the energy-weighted squared spacing of the mode
  (Alexander 2026-11-04). Cz stays the stencil along z. The line systems along z use the same g_a (two tridiagonal factors per
  radial mode; serial cyclic / open, and the partition method across ranks along z).
- Why h_a (`research/o5/radial_rz.py`, the discrete L_m against the Bessel roots of a tube, m = 0, 1, 2, TM and TE): on a
  uniform grid k^2 - lambda = dr^2 lambda^2/12 within 0.9-1.13 (lowest modes of m = 1, 2: 0.4-0.7, an axis term), so
  h_a = dr cuts the k_perp error from 1-2 % to 0.03-0.14 % at k dr = 0.5-0.75. On a stretched grid (24 dr, then 4 % per cell)
  dr removes almost nothing (2.2 % -> 2.0 %); h_a fits within 1.0-1.2 and cuts it from 0.5-2.7 % to 0.03-0.16 % for
  k dr <= 0.26 (k dr = 0.5, where the coarse outer cells barely carry the wave: 10 % -> 3-4 %).
- Tests (tests/test_rz.cpp R-O5): a discrete radial eigenmode times cos(kz z) (uniform and stretched grid, TM m = 0, TE m = 1)
  follows tan^2(w dt/2) = tan^2(kz dz/2) + D^2 lambda_a/4 (1 + 4/3 sin^2(kz dz/2)) g_a to 1.5e-12; the Bessel modes at
  dz = 2 dr against the continuum c^2 (kz^2 + kp^2): TM01 1.2e-2 / 3.5e-3 (box, nr = 16 / 32) -> 5.6e-4 / 1.6e-4 (o5), TE11
  7.5e-3 / 2.1e-3 -> 1.8e-4 / 8.0e-5 (the remaining error is second order in dr: the axis term); random fields in modes 0..2
  at dz/dr = 4 (uniform and stretched) bounded (energy up to 3.3 times its start, 300 steps), div E and div B per mode
  unchanged to 3e-14; hot plasma (shape 1, 2) Gauss law and div B per mode at 1e-14. tests/mpi.sh: stretched grid with
  periodic z and the moving window with the laser through open z, 8 ranks along z = 1 rank.
- Per radial mode the step is the Cartesian 2D step of one transverse mode with lambda_a and g_a, so the stability of
  `research/o5/stab2d.py` carries over (neutral, transient growth at the grid scale).
- Cost: two dense radial transforms per closure (nr^2 per column), four closures per step: about three times the radial
  transform work of a box-closure step. With the RZ TFSF plane: decision 78, section "O5 with the RZ TFSF plane".
- Tests (tests/test_irip.cpp I12, I4; tests/mpi.sh): oblique (kx, ky, kz) modes, both polarisations, also at 45 deg azimuth,
  follow the corrected relation per mode to 2.6e-12 (h_perp = dx/4, 3.2 dx); corrector = solve to 1e-12 (periodic, open x);
  random fields bounded over 400 steps (energy up to 6.3 times its start at h_perp = dx/4, the grid-scale transient growth);
  the Gauss law and div B with plasma at 1e-14 (shapes 1, 2); 8 ranks along x = 1 rank (streaming plasma, open x).


## Tests / benchmarks (Gate O)
1. Reflection from a perfect mirror (step density >> n_c), normal incidence: reflected energy and phase.
2. 1D-like ROM harmonics (wide flat target, periodic y, z or RZ m = 1): spectrum converges in dx, follows the
   relativistic-spikes power law up to the resolved order; harmonic phases converge.
3. Oblique-incidence HHG in 3D vs Smilei or EPOCH with the same setup: spectrum and angular distribution.
4. CHF (dented or pre-curved target) in RZ and 3D: focal intensity, spot size and position converge in dx and
   dr; energy balance (absorbed + reflected + transmitted) to 1e-3.
   Status 2026-11-03: the diagnostic `harmonic_focus` (docs/specs/harmonic_focus.md, decision 67) and the code check in
   `bench/chf/` (provisional physics: a0 = 10, 40 n_c paraboloid, F = 3 um): 2D converged for orders 4.5 .. 10.5 at
   lambda0/128 .. /256, not yet for 10.5 .. 20.5; RZ at lambda0/128 (dr = lambda0/40). With the approved parameters
   (a0 = 10, 5 fs, waist 2 um, 40 n_c, F = 3 um): 2D lambda0/512 converged in position, spot and fluence for both bands,
   balance 1.5e-4; RZ lambda0/256 (dr = lambda0/80) not converged (peak +24 %), balance 1.1e-3 to 48 fs, then a late
   excess of 6e-3 (open); 3D lambda0/64 balance 5.2e-3, 3D lambda0/128 prepared for JUPITER (bench/chf/README.md).
5. TNSA (foil, a0 ~ 10-30, n ~ 50-200 n_c, pre-plasma): proton cutoff vs Smilei/EPOCH within 10 %, Gauss law at
   round-off, total energy drift reported.

### O5 with CPML and the TFSF plane (2026-11-05, decision 75)
- CPML: the solve of iRIP uses on a pml axis the eigenbasis of the stretched transverse operator of the kind (decision 41);
  the closure acts per mode of that basis, g_perp(lambda_y, lambda_z) with its eigenvalues, so L C_perp stays diagonal and the
  line systems keep their form (O5Coef with the eigenvalues). In 2D this is the local stencil 1 + (D^2/6 + h_y^2/12) L_y with
  the stretched operator.
- Measured (tests/test_irip.cpp I9-I11): reflection of a packet into a y layer (h_y = dx/2, against a tall periodic box with
  the same closure), normal / 45 deg: box closure 2.6e-6 / 1.1e-6, o5 1.5e-4 / 7.4e-5 at 16 cells; normal incidence at 32 cells
  1.6e-7 (box) and 1.0e-5 (o5). Without the grid term of g_perp still 1.2e-4: the time-step term D^2 lambda/6 acts on the
  instantaneous stretching b of the layer, while the absorption of the CPML lives in its memory psi (frequency dependent),
  which a factor per mode cannot follow. Accepted (Alexander 2026-11-05); with o5 use thicker layers (~32 cells).
- Random fields in y and z layers (dx/h_perp = 4): bounded and not growing; the grid-scale noise that never reaches the layers
  stays, at 2.6 E_0 after 2000 steps (box closure 0.70 E_0): the transient growth of o5, not an instability. Hot plasma
  through the layers: Gauss law and div B at 2e-15.
- TFSF plane (incident grid): the incident grid runs o5 too (Tfsf::set_o5); the incident Ex*, Bx* of the corrections get the
  closure (C_perp in the incident grid's basis, then Cx, in Tfsf::begin), and the Cx stencil of the main grid across the plane
  gets the incident value of the far cell: Ex*(s - 1) += C_perp X_inc(s)/3, Ex*(s) -= C_perp X_inc(s - 1)/3 (Bx* likewise;
  both passes, Tfsf::o5_fix). C_perp never crosses the x plane. Leakage (tests/test_tfsf.cpp T1, 0 and 30 deg, both
  polarisations, plane wave and focused beam, periodic y and CPML in y): left of the plane <= 1.4e-15 of the peak, total -
  incident <= 2.3e-15. The exact plane waves (tfsf_incident = modes) and the side tube: next section.
- tests/mpi.sh: iRIP o5 with CPML in y and z, and the TFSF laser at 30 deg onto a slab with CPML in y and o5 (8 ranks = 1).

### O5 with the exact plane waves: side tube and tfsf_incident = modes (2026-11-05, decision 76)
- The plane waves of the side tube and of the TFSF plane with tfsf_incident = modes are eigenvectors of the one-step matrix
  per Bloch mode (src/side.cpp mode_step). With o5 (Domain::o5) the box means Ex*, Bx* there are multiplied by the closure of
  the mode, c = Cx(kx) g_perp(ky, kz) = (1 + 4/3 sin^2(kx D/2)) g_perp (o5_factor), and the dispersion becomes
    tan^2(w D/2) = tan^2(kx D/2) + D^2/4 (Lam c + wp^2 / cos^2(w D/2))
  (with the background plasma of decision 63; the one-step matrix confirms it, residual < 1e-10, checked per component). The
  incident Ex*, Bx* of the corrections are the closed means c mu V (stored in the component's pr; kind 3 of the kernels).
- Side tube: in 3D (and with CPML) the closure C of the grid is non-local, so it carries the incident field across the faces.
  The closed Ex*, Bx* of the grid get once per step s = [in] c I - C(chi_in I) (I: the incident means of t_n and t_n+1, chi_in
  the tube mask, [in] the tube points; SideTfsf::o5_means), so a field F = chi_in I + S gives c I + C(S) inside and C(S)
  outside: exact whatever C does near the CPML layers (Alexander chose the per-mode closure, decision 76; the leakage turned
  out to be at round-off, not only small). Cost: two closures and two incident evaluations per step.
- TFSF plane with plane waves: the slab of the plane also holds g_perp mu V of Ex, Bx for the Cx stencil across the plane
  (Tfsf::o5_fix).
- Tests: tests/test_side.cpp T1 (75.5 deg, 30 deg elevation with the corners, a pulse), T3 (flying focus in vacuum and in a
  plasma of 0.01 n_c with the cold fluid): outside the tube <= 1.5e-14, total - incident <= 1.5e-14 of the peak, Gauss law and
  div B <= 1e-14; tests/test_tfsf.cpp T3 (45 deg, 30 deg, a round beam): left of the plane <= 1.4e-15. tests/mpi.sh: the side
  tube pulse and the flying focus with o5 on 8 ranks along x.

### O5: the cap of the transverse term for dy << dx (2026-11-05, decision 77)
- The time-step part of the transverse factor, D^2 lam/6 = (2/3) q with q = D^2 lam/4 = (k_perp c dt/2)^2 (lam the eigenvalue
  of -L_perp), grows like (dx/dy)^2 at the grid scale. Uncapped, the transient growth max_n ||G^n|| over the Brillouin zone is
  1.40 / 2.05 / 3.02 / 6.47 / 12.6 in 2D for dy/dx = 3.2 / 1 / 0.5 / 0.2 / 0.1 (research/o5/stab_limit.py) and 1.53 / 2.44 /
  3.93 / 7.24 / 17.6 in 3D for dy = dz = 3.2 .. 0.1 dx (research/o5/stab3d.py); the spectral radius stays 1 (neutral).
- Now g_perp = 1 + (2/3) min(q, 1) + (h_y^2 lam_y^2 + h_z^2 lam_z^2)/(12 lam) (RZ: g_a = 1 + (2/3) min(D^2 lam_a/4, 1) +
  h_a^2 lam_a/12), always on with closure = o5 (Alexander chose the hard cap at q = 1 and no option). Modes with q > 1 have
  k_perp c dt > 2: they are not resolved in time, and the Taylor term no longer approximates tan^2(sqrt q)/q anyway.
- Transient growth with the cap: 2D 1.40 / 2.05 / 2.12 / 2.15 / 2.16, 3D 1.53 / 2.09 / 2.12 / 2.15 / 2.15 for the same dy:
  bounded by about 2.15 for any dy. Alternatives measured: cap at q = pi^2/4 (where tan^2 cannot be matched) 2.6; the smooth
  (2/3) q/(1 + q) 2.15, but it already changes resolved modes (order 30 at 0.6 rad, dy = dx/4: -2.80e-2 instead of -2.62e-2).
- Resolved modes are unchanged: orders 7, 15, 30 at 0.2 - 0.6 rad, dx = lambda0/128, dy = 3.2 / 1 / 0.25 dx, the same phase
  error to all printed digits (they have q <~ 0.2). In 2D the cap is inactive for dy >= dx (q <= (D/dy)^2), in 3D it acts only
  near the grid corner (q <= 2 at dy = dz = dx).
- In 2D the closure is now applied per mode in the y Fourier basis like in 3D (a capped factor is no stencil). The side tube
  and the TFSF plane waves carry the same capped factor (src/side.cpp o5_factor), so their leakage stays at round-off.
- Tests: the dispersion checks (tests/test_cart2d.cpp C9, tests/test_irip.cpp I12, tests/test_rz.cpp R-O5) predict with the
  capped g (q up to 8 on the dy = dx/4 modes); random fields at dy = dx/10 .. 3.2 dx (2D, 3D) and dz/dr = 4, 10 (RZ) stay below
  2.15^2 = 4.62 E_0, the energy bound of the per-mode growth. Measured: the energy peaks at 1.34 E_0 in 2D (3.4 before the cap,
  at dy = dx/4), 1.35 in 3D, 1.31 in RZ, and 1.35 with CPML layers in y and z at dx/h_perp = 4 (decision 75 test); the
  dispersion checks agree with the capped relation to 4e-14 (2D), 4e-13 (3D), 6e-13 (RZ); the TFSF and side-tube leakage stays
  <= 1.6e-14.

### O5 with the RZ TFSF plane (2026-11-06, decision 78)
- Why: the RZ CHF focus at lambda0/128 is limited by the chromatic focal shift of the scheme (bench/chf/README.md, "Why iRIP's
  focus is 17 % below RIP-CC's"); O5 removes it, and the production set-up feeds the laser through the TFSF plane.
- As in Cart3D (decision 75): the incident FieldsRZ runs o5 too (RzTfsf::set_o5). In RzTfsf::begin its box means
  (Ez, Bz)(t_n + t_n+1) / 2 get C_perp per radial eigenmode (pe_, pb_; rz_o5_closure with cz = false, the incident grid's
  own factors = the main grid's) and then Cz (xs_, bs_), used by the corrections in both passes. C_perp acts per z cell and
  never crosses the plane; the 3-point Cz of the main grid does: after its closures RzTfsf::o5_fix adds + PE(s)/3 to the
  scattered cell s - 1 and - PE(s - 1)/3 to the total cell s (PE: the C_perp-closed incident mean), so the scheme is the
  plain o5 scheme for (total right, scattered + incident left).
- Tests: tests/test_rz_tfsf.cpp R1 with irip o5 (focused beam through the plane, vacuum, z layers, radial absorber):
  left of the plane 3.1e-15 of the peak, total - incident 4.1e-15 (box closure: 2.2e-15, 3.4e-15; without the stencil fix at the plane 1.0e-3); energy balance against the injected energy -9.8e-4 (-5.7e-4 without o5: O5 does not conserve
  the discrete energy). A single harmonic (lambda0 = 0.8/7 um, NA 0.5, the CHF grid dz = 0.00625 um, dr = 0.02 um) fed through
  the plane at 1 um: peak 0.5302, focus 2.658 um, the same to 4 digits as fed through the open edge (peak maps to 3e-6, the
  fluence to 8e-4 from the envelope cut of the TFSF laser at 3 sigma), against the x4 reference 0.5627 at 2.656 um (iRIP
  without o5 through the plane: 0.6061 at 2.842 um). tests/mpi.sh: examples/rz_tfsf.deck with irip o5 on 1 and 4 / 8 ranks
  along z (the plane on a rank boundary).
