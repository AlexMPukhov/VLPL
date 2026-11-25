# Dielectric regions eps(r, z) in the explicit RIP-CC RZ step (blocks `dielectric`)

Status: implemented (decision 100, branch `rz-dielectric`). RIP-CC RZ only (iRIP / iRZ: the `medium` block, decision 84).

Motivation: dielectric-loaded structures (PTFE 2.1, HDPE 2.3, quartz 3.8), also together with the metal of the block
`rings` (decision 83), and dielectric pipes. The synchronous mode travels at nearly c (1 - beta_g ~ 0.01), so its frequency is very sensitive to eps.

## Physics and equations
A lossless, non-dispersive linear dielectric: D = eps E, B = H (code units, c = 1), as the polarization current of P = chi E,
chi = eps - 1:  d_t E = curl B - J - J_P,  J_P = d_t P,  so div(E + P) = rho.
Regions: rectangles r_in < r < r_out, z_start < z < z_end fixed in the lab frame (they move through a moving window as the rings
do); overlapping blocks add their chi; where the rings are metal, the metal rules win (E = 0).

## Discretisation
RIP-CC (`FieldsRZ::pass`, docs/specs/rz_modes.md): E_z, E_phi, B_r at (i, r_j); E_r, B_z, B_phi at (i, r_{j+1/2}); J_z at
(i, r_j), J_r at (i+1/2, r_{j+1/2}), J_phi at (i+1/2, r_j). J_P is time-centred (Crank-Nicolson) at each current point:
- E_z: J_P,z = chi_z (E_z^{n+1} - E_z^n) / dt is local: the predictor and the corrector divide their increment by 1 + chi_z.
- E_r, E_phi: y_a = dt J_P(a) = chi_a mu_z(E^{n+1} - E^n) at a = i + 1/2. Through the characteristics node i gets
  -(y_{i-1/2} + y_{i+1/2}) / 2 in E and -(y_{i-1/2} - y_{i+1/2}) / 2 in B_phi (+ in B_r). With dE_i = m_i (dEv_i -
  (y_{i-1/2} + y_{i+1/2}) / 2) (dEv the change of the vacuum transport with the free current; m_i = 0 where E is held at 0:
  metal points of the rings, E_phi on the axis, the guard nodes beyond an open z end) every (radial row, mode slot, component)
  is the tridiagonal line along z
    y_a (1 + chi_a (m_a + m_{a+1}) / 4) + chi_a / 4 (m_a y_{a-1} + m_{a+1} y_{a+1}) = chi_a / 2 (m_a dEv_a + m_{a+1} dEv_{a+1}),
  diagonally dominant; chi_a = 0 gives y_a = 0.
- With the block metal (conductors that depend on phi, docs/specs/rz_metal_phi.md, decision 103) the masks m depend on the
  angle, which the per-mode lines cannot hold. Then the lines are solved on the 2M - 1 collocation angles (Alexander,
  2026-11-24): E_r and its transported value at phi_k, E_phi at its own angles (the staggered phi_k + dphi / 2 with
  phi_derivative = difference); per (row, angle, component) the same line with the masks Mh (E_r) and Mp (E_phi, and the
  axis) of that angle; the corrections of E and B per angle with the face rules rBp, rBr (B from the vacuum side at a z face,
  nothing where B is held at 0); then back to the modes. The modes and the values on these angles are a bijection (1 +
  2(M - 1) real numbers each, Parseval), so nothing is projected: each angle is the line of decision 100, and stability and
  the energy follow per angle. An axisymmetric metal function (spectral) gives the rings result to round-off (test P6). The
  window entry (E_z of the entering columns from div D) applies the E_z rule of each angle column by column.
- eps from a deck expression (Alexander, 2026-11-24, Q2 (a)): chi of each component over its control volume from 8 x 8
  sub-samples of the function (inside the block's bounding box), with the same rule as the rectangles' cut cells: harmonic
  along the component's own direction (E_r across r with weight 1/r, E_z along z), arithmetic (r-weighted) across. Exact
  for interfaces along r or z, better than a staircase for tilted or curved ones, not second order there (the full
  anisotropic tensor would couple E_z with E_r). The function is evaluated on the device (the compiled program of the
  expression); a window shift moves chi by one column and samples only the edge columns. eps < 1 anywhere is an error.
- eps depending on the azimuth (the expression uses x, y or phi): chi is kept on the 2M - 1 collocation angles (E_z, E_r at
  phi_k, E_phi at its own angles, staggered with the metal block's phi_derivative = difference), each angle's control volume
  [phi -+ dphi / 2] sampled 8 x 8 x 4 (E_phi harmonic over phi, E_z, E_r arithmetic over it). Then every operation with chi
  goes through the angles (a bijection with the modes): the E_z increments of predictor and corrector are divided per angle
  (`rz_diel_ez_angles`), the lines are solved per angle (also without metal), the energy E.P / 2, the mode energies E_m . P_m
  and div D take D = (1 + chi) E on the angles back to the modes, the window entry likewise. The modes couple through chi
  (P = chi E is the collocation product); stability and losslessness hold per angle. Axisymmetric eps keeps the per-mode
  path (bit-identical).
- Step: predictor (1 + chi_z) -> vacuum transport (unchanged) -> exchange -> line solves and the corrections of the owned
  transverse fields (metal z faces: B from the characteristic of the vacuum side including y) -> exchange -> corrector
  (1 + chi_z). Without dielectric blocks nothing changes (bit-identical).
- Line solve (`DielectricRZ::correct`, src/rz_dielectric.cpp): only the radial rows with dielectric. Per rank the interior
  half points are solved by Thomas for three right-hand sides; the last half point of every rank is the interface unknown,
  whose equations form a P x P tridiagonal (cyclic for periodic z) system from 10 values per line and rank (one allgather per
  step), solved on every rank. At least 2 cells per rank. GPU: one thread per line, the allgather through the host.
- chi per point (`eps_cv`): eps over the control volume of each component, the harmonic mean along its normal and the
  arithmetic (r-weighted) mean along the interface: E_r over [z_i, z_{i+1}] x [r_j, r_{j+1}] harmonic in r (weight 1/r: D_r r is
  constant across a shell); E_z over [z_i -+ dz/2] x [rh_{j-1}, rh_j] harmonic in z; E_phi arithmetic. Cells inside or outside
  every block exactly, cut cells with 8 x 8 sub-samples. `ramp` (a length) gives a block linear z ends.
- Charge: J_P enters as a free current at its own point, so the RIP-CC identity gives div(E + P) - rho = const to round-off
  with P = chi E (E_z nodes, mu_z E_perp at i + 1/2). `FieldsRZ::div_e` returns div D when blocks are present (the Gauss check
  of the diagnostics), `energy()` adds E.P / 2.
- Stability and dispersion (von Neumann, research/rz_dielectric/vonneumann.py, 2D TM analogue with K dz <= 2): stable and
  lossless for every chi tested (to 3); dv/v second order, 1.6e-6 at k dz = 0.02, chi = 0.05.
- The grid-scale branch: the z-Nyquist mode has mu_z E = 0 and is not polarized, so near k_z dz = pi the scheme's group velocity
  in a dielectric exceeds c (max 1.05 c at chi 0.1, 1.2 c at 0.5, 1.95 c at 2.8; vonneumann.py). In a static box harmless.
  In a moving window such fields (scattered at the open rear edge) cross the window and grow at the moving open front edge;
  with dielectric blocks and a window therefore (rz_sim.cpp):
  - a front absorbing layer of 32 cells (`FieldsRZ::zabs_hi`, the z-layer kernel, kappa = 1.5 ln 1e3 / 32): keep beams and
    anything physical more than 32 cells behind the front edge (nothing physical is ahead of a beam at v ~ c);
  - E_z of the entering column and of the layer columns from div D of the column before (`rz_window_entry_dielectric`, as
    iRIP's window entry), so the damping and the entry leave no charge (target: the initial residual; entering matter neutral).
  A warning (rank 0; `RZFrontLayerWatch` in rz_sim.cpp) when charged particles are in the layer: at the start for every
  charged species whose load region reaches into it (a plasma entering through the front edge: one note), and during the run
  (checked every 20 steps, summed over the ranks) when particles of another charged species enter it, then at most every
  2000 steps while they stay (Alexander, 2026-11-13: keep the layer, warn if the user violates it).
  The last half point at the open front edge keeps its chi (the guard node held at 0), so P = chi E holds across the shifts.

## Deck interface
```
dielectric [<name>] {   # several blocks; RZ, numerics.solver = rip_cc
  r_in    = 0 um       # inner radius (default 0)
  r_out   = 1 mm       # outer radius (beyond the wall: up to the wall)
  z_start = 6 mm       # lab frame
  z_end   = 106 mm
  eps     = 2.1        # >= 1; or an expression of x, y, z, r, phi (lab frame, units allowed), e.g.
                       #   eps = 1 + 1.5 * (1 - (r / 0.5 mm)^2)    (a graded-index rod)
                       #   eps = x < 0 ? 2.1 : 1                    (a half-filled pipe: chi per angle)
                       # then r_in, r_out, z_start, z_end are an optional bounding box (default: everywhere)
  ramp    = 0 um       # > 0: linear z ends of this length centred on z_start, z_end (constant eps only)
  enabled = true
}
```
Not with: iRIP (rejected; use `medium`), the boosted frame (rejected). The TFSF plane and the beam self-field init assume
vacuum there: start a bunch with `self_field` before the dielectric. Example: the dielectric case of tests/mpi.sh (examples/rz_disk_loaded.deck with dielectric blocks).
Thin metal on a dielectric (rings with r_out - r_in of a few um): the rings need at least one radial cell; a film thinner than
dr is one cell of metal (a perfect conductor: its thickness does not matter once it is opaque), so place r_in, r_out on nodes.

## Tests
- Fast (tests/test_rz_dielectric.cpp):
  - D1 fields only, periodic, modes 2, two blocks with partial cells (eps 1.7, 3): div D unchanged to 1e-14 of max over 400
    steps; the energy (E.D + B^2) / 2 bounded.
  - D2 TM01 in a filled tube: w sqrt(eps) / w(vacuum, same grid) - 1 against the scheme's dispersion relation
    (vonneumann.py tm01): eps 1.1 / 2.25 / 4 at k_z dz = 0.098 and 0.393, within 0.5 % of the predicted 5.9e-5 ... 9.6e-3.
  - D3 a block with eps = 1: fields bit for bit equal.
  - D4 hot plasma with dielectric over the ring region, a collar and a rod (deck path, shapes 1, 2) with (a) rings, (b) no
    metal, (c) the block metal (four ridges depending on phi with z faces): div D - rho and div B off the metal at round-off
    (6e-16 ... 1.2e-15); iRIP rejected.
  - D5 moving window, eps 3.8 lining, a bunch: Gauss 6.8e-13 of max rho, no growth (without the front layer and entry: 0.29
    and +17 % energy).
  - D6 the front-layer warning: a bunch loaded in the layer is warned at the start; a bunch at c catching up with a window at
    0.9 c is warned once when it enters (rate-limited); D5 raises none.
  - D7 eps from a function equal to the rectangles of D1 (indicators, one across the periodic seam) = the rectangle blocks:
    chi to 9e-16, fields after 300 steps to 4.5e-15.
  - D8 graded-index tube eps(r) = 1 + 1.5 (1 - r^2 / R^2): TM01 against the radial ODE shot with RK4
    (research/rz_dielectric/grin_tm01.py, w = 2.155441729447): 7.9e-5 at dr = 0.05, 2.0e-5 at dr = 0.025 (second order).
  - D9 moving window: the D5 lining as a function of (z, r) = the rectangle block (energies to 4e-16, Gauss 7e-13).
  - D10 eps per angle (the graded rod written in x, y) = the same eps per (r, z): fields 5.9e-15, W and W_1 equal, div D 2e-14.
  - D11 a round pipe half filled with eps 2.1 (x < 0), k_z = 0: the lowest TM cutoff against polar finite differences
    (research/rz_dielectric/xsection_eps.py, Richardson over three grids, second order: 1.88013382): 8 modes +4.5e-4,
    16 modes -5.9e-7.
- With the block metal (tests/test_rz_metal_phi.cpp):
  - P6 an axisymmetric metal function (spectral) + three blocks = the rings + the same blocks (lines on the angles vs on the
    modes), random fields, modes 0..2, 300 steps: 3.9e-15 of max.
  - P7 the square guide (|x|, |y| < 0.6 in R = 1, 8 modes) filled with eps 2.1 / 3.8: at k_z = 0 every E update is divided by
    eps, so sqrt(eps) k_c = the vacuum k_c of the same grid (TM11, TE10): 3e-9.
  - P8 the corrugated square pipe of P4 (modes 0..6) with a lining crossing teeth and square, a rod and a block beyond the
    metal, random fields, 3000 steps: W / W0 in [0.92, 1.00], div D and div B off the metal unchanged to 7e-15.
  - tests/mpi.sh: rings + gaps + collar in a moving window, two blocks in a periodic column (cyclic), and the square
    corrugated pipe with a lining across its walls in a moving window (lines on the angles), 1 vs NP ranks.
- Benchmarks (bench/rz_dielectric/README.md): the DWA wake against the modal theory of the lined tube (eps 3.8, 2.1); the
  Cherenkov cone; thin-wall convergence (5 / 10 / 20 cells).

## References
Decision 100. Prior designs: O7 media (docs/specs/linear_medium.md, Crank-Nicolson current, iRIP) and the GEM-PIC-RZ
dielectric (J = -d_zeta(chi E), trapezoidal; public repository). Sub-cell smoothing: harmonic along the normal, arithmetic
along the interface (the effective-permittivity rule of anisotropic averaging).
