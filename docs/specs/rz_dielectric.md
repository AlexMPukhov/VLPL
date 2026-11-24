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
  eps     = 2.1        # >= 1
  ramp    = 0 um       # > 0: linear z ends of this length centred on z_start, z_end
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
  - D4 hot plasma with rings, filled gaps, a collar and a rod crossing a ring face (deck path, shapes 1, 2): div D - rho and
    div B off the metal at round-off (6e-16); iRIP rejected.
  - D5 moving window, eps 3.8 lining, a bunch: Gauss 6.8e-13 of max rho, no growth (without the front layer and entry: 0.29
    and +17 % energy).
  - D6 the front-layer warning: a bunch loaded in the layer is warned at the start; a bunch at c catching up with a window at
    0.9 c is warned once when it enters (rate-limited); D5 raises none.
  - tests/mpi.sh: rings + gaps + collar in a moving window and two blocks in a periodic column (cyclic), 1 vs NP ranks.
- Benchmarks (bench/rz_dielectric/README.md): the DWA wake against the modal theory of the lined tube (eps 3.8, 2.1); the
  Cherenkov cone; thin-wall convergence (5 / 10 / 20 cells).

## References
Decision 100. Prior designs: O7 media (docs/specs/linear_medium.md, Crank-Nicolson current, iRIP) and the GEM-PIC-RZ
dielectric (J = -d_zeta(chi E), trapezoidal; public repository). Sub-cell smoothing: harmonic along the normal, arithmetic
along the interface (the effective-permittivity rule of anisotropic averaging).
