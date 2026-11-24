# Quasi-3D RZ with azimuthal modes (`RZModes`)

Status: steps 1 and 2 of the plan implemented (2026-10-28): fields per mode with the explicit RIP-CC
and the iRIP steps (`src/rz.hpp`, `src/rz_fields.cpp`, `src/rz_irip.cpp`, decision 43), and the particle
kernels: push with gather, charge-conserving deposition per mode, charge density
(`src/rz_particles.cpp`, decision 44); step 2b, the deck path (`grid.geometry = rz`, `src/rz_sim.cpp`,
decision 45: periodic z, conducting wall, plasma); step 3a, open z, the laser in the modes and the
moving window (`src/rz_laser.cpp`, decision 46); step 3b, openPMD thetaMode output and the LWFA benchmark
against FBPIC and Cart3D (decision 48); step 4a, a radial absorbing layer (decision 49); step 4b, shape 2
with the stencil folded across the axis (decision 50); step 4c, checkpoints (decision 51); step 5, iRIP
with ranks along z (the partition method of decision 39). The charge-conserving deposition and the field step per mode are verified in numpy
(`research/rz_modes/`): the Gauss law per mode is conserved to round-off with particles, for explicit
RIP-CC and for the box-closure iRIP. Phase 2,
architecture spec section 2 (D-1), decisions 17, 38-41.

## Physics and conventions
- Fields as azimuthal modes: F(z, r, phi) = F_0(z, r) + 2 Re sum_{m=1}^{M-1} F_m(z, r) e^{i m phi},
  F_m = (1/2 pi) int F e^{-i m phi} dphi (the convention of GEM-PIC-RZ). m = 0 real, m >= 1 complex.
- Components Ez Er Ephi Bz Br Bphi; z is the transport axis (c dt = h_z), as x in Cart3D.
- Particles stay Cartesian (z, x, y) (D-1); r, phi only inside gather and deposit.

## Discretisation
### Grid and staggering (RIP-CC with x -> z, y -> r, z -> phi)
| quantity | position |
|---|---|
| Ez, Jz | (i, r_j) |
| Er, Bphi | (i, r_{j+1/2}) |
| Ephi, Br | (i, r_j) |
| Bz | (i, r_{j+1/2}) |
| rho, J_phi | (i+1/2, r_j) |
| J_r | (i+1/2, r_{j+1/2}) |

r_j = j h_r, j = 0 .. N_r - 1, a conducting wall at r = N_r h_r (outer PML later). Cell volumes
V_j = 2 pi r_j h_r h_z, axis V_0 = pi (h_r/2)^2 h_z. The derivative d/dz(Cart) -> i m / r.

### Field step per mode
The Cartesian RIP-CC step with the transverse operators of research/irip_rz/irip_rz.py:
- corrector: Ez += D [(1/r) d_r (r <Bphi>) - (i m / r) <Br> - Jz], Bz -= D [(1/r) d_r (r <Ephi>) - (i m / r) <Er>];
- transport sources: Psi_r = (i m / r) mu_z Bz* - J_r, Psi_phi = -d_r mu_z Bz* - J_phi,
  Phi_r = -(i m / r) mu_z Ez*, Phi_phi = d_r mu_z Ez*; T+ = Er + Bphi and Ephi - Br move +z, the others -z;
- explicit RIP-CC: the predictor Ez*, Bz*; iRIP: the box averages (Ez^n + X)/2, (Bz^n + Y)/2 with
  (I - D^2/4 mu_z^2 L_m) X = R per mode, L_m the radial operator of the mode (a fixed N_r x N_r matrix,
  diagonalised once per mode: research/irip_rz, real spectrum, energy conserving), the system along z per
  radial eigenmode (Thomas, partition method across ranks) as in Cart3D.
- Axis: m = 0: Ephi = Br = 0 on the axis, (1/r) d_r (r F)(0) = 4 F_{1/2} / h_r; m >= 1: Ez = 0 on the
  axis; Ephi, Br on the axis are not dynamic (m = 1: one-sided rules, Bz(-h/2) = -Bz(h/2), Ez ~ r).
- Explicit RIP-CC needs D^2 (lambda_max(L_m) + wp^2) / 4 < 1 for all m (as Cart3D's
  dx^2 (1/dy^2 + 1/dz^2) + dx^2 wp^2 / 4 < 1; checked numerically: random fields stay bounded at
  D^2 lambda_max / 4 = 0.98 and grow at 1.02); lambda_max grows with m^2 / r^2 near the axis (4.5 x 4/h_r^2
  for 3 modes), so many modes or fine h_r call for iRIP (no limit). `rz_lambda_max` gives it.

### Deposition (charge conserving per mode)
A particle of charge q w moves (z0, x0, y0) -> (z1, x1, y1); r, phi at both ends. Linear weights S_z on
the rho grid (i+1/2) and S_r on r_j: W = S_z S_r, e = e^{-i m phi}. Exactly
    Delta(W e) = (Delta W) e_bar + W_bar (Delta e),   e_bar = (e0 + e1)/2,  W_bar = (W0 + W1)/2.
- Delta W is split by the 2D Esirkepov rule in (z, r): Wz = (S1z - S0z)(S0r + S1r)/2,
  Wr = (S0z + S1z)(S1r - S0r)/2; charge fluxes through the z faces = -cumsum_z Wz, through the r surfaces
  = -cumsum_r Wr from the axis; times e_bar. J_z = flux / (A_j dt), A_j = V_j / h_z;
  J_r = flux / (2 pi r_{j+1/2} h_z dt).
- W_bar Delta e gives J_phi,m = i r_j q w W_bar (e1 - e0) / (m dt V_j) for m >= 1 (it balances
  (i m / r_j) J_phi in the divergence); as dt -> 0 it is q v_phi S e^{-i m phi}. J_phi,0 = q w v_phi S / V_j.
- Then (rho^{n+1}_m - rho^n_m)/dt + div_m J = 0 at every rho node (axis: m = 0), so the Gauss law per
  mode is conserved by both closures (prototype: 2e-16 ... 9e-16 relative, m = 0 .. 3).
- Uniform plasma: rho_j = n for j >= 1, the axis node gets 4/3 n (linear weights, V_0 = pi h_r^2 / 4);
  neutral plasmas cancel. A volume correction would break the Esirkepov fluxes (the weights must sum to
  1); kept as is, documented.
- Shape 2: the support crossing r = 0 is folded (step 4b below).

### Gather
E, B per mode interpolated with the staggering descriptor at (z, r), summed with e^{i m phi}, rotated to
Cartesian (x, y). m = 1 on the axis: Er, Ephi from the one-sided rules (the axis field is transverse).

### Boundaries, MPI, window
- z: periodic or open (inbound T at the edge, laser by modes), moving window as in Cart3D (iRIP: the
  entry rule of decision 40, per mode).
- r: conducting wall first; CPML in r later (decision 41 carries over: the radial operator stays per mode).
- MPI along z only (the radial transforms need whole columns).

## Implementation plan (one PR each)
1. RZ fields and solvers per mode: storage (2M - 1 real arrays per component), RIP-CC and iRIP steps,
   vacuum tests (TM/TE guided modes of the conducting cylinder against the Bessel dispersion, energy).
2. Particles: deposition (above), gather, the existing pushers; Gauss law per mode with plasma, Langmuir.
2b. Deck input (`grid.geometry = rz`), the RZ operators in `Simulation`, loading in the cylinder (profiles),
   particles leaving through the wall, migration along z, diagnostics.
3. Laser in modes (Gaussian m = 1, LG), open z, window (3a); field output (openPMD thetaMode) and the
   LWFA benchmark RZ against FBPIC and Cart3D (3b, the Phase 2 exit test).
4. CPML in r; shape 2 near the axis; RZ in openPMD (thetaMode).
5. iRIP with ranks along z.
6. Stretched radial grid (below): 6a fields, 6b particles, 6c output and checkpoints.

## Implementation so far (step 1)
- Storage: the Cart3D `Field` with axes (z, r, slot), slot 0 = m = 0, slots 2m - 1, 2m = Re, Im of
  mode m; the Domain has N = {Nz, Nr, 2M - 1}, h = {dz, dr, 1}, r and the slots neither periodic nor
  split, so their guards are zero (the wall beyond r_Nr) and the halo, the kernels and the z exchange
  are those of Cart3D (`rz_domain`). The guard check of `Comm::decompose` now applies only to periodic
  or split axes.
- `FieldsRZ::pass`: one pass of the step with complex mode arithmetic per (i, j, m); the explicit step
  uses the predictor, iRIP the box averages (`rz_irip_step`).
- iRIP: per mode and kind the radial operator is built by probing the same formulas on the host,
  checked real and tridiagonal, symmetrised by a diagonal similarity and diagonalised once (QL,
  `src/tridiag_eigen.hpp`, shared with the CPML bases). Per step: the radial transform, the system along
  z per radial mode (cyclic for periodic z, zeros outside for open z), back. One rank along z so far
  (the explicit step runs on ranks along z).
- Tests (`tests/test_rz.cpp`): TM01 and TE11 guided modes of the conducting cylinder against the Bessel
  roots through the iRIP law, relative errors 3.4e-5 and 1.0e-6 at nr = 32 (more than second order in
  dr); iRIP random fields in modes 0..2 at dz/dr = 4: energy conserved to 4e-14, div E and div B per
  mode unchanged to 3e-14; explicit RIP-CC on a stable grid: bounded (max 1.001 W0), div E and div B
  unchanged to 2e-15, TM01 within 1.5e-5 of the continuum frequency.

## Implementation so far (step 2)
- `src/rz_particles.cpp`: `rz_push_deposit` (gather, the pushers of Cart3D through
  `push_momentum_inline` in `src/pushers.hpp`, move, deposition), `rz_rho`, `rz_gather_point` (tests).
  Kernels as in Cart3D: one `par_for` over the particles, atomics, `[=] HD` lambdas on plain views.
- Gather: per mode linear interpolation with the staggering above, values below the axis by mirror
  parity (F_m(-r) = (-1)^m F_m(r) for Ez, Bz; the opposite sign for the radial and azimuthal components),
  then F = F_0 + 2 Re sum F_m e^{i m phi} and the rotation (r, phi) -> (x, y). m = 1 on the axis: E_r1
  and E_phi1 are one transverse vector, E_phi1 = i E_r1 (B_r1 = -i B_phi1).
- Deposition: the split of decision 42 per straight segment. A move longer than one cell in z or in r
  (iRIP: c dt > h_r, or fast particles) is split into n = ceil(max(|dz|/h_z, |d r_perp|/h_r)) equal
  Cartesian sub-segments (the chord bounds the change of r); each conserves charge, so the sum does.
- numerics.rz_deposit = chord (2026-11-13): `deposit_seg_chord` puts the currents of `seg_currents` (the implicit
  mover's splits) into the explicit deposit: m = 0 J_phi = w L / (r_j dt) with L = x0 y1 - y0 x1 instead of
  w r_j dphi / dt, m >= 1 J_r with the chord phase e_c = ((z0 + z1) / (r0 + r1))^m. The default (phase) gives a segment
  that passes the axis at b << dr a J_phi of ~ pi r_j^2 / dt: test R13, dt sum r J_phi0 dV / (q w L) at a move of 0.2 dr:
  b = 2 dr phase 1.06, chord 1; b = 0.2 dr 14.5 and 0.61; b = 0.01 dr 693 and 0.47 (error 1.38 and 0.001 dr^2 q w). R5
  with chord: Gauss law per mode to 1.5e-14. Found in a boosted RZ LWFA run (the axis noise of the wake electrons).
- Tests (`tests/test_rz.cpp`): R4 gather of a uniform transverse E and B (mode 1, also next to the
  axis) to 4e-16; R5 hot plasma in modes 0..2, Gauss law per mode and div B unchanged to round-off for
  RIP-CC (1.7e-14) and iRIP at dz/dr = 2.5 with moves over several radial cells (8e-15); R6 Langmuir
  oscillation of a cold plasma filling the cylinder (quiet start, curl-free J0 perturbation, projection on
  the mode), w within 0.6 % (iRIP) and 0.9 % of the leapfrog law (the linear shapes lower w slightly,
  as in the Cartesian test). A perturbation
  uniform in r would also excite the TM modes of the plasma-filled guide (E_z = 0 at the wall), with
  w^2 = wp^2 + k^2 + kp^2: physics, not an error (the first version of the test saw them).
- Particles beyond the last radial cell are skipped (no wall treatment yet, step 2b).

## Implementation so far (step 2b)
- Deck (README, "RZ with azimuthal modes"): `grid { geometry = rz, modes, nz nr, dz dr, boundary_z =
  periodic, boundary_r = wall, mpi_z }`; species keys in particle axes: `ppc_z ppc_r ppc_theta`,
  `uz ux uy`, `uth_z uth_x uth_y`, `zmin zmax rmin rmax`, `density_function = f(z, r)`, `t_start`.
  Rejected so far: laser, window, pml, self_field, shape 2, the Yee solver, slices / openPMD /
  checkpoints, iRIP on more than one rank (laser and window: step 3a; openPMD 3b; shape 2 4b;
  checkpoints 4c; iRIP over ranks 5).
- `Simulation` with `cfg.rz` (`src/rz_sim.cpp`): the RZ domain (ranks along z), `FieldsRZ`, the operators
  rz_push_deposit, rz_fields, migrate. Loading: regular in (z, r, theta) per cell, weight
  n 2 pi r (dr / ppc_r)(dz / ppc_z) / ppc_theta: rho_0 = q n exactly at the nodes off the axis (the
  midpoint errors of the two sides of a linear weight cancel), 4/3 n in the continuum on the axis node
  (1.5 n at ppc_r = 2).
- Wall: `migrate(..., rwall)` removes particles at r >= nr dr after the push. Their current up to the
  wall is deposited (the r guards hold the stencil), so the Gauss law at the nodes inside stays exact;
  E_z of the wall node is 0 (the conductor), the gather near the wall uses the zero guard of E_r.
- diag.txt as in Cart3D: W_field per mode, W_kin, the change of the Gauss residual per mode (the axis
  node is a Gauss point of m = 0 only), max |rho|, max |div B| per mode.
- Tests: R7 (`tests/test_rz.cpp`) runs a deck: loading exact to 5e-16, Gauss drift 1e-16 with 10 % of the
  electrons absorbed, the RIP-CC stability check; `tests/mpi.sh`: RZ RIP-CC 1 vs 8 ranks along z (equal
  to all printed digits); regression case `rz` (`examples/rz_plasma.deck`, iRIP, 2 modes).

## Implementation so far (step 3a)
- Open z (`grid.boundary_z = open`): the zero guards beyond the edges are the outgoing transport (inbound
  T = 0), particles leave with exact currents (`keep_lo`, decision 21) through the same `migrate`; the
  last cell at the right edge is no Gauss point (as Cart3D). iRIP: the system along z with zeros outside.
- Laser (`laser` blocks, `src/rz_laser.cpp`): the Cartesian `LaserConfig` (focus, Laguerre-Gauss, phase,
  several lasers summed) is evaluated in the transverse plane of the particles at 4 (M + 8) azimuthal
  points of each radial node and projected on the modes; the inbound T in the guard i = -1 at t + dt:
  E_r + B_phi, E_phi - B_r with B_phi = E_r, B_r = -E_phi. `FieldsRZ::inflow` hooks run at the start of
  the step (iRIP: the same guard in both passes). Deck: `polarization = x|y` (the particle axes), round
  and on the axis (no `waist_z`, `y_center`); a linearly polarised LG_p^l needs `modes >= |l| + 2`.
- Moving window (`window` block): the fields and the initial Gauss residual shift one cell along -z
  (`shift_fields_x`), the entering column is loaded (`load_rz` on [n-1, n)), the entry columns are frozen
  in the push step before a shift (`Domain::xfreeze`), and with iRIP Ez, Bz of the entering cell come
  from the Gauss law and div B of the cell before it per mode (`rz_window_entry`, decision 40).
- Diffraction of the laser (R8): the scheme's dispersion with c dt = dz, tan^2(w dt/2) = tan^2(k_z dz/2)
  + dz^2 k_perp^2 / 4 for iRIP, makes the diffraction phase smaller by x / (tan x sec^2 x), x = k dz / 2
  (0.81 at dz = lambda/8, 0.95 at lambda/16); RIP-CC's sin law makes it larger by 2x / sin 2x. The on-axis
  fluence of a round beam at z = 8 um, z_R = 24.5 um, against the paraxial law:

  | dz | dr | fluence ratio / paraxial - 1 |
  |---|---|---|
  | lambda/8 | w/10 | +2.4e-2 |
  | lambda/16 | w/10 | -0.9e-3 |
  | lambda/16 | w/20 | +4.2e-3 |
  | lambda/32 | w/10 | -8.9e-3 |
  | lambda/32 | w/20 | -2.9e-3 |

  converging to the paraxial law (dz and dr errors of opposite sign; the 8 fs pulse's bandwidth is not
  the cause: -7.9e-3 with 24 fs at lambda/32, w/10).
- Tests: R8 laser in vacuum (iRIP, open z): energy against the analytic pulse energy (-8e-4), conserved
  while inside (1e-6), on-axis fluence against paraxial (1e-3), E_y = 0 on the axis for x polarisation,
  3e-6 left after the exit, Gauss law and div B 1e-15. R9 moving window with a laser wake and plasma
  entering, RIP-CC and iRIP: Gauss law and div B 2e-15 (without the iRIP entry rule 1.7e-4), protons =
  shifts x radial cells x ppc exactly. `tests/mpi.sh`: the laser deck and the shortened LWFA deck (window), 1 vs 8 ranks along z (3e-13).
  Regression case `rz_lwfa` (`examples/rz_lwfa.deck`, the RZ twin of `examples/lwfa_window.deck`).

## Implementation so far (step 3b)
- openPMD output (`output.openpmd_every`, fields E B J rho rho_<species>, particles): geometry thetaMode,
  `geometryParameters = "m=M;imag=+"`, axisLabels (r, z), datasets (2M - 1, Nr, Nz) per component r, t, z.
  openPMD reconstructs F = F_0 + sum_m [C_m cos(m theta) + S_m sin(m theta)]; ours is F_0 + 2 Re sum F_m
  e^{i m theta}, so C_m = 2 Re F_m and S_m = -2 Im F_m. Positions in the cell (r, z) after the staggering
  table above; J at t - dt/2. Particles: the openPMD components x, y, z are the particles' Y, Z, X (z is
  axis 0 of the particles). Checkpoints: step 4c.
- Tests: R10 (`tests/test_rz.cpp`) writes a uniform transverse E = E0 x, B = B0 y and reconstructs it with
  the standard's formula from the file (2e-16), with the mesh attributes and the particle axes. The
  openPMD validator passes on an RZ LWFA output; openpmd-viewer reads fields and particles.
- LWFA benchmark (`bench/lwfa_fbpic/rip_rz.deck`, `compare_rz.py`, README there): VLPL-RZ against FBPIC
  on FBPIC's grid, wake Ez (m = 0, r <= 0.5 um) within 4.3-5.9 % (3D: 3.4-4.6 %), laser 0.2-4 %, wake peak
  within 0-5 % of VLPL-RIP 3D: the Phase 2 exit criterion "RZ vs 3D agreement for LWFA". On the axis row
  alone the comparison is limited by particle noise (two samplings differ by 5-14 %).

## Implementation so far (step 4a): radial absorbing layer
- `grid.boundary_r = absorbing` (`grid.r_absorber_cells = 32`, `grid.r_absorber_R0 = 1e-4`): after the field
  step every component of every mode is multiplied by exp(-kappa s^2), s the depth into the outer layer
  at the component's radius (nodes r_j, half nodes r_{j+1/2}); kappa = 1.5 ln(1/R0) / a, a = cells dr / dt
  the steps to cross the layer, so two crossings damp by R0 (`rz_absorb`, `src/rz_fields.cpp`).
  `grid.absorber_particles` (decisions 111, 115): remove: no current in the layer (J_z, J_phi at r_j > r_{j0}, J_r at
  r_{j+1/2} > r_{j0}, j0 = nr - cells) and particles removed at r >= r_{j0}; conserve (default): removed at r_{j0} + (1 + (s - 1)/2) dn,
  current deposited into the layer; keep: absorbed at r = nr dr as before; the removed number and charge are printed at the
  end of the run. In RZ the plasma in the layer drives no instability (plasma-only, 455 fs: E_phi m1 near the axis 1.49e7 V/m
  with all three; with a column clear of the layer 1.44e7); remove adds ~8 C/m^3 of mean charge next to the face by 453 fs
  (14 with conserve / keep; plasma 5e5), Gauss residual 3.2e-9 then round-off. The Gauss law and div B are checked inside the layer only,
  as with the Cart3D CPML. Works for both closures and any incidence (the damping acts per step).
- Not a PML: what remains is the reflection from the gradient of the damping, mostly of radial
  wavelengths longer than the layer. A ring pulse of width 0.6 / k0 in modes 0, 1 (R11) leaves
  9e-4, 2.7e-4, 9e-5 of its energy inside for 16, 24, 32 cells at dz = dr (iRIP), independent of R0
  (1e-3 .. 1e-6), 2.6e-4 for 32 cells at dz = 2 dr; the conducting wall keeps 93 %. A cylindrical
  CPML (stretched r in the 1/r terms per mode) would remove the long-wavelength reflection if needed.
- Tests: R11 (both closures, < 3e-4 at 32 cells); the plasma deck with plasma streaming into the layer:
  Gauss law 2e-16 inside; `tests/mpi.sh`: the RZ laser deck with the layer, 1 vs 8 ranks.

## Implementation so far (step 4b): shape 2
- `numerics.shape = 2` in RZ: quadratic splines in z and r on the charge grid for the deposition and at
  each component's staggering for the gather (`weights<O>`, the kernels templated on the order).
- Below the axis the stencil is laid out on the line r in (-inf, inf) and folded onto r >= 0 at charge
  level: a node -j onto +j with (-1)^m (rho, J_z, J_phi, each then divided by the target node's volume or
  area), a radial flux at half node -(j+1/2) onto +(j+1/2) with -(-1)^m (a flux towards the axis from the
  mirror side is an outward flux through the mirrored surface). The folded fluxes balance the folded
  charges node by node: dW_1 + (-1)^m dW_-1 = -[(F_3/2 - (-1)^m F_-3/2) - (F_1/2 - (-1)^m F_-1/2)], and for
  m = 0 at the axis node dW_0 = -(F_1/2 - F_-1/2) with the same V_0 as the field solver. The gather mirrors
  node quantities (-j -> j) and half-node ones (-1-j -> j) with the parity of the component.
- Wall: a shape-2 particle has weight on the last node r_{nr-1} until r = (nr + 1/2) dr; it is absorbed
  there, not at nr dr (removed at nr dr it took charge from the last node without a current: a Gauss
  error of 6e-4 in the LWFA benchmark when electrons reached the wall).
- Tests: R4 (uniform transverse field, exact for both shapes), R5 (hot plasma crossing the axis, modes
  0..2, Gauss law per mode 1e-14 for RIP-CC and iRIP with shape 2), R7 (wall absorption, shape 2: Gauss
  5e-17), R9 (moving window, iRIP, shape 2). The loading of shape 2 deposits q n to 0.8 % off the axis
  (the midpoint quadrature of the quadratic weights; shape 1: exact).
- LWFA against FBPIC with shape 2: wake peak equal to VLPL-RIP 3D's to 0-1 %, near-axis L2 3.5-4.9 %, axis
  row 4.4-6.2 % (shape 1: 7-11 %): shape 2 removes most of the axis noise (bench/lwfa_fbpic/README.md).

## Implementation so far (step 4c): checkpoints
- The `checkpoint` block and `--restart` as in Cart3D (docs/specs/openpmd.md). The file holds E and B as
  the thetaMode output in float64 (C_m = 2 Re F_m, S_m = -2 Im F_m: the factors are powers of 2, the
  reading divides back exactly), the initial Gauss residual per slot (`rippic_gauss0`), the particles
  (openPMD x, y, z = slots Y, Z, X) and the state group with `geometry = rz` (a Cart3D checkpoint is
  refused, as another grid, modes (N[2] = 2M - 1) or shape). Nothing else carries over between steps:
  the currents are rebuilt each step, the laser inflow is a function of t, the absorber and the iRIP
  work arrays hold no state; the window's freeze of the entry columns follows from t.
- Another number of ranks: every rank keeps the particles of its z slab (only z is split).
- Tests: R12 (rip_cc and irip, shape 2, 3 modes, window, laser, entering plasma, absorber): the restart
  from a checkpoint after the window started equals the uninterrupted run in every field slot, the
  Gauss residual and every particle, exactly on one CPU thread (the RZ deposit adds atomically, so with
  several threads two uninterrupted runs differ at round-off already). `tests/mpi.sh`: the RZ window deck
  from its checkpoint of 8 ranks on 8 (diag.txt byte-identical) and on 3 ranks.

## Implementation so far (step 5): iRIP with ranks along z
- The systems along z per line (radial eigenmode a, slot k) after the radial transform are the Cart3D
  ones along x with real right-hand sides: the partition method of decision 39 (`solve_partitioned` in
  `src/rz_irip.cpp`, the Cart3D one in `src/irip.cpp` for complex lines). Per rank three local Thomas
  solves more (u, v, and for periodic z the Sherman-Morrison vector), one allgather of 8 values per line,
  the reduced block-tridiagonal system of the interfaces on every rank. The radial transforms need no
  communication (r is not split); everything else (passes, window entry rule, laser) was already
  rank-aware. At least 3 cells along z per rank.
- Tests (`tests/mpi.sh`): the periodic plasma deck (cyclic, 1 vs 8 ranks) and the LWFA window deck
  with laser and plasma (open z, 1 vs 8 ranks) agree with one rank to all digits of diag.txt; the Gauss
  law stays at round-off on every rank count.

## Stretched radial grid (step 6, done: 6a fields, 6b particles, 6c output)
Fine cells near the axis (h_r << c dt with iRIP), coarser outside, as GEM-PIC-RZ's `radial_grid = table`.
- **Grid.** Nodes r_0 = 0 < r_1 < ... < r_nr with spacings Delta_j = r_{j+1} - r_j: `grid.dr` for the first
  `grid.r_uniform / dr` cells, then Delta_j = min(dr s^(j - ju + 1), dr_max) (`grid.stretch = s`, default
  1: uniform; `grid.dr_max`); half nodes rh_j = (r_j + r_{j+1}) / 2. The cells next to the axis stay
  uniform (at least 4), so the axis rules and the shape-2 fold are unchanged.
- **Operators (finite volume, energy conserving on any grid).** Per row four weights: the node area
  A_j = (rh_j^2 - rh_{j-1}^2) / 2 (A_0 = rh_0^2 / 2, the axis volume of decision 42), the half-node
  weight B_j = rh_j Delta_j, the gradient spacings Delta_j (node -> half) and delta_j = A_j / r_j
  (half -> node). Divergences (1/r) d_r (r g): (rh_j g_j - rh_{j-1} g_{j-1}) / A_j at nodes,
  (r_{j+1} f_{j+1} - r_j f_j) / B_j at half nodes; gradients (f_{j+1} - f_j) / Delta_j and
  (g_j - g_{j-1}) / delta_j; the m / r terms at their own radius. Then sum_j B_j g_j (d f)_j =
  -sum_j A_j f_j (div g)_j and sum_j A_j f_j (d g)_j = -sum_j B_j g_j (div f)_j: the curl pair is
  adjoint in the energy weights 2 pi A_j dz (nodes) and 2 pi B_j dz (half nodes), so RIP-CC keeps its
  energy law and iRIP its exact energy conservation; the same divergence gives the Gauss law. On a
  uniform grid all weights reduce to the present formulas (A_j = r_j dr, B_j = rh_j dr, delta_j = dr).
  Axis: m = 0: 4 Bphi_{1/2} / dr = rh_0 Bphi_0 / A_0; m = 1: -Bz_0 / rh_0 and -i Ez_1 / r_1.
- **iRIP**: the probed L_m on the grid is still tridiagonal and self-adjoint in the weights, so the
  diagonal similarity and the eigenbasis are unchanged; rz_lambda_max for RIP-CC from the same basis.
- **Particles (6b)** in the index coordinate xi = j + (r - r_j) / Delta_j (piecewise linear): the shapes,
  the fold and the Esirkepov fluxes in xi as now, the charge per node divided by 2 pi A_j dz, the radial
  flux by the face area 2 pi rh_j dz: charge conservation is exact for any grid (it is a statement about
  charges and fluxes in xi). Gather in xi. Loading per cell volume; the wall in xi.
- **Absorber, laser, window entry, diagnostics**: the radii and weights of the grid.
- **Output (6c)**: openPMD has no non-uniform mesh: the thetaMode records keep `gridSpacing` = dr (the
  fine spacing) and carry the node radii as the attribute `rippic_r_nodes` (and a note); checkpoints
  store the grid parameters and refuse another grid.
- **Rule** (CLAUDE.md of GEM-PIC-RZ): k_perp Delta <~ 1 wherever radiation must leave, a slow stretch; the
  setup prints 2 / dr_max (the cutoff) and the radius where k_perp Delta = 1 for the laser's k.
- **Tests.** 6a: energy conservation of random fields on a stretched grid (iRIP to round-off, RIP-CC
  bounded); the guided TM01 / TE11 modes on a stretched grid against the Bessel roots (convergence);
  the reflection test of the Gate (a radially outgoing pulse from the fine region through the stretch
  into the absorber: energy returning inside against the same run on the uniform fine grid; resolved:
  < 1e-4, near the cutoff k Delta ~ 2: large, as the prototype); Gauss law and div B unchanged in vacuum.
  6b: hot plasma Gauss law per mode at round-off on the stretched grid (both shapes); LWFA on the
  stretched grid against the uniform fine grid (wake near the axis to ~1 %).

## Implementation so far (step 6a): fields on a stretched radial grid
- `RadialGrid` (`src/rz.hpp`, `rz_radial_grid` in `src/rz_fields.cpp`): the radii and the four weights per
  row on the host and as one device buffer; the kernels read them through the plain `RadialView`. Every
  RZ field operator (RIP-CC passes, divergences, the energy, the window entry rule, the laser projection)
  and the iRIP basis use it; a uniform grid gives the earlier results to round-off (regression unchanged).
- Deck: `grid.r_uniform`, `grid.stretch`, `grid.dr_max` (README). The setup line prints the cutoff 2/dr_max
  and the radius up to which k0 dr <= 1. `grid.r_absorber_cells` counts cells: on a stretched grid the
  layer is as thick as its (coarse) cells.
- Tests (`tests/test_rz_grid.cpp`): R13 random fields, modes 0..2, 0.05 -> 0.4 at 12 % per cell: iRIP at
  dz = 8 dr_min conserves the energy to 2e-14, RIP-CC (dz from rz_lambda_max) bounded, div E and div B
  unchanged (2e-14). R14 TM01 / TE11 on stretched grids, second order under refinement (2.2e-3 -> 5.2e-4,
  2.7e-4 -> 4.7e-5). R15 (Gate B): a ring wave with k_r = 4 leaves through 0.05 -> 0.2 (k dr_max = 0.8, 4 %
  per cell) with 5.5e-7 of its energy coming back into the fine region (uniform fine grid: 3.4e-6, the
  absorber's floor); stretched to 0.6 (k dr_max = 2.4, beyond the cutoff) 12 % comes back. R16 the laser
  on 60 stretched cells carries the energy of 144 uniform ones to 3.6e-4, the Gauss law at 5e-15.

## Implementation so far (step 6b): particles on a stretched radial grid
- `rz_xi` (src/rz.hpp): the index coordinate xi(r), r / dr on the uniform part (uniform grids keep their
  results), a binary search over the node radii beyond it, the last spacing beyond the arrays (they run to
  nr + 4 for the shapes' guards). Gather, deposit and rho use xi in place of r / dr; volumes 2 pi A_j dz,
  radial faces 2 pi rh_j dz, J_phi with r_j. Moves are split so that no sub-segment changes xi by more than
  one: the chord over the cell size at its closest approach to the axis (the cells grow outward).
- Loading regular in r within each cell; the wall in xi (nr for shape 1, nr + 1/2 for shape 2).
- Tests (`tests/test_rz_grid.cpp`): R17 a hot plasma column (u_th = 0.1, modes 0..2) reaching into the
  stretch, partly absorbed by the wall: Gauss law 1e-16, div B 4e-17 for iRIP shape 1 and 2 and RIP-CC
  shape 2. R18 loading rho = q n: 1e-15 on the uniform part off the axis, 9e-4 in a stretch of 8 % per cell.
  R19 a laser wake (a0 = 1.5, n = 0.01 n_c) on 60 stretched cells against 144 uniform ones: E_z on the
  axis within 1.5 % (L2). `tests/mpi.sh`: the plasma deck with iRIP and shape 2 on a stretched grid, 1 vs
  8 ranks identical.

## Implementation so far (step 6c): output and checkpoints on a stretched radial grid
- openPMD has no non-uniform mesh: the thetaMode records keep `gridSpacing` = dr (the spacing next to the
  axis) and, on a stretched grid, carry the node radii r_0 .. r_nr (code units, in `gridUnitSI`) as the
  attribute `rippic_r_nodes`; half-node components sit at the midpoints (`position` 0.5). A reader that
  ignores the attribute places the outer cells wrongly: read it (openPMD-viewer: use the radii from it).
- Checkpoints store the node radii (`r_nodes` in the state group); a restart with another radial grid is
  refused. Particle patches span the cylinder of the grid's wall radius.
- Test R20 (`tests/test_rz_grid.cpp`): the radii in the output equal the grid's; the hot plasma of R17
  restarted from a checkpoint equals the uninterrupted run exactly (one thread); another stretch refused.

## Axis: density-conserving shapes (decision 53)
- Problem: with the plain splines a uniform density deposits 4/3 (shape 1) and 13/8 (shape 2) of itself onto the
  axis node (node 1, shape 2: 1.005), since the splines' charge-assignment volume of node 0 differs from the
  solver's axis volume pi dr^2 / 4. Seen in B7 as a line of excess electron density on the axis ahead of the
  laser (FBPIC, with its default Ruyten shapes, has none); the same factor couples near-axis charge and current to
  the axis fields.
- The deposition (rz_push_deposit, rz_rho) uses the splines in a mapped index coordinate (rz_xi(.., shape),
  coefficients in src/rz_ruyten.cpp); the map keeps the nodes at xi = j and is monotone, so the Esirkepov fluxes
  in xi conserve the charge exactly as before. The gather stays plain (as FBPIC).
- Loading at the Gauss-Legendre points of each radial cell. A species with density = 0 (filled by ionization) is not
  loaded, as in Cart3D (before 2026-11-09 RZ filled it with zero-weight particles; decision 90).
- rho_m for m >= 1 is not deposited on the axis node (rho_m ~ r^m; checked in R5): before, the output plane showed
  negative density on the axis inside dense near-axis bunches (C_0 +- C_1 with the noisy dipole of the axis half cell).
- The map is used for m = 0 only (decision 106, `numerics.rz_axis_map = m0`): the modes m >= 1 deposit, and with
  rz_gather = matched gather, in the plain radial coordinate, each mode conserving charge with its own weights. With the
  map in every mode a uniform transverse current had 0.833 (shape 1) / 0.626 (shape 2) of its value at the first face and
  the dipole of a charge at 0.1 dr was 1.45 x / 4.7 x too large; now 1.0000 / 1.083 and exact (tests A1, A2 in
  tests/test_rz_axis.cpp; A3 prints the m = 0 gather of a linear field for plain, matched and energy).
- Tests: R7 loading rho = q n on every node including the axis (shape 1 1.7e-15, shape 2 2.6e-3, was 0.6), R18
  the same on a stretched grid (3e-15), R5 / R17 the Gauss law at round-off, R6 Langmuir second order in dr.

## Current filter (decisions 54, 70, 91)
- **Binomial (`numerics.filter = binomial`, the default since decision 91).** Per pass: z step S_z on J_z
  (J_z,i += (dz/4)(w_i - w_{i-1}) with w = d_z J_z alone) and the binomial over the Gauss cells on J_r, J_phi (they sit at
  the cell, as rho); r step as below (T_r on J_z and rho, S on J_perp). Every component then gets the binomial in z and
  J_z the radial T of rho, so a plasma drifting along z keeps J = v rho at the grid scale. The projection and the local z
  step change only the longitudinal current: with the projection a drifting plasma is unstable (streaming-plasma test
  0.78 of K_e at gamma 10, binomial 2.2e-10; bench/nci/README.md). Tests: R25 (D J' = T D J 8e-16; J_z of a uniform column
  x cos^2(k dz / 2) per pass, 6e-16), R27 (window, below).
- **Moving window with the filter (decision 91).** At each shift the last Gauss cell becomes an inner one, so T^p changes on
  the last p + 1 Gauss cells: the Gauss residual changes by -(T_new^p - T_old^p) rho there unless rho is that of the
  entering plasma alone. The entry columns are now frozen for 3 + filter_passes cells (window_freeze_cells), and the charge
  of the entering plasma is a background: a column that leaves the frozen cells has its charge, filtered as rho
  (S of decision 79, T), subtracted from gauss0; the diagnostics add the filtered charge of the columns still frozen.
  B7 with warm plasma (u_th = 0.01, Nm = 32) had 3e-4 of max rho in every mode before; test R27 (warm, electrons 2 and
  ions 1 per cell in z, a ramp, a laser wake, 2 passes): iRIP + projection 3.2e-15, RIP-CC + binomial 4.0e-15.
- `numerics.filter_passes` (default 0) passes, after the deposit (J) and in rz_rho (rho, output), each two
  charge-conserving steps S = 1 + a G D on J, T = 1 + a D G on rho (D the continuity divergence on the Gauss points,
  G = -D^T, D S = T D):
  - z: J_z,i += (dz/4)(w_i - w_{i-1}), w = D J = (J_z,i+1 - J_z,i)/dz + D_m J_perp,i on the Gauss cells (J_r, J_phi at
    the cell, as rho), on the columns between two Gauss cells; rho: the binomial over the Gauss cells, Neumann at
    their ends (open edges);
  - r, per mode: rho, J_z <- (1 + a_m D_m G_m) X and (J_r, J_phi) <- (1 + a_m G_m D_m) V, D_m the radial divergence
    (nodes from half nodes and nodes), G_m its negative adjoint, a_m = 1/lambda_max,TM(m), slice by slice in z.
  Continuity per mode exact, Gauss law at round-off on any boundary; a uniform density unchanged; only the
  longitudinal part of J is changed. m >= 1: the axis node is not a degree of freedom (left as it is). The scratch
  is the transport arrays of the field step.
- Moving window: the first filter_passes cells at the left edge are not Gauss points of the diagnostics (shifted
  from the interior, they hold charge filtered as interior; frozen there, it leaves with them); at the right edge
  exact while the entering plasma is neutral and at rest.
- **Projection (`numerics.filter = projection`, the default from decision 70 to 91).** The local steps above keep D S = T D,
  but "only the longitudinal part of J" does not hold: the z step changes J_z alone (as the local Cart3D filter of
  decision 64) and the r step does not combine into a gradient either, so the change of J radiates. The projection keeps
  T for rho and replaces S: per pass w = D J, r = (T - 1) w (T = T_r T_z, as rz_rho), L psi = r, J <- J + G psi with
  G = (dz, mu_z^2 G_m) the gradient of the RZ Gauss law (dz Ez + mu_z D_m E_perp = rho) and L = D G = dzz + mu_z^2 D_m G_m,
  mu_z the (1, 2, 1)/4 average along z. Then D J' = T D J exactly, and G psi only moves charge.
  - L psi = r: per mode the TM eigenbasis of D_m G_m (iRIP's radial transform, eigenvalues -lambda), per radial eigenmode
    s (psi_i-1 + psi_i+1) + b psi_i = dz^2 r_i with s = 1 - dz^2 lambda/4, b = -2 - dz^2 lambda/2 (|b| > 2|s|, also for
    iRIP's large lambda), the line solver of src/xlines.hpp (cyclic for periodic z, the partition method for ranks along
    z; the Re and Im slots of a mode are one complex line). No singular mode: the wall makes D_m G_m definite.
  - Open z: the Gauss cells [0, N - 1); psi reflected at both ends (psi_-1 = psi_0, psi_N-1 = psi_N-2), J_z on the faces
    0 and N - 1 unchanged, (J_r, J_phi) changed on the Gauss cells only.
  - Moving window: the projection is an elliptic solve along z, so its change of J (the Coulomb field of the filter's
    charge rearrangement) reaches ahead of the pulse at a level that falls with the lowest radial eigenvalue. In R23's
    window case (R = 0.8 um, the pulse meeting the plasma at the front edge) the entering cells are then not quite empty:
    a Gauss residual of 1.2e-13 for ~80 steps, which leaves with the window (local filter 2e-15).
  - Cost per pass: two radial transforms (nr^2 per column and mode, as iRIP) and the lines.
- Test R24 (radiation of the filter's change, as S6 of Cart3D): a random J (modes 0..2, periodic z, wall), one pass,
  dJ = S J - J fed alone into the field solver for one step, then 40 steps without current: B^2 / (E^2 + B^2) =
  1.1e-31 for the projection (RIP-CC and iRIP), 0.17 / 0.15 for the local filter, 0.34 / 0.33 for the raw J; a random
  gradient (dz, mu_z^2 G_m) psi 5e-32 / 7e-32, the plain (dz, G_m) psi 1.9e-2.
- Test R25: D J' = T D J for a random J, periodic and open z, uniform and stretched radial grid, 1 and 2 passes: 5.5e-15.
- Test R23 (2 passes, the projection): a hot plasma column of modes 0..2 run by iRIP and RIP-CC, shapes 1 and 2: Gauss
  law and div B per mode at round-off (1e-16); plasma drifting out through an open edge (3e-17; the first version, a plain
  binomial along z, 2e-2); a laser wake with the moving window and plasma entering and leaving (RIP-CC, projection
  1.5e-13, local filter 3e-15); a uniform rho kept to 4e-15. `tests/mpi.sh`: rz_plasma with 2 passes on 1 and NP ranks
  along z, the projection with iRIP and RIP-CC, the local filter with iRIP.

## Transverse particle shape (2026-11-06, decision 79)
`numerics.rz_transverse_shape = none` (default) | `gaussian`, `rz_shape_width = 1` (in units of dr at the axis), src/rz_smooth.cpp.
The default is `none` (decision 79): the Gaussian also damps the harmonics of CHF (low-band peak -11 % at 0.5 dr, -39 % at
1 dr, lambda0/64, bench/noise_heating), while more particles keep them. A run warns when a mobile species has ppc_theta
< 2 (2M - 1). The Gaussian suits runs whose physics lives at small k_perp (wakes, beams).
- Why: with M modes a macro-particle is a truncated Fourier series in phi, and ppc_theta particles sample 2M - 1 azimuthal
  components; the noise heating of a hot plasma grows with the modes (bench/noise_heating: 4 modes heat ten times the 2D
  rate at the same ppc, 1 mode as 2D).
- S = exp(a^2/2 Lap_perp) per mode, from the field solver's operators: scalars at the nodes (J_z, E_z, rho) by f(L_TM) =
  exp(-lambda a^2/2) in iRIP's TM basis; vectors (J_perp, E_perp: V_r at half nodes, V_phi at nodes) split into gradient
  and curl parts, V + G_m k(L_TM) D_m V - C_m k(L_TE) C_m^T V, k = (1 - f) / lambda (C_m: the curl of a TE scalar, L_TE =
  C_m^T C_m). D_m C_m = 0 and C_m^T G_m = 0 on the grid, so D_m S = f(L_TM) D_m: S J deposits f rho, the continuity per
  mode stays exact (rho of the diagnostics gets f too). S is self-adjoint in the field energy, so the gather of S E and the
  deposit of S J exchange the same energy. B is gathered as it is (no work). A uniform field and the laser (k_perp ~ 0)
  are untouched; a conducting wall is a Dirichlet end for J_z, E_z (they fall to 0 within ~a of it).
- Applied: S E before the push (sez_, ser_, sep_), S J after the deposit (before the current filter), f rho in rz_rho_all
  and the openPMD rho. Per column and mode, so any z decomposition; two radial transforms per scalar, about +30 % run time
  in the noise test.
- Price: at transverse wavenumber k_perp the coupling is scaled by exp(-k_perp^2 a^2/2), at dr = lambda0/40:

| harmonic, angle | k_perp dr | f, a = 0.5 dr | f, a = 1 dr |
|---|---|---|---|
| 5, 30 deg | 0.39 | 0.98 | 0.93 |
| 10, 15 deg | 0.41 | 0.98 | 0.92 |
| 10, 30 deg | 0.79 | 0.93 | 0.73 |
| 20, 30 deg | 1.57 | 0.73 | 0.29 |

- Tests (tests/test_rz_shape.cpp, `build/tests test_rz_shape`): against the heat kernel of the plane, exp(t Lap)[(x + i y)^m
  exp(-r^2/w^2)] = (w^2/(w^2 + 4t))^(1+m) (x + i y)^m exp(-r^2/(w^2 + 4t)), t = a^2/2: scalars of modes 0-3 to 1.1-1.9e-3
  at w = 8 dr, a = 2 dr, 2.9-5.0e-4 at dr/2 (second order); the vector field E_x (mode 1) to 4.7e-3 -> 1.7e-3, first order
  in the two cells next to the axis (the axis term of the radial operators), second order beyond; self-adjoint to 1e-17.
  Heating (bench/noise_heating): the hot column of 4 modes, 1.0e-3 per 10 fs without, |G| <= 7e-5 with a = 1 dr.
  The thermal many-mode test B7 (10 modes) reaches its plateau by step 150 instead of 50 (the smoothed noise starts lower);
  flat to step 850.

## Tests (planned)
- Fast: guided modes TM_0n, TE_1n against J_m roots (dispersion to the scheme's law); Gauss law per mode
  with a hot plasma (round-off); div B per mode; Langmuir; energy in vacuum (iRIP exact).
- Benchmark: LWFA in RZ (M = 2) against Cart3D and FBPIC, done (step 3b, bench/lwfa_fbpic).

## References
- D. Davidson et al., J. Comput. Phys. 281, 1063 (2015): charge-conserving deposition with azimuthal modes.
- R. Lehe et al., Comput. Phys. Commun. 203, 66 (2016): FBPIC, quasi-3D.
- A. F. Lifschitz et al., J. Comput. Phys. 228, 1803 (2009): quasi-3D PIC.
- T. Zh. Esirkepov, Comput. Phys. Commun. 135, 144 (2001).

## The first-cell current of mode 1 (numerics.rz_axis_current, decision 112)
For m = 1 the transverse current has a regular part J_+ = J_r - i J_phi (~ r^0; a uniform J_x has F_r1 = J_x/2, F_phi1 = i J_x/2 in
VLPL's F = F_0 + 2 Re sum F_m e^{i m phi}) and a part J_- = J_r + i J_phi that vanishes as r^2. The deposit (exact for point
particles: a straight passage through the axis has J_- = J_+, the analytic value) fills J_- in the first cell (J_r at r_{1/2},
J_phi at r_1) at the size of J_+; a 3D grid cannot hold that sub-cell structure (3D trapped-bunch run: |J_-|/|J_+| = 0.04 at 0.5 dr,
0.10 at dr; RZ: 1.65 and 12.6). The TE operator of mode 1 has a bound state next to the axis (lambda dr^2 = 6.365 above the band edge
4, B_z 90 % in the first half cell; iRIP omega tau = 0.48 pi, RIP-CC 0.78 pi) whose electric pattern lives on exactly these two
values, so the first-cell J_- drives it: 99.8 % of its drive power (bunch-column run, step 900), and it rings as the line at 0.48 pi.
Any local first-cell rule that is exact for the regular m = 1 field keeps the bound state (first-cell volume 2 dr^2: 4.69, the
centrifugal factor 1 instead of 4: 4.26, both break the exactness by O(1)).
- `numerics.rz_axis_current = regular` (default): after the current exchange in FieldsRZ::step, J_r,0 += d, J_phi,1 += -i k d with
  k = rh_0 r_1 / (m_f A_1), d = -D / (1 + k), D = J_r,0 + i J_phi,1: the first-cell D is zero; the shift is divergence free (rho, the
  m = 1 continuity, the Gauss law unchanged) and keeps the transverse current moment; a uniform transverse current is unchanged. The
  push gather uses E' = P^T E in the first cell (s = c (B_0 E_r,0 + i k A_1 E_phi,1), E_r,0 -= s/B_0, E_phi,1 += i s/A_1, on gather
  copies; a uniform field unchanged), so that the work of the projected current equals the work on the particles: without it the dense
  near-axis plasma test R-AX heated more (W_kin(15 fs)/W_kin(5 fs) matched 1.76, energy 1.67; main 1.16, 1.11; with it 1.26, 1.15).
  Field ionization gathers the raw E. Not with numerics.mover = implicit (decision 82): its fixed-point iteration did not converge with
  the projection (m = 0, 1 hot column: 30 of 30 steps unconverged against 2-7 raw); it keeps the raw current (open).
- `raw`: the deposit as before.
- Results (cheap trapped-bunch model, nci/rzm1, steps 900-1500, r = 0.4 - 1.6 um): the 0.48 pi line share of the near-axis E_phi m1
  81 % (raw) -> 0.2 % (regular), its power 2000 - 4000 x lower, the drive of the bound state 70 x lower in amplitude; E_phi m1 5.2e21
  against 4.5e22 raw and 4.7e21 in 3D (slow part 4.0e21 / 4.2e21 3D); J_r,t m1 3.7e27 against 1.75e29 raw; the m = 0 wake unchanged.
  LWFA (examples/rz_lwfa.deck, a0 = 2.5): laser peak within 0.4 %, near-axis E_r m1 within 0.8 %, field energy within 6e-7 of raw.
- Tests: A6 (tests/test_rz_axis.cpp: straight passages b = 0 .. 2 dr, shapes 1, 2, phase, chord: first-cell D / J_+ 3e-16, m = 1
  continuity 1.5e-14, moment 5e-15; a uniform current no worse than raw), R-AX, the mover tests.

## Axis values of the modes m >= 1 (2026-11-21, Alexander: "m = 1 Ez and Jz must be physically correct everywhere")
E_z, B_z, J_z, rho of mode m vanish as r^m on the axis, the transverse components of m = 1 are finite (a uniform transverse vector:
V_phi1 = i V_r1), those of m >= 2 vanish. In the code: E_z of m >= 1 is held at zero on the axis node (as before), J_z of m >= 1 is set to
zero there after the deposit (before and after the filter, which spread the leftover of the folded deposit into the first node) and in
FieldsRZ::step (rz_axis_zero_jz; not in the implicit mover's trial step, whose fixed point compares its own deposit), rho of m >= 1 has
no axis value. E_phi, B_r of m = 1 on the axis node are not dynamic (zero in the solver, r_0 = 0 removes them); the gather uses the regular
values E_phi1(0) = i E_r1(dr/2), B_r1(0) = -i B_phi1(dr/2). The openPMD output (not the checkpoints, which restart the solver state) writes
the physical axis values: E_z, J_z, rho of m >= 1 zero; E_phi, J_phi, B_r of m = 1 the regular values from the first half node (in the
thetaMode coefficients C_t(0) = S_r(dr/2), S_t(0) = -C_r(dr/2); B: C_r(0) = -S_t(dr/2), S_r(0) = C_t(dr/2)); m >= 2 zero. Test O-AX
(tests/test_openpmd.cpp).
