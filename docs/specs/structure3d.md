# Structure runs in Cart3D: embedded metal, quadrupoles, beams from files, wake diagnostics

Status: implemented (decision 102), RIP-CC, Cart3D (and cartesian2d), CPU / GPU / MPI.

Motivation: a drive bunch in a metallic structure of arbitrary 3D shape, e.g. a corrugated pipe, with
external magnetic quadrupoles acting on the beam; this spec describes the code.

Coordinates: x is the propagation axis of VLPL (the window moves along +x), y and z are transverse. Files and lattices
written for z-propagating codes (GEM-PIC-RZ, elegant, openPMD beams) are mapped (x, y, z)_file -> (y, z, x)_VLPL.

## 1. Embedded perfect conductor (block `metal`)

### Physics
Perfect conductor: tangential E = 0, normal B = 0 on the surface, all fields 0 inside; the normal E and the tangential B
on the surface are the surface charge and current. The metal is fixed in the lab frame (it moves through a moving
window). Particles cross the metal (`metal.particles = pass`, the default) or are absorbed in it (`absorb`), decision 122.

### Discretisation (staircase, src/metal3d.*)
The transverse plane of RIP-CC is a 2D Yee lattice (src/grid.hpp): Ex on nodes (j, k), Ey / Bz on y edges (j + 1/2, k),
Ez / By on z edges (j, k + 1/2), Bx on faces (j + 1/2, k + 1/2); all at integer x.
- **Node mask.** M(i, j, k) = 1 where the deck function `metal(x, y, z) > 0` at the node (absolute coordinates; periodic
  axes wrapped). Evaluated on the device for the local array including the guards (no exchange needed), at the start and
  for the new columns after every window shift (the mask moves with the fields).
- **Metal points.** A y edge is metal if both its nodes are, a z edge likewise, a face if its four nodes are.
- **Rules**, applied after the three stages of the RIP-CC step (plain and CPML step alike):
  - Ey, Ez of a metal edge: 0 (tangential E).
  - Their partners Bz, By (same position): 0 if the edge is metal also at i - 1 and i + 1 (inside along x: then they
    are the normal B of a y or z surface, or inside). At an x face they take the characteristic that arrives from the
    vacuum side: left face (vacuum at i - 1) Bz = T+_y, By = -T-_z; right face Bz = -T-_y, By = T+_z. The outgoing
    characteristic is then minus the incoming one: total reflection, exact for normal incidence (the RZ rings rule,
    docs/specs/rz_metal.md). An edge without metal at i - 1 and i + 1 (one cell thick along x) cannot reflect from both
    sides; it is set to 0 (absorbing) and the setup warns.
  - Ex of a metal node: kept (from the corrector: the normal E of an x face, the surface charge) if the node is an x face
    and its four transverse neighbours are metal; 0 otherwise (tangential to a y or z surface, or inside). The predictor
    Ex* the same.
  - Bx of a metal face: 0 (normal B of an x face, or inside), and Bx*.
- **Charge conservation.** The RIP-CC Gauss identity holds for any values that enter the step; overriding the results at
  metal points changes the Gauss residual only at the rho points whose stencil holds such a point (the surface charge).
  The checks skip (i + 1/2, j, k) with any metal node in i..i+1, j-1..j+1, k-1..k+1 and the div B points with any metal
  node in i..i+1, j..j+1, k..k+1; every other point keeps its residual to round-off.
- **Particles** (`metal.particles`, decision 122). `pass` (default): nothing is done; the fields at the metal points are 0, so a
  particle inside moves on a straight line, and its charge and current there touch only skipped Gauss points. Entry and
  exit change the surface charge: transition radiation. Slow particles cross a wall as well (a real wall stops electrons
  below ~100 keV per 10 um of gold): use `absorb` where the wall is a sink. `absorb`: after the move (stage Migrate, before
  the migration) a particle whose box [floor(r - s), ceil(r + s)] (s = (order + 1)/2 cells per axis) holds only metal nodes
  is removed from the species (`remove_flagged`, src/species.cpp), its kinetic energy added to `Species::lost_energy`
  (photons: w |u|); also at the start (particles loaded inside the metal). Its current up to there was deposited and the
  field step conserves charge, so the conductor keeps the charge as surface charge (a floating conductor: the flux of E
  out of a box around it equals the absorbed charge, test S4b). The Gauss law at the checked points is exact in both modes.
- **Initial fields** (laser, beam self-field) are set to 0 at the forced points before the run starts. The self-field of a
  beam ignores the metal: start the beam in vacuum ahead of the structure.
- **Cost.** Three extra passes over the cells per step that read one mask value per cell (the rules read neighbours only
  at metal nodes). Memory: one more field array (the mask, float64).
- **Stability.** Unchanged (the metal only removes degrees of freedom; the face rule is a lossless reflection).
- **Not supported:** iRIP and Yee (rejected at the deck), RZ (use `rings`), cut cells (later: the staircase error of a
  curved wall is first order in h / R, test S2).

### Deck
```
metal {
  function = "max(abs(y - 2 mm), abs(z - 2 mm)) > 0.75 mm ? 1 : -1"   # > 0 inside the metal; x, y, z absolute, units allowed
}
```

## 2. External quadrupoles (blocks `quadrupole [<name>]`, `quadrupole_lattice`)

Lab-fixed hard-edged magnets acting on the particles only (not on the grid fields; the field of a magnet is static and
curl-free, and the grid would only carry it as a constant): B_y = -G (z - z_c), B_z = -G (y - y_c), so G > 0 focuses
electrons moving along +x in y and defocuses them in z (the GEM-PIC-RZ / elegant convention B_x = -G y, B_y = -G x
mapped to VLPL). Implemented in the gather of every Cart3D push kernel (src/particles.cpp move_one, push_store).

**Edges.** The kick of step n uses the field at x^n, but the particle crosses [x^n - v_x dt/2, x^n + v_x dt/2] during the
kick interval: G is averaged over that interval (the overlap with each magnet), so a magnet acts with its full length
wherever its edges fall between the particle positions (GEM-PIC-RZ PR #38). With the point value instead, an edge inside
a step changes the effective length by up to v_x dt.

```
quadrupole Q1 { x_start = 1 mm;  length = 9.07 cm;  gradient = 1210   # T/m; y_center, z_center: the axis (default box centre) }
quadrupole_lattice { file = lattice.txt;  x_offset = 0 }   # lines: x_start [m]  length [m]  gradient [T/m]  (# comments)
```

Not with the implicit mover or the boosted frame (rejected). RZ: section 5.

## 3. Beams from openPMD particle files (species key `file`)

Port of GEM-PIC-RZ src/beam_import.cpp (src/beam_import.*): `/data/<iteration>/<particlesPath>/<species>/` with
position + positionOffset, momentum (macroWeighted / weightingPower), weighting, as datasets or constant records, times
unitSI; charge and mass checked (electrons or positrons; the species takes the sign of the file). HDF5 builds only.
- Placement: X = x_center + (z - <z>_Q) (the charge centroid at x_center, so the time of the file does not matter),
  Y = y_center + x + y_offset, Z = z_center + y + z_offset (y_center, z_center: the axis, default the box centre);
  (u_x, u_y, u_z) = (u_z, u_x, u_y) of the file.
- Weights: w = N_real k0^3 / n_c. `total_charge` rescales them, `macroparticles` resamples systematically (probability
  ~ weight, equal weights, `resample_smoothing` x the rms of every coordinate as Gaussian jitter). Every rank reads the
  file and keeps its own particles (the resampling uses the same random numbers on all ranks).
- `self_field = true`: the field of the beam at its mean u_x (the RIP-CC steady state as for every beam,
  docs/specs/beam_self_field.md). The species gets no density (nothing enters with the window).
- The summary prints the moments of the file as imported (Q, E, dE/E, centroids, rms sizes, eps_n y, z).

```
species drive {
  file = bunch.h5;  file_species = beam;  file_iteration = 0   # defaults: the only species, the last iteration
  x_center = 3 mm   # required
  y_offset = 10 um;  total_charge = 1 nC;  macroparticles = 200000;  resample_smoothing = 0.05
  self_field = true;  rigid_until = 1 ns   # a rigid driver (decision 98)
}
```

## 4. Wake diagnostics (blocks `wake [<name>]`, `beam_slices [<name>]`, `side_flux`; src/wake.*)

- **wake**: at the probe (y, z) (nearest node; default the axis) the force per unit charge on a test particle with
  v = c along +x, F/q = (Ex, Ey - c Bz, Ez + c By), along the whole window; and its time integral at fixed
  xi = x - c t, V(xi) = int F c dt [V]: the energy gain per charge of a witness at xi. The wake potential per driver
  charge is W = -V / Q. Needs a moving window at v = 1 (RIP-CC, c dt = h_x: a window column is a fixed xi); accumulated
  from the first shift after `start`. With a rigid driver at an offset and the probe there, Vy, Vz are the dipole
  (transverse) wake potentials; probes at other offsets give the quadrupole / C4 parts. dir/wake_<name>.txt, one block
  every `every` steps and at the end: xi [um], Ex, Fy, Fz [V/m], Vx, Vy, Vz [V].
- **beam_slices**: slices of width `dx` (default h_x) along xi of one species: Q [pC], I [kA], <y>, <z>, sigma_y,
  sigma_z [um], <E_kin>, sigma_E [MeV], <u_y>, <u_z> -> dir/slices_<name>.txt every `every` steps.
- **side_flux**: the Poynting energy S = E x B leaving the box y_lo < y < y_hi, z_lo < z < z_hi (node planes; default
  2 cells inside the CPML layers or the edges) through its four side faces over the window length (S at the plane from
  the staggered fields, second order), per face and accumulated (trapezoidal in time), and the field energy inside ->
  dir/side_flux.txt [W, J]. With the absorber outside the box this is the energy that leaves the structure sideways (the
  mode escape through open gaps).

```
wake dipole { y = 2.01 mm;  z = 2 mm;  every = 500 }
beam_slices drive { species = drive;  dx = 5 um;  every = 500 }
side_flux { every = 10 }
```

## 5. RZ (decision 105; src/structure.cpp, src/rz_particles.cpp, src/wake.cpp)

The quadrupoles, the beam files, the wake probe and the beam slices also run in geometry = rz, where z is the transport
axis and the particle axes are (z, x, y) (X holds z, Y holds x, Z holds y). The deck keys are named in the RZ axes:

```
quadrupole Q1 { z_start = 1 mm;  length = 9.07 cm;  gradient = 1210;  x_center = 0;  y_center = 0 }   # default: the axis
quadrupole_lattice { file = lattice.txt;  z_offset = 0;  x_center = 0;  y_center = 0 }   # lines: z_start [m] length [m] gradient [T/m]
species drive { file = bunch.h5;  z_center = 3 mm;  x_offset = 50 um;  y_offset = 0;  self_field = true;  rigid_until = 1 ns }
wake dipole { x = 50 um;  y = 0;  every = 500 }   # the probe (x, y), default the axis
beam_slices drive { species = drive;  dx = 5 um;  every = 500 }
```

- Quadrupoles: B_x = -G (y - y_c), B_y = -G (x - x_c), G > 0 focuses electrons moving along +z in x (the GEM-PIC-RZ /
  elegant convention). The same table and edge average as Cart3D (external.hpp; its
  (x, y, z) are the RZ particle axes (z, x, y)), added to B after the gather in both RZ push kernels (rz_push_deposit,
  rz_push_store). Not with the implicit mover or the boosted frame.
- Beam files: the reader of section 3. The file's (x, y, z) are the RZ (x, y, z): z = z_center + (z_file - <z>_Q),
  x = x_file + x_offset, y = y_file + y_offset (the file's transverse origin on the axis); momenta unchanged. The particles
  are identical to those of the Cart3D import of the same file. Particles outside the radial grid (r >= r_nr) or the z box
  are dropped (warning). `self_field` solves the RZ steady field of the beam at its mean u_z (rz_add_beam_field), all modes.
- wake: the probe at (x, y), F/q = (E_z, E_x - c B_y, E_y + c B_x) with all modes summed at its phi (rz_wake_line in
  src/rz_particles.cpp: the plain linear map in r; every RZ component sits at the z nodes, so along z these are the node
  values); the integral at fixed xi = z - c t as in Cart3D. dir/wake_<name>.txt: xi [um], E_z, F_x, F_y [V/m],
  V_z, V_x, V_y [V]. MPI along z: every rank holds its columns.
- beam_slices: as Cart3D, with <x>, <y>, sigma_x, sigma_y, <u_x>, <u_y>.
- side_flux and the Cart3D metal block are Cart3D only (the RZ metal block is decision 103).

Example: `examples/rz_corrugated_pipe.deck` (a round corrugated pipe with resolved teeth, a bunch imported from a file,
wake probes on the axis and at the offset). Analysis: research/rz_structure/corrugated_wake.py, against Bane-Stupakov.
Result (a = 1 mm, h = 50 um, p = 40 um, g = 20 um, sigma_z = 100 um, 30 mm of pipe): loss factor 8.09 kV/pC/m against
Bane-Stupakov 8.08; dipole kick factor 2.31 kV/pC/m/mm against 2.42 with k1^2 = k0^2 + 2 / a^2 (the corrugated wall shorts
E_phi; decision 105) and 1.58 with the isotropic-impedance k1 = sqrt(2) k0; wavenumbers behind the bunch 5 - 10 % below
the leading-order values, amplitudes 0.96 / 0.98 of them.

Tests (tests/test_rz_structure.cpp, `build/tests test_rz_structure`):
- R1 Cold electrons through a FODO with the axis off r = 0 (edges between the grid points), RZ push with both pushers,
  against thick-lens matrices.
- R2 A beam written as openPMD in the test and imported in RZ:
  - (a) Q, the centroid at z_center + the offsets, eps_n x, y, sigma_z and <gamma> = the moments of the written particles;
  - (b) the particles = the Cart3D import of the same file, particle by particle;
  - (c) beam_slices: the sum of the slice charges = Q, the charge-weighted <x>, <y> = the centroid.
- R3 The probe off the axis against a backward plane wave E_x = -c B_y = g(z + t) (mode 1): V(xi) = sum 2 g(xi + 2 t_n) dt.
- R4 The deck keys in the RZ axes; side_flux rejected in RZ.
- MPI: tests/mpi.sh case `rz structure` (quadrupoles, axisymmetric metal, window, self-field beam, probes, slices; 1 vs N
  ranks along z).

## Tests (tests/test_structure.cpp, `build/tests test_structure`)
- S1 Rectangular pipe (walls on node planes, uniform in x): TM21 and TE11 with kx = 0 and 2 pi / L against the RIP-CC
  relation sin^2(w dt/2) = sin^2(kx dx/2) + dx^2 Lam/4 cos^2(kx dx/2) with the discrete transverse eigenvalue
  Lam = (2/hy)^2 sin^2(m pi hy/2a) + (2/hz)^2 sin^2(n pi hz/2b): exact.
- S2 Round pipe R (staircase), kx = 0: TM01 and TE11 cutoffs against j01 / R and j'11 / R at R / h = 8, 16, 32.
- S3 Normal incidence on the x faces of a slab, both polarizations, both faces: the mirror image to round-off.
- S4 Hot plasma in a corrugated tube, moving window, with and without CPML: Gauss law and div B at round-off, with
  particles = absorb and pass.
- S4b A cold bunch into an isolated plate: absorb: all removed, flux of E around the plate = the absorbed charge; pass:
  none lost, the plate neutral again, <u_x> kept.
- S5 Cold electrons through a FODO (edges between the grid points) against thick-lens matrices, second order in h.
- S6 A beam written as openPMD in the test (unitSI, positionOffset, macroWeighted, constant records) imported: Q,
  centroid, rms length, eps_n, <gamma> = the moments of the written particles; resampled within noise.
- S7 side_flux: a TE field radiating into CPML: energy inside + energy out = initial energy.
- S8 wake: a backward plane wave Ey = -Bz = g(x + t) in the window: V(xi) = sum 2 g(xi + 2 t_n) dt.
- MPI: tests/mpi.sh case `structure` (metal + quadrupoles + window + wake on 1 and 4 ranks).
