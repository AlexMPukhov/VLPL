# VLPL: capabilities, with the keys and one tested example each

For every capability: what it models, the deck blocks and keys, the example deck in `examples/` (all are run by
the repository's skill test), where it works, and what to check. Keys are explained in full in
`deck_reference.md`; the physics behind each module is in the repository under `docs/specs/<name>.md`.

| capability | block(s) | example | geometries |
|---|---|---|---|
| laser wakefield, lab frame | `laser`, `window`, `species` | `lwfa_plain_rz.deck`, `rz_lwfa.deck`, `lwfa_window.deck` | 3D, 2D, RZ |
| boosted frame | `boost`, `lab_diagnostics` | `rz_lwfa_boost.deck` | 3D, 2D, RZ |
| field ionization | `species` with `element`, `ionization` | `rz_ionization.deck` | 3D, 2D, RZ; boosted frame |
| collisions, impact ionization | `collisions` | `collisions_2d.deck` | 3D, 2D, RZ |
| QED, spin, pairs | `qed`, `species` with `spin`, `kind = photon` | `qed_collision.deck` | 3D, 2D, RZ; boosted frame |
| solid targets | `species` with `slab_*`, `laser.tfsf_x` | `solid_target.deck` | 3D, 2D |
| linear media without particles | `medium` | `plasma_mirror_2d.deck`, `rz_medium.deck` | 3D, 2D, RZ (implicit solver) |
| dielectric bodies | `dielectric` | `dwa.deck` | RZ (explicit solver) |
| metal bodies | `metal` (also `rings`) | `rz_corrugated_pipe.deck` | 3D, RZ (explicit solver) |
| beams, self-fields, beam files | `species` with `self_field`, `file` | `rz_corrugated_pipe.deck`, `dwa.deck` | 3D, RZ |
| restarts | `checkpoint`, `--restart` | any deck | all |
| MPI, GPU | build option | any deck | all |

## Laser wakefield (lab frame)

- **Laser:** `laser.a0`, `laser.polarization` (y | z; RZ: x | y), `laser.duration` (FWHM of the intensity),
  `laser.waist` (3D: also `laser.waist_z`; 0 = uniform in z, set it equal to `laser.waist` for a round pulse),
  `laser.focus` (x of the focus), `laser.delay` (time of the peak at x = 0, default 2 x duration). It enters
  through the open edge: `grid.boundary_x = open` (RZ `grid.boundary_z = open`). Several lasers: `laser <name>`
  blocks. Laguerre-Gauss: `laser.mode_p`, `laser.mode_l`. From a LASY file: `laser.file`.
- **Window:** `window.start` (>= 2 x `laser.delay`), `window.velocity = 1` (c).
- **Plasma:** an electron and an ion `species`, `species.density` in n_c, starting ahead of the box
  (`species.xmin`), a ramp by `species.density_function`. Fixed ions: `species.mobile = false` (not in a boosted frame).
- **Witness beam:** a `species` with `ux` (RZ `uz`), `self_field = true`, loaded later with `species.t_start`.
- **Diagnostics:** `diagnostic <name>` with `diagnostic.gamma_min` for the accelerated electrons; RZ:
  `output.axis_every` (E_z on the axis), `output.openpmd_every`; Cartesian: `output.slice_every`.
- **Check:** the laser amplitude after focusing against a0; the wake E_z against E0 = sqrt(n / n_c) in code units
  (linear: (a0^2 / 2) / sqrt(1 + a0^2 / 2) x E0 x a form factor <= 1 for a resonant pulse; blow-out: ~ sqrt(a0) E0).
  max|dGauss| at round-off while plasma enters and leaves.

## Boosted frame

- `boost.gamma`; everything else stays a lab-frame deck. Output in the lab frame: `lab_diagnostics.t_first`,
  `lab_diagnostics.t_every`, `lab_diagnostics.count`, `lab_diagnostics.species`, `lab_diagnostics.gamma_min`.
- Needs: implicit solver for gamma >~ 2, `numerics.shape = 2`, `numerics.filter_passes = 1`, neutral plasma
  pairs, lab cells <= lambda0 / 32, a moving window for long stages. Beams: the loading checks. Laser far from
  focus at the entrance: `laser.injection = antenna`, with the near-field distance d_1%. All in
  `physics_rules.md`, section 5.
- **Check:** the boost summary printed by `--check` (boosted cell, step, plasma streaming, d_1%, warnings); one
  short case against the lab run (`rz_lwfa.deck` is the lab twin of the example).

## Field ionization

- An atom species: `species.element` (H ... Lr), `species.charge_state` (start, default 0),
  `species.ionization = adk`, `species.ionization_electrons` (the species that receives the electrons; give it
  `density = 0`), `species.max_charge_state`, `species.density` (atoms, n_c). Tunnelling ADK per level with the
  NIST energies, Monte Carlo, several levels per step; `species.ionization_bsi = tong_lin` adds the
  barrier-suppression correction. The ionization current takes the ionization energy from the field.
- **Resolution:** the field oscillation must be resolved in time: lambda0 / 32 along the axis.
- **Output:** `ionization.txt`: energy taken, the conserved sum, the charge-state distribution.
- Ionization injection for LWFA = this plus the wakefield setup (a few percent of a high-Z gas).

## Collisions and impact ionization

- `collisions <name>` blocks: `collisions.species1`, `collisions.species2` (default = species1: intra-species),
  `collisions.coulomb_log` (0 = automatic), `collisions.every`. Relativistic binary collisions (Perez et al. 2012,
  Nanbu angles). `collisions.ionizing = true` with an ion species given by `species.element` and
  `species.charge_state` adds electron-impact ionization; new electrons go to `collisions.ionization_electrons`.
- **Resolution:** collisions act within a cell: cells of the order of the Debye length or the gradient scale;
  `species.ppc_x` etc. >= 4 per direction so that every cell has pairs.
- **Output:** `collisions.txt` (pairs, mean Coulomb logarithm, energy exchanged, temperatures per species).

## Strong-field QED and spin

- `qed.species` (the leptons), `qed.model` (monte_carlo | deterministic | none), `qed.photons` (a species with
  `species.kind = photon`), `qed.photon_min_energy` (MeV); spin: `species.spin` (three components, |S| <= 1),
  precession (T-BMT) and radiative polarization; pairs: `qed.pair_electrons`, `qed.pair_positrons`.
- Photon emission in the locally constant field approximation: valid for a0 >> 1; chi is printed in `qed.txt`.
- **Resolution:** the emission probability per step must stay small: the code sub-steps (`qed.substep_prob`); a
  counter-propagating beam needs its own `species.ppc_x`, and in a boosted frame gamma^2 (1 + beta)^2 more.
- **Output:** `qed.txt` (mean spin, chi, emitted energy, Stokes vector), `qed_spectrum.txt`, `qed_pairs.txt`.

## Solid targets (kinetic)

- Tilted foil with pre-plasma: `species.slab_x0`, `species.slab_y0`, `species.slab_angle`,
  `species.slab_thickness`, `species.slab_ramp` (exponential scale length), `species.slab_ramp_extent`. The loaded
  charge is the integral of the profile.
- Laser at an angle through a total-field / scattered-field plane inside the box: `laser.tfsf_x`,
  `laser.angle` (|angle| <= 45 deg), with `grid.boundary_x = absorbing` (`grid.x_absorber_width`).
- Numerics: `numerics.shape = 3` (3D; 2 in 2D), `numerics.filter_passes = 1`, `numerics.filter = projection`
  when the harmonics matter; omega_p dt at the peak density is checked (error above the stability limit).
- **Output:** `output.xflux = true` gives the energy through the x edges (`xflux.txt`: reflectivity);
  `output.xflux_every` records the reflected field for spectra (repository `tools/xflux.py`); `harmonic_focus`
  blocks give the focused harmonic field.
- Above ~100 n_c or when only the optical response matters: a `medium` block instead of particles.

## Linear media without particles (metals, dielectrics, lossy plasma)

Generic optical materials on the grid, with the implicit solver (`numerics.solver = irip`,
`numerics.closure = box`), any density (no omega_p dt limit), exact energy balance.

- `medium.model = drude`: free electrons: `medium.density` (omega_p^2 / omega_0^2 = n / n_c),
  `medium.collision_frequency` (nu / omega_0). A metal or an overdense plasma: density >> 1; a perfect mirror in
  the limit; a lossy conductor with a collision frequency.
- `medium.model = lorentz`: a dielectric with one resonance: `medium.permittivity` (the real eps at omega_0) and
  `medium.resonance` (omega_r / omega_0), or `medium.density` with `medium.resonance`.
- Shape: `medium.profile`, any expression of x, y, z (RZ: z, r) that is 1 (true) inside: slabs, curved mirrors,
  lenses, layers; sub-cell averaged (`medium.supersample`). Several `medium <name>` blocks.
- `medium.model = bound`: the bound electrons of a gas as Maxwell-Bloch atoms (resonant response, saturation, optional ionization).
- Works with particles in the same run, CPML, the moving window, MPI. Not with a TFSF plane, side injection,
  the o5 closure or a boosted frame. Keep media one cell clear of CPML layers.
- **Output:** `medium.txt`: energy in the medium, dissipated energy, the balance.

## Dielectric bodies with beams or fields (RZ, explicit solver)

- `dielectric <name>` blocks: a lossless dielectric of permittivity `dielectric.eps` in the rectangle
  `dielectric.r_in` .. `dielectric.r_out`, `dielectric.z_start` .. `dielectric.z_end` (fixed in the lab frame, also
  with the moving window); `dielectric.ramp` gives linear ends. Stable and lossless for any eps; div D = rho exact.
- Use: dielectric-lined tubes (dielectric wakefield acceleration, Cherenkov radiation of a bunch), windows, fibres.
- Particles must not cross a dielectric (a warning when they do).

## Metal bodies (perfect conductors; explicit solver)

- `metal.function`: perfect conductor wherever the expression of the lab coordinates is > 0: 3D f(x, y, z); RZ
  f(x, y, z, r, phi) (an axisymmetric shape with `metal.phi_derivative = spectral`; shapes that depend on phi
  couple the azimuthal modes and need enough `grid.modes`). Staircase on the grid, particles inside are absorbed.
  Use: pipes, irises, cavities, corrugated and tapered guides, apertures, collimators.
- `rings` (RZ): a shortcut for a periodic stack of metal rings, optionally closed by an outer jacket (a
  disk-loaded waveguide): `rings.r_in`, `rings.r_out`, `rings.period`, `rings.gap`, `rings.z_start`,
  `rings.z_end`, `rings.jacket`.
- The outer radial boundary `grid.boundary_r = wall` is itself a conducting tube at r = nr dr.
- **Resolution:** the smallest feature (tooth, gap) >= 10 cells; staircase errors of cutoff frequencies are first
  order in the cell (a few percent at 16 cells per radius).
- **Diagnostics for structures:** `wake <name>` probes (`wake.x`, `wake.y`, `wake.every`: the force on a test
  charge and its integral, the wake potential), `beam_slices`, `output.plane_z` (all field components at a lab
  plane every step), `output.axis_every`.

## Beams, self-fields, beam files, external magnets

- A beam is a `species` with a drift (`species.ux`; RZ `species.uz` = gamma beta) and a shape
  (`species.density_function`, e.g. `gauss(z - 1 mm, 10 um) * gauss(r, 5 um)`, and the bounds).
  `species.self_field = true` starts it with its steady self-field (Gauss law exact from step 0; needs an open
  transport boundary). `species.rigid_until` freezes its momenta (a rigid driver).
- From an openPMD particle file: `species.file`, placed with `species.z_center` (RZ) or `species.x_center` (3D);
  `scripts/make_bunch.py` writes a Gaussian bunch as such a file. Needs a build with HDF5.
- External quadrupoles act on the particles: `quadrupole <name>` (`quadrupole.gradient` in T/m,
  `quadrupole.length`, start position).

## Restarts

`checkpoint.every` (steps), `checkpoint.keep` (default 2), `checkpoint.stop_after_hours`; continue with
`vlpl deck --restart latest` (`auto`: or start fresh; or a file name). Needs HDF5. Same number of ranks: the
continuation is exact on the CPU; another number: round-off. The text logs continue.

## MPI and GPU

The same deck. MPI: `mpirun -np P vlpl deck`; decomposition automatic or `grid.mpi_x`, `grid.mpi_y`,
`grid.mpi_z`; implicit solver and RZ: along the transport axis only. GPU: a CUDA build, one GPU per rank.
Results agree with the single-rank CPU run to round-off. Details: skill `vlpl-setup`.

## Not in the code

Radiation reaction other than through the `qed` block; nuclear reactions; hydrodynamic (fluid) targets;
transport through thick matter. Reduced models for long underdense stages (envelope laser, quasi-static wakes)
are separate codes (`physics_rules.md`, section 6).
