# Status

Updated: 2026-11-14 (Phase 3: binary collisions and impact ionization, decision 94; RZ metal depending on phi, block metal, and the axis cut numerics.rz_axis_cut, decision 103; structure runs in Cart3D: embedded metal, quadrupoles, openPMD beams, wake diagnostics, decision 102; Phase 1 exit benchmark: emittance of the Fig. 4 beam, `bench/emittance/`, decision 98; exact discrete steady-state self-field init for RIP-CC, the default, decision 99; LASY laser files in RZ, 2D and the boosted frame, decision 97; Breit-Wheeler pairs, decision 96; beam self-field clamped per mode, decision 95; Phase 4 started: spin-resolved QED, decision 93; current filter binomial by default and exact window entry, decision 91; field ionization in the boosted frame, decision 90; boosted-frame antenna, decision 89; boosted frame, decision 87; code renamed VLPL, formerly RIP-PIC, decision 88; Phase 3 started: field ionization, decision 86; O7 linear media and model = bound, decisions 84, 85; RZ embedded metal, decision 83). Phase 1 (production core) of `docs/ROADMAP.md`; Phase 0 done except multi-GPU.

## Works (tested)
- Agent skills (decision 119, 2026-11-24, branch agent-skill): `skills/vlpl-setup`, `skills/vlpl-simulation` (open Agent Skills
  format; docs/agent_skill.md, README "Agent skill"). After editing the README sections "Units" / "Deck" or a copied example run
  `sh tools/sync_skills.sh`; a new deck key that the skill should know is written as `block.key` in the skill files. Test
  tests/agent_skill.sh: static part in `make test`, the decks with the executable in `make agent_skill`.
- HDF5 1.12 or newer (decision 113, 2026-11-21): the Makefile reads the version from H5public.h; older: warning and a
  build without HDF5. Only the 1.12 API in src/h5.cpp. A parallel HDF5 gets the MPI paths of mpicxx in non-MPI builds.
  Test tests/hdf5_version.sh (make test); the CPU cluster: hdf5/1.14.6.2 (parallel, OpenMPI 5.0.10).
- NCI cures other than many binomial passes (decision 109, 2026-11-18, bench/nci/mechanism section 6): test switch
  numerics.filter_flat = L (maximally flat factor after the N passes, Gauss law exact, tests S9b, R25, mpi_fstream, mpi_rzff): at the
  noise level for gamma 10 with ~1e-6 loss at k0, but 100 x the proposal's growth at gamma 5 - 6; the force of the unstable modes is
  98 % E_x (matched gather ruled out); the implicit mover makes the NCI worse. Defaults unchanged.
- NCI mechanism of the boosted frame (decision 108, 2026-11-17, branch nci-mechanism, bench/nci/mechanism): the alias m = +-1 of the
  streaming plasma crosses the RIP / iRIP EM branch (measured frequency = alias to 1e-3 pi, 2D and RZ). Options
  `numerics.filter_compensate`, `numerics.filter_axes = x` (Gauss law exact, tests S9b, R25, mpi.sh); 8 passes + compensator along
  x: NCI at the noise level for gamma >= 10, 0.3 % less plasma response at k0 (two passes 1.9 %). The boosted injection must be
  exactly neutral per lattice group (decision 114: each pre-ionized ion species with its own electrons, same ppc; groups may differ
  in ppc and overlap; Config::check refuses otherwise; tests B7, B7b, B8: Gauss 3e-16 .. 5e-15 in 2D, 3D, RZ with filter 0 - 2); its plane
  is 3 + filter reach cells behind the edge. Defaults unchanged (Alexander decides).
- RZ axis, modes m >= 1 (decision 106, 2026-11-13, branch rz-axis-map, from the audit): the density-conserving radial map for
  m = 0 only; uniform transverse current next to the axis 1.0000 (was 0.833, shape 1), dipole of an offset charge exact (was
  1.45 x / 4.7 x); `numerics.rz_axis_map = all` gives the old deposit. Open: the m = 0 gather of non-uniform fields in the axis
  cell with rz_gather = matched (x 1.25 at dr/2 shape 1, x 1.59 shape 2; Alexander decides), shape 2 +8 % at the first face.
- RIP-CC field solver (explicit, X-dispersionless, exactly charge conserving), 3D Cartesian.
- Boris and Higuera-Cary pushers, shape order 1 and 2, Esirkepov deposition on the charge grid.
- Species: structure of arrays with an attribute registry (F64/F32/I64/I32), migration over all
  attributes.
- Species sorted by tile (stable; `numerics.tile`; `numerics.sort_every` = auto: every step only for a
  deposit that needs it, decision 20); CPU current deposit by
  coloured tiles without atomics (`numerics.deposit`, default auto = tiled on the CPU): bit-identical
  for any thread count.
- Step pipeline of operators (stages inject .. diagnostics), boundary hooks of the field step.
- MPI 3D domain decomposition (1 vs 8 ranks agree to 1e-11), OpenMP.
- CUDA build from the same source (compiles for sm_80/sm_90/sm_120). First real GPU run on 2026-10-26
  (RTX 5090, 1 GPU, no MPI): all 28 tests pass, streaming-plasma diag = CPU to round-off
  (`docs/GPU_TESTING.md`). Second check of main c7310e0 (tile sort, regression harness): all 32
  tests and all 3 regression cases pass on the GPU, GPU = CPU to <= 2e-12. Multi-GPU (MPI) not yet run.
- Exact particle currents at the open x boundaries (decision 21, `docs/specs/open_boundary.md`):
  the Gauss law stays at round-off while plasma leaves through either edge (before: 0.3 at the left).
- Moving window along +x (decision 22, `docs/specs/moving_window.md`, `examples/lwfa_window.deck`):
  plane pulse frozen exactly, Gauss law at round-off with cold and hot plasma entering and leaving
  at v = c, MPI 1 vs 8 ranks identical particle counts. The shift moves the start of each array
  (no copy) and cold plasma enters through kernels (decision 55): 0.41 ms per shift in
  `bench/lwfa_bubble` on the RTX 5090 (was 5.3 ms).
- Density profiles: `density_function` (deck expressions taken over from GEM-PIC-RZ, decision 23,
  `docs/specs/density_profile.md`), absolute coordinates, also for the plasma entering the window.
- Beam self-field at the start (decision 24, `docs/specs/beam_self_field.md`): the discrete boosted
  Poisson problem (DFT in y, z, tridiagonal in x, k_perp = 0 apart), Gauss law at round-off for any
  gamma, also at rest (Poisson initialisation); `t_start` loads a witness beam later in the window
  (`examples/lwfa_beam.deck`). The boost factor per mode is clamped at gamma h_x k_perp = 2 (decision 95; explicit
  RIP-CC only, decision 99): no field tail ahead of an ultrarelativistic beam, exact Gauss law when the window starts
  at t = 0.
  Since decision 99 a RIP-CC run starts with the exact discrete steady state of the RIP-CC step (per Fourier mode,
  `numerics.beam_init = steady`, the default; `gauss` keeps the Gauss-law field): stationary under the step to 6e-14
  (test B7). Decision 98: the self-field also with periodic x (3D DFT), with Yee (its own Gauss law) and iRIP, a
  co-moving neutralizing background in fully periodic boxes, RZ witness injection at t_start, `species.rigid_until`.
- Emittance benchmark (Phase 1 exit, `bench/emittance/`, decisions 98, 99): Fig. 4 of the 2020 paper (10 GeV,
  eps_n = 1 nm, 1 mm of vacuum). eps_n,y at 1 mm: RIP-CC with the steady init 0.9983 / 0.9982 nm (6 / 0.6 nC;
  0.9989 / 0.9988 at h/2), iRIP and iRZ 0.998-1.004 nm, Yee 2.4 um / 158 nm, FBPIC standard 574 / 54.5 nm, FBPIC
  Galilean 0.9993 / 1.000 nm. With the old Gauss-law init RIP-CC jumped once to 11-12.5 nm (6 nC). Witness in a linear
  wake (case B): eps_n 0.097 -> 0.105-0.111 um over the focusing phase in all VLPL solvers, FBPIC 0.119-0.123 um.
  Regression: test B6.
- LWFA benchmark against FBPIC (`bench/lwfa_fbpic/`, a0 = 1, 0.01 n_c, moving window, 300 fs):
  on-axis wake within 3.4-4.6 % (relative L2) and laser amplitude within 1-2 % of FBPIC (RZ, 2 modes).
- Laser injection through the open left x boundary; species slab profiles; x-y field slices;
  `tools/movie.py`.
- CPU push and deposit (decision 29, bit-identical): streaming deck, 1.05 M particles, best of 3:
  shape 2 12.5 -> 8.1 s on 1 thread, 5.2 -> 3.5 s on 4 (1.5x); shape 1 5.7 -> 4.9 s (1.16x);
  `make ARCH=-march=native` another 10-20 %. Moves of more than one cell are split into segments
  (decision 28, for iRIP). Every run prints the wall time per operator.
- Beam diagnostics (`diagnostic <name>` blocks, `docs/specs/beam_diagnostics.md`, decision 30):
  charge, mean and peak energy, rms spread, normalised emittances, sizes, the spectrum dQ/dE, and the
  selected particles (raw float64 per rank), all on the device with centred two-pass moments; MPI
  1 vs 8 ranks identical. The bubble benchmark compares the injected electrons with FBPIC.
- CPML absorbing layers in y and z (`grid.boundary_y/z = pml`, decision 31, `docs/specs/pml.md`):
  reflection 5e-6 at normal and 2e-5 at 45 degrees incidence with 16 cells, stable, Gauss law at
  round-off outside the layers, exact with the moving window, MPI 1 vs 8 ranks.
- Laser modes (decision 32, `docs/specs/laser_modes.md`): focus anywhere along x, Laguerre-Gauss
  (p, l), any number of lasers summed with their own polarisation and phase (circular polarisation).
- openPMD output and checkpoint/restart (decision 34, `docs/specs/openpmd.md`, HDF5 builds):
  - openPMD 1.1.0 files with the ED-PIC extension: E, B, J, rho, rho_<species>, and particles with all
    attributes. They pass the openPMD validator with 0 errors and 0 warnings and read with openPMD-api.
  - Checkpoints hold the whole state. `--restart latest|auto|file` continues bit-exactly on the CPU:
    1-3 threads and 4 ranks in the fast test, the diag.txt of 8 ranks byte-identical in tests/mpi.sh.
    Another number of ranks agrees to round-off.
  - `checkpoint.stop_after_hours` writes a checkpoint and stops before a batch time limit.
  - Fixed on the way: the CPML memory at the left face after a window shift (4 ranks along x were 6 %
    off), and two thread-order dependencies of the tiled deposit (leaving particles, species loaded
    later).
- Laser from a LASY file (decision 35, `docs/specs/laser_file.md`, HDF5 builds): cartesian and thetaMode
  envelopes (any azimuthal modes), E or normalized vector potential, any complex polarization. A file at
  another plane is propagated exactly to the entrance plane, which keeps spatio-temporal effects of
  the bandwidth. Checked against the built-in Gaussian and LG modes (1e-3, the file grid's
  interpolation), a spectrally resolved paraxial beam (1.6e-3 = 1/(kW)²), and real LASY 0.7 files.
  Also RZ (the file kept as azimuthal modes, exact Hankel propagation per mode), 2D (slab files) and the boosted frame
  (through the lab-fixed antenna), decision 97: L8-L10, file = built-in laser to 5e-4 .. 2.5e-3 (RZ / 2D, lab / boosted).
- Lab particle snapshots with a recording cut (`lab_diagnostics.select_uz`, `select_r`, decision 97): without it every
  particle of the listed species is recorded and gathered on every rank.
- Moving window cost (decision 36): a shift is a flat copy plus an exchange along x, and cold plasma
  enters without a random generator. The window operator went from 16 % to 2 % of the lwfa deck on the
  CPU (6.9 -> 0.7 s), bit-identical. It had been 60 % on the GPU (fourteenth check); the GPU number
  after the fix is in the fifteenth check.
- Yee solver as the reference (decision 37, `docs/specs/yee.md`, `numerics.solver = yee`, periodic
  boxes): exact numerical dispersion, Gauss law and div B to round-off, the same particle kernels
  (descriptor `Yee`). Numerical Cherenkov benchmark against RIP-CC in `bench/nci/`.
- iRIP, box closure (decision 38, `docs/specs/implicit_rip.md`, `numerics.solver = irip`): Cart3D,
  periodic boxes, one rank or ranks along x (partitioned tridiagonal solve, decision 39; 1 vs 2, 3, 8
  ranks to 1e-11); open x, laser and moving window (decision 40), Gauss law and div B at round-off;
  CPML layers in y, z (decision 41: separable solve with a dense eigenbasis per pml axis; reflection
  2.6e-6 at h_y = dx/2, as RIP-CC's layer where both run). No Courant limit (tested at h_x/h_perp = 10), zero dispersion along x,
  tan^2(w dt/2) = tan^2(kx dx/2) + dx^2 lambda/4 to 6e-13, energy conserved to 3e-12, Gauss law and
  div B at round-off with particles moving several transverse cells per step.
- RZ with azimuthal modes (decisions 42-48, `docs/specs/rz_modes.md`, deck `grid.geometry = rz`: periodic
  or open z, laser, moving window, conducting wall or radial absorbing layer, openPMD thetaMode output;
  no checkpoints yet):
  fields per mode with RIP-CC and iRIP (guided TM01 / TE11 against the Bessel roots, iRIP energy to
  4e-14), particle gather / push / deposition per mode (Gauss law per mode and div B at round-off with
  a hot plasma in modes 0..2, also with moves over several radial cells; Langmuir within 1 %); the deck
  path with exact loading and the Gauss law at round-off while the wall absorbs particles; the laser
  (pulse energy to 8e-4 of the analytic value, paraxial on-axis fluence to 1e-3); shapes 1 and 2 (2 folded
  across the axis, Gauss law per mode at round-off); the moving window with
  a laser wake (Gauss law 2e-15, exact entering count); 1 vs 8 ranks; openPMD thetaMode output (a known
  field reconstructed from the file to 2e-16); LWFA against FBPIC 4.3-5.9 % (bench/lwfa_fbpic).
- RZ conductors depending on the azimuth (decision 103, `docs/specs/rz_metal_phi.md`, block `metal { function }`,
  2026-11-13): metal where f(x, y, z, r, phi) > 0 on the 2M - 1 angles of the modes; the phi derivative becomes the
  staggered difference (i m -> i m_eff: a cylindrical Yee grid in phi), the rings rules per point and angle; C4 shapes
  couple m and m +- 4 (a square pipe: `examples/rz_square_pipe.deck`). The code equals an independent numpy
  operator for a square guide and a four-ridge pipe to 3e-7, the staircase converges to the continuum at first order;
  axisymmetric masks = rings to 5e-15. `numerics.rz_axis_cut` (any RZ RIP-CC run): mode m held at zero inside
  r < m_eff dr / kappa, so dz ~ dr for many modes. RIP-CC only; MPI along z; CUDA build runs the tests on the RTX 5090.
- RZ embedded perfect conductors (decision 83, `docs/specs/rz_metal.md`, block `rings`, 2026-11-07): metal rings fixed in the lab
  frame as a staircase inside the RIP-CC step (zero inside, tangential E and normal B zero, total reflection at the z faces
  through the characteristics), particles inside removed; a tube = the conducting wall exactly, a disc cavity = the mirror
  periodic box to 5e-15, Gauss law off the metal surface at round-off with plasma (tests/test_rz_metal.cpp); RIP-CC only
  (iRIP needs one radial operator for all columns). Also the RZ beam self-field (`species.self_field`, open z) and the
  lab-plane probe `output.plane_z` / `axis_every` with `tools/rz_rings.py` (`examples/rz_disk_loaded.deck`). The RZ exchange no
  longer zeroes the guards of the mode axis (80 % of the step on large RZ grids, results unchanged).
  `rings.jacket` (2026-11-12, decision 83 amended): a solid tube r_out .. r_out + jacket over the stack closes the gaps: the
  closed corrugated pipe with a real open end at z_end (per-point metal rules, bit-identical without a jacket; M7, M3 with a
  jacket, tests/mpi.sh).
- Generic dielectrics (decision 100, 2026-11-24, branch generic-dielectric; Alexander: "dielectrics and metals have to be
  implemented in general"): RZ with the block metal (lines on the angles, P6-P8), eps from a deck expression (D7-D9), eps
  depending on phi (chi on the angles, D10-D11), and Cart3D / 2D (docs/specs/dielectric3d.md, E1-E4); tests/mpi.sh cases.
- RZ dielectric regions (decision 100, `docs/specs/rz_dielectric.md`, blocks `dielectric`, 2026-11-13): eps(r, z) rectangles in the
  explicit RIP-CC step as a time-centred polarization current (a tridiagonal line along z per radial row, partitioned across the
  ranks), stable and lossless for any eps, div D = rho at round-off, together with the rings (dielectric in the gaps), MPI and
  CUDA. Cut cells: harmonic along the normal, arithmetic along the interface. In a moving window: a 32-cell absorbing layer at the
  front edge and a div-D-consistent entry (the scheme's grid-scale branch outruns c in a dielectric). Tests D1-D5, tests/mpi.sh;
  bench/rz_dielectric/ (DWA wake vs the lined-tube theory to 1e-5 in k, Cherenkov cone 2.7e-3, thin-wall convergence second
  order).
- Track O, O2 (decision 56, `docs/specs/xflux.md`): the reflected and transmitted field at the open x edges
  (`output.xflux`, `xflux_every`; `tools/xflux.py` for harmonic spectra and profiles). Energy balance exact for
  plane waves (4e-13), O(dz^2) with transverse structure (1.6e-4 at lambda0/16); a cold overdense step
  reflects R = 1 with the Fresnel phase to 2e-3 rad (RIP-CC and iRIP).
- Track O benchmark 1 (`bench/rom_harmonics/`): 1D-like ROM harmonics at normal incidence (a0 = 20, 50 n_c, pre-plasma
  0.1 lambda0): the reflected spectrum follows omega^(-8/3) (exponent -2.79 / -2.95 at lambda0/200, /400), converged in dx.
- Track O, O1 (decision 57, `docs/specs/tfsf.md`): TFSF laser source (`laser.tfsf_x`, `angle`), the incident field from an
  auxiliary vacuum grid (any beam, RIP-CC and iRIP): no leakage (1e-15) for focused beams at 0 and 30 deg, periodic or with
  CPML in y, z. Not yet with a moving window or restarts. RZ (decision 68, 2026-11-03): `laser.tfsf_x` at normal incidence with
  a vacuum incident FieldsRZ (the laser through its left edge) and absorbing z layers (`grid.boundary_z = absorbing`), RIP-CC and
  iRIP: leakage 2e-15, total - incident 4e-15, the 1 um layer reflects 8.2e-4 of the peak (`build/tests test_rz_tfsf`,
  `examples/rz_tfsf.deck`, MPI along z). The laser through an open left edge is now inbound energy in xflux.txt (it was never
  counted: E_left = Enet_left); E_inj, column 12, the injected laser energy.
  The incident grid starts divergence-free on the grid (Ex, Bx from the discrete constraints, decision 101): no static
  field stays where the pulse started (before: 7e-2 of the peak).
- Track O, O3 (decision 59, `docs/specs/x_absorber.md`): absorbing x boundaries (`grid.boundary_x = absorbing`), graded
  damping layers at both ends for RIP-CC and iRIP, the incident grid of the TFSF source with the same layers: the reflected field at 45 deg
  4.3e-4 of the peak with the default 4 lambda0 (3.9e-5 at 0 deg), no leakage across the TFSF plane (6e-16).
- Track O, oblique Fresnel reflection (decision 60, `docs/specs/fresnel.md`): a TFSF plane wave of one k_y
  (`laser.waist = 0`, periodic y) onto cold plasma steps, r(omega0) from the O2 records against the Fresnel formulas:
  s at 9 n_c phase error 1.1e-2 (lambda0/32) -> 6.6e-4 (/64) at 45 deg; p first order (0.13 -> 0.074 -> 0.040 rad at 45 deg):
  the one-cell transition layer of the deposited step (Drude-Lekner), 5-10x smaller with the surface between nodes;
  a resolved ramp under p needs shape >= 2 (shape 1: grid heating seeded by resonance absorption);
  underdense |r| to 1e-3 (0 deg) and 1.3 % (30 deg, second order in dy). TFSF lasers limited to 45 deg (Alexander);
  steeper ones enter from a side boundary (block `side`, decision 61).
- Side injection (decision 61, `docs/specs/side_injection.md`, block `side`): a TFSF tube with faces y, z = const inside
  the CPML layers, the incident field a sum of exact discrete plane waves (continuous or a pulse, any angle and elevation),
  RIP-CC and iRIP: leakage and total - incident 1e-14 at 60, 75.5 deg and through the corners; MPI and restart in
  `tests/mpi.sh` (`examples/side_wave.deck`). For lasers beyond 45 deg (decision 60).
- Flying focus through the side tube (decision 62, `side.wave = flying_focus`): the luminal Bessel pulse of GEM-PIC-RZ
  as exact discrete components, the corrections tabulated in the window moving at c: vacuum leakage 3e-15, total -
  incident 5e-15 (RIP-CC, iRIP); on the axis the analytic pulse to 2.4e-4 over 0.3 mm (`bench/flying_focus/`, GEM-PIC-RZ
  within its dispersion: 0.23 rms phase offset, 0.14 um envelope shift).
- Side tube in a background plasma (decision 63, `side.background_density`): the incident wave of the plasma (RIP-CC:
  TE and TM waves of their own k) and a linear cold fluid outside the kinetic region, so the TFSF stays exact: leakage
  and total - incident 1e-14 with the fluid everywhere (plane waves in 0.05 n_c, flying focus in 0.01 n_c, CPU and RTX
  5090); kinetic electrons + fluid shell 1.4e-2 of a weak pulse (vacuum shell 0.25); against GEM-PIC-RZ in
  `bench/flying_focus/` (1e18 cm^-3, 0.3 mm: laser 1.001 a0 and stationary, linear wake within ~10 %; at a0 = 2 the
  wake behind the cavity 35 % weaker than GEM-PIC-RZ's, not resolved). MPI and restart in `tests/mpi.sh`
  (`examples/side_plasma.deck`).
- Track O, O4 (decision 64, `docs/specs/solid_target.md`, `examples/solid_target.deck`): tilted slabs with exponential
  pre-plasma in front and behind (`slab_ramp`, `slab_rear_ramp`), loaded with exact sub-cell means (charge = analytic
  integral to 8e-14 at any tilt), mobile ions on the same lattice (neutral at t = 0), shape 3 (Cart3D; continuity, open
  edges, tiled and box deposits at round-off), the Gauss-law-exact current filter for Cart3D (`numerics.filter_passes`,
  longitudinal current only; RIP-CC and iRIP), omega_p dt checked at the loaded peak density. Since decision 69 the
  filter projects (`numerics.filter = projection`): its change of J is RIP-CC's gradient and radiates nothing (B 2e-31
  against 7e-2 for the old local filter, which raised the HHG orders 8-16 by 2-3.7; kept as `filter = local`).
- Current filter of every component (decision 91, `numerics.filter = binomial`, the default in Cart3D and RZ): J smoothed as
  rho, so a drifting plasma keeps J = v rho at the grid scale. Streaming-plasma NCI test (gamma 10, 282/wp): W_field / K_e
  1.3e-10 (projection 0.98, local 13.8, none 2.7e-9; FBPIC with filter 1.2e-9 standard, 1.7e-10 Galilean), all four
  variants; any boundary and decomposition. Projection and local stay for static targets (HHG, foils: set explicitly).
- Exact window entry (decision 91): the entering plasma is frozen for 3 + filter_passes cells and its charge kept as a
  background in gauss0: Gauss law at round-off with warm, non-neutral plasma entering, with and without the filter (W8, R27;
  before: 2e-7 in bench/lwfa_bubble, 3e-4 of max rho in B7 with warm plasma). The window start warning now needs
  c start + h_x / v <= L (the 1.05e-4 of bench/lwfa_bubble was the laser head cut at the edge one step before the first shift).
- TFSF plane with exact plane waves (decision 65, `laser.tfsf_incident = modes`, docs/specs/tfsf.md): the incident field in the
  two plane cells as exact discrete plane waves (angular spectrum of the Gaussian focus on the k_y, k_z lattice), no incident
  grid; leakage 1e-15 at 0-45 deg, RIP-CC and iRIP, the pulse may start outside the box and the plane may sit next to the
  left x absorber (the box of the oblique HHG benchmark 3.3 times smaller). Restarts work; periodic y, z only.
- Track O, TNSA benchmark set up (`bench/tnsa/`, VLPL against Smilei 5.1, 2D-like: 3 cells in z): C6+ foil 40 n_c with
  pre-plasma, H layer on the rear, a0 = 10 at normal incidence; the same proton dumps and cutoff E_cut(1e-3) in both codes.
  Code check (lambda0/32 x lambda0/16, 200 fs): E_cut(1e-3) 9.66 vs 9.67 MeV (-0.1 %), 3-5 % ahead at 100-150 fs.
  **Benchmark (2D, lambda0/64 x lambda0/32, 4 x 4 per cell, 400 fs): E_cut(1e-3) 17.65 vs 17.43 MeV (+1.3 %), Gate O
  passed**; all cutoffs within 2.1 %, E_p 2.2 %, spectra to 1e-4 per MeV; VLPL 12.7 core-hours, Smilei 8.0.
- 2D Cartesian geometry (decision 66, `docs/specs/cart2d.md`, `grid.geometry = cartesian2d`, `examples/foil_2d.deck`): the
  Cart3D code on one z cell with the z stride 0 (d/dz = 0 exactly in every field kernel), 2D particle kernels (shapes 1-3,
  J_z = q w v_z W_xy), rip_cc and irip, filter, absorbing x, CPML y, TFSF plane (grid or modes), moving window, MPI in x, y.
  Tests (`build/tests test_cart2d`): continuity per shape 1e-15 relative (also split moves), Gauss law and div B 1e-14 with hot plasma
  (rip_cc, irip), vacuum dispersion of (kx, ky) modes to 5e-14, Langmuir within 0.6 %, = the Cart3D slab of 3 z cells to 4e-11
  of the field peak and 5e-13 in the energies (foil + TFSF laser), TFSF modes leakage 6e-16, moving window Gauss law.
  TNSA code check (bench/tnsa, now 2D): 7.2 -> 0.53 core-hours on the same 2 nodes (13.5x), 0.41 on one node (Smilei
  0.27); E_cut(1e-3) 4 % above Smilei at 2 x 2 and 2 x 6 particles per cell (both codes +20 % from 2 x 2 to 2 x 6).
- MPI Gauss law with dense plasma (decision 58): the current ends where the stored position is (r1 = P - lo), so ranks
  with an offset and the moving window keep the Gauss law at round-off (4 n_c slab, 8 ranks: 2e-12 -> 2.5e-15;
  `tests/mpi.sh` case "dense slab").
- Track O, CHF diagnostic and benchmark set-up (decision 67, `docs/specs/harmonic_focus.md`, `bench/chf/`): block
  `harmonic_focus [<name>]`, a causal band-pass in time per node (Butterworth edges, bilinear transform) with the peak of
  |E_band|^2, its time and the fluence per node (RZ at phi = 0 and pi/2), Cart3D, 2D and RZ, MPI (1 vs 8 ranks: RZ exact,
  2D 6e-11); exact against the analytic filter response (3.7e-13) and against a host replay of the grid's field (2D 0, RZ
  1.9e-14); RZ vacuum beam: radial FWHM 1.768 um (analytic 1.766), on-axis fluence FWHM 17.76 um (2 z_R = 17.67).
  xflux.txt has the energy of the radial absorber, the field between the measuring planes and the particles that left.
  Code check (provisional physics: a0 = 10, 5 fs, 40 n_c parabolic mirror, F = 3 um): 2D orders 4.5 .. 10.5 converged at
  lambda0/128 .. /256 (peak 2.38 -> 2.34 a0^2, focus x 2.242 -> 2.244 um, spot 0.092 -> 0.097 um), orders 10.5 .. 20.5 not
  (+56 %); energy balance 5.5e-3, 1.55e-3, 3.1e-4 at lambda0/64, /128, /256 (second order); RZ lambda0/128, dr = lambda0/40:
  28.7 a0^2 at the focus, balance 2.0e-3 (laser through the open left edge); rerun with the RZ TFSF plane and z layers in
  bench/chf/README.md. Approved parameters (Alexander: a0 = 10, 5 fs, waist 2 um, 40 n_c, F = 3 um, fixed ions), runs of
  2026-11-03: 2D lambda0/512 converged in position, spot and fluence for both bands (high-band peak +15 % from /256),
  balance 1.5e-4; RZ dz = lambda0/256, dr = lambda0/80 not converged (low band 30.6 -> 37.9 a0^2, focus 0.09 um closer to
  the target), balance 1.1e-3 to 48 fs, then a late excess of 6e-3 that does not fall with dz (open); 3D lambda0/64 low
  band 33.8 a0^2 between RZ /128 and /256, balance 5.2e-3; 3D lambda0/128 prepared for JUPITER
  (bench/chf/jupiter_3d128.sbatch).
- Lorentz-boosted frame (decision 87, `docs/specs/boosted_frame.md`, blocks `boost`, `lab_diagnostics`, 2026-11-08; physics
  choices provisional, for Alexander's review): the deck stays in the lab frame; dx -> gamma (1 + beta) dx, lasers at the boosted
  entrance plane (Cart3D, 2D, RZ, the RZ TFSF plane), plasma at rest streaming in 3 cells behind the right edge (neutral loading,
  Gauss law exact), beams mapped ballistically with the self-field at the relative gamma; lab-frame field snapshots, particles
  and beam moments by back-transformation. tests/test_boost.cpp: vacuum laser boosted vs lab run 1.3 % (2D RIP-CC), 2.5 % (RZ
  iRIP); a0 = 1 wake at 0.005 n_c, gamma 2, lambda/32: peak Ez 1.9 % (2D), 0.5 % (RZ), laser 2.8 %, 1.1 %; Gauss law 2e-15 with
  plasma streaming through (2D, 3D, RZ); a ballistic gamma = 20 beam in the lab frame (charge 0.8 %, energy exact). Resolution
  rule: the plasma streams into the laser, so the boosted run needs ~(1 + beta) times finer cells (lambda/16: the wake 15-21 %
  too strong). No NCI with RIP-CC or iRIP at gamma 10 and 30 (bench/nci/README.md): no Galilean scheme. MPI (tests/mpi.sh), RTX
  5090. Lab-fixed antenna (`laser.injection = antenna`, decision 89, 2026-11-09, provisional): a moving polarization sheet with
  its charge (Gauss law 2e-15 with streaming plasma, 2D, 3D, RZ); a beam focused 70 um downstream that the entrance plane cuts
  (3.3 %, 4.4 % off the lab run): 0.73 % (2D), 0.44 % (RZ) at lambda/16, 0.1 % at lambda/32. Open: LASY files, the Cart3D
  TFSF plane. Checkpoints with the partial lab snapshots: decision 116 (2026-11-22).
- Structure runs in Cart3D (decision 102, `docs/specs/structure3d.md`, 2026-11-13): embedded perfect conductor
  from a deck function (block `metal`, staircase, plain and CPML step, moving window, MPI, GPU), external quadrupoles in the
  push (edge-averaged), beams from openPMD files (species `file`), wake potential probe, beam slices, side Poynting flux.
  Tests S1-S8 (tests/test_structure.cpp; rectangular pipe = discrete spectrum to 5e-13, round pipe cutoffs first order in h/R,
  exact x-face reflection, Gauss law 1e-14 with plasma and metal, FODO vs matrices 2e-5, import moments 5e-15, Poynting
  balance 3e-8, wake probe 6e-12), tests/mpi.sh case `structure`. An example bunch file imports with the moments of the
  python reference (Q 1000 pC, E 999.477 MeV, eps_n 1.0023 / 0.9995 um). Open: cut cells; RZ (`rings`) unchanged.
- Structure runs in RZ (decision 105, `docs/specs/structure3d.md` section 5, 2026-11-14, branch rz-structure): quadrupoles
  and quadrupole_lattice in the RZ push (B_x = -G y, B_y = -G x), openPMD beams in RZ, the wake probe off the axis (all modes)
  and beam slices. Tests R1-R4 (tests/test_rz_structure.cpp), tests/mpi.sh case `RZ structure`, example
  examples/rz_corrugated_pipe.deck against Bane-Stupakov (research/rz_structure/).
  Loss factor 8.09 vs 8.08 kV/pC/m, kick factor 2.31 vs 2.42 kV/pC/m/mm; open: the dipole reference k1 (decision 105) to be
  checked by Alexander, the wake far behind at longer structures.
- 157 fast tests and the 4 checks of test_hfocus (`make test`, `build/tests <name>` for a subset), regression harness (`make regression`,
  7 decks), MPI test (`tests/mpi.sh`, run locally before every pull request; the hosted CI job only on demand,
  decision 47), CI on the self-hosted runner (decision 33).

## Phase 0 checklist
- [x] Private repository AlexMPukhov/VLPL, BSD-3 licence
- [x] CI active (`.github/workflows/ci.yml`: CPU tests, MPI 1 vs 8 ranks, CUDA compile; since 2026-10-26)
- [x] CLAUDE.md, STATUS.md, decisions.md, spec template
- [x] First real GPU run (2026-10-26, RTX 5090 on the GPU workstation: tests and CPU-vs-GPU diag in
      `docs/GPU_TESTING.md`); first multi-GPU runs on JUPITER 2026-11-01 (1-16 GH200, all checks pass)
- [x] Architecture spec `docs/specs/architecture.md`: geometry (Cart3D / RZModes), solver interface
      with staggering descriptor, species attribute registry, tiles, step pipeline of operators
      (accepted 2026-10-26, PR #1; decisions 10-15)
- [x] Refactor to that architecture without changing results (all tests, laser deck diag identical)
      (steps 1-5: staggering descriptor, attribute registry, operator pipeline, geometry, tile sort)
- [x] Regression harness: stored reference diagnostics with tolerances (`tests/regression/`,
      `make regression`, in CI; `docs/specs/regression.md`)

## Working setup
- GPU and performance work (device kernels, the moving-window shift, migration and particle removal,
  sorting, deposit paths, reductions, transfers, `portable.hpp`, later multi-GPU MPI) is done by a
  Claude session with direct access to the CPU cluster / the GPU workstation (measure -> change -> measure). It records its
  numbers as dated sections in `docs/GPU_TESTING.md`. The cloud developer agent keeps physics features
  and specs; "Next GPU check" handovers are no longer needed.
- Git access: start a cloud session from the Code tab (claude.ai/code) with AlexMPukhov/VLPL as a
  source, or attach it in a running session; the session then clones, commits and pushes with git
  and opens pull requests. (The initial import went through the GitHub connector, one commit per file.)
- Alexander's laptop copy `~/code/RIP-PIC` (hp2) is a plain folder, not a clone; replace it with a
  clone when convenient.

## Next task
Phase 0 is complete, including the multi-GPU (MPI) run on JUPITER (Thomas Wilson, 2026-11-01, scripts
`bench/jupiter/`, results in its README): every correctness check passes (tests/mpi.sh 44 cases 1 vs 4 GPUs,
81 GPU diag files = CPU); one GH200 = 4.9x (3D bubble) / 8.5x (RZ) one CPU node; strong scaling 3D bubble
42 % on 4 and 21 % on 16 GPUs, weak 67-72 %; the losses are the guard / particle exchanges (field solve and
migrate), the push scales. Done since (branch gpu-exchange): guard exchanges with one pack kernel per call, both
directions in flight, pinned host staging or CUDA-aware MPI (`VLPL_CUDA_AWARE=1`), particle messages in both
directions at once, persistent device buffers in the migration, and the exchange time per run in the log
(`exchanges (max over ranks): ...`); tests/mpi.sh 1 vs 2 GPUs (RTX 5090, host staging) all cases pass; the
JUPITER 2026-11-03 (Thomas, merged main): both paths pass all checks (mpi.sh 53 cases 1 vs 4 GPUs, 95 diag
files = CPU); host staging 1.1-1.4x faster than on 2026-11-01, CUDA-aware MPI another 1.25-1.31x on the 3D
bubble (strong 66 % on 4, 37 % on 16 GPUs; weak 95 % / 92 %), RZ +6-11 % (bench/jupiter/README.md). Done since (branch
gpu-overlap, decision 92, docs/GPU_TESTING.md "Multi-GPU overlap"): the current guard exchanges overlap the push of
the interior particles (priority streams, `VLPL_OVERLAP=0` off), every halo in its own stream, the RIP-CC predictor
computed into the guards instead of exchanged, the migration on the device (scan, one pack and one fill kernel, one
message per axis for all species), the step profiler (`VLPL_PROFILE`); CPU results bit-identical to main, GPU 1 vs 2
ranks to round-off. Next: the JUPITER rerun (README "Note for Thomas"), then load balance (Phase 5). GPU deposit: the box version
(opt-in, correct on the RTX 5090 but slower than the atomics: FP64 shared-memory atomics are CAS
loops on sm_120 and sm_90, fifth check in `docs/GPU_TESTING.md`); `auto` keeps the atomics on the GPU
(0.6 s on the full streaming deck), sort_every = auto no longer sorts for them, and the GPU reductions
are block-wise (fixes of the fifth check). The tile sort runs on the device in the GPU build
(CUB radix sort, decision 25). Seventh check: correct, global sorted 2.5 -> 0.9 s, the rest was the
atomic per-tile counting, now replaced by starts from the sorted keys (eighth check handed over).
Eighth check: starts 1.8 us (was 121-223 us), the sort ~0.1 ms per step, global sorted 0.7 s; box
0.8-0.9 s, all of it the box kernel (1.43x the global push, its FP64 CAS loops). Now (decision 26):
the box sums fixed-point int64 with two native 32-bit shared atomics (no CAS in the SASS of sm_90 and
sm_120), exact in any order. Ninth check: exact on the GPU, box kernel 531 -> 444 us, still 1.20x the
global push (370 us). ptxas: the shape-1 box kernel takes 240 registers uncapped (one block of 256
per SM), so it is now capped with `__launch_bounds__` (decision 27: two blocks, 128 registers, no
spills; three blocks as an option); strays are deposited after the box kernel.
Box on the RTX 5090 is closed (tenth check: 433 us at best against 370 us for the global push, the
same with 1, 2 or 3 blocks per SM; `auto` stays global, box opt-in; revisit after a GH200 timing).
Phase 1 (production core, 3D LWFA) has started: exact open-boundary particle currents (PR #18,
decision 21), the moving window (decision 22, `examples/lwfa_window.deck`), density profiles
(decision 23) and the beam self-field with delayed loading (decision 24, external injection,
`examples/lwfa_beam.deck`). The first LWFA benchmark against FBPIC agrees (`bench/lwfa_fbpic/`; a
strongly nonlinear bubble is set up in `bench/lwfa_bubble/` (a0 = 4, 0.005 n_c, matched waist,
mobile ions in both codes, 1 ps; eleventh GPU check done). openPMD output with checkpoint/restart is done
(decision 34; GPU check 14 handed over). The laser from a LASY file is done (decision 35), and Yee as the reference solver (decision 37).
iRIP (decisions 38-41): Cart3D, ranks along x, periodic or open x with laser and window, y and z
periodic or with CPML. RZ with azimuthal modes (Phase 2): spec `docs/specs/rz_modes.md` with the
charge-conserving deposition per mode (decision 42), verified in numpy with both closures. Step 1 done:
fields and the RIP-CC / iRIP steps per mode (decision 43; guided modes against Bessel roots, iRIP energy
to 4e-14). Step 2 done: particle kernels (decision 44; gather, push, charge-conserving deposition per
mode: Gauss law per mode at round-off with RIP-CC and iRIP, Langmuir within 1 %). Step 2b done: the deck
path (`grid.geometry = rz`, decision 45; periodic z, conducting wall absorbing particles, plasma;
`examples/rz_plasma.deck`, MPI along z for RIP-CC). Step 3a done: open z, the laser projected on the
modes and the moving window (decision 46; `examples/rz_laser.deck`, `examples/rz_lwfa.deck`). Step 3b done:
openPMD thetaMode output and the LWFA benchmark (decision 48): RZ against FBPIC 4.3-5.9 % in the near-axis
wake (3D: 3.4-4.6 %), against VLPL-RIP 3D 0-5 % in the wake peak, 15x faster than 3D: the Phase 2 exit
criterion "RZ vs 3D agreement for LWFA" is met. Step 4a done: a radial absorbing layer (decision 49;
1e-4 of a broadband pulse left at 32 cells). Step 4b done: shape 2 folded across the axis (decision 50;
LWFA wake peak equal to 3D's to 0-1 %, axis noise of shape 1 mostly gone; tests/mpi.sh: 1 vs 8 ranks identical). Step 4c done: RZ checkpoints (decision 51; restart exact on one
thread, 8 ranks on 8 byte-identical, on 3 to round-off). Step 4 complete. Step 5 done: iRIP RZ over
ranks along z (the partition method of decision 39; 1 vs 8 ranks identical in diag.txt). With it the iRIP RZ part
of Phase 2 is done, so Phase O (overdense targets) may start (docs/ROADMAP.md). Open in Phase 2:
stretched transverse grids, the boosted frame, Yee in RZ. Step 6a done: the RZ fields on a stretched radial
grid (decision 52; energy to 2e-14, the Gate B reflection test: no reflection when k dr_max <~ 1, 12 % beyond
the cutoff). Step 6b done: particles on the stretched grid in the index coordinate (Gauss law 1e-16, the wake
on 60 stretched cells within 1.5 % of 144 uniform ones). Step 6c done: openPMD output (node radii as
rippic_r_nodes) and checkpoints on the stretched grid. Step 6 complete. Axis (B7 against FBPIC): the line of
excess density on the axis was the plain splines' axis share (4/3, 13/8): density-conserving deposition maps and
Gauss-point loading (decision 53; axis n_e ahead of the laser 1.000, FBPIC 1.007); the remaining axis noise is
sampling noise, removed to FBPIC's level by the charge-conserving filter `numerics.filter_passes` (decision 54);
the late energy excess of the RZ CHF runs (6e-3 after 48 fs) is noise heating of the laser-heated electrons (it falls with
the particles per cell and the filter, not with the cell size; bench/chf/README.md); since
decision 70 its change of J is the gradient of the RZ Gauss law and radiates nothing (B 1e-31 against 0.15-0.17 of the
change's field energy for the old local filter, kept as `numerics.filter = local`).
Side injection (decisions 61-63): next the continuum providers (rotated Gaussian, LASY files); a production flying focus
in a plasma channel (GEM-PIC-RZ ff_plasma.deck).
RZ noise heating (decision 79, 2026-11-06, bench/noise_heating): the late energy excess of the RZ CHF runs is noise heating
that grows with the number of modes at fixed ppc_theta (2M - 1 azimuthal components per ring of particles): 4 modes 1.0e-3
per 10 fs, 2D 1.0e-4. Default: no transverse shape and a warning below ppc_theta = 2 (2M - 1); the option
numerics.rz_transverse_shape = gaussian (width 1 dr) brings it below 7e-5 but damps harmonics. CHF at lambda0/64: the Gaussian damps the harmonics
(low-band peak -11 % at 0.5 dr, -39 % at 1 dr); ppc_theta 30 without it keeps the focus and halves the high-band noise.
CHF energy floor (decision 80, proposed): the rest of the RZ CHF excess is an axis instability of the dense target (a 40 n_c
1 keV plasma with k_p dr = 2.1 heats 18-fold in 29 fs in RZ, not in 2D). numerics.rz_gather = matched (exact for uniform
fields) halves the CHF floor (2.26e-2 -> 1.22e-2 at lambda0/64), energy reaches the 2D level (7.0e-3) but distorts smooth
fields next to the axis. Default since 2026-11-06: matched (Alexander). The slower growth that remains needs k_p dr <= 1.5
at the axis (refine dr there; a warning above), not the z or time centring.
Track O: CHF (Gate O 4): `harmonic_focus`, the RZ TFSF plane (decisions 67, 68) and the runs with the approved parameters in
bench/chf/README.md: 2D converged at lambda0/512, RZ not yet at lambda0/256 (its late energy excess of 6e-3 is noise heating
of the hot electrons), 3D not yet at lambda0/128 (JUPITER, 2026-11-04: low band 53.7 a0^2, +59 % from /64; balance 1.8e-3);
in RZ dr, the number of modes (4 -> 8: +14 %) and the solver (iRIP 17 % below RIP-CC at /128) all still change the focus;
no larger CHF runs for now (code development first, Alexander 2026-11-04); the code questions they raised: iRIP against RIP-CC
at the same grid, the noise heating of hot electrons, the load balance of a target in one slab.
iRIP against RIP-CC answered (bench/chf/README.md, 2026-11-06): the gap is a chromatic focal shift of the scheme.
The surface reflection and the generated energy agree between the solvers. Each harmonic order focuses
downstream of the true focus, by +0.04 / +0.09 / +0.21 um for orders 5 / 7 / 10 with RIP-CC and twice that with
iRIP. O5 removes the shift (<= 0.02 um). The CHF at /128 with open z edges gives 28.8 (RIP-CC), 24.5 (iRIP) and
35.3 a0^2 (iRIP+O5), the last near RIP-CC at /256 (37.9).
O5 with the RZ TFSF plane (decision 78, 2026-11-06): the incident grid runs o5, the Cz stencil across the plane gets the
incident far cell; leakage 3.1e-15 (1.0e-3 without that fix), a harmonic fed through the plane focuses as through the open
edge (2.658 um, reference 2.656 um), MPI along z = 1 rank. Production CHF runs with iRIP+O5 and the plane are possible.
Earlier: O4 done (decision 64, solid targets); next the oblique-incidence HHG benchmark in 3D against Smilei
(`bench/hhg_oblique/`), O5 (dispersion-corrected closure, `numerics.closure = o5`: 2D, decision 72; 3D with the transverse factor per mode, decision 73; RZ per radial eigenmode with the mode-weighted spacing, decision 74; CPML and the TFSF plane, decision 75; the side tube and the TFSF plane waves, decision 76; the cap of the transverse term for dy << dx, decision 77; the RZ TFSF plane, decision 78) and TNSA (bench/tnsa: code check done, benchmark runs next). Shape 3 and the filter with a moving window are open.
Next in Phase 2 (the boosted frame is done, decisions 87, 89 (lab-fixed antenna)):
Yee in RZ, stretched transverse grids in 3D (iRIP with CG).
Performance follow-ups: GEMMs for the dense transforms on the GPU (the mixed-radix FFT is done, decision 69).
The architecture refactor is complete: steps 1-4 bit-identical on the example decks, step 5 (tile
sort, `src/tiles.*`) equal to round-off, bit-identical with `numerics.sort_every = 0`.

## Settled for Phase 2 (research done, no code yet) — read before any iRIP / RZ work
- **iRIP = box (Crank-Nicolson) closure**, decision 17, spec `docs/specs/implicit_rip.md`. The
  characteristic-trapezoid iRIP of the earlier plan is rejected (transient energy growth up to ~420,
  only a modified Gauss law). The box variant conserves energy exactly, the standard Gauss law and
  div B to round-off (RipCC descriptor and Esirkepov reused), zero dispersion along x, stable for any
  number of azimuthal modes and omega_p dt < 2.
- Implicit system (I - D^2/4 mu_x^2 L) X = R couples x slices: transverse eigenbasis (RZ: per-mode dense
  transform, diagonalise once; 3D: FFT/DST, CG with PML) + batched tridiagonal along x, partitioned
  (SPIKE-type) across ranks.
- RZ axis rules (also for rip_cc in RZ): on-axis Ephi, Br are never dynamic DOFs (also m = 1).
- Python prototypes with all numbers: `research/irip_rz/` (README.md first). The "modified Gauss law"
  in `docs/specs/architecture.md` section 2 is superseded by decision 17.

## O6 implicit mover: implemented for Cart3D and 2D (2026-11-06, decision 81), RZ m = 0 (2026-11-07, decision 82)
- `numerics.mover = implicit` (Alexander chose option A of `docs/specs/implicit_mover.md`): Crank-Nicolson particles with the box
  closure's work field gathered by the transpose of the Esirkepov deposit; W + K conserved to `mover_tol`, Gauss law exact, no
  omega_p dt limit. irip with closure = box, shape 2 or 3, Cart3D and cartesian2d, periodic or open x. tests/test_mover.cpp (CPU and
  RTX 5090 alike): hot plasma at omega_p dt = 3 W + K to 1e-13 (24-34 iterations at tol 1e-12), Langmuir = Crank-Nicolson to 1e-4 up
  to omega_p dt = 10, 2D 40 n_c / 1 keV cell (dx/lD = 14): energy 2.5e-14 against +4.2 % explicit in 300 steps.
- RZ (2026-11-07, decision 82): modes = 1 or 2 (m = 0, 1), shape 2, with a C1 density-conserving radial map (dxi/dr 0.20 ... 3.5,
  continuous), the RZ work field per mode (rz_irip_trial, exact), a per-segment force from the exact work, J_phi of m = 0 from the
  angular momentum of the move, and a Newton step for particles within 2 dr of the axis. Tests M6 (work field, 3 modes, stretched
  grid: 0), M7 (omega_p dt = 2.5, uniform and stretched grids: W + K + wall loss 8e-15, Gauss 5e-14, 23-26 iterations, all steps
  converged); tests/mpi.sh RZ case. m = 1 accepted (Alexander): rare steps stall when a path passes within ~0.01 dr of the axis
  (2-5 of 30 in M7, energy error 5e-5 ... 4.8e-3 over 30 steps depending on the OpenMP order, Gauss exact); even m >= 2 rejected (no smooth deposit at the axis).
- Next: pml, TFSF, window, filter (adjoint in the gather); a better preconditioner for omega_p dt >~ 3 (O7's medium solve).

## O7 linear media without particles: implemented (2026-11-07, decision 84, branch o7-media)
- Blocks `medium [<name>]` (drude / lorentz, density or permittivity, collision_frequency, resonance, profile, supersample) for irip
  with closure = box: a Crank-Nicolson current per component at the current positions, driven by the box closure's work field (the
  medium's own transport source eliminated locally), Ex^{n+1}, Bx^{n+1} by restarted GMRES preconditioned by the vacuum solve.
  Cart3D, 2D, RZ (all modes), CPML (media clear of the layers), MPI, particles (explicit; implicit O6 mover in Cart3D). src/medium.*,
  src/medium_rz.cpp, src/medium_sim.cpp; tests/test_medium.cpp, tests/test_medium2.cpp; examples/plasma_mirror_2d.deck, rz_medium.deck.
- Numbers (docs/specs/linear_medium.md "C++ implementation"): W + K + dissipated to <= 8e-13 (random fields and media, omega_p dt <=
  30; RZ 3.7e-14), |r| = 1 to 7e-12 at omega_p dt = 0.5, 3, 20, oblique s-pol Fresnel / Snell / absorption second order (glass R
  error 8.5e-4 at lambda/40), Gauss law with particles 4e-15, the parabolic-mirror focus within lambda/20 of physical optics; MPI
  1 vs 8 ranks equal to all printed digits (2D mirror with CPML, RZ), 9.4e-13 with the implicit mover; RTX 5090: the same numbers,
  all medium tests pass (VLPL_MEDIUM_FAST); CUDA sm_90 and sm_120 compile.
- Open: p-pol on tilted surfaces (O7b, conformal weighting); TFSF, side tube, o5 with media (checkpoints and the moving window with media done 2026-11-08); O6 + media in RZ;
  GMRES memory (restart + 2 vectors of 2 N) and its iteration count at large omega_p dt (a better preconditioner).
- model = bound (2026-11-08, decision 85): the bound electrons of a gas (Rb D1 + D2 built in) as V-type Maxwell-Bloch atoms, an
  option of the medium block: Cayley step in the work field (exact energy exchange), pre-warped lines, coupling a / cos x and
  polarization cP cos(xbar) / cos x (Rabi frequency and weak-field chi exact at the line), Newton around the GMRES solve. Cart3D,
  2D, MPI, particles (explicit, O6). tests/test_bound.cpp: Rabi flopping 1.3e-5 at lambda/64 (node), grid vs analytic 5e-5; weak
  field vs Lorentz chi 6.8e-4 at lambda/40; W + K_b 1.7e-15 in saturation; Gauss 4e-15. examples/rb_gas_2d.deck. Extensions (2026-11-08, branch o7c-bound): relaxation, ADK ionization per level into
  electron-ion pairs, RZ (atoms on n_phi angles), checkpoints with all media (numbers in docs/specs/linear_medium.md). The solve's residual scale now has a floor of 1e-4 |F| (s-pol stalls).

## Phase 3, field ionization: implemented (2026-11-08, decision 86, branch field-ionization; physics choices provisional)
- Species keys element, charge_state, ionization = adk, ionization_electrons, max_charge_state, ionization_current, ionization_bsi
  = tong_lin (docs/specs/field_ionization.md): charge state per particle (I32 attribute, q Z in push, current and charge), ADK
  with l* = n* - 1 and l per level (Smilei's form), held at its maximum beyond barrier suppression, NIST ASD 5.12 energies of all
  charge states of H .. Lr (103 elements, tools/make_ionization_table.py) in data/ionization_energies.def; Monte Carlo with successive levels per step (exponential waiting times in E^n); one
  macro-electron per level at the ion with its velocity; the ionization current as the charge-conserving displacement of the
  electron to the tunnel exit (work = I_k w exactly, Gauss law at round-off); random numbers keyed by a position-hash id.
  Cart3D, 2D, RZ (all modes), MPI, CPU and GPU. src/ionization.*, src/atomic.*, tests/test_ionization.cpp,
  tools/ionization_ref.py, examples/ionization_2d.deck, rz_ionization.deck, bench/ionization_smilei/.
- Numbers (tests/test_ionization.cpp): Landau identity 6e-16, Python rates 3e-11; H in a constant field vs exp(-w t) 1.1 sigma,
  Ar0..4+ at w dt = 2.3 / 0.5 vs the rate equations 1.4 sigma, order independent; C0 -> C4+ after a Gaussian pulse vs Python
  1.2 sigma; field energy lost = work of the current (4.6e-4) = sum I_k w (7e-16), with mobile electrons the current takes
  0.994 of sum I_k w from W + K; Gauss law 1e-16 (Cart3D, 2D, RZ m = 0, 1); u = Z E t / m to 3e-16. tests/mpi.sh 1 vs 8
  ranks: ionization.txt identical (2D and RZ); GPU (RTX 5090) = CPU (test_ionization and the examples' ionization.txt);
  Smilei 5.1: C2+ / C3+ to 0.3 sigma, transmitted field to 1.2e-3 of the pulse (bench/ionization_smilei/).
- Boosted frame (decision 90, 2026-11-09, provisional): the ADK rate per ion from |E| in its rest frame (E, B and its momentum)
  over its proper time dt / gamma_i; the ionization current takes gamma_i I_k; the id hash mixes in the step of the first pass.
  tests/test_boost.cpp B4: charge states boosted (gamma 2) vs lab within 0.15 sigma (2D), 0.91 sigma (RZ), 0.40 sigma (2D,
  three levels per ion); Gauss law <= 3e-17; tests/mpi.sh 1 vs 8 ranks ionization.txt identical (2D) / 1.1e-13 (RZ); lab runs
  bit-identical to main.
- Open: the implicit mover (O6), self_field, ionization of the O7c Maxwell-Bloch medium, E_ion / E_cur across restarts. Binary
  collisions and impact ionization: next section (decision 94).

## Phase 3, binary collisions and impact ionization: implemented (2026-11-10, decision 94, branch collisions; physics choices provisional)
- Blocks `collisions <name>` (species1, species2, coulomb_log = 0 automatic | fixed, every, angle = nanbu | smilei5, ionizing,
  ionization_electrons; docs/specs/collisions.md): the relativistic binary Monte Carlo of Pérez et al. (2012) with Nanbu's angle
  distribution, Smilei 5.1's pairing and normalization (intra / inter species, repeated partners, Nanbu-Yonemura weights), the
  automatic Coulomb log (Lee-More form per pair: Debye length per cell, b_min = max(hbar / 2p*, classical); `quantum_bmin = h`
  for Smilei's h / 2p*) and the low-temperature cap (kept on by default, decided from the dense-plasma analysis in the spec);
  stage Migrate after the
  migration; cells = grid cells, RZ the annuli between the radial nodes. Impact ionization (Pérez / Smilei): RBEB over all
  orbitals (Smilei's neutral-atom orbitals, data/binding_energies.def, shifted to the NIST energies), several levels per step,
  new electrons appended reproducibly (two passes). `mobile = false` species of a collision block are frozen (benchmarks).
  Cart3D, 2D, RZ, MPI, CPU and GPU (CUDA sm_90 / sm_120 compile). src/collisions.*, src/collision_kernels.hpp, src/impact.*,
  tests/test_collisions.cpp, examples/collisions_2d.deck, rz_collisions.deck, bench/collisions_smilei/, dir/collisions.txt.
- Numbers (README "Tests", spec "Results"): conservation per pair 7e-14; <cos chi> = e^-s 1.2e-3; beam slowing on cold
  electrons 9e-4, beam scattering on ions 2.7e-3; Maxwell-Jüttner kept (KS 0.0026 < 0.0058) and reached from a Gaussian; e-i
  relaxation vs Spitzer converges with dt (0.098 -> 0.051); isotropization within 5-7 % of the bi-Maxwellian NRL rate in Cart3D,
  2D and RZ; ids: momenta bit-identical with the particles reversed; impact ionization of C vs n_e <sigma v> 0.3 %, H at
  100 eV 6.0e-17 cm^2 (measured 5.7e-17). Smilei 5.1 (bench/collisions_smilei): with the same angle law 0.4-4 %, stopping
  within 2 % of the NRL drag (Smilei 3-8 % below), impact ionization of Al3+ 0.9 %. Solid-density e-p relaxation
  (bench/collisions_epoch, lnL = 5): VLPL 488, EPOCH 4.20.1 Nanbu-Pérez 487 (full PIC), Smilei 513 /omega. With the
  physical (automatic) lnL: VLPL 937 vs GMS 865 and BPS 944. Automatic lnL vs the Lee-More formula in SI (C9): 4.5e-4.
  RTX 5090 (sm_120): test_collisions and both collision examples run.
- Open: species groups in one block, the boosted frame and the implicit mover with collisions, Higginson's energy-conserving
  weights, a degeneracy correction of lnL (T_e <~ T_F), electron-only screening for e-i pairs, tallies across restarts.

## Phase 4, spin-resolved QED: implemented (2026-11-09, decision 93, branch qed-spin; physics choices provisional)
- Species keys spin = sx sy sz (attributes s0 s1 s2), spin_anomaly; kind = photon; block qed (docs/specs/qed.md): T-BMT as an
  exact rotation per step for every species with spin; Monte-Carlo nonlinear inverse Compton emission (LCFA) resolved in the
  lepton spin and the photon Stokes vector (Algorithm 1 of the group's QV3D draft, the manuscripts' errors corrected), mixed
  states by default (pure optional), the spin after an emission and the non-radiative change, u = r^3 sampling, automatic
  sub-steps; photons into a photon species above a threshold or only tallied; deterministic P_alpha chi^2 g(chi) option.
  The push / move split of decision 15 (push_store, stage Qed, move + deposit with Pusher::None). Cart3D, 2D, RZ, MPI; the
  CUDA build compiles (sm_90, sm_120), not yet run on a GPU. src/qed.*, src/qed_tables.cpp, tests/test_qed.cpp,
  examples/qed_collision.deck, bench/qed_pwfa/.
- Numbers: README "Tests" (spectrum, spin dependence, Stokes vector, Sokolov-Ternov relaxation, energy loss, T-BMT, energy
  balance -2.4e-13); tests/mpi.sh 1 vs 8 ranks: qed_collision diag.txt and qed.txt.
- GPU: one RTX 5090 (sm_120, CUDA 13.0): the QED and pair unit tests pass and tests/mpi.sh on 2 ranks through MPS
  (docs/GPU_TESTING.md, "GPU-only failures on main after gpu-overlap"); several GPUs (JUPITER, sm_90) not yet run with QED.
- iRIP (2026-11-23, branch qed-irip): QED with numerics.solver = irip (explicit mover) gives the RIP-CC results at the same
  step, also with h_perp < c dt; the results converge in c dt = h_x for both solvers alike (qed_cascade: 0.3144 pairs per beam
  electron at lambda/20 for both, 0.3167 at lambda/40, 0.291 at lambda/10); P>1 = 0 in all runs. docs/specs/qed.md "Field
  solvers and the time step", tests/test_pairs.cpp P5b, tests/mpi.sh (the pair cascade with iRIP at 4 x the RIP-CC step, 1 vs 8
  ranks).
- Boosted frame (2026-11-23, decision 117, branch qed-boost, defaults accepted by Alexander): spin, emission and pairs, lab
  energies in qed.txt and the spectra, lab spins and photon polarization in the lab snapshots; qed_cascade lab vs gamma 2:
  0.3144 vs 0.3134 pairs per beam electron, gamma 5 at lambda/20 -19 %: the laser from the lab-fixed antenna
  is 6 % weak 2.7 um downstream of it (halving c dt' removes it; -1.5 % at lambda/80); mpi.sh 1 vs 8 ranks and
  restarts. docs/specs/qed.md "Boosted frame".
- Boosted beams: loading checks (2026-11-24, decision 118, branch qed-boost; Alexander: "1: yes"): error below one lattice plane
  across the beam along the boost axis, warning below 20, loaded vs deck charge (warning above 1 %); test B10.
- Open: the overlapped GPU push and the implicit mover with QED; tallies across restarts.

## Phase 4, Breit-Wheeler pairs: implemented (2026-11-10, decision 96, branch breit-wheeler; physics choices accepted by Alexander, 2026-11-11)
- qed.pair_electrons, qed.pair_positrons, qed.pair_selection (docs/specs/qed.md): the photons of qed.photons decay by Monte Carlo
  (LCFA) into an electron and a positron, resolved in the photon Stokes vector (rate Cg (B0 - xi3 K23)) and the pair spins
  (mixed states by default; formulas by crossing the emission rate of decision 93 and charge conjugation); the photon
  polarization changes without a decay (dichroism); threshold 2 m c^2; the pair shares the photon energy exactly, the field takes
  the momentum m c (1/2eps_+ + 1/2eps_-); own tables of T0, T3 in ln chi; sub-steps. The decayed photons are removed, the pairs
  appended (count, scan, fill; ids from the photon's stream: rank- and order-independent). Pair species are split species: they
  radiate and precess when listed in qed.species (cascades). Cart3D, 2D, RZ, MPI; CUDA compile check. src/qed.*,
  tests/test_pairs.cpp, examples/qed_cascade.deck, examples/qed_cascade_estimate.py.
- Numbers: README "Tests" (tables, Erber limits, polarization ratio, spectrum, selection, pair spins, cascade energy balance).
  Demo bench/qed_cascade/: 0.317 pairs per beam electron vs 0.322 estimated (first generation, photons polarized as emitted;
  0.361 unpolarized).
- Checks (the CPU cluster, 2026-11-10): full test suite and the 3 regression cases pass; runs without pairs bit-identical to main
  (stream, lwfa, beam, ion, rz, rzion, foil decks and the QED-emission deck: console, qed.txt, qed_spectrum.txt); tests/mpi.sh all
  4 segments pass, the pair cascade on 1 vs 8 ranks: qed.txt to 6.9e-17 of each column's maximum (pairs, their photons' energy
  and <S+.e2> equal to all printed digits), energies to 9.3e-13; CUDA build compiles (sm_90, sm_120).
- Transverse pair-spin sign corrected (2026-11-11): the first version had the positron antiparallel, the electron parallel to
  the rest-frame B'. Seipt & King (2020), Chen et al. (2022), Torgrimsson (2021) and an independent Volkov-state calculation
  (tools/volkov_spin.py; it reproduces VLPL's emission channels to 1e-3, i.e. Sokolov-Ternov) give the positron parallel, the
  electron antiparallel to B' (+0.6700 at chi = 1, x = 0.5 vs 0.6699 from the corrected formula); table in docs/specs/qed.md.
  Rates, yields and helicity terms unchanged. Demo rerun: 0.3174 pairs per beam electron, <S+.e2> = -0.380, <S-.e2-> = -0.377.
  Checks on the CPU cluster (2026-11-11): test suite all passed (P4 within 2.4 sigma, P4b <S+.B'> 0.6450 / 0.6447, <S-.B'> -0.6409 / -0.6410
  vs +-0.6427 in a magnetic / crossed field), regression 7/7; runs without pairs bit-identical to main (8 decks + qed.txt,
  qed_spectrum.txt); mpi.sh 4 segments exit 0 (QED pairs 1 vs 8 ranks 6.9e-17, <S+.e2> -0.3913 both); CUDA compile sm_90, sm_120.
- Open: vacuum birefringence; Bethe-Heitler and trident; pair spin
  correlations. Physics choices (mixed pair spins, selection on, energy exact with the momentum to the field, threshold
  2 m c^2 only, x from the full spectrum) accepted by Alexander, 2026-11-11.

## Known problems
- Beam self-field init (bench/emittance, decisions 98, 99): VLPL-RZ has no steady init yet (RZ starts with the
  Gauss-law field; the numbers with decision 95 are in bench/emittance/README.md). The steady init leaves a field tail
  ahead of the beam of 8.5e-11 of max|E| at u = 1e4 (window from t = 0: Gauss drift 2.7e-12; Gauss-law init exactly 0).
  Decision 95's clamp is applied for explicit RIP-CC only (with it iRIP / iRZ jumped 1 -> 12 nm in case A); the
  unclamped iRIP / iRZ field in a window from t = 0 has the zigzag tail up to the right edge (1.6e-4 of max|E| at
  u = 1e4), the Gauss law stays at round-off (2.6e-14 / 1.9e-14 over 12 shifts). A tail-free init for the implicit
  solvers is open (Alexander: "we have to find a solution for the implicit ones"). Yee in a wake needs a Yee moving window. Case B after dephasing: 3D stays at 0.13 um, RZ and FBPIC blow up to
  0.33-0.38 um (not investigated).
- RZ current filter with a moving window: the Gauss residual of m = 1 rises to 3e-11 (1e-10 of max|rho|) in the
  last Gauss cell at the front edge once plasma enters there with the laser's tail (B7; 2e-14 without the
  filter). It levels off after one transit of the box. Ruled out: the filter's z closure at that edge, stale guards.
- Open x boundary: the Gauss residual and max|div B| skip the last cell at an open right edge (not a
  Gauss point: it needs the fields beyond the edge). Leaving particles: fixed (decision 21, test 11b).
- GPU deposition: global atomics (`auto`); the box deposit is correct but 1.17x slower on the RTX
  5090 (tenth GPU check, decisions 26-27). Four halo exchanges per step (pinned host staging, or
  CUDA-aware MPI with `VLPL_CUDA_AWARE=1`), not overlapped with the push.
- Fixed after the third GPU check: `deposit = auto` picks the coloured tiles only with at least as many
  tiles per colour as threads (it was 4x slower than global on 32 x 16 x 16 cells at 20 threads); the
  tiled-deposit tests are skipped in the CUDA build (169 s there). Cost: on the full streaming deck at
  20 threads (16 tiles per colour) auto now takes global (-14 % of tiled lost); `deposit = tiled` forces it.
