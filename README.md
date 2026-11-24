# VLPL

*VLPL, formerly RIP-PIC (renamed 2026-11-08, decision 88). Variants by field solver: VLPL-RIP and VLPL-iRIP (3D / 2D with RIP-CC / iRIP), VLPL-RZ and VLPL-iRZ (quasi-3D RZ). Environment variables are VLPL_<X>; the old RIP_<X> names still work as deprecated aliases.*

Electromagnetic 3D particle-in-cell code built on the RIP ("rhombi in plane") Maxwell solver of
Pukhov, J. Comput. Phys. 418, 109622 (2020). It has no numerical dispersion for waves along x
(c Δt = h_x, so the transverse fields shift exactly one cell per step), and in this version it is
exactly charge conserving (RIP-CC, see below).

Status: field solver, pushers, charge-conserving deposition, MPI 3D decomposition, laser injection
through the open x boundary, a moving window (exact Gauss law while plasma enters and leaves), x-y
field slices and a movie script, CPU (OpenMP) and GPU (CUDA) from one
source, several GPUs with MPI (one per rank). Tested on one GPU (RTX 5090: all tests and the
regression harness = CPU to round-off); several GPUs with MPI not yet run: see docs/GPU_TESTING.md.
The scripts for the first multi-GPU run and the timings on JUPITER / JUWELS Booster: `bench/jupiter/`.
Numerical emittance of a 1 nm beam (Fig. 4 of JCP 418, 109622) and of a witness in a wake, all solvers and FBPIC:
`bench/emittance/`.

Plan: `docs/ROADMAP.md`; current state: `docs/STATUS.md`; conventions: `CLAUDE.md`, `docs/decisions.md`.
Licence: BSD-3 (`LICENSE`).

## Build, test, run

```
make                                   # build/vlpl and build/tests (C++17, OpenMP; HDF5 >= 1.12 if pkg-config finds it)
make HDF5_DIR=/opt/hdf5                # HDF5 elsewhere (openPMD output, checkpoints); HDF5=0: without
                                       # HDF5 1.12 or newer; an older one: warning, build without HDF5 (decision 113)
make test                              # 104 checks against exact or analytic references
make agent_skill                       # the agent skills against build/vlpl: every deck, restart (docs/agent_skill.md)
build/tests pml laser                  # only the test groups whose name contains pml or laser
make ARCH=-march=native BUILD=build-native   # CPU build for this machine (AVX2/AVX-512, FMA; round-off differs)
make MPI=1 BUILD=build-mpi             # MPI (mpicxx), 3D Cartesian domain decomposition
make CUDA=1 CUDA_ARCH=-arch=sm_90 BUILD=build-gpu              # one GPU (nvcc --extended-lambda)
make CUDA=1 MPI=1 CUDA_ARCH=-arch=sm_90 BUILD=build-gpu-mpi    # several GPUs, one per MPI rank
sh tests/mpi.sh build-mpi/vlpl       # 1 rank vs 8 ranks: same energies to 1e-11, Gauss law exact (NP=4: vs 4 ranks)
build/vlpl examples/streaming_plasma.deck --check      # summary only
build/vlpl examples/streaming_plasma.deck              # run, diagnostics in out_stream/diag.txt, time per operator at the end
mpirun -np 8 build-mpi/vlpl examples/streaming_plasma.deck grid.mpi_x=2   # override any key
build/vlpl deck checkpoint.every=1000 checkpoint.stop_after_hours=23.5     # checkpoints, stop before a time limit
build/vlpl deck --restart latest       # continue from the newest checkpoint in output.dir (auto: or start fresh)
build/vlpl examples/laser_reflection.deck              # laser on a tilted overdense layer (~8 min, 2 cores)
build/vlpl examples/lwfa_window.deck                   # laser wakefield in a moving window (3D, 14 min, 2 cores)
build/vlpl examples/lwfa_beam.deck                     # the same with an externally injected 200 MeV beam
build/vlpl examples/beam_vacuum.deck                   # a relativistic beam started with its self-field
python3 tools/movie.py out_reflect --fields Ez n_electrons    # -> out_reflect/movie.mp4 (matplotlib, ffmpeg)
```

Any key can be overridden on the command line as `block.key=value`. Unknown keys, missing units and
units of the wrong kind are errors (deck parser taken over from GEM-PIC-3D).

## Units

c = 1, lengths in 1/k0, times in 1/ω0 (λ0 = `grid.lambda0`), momenta u = p/(mc) = γv, fields in
m_e c ω0/e, densities in n_c, charge in e, mass in m_e. A macro-particle of weight w stands for
w n_c/k0³ particles, so ∇·E = ρ, ∂_t E = ∇×B − J, and ω_p² = n/n_c.

## Grid

All fields live at integer time levels and at integer x:

| quantity | position | note |
|---|---|---|
| Ex, Jx | (i, j, k) | |
| Ey, Bz | (i, j+½, k) | transverse pair co-located: Ey − v_x Bz is exact on the grid |
| Ez, By | (i, j, k+½) | |
| Bx | (i, j+½, k+½) | |
| ρ | (i+½, j, k) | |
| Jy | (i+½, j+½, k) | stored where the transport source lives (μ_x j_y of the paper) |
| Jz | (i+½, j, k+½) | |

Gauss law: (Ex(i+1) − Ex(i))/h_x + μ_x δ_y Ey + μ_x δ_z Ez = ρ at (i+½, j, k).

## Field solver (RIP-CC)

D = h_x = c Δt, J = j^{n+½}, μ = average.

1. Predictor: Ex* = Ex + D/2 (δ_y Bz − δ_z By − Jx), Bx* = Bx − D/2 (δ_y Ez − δ_z Ey).
2. Transport: T⁺_y = Ey + Bz and T⁻_z = Ez − By move +x, T⁻_y = Ey − Bz and T⁺_z = Ez + By move −x,
   one cell per step, plus D times the sources at (i+½, n+½):
   Ψ_y = δ_z μ_x Bx* − Jy, Ψ_z = −δ_y μ_x Bx* − Jz, Φ_y = −δ_z μ_x Ex*, Φ_z = δ_y μ_x Ex*.
3. Corrector: Ex += D (δ_y μ_t Bz − δ_z μ_t By − Jx), Bx −= D (δ_y μ_t Ez − δ_z μ_t Ey),
   with μ_t F = (Fⁿ + Fⁿ⁺¹)/2.

The corrector uses the same time-centred transverse fields as the trapezoidal x-transport. That is
the condition for exact conservation of the Gauss law and of ∇·B; the Ex*, Bx* in the transverse
sources drop out of it. The 2020 paper advanced a second, half-step field set instead, which breaks
both. This version needs one field set (plus four temporary arrays), and its vacuum dispersion is
exactly Eq. (57) of the paper. Stability: h_x²(1/h_y² + 1/h_z²) + h_x² ω_p²/4 < 1, checked at start.

Four guard exchanges per step (J, Ex*/Bx*, new transverse fields, Ex/Bx), guard width 3.

## Particles

Higuera–Cary (default) or Boris pusher; gather of E^n, B^n at x^n with shape order 1 (linear), 2
(quadratic) or 3 (cubic, Cart3D, docs/specs/solid_target.md), each component at its own stagger, so the
co-located pairs (Ey, Bz) and (Ez, By) use identical weights. Current: Esirkepov deposition of the straight
move x^n → x^{n+1} on the charge grid (nodes at ((i+½)h_x, j h_y, k h_z)); this is exactly the continuity equation of the solver. The
initial momenta are drawn per global cell, so the initial state does not depend on the decomposition.

## Deck

```
grid      { lambda0, nx ny nz, dx dy dz, boundary_x = periodic|open|absorbing, t_end, mpi_x mpi_y mpi_z (0 = auto),
            x_absorber_width = 4 lambda0 (or x_absorber_cells), x_absorber_R0 = 1e-4 }   # absorbing: damping
                                     # layers at both x ends, any angle (45 deg: 6e-4 in field);
                                     # lasers only through tfsf_x (docs/specs/x_absorber.md)
numerics  { solver = rip_cc|yee|irip (yee: periodic boxes, cfl = 0.98; docs/specs/yee.md;
            irip: implicit RIP, no Courant limit; ranks along x only so far;
            docs/specs/implicit_rip.md),
            closure = box|o5 (irip: o5 corrects the oblique dispersion, e.g. 9-35 times less phase error for
            harmonics at 0.2-0.4 rad; not energy conserving, transient growth at the grid scale <= 2.15 for any
            dy / dx (the time-step term of the transverse factor is capped at k_perp c dt = 2); 2D, 3D
            (the transverse factor per mode in the y, z Fourier basis) and RZ (per radial eigenmode, also stretched
            radial grids); Cartesian: with CPML (reflection ~60 times the box closure's: 1.5e-4 at 16 cells, 1e-5 at
            32; use about 32 cells), the TFSF plane (incident grid and tfsf_incident = modes) and the side tube (plane
            wave, flying focus, background plasma), all with leakage at round-off; RZ: with the TFSF plane (decision 78);
            decisions 72-77, docs/specs/overdense.md),
            shape = 1|2|3 (3: Cart3D, no moving window; particles kept 2 cells below and 1 above an open edge),
            pusher = higuera_cary|boris, seed,
            beam_init = steady|gauss (self_field beams, RIP-CC: steady, the default, = the exact discrete steady state of the
            RIP-CC step, emittance kept; gauss = the Gauss-law field of decisions 24, 95; iRIP, Yee, RZ, CPML: gauss;
            decision 99, bench/emittance/),
            filter_passes = 0 (passes of the Gauss-law-exact current filter: binomial of the charge; rip_cc, irip; with a
            moving window at v = c (lab, decision 107) or in a boosted frame, where shape 2 with 1 - 2 passes is needed
            against the NCI of the streaming plasma, decision 104; docs/specs/solid_target.md),
            filter = binomial|projection|local (binomial, the default: every component of J smoothed as rho, so a
            drifting plasma keeps J = v rho and stays stable, any boundary and decomposition, decision 91;
            projection: only the longitudinal J, its change a gradient that radiates nothing (harmonics unfiltered),
            y, z periodic, ranks along x only, elsewhere the local filter with a warning, decisions 69, 71; local:
            its change of J radiates at the grid scale. Projection and local make a relativistically drifting plasma
            unstable (W_field / K_e 0.1-1 in the streaming-plasma test, binomial 1e-10): not for beams or LWFA),
            filter_compensate = false (binomial: one compensating step along x (z in RZ) after the passes, response
            T^N ((N + 1) - N T), flat to O(k^4) at the laser; Gauss law exact; decision 108, bench/nci/mechanism),
            filter_flat = 0 (binomial, test switch: L = 2 .. 9 adds the maximally flat factor sum_{j<L} binom(N-1+j, j) S^j
            along x (z in RZ) after the N passes, S = 1 - T; flat to O(S^L) at the laser, Gauss law exact; decision 109),
            filter_axes = all|x (binomial: x passes the filter along the streaming axis only, z in RZ; decision 108;
            boosted frame: 8 passes + compensator along x keep the NCI at the noise level for gamma >= 10, 0.3 % less
            plasma response at the laser (lambda / 32) against 1.9 % for two plain passes; provisional),
            tile = 8 8 8 (cells per tile), sort_every = auto (tile sort every n steps, 0: off,
            auto: every step only if the deposit needs it),
            deposit = auto|global|tiled|box (auto: coloured tiles on the CPU, global on the GPU;
            box: one GPU block per tile with a fixed-point shared-memory box) }
species <name> { charge, mass, density (n_c), ppc_x ppc_y ppc_z, ux uy uz, uth_x uth_y uth_z,
                 xmin xmax ymin ymax zmin zmax,
                 density_function = x < 20 um ? (x - 13 um) / (7 um) : 1,   # factor of density (absolute
                                                       # x, y, z; docs/specs/density_profile.md)
                 self_field = false (steady fields of the beam moving with ux, docs/specs/beam_self_field.md),
                 t_start = 0 (load later, e.g. a witness beam in the window),
                 rigid_until = 0 (momenta frozen until then, current deposited; decision 98) }   # any number of blocks
species <name> { element = N, charge_state = 0, ionization = adk, ionization_electrons = <species>, max_charge_state,
                 ionization_current = true, ionization_bsi = none|tong_lin, ionization_tl_alpha = 6, ionization_m = 0,
                 ionization_potentials = 552.07, 667.05 eV, ionization_l = 0, 0, density (atoms, n_c) }
                                     # field ionization: charge state per particle (q = 1, m = the atomic mass), ADK per
                                     # level (l* = n* - 1, l of the removed electron, |m| = 0), NIST energies of H .. Lr
                                     # (data/ionization_energies.def), Monte Carlo with several levels per step, electrons
                                     # into ionization_electrons, the energy-conserving ionization current (charge-
                                     # conserving displacement); Cart3D, 2D, RZ, MPI, GPU; dir/ionization.txt (decision 86,
                                     # docs/specs/field_ionization.md, examples/ionization_2d.deck, rz_ionization.deck);
                                     # element + charge_state without ionization: an ion of that charge and mass
collisions <name> { species1 = e, species2 = i (default species1), coulomb_log = 0 (automatic) | 5, every = 1,
                     angle = nanbu|smilei5, low_t_cap = true, quantum_bmin = hbar|h, ionizing = false,
                     ionization_electrons = e }
                                     # binary Coulomb collisions (Pérez et al. 2012, Nanbu angles, Smilei 5.1 pairing and
                                     # weights), any number of blocks; ionizing: impact ionization of the ion species (an
                                     # element and charge_state; RBEB cross-sections, NIST energies); mobile = false species
                                     # of a block are frozen (collisions only); Cart3D, 2D, RZ, MPI, GPU; dir/collisions.txt
                                     # (decision 94, docs/specs/collisions.md, examples/collisions_2d.deck, rz_collisions.deck)
species <name> { ..., spin = 0 0 1, spin_anomaly = 1.15965218e-3 }   # spin vector (particle axes, |S| <= 1): T-BMT
species <name> { kind = photon }     # photons (q = 0, ballistic), filled by the qed block; attributes xi1 xi2 xi3 (Stokes),
                                     # pe0 pe1 pe2 (the polarization axis e1)
qed       { species = <leptons>, model = monte_carlo|deterministic|none, photons = <photon species> (optional),
            photon_min_energy = 0 (MeV), spin_state = mixed|pure, spin_precession = true, spin_flip = true,
            substep_prob = 0.1, spectrum_bins = 200, spectrum_u_min = 1e-8, spectrum_e_min = 1e-3, spectrum_e_max = 1e7 (MeV),
            spectrum_every = 0 }     # spin-resolved strong-field QED (decision 93, docs/specs/qed.md): Monte-Carlo photon
                                     # emission (LCFA) resolved in the lepton spin and the photon Stokes vector, spin flips
                                     # and the non-radiative spin change, sub-steps; deterministic = P_alpha chi^2 g(chi);
                                     # T-BMT for every species with spin; Cart3D, 2D, RZ, MPI; RIP-CC and iRIP (explicit mover; docs/specs/qed.md "Field
                                     # solvers and the time step"); the boosted frame (decision 117: lab energies and lab spins,
                                     # docs/specs/qed.md "Boosted frame"); dir/qed.txt (<S>, <chi>,
                                     # emitted energy, <xi>), dir/qed_spectrum.txt (dN/du, dN/d eps with <xi>);
                                     # examples/qed_collision.deck, bench/qed_pwfa/
            + pair_electrons = <e->, pair_positrons = <e+>, pair_selection = true   # Breit-Wheeler pairs (decision 96):
                                     # the photons decay by Monte Carlo (LCFA), resolved in their Stokes vector and the
                                     # pair spins; pair species mobile, charge -1 / +1, mass 1, density = 0; dir/qed.txt
                                     # adds N_pairs, E_pairs, <S+->, dir/qed_pairs.txt (dN/dx, dN/d eps);
                                     # examples/qed_cascade.deck (+ qed_cascade_estimate.py)
laser     { a0, polarization = y|z, duration (intensity FWHM), delay (default 2 duration), waist,
            waist_z (0: uniform in z; set = waist for a round 3D pulse), y_center, z_center,
            focus (x of the focus, default 0), mode_p, mode_l (Laguerre-Gauss), phase,   # boundary_x = open
            tfsf_x, angle }          # tfsf_x: TFSF plane inside the box instead of the left edge (also with CPML), the beam at `angle`
                                     # tfsf_incident = grid|modes: modes = exact discrete plane waves at the plane, no
                                     # incident grid (periodic y, z; the pulse may start outside the box; decision 65)
                                     # (about z, from x towards y); envelope zero beyond 3 sigma (docs/specs/tfsf.md)
                                     # |angle| <= 45 deg; waist = 0 with tfsf_x: a plane wave of one k_y (periodic y,
                                     # sin(angle) ny dy = m lambda0; docs/specs/fresnel.md)
                                     # injection = plane|antenna (boost only): antenna = a current sheet fixed at the lab
                                     # plane x = 0 with its charge (Gauss law exact), moving at -beta c in the boosted
                                     # frame: a beam focused far downstream need not fit the box at the boosted entrance
                                     # plane's lab events (decision 89, docs/specs/boosted_frame.md); a lab run: plane x = 0
laser <name> { ... }                                   # more lasers, summed (docs/specs/laser_modes.md)
side      { faces = yz, margin = 2 (or y_lo, y_hi, z_lo, z_hi), a0, angle, elevation, polarization = y|z,
            duration (0: continuous), delay, x_center, y_center, z_center, phase,
            wave = plane|flying_focus (cone_angle, length_1e or duration, x_center, cep: a luminal Bessel pulse,
            window at c from t = 0),
            background_density (n_c: the incident wave of that plasma), fluid = shell|all (the cold fluid outside the
            kinetic region or everywhere) }
                                     # a plane wave through the faces of a TFSF tube (y, z = const, inside the CPML),
                                     # any angle, exact discrete dispersion (docs/specs/side_injection.md)
laser <name> { file = laser.h5, file_x, delay, t_center, y_center, z_center, a0 }   # from a LASY file
                                                       # (any structure; propagated exactly from file_x,
                                                       # docs/specs/laser_file.md; also RZ (thetaMode kept as modes,
                                                       # exact Hankel propagation per mode), 2D (slab files) and the
                                                       # boosted frame (always through the lab-fixed antenna,
                                                       # injection = antenna), decision 97)
species:    slab_x0, slab_y0, slab_angle, slab_thickness       # optional tilted layer (absolute coordinates)
            slab_ramp, slab_ramp_extent (5 slab_ramp), slab_rear_ramp, slab_rear_ramp_extent, slab_exact
                                     # exponential pre-plasma in front and behind; with a ramp (or slab_exact) the
                                     # loaded charge is the integral of the profile to round-off at any tilt; the
                                     # run stops if omega_p dt at the loaded peak density is unstable (O4,
                                     # docs/specs/solid_target.md, examples/solid_target.deck)
window    { start (default 0), velocity (c, default 1) }   # moving window along +x, needs boundary_x = open
boost     { gamma }                 # Lorentz-boosted frame along +x (RZ: +z); the rest of the deck stays in the lab
                                    # frame (docs/specs/boosted_frame.md, decision 87, examples/lwfa_boost.deck,
                                    # rz_lwfa_boost.deck): dx -> gamma (1 + beta) dx, t_end and window.start lab times,
                                    # lasers at the boosted entrance plane, plasma at rest streaming in 3 + filter reach cells
                                    # behind the right edge (it must be exactly neutral per lattice group: species with the
                                    # same ppc, bounds and slab form a group with sum q Z n f(x) = 0, i.e. every pre-ionized
                                    # ion species brings its electrons with the same ppc and matching weight; groups may differ
                                    # in ppc and overlap, e.g. a fine inner and a coarse outer zone; the run stops otherwise,
                                    # decisions 108, 114),
                                    # beams (ux != 0 or self_field) mapped ballistically; iRIP for gamma >~ 2;
                                    # field ionization with the ADK rate in the ion's rest frame over its proper time
                                    # (decision 90, examples/ionization_boost.deck, rz_ionization_boost.deck);
                                    # the streaming plasma needs numerics.shape = 2 and filter_passes = 1 (2 when it
                                    # stays > 100 / omega_p in the box): without it the NCI grows from the noise (the
                                    # summary warns; decision 104, bench/nci/boost_stability/README.md)
lab_diagnostics { t_first, t_every, count, fields = true, species, gamma_min, particles = true,
                  select_uz, select_r }   # select_uz: record only lab u along the boost axis >= select_uz, select_r:
                                    # |transverse coordinates| < select_r (FBPIC's select; decision 97)
                                    # lab-frame snapshots: dir/lab_fields/snap_KKK.{bin,txt} (tools/lab_fields.py),
                                    # dir/lab_particles, dir/lab_beams.txt (moments in the lab frame)
grid:       boundary_y = pml, boundary_z = pml              # CPML layers inside the box (default periodic)
            absorber_particles = remove|conserve|keep       # particles entering the y/z PML (RZ: the radial absorber): removed,
                                                            # no current in the layer | removed charge-conserving (default) | kept
                                                            # (summary: number, charge); docs/specs/pml.md, decisions 111, 115
pml       { cells = 16, R0 = 1e-6, order = 3 }              # docs/specs/pml.md
output    { dir, every, slice_every, slice_z, slice_fields = Ex Ey Ez Bx By Bz rho n_<species>,
            openpmd_every (0: off), openpmd_fields = E B rho (also J, rho_<species>),
            xflux = true (energy through the open x edges or the inner faces of the x absorbers: xflux.txt,
            outbound and net), xflux_every (records of the
            reflected / transmitted field: xflux/, tools/xflux.py; docs/specs/xflux.md),
            openpmd_species = <names>|all, openpmd_precision = single|double }   # docs/specs/openpmd.md
harmonic_focus [<name>] { n_min = 4.5, n_max = 30, filter_order = 8, xmin .. zmax (RZ zmin, zmax, rmax), every = 0 }
                                     # band-passed field per node: peak of |E_band|^2, its time and the fluence in
                                     # harmonic_focus[_<name>]/, tools/hfocus.py (CHF, docs/specs/harmonic_focus.md)
checkpoint { every (0: off), keep = 2, stop_after_hours = 0 }   # restart with --restart file|latest|auto
medium [<name>] { model = drude|lorentz, density (wp^2 / omega0^2 = n / n_c), collision_frequency = 0 (nu / omega0),
            resonance (lorentz: omega_r / omega0), permittivity (lorentz, instead of density: the real eps at omega0
            without damping, wp^2 = (eps - 1)(omega_r^2 - 1)), profile = <factor of x, y, z (RZ: z, r)>, supersample = 4 }
                                     # O7: a particle-free linear medium (plasma mirror, lossy plasma, glass) for irip with
                                     # closure = box, any omega_p dt; Crank-Nicolson current driven by the box closure's work
                                     # field, W + K + dissipated exact; the profile's sub-cell volume fraction at each current
                                     # position; Cart3D, 2D, RZ (all modes), CPML (media one cell clear of the layers), MPI,
                                     # particles (also numerics.mover = implicit, Cart3D); not with o5, TFSF, side; moving window supported;
                                     # checkpoints with the media state; numerics.medium_tol = 1e-10, medium_maxit = 200, medium_restart = 30 (GMRES);
                                     # dir/medium.txt: K, W_dissipated, the balance, GMRES statistics (decision 84,
                                     # docs/specs/linear_medium.md, examples/plasma_mirror_2d.deck, rz_medium.deck)
medium [<name>] { model = bound, atom = Rb | wavelength1, strength1[, wavelength2, strength2], density (atoms: n_a / n_c
            or '<n> cm^-3'), polarization = z (the dipole axis x|y|z; RZ x), lifetime = 0, dephasing_time = 0 (T1, T2; 0:
            none), ionization = none|adk, ionization_potentials (eV: ground, excited; Rb built in), electrons, ions (species),
            ionization_ppc = 8, n_phi (RZ: angles, default 4 x modes), profile, supersample = 4 }
                                     # option: the bound electrons of a gas as V-type Maxwell-Bloch atoms (Rb: D1 795 nm,
                                     # D2 780 nm; saturation and Rabi cycling near an 800 nm laser, no rotating-wave
                                     # approximation); exact energy exchange with the field (K = the atoms' energy), the
                                     # nonlinear current iterated around the GMRES solve (numerics.medium_bound_tol = 1e-8,
                                     # medium_bound_maxit = 30; outer iterations in dir/medium.txt); Cart3D, 2D, MPI,
                                     # particles; relaxation, ADK ionization per level into electron-ion pairs at rest, RZ
                                     # (atoms on n_phi angles), checkpoints (decision 85, examples/rb_gas_2d.deck, rz_rb.deck)
diagnostic <name> { species, every (default output.every), gamma_min (default 1), bins (200),
            e_max_mev (1000), particles = false }   # beam_<name>.txt: Q, energy, spread, emittances,
                                     # sizes; spectrum_<name>.txt: dQ/dE; particles/ (docs/specs/beam_diagnostics.md)
```

2D Cartesian (x, y) (`docs/specs/cart2d.md`, decision 66; `examples/foil_2d.deck`): fields and particles independent
of z, all six field components and three momentum components kept. The Cart3D code on one z cell with no z guards and
the z stride 0, so d/dz = 0 exactly in every field kernel and a field is one (x, y) plane; the particle kernels have a
2D shape. Weights and every total (diag.txt energies, `diagnostic` charges and spectra, xflux) are per slab of
thickness dz: a 2D run with the dz of a z-uniform Cart3D run equals one z column of it.
```
grid      { geometry = cartesian2d, lambda0, nx ny, dx dy, dz = 1 um (slab thickness of the totals), boundary_x,
            boundary_y = periodic|pml, mpi_x mpi_y, t_end }   # no nz, mpi_z, boundary_z
```
rip_cc and irip (irip: ranks along x), shapes 1-3, current filter, open / absorbing x, CPML in y, lasers through the left
edge or a TFSF plane (incident grid or `tfsf_incident = modes`), moving window, species profiles, diagnostics, openPMD and
checkpoints. Refused: yee, `side`, laser files other than slab files (LASY y of length 1), waist_z / mode_p / mode_l, self_field, ppc_z / zmin / zmax.
The RIP-CC stability condition loses its z term: dx^2/dy^2 + dx^2 wp^2/4 < 1.

RZ with azimuthal modes (`docs/specs/rz_modes.md`; z is the transport axis, c dt = dz; periodic or open
z with laser and moving window, a conducting wall at r = nr dr that absorbs particles (optionally behind
a radial absorbing layer for the fields), openPMD output in thetaMode, checkpoints and restart as below):
```
grid      { geometry = rz, modes = 2 (m = 0 .. modes - 1), lambda0, nz nr, dz dr, boundary_z = periodic|open|absorbing
            (absorbing: damping layers at both z ends, z_absorber_width = 4 lambda0 | z_absorber_cells, z_absorber_R0 = 1e-4;
            the laser then comes through laser.tfsf_x),
            boundary_r = wall|absorbing (r_absorber_cells = 32, r_absorber_R0 = 1e-4), t_end, mpi_z,
            r_uniform, stretch = 1, dr_max (stretched radial grid: dr up to r_uniform, then x stretch per
            cell up to dr_max; keep k_perp dr <~ 1 where radiation must leave, stretch slowly; the openPMD
            records carry the node radii as rippic_r_nodes, the absorber is as thick as its outer cells) }
numerics  { solver = rip_cc|irip (rip_cc: dz^2 (lambda_max + wp^2) / 4 < 1 is checked, lambda_max of the
            radial operators grows with the modes near the axis; irip: no limit), shape = 1|2 (2 is
            folded across the axis, less noise there; recommended), pusher, seed,
            filter_passes = 0 (RZ: passes of the charge-conserving current filter, smoothing the charge in z and
            per mode in r; 1-2 bring the axis noise to FBPIC's, the Gauss law stays exact also at open edges,
            decision 54), filter = binomial|projection|local (binomial, the default: every component smoothed in z,
            J_z in r as rho, stable for drifting plasma, decision 91; projection: the change of J is the gradient of
            the Gauss law and radiates nothing, decision 70, but a drifting plasma is unstable with it; local: the
            old filter, whose change of J radiates). With a moving window the entering plasma is frozen for
            3 + filter_passes cells and its charge kept as a background: the Gauss law stays at round-off with warm
            and non-neutral plasma entering (decision 91),
            rz_transverse_shape = none|gaussian (the noise heating grows with the modes at fixed ppc_theta: use
            ppc_theta >= 2 (2 modes - 1), a warning below; gaussian: a transverse Gaussian particle shape of width
            rz_shape_width = 1 dr on the deposit and the gather of E, which also damps large k_perp such as harmonics;
            decision 79), rz_axis_cut = 0 (kappa > 0, rip_cc: mode m >= 2 held at zero inside r < m_eff dr / kappa, a
            conducting rod per mode, the polar filter of global models: lambda_max no longer grows with the modes, so
            dz ~ dr with many modes; removes nothing a field of mode m ~ r^m could carry there; docs/specs/rz_metal_phi.md),
            rz_gather = plain|matched|energy
            (plain, the default since decision 106: the plain splines, exact for fields linear in r, as FBPIC; matched:
            the gather in the deposit's density-conserving radial map, for dense plasma at the axis (k_p dr > 1.5: half
            the heating of plain, decision 80), but it overestimates a focusing field in the cell next to the axis
            (x 1.25 at dr/2 with shape 1, x 1.6 with shape 2; betatron frequency of a narrow beam too high); energy:
            in addition E_r energy-consistent, not exact for smooth fields next to the axis. The code warns for a dense
            plasma with plain, for matched without a dense plasma, and for a streaming plasma (a drift
            u >= 1) with dr < c dt and no current filter),
            rz_axis_map = m0|all (m0, the default: the density-conserving radial map of the deposit for m = 0 only, the
            modes m >= 1 in the plain radial coordinate; all: the map in every mode, as before decision 106, which
            distorts a uniform transverse current and the dipole of a charge in the cells next to the axis),
            rz_deposit = phase|chord (the explicit deposit next to the axis; phase, the default: J_phi of m = 0 from
            r_j dphi and J_r with the mean phase, which gives a particle passing the axis at b << dr a spurious m = 0 J_phi
            ~ pi r_j^2 / dt (dphi ~ pi); chord: the splits of the implicit mover, J_phi from the angular momentum of the move
            and J_r with the chord phase; both charge-conserving per mode, equal far from the axis),
            rz_axis_current = regular|raw (RZ, decision 112: only the regular m = 1 current in the first radial cell and the
            adjoint in the gather, the default; removes the axis line at 0.48 pi; not with the implicit mover; raw: as before),
            mover = explicit|implicit (implicit, O6: Crank-Nicolson particles with the box closure, field + kinetic
            energy conserved to mover_tol, Gauss law exact, no omega_p dt limit, no grid heating; irip with closure = box,
            shape 2 or 3, Cart3D and cartesian2d, periodic or open x; RZ with modes = 1 or 2 (m = 0, 1), shape 2: the deposit then
            uses a C1 density-conserving radial map (decision 82; load with ppc_r >= 8: the axis node gets 1e-3 of the
            density wrong with 8, 9e-2 with 2); no pml, TFSF, side tube, window or filter yet; momenta at integer times; cost
            4-8 particle passes per step at omega_p dt < 1, ~10-30 at 2-10; decisions 81, 82, docs/specs/implicit_mover.md),
            mover_tol = 1e-6, mover_maxit = 100, mover_anderson = 5 }
species <name> { charge, mass, density, ppc_z ppc_r ppc_theta (default 1 1 4), uz ux uy, uth_z uth_x uth_y,
                 zmin zmax rmin rmax, density_function = f(z, r, x, y, phi), t_start, mobile = true (false: a fixed
                 background, e.g. ions), self_field = false (true: the steady field of a beam along z, open z, at the
                 start; Gauss law at round-off; decision 83) }   # deposition with density-conserving shapes (decision 53)
rings     { r_in, r_out, period, gap, z_start, z_end, jacket }   # embedded perfect conductors: metal rings r_in < r < r_out of
                                     # thickness period - gap, fixed in the lab frame (also with the window); staircase on the
                                     # grid (r_in, r_out snapped to radial nodes, faces to z cells), tangential E = 0 and normal
                                     # B = 0, total reflection at the z faces through the RIP transport, particles inside removed;
                                     # gap = 0: a tube, r_in = 0: discs; rip_cc only (docs/specs/rz_metal.md, decision 83)
                                     # jacket > 0: a solid tube r_out .. r_out + jacket over [z_start, z_end] closes the gaps
                                     # (closed corrugated pipe with an open end at z_end); one convenience geometry: conductors
                                     # of any shape (also axisymmetric) with the block metal below
dielectric [<name>] { r_in, r_out, z_start, z_end, eps, ramp }   # lossless dielectric eps(r, z), rectangles fixed in the lab frame,
                                     # chi = eps - 1 added; metal wins; time-centred polarization current (a tridiagonal line
                                     # along z per row, stable and lossless for any eps), div D = rho at round-off; cut cells
                                     # sub-cell averaged; ramp: linear z ends; with a moving window the last 32 cells at the front
                                     # edge absorb (keep beams behind them); rip_cc only (docs/specs/rz_dielectric.md, decision 100)
                                     # (a warning, at the start and during the run, when charged particles are inside it);
                                     # several blocks combine (any union of rectangles); with metal: only the rings block so far
                                     # (not the metal block); examples bench/rz_dielectric/dwa.deck (a lined tube, the wall the
                                     # grid edge), cherenkov.deck
metal     { function, phi_derivative = difference|spectral, enabled = true }   # RZ: conductors that depend on the azimuth:
                                     # metal where function(x, y, z, r, phi) > 0 (lab frame, also with the window), on the
                                     # 2M - 1 angles of the modes; difference (default): the phi derivative becomes the
                                     # staggered difference over 2 pi / (2M - 1) (i m -> i m_eff, a Yee grid in phi), a
                                     # staircase in r and phi, z faces as rings; couples m and m +- 4 for a C4 shape (a square
                                     # pipe: examples/rz_square_pipe.deck); spectral: axisymmetric masks only (=
                                     # rings); particles inside removed; rip_cc only (docs/specs/rz_metal_phi.md, decision 103)
quadrupole <name> { z_start, length, gradient [T/m], x_center, y_center }   # RZ: B_x = -G (y - y_c), B_y = -G (x - x_c), G > 0
                                     # focuses electrons in x; edges averaged over the kick; quadrupole_lattice { file, z_offset }
species <name> { file = beam.h5, z_center, x_offset, y_offset, self_field, rigid_until, ... }   # openPMD beam (as Cart3D)
wake <name> { x, y, every }          # RZ probe off the axis (all modes): E_z, E_x - c B_y, E_y + c B_x and their integral at
                                     # fixed xi = z - c t; beam_slices as Cart3D (docs/specs/structure3d.md section 5,
                                     # decision 105; example examples/rz_corrugated_pipe.deck)
laser     { a0, polarization = x|y, duration, delay, waist, focus, mode_p, mode_l, phase, y_center, z_center, tfsf_x }
                                     # round, centred on the axis (default) or off it in the transverse plane, projected on the modes
                                     # (LG_p^l: modes >= |l| + 2; off the axis: enough modes for its azimuthal content); boundary_z = open
                                     # tfsf_x: a TFSF plane at z = tfsf_x (normal incidence; the incident field from a vacuum RZ grid
                                     # with the laser through its left edge; focus and delay count from z = 0), the reflected light
                                     # alone left of it (docs/specs/tfsf.md "RZ", examples/rz_tfsf.deck)
window    { start, velocity }       # as in Cart3D, along +z
output    { openpmd_every, openpmd_fields = E B J rho rho_<species>, openpmd_species, openpmd_precision,
            plane_z, axis_every }    # thetaMode (FBPIC's layout, openpmd-viewer); particles x, y, z
                                     # plane_z: all components at the lab plane z every step (dir/plane, also with the window),
                                     # axis_every: Ez on the axis of the box; tools/rz_rings.py (wake, loss, Poynting, far field)
```
Examples: `examples/rz_plasma.deck` (a hot plasma column, iRIP at dz = 2 dr, electrons absorbed by the
wall), `examples/rz_laser.deck` (a focused pulse through the open z boundaries), `examples/rz_lwfa.deck`
(the RZ twin of `examples/lwfa_window.deck`: 62 s on 2 cores), `examples/rz_tfsf.deck` (a pulse through a TFSF plane onto
an overdense foil, absorbing z layers, the energy balance in xflux.txt), `examples/rz_disk_loaded.deck` (a 1 nC proton bunch at
gamma = 10 with its self-field through a closed disk-loaded waveguide with an open end: wake, energy loss, the field at a
plane behind the exit; 2420 x 711 cells), `examples/rz_square_pipe.deck` (a bunch through a square corrugated pipe, block
metal with 17 modes and the axis cut: the C4 coupling of m and m +- 4).

Structure runs in Cart3D (decision 102, docs/specs/structure3d.md): a bunch through a metallic structure of any 3D shape,
inside external quadrupoles, with the wake diagnostics. Files written for z-propagating codes map (x, y, z) -> (y, z, x).
```
metal     { function = f(x, y, z) }  # perfect conductor where f > 0 (absolute coordinates, lab-fixed, also with the window):
                                     # staircase on the RIP-CC grid (tangential E = 0, total reflection at x faces through the
                                     # RIP transport, the Gauss law exact off the surface), particles inside absorbed; rip_cc only
quadrupole <name> { x_start, length, gradient [T/m], y_center, z_center }   # hard-edged, on the particles only:
                                     # B_y = -G (z - z_c), B_z = -G (y - y_c) (G > 0 focuses electrons in y); averaged over the
                                     # x crossed during the kick, so edges between the grid points are exact
quadrupole_lattice { file, x_offset, y_center, z_center }   # lines: x_start [m] length [m] gradient [T/m]
species <name> { file = beam.h5, file_species, file_iteration, x_center, y_center, z_center, y_offset, z_offset,
                 total_charge, macroparticles, resample_smoothing, self_field, rigid_until }
                                     # a beam from an openPMD particle file (electrons / positrons; the charge centroid at
                                     # x_center; resampling as GEM-PIC-RZ); the summary prints its moments
wake <name> { y, z, every, start }   # F/q = (Ex, Ey - Bz, Ez + By) at the probe along the window and its integral over time
                                     # at fixed xi = x - c t (the wake potential, V) -> dir/wake_<name>.txt (window at v = c)
beam_slices <name> { species, dx, every }   # Q, I, centroids, sizes, energy and spread per slice of xi -> dir/slices_<name>.txt
side_flux { y_lo, y_hi, z_lo, z_hi, every }   # Poynting energy leaving the box sideways (per face, accumulated) and the
                                     # field energy inside -> dir/side_flux.txt (default: 2 cells inside the CPML layers)
```
Example: `examples/structure_pipe.deck` (a rigid 1 GeV bunch with an offset through a round corrugated pipe
inside two quadrupoles; 96 x 64 x 64 cells, 300 steps, 8 s on 24 cores).

Laser: E_pol = a0 g(t) exp(−(y−y_c)²/w0² [−(z−z_c)²/w0z²]) sin(t − t_c) at the entrance plane x = 0
(focus there; with `focus` = x_f the paraxial beam at its distance from the focus, Laguerre-Gauss modes
and several lasers: docs/specs/laser_modes.md), injected as the inbound transport vector: the guard layer i = −1 at t_n holds
a(t_n + Δt), since T(0)ⁿ⁺¹ = T(−1)ⁿ. Ex (p) or Bx (s) of the narrow beam follow from the solver.
Slices: float32 files `dir/slices/<field>_<step>.f32` (ny × nx, x fastest), `meta.txt`, `index.txt`.

openPMD (`docs/specs/openpmd.md`): `dir/openpmd/data_<step>.h5`, openPMD 1.1.0 with the ED-PIC
extension (as FBPIC and WarpX), readable with openPMD-viewer and openPMD-api; code-unit values with their
`unitSI`; absolute x in `gridGlobalOffset` and in the particle positions. Checkpoints
`dir/checkpoints/checkpoint_<step>.h5` (also openPMD) hold the whole state: a restart continues the run
bit-exactly on the CPU (same ranks; another number of ranks redistributes the particles, round-off), and
the text logs continue where the checkpoint was written.

`diag.txt`: step, t, W_field, W_kin, W_total, max|ΔGauss| (change of the Gauss residual since t = 0),
max|ρ|, max|∇·B|, particle numbers.

## Tests (`make test`)

Vacuum waves along ±x shifted exactly (error 0); random vacuum fields bounded over 2000 steps with
∇·B unchanged; Boris and Higuera–Cary gyration phase, uniform-E acceleration, E×B drift; Gauss law
and ∇·B conserved to ~1e-14 for hot drifting neutral plasma and a non-neutral electron slab, shapes 1
and 2; Langmuir frequency against the leapfrog value; discrete continuity equation of the Esirkepov
deposition for random moves (shapes 1, 2), also for moves over ~6 cells split into segments (implicit
solvers), and the J positions derived from the staggering descriptor;
attribute registry: extra F64/F32/I64/I32 attributes survive migration (periodic wrap, open x);
pipeline: stage order, insertion order within a stage, boundary hook once per step at t_n;
laser: focus away from the entrance (width and fluence gain against paraxial theory), Laguerre-Gauss
against the q-parameter form, two crossed lasers give circular polarisation;
pml: reflection of a packet at 90 and 45 degrees into y and z layers (5e-6, 2e-5), no energy growth
with random fields, Gauss law outside the layers with plasma leaving through y and z, moving window =
fixed box shifted, layers included;
beam diagnostics: moments above gamma_min = direct sums (1e-12), emittance of a phase-space ellipse
= a b / 2, spectrum exact at the bin centres, particle files = the selected particles;
geometry: Cart3D gather of linear fields exact for shapes 1 and 2; tiles: stable sort, every particle
in its tile range with all attributes, sort_every = 0 removes the operator; open x: laser pulses
(polarisation y and z) leave through the open boundary with div B and Gauss at round-off, and a hot
plasma leaves through either open x edge with the Gauss law at round-off (shapes 1, 2); tiled
deposit: colouring, tiled = global to round-off, 1 and 3 threads bit-identical, strays (sort_every = 3); reach of the deposit
within [c-2, c+3]; box deposit = global (full and partial tiles, transversely hot plasma), strays, weights over 8
decades (fixed-point box); 64-bit block sums from 32-bit adds with carry exact;
2D geometry (test_cart2d): continuity of the 2D deposit per shape (also split moves) and sum J_z = sum q w v_z, Gauss law and
div B at round-off with hot plasma (rip_cc, irip), vacuum dispersion of (kx, ky) modes, Langmuir frequency, = the Cart3D slab
of 3 periodic z cells (foil + TFSF laser), TFSF plane waves without leakage, moving window;
moving window: a plane pulse frozen in the window exactly, Gauss law at round-off in a laser wake with
cold and hot plasma (shapes 1, 2, global and tiled), exact count of an entering slab at v = 1 and 0.5;
deck expressions (values, units, errors); density_function: a linear ramp loaded exactly; beam
self-field: Gauss law at round-off after the init (gamma up to 2000, also at rest = Poisson), the
line-charge field of a long beam, the near field travelling with the beam, a beam loaded later in a window, a
u = 1e4 beam in a window moving from t = 0 (no field ahead of it, Gauss law at round-off, decision 95).
RZ: guided TM01 / TE11 modes against the Bessel roots, iRIP energy and div E, div B per mode, gather of
a uniform field, Gauss law per mode with a hot plasma (RIP-CC and iRIP), Langmuir frequency, and the
deck path (exact loading, Gauss law with particles absorbed by the wall, the RIP-CC stability check),
the laser through open z (pulse energy against the analytic value, paraxial on-axis fluence), the
moving window with a laser wake (Gauss law at round-off, exact count of the entering plasma), openPMD
thetaMode output (a known field reconstructed from the file with the standard's formula), the radial
absorbing layer (a ring pulse leaves to 1e-4), shape 2 across the axis (Gauss law per mode at round-off),
the restart from a checkpoint (= the uninterrupted run, exactly); on a stretched radial grid the energy of
random fields, the guided modes, an outgoing wave that leaves without reflection (and is reflected beyond
the cutoff), the laser against the uniform fine grid, and with particles the Gauss law per mode, the loading and
a laser wake against the uniform fine grid (tests/test_rz_grid.cpp).
- O2, the field through the open x edges (tests/test_xflux.cpp): a plane pulse leaves with W + E_out constant
  to 4e-13 (RIP-CC, iRIP), an RZ Gaussian beam to 1.6e-4 (O(dz^2)), and a cold overdense step (9 n_c) reflects
  R = 1 with the Fresnel phase to 2e-3 rad.
- O1, the TFSF source (tests/test_tfsf.cpp, `build/tests test_tfsf`): a focused beam (waist 1.5 um) and a plane wave at 0 and
  30 deg, RIP-CC and iRIP, periodic or CPML y: leakage 1e-15, total - incident 2e-15; the incident grid starts
  divergence-free (1e-17, its Ex, Bx from the discrete constraints) and leaves no static field behind.
- O3, absorbing x boundaries (tests/test_tfsf.cpp, `build/tests test_x_absorber`): a focused beam from a TFSF plane
  into the default 4 lambda0 layer against a box 15 um longer: reflected field 3.9e-5,
  1.3e-4, 4.3e-4 of the peak at 0, 30, 45 deg (iRIP 4.4e-4), TFSF leakage 6e-16.
- Oblique Fresnel reflection (tests/test_xflux.cpp, `build/tests test_fresnel`): a TFSF plane wave of one k_y onto
  cold plasma steps against the Fresnel formulas at lambda0/32: 9 n_c, 45 deg, s phase error 1.1e-2 (6.6e-4 at
  lambda0/64), p with the surface half way between nodes 3.7e-3 (30 deg) and 1.4e-2 (45 deg) (on a node first
  order: the one-cell transition layer of the deposited step), 0.5 n_c at 0 deg |r| to 1.2e-3, 0.25 n_c at 30 deg s to 1.3 %
  (second order in dy); docs/specs/fresnel.md.
- TFSF plane with exact plane waves (tests/test_tfsf.cpp T3, `build/tests test_tfsf_modes`): leakage 1e-15 at 0, 30, 45 deg, both
  polarisations, RIP-CC and iRIP, round beam, the plane next to the open edge or the x layer with the pulse starting outside
  the box; peak = the incident grid's to 0.35 % at 0 deg.
- Side injection (tests/test_side.cpp, `build/tests test_side`): exact discrete plane waves through the faces of the
  TFSF tube at 60, 75.5 deg and 30 deg elevation (corners), continuous and pulsed, RIP-CC and iRIP, CPML y, z: outside
  the tube <= 1.4e-14 of the peak, total - incident <= 1.5e-14, Gauss law and div B <= 1.6e-14. The flying focus
  (`build/tests test_side_ff`, 5628 exact components in a window moving at c): 3e-15, 5e-15; against GEM-PIC-RZ and
  the analytic pulse in `bench/flying_focus/` (on the axis 2.4e-4 of a0 over 0.3 mm). In a background plasma
  (`build/tests test_side_plasma`, the cold fluid everywhere, TE and TM waves of the solver with the fluid): the same
  round-off (<= 1.3e-14) for the plane waves in 0.05 n_c and the flying focus in 0.01 n_c; kinetic electrons with the
  fluid shell leak 1.4e-2 of a weak pulse (2.5e-1 with a vacuum shell). `bench/flying_focus/rip_plasma.deck` against
  GEM-PIC-RZ in 1e18 cm^-3 (0.3 mm: the laser on the axis 1.001 a0 and stationary; linear wake within ~10 %, at a0 = 2
  the same cavity, the wake behind it 35 % weaker than GEM-PIC-RZ's).
- O4 solid targets (tests/test_solid.cpp, `build/tests test_solid`): the tilted slab with front and rear pre-plasma loads
  the analytic charge to 8e-14 at four tilts (the point-sampled plain slab: 9e-4); a neutral 20 n_c foil with hot electrons
  and mobile ions, shape 3 and the filter: rho(0) 2e-14, Gauss law and div B <= 3e-14 (RIP-CC, iRIP); the filter alone:
  charge waves cos^4(kh/2) for two passes, a discrete curl J unchanged (2e-16), charge kept at open x; omega_p dt at the
  loaded peak refused. Shape 3 runs through the plasma, continuity, gather, open-x, tiled, box, reach and Langmuir tests.
- CHF diagnostic (tests/test_hfocus.cpp, `build/tests test_hfocus`): the band-pass against the analytic bilinear Butterworth
  response (3.7e-13), the maps against a host replay of the grid's field in 2D (exact) and RZ (1.9e-14, axis and both
  angles), the fluence of a plane pulse against the analytic value (3.1e-3, the filter's |H|^2); bench/chf for the benchmark.
- RZ transverse particle shape (tests/test_rz_shape.cpp, `build/tests test_rz_shape`): S = exp(a^2/2 Lap_perp) per mode
  against the heat kernel of the plane (scalars of modes 0-3 to 2e-3 at w = 8 dr, second order; the vector field E_x to
  5e-3, first order next to the axis), self-adjoint to 1e-17; the noise heating of a hot column with 4 modes from 1.0e-3
  to below 7e-5 per 10 fs (bench/noise_heating).
- RZ TFSF plane and z layers (tests/test_rz_tfsf.cpp, `build/tests test_rz_tfsf`): a focused beam through the plane in vacuum,
  nothing left of it (2e-15 of the peak) and the total field = the incident grid's right of it (4e-15), RIP-CC, iRIP and
  iRIP with closure = o5 (3e-15, 4e-15; 1e-3 without the Cz fix at the plane); the
  z layer of 1 um reflects 8.2e-4 of the peak field (against a box 10 um longer); the laser through an open left edge is
  inbound energy in Enet_left: W_in + Enet_left + Enet_right = 0 to 2.4e-4 (RZ beam, O(dz^2)) and 3.4e-13 (Cart3D plane pulse).
- O7 linear media (tests/test_medium.cpp, test_medium2.cpp, `build/tests test_medium`): W + K + dissipated with random fields and
  media to <= 8e-13 (2D, 3D, omega_p dt up to 30) and 3.7e-14 (RZ, 3 modes); |r| = 1 at normal incidence to 7e-12 for omega_p dt =
  0.5, 3, 20; oblique s-pol Fresnel R of glass 1.6e-2, 3.5e-3, 8.5e-4 at lambda/10, 20, 40, Snell angle 0.29, 0.056, 0.002 deg,
  absorption of a 9 n_c collisional plasma 7.4e-2, 2.3e-2, 6.0e-3 (second order); Gauss law with particles plus media 4e-15
  (explicit and implicit movers, RZ 1.3e-14); the focus of a parabolic plasma mirror against physical optics; the deck block,
  the sub-cell weighting and the rejected combinations (docs/specs/linear_medium.md).
- O7 model = bound (tests/test_bound.cpp, `build/tests test_bound`): Maxwell-Bloch atoms; Rabi flopping of the node step against
  the Schroedinger equation without RWA (3 pi / 2: 2.0e-4, 5.0e-5, 1.3e-5 at lambda/16, 32, 64) and on the grid against the
  analytic field (5e-5); the weak-field spectral phase of 60 lambda of Rb vapour against the Lorentz chi of D1 + D2 (2.7e-3,
  6.8e-4 at lambda/20, 40); W + K_atoms in saturation to 1.7e-15; Gauss law with particles and atoms 4e-15; the deck block.
- RZ metal depending on phi (tests/test_rz_metal_phi.cpp, `build/tests test_rz_metal_phi`, decision 103): an axisymmetric
  function = the rings block to 5e-15; the cutoffs of a square guide and a four-ridge pipe = an independent numpy operator
  (research/rz_metal_phi/xsection.py) to 3e-7 for 4, 8, 16 modes, -2.6 % (TM11), -8 % (TE10), -5 % / -4 % (ridges) from the
  continuum at 16 modes; the axis cut: lambda_max dr^2 1610 -> 6.4 with 32 modes, TM11 unchanged; random fields in a
  square corrugated pipe bounded over 3000 steps, Gauss law and div B off the metal 2e-14; plasma with corrugated ridges:
  Gauss law 5e-16, particles absorbed.
- Structure runs (tests/test_structure.cpp, `build/tests test_structure`, decision 102): a rectangular metal pipe has the
  discrete TM21 / TE11 spectrum of RIP-CC exactly (4.9e-13 in w dt, kx = 0 and 2 pi / L); a round pipe (staircase) has the
  TM01 / TE11 cutoffs j01 / R, j'11 / R to -3.0 / -6.6 % at R / h = 8, -1.8 / -3.7 % at 16, -0.99 / -2.1 % at 32 (first
  order); x faces reflect exactly (1.4e-16, both polarizations and faces); plasma in a corrugated tube (periodic, CPML, moving
  window): Gauss law 1.2e-14, div B 2e-16, particles absorbed; electrons through a FODO with edges between the grid points
  against thick-lens matrices: 3.3e-5 / 1.6e-5 of the amplitude at h = 4 / 2; an openPMD beam written in the test and imported:
  Q, centroid, rms length, eps_n, <gamma> to 5e-15 (resampled: 4e-4); side_flux: energy inside + out = initial to 3e-8 (a TE
  field into CPML); the wake probe against a backward plane wave to 6e-12.
- Structure runs in RZ (tests/test_rz_structure.cpp, `build/tests test_rz_structure`, decision 105): electrons through a
  FODO (axis off r = 0, edges between the grid points) in the RZ push against thick-lens matrices: 3.3e-5 / 1.6e-5 of the
  amplitude at dz = 4 / 2 (Boris and Higuera-Cary); an openPMD beam imported in RZ: Q, centroid at z_center + offsets,
  eps_n x, y, sigma_z, <gamma> to 7e-15, particle by particle = the Cart3D import of the same file (3e-13), beam_slices
  sum Q to 7e-10; the wake probe off the axis (mode 1, all modes summed) against a backward plane wave to 5e-11; the deck
  keys in the RZ axes.
  Benchmark examples/rz_corrugated_pipe.deck (resolved teeth, bunch imported from a python-written file = the python
  moments to all printed digits): loss factor 8.09 kV/pC/m vs Bane-Stupakov 8.08, dipole kick factor 2.31 vs 2.42
  kV/pC/m/mm (k1^2 = k0^2 + 2 / a^2, decision 105).
- RZ embedded metal and self-field (tests/test_rz_metal.cpp, `build/tests test_rz_metal`): a tube r_in = R on a larger grid
  equals the conducting wall at R exactly (TM and TE, 300 steps); a cavity closed by discs equals the mirror-symmetric periodic
  box to 5e-15 (faces included), its TM011 frequency within 5e-4 of the continuum; hot plasma between rings (shapes 1, 2): the
  Gauss law off the metal surface 1e-15, div B 1e-16, particles absorbed by the metal; the RZ self-field of a gamma = 10
  beam: Gauss law 2e-13 of max rho at the init, its field energy constant to 2e-3 over 10 fs; a u = 1e4 beam in a moving
  window from t = 0: Gauss law 2e-14 (decision 95); iRIP with rings rejected.
- RZ dielectric blocks (tests/test_rz_dielectric.cpp, `build/tests test_rz_dielectric`, decision 100): div D unchanged to 1e-14 with
  partial cells (eps 1.7, 3); TM01 in a filled tube against the scheme's own dispersion relation (eps 1.1 ... 4, within 0.5 %);
  eps = 1 bit for bit; plasma + rings + gaps + collar: div D - rho 6e-16; a moving window with eps 3.8 and a bunch: Gauss 7e-13,
  no growth at the front edge. Benchmarks in bench/rz_dielectric/ (DWA, Cherenkov cone, thin wall).
- Field ionization (tests/test_ionization.cpp, `build/tests test_ionization`, decision 86): ADK of I = 1/2 Hartree = the Landau
  rate (6e-16), the code tables = the SI formula (4e-15) = the independent Python implementation tools/ionization_ref.py
  (3e-11), the cap beyond the peak; the Monte Carlo in a constant field: H against exp(-w t) and Ar0..Ar4+ at w dt = 2.3, 0.5
  (several levels per step) against the rate equations, within 1.1 and 1.4 sigma (32768 ions), the same ionizations with the
  ions reversed; C0 -> C4+ after a Gaussian pulse against the rate equations in Python, 1.2 sigma (65536 ions); the ionization
  current: field energy lost = its work = sum I_k w (4.6e-4, the time centring), with mobile electrons it takes sum I_k w
  from W + K (-0.994 of it); the Gauss law at round-off (1e-16) while electrons are created (Cart3D, 2D, RZ m = 0, 1); u = Z E
  t / m per particle (3e-16); deck checks.
- Binary collisions and impact ionization (tests/test_collisions.cpp, `build/tests test_collisions`, decision 94): one collision
  conserves P and E to 7e-14 (20000 random pairs, |u| up to 100, s up to isotropic); Nanbu's <cos chi> = e^-s to 1.2e-3; a
  gamma = 3 beam slowing on cold electrons vs d gamma / dt = -n k0 r_e lnL / beta to 9e-4 and a gamma = 2 beam scattering on
  heavy ions vs exp(-nu t) to 2.7e-3; isotropization (T_perp = 2 T_par) within 5-7 % of the bi-Maxwellian NRL rate in Cart3D,
  2D and RZ (inner and outer radial cells alike); e-i relaxation vs NRL / Spitzer converging with dt (0.098 -> 0.051), with
  unequal weights (energy on average, 2.8 %); Maxwell-Jüttner kept and reached; momenta per id bit-identical for reversed
  particle storage; impact ionization of carbon by a 1 keV beam vs n_e <sigma v> 0.3 %, the charge conserved, RBEB of H at
  100 eV 6.0e-17 cm^2 (measured 5.7e-17); deck checks. Against Smilei 5.1: bench/collisions_smilei/README.md.
- Spin-resolved QED (tests/test_qed.cpp, `build/tests test_qed`, decision 93; references independent of the code's tables:
  std::cyl_bessel_k and quadrature): Bessel tables 2e-7 (y <= 10); W(chi) and g(chi) vs quadrature / the Baier-Katkov form
  2e-8; the equilibrium spin at chi = 1e-3 -0.9235 (Sokolov-Ternov -0.9238); photon spectrum dN/du = C0 w at chi = 0.1, 1, 10
  (chi^2 per bin 1.03, 0.75, 0.61); rate(S = +e2) - rate(S = -e2) = 2 C0 g (chi^2 per bin 0.91, 1.19); the Stokes vector per
  u bin vs Eq. 21 and the xi1, xi2 terms (mixed and pure, within 1.8 sigma); the spin along e2 relaxing as the analytic
  solution at chi = 1 and 0.1 (to S_inf = -0.838, -0.905; mixed and pure, within 2.6 sigma); energy loss = P_alpha chi^2 g(chi)
  at chi = 0.1, 1, 10 (Monte Carlo within 1.1 sigma of 0.8 %, deterministic 3e-9); T-BMT rotation = RK4 (4e-15); precession
  in a constant B at (a gamma + 1) omega_c and a gamma omega_c relative to u in Cart3D and RZ (1e-10); energy balance electrons
  + photons -2.4e-13 with 87 % radiated; identical results with the particles reversed; deck checks. Boosted frame (Q11,
  decision 117): the Wigner rotation of perpendicular boosts (g1 + g2)/(1 + g1 g2) to 4e-15, spin and photon polarization axis
  there and back 6e-13; tests/mpi.sh: the cascade at gamma 2 on 1 vs 8 ranks and across restarts (lab files identical); lab vs
  gamma 2 runs of qed_cascade: 0.3144 vs 0.3134 pairs per beam electron (-0.4 sigma, docs/specs/qed.md).
- Breit-Wheeler pairs (tests/test_pairs.cpp, `build/tests test_pairs`, decision 96, physics choices accepted by Alexander
  2026-11-11; rates and helicity transfer agree with Seipt & King, PRA 102, 052805 (2020) and Chen et al., PRD 105, 116013
  (2022); transverse pair spin with their (Sokolov-Ternov) orientation since 2026-11-11, checked by a Volkov calculation
  (tools/volkov_spin.py, docs/specs/qed.md); Bessel functions by their integral
  representation, quadrature, Erber in closed form): tables T0, T3 6e-7; Erber limits (deviation -3.4e-3 at chi = 0.02 falling ~ chi,
  -3.8e-3 at chi = 1e4 falling ~ chi^(-2/3)); polarization ratio 1.995 (chi = 0.02, -> 2), 1.502 (chi = 1e4, -> 3/2); |S_+-| <= 1
  (max 0.99994); pair spectrum dN/dx at chi = 0.3, 1, 10 (chi^2 per bin 1.02, 0.98, 1.33), energy exact per event (4e-16), momentum
  deficit = m c (1/2eps_+ + 1/2eps_-); decay rates of xi3 = +-1 photons (ratio 1.872 vs 1.863, 1 sigma); survival and selection
  xi3 = tanh(Cg T3 t) (0.5394 vs 0.5391); pair spins per x bin (5 photon states, mixed and pure, within 2.4 sigma); positron parallel, electron antiparallel to
  the rest-frame B' (P4b: <S+.B'> 0.645, <S-.B'> -0.641 vs +-0.6427, magnetic and crossed field); cascade in a
  constant B (chi_e = 20, Cart3D and RZ): energy balance with the pair rest energy -8.5e-15 / -3.3e-15, charge, pairs = positrons;
  identical pairs with the beam reversed; with iRIP (P5b, Cart3D and RZ) the RIP-CC events at the same step (4.4e-16) and at 4
  times the step with h_perp = c dt / 4 the same pairs per beam electron within 4 standard errors of 4 seeds (10.09 vs 10.22 Cart3D,
  10.18 vs 10.18 RZ), no event with probability > 1; deck checks.
- Boosted frame (tests/test_boost.cpp, `build/tests test_boost`, decision 87): a vacuum laser back-transformed to a lab time
  against the lab run (1.3 % 2D RIP-CC, 2.5 % RZ iRIP) and the paraxial beam; an a0 = 1 wake (0.005 n_c, gamma 2, lambda/32)
  against the lab run (peak Ez 1.9 % 2D, 0.5 % RZ); the Gauss law with plasma streaming through the window (2e-15, also hot
  with iRIP); a ballistic gamma = 20 beam in the lab frame (charge, energy, centre, length); the loading checks of a boosted beam (B10,
  decision 118: error at 0.765 lattice planes, warning at 5.4, silent at 75); field ionization (decision 90):
  nitrogen charge states after a laser, boosted (gamma 2) vs lab run within 0.15 sigma (2D), 0.91 sigma (RZ), 0.40 sigma
  (2D, three levels per ion), Gauss law <= 3e-17; streaming-plasma stability (decision 104): a warm plasma at gamma 10 in
  a periodic box, W_field / K_e after 100 / omega_p 1.2e-3 (2D) / 6.3e-4 (RZ) without and 4.6e-6 / 2.2e-6 with one filter
  pass; the filter with the window of a boosted 2D run: Gauss law 1.6e-15 with the plasma leaving at the left edge (6.4e-4
  without skipping the first filter_passes cells); the
  summary's warning on and off.
- Boosted-frame antenna (tests/test_boost_antenna.cpp, decision 89): a beam focused 70 um downstream in a box that cuts it at
  the boosted entrance plane's lab events: the antenna's lab snapshot against the lab run 0.73 % (2D RIP-CC), 0.44 % (RZ iRIP)
  at lambda/16 (0.11 %, 0.09 % at lambda/32), the entrance plane 3.3 %, 4.4 %; the Gauss law with the moving sheet and
  streaming plasma 2e-15 (2D RIP-CC and iRIP, 3D RIP-CC, RZ iRIP).
  Its near field (A3, bench/antenna_nearfield): the emitted field is within 1 % of the lab run only beyond
  d_1% = 1.1 gamma (1 + beta) c dt' downstream of the antenna (one boosted step after emission; 0.65 at a tenth of it, 0.9 at
  half), the same curve for gamma 5 - 10 and lambda/20 - lambda/80; printed by the boost summary.
`tests/mpi.sh`: 1 vs 8 ranks (periodic, open x with the plasma leaving, moving window, beam self-field,
RZ along z, also the laser through open z and the moving window).
HDF5 check of the Makefile (`tests/hdf5_version.sh`, run by `make test` after build/tests; decision 113): `make -n`
against stand-in installations: 1.10.6 gives the warning "too old (need 1.12 or newer)" and a build without HDF5,
1.14.6 serial adds -DRIP_HDF5 and -lhdf5, a parallel 1.12.0 adds the MPI paths of mpicxx without MPI=1, none with MPI=1.

Regression harness (`make regression`, `docs/specs/regression.md`): small versions of the example
decks (streaming plasma with shapes 1 and 2, laser reflection) compared row by row with the stored
`diag.txt` in `tests/regression/ref/` (energies 1e-9, Gauss law and div B at round-off, particle
numbers exact); `sh tests/regression/run.sh EXE` runs it with any binary, e.g. on a GPU.

## CPU and GPU

`src/portable.hpp` (after GEM-PIC-3D) writes every kernel once as an `[=] HD` lambda that runs as an
OpenMP loop or as a CUDA kernel; fields and particles live in `Buf<T>` device memory for the whole
run. Guard layers are packed on the device and exchanged through host memory; leaving particles are
listed on the device and only they go through the host. Each rank takes the GPU `local rank mod
#GPUs` of its node. Details and the test plan for real GPUs: `docs/GPU_TESTING.md`. On JUPITER / JUWELS
Booster (Slurm, 4 GPUs per node), `bench/jupiter/` builds the code, checks 1 and 4 GPUs against the CPU build
(unit tests, regression cases, every case of `tests/mpi.sh`) and times the 3D bubble and the RZ B7 deck on
1, 4, 16 GPUs (strong and weak scaling, time per operator); its README is the step-by-step guide.

## Agent skill

The repository ships two [agent skills](https://agentskills.io) in [`skills/`](skills/): instructions that AI
coding agents load on demand, so that an agent can build the code, choose the solver variant, and write, check, run
and debug input decks from a request in plain words. The format (`SKILL.md` with a YAML header, plus `references/`,
`scripts/`, `examples/`) is an open standard read natively by Claude Code, OpenCode, Codex, Cursor, Gemini CLI,
Goose, Cline and others, with commercial or open models.

```
npx skills add AlexMPukhov/VLPL          # the `skills` CLI (skills.sh, needs Node.js): choose skills and agents
```

Without Node.js copy the two folders `skills/vlpl-setup` and `skills/vlpl-simulation` into the skill directory of
the agent (`.agents/skills/` in a project for OpenCode, Codex, Cursor and Gemini CLI, `.claude/skills/` for Claude
Code, or the user-wide `~/.claude/skills/`, `~/.config/opencode/skills/`, `~/.agents/skills/`). Then ask for a
simulation, e.g. "LWFA, a0 = 2, 800 nm, n = 1e18 cm^-3, 2 mm" or "a 30 fs pulse at 45 degrees on a 30 n_c foil".

| skill | content |
|---|---|
| `vlpl-setup` | clone, build (`make`: OpenMP, MPI, CUDA, HDF5), the tests, how to run (threads, ranks, GPUs, restarts), where the output goes, cluster jobs |
| `vlpl-simulation` | helps the user run the code for a physics problem: the questions to ask, which variant (VLPL-RIP, VLPL-iRIP, VLPL-RZ, VLPL-iRZ) with the conditions that decide it, the deck in one page, the rules for resolution, stability, particles, filter and the boosted frame as numbers (loading checks, antenna near-field distance), verification steps (`--check`, warnings, smoke run, cost, convergence); the capabilities with one tested deck each (laser wakefield in RZ and 3D, boosted frame, field ionization, collisions, QED and spin, solid target, linear media, dielectric and metal structures with a beam); diagnostics and units; a table of errors, warnings and symptoms; the deck reference; helper scripts (plasma numbers and starting values of the grid, a first look at an output directory, a bunch file). The agent does not modify the source. |

The skills cannot drift from the code: the deck reference in the skill is generated from the sections "Units" and
"Deck" of this README and eleven of the twelve example decks are copies of repository decks
(`sh tools/sync_skills.sh` after editing those; `make test` fails when a copy is stale or a deck key named in the
skills is not read by the parser), and `make agent_skill` runs every deck of the skills with the real executable,
a restart, and the helper scripts against the code's summary. Details: [`docs/agent_skill.md`](docs/agent_skill.md).

## Code layout

```
src/portable.hpp    par_for, Buf<T> device memory, atomics, reductions (CUDA or OpenMP)
src/grid.hpp        Domain (global/local box), Field (3D device array with guards), FV view, for_owned
src/comm.*          MPI Cartesian topology, guard copy / guard add, reductions (stub without -DRIP_MPI)
src/fields.*        RIP-CC solver and field diagnostics
src/stagger.hpp     staggering descriptors (RipCC): the only place with stagger offsets
src/shape.hpp       particle shapes of order 1, 2 and 3
src/slab.hpp        tilted slab with pre-plasma: profile and exact sub-cell means (O4)
src/filter.*        Cart3D current filter that keeps the Gauss law exact (O4)
src/geometry.hpp    Cart3D, Cart2D: field views, gather, Esirkepov and charge deposit (from the descriptor)
src/tiles.*         tile layout, stable sort of a species by tile (tile_start ranges), tile colours
src/particles.*     pushers, push / charge kernels (templates on the geometry)
src/species.*       particle storage: attribute registry (SoA), migration
src/pipeline.*      step pipeline: Stage, Operator, Pipeline
src/laser.*         laser injection through the open x boundary (boundary hook)
src/sim.*           deck -> Config, particle loading, core operators, diag.txt
src/deck.*          input parser
src/window.*        moving window (field shift, entering plasma)
src/expr.*          deck expressions (from GEM-PIC-RZ), e.g. density profiles
src/beam_field.*    steady self-field of a relativistic beam (DFT in y, z + tridiagonal in x)
src/pml.*           CPML absorbing layers in y and z (coefficients, psi arrays, the RIP-CC step with layers)
src/tfsf.*          TFSF x plane with an auxiliary incident grid (O1)
src/tfsf_modes.*    the TFSF x plane with exact discrete plane waves (laser.tfsf_incident = modes)
src/hfocus.*        harmonic_focus: band-pass in time per node, peak and fluence maps (CHF focus; tools/hfocus.py)
src/rz_tfsf.cpp     RZ TFSF plane (vacuum incident FieldsRZ, corrections per mode) and the z absorbing layers
src/rz_metal.*      RZ embedded perfect conductors (block rings: snapping, particle removal; the field rules in rz_fields.cpp)
src/metal3d.*       Cart3D embedded perfect conductor (block metal: node mask, the rules after each stage of the step, absorption)
src/external.hpp    external quadrupoles in the particle push (edge-averaged); src/beam_import.* openPMD beams (species.file)
src/wake.*          wake, beam_slices, side_flux diagnostics; src/structure.cpp their deck blocks and setup
src/rz_dielectric.* RZ dielectric blocks (chi per component, the polarization-current line solve, the window entry)
src/rz_beam_field.cpp  RZ self-field of a beam (species.self_field)
src/side*.{hpp,cpp} side injection: the TFSF tube with faces y, z and exact discrete plane waves; the flying focus
                    (side_luminal.cpp: luminal Bessel pulse, tabulated in the window frame)
src/beam_diag.*     beam diagnostics: charge, energy, emittance, spectrum, particle output (block `diagnostic`)
src/openpmd.*       openPMD output and checkpoint/restart (HDF5 through rank 0, per-rank reads)
src/h5.*            thin RAII layer over the HDF5 C API
src/laser_file.*    laser from a LASY envelope file (reading, thetaMode resampling, exact propagation)
src/yee.cpp         Yee solver (the reference for comparisons; periodic boxes)
src/irip.cpp        iRIP, box closure (implicit RIP-CC without the Courant limit; ranks along x)
src/irip_pml.cpp    iRIP with CPML layers: the pass with the layers, the eigenbasis of a pml axis
src/implicit_mover.* O6: the energy-conserving implicit particle mover (numerics.mover = implicit)
src/bloch.hpp       O7 model = bound: the node step of the V-type Maxwell-Bloch atom (Cayley, pre-warped lines)
src/ionization.*    field ionization: ADK / Tong-Lin rates per level, the Monte-Carlo step, new electrons, the ionization current,
                    dir/ionization.txt; src/atomic.* the element table (data/ionization_energies.def, NIST ASD 5.12, H .. Lr; tools/make_ionization_table.py)
src/qed.*           spin-resolved QED: T-BMT, Monte-Carlo photon emission with spin and Stokes vector (qed.hpp kernels),
                    the qed operator, photons, dir/qed.txt; qed_tables.cpp the Bessel and rate tables, Breit-Wheeler pairs
                    (qed_pair, bw_step)
src/collisions.*    binary collisions: cell lists, pairing, the Debye length, the operator, dir/collisions.txt (collision_kernels.hpp:
                    the pair collision, Nanbu's angle); src/impact.* impact ionization (RBEB tables, data/binding_energies.def)
src/medium.*        O7: linear media without particles (blocks medium: build, current, GMRES); medium_rz.cpp per mode,
                    medium_sim.cpp the checks, the setup and dir/medium.txt
src/rz*.{hpp,cpp}   RZ with azimuthal modes: fields per mode, RIP-CC and iRIP steps, particle kernels; rz_smooth.cpp the transverse particle shape (decision 79)
                    (gather, push, charge-conserving deposition per mode), the deck path (rz_sim.cpp),
                    the laser projected on the modes (rz_laser.cpp)
src/pushers.hpp     the momentum pushers (Boris, Higuera-Cary), shared by Cart3D and RZ
src/fsutil.hpp      mkdir -p, directory listing, text logs that continue after a restart (POSIX)
tests/              tests.cpp, test_*.cpp (window, profile, beam, diag, pml, laser, openpmd, lasy, yee, irip, rz), mpi.sh, regression/
examples/           streaming_plasma.deck (numerical Cerenkov test of the paper, Sec. 7.1), rz_*.deck (RZ),
                    laser_reflection.deck (narrow pulse on a tilted overdense layer),
                    lwfa_window.deck (laser wakefield in a moving window), lwfa_beam.deck (with an
                    externally injected beam), beam_vacuum.deck (beam with its self-field),
                    solid_target.deck (O4: tilted 30 n_c foil with pre-plasma, TFSF laser at 45 deg, shape 3, filter),
                    qed_collision.deck (spin-resolved QED: 1 GeV electrons with spin through an a0 = 100 pulse),
                    collisions_2d.deck, rz_collisions.deck (collisions and impact ionization in a dense plasma)
                    qed_cascade.deck (Breit-Wheeler pairs: 10 GeV electrons through an a0 = 80 pulse, chi_e up to 9.5)
skills/             agent skills vlpl-setup, vlpl-simulation (docs/agent_skill.md); tools/sync_skills.sh, tests/agent_skill.sh
tools/movie.py      mp4 movie of the slices
tools/make_ionization_table.py  builds data/ionization_energies.def from the NIST ASD downloads (queries in the file)
tools/ionization_ref.py  independent ADK rates and rate equations in Python (references of tests/test_ionization.cpp)
tools/make_binding_table.py  builds data/binding_energies.def (orbital energies) from Smilei 5.1's IonizationTables.h
```

## Status and validation

VLPL is research software under active development; this repository holds snapshots of the development tree. Every
feature comes with an automatic test against an independent reference (an exact solution, a conservation law, another
code or a stored benchmark curve with a tolerance).

- **Unit and physics tests** (`make test`, `tests/`): exact charge conservation of RIP-CC and iRIP (the Gauss law at
  round-off with plasma entering and leaving the box, in 3D, 2D and RZ, with filters, moving windows, the boosted frame,
  metal and dielectrics), dispersion relations of the solvers, pushers against analytic orbits, ionization rates,
  collision rates, QED rates against tabulated values, restart = uninterrupted run.
- **Regression harness** (`make regression`, `tests/regression/`): stored diagnostics of reference decks with tolerances.
- **MPI** (`sh tests/mpi.sh <mpi build>/vlpl`): 1 rank against N ranks for every solver, geometry and module, and
  restarts on a different number of ranks.
- **GPU**: the CUDA build (one source with the CPU build) passes the unit tests and the regression harness on an RTX 5090
  (GPU = CPU to round-off); multi-GPU runs and scaling on JUPITER / JUWELS Booster (GH200, A100): `bench/jupiter/`.
- **Benchmarks against other codes and theory** (`bench/`, each with a README, the decks and the analysis scripts):
  laser wakefield acceleration and numerical emittance against FBPIC (`lwfa_fbpic`, `lwfa_bubble`, `emittance`),
  field ionization, collisions and TNSA against Smilei and EPOCH (`ionization_smilei`,
  `collisions_smilei`, `collisions_epoch`, `tnsa`), high harmonics from oblique incidence against Smilei
  (`hhg_oblique`, `rom_harmonics`), the numerical Cherenkov instability in the boosted frame (`nci`), QED cascades and
  beam-plasma QED (`qed_cascade`, `qed_pwfa`), dielectric structures against modal theory (`rz_dielectric`), flying
  focus, noise heating and the lab-frame antenna (`flying_focus`, `noise_heating`, `antenna_nearfield`, `chf`).
- **Site scripts**: the Slurm and run scripts in `bench/` and `research/` were written for the developers' machines;
  `$WORK` stands for the work directory there and `<partition>` for a Slurm partition. The results in the READMEs were
  obtained on a CPU cluster (Ivy Bridge and EPYC nodes), an RTX 5090 workstation and JUPITER / JUWELS Booster.
- **Design notes**: one specification per module in `docs/specs/` (equations, discretisation, tests) and the reasons
  behind the conventions in `docs/decisions.md`.

Cite the software as A. Pukhov, VLPL, Zenodo, doi:10.5281/zenodo.24503313 (all versions; v1.0.0: doi:10.5281/zenodo.24503314),
see CITATION.cff. The method papers: A. Pukhov, J. Comput. Phys. 418, 109622 (2020) (RIP); A. Pukhov, T. C. Wilson, the iRIP paper (in
preparation), whose verification scripts, decks and figure scripts are in `bench/irip_paper/`.

## Known limitations and next steps

- Open x boundary: waves leave exactly at normal incidence (inbound T = 0). Particles leave with
  exact currents (kept one cell beyond the lower edge, `docs/specs/open_boundary.md`), so the Gauss
  law stays at round-off. y and z: periodic or CPML layers (`docs/specs/pml.md`).
- Moving window: at v = c the Gauss law is exact if the window starts at t = 0 or the plasma begins
  ahead of the box (`docs/specs/moving_window.md`); freshly entered plasma is held for 3 steps.
- Boosted frame (decision 87): lasers enter at the boosted entrance plane, i.e. at lab events x in [X0, 0] with
  |X0| = gamma beta gamma (1 + beta) window.start, where a beam focused far downstream is wider (the summary warns
  when the transverse box cuts it); `laser.injection = antenna` (a lab-fixed current sheet, decision 89) removes this; its
  field is weak within d_1% = 1.1 gamma (1 + beta) c dt' of lab x downstream of it (the summary prints it: place collisions
  and other interactions beyond, bench/antenna_nearfield). Not
  yet: LASY files, Cart3D TFSF, side tube, media, the implicit mover. Checkpoints hold the partial lab snapshots (decision 116). Field
  ionization (decision 90): ionization.txt is in the boosted frame, and its column W_field + W_kin + E_cur is no energy
  check there. Hot plasma with
  RIP-CC: the Gauss law is exact only once the window moves (the lab window's limitation); iRIP and cold plasma: exact.
- I/O: serial HDF5 through rank 0 (parallel HDF5 when the output time at many GPUs matters).
- Yee: periodic boxes only so far (no open x, laser, window, CPML); Mur and a Yee CPML when a
  comparison needs them.
- iRIP: ranks along x only so far; RZ follows. The y, z transforms are fast only for powers of two
  (other sizes: a direct O(n^2) sum); with CPML a pml axis uses a dense O(n^2) transform.
- Next: relativistic beam test of the co-located transverse force (paper Fig. 4/5).
- Performance: the deposition uses atomics (about 0.2 µs per particle and step on 2 CPU cores);
  particle sorting/tiling, CUDA-aware MPI and merging the four guard exchanges are the next steps.
