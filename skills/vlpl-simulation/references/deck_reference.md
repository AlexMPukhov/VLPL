<!-- Generated from README.md (sections "Units" and "Deck") by tools/sync_skills.sh. Do not edit this copy. -->
# VLPL deck reference

All blocks and keys of the input deck with their defaults, as in the README of the repository. Paths such as
`docs/specs/<name>.md`, `examples/<name>.deck` and `tools/<name>.py` refer to the VLPL repository; "decision N" is
an entry of its `docs/decisions.md`. The file is long: search it for the block or key you need.

Notation: `block { key, key = default, a|b ... }` lists the keys of a block with defaults and choices, and `nx ny nz`
stands for three keys. It is not deck syntax: in a deck write one `key = value` per line, lengths, times and angles
with a unit (`um`, `fs`, `deg`, ...), densities in n_c without a unit.

## Units

c = 1, lengths in 1/k0, times in 1/ω0 (λ0 = `grid.lambda0`), momenta u = p/(mc) = γv, fields in
m_e c ω0/e, densities in n_c, charge in e, mass in m_e. A macro-particle of weight w stands for
w n_c/k0³ particles, so ∇·E = ρ, ∂_t E = ∇×B − J, and ω_p² = n/n_c.

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
            decision 99),
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
            T^N ((N + 1) - N T), flat to O(k^4) at the laser; Gauss law exact; decision 108),
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
                                     # examples/qed_collision.deck
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
                                    # summary warns; decision 104)
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
                                     # or eps = <expression of x, y, z, r, phi with units> (e.g. 1 + 1.5 * (1 - (r / 0.5 mm)^2);
                                     # with x, y or phi chi lives on the 2M - 1 angles),
                                     # then the rectangle keys are an optional bounding box; sampled 8 x 8 per control volume,
                                     # chi = eps - 1 added; metal wins; time-centred polarization current (a tridiagonal line
                                     # along z per row, stable and lossless for any eps), div D = rho at round-off; cut cells
                                     # sub-cell averaged; ramp: linear z ends; with a moving window the last 32 cells at the front
                                     # edge absorb (keep beams behind them); rip_cc only (docs/specs/rz_dielectric.md, decision 100)
                                     # (a warning, at the start and during the run, when charged particles are inside it);
                                     # several blocks combine (any union of rectangles); with metal: the rings block or the
                                     # metal block (also depending on phi: the lines solved on the 2M - 1 angles);
                                     # examples bench/rz_dielectric/dwa.deck (a lined tube, the wall the
                                     # grid edge), cherenkov.deck
metal     { function, phi_derivative = difference|spectral, enabled = true }   # RZ: conductors that depend on the azimuth:
                                     # metal where function(x, y, z, r, phi) > 0 (lab frame, also with the window), on the
                                     # 2M - 1 angles of the modes; difference (default): the phi derivative becomes the
                                     # staggered difference over 2 pi / (2M - 1) (i m -> i m_eff, a Yee grid in phi), a
                                     # staircase in r and phi, z faces as rings; couples m and m +- 4 for a C4 shape; spectral: axisymmetric masks only (=
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

Structure runs in Cart3D (decision 102, docs/specs/structure3d.md): a bunch through a metallic structure of any 3D shape,
inside external quadrupoles, with the wake diagnostics. Files written for z-propagating codes map (x, y, z) -> (y, z, x).
```
metal     { function = f(x, y, z) }  # perfect conductor where f > 0 (absolute coordinates, lab-fixed, also with the window):
                                     # staircase on the RIP-CC grid (tangential E = 0, total reflection at x faces through the
                                     # RIP transport, the Gauss law exact off the surface), particles inside absorbed; rip_cc only
dielectric [<name>] { eps }          # lossless dielectric, Cart3D and 2D (docs/specs/dielectric3d.md, decision 100): eps a number
                                     # (everywhere) or an expression of x, y, z (lab frame, units), blocks add their eps - 1;
                                     # time-centred polarization current (lines along x, ranks along x by a partition solve),
                                     # stable and lossless for any eps, div D = rho at round-off; chi per component from 4^3
                                     # sub-samples (harmonic along its direction); with the metal block; CPML; with a moving
                                     # window the last 32 cells at the front edge absorb (a warning for charged particles in them);
                                     # rip_cc only, no side tube, no boosted frame
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

