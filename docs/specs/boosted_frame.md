# Lorentz-boosted frame

Status: implemented (decision 87, branch `boosted-frame`; the lab-fixed antenna decision 89, branch `boost-antenna`; field ionization decision 90, branch `boost-ionization`); physics choices
provisional, for Alexander's review.

## Physics and equations
The run is done in a frame S' moving at beta c along +x (RZ: +z) with respect to the lab S (Vay 2007). A laser of lab
frequency omega0 becomes omega' = gamma (1 - beta) omega0, a plasma at rest becomes a plasma streaming at -beta c with
density gamma n, and the interaction length of an LWFA contracts while the laser stretches. With the window moving at c
in both frames the number of steps falls by about gamma^2 (1 + beta)^2 for the same cells per laser wavelength
(Vay 2007; Martins et al. 2010).

Events, code units (c = 1):

    x = gamma (x' + beta t') + X0,   t = gamma (t' + beta x') + T0,   X0 = T0 = -gamma beta t'_ref,

t'_ref the boosted time at which the box's left edge x' = 0 reaches the lab plane x = 0: the window start (boosted), or
the end of the laser injection without a window. Consequences:
- The lasers are injected at lab events x <= 0 (no plasma of the lab deck there).
- After the window starts, the box's left edge is the lab light line t - x = window.start and its right edge
  t - x = window.start - L: the boosted box covers exactly the lab moving window (L the lab box length).

Fields: E'_y = gamma (E_y - beta B_z), E'_z = gamma (E_z + beta B_y), B'_y = gamma (B_y + beta E_z),
B'_z = gamma (B_z - beta E_y), E_x, B_x unchanged (RZ: r, phi for y, z, per azimuthal mode). Momenta:
u'_x = gamma (u_x - beta gamma_p), gamma'_p = gamma (gamma_p - beta u_x). Densities: n' = gamma (1 - beta v_x) n.

## Discretisation and inputs (the deck stays in the lab frame)
- **Grid**: dx' = gamma (1 + beta) dx_lab (co-moving length, as FBPIC's `copropag_length`): the same cells per Doppler-shifted
  wavelength and per wake period; transverse cells unchanged. RIP-CC keeps c dt' = dx'. RIP-CC needs
  dx'^2 (1/dy^2 + 1/dz^2) < 1 (3D, 2D) and in RZ dz' well below dr: for gamma >~ 2 iRIP is the solver of choice (no Courant
  limit, also dispersion-free along x).
- **Time**: `grid.t_end` is a lab time; the run ends two steps after the lab time t_end has left the box at its left edge
  (then every lab snapshot up to t_end is complete). `window.start` (lab) -> gamma (1 + beta) start: the end of the injection
  at x' = 0 maps as the retarded time t - x.
- **Laser** (`LaserConfig::field_boosted`): the inbound characteristic at the boosted entrance plane x' = 0 is
  E' + B' = gamma (1 - beta) (E + B)_lab, evaluated at the lab event of (x' = 0, t') with the paraxial beam at the plane x
  (`LaserConfig::field_at`: focus distance x - x_f, retarded time t - x). Frequency, duration and Rayleigh length transform by
  themselves. Focus and delay refer to the lab plane x = 0. Open-edge injection in Cart3D, 2D and RZ, the RZ TFSF plane
  (`tfsf_x` scaled as a box length).
  Constraint: the injection runs over lab x in [X0, 0], |X0| = gamma beta gamma (1 + beta) window.start; the converging
  beam there is wider (w0 sqrt(1 + ((x_f - X0)/z_R)^2)) and must fit the transverse box. The summary warns when the box
  holds the beam at its focus but not at its entrance events (`boost_beam_cut`, test B5); a beam wider than the box already
  at its focus (a plane wave over a periodic y) is the deck's choice, the same in the lab run, and not warned about.
  `laser.injection = antenna` removes this constraint (below).
- **Antenna** (`laser.injection = antenna`, src/boost_antenna.cpp, decision 89; provisional): a current sheet fixed at the lab
  entrance plane x = 0, as the antennas of WarpX and FBPIC, which moves at -beta c through the boosted box. The lab run of the
  same deck injects at x = 0 as before (the antenna is that plane), so focus and delay keep their meaning.
  - Lab source: J_perp = K delta(x), K = -2 E_perp,laser(t, 0, y, z) (the tangential B of the forward beam fixes K:
    B_z(0+) = -K_y / 2, paraxial B_z = E_y). It radiates the beam forwards and its mirror image backwards (out through the
    open left edge). Its charge: d sigma / dt = -div_perp K, i.e. a polarization sheet P = Pi delta(x), Pi = int K dt,
    rho = -div P, J = dP/dt.
  - Boosted frame: x'_a(t') = -X0 / gamma - beta t', Pi~(t') = Pi(t_lab(t')) on the antenna's world line,
    rho' = sigma~ delta(x' - x'_a), J'_perp = dPi~/dt' delta, J'_x = -beta rho' (the convection of its charge),
    dPi~/dt' = K(t_lab) / gamma (time dilation). In the boosted frame the forward wave carries gamma (1 - beta) of the lab
    field and the backward wave gamma (1 + beta) of it (Doppler; it leaves through the open left edge).
  - Discretisation (exact discrete continuity with the Gauss stencil of RIP-CC / iRIP, as the media's -div P):
    Pi^{n+1} = Pi^n + (dt / gamma) K(t_lab(t'_{n+1/2})) on the transverse plane (state; restarts re-sum it from step 0);
    sigma^n = -div_perp Pi^n at the rho points; w^n_i the (stretched, below) spline of x'_a(t_n) at the rho nodes x_{i+1/2};
    rho^n = sigma^n w^n_i / h_x; J_perp = (Pi^{n+1} - Pi^n) / dt (w^n_i + w^{n+1}_i) / (2 h_x) at i + 1/2;
    J_x(i) = -(sigma^n + sigma^{n+1}) / 2 (C^{n+1}(i) - C^n(i)) / dt with C(i) = sum_{i' < i} w_{i'} at the integer nodes
    (the Esirkepov split in x). Then rho^{n+1} - rho^n = -dt div J identically, and the solver keeps the Gauss law with
    rho = rho_particles + rho_antenna at round-off; compute_rho / rz_rho_all add rho^n. RZ: per mode with the RZ continuity
    (div_n = (1/A_j)(rh_j Pi_r,j - rh_{j-1} Pi_r,j-1) + (i m / r_j) Pi_phi,j; the axis a Gauss point of m = 0 only), the laser
    projected on the modes as rz_laser.cpp. MPI: every rank sums Pi on its transverse plane plus one lower guard row (the
    neighbour's points, wrapped at periodic edges, zero beyond open edges) with the same arithmetic, so Pi is identical on all
    ranks, and adds the sheet's current to its owned cells after the guard sums of the deposit (operator boost_antenna, stage
    Push).
  - The spline weights w: RIP moves the forward wave exactly c dt per step, the sheet moves -beta c dt, so a forward
    characteristic meets the sheet's nodes s = (1 + beta) c dt / h_x cells apart. The quadratic B-spline is stretched to s cells
    and normalised per step (a partition of unity at spacing s: every characteristic gets the weight 1/s; the unit-width spline
    modulated the beam at the grid scale, 7 % error). Its time smoothing (variance 1/4 + (beta / (2 (1 + beta)))^2 steps) is
    pre-compensated: K is used as K - (var / 2)(K_+ - 2 K + K_-) with K_+- one step later / earlier on the world line.
  - Requirement: the antenna is inside the box when the laser starts (lab t = 0): x'_a = gamma beta window.start (lab) <
    box - 6 cells (an error otherwise); it leaves through the left edge when the window starts (the lab light line
    t - x = window.start passes x = 0). RIP-CC and iRIP (not Yee); without tfsf_x.
  - Near field (measured, bench/antenna_nearfield/README.md): the emitted field reaches the lab field only one boosted step
    after its emission. Within d_1% = 1.1 gamma (1 + beta) c dt' = 1.1 (gamma (1 + beta))^2 h_lab of lab x downstream of the
    antenna it is weaker than the lab run's (0.6 - 0.66 at a tenth of d_1%, 0.78 at 0.3, 0.9 at half, 0.97 at 0.8); beyond it
    within 1 %. The deficit is the same curve in x / d_1% for gamma 5 - 10 and lab cells lambda/20 - lambda/80, fixed in distance
    (not in time), and independent of the transverse cell, the particle shape, the current filter (the antenna current is added
    after it) and the solver: the extent of the moving source in the boosted frame. The boost summary prints d_1% per antenna
    laser; interactions (collisions, ionization fronts) belong further downstream, or c dt' must be smaller (QED at gamma 5,
    lambda/20: d_1% = 4.3 um; a boosted RZ LWFA deck at gamma 6, lambda0/32: 3.9 um, inside the first < 0.001 n0 of the ramp).
    Separate effect: a tightly focused beam loses peak amplitude at large c dt' wherever it is (2D, waist 1.5 um, focus 10 boosted
    steps away: -3.3 % at gamma 5, lambda/20, -0.45 % at lambda/40, -0.07 % at lambda/80; waist 3 um: 0.02 %): its angles,
    opened to ~gamma (1 + beta) lambda / (pi w) in the boosted frame, need a smaller c dt'.
- **Plasma** (at rest in the lab, `boost_kind = 1`): a lattice in q = x' + beta t' = (x - X0)/gamma, regular in cells of
  dx' with ppc per cell, comoving with the plasma. The profile, the bounds and `density_function` are evaluated at the lab
  position x = gamma q + X0. Momenta: the lab momentum (u = 0 plus the thermal spread) boosted per particle. Weight: the cold
  density factor gamma (1 - beta v_x) of the drift, the same for every species at a lattice point (neutral loading, exact
  Gauss law).
  - Injection (operator `boost_inject`, after the window): the plane x'_inj = right edge - 3 dx' (window position at step n);
    in step n -> n + 1 the lattice points with q in [q(n), q(n + 1)), q(n) = x'_inj(n) + beta t'_n, enter at
    x' = q - beta t'_{n+1}. A shift of the window and the streaming plasma together bring (1 + beta) dx' of plasma per step.
  - Nothing is frozen: a particle behind the plane ends a step at least 2 cells from the edge, so no current reaches the last
    cell (the argument of the lab window, docs/specs/moving_window.md). The window does not load the entering column.
- **Beams** (`boost_kind = 2`: `self_field` or ux != 0): loaded once at the boosted time of their centre's event (lab
  time `t_start`, default 0), positions by the ballistic map of the lab beam to that boosted time: the lab position at lab time
  t0 is x_e = ax x' + bx with ax = gamma (1 - beta v), momenta boosted per particle, the self-field with the relative gamma
  (the boosted u0). The head and tail are mapped ballistically over gamma beta L_beam' of boosted time.
  Loading checks (decision 118): a beam is a = 1 / (gamma (1 - beta v)) times its lab length in the boosted frame (moving against
  the boost: gamma^2 (1 + beta)^2 shorter in boosted cells), so the lattice holds n = (xmax - xmin) a ppc_x / dx' planes of it.
  n < 1: an error (with the ppc_x needed); n < 20: a warning (charge quantized, ~1/n); after the load the loaded charge is printed
  against the deck's and a difference above 1 % is a warning. Test B10.
- LASY files (decision 97): emitted by the antenna (the file at the lab plane x = 0), RZ, 2D and Cart3D; a moving window is
  required. Lab particle snapshots: `select_uz`, `select_r` record only the particles that pass the cut.
- **Field ionization** (decision 90, docs/specs/field_ionization.md "Boosted frame"): ionizable species stream in like any
  plasma (mobile; ions at rest in the lab). The ADK rate is taken per ion from |E| in its rest frame (from E and B at the
  ion and its momentum: the lab field for an ion at rest in the lab) over its proper time dt / gamma_i; the new electrons
  get the ion's velocity; the ionization current takes (gamma_i + u . S_rest) I_k, gamma (1 - beta) I_k for a forward laser
  (the energy of the rest-frame 4-momentum (I_k, I_k S_rest) in this frame; gamma_i I_k until 2026-11-28), so the Gauss law stays
  exact. The ids of the random numbers mix in the step of the ion's first ionization pass (decomposition-independent).
  `ionization.txt` is written in the boosted frame (boosted step and time; the charge states of the ions in the box).
- **QED** (decision 117, docs/specs/qed.md "Boosted frame"): spin, photon emission and Breit-Wheeler pairs run unchanged
  (chi invariant); photon_min_energy and the energies of qed.txt and the spectra are lab energies, the deck spin a lab
  rest-frame spin; the photon species are empty at the start and not boosted. A beam moving against the boost is
  gamma^2 (1 + beta)^2 shorter in boosted cells: raise its ppc_x by that factor.
- **Streaming-plasma stability** (decision 104, bench/nci/boost_stability/README.md): the plasma streaming at -beta c through
  the boosted grid is NCI-unstable without the binomial current filter on the grids of real boosted LWFAs (h_perp < c dt);
  use numerics.shape = 2 and filter_passes = 1 (2 when the plasma stays omega_p T = k_p L_lab > 100 in the box). The filter
  is allowed with the window of a boosted 2D/3D run (nothing is frozen at the right edge; the Gauss check skips the first
  filter_passes cells at the left edge, where the plasma leaves; the mechanism and the streaming-axis filter of decision 108:
  bench/nci/mechanism/README.md; the injected plasma must be exactly neutral per lattice group, boost_check_neutral:
  species with the same ppc, bounds and slab cancel point by point, each pre-ionized ion species with its own electrons; groups
  may differ in ppc and overlap, decisions 108, 114; it enters 3 + filter reach cells behind the edge, decision 108). The summary warns (boost_nci_risk) when a deck has
  numerics measured unstable.
- Not supported (error): side tube, media, Cart3D TFSF, oblique lasers, `mobile = false` (a lab background moves at -beta c), drifting plasmas,
  the implicit mover, rings.

## Lab-frame diagnostics (block `lab_diagnostics`, src/boost_diag.cpp)
- **Fields** at the lab times T_k (as WarpX's and FBPIC's back-transformed diagnostics): in boosted step n the lab time T is on the
  plane x'_T = ((T - T0)/gamma - t'_n)/beta. The six components are interpolated there (cubic Lagrange between the nodes; all
  RIP-CC / iRIP components sit at integer x) and transformed to the lab; the transverse pairs are co-located (Ey, Bz; Ez, By;
  RZ Er, Bphi; Ephi, Br), so no transverse interpolation. The planes of one snapshot lie dt'/(gamma beta) apart in the lab
  (one per boosted step, about (1 + beta)/beta lab cells) and cover the window. Output `dir/lab_fields/snap_KKK.bin` (float64
  [component][n2][n1][nx]) and `.txt` (lab time, x0, dx, ...); `tools/lab_fields.py` reads them.
  **RZ mode convention:** the RZ snapshots hold the raw mode slots of the grid (slot 0 = F_0, slots 2m-1, 2m = Re F_m, Im F_m) with
  F(phi) = F_0 + 2 Re sum_m F_m e^{i m phi} (rz_modes.md): the m >= 1 slots are HALF the cos / sin amplitudes, e.g. E_x on the axis
  plane phi = 0 of a linearly polarized laser is E_r slot 0 + 2 x slot 1, not slot 1. The openPMD output and the checkpoints carry
  C_m = 2 Re F_m, S_m = -2 Im F_m instead. `tools/lab_fields.py: field_at_angle` returns the physical field at an angle.
- **Particles** of the listed species when they cross the lab time T (`dir/lab_particles/<species>_KKK.bin`, x y z ux uy uz w
  in the lab frame): t_lab = gamma (t' + beta x') + T0 crosses T within the step; the position is interpolated on the step's
  straight move, the momentum is the step's u, boosted back. QED (decision 117): photons are massless (velocity u/|u|,
  u_x = gamma (u'_x + beta |u'|)); species with spin get the extra columns s0 s1 s2 (the lab rest-frame spin), photons xi1 xi2
  xi3 pe0 pe1 pe2 (Stokes vector, lab polarization axis), named in the .txt and kept in the checkpoint (`widths`).
- **Beam moments** in lab quantities from those particles (`dir/lab_beams.txt`: charge, mean energy, spread, emittances,
  centre and sizes), the same moments as `diagnostic` blocks (beam_diag.hpp).
- A snapshot is written when its lab time has left the box at the left edge (complete = 1) or at the end of the run
  (complete = 0). Rank 0 gathers one snapshot at a time (memory of one lab snapshot).
- **Checkpoints and restarts** (decision 116): a checkpoint holds the state of the lab diagnostics (group `rippic/op_lab_fields`,
  `rippic/op_lab_particles`): the status of every snapshot (open, written, incomplete), the planes of the open field snapshots
  gathered over the ranks (with the lab x of every plane), the particle records of the open particle snapshots in rank order with
  the count per rank, and the length of `lab_beams.txt`. A restart continues them: the same number of ranks gives files identical
  to the uninterrupted run (byte for byte on the CPU with deterministic sums, e.g. one OpenMP thread per rank: the RZ deposit
  adds with atomics, so several threads or a GPU differ at round-off even between two uninterrupted runs); another number of
  ranks puts the restored planes and records on rank 0 (particle records in another order; fields and moments to round-off,
  as between rank counts anyway). `lab_beams.txt` is cut back to its length at the
  checkpoint, so lines written by the stopped run after it (e.g. its incomplete snapshots at the end) disappear. From a
  checkpoint without this state (written before decision 116) or with other lab times / species in the deck, the snapshots
  that had begun before the checkpoint are written with `complete 0` (a warning in the log names them), the ones already
  written are left as they are.

## Deck interface
```
boost {
  gamma = 5            # boost along +x (RZ: +z); everything else in the deck is in the lab frame
}
lab_diagnostics {
  t_first   = 1 ps     # lab time of the first snapshot
  t_every   = 0.5 ps   # lab spacing (count > 1)
  count     = 4
  fields    = true     # E, B in the lab frame
  species   = beam     # particles at the lab times, and lab_beams.txt (optional)
  gamma_min = 10       # moments of the particles with lab gamma >= gamma_min
  particles = true     # write the particles (false: only lab_beams.txt)
}
```

## Tests
- Fast (tests/test_boost.cpp, `build/tests test_boost`): see the numbers in decision 87.
  Resolution: the plasma streams into the laser, so the boosted run needs about (1 + beta) times finer cells than a lab run
  for the same ponderomotive drive (lambda/16, gamma 2: wake 15-21 % too strong; lambda/32: 0.5-1.9 %).
  - B1: vacuum laser, the lab snapshot against the lab run on the same lab grid and against the paraxial beam (2D RIP-CC, RZ iRIP).
  - B2: wake (a0 = 1, 0.005 n_c, lambda/32), 2D and RZ, on-axis Ez and laser peak against the lab run; Gauss law.
  - B10 (decision 118): loading checks of a boosted beam: error at n = 0.765 planes (gamma 5), warning and charge off > 1 % at
    n = 5.4 (gamma 2, ppc_x 2), silent and within 1 % at n = 75 (ppc_x 28).
  - B3: Gauss law with plasma streaming through the window (2D RIP-CC cold, 2D iRIP hot, 3D RIP-CC cold); a ballistic beam
    in the lab frame (charge, energy, centre, length).
  - B4: field ionization (decision 90): the rest-frame field against the Lorentz transformation (1.8e-15); a laser through a
    nitrogen slab, charge-state fractions boosted (gamma 2) vs lab within the statistics: 2D 0.15 sigma, RZ 0.91 sigma
    (a0 = 0.0105, N0 / N1+), 2D 0.40 sigma (a0 = 0.03, N2+ / N3+); Gauss law <= 3.4e-17.
  - B5: the summary warning that the box cuts the beam at its entrance events: only when the box holds it at its focus.
  - B6 (decision 104): the warm streaming plasma at gamma 10 (2D, RZ, periodic, 100 / omega_p) unstable without and stable
    with one filter pass; the filter with the window of a boosted 2D run (Gauss law with the plasma leaving at the left
    edge); the NCI warning of the summary.
- Antenna (tests/test_boost_antenna.cpp, `build/tests test_boost_antenna`, numbers in decision 89):
  - A1: a beam focused 70 um downstream (waist 3 um), box +-17.5 um: the lab snapshot of the boosted run (gamma 2) with the
    antenna and with the entrance plane against the lab run and the paraxial beam (2D RIP-CC, RZ iRIP); Gauss law in vacuum.
  - A2: Gauss law with the moving sheet and cold plasma streaming through the window (2D RIP-CC, 2D iRIP, 3D RIP-CC, RZ iRIP).
  - A3: near field (gamma 5, lambda/20, iRIP, 2D vacuum, 4 lab snapshots against the lab run): every point from 1.1 boosted
    steps after emission on within 1 % of the lab run, the deficit inside as measured (0.55 - 0.75 in the first quarter step,
    < 0.97 at 0.5 - 0.75 steps), the same curve in two snapshots (fixed in distance). The scan behind it: `VLPL_NF_SCAN=1
    build/tests test_boost_antenna` with NF_* parameters (bench/antenna_nearfield).
- MPI: tests/mpi.sh case "boosted frame" (examples/lwfa_boost.deck, rz_lwfa_boost.deck, 1 vs N ranks: diag.txt,
  lab_beams.txt, lab fields), also with laser.injection = antenna.
  Field ionization in the boosted frame (decision 90): examples/ionization_boost.deck, rz_ionization_boost.deck (ionization.txt).
- NCI: bench/nci/run_boost.sh (streaming plasma at gamma 10 and 30, RIP-CC and iRIP, 3D and RZ, cold, h_perp >= 3 c dt:
  stable); bench/nci/boost_stability/README.md (decision 104: warm plasma, gamma 2 - 20, h_perp down to c dt / 5, and a0 = 2, 4
  boosted LWFAs vs the lab run): unstable without the binomial filter; shape 2 with one pass (two for a residence
  > 100 / omega_p) stable. B6 in tests/test_boost.cpp.
- Speed-up: examples/rz_wake_boost.deck (decision 87): 4.3x on the same grid at gamma 2, 5.4x at gamma 4 for a 130 um
  wake; the step ratio tends to gamma^2 (1 + beta)^2 for interaction lengths much longer than the box.

## References
- J.-L. Vay, Phys. Rev. Lett. 98, 130405 (2007): boosted frame.
- J.-L. Vay et al., J. Comput. Phys. 230, 5908 (2011); R. Lehe et al., Phys. Rev. E 94, 053305 (2016): NCI in boosted frames.
- WarpX BackTransformed diagnostics; FBPIC BoostConverter and BackTransformedFieldDiagnostic.
- A. Pukhov, J. Comput. Phys. 418, 109622 (2020): RIP, dispersion-free along x.
