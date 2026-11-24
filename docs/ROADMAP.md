# VLPL strategic plan (2026-10-26)

Goal: a production 3D and quasi-3D (RZ) electromagnetic PIC code with an X-dispersionless field
solver family, multi-species atomic and QED physics, CPU and multi-GPU, developed "zero-touch" by
Claude. Alexander's time is spent only at milestone gates (physics review of a one-page report).
Applications (Alexander, 2026-10-28): relativistic laser-plasma interaction in general, not only plasma
accelerators: also overdense targets, high-harmonic generation (HHG) and coherent harmonic focusing (CHF),
and ion acceleration from foils (TNSA); plasma mirrors and optics as particle-free linear media (O7). Track O below; physics constraints in `docs/specs/overdense.md`.

## 1. Principles that keep the development cheap

1. **The test suite is the specification.** No feature is merged without an automatic test against
   an analytic result, a conservation law, or a published benchmark. Tests are the only thing that
   lets an agent change code it did not just read. Fast tests (< 1 min, CPU) run on every change;
   physics benchmarks (minutes to hours, GPU) run nightly and produce a pass/fail table.
2. **Design the data layouts once, before the physics piles up.** The expensive rewrites are
   geometry (3D vs RZ modes), particle attributes (ionization state, spin, ids), and tiling for GPU
   performance. These interfaces are fixed in Phase 0; physics modules are then plug-ins.
3. **Small modules, short files, one feature per session.** Each module (< ~600 lines) has its own
   spec (`docs/specs/<feature>.md`, half a page: equations, discretisation, tests, references) and
   its own tests. A session reads `CLAUDE.md`, `docs/STATUS.md`, the spec and the files it touches,
   nothing else.
4. **State lives in the repository, not in conversations.** `docs/STATUS.md` (what works, what is
   next, open problems) is updated at the end of every session; design decisions go to
   `docs/decisions.md` with their reason, so nothing is derived twice.
5. **GPU first, optimise late.** Every kernel goes through `portable.hpp` from day one, so there is
   never a porting phase; performance tuning comes after the physics of a phase is verified.
6. **Reuse what is proven.** Own code from GEM-PIC-3D / GEM-PIC-RZ / QUARZ (deck, portability layer,
   azimuthal-mode machinery, openPMD rectilinear-grid convention, stretched-grid rules); external,
   permissively licensed, well-tested libraries where they save a large verification effort
   (openPMD-api with HDF5/ADIOS2 for I/O, PICSAR-QED (BSD, header-only, GPU) for QED rates and
   tables). Each external dependency stays optional behind a thin interface.
7. **Division of work.** Cloud sessions with Alexander: physics, specs, review, papers. Team agent
   (Claude Code on the CPU cluster / the GPU workstation): implementation, long tests, GPU and scaling runs, through
   `HANDOVER.md` files.

## 2. Architecture targets (fixed in Phase 0)

- **Geometry** template parameter: `Cart3D` and `RZModes` (complex azimuthal modes m = 0..M-1,
  longitudinal axis x -> z). Fields, deposition and gather are geometry-specific; particles,
  physics modules and I/O are geometry-agnostic.
- **Field solver** interface with a declared staggering: `rip_cc` (explicit, now), `irip` (implicit,
  box / Crank-Nicolson closure, decision 17, `docs/specs/implicit_rip.md`: exactly energy and charge
  conserving; lifts h_perp >= c dt; needed for fine radial grids and many modes in RZ),
  `yee` (reference for comparisons and papers), optionally `ndf` (VLPL). Gather and Esirkepov
  stencils are generated from the staggering descriptor.
- **Species**: structure of arrays with a registry of attributes (x, u, w always; id, charge state,
  spin vector, optical depths for QED, tracking flags on demand). Kinds: charged particle, ion with
  charge state, photon (neutral, ballistic, optional polarisation). Particles kept in tiles sorted by
  cell (GPU deposition without global atomics, cheap binary collisions per cell).
- **Step pipeline** of operators: inject -> ionize -> push (+RR) -> QED emission / pair creation ->
  deposit -> fields -> collisions -> migrate -> boundaries -> diagnostics. Each physics module is
  one operator with its own deck block.
- **I/O**: openPMD (fields incl. non-uniform grids, particles, checkpoints); restart bit-exact.
- **Input**: current deck format plus an expression parser for profiles (as in the GEM-PIC-RZ patch).

## 3. Phases, milestones and exit criteria

| Phase | Content | Exit criterion (automatic unless noted) |
|---|---|---|
| **0 Foundation** (2-3 sessions) | private GitHub repo, CI (CPU + MPI on every push; GPU nightly on the GPU workstation), `CLAUDE.md`, `STATUS.md`, specs template, ctest-style regression harness with tolerances; architecture refactor (geometry/solver/species/operator interfaces, tiles); first real GPU run | all current tests green on CPU, MPI, 1 GPU, 4 GPUs; GPU == CPU to round-off |
| **1 Production core, 3D LWFA** (8-12) | moving window (natural: one cell per step), plasma injection, PML in y/z, exact open-boundary particle currents, Poisson init, laser modes (Gaussian/Laguerre, from file), openPMD output + checkpoint/restart, diagnostics (spectra, probes, tracking), Yee as reference solver, sorting, CUDA-aware MPI, merged halos | benchmarks: NCI and emittance tests of the 2020 paper reproduced with RIP-CC vs Yee; LWFA bubble vs FBPIC; restart bit-exact; scaling on 16 GPUs. **Gate A** (Alexander): code paper RIP-CC |
| **2 Solvers and geometry** (8-10) | iRIP, box closure (transverse eigenbasis: RZ per-mode dense transform, 3D FFT/DST, CG with PML; batched + partitioned tridiagonal along x), stretched transverse grids, boosted frame, quasi-3D RZ with azimuthal modes for rip_cc / irip / yee | RZ vs 3D agreement for LWFA; iRIP stable and energy conserving at dx/h_perp = 10; standard Gauss law to round-off; no reflections from stretched grids (radiation-cone test). **Gate B** |
| **O Overdense targets and harmonics** (6-8; after the iRIP RZ part of Phase 2, parallel to 3) | O1 total-field/scattered-field (TFSF) laser source: oblique incidence in 3D, normal incidence in RZ; O2 reflected-field diagnostic at the x boundaries (time series of the outbound transport components, harmonic spectrum, transverse profile); O3 absorbing x boundaries for oblique waves (CPML along x for rip_cc / irip); O4 solid-density targets: tilted slabs with pre-plasma, ions, shape 2-3, the binomial current filter for irip, a check of omega_p dt < 2 at the peak density; O5 dispersion-corrected closure option for irip (paraxial term, `docs/specs/overdense.md`); O6 research: an energy-conserving implicit particle mover with the box closure (lifts omega_p dt < 2, suppresses grid heating at unresolved Debye lengths; prototype and options 2026-11-06: `docs/specs/implicit_mover.md`); O7 linear-response media without particles (cold Drude plasma, lossy plasma, Lorentz dielectrics) for irip (implemented 2026-11-07, decision 84): Crank-Nicolson current driven by the box-closure work field, exactly energy conserving and unconditionally stable for any omega_p dt, Krylov solve preconditioned by the vacuum box solver (`docs/specs/linear_medium.md`; prototyped in 2D 2026-10-28); O7b conformal (cut-cell) medium weighting for surfaces tilted against the grid; O7c bound electrons of a gas as Maxwell-Bloch atoms (`model = bound`, Rb D1 + D2, relaxation, ADK ionization into species, RZ, checkpoints; implemented 2026-11-08, decision 85) | 1D-like ROM harmonics: reflected spectrum converges in dx and follows the relativistic-spikes power law (Baeva et al. 2006) up to the resolved order; oblique-incidence HHG in 3D vs a reference code (Smilei or EPOCH, same setup); CHF in RZ and 3D: focal intensity and spot converge in dx and dr, energy balance (absorbed + reflected + transmitted) to 1e-3; TNSA: proton cutoff energy vs Smilei/EPOCH within 10 %; O7: energy balance W + K + dissipated to the solver tolerance, |r| = 1 for lossless overdense layers at any omega_p dt, oblique Fresnel reflection/absorption and Snell refraction second order in dx (s-pol and grid-aligned surfaces), parabolic-mirror focus position, Gauss law with particles + medium to round-off. **Gate O** |
| **3 Atomic physics, many species** (6-8) | ions with charge state, field ionization (ADK/PPT, multi-level, energy-conserving current; implemented 2026-11-08, decision 86, `docs/specs/field_ionization.md`), binary collisions (Nanbu/Pérez, relativistic, inter- and intra-species, Coulomb log) and collisional ionization (implemented 2026-11-10, decision 94, `docs/specs/collisions.md`), NIST data tables | Smilei/EPOCH ionization benchmarks, temperature relaxation and slowing-down vs theory, conservation of momentum/energy in collisions. **Gate C** |
| **4 QED** (8-10) | photon species, radiation reaction (Landau-Lifshitz, quantum-corrected), Monte-Carlo photon emission (LCFA), Breit-Wheeler pairs, spin: T-BMT precession for all species, spin- and polarisation-resolved emission and pair creation (Seipt / Li et al.), optional Bethe-Heitler and trident (implemented: spin, photons, emission 2026-11-09, decision 93; Breit-Wheeler pairs with photon polarization and pair spins 2026-11-10, decision 96; `docs/specs/qed.md`) | photon spectra vs synchrotron function, pair yields vs published cascades, radiative polarisation (Sokolov-Ternov) build-up, energy balance to 1e-3. **Gate D** |
| **5 Hardening and release** (continuous) | dynamic load balancing, performance on JUPITER (GH200), user manual, examples gallery, input validation, v1.0 release (Zenodo), code paper(s) | documented examples run unchanged; weak scaling >= 80 % to 256 GPUs |

Session numbers are rough: one session is one focused feature with tests. Phases 3 and 4 can
partly run in parallel once the species and operator interfaces of Phase 0 exist.

## 4. Verification ladder (every physics feature climbs it)

1. Unit test against an exact result (single particle, single mode, conservation identity).
2. Convergence test (rate in dx, dt, ppc).
3. Published benchmark (paper figure or other code), compared automatically with a stored curve.
4. Physics review at the gate: one page with plots and a movie, read by Alexander.

## 5. Risks

- **RZ with RIP:** explicit RIP-CC needs h_r >= c dt; fine radial resolution needs iRIP first.
  Resolved in the prototype (2026-10-26, `research/irip_rz/`): the box iRIP is unconditionally stable
  for any number of modes; the cost is the x-coupled implicit solve, O(Nr^2) per slice and mode.
- **Harmonics focused at large angles (track O):** the box-iRIP phase error of harmonic N at angle alpha is
  about -(N k0 dx)^2 sin^2(alpha)/6, set by c dt, not by h_perp (refining the transverse grid does not help).
  CHF needs dx <~ lambda0/256 for N ~ 20, or the dispersion-corrected closure (O5).
- **Solid densities:** omega_p dt < 2 ties dx = c dt to the density (dx < lambda0/(pi sqrt(n/n_c))); grid
  heating at unresolved Debye lengths. Mitigation: shapes 2-3, filter, later the implicit mover (O6).
- **Linear media on tilted surfaces (O7):** p-polarised light on surfaces tilted against the grid couples into
  surface waves at the staircase, a first-order error (2D prototype, 100 n_c, 22.5 deg: 3 % at lambda/20, 1.7 % at
  lambda/40; s-pol and x-aligned surfaces are second order). Mitigation: conformal weighting (O7b), a short
  physical density ramp, resolution.
- **Collisions / QED on GPU:** need per-cell particle lists -> tiles are a Phase 0 prerequisite.
- **Agent context limits:** mitigated by small modules, STATUS.md, specs, and tests as the oracle.
- **No GPU in the cloud sessions:** GPU correctness is checked nightly on the GPU workstation / JUWELS; a GPU
  regression blocks the next phase.

## 6. Decisions requested from Alexander (defaults in brackets)

1. Repository: private GitHub `AlexMPukhov/VLPL`, public at the first code paper [yes]; licence [BSD-3].
2. External dependencies allowed as optional: openPMD-api (HDF5/ADIOS2), PICSAR-QED [yes].
3. First application driving Phase 1 priorities [LWFA with external injection and beam quality];
   second application track (Alexander, 2026-10-28): overdense targets, HHG/CHF and TNSA (track O).
4. Target machines in priority order [the GPU workstation for correctness, JUWELS Booster / JUPITER for production].
