# Design decisions (newest last)

Each entry: decision — reason. Check here before changing a convention.

1. **RIP-CC instead of the two-chain scheme of JCP 418, 109622 (2020).** One field set at integer
   times; predictor Ex*, Bx* for the transverse sources, corrector with time-centred transverse
   fields. — The 2020 scheme conserves Gauss / div B only if the half-step fields equal the time
   average; RIP-CC conserves them to round-off for any predictor, has the same dispersion (Eq. 57)
   and stability (Eq. 59), and needs half the memory.
2. **c dt = h_x always.** — The transport is an exact one-cell shift only then (zero dispersion along x).
3. **Charge grid at (i+1/2, j, k); Jy, Jz stored at i+1/2.** — The solver only uses mu_x jy, mu_x jz;
   storing them at i+1/2 makes standard Esirkepov deposition exactly the continuity equation of the
   solver.
4. **Transverse pairs (Ey, Bz), (Ez, By) co-located and gathered with identical weights.** —
   Ey - vx Bz of a relativistic beam is exact on the grid (emittance preservation).
5. **Laser injection through the inbound transport vector**: guard layer i = -1 at t_n holds
   a(t_n + dt). — Exact for the transport; longitudinal fields follow from the solver.
6. **Portability layer from GEM-PIC-3D** (`par_for` lambdas: OpenMP or CUDA). — One source for CPU and
   GPU; no porting phase.
7. **Halo exchange and particle migration through host memory**; only leaving particles go to the
   host. — Works with any MPI; CUDA-aware MPI is a later optimisation.
8. **Initial momenta drawn per global cell** (seed from the global cell index). — Initial state
   independent of the decomposition, so MPI runs are comparable with serial ones.
9. **Repository private until the first code paper; BSD-3 licence; optional external dependencies
   openPMD-api and PICSAR-QED allowed** (Alexander, 2026-10-26, roadmap defaults).
10. **Particles are Cartesian in both geometries; particle axis 0 is the transport axis** (RZModes:
    (z, x, y); r, phi only inside gather/deposit). — Pushers, QED, spin and collisions never see the
    geometry; the axis singularity stays in two kernels. (Spec D-1, `docs/specs/architecture.md`.)
11. **The staggering is a compile-time descriptor declared by the solver** (`src/stagger.hpp`); gather,
    Esirkepov and rho are generated from it, J_a = rho + e_a/2. — One place for offsets; Yee or iRIP
    get a consistent charge-conserving deposition for free. (D-2)
12. **Species: structure of arrays with a registry of named attributes** (F64, F32, I64, I32; core slots
    x, u, w fixed at 0..6). — Migration, sorting, I/O and checkpoints loop over the registry, so new
    physics adds attributes without touching them. (D-3)
13. **Counter-based random numbers** from (seed, step, particle id, operator stream). — Identical
    stochastic physics on CPU, GPU and any rank count; stochastic operators therefore require `id`. (D-4)
14. **Each species is one SoA sorted by tile** (8^3 cells in 3D, 32 x 8 in RZ; `tile_start` offsets). —
    Shared-memory deposition without global atomics per particle, per-cell lists for collisions. (D-5)
15. **Push and move are split only when an ionization or QED operator is active**; otherwise the fused
    push+deposit kernel runs. — The move kernel needs no stored old position; the extra pass costs
    only in runs that need it. (D-6)

Accepted by Alexander on 2026-10-26 (merge of the architecture spec, PR #1).
16. **The tile sort is stable and, for now, sorts on the host** (keys and the permutation of all
    attributes on the device, the counting sort over the keys on the host). — A stable sort keeps runs
    reproducible and independent of the thread count; an atomic device counting sort would not be.
    The device sort (scan) comes with the per-tile deposition, after the real-GPU tests.
17. **The implicit solver `irip` uses the space-time centred ("box", Crank-Nicolson) closure: both movers
    take Ex_bar = mu_x mu_t Ex, Bx_bar = mu_x mu_t Bx; not the characteristic trapezoid.** — The trapezoid
    (right movers (F(i+1,n+1) + F(i,n))/2, left movers mirrored) conserves no energy: transient gain
    ~min(kappa, 2 kappa sin(kx D/2)), up to ~420 for RZ m = 50, and only a modified Gauss law. The box
    closure conserves energy exactly (sup ||M^n|| = 1), the standard RIP-CC Gauss law and div B to
    round-off (so the RipCC descriptor and Esirkepov deposition are reused unchanged), keeps zero
    dispersion along x, is stable for omega_p dt < 2. Price: (I - D^2/4 mu_x^2 L) couples neighbouring x
    slices -> transverse eigenbasis (RZ: per-mode dense transform; 3D: FFT/DST) + batched tridiagonal
    along x, partitioned across ranks. Supersedes "modified Gauss law" in `docs/specs/architecture.md`
    section 2. Spec `docs/specs/implicit_rip.md`; evidence `research/irip_rz/` (Alexander + cloud
    session, 2026-10-26).
18. **CPU current deposit by coloured tiles without atomics** (`numerics.deposit = auto|global|tiled`;
    auto = tiled in the CPU build). Tiles of one colour (parity of the tile coordinates) write disjoint
    cells (reach [c-2, c+3] per particle, tiles >= 6 cells); particles that left their tile since the
    last sort take the atomic path. — No atomics in the hot loop, and a CPU run no longer depends on the
    thread count (bit-reproducible). The GPU keeps the atomic deposit until the shared-memory version is
    tested on a real GPU. (`docs/specs/tile_deposit.md`)
19. **Block-parallel loops with a scratch array (`par_blocks` in `portable.hpp`) for per-tile work on
    the GPU; first use: the box deposit** (`numerics.deposit = box`). — Shared-memory accumulation per tile
    replaces ~144 global atomics per particle by ~3 (T+5)^3 per tile. The CPU fallback (one OpenMP
    iteration per block, nt = 1) keeps the GPU algorithm under the CPU tests. Body and flush are HD
    lambdas like par_for kernels; additions to the scratch use `block_add` / shared atomics. The GPU
    default stays `global` until the box deposit is timed on a real GPU.
20. **`numerics.sort_every = auto` (default): sort by tile every step only when the chosen deposit needs
    sorted particles (tiled, box); with the global deposit no sort.** Supersedes the default 1 of
    decision 14. — The host tile sort cost 1.9 of 2.5 s of a GPU run that used the global deposit (fourth
    GPU check, RTX 5090: 0.6 s without the sort), and the global deposit gains nothing from it (CPU,
    streaming deck 200 fs, 2 threads, global: 15.4 s sorted, 13.9 s unsorted). An
    explicit number still forces sorting (e.g. for cache locality on the CPU).
21. **At an open lower boundary a leaving particle is kept until x < lo - h (`Domain::keep_lo`), then
    deleted; at an open upper boundary it is deleted at x >= hi.** — Deleting at x < lo dropped its
    charge from cell 0, a Gauss point (rho at i + 1/2), without a current: Gauss residual 0.3 there.
    One extra cell lets the Esirkepov current carry the charge out exactly (shapes 1 and 2, the reach
    fits ng = 3). At the upper edge the last cell is no Gauss point and a deleted particle touches
    nothing else. Simpler than depositing a synthetic exit current, and it works unchanged for the
    moving window. (`docs/specs/open_boundary.md`; number 19 is the box deposit, PR #17.)
22. **Moving window: the fields shift in storage by one cell (`Domain::wx` counts the shifts), particles
    keep absolute positions; at v = c the particles of the three entry columns are not pushed in a step
    that ends with a shift (`Domain::xfreeze`).** — An index offset would avoid the copy but every
    kernel would need it; one copy of 6 fields per step is cheap next to the push. At v = c no particle
    moves towards the right edge, and with the entry columns at rest no current reaches the last cell
    (not a Gauss point), so the extended and the truncated update agree and the Gauss law stays exact
    for hot plasma (without: a static residual ~ rho u_th per entering column, 9e-4 at u_th = 0.05).
    Cost: freshly entered plasma is held for 3 steps. (`docs/specs/moving_window.md`)
23. **Deck expressions: the parser of GEM-PIC-RZ (`src/expr.*`, BSD, same author) with VLPL's unit
    table; evaluated on the host when cells are loaded. A density profile is a factor of `density` in
    absolute coordinates and changes the weights, not the lattice positions.** — Reuse instead of a
    new parser (roadmap principle 6); the same syntax as in GEM-PIC-RZ decks. Weights keep the regular
    lattice, so electrons and ions stay exactly neutral and the moving window stays exact.
    (`docs/specs/density_profile.md`)
24. **Beam self-field: solve the discrete RIP-CC Gauss operator dxx/gamma^2 + mu_x^2 (dyy + dzz) for phi
    at the rho points (DFT in y, z, tridiagonal in x), the k_perp = 0 mode apart with Ex0 = 0 ahead of a
    moving beam; one global solve on every rank.** — The discrete operator, not the continuum one, gives
    the Gauss law to round-off; the k_perp = 0 split keeps it at round-off for any gamma (through phi:
    6e-9 at u = 2000) and lets the window keep zero fields at its right edge. Replicated global solve:
    simple and decomposition-independent, memory 8 N bytes per rank (a distributed solve when needed).
    Known: the operator has slowly decaying zigzag tails along x for large gamma (decay 4 / (gamma h_x
    sqrt(K)) per cell), so an under-resolved beam leaves a small field ahead of it (3e-10 at one cell).
    (`docs/specs/beam_self_field.md`)
25. **GPU tile sort on the device: CUB's stable LSD radix sort of (tile key, index) pairs, tile
    starts read off the sorted keys (no atomics), the attribute permutation through one persistent scratch buffer.**
    Supersedes the host part of decision 16. — The host sort cost about 2 s of the 2.6 s of a box
    run (sixth GPU check: keys down, permutation up, one allocation per attribute). A radix sort of
    pairs is stable, so the permutation is the same as the host counting sort's and runs stay
    independent of the scheduling; CUB ships with the CUDA toolkit (no new dependency). The CPU
    build keeps the counting sort. Seventh GPU check: per-tile counts by int atomics cost 121-223 us
    per call (all particles of a tile hit one address) against 12 us for the sort itself, so the
    starts now come from the sorted keys: position q writes start[t] = q for t in (key[q-1], key[q]].
26. **Box deposit in fixed point: int64 box sums with a per-tile power-of-two scale, added as two
    native 32-bit shared atomics with carry.** — The eighth GPU check left the box kernel as the only
    reason box is slower than global (531 against 371 us): FP64 and int64 atomics in shared memory
    are CAS loops (`ATOMS.CAST.SPIN.64`, fifth check), and the threads of a block retry on the same
    few cells. The 32-bit integer add is native. The scale from the bound B = N max|q w| / dV of a
    tile keeps the Gauss law at round-off (tests 15: 3e-15 to 9e-15, as with doubles). Price: the
    error is absolute (B 2^-62 per addend), so a light cell next to heavy ones in the same tile loses relative
    accuracy (1e-6 at a contrast of 1e9, ~1e-12 at 1e3); global and tiled deposits are unchanged.
    Bonus: the box sums are independent of the thread order. One extra reduction (max |w|) per call.
27. **Box kernel register cap: `__launch_bounds__(256, 2)` for shape 1, none for shape 2; strays after
    the box kernel.** — The ninth GPU check found the box kernel 1.20x the global push after the
    fixed-point change. ptxas (sm_120) shows why occupancy matters: uncapped, the shape-1 box kernel
    takes 240 registers, one block of 256 threads per SM (17 %), against 80 and three blocks for the
    global push. A cap of two blocks gives 128 registers with no spills; three blocks (80 registers)
    spill 56 bytes, 120 with the stray deposit still inside the kernel, so strays now go to a separate
    atomic pass (as in the tiled path). Both caps are compile-time options (`RIP_BOX_MINB1`,
    `RIP_BOX_MINB2`) until a GPU timing decides; shape 2 spills 432 bytes at two blocks and stays at one.
28. **Moves of more than one cell are split into straight segments of less than one cell, each
    deposited by the Esirkepov kernel (`cart3d::esirkepov_segments`).** — The kernel's stencil holds
    L = W + 1 nodes per axis, so it is valid for a move of at most one cell; an implicit solver (iRIP)
    allows c dt > h, and particles then cross many cells per step. Each segment conserves charge
    exactly, so their sum does (test 6b: moves of 5.6 cells, continuity at round-off). n = 1 for every
    explicit RIP-CC step (dt <= h), the plain call, bit-identical. Equal straight segments, not cuts at
    the cell faces: simpler, the same on the GPU, and the current of a straight move is the same
    straight path either way. Open for iRIP: the tiled and box deposits require dt <= h (checked at
    setup; long moves need the global deposit), and the guard layers and the MPI migration must cover
    the longest move.
29. **CPU push and deposit: gather weights per axis and offset, pointer strides, shape arrays in
    registers, plain adds of zeros, two passes per chunk of 64 particles; `ARCH` build option.** —
    The push and deposit took 80 % of a step at ~1800 instructions per particle (callgrind), ~560 of
    them index arithmetic. The gather computed the shape weights for every component (18 evaluations);
    with RIP-CC offsets of 0 or 1/2 cell it needs 5. The deposit stored the shape arrays at a computed
    index (stack memory) and branched on zero contributions. The tiled deposit now moves a chunk of 64
    particles before depositing them, which overlaps the latency-bound push of independent particles.
    All bit-identical to the previous code in the default build. Vectorising the push over particles
    (OpenMP simd) was tried and dropped: GCC finds control flow in the gather, and the branch-free
    variant was slower. `make ARCH=-march=native` adds ~10 % to 25 % (FMA changes round-off; the
    regression references pass); the default stays portable x86-64, because the cluster compute nodes
    differ from the head node.
30. **Beam diagnostics as an operator per `diagnostic <name>` block: device reductions with centred
    two-pass moments, text files for moments and spectra, raw float64 per rank for particles.** — The
    LWFA benchmarks need the injected beam (charge, energy, spread, emittance). Two passes (means,
    then Σ w (a - ⟨a⟩)²) because the one-pass form loses the spread of a high-energy beam to
    cancellation (GEM-PIC-RZ's dE/E lesson). Raw binary per rank in index order, not openPMD yet:
    no library dependency, readable by numpy, deterministic per rank. openPMD output (Phase 1)
    replaces it later. The selection is γ >= γ_min only; position windows can be added when a
    deck needs them.
31. **Absorbing transverse boundaries: CPML with κ = 1, α = 0 in layers inside the box, one ψ per
    derivative call site of the RIP-CC step, ψ shifted with the moving window; zero fields behind.** —
    The 2D prototype (`research/pml/`) compared CPML, CPML with graded κ and α, and a matched sponge
    on RIP-CC. Plain CPML reflects 1e-6 to 2e-4 from normal incidence down to 30° from x, and 2-3e-3
    at 20°. κ/α reflect up to 5e-2 at h_⊥ = 5 dx, the LWFA case. The sponge reflects 1e-2 at 45°.
    - Each call site needs its own ψ, because every evaluation updates ψ once per step. So with
      layers the transport sources are computed once per location into scratch arrays; the plain
      step stays unchanged and bit-identical.
    - The Gauss law is not conserved in the layers, and is checked one cell inside them.
    - ψ fields are allocated only on ranks that touch a layer.
32. **Laser modes at the entrance plane: the paraxial Gaussian beam at its distance from the focus,
    separable in y and z, Laguerre-Gauss for round beams, and several lasers summed with their own
    polarisation and phase.** — Every deck so far focused at x = 0, the entrance plane. LWFA needs
    the focus inside the plasma, and structured light needs Laguerre-Gauss. The original expression
    is kept as the exact path for focus 0 and phase 0 (bit-identical decks). The guard layer is zeroed
    once and summed over the lasers. Laser from file (LASY) waits for the HDF5 / openPMD work.
33. **CI on a self-hosted runner (the GPU workstation) for pull requests and pushes to main; MPI on a hosted
    runner for pull requests only; no runs for docs-only changes.** — The free 2000 Actions minutes
    were 90 % used by 2026-10-27. Every push to any branch had run three hosted jobs, twice for a
    branch with an open pull request. The runner on the GPU machine also turns the manual GPU checks
    of the test suite into CI on every pull request. Sessions keep work local and push once per pull
    request (the CI notes of the development repository).
34. **I/O: openPMD 1.1.0 over the HDF5 C API (no openPMD-api), one file per iteration, code-unit
    values with unitSI; output with ED-PIC, checkpoints in base openPMD holding the whole state; all
    blocks stream through rank 0 (serial HDF5); restart bit-exact for the same decomposition.**
    (docs/specs/openpmd.md)
    - The HDF5 C library is on every machine we use (container, the GPU workstation, JUPITER), and openPMD-api
      is a heavy C++ dependency. The subset we write is small, and the openPMD validator checks it
      (0 errors, 0 warnings). openPMD-api 0.17 reads the files.
    - Code units with unitSI keep a checkpoint exact, and readers convert anyway. Output `weighting`
      is the exception: ED-PIC requires real particle numbers with unitSI = 1 (openPMD-viewer,
      FBPIC comparisons). The checkpoint therefore stores the raw weight as base openPMD.
    - A checkpoint is an openPMD file too, readable by the same tools. The state outside the standard
      sits in `/data/<step>/rippic` and in meshes named `rippic_*`: Gauss reference, CPML memories,
      tile starts, loaded flags, window shift, rank boxes.
    - Serial HDF5 through rank 0: works with any HDF5 build, and rank 0 holds at most one block. It
      does not scale to many ranks; parallel HDF5 is the upgrade once the output time at 64 GPUs
      matters.
    - Bit-exact means the same decomposition and a deterministic deposit (CPU tiles). On the GPU the
      atomics sum in another order. Another rank count redistributes the particles, which changes
      the summation order, so the restart agrees to round-off.
    - The restart test found three bugs, fixed here:
      - the CPML transport memory at i = -1 kept a value one cell off after a window shift, with
        6 % error in field energy for a run split along x;
      - leaving particles were strays of the tiled deposit;
      - species loaded later were not sorted.
      The second and third made the CPU deposit thread-order dependent in window runs, against the
      claim "bit-identical for any thread count".
35. **Laser from a LASY file: the envelope at the entrance plane, interpolated per step at the guard
    cells, injected as the inbound transport vector; a file at another plane is propagated there
    exactly (FFT in t, y, z, vacuum) once at the start.** (docs/specs/laser_file.md)
    - LASY's envelope at a plane versus time is exactly what the boundary injection needs, so a file
      at the entrance plane is used without any assumption about the propagation.
    - Propagating once, at the start, keeps the step cost the same as for the built-in lasers: one
      trilinear interpolation of a complex envelope per guard cell.
    - The file grid is resampled (thetaMode) and propagated on the host, then cropped to each rank's
      cells and uploaded. Every rank at the left edge does this itself. That duplicates work, which
      costs little next to a run, and needs no communication. Memory for the padded grid (twice the
      time window, twice the transverse extent) is needed once per such rank at setup.
    - Exact rather than paraxial propagation: the test showed that a short pulse already differs by
      3.5 % from a monochromatic paraxial beam through its bandwidth (each frequency has its own
      Rayleigh length). The FFT propagator keeps that.
    - LASY's x is our y and LASY's y our z, so a LASY profile keeps its orientation about the
      propagation axis.
36. **Moving-window shift as a flat copy of each array, exchange along x only, no generator for a
    cold species.** (docs/specs/moving_window.md)
    - The fourteenth GPU check found the window operator at 60 % of a plain GPU run, 2.8 ms per
      shift. On the CPU it was 16 %, 37 ms per shift on 4 threads.
    - The time went to:
      - an indexed copy (three integer divisions per element, 7 arrays);
      - a full guard exchange in three directions;
      - a Mersenne twister seeded per cell of the entering column, also for cold plasma.
    - x is the fastest index, so the shift is the array moved by one element. The y and z guards move
      with the data and stay valid, so an exchange along x completes it. All bit-identical (diag.txt
      of window runs with cold and hot plasma and with CPML, 1 thread; tests/mpi.sh on 8 ranks).
    - A hot species keeps its per-cell generator: the seeding defines the decomposition-independent
      random numbers. A counter-based generator would be cheaper, but it would change every hot
      deck's numbers.
37. **Yee as a second solver of the same code, B kept at integer time levels by two half steps; periodic
    boxes first.** (docs/specs/yee.md)
    - The comparisons of the code paper (numerical Cherenkov, emittance) must differ only in the
      field solver.
    - The particle kernels were already templates on the staggering descriptor, so Yee only adds:
      - its descriptor (E at edges, B at faces, ρ at nodes, J_a = ρ + e_a/2);
      - the field step;
      - its Gauss residual and div B;
      - a time step in `Domain` (`tau`; 0 keeps dt = h_x for RIP-CC, bit-identical).
    - Two half steps of Faraday's law keep B at integer levels. The push and every diagnostic see
      E^n, B^n as with RIP-CC, and checkpoints stay the same. It is the standard leapfrog, one curl
      more per step.
    - The boundary physics (laser transport vector, CPML, one-cell window, beam self-field) is
      RIP-CC's. A Yee deck that needs it is rejected rather than run with a wrong boundary. Mur and a
      Yee CPML follow when a comparison needs them.
38. **iRIP first as Cart3D on a periodic box and one rank: FFT in y, z, a cyclic tridiagonal solve along
    x, and a second full RIP-CC pass.** (docs/specs/implicit_rip.md, src/irip.cpp)
    - The periodic box is where the transverse operator has an exact, cheap eigenbasis (2D DFT), so the
      scheme is tested without a solver tolerance: dispersion law, energy invariant and Gauss law at
      round-off. MPI along x (partitioned tridiagonal), boundaries (CG with PML) and RZ build on it.
    - The new Ex, Bx come from a second pass of the unchanged RIP-CC kernels with the box averages, not
      from the solve: both movers see the same Ex_bar, Bx_bar, so the Gauss law and div B are exact
      even if the solve were inexact. Cost: one transport + corrector more per step.
    - The own small FFT (radix 2, direct sum for other sizes) avoids a library dependency for the first
      version; cuFFT / FFTW when the timing matters.
    - Guards grow with the largest particle move, 2 + ceil(dt / h_min): with h_perp < c dt a particle
      crosses several transverse cells per step. RIP-CC and Yee keep 3.
39. **iRIP with MPI: ranks along x only, partition method with one allgather per solve.**
    (docs/specs/implicit_rip.md)
    - Whole y, z planes per rank keep the DFT local; splitting y, z too would need a distributed FFT
      (transposes). The production runs (LWFA, beams) are long along x, so x is the axis to split.
    - The inverse of the system along x decays like (1 - 2/kappa)^|i|, slowly for kappa = k_perp D/2
      >> 1: an overlapping (truncated) local solve is not exact. The partition method is: local Thomas
      solves plus a block-tridiagonal system of the 2 (P - 1) interface values per mode.
    - Every rank gathers the edge values of all ranks (10 doubles per mode and rank) and solves the
      reduced systems of all modes itself: one collective per field and step, no second
      communication. The work grows like P per mode; above ~64 ranks a recursive (SPIKE) reduction or
      a split of the modes over the ranks is the next step.
    - One rank keeps the single-rank code path (bit-identical to decision 38).
40. **iRIP with open x, laser and moving window: truncated system along x, the laser as RIP-CC's
    inbound T, and the entering cell of the window fixed by the constraints.** (docs/specs/implicit_rip.md)
    - The open edge's zero guards are exactly a truncation of the implicit system (zeros outside): the
      same matrix without the cyclic corner terms, still SPD. Energy only leaves (tested).
    - RIP-CC's exact window entry rests on causality: nothing reaches the last cell, so the truncated
      update equals the extended one. iRIP's solve couples all cells along x (the inverse decays like
      |z|^distance, |z| -> 1 for kappa >> 1), so the last cells carry a precursor and the entering cell
      broke the Gauss law at 3e-5. Extending the solve ahead would need hundreds of cells for round-off.
      Instead the entering cell's Ex, Bx are set from the Gauss law and div B at the point it closes:
      local, exact, and 0 for RIP-CC.
    - The beam self-field stays rejected: it solves RIP-CC's discrete equilibrium of a moving beam, which
      is not iRIP's, and would launch a transient.
41. **iRIP with CPML in y, z: the separable solve with a dense eigenbasis per pml axis, b floored at 1e-4,
    and the window's entry rule only outside the layers.** (docs/specs/implicit_rip.md, src/irip_pml.cpp)
    - The spec planned CG on the SPD operator. But the CPML derivative is linear in the current values
      with a factor b that depends on y alone (or z alone), so the operator stays a sum of 1D operators:
      an exact direct solve, the same structure as the DFT path and the planned RZ path (dense per-mode
      transforms), no iteration count, no tolerance.
    - Each 1D operator is C T, similar to a symmetric matrix: real eigenbasis, computed once.
    - b = exp(-sigma dt) reaches 1e-20 deep in a layer when c dt >> h_perp, and the eigenbasis then
      spans 1e20 (solve accurate to 1e-6 only). Flooring b at 1e-4 (absorption per step still 1e4) keeps
      it at round-off. RIP-CC's layers are unchanged.
    - Inside the layers the Gauss law does not hold, so the window's entry rule (decision 40) is
      applied only outside them; applied everywhere it fed back and the fields grew to 1e96.
    - The dense transforms are O(n^2) per line: fine for the transverse sizes of VLPL's 3D runs on
      the CPU; on the GPU they should become cuBLAS GEMMs (follow-up, with the timing).
42. **RZ with azimuthal modes: charge-conserving deposition by a 2D Esirkepov split in (z, r) times the
    mean phase, plus J_phi from the phase change.** (docs/specs/rz_modes.md, research/rz_modes/)
    - Delta(W e^{-i m phi}) = (Delta W) e_bar + W_bar Delta e is exact. The first part is the ordinary
      2D Esirkepov decomposition with cylindrical volumes, the second the azimuthal current at the rho
      node, where the solver's divergence has (i m / r) J_phi. So the discrete continuity holds per mode
      to round-off (prototype 1e-16), and with it the RIP-CC Gauss law per mode for explicit RIP-CC and
      iRIP. No current correction (FBPIC's curl-free projection is global per mode; this is local).
    - Of the two candidates of the architecture spec (this split, after Davidson et al. 2015, or direct
      deposition plus a per-mode correction) it is the local one: no extra solve, no MPI coupling.
    - Linear radial weights: rho = n off the axis, 4/3 n on the axis node for a uniform plasma; a volume
      correction would break the fluxes. Neutral plasmas cancel; documented.
43. **RZ fields stored in the Cart3D `Field` with the azimuthal mode slots as the third axis; iRIP per
    mode with a radial eigenbasis built by probing.** (docs/specs/rz_modes.md, src/rz_*.cpp)
    - Slot 0 = m = 0, slots 2m - 1, 2m = Re, Im: the existing Field, guards, halo exchange along z and
      kernel helpers serve RZ unchanged; r and the slots are not periodic and not split, so their guards
      are the zero wall. The cost is guard cells on the slot axis (2 ng per component); a per-axis guard
      width can remove them later.
    - The architecture's `FieldSet<G>` and geometry template come with the particle kernels (step 2),
      where gather and deposition need them; step 1 keeps the RZ fields beside Cart3D.
    - The radial operator of iRIP is built on the host by applying the same operator formulas to unit
      vectors, checked real and tridiagonal: one source of truth for the operators, no second hand
      derivation of the matrix.
44. **RZ particles: Cartesian particles with the per-mode gather and deposition in one kernel each;
    long moves split into straight Cartesian sub-segments.** (docs/specs/rz_modes.md, src/rz_particles.cpp)
    - Particles keep (z, x, y) and Cartesian momenta (D-1), so the pushers of Cart3D serve unchanged
      (`src/pushers.hpp`, shared inline code); r and phi exist only inside gather and deposition.
    - A move over more than one cell in z or r (iRIP's long steps, fast particles) is cut into equal
      straight sub-segments in Cartesian space: each is exactly charge conserving (decision 42), so
      their sum is, and no segment needs more than the 2 x 2 (z, r) stencil. A helical (r, phi) path
      would be no more exact (the motion within a step is straight in the pusher's model).
    - The RZ kernels stand beside the Cart3D descriptor kernels for now; folding both into the
      geometry template of the architecture spec waits for the deck integration (step 2b), when the
      operators of `Simulation` call them.
45. **RZ in the Simulation as a separate path beside Cart3D; the wall absorbs particles in the migration.**
    (docs/specs/rz_modes.md, src/rz_sim.cpp)
    - `Simulation` holds `FieldsRZ* rz` beside `Fields* fields` and its own three operators; the rest of
      the pipeline (diagnostics, beam diagnostics, delayed loading) is shared. Folding both geometries
      into one templated field set (architecture spec) would touch every Cart3D operator for one
      branch in each of a few places; it waits until the laser and the window need it (step 3).
    - Species keys in RZ are those of the particle axes (z, x, y) for momenta, and (z, r, theta) for the
      loading: the pushers and the beam diagnostics see Cartesian particles in both geometries.
    - The conducting wall absorbs particles at r >= nr dr in `migrate` (one list kernel with the z
      exchange): the current up to the wall is deposited before, so the Gauss law inside holds without a
      surface-charge model. Re-emission or a reflecting wall can come later as an option.
    - Stability of explicit RIP-CC in RZ is checked with the largest eigenvalue of the radial operators
      (from the same probed matrices as iRIP's bases), not with a formula: near the axis it grows with
      the modes (4.5 x 4/dr^2 at 3 modes).
46. **RZ laser: the Cartesian LaserConfig projected on the modes by an azimuthal quadrature; inflow
    hooks owned by FieldsRZ.** (docs/specs/rz_modes.md, src/rz_laser.cpp)
    - One source of laser formulas for both geometries: focus, Laguerre-Gauss modes, phase and several
      lasers come to RZ without a second derivation per mode. The quadrature with 4 (M + 8) points per
      radial node is exact for the modes kept as long as the pulse's azimuthal content stays below
      3 M + 32 (a linearly polarised LG_p^l: modes |l| +- 1); its cost is one kernel of nr M threads per
      step at the left edge.
    - The hooks belong to `FieldsRZ` (`inflow`, run at the start of the step) instead of the Cart3D
      `BoundaryHooks`, whose signature takes `Fields&`: the step needs them before both iRIP passes and
      after the predictor of RIP-CC, and the start of the step serves both (the predictor does not read
      the guard layer).
    - The window reuses the Cart3D flat shift (`shift_fields_x`: z is the fastest index in RZ too) and
      the entry rule of decision 40, per mode.
47. **The hosted MPI job runs only on demand; sessions run tests/mpi.sh locally before every pull request.**
    (the CI notes of the development repository)
    - On 2026-10-28 the job (~12 min per run, 13 pull-request runs in a day) used up the account's
      Actions minutes for the month. The cpu and gpu jobs run on the self-hosted runner and cost nothing.
    - The MPI checks are not lost: every session already ran tests/mpi.sh before pushing (decision 33),
      and the pull request names the result. The job stays in the workflow for a manual run.
    - To restore it on every pull request at no cost: a self-hosted runner on the CPU cluster (which has MPI), named
      by the repository variable `MPI_RUNNER`; the ranks go to a compute node through srun
      (`tests/mpirun_srun.sh`). Unset, the job stays manual and hosted (the CI notes of the development repository).
48. **RZ openPMD output in thetaMode with cos/sin coefficients C_m = 2 Re F_m, S_m = -2 Im F_m; the RZ
    LWFA benchmark compares the wake averaged over the first two radial cells.** (docs/specs/rz_modes.md,
    src/openpmd.cpp, bench/lwfa_fbpic)
    - thetaMode is what FBPIC and WarpX write and openpmd-viewer reads: the same tools compare the codes.
      The factor 2 and the sign follow from F = F_0 + 2 Re sum F_m e^{i m theta} against the standard's
      cos/sin series; checked by reconstructing a known uniform field from the file, not by our own
      inverse.
    - Particles keep their Cartesian records; only the axes are relabelled (X holds z).
    - With linear shapes the axis row is noise limited (the axis node has the smallest volume and few
      particles): two samplings of the same RZ run differ by 5-14 % there. The near-axis average
      (r <= 0.5 um, two cells) measures the physics; the axis row is printed beside it.
    - Tried and dropped: an axis volume V_0 = pi dr^2/3 (the volume of the linear weight, so that a
      uniform plasma deposits n instead of 4/3 n on the axis node; conservative like the present one).
      The benchmark got worse (0.121 against 0.108 on the axis row), and the vacuum TM01 eigenvector near
      the axis less exact. Shape 2 near the axis (step 4) is the way to lower the axis noise.
49. **RZ radial boundary: a graded damping layer (exp(-kappa s^2) per step) instead of a cylindrical CPML,
    for now.** (docs/specs/rz_modes.md, src/rz_fields.cpp)
    - It works unchanged for explicit RIP-CC and iRIP and for every mode: no change to the solver, its
      transverse operators or the iRIP radial eigenbases (a CPML in r would make the radial operator
      depend on psi and on a complex stretched radius in the m/r terms, decision 41 handled this only for
      Cartesian axes).
    - Measured cost of the shortcut: 1e-4 of a broadband pulse's energy stays at 32 cells (the Cart3D
      CPML reflects 2.6e-6 at normal incidence). Enough for wakefield runs, where the layer's job is to
      remove the laser's diffraction and the wake's radiation; a cylindrical CPML stays an option.
50. **RZ shape 2: the stencil folded across the axis at charge level; the wall absorbs at (nr + 1/2) dr.**
    (docs/specs/rz_modes.md, src/rz_particles.cpp)
    - Folding the charges and the Esirkepov fluxes (before dividing by volumes and areas) with the mode
      parities keeps the discrete continuity per mode exact without any change to the field solver or
      to the axis volume of decision 42; no separate near-axis shape (Ruyten-type corrections) is needed
      for charge conservation.
    - Removal radius from the shape's reach, as the open x boundary keeps particles until their weight
      in the last Gauss cell is zero (decision 21).
    - Shape 2 is the recommended RZ shape for wakefield runs: in the FBPIC benchmark it removes most of
      the axis noise of shape 1 (axis row 4.4-6.2 % against 7-11 %) at 1.8x the cost per step.
51. **RZ checkpoints store E, B as the thetaMode output (float64), not as raw slots.**
    (docs/specs/rz_modes.md step 4c, src/openpmd.cpp)
    - The coefficients C_m = 2 Re F_m, S_m = -2 Im F_m differ from the slots by powers of 2, so the restart
      is still bit-exact, and a checkpoint stays a valid openPMD file that openPMD-viewer reads like the
      output (as the Cart3D checkpoints). Only the initial Gauss residual is written as raw slots
      (`rippic_gauss0`, VLPL's own record).
    - The state group records `geometry`; checkpoints without it are Cart3D ones (older files).
    - The fast test runs on one thread: the RZ deposit adds atomically (decision 44), so "exact" is
      defined against a deterministic summation order; a tiled RZ deposit is not worth it for this alone.
52. **Stretched radial grid: finite-volume weights that keep the curl pair adjoint; particles in the index
    coordinate.** (docs/specs/rz_modes.md step 6, src/rz_fields.cpp)
    - Node areas A_j = (rh_j^2 - rh_{j-1}^2) / 2, half-node weights rh_j dn_j, gradient spacings dn_j and
      A_j / r_j: the discrete d_r and (1/r) d_r (r .) are negative adjoints in the energy weights, so the
      energy law of RIP-CC and the exact energy conservation of iRIP hold on any grid (2e-14 measured),
      and the divergence of the Gauss law is the same operator (charge per node / 2 pi A_j dz). Half nodes
      at the midpoints. A coordinate map with metric factors would also work but needs them in every
      kernel; the four arrays are simpler and exact.
    - The cells next to the axis stay uniform (at least 4): the axis rules and the shape-2 fold of
      decision 50 carry over unchanged.
    - Geometric stretch with a cap (r_uniform, stretch, dr_max) rather than a free table: it enforces the
      "stretch slowly" rule; a table can be added the same way if needed.
53. **RZ particle shapes: density-conserving (Ruyten-type) maps for the deposition; Gauss-point loading in r.**
    (docs/specs/rz_modes.md "Axis: density-conserving shapes", src/rz_ruyten.cpp, src/rz_particles.cpp)
    - The plain splines deposit a uniform density onto the axis node with 4/3 (shape 1) and 13/8 (shape 2) of its
      value, because their charge-assignment volume of node 0 (pi dr^2/3, ...) is not the field solver's axis
      volume pi dr^2/4 (decision 42). Visible as a line of excess density on the axis ahead of the laser (B7 against
      FBPIC), and a first-order error of the coupling of near-axis charge and current to the axis fields.
    - Not by changing the axis volume: that makes the axis divergence inaccurate (tried, the FBPIC agreement got
      worse). Instead the deposition uses the splines in a mapped index coordinate, cell j: xi = j + u + b_j u (1-u)
      (shape 2: xi = u^p in the axis cell), with b_j such that every node receives its finite-volume share of a
      uniform density: shape 1 b_j = 1/(2(2j+1)) on a uniform grid (one equation per cell, any grid); shape 2
      p = 0.328, b_j from the coupled equations of all nodes (banded Newton at setup, started from the
      first-moment rule, uniform limit b_j = 1/(2j+1)). The maps are monotone, so the Esirkepov fluxes in xi stay exactly charge
      conserving (Gauss law unchanged at round-off). As FBPIC (use_ruyten_shapes) the gather stays plain: the
      mapped gather is no better (Langmuir -1.2 % against -1.1 % at kp dr = 0.3).
    - Loading at Gauss-Legendre points in r per cell: the shape-1 deposited charge of a cell is cubic in the
      position, so 2 points load a uniform density exactly onto every node (the axis node too); shape 2 to the
      quadrature across the spline's kinks (2.6e-3 with 2 points).
    - Cost: a Langmuir mode loses O((kp dr)^2) of w as before (second order, -3.3e-3 at kp dr = 0.2).
    - rho of the modes m >= 1 is not deposited on the axis node (rho_m ~ r^m; the solver has no m >= 1 degree of
      freedom there, decision 42): the dipole of the charge in the axis half cell over its small volume gave values
      30x those of the next node in B7, and the theta = 0 / pi output plane C_0 +- C_1 showed negative electron
      density on the axis. Output only: the fields never read it.
54. **RZ current filter that keeps the Gauss law exact (`numerics.filter_passes`, default 0).**
    (docs/specs/rz_modes.md "Current filter", src/rz_filter.cpp)
    - The near-axis noise left after decision 53 is sampling noise (scales as 1/sqrt(ppc); the same with RIP-CC,
      shape 2, a finer dr, plain or mapped gather). FBPIC is quieter there because it filters rho and J by
      default (binomial in z and r) and its radial transform; B7 axis E_z noise behind the bubble: VLPL 1.6e-2,
      FBPIC 2.8e-3 E_0.
    - Per pass two steps of the form S = 1 + a G D on J, T = 1 + a D G on rho, D the divergence of the continuity
      restricted to the Gauss points and G = -D^T: then D S = T D identically, so the filtered rho and J keep the
      continuity equation per mode exactly on any boundary and decomposition (the Gauss law stays at round-off,
      unlike a plain binomial in r, which also moves charge onto the axis).
      - z: J_z,i += (dz/4)(w_i - w_{i-1}), w = D J (the full divergence, J_r and J_phi at the cell as rho) on the
        Gauss cells, on the columns between two of them; rho: the binomial over the Gauss cells, Neumann at their
        ends. In the interior, for J_z of a charge wave, the binomial.
      - r, per mode: rho and J_z by T = 1 + a_m D_m G_m, (J_r, J_phi) by S = 1 + a_m G_m D_m, D_m the radial
        divergence, G_m = -D_m^T in the finite-volume weights, a_m = 1 / lambda_max,TM(m) (T, S in [0, 1]); it acts
        slice by slice in z, so it commutes with the z step and every edge.
      - A uniform density is unchanged (no new axis excess); on a uniform grid away from the axis T is the binomial.
      - Both steps filter the longitudinal part of J only (S J = J if D J = 0); the curl part, the radiation, is left
        as it is. The first version used a plain binomial of every component along z: at an open edge it mixed in
        the guard cell, where the charge of leaving particles vanishes without a current, and the Gauss law drifted
        (2e-2 after 10 fs of plasma drifting out of a static box, 4e-3 in the edge cell of rz_lwfa).
      - With a moving window each shift turns interior cells into edge cells (left) and edge cells into interior
        cells (right), which the closures filter differently. Left: the offset stays frozen in the first
        filter_passes cells (a div E mismatch cannot propagate) and leaves with them; the diagnostics skip them.
        Right: exact when the plasma in the last cells is neutral and at rest (what the iRIP entry rule assumes too);
        with RIP-CC nothing reaches ahead of the pulse (window test at 3.5e-15), with iRIP the implicit coupling along
        z reaches the front at ~1e-7 of rho in a 6.4 um box, vanishing in B7's 40 um.
    - rho output (rho_<species>) is filtered the same way so it matches the J the fields see.
    - B7 (n = 2.3e-3 n_c, Nm = 2): axis E_z noise rows 0/1/2: no filter 1.6e-2 / 5.8e-3 / 3.8e-3, 1 pass
      4.2e-3 / 3.3e-3 / 1.9e-3, 2 passes 3.1e-3 / 2.6e-3 / 1.6e-3 (FBPIC 2.8e-3 / 2.0e-3 / 1.1e-3; first version of
      the z step). Off by default:
      it damps short wavelengths (k dz = pi removed per pass), so it is a choice of the deck as in FBPIC.

55. **Window shift without copies: the field array starts at a movable offset in its buffer; cold plasma
    enters through kernels.** (docs/specs/moving_window.md, docs/GPU_TESTING.md 2026-10-28)
    - After decision 36 the shift in `bench/lwfa_bubble` (600 x 160 x 160 cells) was still 5.3 ms on
      the RTX 5090: 1.2 ms for the flat copies of 7 arrays (at the memory bandwidth) and the rest the
      host loop over the entering column (410 k particles, the density function on the host, upload).
    - A shift by one x cell is the array seen one element further on. `Field` keeps one (x, y) plane of
      slack behind the array and `Field::shift_x` moves the start; only when the slack is used up (once
      per sx * sy shifts) one flat copy goes through the scratch field. The element exposed at the end
      is a corner of the right x guard, which the exchange along x fills, as with the copy.
    - Cold species load in kernels (`src/load.*`): a count per cell, a scan, a write per cell, the same
      particles in the same order as the host loop. The deck expressions are evaluated by one HD
      function (`expr_eval`) on the host and on the device. On the device `sin` etc. may differ from
      glibc in the last bit; arithmetic and comparisons, positions and the slab test (no FMA) are exact.
    - A hot species keeps the host loop with its per-cell generator (decision 36).
    - Bit-identical on the CPU (diag.txt of lwfa_window); on the GPU equal to all printed digits.

56. **O2: the reflected and transmitted field as the outbound transport components at the open x edges
    (`output.xflux`, `output.xflux_every`).** (docs/specs/xflux.md, 2026-10-29)
    - The component moving along -x in the left edge cell is the reflected field exactly, also while the laser
      enters (the laser is only in the inbound component of the guard layer); no separation of incident and
      reflected fields by subtraction or by a reference run is needed.
    - The energy of the outbound component in the edge cell (T^2 / 4 per point, the cell volume, the RZ mode
      weights) leaves during the next step: counted one step later. Exact for plane waves (round-off); with
      transverse structure a discretisation term O(dz^2) (1e-4 at lambda0/16 for a 3 um waist).
    - Records as raw float64 per rank with a text meta file (as the particles of the beam diagnostics), no
      HDF5 dependency; `tools/xflux.py` for spectra and harmonic profiles.
    - Not with a moving window (refused): the edge cell does not hold the whole outgoing wave there.

57. **O1: TFSF laser source with the incident field from an auxiliary vacuum grid; TFSF lasers with a cut envelope.**
    (docs/specs/tfsf.md, 2026-10-30)
    - Variant B of the draft (Alexander): a second `Fields` on the same domain and solver carries the incident beam,
      set up once left of the plane and advanced every step. Exact for any beam and both solvers (it is the discrete
      solution), no dispersion formulas per solver; the cost is one more vacuum field step. Variant A (analytic plane
      waves with the discrete k_x) was exact only for plane waves.
    - The corrections at the plane add the incident T+ (T-) of the cell across and half the incident Ex*, Bx* in the
      sources at the plane; iRIP takes the box mean of t_n and t_n+1 in both passes, the implicit solve is unchanged.
    - The incident grid is zero from the plane on at t = 0 (else a free error wave leaks back across the plane), and
      the envelope of a TFSF laser is exactly zero beyond 3 sigma (C2 spline from 2 sigma; Alexander), so nothing is cut
      at normal incidence. Leakage and total - incident at round-off (1e-15).

58. **The end point of the particle move is taken back from the stored global position before the current
    deposit (`r1 = P - lo` after `P = r1 + lo`, Cart3D `move_one`, RZ push).** (2026-10-30)
    - Positions are global, the push works in rank-local coordinates (lo = the rank's offset, plus the window shift).
      fl(r1 + lo) - lo != r1 when lo != 0: the next step and the charge deposit start up to ulp(x) / 2 away from where
      the current ended, a charge jump without a current, ~ n ulp(x) / h_x per particle and step.
    - Seen as an MPI-only Gauss drift with dense plasma (lo = 0 on one rank without a window): 4 n_c slab, laser from the
      left edge, 2 x 2 x 2 ranks: 2.3e-12 against 2e-15 on one rank, independent of a0 (any motion rounds), proportional
      to the density, starting when the laser reaches the slab, only on ranks with an offset, inside the plasma (not at
      rank faces). With the fix 2.5e-15. The moving window on one rank: 5.5e-15 -> 2.0e-15 (lwfa_window case).
    - Serial runs without a window are bit-identical (lo = 0). Left: a migrating particle is seen by the new rank at
      fl(P - lo_B), one round-off jump per crossing (not per step); exact only with positions in cell units and integer
      offsets, not worth the refactor.

59. **O3: absorbing x boundaries as graded damping layers (`grid.boundary_x = absorbing`), not a CPML along x; the laser
    enters through the TFSF plane.** (docs/specs/x_absorber.md, 2026-10-30)
    - Option B of the draft (Alexander): after every field step all six components in L cells at each x end are
      multiplied by exp(-kappa s^2), s the depth in the layer / L, kappa = 1.5 ln(1/R0) / (L dx / dt) (a wave in and
      out again at normal incidence: R0), as the radial layer of RZ (decision 49). Nothing in the RIP-CC transport or in
      the iRIP box solve changes; a CPML along x would need psi memories in both (iRIP: in its eigenbasis along x).
      The price is thickness: damping E and B alike is matched only at normal incidence, an oblique wave sees an
      impedance step that only a slow ramp keeps small. At 45 deg the reflected field is 1.4e-1 (open edge), 9.5e-2,
      1.4e-2, 5.6e-4 for 1, 2, 4 lambda0; default 4 lambda0 per side (Alexander), a CPML along x would need 8-16 cells.
    - Lasers at any angle come in through the TFSF plane (O1): the incident grid has the same layers, so the
      correction at the plane stays exact. A laser through the left edge, a laser file and a moving window are refused.
    - O2 then measures at the inner faces of the layers and also writes the net (Poynting) flux, outbound minus the
      inbound energy entering the measuring cell, which is right at any angle.

60. **Oblique Fresnel test with a plane wave of one k_y (`laser.waist = 0`); TFSF lasers at most 45 deg, steeper ones
    are to enter from a side boundary (Alexander).** (docs/specs/fresnel.md, 2026-10-31)
    - A Fresnel coefficient belongs to one k_y: `waist = 0` makes the TFSF laser a plane wave of k_y = k0 sin(angle)
      (periodic y with sin(angle) L_y = m lambda0, checked), its envelope moving along x at cos(angle). r(omega0) is the
      outbound record at the left measuring plane over that at the right plane without plasma, projected on
      exp(i k_y y), with the surface half way between the planes (exact cancellation of the propagation phases).
    - 12 fs pulses (the components near the cutoff omega = c k_y travel slowly along x); the underdense slab runs into
      the right absorber (no rear-face echo). 0.5 n_c at 30 deg was replaced by 0.25 n_c: the transmitted wave has its
      cutoff at 0.87 omega0 there, and its near-grazing components return over hundreds of fs.
    - RIP-CC, lambda0/32 -> /64: s, 9 n_c, phase error 1.3e-2 -> 1.1e-3 (30 deg), 1.1e-2 -> 6.6e-4 (45 deg); p only
      first order, 4.9e-2 -> 3.0e-2 (30 deg), 0.13 -> 0.074 (45 deg), likely the O(dx) position of the surface charge
      seen by the normal field (not proven); 0.25 n_c at 30 deg s: |r| 1.3 % low, second order in dy, not in dx.
    - Alexander 2026-10-31: lasers are not injected through the front (TFSF x plane) at more than 45 deg; a steeper
      laser should enter from the side boundary as in GEM-PIC-RZ. `tfsf_x` with |angle| > 45 deg is refused until the
      side injection exists.
   Addendum 2026-11-01: the first-order p phase error is the one-cell transition layer that the deposition makes of
   the step (Drude-Lekner thin-layer term, the exact ramp of one cell reproduces it); it depends on the sub-cell
   position of the surface. test_fresnel checks p with the surface half way between nodes (30 deg < 1e-2, 45 deg
   < 3e-2 rad at lambda0/32). A resolved ramp with shape 1 is unstable under p (resonance absorption seeds the
   finite-grid instability); shape 2 is stable. docs/specs/fresnel.md.

61. **Side injection: a TFSF tube with faces y, z = const and the incident field as a sum of exact discrete plane waves
    (block `side`).** (docs/specs/side_injection.md, 2026-10-31; follows decision 60: the x plane of O1 takes at most 45 deg)
    - Option S of the draft (Alexander): flying focus, luminal beatwaves and multi-mode LWFA in channels (low density)
      need converging waves that enter along the whole box and a moving window; steep oblique incidence needs > 45 deg.
      An x plane with an incident grid (O1) can do neither. Faces y and z (a closed tube), at least one cell outside the
      CPML layers: no CPML memory in the corrections.
    - The RIP-CC y, z derivatives are two-point differences and the x transport never mixes y, z cells: a difference
      across a face gets the incident value of the far point (predictor, transport sources, corrector). iRIP: known
      values in the right-hand side of both passes, the implicit operator unchanged.
    - The incident field is a sum of plane waves with the discrete vacuum dispersion of the solver (RIP-CC
      sin^2(wD/2) = sin^2(kxD/2) + D^2 Lam/4 cos^2(kxD/2), iRIP tan^2(wD/2) = tan^2(kxD/2) + D^2 Lam/4) and the
      polarisation from the eigenspace of the one-step matrix: leakage and total - incident at round-off (1e-14) at
      any angle. Continuum formulas would leak O(dx^2) (1e-2 at lambda/32 for steep waves, research/side_tfsf/).
    - Everything is a closed form at the cells next to the faces (phase tables per component, a time factor per
      step): no incident grid, no exchange, restarts store nothing. First provider: the plane wave, continuous or a
      pulse (Fourier sum; with periodic x the components share k_x). Next: the luminal Bessel pulse as a table in the
      window, continuum providers (rotated Gaussian, LASY).
    - The plasma must end 2 cells inside the faces (refused otherwise; outside the tube particles see only the
      scattered field); a warning if the incident field reaches the open x ends of the tube (Alexander).

62. **The flying focus through the side tube: a luminal Bessel pulse of exact discrete components, tabulated in the
    window frame (`side.wave = flying_focus`).** (docs/specs/side_injection.md, 2026-10-31)
    - Alexander: the exact table (not a radial table interpolated, which would leak O(h^2)); plasma outside the tube to
      come (stage B: real hydrogen is ionised far out, the faces must lie in preformed plasma, as GEM-PIC-RZ found); the
      references: vacuum leakage, GEM-PIC-RZ's vacuum and plasma decks.
    - All components share w - k_x = kappa; with the window at c from t = 0 the phase is k_x (local x) + k_perp . r_perp
      - kappa t, so the corrections at the face cells are a table times e^{-i kappa t} (8 complex per face cell), built
      once: per frequency the sum over the azimuths at each face column, then over the frequencies along x. A production
      box (4500 x 320 x 320, ~10^4 components) needs ~2 GB for the table in all and a one-time sum of ~10^13 operations;
      summing the components every step would cost ~10^13 operations per step.
    - |k_perp| from the discrete dispersion at each azimuth, so the pattern is exactly luminal on the grid. The initial
      field fills the whole tube (separable sum). The window must hold the pattern at both ends (the tube is open there).

63. **A background plasma for the side tube: the incident wave of the plasma and a linear cold fluid outside the
    kinetic region (`side.background_density`, `side.fluid = shell | all`).** (docs/specs/side_injection.md, 2026-11-01)
    - Alexander: the cold-fluid shell (proposed against particles outside the tube, which would gather scattered +
      incident fields and carry j - j_inc: for the flying focus a table over the whole shell, ~3 GB in a production box,
      and only as exact as the particle response matches the linear one). The fluid is driven by the stored field, so
      outside the tube it carries exactly j - j_inc and the TFSF stays exact: leakage at round-off (1e-14) as in vacuum.
    - The fluid at the points of J, j^{n+1/2} = j^{n-1/2} + D wp^2 F (Ex, mux Ey, mux Ez)^n. The discrete dispersion of the
      one-step matrix with it: RIP-CC splits TE (Ex = 0, Lam + wp^2) and TM (Bx = 0, Lam + wp^2 (1 - D^2 Lam / 4)), iRIP
      stays degenerate (Lam + wp^2 / cos^2(wD/2)); every provider component becomes a TE and a TM wave with RIP-CC.
    - F: the complement of the coverage by a uniform particle plasma at the kinetic edge (2 cells inside the faces), 0 in
      the CPML; fluid = all for tests and a linear plasma. The kinetic plasma must meet the edge with the background
      density; the fluid holds where the field at the faces is weak (linear response).

64. **O4 solid-density targets: the tilted slab gets an exponential pre-plasma with exact sub-cell weights; shape 3;
    a Cart3D current filter of the longitudinal current only; omega_p dt checked at the loaded peak density.**
    (docs/specs/solid_target.md, 2026-11-01)
    - Pre-plasma: front (and optionally rear) exponential ramps of the slab normal coordinate, cut at an extent (default
      5 scale lengths). With a ramp (or `slab_exact`) every macro-particle stays on the regular lattice but carries the
      exact mean of the profile over its sub-cell (closed form: the second antiderivative of the piecewise exponential,
      four corner values over (cos sin)), so the loaded charge is the integral of the profile to round-off at any tilt
      (8e-14 in the test; the point-sampled plain slab misses 9e-4 at 22.5 deg). Electrons and ions on the same keys are
      neutral at t = 0. The plain slab keeps the point test (bit-identical results of the existing decks).
    - Shape 3 (cubic B-spline) through the same templates; its support of 2 cells needs particles kept 2 cells below an
      open lower edge and 1 above an upper one (Domain::keep_below / keep_above; decision 21 had 1 and 0), guards
      ng = 4 + ceil(dt/h) (5 for RIP-CC), and the tile box one cell lower along x ([a - 3, b + 2], T >= 6 keeps the
      colouring safe). Not in RZ, not with a moving window (the entry columns assume shape <= 2).
    - Filter: as decision 54, S_a = 1 + (h_a^2/4) G_a D per axis on the current, T_a = the binomial on the charge over the
      Gauss nodes; D S = T D identically, so the Gauss law with T rho stays at round-off on any boundary and decomposition.
      It smooths only the longitudinal current: grid heating is electrostatic aliasing, while the transverse surface
      current radiates the harmonics; a plain binomial of all components would damp harmonic N by cos^2(N k0 h/2) per
      pass (0.66 for N = 20 at lambda0/100). rip_cc and irip; not Yee, not with a moving window.
    - omega_p dt: Config::check sees `density` only; after the initial load the largest loaded density factor per species
      (all ranks) gives the peak omega_p, refused when unstable, a warning above omega_p dt = 1, printed with the skin
      depth in cells.
    - Shape 1 with a pre-plasma (finding of the GPU session, 2026-11-01): a cold 0 -> 9 n_c ramp over lambda0/4 hit by p-pol
      light at 45 deg (a0 = 1e-3) grows exponentially after the pulse (energy x600 at lambda0/32, x6000 at /64 by 130 fs;
      Boris and iRIP alike), shape 2 decays: resonance absorption at eps = 0 drives plasma waves that break, and the
      velocity spread with lambda_D << dx feeds the shape-1 finite-grid instability. A slab ramp with shape 1 warns.
    - Grid heating test (lambda_D = 0.17 dx, 300 steps): dK/K +0.68 with shape 1, -0.037 with shape 3 and -0.031 with
      shape 3 plus one filter pass (the initial exchange with the field fluctuations, no heating).

65. **The TFSF x plane may take its incident field as exact discrete plane waves (`laser.tfsf_incident = modes`) instead
    of the incident grid; decision 59 stays (no injection through the left edge of an absorbing x boundary).**
    (docs/specs/tfsf.md "Exact plane waves", src/tfsf_modes.*, 2026-11-01; Alexander: reconsider decision 59, option 3)
    - The incident grid needs the whole pulse between the left x absorber and the plane at t = 0: at 45 deg ~17 um for a
      13 fs pulse with a 2 um waist, a box 3.3 times Smilei's (which injects at its boundary) and twice the field work:
      the first oblique HHG run cost 4.5 times Smilei's (100 against 20 core-hours at lambda0/32 x lambda0/16).
    - Options weighed with Alexander: (1) injection through the absorber at the left edge (T+ only: wrong by
      (1 - cos a) / (1 + cos a) in field at oblique incidence, 17 % at 45 deg); (2) a continuum beam formula at the plane
      (O(dx^2) error leaking into the reflected record); (3) exact discrete plane waves, chosen.
    - The corrections read the incident field only in the two cells next to the plane: a Gaussian beam as its angular
      spectrum on the k_y, k_z lattice of the periodic box times the Fourier sum of the envelope, each component an
      eigenmode of the one-step matrix (plane_mode of side.cpp, decision 61), evaluated separably per step. At t = 0 the
      incident field is added to the total-field region, so the pulse may start anywhere, also outside the box.
    - Leakage at round-off (1e-15 of the peak) at 0, 30, 45 deg, both polarisations, RIP-CC and iRIP, round or z-uniform
      beams, the plane 8 cells from an open edge or the left x layer; peak over the run = the incident grid's to 0.35 %
      (the beam models differ by 4.5 % in L2: paraxial entrance formula with a cut envelope against the exact focus).
    - What the boundaries do to the incident field differs: the incident grid contains the reflection of the incident beam
      at the open right edge or the x layers, so it cancels; with plane waves that echo is scattered field and crosses
      into the record (e.g. 0.07 of the field at 30 deg off an open edge; ~1e-6 from the low-level wide-angle content of a
      Gaussian focus off an edge 11 um away). Use absorbing x and keep the right layer far enough.
    - Periodic y, z only (no CPML; the plane waves are those of the periodic box); Gaussian beams (no Laguerre-Gauss).

66. **2D Cartesian geometry = the Cart3D code on one z cell with the z stride 0 (`grid.geometry = cartesian2d`), not a
    separate Cart2D field geometry.** (docs/specs/cart2d.md, 2026-11-02; Alexander chose this option)
    - The 2D-like Cart3D decks (3 periodic z cells, the fewest the guards allow) paid for a grid 9 planes thick: the TNSA code
      check cost 7.2 core-hours against Smilei 2D's 0.27, independent of the particles (rip_cc 66 %, push_deposit 33 %).
    - `Domain::flat`: no z guards, `FV::sy = 0` (z stride sx sy = 0), so f(i, j, k +- 1) = f(i, j, k) and every existing field kernel (RIP-CC, iRIP
      with a length-1 z DFT, CPML in y, TFSF, x absorber, filter, diagnostics) computes d/dz = 0 exactly, unchanged. The
      option weighed: a separate Cart2D geometry with 2D copies of every field module (more code, every later feature
      twice), for at most the few redundant z flops in the field step (the step is memory bound; memory is one plane).
    - Particles keep z and u_z (z wraps over the slab on the device). Only the particle kernels have a 2D form (`Cart2D` in
      src/geometry.hpp): shape in (x, y); J_z = q w v_z W_xy, the z sum of the Cart3D J_z of the same move.
    - Totals are per slab of thickness `grid.dz` (default 1 um): weights n h_x h_y h_z / ppc. With the dz of a z-uniform
      Cart3D run, the 2D run equals one z column of it (test: fields 4e-11 of the peak, energies 5e-13 after 14 fs of a
      laser on a foil; later the round-off grows in both, as between the z planes of the 3D run itself).
    - The plane-wave TFSF source takes its frequency lattice from the x-y diagonal of the box in 2D (no path along z); its
      sampling changes the incident field by ~5e-5 of a0 between boxes (also in 3D between different dz), so 2D and the
      Cart3D slab with `tfsf_incident = modes` agree to that level, with the incident grid to round-off.
    - Refused in 2D for now: yee, side injection, laser files, z-profiled lasers (waist_z, Laguerre-Gauss), self_field.

67. **The CHF focus is measured inside the box by a band-pass in time per node (`harmonic_focus`), not by propagating the
    O2 records back, and RZ needs no TFSF plane at normal incidence** (2026-11-03, docs/specs/harmonic_focus.md).
    - What Gate O asks (focal intensity, spot size, position of the reflected harmonics) is a property of the field in front
      of the target, where the incident laser overlaps with the reflected light. Options weighed: (a) the field history of
      the focal region written every step and filtered afterwards (tens of GB per run at lambda0/128); (b) the O2 records at
      the left edge propagated back to the focus in post-processing (an analytic propagator, so the grid's dispersion of the
      harmonics between the focus and the edge would not be in it, and the open edge returns part of an oblique wave);
      (c) a causal band-pass on the fly: chosen. Butterworth edges by the bilinear transform, prewarped at n_min, n_max,
      cascaded second-order sections in direct form II transposed, one state per node and component: 96 doubles per node in
      RZ, 5 % of the step in the RZ vacuum test, exact against a host replay of the grid's own field (2D 0, RZ 1.9e-14).
    - A causal filter has a phase: the peak of the filtered pulse depends on it (a 6 fs pulse through 0.5 .. 1.5: 4 % lower),
      the fluence does not (Parseval). Both are written; a resolution series compares them at the same band and filter order,
      with the band edges between harmonic orders (4.5, 10.5, 20.5).
    - RZ, linear polarisation: the field is reconstructed at phi = 0 and pi/2 (the polarisation plane and across it); on the
      axis the transverse field is the Cartesian 2 (Re, -Im) E_r,1 at both angles (E_phi of the axis is no DOF, decision 43).
    - O1 in RZ at normal incidence: the laser through the open left z edge is exact along z, but the edge returns
      (1 - cos a) / (1 + cos a) of a wave leaving at the angle a (3 % in amplitude at 20 deg); the CHF harmonics diverge from
      their focus. First judged not needed; built the same day on Alexander's request (decision 68).
    - Energy balance (Gate O: absorbed + reflected + transmitted to 1e-3): xflux.txt gets the energy the radial absorber of
      RZ removes (E_abs_r: W before minus after the damping, every step), the field energy between the measuring planes
      (W_in: without the x absorbers, whose content Enet has already counted as gone) and the kinetic energy of the
      particles removed at open edges or the RZ wall (E_part_out, summed in the migration: 1.2 % of the laser energy in the
      RZ CHF run, which was off by -1.1 % without it).

68. **RZ gets a TFSF plane with an auxiliary incident grid and absorbing z layers; the laser through an open left edge is
    inbound energy in xflux** (Alexander 2026-11-03, docs/specs/tfsf.md "RZ", docs/specs/x_absorber.md "RZ", xflux.md).
    - Why: the open left z edge returns (1 - cos a) / (1 + cos a) of oblique outgoing light, and the CHF harmonics leave at
      20-40 deg. Damping layers at the z ends (as the Cart3D x layers, decision 59) would damp an edge-injected laser, so the
      laser comes in through a TFSF plane right of the left layer.
    - The incident field: variant B of decision 57 (an auxiliary vacuum grid of the same scheme), not exact plane waves
      (decision 65): in RZ a focused beam is no finite sum of discrete plane waves of the modes, and the auxiliary grid costs
      one vacuum field step (the RZ CHF run is particle-bound: 88 % push). The laser enters the incident grid through its open
      left edge (no initial pulse to set up, no overlap with the plane at t = 0); it has the radial absorber and the right z
      layer of the main grid but not the left one. The corrections are the RZ form of tfsf_correct_kernel per mode; iRIP takes
      the incident box means before the radial damping, as its own implicit solve does (taken after it, the leakage was
      1.4e-6 instead of 2e-15).
    - Normal incidence only; no restarts and no moving window with the RZ plane yet (refused).
    - xflux: the inflow hooks write the laser into the guard inside the field step and the guard no longer holds it after the
      step, so Enet_left had never counted the injected energy (Cart3D and RZ, E_left = Enet_left exactly). One more inflow hook
      (registered by xflux after the laser hooks) measures the inbound component in the guard and counts it as inbound at the
      left plane in the same step; E_inj (column 12) is its sum, with an RZ TFSF plane the energy injected into the incident
      grid. Counted one step later (as the other inbound terms), the balance was off by up to 5e-2 while the pulse entered.

69. **The Cart3D current filter projects: the change of J is RIP-CC's gradient G psi, solved per transverse mode; the
    local filter of decision 64 stays as `numerics.filter = local`** (Alexander 2026-11-03, docs/specs/solid_target.md).
    - Why: in the oblique HHG case one filter pass raised the reflected orders 8-16 by 2-3.7 (bench/hhg_oblique). The local
      filter changes one component at a time (J_a += (h_a/4) d_a D J): that keeps D J' = T D J, but a change of J_a alone
      is no gradient, so every pass adds a grid-scale transverse current that radiates. Fed alone into the solver, 7 % of
      the field energy of that change ends in B; in a thermal slab it raises the radiated noise at orders 7-16 by 1.1-1.8.
    - The projection: J <- J + G psi with L psi = (T - 1) D J. The plain discrete gradient (dx, dy, dz), the adjoint of D,
      still radiates in RIP-CC (4 % in B): RIP-CC's Gauss law has mu_x on the transverse terms, and its electrostatic
      field is the gradient G = (dx, mu_x^2 dy, mu_x^2 dz) with L = D G = dxx + mu_x^2 (dyy + dzz), the operator of the beam
      self-field. With G the change leaves B at 2e-31 (RIP-CC and iRIP).
    - Solve: DFT in y, z and a tridiagonal system per mode along x with iRIP's partition method for ranks along x; the line
      solver moved to src/xlines.hpp with the mode coefficients as a parameter (iRIP bit-identical). Open x: Neumann on the
      Gauss nodes [0, N - 1); the transverse mode 0 is singular and pinned (psi_0 = 0; its dropped row holds since T keeps
      the charge). Restrictions as iRIP's: y, z periodic, ranks along x only; the local filter stays for other set-ups.
    - The non-power-of-two DFT was the direct O(n^2) sum with a sin and cos per term (the projection filter four times
      slower than the whole field step at ny = 120); it is now a mixed-radix Stockham FFT (O(n sum p)), also for iRIP
      (power-of-two sizes bit-identical, others at round-off).

70. **The RZ current filter projects as well: J <- J + G psi with G = (dz, mu_z^2 G_m) the gradient of the RZ Gauss law,
    L psi = (T - 1) D J per mode in the TM eigenbasis and along z per radial eigenmode; the filter of decision 54 stays as
    `numerics.filter = local`** (Alexander 2026-11-04, docs/specs/rz_modes.md "Current filter").
    - Why: test R24 (as S6 of decision 69): fed alone into the field solver, the change of J of one pass of the local filter
      puts 0.17 (RIP-CC) / 0.15 (iRIP) of its field energy into B, half of a raw random J, more than the local Cart3D filter
      (7e-2). Its z step changes J_z alone and its r step is no gradient either. Decks that used it: bench/chf (all three
      series, filter_passes = 1, also the RZ run with the open late energy excess) and bench/fbpic_nm_scan (2); how much it
      changed their results is not measured.
    - The gradient: as in Cart3D the plain (dz, G_m) psi still radiates (1.9e-2); with RIP-CC's mu_z on the transverse terms
      of the Gauss law the curl-free gradient is (dz, mu_z^2 G_m) psi (5e-32 / 7e-32). The projection: 1.1e-31, and
      D J' = T D J to 5.5e-15 (R25); T on rho unchanged, so the axis smoothing of decision 54 is the same.
    - Solve: iRIP's TM radial basis per mode (exported as rz_tm_basis) and the line solver of src/xlines.hpp with the
      projection's coefficients (s = 1 - dz^2 lambda/4, b = -2 - dz^2 lambda/2; Neumann at the open z ends), the partition
      method for ranks along z. Unlike Cart3D no restrictions: periodic or open z, any ranks along z, the moving window.
    - Moving window: the elliptic solve reaches ahead of the pulse (the Coulomb field of the filter's charge rearrangement);
      in R23 the entering cells get a Gauss residual of 1.2e-13 for ~80 steps while the pulse meets the plasma at the front
      edge (local filter 2e-15). Accepted; R23's window limit for the projection is 1e-12.

71. **Where the Cart3D projection filter cannot run, the local filter is used with a warning instead of refusing the deck**
    (Alexander 2026-11-04).
    - The projection (decision 69) needs y, z periodic (no CPML, no side tube) and ranks along x only, as iRIP: the FFT in y, z
      needs whole lines on every rank, the lines along x use the partition method. With it as the default, every RIP-CC deck
      with a filter and ranks in y or z (the automatic decomposition) or CPML in y, z was refused at the start: tests/mpi.sh's
      2D foil on 8 ranks in x, y, and the CHF 3D lambda0/128 job for JUPITER (16 GPUs). RZ is not affected (ranks along z only;
      its projection has no restriction, decision 70).
    - Now Simulation picks the local filter in those cases and prints a warning (its change of J radiates at the grid scale);
      numerics.filter = local silences it. Lifting the restriction (transposes for ranks in y, z; a sine / cosine or dense
      transform for non-periodic y, z) is left until the local filter's radiation matters in a production case.
    - The CHF series (bench/chf/series.py) sets numerics.filter = local explicitly in all three geometries: its 2D, RZ and 3D
      lambda0/64 results were made with the local filters, so lambda0/128 stays comparable.

72. **O5: iRIP gets an optional dispersion-corrected closure, `numerics.closure = o5`: Ex*, Bx* <- Cx C_perp Ex*, Bx* with
    Cx = 1 - (hx^2/3) delta_x^2 and C_perp = 1 - sum_a (D^2/6 + h_a^2/12) delta_a^2; first in 2D** (Alexander 2026-11-04,
    docs/specs/overdense.md "O5").
    - Why: the box closure has 2-3 times RIP-CC's oblique dispersion; in the CHF benchmark the phase error over the 3 um from
      the paraboloid to the focus (order 7 at angle 0.4: 0.69 rad iRIP, 0.30 RIP-CC at dz = lambda0/128, dr = lambda0/40) is a
      spherical aberration of the harmonic focus (iRIP's peak 17 % below RIP-CC's on the same grid, bench/chf/README.md).
    - The paraxial factor Cx planned before only brings iRIP to RIP-CC's level; most of the rest is the transverse grid. The
      transverse factor cancels the leading time-step and transverse-grid errors as well: 0.027 rad instead of 0.30 / 0.69.
      Any closure shared by both movers keeps the Gauss law (Theorem 1); checked with plasma at 1e-14.
    - Price: not energy conserving. Every mode stays neutral (spectral radius 1), but the plain energy of grid-scale modes
      grows transiently, by ||G^n|| <= 1.4 (h_perp = 3.2 dx), 2.05 (dx), 6.5 (dx/5); random fields reach 3.4 times their
      energy. Off by default.
    - The pentadiagonal system per mode factors into two tridiagonal ones with constant coefficients, so the line solver and
      its partition method are reused unchanged.

73. **O5 in 3D: the transverse factor of the closure per mode in the y, z Fourier basis**, g_perp = 1 + D^2 lambda/6 +
    (h_y^2 lambda_y^2 + h_z^2 lambda_z^2) / (12 lambda) (Alexander 2026-11-04, option B; docs/specs/overdense.md "O5 in 3D").
    - Why: one local factor shared by Ex and Bx (needed for the Gauss law) cannot cancel the per-axis transverse-grid error; it
      leaves an azimuthal anisotropy of the phase error as large as the box closure's (order 7 at 0.4 rad: spread 0.11 rad;
      per mode 0.005 rad). Anisotropy distorts a 3D harmonic focus differently per azimuth.
    - Price: four extra transverse transform pairs per step (about three times the transform work of a box-closure step).
      Transient growth at the grid scale 1.5 (h_perp = 3.2 dx) .. 7.2 (dx/4), every mode neutral. 2D unchanged (the stencil).

74. **O5 in RZ: the transverse factor per radial eigenmode with the mode-weighted spacing**, g_a = 1 + D^2 lambda_a/6 +
    h_a^2 lambda_a/12, h_a^2 the energy-weighted mean of the local squared spacing over the mode (Alexander 2026-11-04;
    docs/specs/overdense.md "O5 in RZ").
    - Why: the discrete radial operator follows the Cartesian grid error dr^2 lambda^2/12 on a uniform grid, but on a
      stretched grid the axis spacing dr removes almost nothing of the error (2.2 % -> 2.0 % in k_perp); the mode-weighted
      spacing removes most of it (0.5-2.7 % -> 0.03-0.16 % for k dr <= 0.26). On a uniform grid h_a = dr (the 2D operator).
    - Price: two dense radial transforms per closure, four closures per step; not energy conserving (random fields up to 3.3
      times their energy), every mode neutral as in 2D. Not with the RZ TFSF plane yet (since decision 78: yes).

75. **O5 with CPML and the TFSF plane: the closure per mode of the stretched operator's eigenbasis; the higher CPML reflection
    is accepted** (Alexander 2026-11-05; docs/specs/overdense.md "O5 with CPML and the TFSF plane").
    - CPML: the only per-mode form compatible with iRIP's solve on a pml axis. The layers reflect about 60 times more than with
      the box closure (1.5e-4 at 16 cells, 1e-5 at 32 cells, normal incidence): the closure sees the instantaneous stretching,
      not the CPML memory. Accepted; with o5 use layers of about 32 cells.
    - TFSF plane: exact (leakage at round-off) with the incident grid running o5; the exact plane waves (tfsf_incident = modes)
      and the side tube follow.

76. **O5 with the side tube and the exact plane waves: the per-mode closure, with a per-step correction of the closed Ex*, Bx*
    in the tube** (Alexander 2026-11-05: per mode, measure the leakage; docs/specs/overdense.md "O5 with the exact plane
    waves").
    - The plane waves carry the closure in their one-step matrix and dispersion; the correction [in] c I - C(chi_in I) makes
      the scheme exact for (incident in the tube + scattered) whatever the non-local closure does near the CPML layers.
    - Measured: the leakage stays at round-off (<= 1.5e-14) for plane waves, pulses, the tube corners, the flying focus in
      vacuum and in a plasma; the local-stencil alternative (anisotropic in 3D) is not needed.

77. **O5: the time-step part of the transverse factor is capped at q = D^2 lam/4 <= 1 (k_perp c dt <= 2), always on**
    (Alexander 2026-11-05: hard cap at q = 1, no option; docs/specs/overdense.md "the cap of the transverse term").
    - Without it the transient growth at the grid scale grows like (dx/dy)^2 (12.6 in 2D, 17.6 in 3D at dy = dx/10); with it
      <= 2.15 for any dy. Resolved modes (q <~ 0.2) are unchanged to all printed digits.
    - The 2D closure is now per mode in the y Fourier basis (no stencil); o5 results change only for modes with q > 1.

78. **O5 with the RZ TFSF plane: the incident grid runs o5; its box means get C_perp per z cell, then Cz; the Cz stencil of
    the main grid across the plane gets the incident value of the far cell** (Alexander 2026-11-06: "go ahead";
    docs/specs/overdense.md "O5 with the RZ TFSF plane", docs/specs/tfsf.md "RZ").
    - Why: the CHF focus at lambda0/128 in RZ is limited by the chromatic focal shift of the scheme (bench/chf/README.md,
      2026-11-06); O5 removes it, and the production set-up feeds the laser through the TFSF plane (decision 68).
    - As Cart3D (decision 75): C_perp acts per z cell, so it is the same on either side of the plane; only the 3-point Cz
      crosses it. Measured (tests/test_rz_tfsf.cpp R1): left of the plane 3.1e-15 of the peak, total - incident 4.1e-15 (box closure: 2.2e-15, 3.4e-15; without the stencil fix at the plane 1.0e-3). A
      single harmonic (order 7, NA 0.5) fed through the plane focuses as when fed through the open edge (peak and position
      to 4 digits): 2.658 um against 2.656 um of the converged reference, iRIP without o5 2.842 um. MPI along z = 1 rank.

79. **RZ noise heating: ppc_theta must grow with the modes; default `numerics.rz_transverse_shape = none` with a warning
    below ppc_theta = 2 (2M - 1); a transverse Gaussian particle shape (`gaussian`, rz_shape_width = 1 dr) as an option**
    (Alexander 2026-11-06: first the Gaussian as the default, chosen over constant or r-dependent sigma factors, which would
    damp the laser's m = 1 coupling or a uniform field near the axis; then, after the CHF check below, "Default none + ppc
    warning"; docs/specs/rz_modes.md "Transverse particle shape").
    - Why: part of the late energy excess of the RZ CHF runs is noise heating (the rest: an axis instability of the dense
      target, decision 80), and in RZ the noise heating grows with the number of modes at fixed
      ppc_theta: a macro-particle is a truncated Fourier series in phi, ppc_theta particles sample 2M - 1 azimuthal
      components. Hot plasma column, 4 modes, CHF cells (bench/noise_heating): 1.0e-3 of the energy per 10 fs, 2D with
      the same cells 1.0e-4, RZ with 1 mode 0.8e-4; ppc_theta 8 -> 28 -> 56: 1.0e-3 -> 2.2e-4 -> 4e-5. Not the cause: the
      RIP-CC stability limit, the deposit/gather mismatch of the Ruyten maps (rz_gather = matched: -2 %), filter passes.
      More particles near the axis only do not help (r < 0.8 um: -14 %): the heating is in the whole column.
    - S = exp(a^2/2 Lap_perp) per mode from the field solver's operators, on J (deposit), E (gather) and rho (diagnostics):
      self-adjoint in the field energy (deposit and gather fit, 1e-17), continuity exact (Gauss law at 1e-13 in the runs),
      smooth fields untouched. Heating with a = 1 dr: |G| <= 7e-5 per 10 fs (4 and 8 modes, RIP-CC, iRIP, O5). Cost +30 %.
    - Price: deposit and gather at transverse wavenumber k_perp are scaled by exp(-k_perp^2 a^2 / 2): at dr = lambda0/40,
      0.73 for harmonic 10 at 30 deg (0.93 with a = 0.5 dr, which heats 3.7e-4). CHF at lambda0/64 (dr = lambda0/19,
      bench/noise_heating): the low-band peak -11 % (0.5 dr) and -39 % (1 dr) against `none` with ppc_theta 30, which
      keeps the focus (-0.4 %) and cuts the high-band noise fluence 5.4 -> 2.3. For harmonics at large k_perp use
      `none` with more ppc_theta.

80. **RZ gather next to the axis: `numerics.rz_gather = matched` by default (plain | matched | energy); the CHF floor is an
    axis instability of the dense target** (Alexander 2026-11-06: "Yes, matched default"; bench/noise_heating "The rest
    of the CHF excess").
    - The CHF excess at lambda0/64 = noise (b / ppc_theta) + a floor of 1.6e-2 at 59 fs that grows steeply with dr. A uniform
      40 n_c plasma at 1 keV with k_p dr = 2.1 is unstable next to the axis in RZ (W_kin x 18 in 29 fs) and stable in 2D on
      the same cells, stable without plasma at the axis and at 10 n_c.
    - The plain gather differs from the density-conserving deposit (decision 53) next to the axis. `matched` (the deposit's
      radial map) and `energy` (E_r with the derivative of the deposit's shape) remove most of it (x 2.1 / 2.4); a slower
      growth after 20 fs remains open. `matched` is exact for uniform fields; `energy` is not next to the axis (up to
      -62 % within the first cell).
    - CHF lambda0/64, ppc_theta 30, excess at 59 fs: plain 2.26e-2, matched 1.22e-2, energy 7.0e-3 (2D: 7.8e-3); the
      low-band focus unchanged (matched -1.4 %, energy +3 %).
    - The slower growth that remains is a resolution requirement, not the z or time centring: stable for k_p dr <= 1.5 in
      the cells next to the axis; a radial grid refined only there removes it (+10 % cells). A run warns for k_p dr > 1.5.

81. **O6: the energy-conserving implicit mover, `numerics.mover = implicit`: Crank-Nicolson particles with the work field of the
    box closure gathered by the transpose of the Esirkepov deposit; a fixed point in the grid current with one particle pass per
    iteration, the lumped cold preconditioner and Anderson; fixed path segments ceil(c dt / h_min)** (Alexander 2026-11-06: "go on
    with A"; docs/specs/implicit_mover.md, research/o6_mover/, src/implicit_mover.cpp)
    - Why: W + K exactly conserved (to the tolerance) kills grid heating and any numerical growth that draws on the energy, and lifts
      omega_p dt < 2; the Esirkepov deposit and the field step are unchanged, so the Gauss law stays exact.
    - The work field of the 3D box step (irip_trial, src/irip.cpp): Et_x = mu_t Ex, Et_y = [(Ey + Bz)(i) + (Ey - Bz)(i+1)]/2 + D/2 Psi_y
      - D^2/4 delta_y delta_x mu_t Ex, Et_z likewise with (Ez - By)(i) + (Ez + By)(i+1), Psi_z and delta_z (research/o6_mover/work3d.py;
      C++ test M1: W^{n+1} - W^n = -dt dV J . Et to 0 / 9e-16).
    - Fixed segments: the adaptive split of esirkepov_segments makes the current a discontinuous function of the iterate where a move
      crosses one cell; the solve stalled at a residual of 1e-3 (hot plasma, h_perp = c dt / 2). With n = ceil(c dt / h_min) segments
      for every particle (|vbar| < c keeps each below one cell) it converges geometrically (24-34 iterations to 1e-12).
    - One particle pass per iteration (u1 updated from the current Et, then the new path deposited) beats converging each particle
      first (prototype: 7.8 against 21 passes per step).
    - Momenta are at integer times with this mover (u^n), at half steps with the explicit push.
    - First version: irip with closure = box, shape 2 or 3 (the path gather of shape 1 is piecewise constant), Cart3D and 2D, periodic or
      open x (the inflow hooks run once per step, in the first trial); rejected: RZ (the shape-2 Ruyten axis map has an unbounded
      slope, which stalls the solve: an axis map decision first), pml, TFSF, side tube, window, current filter (the gather would need
      the adjoint filter), x absorbers, the tiled and box deposits.

82. **O6 in RZ: the implicit mover with a C1 density-conserving radial map, for m = 0** (Alexander 2026-11-07: "RZ axis map
    implement: the bounded-slope density-conserving map", then "Research a C1 map"; research/o6_mover/README.md, src/rz_irip.cpp
    rz_irip_trial, src/rz_particles.cpp, src/rz_ruyten.cpp rz_c1_coefficients)
    - The map (rz_xi shape 4, only with numerics.mover = implicit): a cubic Hermite per radial cell with the node slopes dxi/dr
      s0 / dr = 3.5 / dr at the axis, 0.4 x 2 / (dn_0 + dn_1) at node 1, 2 / (dn_{j-1} + dn_j) beyond, plus a bump ca_j u^2 (1 - u)^2
      per cell from the node shares (Newton, banded, 1e-14): density-conserving as decision 53 and dxi/dr continuous, 0.20 ... 3.5
      on a uniform grid (research/o6_mover/c1scan.py). Why: the energy-consistent radial force is proportional to dxi/dr; decision
      53 has u^0.328 in the axis cell (unbounded) and slope jumps at every cell edge, and the bounded but discontinuous exponential
      map still stalled the solve (prototype, omega_p dt = 1.6: 31-62 iterations to the cap, |dE|/E up to 2.6e-5). Loading:
      the Gauss-point loading samples the steep axis cell: rho of a uniform plasma on the axis node 9e-2 off with ppc_r = 2, 1e-2
      with 4, 1e-3 with 8 (decision 53's map: 3e-3 with 2; research/o6_mover/c1_loading.py).
    - The work field of the RZ box step per mode (rz_irip_trial): Et_z = Ez*, Et_r = [(E_r + B_phi)(a) + (E_r - B_phi)(a+1)]/2
      + D/2 Psi_r - D/4 d_r [Ez*(a+1) - Ez*(a)], Et_phi = [(E_phi - B_r)(a) + (E_phi + B_r)(a+1)]/2 + D/2 Psi_phi
      + D/4 G_phi [Ez*(a+1) - Ez*(a)] (G_phi = -(i m / r)); exact (test M6: 0 / 2e-15 with 3 modes on a stretched grid).
    - The force from the work per path segment: f_s = f_ref + (W_s - f_ref . ds) ds / |ds|^2 with f_ref the exact derivative of the
      segment work at its midpoint, so f_s . ds = W_s exactly without dividing by rhat . ds or phihat . ds (which vanish for segments
      next to the axis).
    - J_phi of m = 0 from the angular momentum of the move, wbar L / r_j with L = x0 y1 - y0 x1, instead of r_j dphi: m = 0 has no
      continuity constraint on J_phi, and r_j dphi jumps when a segment passes over the axis (dphi -> +-pi) and makes the force grow
      as r_j / r there. For m >= 1 the implicit deposit splits Delta(S_z S_r e) without the cross terms of deposit_seg (J_z from the
      folded complex radial profiles S_r e of each end, J_r with the reference phase ((z0 + z1)/(r0 + r1))^m, J_phi the rest), still
      charge-conserving per mode. The explicit deposit is unchanged.
    - Near the axis the radial force is the field gradient times xi'^2 (~12 at the axis): the Picard step of a particle whose path
      comes within 2 dr of the axis stops contracting (a period-2 oscillation the global Anderson on J does not see). Those particles
      take one Newton step per iteration (difference Jacobian, 3 more evaluations of their map); same fixed point.
    - Result (tests M6, M7): m = 0 at omega_p dt = 2.5, dz = 2 dr, hot column with paths through the axis, uniform and stretched
      grids: W + K (+ the energy of electrons taken out at the wall) to 8e-15, Gauss 5e-14, 23-26 iterations per step (tol 1e-12),
      every step converged. Accepted: modes = 1 or 2 (m = 0, 1), shape 2, irip with closure = box, z layers and the radial absorber;
      rejected: modes >= 3, o5, TFSF, window, filter, rz_transverse_shape.
    - m >= 1 open: with m = 1 the solve converged in 28 of 30 steps of the same column and stalled at 5e-5 and 4e-4 in two (a path
      within ~0.01 dr of the axis, where the m = 1 split is Lipschitz but curved on that scale; backtracking in the particle Newton step
      made it worse); even m >= 2 have no smooth deposit at the axis (the fold of decision 50 gives a particle on the axis a charge of
      mode m with its arbitrary phase) and stall regularly. With m = 1: |d(W + K)|/E 7e-5 ... 1.3e-4 over the 30 steps of M7 (m = 0:
      7e-15). Decision (Alexander, 2026-11-07): m = 1 accepted with the rare unconverged steps (counted in MoverStats; the Gauss law
      stays at round-off because every iterate deposits charge-conservingly); modes >= 3 rejected. Test M7 runs m = 0, 1 as well.
      Spread: 10 repeats of M7 with m = 0, 1 (the stalled steps depend on the OpenMP summation order): |d(W + K)|/E 5e-5 ... 4.8e-3 over 30 steps (median ~1.4e-4), 2 ... 5 of 30 steps unconverged; Alexander kept m = 1 with a loose test (W + K to 1e-2, at most 8
      unconverged steps of 30: in effect a check of the Gauss law and the density).
    - m = 0 is not yet robust at larger c dt / dr: the RZ deck of tests/mpi.sh (dz = 4 dr, stretched to dr_max = 6 dr, omega_p dt =
      2.5, ppc_r = 2) stalled in 4 of 15 steps with mover_maxit = 100 (up to 1.4e-2), |dW|/W 1.3e-5 in 15 steps; with a Newton step
      for every particle 2 stalls, with mover_anderson = 10 and maxit 300 one (step 2 at 1.9e-4). In the stalled step many slow
      particles near the axis keep changing by ~1e-4 per iteration: the global fixed point, not one particle's map. Open.

83. **RZ embedded perfect conductors (block `rings`) as a staircase inside the RIP-CC step; the z faces reflect through the
    characteristics; RIP-CC only; the RZ self-field; mode-slot guards not exchanged** (2026-11-07, for Alexander's request of a
    full-Maxwell model of corrugated structures; docs/specs/rz_metal.md, src/rz_metal.*, src/rz_beam_field.cpp)
    - Why the staircase rules: zero inside, tangential E and normal B zero on the surface, the normal E (corrector) and the
      tangential B kept. In the collocated RIP-CC grid the z face of a ring is a column holding all six components; zeroing
      E_r there and keeping the scheme's B_phi = (T+ - T-)/2 would halve the reflected wave (T- from inside the metal is 0).
      B_phi = T+ (left face) makes the outgoing T- = -T+: exact total reflection for normal incidence. Test: a cavity closed by
      discs equals the mirror-symmetric periodic box to 5e-15.
    - Gauss law: the RIP-CC identity holds for any field values entering the step, so overriding results at metal points only
      changes the residual at rho nodes whose stencil holds such a point (the surface charge): column i or i + 1 metal and
      J <= j <= K; they are left out of the check, every other node keeps round-off (1e-15 with a hot plasma, shapes 1 and 2).
    - Particles are removed when their deposit (in the density-conserving radial map of decision 53) touches only metal or
      surface nodes, so the removal changes no checked node.
    - iRIP rejected: its z lines need one radial operator for all columns (one eigenbasis per mode); the metal makes it depend
      on the column. The explicit limit stays (m = 0, uniform dr: dz <= 0.91 dr; e.g. dz = 2.09 um at dr = 2.5 um).
    - RZ self-field: phi per radial TM eigenmode and a tridiagonal line along z (the RZ form of docs/specs/beam_self_field.md),
      Gauss law 2e-13 of max rho. With a moving window the field ahead of the beam reaching the front edge leaves a frozen
      residual of its size in the last Gauss cell when the first column enters (5e-8 of max rho for gamma = 10, 0.8 mm ahead
      with R = 6.5 mm), as in Cart3D; it leaves with the window.
    - Mode-slot guards (Domain::slots): RZ fields keep 3 guard slots on each side of the mode axis, which no kernel addresses;
      the exchange zeroed them every call (62 M doubles for 6 fields on 2420 x 711 cells: 80 % of the step). Not exchanged now;
      results unchanged (regression and tests identical).
    - Amended 2026-11-12, `rings.jacket`: the closed corrugated pipe with an open end needs a second metal rectangle (a tube
      r_out .. r_out + jacket over [z_start, z_end]), so the column types became per-point rules (a node is metal if it lies
      in any rectangle; zero if both z neighbours are metal or it lies on an r surface; else a z face). Without the jacket
      they reduce exactly to the column rules (regression and all tests bit-identical). Alternatives rejected: a second
      rings block (two sets of column types, wrong faces where the two touch); a grid wall at r = b (no open end, so no
      exit radiation of the closed pipe).
84. **O7: linear-response media without particles for iRIP (`medium [<name>]` blocks): Drude, collisional and Lorentz media as a
    Crank-Nicolson current driven by the box closure's work field, the medium's own transport source eliminated locally, GMRES on
    Ex^{n+1}, Bx^{n+1} preconditioned by the vacuum box solve** (docs/specs/linear_medium.md, prototype research/linear_medium/,
    src/medium.*, src/medium_rz.cpp, src/medium_sim.cpp; the physics settled with Alexander on 2026-10-28)
    - Why: plasma mirrors and optics of any omega_p dt without particles; W + K exact (decision 81's work field), so the scheme is
      unconditionally stable and dissipates only through nu; the Gauss law holds for any current (the medium charge is -div P).
    - Per current component at the position of that current (Cart3D Jx (i, j, k), Jy (i+1/2, j+1/2, k), Jz (i+1/2, j, k+1/2); RZ Jz
      (i, r_j), Jr (i+1/2, r_{j+1/2}), Jphi (i+1/2, r_j)): wp2, w02, nu as fields (21 field arrays with the state and scratch), the
      sub-cell volume fraction of the profile with supersample^3 samples (RZ: r-weighted between the neighbouring half nodes /
      nodes; Jphi on the axis zero). Overlapping blocks: wp2 summed, w02 and nu wp2-weighted (one oscillator per point).
    - The solve: the iterate (X, Y) gives the box means, Et0 (the work field with the external current only), the local u and the
      pass with J + u; the operator X - G(X) is applied as (G(0) - G(s v)) / s + v with s = |G(0)| (no cancellation for small
      Krylov vectors). Restarted GMRES (numerics.medium_restart = 30, medium_tol = 1e-10 relative to |G(0)|, medium_maxit = 200),
      right-preconditioned by the Ex-diagonal scaling and the vacuum solve (DFT / eigenbasis + lines along x; RZ per mode). The
      start: E^n (or the last solution in the same step: O6 calls the solve once per mover iteration). medium.txt: K, the
      dissipated energy, W + K + dissipated, the GMRES statistics.
    - Supported: Cart3D, cartesian2d, RZ (all modes, axisymmetric media), periodic or open x / z, CPML in y, z (media must stay
      one cell clear of the layers, checked), z and x absorbers, particles (explicit mover; the implicit O6 mover in Cart3D: the
      trial step solves the medium, the particles see Et with the medium's own source), MPI (ranks along x / z). Rejected for
      now: closure = o5, TFSF planes, the side tube, the moving window, checkpoints / restart (J, P not written), O6 + media in
      RZ, solvers other than irip.

85. **O7, model = bound: the bound electrons of a gas (e.g. Rb vapour near an 800 nm laser) as V-type Maxwell-Bloch atoms, an
    option of the `medium` block (not the default), coupled through the box closure's work field with an exact energy exchange;
    the nonlinear atom current is linearized per node inside the GMRES solve and iterated (Newton) until it is consistent**
    (Alexander, 2026-11-08: "Maxwell-Bloch atom" rather than a linear Lorentz preset; docs/specs/linear_medium.md "Bound
    electrons", src/bloch.hpp, src/medium.cpp, tests/test_bound.cpp, examples/rb_gas_2d.deck)
    - Why: Rb D1 (795 nm) and D2 (780 nm) are 0.6 % and 2.5 % from 800 nm; at 1e12 W/cm^2 the Rabi frequency is ~30 times the D1
      detuning, the transitions saturate and Rabi-cycle, a Lorentz oscillator does not apply. The model of GEM-PIC-RZ's
      `bound_electrons` block (same physics choices: V-system, scalar dipole along the laser polarization, effective dipoles from
      the oscillator strengths, d^2 = f hbar e^2 / (2 m omega), no relaxation by default, carrier resolved without RWA).
    - Node step: rho -> U rho U^+, U the Cayley map of (dt/2) H(Et), Et the work field (decision 81). U commutes with H(Et), so
      tr(rho H(Et)) is conserved and the atoms take exactly the field's work Et u dt: [W + K]^{n+1} = [W + K]^n with K = ef n_a sum
      w_j' p_j (ef = cP'/(2a') the same for both lines). Frequencies pre-warped (exact free precession); coupling a' = a / cos x
      and polarization cP' = cP cos(xbar) / cos x (x = w dt / 2): Rabi frequency and weak-field chi exact at the line, second order
      elsewhere. GEM-PIC-RZ's scaling (coupling only, tan x / (x cos x)) would break the exact energy exchange.
    - Coupling to the fields: the atoms sit on the positions of the current component of the dipole axis (`polarization`), their
      current u_b(Et) enters like a linear medium's: u_b = u* + s (Et - Et*) with s = du/dEt by central differences of the node
      step, the medium's own -dt/2 u_b in Et_y, Et_z eliminated locally together with a linear medium on the same component; the
      GMRES solve as in decision 84; Et* <- Et of the solution until max |Et - Et*| <= medium_bound_tol max |Et| (1e-8). The
      polarization charge of the atoms is -div of the polarization the fields saw (bP += dt u_b): Gauss law to round-off.
    - Scope: Cart3D, cartesian2d and RZ, MPI, particles (explicit; the O6 implicit mover in Cart3D), CUDA kernels (the node step
      is HD). One `bound` block per run.
    - Extensions (Alexander, 2026-11-08: "go on" with RZ, ionization, relaxation, checkpoints):
      - relaxation T1, T2 and ADK ionization per level in a symmetric Strang splitting around the Cayley step; dissipated =
        field work minus the change of the atoms' energy (exact bookkeeping, includes the ionization energy);
      - ionization into species: the ionized density per atom position becomes an electron and an ion at rest once it reaches
        n_a / ionization_ppc (my choice of the creation rule, as GEM-PIC-RZ's macro-atoms do not exist here: deterministic,
        no charge deposited by a pair, the remainder waits on the grid);
      - RZ: atoms at the Jr positions on n_phi angles, the gather Et_x from the modes (axis rule E_phi,1(0) = i E_r,1(rh_0)) and
        the deposit as its transpose in the RZ energy inner product (exact for n_phi >= 2 m_max + 1); the atoms' own source
        lagged by one outer iteration (it couples neighbouring nodes), converged with Et*;
      - checkpoints with all media (bit-identical restarts on the same ranks).
      - the moving window with all media (the state shifts with the fields, the entering column from the profiles, the media
        charge in the iRIP entry cell);
    - Provisional, for Alexander's review (decided in his absence, 2026-11-08):
      - the coupling and polarization scales a / cos x, cP cos(xbar) / cos x (Rabi error 1.3e-5 at lambda/64 instead of 1.4e-3 with
        a common 1/sqrt(cos x); both keep the energy exchange exact);
      - the pair-creation rule (quantum n_a / ionization_ppc, default 8 pairs per atom position, electrons at rest at the atom);
      - ADK at |Et| of the dipole component only (linear polarization along the dipole axis), constant beyond the ADK maximum;
      - RZ: atoms at the Jr positions with E_phi interpolated linearly; the lagged own source (6 outer iterations per step in the
        dense random test, ~2-4 in the decks);
      - one bound block per run; atoms clear of absorbers, CPML layers and the last radial cells.
      - open (found by the field-ionization session, no change yet): the ADK prefactor of bloch.hpp (taken over from GEM-PIC-RZ)
        is C^2 = 2^{2n*} / (n* Gamma(n* + 1) Gamma(n*)), the l* = 0 form, with no (2l + 1) factor for p states, while
        src/ionization.cpp (decision 86) uses the standard l* = n* - 1: the Rb rates here are 2.4x (5s) and 1.4x (5p) higher.
        Which form Alexander wants for the bound atoms (and whether GEM-PIC-RZ's comment "l* = n* - 1" should be corrected there).
    - With it (all media): the GMRES residual is measured relative to max(|G(0)|, 1e-4 |F|) (|F| the norm of the field state):
      for an s-polarised wave along x, Ex, Bx ~ 0, |G(0)| is round-off and the relative residual stalled at ~2e-9 (200 GMRES
      iterations every step); now 0.5 per solve.
    - Numbers: docs/specs/linear_medium.md "Bound electrons" (Rabi flopping second order, 1.3e-5 at lambda/64; weak-field phase vs
      the Lorentz chi 6.8e-4 at lambda/40, ~25 times better than the unwarped linear Lorentz medium near the line; W + K_b to
      1.7e-15 in saturation; Gauss law 4e-15).

86. **Phase 3, field ionization: a charge state per particle (species q = 1, attribute charge_state), ADK per level with the
    PPT/ADK coefficients for arbitrary l (l* = n* - 1, |m| = 0), Monte-Carlo sampling of successive levels by exponential waiting
    times in E^n, new electrons at the ion with its velocity, and the energy-conserving ionization current as the
    charge-conserving displacement of the new electron to the tunnel exit. All physics choices provisional, for Alexander's
    review** (2026-11-08, Alexander away: "best judgement, the standard well-documented option"; docs/specs/field_ionization.md,
    src/ionization.*, src/atomic.*, data/ionization_energies.def, tests/test_ionization.cpp, bench/ionization_smilei/)
    - Charge state per particle, not one species per charge state (EPOCH's way): push, current and charge deposit multiply q
      by Z[p] when the species has the attribute (a null pointer otherwise: species without it are bit-identical); one ion
      species per element, the tiles, migration, openPMD and checkpoints carry Z like any attribute. The box deposit's
      fixed-point scale uses max_charge_state.
    - Rate (provisional): the static ADK rate (the carrier is resolved, no cycle average) w = C^2 f(l, m) I (2 F0/F)^(2n*-|m|-1)
      exp(-2 F0/(3F)) with C^2 = 2^(2n*) / (n* Gamma(n* + l* + 1) Gamma(n* - l*)), l* = n* - 1 (ADK 1986), f(l, m) = (2l + 1)
      (l + |m|)! / (2^|m| |m|! (l - |m|)!), l the orbital of the removed electron from the ground configuration, |m| = 0: the form
      of Smilei (ionization_model = "tunnel"), so that the Smilei benchmark compares the same model. Hydrogen-like I = 1/2:
      the Landau rate exactly (test I1a, 6e-16). GEM-PIC-RZ uses the s-state form (f = 1, and C^2 with Gamma(n* + 1) Gamma(n*),
      i.e. l* = 0, although its comment says l* = n* - 1): this / GEM-PIC-RZ = 1.000 (H), 1.28 (He), 3.10 (N0, l = 1), 0.39
      (N4+, n* = 1.86), 1.06 (N5+), 3.22 (Ar0), 0.19 (Ar7+), 1.36 (Kr8+), 2.83 (Xe0), 0.80 (Au0). The threshold field moves by a
      few % only (exponential field dependence). The deck keys follow GEM-PIC-RZ's ion block (element, charge_state,
      ionization = adk, max_charge_state, ionization_potentials), so either convention could be adopted in both codes;
      Alexander to decide (and whether GEM-PIC-RZ's comment or formula is the intended one).
    - Beyond barrier suppression (provisional): the ADK expression is held at its maximum F = (2/3) F0 / (2n* - |m| - 1) (it
      falls beyond, unphysically; GEM-PIC-RZ's convention). Option ionization_bsi = tong_lin: the Tong-Lin factor exp(-alpha Z^2
      F / (I F0)), alpha = 6 (Tong & Lin 2005; Smilei's tunnel_TL default), also held at the maximum of the corrected
      expression. Not chosen: Posthumus or Bauer-Mulser linear BSI rates (element-specific fits; the Smilei tunnel_BSI model of
      Ouatu et al. 2022 could follow if needed).
    - Sampling (provisional): Monte Carlo per ion with successive levels by exponential waiting times tau_k = -ln u_k / w_k(|E^n|)
      while sum tau < dt: the exact sample of the rate equations for a field constant over the step, at any w dt (I2b at w dt =
      2.3, 0.5: several levels per step, 1.4 sigma). Rejected: the rate-integrated (deterministic) scheme, which needs fractional
      charge states or splitting the macro-ions; Nuter's one-draw cumulative Bateman probabilities (equivalent, but ill-
      conditioned for close rates). The field is E^n held over the step (left point): E^{n+1} is unknown before the push; for the
      pulse of test I3 at lambda/40 the held field changes the final fractions by 8e-6 (Python, continuous vs held), GEM-PIC-RZ's
      trapezoid over its two slices matters there because its steps are long.
    - Data (Alexander, 2026-11-08: "add ALL elements ionization states and energies from NIST"): every charge state of H .. Lr
      (103 elements, 5357 levels) and D from NIST ASD 5.12, retrieved 2026-11-08, in data/ionization_energies.def (built by
      tools/make_ionization_table.py from the NIST downloads; Rf .. Ds lack stages in the ASD). l of each level: the last
      subshell of NIST's ground configuration of the ion; masses: NIST standard atomic weights. Identical energies to
      GEM-PIC-RZ's 21 elements.
    - Random numbers: u_k = hash(seed, step, species, id, k) (splitmix64), id = a hash of the ion's position when ionization
      first sees it (its load position: the same on every decomposition), so CPU, GPU, any thread count, rank count or particle
      order draw the same numbers (I2c: identical counts with the ions reversed).
    - New electrons (provisional): one macro-electron per ionized level, weight of the ion, u = the ion's u (u^{n-1/2}): with it
      the leapfrog's discrete kinetic energy u^{n-1/2} . u^{n+1/2} / 2 of the new electron is zero at its birth, consistent with
      the field work from t_n on. Appended in the order of their ions (count, exclusive scan (CUB on the GPU), fill).
    - Ionization current (provisional): the electron is born at x_ion + d, d = I_k E^n / (q_e |E^n|^2) (the tunnel exit), and the
      Esirkepov current of the displacement (RZ: one charge-conserving segment per mode) goes into J^{n+1/2} with the push. Its
      work is q_e w d . E^n = w I_k exactly (E_cur / E_ion - 1 = 7e-16), the field loses that energy to the time centring (I4a:
      4.6e-4), and the current is Nuter's J = I_k w E / (|E|^2 dt) together with its dipole charge, so the Gauss law stays at
      round-off (I5: 1e-16 in Cart3D, 2D and RZ). Smilei 5.1 deposits Nuter's current at the ion without that charge
      (Projector::ionizationCurrents: the divergence of every event's current is unbalanced) and none in its AM (RZ) geometry.
      |d| is capped at half a cell (never reached at physical fields: 0.5 nm for H at its threshold).
    - Not in this block: the implicit mover (O6: the current would have to enter its nonlinear solve), self_field of ionizable
      species, ionization of the Maxwell-Bloch medium (O7c), collisional ionization (next block of Phase 3), restarting E_ion /
      E_cur (they start from 0 after a restart; the charge states are in the checkpoint).
87. **Lorentz-boosted frame (`boost.gamma`, block `lab_diagnostics`): the deck stays in the lab frame and is transformed;
    lasers enter at the boosted entrance plane, plasma at rest streams in through a plane 3 cells behind the right edge,
    lab-frame snapshots by back-transformation (WarpX / FBPIC style); no NCI mitigation** (2026-11-08, Phase 2 leftover;
    the physics choices below are provisional, for Alexander's review; docs/specs/boosted_frame.md, src/boost.*,
    src/boost_diag.cpp, tests/test_boost.cpp, examples/lwfa_boost.deck, examples/rz_lwfa_boost.deck, bench/nci/README.md)
    - Frames: x = gamma (x' + beta t') + X0, t = gamma (t' + beta x') + T0 with X0 = T0 = -gamma beta t'_ref (t'_ref the window
      start, boosted): after the window starts, the box covers exactly the lab moving window (light lines t - x =
      window.start and window.start - L); the lasers are injected at lab events x <= 0, before the plasma of a lab deck.
    - Grid: dx' = gamma (1 + beta) dx_lab (FBPIC's copropag_length): the same cells per Doppler-shifted wavelength; t_end and
      window.start are lab times. For gamma >~ 2 RIP-CC's Courant condition (3D, 2D) and RZ condition need wide transverse
      cells, iRIP has none: iRIP is the solver for boosted runs.
    - Laser (provisional): the inbound characteristic at x' = 0 is gamma (1 - beta) (E + B)_lab of the paraxial beam at the
      lab event (LaserConfig::field_boosted, field_at): frequency, duration, Rayleigh length transform by themselves. The
      price: the injection runs over lab x in [X0, 0], where a beam focused downstream is wider and must fit the transverse
      box (summary warning). The alternative of WarpX / FBPIC, an antenna fixed in the lab (moving at -beta c here), has no
      such constraint but needs a moving current sheet with its charge (the exact Gauss law of VLPL); proposed as the next
      step. Open-edge injection in Cart3D, 2D and RZ, and the RZ TFSF plane; LASY files and the Cart3D TFSF plane refused.
    - Plasma injection (provisional): a lattice comoving with the plasma (q = x' + beta t', ppc per boosted cell), the profile
      evaluated at the lab position gamma q + X0; each step the lattice points that crossed the plane x'_inj = right edge -
      3 dx' enter. Nothing is frozen (the lab window's 3 frozen columns would leave gaps in a streaming plasma); no current
      reaches the last cell. Weights carry the cold density factor gamma (1 - beta v_x) of the drift, the same for every
      species at a lattice point (neutral loading): the per-particle flux factor of a thermal spread would put net charge at
      each injected point (Gauss residual 6e-3 at u_th = 0.02); its omission is an O(beta u_th) error in the boosted thermal
      distribution. Plasmas must be at rest in the lab and mobile (an immobile lab background moves at -beta c).
    - Beams (self_field or ux != 0): loaded once at the boosted time of their centre's event, ballistically mapped from the
      lab time t_start (lab length / (gamma (1 - beta v))), momenta boosted, the self-field with the relative gamma.
    - Lab diagnostics (provisional, the WarpX scheme): one plane per boosted step and snapshot, cubic Lagrange along x' (all
      components at integer x; the transverse pairs are co-located, so the transformation needs no transverse interpolation),
      planes dt'/(gamma beta) apart in the lab; particles at the step's crossing of the lab time, interpolated on the move;
      beam moments in the lab frame (lab_beams.txt). Linear interpolation lost 2 % of a carrier at 16 cells per wavelength.
    - NCI (measured, no mitigation): the streaming plasma at gamma ~ 10 and 30, RIP-CC and iRIP, 3D and RZ, shapes 1 and 2,
      also iRIP with h_x = h_y: W_field / W_kin <= 1.5e-11 after omega_p t = 282, late rates <= 0.024 omega_p (Yee: 9 orders
      at 0.24 omega_p). Both solvers are dispersion-free along x, so no Galilean scheme or filter (bench/nci/README.md).
    - Resolution (measured): in the boosted frame the plasma streams into the laser, so a particle crosses a laser cycle in
      N/(1 + beta) steps (N cells per wavelength), against N in the lab: the quiver motion, and with it the ponderomotive drive,
      is time-resolved (1 + beta) times worse. a0 = 1 wake, 0.005 n_c, gamma = 2: at lambda/16 the boosted wake was 15 % (2D)
      and 21 % (RZ) above the lab run's (which hardly changes with dx: 0.0150 -> 0.0148), at lambda/32 1.9 % and 0.5 %
      (scan in RZ at lambda/16: 6 %, 8.6 %, 21 % at gamma 1.05, 1.3, 2). Rule (provisional): dx_lab <= lambda/(16 (1 + beta))
      for the same drive accuracy as a lab run at lambda/16; the speed-up quoted at equal accuracy (summary of the PR).
    - Tests (tests/test_boost.cpp): vacuum laser (4 cycles, waist 5 um, focus 20 um, gamma 2) boosted vs lab run 1.25e-2 (2D
      RIP-CC), 2.45e-2 (RZ iRIP), vs the paraxial beam 2.7e-2, 5.4e-2 (the lab run 1.8e-2, 3.2e-2: the monochromatic formula
      imposed on different lab planes for a broadband pulse); wake (above) with the laser peak 2.8 %, 1.1 %; Gauss law 2e-15 with
      plasma streaming through (2D RIP-CC cold, 2D iRIP hot, 3D RIP-CC cold, RZ iRIP 1.3e-14); ballistic gamma = 20 beam in the
      lab frame: charge +0.8 % (lattice), mean energy exact, centre 1e-3 / k0, length +0.8 %.
    - Speed-up (examples/rz_wake_boost.deck, the B2 wake over 130 um, RZ iRIP, 24 threads of one cluster node): lab lambda/16 2600
      steps 48.5 s, lab lambda/32 5200 steps 182 s; boosted gamma 2: lambda/16 628 steps 11.1 s (4.4x, but the wake 15-21 % off),
      lambda/32 1255 steps 42.2 s (4.3x against the lab run on the same grid, 1.15x against the lab lambda/16 run of the same
      accuracy); gamma 4, lambda/32: 1018 steps 33.5 s (5.4x). This short case is dominated by the static start (the window
      start maps to gamma (1 + beta) times its lab value); for an interaction length much longer than the box the step ratio
      tends to gamma^2 (1 + beta)^2, and with the (1 + beta) finer cells the cost ratio at equal accuracy to about gamma^2.
    - MPI (tests/mpi.sh): 3D RIP-CC 1 vs 8 ranks energies 1.6e-12, lab beams 4.9e-10, lab fields 8.3e-15; RZ iRIP 8 ranks along
      z 2.4e-13, lab beams identical, lab fields 1.7e-14; Gauss law <= 1.8e-15. RTX 5090: test_boost and both examples as on
      the CPU (lab fields 4e-15, lab_beams.txt identical); CUDA sm_90 and sm_120 compile.

88. **The code is VLPL (formerly RIP-PIC); the method names stay (Alexander, 2026-11-08).** The GitHub repository
    is now AlexMPukhov/VLPL (formerly AlexMPukhov/RIP-PIC), and the code follows: the code name in all documents, the banner and
    the log, the openPMD `software` attribute ("VLPL"), the binary (`build/vlpl`, formerly `build/rippic`), the job and log
    names of the scripts, and the environment variables (`VLPL_CUDA_AWARE`, `VLPL_MEDIUM_FAST`, `VLPL_MEDIUM_DEBUG`,
    `VLPL_MEDIUM_DIAG`, `VLPL_MOVER_DEBUG`, `VLPL_CI_ENV`, `VLPL_CI_MAKE_ARGS`).
    - Names with the solver (Alexander, 2026-11-08): VLPL-RIP (explicit solver RIP-CC), VLPL-iRIP (implicit iRIP), VLPL-RZ
      (RZ with RIP-CC), VLPL-iRZ (RZ with iRIP); VLPL alone is the code as a whole (repository, binary, openPMD software).
      The summary banner names the variant of the run (VLPL with the Yee solver).
    - Kept: RIP (the "rhombi in plane" solver of Pukhov, JCP 418, 109622 (2020)), RIP-CC and iRIP are method names, and so
      are `numerics.solver = rip_cc | irip` and the rip_* / irip_* identifiers; deck keywords are unchanged.
    - Compatibility: the old variable names (formerly RIP_<X>) are deprecated aliases: the binary reads them when VLPL_<X> is
      unset and warns once on rank 0 (src/env.hpp); bench/jupiter/env.sh maps the old CUDA-aware variable and notes old
      build folders that have only the old binary; the CI workflow accepts the old runner variables. Not renamed, to keep
      old output files and checkpoints readable: the on-disk record names (formerly RIP-PIC's own records: meshes
      rippic_gauss0, rippic_pml_*, rippic_fluid_j*, attribute rippic_r_nodes, checkpoint group "rippic"). Compile macros
      RIP_MPI, RIP_HDF5, RIP_BOX_MINB* are internal and stay. Paths of existing folders on the CPU cluster and file names of old logs
      in the documents stay as they are.
    - The mechanical part is the script a rename script (development repository) (idempotent; re-run after merging branches that were
      started before the rename).

89. **Boosted frame: the lab-fixed laser antenna (`laser.injection = antenna`), a moving polarization sheet with its charge,
    charge-conserving in the Gauss stencil of RIP-CC / iRIP** (2026-11-09; provisional, for Alexander's review;
    docs/specs/boosted_frame.md "Antenna", src/boost_antenna.cpp, tests/test_boost_antenna.cpp)
    - Why: the entrance plane of decision 87 injects at lab x in [X0, 0], |X0| = gamma beta gamma (1 + beta) window.start
      (83 um in the B1 test), where a beam focused far downstream is wide and gets cut by the transverse box. WarpX and
      FBPIC use an antenna fixed in the lab, which moves at -beta c in the boosted frame.
    - Source (provisional): a current sheet at the lab plane x = 0, K = -2 E_perp of the deck's paraxial beam at x = 0 (the
      tangential B of the forward beam fixes the sheet current; paraxial B = E), i.e. the field the lab run injects at that
      plane; focus and delay keep their meaning, a lab run of the same deck injects at x = 0 as before. The sheet radiates
      the mirror beam backwards as well; it leaves through the open left edge (in the boosted frame it carries
      gamma (1 + beta) / (gamma (1 - beta)) times the forward field and is under-resolved, but RIP transports it along x
      exactly). Alternative not taken: a one-way (Huygens) pair of sheets, which needs magnetic currents or two sheets a cell
      apart, neither aligned with a sheet that moves by beta h per step.
    - Charge (the exact Gauss law of VLPL): the sheet is a polarization P = Pi delta(x), Pi = int K dt, with rho = -div P,
      J = dP/dt; boosted: rho' = sigma delta(x' - x'_a), J'_perp = dPi/dt' delta, J'_x = -beta rho'. Discretely: Pi summed on
      the transverse plane each step (midpoint K at the lab time of the antenna's world line, times dt / gamma), sigma =
      -div_perp Pi with the Gauss stencil, a quadratic spline in x, J_perp with the time-averaged spline weights, J_x by the
      Esirkepov split of the moving spline (C^{n+1} - C^n of its cumulative weights): rho^{n+1} - rho^n = -dt div J
      identically, as the media's -div P (decision 84). No particles (WarpX uses macroparticle pairs): no per-particle
      state, no MPI handover; the state is Pi on the transverse plane, re-summed from step 0 after a restart.
    - The weights (provisional, numerics): RIP moves the forward wave exactly c dt per step and the sheet moves -beta c dt, so
      a forward characteristic meets the sheet's nodes s = (1 + beta) c dt / h_x cells apart. A unit-width quadratic spline is
      no partition of unity at that spacing: the first version modulated the emitted beam at the grid scale and was 7.4 % (2D)
      and 5.8 % (RZ) off the lab run. The spline is therefore stretched to s cells and normalised per step: every
      characteristic gets the weight 1/s. Its smoothing in time (variance 1/4 + (beta / (2 (1 + beta)))^2 steps) is
      pre-compensated, K - (var / 2) (K_+ - 2 K + K_-) with K one step earlier and later on the world line (fourth order).
    - Measured (tests/test_boost_antenna.cpp; vacuum, a0 = 0.01, 4 cycles, waist 3 um, focus 70 um, z_R = 35 um, gamma 2,
      lambda/16; the box +-17.5 um holds the beam at x = 0 (6.7 um) but not at X0 = -83 um (13.3 um)): lab snapshot of the
      boosted run against the lab run 7.3e-3 (2D RIP-CC) and 4.4e-3 (RZ iRIP, 2 modes) with the antenna, 3.3e-2 and 4.4e-2
      with the entrance plane; at lambda/32 1.1e-3 and 0.94e-3 (antenna) against 3.3e-2 and 4.8e-2 (the plane's cut does not
      converge). Against the paraxial beam the antenna is as close as the lab run itself (3.2e-2 vs 3.2e-2, 9.2e-2 vs 9.1e-2
      at lambda/16: the monochromatic formula, and in RZ the wall at 17.5 um). The residual at lambda/16 is an amplitude
      factor (fit 0.993 in 2D, 0.996 in RZ, correlation 1.00000). Gauss law with the moving sheet and cold plasma streaming
      through the window (gamma 3, a0 = 1, focus 3 um, off the axis): 1.8e-15 (2D RIP-CC), 1.8e-15 (2D iRIP), 2.2e-15 (3D
      RIP-CC), 2.5e-15 (RZ iRIP, 2 modes); in vacuum 5e-18. Cost: the operator is < 1 % of the step (3D example).

90. **Field ionization in the boosted frame: the ADK rate per ion from |E| in its rest frame over its proper time; the
    ionization current takes (gamma_i + u . S_rest) I_k (corrected 2026-11-28, below); the step of the first pass in the
    id** (2026-11-09; provisional, for Alexander's
    review; docs/specs/field_ionization.md "Boosted frame", docs/specs/boosted_frame.md, src/ionization.*, tests/test_boost.cpp
    B4, examples/ionization_boost.deck, examples/rz_ionization_boost.deck). Decision 87 refused ionizable species: ions at
    rest in the lab stream at -beta c in the boosted frame, and the field there (a laser: gamma (1 - beta) E_lab) is not the
    field that ionizes them.
    - Rate: |E_rest|^2 = |gamma_i E + u x B|^2 - (u . E)^2 from E^n and B^n gathered at the ion (the push's gather) and its
      u^{n-1/2}; the waiting times over the proper time dt / gamma_i. For an ion at rest in the lab this is exactly the lab
      field and the lab time (the rate equations are those of the lab run).
    - New electrons: the ion's position and velocity, as in the lab (one macro-electron per level, the ion's weight).
    - Correction (2026-11-28, Alexander: "fix the bug first"): the rest-frame transfer is not (I_k, 0) but (I_k, I_k S_r),
      S_r = E_r x B_r / |E_r|^2: the tunnel-exit dipole also takes the momentum q_e d_r x B_r (the photons' I_k / c for a
      plane wave). Its energy in the simulation frame is (gamma_i + u . S_r) I_k (ion_energy_factor), gamma (1 - beta) I_k
      for a forward laser instead of gamma I_k: the old rule took 1 / (1 - beta) too much (17.5 at gamma = 3), seen as a
      laser depletion about three times too large in a gamma = 3 helium run. The text below is the original (wrong) rule.
      Tests: B4 ion_energy_factor vs the photon Doppler factor gamma (1 - V . k) (1000 random plane waves and frames); the
      energy paid per ionization in the boosted nitrogen runs vs the lab runs.
    - Ionization current (original, replaced by the correction above): the energy of the rest-frame transfer (I_k, 0) is gamma_i I_k in the simulation frame; the
      displacement d = gamma_i I_k E / (q_e |E|^2) along the simulation-frame E does exactly that work, and the Gauss law
      stays at round-off. E_ion = sum gamma_i I_k w (simulation frame; E_cur / E_ion = 1). The momentum gamma_i v I_k of the
      transfer and the mass change of the ion are neglected (as in the lab). W_field + W_kin + E_cur in ionization.txt is no
      check in the boosted frame (plasma kinetic energy flows through the box; a new electron is born with the kinetic energy
      (gamma_i - 1) m_e c^2 of the streaming velocity while the ion keeps its mass).
    - Random numbers: the id hash of the load position also mixes in the step of the ion's first ionization pass (the step
      after its injection; the same for any decomposition): plasma enters through the injection plane in every step, and two
      lattice points loaded at the same x' in different steps would otherwise share an id.
    - Only in the boosted frame (`boost` on). Lab runs keep the simulation-frame |E| and dt and the old hash: bit-identical
      to main (examples/ionization_2d.deck and rz_ionization.deck: diag.txt and ionization.txt identical). Choices for
      Alexander: the rest-frame rate only with `boost` (a lab run with relativistic ions would need it too; switching it on
      there would change lab results of moving ions); gamma_i I_k as the energy taken; the step in the id.
    - Numbers (2026-11-09, the CPU cluster CPU; tests/test_boost.cpp B4, gamma 2, iRIP, lambda/32, nitrogen 1e-4 n_c, 8 fs pulse): rest-frame
      field vs the Lorentz transformation 1.8e-15; charge states boosted vs lab (sigma from both runs' N_eff, 3e4 - 6e4 ions):
      2D a0 = 0.0105 N1+ 0.5244 / 0.5238 (0.15 sigma), RZ 0.4552 / 0.4578 (0.91 sigma), 2D a0 = 0.03 N3+ 0.0166 / 0.0161
      (0.40 sigma, up to three levels per ion); Gauss law <= 3.4e-17; E_cur / E_ion - 1 <= 2.2e-16; 748 boosted against 3088 lab
      steps. tests/mpi.sh 1 vs 8 ranks: 2D ionization.txt identical (17163 events), RZ 1.1e-13 (39236 events in both). CUDA
      sm_120 and sm_90 compile; RTX 5090: B4 with the same fractions and event counts as the CPU. Lab runs: byte-identical
      to main with 1 thread (diag.txt, ionization.txt of both ionization examples); with 32 threads only max|dGauss| differs
      at 1e-14, as between two runs of main.
    - Fixed with it (Alexander, 2026-11-09): (1) the RZ loader filled a species with density = 0 (the electrons of an
      ionizable species) with zero-weight particles (102400 in examples/rz_ionization.deck); in the boosted frame they
      streamed in through the injection plane every step (1.6 M in examples/rz_ionization_boost.deck, 4x the run time). Now
      skipped as in Cart3D's load_cells; B4 checks that the electron species starts empty and ends with one macro-electron
      per event. RZ lab results change only in the particle count of diag.txt (zero weights carry no charge or energy).
      (2) The boost summary warned that the box cuts the beam at its lab entrance events also for a beam already wider than
      the box at its focus (a plane wave over a periodic y); now only when the box holds the beam at its focus
      (boost_beam_cut, test B5).
91. **The current filter smooths every component (`numerics.filter = binomial`, the new default); the window keeps the plasma
    that enters frozen for 3 + filter_passes cells and its charge as a background** (Alexander 2026-11-09; found by the
    paper session, docs/specs/solid_target.md, docs/specs/rz_modes.md "Current filter", docs/specs/moving_window.md).
    - Why (NCI): the projection (69, 70) and the local filter (54, 64) change only the longitudinal current, while rho gets
      T = T_x T_y T_z. For a drifting plasma J = v rho then no longer holds at the grid scale: the electric force of the
      smoothed charge and the magnetic force of the unsmoothed transverse current no longer cancel to 1/gamma^2. A relativistic
      drift then becomes violently unstable. Streaming-plasma test (examples/streaming_plasma.deck, u = -10 / -50,
      h_perp = 3 c dt, 1 pass), W_field / K_e at 282/wp:
      3D RIP-CC: none 2.7e-9, projection 0.98, local 13.8, binomial 1.3e-10.
      3D iRIP: none 1.2e-8, projection 1.23, local 31.6, binomial 1.7e-10.
      RZ RIP-CC: none 2.6e-9, projection 0.78, local 2.3e-10, binomial 2.2e-10.
      RZ iRIP: none 4.7e-9, projection 0.82, local 2.4e-10, binomial 2.2e-10.
      gamma 50, 3D RIP-CC: none 2.1e-10, projection 0.085, local 0.24, binomial 2.3e-11.
      gamma 50, RZ RIP-CC: none 1.9e-10, projection 0.055, binomial 4.5e-11.
      Growth rate (energy): 0.40 wp (gamma 10), 0.19 wp (gamma 50) for the projection and the 3D local filter; none for the
      binomial filter. At h_perp = c dt / 4 (iRIP, gamma 10): none 6.6e-4 (3D) / 1.3e-3 (RZ), projection 0.36 / 0.12, binomial
      2.5e-10 / 5.2e-10. FBPIC with its binomial filter: 1.2e-9 (standard), 1.7e-10 (Galilean); bench/nci/README.md.
      The RZ local filter is stable because its radial step smooths J_z by the T of rho; only its z step is longitudinal.
    - The binomial filter: per pass and axis a, J_a <- S_a J_a with S_a = 1 + (h_a^2/4) G_a D_a (the step of decision 64, with
      w = d_a J_a only), and J_b <- T_a J_b for b != a (the binomial along a over the Gauss range of a, Neumann at its ends;
      J_b lies at the a-position of the charge nodes). Then D J' = T D J exactly on every boundary and decomposition. In the
      interior every component gets the binomial (1, 2, 1)/4 along every axis, as rho, so J = v rho holds at the grid scale.
      There is no transform, so there is no restriction on y, z (CPML, ranks in y, z). RZ: the z step is S_z on J_z and T_z on
      J_r, J_phi; the r step is the one of decision 54 (T_r on J_z and rho, S on J_perp).
      Tests: S9 (identity 4e-16; a plane wave in every component x prod cos^2(k_a h_a/2) to 3e-15), R25 (RZ identity 8e-16,
      transfer 6e-16), S10 (NCI on 16 x 8 x 8: projection 0.59, binomial 1.0e-10, none 1.4e-9).
    - Price: the transverse current is smoothed too, so radiation from grid-scale currents is damped (harmonic N of a laser
      by cos^2(N k0 h / 2) per pass), as in FBPIC. Decks where the harmonics must stay unfiltered keep the projection
      explicitly: bench/hhg_oblique (rip.deck, rip_2d.deck), examples/foil_2d.deck, examples/solid_target.deck. The projection
      and the local filter stay selectable; they must not be used with a relativistically drifting plasma (beams, injected
      LWFA electrons). bench/chf sets local explicitly (unchanged). B7 (bench/fbpic_nm_scan) now uses the default binomial.
    - Window, filter (the B7 finding: 3e-4 of max rho in every mode with warm plasma, 1.5e-12 without the filter): at every
      shift the last Gauss cell becomes an inner one. T^p then changes on the last p + 1 Gauss cells, and the Gauss residual
      changes by -(T_new^p - T_old^p) rho there. The 3 frozen entry columns of decision 22 kept the charge of moving particles
      out of the last 2 nodes only. Now the entering plasma is frozen for F = 3 + filter_passes cells (window_freeze_cells; 3
      without the filter, bit-identical).
    - Window, non-neutral entering columns (the 1.95e-7 of bench/lwfa_bubble while its ramp entered, no filter): electrons 2 x 2 x 2 and protons 1 x 1 x 1 per cell put different charges on the nodes where the density is
      not uniform (ramp, front edge). That charge appears with no current and no field, exactly like a non-neutral initial
      load, which gauss0 holds as a background. The column that leaves the frozen cells now has its charge (filtered as rho)
      subtracted from gauss0 (window_fold_entered; F + 1 cells behind the edge, out of reach of the changing closure). The
      Gauss check adds the charge of the columns still frozen (window_frozen_rho). The fields are unchanged (RIP-CC and the
      iRIP entry cell bit-identical); gauss0 is in the checkpoint, and the frozen part is recomputed after a restart.
      Tests: W8 (Cart3D, warm non-neutral ramp entering, RIP-CC / iRIP: 5.9e-16 / 6.4e-16, net entering charge 6.3e-3; before
      ~rho of the column), R27 (RZ, warm non-neutral, 2 passes: iRIP + projection 3.2e-15, RIP-CC + binomial 4.0e-15),
      tests/mpi.sh (1 vs 8 ranks, restarts).
    - Window start and the laser head (the 1.05e-4 of bench/lwfa_bubble, also in max|div B|, from step 620, before any
      plasma entered): the first shift comes after the step that ends at start + h_x / v. A pulse injected from t = 0 then has
      its head at c (start + h_x / v), and with c start = L (29.98 um of 30 um in the deck) one cell of the head was cut at the
      open edge. The start warning now tests c start + h_x / v > L (it tested c start > L), and the deck starts at 99.8 fs.
      Full bubble run on the RTX 5090 after both fixes: max|dGauss| 1e-14 while the plasma enters, at most 2.8e-12 later
      (the injected bunch, as in October), max|div B| 1.2e-14; charge and spectrum at 900 fs unchanged (118 pC, 75 / 188 MeV).
      B7 at N_m = 32 with warm plasma entering: 6.5e-16 (binomial) and 6.4e-16 (projection), before 1.8e-7 (3e-4 of max rho).
      B7 with the laser (now the binomial filter, two passes): W_0 3.94e-4 J at N_m = 2 (FBPIC 3.78e-4, projection 4.10e-4),
      W_1 0.268 J; N_m = 10: W_9 2.2e-7 J; N_m = 16: W_15 8.6e-8 J.
    - Cost: a pass over the particles per shift (the column that leaves the frozen cells) and per diagnostic (the frozen
      columns). B7 N_m = 2 on 16 epyc cores: the moving_window operator 0.8 -> 2.9 s over 1199 steps (wall 58.8 -> 62.7 s).
    - Not changed: Cart3D still refuses the filter with a moving window (the left-edge rule of decision 54 is RZ only).
      A window at v < c and the boosted frame have no frozen columns: the background bookkeeping is off there.

92. **Multi-GPU: the current guard exchanges overlap the push of the interior particles, the migration runs on the
    device, the RIP-CC predictor is not exchanged** (2026-11-09, branch gpu-overlap; docs/GPU_TESTING.md "Multi-GPU
    overlap", bench/jupiter/README.md)
    - Why: JUPITER 2026-11-03, bubble_strong on 16 GH200 (CUDA-aware): 10.5 ms per step, of it particle migration 3.6 ms
      and field solve 2.2 ms, nearly all guard exchanges (17 halo calls per step of a window run, each a pack, a stream synchronisation and
      an MPI round), push 4.6 ms. Nothing overlapped: every halo ended in a device synchronisation, the migration sorted
      the list of leavers on the host and sent 12 messages (2 species x 3 axes x size + data).
    - Push (GPU, several ranks; `VLPL_OVERLAP=0` switches it off): the boundary particles (cell within 2 g of a face of
      the owned box, g the guard width) are pushed first in a high-priority stream; that kernel marks the interior ones
      (one byte per particle, from the positions before the push: a boundary particle moved into the interior must not be
      pushed twice); the interior particles follow in a low-priority stream while the current guards are summed
      (add_guards) and copied back (the first exchange of the RIP-CC step, now done in the push; Fields::j_ready). An
      interior particle reads and writes only cells in [g, n - g), the exchanges write the guards and the edge layers
      [0, g), [n - g, n): no race. Per particle the same kernel; the currents differ only in the order of the atomic
      additions, which on the GPU is not fixed anyway. Off on the CPU (one host thread: nothing to overlap) and whenever
      something else writes the currents between push and field step (filter, ionization, side tube, media, boosted
      antenna), with the tiled / box deposits and the implicit mover. RZ keeps its own push (not overlapped).
    - Guard exchanges (Comm::halo): every call runs in a high-priority non-blocking stream that starts after the caller's
      stream (event) and that the caller's stream waits for afterwards; staging copies asynchronous; the host waits only
      for that stream before MPI. An axis periodic with this rank: one kernel for both directions (was a pack and two
      unpacks). Bit-identical.
    - RIP-CC predictor (Exs, Bxs): computed one cell into the guards from the guards of E, B, J instead of exchanged (the
      transport reads it at radius 1): the same numbers as the exchange gives, 3 halo calls per step fewer. Not with a side
      tube (its face corrections need the exchange). Bit-identical (CPU decks against main).
    - Migration: flags on the device, CUB scan, one pack kernel and one hole-fill kernel over all attributes (4-byte words:
      rows with F32 / I32 attributes put F64 at 4-byte offsets), the rows of all species to the host in one copy; one
      message per axis and direction for all species (byte counts first); one MPI round per message (MPI_Mprobe, no size
      message); the host keeps staying rows as segments of the buffers they are in. The particle order and the summation
      order of the lost energy are those of the former algorithm: bit-identical (test_overlap O1, CPU decks against main).
    - Not done (would change results or conventions): the 26-neighbour exchange (one round instead of three, but the
      add order of corner guards changes: round-off), moving the migration before the field step (the window shift moves
      the box first), load balance (Phase 5). The remaining critical-path exchanges per step of a window run: T and Ex, Bx
      of the RIP-CC step (3 + 3 calls) and the window's two x exchanges.
    - Profiler: `VLPL_PROFILE=1` (src/prof.hpp) prints ms per step of the operators and their parts, max and mean over the
      ranks, from step `VLPL_PROFILE_FROM`; `VLPL_PROFILE_SYNC=1` synchronises every scope and puts a barrier in front of
      every operator (exact attribution, no overlap). NVTX ranges in the GPU build.
93. **Phase 4, spin-resolved QED: T-BMT as an exact rotation per step for every species with `spin`; Monte-Carlo
    nonlinear inverse Compton emission (LCFA) resolved in the lepton spin and the photon Stokes vector, mixed spin states
    (|S| <= 1) by default, after Algorithm 1 of the group's QV3D QED draft; own rate tables instead of PICSAR-QED; photons as
    a species of kind photon (optional, else tallies). Physics choices provisional, for Alexander's review** (2026-11-09,
    branch qed-spin; docs/specs/qed.md, src/qed.hpp, src/qed.cpp, src/qed_tables.cpp, tests/test_qed.cpp,
    examples/qed_collision.deck, bench/qed_pwfa/)
    - Pipeline (decision 15, first use): a species with spin or emission is pushed in stage Push by `push_store` /
      `rz_push_store` (gather and push only, E^n, B^n, u^{n-1/2} stored per particle), changed in stage Qed (T-BMT, then the
      emission), and moved and deposited in stage MoveDeposit by the unchanged kernels with `Pusher::None` (no gather, no
      push). Other species keep the fused kernel; without spin / QED nothing changes (bit-identical). The guard sums, filters
      and ionization currents follow the last deposit (`push_deposit_tail`). Not with the overlapped GPU push, the implicit
      mover, the boosted frame.
    - T-BMT: E^n, B^n and u^n = the direction of u^{n-1/2} + u^{n+1/2} with the mean of their lengths (the chord average is
      short by (omega_c dt)^2 / 8: 7.7e-3 rad error after 9.6 rad at omega_c dt = 0.08, now round-off).
    - Emission: the rate C0/4 (w + <g, S> + <pf, Sf> + <pxi, xi>) of the manuscripts with their errors corrected (spec). Mixed
      states (default; Alexander's group used them in arXiv:2605.19814): S <- pf / p0, xi = pxi / p0, the non-radiative
      update normalized by 1 - C0 dt (W + <G, S>); `spin_state = pure` draws +-pf/|pf|. u = r^3, sub-steps so that the
      probability per sub-step stays below 1 (from max_r 3 r^2 (w + u K13)) and C0 (W + Gs) dt_s <= substep_prob = 0.1.
      Lepton: gamma <- (1 - u) gamma along u (energy exact, momentum not); photon along u at x^n.
    - Tables: the rate depends on (chi, u) only through y = 2u / (3 (1 - u) chi) and prefactors in u, so the Bessel functions
      are tabulated in ln y (4096 points, quadratic interpolation: 2e-7 for y <= 10) and the integrals W, Gs, max, P in
      ln chi (801 points, 1e-6 .. 1e4). PICSAR-QED (decision 9 allows it) has no spin- or polarization-resolved rates.
    - Photons: `kind = photon` (q = 0, not gathered or deposited, ballistic in stage MoveDeposit, migrated, written, energy
      sum w |u| in diag.txt); `qed.photons` optional with `photon_min_energy`; without it only the tallies (qed.txt) and the
      spectra (qed_spectrum.txt). Not done: Breit-Wheeler pairs, photon polarization in pair creation (next Phase 4 block).
    - Random numbers: a hash of (seed, step, species, particle id, sub-step, draw) (as decision 86): CPU = GPU = any rank
      count or particle order.
    - Deterministic option `model = deterministic`: d gamma/dt = -P_alpha chi^2 g(chi) from the same tables, no spin effects.
94. **Phase 3, binary collisions: the relativistic Monte Carlo of Pérez et al. (2012) with Nanbu's angle distribution, the
    pairing and the normalization of Smilei 5.1, in stage Migrate after the migration; impact ionization after Pérez / Smilei
    (RBEB cross-sections over all orbitals, Smilei's neutral-atom orbitals shifted to the NIST energies). Physics choices
    provisional, for Alexander's review** (2026-11-10, branch collisions; docs/specs/collisions.md, src/collisions.*,
    src/collision_kernels.hpp, src/impact.*, data/binding_energies.def, tests/test_collisions.cpp, bench/collisions_smilei/)
    - Where in the step: after the migration and the tile sort (stage Migrate), not before it (architecture.md section 6 put
      collisions before the migration): there every particle is in a cell of its own rank, so a cell's partners never sit on
      two ranks. The collisions change u^{n+1/2} at x^{n+1}.
    - Pairing (Smilei 5.1): per cell, a shuffle as the order of a hash of (seed, step, species, id), intra-species pairs (an odd
      particle twice, weight correction 1/2), inter-species every particle of the larger group with one of the smaller (1/k,
      1/(k+1) for the repeated ones); s normalized with D = dt every N_c max(w) / V (Smilei 5.1; Pérez's n1 n2 / n12 is the
      same for equal weights). Unequal weights: Nanbu-Yonemura (energy conserved on average: 2.8 % of the exchanged energy in
      test C4b; Higginson's energy-conserving variant not done). Random numbers keyed by (seed, step, block, absolute cell,
      pair): independent of ranks, threads, particle order and CPU / GPU (C7).
    - Angles: Nanbu's distribution with Pérez's fits by default (the task's choice; <cos chi> = e^-s to 1.2e-3), Smilei 5.1's
      newer fit as `angle = smilei5` (2e-2 from e^-s; with it VLPL = Smilei to 0.4-4 % on Smilei's benchmarks).
    - Coulomb log: automatic by default, the Lee-More form per pair: Debye length per cell from all species, ion-sphere floor,
      b_min = max(hbar / (2 p*), classical), lnL >= 2. Or fixed per block. The quantum b_min is hbar / (2 p*), as Lee-More, GMS,
      BPS and the NRL formulary, not Smilei 5.1's h / (2 p*): that one is 2 pi larger and lowers lnL by 1.84 wherever the
      quantum limit sets b_min (T_e >~ 10 eV). `quantum_bmin = h` keeps Smilei's form for comparisons. Test C9 checks the log
      against the SI formula to 4.5e-4. No degeneracy correction: none is needed at T_e / T_F = 12 (bench case).
    - Pérez's low-temperature limit s <= s' stays on by default (as Smilei and EPOCH); `low_t_cap = false` switches it off.
      Alexander delegated this decision (2026-11-10: "I don't have to decide it"); the numbers are in docs/specs/collisions.md,
      "Dense plasmas" (bench/collisions_epoch/dense_plasma.py). Solid-density hydrogen (1.1e23 cm^-3, 100 / 10 eV):
      Gamma_ii = 1.1, T_e / T_F = 12, physical lnL 2.3-2.8. Relaxation 1/e times (1/omega) with the dense-plasma logs:
      GMS 865, BPS 944 (both agree with MD within its error bars), Lee-More 756; VLPL with the automatic lnL 937 with the
      limit, 885 without (dt / 4). The limit thus changes the result by 6 %, less than the 9 % model spread. It removes 7 % of the
      e-p exchange at the physical lnL (13 % at the benchmark's fixed lnL = 5), and 46 % of the p-p exchange, where the protons
      are coupled and a Coulomb log has no meaning. It is a physical bound (mean free path >= interatomic distance), not a
      stability limit, and with it the result is dt-converged (EPOCH 487 -> 488, VLPL 488 -> 487 at dt / 4 with lnL = 5).
      Hence kept. The larger error was the quantum b_min, fixed above (Smilei's form: 1136, 20-31 % slower than BPS / GMS).
      The ~20 % gap to NRL at lnL = 5 was a gap of the fixed log, not of the method. EPOCH's Nanbu-Pérez scheme applies the
      same limit (`s12 = MIN(s12, s_prime)`) and gives VLPL's 1/e time to 0.2 % at lnL = 5.
    - Cells: the grid cells; RZ the annuli between the radial nodes, all angles together (as Smilei AM). Ids registered only
      when there are collision blocks (runs without them are bit-identical); new electrons of field ionization then get their id
      from the ion id and the level (not from their position, which differs at round-off between decompositions).
    - Impact ionization: tables per charge state as Smilei (100 energies 1 eV .. 10 MeV), the orbital energies of Smilei's
      neutral-atom table (University of Guelph data; Z <= 100) shifted to the NIST ionization energy of the ion (decision 86),
      several ionizations per step (Nuter et al.), the ion's recoil neglected. The per-cell kernel runs twice (count on a saved
      state, then write after a scan) so the new electrons are appended in a reproducible order on the GPU. An ion species of an
      ionizing block gets a charge state per particle (ion.states) without field ionization; with ionization = adk both act.
    - Frozen species: `mobile = false` species of collision blocks may carry thermal momenta and do not count in the omega_p dt
      limits (Smilei's time_frozen): pure-collision benchmarks with the production binary.
    - The tests compare with the NRL rates where they are exact: near t = 0 / Maxwellian plasmas (intra lnL = 1000 as Smilei's
      thermalisation benchmark), test particles (beam slowing and scattering 1e-3), and show the dt convergence (C4). Strongly
      anisotropic isotropization stays 5-7 % below the bi-Maxwellian NRL rate independent of dt.
95. **Beam self-field: the boost factor of each transverse mode is clamped at the cell-resolved pancake, gamma^2 h_x^2 K <= 4
    (1 / gamma^2 -> max(1 / gamma^2, h_x^2 K / 4)), Cart3D and RZ** (2026-11-10, branch beam-startup-gauss;
    src/beam_field.cpp, src/rz_beam_field.cpp, docs/specs/beam_self_field.md, tests B5, M5)
    - Symptom: bench/qed_pwfa (driver 500 GeV, witness 1 TeV, window from t = 0) had max|dGauss| 7e-5 .. 1.2e-4 from the
      first step until the window had moved one box length. A beam alone in vacuum: 1.8e-4 of max rho at u = 1e4 and
      1e6, 3.7e-6 at u = 100, 1.8e-13 at u = 10; half of it in a box twice as long; 6e-16 with the window standing still.
    - Cause: a boundary effect of the boosted Poisson operator. For gamma h_x k_perp > 2 its off-diagonal
      a = 1/(gamma^2 h_x^2) - K/4 is negative; at large gamma the operator becomes the x average mu_x^2 of the RIP-CC Gauss
      law, whose homogeneous solutions (-1)^i (A + B i) do not decay. Fitted to phi = 0 at both x ends they put a zigzag
      field ahead of the beam that falls only linearly towards the right edge (amplitude ~ 1/L); the first window shift
      cuts it there, and the cut stays in the lab frame as a static residual until it leaves through the left edge.
      The continuum pancake ahead of the beam is 1/(gamma k_perp) < h_x/2 thick there: the tail is a discrete artefact.
    - Fix: per mode 1 / gamma_eff^2 = max(1 / gamma^2, h_x^2 K / 4), so a >= 0. At a = 0 the solve is local: phi = rho / K
      (the slice Poisson problem), Ex = -h_x (rho(i) - rho(i-1)) / 4 inside the beam only, exactly zero fields ahead of and
      behind the beam (k_perp = 0 apart, unchanged). Ex is formed per mode (phi / gamma_eff^2 transformed back with phi).
      Modes with gamma h_x k_perp <= 2 are unchanged, so u <~ 10 beams are unchanged up to round-off. The Gauss law stays
      exact at the init. RZ the same per radial eigenmode (1 / gamma^2 -> max(1 / gamma^2, dz^2 lam / 4)).
    - Results: the reproducer 1.8e-4 -> <= 6e-15 for every gamma, box length, transverse resolution; RZ (u = 1e4,
      window) <= 6.5e-15; bench/qed_pwfa (dy = dz = 2 um, 480 steps) 7.2e-5 -> <= 3.5e-14. The near field of the gamma 50
      beam of B3 is 8 times steadier (change over 60 steps 9.9e-3 -> 1.2e-3 of max): the clamped modes are closer to the
      discrete steady state of RIP-CC. B2 (line-charge E_r) unchanged (0.67 %), B4 1.1e-12 -> 8e-16. qed_pwfa: initial
      W_field -0.09 % (the tail), at step 400 W_field 7e-5 and W_kin 8e-9 relative to the old run.
    - Regression `beam` (examples/beam_vacuum.deck, gamma 400, standing box) re-blessed: W_field at the init -0.65 %
      (20.208 -> 20.077), W_kin unchanged at the init, the drift of W_total over 30 steps 3.1e-8 -> 1.3e-10 relative.
    - Physics: the initial field of the high-k_perp modes of an ultrarelativistic beam changes from the zigzag tail to the
      local field (for Alexander's review before merging).
96. **Phase 4, Breit-Wheeler pairs: photons of the `kind = photon` species decay by Monte Carlo (LCFA) into an electron and a
    positron of user-named species, resolved in the photon Stokes vector and the pair spins; rates from own tables; the pair
    spin formulas derived by crossing the emission rate of decision 93 (the QV3D draft has no pair formulas). Physics choices
    (mixed pair spins, selection on, energy exact with the momentum to the field, threshold 2 m c^2 only, x from the full
    spectrum) accepted by Alexander, 2026-11-11** (2026-11-10, branch breit-wheeler; docs/specs/qed.md, src/qed.hpp `qed_pair`,
    src/qed.cpp `bw_step`, src/qed_tables.cpp, tests/test_pairs.cpp, examples/qed_cascade.deck, examples/qed_cascade_estimate.py)
    - Rate d2W/(dx dt) = Cg (B0 - xi3 K23(rho)), B0 = (x^2 + x_-^2)/(x x_-) K23 + IK13, rho = 2/(3 chi_gamma x x_-), Cg = alpha /
      (sqrt3 pi hw eps_gamma), x = eps_+ / eps_gamma; basis n, e1 = the transverse part of E + n x B (the force on the
      positron), e2 = n x e1. The pair spins b_+- (spec) were first derived from the spin- and polarization-resolved
      emission rate by crossing (initial lepton -> positron with S -> S_+ (the longitudinal component with a sign), emitted
      photon -> absorbed: xi2 -> -xi2, overall sign) and from charge conjugation for the electron (transverse sign corrected,
      below). Checks of the formulas: |b_+-| <= p0 for all x,
      chi, xi (saturated at 0.99994: pure pair states at the edge); the unpolarized rate is the standard one (Erber limits,
      3 sqrt3 / (16 sqrt2) chi e^(-8/(3 chi)) and 0.3796 chi^(2/3)); the polarization ratio -> 2 (chi -> 0) and 3/2 (chi -> inf);
      the harder lepton of a circularly polarized photon takes its helicity (angular momentum). Literature check (2026-11-11, docs/specs/qed.md
      "Comparison with the literature"): the rate, its xi3 dependence, the magnitudes of all spin terms and the longitudinal
      (helicity) terms agree with D. Seipt and B. King, PRA 102, 052805 (2020) and Y.-Y. Chen et al., PRD 105, 116013 (2022);
      the transverse part (e1, e2 terms) of b_+- had the opposite sign there (the pair preferably in the lower magnetic energy
      state: positron parallel, electron antiparallel to the rest-frame B' = -e2 |B'|). Corrected 2026-11-11 after an independent
      first-principles check (Alexander: "check the signs carefully! Seipt & Chen also might make a mistake", then "use the
      correct sign that agrees with Sokolov-Ternov"): tools/volkov_spin.py (Volkov states, explicit Dirac spinors, hole-theory
      positron = C conjugate) reproduces VLPL's emission channels to 1e-3 (electrons polarize antiparallel to B') and gives the
      positron of a pair parallel, the electron antiparallel to B' (<S_+ . B'> = +0.6700 at chi = 1, x = 0.5; formula 0.6699).
      b_perp -> -b_perp for both leptons; the crossing had mapped S_perp -> +S_+perp, but the outgoing antifermion
      carries the reversed spin (Chen et al.: zeta' = -zeta_+). Rates, yields, the photon selection and the helicity terms are
      unchanged; the pair-spin outputs change sign. Tests P4 (published sign per x bin) and P4b (positron parallel, electron
      antiparallel to B' from E and B, magnetic and crossed field).
    - Monte Carlo as the emission: x uniform, a decay if r' < Cg (B0 - xi3 K23) dt_s; sub-steps so that Cg Mb dt_s <= 1 and
      Cg (T0 + T3) dt_s <= substep_prob (T0, T3, Mb tabulated in ln chi, 0.01 .. 1e4, times e^(8/(3 chi))). Below chi_gamma
      0.01 (rate ~ e^-267) and at eps_gamma <= 2 m c^2: no pairs.
    - Mixed states (default, as decision 93): S_+- = b_+- / p0; `spin_state = pure` draws +-b/|b|. The photon Stokes vector is
      turned from its stored basis (pe = e1 at the emission) into the local e1 and changes without a decay by the selection
      xi <- [xi (1 - a T0) + a T3 (0,0,1)] / (1 - a (T0 - xi3 T3)) (the vacuum's dichroism; `pair_selection = false` turns it
      off). The vacuum birefringence (the real part, xi1 <-> xi2) is not included.
    - Kinematics: eps_+ = x eps_gamma, eps_- = eps_gamma - eps_+ (exact; both >= m c^2), both along n with |p| = sqrt(eps^2 -
      1): the field takes the momentum m c (1/(2 eps_+) + 1/(2 eps_-)) (energy and momentum cannot both be conserved for a free
      photon -> pair; the energy is). The pair rest energy 2 m c^2 w leaves the photon energy: diag.txt's W_kin does not count
      it; qed.txt has N_pairs (sum w) for the balance.
    - Particles: the photons decay in stage Qed after the emission (also those of this step), by count, scan, fill; the decayed
      photons are removed (stable order, Species::permute_device + resize); the pairs are appended to the pair species at the
      photon's position x^n (neutral: the Gauss law stays), with ids from the photon's random-number stream (photon ids from
      the emitting lepton's stream when pairs are on): any rank count, thread count or particle order gives the same pairs.
      The pair species are split species (pushed by push_store; T-BMT and emission when they are in qed.species).
    - Demo (bench/qed_cascade/): 10 GeV electrons through an a0 = 80 pulse (chi_e up to 9.3): 0.317 +- 0.004 pairs per beam
      electron vs 0.322 from an independent first-generation estimate with the photons polarized as emitted (0.361 unpolarized),
      energy radiated 0.727 vs 0.731.
    - Not done: a GPU run (the kernels are par_for / HD and the CUDA build compiles), Bethe-Heitler and trident, vacuum
      birefringence, pair-pair spin correlations (only the means of each lepton are kept).
97. **LASY laser files in RZ, 2D and the boosted frame; a recording cut for the lab particle snapshots** (2026-11-10, branch
    lasy-rz-boost; src/laser_file_rz.cpp, src/laser_file_io.hpp, src/rz_laser.cpp, src/boost_antenna.cpp, src/boost_diag.cpp;
    docs/specs/laser_file.md "RZ, 2D and the boosted frame", tests L8-L10 in tests/test_lasy.cpp, test_boost lab select)
    - Asked for by Alexander ("LASY input in all codes"), needed for an
      external FBPIC case: RZ, gamma_boost 6, the laser from a LASY file through an antenna at z = 0.
    - RZ keeps the file as azimuthal modes a_k(u, r) (env = sum a_k e^{-i k phi}, phi of the particles, LASY's x = the
      particles' x as the built-in `polarization = x`), not resampled on a plane: its file (896 radii up to 300 um) on a
      Cartesian (y, z) grid would take 13 GB. Cartesian files are projected on polar nodes around their origin.
    - `file_x != 0` in RZ: exact vacuum propagation per mode (Hankel transform of order |k|, the quadrature of GEM-PIC-RZ,
      FFT in u). The radial wavenumber spacing is dk = 2 pi / (4 (R_file + R_out)): with GEM-PIC-RZ's 2 (R_file + R_out) the
      images of the discrete inverse transform left 9e-3 of the peak at r = R_out (found in the QUARZ test; GEM-PIC-RZ has
      the same factor 2: worth checking there).
    - Boosted frame: a file is always emitted by the lab-fixed antenna (decision 89), K = -2 E_perp of the file at the lab
      plane x = 0 at the lab time on the antenna's world line, with the same time pre-compensation as the built-in lasers.
      A moving window is required; the file lasers are built on every rank.
    - 2D: slab files only (LASY y axis of length 1).
    - `lab_diagnostics.select_uz`, `select_r`: the lab particle snapshots record only crossings with the lab u along the
      boost axis >= select_uz and |transverse coordinates| < select_r (Cartesian: from the box centre). Without a cut every
      particle of the listed species is recorded and each completed snapshot is summed on every rank (allreduce): the
      background electrons of the boosted RZ LWFA case (~4e8 macro-particles) would need ~24 GB per snapshot on every rank.
      Off by default: bit-identical. Lab times with particle snapshots: at most 64 (was 32; that case has 51).
    - Results: L8 (RZ view vs the built-in pulses) Gaussian 4.9e-4, LG (0, 1) 5.6e-4, cartesian file 1.4e-3 of a0;
      L9 (RZ propagation over z_R vs the spectrally resolved paraxial beam) 2.2e-3 (paraxial error 1/(kW)^2 = 1.7e-3);
      L10 file laser vs built-in laser in the PIC loop: RZ lab 5.6e-4, RZ boosted antenna (gamma 2, iRIP) 2.5e-3 with the
      Gauss law at 2e-16, 2D lab 5.7e-4, 2D boosted antenna 3.1e-4 (Gauss law 2.7e-16).

98. **Emittance benchmark (Phase 1 exit, `bench/emittance/`): the self-field for every solver, a co-moving neutralizing
    background in periodic boxes, `species.rigid_until` (Alexander chose the cases, 2026-11-09).**
    - Cases (Alexander): A = Fig. 4 of the 2020 paper as published (10 GeV, eps_n = 1 nm, 6 / 0.6 nC, vacuum, 1 mm) with
      every solver including Yee; B = an eps_n = 0.1 um witness in a linear wake (a0 = 1, 0.01 n_c, matched channel)
      with RIP, iRIP, RZ, iRZ and FBPIC. No Yee in a wake until Yee has a moving window.
    - Self-field with periodic x: a 3D DFT of the same discrete equations (RIP-CC / iRIP); with Yee its own Gauss law
      (phi at the nodes), B = beta x E moved to x_{i+1/2} by exp(i theta_x / 2) with the signed wavenumber (the
      unsigned one flips the sign of the k < 0 half: B came out at 0.10 of beta E). This is the Gauss-law field, not a
      discrete steady state (Yee has none for a beam faster than its slow waves). k = 0 is dropped. No clamp of
      decision 95 in the periodic solve.
    - Fully periodic Cartesian box with a self_field species: the mean of J is removed every step (a uniform background
      of charge -<rho> moving with the mean current; div J unchanged, Gauss law exact). Without it the beam's net
      current drives Ex = -<Jx> t (measured field energy 1.7e-5 = (1/2) <Jx>^2 t^2 V predicted 1.8e-5) and decelerates it.
    - iRIP accepts self_field (the RIP-CC solution; iRIP conserves the same Gauss law; test B8); RZ accepts self_field
      with t_start (the solve adds to the fields present; test M6).
    - `species.rigid_until`: the push's momenta are restored before that time (ballistic motion, current deposited).
      A diagnostic of the init transient, not a cure for RIP-CC (below).
    - Finding: RIP-CC's discrete steady state differs from the Gauss-law init by O(h^2) and the mismatch travels with
      the beam (forward waves move exactly one cell per step). In case A this kicked the 1 nm beam to 12.3 nm (6 nC) in
      the first 20 um, flat afterwards; x 1/4.8 at h_x / 2; a 30 um rigid warm-up left 1.2-4.4 nm. iRIP and iRZ:
      1.00 nm over 1 mm. Cure: decision 99.
    - Results (bench/emittance/README.md): eps_n,y at 1 mm, 6 / 0.6 nC: Yee 2371 / 158 nm, FBPIC standard 574 / 54.5 nm,
      FBPIC Galilean 0.9993 / 1.000 nm, iRIP 0.9983 / 0.9981 nm, iRZ 1.001 / 1.001 nm. Case B (focusing phase, 0.25 mm):
      0.097 -> 0.105-0.111 um in all VLPL solvers, FBPIC 0.119-0.123 um.
    - Test B6 (regression, the case A beam scaled down, 15 um): growth of eps_n,y RIP-CC -3.0e-6 (steady init),
      iRIP -2.0e-5, Yee +19.8.
    - Numbered 98 because 91 (current filter), 92 (gpu-overlap), 94 (collisions), 95, 96 and 97 (LASY) were taken
      meanwhile; the steady init is 99.

99. **The self-field of a RIP-CC beam starts as the exact discrete steady state of the RIP-CC step
    (`numerics.beam_init = steady`, the default; `gauss` keeps the Gauss-law field of decisions 24 and 95). Alexander
    approved the method (2026-11-09: "agree").**
    - Why: decision 98's finding. The Gauss-law field differs from RIP-CC's steady state by O(h^2), and the difference
      travels with the beam. Fig. 4 beam: 1 nm -> 12.3 nm.
    - Method (`docs/specs/beam_self_field.md` "Steady init"):
      - Per Fourier mode of the array index, `ModeStep` applies the vacuum step of `Fields::step_core` term by term. It is
        probed with the 6 unit vectors and the source to give M and c.
      - The rigid beam moves beta cells per step: lambda = exp(-i beta tx), Jx^ from the discrete continuity equation.
        The steady field solves (M - lambda) F^ = -c Jx^.
      - Closed forms: k_perp = 0 (sheet field only), tx = 0 (Gauss-law field, exactly stationary), k = 0 dropped.
      - Open x: zero padding to a power of two >= 2 N_x, then the open box's sheet field Ex0.
    - Scope: RIP-CC with beta > 0, no CPML in y, z. iRIP, Yee, RZ and CPML runs keep the Gauss-law field. RZ is the
      follow-up.
    - Result, case A at 1 mm: 0.9983 nm (6 nC) and 0.9982 nm (0.6 nC) from 0.9982 nm; at h/2 0.9989 and 0.9988 nm from
      0.9988 nm; periodic and in the window (Gauss-law init: 11.2 / 1.50 nm periodic, 11.7 / 1.54 nm window).
    - With decision 95 (merged meanwhile): its clamp makes the open-x Gauss-law init the local slice solution for every
      mode with gamma h_x k_perp > 2, which RIP-CC keeps too: at u = 1e6 both inits are stationary to 6e-14 in a window
      (B7). The steady init has a small field tail ahead of the beam (8.5e-11 of max|E| at u = 1e4, B5), so the first
      window shifts give a Gauss drift of 2.7e-12 (Gauss-law init 2.6e-14); B4 8.4e-12 (8e-16). Case A with 95 and the
      Gauss-law init: window 0.998 nm at 0.2 mm (6 and 0.6 nC), RZ RIP-CC 1.0009 / 1.0006 nm at 1 mm (before 12.45 / 1.59).
    - The steady init stays the default (Alexander, 2026-11-11: "agree"): exact at every gamma and in periodic boxes (no
      clamp there). Below the clamped regime the Gauss-law init is off (B9, window, max|F_gauss - F_steady| of max|F|):
      1.0e-2 at u = 5, 5.9e-3 at 20, 8.8e-4 at 60, 3.2e-12 at 1e6.
    - Decision 95's clamp is limited to explicit RIP-CC, Cart3D and RZ (Alexander, 2026-11-11: "limit it to the explicit
      codes"). With the clamp, iRIP and iRZ jumped like RIP-CC before: case A 6 nC 1 -> 11.7 nm (iRIP, 0.2 mm) and
      12.5 nm (iRZ, 1 mm); the clamped field is the local state of the explicit step, not of the implicit one. Unclamped
      again: iRIP 0.998 nm at 0.2 mm, iRZ 1.0007 / 1.0005 nm at 1 mm (6 / 0.6 nC), as before the merge; B8 iRIP open-x
      near-field change 9.7e-3 -> 1.4e-3. In a window from t = 0 the unclamped iRIP field has the zigzag tail up to the
      right edge (1.6e-4 of max|E| at u = 1e4, B5), but the iRIP shifts keep the Gauss law (2.6e-14 over 12 shifts; iRZ
      M5 1.9e-14). Open (Alexander: "we have to find a solution for the implicit ones"): a tail-free init for iRIP / iRZ,
      e.g. the exact discrete steady state of the iRIP step as for RIP-CC.
    - Tests:
      - B7: one real RIP-CC step leaves the steady field unchanged (shifted by one cell) to 6e-14 of max|F|, periodic and
        in the window; the periodic Gauss-law init changes by 1.1e-3; Gauss law 1.1e-15.
      - B6: the scaled Fig. 4 beam keeps eps_n to -3e-6 (Gauss-law init: +6.35).
      - B5 (decision 95) runs both inits and iRIP: Gauss-law exactly zero ahead, steady < 1e-9 ahead and Gauss < 1e-10,
        iRIP Gauss < 1e-12; M5 also with iRZ.
      - B9: Gauss-law vs steady init in a window at u = 5, 20, 60 (differ by > 1e-4) and 1e6 (equal to 1e-10).
    - Changes results of RIP-CC runs with self_field. The regression case `beam` is re-blessed.

100. **Dielectric regions eps(r, z) in the explicit RIP-CC RZ step as a time-centred (Crank-Nicolson) polarization current
     (blocks `dielectric [<name>]`; Alexander chose option (a), 2026-11-13). For dielectric loading
     (PTFE 2.1 ... quartz 3.8): chi = eps - 1 up to 3.**
     - Options (von Neumann analysis of the 2D TM analogue of `FieldsRZ::pass`, research/rz_dielectric/vonneumann.py, k dz = 0.02):
       (a) J_P = chi mu_z(E^{n+1} - E^n) / dt, a tridiagonal line along z per (row, slot, component): stable for every chi
       tested (to 3), lossless, dv/v = 1.6e-6 at chi 0.05; (a') two Jacobi sweeps of it: stable to chi 2.9, Q 2e6 at chi 0.05
       (one or three sweeps unstable); (c) lagged J_P = chi mu_z(E^n - E^{n-1}) / dt: Q = 1100 at chi 0.05, stable to chi = 1;
       (c2) second-order lag: stable only to chi = 0.333; (b) the D formulation couples i - 1, i, i + 1 the same way as (a).
     - Method (docs/specs/rz_dielectric.md): E_z local ((1 + chi) in predictor and corrector); E_r, E_phi: y_a = dt J_P(a) at
       the current points a = i + 1/2 from the line y_a (1 + chi_a (m_a + m_{a+1}) / 4) + chi_a / 4 (m_a y_{a-1} + m_{a+1} y_{a+1})
       = chi_a / 2 (m_a dEv_a + m_{a+1} dEv_{a+1}) (m = 0 where metal holds E at 0), applied through the characteristics
       (metal z faces: B from the vacuum side). Ranks along z: partition solve, one allgather per step, cyclic for periodic z.
       J_P enters as a free current at its own point, so div(E + P) = rho holds to round-off; no extra state.
     - chi per point: harmonic mean along the component's normal (E_r across r with weight 1/r, E_z across z), arithmetic
       (r-weighted) along the interface; 8 x 8 sub-samples only in cut cells. A block's `ramp` (length) smooths its z ends.
     - Moving window (found with the DWA benchmark at eps 3.8): the mu_z average makes the z-Nyquist mode unpolarized, so near
       k_z dz = pi the scheme's group velocity in a dielectric exceeds c (1.05 c at chi 0.1, 1.95 c at chi 2.8). Grid-scale
       fields scattered at the open rear edge cross the window and grow at the moving open front edge (DWA eps 3.8: Gauss
       error 0.6 of max rho after 2 mm, 3.7 GV/m ahead of the bunch; eps 2.2 after 3.6 mm, 1.5 after 6 mm; 1.1 clean over 6 mm;
       a static box clean). Remedy, only with dielectric blocks and a window: (1) an absorbing layer of 32 cells at the front
       edge (the z-layer kernel, kappa = 1.5 ln 1e3 / 32; nothing physical is ahead of a beam at v ~ c: keep beams more than
       32 cells behind the front edge), (2) E_z of the entering column and of the layer from div D (as iRIP's window entry),
       so the damping leaves no charge. Then: Gauss 1.8e-12 over 6 mm, the field ahead 1e-4 ... 2e-3 of the wake. Without
       dielectric nothing changes (bit-identical).
     - Alexander (2026-11-13): "if there is no other remedy, stay with this. however, through a warning if the user violates
       it!" Charged particles inside the layer are warned on rank 0: at the start (load regions; a plasma entering through the
       front edge gets one note) and during the run when they enter, at most every 2000 steps while they stay (test D6).
     - Tests: tests/test_rz_dielectric.cpp D1-D6, tests/mpi.sh (window + rings + gaps + collar; periodic cyclic), the
       benchmarks in bench/rz_dielectric/ (README there: DWA against the lined-tube theory, the Cherenkov cone, thin-wall
       convergence).
     - Generic metal (Alexander, 2026-11-24: "dielectrics and metals have to be implemented in general", not only with the
       rings): the dielectric blocks with the block metal (decision 103, also depending on phi). Options: (a) the lines on the
       2M - 1 collocation angles, (b) the per-mode lines iterated with the angle-dependent zeros (Jacobi on the mode coupling:
       two sweeps stable only to chi 2.9, one or three unstable). Alexander chose (a) ("Solve on angles", 2026-11-24). The
       modes and the angle values are a bijection (Parseval), so each angle is the line above with that angle's masks (Mh for
       E_r, Mp for E_phi at its staggered angles), the corrections per angle with the face rules rBp / rBr, then back to the
       modes; the window entry applies the E_z rule per angle column by column. Without the metal block nothing changes. Tests:
       P6 axisymmetric metal function + dielectric = rings + dielectric (3.9e-15), P7 square guide filled with eps: sqrt(eps)
       k_c = vacuum k_c (3e-9), P8 corrugated square pipe + dielectric, 3000 steps (energy bounded, div D 7e-15), D4 with no
       metal and with phi ridges, tests/mpi.sh square pipe + lining in a window.
     - eps from a deck expression of x, y, z, r, phi (2026-11-24; Alexander: Q2 (a), the per-component rule of the rectangles
       from sub-samples: harmonic along the component's direction, arithmetic across; the full anisotropic tensor not
       done). Axisymmetric: chi per (r, z) as before (D7 function = rectangles 4.5e-15, D8 graded tube vs the radial ODE
       7.9e-5 / 2.0e-5, second order, D9 window). Depending on phi: chi on the 2M - 1 angles, every operation with chi
       through the angles (D10 = the axisymmetric path 6e-15, D11 half-filled pipe vs polar finite differences: 16 modes
       -5.9e-7).
     - Cart3D and 2D (2026-11-24; Alexander: Q3, the same scheme, lines along x, the front layer and its warning;
       docs/specs/dielectric3d.md, src/dielectric3d.*): eps a number or an expression of x, y, z; E_x local, E_y / E_z lines
       along x per (j, k), the partition solve over the row of ranks along x (Comm::allgather_x, an MPI_Cart_sub row: ranks
       along y, z independent); the metal block's x faces; the CPML step too; chi per component from 4^3 sub-samples
       (harmonic along its direction), periodic axes wrapped (the guards must hold their partners' chi: without the wrap a
       non-uniform eps broke div D by 0.1). Tests E1-E4 (div D 5e-15; normal incidence: reflection -1/3 to 4e-16, the
       transmission to the grid dispersion; oblique p / s against Fresnel 8e-3 / 6e-4; window with the warning), tests/mpi.sh.

101. **The longitudinal fields of the TFSF incident grid from the discrete constraints.** (docs/specs/tfsf.md, 2026-11-01; renumbered from 64 at the merge with main, 2026-11-13)
    - Alexander: a static field left where the pulse started usually means an incorrectly initialized longitudinal
      field. So it was: the beam formula set only the Ex, Bx of the tilt, not those of the focusing (~ a / (k w0)); the
      div E, div B of that start (1.4e-2 dx / peak at w0 = 1.5 um, 0 deg) stayed as a static Ex, Bx of 7e-2 of the peak.
    - Ex, Bx are now summed from the plane towards -x from the scheme's own Gauss law and div B (no charge), so the
      incident grid starts divergence-free on the grid (1e-17) and no static field remains. The transverse fields stay
      those of the formula. Outside the CPML layers only (as the iRIP entry cell, decision 41): in a layer, or at the zero
      guard of an open edge, the sum would turn the cut of the pulse into a large Ex, Bx.
    - The incident field at the plane changes by the corrected longitudinal field; the TFSF stays exact (leakage 1e-15).

102. **Structure runs in Cart3D: an embedded perfect conductor from a deck function (block `metal`), external quadrupoles in
    the particle push, beams from openPMD particle files, and wake diagnostics** (2026-11-13, numbered 102 at the merge with main;
    `docs/specs/structure3d.md`). Asked for by Alexander for 3D studies of metallic structures (wakefields of
    corrugated pipes and other structures).
    - Metal: a node mask (function > 0 at the Ex nodes), evaluated on the device including the guards (no exchange) and for
      the entering columns after each window shift. The rules are post-passes after the predictor, the transport and the
      corrector (one code path for the plain and the CPML step, nothing changes without the block): tangential E of metal
      edges 0; their B partners 0 inside along x, the characteristic from the vacuum side at an x face (the RZ rings rule,
      decision 83); Ex kept only at x faces in the transverse interior (the surface charge); Bx of metal faces 0. The
      Gauss and div B checks skip the points whose stencil touches a metal node. Staircase only: the curved-wall error is
      first order (TM01 / TE11 of a round pipe: -0.99 / -2.1 % at R / h = 32). Cut cells: later, if the 3D study needs them.
    - One cell of metal along x cannot reflect from both sides; such edges are set to 0 (they absorb) and the setup warns.
    - Particles inside the metal: weight and momentum 0 (not removed: no migration message needed, any rank count; the
      kinetic energy goes to lost_energy). Their charge is then 0 only at skipped Gauss points.
    - Quadrupoles: on the particles only (static curl-free field), in move_one / push_store, averaged over the x crossed in
      the kick interval (GEM-PIC-RZ PR #38). Not with the implicit mover, the boosted frame, RZ.
    - Beam files: a port of GEM-PIC-RZ's reader on src/h5; every rank reads the file; the charge sign comes from the file;
      the self-field uses the mean u_x.
    - Wake: F/q at a probe along the window and its time integral at fixed xi (needs a window at v = c, so a column is a
      fixed xi); beam slices; the Poynting flux through four side planes.
    - Found on the way (not changed, also on main without metal): a hot plasma (u_th = 0.2) that fills a moving window from
      t = 0 drifts the Gauss check by 0.12 at the first fold of the entering charge (decision 91); cold plasma 2e-16.
103. **RZ metal that depends on the azimuth (block `metal`): the modes on 2M - 1 angles, the phi derivative as the staggered
    difference (i m -> i m_eff) so that the angles form a Yee grid, the rings rules per point and angle; the axis cut
    `numerics.rz_axis_cut` against the m^2 / r^2 limit of many modes** (2026-11-13, branch `rz-metal-phi`,
    docs/specs/rz_metal_phi.md).
    - Why: structures with a discrete rotational symmetry (e.g. a square or other C4-symmetric guide, C4:
      m couples with m +- 4) with a few modes instead of full 3D; the rings block is axisymmetric only.
    - The metal on the 2M - 1 angles where the modes are represented exactly (invertible transform, Parseval): the rules act
      on the angle values after every pass, the changes go back to the modes.
    - Rejected: the spectral phi derivative with the metal zeroed on the angles. The zeros inside the metal enter the vacuum
      through the global derivative and B_z loses its Neumann condition at phi surfaces: the square guide's TE10 came out at
      0.81 .. 1.68 for M = 4 .. 16 (analytic 2.62) with spurious low modes (research/rz_metal_phi/xsection.py). The staggered
      difference over dphi = 2 pi / (2M - 1) is diagonal in the modes too (m_eff = (2 / dphi) sin(m dphi / 2)) and local on
      the angles: TE10 2.25 .. 2.47 for M = 4 .. 32 (staircase error). Kept as `phi_derivative = spectral` for axisymmetric
      masks (= rings to round-off).
    - Rejected: more angles than 2M - 1 with a least-squares transform back: then the rules are no projection, they damp.
    - Every RZ operator reads m_eff (field step, divergences, J_phi of the deposit, radial bases, self-field, filters,
      media); without a metal block m_eff = m (results unchanged).
    - The odd number of angles is never C4 symmetric: degenerate pairs split by the staircase (falls with M). A mode-family
      selection (only m = m0 + 4k stored) would give C4-symmetric grids and 4x fewer modes: not done (a change of the slot
      layout everywhere).
    - Axis cut: mode m >= 2 zero inside r < m_eff dr / kappa (a conducting rod per mode, the polar filter of global
      models). Physics approximation (fields of mode m ~ r^m there), off by default; with it dz ~ dr for any number of modes
      instead of dz <~ 2 dr / m.

104. **Boosted-frame stability: the streaming plasma needs the binomial current filter; the filter is allowed with a boosted
    window in 2D/3D; the summary warns about unstable numerics** (2026-11-14, branch boost-stability; asked for by Alexander:
    "ensure the boosted frame runs are stable both linearly and non-linearly"; bench/nci/boost_stability/README.md). Provisional.
    - Why: decision 87 tested a cold plasma (u_th = 1e-4) on coarse transverse cells (h_perp >= 3 c dt). On the grids of a real
      boosted LWFA (dx' = gamma (1 + beta) dx_lab > h_perp) with a seed (u_th = 1e-2) the plasma is unstable: periodic box,
      gamma 5 - 20, 2D, 3D, RZ, RIP-CC and iRIP, the field energy grows from ~1e-6 to 1e-3 - 4e-2 of the drift energy within
      300 / omega_p without filter (shape 2 x3 - x10 lower), heating the plasma; in a boosted LWFA with the window (a0 = 4,
      gamma 10, 2D) the noise ahead of the laser reaches 1.2e-2 rms (20 % density modulation) within ~9 / omega_p of the
      plasma's proper time and modulates the laser; RZ adds 17 - 48 % excess electron density on the axis. Shape 2 with two
      binomial passes is flat everywhere (<= 6e-6); one pass is flat for ~100 / omega_p (slow growth after, where h_perp < c dt);
      shape 1 grows even with two passes at gamma >= 5; the RZ gathers plain and matched agree to 2 % in every case.
    - Cart3D/2D: filter_passes was refused with every moving window (decision 91 handles the lab window's frozen columns in RZ
      only), so a boosted Cartesian run had no remedy. The boosted window freezes nothing (the plasma enters 3 cells behind the
      edge); allowed there. Plasma leaving through the open left edge leaves the filter's closure offset in the first
      filter_passes cells (2.4e-3 at the end of a 2-pass run); the Gauss check skips them, as RZ (rz_gauss_istart). The lab
      window with the filter stays refused (allowed since decision 107).
    - Warning (boost_nci_risk): filter = projection / local; shape 1 at gamma > 2.5; no filter at gamma > 3; one pass when the
      plasma's residence k_p L_lab > 100. Defaults unchanged (filter_passes = 0, Cartesian shape 1): a boosted default of shape 2
      and one pass is proposed to Alexander.
    - Results against the lab run (a0 = 2, 4, n0 = 0.004 n_c, 100 um up-ramp, 0.6 mm, gamma 5 and 10, 2D and RZ): see the
      README; the laser agrees to <= 1.5 %, the a0 = 2 wake to 3 - 6 %, the a0 = 4 wake to 12 - 26 % (the bubble back, where
      electrons are self-injected); the unfiltered gamma 10 run injects a much stronger bunch (80 - 105 %).
    - MPI with the filter (one pass, 1 vs 4 ranks): lwfa_boost (3D) energies 4e-13, lab fields 7e-15; rz_lwfa_boost 9e-14, 7e-15;
      ionization_boost (2D) identical.
    - Open: with two passes and warm plasma a Gauss offset of 1e-10 .. 3e-9 (1e-8 of the electron charge density, two interior
      cells) appears at boosted step ~1450 in three runs (2D gamma 5, 10, RZ gamma 10); one pass and cold two-pass runs <= 2e-14;
      cause not found (found in decision 108: the injection plane inside the filter's right-edge closure). 3D boosted LWFA, gamma 4,
      gamma > 20 untested.

105. **Structure runs in RZ: the external quadrupoles, the openPMD beam import, the wake probe and the beam slices of
    decision 102 ported to geometry = rz** (2026-11-14, branch `rz-structure`; docs/specs/structure3d.md section 5). Asked
    for by Alexander ("Yes, port them"): VLPL-RZ is the code for a round corrugated pipe with resolved teeth (and C4
    structures with modes, decision 103) with a focusing lattice and imported beams.
    - One table and one edge average for both geometries (external.hpp): the Cart3D (x, y, z) are the RZ particle axes
      (z, x, y), so `QuadView::add(X, Y, Z, ..)` gives B_x = -G (y - y_c), B_y = -G (x - x_c) in RZ: G > 0 focuses electrons
      in x, the GEM-PIC-RZ / elegant convention. Added after the gather in rz_push_deposit and rz_push_store; the implicit
      mover and the boosted frame are rejected as in Cart3D.
    - Deck keys in the RZ axes (z_start, z_offset, x_center, y_center; species z_center, x_offset, y_offset; wake x, y)
      rather than the Cart3D names, so that a deck reads in its own axes; a Cart3D key such as x_start in an RZ deck stops with "unknown key".
    - Beam import: the same reader and placement, the file's transverse origin on the axis; the particles equal the
      Cart3D import of the same file. Particles beyond the outer radial node are dropped with the warning.
    - Wake probe: (x, y) anywhere (not the nearest node: the RZ fields are interpolated linearly in r with all modes summed
      at the probe's phi); along z the node values, as every RZ component sits at the z nodes.
    - Not ported: side_flux (Cart3D faces) and the Cart3D metal block (RZ has decision 103).
    - Benchmark (examples/rz_corrugated_pipe.deck, research/rz_structure/): round corrugated pipe a = 1 mm, h = 50 um,
      p = 40 um, g = 20 um with the teeth resolved (axisymmetric metal function, spectral phi derivative; dz = 2 um,
      dr = 5 um, modes 0, 1), a rigid 100 pC bunch (sigma_z 100 um, gamma 2000) imported from a python-written openPMD
      file (the import line = the python moments: Q, E, dE/E, sigma, eps_n to all printed digits), offset 50 um, 30 mm of
      pipe (14990 steps, 33 min on 24 Ivy cores). At 29.5 mm into the pipe: loss factor 8.09 kV/pC/m (Bane-Stupakov
      8.08, +0.2 %; the slot-line modal k0: 8.34, -3 %), steady from 9.5 mm on (8.19, 8.18, 8.09); wake behind the bunch:
      k0 8.37 / mm (BS 8.94, modal 8.76), amplitude 0.96 of BS. Dipole: kick factor 2.31 kV/pC/m/mm (2.31, 2.36, 2.31
      along z), linear in the offset (25 vs 50 um: -231.0 vs -230.8 V/m per um), 0.2 % with a centred bunch, decaying to
      2.7 % in a smooth pipe (transient); behind the bunch k1 8.18 / mm, amplitude 0.98.
    - The dipole reference (my derivation, for Alexander to check): a corrugated wall shorts E_phi (teeth; evanescent in
      the grooves), so with E_z = A x at v = c inside (H_z = -A y, the uniform E_perp part fixed by E_phi(a) = 0)
      E_z / (Z0 H_phi) = 2 i / (k a) / (1 - 2 / (k a)^2) at the wall: k1^2 = 2 p / (a h g) + 2 / a^2 (9.06 / mm), almost
      the monopole k0. Kick factor with it 2.42 (sim -4.5 %), modal 2.47. The isotropic-impedance form k1 = sqrt(2) k0
      (resistive wall, thin dielectric) gives 1.58, 46 % below the simulation. The remaining 5 - 8 % in the wavenumbers
      (sim below theory) is of the order of the leading-order corrections in h / a, p / a, g / p; the wake far behind the
      bunch is still building up at 30 mm.
106. **RZ: the density-conserving radial map for m = 0 only; the modes m >= 1 deposit and gather in the plain radial
     coordinate (`numerics.rz_axis_map = m0`, the default; `all`: the map in every mode, as before).** (audit 2026-11-13,
     branch rz-axis-map; src/rz_particles.cpp deposit_seg, rho_impl, interp; tests/test_rz_axis.cpp)
    - Decision 53's map makes every node receive its finite-volume share of a uniform density: a statement about m = 0. In
      the modes m >= 1 it was applied too, where it is an error of first order in the axis cells: a uniform transverse
      current (a rigid shift of a uniform plasma, the laser quiver; J_r1 = J_x / 2) came out as 0.833 (shape 1: the flux
      weight dxi/du = 3/2 - u of the axis cell, 5/6) and 0.626 (shape 2) at the first radial face, and the dipole of a
      charge at x0 = 0.1 dr (node 1 of mode 1) as 1.45 x (shape 1) and 4.7 x (shape 2: (x0/dr)^0.33, not linear in the
      offset): the seed of hosing and of transverse wakes.
    - With the plain coordinate for m >= 1: the uniform current 1.0000 (shape 1; J_r1 3e-8, J_phi1 2e-8), shape 2 1.083 at
      the first face and 4e-2 in J_phi1 (the fold of the fluxes across the axis, decision 50; open), the dipole exact
      (2e-16, shapes 1 and 2). The continuity holds per mode with the mode's own weights (the Gauss law at round-off as
      before); the charge check and the output rho use the same weights per mode.
    - rz_gather = matched gathers each mode in the map of its deposit: m = 0 in the density-conserving map (decision 80,
      unchanged), m >= 1 plain. rz_gather = energy likewise.
    - Not changed: the implicit mover (C1 map in every mode, decision 82); the m = 0 gather. With matched a linear radial
      field E_r = g r (m = 0: ion column, bubble) is gathered as g dr xi(r) in the axis cell: x 1.25 at r = dr/2, x 1.45 at
      0.1 dr (shape 1), x 1.59 and x 4.7 (shape 2); plain is exact (test A3). Alexander's decision (decision 80 chose
      matched for the axis instability of dense targets).
    - The removal of particles inside embedded metal (rz_metal_remove) tests the lower edge of the shape in the plain
      coordinate (the modes m >= 1 reach lower than the mapped m = 0 shape).
    - **numerics.rz_gather = plain is the default again** (Alexander 2026-11-14: "go with option 1 for now. we need an option
      that works good for boosted frames (streaming plasma) and a separate option that is stable for the overdense plasma.
      this can be a user choice. a warning should be thrown is the code detects streaming plasma or overdense target with an
      unmatched option"). matched stays the choice for dense targets (decision 80). Warnings at the start of an RZ run: a
      dense plasma at the axis (k_p dr > 1.5) with plain; matched or energy without a dense plasma; a streaming plasma
      (boosted frame, or a species drifting with |u| >= 1 that is not a self_field beam) with dr < c dt and filter_passes = 0.
    - Streaming plasma (bench/nci/rz_nci.deck, gamma 10, 282 / omega_p, W_field / K_e), plain = matched to 3 digits in every
      case, so the gather does not matter there; the filter does:
      RIP-CC shape 1: 2.6e-9 (no filter), 2.2e-10 (one binomial pass); shape 2: 4.3e-10, 1.1e-10.
      iRIP shape 1: 4.7e-9, 2.2e-10; shape 2: 6.0e-10, 1.1e-10.
      iRIP, dr = c dt / 4: shape 1 1.3e-3 (growth 0.050 omega_p), 5.2e-10 with the filter; shape 2 3.4e-5 (growth 0.058
      omega_p), 1.9e-10 with the filter. Linear regime, seeded by the particle noise only; seeded and large-amplitude tests:
      the boosted-frame session.
    - Test A4 (tests/test_rz_axis.cpp): the betatron frequency of a particle in a linear focusing field, plain against the
      exact value; matched printed.

107. **The current filter with the lab moving window in Cart3D and 2D; the boosted-vs-lab energy offset of the Lee-Vay case
    is the filter** (2026-11-16, branch lab-window-filter; Alexander: "do it", to pin down the +3 .. 6 % energy offset of
    bench/nci/boost_stability/README.md section 2c before any gamma_b rule; section 2d there). Provisional.
    - Code: `numerics.filter_passes` is accepted with a window at v = c (lab frame); a window at v < c stays refused. The
      frozen-column handling of decision 91 (window_freeze_cells = 3 + passes, window_fold_entered, window_frozen_rho) was
      already shared by RZ and Cart3D (src/window.cpp); only the refusal in Config::check and the left-edge rule of the Gauss
      check were RZ / boosted only. gauss_drift now skips the first filter_passes cells at the open left edge for every window
      with the filter (as rz_gauss_istart), where the leaving plasma leaves the closure offset of the filter.
    - Test W9 (tests/test_window.cpp): W8's warm non-neutral column with a ramp, binomial filter 1 and 2 passes, 3D and 2D,
      RIP-CC and iRIP, 128 shifts (the plasma front leaves at the left edge): max|dGauss| 5.4 - 5.8e-16 (3D), 5.6e-17 (2D).
      tests/mpi.sh: the moving-window non-neutral case with 2 passes, 1 vs 8 ranks: energies identical, Gauss 1.1e-15.
      Lee-Vay lab runs with one pass (2D, 1600 - 3200 x 512 cells, 12600 - 25200 steps): max|dGauss| 3.1 - 5.9e-13, as
      without the filter (7.7e-13 - 1.1e-12, the injected bunch).
    - Result (lab time 1000 fs, plasma depth 247 um, bunch gamma >= 50): with the filter in both frames boosted (gamma 2, 3)
      and lab agree at lambda / 64 to +0.2 .. +2.7 % in energy (median +0.9 %) and -1.3 .. -2.1 % in charge; without the
      filter in both (gamma 2, lambda / 32) to -0.8 % and -0.2 %. The +3 .. 6 % of section 2c compared a filtered boosted
      run with an unfiltered lab run. One pass itself moves the lab bunch by +1.5 .. +3 % in energy and +1 .. +3 % in charge.
    - Not changed: defaults (filter_passes = 0); the RZ path. Open: the remaining -2 % in charge at lambda / 64 and the
      +2.7 % of gamma 3 RIP-CC; lambda / 128.

108. **Mechanism of the NCI in VLPL: the alias m = +-1 of the streaming plasma crosses the EM branch of RIP / iRIP; options
    `numerics.filter_compensate` and `filter_axes = x`; the boosted injection must be exactly neutral** (2026-11-17, branch
    nci-mechanism; Alexander: "go on with NCI research in VLPL"; bench/nci/mechanism/README.md). Provisional; no default changed.
    - Spectrum (2D periodic streaming plasma, fields every step, the growth rate and frequency of every (k_x, k_y) mode; RZ on the
      boosted RZ LWFA grid): the strongest unstable modes lie on the crossing of the EM branch with the beam alias m = +-1, omega tau =
      beta (k_x D + 2 pi m); the measured frequency equals the alias to 0.0003 - 0.002 pi and is 0.01 - 0.04 pi off the m = 0 line
      (iRIP and RIP-CC, shapes 1, 2, gamma 5, 10, h above and below c dt; RZ RIP-CC gamma 6: k_z dz = 0.875 pi, 0.8914 vs 0.8907 pi,
      0.052 omega_p). With c dt = D the alias lies (1 - |beta|)(2 pi - |k_x| D) above the light line, the EM branch (D^2 Lambda / 4)
      cot(k_x D / 2) above it (iRIP x cos^2): they meet from Nyquist (large k_perp) down to a k_x set by the plasma term, which
      falls with gamma^4 n0 (k0 dx_lab)^2. No growth off these curves (seed isotropic in the plasma rest frame): no E / v x B
      mismatch instability, so no matched gather is needed. The anisotropic seed of decision 104 (u_th = 1e-2 in every boosted
      component) also drives low-k beam-line modes (physical, Weibel-type).
    - Rates (amplitude, unfiltered): iRIP gamma 10 h 0.25 um 0.058 omega_p (shape 1 0.068), h 1 um 0.020, RIP-CC h 1 um 0.011,
      gamma 5 0.048; proportional to omega_p (n0 0.002 - 0.008 n_c within 3 %).
    - Options: filter_compensate (J <- J + c (L J - J), c = -N, along x after N passes; a polynomial in S_a, so d P(S) = P(T) d and
      the Gauss law stays exact) and filter_axes = x (passes along the streaming axis only). Tests S9b (Cartesian: 4 - 7e-16,
      transfer function 3 - 4e-15) and R25 (RZ, kinds 3, 4).
    - Cures (W_field / K_e at 600 / omega_p, noise 1e-8 - 6e-8): two plain passes 3.2e-7 (gamma 10), 2.1e-4 (gamma 5), RZ LWFA
      3.0e-5 at 533 / omega_p; 8 passes + compensator along x 5.4e-8, 2.9e-6, RZ 3.7e-7; the compensator alone is worse at equal
      passes (it lifts k_x D = 0.4 - 0.8 pi, where the resonance lies at small k_perp). Plasma response at k0 (lambda / 32),
      measured with the EM mode of a plasma at rest: two passes -1.9 %, four -3.8 % (= cos^(2N)(k0 D / 2) to 0.3 %), 8 +
      compensator -0.3 %. gamma ~ 5 with h < c dt keeps a residual at k_x D ~ pi/4 (0.0045 omega_p) that no filter flat at k0 removes.
    - Proposal to Alexander (not implemented as default): boosted runs with shape 2, filter_passes = 8, filter_compensate, filter_axes
      = x; gamma 3 to be added to the warning of decision 104 (1.2e-3 at 600 / omega_p without filter).
    - **Neutral boosted injection** (Alexander: "the boosted injection has to be absolutely neutral. This is a precondition."):
      BoostInject has no creation current, so any net charge of the entering plasma stays at the injection plane (a He / N deck,
      electron density rounded by 2e-6: Gauss residual 1.2e-3 of the electron charge density, field energy 2e-3 without a laser).
      Config::check now refuses a boosted plasma (species at rest in the lab with a density) unless (a) all its species share ppc,
      bounds and slab (the loaders place particles on deterministic lattice points, so the charges then cancel point by point) and
      (b) sum q Z0 n f(x) = 0 at every point of the lab path to 1e-12 (Z0: the initial charge state of an ionizable species);
      the error names the mismatch and, for one electron species, the neutral density. A deliberately non-neutral plasma or
      electrons alone are refused (their missing ions would also miss their current); a creation current is not implemented.
    - The injection plane is now 3 + R cells behind the right edge, R = filter_passes (+1 with the compensator), as the frozen
      columns of the lab window (decision 91): with two passes the plane 3 cells behind lay inside the filter's right-edge closure,
      which changes at every shift, and warm electrons left up to 3e-9 of the charge density in the Gauss law (the open item of
      decision 104). Without a filter unchanged (bit-identical).
    - Tests B7 (refusals and their messages; boost_net_charge of the rounded density = (n_p - n_e) / (n_p + n_e) to 1e-6) and B8
      (2D, 3D, RZ; e + He+ + N3+, both ionizable; 20 um ramp; warm electrons; gamma 10; filter 0, 1, 2: max|dGauss| 3e-16 .. 5e-15).
    - 2026-11-18, Alexander: electron-only boosted runs stay refused ("keep refusing electron-only injection!"); "virtual"
      co-moving ion charges or an analytic streaming background were assessed (about 25 % cheaper than real ions, or the electron
      shot noise returns) and rejected: "we win not much with the virtual background". Use ions with the same ppc as the electrons.

109. **NCI cures other than many binomial passes: the maximally flat filter `numerics.filter_flat` (test switch), the force of the
    unstable modes, the implicit mover** (2026-11-18, branch nci-mechanism; Alexander on the filter figures: "too much filtering
    will destroy physics at some point. any other solution?", then "go!" for short-box tests and test switches;
    bench/nci/mechanism/README.md section 6). Provisional; no default changed.
    - `numerics.filter_flat = L` (2 .. 9; filter = binomial, filter_passes = N >= 1, not with filter_compensate): after the N passes the
      maximally flat (Herrmann) factor P(S) = sum_{j<L} binom(N-1+j, j) S^j along x (z in RZ), S = 1 - T: response T^N P(1 - T), flat to
      O(S^L) at k = 0, zero of order N at Nyquist. P is applied as its real factors (Domain::filt_fk/fa/fb, set_flat_filter: roots by
      Durand-Kerner): a real root is one step with cf = 1/r (the compensator is L = 2, cf = -N); a complex pair (1 - a S + b S^2) is
      two plain steps and the combination (1 - a + b) x0 + (a - 2 b) x1 + b x2 (Field::assign, Field::lincomb). Linear in the steps,
      so D J' = T' D J stays exact; filter reach N + L - 1 (window freeze, boosted injection plane, Gauss range). Tests: S9b (the
      factors reproduce T^N P for N 1 - 16, L 2 - 9; Gauss identity and the plane-wave transfer for L = 4, 6), R25 kinds 5, 6 (RZ);
      tests/mpi.sh mpi_fstream (3D, 1 vs 8 ranks 1.2e-11) and mpi_rzff (RZ, 0).
    - Removed at the laser (lambda / 32): N=6 L=4 1.0e-6, N=8 L=4 2.7e-6, N=12 L=6 8.9e-9 (8 passes + compensator 3.2e-3, two passes
      1.9e-2). W_field / K_e at 533 / omega_p: gamma 10 (2D iRIP) 1.8e-7, 1.0e-7, 1.0e-7 (proposal 5.4e-8, noise level); gamma 5 3.2e-4,
      4.2e-5, 4.2e-5 (proposal 1.3e-6, two passes 1.6e-4); RZ LWFA grid 2.2e-4, 6.0e-5, 4.8e-5 (proposal 3.7e-7, two passes 3.0e-5).
      At gamma 5 - 6 the unstable band reaches down to ~0.3 pi (the 4th - 5th laser harmonic at lambda / 32): a filter that leaves the
      harmonics alone passes it. Flat filters suit gamma_b >= 10 only.
    - Force of the unstable modes (gamma 10, 2D iRIP, no filter; bench/nci/mechanism/force.py on the fields of every step): 98 % of
      |F|^2 on the plasma is the longitudinal E_x (no v x B part); the transverse rest |E_perp + v x B| / |E_perp| = 0.09 (a continuum
      wave of the same omega, k: 0.076 weighted, 0.0135 for the strongest mode). VLPL's gather is already matched (decision 4), so a
      matched gather could change at most the 2 % transverse share: ruled out.
    - The energy-conserving implicit mover (decision 82, no filter possible with it): total energy kept to 1e-8, but W_field / K_e
      16 (gamma 10) and 1.9 (gamma 5) at 533 / omega_p: worse than the explicit mover (3.5e-3, 1.2e-2); the field draws on the drift
      energy. Ruled out.
    - Not tested, reasoned: filtering only the streaming species or only the gathered fields (the laser quiver and wake currents belong
      to that plasma: the same change at k0); a comoving grid (light moves 1 +- beta cells per step: RIP's exact shift is lost, a new
      solver); higher shape (k and k - 2 pi / dx have the same shape factor near Nyquist).
    - What keeps the physics: move the band, not filter it: gamma_b >= 10 (band at 0.7 - 1 pi) or a finer grid (lambda / 64), at the
      cost of run time. Alexander decides.

111. **Particles that enter the transverse absorbing layers are removed, with no current in the layers
    (`grid.absorber_particles = remove`; the default until decision 115)** (2026-11-20, branch pml-particle-removal; Alexander: "Simple solution: remove
    all particles reaching the PML layer", then "keep the current deposition out of the PML and check whether it creates any
    additional source of errors / instabilities in a short run"). Number 110 is taken on the branch trapped-split (off main).
    - Cause (nci/trap/d3, 3D trapped-bunch model of the boosted RZ LWFA case: plasma streaming at u = -5.9 through a 400 x 80 x 80 window,
      iRIP, binomial filter 2 passes, shape 2): with the plasma filling the box into the 16-cell y/z CPML a near-uniform transverse
      field grows along the plasma's path (E_phi m1 at r = 0.41 um: 3.7e6 V/m at 50 fs, 3.1e8 at 302 fs, 3.1e10 at 906 fs, 4.4e10 at
      1509 fs); with periodic y, z, or with the plasma in a column r < 8 um clear of the layers, 7.0e6 at 1509 fs. The settled stability
      of the streaming plasma (decision 104) holds; the layers damp fields, not the currents of the particles in them. RZ: the radial
      absorber with plasma in it drives nothing (1.50e7 into the layer, 1.44e7 column, 453 fs).
    - remove: currents in the layers set to zero after the deposit and the filter, particles removed once in a layer (the charge
      continuity is broken at the face); conserve: current into the layer, removed once the shape left the physical nodes + 1 cell
      (charge conserving); keep: the old behaviour. The removed number and charge per species are printed at the end of the run.
    - Tests: P5 (tests/test_pml.cpp, all three modes; remove: Gauss 4.6e-16 inside, 2.6e-2 next to the faces); short runs (nci/pmlrm,
      455 fs): 3D plasma-only filling into the layers: remove 6.8e6, conserve 6.8e6, keep 5.7e9 V/m at 453 fs; the face error of remove:
      Gauss residual 1.4e-8 constant, round-off after ~400 fs, mean rho next to the faces 2.2 C/m^3 (plasma 5e5), no growth; RZ: the
      same within 7 % for the three modes.
    - Tests, 2026-11-22 (branch test-repair-111-112; Alexander: option 1): in tests/mpi.sh the moving-window + PML cases (iwinpml,
      winpml and its restarts) run with absorber_particles = conserve, so the MPI check stays a Gauss law at round-off; remove
      leaves a residual ~4e-3 at the layer faces (the same on 1 and 8 ranks). A separate case (mpi_winpmlr) checks the rank
      invariance of remove: energies 1 = NP ranks to 1e-8 and the Gauss residual itself to 1e-8 relative, with a loose Gauss
      limit 1e-2. Inside the physical region remove changes the fields measurably against conserve at the end of these cases
      (energy of the difference field 0.6 % / 12.7 % of the field energy inside, mostly in the 2 cells next to the faces, E and
      B parts about equal; 0.04 % / 0.27 % of the peak field energy): reported to Alexander, no change.

112. **RZ, mode 1: only the regular transverse current in the first radial cell, its adjoint in the gather
    (`numerics.rz_axis_current = regular`, the default)** (2026-11-20, branch rz-axis-regular; Alexander: "this is the most plausible
    error. fix it"; nci/rzm1, docs/specs/rz_modes.md "The first-cell current of mode 1").
    - Cause of the near-axis m = 1 line of the trapped-bunch noise (0.78 pi with RIP-CC, 0.48 pi with iRIP): the TE bound state of
      mode 1 next to the axis (lambda dr^2 = 6.365) driven by the r^2 part J_r + i J_phi of the first-cell current, which the deposit
      fills (exactly, for point particles) and a 3D grid cannot (3D: |J_-|/|J_+| 0.10 at dr, RZ 12.6); 99.8 % of the drive from the
      first cell. Checked and not the cause: the gather of uniform transverse E and B (exact to 2e-16 down to 0.1 dr), the m = 1 charge
      conservation (Gauss residual per mode 3e-17), the pusher (Cartesian), the filter on regular data (unchanged), the 3D comparison
      (no line; its m = 1 excess was the PML artifact of decision 111).
    - Rule: a divergence-free shift of J_r,0 and J_phi,1 that removes the first-cell r^2 part (charge conservation exact), and the
      adjoint shift of E_r,0, E_phi,1 for the push gather (energy consistency; the dense-plasma axis test R-AX needs it). Off with the
      implicit mover (its iteration does not converge with it; open). `raw` keeps the old deposit.
    - Results: the 0.48 pi line 81 % -> 0.2 % of the near-axis E_phi m1 power without any damping, E_phi m1 at the 3D level (5.2e21 vs
      4.7e21), J_r,t m1 47 x lower; LWFA laser and wake unchanged within the noise (peak 0.4 %, field energy 6e-7). Replaces the damping of
      the bound state (rz_axis_damp, branch rz-axis-damp, not merged).

113. **HDF5 1.12 or newer** (2026-11-21, branch hdf5-min-version; Alexander, for all group codes: computing sites drop the
    older release series; same rule as GEM-PIC-RZ and QUARZ).
    - The Makefile reads the version from H5public.h (HDF5_DIR/include, the -I paths of pkg-config, CPLUS_INCLUDE_PATH,
      CPATH, the system paths). Older than 1.12, or no header: a warning with the version and a build without HDF5
      (openPMD output, checkpoints, beams and lasers from files disabled; the same as without HDF5), not an error.
      `src/h5.hpp` stops with `#error` if an older hdf5.h is reached anyway (flags given by hand).
    - Only the 1.12 API: `H5Literate2` / `H5L_info2_t` (group listing), `H5Oget_info_by_name3` / `H5O_info2_t`
      (`h5::is_group`); no version branches.
    - A parallel HDF5 (H5_HAVE_PARALLEL in H5pubconf.h) includes mpi.h in hdf5.h: a build without MPI=1 adds the -I, -L
      and -l words of `$(MPICXX) --showme` (OpenMPI) or `-show` (MPICH). MPI=1 and CUDA=1 MPI=1 compile through mpicxx.
    - Test: `tests/hdf5_version.sh` (in `make test`): `make -n` against stand-in installations 1.10.6 (warning, no
      -DRIP_HDF5 / -lhdf5), 1.14.6 serial, 1.12.0 parallel with and without MPI=1. The HDF5 code paths themselves are
      covered by the openPMD / LASY / beam-import tests and the restart cases of tests/mpi.sh.
    - Consequence: Ubuntu 24.04's libhdf5-dev is 1.10.10, so the hosted MPI job of the CI (only by hand, decision 47)
      runs without HDF5 there; the self-hosted runners use HDF5 1.14.6 (the GPU workstation) and 1.14.6.2 (the CPU cluster).
114. **Boosted injection: neutral per lattice group; each pre-ionized ion species brings its own electrons with the same ppc
    and matching weight** (2026-11-22, branch boost-lattice-groups; Alexander: "I suggest the ppc should be the same for all
    species in every cell ... The number of ppc can vary from cell to cell"; "only the relevant species must have the same ppc.
    e.g. you have pre-ionized nitrogen, then you should use the same ppc of electrons from this ion as the ppc for ions. if,
    e.g., there is pre-ionized helium with some ppc, you need to add electrons from helium with the same ppc and weight").
    Refines decision 108 (a), which required one lattice for all injected species and so refused the wide boosted RZ LWFA deck (inner
    plasma r < 25 um at ppc 3 x 2 x 8, outer 25 .. 141.75 um at 1 x 1 x 6).
    - Rule: the injected species (at rest in the lab, density != 0) are grouped by lattice (ppc, bounds and slab: the loaders put
      the particles of such species on the same deterministic points). Every group must be neutral on its own,
      sum_{s in group} q_s Z_s n_s f_s(x) = 0 at every point to 1e-12 (Z_s: initial charge state of an ionizable species), so
      the charges cancel particle position by particle position. Groups may differ in ppc and may lie side by side (radial
      zones) or overlap in the same cells (He2+ with its electrons at one ppc, N3+ with its electrons at another). One electron
      species may neutralize several ions on its own lattice (the boosted RZ LWFA deck: elec with He and N; elec_o with He_o and N_o).
    - A deck expresses it by giving every ion species (or every zone) an electron species with the same ppc_*, zmin zmax rmin
      rmax (xmin .. zmax) and slab keys and the density sum Z n of its ions, with the same density_function.
    - Refusal: a group that is not neutral; the message names the group (its first species and ppc), the residual, the lab
      position, all species of the group and, for one electron species in it, the neutral density. A single lattice behaves as
      before (decision 108's messages are kept; electrons on a lattice of their own are refused, since that group is not neutral).
    - src/boost.cpp: boost_same_lattice, boost_net_charge(c, where, key) (one group, transverse samples within the group's
      bounds), boost_check_neutral per group.
    - Test B7b (tests/test_boost.cpp, boost_groups): RZ, gamma 10, iRIP, shape 2, filter 2, 2 modes, 20 um lab ramp; two radial
      zones (e + He2+ at 2 x 2 x 8 for r < 1.6 um, e_o + He_o at 1 x 1 x 6 outside) and two overlapping groups (He2+ + eHe at
      2 x 2 x 8, N3+ (adk) + eN at 1 x 1 x 6 in the same cells): accepted, Gauss law max|dGauss| 2.8e-16 and 3.3e-16 over 171
      steps (zone boundary included); a zone with the electron density rounded by 2e-6 refused, message naming e_o and He_o.
      B7 and B8 unchanged (B8 2.8e-15 .. 2.9e-16).
    - Not enforced, open (questions of Alexander, 2026-11-22): how many particles per ring (VLPL warns below
      ppc_theta = 2 (2M - 1), decision 79), the weight jump at a zone boundary (heavy outer electrons pulled into the fine
      region), a radial ppc table.

115. **`grid.absorber_particles = conserve` is the default** (2026-11-22, branch test-repair-111-112; Alexander: "agree").
    remove and keep stay as options.
    - Why: remove (decision 111) stops the current at the layer face, so the charge of a particle that enters a layer vanishes
      there and its field stays next to the faces inside the physical region: the Gauss law is broken there (P5: residual 2.6e-2
      next to the faces, 4.6e-16 further inside; tests/mpi.sh: max|dGauss| 1.3e-3 in the moving window + PML case, 6.6e-3 rising
      to 0.28 in the 2D foil with CPML in y, the same on 1 and 8 ranks). At the end of the window + PML cases the difference field
      remove - conserve inside the physical region holds 0.6 % (iRIP) / 12.7 % (RIP-CC) of the field energy there, mostly in the
      2 cells next to the faces, and stays frozen in time. conserve deposits the current into the layer (charge conserving) and
      removes a particle once its shape left the physical nodes + 1 cell: the Gauss law in the physical region at round-off, and it
      stops the 3D streaming instability of decision 111 as well (plasma-only, 453 fs: 6.8e6 V/m with remove and with conserve,
      5.7e9 with keep; RZ: the three within 7 %).
    - Changes: the default in src/sim.cpp and Config (src/sim.hpp); README, docs/specs/pml.md, docs/specs/rz_modes.md. Tests: the
      default is checked in test_pml; tests/mpi.sh runs the window + PML cases (iwinpml, winpml and its restarts) and the 2D foil with
      CPML in y with the default under the 1e-12 Gauss limit, and keeps one explicit remove case (mpi_winpmlr: energies and the Gauss
      residual itself rank invariant, loose limit 1e-2). Regression references: all 7 match with the new default (no re-bless).
116. **The partial lab snapshots of a boosted run in the checkpoint** (2026-11-22, branch lab-restart; Alexander: "yes, do it!").
    Numbered 116: 114 is boost-lattice-groups, 115 the conserve default (both merged).
    - Bug: the lab_diagnostics state (the planes of the open field snapshots, the particle records of the open particle snapshots,
      the done flags, the open lab_beams.txt) lived only in the memory of LabFields / LabParticles (src/boost_diag.cpp). After
      `--restart` the open snapshots lost their planes / records from before the checkpoint and were still written with
      `complete 1`; the particle snapshots finished before the checkpoint were rewritten EMPTY (their done flags were lost, so the
      first step finalized them again with no records), and lab_beams.txt was truncated to its header.
    - Choice: the state goes into the checkpoint, not a reopening of the snapshot files. A checkpoint is one consistent state; a
      snapshot file on disk may be newer than the checkpoint (finished or written as incomplete by the stopped run after it), so
      reading it back would mix two states. Generic hook: `Operator::save` (collective; named float64 arrays) is written by the
      checkpoint to `rippic/op_<name>/<key>`; the operator reads it back in setup with `read_op_state` (openpmd.hpp).
    - lab_fields: times, status per snapshot (0 open, 1 written, 2 incomplete), for each open snapshot the lab x of all planes
      (`xl_K`) and the planes gathered over the ranks as the global array (`planes_K`, [component][n2][n1][plane], step order; the
      same gather as the final write). A restart puts them on rank 0 in front of the new planes: any number of ranks.
    - lab_particles: times, species, status, the number of ranks, the length of lab_beams.txt, and for each open snapshot and
      species the records gathered in rank order (`rec_K_Q`) with the count per rank (`cnt_K_Q`). The same number of ranks: every
      rank gets its own records back (the files are byte-identical); another number: all on rank 0 (records in another order,
      moments to round-off, as between rank counts anyway). lab_beams.txt is cut to its length at the checkpoint (the lines the
      stopped run wrote later are dropped).
    - Without usable state (a checkpoint written before this decision, or other lab times / species in the deck): status from
      the geometry at the checkpoint's step (written if the lab time had left the box, open-but-incomplete if the snapshot had
      planes / its lab time was inside the box up to the injection plane, else open); incomplete snapshots are written with
      `complete 0` and a warning in the log, written ones are not touched, and lab_beams.txt keeps the lines of the written ones.
    - Cost: at a checkpoint every open snapshot is gathered like its final write (the global array on every rank for the
      allreduce, as finalize already does); the checkpoint grows by the open snapshots (float64).
    - Byte identity needs deterministic sums: the RZ deposit adds with atomics, so with several OpenMP threads (or on the GPU)
      two UNINTERRUPTED runs already differ at round-off (B9 deck, 8 threads: fields 2e-15, particle records 7e-15 of the
      maximum). B9 therefore runs with one thread (tests/mpi.sh runs with OMP_NUM_THREADS=1 anyway); the GPU check is 1e-8.
    - Tests (2026-11-22): tests/test_boost.cpp B9 (RZ, 2 modes, RIP-CC, rz_deposit = chord, shape 2, filter 2, 266 boosted steps,
      checkpoints at steps 149 (1 snapshot open) and 199 (3 open)): restarted once and twice, all 13 lab files (fields,
      particles, lab_beams.txt; 1.9 MB) byte-identical to the uninterrupted run; restart with other lab times: 4 files
      complete 0, 2 complete 1 (not begun). tests/mpi.sh "Lab snapshots across checkpoint restarts", 8 ranks: RZ (the same
      numerics) and 3D (examples/lwfa_boost.deck), stopped by a shorter t_end and continued with --restart latest once and
      twice: 13 of 13 lab files identical (6.6 MB RZ, 32.7 MB 3D); 8 ranks restarted on 3: lab_beams.txt 0 (RZ) / 8.7e-11 (3D),
      lab fields 1.1e-15 / 4.7e-15, particle column sums 7e-14 / 4e-14 (the uninterrupted 3D run on 1 vs 8 ranks: 4.9e-10,
      8.3e-15, 8e-14).
117. **QED (spin, photon emission, Breit-Wheeler pairs) in the boosted frame; lab energies, lab spins and lab photon
    polarization in the outputs. The eight defaults accepted by Alexander, 2026-11-23** (branch qed-boost; Alexander:
    "implement QED in the boosted frame", then on the eight defaults below: "I agree with defaults"; as every decision, held
    until counter-evidence arrives; docs/specs/qed.md "Boosted frame", src/qed.*, src/boost_diag.cpp, one line in
    src/boost.cpp)
    - The QED operator is unchanged: the code units stay those of the lab omega0 (hw fixed), chi and chi_gamma are invariants,
      and C0 = c0f / gamma' per boosted time gives W dt' = invariant x proper time, so the numbers of photons and pairs do not
      depend on the frame; u = eps_gamma / eps_e and x = eps_+ / eps_gamma are invariant for collinear emission and decay.
    - Choices (the defaults, accepted by Alexander, 2026-11-23): (1) photon_min_energy is a lab energy, eps_lab = gamma_b eps' (1 + beta_b n'_x)
      (n' the photon direction, x the boost axis; RZ z); (2) the energies of qed.txt (E_rad, E_photons, E_pairs), the photon
      spectra dN/d eps and the pair-lepton spectra are lab energies; counts, u and x bins as they are; the rows stay at the
      boosted steps, and the instantaneous columns (W, <S>, <chi>, chi_max, N_photons, E_photons_now) are the box at the boosted
      time; diag.txt in the boosted frame; (3) the deck `spin` is the rest-frame spin seen from the lab; the code keeps the
      rest-frame spin reached by the pure boost from the simulation frame, converted at setup with the species' mean lab momentum
      through the spin 4-vector (qed_spin_boost: exact for a cold beam, the identity along the boost axis or at rest in the lab;
      the Wigner rotation of a momentum spread at load is neglected); (4) the photon Stokes vector is unchanged (the basis
      (e1, n x e1) goes over covariantly, the helicity is invariant), the polarization axis e1 is the polarization 4-vector
      boosted with the gauge fixed transverse (qed_pol_boost); (5) the lab particle snapshots: photons massless (velocity u/|u|,
      u_x = gamma_b (u'_x + beta_b |u'|)); extra columns s0 s1 s2 (lab rest-frame spin, Wigner rotation included) for species
      with spin and xi1 xi2 xi3 pe0 pe1 pe2 (lab) for photons, named in the .txt; other species unchanged; the checkpoint of the
      open snapshots (decision 116) carries the record widths (`widths`; absent = 7); (6) the pair threshold om > 2 in the
      simulation frame (where the pair is made; the rate is negligible there anyway), the chi_gamma cut 0.01 invariant;
      (7) sub-steps per boosted step as before (P>1 = 0 in all runs below); (8) random numbers per boosted step: lab and
      boosted runs are different realizations, compared statistically.
    - Limits of the accepted defaults (documented in docs/specs/qed.md "Boosted frame"): (a) the deck spin is converted at load
      with the species' MEAN lab momentum: exact for a cold beam, the per-particle Wigner rotation of a momentum spread is
      neglected at load (angle of the order of the spread angle); the lab snapshots convert every particle exactly; (b) the rows
      of qed.txt are at boosted times: the counts and the lab energies are frame-independent sums since the start, but the
      instantaneous columns (W, <S>, <chi>, chi_max, N_photons, E_photons_now) describe the box at a boosted time and are no lab
      quantities (chi_max 0.20 boosted vs 1.17 lab at gamma 5, qed_collision: the beam crosses the pulse peak between two rows);
      lab-time states come from the lab snapshots. No test below gave evidence against one of the eight defaults.
    - Beyond the eight defaults (provisional, for Alexander): none in the physics; implementation details only (the photon
      species is skipped by apply_boost since it is empty at the start; the advice ppc_x x gamma^2 (1 + beta)^2 for beams moving
      against the boost).
    - Boost files: src/boost_diag.cpp (photons, extra columns, checkpoint widths) and src/boost.cpp (apply_boost skips the
      photon species, which are empty at the start; boost_summary does not list them). src/boost.hpp, boost_antenna.cpp untouched.
    - Beams moving against the boost are gamma_b^2 (1 + beta_b)^2 shorter in boosted cells (13.9 at gamma_b = 2, 98 at 5): the
      1.5 um beam of the QED decks is 2.7 / 0.38 boosted cells long, so ppc_x must grow by that factor to keep the macro-particles
      (at gamma_b = 5 with the deck's ppc_x no lattice point fell into the beam: nothing was loaded).
    - Tests (the CPU cluster, 4 seeds each, iRIP and the lab-fixed antenna in all runs; the lab run = gamma 1 on the same lab grid):
      examples/qed_cascade.deck (10 GeV, a0 = 80), pairs per beam electron: lambda/20 lab 0.3144 +- 0.0013, gamma 2 0.3134 +-
      0.0021 (-0.4 sigma), gamma 5 0.2532 (-19 %); lambda/40 lab 0.3166, gamma 2 0.3204 (+2.3 sigma), gamma 5 0.3114 (-1.7 %);
      lambda/80 lab 0.3198 +- 0.0012, gamma 5 0.3150 +- 0.0010 (-1.5 %, -3.0 sigma). E_rad / E_0 (lab energies): 0.7245 / 0.7252
      / 0.6896; 0.7294 / 0.7296 / 0.7246; 0.7284 / - / 0.7291. Pair spins <S+.e2> -0.375 / -0.371 / -0.377 (lambda/20) within
      1 sigma; the electrons' lab spin <S2> at the lab time 29.35 fs from the lab snapshot vs the lab run's qed.txt: 0.5950 /
      0.5941 / 0.6210 (lambda/20), 0.5908 / - / 0.5914 (lambda/80). examples/qed_collision.deck (1 GeV, a0 = 100): photons per
      electron 14.83 / 14.87 / 12.98 (lambda/20), 15.03 / - / 14.87 (lambda/80); <S2> 0.786 / 0.778 / 0.810, 0.777 / - / 0.783.
      Spectra dN/du, dN/d eps_lab, dN/dx: chi^2 per bin 0.8 - 3.5 at gamma 2 (lambda/20), 0.7 - 3.9 at gamma 5 (lambda/80).
      Energy balance at the lab time from the lab snapshot (kinetic + photons + 2 m c^2 per pair): -2.056e-5 vs the lab run's
      -2.067e-5 (cascade, gamma 2), -4.20e-4 vs -4.18e-4 (collision; the laser's work). So gamma 2 agrees with the lab run on the
      lab grid within ~2.5 sigma; gamma 5 converges to it with the lab cell (the deficit 19 % -> 1.7 % -> 1.5 % in the pairs, 8 %
      -> 0.7 % -> 0.1 % in the radiated energy, while the lab run itself changes by ~1 % per halving). Cause (diagnosed
      2026-11-23 with lab field snapshots, the beam's chi every step and one-at-a-time refinements; the first reading "field
      resolution, c dt' grows" was imprecise: cells per wavelength and the laser phase per step of a counter-moving electron
      are invariant): at gamma 5 the laser emitted by the lab-fixed antenna is too weak during the first few um downstream of
      the antenna. Its carrier amplitude on the axis at the lab times 21 / 25 / 28 fs (pulse centre 1.9 / 2.7 / 3.6 um from the
      antenna) is 0.82 / 0.94 / 0.99 of the lab run's (gamma 2: 0.999 at all times). The collision is at 2.7 um, so the beam's
      peak chi (every step) is 8.64 vs 9.28 (-7 %), and the pairs (~chi^3 there) fall by 19 %. Halving c dt' (lambda/40) gives
      0.998 - 1.004 from 24 fs on (peak chi 9.40, pairs 0.311 vs the lab 0.313); halving h_perp (0.94 at 25 fs, peak chi 8.66),
      doubling ppc_x (peak chi 8.64) and a 4x longer beam (pairs -43 %, the beam meets the pulse even closer to the antenna)
      do not help. The global peak of the laser on the beam's plane is right (79.6 vs 79.2): it is reached downstream.
      Entrance-plane injection is no alternative at gamma 5 (the pulse enters far upstream: peak 3.9 on the beam's plane). So:
      not the QED operator, and not the transverse grid; the antenna's start-up region at large c dt' (src/boost_antenna.cpp,
      the boosted-frame session's code: reported there). The collision deck (waist 1.5 um) has in addition a lower global
      peak (94.9 vs 106.2), also removed only by halving c dt' (99.9). chi_max of qed.txt is no lab quantity: the maximum over the particles in the box at the output rows (the whole
      beam crosses the pulse peak between two rows in the boosted frame; 0.20 vs 1.17 at gamma 5); dN/du carries the chi
      dependence and agrees.
    - tests/test_qed.cpp Q11: the Wigner rotation of two perpendicular boosts, cos theta = (g1 + g2)/(1 + g1 g2), to 4e-15; the
      spin along the boost axis unchanged; there and back 6e-13; the polarization axis of a photon along y polarized along x,
      (1/gamma, beta, 0), and transverse and unit after random boosts. tests/mpi.sh "QED in the boosted frame": the small-box
      cascade at gamma 2 on 1 vs 8 ranks (energies 3.8e-13, qed.txt 1.7e-11, pair weight 1.9372082992e-02 on both, the lab
      snapshot of the fields and the four species: column sums 7.3e-14); restarted on 8 ranks: 11 of 11 lab files identical;
      8 ranks restarted on 3: column sums 9.8e-14.
118. **Boosted beams: loading checks (error below one lattice plane, warning below 20, loaded vs deck charge); the loading itself
    unchanged** (2026-11-24, branch qed-boost; Alexander: "1: yes" to the checks proposed after the decision-117 tests)
    - Why: a beam moving against the boost is gamma^2 (1 + beta)^2 shorter in boosted cells (14 at gamma 2, 98 at 5). The boosted
      lattice (boost_lattice: regular in boosted cells, ppc_x per cell) then holds few or no planes of it: the 1.5 um beam of the
      QED decks (ppc_x 2, dx 0.04 um; 0.77 planes at gamma 5, 5.4 at gamma 2) loaded NOTHING at gamma 5 and 11 % more
      charge than the lab run at gamma 2, silently.
    - n_planes = (xmax - xmin) a ppc_x / dx', a = 1 / (gamma (1 - beta v)) (boost_beam_planes): apply_boost throws below 1 (the
      message names n, the contraction and the ppc_x for 1 and for 20 planes); the summary warns below 20 (charge quantized,
      error ~ 1/n; boost_beam_warning). After the load (load_species, collective) the loaded charge (sum w) is printed against
      the deck's (boost_beam_deck_charge: exact along the boost axis, transversely on the loader's lattice in Cart3D / 2D, exact
      in r in RZ) and a difference above 1 % is a warning (boost_beam_charge_check). Beams from files and empty species are not
      checked. Files: src/boost.cpp, src/boost.hpp (the checks), src/sim.cpp (two calls in load_species).
    - Test tests/test_boost.cpp B10: the QED decks' beam at gamma 5, ppc_x 2: error, n = 0.765 (analytic 1.5 ppc_x / (s^2 dx),
      s = gamma (1 + beta)), the message names ppc_x >= 3 (53 for 20); at gamma 2, ppc_x 2: n = 5.385, the warning, the loaded
      charge -7.15 % of the deck's (the check prints it and warns); ppc_x 28: 75.4 planes, no warning, +0.81 %.
119. **Agent skills `vlpl-setup` and `vlpl-simulation` in `skills/` (open Agent Skills format), kept in step with the code by
    a test** (2026-11-24, branch agent-skill; Alexander, 2026-11-24: "does VLPL have skills? we should add them"; the pattern
    of GEM-PIC-RZ and QUARZ, his decision of 2026-11-18 to give all codes skills)
    - What: two SKILL.md folders (docs/agent_skill.md). The deck reference in the skill is generated from the README sections
      "Units" and "Deck" (tools/sync_skills.sh), eleven example decks are copies of repository decks, one (lwfa_plain_rz) is
      the skill's own with all numbers from scripts/vlpl_numbers.py.
    - Test: tests/agent_skill.sh. The static part (format, copies in sync, every `block.key` of the skill files read by the
      parser and in the reference) runs in `make test`: editing the README "Deck" section or a copied example without
      `sh tools/sync_skills.sh` fails it. `make agent_skill` runs every deck of the skills for 20 steps, a restart, the helper
      scripts against the code's summary and, with an MPI executable, 2 ranks against 1.
    - Rules of the skills: dialogue with the user first, the agent does not modify the source, long runs only after the
      user's yes, resolution and stability rules as numbers. The skills belong to the public release: nothing private goes
      in (they pass the gate of tools/export_public.sh unchanged, and the generated copies stay in sync after its rewrite
      rules). Naming (Alexander, 2026-11-18): folders vlpl-setup, vlpl-simulation; variants VLPL-RIP, -iRIP, -RZ, -iRZ.
    - Open, for Alexander: the starting values that are not measured for VLPL (transverse cell <= waist / 20 and
      (c / omega_p) / 5, ppc for cold plasma, the window length, the box of 3 spots, omega_p dt <= 0.2 for wakefields) are
      common PIC practice proposed by the session, listed in docs/agent_skill.md "What the rules are based on".

120. **The omega_p dt start-up checks count a boosted beam with its density in the boosted frame, n' = n gamma_b (1 - beta_b
    beta_beam) (Alexander, 2026-11-24: "Yes, fix and rerun")** (branch boost-check-density; src/sim.hpp
    `Config::check_density_factor`, the three sums in src/sim.cpp: Config::check (Cartesian), Config::check_rz,
    Simulation::check_peak_density; tests/test_boost.cpp B11).
    - Before: the deck density is a lab value, apply_boost leaves it so, and the checks summed it unconverted for every
      species, with the boosted c dt'. A gamma 1e4 beam at gamma_b = 10 was checked with 20 times its boosted density: lab
      n = 11.7 n_c, lab dz = lambda/51.2 (c dt' = 0.3117 um) gave sqrt(11.7) k0 c dt' = 8.37 and the refusal "omega_p dt >= 2
      (iRIP)", although the boosted omega_b' dt' = 1.875 (found in the bunch-in-vacuum density scan of the iRIP paper review).
    - Now, per boost kind: boost_kind 2 (beam, ballistic map): n gamma_b (1 - beta_b beta_beam), beta_beam the lab velocity
      (0.0501 n at gamma 1e4, gamma_b 10; a beam against the boost becomes denser, up to 2 gamma_b n). boost_kind 1 (plasma
      at rest in the lab): unchanged, the lab density n. Its boosted density is gamma_b n, but the plasma streams at -beta_b c,
      and the response that enters the explicit current is the transverse one, n' / gamma_b = n (longitudinal n / gamma_b^2),
      so the lab density is already the right number; changing it would refuse boosted LWFA decks that run stably. Without
      boost: unchanged. The ionization term (wp2_electrons) gets the same factor as its species.
    - Only the check changes, not the solver or its limit: iRIP and Yee omega_p dt < 2, RIP-CC
      dx^2 (1/dy^2 + 1/dz^2) + (omega_p dt)^2 / 4 < 1 (RZ: with lambda_max), now evaluated with the boosted beam density.
    - Kept conservative on purpose: the check uses the rest mass. A relativistic beam responds with n' / gamma' (transverse;
      = n / gamma, the proper density, a Lorentz invariant), which would relax the check by gamma' (500 for the beam above);
      not applied, as the limit of a beam with a momentum spread was not measured. For Alexander if it ever matters.
    - Test B11: the boosted RZ / iRIP deck above is accepted at lab n = 11.7 n_c (omega_b' dt' = 1.875) and refused at 20 n_c
      (2.45); the factor equals gamma_b (1 - beta_b beta_beam) to 1e-9; a plasma at rest in the same deck has factor 1; the
      lab-frame deck with dx = c dt' is refused at 11.7 n_c and accepted at 0.5 n_c, as before.
122. **Metal blocks: particles cross the metal by default; absorption is an option (`particles = pass | absorb`; Alexander,
    2026-11-25: "why should we remove particles from metals or set their charges to zero? I suggest we introduce a switch")**
    (branch metal-particles; blocks `metal` in Cart3D / 2D and RZ, `rings`; src/metal3d.cpp, src/structure.cpp, src/rz_sim.cpp,
    src/species.cpp `remove_flagged`; tests S4, S4b, M3, P5).
    - Found by Oliver in a foam-in-cone pair-production run: positrons and photons that reached the cone wall ended with
      w = 0 and gamma = 1. The metal absorbed every species, photons included, and kept the dead particles in the arrays
      (pushed and written every step). Absorption was a simplification for the wakefield structures (the wall as a sink for
      halo), not a requirement of the scheme: the Gauss checks skip the points with metal in their stencil.
    - `pass` (the default): nothing is done to the particles. The fields at the metal points are 0, so a particle inside moves
      on a straight line; entry and exit change the surface charge (transition radiation). For 100 MeV - 1 GeV particles and
      um-thick walls this is the physics to first order (gold, 10 um: ~22 keV collisional and ~0.3 % radiative loss, 0.23 % of
      the photons convert; multiple scattering ~7 mrad at 100 MeV is NOT modelled). Slow particles cross a wall too, where a
      real wall stops them (electrons below ~100 keV in 10 um of gold): the reason `absorb` stays.
    - `absorb`: as before for a particle whose shape lies inside the metal, but it is removed from the species. The charge
      it carried stays on the conductor as surface charge: its current up to the surface was deposited and the field step
      conserves charge (a floating conductor; S4b: flux of E around an isolated plate = the absorbed charge).
    - Fixed on the way in the Cart3D absorption: the guard test left half-cell sheets at the upper rank edges untested
      (shape 2: 1 rank != N ranks); in 2D particles in the upper half of the slab were never absorbed; photons were booked
      with m (gamma - 1) instead of w |u|.
    - Results change for decks with a metal block or rings whose particles reach the metal (the default is now pass): set
      `particles = absorb` for the old behaviour. The outer RZ wall and the open edges absorb as before.
    - Planned, not in this step (Alexander, 2026-11-25): the same choice for a metal with finite conductivity (a Drude medium:
      local plasma frequency and collision frequency) in every solver, the perfect conductor in the implicit solvers, and a
      material option (stopping power, scattering, bremsstrahlung, Bethe-Heitler, trident).

123. **The Drude / Lorentz medium requires iRIP (Alexander, accepted on 2026-11-25: "if the drude model works with the iRIP
    scheme, we can accept it and write: Drude requires iRIP").** The `medium` blocks stay with `numerics.solver = irip`
    (decision 84); rip_cc keeps refusing them. A metal with finite conductivity is a `medium` block in an iRIP run; particles
    cross it in its self-consistent fields.
    - Tried on 2026-11-25 in the explicit step (branch `medium-explicit`, not for merging; Cart3D / 2D): the time-centred
      current J^{n+1} - J^n = dt (wp2 Ebar - w02 Pbar - nu u), E_x local with (1 + chi + b), E_y and E_z in the lines along x of
      the dielectric blocks (decision 100) with chi -> chi + b, b = wp2 dt^2 / (4 (1 + nu dt / 2 + w02 dt^2 / 4)).
    - What works: a uniform medium for any omega_p dt (von Neumann, wp^2 dt^2 = 0.01 ... 1e6: |g| - 1 at round-off); no damping
      without collisions (lossless overdense slab at normal incidence: |R - 1| = 7e-12, 2e-12, 4e-12 at omega_p dt = 0.5, 3, 20);
      s-polarized Fresnel reflection, refraction angle and collisional absorption at second order; Gauss law with particles 4e-15.
    - What fails: p-polarization (medium in E_x and E_y together) with a density that varies along x. Energy after 8000 steps
      of random fields, lossless: sharp slab 1.0e5 (2D) and 2.5e5 (3D) at omega_p dt = 3, 2e6 at 30; a smooth cosine density
      133 (2D) and 4.7e8 (3D); with nu dt = 1 the dissipated energy reaches 300 - 600 E0. The matrix of one step of the real
      code (8 x 6 and 16 x 10 cells): medium in E_x only, E_y only or E_z only is stable for every profile; E_x + E_y with a
      gradient grows by up to 5e-3 per step, the mode is the local plasma oscillation in the gradient (84 % of its norm in J).
    - Ruled out as cures: iterating the predictor of E_x towards the time mean (1, 2, 3, 5 passes: the slab stays at 9e-4 per
      step); driving the transverse current with the exact work field of the transport, (M+ + M-) / 2 (local current, stable
      in the small matrices, but the 8000-step runs still grow: slab 3e4, cosine 2e3 in 2D). The explicit step conserves a
      modified energy only; iRIP has the exact balance W + K + dissipated for any medium shape (decision 84), which is the
      reason for the choice.
    - The dielectric blocks of decision 100 are not affected: a sharp eps = 4 slab stays at 1.13 (2D), 1.14 (3D) of the
      initial energy over 8000 steps in the same test.
    - Scripts and the diagnosis switches: branch `medium-explicit`, research/drude_cc/ (vonneumann.py, eig.py; line.py is a
      prototype that mispredicted the dielectric: not to be used for interfaces), tests/test_medium.cpp (VLPL_MEDCC_*).
    - Open (Alexander's stages): the perfect conductor (`metal`, `rings`) in the implicit solvers; a material model.

124. **Metal blocks in the implicit solvers are a dense Drude medium (Alexander, 2026-11-25: "Yes, start stage 3"; proposal:
    the perfect metal in the implicit solvers as a Drude medium of very high density).** Branch metal-implicit;
    src/structure.cpp `metal_medium`, src/sim.cpp (RZ); tests L9a-e (tests/test_medium2.cpp), tests/mpi.sh.
    - With `numerics.solver = irip` the blocks `metal` (Cart3D / 2D: function(x, y, z) > 0; RZ: a function of z and r) and
      `rings` (RZ: rings, gaps, jacket) become a `medium` block of that name (decision 84) with the block's shape as profile.
      rip_cc keeps the staircase perfect conductor (decisions 83, 102, 103).
    - Density: `density` (n / n_c) if given, otherwise omega_p dt = 10: a skin depth of dx / 10, so the surface is a mirror
      and nothing reaches the third cell (L9b: |R - 1| = 3e-12, field energy deeper than two cells <= 6e-13 of the pulse).
      `collision_frequency` (nu / omega0, default 0) gives the metal with finite conductivity in the same block.
    - Why a medium and not E = 0 in the implicit solve: the medium has the exact energy balance and the Gauss law of
      decision 84 for any shape, costs no new solver, and the surface enters by its sub-cell volume fraction instead of a
      staircase. The price: GMRES iterations (25 - 28 per solve in L9c, L9d with a hot plasma inside), and the medium keeps
      one cell clear of CPML layers and of the radial absorber.
    - Particles cross the metal in its self-consistent fields (`particles = pass`, the default).
    - `particles = absorb` (Alexander, 2026-11-25: "Yes, add absorb"), Cart3D / 2D: a particle whose shape lies inside the
      block (the node mask of decision 102, `Simulation::metal_mask_`) is removed; the charge density it deposited is kept
      at rest in `Simulation::rho_abs_` and added to the particle charge in `compute_rho`, so the Gauss law (with the
      polarization charge of the medium) is untouched by the removal and the medium screens the stored charge as it
      screened the particle (src/structure.cpp `metal_absorb_medium`: two extra charge deposits of a species, only in a
      step in which it loses particles). Test L9e. Limits: refused with a moving window (the stored charge is not shifted)
      and in RZ (`rings`, `metal`). The stored charge is kept on the owned cells (the removed charge of the guard cells
      goes to its owners, a collective exchange in every step in which any rank absorbed a particle) and is written to
      the checkpoints (mesh `vlpl_metal_absorbed`): a restart on the same ranks is exact, on other rank counts to
      round-off (tests/mpi.sh).
    - RZ: the medium has the same coefficients for all azimuthal modes, so a metal function of x, y or phi is refused in irip
      (rip_cc has the block metal on the collocation angles, decision 103).

125. **The common input form of the VLPL family in VLPL (Alexander, 2026-11-26: the deck standard for VLPL, GEM-PIC-RZ
    and QUARZ accepted as proposed; "Yes, VLPL next"; "add these features to VLPL").** Branch deck-standard;
    src/deck.*, `Deck::apply_standard_names`, block `bunch` (src/structure.cpp `species_bunch_from_deck`,
    src/beam_import.* `BunchSource`, `MomentSums`); tests/test_deck.cpp; docs/specs/deck_standard.md;
    examples/rz_bunch.deck.
    - Syntax: a statement ends at the end of the line or at `;`, a block may stand on one line; keys and block kinds
      in any case (the name given to a block keeps its case: it is in file names); a key twice in a block is an error
      (before: the last one won silently). Every deck that ran before runs unchanged.
    - Units: energies eV .. TeV, emittances mm*mrad, more spellings (µm, micron, as, s, fC, uC, urad); a density is a
      bare number in n_c as always, or carries cm^-3, m^-3 (converted with grid.lambda0) or n_c.
    - Common names as aliases, rewritten before the configuration is read: grid.m_max (modes = m_max + 1), laser
      wavelength (sets grid.lambda0 or must equal it), tau_fwhm (duration), z_focus (focus), species charge with a unit
      (total_charge). Two names of one quantity are an error.
    - Block `bunch <name>`: a Gaussian bunch by charge, energy, spread, sizes and normalized emittances, at its waist,
      in beam coordinates (z along the propagation) in every geometry; it goes through the beam-import path (self-field
      init, t_start, rigid_until). Why a generator inside the code: GEM-PIC-RZ and QUARZ have one, and a chain test
      should not need a file for its first stage. Memory rule: every rank draws the same sequence (mt19937_64, 7 normal
      numbers per particle) in portions of 2^18 and keeps its own particles; the printed moments are one-pass sums
      about the nominal centre. Electrons and positrons only.
    - A beam from a file is read block-wise as well (Alexander, 2026-11-26: reading the whole file on every rank "is a
      straight bug. fix it!"): `BeamFileSource` in src/beam_import.*, 2^18 particles at a time, up to three passes
      (total weight and the rms for the jitter; charge centroid and mean gamma; delivery); arithmetic and the order of
      the random numbers as in the former import, so the particles are unchanged. Test S6b (tests/test_structure.cpp).
    - Not changed: `output.every` keeps its
      VLPL meaning (rows of diag.txt). Positions stay absolute lab coordinates.
    - Numbers (the CPU cluster epyc, job 18147, binaries of a46d7bf built 2026-11-26 20:42 / 20:46): compact form and common names =
      the old start-up summary; 1.742e19 cm^-3 -> n / n_c to 3.4e-9; 200000 generated particles against the deck:
      Q 3.8e-12, E 7.7e-8, dE/E 2.6e-4, sigma 3.1e-3 / 1.1e-3 / 3.3e-5, eps_n 8.3e-4 / 4.5e-4; portions = one portion;
      one-pass moments = file_moments to 3.8e-12. examples/rz_bunch.deck (1 mm, conducting wall): sigma 2.996 / 3.002 ->
      3.1086 / 3.1140 um (free drift with the sampled moments 3.1099 / 3.1155, 0.05 % below), eps_n unchanged to 6
      digits, E 203.88875 -> 203.88933 MeV (+0.0006 MeV), dE/E 1.00e-4 -> 1.03e-4.
