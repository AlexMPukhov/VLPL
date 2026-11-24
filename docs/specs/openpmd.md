# openPMD output and checkpoint/restart

Status: implemented (`src/openpmd.*`, `src/h5.*`, decision 34). Needs a build with HDF5.

## What is written
openPMD 1.1.0 over HDF5, one file per iteration (`iterationEncoding = fileBased`).
Values are stored in code units, each with its `unitSI`, so nothing is converted on the way out:
readers (openPMD-api, openPMD-viewer, yt, ParaView) multiply by `unitSI`.
The one exception is the output's `weighting`, see below.

- **Output** `dir/openpmd/data_<step>.h5`, every `output.openpmd_every` steps and at step 0.
  - The ED-PIC extension (`openPMDextension = 1`) is used, the convention of FBPIC, WarpX and PIConGPU.
  - Meshes:
    - `E`, `B` at t_n.
    - `J` = j^{n-1/2}, the current of the last step, `timeOffset = -dt/2`.
    - `rho` and `rho_<species>`, charge densities deposited at output time.
  - Particles of the species listed in `output.openpmd_species` (or `all`):
    - `position` (absolute x) and `positionOffset` (constant 0).
    - `momentum` (u = γv/c with `unitSI = m m_e c`, so kg m/s; u^{n-1/2}, `timeOffset = -dt/2`).
    - `weighting`: real particles per macro-particle, `unitSI = 1` as ED-PIC requires.
    - `charge` and `mass` as constant records.
    - Every further registered attribute as a scalar record of its own type.
  - Precision: float32 by default, `openpmd_precision = double` for float64.
- **Checkpoint** `dir/checkpoints/checkpoint_<step>.h5`, every `checkpoint.every` steps.
  It is also written when `checkpoint.stop_after_hours` is reached.
  - Base openPMD (`openPMDextension = 0`), always float64.
  - The raw weight is stored, with `unitSI = n_c / k0^3`.
  - Contents:
    - `E`, `B`.
    - `rippic_gauss0`: the initial Gauss residual, which the diagnostics subtract.
    - The twelve CPML memories `rippic_pml_*`, with one extra cell below x = 0 (below).
    - All particles of all loaded species with all registered attributes.
    - `particlePatches`: one patch per rank, its box and its particles.
    - The group `/data/<step>/rippic`:
      - step, time and window shift `wx`;
      - the grid (N, h), shape order and number of ranks;
      - the species names and which are loaded;
      - `rank_box` (offset, size per rank);
      - the tile starts of the last sort per rank and species.
- Each file is written as `<name>.tmp` and renamed when complete, so an interrupted write never
  looks like a checkpoint. `checkpoint.keep` newest checkpoints are kept (default 2).

Both kinds pass `openPMD_check_h5` (the openPMD validator) with 0 errors and 0 warnings, the
output with `--EDPIC`. They also read with openPMD-api 0.17.

## Layout
- Meshes have `dataOrder = C` with `axisLabels = (z, y, x)`, the storage order of the arrays
  (x fastest).
  - `gridSpacing` = (h_z, h_y, h_x) in 1/k0 (`gridUnitSI = 1/k0`).
  - `gridGlobalOffset` = (0, 0, wx h_x): the absolute x of index 0 in the moving window.
- The staggering positions within the cell come from the RipCC descriptor (`src/stagger.hpp`), in
  (z, y, x) order:
  - E_x (0,0,0), E_y (0,½,0), E_z (½,0,0);
  - B_x (½,½,0), B_y (½,0,0), B_z (0,½,0);
  - ρ (0,0,½); J_x (0,0,0), J_y (0,½,½), J_z (½,0,½).
- Units: E in m_e c ω0/e, B in m_e ω0/e, ρ in e n_c, J in e n_c c, lengths 1/k0, times 1/ω0,
  with n_c = ε0 m_e ω0²/e² (CODATA 2018).

## Parallel I/O
- Serial HDF5. Every rank packs its owned block (or its particle attribute) on the host. The
  blocks stream through rank 0 in rank order, in messages of at most 256 MB. Rank 0 alone writes
  them as hyperslabs.
- Rank 0 holds at most one block of one field component, or one chunk of particles. Memory does
  not grow with the number of ranks; the time does.
- Ranks without a CPML memory (away from the layers) send nothing, and the file keeps the fill
  value 0. That is the value of ψ outside the layers.
- On a restart every rank opens the file read-only and reads its own hyperslab.
- Parallel HDF5 (collective writes) is the upgrade when the output time at 64 GPUs matters.

## Restart (bit-exact)
`vlpl deck --restart <file>`; `--restart latest` takes the newest checkpoint in
`dir/checkpoints`, and `--restart auto` does the same or starts fresh when there is none (for a
chain of batch jobs). The deck must have the same grid (n, d, λ0), shape order and species. Any
other parameter may change (t_end, output, threads, ranks).

What makes the continuation exact:
- The fields are restored in the owned cells. The guards come from the exchange, as at the end of
  every step (the scratch arrays of the step carry no state).
- The transport ψ of the CPML is also kept at i = -1: rip_cc_step_pml updates the source at the
  left face of cell 0 itself. Its file x range is therefore [-1, N). On a restart with the same
  ranks, i = -1 of an interior rank is its left neighbour's cell n-1. That holds since the window
  shift now exchanges ψ after shifting (before, the guard kept the value of one cell further left;
  that made 4 ranks along x differ from 1 rank by 6 % in the field energy of a window run with
  layers).
- The particles keep their order per rank, and the tile starts of the last sort are restored: the
  tiled and box deposits use them until the next change.
- `wx`, the step, the time, the loaded flags of delayed species and the initial Gauss residual are
  restored.
- Laser hooks, plasma loading (seeded per absolute cell) and the window freeze depend only on t
  and wx.
- The text logs continue: `diag.txt`, `beam_*.txt`, `spectrum_*.txt`, the particle and slice
  indexes. Lines of later steps, written before the old run stopped, are dropped.

Exactness also needs a deterministic deposit. On the CPU the coloured tiles are bit-identical for
any thread count. Two holes found by the restart test are closed:
- Particles in the cell below an open lower edge (leaving through it) were strays, deposited
  with atomics in thread order (one column per step in a moving window). They are now clamped into
  the first tile, exactly as the sort does it.
- A species loaded later (`t_start`) is sorted right away; its first push took the atomic path.

The GPU deposit uses atomics (`auto` = global), so there the continuation agrees to round-off.

Another decomposition is allowed: every rank reads the particles of all patches in chunks of 2^20
and keeps those in [keep_lo, hi), as migrate would. The sums then run in another order: the result
agrees to round-off, not bit for bit.

- RZ (`grid.geometry = rz`): E, B in thetaMode, the Gauss residual per slot, particles of the z slabs;
  docs/specs/rz_modes.md step 4c, decision 51.

## Deck interface
```
output {
  openpmd_every     = 100           # steps, 0 = off (default)
  openpmd_fields    = E B rho       # E B J rho rho_<species> (default E B rho)
  openpmd_species   = electrons     # particle output: species names or all (default none)
  openpmd_precision = single        # single (default) or double
}
checkpoint {
  every            = 1000           # steps, 0 = off (default)
  keep             = 2              # newest checkpoints kept, 0 = all
  stop_after_hours = 23.5           # write a checkpoint and stop after this wall time, 0 = never
}
```
`make` builds with HDF5 when pkg-config finds it. Otherwise give `HDF5_DIR=/path` (with include/ and
lib/), or `HDF5=0` to build without. A deck with openPMD output or checkpoints is rejected by a
build without HDF5.

## Tests
- Fast (tests/test_openpmd.cpp; skipped without HDF5):
  - O1, output of a window run with laser, CPML, plasma and beams, read back with plain HDF5 calls:
    - E, B, J of this rank's block are exactly the run's arrays, and ρ agrees to 2e-15 (deposited
      again).
    - Σ ρ_s = ρ to 3e-15.
    - The beam particles (position, momentum, weighting = w n_c/k0³) are exact.
    - The unitSI of E, B, J, ρ, length, time, momentum and charge are exact against the CODATA
      constants.
    - The staggering positions and gridGlobalOffset = wx h_x are checked.
  - O2, restart: the uninterrupted run against one that writes a checkpoint at step 60, runs 7
    steps more (as if it stopped later) and is restarted from it.
    - The deck has a moving window, CPML in y and z, a laser, a hot plasma with an up-ramp, and two
      beams loaded at t_start before and after the checkpoint.
    - Fields, CPML memories and all particles are compared at step 117, and so are the beam
      diagnostic files.
    - Difference 0 on 1, 2, 3 threads and on 4 ranks (CPU); round-off on the GPU.
- MPI (tests/mpi.sh):
  - Window with CPML, 1 vs 8 ranks.
  - Restart of the 8-rank checkpoint on 8 ranks: diag.txt byte-identical to the uninterrupted run.
  - The same checkpoint on 1 and 3 ranks: round-off.
- `openPMD_check_h5` on both kinds of file (by hand, docs/STATUS.md).

## References
- openPMD standard 1.1.0 and the ED-PIC extension, https://github.com/openPMD/openPMD-standard
- A. Huebl et al., openPMD 1.1.0, Zenodo 10.5281/zenodo.1167843.
