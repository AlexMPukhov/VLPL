# GPU build: what is verified and what to run on real hardware

Status (2026-10-26): the CUDA build (serial and MPI) compiles with nvcc 12.0 for sm_80 without
warnings. The CPU and MPI builds use the same kernels (the portability layer turns them into OpenMP
loops) and pass all tests. **First real GPU run** on one RTX 5090: all tests pass and the
streaming-plasma deck agrees with the CPU build to round-off (section "First run on the RTX 5090"
below). Not yet run: several GPUs with MPI.

## Design

- `src/portable.hpp`: `par_for` (CUDA kernel or OpenMP loop), `Buf<T>` device memory, atomics,
  `par_sum` / `par_max_abs` reductions. All kernels are `[=] HD` lambdas capturing plain views
  (`FV` for fields, raw pointers for particles).
- Fields and particles live on the device the whole run. The host sees them only for setup, the
  slices (one z-plane per output) and the tests.
- Guard exchange (`Comm::halo`): per axis, one kernel packs the guard layers of all fields of the call
  (both directions) into one device buffer; both directions are in flight at once (MPI_Irecv / Isend),
  through pinned host buffers (one copy each way per direction), or straight from device memory with a
  CUDA-aware MPI (`VLPL_CUDA_AWARE=1` in the environment); one kernel unpacks. With a periodic axis that
  is not decomposed the copy stays on the device.
- Particle migration: a kernel lists the leaving particles (atomic counter) into a persistent device
  buffer; only those go to the host, the holes are refilled from the tail on the device, the MPI
  exchange runs on the host along x, y, z (both directions at once, `Comm::exchange_bytes`), and the
  arrivals go up in one copy and are unpacked by one kernel per attribute. With one rank and all axes
  periodic the particles are wrapped in place.
- At the end of a run rank 0 prints `exchanges (max over ranks): guard cells <s> in <n> calls (MPI wait
  <s>), particle messages <s>; <mode>`: the slowest rank's time in the guard exchanges (pack, copies,
  MPI, unpack), the part waiting for MPI, and the particle messages.
- One GPU per rank: local rank (MPI_COMM_TYPE_SHARED) modulo the number of visible devices.

## To run

Single GPU (e.g. the GPU workstation, RTX 5090, CUDA 13.0, no MPI):

```
make CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu
build-gpu/tests                                     # must print "all passed"
build-gpu/vlpl examples/laser_reflection.deck output.dir=gpu_reflect
python3 tools/movie.py gpu_reflect                  # compare with out_reflect/movie.mp4
```

Several GPUs (e.g. JUWELS Booster sm_80, JUPITER GH200 sm_90; 4 GPUs per node):

```
make CUDA=1 MPI=1 CUDA_ARCH=-arch=sm_90 BUILD=build-gpu-mpi
srun -n 4 --gpus-per-task=1 build-gpu-mpi/vlpl examples/laser_reflection.deck grid.mpi_x=2 grid.mpi_y=2
sh tests/mpi.sh build-gpu-mpi/vlpl                # with MPIRUN="srun" if needed
```

Expected: the same diag.txt as the CPU run up to round-off (the order of the atomic additions
differs), max|dGauss| and max|divB| ~1e-14 until particles reach an open boundary.

On JUPITER / JUWELS Booster this, with the comparison against the CPU build and the timings, is the
Slurm package `bench/jupiter/` (its README is the guide; section "Script package for JUPITER" below).

## Known performance limits (first version)

- Current deposition with double-precision atomics, no particle sorting.
- Four guard exchanges per step, through pinned host memory unless `VLPL_CUDA_AWARE=1` (not
  overlapped with the push yet).
- Reductions for diagnostics use one global atomic; fine at `output.every` >~ 10.

## First run on the RTX 5090

2026-10-26, commit 1a00563 (main), the GPU workstation: NVIDIA GeForce RTX 5090 (cc 12.0, 32 GB),
driver 580.95.05, CUDA 13.0 (nvcc V13.0.88), host compiler g++ 14.1.0, 20 CPU cores, no MPI.
Non-interactive jobs there need
`LD_LIBRARY_PATH=/opt/isl-0.24/lib:/opt/mpc-1.2.1/lib:/opt/mpfr-4.2.1/lib:/opt/gmp-6.2.1/lib:/opt/cuda-13.0/lib64:/opt/gcc-14.1.0/lib64`
and `/opt/gcc-14.1.0/bin` first in `PATH` (otherwise g++'s cc1plus fails without a message).

Builds (both without warnings):

```
make -j BUILD=build-cpu                                                       # 5 s
make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu   # 13 s
```

### Tests

`build-cpu/tests` (OpenMP, 20 threads) and `build-gpu/tests`: **all 28 pass on both, 0 failures**
(CPU 5.5 s, GPU 2.4 s wall time). The measured numbers:

| test | CPU | GPU |
|---|---|---|
| vacuum: +x wave (Ey,Bz) shifted exactly 1 cell/step | max err 0.0e+00 | 0.0e+00 |
| vacuum: -x wave (Ez,By) shifted exactly 1 cell/step | max err 0.0e+00 | 0.0e+00 |
| vacuum random fields: energy bounded over 2000 steps | max W/W0 = 1.0469 | 1.0469 |
| vacuum random fields: div B unchanged | max change 8.8e-14 | 8.4e-14 |
| Boris: gyration, \|u\| conserved and phase exact | dgamma 2.3e-14, phase err 1.5e-13 | same |
| Boris: uniform E, u = q E t | err -1.6e-13 | -1.6e-13 |
| Higuera-Cary: gyration | dgamma 1.4e-14, phase err 3.6e-14 | same |
| Higuera-Cary: uniform E, u = q E t | err -1.6e-13 | -1.6e-13 |
| Higuera-Cary: E x B drift velocity E/B | v_d = 0.50003 (0.5) | 0.50003 |
| plasma shape 1, neutral, drift -3: Gauss law | max\|dGauss\| 5.5e-15 | 5.1e-15 |
| plasma shape 1, neutral, drift -3: div B = 0 | 1.8e-16 | 2.1e-16 |
| plasma shape 1, electron slab: Gauss law | 8.3e-16 | 1.8e-15 |
| plasma shape 1, electron slab: div B = 0 | 1.4e-16 | 1.2e-16 |
| plasma shape 2, neutral, drift -3: Gauss law | 1.1e-14 | 9.9e-15 |
| plasma shape 2, neutral, drift -3: div B = 0 | 2.2e-16 | 1.5e-16 |
| plasma shape 2, electron slab: Gauss law | 9.4e-16 | 8.9e-16 |
| plasma shape 2, electron slab: div B = 0 | 1.0e-16 | 9.6e-17 |
| Langmuir oscillation frequency | w = 2.0086 (expected 2.0136) | 2.0086 |
| staggering: J_a = rho + e_a/2 matches grid.hpp | ok | ok |
| continuity, shape 1 | max err 1.4e-13 (\|drho/dt\| ~ 1.3e+02) | 2.2e-13 |
| continuity, shape 2 | max err 3.0e-13 (\|drho/dt\| ~ 1e+02) | 3.4e-13 |
| registry: require idempotent, init values, type change rejected | ok | ok |
| registry: attributes survive migration (wrap and open x) | 2116 left through x, 0 mismatches | same |
| pipeline: default operators | push_deposit -> rip_cc -> migrate | same |
| pipeline: stage order, insertion order, setup once | ok | ok |
| pipeline: boundary hook once per step at t_n | 3 calls | 3 calls |
| geometry Cart3D: gather of linear fields exact, shape 1 | max err 7.1e-15 | 7.1e-15 |
| geometry Cart3D: gather of linear fields exact, shape 2 | max err 1.1e-14 | 1.1e-14 |

### CPU vs GPU: streaming plasma

`examples/streaming_plasma.deck` (64 x 32 x 32 cells, 2 x 524288 particles, u = -10, Higuera-Cary,
shape 1), run with both binaries. Relative differences |GPU - CPU| / |CPU| of the diag.txt columns:

| run | rows | W_field | W_kin | W_total | max\|dGauss\| CPU / GPU | max\|divB\| CPU / GPU | N_e, N_p |
|---|---|---|---|---|---|---|---|
| `grid.t_end=200fs output.every=20` (60 steps) | 4 | <= 4.1e-13 | 1.1e-11 at step 0, 0 after | 1.1e-11 at step 0, 0 after | 2.5e-16 / 3.2e-16 | 1.2e-21 / 1.1e-21 | 524288, 524288 equal in all rows |
| full deck, 1.5 ps (448 steps, every 10) | 46 | <= 1.1e-12 | 1.1e-11 at step 0, <= 4e-13 after | same as W_kin | 8.0e-16 / 8.5e-16 | 7.8e-21 / 7.7e-21 | equal in all rows |

The 200 fs rows (W_field, W_kin, W_total):

```
step  W_field CPU / GPU                          W_kin CPU / GPU
0     0 / 0                                      2.425858672173e+10 / 2.425858672146e+10
20    7.354997174429e-04 / 7.354997174426e-04    2.425858672173e+10 / 2.425858672173e+10
40    2.769776888454e-03 / 2.769776888453e-03    2.425858672173e+10 / 2.425858672173e+10
60    5.652361466878e-03 / 5.652361466878e-03    2.425858672173e+10 / 2.425858672173e+10
```

The numerical Cerenkov field energy grows identically: W_field(1.5 ps) = 3.567823030287e-02 on both.
The W_kin values after step 0 agree to all 13 printed digits (4e-13 is the print resolution).
The Gauss-law drift stays at round-off (< 1e-15 against max|rho| ~ 1e-5) on the GPU.

**Step-0 W_kin offset of 1.1e-11:** this comes from how the diagnostic sum is rounded, not from a
physics difference. `par_sum` on the GPU adds all particles to one double with `atomicAdd`. That
is one long chain of additions, while the CPU adds 20 OpenMP partial sums. At t = 0 every proton has
exactly u = -10, so all terms are identical and every addition to the growing accumulator rounds the
same way. Python check with the deck's terms: one sequential chain gives -1.11e-11, 20 chunks give
-1.7e-13. Once the momenta spread (from step 20 on), the rounding errors cancel and the columns agree.
A tree reduction in `par_sum` (block-wise partial sums) would remove this; it is not needed for
correctness.

### Timings (indicative, not a benchmark)

Full streaming-plasma deck, 448 steps, 1.05 M particles: GPU 0.6 s of time stepping (2.2 s wall
time with CUDA start-up) vs CPU 14.3 s on 20 OpenMP threads, a factor of about 24. 200 fs: GPU 0.1 s,
CPU 2.0 s.

Logs and outputs: `$WORK/rip-pic/{logs,cpu_stream,gpu_stream,cpu_stream_full,gpu_stream_full}`
on the CPU cluster (GPU workstation: `$WORK/...`).

## Next GPU check (handover for the agent on the CPU cluster / the GPU workstation)

Done on 2026-10-26: all pass, results in "Second check on the RTX 5090" below.

Current main since the first run adds the tile sort (`src/tiles.*`: tile keys and the permutation of
all particle attributes run as device kernels), the open-edge fixes of the diagnostics and the
regression harness. On cuda (environment as in "First run on the RTX 5090"):

```
git pull                                    # main
make -j BUILD=build-cpu && make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu
build-cpu/tests && build-gpu/tests          # all must pass (32 tests)
sh tests/regression/run.sh build-gpu/vlpl # GPU against the CPU references: "all cases match"
REGRESSION_OUT=regression_cpu sh tests/regression/run.sh build-cpu/vlpl
```

Report: the test lines that differ between CPU and GPU, the regression result per case with its time
(CPU vs GPU), and, if a case fails, the first rows that `tests/regression/compare.py` prints. Results go
into a new section here and into `docs/STATUS.md`, on a branch with a pull request.

## Second check on the RTX 5090: tile sort and regression harness

2026-10-26, main c7310e0, same machine and environment as the first run (python 3.6.8 for
`compare.py`). Both builds compile without warnings (CPU 5 s, GPU 14 s).

**Tests: 32/32 pass on CPU and GPU** (CPU 7.2 s, GPU 2.7 s wall time). The four new tests give the
same numbers on both:

| test | CPU | GPU |
|---|---|---|
| pipeline: default operators | push_deposit -> rip_cc -> migrate -> tile_sort | same |
| tiles: sorted ranges, attributes kept, stable order | 27 tiles, 0 bad | 27 tiles, 0 bad |
| tiles: sort_every = 0 removes the tile_sort operator | ok | ok |
| open x, pol z: pulse leaves, div B, Gauss at round-off | max\|divB\| 2.3e-15, max\|dGauss\| 5.6e-17, W_end/W_max 5.6e-04 | 2.6e-15, 5.6e-17, 5.6e-04 |
| open x, pol y: pulse leaves, div B, Gauss at round-off | max\|divB\| 5.6e-17, max\|dGauss\| 1.9e-15, W_end/W_max 5.6e-04 | 5.6e-17, 2.1e-15, 5.6e-04 |

The only CPU/GPU differences are in the round-off values (last digit): vacuum div B change 8.8e-14 / 8.4e-14;
Gauss law shape 1 neutral 6.0e-15 / 6.6e-15, slab 6.7e-16 / 1.6e-15; shape 2 neutral 1.1e-14 /
9.5e-15, slab 7.2e-16 / 8.9e-16; div B 2.0e-16 / 2.3e-16, 1.7e-16 / 1.6e-16, 1.9e-16 / 1.7e-16,
1.2e-16 / 8.8e-17; continuity 1.7e-13 / 2.0e-13 (shape 1), 3.2e-13 / 3.1e-13 (shape 2). All other
lines are identical.

**Regression: all cases match the references, on the GPU and on the CPU**; the self-test (a 1e-6
change of W_field) is detected on both. Times are for the time stepping ("done:" line of the run),
in brackets the wall time from `run.sh`, which includes the CUDA start-up:

| case | steps | CPU (20 threads) | GPU | GPU - CPU: W_field, W_kin, W_total | N equal | max\|dGauss\| CPU / GPU |
|---|---|---|---|---|---|---|
| stream | 179 | 1.3 s (1 s) | 0.2 s (2 s) | 1.4e-12, 6.6e-13, 6.6e-13 | yes | 3.1e-16 / 2.6e-16 |
| stream_s2 | 90 | 2.5 s (3 s) | 0.1 s (2 s) | 2.0e-12, 6.6e-13, 6.6e-13 | yes | 3.6e-16 / 3.3e-16 |
| reflect | 225 | 6.2 s (6 s) | 0.3 s (2 s) | 0, 1.6e-12, 0 | yes | 2.6e-14 / 2.4e-14 |

The differences are the largest relative deviation over all rows, with W_field measured against 1e-12 of its
maximum (as in `compare.py`; the field energy starts at 0). The GPU W_field differs from the stored
references by at most 2.0e-12. The tile sort, which runs as device kernels, works on the GPU: the
tile test passes, and the decks agree with the CPU to round-off with sorting on (the default).

Logs and outputs: `$WORK/rip-pic/{logs2,regression_cpu,regression_gpu}` on the CPU cluster.

## Third check on the RTX 5090: tiled CPU deposit (main 16bb4fc)

2026-10-26, main 16bb4fc (PR #12, coloured-tile CPU deposit), same machine and environment as the
first run. Both builds compile without warnings (CPU 6 s, GPU 15 s).

**Tests: 37/37 pass on CPU and GPU.** The five new tiled-deposit tests pass on both:

| test | CPU | GPU |
|---|---|---|
| colouring (each tile once, no neighbours in a colour), safety | 36 tiles | 36 tiles |
| tiled = global, shape 1, Gauss at round-off | max rel dJx 4.1e-15, dW 5.6e-16, max\|dGauss\| 7.4e-15 | 5.3e-15, 0, 6.2e-15 |
| tiled = global, shape 2, Gauss at round-off | max rel dJx 7.9e-15, dW 4.3e-16, max\|dGauss\| 9.4e-15 | 7.0e-15, 3.3e-15, 9.9e-15 |
| 1 and 3 threads bit-identical | dJx 0, dEx 0 | dJx 0, dEx 0 |
| sort_every = 3, strays deposited atomically | 794287 strays, max\|dGauss\| 7.0e-15 | 794287 strays, 6.4e-15 |

The other lines differ between CPU and GPU only in the round-off values (1e-13 to 1e-17), as in the
second check.

**Regression: all cases match the references on the GPU and on the CPU**, and the self-test passes on
both. Time stepping ("done:" line):

| case | steps | CPU (20 threads, deposit = auto = tiled) | GPU |
|---|---|---|---|
| stream | 179 | 5.1 s (second check, global: 1.3 s) | 0.2 s |
| stream_s2 | 90 | 4.6 s (2.5 s) | 0.1 s |
| reflect | 225 | 7.4 s (6.2 s) | 0.3 s |

### Problem 1: the tiled tests take 169 s in the GPU build

`build-gpu/tests` now takes 2 min 51 s, against 2.7 s in the second check (`build-cpu/tests`: 16 s).
The machine was idle. The host CPU time was 2 min 49 s, so the time goes into host work. Time per test line in the GPU build vs the CPU build (all other lines take < 1 s on the GPU):

| test | GPU build | CPU build |
|---|---|---|
| tiled = global, shape 1 | 19 s | 1 s |
| tiled = global, shape 2 | 44 s | 3 s |
| 1 and 3 threads bit-identical | 88 s | 2 s |
| sort_every = 3, strays | 18 s | 1 s |

These tests force the tiled path, which the GPU build otherwise never uses (`auto` = global there).
The fast tests should stay under 1 min (CLAUDE.md), and the thread-count test has no meaning in a
GPU build. Suggestion: skip these four tests in the CUDA build, or shrink them there, until the
shared-memory GPU deposit exists.

### Problem 2: tiled deposit slower than global with many threads on small grids

`streaming_plasma.deck` at regression size (32 x 16 x 16 cells, 179 steps, default `tile = 8 8 8`),
time stepping on the 20-core machine:

| threads | global | tiled |
|---|---|---|
| 1 | 4.5 s | 3.3 s (-27 %) |
| 4 | 1.6 s | 1.8 s (+13 %) |
| 20 | 1.4 s | 5.7 s (4x slower) |

With 8^3 tiles this grid has 4 x 2 x 2 = 16 tiles, so 2 tiles per colour: at most 2 of the 20
threads work while the others wait, and the strays pass runs serially after them. On the full deck (64 x 32 x 32, 128 tiles,
16 per colour, 200 fs) tiled still wins: 1 thread 11.4 -> 8.2 s (-28 %), 20 threads 2.8 -> 2.4 s
(-14 %); GPU 0.3 s. Suggestion: let `deposit = auto` choose tiled only when every colour has at least
about as many tiles as there are threads (or fall back to global otherwise).

Logs and outputs: `$WORK/rip-pic/logs3` (ttime_*.log: time per test line; small.log:
deposit timings) and `regression3_{cpu,gpu}` on the CPU cluster.

## Next GPU check: box deposit (handover for the agent on the CPU cluster / the GPU workstation)

Done on 2026-10-26 (branch tip 88be38c): all pass, box is not faster than global; results in
"Fourth check on the RTX 5090" below.

New since the second check: `par_blocks` (`src/portable.hpp`, one thread block per tile with a scratch
array in shared memory, opt-in above 48 KB) and `numerics.deposit = box` (`docs/specs/tile_deposit.md`,
decision 19). It is the first kernel that uses shared memory and `__syncthreads`; on the CPU it has
only run through the fallback. Please check on cuda (environment as before):

```
git pull                                     # main (or the branch of the pull request)
make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu && make -j BUILD=build-cpu
build-gpu/tests && build-cpu/tests           # all must pass; the "reach" and "box deposit" lines run the GPU kernel
printf '#!/bin/sh\nexec build-gpu/vlpl "$@" numerics.deposit=box\n' > gpu_box.sh && chmod +x gpu_box.sh
sh tests/regression/run.sh ./gpu_box.sh      # box deposit on the GPU against the CPU references
for dep in global box; do for t in "6 6 6" "8 8 8"; do
  build-gpu/vlpl examples/streaming_plasma.deck numerics.deposit=$dep "numerics.tile=$t" output.dir=gpu_$dep | grep done
done; done                                   # full deck, 448 steps: time per mode and tile size
```

Report: test lines that differ between CPU and GPU; regression per case; the four timings (and
`nvidia-smi` / compute-sanitizer output if a kernel fails: `compute-sanitizer --tool memcheck` on
`build-gpu/tests`). If box is faster than global, say by how much: the GPU default of
`numerics.deposit = auto` switches to box only after this.

## Fourth check on the RTX 5090: box deposit (branch gpu-box-deposit, 88be38c)

2026-10-26, branch `gpu-box-deposit` at 88be38c (box deposit, merged with main's thread-aware
`deposit = auto`), same machine and environment as the first run. Both builds compile without
warnings (GPU 17 s, CPU 6 s). The first run of this check, on the branch's first commit 03485b5,
gave the same box results; the numbers below are from 88be38c.

**Tests: all pass on GPU and CPU** (`build-gpu/tests` 4.2 s, the tiled-deposit tests are skipped
there: "skipped (GPU build)"; `build-cpu/tests` 21 s). The GPU used to need 172 s with the tiled tests,
so problem 1 of the third check is fixed. The "reach" and "box deposit" lines, which run the
shared-memory kernel on the GPU:

| test | CPU (fallback) | GPU (par_blocks kernel) |
|---|---|---|
| reach of the deposit, shape 1 | x [-2,2] y [-1,2] z [-1,2] | same |
| reach of the deposit, shape 2 | x [-2,2] y [-2,3] z [-2,3] | same |
| box = global, shape 1 | max rel dJx 6.1e-15, dW 3.7e-16, max\|dGauss\| 6.1e-15 | 5.0e-15, 3.3e-15, 6.2e-15 |
| box = global, shape 1, partial tiles | 5.3e-15, 1.3e-16, 4.7e-15 | 5.8e-15, 0, 5.4e-15 |
| box = global, shape 2 | 5.1e-15, 1.2e-16, 7.5e-15 | 5.2e-15, 9.7e-16, 6.4e-15 |
| box = global, shape 2, partial tiles | 6.2e-15, 4.0e-16, 6.6e-15 | 6.1e-15, 2.7e-15, 6.2e-15 |
| box, sort_every = 3, strays deposited globally | 834542 strays, max\|dGauss\| 6.3e-15 | 834542 strays, 6.4e-15 |

All other lines differ between CPU and GPU only in round-off digits, as in the earlier checks.

**compute-sanitizer** (CUDA 13.0) on `build-gpu/vlpl` with `deposit = box`, `tile = 6 6 6`,
streaming plasma on 32 x 16 x 16 cells, 18 steps, shapes 1 and 2: memcheck 0 errors, racecheck
0 hazards (shared memory), synccheck 0 errors (`__syncthreads`). memcheck on the whole
`build-gpu/tests` of 03485b5 did not finish in 30 min (the tiled tests then still ran on the GPU
build), so the sanitizers were run on the deck instead.

**Regression: all cases match the references** with the box deposit on the GPU (`./gpu_box.sh`),
with the default GPU build (global) and on the CPU; the self-test passes on all three. Time stepping:

| case | steps | GPU box | GPU global | CPU (20 threads, auto) |
|---|---|---|---|---|
| stream | 179 | 0.5 s | 0.2 s | 1.3 s |
| stream_s2 | 90 | 0.4 s | 0.1 s | 2.6 s |
| reflect | 225 | 0.4 s | 0.3 s | 7.2 s |

### Timings: box is slower than global on the RTX 5090

Full `streaming_plasma.deck` (64 x 32 x 32 cells, 1.05 M particles, 448 steps), time stepping:

| GPU | tile 6 6 6 | tile 8 8 8 |
|---|---|---|
| deposit = global | 2.5 s | 2.6 s |
| deposit = box | 3.4 s | 3.4 s |

So **box is 30-36 % slower than global**, and the tile size does not matter. Breakdown with the tile sort
frequency (tile 8 8 8):

| GPU | sort every step | sort every 10 steps | no sort |
|---|---|---|---|
| global | 2.5 s | 0.8 s | 0.6 s |
| box | 3.4 s | 1.7 s | (needs sorting) |

- The **tile sort dominates**: 1.9 of the 2.5 s with global, about 4 ms per step, because it runs on
  the host. Without it the global step takes 0.6 s, as in the first run.
- The box deposit itself costs about 0.9 s more than the atomic deposit over 448 steps (1.7 vs 0.8 s
  at sort every 10 steps), about 2 ms per step. On this card, double-precision atomics into global
  memory are cheaper than the box kernel in its current form.
- Suggestion: keep `deposit = auto` = global on the GPU. Move the tile sort to the device first: with
  the host sort, any deposit that needs sorted particles loses. Profile the box kernel before
  switching (occupancy with the shared-memory scratch, the number of particles per block). `nsys` 2025.3
  is installed on the GPU workstation for that.
- The RTX 5090 has 1/64 FP64 throughput. On the GH200 (JUPITER) the balance between atomics and the
  shared-memory kernel can differ, so time it there before deciding.

### Fix of the third check's problem 2 (CPU, 20 cores)

Regression-size streaming plasma (32 x 16 x 16, 179 steps), `deposit = auto`: 1 thread chooses tiled
(3.1 s), 4 and 20 threads choose global (1.6 s, 1.4 s; tiled took 5.7 s at 20 threads before).

Logs and outputs: `$WORK/rip-pic-box/{logs4,regression4_box,regression4_gpu,regression4_cpu}`
on the CPU cluster (`logs`: the first run on 03485b5 and the 30-min memcheck).

## Next GPU check: why is the box deposit slower? (handover for the agent on the CPU cluster / the GPU workstation)

Done on 2026-10-26 (branch tip 7f97fca), without ncu (no counter access on the GPU workstation); results in
"Fifth check on the RTX 5090" below.

Current state (branch `gpu-box-deposit` after the merge of main and `sort-auto`): `numerics.sort_every =
auto` no longer sorts with the global deposit (decision 20), so the GPU default (global) should now take
about 0.6 s on the full streaming deck instead of 2.5 s. `deposit = box` stays opt-in. Its kernel cost
~2 ms more per step than the atomics (fourth check: 1.7 vs 0.8 s with sorting every 10 steps). On the GPU workstation:

```
make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu && build-gpu/tests
D=examples/streaming_plasma.deck
build-gpu/vlpl $D output.dir=g_auto | grep -E "deposit|done"              # expect global, ~0.6 s
for t in "4 4 4" "6 6 6" "8 8 8"; do build-gpu/vlpl $D numerics.deposit=box "numerics.tile=$t" output.dir=g_box | grep done; done
ncu --set full -k regex:rip_block_kernel -c 3 -o box_prof build-gpu/vlpl $D numerics.deposit=box "numerics.tile=6 6 6" grid.t_end=50fs output.dir=g_prof
ncu --set full -k regex:rip_kernel -c 20 -o glob_prof build-gpu/vlpl $D numerics.deposit=global grid.t_end=50fs output.dir=g_prof2
```

Report from `ncu`: for the box kernel the achieved occupancy and its limiter (shared memory per block,
registers), the time in shared-memory atomics (are FP64 shared atomics native on sm_120 or a CAS
loop: `atom.shared.add.f64` vs `atom.shared.cas.b64` in the SASS, `cuobjdump -sass build-gpu/vlpl`),
the zeroing and flush share of the kernel, and particles per block; for the global push kernel the
same occupancy numbers and the global-atomic throughput. With these I decide between: smaller tiles
(less shared memory, more blocks), per-warp sub-boxes, a float-free reduction order, or dropping box
on this card and timing it on the GH200 (JUPITER), where FP64 is not throttled.

## Fifth check on the RTX 5090: why the box deposit is slower (branch gpu-box-deposit, 7f97fca)

2026-10-26, branch `gpu-box-deposit` at 7f97fca (with `sort_every = auto`, decision 20), same
machine and environment as before.

**`ncu` cannot read the counters on cuda.** It stops with `ERR_NVGPUCTRPERM`. The driver has
`RmProfilingAdminOnly: 1`, and sudo needs a password. Enabling it needs an admin of the GPU workstation:
the driver option `NVreg_RestrictProfilingToAdminUsers=0` (modprobe.d) and a driver reload.
Without the counters, this check uses the following instead:
- `nsys` (CUDA trace: time per kernel and per API call);
- `cuobjdump -sass` (which atomic instructions are emitted);
- `cuobjdump -res-usage` with the device limits (theoretical occupancy).

Achieved occupancy, the zeroing and flush share and the global-atomic throughput need `ncu` and are
estimated below, not measured.

### Tests and timings

`build-gpu/tests`: all pass (4.1 s). Full `streaming_plasma.deck`, 448 steps, time stepping:

| GPU | time |
|---|---|
| default (`auto`: global deposit, no sorting) | **0.6 s** (was 2.5 s; decision 20 works as expected) |
| global, `sort_every = 1` | 2.5 s |
| box, tile 4 4 4 | 3.2 s |
| box, tile 6 6 6 | 3.4 s |
| box, tile 8 8 8 | 3.4 s |

### Where the time goes (nsys, 60 steps, 2 species, i.e. 120 calls per kernel)

| per call (µs) | global | box T=4 | box T=6 | box T=8 |
|---|---|---|---|---|
| push + deposit kernel (`rip_kernel push_impl` / `rip_block_kernel push_box_impl`) | 369 | 555 | 616 | 629 |
| stray count `par_sum(n, S[p])` | — | 656 | 656 | 656 |
| all GPU kernels, 60 steps | 62 ms | 166 ms | 173 ms | 174 ms |
| wall time, 60 steps | 0.1 s | 0.4 s | 0.4 s | 0.5 s |
| `cudaMalloc` + `cudaFree` calls (time) | 108 (2.3 ms) | 3024 (61 ms) | 3024 (62 ms) | 3024 (61 ms) |

Per step on the box path (T = 6), against 0.74 ms for the push and deposit kernels of the global path:

| step part | time | note |
|---|---|---|
| box kernel | 1.23 ms | 1.5–1.7× the global push kernel |
| stray `par_sum` | 1.31 ms | more than the box kernel itself |
| `Buf` allocations | ~1 ms | `ts` and `stray`, allocated and freed on every call |
| host tile sort and syncs | ~4 ms | the rest of the wall time |

Notes:
- `par_sum` adds every particle's flag into one global double with `atomicAdd` (`portable.hpp`). The
  1.05 M atomics all go to the same address and are executed one after another: 656 µs, independent of the tile size. The same
  single-address reduction costs 658 µs in the kinetic-energy diagnostic. A block-wise reduction
  (or counting the strays with an atomic per block into an int) would remove almost all of it.
- `Buf<int> ts` and `Buf<int> stray` (N ints) are allocated and freed on every call, with a
  memset and a host-to-device copy of `tile_start`. They could be kept in the species, like the
  tile_start ranges.
- The host tile sort takes about 4 ms per step, as in the fourth check, so any deposit that needs
  sorted particles every step loses on the GPU until the sort runs on the device.

### FP64 atomics in shared memory are a CAS loop on sm_120

`cuobjdump -sass` of minimal test kernels (nvcc 13.0, `logs5/atom2.cu`), the same for `-arch=sm_120`
(RTX 5090) and `-arch=sm_90` (GH200):

| `atomicAdd` on | SASS (sm_120 and sm_90) | |
|---|---|---|
| `double` in shared memory | `ATOMS.CAST.SPIN.64` | compare-and-swap loop |
| `float` in shared memory | `ATOMS.CAST.SPIN` | compare-and-swap loop |
| `unsigned long long` in shared memory | `ATOMS.CAST.SPIN.64` | compare-and-swap loop |
| `int` / `unsigned` in shared memory | `ATOMS.ADD` | native |
| `double` in global memory | `REDG.E.ADD.F64.RN.STRONG.GPU` | native L2 reduction |

So the FP64 add in shared memory is a compare-and-swap retry loop, on this card and on the GH200,
while the add in global memory is a native L2 reduction. In shared memory only the 32-bit integer add
is native. In the vlpl binary:

| kernel | shared atomics (`ATOMS.CAST.SPIN.64`) | global atomics (`REDG.E.ADD.F64`) |
|---|---|---|
| `rip_block_kernel` box, shape 1 | 54 | 55 (strays, flush) |
| `rip_block_kernel` box, shape 2 | 144 | 37 |
| `rip_kernel` push (global), shape 1 | 0 | 54 |
| `rip_kernel` push (global), shape 2 | 0 | 36 |

(These are static instruction counts in the SASS.) With 256 threads of a block working on neighbouring particles of one tile,
many CAS loops hit the same few shared addresses and retry. That explains why the box kernel is
slower than the global atomics even though it saves global traffic.

### Occupancy (theoretical, from `-res-usage` and the device limits)

RTX 5090: 170 SMs, 65536 registers, 100 KB shared memory (+ 1 KB reserved per block), 1536 threads and
24 blocks per SM; both kernels use 256 threads per block.

| kernel (shape 1 deck) | registers | shared memory per block | blocks per SM: registers / shared / threads | occupancy | limiter |
|---|---|---|---|---|---|
| global push, shape 1 | 80 | — | 3 / — / 6 | 3 blocks = 50 % | registers |
| box T = 4 (2048 blocks, 512 particles per species and block) | 80 | 17.5 KB | 3 / 5 / 6 | 50 % | registers |
| box T = 6 (396 blocks, 1728 per block) | 80 | 31.9 KB | 3 / 3 / 6 | 50 % | registers and shared |
| box T = 8 (512 blocks, 4096 per block) | 80 | 52.7 KB | 3 / 1 / 6 | 1 block = 17 % | shared memory |
| global push, shape 2 | 254 | — | 1 / — / 6 | 17 % | registers |
| box, shape 2 | 168 (+224 B stack) | as above | 1 / ... | ≤ 17 % | registers |

Occupancy alone does not explain the slow-down: T = 4 and T = 6 have the same 50 % as the global
kernel and are still 1.5–1.7× slower. T = 8 drops to 17 % but is only 2 % slower than T = 6.

Zeroing and flush, estimated: per block they touch 3 (T+5)^3 doubles, 2.2 k (T = 4), 4.0 k (T = 6)
and 6.6 k (T = 8). The particles of the block do roughly 512 / 1728 / 4096 × (the 54 shared-atomic
instructions of the shape-1 deposit) CAS loops. So zeroing and flush are at most ~10 % (T = 4) and
a few % (T = 6, 8) of the shared-memory operations: not the bottleneck.

### What this suggests (decision is the developer's)

1. Fix the overheads first (independent of box vs global): a block-wise `par_sum` instead of one
   address; no `Buf` allocation per call. That saves about 2.3 ms per step on the box path. It also
   speeds up the kinetic-energy diagnostic (658 µs per call).
2. The box kernel itself is slower because FP64 shared atomics are CAS loops, on sm_120 and on sm_90.
   FP32 does not help: the shared FP32 add is a CAS loop too. Options:
   - per-warp sub-boxes or a warp-level pre-reduction (fewer threads per address, fewer retries);
   - 32-bit integer fixed-point accumulation in shared memory (the only native `ATOMS.ADD`). This needs
     a scale per tile and care with overflow and charge conservation; it is a physics decision;
   - keep global atomics on the GPU. The same CAS loop is emitted for sm_90, so the box kernel has the
     same handicap on the GH200. There the global FP64 atomics and FP64 arithmetic are faster, so a
     GH200 timing is still needed before deciding.
3. Smaller tiles do not help much: T = 4 is 10 % faster than T = 6 in the kernel, and the per-call
   overheads do not change.

Logs and outputs: `$WORK/rip-pic-box/logs5` on the CPU cluster (nsys reports `nsys_*.nsys-rep`, CSV
summaries `stats_*`, `sass.txt`, `resusage.txt`, the atomic test kernels `atom2.cu`, the device query `devq.cu`).

## Next GPU check (sixth): timings after the fixes of the fifth check (handover for the agent on the CPU cluster / the GPU workstation)

Branch `gpu-box-deposit` after "Fifth check" fixes 1 and 2:
- `par_sum` / `par_max_abs` on the GPU are a block-wise tree reduction (256 threads, at most 1024
  blocks, one partial per block, combined on the host in block order): no atomics on one address,
  no allocation per call, and the result is independent of the scheduling. New test "reductions"
  (exact sums, sizes 1 .. 3e6).
- The tiled and box deposits keep their device scratch (tile ranges, colour order, stray list and
  counter) between calls, and count strays with an int atomic touched by strays only (no reduction
  over all particles); the tiled path deposits the listed strays only.

On cuda (environment as before):
```
git checkout main && git pull            # gpu-box-deposit was merged (PR #17)
make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu && build-gpu/tests
sh tests/regression/run.sh build-gpu/vlpl
D=examples/streaming_plasma.deck
build-gpu/vlpl $D output.dir=g_auto | grep -E "done|steps"                          # global, no sort
for T in 4 6 8; do build-gpu/vlpl $D numerics.deposit=box "numerics.tile=$T $T $T" output.dir=g_box$T | grep -E "done|steps"; done
nsys profile -o box6 build-gpu/vlpl $D numerics.deposit=box "numerics.tile=6 6 6" grid.t_end=0.1ps output.dir=g_n
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum box6.nsys-rep
```
Report: all tests and regression cases; the times (fifth check: 0.6 s global, 3.2 / 3.4 / 3.4 s box);
per call of the box kernel, the reductions and the `cudaMalloc` count. Expected: the stray `par_sum`
(656 us) and the per-call allocations (~1 ms per step) are gone, so box T = 6 drops by about 2.3 ms per
step; the box kernel itself stays slower than global (CAS-loop shared FP64 atomics, fifth check).
Results into a new section here, on a branch from main with a pull request.

## Next GPU check: Phase 1 core (handover for the agent on the CPU cluster / the GPU workstation)

Main after PR #21 (it contains #18 open-boundary currents, #19 moving window, #20 density profiles,
#21 beam self-field and the FBPIC benchmark #22). New device code: the moving
window shift (`shift_fields_x`, one copy kernel per field), the frozen entry columns in the push
kernel (`Domain::xfreeze`), and the kept particles below the open left edge (migration). The
density profiles and the beam self-field solve are host code at loading. On cuda (environment as
in "First run on the RTX 5090"):

```
git checkout main && git pull
make -j BUILD=build-cpu && make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu
build-cpu/tests && build-gpu/tests          # all must pass (75 tests on the CPU; the tiled ones are skipped on the GPU)
sh tests/regression/run.sh build-gpu/vlpl # GPU against the CPU references (5 cases, new: lwfa, beam)
REGRESSION_OUT=regression_cpu sh tests/regression/run.sh build-cpu/vlpl
# the full examples (CPU 2 cores: lwfa_window 14 min): time and max|dGauss| (column 6 of diag.txt)
for d in lwfa_window lwfa_beam beam_vacuum; do
  s=$(date +%s); build-gpu/vlpl examples/$d.deck output.dir=g_$d output.slice_every=0 > /dev/null; echo "$d $(( $(date +%s) - s )) s"
  awk '!/^#/{print $6}' g_$d/diag.txt | sort -g | tail -1
done
```

Report: the test lines that differ between CPU and GPU, the regression result per case with its time,
and for the three examples the GPU time and the largest max|dGauss| (expected <= 3e-14). Results go
into a new section here, on a branch from main with a pull request.

## Sixth check on the RTX 5090: timings after the fixes of the fifth check (main 260b1dc)

Done on 2026-10-27 on the GPU workstation (RTX 5090, CUDA 13.0, `-arch=sm_120`), main 260b1dc (contains
gpu-box-deposit 8832d9e, PR #17). Logs in `$WORK/rip-pic-6/logs6/` on the CPU cluster.

- `build-gpu/tests`: all pass, 0 failures (63 test lines + "tiled deposit tests skipped (GPU build)",
  5.8 s). New test "reductions: par_sum and par_max_abs exact" ok, max error 0.0e+00.
- `tests/regression/run.sh build-gpu/vlpl`: all 5 cases match the references (table in the next
  section).
- Streaming deck (`examples/streaming_plasma.deck`, 448 steps), wall time:

| run | fifth check | sixth check |
|---|---|---|
| default (global, no sort) | 0.6 s | 0.6 s |
| box, tile 4 | 3.2 s | 2.6 s |
| box, tile 6 | 3.4 s | 2.6 s |
| box, tile 8 | 3.4 s | 2.7 s |

  Box tile 6 gains 0.8 s = 1.8 ms per step (expected about 2.3 ms).

- `nsys` of the first 60 steps (`grid.t_end=200fs`, as in the fifth check), per call:

| item | fifth check | sixth check |
|---|---|---|
| box kernel (`push_box_impl`), tile 6 | 616 us | 532 us |
| global push (`push_impl`) | 369 us | 370 us |
| stray `par_sum` | 656 us | gone (strays counted by an int atomic) |
| W_kin reduction (`rip_reduce_kernel`, kinetic_energy) | 656 us | 20 us |
| `cudaMalloc` + `cudaFree` calls, box tile 6 | 3024 | 1143 + 1140 |
| `cudaMalloc` calls, global | | 43 |

  Fixes 1 and 2 work: the stray reduction and the deposit's per-call allocations are gone, and the
  diagnostic reductions are 30x faster. The box kernel itself is 1.44x slower than the global push
  (was 1.67x), as expected from the CAS-loop shared FP64 atomics.

- What is left of the box overhead is the host tile sort, which box needs every step:
  `sort_by_tile` allocates the key buffer (`Buf<int> dk`) and `Species::permute` allocates the
  permutation plus one temporary per attribute (`attr_gather_kernel`: 854 calls in 60 steps, ~14 per
  step); that is the remaining 19 allocations per step. `cudaMemcpy` takes 102 ms in 2328 calls
  (global: 13 ms in 1844), because the keys go to the host and the permutation comes back. So box
  2.6 s = 0.6 s of the default run + ~2 s of host sort, and only the device tile sort can tell
  whether box pays.

## Phase 1 core check on the RTX 5090 (main 260b1dc)

Done on 2026-10-27, same machine and commit (moving window, open-boundary currents, density
profiles, beam self-field). Both builds compile without errors. The CPU runs use 20 threads
(`OMP_PROC_BIND=false`).

**Tests.** `build-cpu/tests`: 75 ok, 0 failures, 34.5 s. `build-gpu/tests`: 63 ok, 0 failures,
5.8 s. The lines that differ:
- CPU only (12 lines): the four "open x, ..., tiled" lines, the six lines of the tiled deposit
  (colouring, tiled = global shape 1 / 2, 1 and 3 threads bit-identical, sort_every = 3 strays,
  deposit = auto), and the two "window, wake, shape 1 / 2, tiled" lines. On the GPU these are
  replaced by "tiled deposit tests skipped (GPU build)".
- All other lines agree to round-off. The quoted errors differ only in the last digits, e.g. div B
  change 8.8e-14 (CPU) vs 8.4e-14 (GPU). "window, vacuum: plane pulse frozen" has dW/W 4.1e-16 on the
  CPU and 0.0e+00 on the GPU. "beam loaded at t_start in the window" gives max|dGauss| 1.1e-12 on both.

**Regression** (`tests/regression/run.sh`; the GPU against the stored CPU references, the CPU into
`regression_cpu`):

| case | GPU | CPU |
|---|---|---|
| stream | ok, 2 s | ok, 1 s |
| stream_s2 | ok, 1 s | ok, 2 s |
| reflect | ok, 2 s | ok, 8 s |
| lwfa (new) | ok, 2 s | ok, 3 s |
| beam (new) | ok, 2 s | ok, 3 s |
| total | 9.4 s | 16.4 s |

The self-test (a 1e-6 change of W_field is detected) passes in both.

**Full examples** (`output.slice_every=0`):

| deck | GPU time | CPU time (20 threads) | max max\|dGauss\| GPU | CPU |
|---|---|---|---|---|
| lwfa_window | 9 s | 132 s | 2.22e-14 | 2.32e-14 |
| lwfa_beam | 12 s | 162 s | **2.970e-12** | **2.970e-12** |
| beam_vacuum | 2 s | 5 s | 3.3e-16 | 4.0e-16 |

CPU vs GPU `diag.txt`, all rows:
- W_field, W_total and the particle counts agree to all 12 printed digits.
- max|dGauss| differs by at most 2.1e-15, max|rho| by at most 9e-19.

**lwfa_beam exceeds the expected 3e-14, on the CPU and on the GPU alike.**
- Up to step 600 the residual is 1.1e-14. The beam is loaded at step 600 (t = 235.5/omega0 = 100 fs).
- From then on max|dGauss| = 2.97e-12, constant to 3 digits, while max|rho| is about 4e-2.
- Between steps 840 and 860 it drops back to 1.9e-14. That is about 256 steps after loading, i.e. one
  window length (nx = 256, one cell per step).
- So the residual is created at loading, does not grow, stays fixed in the lab frame and leaves
  through the back of the window, while the beam moves with the window.
- It matches the test "beam loaded at t_start in the window" (1.1e-12).
- It is a property of the beam loading, not of the GPU. The developer should check the loading at
  t_start inside the moving window.

## Next GPU check (seventh): device tile sort (handover for the agent on the CPU cluster / the GPU workstation)

Main after PR #25 (device tile sort, merged before a GPU run): in the GPU build `sort_by_tile` sorts on the device
(decision 25): tile keys, CUB `DeviceRadixSort::SortPairs` of (key, index) (stable, the same
permutation as the host counting sort), per-tile counts by int atomics (only the nt counts go to the
host), and `Species::permute_device` gathers every attribute through one persistent scratch buffer
(no allocation per attribute and sort). Compiles for sm_80, sm_90 + MPI, sm_120; the CUB path has
not run yet (no GPU in the cloud session). On the GPU workstation:
```
git checkout main && git pull
make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu && build-gpu/tests
sh tests/regression/run.sh build-gpu/vlpl
D=examples/streaming_plasma.deck
build-gpu/vlpl $D output.dir=g_auto | grep -E "done|steps"                         # global, no sort
build-gpu/vlpl $D numerics.sort_every=1 output.dir=g_s1 | grep -E "done|steps"     # global, sorted
for T in 4 6 8; do build-gpu/vlpl $D numerics.deposit=box "numerics.tile=$T $T $T" output.dir=g_box$T | grep -E "done|steps"; done
nsys profile -o sort7 build-gpu/vlpl $D numerics.deposit=box "numerics.tile=6 6 6" grid.t_end=0.1ps output.dir=g_n
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum sort7.nsys-rep
```
Report: the "tiles: sorted ranges, attributes kept, stable order" test line (it runs the CUB path),
all tests and regression cases; times against the sixth check (global 0.6 s; global sorted 2.5 s;
box 2.6 / 2.6 / 2.7 s); per call of the sort kernels and the `cudaMalloc` count. Expected: the sort
costs a fraction of a ms per step, so global sorted comes close to 0.6 s and box shows its own cost
(the box kernel was 1.44x the global push). Results into a new section here, on a branch from main with a pull
request. If the tiles test fails on the GPU, the device sort on main is broken: say so first.

## Seventh check on the RTX 5090: device tile sort (branch device-tile-sort, 963d2d4)

Done on 2026-10-27 on the GPU workstation (RTX 5090, CUDA 13.0, `-arch=sm_120`), branch `device-tile-sort`
963d2d4 (code commit 0029c36). Logs in `$WORK/rip-pic-7/logs7/` on the CPU cluster.

**Correctness**
- The GPU build compiles.
- The CUB path runs and is correct: "tiles: sorted ranges, attributes kept, stable order" is ok
  (27 tiles, 0 bad).
- `build-gpu/tests`: 63 ok, 0 failures, 5.7 s. The tiled-deposit tests are skipped in the GPU build,
  as before.
- `tests/regression/run.sh build-gpu/vlpl`: all 5 cases match the references, in 9.4 s:
  stream 2 s, stream_s2 1 s, reflect 2 s, lwfa 2 s, beam 2 s. The self-test passes.

**Streaming deck** (`examples/streaming_plasma.deck`, 448 steps, wall time):

| run | sixth check (host sort) | seventh check (device sort) |
|---|---|---|
| default (global, no sort) | 0.6 s | 0.6 s |
| global, `sort_every=1` | 2.5 s | 0.9 s |
| box, tile 4 | 2.6 s | 0.9 s |
| box, tile 6 | 2.6 s | 0.9 s |
| box, tile 8 | 2.7 s | 1.1 s |

The device sort removes about 1.6-1.7 s, i.e. 3.6 ms per step. Box now costs the same as global with
sorting, but still 0.3 s (0.7 ms per step) more than the default global run without sorting.

**`nsys`, first 60 steps** (`grid.t_end=200fs`, two species, so two sorts per step), per call:

| kernel | box tile 6 | global, sort_every = 1 (tile 8) |
|---|---|---|
| index init + per-tile counts by `atomic_inc` (`tiles.cpp:54`) | **121 us** | **223 us** |
| `tile_keys_kernel` | 6.7 us | 6.7 us |
| CUB radix sort (onesweep + histogram + exclusive sum) | 2 x 8.8 + 2.1 + 0.9 us | 9.1 + 2.0 + 0.9 us |
| "moved" `par_sum` (`tiles.cpp:66`) | 4.6 us | 4.6 us |
| `attr_gather_kernel` (854 calls = 7 per sort) | 2.9 us | 2.9 us |
| sort per step (2 species), total | ~0.33 ms | ~0.53 ms |
| box kernel / global push | 531 us | 371 us |
| `cudaMalloc` calls in 60 steps | 53 (sixth: 1143) | 51 |

- The allocations per sort are gone. That leaves 53 calls, against 43 in the default run: the setup
  plus CUB's temporary buffer.
- `cudaMemcpy` calls: 3182 for box tile 6, 2942 for global sorted and 1844 for the default run, i.e.
  about 18-22 more per step. Their time (about 100 ms) mostly waits for the preceding kernels, because
  the copies are synchronous.

**The sort is now dominated by the counting kernel, not by CUB.** The per-tile counts at
`tiles.cpp:54` take 121-223 us per call, against 12 us for the whole CUB sort.
- Each particle does `atomic_inc(&C[K[p]])` on its tile's counter.
- The species is already sorted, so neighbouring threads hit the same counter, and all threads of a
  warp serialise on one address.
- It is slower with fewer tiles: 223 us with 128 tiles (tile 8), 121 us with 396 tiles (tile 6).

Suggestion for the developer: take the counts from the sorted keys after the CUB sort. A tile starts
where `key2[q] != key2[q-1]`, which needs one kernel without atomics, plus a scan or a search for the
starts. Alternatives are warp-aggregated atomics or a shared-memory histogram per block. This would
bring the sort to about 0.1 ms per step, so global with sorting would come close to 0.6 s.

What box itself costs is then the kernel: 531 us against 371 us for the global push (1.43x, as in the
sixth check).

## Next GPU check (eighth): tile starts without atomics (branch tile-starts)
The per-tile counting by `atomic_inc` (121-223 us per call in the seventh check) is gone: after the
CUB sort, one kernel over the sorted keys writes the tile starts (position q sets start[t] = q for
the tiles t in (key[q-1], key[q]]), and the nt + 1 starts go to the host. The permutation and so all
results are unchanged. On the GPU workstation:
```
git fetch && git checkout tile-starts && git pull
make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu && build-gpu/tests
sh tests/regression/run.sh build-gpu/vlpl
D=examples/streaming_plasma.deck
build-gpu/vlpl $D output.dir=g_auto | grep -E "done|steps"                         # global, no sort
build-gpu/vlpl $D numerics.sort_every=1 output.dir=g_s1 | grep -E "done|steps"     # global, sorted
for T in 4 6 8; do build-gpu/vlpl $D numerics.deposit=box "numerics.tile=$T $T $T" output.dir=g_box$T | grep -E "done|steps"; done
nsys profile -o sort8 build-gpu/vlpl $D numerics.deposit=box "numerics.tile=6 6 6" grid.t_end=0.1ps output.dir=g_n
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum sort8.nsys-rep
```
Report: the tiles test line, all tests and regression cases; times against the seventh check (global
0.6 s; global sorted 0.9 s; box 0.9 / 0.9 / 1.1 s); per call of the starts kernel (expected a few us,
like the CUB sort's 12 us). Expected: global sorted about 0.6-0.7 s, box at the cost of its kernel
(1.43x the global push). Results into a new section here, on a branch from `tile-starts` with a pull
request into it (or from main if it is merged by then).

## Eighth check on the RTX 5090: tile starts without atomics (main b587e3a)

Done on 2026-10-27 on the GPU workstation (RTX 5090, CUDA 13.0, `-arch=sm_120`), main b587e3a. It contains
`tile-starts` 55f6e47 (PR #28), so main was used instead of the branch. Logs in
`$WORK/rip-pic-8/logs8/` on the CPU cluster.

**Correctness**
- "tiles: sorted ranges, attributes kept, stable order" is ok (27 tiles, 0 bad).
- `build-gpu/tests`: 63 ok, 0 failures, 5.7 s.
- `tests/regression/run.sh build-gpu/vlpl`: all 5 cases match the references, in 9.4 s:
  stream 2 s, stream_s2 1 s, reflect 2 s, lwfa 2 s, beam 3 s. The self-test passes.

**Streaming deck** (448 steps, wall time):

| run | seventh check | eighth check |
|---|---|---|
| default (global, no sort) | 0.6 s | 0.6 s |
| global, `sort_every=1` | 0.9 s | 0.7 s |
| box, tile 4 | 0.9 s | 0.9 s |
| box, tile 6 | 0.9 s | 0.8 s |
| box, tile 8 | 1.1 s | 0.9 s |

**`nsys`, first 60 steps** (`grid.t_end=200fs`, two species, so two sorts per step), per call:

| kernel | box tile 6 (396 tiles) | global, sort_every = 1 (tile 8, 128 tiles) |
|---|---|---|
| tile starts from the sorted keys (lambda instance 2) | **1.76 us** (was 121 us with atomics) | **1.76 us** (was 223 us) |
| index init (lambda instance 1) | 1.56 us | 1.56 us |
| `tile_keys_kernel` | 6.7 us | 6.7 us |
| CUB radix sort | 2 x 8.8 + 2.1 + 0.9 = 20.6 us (9 key bits: two passes) | 9.1 + 2.1 + 0.9 = 12.1 us (7 bits: one pass) |
| "moved" `par_sum` | 4.6 us | 4.6 us |
| `attr_gather_kernel`, 7 per sort | 7 x 2.9 = 20 us | 20 us |
| sort per species | ~55 us | ~47 us |
| sort per step (2 species) | ~0.11 ms | ~0.09 ms |
| box kernel / global push | 531 us | 371 us |
| `cudaMalloc` calls in 60 steps | 53 | 51 (default run: 43) |

**Result**
- The starts kernel takes 1.8 us per call, 70-130x less than the atomic counts.
- The whole device sort now costs about 0.1 ms per step.
- Global with sorting takes 0.7 s, within the expected 0.6-0.7 s. The 0.1 s it adds is 0.22 ms per
  step: the 0.1 ms of sort kernels, plus about 20 extra synchronous `cudaMemcpy` calls and launches
  per step (2942 against 1844 calls in 60 steps).
- Box now runs at the cost of its kernel. With tile 6 it adds 2 x (531 - 371) us = 0.32 ms per step,
  i.e. 0.14 s on the deck, on top of global sorted, which gives the measured 0.8 s.
- So on the RTX 5090 the global deposit without sorting (0.6 s) stays the fastest. The box kernel
  (1.43x the global push) is the only reason box is slower.

## Next GPU check (ninth): fixed-point box deposit (branch box-fixed)
The box now sums the currents as int64 fixed point (decision 26). Each add is two native 32-bit
shared atomics (`ATOMS.ADD`) with an exact carry, instead of the FP64 CAS loop. The SASS of the sm_90
and sm_120 builds has no `ATOMS.CAST.SPIN` left (cloud session, `cuobjdump -sass`). Global and tiled
are unchanged. On the GPU workstation:
```
git fetch && git checkout box-fixed && git pull
make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu && build-gpu/tests
sh tests/regression/run.sh build-gpu/vlpl
/opt/cuda-13.0/bin/cuobjdump -sass build-gpu/vlpl | grep -oE "ATOMS\.[A-Z0-9.]+" | sort | uniq -c
D=examples/streaming_plasma.deck
build-gpu/vlpl $D output.dir=g_auto | grep -E "done|steps"                         # global, no sort
build-gpu/vlpl $D numerics.sort_every=1 output.dir=g_s1 | grep -E "done|steps"     # global, sorted
for T in 4 6 8; do build-gpu/vlpl $D numerics.deposit=box "numerics.tile=$T $T $T" output.dir=g_box$T | grep -E "done|steps"; done
nsys profile -o box9 build-gpu/vlpl $D numerics.deposit=box "numerics.tile=6 6 6" grid.t_end=200fs output.dir=g_n
nsys stats --report cuda_gpu_kern_sum box9.nsys-rep
```
Report:
- the lines "block_add: 64-bit sums ... exact" (the concurrent 32-bit adds with carry on the GPU) and
  "box deposit, weights over 8 decades", plus all tests and regression cases;
- the `ATOMS` counts (expected `ATOMS.ADD` and `ATOMS.POPC.INC`, no `ATOMS.CAST.SPIN`);
- times against the eighth check (global 0.6 s; global sorted 0.7 s; box 0.9 / 0.8 / 0.9 s);
- the box kernel per call against the global push (eighth check: 531 against 371 us).

Expected: the box kernel below the global push, so box at tile 6 or 8 under 0.7 s. If so, box can
become `auto` on the GPU; if not, the remaining cost is the second atomic or the occupancy (shared
memory per block). Results into a new section here, on a branch from `box-fixed` with a pull request
into it (or from main if it is merged by then).

## Ninth check on the RTX 5090: fixed-point box deposit (main 36ffaf7)

Done on 2026-10-27 on the GPU workstation (RTX 5090, CUDA 13.0, `-arch=sm_120`), main 36ffaf7. It contains
`box-fixed` be1102b (PR #30), so main was used. Logs in `$WORK/rip-pic-9/logs9/` on the CPU cluster.

**Correctness**
- `build-gpu/tests`: 65 ok, 0 failures, 5.8 s. The two new lines:
  - "block_add: 64-bit sums from 32-bit adds with carry, exact": ok, 0 wrong cells. This is the
    concurrent 32-bit adds with carry, run on the GPU.
  - "box deposit, weights over 8 decades: box = global to round-off of max|J|, Gauss": ok, max rel
    dJx 3.4e-15, per cell 9.5e-07 (|J| down to 1e-09 of max), max|dGauss| 3.3e-15.
- The other box lines: "box deposit = global" for shape 1 and 2, full and partial tiles, max rel dJx
  5.7e-15 .. 8.4e-15 and max|dGauss| <= 7.6e-15. "sort_every = 3, strays deposited globally":
  834542 strays, max|dGauss| 8.7e-15.
- `tests/regression/run.sh build-gpu/vlpl`: all 5 cases match the references, in 9.4 s:
  stream 1 s, stream_s2 2 s, reflect 2 s, lwfa 2 s, beam 2 s. The self-test passes.

**SASS** (`cuobjdump -sass build-gpu/vlpl`, the whole binary):
- 419 `ATOMS.ADD` and 39 `ATOMS.POPC.INC.32`, and **no `ATOMS.CAST.SPIN`**.
- In the two box kernels: 396 `ATOMS.ADD`, 92 `REDG.E.ADD.F64.RN` and 2 `REDG.E.ADD`. The global
  reds are the flush and the strays.
- In the two global push kernels: 90 `REDG.E.ADD.F64.RN`.

**Streaming deck** (448 steps, shape 1, wall time):

| run | eighth check | ninth check |
|---|---|---|
| default (global, no sort) | 0.6 s | 0.6 s |
| global, `sort_every=1` | 0.7 s | 0.7 s |
| box, tile 4 | 0.9 s | 0.8 s |
| box, tile 6 | 0.8 s | 0.8 s |
| box, tile 8 | 0.9 s | 0.8 s |

**`nsys`, first 60 steps** (`grid.t_end=200fs`), per call:

| kernel | eighth check | ninth check |
|---|---|---|
| box kernel (`push_box_impl`, shape 1), tile 6 | 531 us | **444 us** |
| global push (`push_impl`, shape 1) | 371 us | 370 us |
| new `par_max_abs` of the weights (box scale) | | 5.2 us |

**The box kernel is 16 % faster but still 1.20x the global push** (it was 1.43x). Box at tile 6 or
8 takes 0.8 s, not under 0.7 s, so the expected switch of `auto` on the GPU is not supported by these
numbers.

**Where the rest likely goes** (`cuobjdump -res-usage`; occupancy is computed, not measured, because
`ncu` still has no counter access on the GPU workstation):
- **Registers.** The shape-1 box kernel now uses **94 registers** (STACK 96), the global push 80.
  With 256 threads per block, 94 registers allow 2 blocks per SM, i.e. 33 % occupancy. 80 registers
  allow 3 blocks, i.e. 50 %.
- **Shared memory.** The box scratch is 3 (T+5)^3 x 8 bytes. That is 17.5 KB at tile 4, 31.9 KB at
  tile 6 and 52.7 KB at tile 8, plus 1 KB reserved per block. It limits tile 8 to 1 block per SM
  (17 %). Tiles 4 and 6 would allow 5 and 3 blocks, so registers are the limit there.
- **Shape 2** (not on this deck): the box kernel uses 252 registers (STACK 224), against 254 for the
  global push (`push_impl<..., 2, false>`).
- Bringing the shape-1 box kernel to <= 80 registers would give it the global push's 50 % occupancy,
  e.g. with `__launch_bounds__(256, 3)` or `-maxrregcount` for this kernel; it would need checking for
  spills. That would show whether the second 32-bit add or the occupancy is the remaining cost.

## Next GPU check (tenth): occupancy of the box kernel (branch box-occupancy)
The ninth check's 94 registers belong to another kernel. ptxas (`-Xptxas -v`, sm_120) gives the
shape-1 box kernel (`rip_block_kernel`) **240 registers uncapped**, which is one block of 256 per SM
(17 %); the global push has 80 (three blocks). `par_blocks<MinB>` now sets `__launch_bounds__(256,
MinB)`, and strays are deposited after the box kernel (one copy of the deposit in it). ptxas:

| `RIP_BOX_MINB1` | registers | spills | blocks per SM (registers) |
|---|---|---|---|
| 1 (no cap, as before) | 240 | 0 | 1 |
| 2 (default) | 128 | 0 | 2 |
| 3 | 80 | 56 B | 3 |

On cuda (three builds; tile 8 is limited to one block by its 53 KB of shared memory, so use 4 and 6):
```
git fetch && git checkout box-occupancy && git pull
for M in 1 2 3; do
  make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu-m$M NVCCFLAGS="-DRIP_BOX_MINB1=$M -Xptxas -v" 2>&1 | grep -A2 "rip_block_kernel" | grep -E "Used|spill" | head -4
done
build-gpu-m2/tests && sh tests/regression/run.sh build-gpu-m2/vlpl
D=examples/streaming_plasma.deck
build-gpu-m2/vlpl $D output.dir=g_auto | grep -E "done|steps"                          # global, no sort
for M in 1 2 3; do for T in 4 6; do echo "MINB1=$M tile $T"; build-gpu-m$M/vlpl $D numerics.deposit=box "numerics.tile=$T $T $T" output.dir=g_m${M}_$T | grep -E "done|steps"; done; done
for M in 1 2 3; do nsys profile -o box10_m$M build-gpu-m$M/vlpl $D numerics.deposit=box "numerics.tile=6 6 6" grid.t_end=200fs output.dir=g_n$M; nsys stats --report cuda_gpu_kern_sum box10_m$M.nsys-rep | grep -E "rip_block_kernel|rip_kernel" | head -4; done
```
Report: the ptxas lines of the three builds (registers, spills), all tests and regression cases
(MINB1 = 2), the wall times, and the box kernel per call against the global push (370 us) for each
MINB1. Expected: the box kernel faster with 2 or 3 blocks per SM; if it gets below the global push,
`auto` can take box on the GPU. If the shape-2 deck matters, also time `RIP_BOX_MINB2=2` once (432 B
of spills). Results into a new section here, on a branch from `box-occupancy` with a pull request
into it (or from main if it is merged by then).

## Tenth check on the RTX 5090: occupancy of the box kernel (main 7d6ea1b)

Done on 2026-10-27 on the GPU workstation (RTX 5090, CUDA 13.0, `-arch=sm_120`), main 7d6ea1b. It contains
`box-occupancy` bc80a39 (PR #32), so main was used. There were four builds:
- `RIP_BOX_MINB1` = 1, 2 and 3;
- one more with `RIP_BOX_MINB2=2` for shape 2.

Logs are in `$WORK/rip-pic-10/logs10/` on the CPU cluster.

**Correction to the ninth check:** its "94 registers" for the shape-1 box kernel came from a
`cuobjdump -res-usage` entry whose name was cut off, i.e. from another kernel. The ptxas numbers
below are the right ones.

**ptxas** (`-Xptxas -v`), `rip_block_kernel<push_box_impl<..., O>>`:

| build | shape 1 (O = 1) | shape 2 (O = 2) |
|---|---|---|
| MINB1 = 1 | 232 registers, 0 spills, 96 B stack | 252 registers, 0 spills, 128 B stack |
| MINB1 = 2 (default) | 128 registers, 0 spills, 96 B stack | 252 registers, 0 spills |
| MINB1 = 3 | 80 registers, **0 spills** (the cloud build reported 56 B), 96 B stack | 252 registers, 0 spills |
| MINB2 = 2 | (128, as default) | 128 registers, **560 B spill stores, 600 B spill loads**, 688 B stack |

**Correctness** (MINB1 = 2):
- `build-gpu-m2/tests`: 65 ok, 0 failures, 5.7 s. "block_add ... exact" is ok (0 wrong cells), and
  "box deposit, weights over 8 decades" is ok (max rel dJx 3.4e-15).
- `tests/regression/run.sh build-gpu-m2/vlpl`: all 5 cases match the references, in 9.4 s:
  stream 1 s, stream_s2 1 s, reflect 2 s, lwfa 2 s, beam 2 s. The self-test passes.

**Streaming deck, shape 1** (448 steps, wall time). Global takes 0.6 s and global sorted 0.7 s.
**Box takes 0.8 s for every MINB1 (1, 2, 3) and every tile (4, 6, 8).**

**Box kernel per call** (`nsys`, first 60 steps), against the global push at 370 us:

| MINB1 | registers | blocks per SM (registers) | tile 4 | tile 6 |
|---|---|---|---|---|
| 1 | 232 | 1 | 465 us | 433 us |
| 2 | 128 | 2 | 452 us | 434 us |
| 3 | 80 | 3 | 456 us | 508 us |

**Shape 2** (same deck with `numerics.shape=2`):
- Wall time: global 1.0 s; box 1.3 s (tile 4) and 1.2 s (tile 6), the same for both builds.
- Per call at tile 6: the global push takes 886 us, box 947 us uncapped (252 registers) and 944 us with
  MINB2 = 2 (128 registers, spills).

**Result: occupancy is not what limits the box kernel.**
- Going from 1 to 2 blocks per SM changes nothing: 433 vs 434 us at tile 6.
- Going to 3 blocks makes tile 6 slower (508 us). Tile 6 has 32 KB of shared memory per block, so
  3 blocks fill the shared memory.
- No variant gets below the global push. The best is 1.17x (433 us) for shape 1, and 1.07x for shape 2.
- So these numbers do not support making box `auto` on the GPU (RTX 5090). MINB1 = 2 as the default
  does no harm.

What is left is likely the work per tile that does not depend on the particles. That is a guess, not
measured (`ncu` is still blocked):
- Zeroing and flushing the box, 3 (T+5)^3 cells per tile with global `REDG.ADD.F64`. At tile 6 that is
  3993 cells per tile, about 1.6 M global adds per call for 396 tiles, 18x the cells of the tile.
- The `__syncthreads`, and load imbalance between tiles.

Two ways to test this:
- skip all-zero cells in the flush, which also helps at the plasma edges;
- time a deck with many more particles per cell, where the fixed cost per tile matters less.

The GH200 (full-rate FP64) may also rank box and global differently.

## Next GPU check (eleventh): bubble-regime LWFA against FBPIC (branch lwfa-nonlinear)
Box tuning on the RTX 5090 is closed after the tenth check; `auto` stays global. This check is
physics: the strongly nonlinear benchmark `bench/lwfa_bubble/` (README there). Setup:
- a0 = 4, 20 fs, matched waist 7.2 um, 0.005 n_c with a 32-52 um up-ramp;
- mobile protons in both codes;
- a 30 um moving window, 6000 steps (1 ps).

VLPL: 600 x 160 x 160 cells, 138 M particles, ~10 GB on the GPU. FBPIC: RZ, Nr = 80, 2 modes,
3 M particles. Run FBPIC on the GPU (`--cuda`, needs cupy and numba-cuda for sm_120). If that does
not work, run it on the epyc node of the CPU cluster (`sbatch -p epyc`, all cores, numba threads).
```
git fetch && git checkout lwfa-nonlinear && git pull
make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu && build-gpu/tests | tail -1
date +%s; build-gpu/vlpl bench/lwfa_bubble/rip.deck > rip_bubble.log; date +%s          # out_bubble_rip/
python3 bench/lwfa_bubble/fbpic_bubble.py out_bubble_fbpic --cuda > fb_bubble.log         # or on epyc without --cuda
python3 bench/lwfa_bubble/compare.py out_bubble_rip out_bubble_fbpic | tee compare_bubble.txt
tail -n 3 out_bubble_rip/diag.txt                                                       # Gauss drift (column 6)
```
Report:
- the compare table: per 100 fs, the laser field, the bubble length, the accelerating field, the
  cavity density and the first-bucket L2;
- the injected-electron table: charge, mean and maximum energy for γ >= 20, from VLPL's
  `beam_injected.txt` and spectrum against FBPIC's particles. It needs main after the beam
  diagnostics (branch diag-beams), for both runs;
- its exit status;
- the wall times of both codes;
- the largest max|dGauss| in diag.txt;
- if possible, on-axis plots of Ex and n_e at 400, 700 and 1000 fs, both codes in one panel
  (for the README).

The tolerances (laser 10 %, length 5 %, L2 0.25) are first guesses: report the numbers even if it
fails. Results into a new section here and into the "Results" section of
bench/lwfa_bubble/README.md, on a branch from `lwfa-nonlinear` with a pull request into it (or from
main if it is merged by then).

## Next GPU check (twelfth, short): rewritten gather and deposit (branch cpu-perf)
The CPU work of decisions 28 and 29 also changes the device code: the gather weights per axis and
offset, the pointer strides, and the shape arrays by selects in the Esirkepov kernel. On the CPU the
results are bit-identical. On cuda (can go with the eleventh check):
```
git fetch && git checkout cpu-perf && git pull
make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu && build-gpu/tests | tail -3
sh tests/regression/run.sh build-gpu/vlpl
build-gpu/vlpl examples/streaming_plasma.deck output.dir=g_auto | tail -2               # time per operator
build-gpu/vlpl examples/streaming_plasma.deck numerics.shape=2 output.dir=g_s2 | tail -2
```
Report:
- the test summary, including "continuity with moves over several cells (split)";
- the regression results;
- the two wall times and the per-operator times, against the eighth check (global 0.6 s).

Expected: everything passes, and the times are equal or better. Results into a new section here.

## Twelfth check on the RTX 5090: rewritten gather and deposit (main 327e271)

Done on 2026-10-27 on the GPU workstation (RTX 5090, CUDA 13.0, `-arch=sm_120`), main 327e271. It contains
`cpu-perf` (PR #35), so main was used. Logs in `$WORK/rip-pic-11/logs11/` on the CPU cluster.

**Correctness**
- `build-gpu/tests`: 71 ok, 0 failures, 5.8 s, including:
  - "continuity with moves over several cells (split), shape 1": ok, moves up to 5.6 cells, max err
    2.1e-12 (|drho/dt| ~ 4.7e+03);
  - the same for shape 2: ok, max err 2.8e-12 (|drho/dt| ~ 3.7e+03).
- `tests/regression/run.sh build-gpu/vlpl`: all 5 cases match the references, in 9.4 s:
  stream 1 s, stream_s2 2 s, reflect 2 s, lwfa 2 s, beam 2 s. The self-test passes.

**Streaming deck** (448 steps):

| run | before | twelfth check | per operator (twelfth) |
|---|---|---|---|
| shape 1 (default, global) | 0.6 s (eighth) | 0.6 s | push_deposit 0.39 s (69 %), rip_cc 0.15 s (27 %), migrate 0.01 s, other 0.01 s |
| shape 2 | 1.0 s (tenth) | 1.0 s | push_deposit 0.79 s (81 %), rip_cc 0.15 s (16 %), migrate 0.01 s, other 0.02 s |

Everything passes; the times are unchanged.

## Eleventh check on the RTX 5090: bubble-regime LWFA against FBPIC (main 327e271)

Done on 2026-10-27 on the GPU workstation, main 327e271. It contains `lwfa-nonlinear` (PR #34) and the beam
diagnostics (PR #36). Both codes ran on the RTX 5090. Logs, comparisons and plots are in
`$WORK/rip-pic-11/logs11/` on the CPU cluster; the outputs are `out_bubble_*` there.

**Setup change (Alexander): the laser sits 3.3 sigma behind the front edge.** The first run used
`window.start = 80 fs`, which left the laser centre 18 um behind the front edge and 12 um from the
back. The bubble then ran into the back edge of the box: compare.py found the "bubble back" at the
box edge and skipped every output from 500 fs on (only cavity behind the laser).
- A second run with `window.start = 104 fs` (exactly 3 sigma) let the laser head reach the open right
  edge before the window moved. VLPL warns about this ("c window.start > box length"), and
  max|dGauss| rose to 1.0e-3.
- Final setup: `window.start = 100 fs` with the default 40 fs delay. c window.start = 29.98 um is
  within the 30 um box. The laser centre is 12.0 um = 3.3 sigma (field rms 12 fs = 3.6 um) behind the
  front edge.
- FBPIC's box is shifted to the same place: [-29.98, 0.02] um at t = 0, i.e. [0, 30] um at 100 fs.
- `rip.deck`, `fbpic_bubble.py` and the README are updated. All numbers below are from this setup.

**FBPIC on the RTX 5090 (sm_120)**
- Environment: FBPIC 0.27.1, Python 3.12 (uv, `$WORK/bench_py/venv`), numba 0.68.0,
  numba-cuda 0.30.4 (cu13), cupy-cuda13x 14.2.0, numpy 2.4.6.
- The pip CUDA libraries must match the driver's CUDA 13.0: `cuda-toolkit[...]==13.0.*`.
- Combinations that do not work:
  - numba-cuda for CUDA 12 (NVVM 12.9): the deposit kernels fail with `ERROR_INVALID_IR`
    (`atomicrmw fadd` on doubles);
  - the CUDA 13.4 pip libraries: `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`;
  - numpy 2.5: it removed `np.row_stack`, which numba-cuda 0.30.4 still uses.

**Runs** (6000 steps, 1 ps; VLPL stops at step 5999, so its last output is 900 fs):

| | VLPL | FBPIC |
|---|---|---|
| exit status | 0 | 0 |
| wall time | 1336 s | 160 s (with numba compilation) |
| grid | 3D 600 x 160 x 160 = 15.4 M cells | RZ 600 x 80, 2 modes |
| particles per cell | e 2 x 2 x 2, p 1 | 2 x 2 x 8 (z, r, theta), both species |
| macro-particles (full box) | 138 M (122.4 M e + 15.4 M p) | 3.1 M |
| per operator | push_deposit 1021 s (78 %), moving_window 209 s (16 %), migrate 60 s (4 %), rip_cc 19 s (1 %) | |

**compare.py** (exit status 1: only the first-bucket L2 exceeds its tolerance):

| t [fs] | E_las RIP | FB | L_b RIP [um] | FB | E_acc RIP | FB | n_cav RIP | FB | L2 |
|---|---|---|---|---|---|---|---|---|---|
| 300 | 3.727 | 3.692 | 11.35 | 11.40 | 0.2004 | 0.1794 | 0.133 | 0.190 | 0.784 |
| 400 | 3.797 | 3.755 | 11.80 | 11.85 | 0.2022 | 0.1850 | 0.103 | 0.201 | 0.825 |
| 500 | 4.093 | 3.999 | 12.10 | 12.10 | 0.2088 | 0.1978 | 0.028 | 0.130 | 0.693 |
| 600 | 4.541 | 4.383 | 12.10 | 12.05 | 0.1849 | 0.1958 | 0.014 | 0.083 | 0.649 |
| 700 | 4.887 | 4.676 | 11.85 | 11.90 | 0.2018 | 0.2047 | 0.003 | 0.052 | 0.640 |
| 800 | 5.116 | 4.881 | 11.40 | 11.55 | 0.1914 | 0.2038 | 0.001 | 0.041 | 0.521 |
| 900 | 5.316 | 5.041 | 11.30 | 11.65 | 0.1769 | 0.2011 | 0.000 | 0.050 | 0.523 |

- The laser field agrees within 1-5.5 %. VLPL's grows a little more as the pulse self-focuses.
- The bubble length agrees within 0.35 um (3 %), and the accelerating field within 12 %.
- The first-bucket L2 (0.52-0.83) is set by the sharp spike at the bubble back.
  - VLPL's spike sits 0.1-0.3 um behind FBPIC's (`bubble_onaxis.png`). From 700 fs on, its
    shape is also changed by the larger injected bunch (beam loading).
  - On the steep spike, that shift alone gives a large L2, while Ex over the rest of the bucket lies
    on top of FBPIC's.
  - An L2 over the bucket without the spike (or after aligning the spike) would test the wake shape.
    The 0.25 tolerance was a first guess.
- The cavity density: VLPL's cavity empties completely (n_e/n_0 <= 0.03 from 500 fs on). FBPIC
  keeps 0.04-0.13 on axis.

**Injected electrons (gamma >= 20)**, VLPL against FBPIC with 1x, 4x and 16x particles (FBPIC
2 x 2 x 8, 2 x 4 x 16 and 4 x 4 x 32 per cell):

| t [fs] | Q RIP [pC] | FB 1x | 4x | 16x | <E> RIP [MeV] | FB 1x | 16x | E_max RIP [MeV] | FB 1x | 16x |
|---|---|---|---|---|---|---|---|---|---|---|
| 400 | 46.1 | 4.2 | 4.6 | 4.1 | 13.6 | 10.6 | 10.8 | 29.4 | 12.7 | 22.8 |
| 500 | 87.0 | 44.5 | 44.5 | 43.5 | 20.3 | 21.0 | 21.1 | 51.4 | 39.6 | 40.8 |
| 600 | 122.6 | 71.7 | 69.8 | 68.6 | 28.3 | 31.9 | 32.7 | 83.6 | 64.5 | 66.2 |
| 700 | 131.8 | 77.3 | 76.9 | 75.4 | 40.0 | 47.5 | 48.5 | 117.9 | 87.3 | 89.1 |
| 800 | 129.1 | 77.5 | 77.1 | 75.6 | 55.6 | 66.3 | 68.0 | 154.4 | 110.8 | 112.8 |
| 900 | 118.3 | 77.5 | 77.1 | 75.6 | 75.2 | 84.4 | 86.7 | 187.7 | 133.7 | 135.9 |

- FBPIC has converged in particle number: 77.5 / 77.1 / 75.6 pC at 900 fs. In compare.py, the laser
  field, the bubble length and E_acc of the three FBPIC runs agree to < 0.5 %.
- VLPL injects 1.5-1.7x more charge (118-132 pC). It injects earlier (46 pC at 400 fs against 4 pC)
  and reaches a higher maximum energy (188 against 136 MeV) at a lower mean energy.
- Alexander's assessment: the larger trapped charge is probably caused by the coarse grid. Both codes use the same
  transverse cell, 0.25 um = 5 dx, but VLPL is 3D Cartesian. This is not a problem for the benchmark.

**Plots** (`bench/lwfa_bubble/`, at 400, 700 and 900 fs, both codes):
- `bubble_onaxis.png`: on-axis Ex and n_e;
- `bubble_ne.png`: the electron density in the plane perpendicular to the polarisation.

The bubbles have the same shape and radius (7-8 um), the same sheath, and the same injected bunch at
the back.

**Gauss law (VLPL).** The largest max|dGauss| is **1.05e-4**:
- It appears at step 660, when the first plasma enters the box: the box front passes 32 um at step
  640. It stays at 1.051e-4 until step 1140 and then leaves through the back after one window length,
  so it is fixed in the lab frame.
- A second residual of 1.953e-7 lasts until step 1620.
- From step 1680 on it is at round-off: 2e-14 .. 6e-13 with max|rho| up to 2.9 (the injected bunch),
  and 2.9e-12 at the end.
- With the window starting at 80 fs, the same plasma entry gave only the 1.953e-7 (steps 540-1500).
- The CPU (20 threads) reproduces the 1.953e-7 exactly over the same steps: full-resolution deck to
  270 fs, 2942 s on the CPU vs 236 s on the GPU. So it is not a GPU effect.
- The 1.05e-4 of the final setup was not run on the CPU.
- The reduced deck (32 x 32 cells of 0.75 um, 1 ppc) shows neither: max|dGauss| 2.1e-14 (GPU) and
  2.2e-14 (CPU).
- The developer should look at where the plasma enters at the right edge when the moving window
  starts.

**Speed per particle** (RTX 5090, FP64 at 1/64 rate in both codes):
- VLPL carries 119 M particles on average over the run, i.e. 7.1e11 particle-steps:
  - push_deposit: 1.4 ns per particle-step;
  - the whole run: 1.9 ns per particle-step.
- FBPIC: the 1x, 4x and 16x runs take 154, 545 and 1986 s (wall times with compilation, from the
  "Total time" line). A straight line through them gives:
  - 120-130 s per 1x of particles, about 8 ns per particle-step (assuming the same 86 % average
    filling as VLPL);
  - about 25 s of fixed cost: compilation, field solve, output.
- So VLPL's particle work is about 5x cheaper per particle on this GPU. FBPIC's per-particle work
  is larger, though: cubic shape on 4 x 4 (z, r) points with 2 complex modes and e^{i m theta}
  factors, plus a sort every step.

**On the CPU**, the tile sort takes 41 % of the 270 fs run of this deck (1193 s against 1164 s for
push_deposit, 20 threads). That may be worth a look.

## Next GPU check (thirteenth, short): CPML layers in y and z (branch pml-yz)
The PML step has new device kernels (decision 31): the predictor and the corrector with ψ, and the
transport sources once per location. On cuda (can go with the eleventh and twelfth checks):
```
git fetch && git checkout pml-yz && git pull
make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 BUILD=build-gpu && build-gpu/tests | grep -E "pml|failure"
D=examples/streaming_plasma.deck
build-gpu/vlpl $D output.dir=g_per | tail -2
build-gpu/vlpl $D grid.boundary_y=pml grid.boundary_z=pml pml.cells=8 "species electrons.uth_y=0.3" output.dir=g_pml | tail -2
```
Report the five pml test lines and the per-operator times of the two runs (rip_cc with and without
layers). Expected: the same numbers as on the CPU (5.1e-6 and 2.0e-5, difference 0 in the window
test), and rip_cc at most a few tens of percent slower with layers.

## Next GPU check (fourteenth, short): openPMD output and restart (branch openpmd)
Output and checkpoints go through host copies of device fields and particles (decision 34). The
restart must restore the state on the device. On the GPU workstation, with the HDF5 in $WORK/gpu/opt
(the prefix with include/hdf5.h):
```
git fetch && git checkout openpmd && git pull
H5=$(dirname $(dirname $(find $WORK/gpu/opt -name hdf5.h | head -n 1)))
make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 HDF5_DIR=$H5 BUILD=build-gpu
build-gpu/tests openpmd
L=examples/lwfa_window.deck
build-gpu/vlpl $L grid.t_end=100fs output.slice_every=0 output.openpmd_every=200 output.openpmd_species=electrons \
    checkpoint.every=300 output.dir=g_a | tail -n 3
mkdir -p g_b/checkpoints && cp g_a/checkpoints/checkpoint_000300.h5 g_b/checkpoints/ && cp g_a/diag.txt g_b/
build-gpu/vlpl $L grid.t_end=100fs output.slice_every=0 output.openpmd_every=200 output.openpmd_species=electrons \
    output.dir=g_b --restart latest | tail -n 3
paste g_a/diag.txt g_b/diag.txt | tail -n 3
ls -l g_a/openpmd g_a/checkpoints
```
Please report:
- the two openpmd test lines;
- the last lines of both runs, with the time per operator;
- the last three lines of the pasted diag files;
- the file sizes.

Expected:
- O1 exact (E B J 0, beam particles 0), ρ about 1e-15.
- O2 "ok" with a round-off difference (the GPU deposit sums with atomics).
- The diag files agree to about 12 digits.
- The run time with output close to the run without.
- Sizes for the full 256 x 64 x 64 window: an output file has about 30 MB of fields (E, B, rho in
  single precision) plus 28 bytes per electron. A checkpoint has about 60 MB of fields plus 56 bytes
  per particle.

## Thirteenth check on the RTX 5090: CPML layers in y and z (gpu-check-11 af931cd = main 0777bcf + checks 11, 12)

Done on 2026-10-27 on the GPU workstation (RTX 5090, CUDA 13.0, `-arch=sm_120`), branch gpu-check-11 af931cd.
It contains main 0777bcf with `pml-yz` (PR #37). Logs in `$WORK/rip-pic-11/logs13/` on
the CPU cluster.

**Tests.** `build-gpu/tests`: 79 ok, 0 failures, 7.2 s. The five pml lines, GPU and CPU (20 threads,
same commit):

| test line | GPU | CPU |
|---|---|---|
| packet into a y layer (16 cells), reflection | normal 5.4e-06, 45 deg 2.0e-05 | normal 5.4e-06, 45 deg 2.0e-05 |
| packet into a z layer, reflection = y layer | 45 deg 2.0e-05 (y: 2.0e-05) | the same |
| moving window = fixed box shifted, layers included | 60 shifts, max difference 0.0e+00 | the same |
| random fields, layers in y and z, 1000 steps | largest rise per 100 steps 0.0e+00, E_end / E_0 4.30e-01 | the same |
| hot plasma leaving through y and z, Gauss law outside the layers | max\|dGauss\| 5.4e-16, 29 % left | 5.2e-16, 29 % left |

- GPU = CPU on all five lines.
- The handover expected 5.1e-6 for the normal-incidence reflection. Both builds give 5.4e-6 on this
  commit, so the expected value is out of date.
- `tests/regression/run.sh build-gpu/vlpl`: all 5 cases match the references, in 9.5 s.

**Streaming deck** (448 steps, GPU):

| run | wall | push_deposit | rip_cc | migrate | other |
|---|---|---|---|---|---|
| periodic (as in the handover) | 0.6 s | 0.39 s | 0.15 s | 0.01 s | 0.01 s |
| periodic, `electrons.uth_y=0.3` | 0.6 s | 0.39 s | 0.15 s | 0.01 s | 0.01 s |
| PML in y and z, 8 cells, `uth_y=0.3` | **1.0 s** | 0.39 s | **0.10 s** | **0.53 s** | 0.02 s |
| the same PML run on the CPU (20 threads) | 16.4 s | 12.54 s | 1.88 s | 1.43 s | 0.56 s |

- **rip_cc with layers is not slower: 0.10 s against 0.15 s periodic.** The open y and z boundaries
  need no periodic halo copy, which likely saves more than the PML kernels cost.
- **migrate goes from 0.01 s to 0.53 s (1.2 ms per step) with open y and z boundaries.** This is now
  half of the step. It is the removal of the particles that leave through the layers, not the hot
  plasma itself: periodic with uth_y = 0.3 stays at 0.01 s. It is likely done on the host, like the
  old tile sort, and is worth moving to the device.
- **CPU vs GPU** `diag.txt` of the PML run (hot plasma):
  - W_field, W_kin and W_total agree to 1.2e-6 relative.
  - N_electrons differs by at most 5 of 352 000: round-off decides a few exits in a hot plasma.
  - max|dGauss| is at most 1.8e-13 in both.

## Fourteenth check on the RTX 5090: openPMD output and restart (main 0d901d4)

Done on 2026-10-27 on the GPU workstation (RTX 5090, CUDA 13.0, `-arch=sm_120`), main 0d901d4. It contains
`openpmd` (PR #42), so main was used. Logs in `$WORK/rip-pic-14/logs14/` on the CPU cluster.

**HDF5 path.** The handover's `find ... | head -n 1` picks the source tree
(`gpu/opt/src/hdf5-1.14.6/src/hdf5.h`) first. The build used `HDF5_DIR=$WORK/gpu/opt/hdf5`
(`include/hdf5.h`, `lib/`) and that `lib` in `LD_LIBRARY_PATH`. A `-path '*/include/hdf5.h'` in the
find would pick the right one.

**Tests**
- `build-gpu/tests openpmd`:
  - "openpmd: output read back = the arrays of the run, units, staggering": ok, E B J 0.0e+00, rho
    3.5e-15, units 0.0e+00, rho - sum rho_s 3.6e-15, 160 beam particles (max error 0.0e+00).
  - "openpmd: restart = uninterrupted run (window, pml, laser, plasma, delayed beams)": ok,
    checkpoint at step 60, window shifts 76, max relative difference 2.2e-14, beam files equal.
- `build-gpu/tests`: 85 ok, 0 failures, 9.1 s.
- `tests/regression/run.sh build-gpu/vlpl`: all 5 cases match the references. The self-test passes.

**Runs** (`examples/lwfa_window.deck`, `grid.t_end=100fs`, `output.slice_every=0`):

| run | steps | wall | push_deposit | rip_cc | moving_window | migrate | other |
|---|---|---|---|---|---|---|---|
| without output | 600 | 2.9 s | 0.79 s | 0.26 s | 1.70 s (60 %) | 0.09 s | 0.02 s |
| g_a: openPMD every 200, checkpoint every 300 | 600 | 6.4 s | 0.79 s | 0.26 s | 1.72 s | 0.09 s | **3.56 s** |
| g_b: restart from step 300 | 300 | 4.4 s | 0.73 s | 0.13 s | 1.44 s | 0.08 s | 2.03 s |

- Both runs exit with status 0.
- **Output is not close to free:** it adds 3.5 s to a 2.9 s run. g_a writes 4 openPMD files and 2
  checkpoints, 420 MB in all, to $WORK, which is mounted over the network on the GPU workstation. That is about
  0.6 s or 120 MB/s per write. On a cluster file system, or with fewer writes, the share falls.
- g_b's "other" includes reading the checkpoint and writing 2 openPMD files.
- `moving_window` takes 60 % of the plain run (2.8 ms per shift). In the bubble benchmark (eleventh
  check) it was 16 %. This is the next GPU cost to look at, together with migrate at open y/z
  boundaries (thirteenth check).

**Restart = uninterrupted run** (`paste g_a/diag.txt g_b/diag.txt | tail -n 3`):

| step | W_field a = b | W_kin a = b | max\|dGauss\| a / b | max\|divB\| a / b | N_e, N_p a = b |
|---|---|---|---|---|---|
| 560 | 1.201815418726e+05 | 6.386466777138e+02 | 1.183e-14 / 1.067e-14 | 4.938e-15 / 4.668e-15 | 1034582, 1052672 |
| 580 | 1.201081698875e+05 | 6.357304935165e+02 | 1.104e-14 / 1.099e-14 | 5.114e-15 / 4.865e-15 | 1034328, 1052672 |
| 600 | 1.200343309173e+05 | 6.339014876381e+02 | 1.125e-14 / 1.078e-14 | 5.090e-15 / 4.849e-15 | 1033998, 1052672 |

- W_field, W_kin, W_total and the particle counts agree in all 13 printed digits.
- The round-off quantities (max|dGauss|, max|divB|) differ at the 1e-15 level, from the atomic sums of
  the GPU deposit.

**File sizes** (256 x 64 x 64 window):

| file | size | expected (fields + per particle) |
|---|---|---|
| openpmd data_000000.h5, data_000200.h5 | 29.4 MB | about 30 MB of fields; no electrons in the box yet |
| openpmd data_000400.h5 | 47.3 MB | |
| openpmd data_000600.h5 | 58.3 MB | 30 MB + 28 B x 1.034 M electrons = 58.9 MB |
| checkpoint_000300.h5 | 84.5 MB | |
| checkpoint_000600.h5 | 175.6 MB | 60 MB + 56 B x 2.087 M particles = 176.9 MB |

The restart's data_000400.h5 and data_000600.h5 have the same sizes as the uninterrupted run's.
## Next GPU check (fifteenth, short): the moving window after decision 36 (branch window-cost)
The fourteenth check found `moving_window` at 60 % of the plain lwfa run (1.70 s of 2.9 s, 2.8 ms per
shift). The shift is now a flat copy plus an exchange along x, and cold plasma enters without a random
generator. On the GPU workstation:
```
git fetch && git checkout window-cost && git pull
make -j CUDA=1 NVCC=/opt/cuda-13.0/bin/nvcc CUDA_ARCH=-arch=sm_120 HDF5_DIR=$WORK/gpu/opt/hdf5 BUILD=build-gpu
build-gpu/vlpl examples/lwfa_window.deck grid.t_end=100fs output.slice_every=0 output.dir=g15 | tail -n 2
```
Please report the time per operator. Expected: moving_window well below 1.70 s (a few hundred ms or
less), and the diag.txt identical to the fourteenth check's run without output, up to the atomic
round-off of the GPU deposit.

## GPU session on the GPU workstation: moving-window shift on the device (branch gpu-window, 2026-10-28)

From now on GPU and performance work runs in a session with direct access to the GPU workstation (measure ->
change -> measure). This replaces the handover for the fifteenth check: `window-cost` (decision 36) is
on main, and its numbers are in the "main" rows below.

**Profile of main 6b2586a** (nsys, `bench/lwfa_bubble/rip.deck grid.t_end=130fs`, 180 shifts):
- 7 flat copies per shift (`shift_fields_x`): 171 us each, 1.2 ms per shift. That is 252 MB moved per
  array at 1.5 TB/s, so a faster copy is not possible.
- The rest of the 5.3 ms per shift ran on the host: the loop over the entering column (160 x 160 cells,
  8 particles per cell, 2 species, the density function per particle), then the upload (22 us per
  attribute array).

**Change** (decision 55):
- `Field::shift_x` moves the start of the array inside its buffer, which has one (x, y) plane of slack.
  It copies only when the slack is used up.
- Cold species are loaded in kernels (`src/load.*`): count per cell, `cub::DeviceScan`, then a write per
  cell, with the deck expression evaluated on the device (`expr_eval`). The particles and their order are
  the same as before.

**Times** (`output.every=100000 output.slice_every=0`):

| run | build | steps | shifts | moving_window main | moving_window branch | per shift |
|---|---|---|---|---|---|---|
| bench/lwfa_bubble to 200 fs | GPU | 1199 | 600 | 3.20 s | 0.25 s | 5.3 -> 0.41 ms |
| lwfa_window to 100 fs | GPU | 600 | 360 | 0.09 s | 0.04 s | 0.25 -> 0.11 ms |
| lwfa_window to 100 fs | CPU, 20 threads | 600 | 360 | 2.07 s | 0.97 s | 5.8 -> 2.7 ms |

- After the change, nsys counts 0.06 ms per shift in the loader kernels and the scan, and no field
  copies.
- The rest of the 0.41 ms is the x exchange (`slab_op`, item 5 of the brief), the D2H copy of the count,
  and launch gaps.
- The bubble run (to 200 fs) is now dominated by push_deposit (50.6 s) and rip_cc (3.9 s).

**Results unchanged:**
- CPU: the `diag.txt` of lwfa_window to 100 fs is byte-identical to main.
- GPU, bubble to 200 fs: W_field, W_total and the particle counts (114483200 electrons, 14310400 protons)
  agree in all printed digits. W_kin differs in the 13th digit (the atomic deposit).
- GPU, lwfa_window: equal in all printed digits apart from max|dGauss| and max|divB| (1e-15 level).
- The bubble's max|dGauss| is 1.051e-4 on main too: that is item 3 (plasma entry), not this change.

**Tests:**
- `build-gpu/tests` and `build-cpu/tests`: all passed, 0 failures. Two of them are new:
  - "window, field storage: shift_x = flat copy over two refills of the slack": 205 shifts, difference 0.
  - "window, cold plasma loaded in kernels = host loop (bounds, slab, density function)": 2150
    particles, positions and momenta exact; weights exact on the CPU and within 1.3e-16 relative on the
    GPU (device `sin`).
- `tests/regression/run.sh`: all 5 cases match on build-gpu and build-cpu.
- `-arch=sm_90` compiles.
- `tests/mpi.sh build-mpi/vlpl` (the CPU cluster, OpenMPI 5.0.10, after merging main e010efb): all 24 cases pass, among them the moving
  window (with CPML, with iRIP, in RZ) 1 vs 8 ranks and the restarts of 8 ranks on 8, 1 and 3 ranks. After
  that merge also: all tests (138 lines, among them the RZ moving window and restart, which shift through
  the same `shift_fields_x`), the regression on both builds, lwfa_window CPU diag.txt = main, sm_90.
- After merging main b9fda69 (2026-10-29: RZ axis shapes, charge-conserving filter, B7; the window decision
  renumbered 53 -> 55): make test passes on the CPU and GPU builds (0 failures), the regression on both
  builds, tests/mpi.sh all 26 cases (now with the RZ current filter case; 1 vs 8 ranks 0.0e+00, Gauss drift
  <= 8e-15), sm_90 compiles.

## Script package for JUPITER / JUWELS Booster (branch jupiter-scripts, 2026-10-29)

**Results 2026-11-01** (JUPITER, 1-16 GH200): all checks pass; timings and reading in `bench/jupiter/README.md`,
section "Results on JUPITER, 2026-11-01"; logs in `bench/jupiter/results_2026-11-01/`.

`bench/jupiter/` is the Slurm package for the first multi-GPU run, made for Thomas Wilson (JSC), after
GEM-PIC-RZ's `benchmarks/jupiter/`. Its README is the step-by-step guide.
- `env.sh` holds the settings: `ACCOUNT` (must be set), partition, `CUDA_ARCH` (sm_90 GH200, sm_80 A100),
  cores per rank and `MODULES`.
- `build.sh` builds two copies with the same options: `build-gpu-mpi` (CUDA, MPI, HDF5) and
  `build-cpu-mpi` (OpenMP, MPI, HDF5).
- `check.sbatch` (1 node, 4 GPUs) checks correctness:
  - the unit tests of both builds;
  - `tests/regression/run.sh` with both binaries;
  - `tests/mpi.sh` with one GPU per rank, 1 vs `NP` = 4 ranks (`NP=8` with `--nodes=2`), and the same with
    the CPU build;
  - then every GPU `diag.txt` against the CPU one (`tests/regression/compare.py`).
- `submit_scaling.sh` + `scaling.sbatch` time three runs on 1, 4, 16 GPUs (64 optional):
  - `bench/lwfa_bubble` to 300 fs, the same box on every count (strong);
  - the same with ny = nz = 160 k on k^2 GPUs, ranks only across the laser (weak);
  - the B7 deck `bench/fbpic_nm_scan/rip_rz.deck` at Nm = 10, ranks along z (strong).
- `cpu_reference.sbatch` runs the two strong cases on one CPU node.
- `collect.sh` turns all logs into a Markdown table: time per step, speed-up, efficiency, and ms per step of
  every operator from the "time per operator" line.
- `tests/mpi.sh` takes `NP` (default 8, unchanged). Its HDF5 probe (`--check`) now starts through `$MPIRUN`,
  because compute nodes start MPI programs only with srun.

**Dry run on the CPU cluster.** One Slurm job on partition <partition> (1 node, 24 cores) ran the scripts with stand-ins:
- `sbatch` runs the job script at once with the `SLURM_*` variables;
- `srun -n N` becomes `mpirun -np N`;
- `nvidia-smi` prints a GH200 line;
- `nvcc` drops the CUDA options and compiles with `-ccbin`, so the "GPU" build is a CPU MPI build made
  by the same make command.

Results:
- `build.sh`: both builds.
- `check.sbatch`, about 73 min:
  - unit tests: all pass apart from `block_add` (g++ 15; fixed since, see below);
  - regression: all 7 cases ok with both binaries;
  - `tests/mpi.sh` 1 vs 4 ranks: all 26 cases on both builds (energies <= 1.3e-11, Gauss drift <= 7.5e-15,
    restarts of 4 ranks on 4 = the uninterrupted run exactly);
  - 52 of 52 diag.txt GPU stand-in = CPU.
- `submit_scaling.sh 1 4 16` with `T_BUBBLE=5fs T_RZ=5fs`: all runs completed, and `collect.sh` gives the
  table.

The dry run found one bug, now fixed: srun forwarded the case list of `tests/regression/run.sh` (stdin) to
the rank, so only the first case ran. The wrapper now reads stdin from /dev/null.

**Real GPU, one rank** (the GPU workstation, RTX 5090, sm_120 build, the srun stand-in starts the binary): `scaling.sbatch`
with NGPU = 1, `T_BUBBLE=110fs T_RZ=150fs`. All three runs completed and `collect.sh` made the table. The
GPU was shared with the B7 runs, so the times only show that the logs are parsed:

| case | steps | s/step | per operator, ms/step |
|---|---|---|---|
| bubble_strong | 660 | 0.0285 | push_deposit 5.02, rip_cc 11.61, moving_window 3.20, migrate 8.39, other 0.21 |
| bubble_weak (= strong on 1 GPU) | 660 | 0.0280 | push_deposit 4.73, rip_cc 11.56, moving_window 3.20, migrate 8.32, other 0.23 |
| rz_nm10 | 899 | 0.0419 | rz_push_deposit 7.64, rz_fields 16.96, moving_window 8.03, migrate 8.19, other 1.08 |

**sm_90 compile checks:**
- `make CUDA=1 CUDA_ARCH=-arch=sm_90` (nvcc 13.0, g++ 14.1) builds. The only warning is the host warning
  about parentheses at `tests/test_rz_grid.cpp:516`, which `main` already has.
- CUDA + MPI: all 29 sources compile for sm_90 with `-DRIP_MPI` against the OpenMPI 5.0.10 headers. They
  are not linked, because there is no MPI on the GPU workstation.

`make test` and `make regression` (g++ 14.1 on the GPU workstation): all passed, 0 failures, all 7 cases.

**Fixed on the way: `block_add` with g++ 15.**
- With g++ 15.3 (the CPU cluster) the unit test `block_add: 64-bit sums from 32-bit adds with carry` failed in every CPU
  build, `main` included (4 wrong cells). It passed with `-fno-strict-aliasing` and with g++ 14.
- Cause: the CPU path of `block_add` (`src/portable.hpp`) wrote the two 32-bit halves through `unsigned*`
  into scratch allocated as `double` and read back as `long long`. That breaks strict aliasing, so g++ 15
  -O3 could move the read before the writes. The box deposit (opt-in) uses this path.
- Fix: on the CPU (one thread per block) it is now one 64-bit add through `memcpy`. The device path (two
  native 32-bit shared atomics) is unchanged.
- After the fix, g++ 15.3 on one cluster node: `block_add` ok, `make test` 0 failures, `make regression` all
  cases, `build-mpi/tests` 0 failures, `tests/mpi.sh` all 26 cases (1 vs 8 ranks: energies <= 1.2e-11, Gauss drift <= 7.8e-15, restarts exact).
- RTX 5090 (sm_120): `build-gpu/tests` 0 failures, regression all cases; sm_90 compiles.

Not covered by these checks: several real GPUs, the link of the CUDA + MPI build, the JSC module names and
the Slurm GPU binding. That is what Thomas's runs will show.

## Multi-rank check on the RTX 5090: guard and particle exchanges (branch gpu-exchange)

What changed (after the JUPITER runs of 2026-11-01, where the exchanges were the losses; design above):
- `Comm::halo`: one pack kernel for all fields of a call (was one per field), both directions of an axis in
  flight at once (MPI_Irecv / Isend instead of two blocking MPI_Sendrecv), pinned host staging buffers (were
  pageable `std::vector`s), or device pointers straight to a CUDA-aware MPI with `VLPL_CUDA_AWARE=1`.
- Particle migration: the device list of leaving particles and the pack scratch are persistent (were a
  `cudaMalloc` and a memset of n ints every step), the arrivals go up in one copy (were one per attribute),
  and the messages of both directions are in flight at once (`Comm::exchange_bytes`, was four blocking
  sendrecvs per axis).
- The log ends with `exchanges (max over ranks): guard cells <s> in <n> calls (MPI wait <s>), particle
  messages <s>; <mode>`; `bench/jupiter/collect.sh` puts it in ms per step into its table.

Checks (RTX 5090, sm_120, CUDA 13.0; the CUDA + MPI build with the CUDA-aware OpenMPI 5.0.11 of the conda
environment `fbpic12`):
- `build-gpu/tests`: 0 failures; regression: all cases.
- `NP=2 RESTART_EXACT=0 sh tests/mpi.sh build-gpu-mpi/vlpl` (2 ranks sharing the GPU, host staging):
  all 40 cases pass (energies <= 1e-8 relative, most 0 or <= 2e-10; Gauss drift <= 4.4e-14; restarts skipped:
  no HDF5 in this build).
- `VLPL_CUDA_AWARE=1`: not testable on cuda. The OpenMPI's CUDA accelerator component fails in `opal_init`
  (`opal_accelerator_base_select failed` with `--mca accelerator cuda`), so the transports treat device
  pointers as host memory and segfault. To be tested on JUPITER (`check.sbatch` with `VLPL_CUDA_AWARE=1`,
  bench/jupiter/README.md).
- sm_90 (GH200): the CUDA + MPI build compiles.
- CPU (g++ 15.3, epyc node): `build-cpu/tests` 0 failures, regression all cases, `tests/mpi.sh` all 48 cases
  (1 vs 8 ranks: energies <= 1.2e-9, Gauss drift <= 4.2e-14, restarts exact), 80 min.
- CPU, one rank (O4 foil, 480 x 128 x 10 cells, 18 steps, cluster node): 53.5 s against 54.1 s on `main`.
  A first version was 27 % slower: with nz = 10 and the guards of shape 3 a z slab is almost the whole
  field, and the fused pack kernel (a division per cell) and a copy of the self-periodic slab cost a pass
  each; the CPU now uses the per-field loops and the self-periodic slab is unpacked from the send buffer.
- Side remark: `mpirun -np 1` binds the rank to one core (OpenMPI's default for <= 2 ranks), so its OpenMP
  threads share that core: the same run takes 54 s with 1, 2 or 4 threads, 28 s with `--bind-to none` and 2.

Not covered: real multi-GPU timings (JUPITER), the CUDA-aware path, inter-node messages.

## Multi-GPU overlap (branch gpu-overlap, 2026-11-09; decision 92)

What changed (after the JUPITER runs of 2026-11-03, where on 16 GH200 the particle migration took 3.6 ms and the
field solve 2.2 ms of a 10.5 ms step, both mostly exchanges):
- **Push overlapped with the current exchanges** (GPU, several ranks; `VLPL_OVERLAP=0` turns it off): the boundary
  particles (cell within 2 guard widths of a face) are pushed first in a high-priority stream and mark the interior
  ones; the interior particles follow in a low-priority stream while the current guards are summed (`add_guards`) and
  copied back (the first exchange of the RIP-CC step, moved into the push): 6 of the 14 halo calls of a step.
- **Every guard exchange in its own high-priority stream** ordered by events after the caller's stream (no device-wide
  synchronisation, asynchronous staging copies); an axis periodic with the rank itself: one kernel instead of a pack
  and two unpacks.
- **RIP-CC predictor computed into the guards** (one cell) instead of exchanged: 3 halo calls per step fewer.
- **Migration on the device**: flags, CUB scan, one pack and one hole-fill kernel for all attributes, one copy to the
  host for all species; one message per axis and direction for all species, one MPI round each (MPI_Mprobe); the host
  keeps the staying rows as segments (no copies). 12 MPI rounds per step became 3.
- **Step profiler** `VLPL_PROFILE=1` (`VLPL_PROFILE_FROM`, `VLPL_PROFILE_SYNC`, NVTX ranges): ms per step of every
  operator and its parts, max / mean over the ranks.

### Correctness

- CPU (g++ 15.3, the CPU cluster): the new build against `main` 755e386 on the same decks and rank counts: `diag.txt` **identical
  byte for byte** in all 9 cases (moving window LWFA on 1, 4, 8 ranks; streaming plasma on 1 and 8; open x on 8; RZ
  plasma on 1 and 4; the bubble deck, small box, window, 8 ranks). So the migration order, the self-periodic exchange
  and the predictor in the guards change nothing on the CPU, where the push overlap is off.
- CPU `tests/mpi.sh` 1 vs 8 ranks (4 Slurm segments): all cases pass, restarts included.
- GPU (RTX 5090, sm_120, CUDA 13.0, OpenMPI 5.0.11 of the conda environment fbpic12): `tests/mpi.sh` 1 vs 2 ranks on
  the one GPU (two processes through a user MPS daemon, host staging), with the overlap on: **all 91 comparisons pass,
  exit 0** (energies 0 ... 9.9e-9 relative, most 0 or <= 2.4e-10; Gauss drift <= 1.1e-13); the 4 restart cases skipped (that build
  had no HDF5). The first version failed here (streaming plasma: W_field 1.3e4 instead of 7.4e-4): a boundary particle
  moved into the interior by the first kernel was pushed again by the second; the boundary kernel now marks the
  interior particles from the positions before the push (test_overlap O4 checks that no particle is pushed twice).
- Unit tests `test_overlap` (CPU and GPU): the device migration equals the former host algorithm exactly (order, four
  attribute types, lost energy); all species in one migration equal one by one; the self-periodic exchange gives the
  exact copies / sums; the two-part push equals the one-kernel push (particles exactly, currents to round-off).
- sm_90 (GH200): the CUDA + MPI build compiles.
- nsys (2 ranks, 2 x 1 x 1): 64 % of the GPU time of the exchange operations (pack, unpack, staging copies) runs while
  the interior push kernel runs.

### Timings on the RTX 5090

The bubble deck with the plasma from x = 0 and the window from 20 fs (600 x 96 x 96 cells, 5.0e7 particles), 240
steps, counted from step 150 (the window moving, the box full). One RTX 5090; the 2-rank runs share it through MPS;
the GPU was shared with other sessions' jobs during all runs (FBPIC, test suites): the totals scatter by +-15 %
between repetitions. FP64 on the GeForce is 1/64, so the push is 150-230 ms per step here and dominates.

Per operator with `VLPL_PROFILE_SYNC=1` (every scope synchronised, so no overlap; ms per step, slowest rank):

| run | step | push_deposit | rip_cc | moving_window | migrate (device / host route / MPI) |
|---|---|---|---|---|---|
| 1 rank, main | 212 | 168.0 | 59.2 | 20.7 | 19.5 (16.8 / 0.4 / 0.0) |
| 1 rank, gpu-overlap | 211 | 185.9 | 36.1 | 25.6 | 14.4 (11.2 / 3.3 / 0.0) |
| 2 ranks along x, main | 252 | 185.5 | 77.5 | 28.6 | 26.5 (19.0 / 5.7 / 4.8) |
| 2 ranks along x, gpu-overlap | 219 | 198.2 (incl. both current exchanges 25.7) | 28.2 | 27.5 | 15.1 (9.2 / 5.9 / 1.0) |
| 2 ranks along y, main | 231 | 177.6 | 66.2 | 21.7 | 20.3 (17.3 / 0.3 / 0.1) |
| 2 ranks along y, gpu-overlap | 221 | 208.0 (incl. both current exchanges 27.5) | 31.6 | 22.6 | 12.3 (9.7 / 3.5 / 0.9) |

(step = wall time per step over the run; main = `main` 755e386 plus the profiler only. In the synchronised profile
every halo sub-scope waits for the device, which on a GPU shared by several processes costs ~1-2 ms per call: the
absolute exchange times are inflated, the differences are real. The push differences, 168 -> 186 ms on one rank where
nothing changed in the push, are the other jobs on the GPU.)

- **Migration** 19.5 -> 14.4 ms (1 rank), 26.5 -> 15.1 ms (2 ranks along x): the device part (list, sort, pack, holes
  = 16.8 ms before) is 11.2 ms; MPI 4.8 -> 1.0 ms (3 rounds instead of 12). Unsynchronised, the migrate scope also
  absorbs the wait for the other rank: 2 ranks along x 121 / 79 ms (main, two repetitions) -> 51 / 35 ms (overlap on)
  and 112 / 88 ms (overlap off).
- **Field step** 59 -> 36 ms (1 rank), 78 -> 28 ms (2 ranks along x), 66 -> 32 ms (2 ranks along y): the predictor is
  not exchanged, the current copy-back moved into the push, the self-periodic axes take one kernel.
- **Totals** (unsynchronised, s for 240 steps, two repetitions): 1 rank 55.3 / 58.6 (main), 56.3 / 58.2 (gpu-overlap);
  2 ranks along x 61.6 / 53.8, 59.3 / 43.1 (overlap off 56.0 / 52.0); along y 62.7 / 51.7, 53.4 / 58.8 (overlap off
  58.7 / 62.9). With two ranks on one GPU the other rank's push fills the GPU while a rank exchanges, so the overlap
  cannot show here; the time per step is set by the push (FP64 on the GeForce).

What these numbers cannot show: the gain of the overlap with one GPU per rank and real network latencies. That needs
the JUPITER rerun (bench/jupiter/README.md, "Note for Thomas": strong and weak series with the overlap on and off,
CUDA-aware; the steady-state table of collect.sh splits the step into push, the overlapped exchanges, what is left
of the interior push, field step, window and the migration parts).

## GPU-only failures on main after gpu-overlap: causes and fixes (branch gpu-fixes, 2026-11-11)

The real-GPU check of `main` (632c24a, RTX 5090, sm_120, CUDA 13.0) found 6 unit-test failures that the CPU build
does not have, and a crash of the GPU `tests/mpi.sh` in the Breit-Wheeler case. `gpu-overlap` (ae88efc) is clean.
Three independent causes, and a fourth in the test script that only the GPU `mpi.sh` reaches:

1. **RZ restart = uninterrupted run (rip_cc, irip): a test problem, not a restart problem.** Bisected to the
   filter-gauss merge (decision 91). Fields and particles of the restarted run agree with the uninterrupted run to
   3e-14; only the stored initial Gauss residual `gauss0` differed (0.4-0.85 relative). Since decision 91 every
   window shift adds the charge of the column that leaves the frozen cells into `gauss0` (`window_fold_entered`),
   deposited with atomic additions. For the neutral plasma of the test that charge cancels to round-off of rho,
   so on the GPU (other summation order) the difference is round-off of rho too; the test divided it by the
   largest value of `gauss0`, itself round-off. The test now compares `gauss0` relative to max |rho| of the
   electrons: 3.1e-14 / 2.6e-14 on the GPU, still exactly 0 on one CPU thread.
2. **Breit-Wheeler table on the host: crash with MPI.** `qed_setup` copied the emission tables `fy`, `fc` to the
   device but left `fb` (T0, T3, Mb of the pair rate) as a host pointer in the kernel's table view. The RTX 5090
   machine runs in HMM addressing mode, so a single process reads host memory from a kernel and the unit tests
   ran; two ranks sharing the GPU through MPS have no HMM: "illegal memory access". Whether a GH200 (JUPITER,
   coherent host memory) would have read it was not checked. `fb` is now uploaded like the others.
3. **Pairs P5 / P6 (energy balance -14 to -16 %, the reversed beam gives other pairs): a ptxas bug for sm_120.**
   The random numbers of a sub-step loop (`qed_emit`, `qed_pair`: `ion_uniform(key + 4 k)`) came out different from
   the CPU's for the same key on the device: `kk` had the right bits, `ion_uniform(kk)` inside the loop did not.
   The counting pass and the writing pass of the pair decay then disagreed (in step 0 of P5, 21 of 114 positrons
   were never written, while their photons were removed): energy lost, pair weights wrong. Evidence, one emitting
   particle and 2000 particles with identical inputs, host vs device:
   | device code | particles that differ from the host |
   |---|---|
   | sm_120 SASS, CUDA 13.0 ptxas (also with `-Xptxas -O1`, also with `-fmad=false`) | 1392 of 2000 |
   | sm_120 SASS, CUDA 12.9 ptxas | 1392 of 2000 |
   | compute_90 PTX compiled to sm_120 by CUDA 13.0 ptxas | 1392 of 2000 |
   | compute_90 or compute_120 PTX, JIT-compiled by the driver (580.95) | 0 |
   | sm_120 SASS, CUDA 13.5 ptxas | 0 |
   | sm_120 SASS, CUDA 13.0, `ion_mix` not inlined on the device | 0 |
   So NVVM's PTX is correct and the offline ptxas of CUDA 12.9 / 13.0 miscompiles the inlined splitmix64 for sm_120.
   Fix: `ion_mix` is `__noinline__` in device code (src/ionization.hpp); with CUDA 13.0 it gives the CPU's numbers
   (checked on sm_120 only). Whether the sm_90 ptxas of CUDA 12.9 / 13.0 has the same bug could not be checked here
   (no sm_90 GPU): on JUPITER, compare the pair tests of test_pairs with the CPU, or build with CUDA >= 13.5.
   With the fix P5 balance -5.1e-15 (Cart3D, 5255 positrons, as the CPU) and -5.9e-15 (RZ), P6 identical pairs.
4. **GPU `tests/mpi.sh` with `RESTART_EXACT=0`: the O7 restarts compared `medium.txt` with the `diag.txt` checker**
   (`same_run` -> `compare`, which reads column 5 as the Gauss drift; in `medium.txt` it is the number of solves,
   40): "Gauss law drift 40 on 2 ranks" although the restarted `medium.txt` was byte-identical. On the CPU
   (`RESTART_EXACT=1`, `cmp`) this path is never taken. `same_run` now uses `mcompare` for `medium.txt`.

The RZ implicit mover test "hot plasma column, m = 0, 1" (rare unconverged steps by design, decision 82) failed once
in the first check (|E/E0 - 1| 2.9e-2, 3e-2 of E0 lost at the wall) and passed in all 10 later runs (3 at 57bf17e,
3 at 27d15fa, 1 on `main`, 1 at d190, 2 on gpu-fixes; |E/E0 - 1| up to 5.5e-4): which steps stay unconverged depends on the summation order, which on
the GPU changes from run to run. Not changed here.

Check of gpu-fixes after the merge of main 96de99b (RTX 5090, CUDA 13.0): the GPU build for sm_120, the CPU build and
the GPU MPI builds for sm_120 and sm_90 compile; GPU unit tests all passed, 0 failures (rz restart 3.1e-14 / 2.2e-14;
mover m = 0, 1 |E/E0 - 1| 2.0e-5, 2 of 30 steps unconverged; P5 balance -5.3e-15 Cart3D, 5255 positrons, -5.9e-15 RZ;
P6 identical, 4021 positrons); GPU regression all cases match; GPU `tests/mpi.sh` NP=2 through MPS, RESTART_EXACT=0:
exit 0, 146 comparisons (Breit-Wheeler cascade: energies identical, qed.txt 2.7e-17, pair weight 1.9496746777e-02 on
1 and 2 ranks); CPU unit tests 0 failures, CPU regression all cases match, CPU `tests/mpi.sh` NP=8 (before the merge,
4 Slurm segments) exit 0.
