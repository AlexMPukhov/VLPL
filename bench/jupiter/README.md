# VLPL on JUPITER / JUWELS Booster: first multi-GPU runs and timings

VLPL runs one MPI rank per GPU. Every rank keeps its block of the grid and its particles on its GPU for the
whole run. Guard cells go to the neighbours through pinned host memory, or straight from GPU memory with
`VLPL_CUDA_AWARE=1` (CUDA-aware MPI); leaving particles go through host memory.
The first multi-GPU runs (JUPITER, 1-16 GH200, 2026-11-01) passed every correctness check; results and timings in
the section "Results on JUPITER, 2026-11-01" below. The MPI code is also tested on CPUs (`tests/mpi.sh`). These scripts do
the first multi-GPU run, in two parts:

| part | script | resources | what it shows |
|---|---|---|---|
| 1 | `check.sbatch` | 1 node, 4 GPUs, ≤ 2.5 h (about 1 h) | GPU = CPU build, on 1 and on 4 GPUs (every case of `tests/mpi.sh`) |
| 2 | `submit_scaling.sh` | 1, 4, 16 GPUs (1, 1, 4 nodes) + 1 CPU node; 64 GPUs optional | time per step and per operator, strong and weak scaling |

Run part 2 only after part 1 has passed.

### Note for Thomas: what to run for the overlap version (branch gpu-overlap, 2026-11-09)

Since the 2026-11-03 runs the current guard exchanges run while the interior particles are pushed, the particle
migration runs on the device with one message per axis, and every log ends with a step profile (decision 92). Please
rerun with the branch `gpu-overlap` (commit ae88efc, the version checked for this note; `main` has newer physics,
e.g. the binomial current filter on by default, which would change the decks and the timings against 2026-11-03).
In the clone of 2026-11-03 (the repository was renamed RIP-PIC -> VLPL; GitHub redirects the old address):

```
git fetch origin && git checkout gpu-overlap && git pull     # git log -1 shows ae88efc
sh bench/jupiter/build.sh                                    # rebuild both copies
sbatch --account=<project> --export=ALL,VLPL_CUDA_AWARE=1 bench/jupiter/check.sbatch     # correctness, about 1 h
CUDA_AWARE=1 sh bench/jupiter/submit_scaling.sh 1 4 16                                    # strong + weak, CPU node too
CUDA_AWARE=1 OVERLAP=0 NO_CPU=1 sh bench/jupiter/submit_scaling.sh 4 16                   # the same without the overlap
sh bench/jupiter/collect.sh > table.md
```

- Every GPU job runs `bubble_strong`, `bubble_weak` and `rz_nm10` (as before), so the second line gives both the
  strong and the weak series; the third line is the comparison without the overlap (rows "N GPU CUDA-aware no
  overlap"). If time allows, also the host-staging pair: `sh bench/jupiter/submit_scaling.sh 4 16` and
  `OVERLAP=0 NO_CPU=1 sh bench/jupiter/submit_scaling.sh 4 16`.
- `collect.sh` prints a second table, "Steady state": ms per step of the operators and of the exchange parts on the
  slowest rank, counted from step 1250 (the box full of plasma). Please send `table.md` and the logs as before.
- If a run of `check.sbatch` fails only with the overlap, `VLPL_OVERLAP=0` exported in the job switches it off.

## 0. Once: get the code and build it

On a **login node of the system you run on**. JUPITER is aarch64 (Grace + GH200); JUWELS Booster is x86 +
A100.

```
git clone https://github.com/AlexMPukhov/VLPL.git && cd VLPL
git checkout gpu-overlap              # the version of the note above; later runs: stay on main
# edit bench/jupiter/env.sh: ACCOUNT (your compute project)
#   JUWELS Booster also: CUDA_ARCH=sm_80 and CPUS_PER_TASK=12
sh bench/jupiter/build.sh             # a few minutes
```

`build.sh` loads `MODULES` (default `GCC OpenMPI CUDA HDF5`) and builds two copies, each with `vlpl` (the
code) and `tests` (the unit tests):

- **`build-gpu-mpi`:** CUDA for `CUDA_ARCH` (sm_90 = GH200), MPI, HDF5;
- **`build-cpu-mpi`:** CPU with OpenMP, MPI, HDF5. This is the reference for every GPU result, and the CPU timing.

If a module has another name, set `MODULES`; find the names with `module spider CUDA` and `module spider HDF5`
(e.g. `MODULES="Stages/2025 GCC OpenMPI CUDA HDF5"`). Without HDF5: `HDF5=0 sh bench/jupiter/build.sh`. Then the
restart cases of part 1 are skipped.

The other settings in `env.sh`:

| variable | default | meaning |
|---|---|---|
| `ACCOUNT` | `CHANGE_ME` | compute project (`sbatch --account`); **must be set** |
| `PARTITION` | `booster` | GPU partition; `develbooster` for short tests (check `sinfo -s`) |
| `CPU_PARTITION` | = `PARTITION` | partition of the CPU reference (its GPUs stay idle) |
| `GPUS_PER_NODE` | 4 | GPUs per node |
| `CPUS_PER_TASK` | 72 | host cores per rank (JUPITER 4 × 72, JUWELS Booster 4 × 12) |
| `CUDA_ARCH` | `sm_90` | `sm_80` on JUWELS Booster |
| `CPU_ARCH` | empty | `ARCH` of the CPU build, e.g. `-mcpu=neoverse-v2` (Grace) for a faster CPU reference |

Every variable can also be given on the command line, e.g. `ACCOUNT=myproj sh bench/jupiter/submit_scaling.sh`.

## 1. Correctness on the GPUs

From the VLPL directory (the jobs find the code through the submit directory):

```
sbatch --account=<project> bench/jupiter/check.sbatch          # --partition=develbooster for a quicker start
```

The log `vlpl-check.<job>.log` has five sections; outputs are in `bench/jupiter/runs/check.<job>/`.

0. **`nvidia-smi`:** the 4 GPUs of the node.
1. **Unit tests** of `build-gpu-mpi` and of `build-cpu-mpi`, one rank each. Both must end with
   `all passed`. Numbers that differ in the last digits between the two are expected (the order of the
   atomic additions differs on the GPU; all differing lines are in `tests_diff.txt`); a `FAIL` is not, and
   the log lists every `FAIL` line.
2. **Regression cases** (`tests/regression/run.sh`), GPU and CPU, one rank each: 7 cases, each `ok` against
   the stored references.
3. **`tests/mpi.sh` with one GPU per rank:** every case runs on 1 and on 4 ranks (4 GPUs) and must agree:
   - the 3D cases: RIP-CC, Yee, iRIP, open x, CPML, moving window, beam self-field;
   - the restart from a checkpoint of 4 ranks on 4, 1 and 3 ranks;
   - the RZ cases (azimuthal modes, ranks along z).

   Each case prints one `mpi (...)` line, e.g. `1 vs 4 ranks, max relative difference of the energies
   1.2e-13`. The same then runs with the CPU build.
4. **Every `diag.txt` of the GPU runs against the CPU run of the same case** (energies, Gauss law, div B,
   particle numbers; `tests/regression/compare.py`): one `ok` line each.

The last line is `summary: 0 failed check(s)` when all is well.
Expected time: about 1 h. Most of it is the CPU part of section 3 (one core per rank; 33 min on a 24-core x86 node); the GPU part should take minutes.

**Optional, 8 GPUs on 2 nodes** (MPI between nodes):
`sbatch --account=<project> --nodes=2 --export=ALL,NP=8 bench/jupiter/check.sbatch`.

## 2. Timings

```
sh bench/jupiter/submit_scaling.sh          # 1, 4, 16 GPUs and the CPU reference: 4 jobs
sh bench/jupiter/collect.sh                 # the table, when the jobs are done
```

**Three cases.** Every run prints, at the end, `done: <steps> steps, <s> s` and
`time per operator: push_deposit ... s, rip_cc ... s, moving_window ..., migrate ..., other ...` (rank 0).

| case | deck | size | scaling |
|---|---|---|---|
| `bubble_strong` | `bench/lwfa_bubble/rip.deck` (3D LWFA, a0 = 4, moving window) to 300 fs | 600 × 160 × 160 cells, 1.4·10⁸ particles, 1800 steps | the same box on 1, 4, 16 GPUs (strong) |
| `bubble_weak` | the same, `ny = nz = 160 k` for k² GPUs, ranks only across the laser | the same cells and particles per GPU | weak |
| `rz_nm10` | `bench/fbpic_nm_scan/rip_rz.deck` (RZ, 10 azimuthal modes, iRIP) | 800 × 50 cells × 10 modes, 2600 steps | ranks along z, strong (up to 16 GPUs) |

In the bubble deck the plasma enters the box from 107 fs on and fills it from 207 fs on, so the work per step
grows during the run.

**The CPU reference** (`cpu_reference.sbatch`): `bubble_strong` and `rz_nm10` on one node without GPUs,
4 ranks × 72 OpenMP threads (JUWELS Booster: `RANKS=4 THREADS=12`).

`collect.sh` prints a Markdown table: for every case and run the decomposition, the time per step,
the speed-up and efficiency against 1 GPU, and the time per step of every operator.

**Short trial first:**
`T_BUBBLE=120fs T_RZ=150fs NO_CPU=1 sh bench/jupiter/submit_scaling.sh 1 4`
(720 and 900 steps, a few minutes per job).

**64 GPUs (optional):** `sh bench/jupiter/submit_scaling.sh 64` (16 nodes; `bubble_strong` and `bubble_weak`).

**Budget, upper limits** (the `--time` of `submit_scaling.sh`):
- 1 GPU 1.5 h, 4 GPUs 1 h, 16 GPUs 0.75 h: at most about 54 GPU-hours, with 64 GPUs another 48;
- CPU reference 2 h on one node.

The real times should be much shorter: on the RTX 5090, `rz_nm10` takes 2.5 min on one GPU.

**Exchanges: host staging against CUDA-aware MPI.** Every run also prints
`exchanges (max over ranks): guard cells <s> s in <n> calls (MPI wait <s> s), particle messages <s> s; <mode>`;
`collect.sh` gives them in ms per step in the last column. To compare the two paths of the guard cells, run the
4- and 16-GPU jobs twice:
```
NO_CPU=1 sh bench/jupiter/submit_scaling.sh 4 16                # through pinned host memory (default)
CUDA_AWARE=1 NO_CPU=1 sh bench/jupiter/submit_scaling.sh 4 16   # straight from GPU memory: rows "N GPU CUDA-aware"
```
CUDA-aware needs an MPI built with CUDA support (`ompi_info --parsable --all | grep mpi_built_with_cuda_support:value`
must say `true`; with ParaStation MPI: `module load MPI-settings/CUDA`). The results must be the same up to round-off
(check part 1 with `VLPL_CUDA_AWARE=1` exported, if in doubt).

## Results on JUPITER, 2026-11-01 (Thomas Wilson)

GH200 nodes (4 GPUs, 72 cores per GPU), the scripts of this directory on `main` of 2026-10-31 (before the side-tube
plasma, #84, and the Fresnel test change, #83; the logs do not record the commit). Logs and the `collect.sh` table:
`results_2026-11-01/`.

**Part 1, correctness: 0 failed checks** (`rippic-check.2636303.log`, one node, 4 GPUs, 1.6 h).

- Unit tests: all pass on the GPU and on the CPU build. The printed numbers of the two differ in the last digits
  only (order of the atomic additions).
- Regression cases: GPU and CPU match the references; the self-test detects a 1e-6 change of W_field.
- `tests/mpi.sh`, 44 cases, 1 vs 4 GPUs (one per rank): energies agree to 1e-10 (dense slab 1.2e-10, most cases
  0 or about 1e-12), Gauss drift 1e-16 ... 3e-14; restarts 4 → 1 and 4 → 3 ranks pass, the RZ restart is exact.
- Every `diag.txt` of the GPU runs against the CPU run of the same case: 81 compared, 0 differ.

**Part 2, timings** (`collect.txt`; seconds per step, operator times of rank 0 in ms per step):

| case | run | ranks (x y z) | s/step | speed-up | efficiency | push/deposit | field solve | migrate | other operators |
|---|---|---|---|---|---|---|---|---|---|
| bubble_strong | 1 GPU | 1×1×1 | 0.0641 | 1 | | 56.9 | 1.9 | 4.9 | 0.5 |
| bubble_strong | 4 GPU | 2×2×1 | 0.0377 | 1.70 | 42 % | 19.1 | 7.8 | 10.1 | 0.9 |
| bubble_strong | 16 GPU | 4×2×2 | 0.0193 | 3.32 | 21 % | 5.7 | 5.0 | 8.3 | 0.4 |
| bubble_strong | CPU node 4×72 | 2×2×1 | 0.312 | 0.21 | | 165.5 | 17.6 | 12.1 | 116.9 (tile sort 114.5) |
| bubble_weak | 1 GPU | 1×1×1 | 0.0635 | | | 56.3 | 1.9 | 4.9 | 0.5 |
| bubble_weak | 4 GPU | 1×2×2 | 0.0952 | | 67 % (weak) | 62.9 | 21.1 | 10.3 | 1.0 |
| bubble_weak | 16 GPU | 1×4×4 | 0.0885 | | 72 % (weak) | 61.1 | 15.4 | 11.1 | 0.9 |
| rz_nm10 | 1 GPU | 1×1×1 | 0.0174 | 1 | | 13.2 | 2.7 | 0.8 | 0.7 |
| rz_nm10 | 4 GPU | 4×1×1 | 0.0132 | 1.31 | 33 % | 5.7 | 4.5 | 2.7 | 0.4 |
| rz_nm10 | 16 GPU | 16×1×1 | 0.0099 | 1.75 | 11 % | 2.4 | 5.2 | 2.0 | 0.4 |
| rz_nm10 | CPU node 4×72 | 4×1×1 | 0.147 | 0.12 | | 140.3 | 3.5 | 1.6 | 1.8 |

(push/deposit = `push_deposit` or `rz_push_deposit`; field solve = `rip_cc` or `rz_fields`; other = moving window,
beam diagnostics, tile sort, rest. Speed-up = t(1 GPU) / t; for the CPU rows t(1 GPU) / t(CPU node).)

Reading:

- **One GH200 = 4.9× (3D bubble) and 8.5× (RZ, 10 modes) one CPU node** (4 ranks × 72 threads). On the CPU the
  tile sort alone (115 ms per step) costs more than a whole GPU step.
- **The particle push scales**: bubble_strong 56.9 → 19.1 → 5.7 ms on 1, 4, 16 GPUs (3.0× and 10× of 4× and 16×;
  rank 0's share, so part of the gap is load imbalance of the bubble across the ranks). In the weak series the push
  stays at 56-63 ms.
- **The losses are the exchanges between the ranks**: the field solve with its guard exchanges (1.9 ms on one GPU,
  5-21 ms on several) and the particle migration (4.9 → 8-11 ms). On 16 GPUs they are 13 of the 19 ms per step of
  bubble_strong; in the weak series the field solve (15-21 ms) is most of the loss to 67-72 %.
- **rz_nm10 is too small for 16 GPUs**: 17 ms per step on one GPU; on 16 GPUs the field step (5.2 ms) is twice the
  push (2.4 ms). Its strong scaling needs a larger RZ problem.
- Next for the multi-GPU code: guard and particle exchanges without the host copies (CUDA-aware MPI), overlap of
  the exchanges with the push, and load balance of the decomposition.

## Results on JUPITER, 2026-11-03 (Thomas Wilson): host staging against CUDA-aware MPI

The same nodes and cases as on 2026-11-01, on `main` after the exchange rewrite (#92, Comm::halo with one pack kernel,
both directions in flight, persistent migration buffers). The MPI is built with CUDA support
(`mpi_built_with_cuda_support:value:true`). Logs and the `collect.sh` table: `results_2026-11-03/`.

**Part 1, correctness: 0 failed checks**, on both paths:
- `rippic-check.2662797.log` (host staging, steps 1-4): unit tests and regression on GPU and CPU pass;
  `tests/mpi.sh` 53 cases 1 vs 4 GPUs and on the CPU build exit 0 (energies agree to 1e-10 or better); 95 GPU
  `diag.txt` = CPU.
- `rippic-check.2662810.log` (`VLPL_CUDA_AWARE=1`, steps 3-4): `tests/mpi.sh` 53 cases 1 vs 4 GPUs exit 0 (same
  differences as host staging, at most 1.6e-10 on the O4 foil), 95 GPU `diag.txt` = CPU.

**Part 2, timings** (seconds per step; operator times of rank 0 in ms per step; 2026-11-01 = before #92):

| case | GPUs | 2026-11-01 | host staging | CUDA-aware | efficiency (host / CUDA-aware) | field solve, ms (host / CUDA-aware) |
|---|---|---|---|---|---|---|
| bubble_strong | 1 | 0.0641 | 0.0615 | | | 1.75 |
| bubble_strong | 4 | 0.0377 | 0.0292 | 0.0233 | 53 % / 66 % | 5.34 / 0.99 |
| bubble_strong | 16 | 0.0193 | 0.0138 | 0.0105 | 28 % / 37 % | 4.79 / 2.18 |
| bubble_weak | 4 | 0.0952 | 0.0843 | 0.0646 | 73 % / 95 % (weak) | 18.42 / 2.71 |
| bubble_weak | 16 | 0.0885 | 0.0781 | 0.0666 | 79 % / 92 % (weak) | 14.36 / 4.76 |
| rz_nm10 | 1 | 0.0174 | 0.0168 | | | 2.73 |
| rz_nm10 | 4 | 0.0132 | 0.0110 | 0.0104 | 38 % / 40 % | 4.11 / 3.73 |
| rz_nm10 | 16 | 0.0099 | 0.0070 | 0.0063 | 15 % / 17 % | 4.12 / 3.80 |

(efficiency = t(1 GPU) / (N t) for the strong cases, t(1 GPU) / t for the weak ones, with the 1-GPU time of this
series; field solve = `rip_cc` or `rz_fields`, which contains the guard exchanges.)

Reading:

- **The exchange rewrite alone** (host staging, against 2026-11-01): 1.1-1.4x per step on several GPUs, weak
  scaling of the bubble 67-72 % -> 73-79 %.
- **CUDA-aware MPI on top**: bubble_strong 1.25x (4 GPUs) and 1.31x (16 GPUs), bubble_weak 95 % and 92 % weak
  efficiency. The gain is in the field solve: its guard exchanges no longer go through host memory
  (bubble_weak on 4 GPUs 18.4 -> 2.7 ms per step, one GPU alone 1.75 ms). rz_nm10 gains 6-11 %: its field step changes little
  (4.1 -> 3.7 ms on 4 GPUs), so the guard copies are not its main cost.
- **What is left on 16 GPUs** (bubble_strong, CUDA-aware, 10.5 ms per step): push 4.6 ms (ideal 56.6/16 = 3.5 ms,
  the rest is the bubble's load imbalance), particle migration 3.6 ms, field solve 2.2 ms.
- **The guard-cell column of `collect.sh` overstates the exchange on several GPUs**: its timer apparently starts before
  the pack kernel, which waits in the stream for the preceding push, so on bubble_weak it reads 61-80 ms per step,
  more than the push itself. The field-solve column and the MPI wait in brackets are the useful numbers.
- Next for the multi-GPU code: overlap of the exchanges with the push, load balance along the bubble, the particle
  migration (3.6 ms on 16 GPUs), and a larger RZ case for its strong scaling. With an MPI that has CUDA support,
  `VLPL_CUDA_AWARE=1` should be the default on JUPITER.

## What to send back

- `vlpl-check.<job>.log`. If a check failed, also the directory `bench/jupiter/runs/check.<job>` (or at least
  its `*.txt` files).
- The output of `collect.sh`, and the logs `vlpl-scale.*.log` and `vlpl-cpu.*.log`.
- Any error at the start (module, `no CUDA device found`, MPI, out of memory). The code prints
  `error (rank N): ...`.

## Known limits of this version

- **Guard exchanges** through pinned host memory or (`VLPL_CUDA_AWARE=1`) from GPU memory; the particles always
  through host memory. Since gpu-overlap (decision 92) the two current exchanges (sums and copies, 6 calls) overlap the
  push of the interior particles; the predictor is no longer exchanged; the transport, Ex / Bx and the window's two x
  exchanges (8 calls per step of a window run) are still on the critical path. RZ (`rz_nm10`) has no push overlap.
- **One GPU per rank.** A rank takes the GPU number (its rank on the node) modulo (the GPUs it sees). With
  `--gres=gpu:4` and 4 ranks per node every rank gets its own GPU.
- **iRIP** (`rz_nm10`, and the iRIP cases of part 1) allows ranks only along x (3D) or z (RZ): its implicit
  solve is a partitioned tridiagonal solve across those ranks.
