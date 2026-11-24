---
name: vlpl-setup
description: Install, build and test the particle-in-cell plasma simulation code VLPL (electromagnetic PIC with the dispersion-free RIP field solver, 3D / 2D Cartesian and quasi-cylindrical RZ geometry, C++17) and choose how to run it - OpenMP threads, MPI ranks, or NVIDIA GPUs (CUDA, one GPU per rank). Use when the vlpl executable is not built yet, when the build or the tests fail, when the user asks how to run VLPL on several cores, on a cluster or on a GPU, how to restart a run, or where its output files go. To choose the solver, write or debug an input deck use the skill vlpl-simulation.
license: BSD-3-Clause
compatibility: Needs git, GNU make >= 3.82 and a C++17 compiler with OpenMP (Linux, macOS, or WSL on Windows). Optional - MPI, the HDF5 C library 1.12 or newer, the CUDA toolkit (nvcc), python3 with numpy and matplotlib.
metadata:
  repository: https://github.com/AlexMPukhov/VLPL
---

# VLPL: build, test, run

VLPL is one executable, `build/vlpl`, that reads one ASCII input deck. There is nothing to install besides the
build. Tell the user what you are going to do before you clone or build. Build and run the code as it is: **do not
modify its source**; a failing test or a missing feature is reported to the user (and as an issue at
https://github.com/AlexMPukhov/VLPL/issues), not patched.

The name: VLPL is the code. With the field solver it is called VLPL-RIP (explicit, Cartesian), VLPL-iRIP (implicit,
Cartesian), VLPL-RZ (explicit, RZ) and VLPL-iRZ (implicit, RZ); all four are the same executable, selected in the
deck (`grid.geometry`, `numerics.solver`). The first line of every run prints which one runs.

## 1. Is it there already?

```
<repo>/build/vlpl                      # without arguments: prints "usage: vlpl deck [--check] ..."
```

If the user has a clone, build it in place (section 2 without the clone). Wherever this skill writes `build/vlpl`,
use the path of the user's executable.

## 2. Clone and build

```
git clone https://github.com/AlexMPukhov/VLPL.git
cd VLPL
make -j2                               # build/vlpl and build/tests (CPU, OpenMP)
```

The Makefile compiles all sources in one compiler call per executable: the build takes about 20 minutes
(measured with g++ 15 on an 8-core login node) and prints nothing while it works. `-j2` builds the two executables at the same time.

| build | command | result |
|---|---|---|
| CPU, OpenMP threads | `make` | `build/vlpl` |
| CPU tuned to this machine | `make ARCH=-march=native BUILD=build-native` | about 10 % faster; results differ at round-off; not portable to other CPUs |
| MPI (domain decomposition) | `make MPI=1 BUILD=build-mpi` | `build-mpi/vlpl`, needs `mpicxx` |
| one NVIDIA GPU | `make CUDA=1 CUDA_ARCH=-arch=native BUILD=build-gpu` | `build-gpu/vlpl`, needs `nvcc` |
| several GPUs, one per MPI rank | `make CUDA=1 MPI=1 CUDA_ARCH=-arch=sm_90 BUILD=build-gpu-mpi` | `build-gpu-mpi/vlpl` |

- **Required:** a C++17 compiler with OpenMP (g++ >= 9 or clang) and GNU make >= 3.82 (macOS ships 3.81: use
  `gmake` from Homebrew). On Debian / Ubuntu: `apt install g++ make libhdf5-dev libopenmpi-dev`.
- **HDF5 (optional, 1.12 or newer):** found through `pkg-config`, or give `make HDF5_DIR=/path/to/hdf5` (the
  directory with `include/` and `lib/`). Without HDF5 (`make HDF5=0`, or a warning "building without HDF5" when it
  is missing or too old) the code runs, but openPMD output (`output.openpmd_every`), checkpoints and restarts
  (block `checkpoint`), laser files and beam files are refused at run time.
- **GPU:** `CUDA_ARCH` is the nvcc architecture flag: `-arch=native` on the machine that has the GPU, or the
  compute capability of the target (`-arch=sm_80` A100, `-arch=sm_90` H100 / GH200, `-arch=sm_120` RTX 50xx). With
  an nvcc older than the host compiler add `NVCCFLAGS=-allow-unsupported-compiler`. The GPU executable needs no
  switch in the deck: every kernel runs on the device.
- **Windows:** build and run inside WSL (Ubuntu); the commands are the same.
- **Python (only for plotting and some helper scripts):** python3 with numpy and matplotlib; movies need ffmpeg.

## 3. Test

```
make test                              # the unit tests against exact or analytic references (exit code 0 = all passed)
build/tests laser pml                  # only the test groups whose names contain these words
make regression                        # small versions of the example decks against stored diag.txt (needs python3)
make agent_skill                       # the decks and keys of these skills against the executable
sh tests/mpi.sh build-mpi/vlpl         # MPI build: 1 rank against 8 ranks (long; NP=4 for 4 ranks)
```

- The unit tests print one line per check (`... ok`) and take long: more than an hour on a 24-core node for the
  whole suite. For a quick check after a build run a few groups by name (`build/tests laser window`) and
  `make agent_skill` (a minute). On a cluster run them on a compute node (through the scheduler), not on the
  login node.
- With MPI as root (containers): `OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`. In containers and
  WSL set `OMP_PROC_BIND=false` if the threads end up on one core.
- If a test fails on an unmodified clone, report its name and output to the user; do not edit the tests.

## 4. Run

```
build/vlpl my.deck --check                        # read the deck, print the summary and the warnings, exit: always first
OMP_NUM_THREADS=8 build/vlpl my.deck              # run on 8 threads
mpirun -np 8 build-mpi/vlpl my.deck               # MPI: the decomposition is chosen automatically
mpirun -np 8 build-mpi/vlpl my.deck grid.mpi_x=8  # any deck value can be overridden: block.key=value
build/vlpl my.deck "species electrons.ppc_x=2"    # a named block: quote the argument (it contains a space)
build/vlpl my.deck checkpoint.every=1000          # write checkpoints (needs HDF5)
build/vlpl my.deck --restart latest               # continue from the newest checkpoint in output.dir
```

- **Threads and ranks:** `OMP_NUM_THREADS` threads per rank; ranks x threads = the cores of the job. Cartesian
  3D decomposes along x, y, z (`grid.mpi_x`, `grid.mpi_y`, `grid.mpi_z`, 0 = automatic), 2D along x and y, RZ along z
  only (`grid.mpi_z`). **The implicit solver (`numerics.solver = irip`) decomposes along x only** (RZ: along z):
  leave the automatic choice or set `grid.mpi_y=1 grid.mpi_z=1`.
- **GPU:** `build-gpu/vlpl my.deck` uses the first GPU; with `mpirun -np P build-gpu-mpi/vlpl` every rank takes
  the GPU "local rank modulo the number of GPUs of its node": start as many ranks per node as it has GPUs.
- **Results do not depend on the backend:** MPI runs reproduce the single-rank run to round-off (the initial
  particles are drawn per global cell), GPU runs reproduce the CPU run to round-off.
- **Exit codes:** 0 = finished or stopped by `checkpoint.stop_after_hours`; 2 = an error, printed as
  `error (rank N): <message>` on stderr. The message names the deck key; the skill vlpl-simulation has a table of
  the messages.
- At the end the run prints `done: N steps, T s` and the time per operator (push, deposit, fields, ...).

## 5. Where the output goes

Everything is written to `output.dir` (default `out`), relative to the directory where `vlpl` is started.

| file | content |
|---|---|
| `diag.txt` | every `output.every` steps: step, t, W_field, W_kin, W_total, max\|dGauss\|, max\|rho\|, max\|divB\|, particle numbers |
| `slices/<field>_<step>.f32` | Cartesian only, every `output.slice_every` steps: x-y slices (float32), `meta.txt`, `index.txt`; movie with `python3 tools/movie.py <dir>` |
| `openpmd/data_<step>.h5` | every `output.openpmd_every` steps: fields and particles in openPMD 1.1 (RZ: thetaMode, as FBPIC), readable with openPMD-viewer |
| `checkpoints/checkpoint_<step>.h5` | block `checkpoint`: the whole state, the newest `checkpoint.keep` files are kept |
| `beam_<name>.txt`, `spectrum_<name>.txt` | block `diagnostic`: charge, energy, spread, emittance, sizes; dQ/dE |
| `ionization.txt`, `collisions.txt`, `qed.txt`, `medium.txt`, `xflux.txt` | written when the deck uses that physics |
| `lab_fields/`, `lab_particles/`, `lab_beams.txt` | boosted-frame runs: lab-frame snapshots (block `lab_diagnostics`) |

All numbers in the text files are in code units (lengths 1/k0 = lambda0 / 2 pi, times 1/omega0, fields
m_e c omega0 / e, densities n_c); the columns and the conversion are in the skill vlpl-simulation. Field and particle dumps are large: estimate cells x components x 4 or 8 bytes per
openPMD step before a long run.

## 6. Cluster jobs

Never run simulations or the tests on a login node: build there, then submit. Put no build step into the job
script. With a wall-time limit let the run stop itself and continue from the checkpoint:

```
# job.sh -- started with: sbatch job.sh      (deck: checkpoint { every = 2000 })
srun build-mpi/vlpl run.deck --restart auto checkpoint.stop_after_hours=23.5 > log_$SLURM_JOB_ID.txt 2>&1
grep -q "^stopped after" log_$SLURM_JOB_ID.txt && sbatch job.sh      # the time limit was reached: next job
```

`--restart auto` starts fresh when there is no checkpoint yet and continues otherwise; the text logs continue where
the checkpoint was written. A restart on the same number of ranks continues the run exactly (CPU); another number of
ranks agrees to round-off.

## Citing

The RIP field solver: A. Pukhov, J. Comput. Phys. 418, 109622 (2020). The README of the repository gives the
current reference of the code itself.
