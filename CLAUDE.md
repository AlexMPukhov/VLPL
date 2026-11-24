# CLAUDE.md — working rules for agent sessions on VLPL

VLPL is developed "zero-touch" by Claude for Alexander Pukhov (HHU Düsseldorf). He reviews at
milestone gates only. Plan: `docs/ROADMAP.md`. Current state and next task: `docs/STATUS.md`.

Name: the code is VLPL (formerly RIP-PIC, decision 88; binary build/vlpl, environment variables VLPL_<X>). With the solver:
VLPL-RIP, VLPL-iRIP (Cartesian, RIP-CC / iRIP), VLPL-RZ, VLPL-iRZ (RZ). RIP, RIP-CC and iRIP are
the method names of the field solvers: keep them.

## Start of every session
1. Read `docs/STATUS.md` (what works, what is next, open problems) and this file.
2. Read the spec of the task in `docs/specs/` (write it first if it does not exist, from
   `docs/specs/TEMPLATE.md`: equations, discretisation, tests, references; half a page).
3. Read only the source files the task touches. `docs/decisions.md` explains why things are as they
   are: check it before changing a convention.

## End of every session
- All tests green: `make -j && make test && make regression` (a change meant to change results
  re-blesses `tests/regression/ref/` with `sh tests/regression/run.sh build/vlpl --bless` and says
  why in the pull request); before every pull request also `make MPI=1 BUILD=build-mpi &&
  sh tests/mpi.sh build-mpi/vlpl` (CI runs it only on demand, decision 47; read every line of its
  output and of diag.txt, not only the last); with kernel changes also the CUDA compile
  (`make CUDA=1 CUDA_ARCH=-arch=sm_80 BUILD=build-gpu`, add `NVCCFLAGS=-allow-unsupported-compiler`
  with an old nvcc).
- Update `docs/STATUS.md` (and `docs/decisions.md` for every new convention or trade-off).
- Agent skills (`skills/`, docs/agent_skill.md): after editing the README sections "Units" or "Deck", or an example copied
  into the skill, run `sh tools/sync_skills.sh` (`make test` fails otherwise); with a new feature add its keys and, if it is a
  capability of its own, an example to `skills/vlpl-simulation`, then `make agent_skill`. The skills are public text.
- One feature per branch / pull request; CI (`.github/workflows/ci.yml`) must pass. Keep work in local commits and push once per pull
  request, after the local checks (decision 33). Never start the hosted MPI job without being asked:
  it costs Actions minutes (decision 47).
- Commits: author Alexander Pukhov, plus the attribution lines the session
  asks for.

## Rules for code
- **Every feature comes with an automatic test** (exact result, conservation law, or stored benchmark
  curve with tolerance). Fast tests in `tests/tests.cpp` (< 1 min total on CPU); long physics
  benchmarks go to `bench/` (nightly, GPU).
- **GPU first.** All loops over cells or particles are `par_for` / `for_owned` / `par_sum` kernels from
  `src/portable.hpp`, written as `[=] HD` lambdas that capture only plain values and views (`FV`,
  raw pointers, small POD structs). No kernel lambdas inside private member functions, no `this`
  capture in kernels, no `std::` math in device code (use `sqrt`, `floor`, `exp` ...), no STL inside
  kernels. Host access to device data only through `Field::host()/upload()`, `Species::get()/set()`.
- Files stay small (< ~600 lines); one physics module per file pair, one deck block per module.
- Units: c = 1, lengths 1/k0, times 1/omega0, momenta u = gamma v, fields m c omega0 / e, densities
  n_c (see README). Deck values carry SI units and are converted in `sim.cpp`.
- Staggering and the RIP-CC step are documented in `src/grid.hpp` and `src/fields.hpp`; the solver is
  exactly charge conserving with the Esirkepov deposition on the charge grid — any change must keep
  the Gauss-law test at round-off.

## Machines
- Cloud sessions: 2 cores, no GPU, MPI (OpenMPI, `OMPI_ALLOW_RUN_AS_ROOT=1
  OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`) and nvcc can be installed for compile checks.
- GPU correctness: the GPU workstation (RTX 5090, sm_120, no MPI) and JUWELS Booster / JUPITER
  (sm_80 / sm_90); see `docs/GPU_TESTING.md`.
