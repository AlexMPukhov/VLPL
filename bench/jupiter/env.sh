# Settings of the JUPITER / JUWELS Booster runs (sourced by the other scripts). Edit ACCOUNT first.
# Values in the environment take precedence: e.g.  ACCOUNT=myproj sh bench/jupiter/submit_scaling.sh
ACCOUNT=${ACCOUNT:-CHANGE_ME}            # compute project (sbatch --account)
PARTITION=${PARTITION:-booster}          # GPU partition (JUPITER and JUWELS Booster: booster; short tests: develbooster)
CPU_PARTITION=${CPU_PARTITION:-$PARTITION}  # partition of the CPU reference (the same nodes, GPUs unused)
GPUS_PER_NODE=${GPUS_PER_NODE:-4}        # 4 GH200 per JUPITER node, 4 A100 per JUWELS Booster node
CPUS_PER_TASK=${CPUS_PER_TASK:-72}       # host cores per rank: JUPITER 72 (4 x 72 Grace cores), JUWELS Booster 12
CUDA_ARCH=${CUDA_ARCH:-sm_90}            # sm_90: GH200 (JUPITER), sm_80: A100 (JUWELS Booster)
CPU_ARCH=${CPU_ARCH:-}                   # ARCH of the CPU build, e.g. -mcpu=neoverse-v2 (Grace); empty: portable
MODULES=${MODULES-"GCC OpenMPI CUDA HDF5 Python SciPy-bundle"}   # module names as on the JSC systems (module spider
                                             # if absent); Python + SciPy-bundle: numpy for the postprocessing scripts;
                                             # MODULES="" loads none (environment already set up)
# the VLPL checkout: batch jobs are submitted from it (Slurm runs a copy of the job script elsewhere)
if [ -z "${SRC:-}" ]; then
  if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/src/sim.cpp" ]; then SRC=$SLURM_SUBMIT_DIR
  else SRC=$(cd "$(dirname "$0")/../.." && pwd); fi
fi
BUILD_GPU=${BUILD_GPU:-$SRC/build-gpu-mpi}  # CUDA + MPI build (one GPU per rank)
BUILD_CPU=${BUILD_CPU:-$SRC/build-cpu-mpi}  # CPU + MPI build (the references and the CPU timing)
RUNS=${RUNS:-$SRC/bench/jupiter/runs}    # output of all jobs
# the code was formerly RIP-PIC: old variable and binary names
if [ -n "${RIP_CUDA_AWARE:-}" ] && [ -z "${VLPL_CUDA_AWARE:-}" ]; then echo "warning: RIP_CUDA_AWARE is deprecated, use VLPL_CUDA_AWARE"; export VLPL_CUDA_AWARE=$RIP_CUDA_AWARE; fi
for b in "$BUILD_GPU" "$BUILD_CPU"; do if [ -x "$b/rippic" ] && [ ! -x "$b/vlpl" ]; then echo "note: $b has only the binary rippic (formerly RIP-PIC): rebuild with sh bench/jupiter/build.sh"; fi; done

load_modules() {
  if [ -n "$MODULES" ] && command -v module > /dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load $MODULES || { echo "module load $MODULES failed: set MODULES (see 'module spider CUDA')"; return 1; }
  fi
}
