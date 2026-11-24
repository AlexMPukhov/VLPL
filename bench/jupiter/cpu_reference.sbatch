#!/bin/bash
# CPU reference for the timing table: bubble_strong and rz_nm10 of scaling.sbatch on ONE node without GPUs, CPU
# build (MPI + OpenMP), RANKS ranks x THREADS threads. Submitted by submit_scaling.sh.
# JUPITER node (4 Grace CPUs, 4 x 72 cores): 4 x 72; JUWELS Booster node (2 x 24 cores): RANKS=4 THREADS=12.
#SBATCH --job-name=vlpl-cpu
#SBATCH --nodes=1
#SBATCH --output=vlpl-cpu.%j.log
set -u
. "${SLURM_SUBMIT_DIR:?submit from the VLPL directory}/bench/jupiter/env.sh"
load_modules || exit 1
RANKS=${RANKS:-4}; THREADS=${THREADS:-$CPUS_PER_TASK}
T_BUBBLE=${T_BUBBLE:-300fs}; T_RZ=${T_RZ:-433.60fs}
EXTRA=${EXTRA:-}
export OMP_NUM_THREADS=$THREADS OMP_PROC_BIND=close OMP_PLACES=cores
C=$BUILD_CPU/vlpl
W=$RUNS/cpu_${RANKS}x${THREADS}.$SLURM_JOB_ID; mkdir -p "$W"; cd "$W" || exit 1
echo "CPU node $(hostname): $RANKS ranks x $THREADS threads"
run() {  # <case> <deck> <overrides...>
  label=$1; deck=$2; shift 2
  t0=$(date +%s)
  # shellcheck disable=SC2086
  srun --ntasks="$RANKS" --cpus-per-task="$THREADS" --cpu-bind=cores "$C" "$deck" "$@" $EXTRA output.dir="$label" \
    > "$label.log" 2>&1
  rc=$?
  echo "-- $label: $(grep '^done' "$label.log" || echo "no 'done' line, exit $rc") ($(( $(date +%s) - t0 )) s with start-up)"
  grep 'time per operator' "$label.log"
}
run bubble_strong "$SRC/bench/lwfa_bubble/rip.deck" grid.t_end=$T_BUBBLE output.every=100 output.slice_every=0
run rz_nm10 "$SRC/bench/fbpic_nm_scan/rip_rz.deck" grid.modes=10 "species electrons.ppc_theta=40" \
  "species ions.ppc_theta=40" grid.t_end=$T_RZ output.every=100 output.openpmd_every=0
echo "logs in $W; table: sh bench/jupiter/collect.sh"
