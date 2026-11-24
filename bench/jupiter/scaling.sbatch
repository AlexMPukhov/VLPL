#!/bin/bash
# Timings on NGPU GPUs, one rank per GPU. Submitted by submit_scaling.sh with NGPU = 1, 4, 16 (64) and the
# matching --nodes; every run writes its log (with "done: ... steps, ... s" and "time per operator") to
# $RUNS/scale_<NGPU>gpu.<job>/<case>.log, and collect.sh turns all logs into the table.
#   bubble_strong  bench/lwfa_bubble/rip.deck (3D, 600 x 160 x 160 cells, 1.4e8 particles) to T_BUBBLE, the same
#                  box on every NGPU (strong scaling); decomposition chosen by MPI_Dims_create
#   bubble_weak    the same with ny = nz = 160 k, k^2 = NGPU (a wider plasma, the same cells and particles per
#                  GPU), ranks only across the laser: grid.mpi_x = 1, mpi_y = mpi_z = k (weak scaling)
#   rz_nm10        the B7 RZ deck bench/fbpic_nm_scan/rip_rz.deck at Nm = 10 (iRIP, 800 x 50 cells, 2600 steps),
#                  ranks along z (strong scaling; up to RZ_MAX_GPUS ranks: 800 cells along z)
#SBATCH --job-name=vlpl-scale
#SBATCH --output=vlpl-scale.%j.log
set -u
. "${SLURM_SUBMIT_DIR:?submit from the VLPL directory}/bench/jupiter/env.sh"
load_modules || exit 1
NGPU=${NGPU:?set NGPU (submit_scaling.sh does)}
T_BUBBLE=${T_BUBBLE:-300fs}        # 1800 steps; plasma from 107 fs, the box full of plasma from 207 fs
T_RZ=${T_RZ:-433.60fs}             # the whole B7 run (2600 steps)
RZ_MAX_GPUS=${RZ_MAX_GPUS:-16}
EXTRA=${EXTRA:-}                   # more overrides for all runs
NT=${SLURM_CPUS_PER_TASK:-$CPUS_PER_TASK}
export OMP_NUM_THREADS=$NT OMP_PROC_BIND=close OMP_PLACES=cores
G=$BUILD_GPU/vlpl
CA=""; [ "${VLPL_CUDA_AWARE:-0}" = 1 ] && CA=_ca   # exchanges straight from GPU memory (CUDA-aware MPI)
NO=""; [ "${VLPL_OVERLAP:-1}" = 0 ] && NO=_noov    # without the overlap of the current exchanges with the push
export VLPL_OVERLAP=${VLPL_OVERLAP:-1}
# step profile (src/prof.hpp): ms per step of the operators and their parts, max / mean over the ranks, at the end
# of every log; counted from step PF (bubble: the box is full of plasma from 207 fs = step 1241), no synchronisation
PROFILE=${PROFILE:-1}
PF_BUBBLE=${PROFILE_FROM:-$(echo "${T_BUBBLE%fs}" | awk '{ s = int($1 / 0.16678); print (s / 2 < 1250 ? int(s / 2) : 1250) }')}
PF_RZ=${PROFILE_FROM:-0}
export VLPL_CUDA_AWARE=${VLPL_CUDA_AWARE:-0}
W=$RUNS/scale_${NGPU}gpu$CA$NO.$SLURM_JOB_ID; mkdir -p "$W"; cd "$W" || exit 1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader > gpu.txt
echo "$NGPU GPU(s) on $SLURM_JOB_NUM_NODES node(s): $(head -1 gpu.txt), $NT cores per rank"
run() {  # <case> <profile from step> <deck> <overrides...>
  label=$1; pf=$2; deck=$3; shift 3
  t0=$(date +%s)
  # shellcheck disable=SC2086
  VLPL_PROFILE=$PROFILE VLPL_PROFILE_FROM=$pf \
  srun --ntasks="$NGPU" --cpus-per-task="$NT" "$G" "$deck" "$@" $EXTRA output.dir="$label" > "$label.log" 2>&1
  rc=$?
  echo "-- $label: $(grep '^done' "$label.log" || echo "no 'done' line, exit $rc") ($(( $(date +%s) - t0 )) s with start-up)"
  grep -e 'time per operator' -e 'exchanges (max' "$label.log"
}
B=$SRC/bench/lwfa_bubble/rip.deck
BO="grid.t_end=$T_BUBBLE output.every=100 output.slice_every=0"
run bubble_strong $PF_BUBBLE "$B" $BO
k=1; while [ $((k * k)) -lt "$NGPU" ]; do k=$((k + 1)); done
if [ $((k * k)) = "$NGPU" ]; then
  run bubble_weak $PF_BUBBLE "$B" $BO grid.ny=$((160 * k)) grid.nz=$((160 * k)) grid.mpi_x=1 grid.mpi_y=$k grid.mpi_z=$k
else
  echo "-- bubble_weak: skipped (NGPU = $NGPU is not a square)"
fi
if [ "$NGPU" -le "$RZ_MAX_GPUS" ]; then
  run rz_nm10 $PF_RZ "$SRC/bench/fbpic_nm_scan/rip_rz.deck" grid.modes=10 "species electrons.ppc_theta=40" \
    "species ions.ppc_theta=40" grid.t_end=$T_RZ output.every=100 output.openpmd_every=0
fi
echo "logs in $W; table: sh bench/jupiter/collect.sh"
