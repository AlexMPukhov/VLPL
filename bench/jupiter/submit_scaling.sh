#!/bin/sh
# Submit the timing series: 1, 4, 16 GPUs (one rank per GPU; 64 optional) and the CPU reference on one node.
# Every GPU job runs bubble_strong, bubble_weak and rz_nm10 (scaling.sbatch).
# Run from the VLPL directory after build.sh (and after check.sbatch has passed):
#   sh bench/jupiter/submit_scaling.sh            1, 4, 16 GPUs and the CPU node: 4 jobs
#   sh bench/jupiter/submit_scaling.sh 1 4 16 64  with 64 GPUs (16 nodes)
#   NO_CPU=1 sh bench/jupiter/submit_scaling.sh 1 without the CPU reference
#   CUDA_AWARE=1 NO_CPU=1 sh bench/jupiter/submit_scaling.sh 4 16   exchanges straight from GPU memory
#                                                (VLPL_CUDA_AWARE=1; logs in scale_<N>gpu_ca.<job>)
#   OVERLAP=0 ...      without the overlap of the current guard exchanges with the interior push (VLPL_OVERLAP=0;
#                      logs in scale_<N>gpu[_ca]_noov.<job>): the comparison run for the overlap
#   PROFILE=0 ...      without the step profile (on by default: VLPL_PROFILE=1 from the step PROFILE_FROM of the
#                      bubble runs, default 1250 = 208 fs, when the box is full of plasma; it adds no synchronisation)
. "$(dirname "$0")/env.sh"
[ "$ACCOUNT" = CHANGE_ME ] && { echo "set ACCOUNT (env.sh or ACCOUNT=<project> sh $0)"; exit 1; }
cd "$SRC" || exit 1
mkdir -p "$RUNS"
for n in ${*:-1 4 16}; do
  nodes=$(( (n + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
  tpn=$(( n < GPUS_PER_NODE ? n : GPUS_PER_NODE ))
  # upper limits; 1 GPU: three runs of a few minutes each on a GH200 (RTX 5090: about 5 min each)
  case $n in 1) t=01:30:00 ;; 4) t=01:00:00 ;; *) t=00:45:00 ;; esac
  sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=$nodes --ntasks-per-node=$tpn \
    --gres=gpu:"$GPUS_PER_NODE" --cpus-per-task="$CPUS_PER_TASK" --time=$t \
    --export=ALL,NGPU="$n",VLPL_CUDA_AWARE="${CUDA_AWARE:-0}",VLPL_OVERLAP="${OVERLAP:-1}",PROFILE="${PROFILE:-1}" \
    bench/jupiter/scaling.sbatch
done
if [ -z "${NO_CPU:-}" ]; then
  # the booster partitions may refuse jobs without GPUs: the CPU job takes the node's GPUs and leaves them idle
  sbatch --account="$ACCOUNT" --partition="$CPU_PARTITION" --ntasks-per-node="${RANKS:-4}" --gres=gpu:"$GPUS_PER_NODE" \
    --cpus-per-task="${THREADS:-$CPUS_PER_TASK}" --time=02:00:00 bench/jupiter/cpu_reference.sbatch
fi
echo "logs go to $RUNS/scale_*gpu.<job> and $RUNS/cpu_*.<job>; table: sh bench/jupiter/collect.sh"
