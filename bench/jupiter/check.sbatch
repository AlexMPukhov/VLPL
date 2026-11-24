#!/bin/bash
# Correctness of VLPL on real GPUs, 1 node with 4 GPUs (~1-1.5 h, most of it the CPU references). Every GPU result is compared with the CPU
# build on the same node, as in the RTX 5090 checks of docs/GPU_TESTING.md. Submit from the VLPL directory:
#   sbatch --account=<project> bench/jupiter/check.sbatch
#   sbatch --account=<project> --nodes=2 --export=ALL,NP=8 bench/jupiter/check.sbatch   (8 GPUs on 2 nodes)
# 1. nvidia-smi; build-gpu-mpi/tests and build-cpu-mpi/tests (one rank): all must pass; the lines that differ
# 2. tests/regression/run.sh with the GPU and with the CPU binary (one rank): every case against the stored
#    references (CPU, x86), and the GPU diag.txt against the CPU diag.txt of this node
# 3. tests/mpi.sh with one GPU per rank: 1 vs NP ranks (default 4, all GPUs of the node) in every case (3D,
#    iRIP, open x, CPML, moving window, beam self-field, restart, RZ), then the same with the CPU build, and
#    every diag.txt of the GPUs against the CPU's (tests/regression/compare.py)
#    On the GPU a restart is compared to round-off (RESTART_EXACT=0 of tests/mpi.sh): the atomic additions of the
#    deposit have no fixed order, so the bytes of diag.txt differ; on the CPU it must be exact.
# The last line is "summary: 0 failed check(s)" when all is well.
# Parts only, e.g. after a cluster error in the CPU half of step 3, into the folder of the earlier job (its GPU runs
# are kept; step 4 compares them with the new CPU runs):
#   sbatch --account=<project> --export=ALL,STEPS="3 4",BUILDS=cpu,REUSE=<earlier job id> bench/jupiter/check.sbatch
# STEPS (default "1 2 3 4"), BUILDS (default "gpu cpu", for steps 1-3), REUSE (default: a new folder for this job).
# The guard exchanges straight from GPU memory (CUDA-aware MPI) against the CPU runs of an earlier job:
#   sbatch --account=<project> --export=ALL,VLPL_CUDA_AWARE=1,STEPS="3 4",BUILDS=gpu,REUSE=<earlier job id> bench/jupiter/check.sbatch
#SBATCH --job-name=vlpl-check
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=02:30:00
#SBATCH --output=vlpl-check.%j.log
set -u
. "${SLURM_SUBMIT_DIR:?submit from the VLPL directory}/bench/jupiter/env.sh"
load_modules || exit 1
NP=${NP:-4}
NT=${SLURM_CPUS_PER_TASK:-12}
export OMP_NUM_THREADS=$NT OMP_PROC_BIND=close OMP_PLACES=cores
G=$BUILD_GPU; C=$BUILD_CPU; CMP=$SRC/tests/regression/compare.py
STEPS=${STEPS:-1 2 3 4}; BUILDS=${BUILDS:-gpu cpu}
W=$RUNS/check.${REUSE:-$SLURM_JOB_ID}; mkdir -p "$W/gpu" "$W/cpu"; cd "$W" || exit 1
fails=0
sec() { echo "   ($(( $(date +%s) - T0 )) s)"; }
on() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
# one rank of an MPI build through srun (the compute nodes start MPI programs only with srun); stdin from
# /dev/null, else srun forwards the case list that tests/regression/run.sh reads to the rank
for b in gpu cpu; do
  [ $b = gpu ] && exe=$G/vlpl || exe=$C/vlpl
  printf '#!/bin/sh\nexec srun -n 1 --cpus-per-task=%s %s "$@" < /dev/null\n' "$NT" "$exe" > "$W/vlpl_$b"
  chmod +x "$W/vlpl_$b"
  ln -sfn "$SRC/examples" "$W/$b/examples"   # tests/mpi.sh and the regression cases read examples/*.deck
done

echo "== 0. node and GPUs ($(hostname), $(date)); steps $STEPS, builds $BUILDS"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "   GPU build $G, CPU build $C, NP = $NP, $NT cores per rank, VLPL_CUDA_AWARE=${VLPL_CUDA_AWARE:-0}"

if on 1; then
echo "== 1. unit tests, GPU and CPU (one rank)"
T0=$(date +%s)
for b in $BUILDS; do
  [ $b = gpu ] && d=$G || d=$C
  (cd $b && srun -n 1 --cpus-per-task=$NT "$d/tests" > "$W/tests_$b.txt" 2>&1); rc=$?
  echo "-- $b: $(grep -E 'passed|failure' "$W/tests_$b.txt" | tail -1) (exit $rc)"
  [ $rc = 0 ] || fails=$((fails + 1))
done
if [ -f "$W/tests_cpu.txt" ] && [ -f "$W/tests_gpu.txt" ]; then
python3 - "$W/tests_cpu.txt" "$W/tests_gpu.txt" "$W/tests_diff.txt" <<'PY'
import sys
a = open(sys.argv[1]).read().splitlines(); b = open(sys.argv[2]).read().splitlines()
d = [(x, y) for x, y in zip(a, b) if x != y]
with open(sys.argv[3], "w") as f:
    for x, y in d:
        f.write("CPU: %s\nGPU: %s\n" % (x.strip(), y.strip()))
print("-- %d of %d lines differ between CPU and GPU in the last digits (expected: the order of the atomic"
      " additions), all in tests_diff.txt; the FAIL lines:" % (len(d) + abs(len(a) - len(b)), max(len(a), len(b))))
for name, t in (("CPU", a), ("GPU", b)):
    for x in t:
        if " FAIL " in x:
            print("   %s: %s" % (name, x.strip()))
PY
fi
sec
fi

if on 2; then
echo "== 2. regression cases, GPU and CPU (one rank) against the references and against each other"
T0=$(date +%s)
for b in $BUILDS; do
  (cd $b && sh "$SRC/tests/regression/run.sh" "$W/vlpl_$b" > "$W/regression_$b.txt" 2>&1); rc=$?
  echo "-- $b:"; sed 's/^/   /' "$W/regression_$b.txt"
  [ $rc = 0 ] || fails=$((fails + 1))
done
sec
fi

if on 3; then
echo "== 3. tests/mpi.sh: 1 vs $NP ranks, one GPU per rank; then the CPU build"
T0=$(date +%s)
for b in $BUILDS; do
  [ $b = gpu ] && exe=$G/vlpl || exe=$C/vlpl
  [ $b = gpu ] && rx=0 || rx=1          # GPU: restarts to round-off (atomic deposit order)
  (cd $b && NP=$NP RESTART_EXACT=$rx MPIRUN="sh $SRC/tests/mpirun_srun.sh" SRUN_OPTS="--cpus-per-task=1" \
     sh "$SRC/tests/mpi.sh" "$exe" > "$W/mpi_$b.txt" 2>&1 < /dev/null); rc=$?
  echo "-- $b ($(grep -c '^mpi' "$W/mpi_$b.txt") cases, exit $rc):"; sed 's/^/   /' "$W/mpi_$b.txt"
  [ $rc = 0 ] || fails=$((fails + 1))
done
sec
fi

if on 4; then
echo "== 4. every diag.txt of the GPU runs against the CPU run of the same case (tests/regression/compare.py)"
n=0; bad=0
for f in $(cd gpu && find . -name diag.txt | sort); do
  n=$((n + 1))
  if [ ! -f "cpu/$f" ]; then echo "-- $f: no CPU run"; bad=$((bad + 1)); continue; fi
  if out=$(python3 "$CMP" "cpu/$f" "gpu/$f" 2>&1); then echo "-- $f: ok"
  else echo "-- $f: DIFFERS"; echo "$out"; bad=$((bad + 1)); fi
done
echo "-- $n diag.txt compared, $bad differ"
fails=$((fails + bad))
fi

echo "== summary: $fails failed check(s); outputs in $W"
