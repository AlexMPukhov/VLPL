#!/bin/bash
# CHF benchmark, 3D at lambda0/128 (bench/chf/README.md, section "3D lambda0/128 on JUPITER"): the target run and its vacuum
# twin on GH200 GPUs, one rank per GPU. Build first with bench/jupiter/build.sh (build-gpu-mpi), set ACCOUNT in
# bench/jupiter/env.sh, then submit from the VLPL directory:
#   sbatch --account=<project> bench/chf/jupiter_3d128.sbatch                                     # target, 4 nodes = 16 GPUs
#   sbatch --account=<project> --nodes=1 --export=ALL,KIND=v bench/chf/jupiter_3d128.sbatch       # vacuum twin, 4 GPUs
# Grid 1280 x 826 x 826 (8.7e8 cells), 2500 steps; ~1e9 particles per species; memory ~300 GB in all (16 GPUs: < 20 GB each).
# Output: bench/jupiter/runs/chf3d128.<t|v>/{run.log, out/xflux.txt, out/diag.txt, out/harmonic_focus_low|high/}; the maps are
# ~0.9 GB per band. Analysis (python3 with numpy): python3 bench/chf/analyse.py <t dir> <v dir> --a0 10
#SBATCH --job-name=vlpl-chf3d
#SBATCH --partition=booster
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --output=vlpl-chf3d.%j.log
set -u
. "${SLURM_SUBMIT_DIR:?submit from the VLPL directory}/bench/jupiter/env.sh"
load_modules || exit 1
KIND=${KIND:-t}                      # t: the target run, v: its vacuum twin (the incident energy of the balance)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72} OMP_PROC_BIND=close OMP_PLACES=cores
D=$RUNS/chf3d128.$KIND
mkdir -p "$RUNS/chf3d128.decks" "$D"
python3 "$SRC/bench/chf/series.py" 3d 128 "$RUNS/chf3d128.decks" > /dev/null || exit 1
cp "$RUNS/chf3d128.decks/${KIND}128/chf.deck" "$D/chf.deck"
cd "$D" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "ranks $SLURM_NTASKS, kind $KIND, $(date)"
T0=$(date +%s)
srun --ntasks="$SLURM_NTASKS" --cpus-per-task="$OMP_NUM_THREADS" "$BUILD_GPU/vlpl" chf.deck > run.log 2>&1 < /dev/null
rc=$?
echo "exit $rc, wall $(( $(date +%s) - T0 )) s"
grep -a "done:\|time per\|exchanges\|warning\|error" run.log | head -20
tail -2 out/diag.txt
exit $rc
