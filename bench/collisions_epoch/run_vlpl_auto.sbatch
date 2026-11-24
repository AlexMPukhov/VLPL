#!/bin/bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH -t 04:00:00
#SBATCH --array=0-3
#SBATCH -o va_%A_%a.log
# VLPL on the solid-density case with the automatic (per-pair, Lee-More-type) Coulomb log instead of lnL = 5:
# 1.5 ps (the relaxation is ~2 times slower than with lnL = 5); low_t_cap on / off, c dt = 0.02 / k0 and dt / 4 (dx / 4, the same particles per cell). sbatch run_vlpl_auto.sbatch (EXE: vlpl)
EXE=${EXE:-../../build/vlpl}
export OMP_NUM_THREADS=24 OMP_PROC_BIND=false
A=("collisions ee.coulomb_log=0" "collisions ep.coulomb_log=0" "collisions pp.coulomb_log=0" grid.t_end=1500fs)
NOCAP=("collisions ee.low_t_cap=false" "collisions ep.low_t_cap=false" "collisions pp.low_t_cap=false")
DT4=(grid.dx=0.000795775um grid.nx=48 output.every=2000)
case $SLURM_ARRAY_TASK_ID in
    0) $EXE solid.deck "${A[@]}" output.dir=out_vlpl_auto ;;
    1) $EXE solid.deck "${A[@]}" "${NOCAP[@]}" output.dir=out_vlpl_auto_nocap ;;
    2) $EXE solid.deck "${A[@]}" "${DT4[@]}" output.dir=out_vlpl_auto_dt4 ;;
    3) $EXE solid.deck "${A[@]}" "${NOCAP[@]}" "${DT4[@]}" output.dir=out_vlpl_auto_nocap_dt4 ;;
esac > va_run_$SLURM_ARRAY_TASK_ID.out 2>&1
echo "run $SLURM_ARRAY_TASK_ID exit $?"
