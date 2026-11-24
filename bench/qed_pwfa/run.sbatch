#!/bin/bash
# The three runs of bench/qed_pwfa (one array task each, one 24-core cluster node, ~4 h each; MPI over 4 nodes was slower):
#   0 both effects, 1 T-BMT only (qed.model=none), 2 radiation only (qed.spin_precession=false)
#   VLPL=/path/to/build/vlpl sbatch bench/qed_pwfa/run.sbatch      (from the repository root; outputs in ./out_<run>)
#SBATCH -J qed_pwfa
#SBATCH -N 1
#SBATCH -c 24
#SBATCH -t 14:00:00
#SBATCH -o qed_pwfa_%a_%A.log
#SBATCH --array=0-2
export OMP_NUM_THREADS=24 OMP_PROC_BIND=false
case $SLURM_ARRAY_TASK_ID in
  0) N=both; X="" ;;
  1) N=tbmt; X="qed.model=none" ;;
  2) N=rad;  X="qed.spin_precession=false" ;;
esac
${VLPL:-build/vlpl} bench/qed_pwfa/pwfa_spin.deck output.dir=out_$N $X
