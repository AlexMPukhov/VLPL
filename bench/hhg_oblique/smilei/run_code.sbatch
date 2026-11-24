#!/bin/bash
#SBATCH -J hhg_sm_code
#SBATCH -p <partition>
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH -t 12:00:00
#SBATCH -o $WORK/hhg_oblique/smilei/logs/%x_%j.log
# Code-check resolution (lambda/32 x lambda/16 x lambda/16, HHG_RES=code) run; HHG_VACUUM=1 (vacuum) or 0 (foil), HHG_A0 = laser a0 parameter.
# sbatch --export=ALL,HHG_VACUUM=1,HHG_A0=2,RUN=vac_code run_code.sbatch
. $WORK/smilei/smilei_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK HHG_RES=code
d=$WORK/hhg_oblique/smilei/$RUN; rm -rf $d; mkdir -p $d; cd $d
cp ../hhg_oblique.py .
hostname; env | grep ^HHG_; date
t0=$(date +%s)
mpirun -np $SLURM_NTASKS --map-by slot:PE=$SLURM_CPUS_PER_TASK --bind-to core smilei hhg_oblique.py > smilei.log 2>&1
rc=$?; t1=$(date +%s)
echo "exit $rc wall $((t1-t0)) s, core-hours $(python3 -c "print(round(($t1-$t0)*$SLURM_NNODES*24/3600,2))") ($SLURM_NNODES nodes x 24 cores)"
date; tail -30 smilei.log
