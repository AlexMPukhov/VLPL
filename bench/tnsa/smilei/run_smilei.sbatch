#!/bin/bash
#SBATCH -J tnsa_sm
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH -t 06:00:00
#SBATCH -o $WORK/tnsa/logs/%x_%j.log
# Smilei side of bench/tnsa. sbatch --export=ALL,TNSA_RES=code,TNSA_T=200,RUN=sm_code run_smilei.sbatch
# (TNSA_VACUUM=1 for the incident field alone). Runs in $WORK/tnsa/$RUN.
. $WORK/smilei/smilei_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_SCHEDULE=dynamic
d=$WORK/tnsa/$RUN; rm -rf $d; mkdir -p $d; cd $d
cp $WORK/rip-pic-tnsa/bench/tnsa/smilei/tnsa.py .
hostname; env | grep ^TNSA_; date
t0=$(date +%s)
mpirun -np $SLURM_NTASKS --map-by slot:PE=$SLURM_CPUS_PER_TASK --bind-to core smilei tnsa.py > smilei.log 2>&1
rc=$?; t1=$(date +%s)
echo "exit $rc wall $((t1-t0)) s, core-hours $(python3 -c "print(round(($t1-$t0)*$SLURM_NNODES*24/3600,2))") ($SLURM_NNODES nodes x 24 cores)"
tail -25 smilei.log
[ "${TNSA_VACUUM:-0}" = 1 ] || python3 $WORK/rip-pic-tnsa/bench/tnsa/spectrum.py smilei . -o tnsa_smilei.txt
