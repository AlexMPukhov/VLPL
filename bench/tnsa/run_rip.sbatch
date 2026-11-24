#!/bin/bash
#SBATCH -J tnsa_rip
#SBATCH -p <partition>
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH -t 12:00:00
#SBATCH -o $WORK/tnsa/logs/%x_%j.log
# VLPL side of bench/tnsa: sbatch --export=ALL,RUN=rip_code,ARGS="" run_rip.sbatch
# ARGS: deck overrides (resolution, t_end, ...); ranks along y only (the foil is thin in x).
cd $WORK/tnsa
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PROC_BIND=false
R=$WORK/rip-pic-tnsa
hostname; echo "RUN=$RUN ARGS=$ARGS"; date
t0=$(date +%s)
eval "mpirun --map-by ppr:$SLURM_NTASKS_PER_NODE:node:PE=$SLURM_CPUS_PER_TASK --bind-to core $R/build-mpi/vlpl $R/bench/tnsa/rip.deck \
  grid.mpi_x=1 grid.mpi_y=$SLURM_NTASKS grid.mpi_z=1 output.dir=$RUN $ARGS" > $RUN.log 2>&1
rc=$?; t1=$(date +%s)
echo "exit $rc wall $((t1-t0)) s, core-hours $(python3 -c "print(round(($t1-$t0)*$SLURM_NNODES*24/3600,2))") ($SLURM_NNODES nodes x 24 cores)"
grep -a "warning\|done\|time per" $RUN.log
. $WORK/smilei/smilei_env.sh
python3 $R/bench/tnsa/spectrum.py rip $RUN -o $RUN/tnsa_rip.txt
