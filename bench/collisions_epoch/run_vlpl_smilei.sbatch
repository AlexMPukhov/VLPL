#!/bin/bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH -t 02:00:00
#SBATCH -o vs_%j.log
# VLPL and Smilei 5.1 runs of the solid-density benchmark: sbatch run_vlpl_smilei.sbatch from bench/collisions_epoch (EXE: vlpl)
EXE=${EXE:-../../build/vlpl}
OMP_NUM_THREADS=24 OMP_PROC_BIND=false $EXE solid.deck > vlpl_solid.log 2>&1; echo "vlpl exit $?"
OMP_NUM_THREADS=24 OMP_PROC_BIND=false $EXE solid.deck "collisions ee.angle=smilei5" "collisions ep.angle=smilei5" "collisions pp.angle=smilei5" output.dir=out_vlpl_solid_s5 > vlpl_solid_s5.log 2>&1; echo "vlpl smilei5 exit $?"
. $WORK/smilei/smilei_env.sh
export OMP_NUM_THREADS=6
mkdir -p out_smilei_solid && cp smilei_solid.py out_smilei_solid/ && (cd out_smilei_solid && mpirun -np 4 smilei smilei_solid.py > smilei.log 2>&1); echo "smilei exit $?"
