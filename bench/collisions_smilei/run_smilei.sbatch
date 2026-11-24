#!/bin/bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH -t 02:00:00
#SBATCH -o smilei_%j.log
# Smilei 5.1 runs of the collision benchmarks (the CPU cluster): sbatch run_smilei.sbatch from bench/collisions_smilei; out_smilei_<case>/
. $WORK/smilei/smilei_env.sh
export OMP_NUM_THREADS=6
for c in ${CASES:-thermal iso beam stop ion}; do
    mkdir -p out_smilei_$c && cp smilei_coll.py out_smilei_$c/ && (cd out_smilei_$c && CASE=$c mpirun -np 4 smilei smilei_coll.py > smilei.log 2>&1)
    echo "$c exit $?"
done
