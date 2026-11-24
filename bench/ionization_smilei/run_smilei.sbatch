#!/bin/bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH -t 01:00:00
#SBATCH -o smilei_%j.log
# Smilei 5.1 run of the benchmark (the CPU cluster): sbatch run_smilei.sbatch from bench/ionization_smilei; output in out_smilei/
. $WORK/smilei/smilei_env.sh
mkdir -p out_smilei && cp smilei_ion.py out_smilei/ && cd out_smilei
export OMP_NUM_THREADS=6
mpirun -np 4 smilei smilei_ion.py > smilei.log 2>&1
