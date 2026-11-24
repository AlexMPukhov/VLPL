#!/bin/bash
#SBATCH -p epyc
#SBATCH -n 8
#SBATCH -c 1
#SBATCH -t 2:00:00
#SBATCH -o out/t2v_%j.log
. $WORK/smilei/smilei_env.sh >/dev/null 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $WORK/rip-pic-o6/research/o6_mover
r() { srun -n1 -c1 --exact python3 t2_heating.py implicit 2 t=5.8 "$@" > out/t2v_$(echo "$@" | tr ' =' '_-').txt 2>&1 & }
r inner=50 ptol=1e-10 tol=1e-10
r inner=50 ptol=1e-10 tol=1e-6
r inner=50 ptol=1e-10 tol=1e-4
r inner=1 tol=1e-10
r inner=1 tol=1e-6
r inner=2 ptol=1e-10 tol=1e-6
r inner=1 tol=1e-4
r inner=1 tol=1e-3
wait
