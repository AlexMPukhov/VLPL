#!/bin/bash
#SBATCH -p epyc
#SBATCH -n 8
#SBATCH -c 1
#SBATCH -t 4:00:00
#SBATCH -o out/cost_%j.log
. $WORK/smilei/smilei_env.sh >/dev/null 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $WORK/rip-pic-o6/research/o6_mover
r() { srun -n1 -c1 --exact python3 t2_heating.py "$@" > out/cost_$(echo "$@" | tr ' =' '_-').txt 2>&1 & }
# omega_p dt = 2, 5, 10 at the CHF cell (n = (wp dt / dx)^2), 1 keV, 4 fs
r implicit 2 n=415 t=4 inner=1 tol=1e-6
r implicit 2 n=415 t=4 inner=1 tol=1e-10
r implicit 2 n=2593 t=4 inner=1 tol=1e-6
r implicit 2 n=2593 t=4 inner=1 tol=1e-10
r implicit 2 n=10370 t=4 inner=1 tol=1e-6
r implicit 2 n=10370 t=4 inner=1 tol=1e-10
r explicit 2 n=415 t=4
r implicit 2 t=29 inner=1 tol=1e-6
wait
