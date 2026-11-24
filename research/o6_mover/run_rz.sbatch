#!/bin/bash
#SBATCH -p epyc
#SBATCH -n 5
#SBATCH -c 2
#SBATCH -t 8:00:00
#SBATCH -o out/rz_%j.log
. $WORK/smilei/smilei_env.sh >/dev/null 2>&1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd $WORK/rip-pic-o6/research/o6_mover
srun -n1 -c2 --exact python3 rz_axis.py explicit > out/rz_explicit.txt 2>&1 &
srun -n1 -c2 --exact python3 rz_axis.py matched > out/rz_matched.txt 2>&1 &
srun -n1 -c2 --exact python3 rz_axis.py implicit inner=1 tol=1e-8 > out/rz_implicit.txt 2>&1 &
srun -n1 -c2 --exact python3 rz_axis.py explicit dr=0.085 > out/rz_explicit_dr2.txt 2>&1 &
srun -n1 -c2 --exact python3 rz_axis.py implicit dr=0.085 inner=1 tol=1e-8 > out/rz_implicit_dr2.txt 2>&1 &
wait
