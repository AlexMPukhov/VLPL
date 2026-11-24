#!/bin/bash
#SBATCH -p epyc
#SBATCH -n 6
#SBATCH -c 2
#SBATCH -t 4:00:00
#SBATCH -o out/t12_%j.log
. $WORK/smilei/smilei_env.sh >/dev/null 2>&1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd $WORK/rip-pic-o6/research/o6_mover
srun -n1 -c2 --exact python3 t1_langmuir.py 2 > out/t1_s2.txt 2>&1 &
srun -n1 -c2 --exact python3 t1_langmuir.py 1 > out/t1_s1.txt 2>&1 &
srun -n1 -c2 --exact python3 t2_heating.py explicit 2 > out/t2_exp_s2.txt 2>&1 &
srun -n1 -c2 --exact python3 t2_heating.py implicit 2 > out/t2_imp_s2.txt 2>&1 &
srun -n1 -c2 --exact python3 t2_heating.py explicit 1 > out/t2_exp_s1.txt 2>&1 &
srun -n1 -c2 --exact python3 t2_heating.py explicit 2 n=40 uth=0.044 dx=0.0125 dy=0.0125 ny=32 ppy=2 > out/t2_exp_s2_sq.txt 2>&1 &
wait
