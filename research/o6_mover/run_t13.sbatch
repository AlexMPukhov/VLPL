#!/bin/bash
#SBATCH -p epyc
#SBATCH -n 3
#SBATCH -c 1
#SBATCH -t 4:00:00
#SBATCH -o out/t13_%j.log
. $WORK/smilei/smilei_env.sh >/dev/null 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $WORK/rip-pic-o6/research/o6_mover
srun -n1 -c1 --exact python3 t1_langmuir.py 2 > out/t1_s2.txt 2>&1 &
srun -n1 -c1 --exact python3 t1_langmuir.py 1 > out/t1_s1.txt 2>&1 &
srun -n1 -c1 --exact python3 t3_emwave.py > out/t3.txt 2>&1 &
wait
