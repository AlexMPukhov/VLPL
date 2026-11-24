#!/bin/bash
#SBATCH -p epyc
#SBATCH -n 6
#SBATCH -c 1
#SBATCH -t 2:00:00
#SBATCH -o out/conv_%j.log
. $WORK/smilei/smilei_env.sh >/dev/null 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $WORK/rip-pic-o6/research/o6_mover
for a in "0.3 0.05 30" "0.1 0.05 30" "0.3 0.001 30" "0.3 0.05 3" "0.1 0.05 270" "0.3 0.05 30 1e-6"; do
  srun -n1 -c1 --exact python3 rz_conv.py $a > "out/conv_$(echo $a | tr ' ' '_').txt" 2>&1 &
done
wait
