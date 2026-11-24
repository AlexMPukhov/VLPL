#!/bin/bash
#SBATCH -p epyc
#SBATCH -n 8
#SBATCH -c 1
#SBATCH -t 2:00:00
#SBATCH -o out/c1_%j.log
. $WORK/smilei/smilei_env.sh >/dev/null 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $WORK/rip-pic-o6/research/o6_mover
for m in plain exp c1; do python3 c1_density.py $m; done
for a in "0.1 0.05 30" "0.3 0.05 3" "0.1 0.05 270" "0.3 0.05 30"; do
  srun -n1 -c1 --exact python3 rz_conv.py $a 1e-9 c1 > "out/c1conv_$(echo $a | tr ' ' '_').txt" 2>&1 &
done
wait
tail -n 2 out/c1conv_*.txt
