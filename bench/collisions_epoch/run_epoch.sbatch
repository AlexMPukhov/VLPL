#!/bin/bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH -t 08:00:00
#SBATCH -o epoch_%j.log
# EPOCH 4.20 runs of the solid-density benchmark (the CPU cluster): sbatch run_epoch.sbatch from bench/collisions_epoch
# (EPOCH: the epoch1d binary, built with PARTICLE_SHAPE_BSPLINE3); out_epoch_{nanbu,sk,nocoll}/
. $WORK/smilei/smilei_env.sh
EPOCH=${EPOCH:-$WORK/epoch/epoch1d/bin/epoch1d}
for v in ${RUNS:-nanbu sk nocoll nanbu_dt4}; do
    mkdir -p out_epoch_$v
    case $v in
        nanbu)  sed -e "s/use_nanbu = T/use_nanbu = T/" epoch_solid.deck > out_epoch_$v/input.deck ;;
        nanbu_dt4) sed -e "s/^  t_end = .*/&\n  dt_multiplier = 0.2375/" epoch_solid.deck > out_epoch_$v/input.deck ;;   # 0.95 / 4
        sk)     sed -e "s/use_nanbu = T/use_nanbu = F/" epoch_solid.deck > out_epoch_$v/input.deck ;;
        nocoll) sed -e "s/use_collisions = T/use_collisions = F/" epoch_solid.deck > out_epoch_$v/input.deck ;;
    esac
    echo out_epoch_$v | mpirun -np 12 $EPOCH > out_epoch_$v/epoch.log 2>&1
    echo "$v exit $?"
done
