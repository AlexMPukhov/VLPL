#!/bin/bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -t 01:00:00
#SBATCH -o vlpl_%j.log
# VLPL runs of the collision benchmarks (the CPU cluster): sbatch run_vlpl.sbatch from bench/collisions_smilei (EXE: the vlpl binary)
EXE=${EXE:-../../build/vlpl}
export OMP_NUM_THREADS=12 OMP_PROC_BIND=false
for c in ${CASES:-thermal iso beam stop ion}; do
    $EXE $c.deck > vlpl_$c.log 2>&1
    echo "$c exit $?"
done
# the same with the angle fit of Smilei 5.1 (angle = smilei5) where s per step is large
for c in ${CASES5:-thermal iso beam stop}; do
    set --
    for b in $(grep -o "^collisions [a-z0-9]*" $c.deck | cut -d" " -f2); do set -- "$@" "collisions $b.angle=smilei5"; done
    $EXE $c.deck output.dir=out_vlpl_${c}_s5 "$@" > vlpl_${c}_s5.log 2>&1
    echo "$c (smilei5) exit $?"
done
