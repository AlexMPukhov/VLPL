#!/bin/bash
#SBATCH -J nf-scan
#SBATCH -p epyc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 03:00:00
#SBATCH -o $WORK/nfruns/scan1/nf.%j.log
# Near field of the lab-fixed antenna (VLPL branch antenna-near-field): vacuum scan, VLPL_NF_SCAN=1 build/tests.
# No build here (built on the cluster head node).
. $WORK/gempic-mem/env.sh
export OMP_PROC_BIND=false OMP_NUM_THREADS=8
W=$WORK/vlpl-nearfield; B=$W/build-mpi/tests; D=$WORK/nfruns/scan1
cd $W && echo "commit $(cat $D/COMMIT)"; stat -c "binary %y" $B
N=$(find $W/src $W/tests $W/Makefile -newer $B | wc -l); echo "sources newer than the binary: $N"; [ "$N" = 0 ] || exit 3
run() {   # name, env assignments...
  local n=$1; shift
  mkdir -p $D/$n && cd $D/$n || return 1
  ( t0=$(date +%s); env VLPL_NF_SCAN=1 NF_OUT=$D/$n/scan.txt "$@" mpirun -np 1 --bind-to none $B test_boost_antenna > console.txt 2>&1
    echo "$n rc=$? wall $(( $(date +%s) - t0 )) s" ) &
}
# 1. near field: gamma x lab cell (iRIP, shape 2, no filter, waist 8 um, dy 0.4 um)
run g2n20 NF_GAMMA=2 NF_N=20
run g2n40 NF_GAMMA=2 NF_N=40
run g5n20 NF_GAMMA=5 NF_N=20
run g5n40 NF_GAMMA=5 NF_N=40
run g6n20 NF_GAMMA=6 NF_N=20
run g6n40 NF_GAMMA=6 NF_N=40
run g5n80 NF_GAMMA=5 NF_N=80
run g10n20 NF_GAMMA=10 NF_N=20 NF_L=120
wait
# shape, filter, solver, transverse cell, emission time
run g5n20s1 NF_GAMMA=5 NF_N=20 NF_SHAPE=1
run g5n20f2 NF_GAMMA=5 NF_N=20 NF_FILTER=2
run g2n20cc NF_GAMMA=2 NF_N=20 NF_SOLVER=rip_cc
run g5n20cc NF_GAMMA=5 NF_N=20 NF_SOLVER=rip_cc NF_DY=0.8
run g5n20dy2 NF_GAMMA=5 NF_N=20 NF_DY=0.2
run g5n20d3 NF_GAMMA=5 NF_N=20 NF_DELAYF=3
run g6n20s1f2 NF_GAMMA=6 NF_N=20 NF_SHAPE=1 NF_FILTER=2
wait
# 2. tight waist: the pulse centre at the focus 40 um downstream (tau 15 fs)
T="NF_TAU=15 NF_FOCUS=40 NF_L=60 NF_LY=30 NF_COUNT=1"
run w15g2n20 $T NF_W=1.5 NF_DY=0.05 NF_GAMMA=2 NF_N=20 NF_TFIRST=163.4
run w15g5n20 $T NF_W=1.5 NF_DY=0.05 NF_GAMMA=5 NF_N=20 NF_TFIRST=163.4
run w15g5n40 $T NF_W=1.5 NF_DY=0.05 NF_GAMMA=5 NF_N=40 NF_TFIRST=163.4
run w3g5n20 $T NF_W=3 NF_DY=0.1 NF_GAMMA=5 NF_N=20 NF_TFIRST=163.4
run w3g5n40 $T NF_W=3 NF_DY=0.1 NF_GAMMA=5 NF_N=40 NF_TFIRST=163.4
run w15g5n80 $T NF_W=1.5 NF_DY=0.05 NF_GAMMA=5 NF_N=80 NF_TFIRST=163.4
wait
echo "end $(date)"
