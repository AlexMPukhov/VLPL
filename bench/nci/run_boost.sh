#!/bin/sh
# NCI in a plasma streaming at gamma ~ 10 (30) as in a boosted frame (decision 87): RIP-CC and iRIP, 3D and RZ.
#   sh bench/nci/run_boost.sh build/vlpl      -> nci_<case>/diag.txt; python3 bench/nci/growth.py nci_*
EXE=${1:-build/vlpl}
D=examples/streaming_plasma.deck
R=bench/nci/rz_streaming_plasma.deck
run() { name=$1; shift; $EXE "$@" output.every=5 output.dir=nci_$name > nci_$name.log 2>&1; tail -1 nci_$name/diag.txt; }
run rip_cc        $D numerics.solver=rip_cc
run irip          $D numerics.solver=irip
run rip_cc_s2     $D numerics.solver=rip_cc numerics.shape=2
run irip_s2       $D numerics.solver=irip numerics.shape=2
run rip_cc_u30    $D numerics.solver=rip_cc "species electrons.ux=-30" "species protons.ux=-30"
run irip_u30      $D numerics.solver=irip "species electrons.ux=-30" "species protons.ux=-30"
# boosted-LWFA-like cells: hx = 1.9 c/wp = hy (beyond RIP-CC's Courant limit; iRIP only)
run irip_hx3      $D numerics.solver=irip "grid.dx=2.992 um"
run irip_hx3_s2   $D numerics.solver=irip "grid.dx=2.992 um" numerics.shape=2
run rz_rip_cc     $R numerics.solver=rip_cc
run rz_irip       $R numerics.solver=irip
run rz_irip_hx3   $R numerics.solver=irip "grid.dz=2.992 um"
