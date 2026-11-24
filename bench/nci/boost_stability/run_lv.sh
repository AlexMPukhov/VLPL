#!/bin/sh
# Section 2c of README.md: the Lee and Vay (CPC 238, 102, 2019) 1e19 case in 2D, lab and boosted (gamma 2, 3), lambda / 32 and 64,
# iRIP and RIP-CC (RIP-CC boosted only where its Courant limit allows). ~250 core-hours; section 2d ~170 more.
#   sh bench/nci/boost_stability/run_lv.sh build/vlpl OUTDIR [NPAR=4];  python3 bench/nci/boost_stability/lv_ana.py OUTDIR
X=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); O=${2:-bs_lv}; P=${3:-4}
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p $O; O=$(cd $O && pwd); : > $O/cmds.txt
mk() { n=$1; g=$2; shift 2
  if [ $g = 0 ]; then awk '/^lab_diagnostics/{s=1} !s{print} s&&/^}/{s=0}' $H/lv.deck > $O/$n.deck; ob=output.openpmd_species=electrons
  else { cat $H/lv.deck; printf 'boost {\n  gamma = %s\n}\n' $g; } > $O/$n.deck; ob=numerics.filter_passes=1; fi
  echo "cd $O && $X $n.deck output.dir=$O/$n $ob $* > $O/$n.log 2>&1" >> $O/cmds.txt; }
R64="grid.nx=3200 grid.dx=0.0125um output.openpmd_every=2400"
mk lab_irip_32_long 0 grid.t_end=1850fs
mk lab_irip_64 0 $R64
mk lab_ripcc_32 0 numerics.solver=rip_cc
mk lab_ripcc_64 0 numerics.solver=rip_cc $R64
for g in 2 3; do mk b${g}_irip_32 $g; mk b${g}_irip_64 $g $R64; done
mk b2_ripcc_32 2 numerics.solver=rip_cc
mk b2_ripcc_64 2 numerics.solver=rip_cc $R64
mk b3_ripcc_64 3 numerics.solver=rip_cc $R64
# section 2d (decision 107): the lab runs with one filter pass, and a boosted run without filter (~170 core-hours more)
mk lab_f1_irip_32 0 numerics.filter_passes=1
mk lab_f1_ripcc_32 0 numerics.filter_passes=1 numerics.solver=rip_cc
mk lab_f1_irip_64 0 numerics.filter_passes=1 $R64
mk lab_f1_ripcc_64 0 numerics.filter_passes=1 numerics.solver=rip_cc $R64
mk b2nof_irip_32 2 numerics.filter_passes=0
xargs -d '\n' -P $P -I{} sh -c '{}' < $O/cmds.txt
