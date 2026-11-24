#!/bin/sh
# Nonlinear check of the boosted frame (README.md here, decision 104): LWFA a0 = 2 and 4, n0 = 0.004 n_c after a 100 um up-ramp,
# 0.6 mm in the lab, through the antenna; the lab run of the same deck as the reference (lab_diagnostics at the lab steps of its
# openPMD output). 2D and RZ (2 modes), iRIP shape 2. Boosted: gamma 5 and 10, cold and warm plasma (u_th = 1e-2 as the noise
# seed), current filter 0, 1, 2 passes; RZ gathers plain and matched. Cost: 2D lab ~1 h, RZ lab ~3 h, 2D boosted ~10 min,
# RZ boosted ~1.5 h on 6-8 threads each (~150 core-hours in all; the warm 2D gamma 10 runs alone show the instability).
#   sh bench/nci/boost_stability/run_nonlinear.sh build/vlpl OUTDIR [NPAR=4]  (OMP_NUM_THREADS per run)
#   python3 bench/nci/boost_stability/plot.py OUTDIR lab c2 4 3 c2_g5_a4,c2_g10_a4_warm,c2_g10_a4_f1_warm   (lab-frame figure)
#   python3 bench/nci/boost_stability/plot.py OUTDIR boost c2_g10_a4_warm                                  (boosted frame)
#   python3 bench/nci/boost_stability/noise.py OUTDIR 600 c2_g10_a4 c2_g10_a4_warm c2_g10_a4_f1_warm          (noise ahead)
X=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); O=${2:-bs_nonlinear}; P=${3:-4}
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p $O; O=$(cd $O && pwd); : > $O/cmds.txt
W="'species electrons.uth_x=1e-2' 'species electrons.uth_y=1e-2' 'species electrons.uth_z=1e-2'"
mk() {   # name geo a0 gamma(0: lab) overrides...
  n=$1; geo=$2; a0=$3; g=$4; shift 4
  if [ $g = 0 ]; then awk '/^lab_diagnostics/{s=1} !s{print} s&&/^}/{s=0}' $H/nl_$geo.deck > $O/$n.deck
  else { cat $H/nl_$geo.deck; printf 'boost {\n  gamma = %s\n}\n' $g; } > $O/$n.deck; fi
  ob=; [ $g != 0 ] && ob=output.openpmd_every=100
  echo "cd $O && $X $n.deck laser.a0=$a0 output.dir=$O/$n $ob $* > $O/$n.log 2>&1" >> $O/cmds.txt
}
for a in 2 4; do
  mk c2_lab_a$a c2 $a 0
  mk rz_lab_a${a}_plain rz $a 0 numerics.rz_gather=plain
  for g in 5 10; do
    mk c2_g${g}_a$a c2 $a $g
    mk rz_g${g}_a${a}_plain_f1 rz $a $g numerics.rz_gather=plain numerics.filter_passes=1
  done
done
mk rz_lab_a4_matched rz 4 0 numerics.rz_gather=matched
mk rz_g10_a4_matched_f1 rz 4 10 numerics.rz_gather=matched numerics.filter_passes=1
mk rz_g10_a4_plain_f0 rz 4 10 numerics.rz_gather=plain
mk rz_g10_a4_matched_f0 rz 4 10 numerics.rz_gather=matched
mk rz_g5_a4_plain_f0 rz 4 5 numerics.rz_gather=plain
mk c2_lab_a4_warm c2 4 0 "$W"
for g in 5 10; do
  mk c2_g${g}_a4_warm c2 4 $g "$W"
  for f in 1 2; do mk c2_g${g}_a4_f${f}_cold c2 4 $g numerics.filter_passes=$f; mk c2_g${g}_a4_f${f}_warm c2 4 $g numerics.filter_passes=$f "$W"; done
done
for v in "plain_f1 numerics.rz_gather=plain numerics.filter_passes=1" "plain_f0 numerics.rz_gather=plain" "matched_f1 numerics.rz_gather=matched numerics.filter_passes=1"; do
  set -- $v; n=$1; shift; mk rz_g10_a4_${n}_warm rz 4 10 "$@" "$W"
done
xargs -d '\n' -P $P -I{} sh -c '{}' < $O/cmds.txt
