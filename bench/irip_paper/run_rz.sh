#!/bin/sh
# The quasi-cylindrical runs of the irip paper (VLPL-RZ / VLPL-iRZ), one after the other on one GPU or CPU node.
# Usage (from bench/irip_paper): X=../../build-gpu/vlpl sh run_rz.sh [outdir]   (default X = ../../build/vlpl)
# Run with VLPL main 2bb707f (RZ reruns of 2026-11-22; numerics.rz_axis_current = regular, the default).
H=$(cd "$(dirname "$0")" && pwd); B=$(cd "$H/.." && pwd); X=${X:-$H/../../build/vlpl}; O=${1:-$H/out}
mkdir -p "$O/nci"; cd "$O"
run() { d=$1; n=$2; shift 2; echo "start $n"; "$X" "$@" output.dir=$d/$n > $d/$n.log 2>&1; echo "end $n $?"; }

# NCI streaming plasma, m = 0, 1 (Table ncivlpl, Fig. nci_vlpl): no filter, binomial and projection filter (one pass)
DR=$H/decks/nci_rz.deck; FINER="grid.nr=64 grid.dr=0.25068um"
for f in f0 f1 fp; do
  case $f in f0) o="numerics.filter_passes=0";; f1) o="numerics.filter_passes=1";; fp) o="numerics.filter_passes=1 numerics.filter=projection";; esac
  for sv in rip_cc irip; do
    run "$O/nci" rz_${sv}_g10_h3_$f "$DR" numerics.solver=$sv $o
    run "$O/nci" rz_${sv}_g50_h3_$f "$DR" numerics.solver=$sv $o "species electrons.uz=-50" "species protons.uz=-50"
  done
  run "$O/nci" rz_irip_g10_hq_$f "$DR" numerics.solver=irip $o $FINER
done

# LWFA against FBPIC (Fig. lwfa): lambda/16 and lambda/32, both closures
L=$B/lwfa_fbpic/rip_rz.deck
run "$O" lwfarz_ripcc "$L"
run "$O" lwfarz_irip "$L" numerics.solver=irip
run "$O" lw_rz_d2 "$L" grid.nz=512 "grid.dz=0.025 um"
run "$O" lw_irz_d2 "$L" grid.nz=512 "grid.dz=0.025 um" numerics.solver=irip

# FBPIC's LWFA example with many modes (Table nm, Fig. b7)
M=$B/fbpic_nm_scan/rip_rz.deck
b7() { n=$1; nm=$2; nt=$3; shift 3; run "$O" $n "$M" grid.modes=$nm "species electrons.ppc_theta=$nt" "species ions.ppc_theta=$nt" "$@"; }
b7 b7b_irip_nm2 2 8;  b7 b7b_irip_nm5 5 20;  b7 b7b_irip_nm10 10 40;  b7 b7b_irip_nm10_f0 10 40 numerics.filter_passes=0
b7 b7b_irip_nm10_pt80 10 80;  b7 b7b_irip_nm10_pt160 10 160;  b7 b7b_irip_nm16 16 64
b7 b7b_irip_nm2_proj 2 8 numerics.filter=projection;  b7 b7b_irip_nm10_proj 10 40 numerics.filter=projection
b7 b7b_irip_nm16_proj 16 64 numerics.filter=projection
hot() { n=$1; shift; run "$O" $n "$M" grid.modes=32 grid.t_end=333.56fs laser.enabled=false \
    "species electrons.ppc_theta=128" "species ions.ppc_theta=128" "species electrons.density_function=1" "species ions.density_function=1" \
    "species electrons.uth_z=0.01" "species electrons.uth_x=0.01" "species electrons.uth_y=0.01" output.openpmd_every=0 "$@"; }
hot b7b_hot_irip_nm32;  hot b7b_hot_irip_nm32_f0 numerics.filter_passes=0

# Bubble on FBPIC's grid and the h_r scan (Table bubblerz)
D=$H/decks/bubble_rz.deck
brun() { n=$1; shift; mkdir -p "$O/$n"; echo "start $n"; ( cd "$O/$n" && "$X" "$@" output.dir=out ) > "$O/$n.log" 2>&1; echo "end $n $?"; }
brun bub_irz_h5 "$D";  brun bub_rz_h5 "$D" numerics.solver=rip_cc;  brun bub_irz_h5_f1 "$D" numerics.filter_passes=1
brun bub_irz_h2 "$D" grid.nr=200 "grid.dr=0.1 um";  brun bub_rz_h2 "$D" grid.nr=200 "grid.dr=0.1 um" numerics.solver=rip_cc
brun bub_irz_h1 "$D" grid.nr=400 "grid.dr=0.05 um";  brun bub_irz_h05 "$D" grid.nr=800 "grid.dr=0.025 um"
