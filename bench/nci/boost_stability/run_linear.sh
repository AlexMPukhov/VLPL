#!/bin/sh
# Linear stability of the boosted-frame plasma (README.md here, decision 104): a neutral e-p plasma streaming at -beta c through a
# periodic box on the grids of a boosted LWFA. Lab LWFA: lambda0 = 1 um, dx_lab = lambda / 32, n0 = 0.004 n_c (gamma 5, 10) or
# 0.002 n_c (gamma 20); boosted dx' = gamma (1 + beta) dx_lab; in the box n = gamma n0, u = -gamma beta; seed u_th = 1e-2.
# 300 / omega_p (omega_p of n0). Scan: gamma 5, 10, 20; h_perp = 0.25, 1 um; filter_passes 0, 1, 2; shape 1, 2; RZ gather plain,
# matched; RIP-CC (where its Courant limit allows) and iRIP; 2D, 3D (subset) and RZ: 148 runs, ~20 core-hours.
#   sh bench/nci/boost_stability/run_linear.sh build/vlpl OUTDIR [NPAR=6]   (OMP_NUM_THREADS per run)
#   sh bench/nci/boost_stability/table.sh OUTDIR
X=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); O=${2:-bs_linear}; P=${3:-6}
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p $O; O=$(cd $O && pwd); : > $O/cmds.txt
for g in 5 10 20; do
  n0=0.004; [ $g = 20 ] && n0=0.002
  set -- $(awk -v g=$g -v n0=$n0 'BEGIN{b=sqrt(1-1/g^2); printf "%.6f %.6f %.6f %.4f", g*(1+b)/32, -g*b, g*n0, 300/sqrt(n0)/(2*3.14159265)*3.33564}')
  dx=$1; u=$2; n=$3; T=$4
  for h in 0.25 1; do
    for geo in c2 c3 rz; do for sv in irip rip_cc; do for fp in 0 1 2; do for sh in 1 2; do for ga in plain matched; do
      [ $geo != rz ] && [ $ga = matched ] && continue
      if [ $sv = rip_cc ]; then        # RIP-CC: Courant limit dx'^2 / h_perp^2 < 1 (2D), < 1/2 (3D); RZ: dz' well below dr
        [ $h = 0.25 ] && continue; [ $g = 20 ] && continue
        [ $geo = c3 ] && [ $fp != 0 ] && continue; [ $geo = rz ] && [ $fp = 2 ] && continue
      fi
      if [ $geo = c3 ]; then [ $sv = irip ] && { [ $h = 1 ] || [ $g = 5 ] || [ $fp = 2 ]; } && continue; fi
      name=${geo}_${sv}_g${g}_h${h}_f${fp}_s${sh}; [ $geo = rz ] && name=${name}_$ga
      a="numerics.solver=$sv numerics.shape=$sh numerics.filter_passes=$fp output.every=10 output.dir=$O/$name grid.t_end=${T}fs"
      if [ $geo = rz ]; then
        a="$a grid.dz=${dx}um grid.dr=${h}um numerics.rz_gather=$ga 'species electrons.uz=$u' 'species protons.uz=$u'"
      else
        a="$a grid.dx=${dx}um grid.dy=${h}um 'species electrons.ux=$u' 'species protons.ux=$u'"
        [ $geo = c3 ] && a="$a grid.dz=${h}um"
      fi
      a="$a 'species electrons.density=$n' 'species protons.density=$n'"
      echo "$X $H/lin_$geo.deck $a > $O/$name.log 2>&1" >> $O/cmds.txt
    done; done; done; done; done
  done
done
xargs -d '\n' -P $P -I{} sh -c '{}' < $O/cmds.txt
sh $H/table.sh $O
