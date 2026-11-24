#!/bin/sh
# NCI mechanism scan (README.md here): 2D periodic streaming e-p plasma (../boost_stability/lin_c2.deck), fields dumped every step;
# per run nci_spec.py writes RUN.spec (the strongest modes, their growth and frequency against the beam aliases) and RUN.png,
# then the dumps are removed.
#   TW=250 NPAR=6 PY=python3 sh scan.sh VLPL OUTDIR LIST
#   LIST lines: name solver shape passes gamma n0 Nlab h_um uth nx [deck arguments ...]
#     gamma: boost; n0: lab density (n_c, lambda0 = 1 um); Nlab: lab cells per lambda0 (dx' = gamma (1 + beta) / Nlab um);
#     h_um: transverse cell; uth: the seed (all electron components; a rest-frame isotropic seed sets uth_x = gamma uth_perp)
#   python3 wk_table.py OUTDIR   (W_field / K_e at omega_p t = 10, 100, 300, 600)
X=$1; O=$2; L=$3; H=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python3}
mkdir -p $O; O=$(cd $O && pwd); : > $O/cmds.txt
while read name sv sh fp g n0 N h uth nx extra; do
  case "$name" in ""|\#*) continue;; esac
  set -- $(awk -v g=$g -v n0=$n0 -v N=$N -v TW=${TW:-250} 'BEGIN{b=sqrt(1-1/g^2); printf "%.6f %.6f %.6f %.3f", g*(1+b)/N, -g*b, g*n0, TW/sqrt(n0)/1.883652}')
  dx=$1; u=$2; n=$3; T=$4
  a="numerics.solver=$sv numerics.shape=$sh numerics.filter_passes=$fp output.every=10 output.dir=$O/$name grid.t_end=${T}fs"
  a="$a grid.nx=$nx grid.dx=${dx}um grid.dy=${h}um output.openpmd_every=1 'output.openpmd_fields=E B'"
  a="$a 'species electrons.ux=$u' 'species protons.ux=$u' 'species electrons.density=$n' 'species protons.density=$n'"
  a="$a 'species electrons.uth_x=$uth' 'species electrons.uth_y=$uth' 'species electrons.uth_z=$uth' $extra"
  echo "$X $H/../boost_stability/lin_c2.deck $a > $O/$name.log 2>&1; $PY $H/nci_spec.py $O/$name $sv $u --n0 $n0 --png $O/$name.png > $O/$name.spec 2>&1 && rm -r ${O:?}/${name:?}/openpmd" >> $O/cmds.txt
done < $L
xargs -d '\n' -P ${NPAR:-4} -I{} sh -c '{}' < $O/cmds.txt
