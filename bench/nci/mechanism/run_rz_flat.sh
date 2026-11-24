#!/bin/sh
# The NCI on the boosted RZ LWFA grid (README.md here): RZ, rip_cc, gamma 6, dz' = 0.30163 um (lab lambda0 / 32, lambda0 = 0.81 um),
# dr = 0.405 um, 128 x 16 cells, 2 modes, n0 = 3.118936e-4 n_c, shape 2, seed isotropic in the plasma rest frame (u_th 1e-3).
#   NPAR=6 PY=python3 sh run_rz.sh VLPL OUTDIR;  python3 wk_rz.py OUTDIR 3.118936e-4
X=$1; O=${2:-nci_rz}; H=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python3}
mkdir -p $O; O=$(cd $O && pwd)
T6=$(awk 'BEGIN{printf "%.1f", 600/sqrt(3.118936e-4)/(2*3.14159265)*3.33564*0.81}')
T2=$(awk 'BEGIN{printf "%.1f", 250/sqrt(3.118936e-4)/(2*3.14159265)*3.33564*0.81}')
B="grid.lambda0=0.81um grid.dz=0.30163um grid.dr=0.405um grid.nz=128 grid.nr=16 numerics.solver=rip_cc numerics.shape=2 output.every=50"
B="$B species\ electrons.density=1.871362e-3 species\ protons.density=1.871362e-3 species\ electrons.uz=-5.91608 species\ protons.uz=-5.91608"
B="$B species\ electrons.ppc_z=3 species\ protons.ppc_z=3 species\ electrons.uth_z=6e-3 species\ electrons.uth_x=1e-3 species\ electrons.uth_y=1e-3"
C=numerics.filter_compensate=true; Z=numerics.filter_axes=x
D=$H/../boost_stability/lin_rz.deck
: > $O/cmds.txt
for v in "f6L4z 6 numerics.filter_flat=4 $Z" "f8L4z 8 numerics.filter_flat=4 $Z" "f12L6z 12 numerics.filter_flat=6 $Z"; do
  set -- $v; n=$1; p=$2; shift 2
  echo "$X $D $B grid.t_end=${T6}fs numerics.filter_passes=$p $* output.dir=$O/$n > $O/$n.log 2>&1" >> $O/cmds.txt
done
xargs -d '\n' -P ${NPAR:-6} -I{} sh -c '{}' < $O/cmds.txt
