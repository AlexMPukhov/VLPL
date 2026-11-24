#!/bin/sh
# Laser-plasma coupling through the filter (README.md here): warm plasma at rest (n = 0.1 n_c, u_th = 0.05), periodic 128 x 8 cells,
# dx = lambda0 / 32, rip_cc; the EM mode kx D = 2 pi / 32 (= k0 at lambda / 32) from the thermal noise, em_coupling.py.
#   PY=python3 sh run_em.sh VLPL OUTDIR
X=$1; O=${2:-nci_em}; H=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python3}
mkdir -p $O; cd $O
C=numerics.filter_compensate=true
for v in "f0 0" "f1 1" "f2 2" "f4 4" "f1c 1 $C" "f2c 2 $C" "f4c 4 $C" "f8c 8 $C"; do
  set -- $v; n=$1; p=$2; shift 2
  $X $H/../boost_stability/lin_c2.deck numerics.solver=rip_cc numerics.shape=2 numerics.filter_passes=$p $* grid.nx=128 grid.ny=8 \
    grid.dx=0.03125um grid.dy=0.0625um grid.t_end=${TEND:-312.7fs} "species electrons.ux=0" "species protons.ux=0" \
    "species electrons.density=0.1" "species protons.density=0.1" "species electrons.uth_x=0.05" "species electrons.uth_y=0.05" \
    "species electrons.uth_z=0.05" output.every=100 output.dir=$n output.openpmd_every=1 "output.openpmd_fields=E B" > $n.log 2>&1
  $PY $H/em_coupling.py $n 4
done
