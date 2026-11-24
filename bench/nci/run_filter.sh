#!/bin/sh
# The streaming-plasma NCI test with the current filter (decision 91): no filter, projection, local and binomial (one pass),
# RIP-CC and iRIP, 3D (examples/streaming_plasma.deck) and RZ (bench/nci/rz_nci.deck: nr 16, dr = 3 c dt, the paper set-up), gamma 10 and 50,
# and the fine transverse grid h_perp = c dt / 4 (iRIP, gamma 10). Prints W_field / K_e at the end and the growth rate.
# Usage: sh bench/nci/run_filter.sh build/vlpl [outdir]   (about 1 h on 16 cores of the epyc node)
X=${1:-build/vlpl}; O=${2:-nci_filter}
mkdir -p $O
D3=examples/streaming_plasma.deck; DR=bench/nci/rz_nci.deck
run() { n=$1; shift; [ -f $O/$n/diag.txt ] || $X "$@" output.dir=$O/$n > $O/$n.log 2>&1; }
for g in 10 50; do for sv in rip_cc irip; do for fm in none projection local binomial; do
    fp=1; fa=numerics.filter=$fm; [ $fm = none ] && { fp=0; fa=numerics.filter=binomial; }
    run c3_${sv}_g${g}_$fm $D3 numerics.solver=$sv numerics.filter_passes=$fp $fa "species electrons.ux=-$g" "species protons.ux=-$g"
    run rz_${sv}_g${g}_$fm $DR numerics.solver=$sv numerics.filter_passes=$fp $fa "species electrons.uz=-$g" "species protons.uz=-$g"
done; done; done
for fm in none projection binomial; do
    fp=1; fa=numerics.filter=$fm; [ $fm = none ] && { fp=0; fa=numerics.filter=binomial; }
    run c3_irip_hq_$fm $D3 numerics.solver=irip numerics.filter_passes=$fp $fa grid.ny=128 grid.nz=128 grid.dy=0.25068um grid.dz=0.25068um
    run rz_irip_hq_$fm $DR numerics.solver=irip numerics.filter_passes=$fp $fa grid.nr=64 grid.dr=0.25068um
done
# W_field / K_e (K_e = W_kin(0) / (1 + 1836), both species at the same gamma) and the energy growth rate in wp (n = 0.01 n_c)
for d in $O/*/; do
    awk -v n=$(basename $d) '!/^#/ { if (k0 == 0 && $4 > 0) k0 = $4; r = $3 / k0 * 1837; w = r
        if (r > 1e-6 && r < 1e-3) { if (t1 == 0) { t1 = $2; r1 = r }; t2 = $2; r2 = r } }
        END { printf "%-24s W_field/K_e %9.2e", n, w; if (t2 > t1) printf "  growth %.3f wp", log(r2 / r1) / ((t2 - t1) * 0.1); print "" }' $d/diag.txt
done
