#!/bin/sh
# Case A (Fig. 4 of Pukhov 2020): the 10 GeV, 1 nm beam in vacuum with every VLPL solver, 6 and 0.6 nC.
#   sh bench/emittance/run_vacuum.sh build/vlpl [case ...]     -> emA_<case>/beam_beam.txt
# Without cases: all. OMP_NUM_THREADS as available (epyc: 32 threads, 6-25 min per 3D case at 1 mm).
EXE=${1:-build/vlpl}; shift
D=bench/emittance/vacuum.deck
R=bench/emittance/rz_vacuum.deck
LO="species beam.density=0.0761"                       # 0.6 nC
PER="grid.boundary_x=periodic"                          # periodic x
WS="window.start=0 fs"                                  # + window.velocity=1: the window at c
# Yee: c dt = 0.7 h_x as in the paper: cfl = 0.07 um * sqrt(1/0.1^2 + 2/0.15^2) um^-1
YEE="numerics.solver=yee numerics.cfl=0.96214"
run() {           # PRE: output prefix (emA), TEND: a shorter t_end (e.g. "20 fs")
    name=$1; shift
    case " $SEL " in *" $name "*|"  ") ;; *) return ;; esac
    o=${PRE:-emA}_$name
    if [ -n "$TEND" ]; then set -- "$@" "grid.t_end=$TEND"; fi
    if [ -n "$XTRA" ]; then set -- "$@" "$XTRA"; fi          # XTRA: one more override
    $EXE "$@" output.dir=$o > $o.log 2>&1
    echo "$name: $(tail -1 $o/beam_beam.txt | awk '{print "eps_ny", $8, "eps_nz", $9}')"
}
SEL="$*"
run yee_6         $D $PER $YEE
run yee_06        $D $PER $YEE "$LO"
run ripp_6        $D $PER
run ripp_06       $D $PER "$LO"
run ripw_6        $D "$WS" window.velocity=1
run ripw_06       $D "$WS" window.velocity=1 "$LO"
run irip_6        $D "$WS" window.velocity=1 numerics.solver=irip
run irip_06       $D "$WS" window.velocity=1 numerics.solver=irip "$LO"
# h_perp = c dt / 2: h_x = 0.3 um (sigma_x = 4.7 cells), 86 cells = 25.8 um
run irip3_6       $D "$WS" window.velocity=1 numerics.solver=irip "grid.dx=0.3 um" grid.nx=86 "species beam.density_function=gauss(x - 12.9 um, 1.41 um) * gauss(sqrt((y - 9.6 um)^2 + (z - 9.6 um)^2), 1.41 um) * (abs(x - 12.9 um) < 5.64 um) * (sqrt((y - 9.6 um)^2 + (z - 9.6 um)^2) < 5.64 um)"
run rz_6          $R "$WS" window.velocity=1
run rz_06         $R "$WS" window.velocity=1 "$LO"
run irz_6         $R "$WS" window.velocity=1 numerics.solver=irip
run irz_06        $R "$WS" window.velocity=1 numerics.solver=irip "$LO"
# iRZ at dr = c dt / 2 (dr = 0.05 um, 192 cells: the same 9.6 um)
run irz_dr05_6    $R "$WS" window.velocity=1 numerics.solver=irip "grid.dr=0.05 um" grid.nr=192
