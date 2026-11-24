#!/bin/sh
# Case B: the 0.1 um witness in the guided linear wake (wake.deck, rz_wake.deck) with the VLPL solvers.
#   sh bench/emittance/run_wake.sh build/vlpl [case ...]      -> emB_<case>/beam_beam.txt
# Without cases: all. PRE: output prefix (emB), TEND: a shorter t_end. 3D: ~1 h each on the 32-core epyc node.
EXE=${1:-build/vlpl}; shift
D=bench/emittance/wake.deck
R=bench/emittance/rz_wake.deck
run() {
    name=$1; shift
    case " $SEL " in *" $name "*|"  ") ;; *) return ;; esac
    o=${PRE:-emB}_$name
    if [ -n "$TEND" ]; then set -- "$@" "grid.t_end=$TEND"; fi
    if [ -n "$XTRA" ]; then set -- "$@" "$XTRA"; fi          # XTRA: one more override
    $EXE "$@" output.dir=$o > $o.log 2>&1
    echo "$name: $(tail -1 $o/beam_beam.txt | awk '{print "E", $5, "eps_ny", $8, "eps_nz", $9}')"
}
SEL="$*"
run rip           $D
run irip          $D numerics.solver=irip
# h_perp = 0.75 c dt: dx = 0.1 um (lambda / 8), 160 cells
run irip_dx01     $D numerics.solver=irip "grid.dx=0.1 um" grid.nx=160
run rz            $R
run irz           $R numerics.solver=irip
# h_perp = c dt / 2: dr = 0.025 um, 360 cells
run irz_dr025     $R numerics.solver=irip "grid.dr=0.025 um" grid.nr=360
