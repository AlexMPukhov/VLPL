#!/bin/sh
# Boosted-frame drift boxes (paragraph "Boosted-frame grids"): one line of cases.txt per run.
# Usage (from bench/irip_paper/boost_nci): X=../../../build/vlpl sh run.sh <line number> ; reduce: python reduce2.py g10_f2 ... (all: reduce.py)
H=$(cd "$(dirname "$0")" && pwd); X=${X:-$H/../../../build/vlpl}
L=$(sed -n "${1}p" "$H/cases.txt"); N=${L%% *}; A=${L#* }
eval "\"$X\" \"$H/box3d.deck\" output.dir=$H/$N $A" > "$H/$N.log" 2>&1; echo "rc=$? $N"
