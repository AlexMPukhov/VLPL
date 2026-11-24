#!/bin/sh
# Regression harness (docs/specs/regression.md): runs the cases of tests/regression/cases.txt with EXE
# and compares each diag.txt with tests/regression/ref/<name>.txt.
#   sh tests/regression/run.sh build/vlpl [--bless]
EXE=${1:-build/vlpl}
BLESS=${2:-}
DIR=$(dirname "$0")
OUT=${REGRESSION_OUT:-regression_out}
mkdir -p "$OUT"; rm -f "$OUT/.fail"
fail=0
grep -v '^#' "$DIR/cases.txt" | while IFS='|' read -r name deck args; do
    name=$(echo "$name" | tr -d ' ')
    deck=$(echo "$deck" | tr -d ' ')
    [ -z "$name" ] && continue
    # overrides separated by ';' (a block name may contain a space: "species electrons.ppc_x=2")
    set -- "$deck"
    old_ifs=$IFS; IFS=';'
    for a in $args; do
        a=$(echo "$a" | sed 's/^ *//; s/ *$//')
        [ -n "$a" ] && set -- "$@" "$a"
    done
    IFS=$old_ifs
    rm -rf "$OUT/$name"
    start=$(date +%s)
    if ! "$EXE" "$@" "output.dir=$OUT/$name" > "$OUT/$name.log" 2>&1; then
        echo "regression $name: FAIL (run failed, see $OUT/$name.log)"; echo 1 > "$OUT/.fail"; continue
    fi
    secs=$(( $(date +%s) - start ))
    if [ "$BLESS" = "--bless" ]; then
        cp "$OUT/$name/diag.txt" "$DIR/ref/$name.txt"
        echo "regression $name: reference written (${secs} s)"
    elif python3 "$DIR/compare.py" "$DIR/ref/$name.txt" "$OUT/$name/diag.txt"; then
        echo "regression $name: ok (${secs} s)"
    else
        echo "regression $name: FAIL"; echo 1 > "$OUT/.fail"
    fi
done
[ "$BLESS" = "--bless" ] && exit 0
python3 "$DIR/compare.py" --selftest "$DIR/ref/stream.txt" || echo 1 > "$OUT/.fail"
if [ -f "$OUT/.fail" ]; then rm -f "$OUT/.fail"; echo "regression: FAILED"; exit 1; fi
echo "regression: all cases match the references"
