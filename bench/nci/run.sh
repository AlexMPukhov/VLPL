#!/bin/sh
# Numerical Cherenkov benchmark (docs/specs/yee.md): the streaming plasma of examples/streaming_plasma.deck
# with RIP-CC and with Yee; then python3 bench/nci/plot.py.   sh bench/nci/run.sh build/vlpl
EXE=${1:-build/vlpl}
for s in rip_cc yee; do
    $EXE examples/streaming_plasma.deck numerics.solver=$s output.every=5 output.dir=nci_$s
    cp nci_$s/diag.txt bench/nci/diag_$s.txt
done
