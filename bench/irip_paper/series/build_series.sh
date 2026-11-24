#!/bin/sh
# Build series.cpp against the VLPL sources (CPU, OpenMP) and run the four cases of Figs. energy3d and energyrz.
# Usage (from bench/irip_paper/series): sh build_series.sh   -> series_{cart_random,rz_random,cart_hot,rz_hot}.txt
set -e
H=$(cd "$(dirname "$0")" && pwd); V=$(cd "$H/../../.." && pwd)
mkdir -p "$H/obj" && cd "$H/obj"
for f in $(ls "$V"/src/*.cpp | grep -v main.cpp); do g++ -O2 -std=c++17 -fopenmp -I"$V/src" -c "$f" -o "$(basename "$f" .cpp).o"; done
g++ -O2 -std=c++17 -fopenmp -I"$V/src" -c "$H/series.cpp" -o series.o
g++ -fopenmp -o "$H/series" ./*.o
cd "$H"; export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OMP_PROC_BIND=false
for c in cart_random rz_random cart_hot; do ./series $c series_$c.txt > $c.log 2>&1; done
./series rz_hot series_rz_hot.txt ../decks/rz_hot.deck > rz_hot.log 2>&1
