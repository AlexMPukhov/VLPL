#!/bin/bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -t 01:00:00
#SBATCH -o rip_%j.log
# VLPL run of the benchmark (the CPU cluster): sbatch run_rip.sbatch from bench/ionization_smilei (EXE: the vlpl binary)
EXE=${EXE:-../../build/vlpl}
export OMP_NUM_THREADS=12 OMP_PROC_BIND=false
$EXE ion.deck > rip.log 2>&1
$EXE ion.deck "species hydrogen.ionization_current=false" "species carbon.ionization_current=false" output.dir=out_rip_nocur > rip_nocur.log 2>&1
