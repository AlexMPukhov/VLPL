#!/bin/sh
# mpirun-style front end for Slurm (MPIRUN of tests/mpi.sh on a cluster whose head node must not run the
# ranks, e.g. the CI runner on the CPU cluster, the CI notes of the development repository):  mpirun_srun.sh -np N exe args...  ->  srun -n N exe args...
# Extra srun options (partition, time) come from SRUN_OPTS, e.g. SRUN_OPTS="-p <partition> -t 10".
np=1
while [ $# -gt 0 ]; do
    case "$1" in
        -np|-n) np=$2; shift 2 ;;
        --oversubscribe) shift ;;
        *) break ;;
    esac
done
exec srun -n "$np" ${SRUN_OPTS:-} "$@"
