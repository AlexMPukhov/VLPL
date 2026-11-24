#!/bin/sh
# Build VLPL for the JUPITER / JUWELS Booster runs, on a login node of the same system (JUPITER: aarch64
# Grace + GH200; JUWELS Booster: x86 + A100):
#   build-gpu-mpi  CUDA + MPI (one GPU per rank), HDF5 if found: vlpl and tests
#   build-cpu-mpi  CPU + MPI + OpenMP, the same options: the references of check.sbatch and the CPU timing
# usage: sh bench/jupiter/build.sh           (settings in env.sh; HDF5=0 builds without HDF5)
set -e
. "$(dirname "$0")/env.sh"
load_modules
cd "$SRC"
command -v nvcc > /dev/null || { echo "nvcc not found: load the CUDA module (MODULES in env.sh)"; exit 1; }
command -v mpicxx > /dev/null || { echo "mpicxx not found: load the MPI module (MODULES in env.sh)"; exit 1; }
echo "nvcc:   $(nvcc --version | tail -1)"
echo "mpicxx: $(mpicxx --version | head -1)"
# HDF5 (openPMD output, checkpoints and the restart cases of tests/mpi.sh): the JSC module sets EBROOTHDF5
H5=""
if [ "${HDF5:-}" = 0 ]; then H5="HDF5=0"
elif [ -n "${HDF5_DIR:-${EBROOTHDF5:-}}" ]; then H5="HDF5_DIR=${HDF5_DIR:-$EBROOTHDF5}"; fi
echo "HDF5:   ${H5:-pkg-config}"
# shellcheck disable=SC2086
make -j 16 CUDA=1 MPI=1 CUDA_ARCH=-arch=$CUDA_ARCH BUILD="$BUILD_GPU" $H5
# shellcheck disable=SC2086
make -j 16 MPI=1 ARCH="$CPU_ARCH" BUILD="$BUILD_CPU" $H5
echo
echo "built: $BUILD_GPU/{vlpl,tests} (GPU), $BUILD_CPU/{vlpl,tests} (CPU)"
echo "next:  sbatch --account=\$ACCOUNT bench/jupiter/check.sbatch   (see bench/jupiter/README.md)"
