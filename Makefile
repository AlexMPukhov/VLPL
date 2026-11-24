# make                 CPU build with OpenMP (build/vlpl, build/tests)
# make MPI=1           with MPI (mpicxx): BUILD=build-mpi
# make ARCH=-march=native  CPU build for this machine's instruction set (AVX2/AVX-512, FMA: ~10 % faster,
#                      results differ at round-off); e.g. ARCH=-march=ivybridge for other nodes. Default: x86-64
# make agent_skill     the agent skills (skills/) against build/vlpl: decks, keys, restart (docs/agent_skill.md)
# make CUDA=1          GPU build (nvcc), e.g. CUDA_ARCH=-arch=sm_90 (GH200), sm_120 (RTX 5090); add MPI=1 for
#                      several GPUs (one GPU per rank): make CUDA=1 MPI=1 BUILD=build-gpu-mpi
# HDF5 (openPMD output and checkpoints, docs/specs/openpmd.md): on if pkg-config finds hdf5 or HDF5_DIR
#                      is given (HDF5_DIR=/path/to/hdf5 with include/ and lib/); HDF5=0 builds without;
#                      HDF5 1.12 or newer (an older one: warning, build without HDF5)
SRC = src/deck.cpp src/comm.cpp src/fields.cpp src/laser.cpp src/particles.cpp src/species.cpp src/tiles.cpp src/pipeline.cpp src/sim.cpp src/window.cpp src/expr.cpp src/beam_field.cpp src/beam_diag.cpp src/pml.cpp src/h5.cpp src/openpmd.cpp src/laser_file.cpp src/laser_file_rz.cpp src/yee.cpp src/irip.cpp src/irip_pml.cpp src/implicit_mover.cpp src/rz_fields.cpp src/rz_irip.cpp src/rz_particles.cpp src/rz_sim.cpp src/rz_laser.cpp src/rz_ruyten.cpp src/rz_filter.cpp src/rz_smooth.cpp src/rz_axis_current.cpp src/absorber_particles.cpp src/load.cpp src/xflux.cpp src/tfsf.cpp src/tfsf_modes.cpp src/side.cpp src/side_luminal.cpp src/filter.cpp src/hfocus.cpp src/rz_tfsf.cpp src/medium.cpp src/medium_sim.cpp src/medium_rz.cpp src/medium_rz_bound.cpp src/rz_beam_field.cpp src/rz_metal.cpp src/rz_metal_phi.cpp src/rz_dielectric.cpp src/atomic.cpp src/ionization.cpp src/boost.cpp src/boost_diag.cpp src/boost_antenna.cpp src/prof.cpp src/qed.cpp src/qed_tables.cpp src/metal3d.cpp src/beam_import.cpp src/wake.cpp src/structure.cpp src/collisions.cpp src/impact.cpp
HDR = $(wildcard src/*.hpp)
BUILD ?= build
.RECIPEPREFIX = >
ifeq ($(CUDA),1)
  NVCC ?= nvcc
  CUDA_ARCH ?= -arch=native
  CXX := $(NVCC)
  CXXFLAGS = -O3 -std=c++17 $(CUDA_ARCH) -x cu --extended-lambda --expt-relaxed-constexpr -Xcompiler -Wall $(NVCCFLAGS)
  ifeq ($(MPI),1)
    CXXFLAGS += -ccbin mpicxx -DRIP_MPI -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
  endif
else
  ifeq ($(MPI),1)
    CXX := mpicxx
    CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -fopenmp $(ARCH) -DRIP_MPI -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
  else
    CXX ?= g++
    CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -fopenmp $(ARCH)
  endif
endif

HDF5 ?= $(if $(HDF5_DIR),1,$(shell pkg-config --exists hdf5 2>/dev/null && echo 1 || echo 0))
# HDF5 1.12 or newer (decision 113). The version is read from H5public.h (HDF5_DIR/include, the -I paths of
# pkg-config, CPLUS_INCLUDE_PATH, CPATH, the system paths); an older HDF5 or no header: a warning and a build
# without HDF5. A parallel HDF5 (H5_HAVE_PARALLEL) includes mpi.h: a build without MPI=1 takes the include and
# library paths of $(MPICXX) (mpicxx --showme or -show). Test: tests/hdf5_version.sh (part of make test).
ifeq ($(HDF5),1)
  ifdef HDF5_DIR
    H5_CFLAGS := -I$(HDF5_DIR)/include
    ifeq ($(CUDA),1)
      H5_LIBS := -L$(HDF5_DIR)/lib -Xlinker -rpath=$(HDF5_DIR)/lib -lhdf5
    else
      H5_LIBS := -L$(HDF5_DIR)/lib -Wl,-rpath,$(HDF5_DIR)/lib -lhdf5
    endif
  else
    H5_CFLAGS := $(shell pkg-config --cflags hdf5 2>/dev/null)
    H5_LIBS := $(shell pkg-config --libs hdf5 2>/dev/null)
  endif
  H5_INCDIRS := $(patsubst -I%,%,$(filter -I%,$(H5_CFLAGS))) $(subst :, ,$(CPLUS_INCLUDE_PATH) $(CPATH)) \
                /usr/include /usr/local/include /usr/include/hdf5/serial
  H5_HEADER := $(firstword $(wildcard $(addsuffix /H5public.h,$(H5_INCDIRS))))
  ifeq ($(H5_HEADER),)
    $(warning warning: HDF5: H5public.h not found (HDF5_DIR, pkg-config hdf5): building without HDF5 (openPMD output, checkpoints and file input disabled))
    override HDF5 := 0
  else
    # "<version> <1 if >= 1.12> <1 if parallel>"
    H5_INFO := $(shell awk '/define H5_VERS_MAJOR/{a=$$3} /define H5_VERS_MINOR/{b=$$3} /define H5_VERS_RELEASE/{c=$$3} \
                 /define H5_HAVE_PARALLEL 1/{p=1} END{print a"."b"."c, (a>1||(a==1&&b>=12))?1:0, p?1:0}' \
                 $(H5_HEADER) $(dir $(H5_HEADER))H5pubconf.h 2>/dev/null)
    H5_VERSION := $(word 1,$(H5_INFO))
    ifneq ($(word 2,$(H5_INFO)),1)
      $(warning warning: HDF5 $(H5_VERSION) in $(dir $(H5_HEADER)) is too old (need 1.12 or newer): building without HDF5 (openPMD output, checkpoints and file input disabled); set HDF5_DIR to HDF5 1.12 or newer)
      override HDF5 := 0
    else
      CXXFLAGS += -DRIP_HDF5 $(H5_CFLAGS)
      LDLIBS = $(H5_LIBS)
      ifeq ($(word 3,$(H5_INFO))$(filter 1,$(MPI)),1)
        MPICXX ?= mpicxx
        H5_MPI := $(shell $(MPICXX) --showme 2>/dev/null || $(MPICXX) -show 2>/dev/null)
        ifeq ($(filter -I%,$(H5_MPI)),)
          $(warning warning: HDF5 $(H5_VERSION) is a parallel build, but $(MPICXX) gives no MPI include path (mpi.h))
        endif
        CXXFLAGS += $(filter -I%,$(H5_MPI)) -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
        LDLIBS += $(filter -L%,$(H5_MPI)) $(filter -l%,$(H5_MPI))
      endif
    endif
  endif
endif

all: $(BUILD)/vlpl $(BUILD)/tests
$(BUILD)/vlpl: $(SRC) src/main.cpp $(HDR) data/ionization_energies.def data/binding_energies.def
> @mkdir -p $(BUILD)
> $(CXX) $(CXXFLAGS) -o $@ $(SRC) src/main.cpp $(LDLIBS)
TESTS = tests/tests.cpp tests/test_window.cpp tests/test_profile.cpp tests/test_beam.cpp tests/test_diag.cpp tests/test_pml.cpp tests/test_laser.cpp tests/test_openpmd.cpp tests/test_lasy.cpp tests/test_yee.cpp tests/test_irip.cpp tests/test_rz.cpp tests/test_rz_grid.cpp tests/test_xflux.cpp tests/test_tfsf.cpp tests/test_side.cpp tests/test_solid.cpp tests/test_cart2d.cpp tests/test_hfocus.cpp tests/test_rz_tfsf.cpp tests/test_rz_shape.cpp tests/test_rz_axis.cpp tests/test_mover.cpp tests/test_medium.cpp tests/test_medium2.cpp tests/test_bound.cpp tests/test_rz_metal.cpp tests/test_rz_metal_phi.cpp tests/test_rz_dielectric.cpp tests/test_ionization.cpp tests/test_boost.cpp tests/test_boost_antenna.cpp tests/test_overlap.cpp tests/test_qed.cpp tests/test_pairs.cpp tests/test_structure.cpp tests/test_rz_structure.cpp tests/test_collisions.cpp
$(BUILD)/tests: $(SRC) $(TESTS) $(HDR) tests/common.hpp data/ionization_energies.def data/binding_energies.def
> @mkdir -p $(BUILD)
> $(CXX) $(CXXFLAGS) -Isrc -o $@ $(SRC) $(TESTS) $(LDLIBS)
test: $(BUILD)/tests
> $(BUILD)/tests
> sh tests/hdf5_version.sh
> sh tests/agent_skill.sh --static
# agent skills (skills/, docs/agent_skill.md): every deck and key of the skills against the executable; MPIVLPL = an MPI
# executable (optional: 2 ranks against 1), the environment variable MPIRUN its launcher (default mpirun)
agent_skill: $(BUILD)/vlpl
> sh tests/agent_skill.sh $(BUILD)/vlpl $(MPIVLPL)
regression: $(BUILD)/vlpl
> sh tests/regression/run.sh $(BUILD)/vlpl
clean:
> rm -rf build build-mpi build-gpu build-gpu-mpi regression_out
.PHONY: all test regression agent_skill clean
