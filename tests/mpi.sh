#!/bin/sh
# MPI check: the streaming-plasma deck (shortened) on 1 and on 8 ranks (2 x 2 x 2) must give the same
# diagnostics up to round-off (summation order of the current deposition differs), also with the Yee solver
# (docs/specs/yee.md) and with iRIP (8 ranks along x). Second case: the
# same plasma, transversely hot, leaving through the open x boundary (docs/specs/open_boundary.md),
# with the Gauss law at round-off on 8 ranks; the same with CPML layers in y and z (docs/specs/pml.md).
# Third case: the LWFA deck in a moving window (small box,
# hot plasma with a density ramp entering at the right edge, docs/specs/moving_window.md; also with the current filter,
# decision 107). Fourth case:
# a relativistic beam started with its self-field (the global solve, docs/specs/beam_self_field.md), with
# its beam diagnostics (docs/specs/beam_diagnostics.md). Fifth: the moving window with CPML layers
# (the CPML memory at the left face moves with the window); with HDF5 also the restart from a
# checkpoint of 8 ranks on 8 ranks (exactly the uninterrupted run) and on 1 and 3 ranks
# (docs/specs/openpmd.md). Last: RZ with azimuthal modes (RIP-CC and iRIP, ranks along z; plasma with
# electrons absorbed by the wall, shape 1 and 2, a stretched radial grid, the laser through open z, the moving window, the restart of 8 ranks on 8 and 3;
# docs/specs/rz_modes.md); the energy through the open x edges (docs/specs/xflux.md). Dense slab: an overdense layer
# (6 n_c) hit by the laser, the Gauss drift on 8 ranks below 1e-13 (decision 58: 2e-12 before, 4e-14 as on 1 rank).
# O6: the implicit mover on the streaming plasma (periodic and open x, docs/specs/implicit_mover.md) and in RZ (m = 0).
# Side injection: a plane-wave pulse through the TFSF tube (docs/specs/side_injection.md), RIP-CC, iRIP, restart.
# O4 foil: a tilted target with pre-plasma and mobile ions, shape 3 and the current filter (docs/specs/solid_target.md).
# 2D: the foil of examples/foil_2d.deck (docs/specs/cart2d.md), RIP-CC, iRIP and CPML in y, restart.
# Field ionization: examples/ionization_2d.deck and rz_ionization.deck (docs/specs/field_ionization.md), ionization.txt too.
# Boosted frame: 3D RIP-CC and RZ iRIP with lab-frame snapshots (docs/specs/boosted_frame.md).
# Spin-resolved QED: examples/qed_collision.deck (docs/specs/qed.md), qed.txt too.
# Binary collisions: examples/collisions_2d.deck and rz_collisions.deck (docs/specs/collisions.md), collisions.txt too.
# Breit-Wheeler pairs: examples/qed_cascade.deck in a small box (decision 96), qed.txt too; the same with iRIP at 4 x the step,
# and in the boosted frame with lab snapshots, also across restarts (decision 117).
#   make MPI=1 BUILD=build-mpi && sh tests/mpi.sh build-mpi/vlpl
# NP: the number of ranks compared with 1 (default 8; "8 ranks" above means NP ranks), e.g. NP=4 on one node with
# 4 GPUs (bench/jupiter/check.sbatch).
EXE=${1:-build-mpi/vlpl}
MPIRUN=${MPIRUN:-mpirun --oversubscribe}
RESTART_EXACT=${RESTART_EXACT:-1}   # a restart on the same ranks: 1 byte-identical, 0 to round-off (GPU: atomic deposit order)
NP=${NP:-8}
export NP
set -e
export OMP_NUM_THREADS=1
compare() {     # 1-rank diag, N-rank diag, label[, limit of the Gauss drift on N ranks (default 1e-12)]
python3 - "$1" "$2" "$3" "${4:-1e-12}" <<'PY'
import os, sys
a=[l.split() for l in open(sys.argv[1]) if not l.startswith('#')]
b=[l.split() for l in open(sys.argv[2]) if not l.startswith('#')]
assert len(a)==len(b), "different number of lines"
worst=0
for x,y in zip(a,b):
    for c in (2,3,4):                       # W_field, W_kin, W_total; relative to 1e-14 W_total at least (W_kin
        # of a cold target is ~1e-21 W_total in the first steps, pure round-off)
        worst=max(worst,abs(float(x[c])-float(y[c]))/max(abs(float(x[c])),1e-14*abs(float(x[4])),1e-300))
    assert x[8:]==y[8:], "particle numbers differ: %s %s"%(x[8:],y[8:])
    assert float(y[5])<float(sys.argv[4]), "Gauss law drift %s on %s ranks"%(y[5],os.environ["NP"])
print("mpi%s: 1 vs %s ranks, max relative difference of the energies %.1e, Gauss drift %s, particles %s -> %s"
      % (sys.argv[3], os.environ["NP"], worst, b[-1][5], a[0][8:], b[-1][8:]))
sys.exit(0 if worst<1e-8 else 1)
PY
}
# the restart on the same ranks against the uninterrupted run: identical bytes (RESTART_EXACT=1), or to round-off
same_run() {
    if [ "$RESTART_EXACT" = 1 ]; then
        if cmp -s "$1" "$2"; then echo "mpi$3: $NP ranks from the checkpoint = the uninterrupted run, exactly"
        else echo "mpi$3: FAILED, $2 differs from $1 (RESTART_EXACT=1)"; exit 1; fi
    else     # medium.txt has its own columns (column 5 is the number of solves, not the Gauss drift)
        case "$1" in
            *medium.txt) mcompare "$1" "$2" "$3 (to round-off, RESTART_EXACT=0)" ;;
            *) compare "$1" "$2" "$3 (to round-off, RESTART_EXACT=0)" ;;
        esac
    fi
}
D=examples/streaming_plasma.deck
for np in 1 $NP; do $MPIRUN -np $np $EXE $D grid.t_end=200fs output.dir=mpi_$np output.every=20 > /dev/null; done
compare mpi_1/diag.txt mpi_$NP/diag.txt ""
for np in 1 $NP; do $MPIRUN -np $np $EXE $D numerics.solver=yee grid.t_end=100fs output.dir=mpi_yee_$np output.every=10 > /dev/null; done
compare mpi_yee_1/diag.txt mpi_yee_$NP/diag.txt " (Yee solver)"
# iRIP (docs/specs/implicit_rip.md): h_perp = h_x / 4, ranks along x only (8 x 1 x 1, the partitioned
# tridiagonal solve), transversely hot
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $D numerics.solver=irip grid.dy=0.25um grid.dz=0.25um grid.t_end=100fs output.every=10 \
        "species electrons.uth_y=0.3" "species electrons.uth_z=0.3" output.dir=mpi_irip_$np > /dev/null
done
compare mpi_irip_1/diag.txt mpi_irip_$NP/diag.txt " (iRIP solver, $NP ranks along x)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $D grid.boundary_x=open grid.t_end=200fs output.every=20 "species electrons.uth_y=0.3" \
        "species electrons.uth_z=0.3" output.dir=mpi_open_$np > /dev/null
done
compare mpi_open_1/diag.txt mpi_open_$NP/diag.txt " (open x)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $D numerics.solver=irip grid.dy=0.25um grid.dz=0.25um grid.boundary_x=open grid.t_end=100fs \
        output.every=10 "species electrons.uth_y=0.3" "species electrons.uth_z=0.3" output.dir=mpi_iopen_$np > /dev/null
done
compare mpi_iopen_1/diag.txt mpi_iopen_$NP/diag.txt " (iRIP, open x)"
# O6 implicit mover (docs/specs/implicit_mover.md): the same plasma, shape 2, periodic and open x (the nonlinear solve with
# global norms: the iteration count must not depend on the decomposition beyond the tolerance)
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $D numerics.solver=irip numerics.mover=implicit numerics.shape=2 numerics.mover_tol=1e-11 \
        grid.ny=8 grid.nz=8 grid.dy=0.25um grid.dz=0.25um grid.t_end=60fs output.every=6 "species electrons.uth_y=0.3" "species electrons.uth_z=0.3" \
        output.dir=mpi_imov_$np > /dev/null
    $MPIRUN -np $np $EXE $D numerics.solver=irip numerics.mover=implicit numerics.shape=2 numerics.mover_tol=1e-11 \
        grid.ny=8 grid.nz=8 grid.dy=0.25um grid.dz=0.25um grid.boundary_x=open grid.t_end=60fs output.every=6 \
        "species electrons.uth_y=0.3" "species electrons.uth_z=0.3" output.dir=mpi_imovo_$np > /dev/null
done
compare mpi_imov_1/diag.txt mpi_imov_$NP/diag.txt " (implicit mover, iRIP, $NP ranks along x)"
compare mpi_imovo_1/diag.txt mpi_imovo_$NP/diag.txt " (implicit mover, iRIP, open x)"
# O5 in 3D (decision 73): the transverse factor per mode in the y, z Fourier basis
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $D numerics.solver=irip numerics.closure=o5 grid.dy=0.25um grid.dz=0.25um grid.boundary_x=open \
        grid.t_end=100fs output.every=10 "species electrons.uth_y=0.3" "species electrons.uth_z=0.3" output.dir=mpi_io5_$np > /dev/null
done
compare mpi_io5_1/diag.txt mpi_io5_$NP/diag.txt " (iRIP, closure o5 in 3D, open x)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $D grid.boundary_y=pml grid.boundary_z=pml pml.cells=8 grid.t_end=200fs output.every=20 \
        "species electrons.uth_y=0.3" "species electrons.uth_z=0.3" output.dir=mpi_pml_$np > /dev/null
done
compare mpi_pml_1/diag.txt mpi_pml_$NP/diag.txt " (pml y, z)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $D numerics.solver=irip grid.dy=0.25um grid.dz=0.25um grid.boundary_y=pml grid.boundary_z=pml \
        pml.cells=8 grid.t_end=100fs output.every=10 "species electrons.uth_y=0.3" "species electrons.uth_z=0.3" \
        output.dir=mpi_ipml_$np > /dev/null
done
compare mpi_ipml_1/diag.txt mpi_ipml_$NP/diag.txt " (iRIP, pml y, z)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $D numerics.solver=irip numerics.closure=o5 grid.dy=0.25um grid.dz=0.25um grid.boundary_y=pml \
        grid.boundary_z=pml pml.cells=8 grid.t_end=100fs output.every=10 "species electrons.uth_y=0.3" "species electrons.uth_z=0.3" \
        output.dir=mpi_io5pml_$np > /dev/null
done
compare mpi_io5pml_1/diag.txt mpi_io5pml_$NP/diag.txt " (iRIP, closure o5, pml y, z)"
# dense slab (decision 58): the current must end where the stored position is, also on ranks with an offset
R=examples/laser_reflection.deck
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $R grid.nx=192 grid.ny=96 grid.nz=8 grid.t_end=30fs laser.y_center=2.4um \
        "species electrons.slab_x0=2.5um" "species electrons.slab_y0=2.4um" "species electrons.slab_angle=0" \
        "species protons.slab_x0=2.5um" "species protons.slab_y0=2.4um" "species protons.slab_angle=0" \
        "species electrons.ppc_x=2" "species electrons.ppc_y=1" "species protons.ppc_x=2" "species protons.ppc_y=1" \
         output.every=10 output.slice_every=0 output.dir=mpi_slab_$np > /dev/null
done
compare mpi_slab_1/diag.txt mpi_slab_$NP/diag.txt " (dense slab)" 1e-13
# O4 (docs/specs/solid_target.md): the tilted foil with pre-plasma and mobile protons, shape 3, current filter, TFSF laser at
# 45 deg, absorbing x; RIP-CC on NP ranks (2 x 2 x 2, the local filter: the projection needs ranks along x only), RIP-CC and
# iRIP on NP ranks along x with the projection filter (decision 69, the partitioned line solves); coarse (lambda0/20, 10 n_c)
# to stay short (the deck as written runs 3.8 M cells)
SO="grid.nx=480 grid.dx=0.05um grid.ny=128 grid.dy=0.15625um grid.nz=10 grid.t_end=60fs output.every=50"   # ny a power of 2 (iRIP DFT); nz: >= 5 per rank (shape 3 guards)
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/solid_target.deck $SO "species electrons.density=10" "species protons.density=10" \
        numerics.filter=local output.xflux_every=0 output.xflux=true output.dir=mpi_solid_$np > /dev/null
    $MPIRUN -np $np $EXE examples/solid_target.deck $SO "species electrons.density=10" "species protons.density=10" \
        grid.mpi_y=1 grid.mpi_z=1 output.dir=mpi_psolid_$np > /dev/null
    $MPIRUN -np $np $EXE examples/solid_target.deck $SO "species electrons.density=10" "species protons.density=10" \
        numerics.solver=irip grid.mpi_y=1 grid.mpi_z=1 output.dir=mpi_isolid_$np > /dev/null
    $MPIRUN -np $np $EXE examples/solid_target.deck $SO "species electrons.density=10" "species protons.density=10" \
        numerics.filter=binomial output.dir=mpi_bsolid_$np > /dev/null
done
compare mpi_solid_1/diag.txt mpi_solid_$NP/diag.txt " (O4 foil: shape 3, filter, mobile ions, TFSF 45 deg)" 1e-11
compare mpi_psolid_1/diag.txt mpi_psolid_$NP/diag.txt " (O4 foil, projection filter, $NP ranks along x)" 1e-11
compare mpi_isolid_1/diag.txt mpi_isolid_$NP/diag.txt " (O4 foil, iRIP, projection filter, $NP ranks along x)" 1e-11
compare mpi_bsolid_1/diag.txt mpi_bsolid_$NP/diag.txt " (O4 foil, binomial filter, ranks in x, y, z)" 1e-11
# the streaming plasma (gamma 10) with the binomial filter (decision 91) on ranks in x, y, z
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $D grid.t_end=200fs numerics.filter_passes=2 output.dir=mpi_bstream_$np output.every=20 > /dev/null
done
compare mpi_bstream_1/diag.txt mpi_bstream_$NP/diag.txt " (streaming plasma, binomial filter 2 passes)"
# the compensated filter along x only (decision 108): 4 passes + compensator, ranks in x, y, z
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $D grid.t_end=200fs numerics.shape=2 numerics.filter_passes=4 numerics.filter_compensate=true \
        numerics.filter_axes=x output.dir=mpi_cstream_$np output.every=20 > /dev/null
done
compare mpi_cstream_1/diag.txt mpi_cstream_$NP/diag.txt " (streaming plasma, 4 passes + compensator along x)"
# the maximally flat filter along x (decision 109): 6 passes and L = 4 (one real factor and a complex pair), ranks in x, y, z
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $D grid.t_end=200fs numerics.shape=2 numerics.filter_passes=6 numerics.filter_flat=4         numerics.filter_axes=x output.dir=mpi_fstream_$np output.every=20 > /dev/null
done
compare mpi_fstream_1/diag.txt mpi_fstream_$NP/diag.txt " (streaming plasma, flat filter N = 6, L = 4 along x)"
L=examples/lwfa_window.deck
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $L grid.nx=64 grid.ny=16 grid.nz=16 grid.t_end=80fs laser.waist=1um laser.waist_z=1um laser.duration=2.5fs window.start=10fs output.every=20 \
        output.slice_every=0 "species electrons.xmin=4um" "species protons.xmin=4um" "species electrons.uth_x=0.05" \
        "species electrons.uth_y=0.05" "species electrons.uth_z=0.05" output.dir=mpi_win_$np \
        "species electrons.density_function=x < 6 um ? (x - 4 um) / (2 um) : 1" \
        "species protons.density_function=x < 6 um ? (x - 4 um) / (2 um) : 1" > /dev/null
done
compare mpi_win_1/diag.txt mpi_win_$NP/diag.txt " (moving window)"
# a non-neutral warm column entering (decision 91): protons 1 per cell against electrons 2 x 2 x 2, ramp; the charge that
# enters is a background (gauss0), the Gauss drift at round-off on 1 and NP ranks
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $L grid.nx=64 grid.ny=16 grid.nz=16 grid.t_end=80fs laser.waist=1um laser.waist_z=1um laser.duration=2.5fs \
        window.start=10fs output.every=20 output.slice_every=0 "species electrons.xmin=4um" "species protons.xmin=4um" \
        "species electrons.ppc_x=2" "species electrons.ppc_y=2" "species electrons.ppc_z=2" "species protons.ppc_x=1" \
        "species protons.ppc_y=1" "species protons.ppc_z=1" "species electrons.uth_x=0.05" "species electrons.uth_y=0.05" \
        "species electrons.uth_z=0.05" output.dir=mpi_winnn_$np \
        "species electrons.density_function=x < 6 um ? 0.3 + 0.7 * (x - 4 um) / (2 um) : 1" \
        "species protons.density_function=x < 6 um ? 0.3 + 0.7 * (x - 4 um) / (2 um) : 1" > /dev/null
done
compare mpi_winnn_1/diag.txt mpi_winnn_$NP/diag.txt " (moving window, non-neutral warm column entering)"
# the same with the binomial current filter, 2 passes, in the lab window (decision 107): frozen for 3 + 2 cells
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $L grid.nx=64 grid.ny=16 grid.nz=16 grid.t_end=80fs laser.waist=1um laser.waist_z=1um laser.duration=2.5fs \
        window.start=10fs output.every=20 output.slice_every=0 "species electrons.xmin=4um" "species protons.xmin=4um" \
        "species electrons.ppc_x=2" "species electrons.ppc_y=2" "species electrons.ppc_z=2" "species protons.ppc_x=1" \
        "species protons.ppc_y=1" "species protons.ppc_z=1" "species electrons.uth_x=0.05" "species electrons.uth_y=0.05" \
        "species electrons.uth_z=0.05" numerics.shape=2 numerics.filter_passes=2 output.dir=mpi_winf_$np \
        "species electrons.density_function=x < 6 um ? 0.3 + 0.7 * (x - 4 um) / (2 um) : 1" \
        "species protons.density_function=x < 6 um ? 0.3 + 0.7 * (x - 4 um) / (2 um) : 1" > /dev/null
done
compare mpi_winf_1/diag.txt mpi_winf_$NP/diag.txt " (moving window, binomial filter 2 passes, non-neutral warm column)"
# the same with iRIP: h_perp = 0.8 dx (beyond RIP-CC's limit), open x, laser and window, 8 ranks along x
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $L numerics.solver=irip grid.nx=64 grid.ny=32 grid.nz=32 grid.dy=0.04um grid.dz=0.04um \
        grid.t_end=80fs laser.waist=0.3um laser.waist_z=0.3um laser.duration=2.5fs window.start=10fs output.every=20 \
        output.slice_every=0 "species electrons.xmin=4um" "species protons.xmin=4um" "species electrons.uth_x=0.05" \
        "species electrons.uth_y=0.05" "species electrons.uth_z=0.05" output.dir=mpi_iwin_$np \
        "species electrons.density_function=x < 6 um ? (x - 4 um) / (2 um) : 1" \
        "species protons.density_function=x < 6 um ? (x - 4 um) / (2 um) : 1" > /dev/null
done
compare mpi_iwin_1/diag.txt mpi_iwin_$NP/diag.txt " (iRIP: moving window, laser, open x)"
# Plasma that reaches a transverse PML, with the default grid.absorber_particles = conserve (decision 115): the Gauss law stays at
# round-off (the 1e-12 limit also checks that the default is conserve). remove (decision 111) takes particles out once they are in
# a layer and leaves their charge at the layer face: a Gauss residual ~1e-3 there, the same on 1 and NP ranks; its rank invariance
# is checked separately below (mpi_winpmlr) with a loose Gauss limit.
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $L numerics.solver=irip grid.nx=64 grid.ny=32 grid.nz=32 grid.dy=0.04um grid.dz=0.04um \
        grid.t_end=80fs laser.waist=0.3um laser.waist_z=0.3um laser.duration=2.5fs window.start=10fs output.every=20 \
        output.slice_every=0 "species electrons.xmin=4um" "species protons.xmin=4um" "species electrons.uth_y=0.05" \
        "species electrons.uth_z=0.05" grid.boundary_y=pml grid.boundary_z=pml pml.cells=6 \
        output.dir=mpi_iwinpml_$np > /dev/null
done
compare mpi_iwinpml_1/diag.txt mpi_iwinpml_$NP/diag.txt " (iRIP: moving window, laser, pml)"
WP="grid.boundary_y=pml grid.boundary_z=pml pml.cells=4"
WD="grid.nx=64 grid.ny=16 grid.nz=16 grid.t_end=80fs laser.waist=1um laser.waist_z=1um laser.duration=2.5fs window.start=10fs
    output.every=20 output.slice_every=0"
CK=""                                         # checkpoints only in a build with HDF5
$MPIRUN -np 1 $EXE $L --check checkpoint.every=200 > /dev/null 2>&1 && CK="checkpoint.every=200"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $L $WD "species electrons.xmin=4um" "species protons.xmin=4um" "species electrons.uth_y=0.05" \
        "species electrons.uth_z=0.05" $WP $CK output.dir=mpi_winpml_$np > /dev/null
done
compare mpi_winpml_1/diag.txt mpi_winpml_$NP/diag.txt " (moving window and pml)"
# absorber_particles = remove (decision 111) on the same case: its Gauss residual at the layer faces (~1e-3) is
# physics of the rule, not of MPI, so the Gauss limit is loose (1e-2); instead the residual itself must be rank invariant:
# column max|dGauss| of 1 and NP ranks equal to round-off (relative 1e-8 where it is above 1e-12).
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $L $WD "species electrons.xmin=4um" "species protons.xmin=4um" "species electrons.uth_y=0.05" \
        "species electrons.uth_z=0.05" grid.boundary_y=pml grid.boundary_z=pml pml.cells=4 grid.absorber_particles=remove \
        output.dir=mpi_winpmlr_$np > /dev/null
done
compare mpi_winpmlr_1/diag.txt mpi_winpmlr_$NP/diag.txt " (moving window and pml, absorber_particles = remove)" 1e-2
python3 - mpi_winpmlr_1/diag.txt mpi_winpmlr_$NP/diag.txt <<'PY'
import os, sys
a=[l.split() for l in open(sys.argv[1]) if not l.startswith('#')]
b=[l.split() for l in open(sys.argv[2]) if not l.startswith('#')]
worst=max(abs(float(x[5])-float(y[5]))/abs(float(x[5])) for x,y in zip(a,b) if abs(float(x[5]))>1e-12)
print("mpi (remove: Gauss residual 1 vs %s ranks): max relative difference %.1e, residual at the end %s" % (os.environ["NP"], worst, b[-1][5]))
sys.exit(0 if worst<1e-8 else 1)
PY
if [ -n "$CK" ]; then
    for np in $NP 1 3; do
        rm -rf mpi_restart_$np; mkdir -p mpi_restart_$np/checkpoints
        cp mpi_winpml_$NP/checkpoints/checkpoint_000200.h5 mpi_restart_$np/checkpoints/
        cp mpi_winpml_$NP/diag.txt mpi_restart_$np/
        $MPIRUN -np $np $EXE $L $WD "species electrons.xmin=4um" "species protons.xmin=4um" "species electrons.uth_y=0.05" \
            "species electrons.uth_z=0.05" $WP output.dir=mpi_restart_$np --restart latest > /dev/null
    done
    same_run mpi_winpml_$NP/diag.txt mpi_restart_$NP/diag.txt " (restart from step 200)"
    compare mpi_winpml_1/diag.txt mpi_restart_1/diag.txt " (restart of $NP ranks on 1)"
    compare mpi_winpml_1/diag.txt mpi_restart_3/diag.txt " (restart of $NP ranks on 3)"
else
    echo "mpi (restart): skipped (built without HDF5)"
fi
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/beam_vacuum.deck grid.t_end=5fs output.every=10 output.dir=mpi_beam_$np \
        "diagnostic all.species=beam" "diagnostic all.every=10" "diagnostic all.e_max_mev=400" > /dev/null
done
compare mpi_beam_1/diag.txt mpi_beam_$NP/diag.txt " (beam self-field)"
python3 - mpi_beam_1/beam_all.txt mpi_beam_$NP/beam_all.txt <<'PY'
import os, sys
a=[[float(v) for v in l.split()] for l in open(sys.argv[1]) if not l.startswith('#')]
b=[[float(v) for v in l.split()] for l in open(sys.argv[2]) if not l.startswith('#')]
assert len(a)==len(b) and len(a)>0, "different number of lines"
worst=max(abs(x-y)/max(abs(x),1e-300) for u,v in zip(a,b) for x,y in zip(u,v))
print("mpi (beam diagnostics): 1 vs %s ranks, max relative difference %.1e (Q %.6g pC, E %.6g MeV)" % (os.environ["NP"], worst, a[-1][3], a[-1][4]))
sys.exit(0 if worst<1e-8 else 1)
PY
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_plasma.deck numerics.solver=rip_cc grid.dz=0.02um grid.t_end=20fs output.every=20 \
        output.dir=mpi_rz_$np > /dev/null
done
compare mpi_rz_1/diag.txt mpi_rz_$NP/diag.txt " (RZ, rip_cc, $NP ranks along z)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_plasma.deck numerics.solver=rip_cc numerics.shape=2 grid.dz=0.02um grid.t_end=20fs \
        output.every=20 output.dir=mpi_rz2_$np > /dev/null
done
compare mpi_rz2_1/diag.txt mpi_rz2_$NP/diag.txt " (RZ, shape 2, rip_cc, $NP ranks along z)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_plasma.deck numerics.solver=irip grid.dz=0.02um grid.t_end=20fs output.every=20 \
        output.dir=mpi_rzi_$np > /dev/null
done
compare mpi_rzi_1/diag.txt mpi_rzi_$NP/diag.txt " (RZ, irip, periodic z, $NP ranks along z: partitioned cyclic solve)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_plasma.deck numerics.solver=irip numerics.shape=2 grid.nr=24 grid.dr=0.05um \
        grid.r_uniform=0.5um grid.stretch=1.08 grid.dr_max=0.3um grid.t_end=40fs output.every=20 output.dir=mpi_rzs_$np > /dev/null
done
compare mpi_rzs_1/diag.txt mpi_rzs_$NP/diag.txt " (RZ, irip, shape 2, stretched radial grid, $NP ranks along z)"
# O6 implicit mover in RZ (decision 82): m = 0, the C1 radial map, omega_p dt = 2.5, dz = 2 dr, a stretched radial grid, ranks along
# z (at dz = 4 dr rare steps stall, docs/specs/implicit_mover.md "Open"); the plasma kept inside the wall (whether a particle at the
# wall is taken out depends on the solve tolerance)
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_plasma.deck numerics.solver=irip numerics.shape=2 numerics.mover=implicit numerics.mover_tol=1e-11 \
        grid.modes=1 grid.nr=16 grid.dr=0.1um grid.r_uniform=0.6um grid.stretch=1.1 grid.dr_max=0.3um grid.t_end=10fs output.every=5 \
        "species electrons.density=4" "species ions.density=4" "species electrons.uth_x=0.1" "species electrons.uth_y=0.1" \
        "species electrons.rmax=1um" "species ions.rmax=1um" \
        output.dir=mpi_rzm_$np > /dev/null
done
compare mpi_rzm_1/diag.txt mpi_rzm_$NP/diag.txt " (RZ, implicit mover, m = 0, stretched radial grid, $NP ranks along z)"
# O5 in RZ (decision 74): the closure per radial eigenmode, two line solves per mode (partitioned, cyclic)
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_plasma.deck numerics.solver=irip numerics.closure=o5 numerics.shape=2 grid.nr=24 grid.dr=0.05um \
        grid.r_uniform=0.5um grid.stretch=1.08 grid.dr_max=0.3um grid.t_end=40fs output.every=20 output.dir=mpi_rzo5_$np > /dev/null
done
compare mpi_rzo5_1/diag.txt mpi_rzo5_$NP/diag.txt " (RZ, irip, closure o5, stretched radial grid, $NP ranks along z)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_plasma.deck numerics.solver=irip numerics.filter_passes=2 numerics.filter=projection grid.dz=0.02um \
        grid.t_end=20fs output.every=20 output.dir=mpi_rzf_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_plasma.deck numerics.solver=irip numerics.filter_passes=2 grid.dz=0.02um grid.t_end=20fs \
        output.every=20 output.dir=mpi_rzfb_$np > /dev/null
done
compare mpi_rzf_1/diag.txt mpi_rzf_$NP/diag.txt " (RZ, irip, projection filter 2 passes, $NP ranks along z)"
# the maximally flat filter along z (decision 109), RIP-CC, ranks along z
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_plasma.deck numerics.solver=rip_cc numerics.filter_passes=6 numerics.filter_flat=4 numerics.filter_axes=x         grid.dz=0.02um grid.t_end=20fs output.every=20 output.dir=mpi_rzff_$np > /dev/null
done
compare mpi_rzff_1/diag.txt mpi_rzff_$NP/diag.txt " (RZ, rip_cc, flat filter N = 6, L = 4 along z, $NP ranks along z)"
compare mpi_rzfb_1/diag.txt mpi_rzfb_$NP/diag.txt " (RZ, irip, binomial filter 2 passes, $NP ranks along z)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_plasma.deck numerics.solver=rip_cc numerics.filter_passes=2 numerics.filter=projection grid.dz=0.02um \
        grid.t_end=20fs output.every=20 output.dir=mpi_rzfc_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_plasma.deck numerics.solver=irip numerics.filter_passes=2 numerics.filter=local grid.dz=0.02um \
        grid.t_end=20fs output.every=20 output.dir=mpi_rzfl_$np > /dev/null
done
compare mpi_rzfc_1/diag.txt mpi_rzfc_$NP/diag.txt " (RZ, rip_cc, projection filter 2 passes, $NP ranks along z)"
compare mpi_rzfl_1/diag.txt mpi_rzfl_$NP/diag.txt " (RZ, irip, local filter 2 passes, $NP ranks along z)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_laser.deck grid.boundary_r=absorbing grid.r_absorber_cells=16 output.every=40 \
        output.dir=mpi_rzl_$np > /dev/null
done
compare mpi_rzl_1/diag.txt mpi_rzl_$NP/diag.txt " (RZ laser, open z, radial absorber, rip_cc, $NP ranks along z)"
# O2: the energy leaving through the open x edges (output.xflux, docs/specs/xflux.md): 1 vs NP ranks, the edge
# planes split over 4 ranks (Cart3D) and the edges on the first and last rank (RZ)
xcompare() {
python3 - "$1" "$2" "$3" <<'PY'
import os, sys
a = [[float(x) for x in l.split()] for l in open(sys.argv[1]) if not l.startswith("#")]
b = [[float(x) for x in l.split()] for l in open(sys.argv[2]) if not l.startswith("#")]
assert len(a) == len(b) and len(a) > 0, "different lengths"
m = 0.0
for c in (4, 5):
    s = max(max(abs(r[c]) for r in a), 1e-300)
    m = max(m, max(abs(x[c] - y[c]) for x, y in zip(a, b)) / s)
print("mpi%s: 1 vs %s ranks, max relative difference of E_left, E_right %.1e (E_left %.4e, E_right %.4e)" % (sys.argv[3], os.environ["NP"], m, a[-1][4], a[-1][5]))
sys.exit(0 if m < 1e-9 else 1)
PY
}
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $D grid.boundary_x=open grid.t_end=200fs output.every=20 "species electrons.uth_y=0.3" \
        "species electrons.uth_z=0.3" output.xflux=true output.dir=mpi_xf_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_laser.deck output.every=40 output.xflux=true output.dir=mpi_rzxf_$np > /dev/null
done
xcompare mpi_xf_1/xflux.txt mpi_xf_$NP/xflux.txt " (xflux: open x, plasma leaving)"
xcompare mpi_rzxf_1/xflux.txt mpi_rzxf_$NP/xflux.txt " (xflux: RZ laser through open z)"
# O1: the TFSF source (docs/specs/tfsf.md), a focused beam at 30 deg onto a thin slab, the plane inside one rank block
# (0.5 n_c; output every 100 steps: in the first steps of the plasma response W_kin ~ 1e-17 differs at
# round-off by 1e-7 relative)
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/tfsf_oblique.deck grid.t_end=50fs output.every=100 "species electrons.density=0.5" \
        "species protons.density=0.5" output.dir=mpi_tfsf_$np > /dev/null
done
compare mpi_tfsf_1/diag.txt mpi_tfsf_$NP/diag.txt " (TFSF laser at 30 deg onto a slab)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/tfsf_oblique.deck grid.t_end=50fs output.every=100 "species electrons.density=0.5" \
        "species protons.density=0.5" grid.boundary_y=pml pml.cells=8 output.dir=mpi_tfsfpml_$np > /dev/null
done
compare mpi_tfsfpml_1/diag.txt mpi_tfsfpml_$NP/diag.txt " (TFSF laser at 30 deg onto a slab, CPML in y)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/tfsf_oblique.deck grid.t_end=50fs output.every=100 "species electrons.density=0.5" \
        "species protons.density=0.5" grid.boundary_y=pml pml.cells=8 numerics.solver=irip numerics.closure=o5 \
        output.dir=mpi_tfsfo5_$np > /dev/null
done
compare mpi_tfsfo5_1/diag.txt mpi_tfsfo5_$NP/diag.txt " (TFSF laser at 30 deg onto a slab, CPML in y, irip, closure o5)"
# O3: the same with absorbing x boundaries (docs/specs/x_absorber.md; layers of 32 cells, the left one inside rank 0)
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/tfsf_oblique.deck grid.t_end=50fs output.every=100 "species electrons.density=0.5" \
        "species protons.density=0.5" grid.boundary_x=absorbing grid.x_absorber_cells=32 output.dir=mpi_tfsfxa_$np > /dev/null
done
compare mpi_tfsfxa_1/diag.txt mpi_tfsfxa_$NP/diag.txt " (TFSF laser at 30 deg onto a slab, absorbing x)"
xcompare mpi_tfsfxa_1/xflux.txt mpi_tfsfxa_$NP/xflux.txt " (xflux at the inner faces of the x absorbers)"
# the plane wave of one k_y (laser.waist = 0, sin 30 deg * 9.6 um = 6 lambda0) onto the slab with absorbing x, linear
# (a0 = 0.01; at a0 = 0.5 the plane wave drives the whole 4 n_c slab and round-off grows to 1e-3 by 50 fs)
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/tfsf_oblique.deck output.every=100 laser.waist=0um laser.delay=12fs laser.a0=0.01 grid.t_end=40fs grid.boundary_x=absorbing grid.x_absorber_cells=32 output.dir=mpi_tfsfpw_$np > /dev/null
done
compare mpi_tfsfpw_1/diag.txt mpi_tfsfpw_$NP/diag.txt " (TFSF plane wave at 30 deg onto a slab, absorbing x)"
xcompare mpi_tfsfpw_1/xflux.txt mpi_tfsfpw_$NP/xflux.txt " (xflux of the reflected plane wave)"
# the same focused beam at 30 deg with the exact plane waves at the TFSF plane (decision 65), absorbing x
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/tfsf_oblique.deck grid.t_end=50fs output.every=100 "species electrons.density=0.5" \
        "species protons.density=0.5" grid.boundary_x=absorbing grid.x_absorber_cells=32 laser.tfsf_incident=modes \
        output.dir=mpi_tfsfm_$np > /dev/null
done
compare mpi_tfsfm_1/diag.txt mpi_tfsfm_$NP/diag.txt " (TFSF plane with exact plane waves at 30 deg onto a slab, absorbing x)"
xcompare mpi_tfsfm_1/xflux.txt mpi_tfsfm_$NP/xflux.txt " (xflux, plane-wave TFSF source)"
# side injection (docs/specs/side_injection.md): a plane-wave pulse at 75.5 deg through the faces of the TFSF tube onto
# an underdense layer, CPML in y and z, faces and the tube corners on several ranks; RIP-CC and iRIP (ranks along x); with
# HDF5 the restart of NP ranks on NP (exact) and on 3 ranks (the incident field is a closed form in t: nothing to store)
SW="output.every=20"                            # 240 steps (40 fs), checkpoint at step 200
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/side_wave.deck $SW $CK output.dir=mpi_side_$np > /dev/null
    $MPIRUN -np $np $EXE examples/side_wave.deck $SW numerics.solver=irip output.dir=mpi_iside_$np > /dev/null
done
compare mpi_side_1/diag.txt mpi_side_$NP/diag.txt " (side tube: plane-wave pulse at 75.5 deg, rip_cc)"
compare mpi_iside_1/diag.txt mpi_iside_$NP/diag.txt " (side tube: plane-wave pulse at 75.5 deg, irip, $NP ranks along x)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/side_wave.deck $SW numerics.solver=irip numerics.closure=o5 output.dir=mpi_o5side_$np > /dev/null
done
compare mpi_o5side_1/diag.txt mpi_o5side_$NP/diag.txt " (side tube: plane-wave pulse at 75.5 deg, irip, closure o5, $NP ranks along x)"
if [ -n "$CK" ]; then
    for np in $NP 3; do
        rm -rf mpi_siderestart_$np; mkdir -p mpi_siderestart_$np/checkpoints
        cp mpi_side_$NP/checkpoints/checkpoint_000200.h5 mpi_siderestart_$np/checkpoints/
        cp mpi_side_$NP/diag.txt mpi_siderestart_$np/
        $MPIRUN -np $np $EXE examples/side_wave.deck $SW output.dir=mpi_siderestart_$np --restart latest > /dev/null
    done
    same_run mpi_side_$NP/diag.txt mpi_siderestart_$NP/diag.txt " (side tube, restart from step 200)"
    compare mpi_side_1/diag.txt mpi_siderestart_3/diag.txt " (side tube, restart of $NP ranks on 3)"
fi
# the flying focus through the side tube (decision 62): the tables of each rank in the window moving at c
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/side_flying_focus.deck grid.t_end=10fs output.dir=mpi_sideff_$np > /dev/null
    $MPIRUN -np $np $EXE examples/side_flying_focus.deck grid.t_end=10fs numerics.solver=irip output.dir=mpi_isideff_$np > /dev/null
done
compare mpi_sideff_1/diag.txt mpi_sideff_$NP/diag.txt " (side tube: flying focus in the moving window, rip_cc)"
compare mpi_isideff_1/diag.txt mpi_isideff_$NP/diag.txt " (side tube: flying focus in the moving window, irip, $NP ranks along x)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/side_flying_focus.deck grid.t_end=10fs numerics.solver=irip numerics.closure=o5 \
        output.dir=mpi_o5sideff_$np > /dev/null
done
compare mpi_o5sideff_1/diag.txt mpi_o5sideff_$NP/diag.txt " (side tube: flying focus in the moving window, irip, closure o5, $NP ranks along x)"
# ... in a background plasma (decision 63): kinetic electrons in the tube, the cold fluid outside; with HDF5 the restart
# (the fluid current is part of the checkpoint and moves with the window)
SP="grid.t_end=10fs output.every=10"
CKP=""
[ -n "$CK" ] && CKP="checkpoint.every=30"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/side_plasma.deck $SP $CKP output.dir=mpi_sidepl_$np > /dev/null
    $MPIRUN -np $np $EXE examples/side_plasma.deck $SP numerics.solver=irip output.dir=mpi_isidepl_$np > /dev/null
done
compare mpi_sidepl_1/diag.txt mpi_sidepl_$NP/diag.txt " (side tube: flying focus in a plasma, fluid shell, rip_cc)"
compare mpi_isidepl_1/diag.txt mpi_isidepl_$NP/diag.txt " (side tube: flying focus in a plasma, fluid shell, irip, $NP ranks along x)"
if [ -n "$CK" ]; then
    rm -rf mpi_sideplrs_$NP; mkdir -p mpi_sideplrs_$NP/checkpoints
    cp mpi_sidepl_$NP/checkpoints/checkpoint_000030.h5 mpi_sideplrs_$NP/checkpoints/
    cp mpi_sidepl_$NP/diag.txt mpi_sideplrs_$NP/
    $MPIRUN -np $NP $EXE examples/side_plasma.deck $SP output.dir=mpi_sideplrs_$NP --restart latest > /dev/null
    same_run mpi_sidepl_$NP/diag.txt mpi_sideplrs_$NP/diag.txt " (side tube in a plasma, restart from step 30)"
fi
RZW="grid.nz=128 grid.t_end=80fs laser.duration=5fs window.start=20fs output.every=20"
CKZ=""
[ -n "$CK" ] && CKZ="checkpoint.every=300"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_lwfa.deck $RZW $CKZ output.dir=mpi_rzw_$np > /dev/null
done
# half the box of the example: a shorter pulse and an earlier window, so that the pulse has not reached
# the right edge before the window moves (else its longitudinal field stays in the last cell)
compare mpi_rzw_1/diag.txt mpi_rzw_$NP/diag.txt " (RZ moving window and laser, rip_cc, $NP ranks along z)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_lwfa.deck numerics.solver=irip $RZW output.dir=mpi_rziw_$np > /dev/null
done
compare mpi_rziw_1/diag.txt mpi_rziw_$NP/diag.txt " (RZ moving window and laser, irip, $NP ranks along z)"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_lwfa.deck numerics.solver=irip numerics.closure=o5 $RZW output.dir=mpi_rzo5w_$np > /dev/null
done
compare mpi_rzo5w_1/diag.txt mpi_rzo5w_$NP/diag.txt " (RZ moving window and laser, irip, closure o5, $NP ranks along z)"
# RZ embedded metal rings, the beam self-field (global solve) and the plane probe (decision 83): the disks of
# examples/rz_disk_loaded.deck without the jacket on a coarse grid, the bunch already in the window and the rings entering it
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_disk_loaded.deck grid.nz=320 grid.nr=130 grid.dr=10um grid.dz=7.6666667um grid.r_uniform=1.12mm \
        grid.dr_max=25um grid.t_end=20ps "species bunch.density_function=gauss(z - 1.9 mm, 50 um) * gauss(r, 50 um)" \
        "species bunch.zmin=1.6mm" "species bunch.zmax=2.2mm" rings.z_start=2.2mm rings.jacket=0um output.plane_z=4mm output.axis_every=100 \
        output.every=50 output.dir=mpi_rings_$np > /dev/null
done
# the Gauss limit 1e-8: the self-field ahead of the bunch reaches the front edge of the window and leaves a frozen residual
# there (decision 83; 3.8e-9 = 1.3e-7 of max rho here, the same on 1 rank), which leaves with the window
compare mpi_rings_1/diag.txt mpi_rings_$NP/diag.txt " (RZ rings, self-field, moving window, plane probe, $NP ranks along z)" 1e-8
# the same with the jacket (the closed disk-loaded guide, rings.jacket) whose open end at z_end = 4 mm passes through the window
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_disk_loaded.deck grid.nz=320 grid.nr=130 grid.dr=10um grid.dz=7.6666667um grid.r_uniform=1.12mm \
        grid.dr_max=25um grid.t_end=20ps "species bunch.density_function=gauss(z - 1.9 mm, 50 um) * gauss(r, 50 um)" \
        "species bunch.zmin=1.6mm" "species bunch.zmax=2.2mm" rings.z_start=2.2mm rings.z_end=4mm \
        output.plane_z=4.5mm output.axis_every=100 output.every=50 output.dir=mpi_jacket_$np > /dev/null
done
compare mpi_jacket_1/diag.txt mpi_jacket_$NP/diag.txt " (RZ rings with a jacket, open end, $NP ranks along z)" 1e-8
# RZ dielectric blocks (decision 100): the disk-loaded guide with the cells filled and a liner
# inside the aperture on the coarse grid of the rings case
# (open z, moving window, the dielectric entering it: partitioned line solves along z), and the periodic plasma column
# with two blocks, one across the periodic seam (the cyclic reduced system across the ranks)
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_disk_loaded.deck grid.nz=320 grid.nr=130 grid.dr=10um grid.dz=7.6666667um \
        grid.r_uniform=1.2mm grid.dr_max=25um grid.t_end=20ps "species bunch.density_function=gauss(z - 1.9 mm, 50 um) * gauss(r, 50 um)" \
        "species bunch.zmin=1.6mm" "species bunch.zmax=2.2mm" rings.z_start=2.2mm \
        "dielectric fill.r_in=0.8mm" "dielectric fill.r_out=1.0mm" "dielectric fill.z_start=2.2mm" "dielectric fill.z_end=100mm" "dielectric fill.eps=1.5" \
        "dielectric liner.r_in=0.7mm" "dielectric liner.r_out=0.8mm" "dielectric liner.z_start=2.0mm" "dielectric liner.z_end=100mm" "dielectric liner.eps=2" output.plane_z=4mm \
        output.axis_every=100 output.every=50 output.dir=mpi_diel_$np > /dev/null
done
compare mpi_diel_1/diag.txt mpi_diel_$NP/diag.txt " (RZ rings + dielectric fill and liner, moving window, $NP ranks along z)" 1e-8
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_plasma.deck numerics.solver=rip_cc grid.dz=0.02um grid.t_end=20fs output.every=20 \
        "dielectric a.r_in=0.4um" "dielectric a.r_out=1.7um" "dielectric a.z_start=0.33um" "dielectric a.z_end=1.07um" \
        "dielectric a.eps=1.4" "dielectric b.r_in=2.05um" "dielectric b.r_out=3.5um" "dielectric b.z_start=1.01um" \
        "dielectric b.z_end=1.53um" "dielectric b.eps=2" output.dir=mpi_dielp_$np > /dev/null
done
compare mpi_dielp_1/diag.txt mpi_dielp_$NP/diag.txt " (RZ dielectric blocks, periodic z, $NP ranks along z: cyclic line solves)"
# RZ metal depending on phi (block metal, decision 103): the square corrugated pipe of examples/rz_square_pipe.deck
# on a coarse grid (5 modes, the axis cut), an offset bunch with its self-field, the corrugated walls entering the moving window
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_square_pipe.deck grid.modes=5 grid.nz=160 grid.nr=60 grid.dr=50um grid.dz=35um \
        grid.r_absorber_cells=8 grid.t_end=20ps "species bunch.density_function=gauss(z - 4.5 mm, 100 um) * gauss(sqrt((x - 100 um)^2 + y^2), 50 um)" \
        "species bunch.rmax=0.4mm" "species bunch.ppc_theta=18" output.openpmd_every=0 output.axis_every=0 \
        output.every=50 output.dir=mpi_sqpipe_$np > /dev/null
done
compare mpi_sqpipe_1/diag.txt mpi_sqpipe_$NP/diag.txt " (RZ metal depending on phi, square pipe, self-field, window, $NP ranks along z)" 1e-8
# the same with a dielectric lining across the corrugated square walls (decision 100 with the block metal: the lines on the
# angles, partitioned along z; the entering columns with the metal of each angle), the bunch behind the 32-cell front layer
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_square_pipe.deck grid.modes=5 grid.nz=160 grid.nr=60 grid.dr=50um grid.dz=35um \
        grid.r_absorber_cells=8 grid.t_end=20ps "species bunch.density_function=gauss(z - 3 mm, 100 um) * gauss(sqrt((x - 100 um)^2 + y^2), 50 um)" \
        "species bunch.zmin=2.5mm" "species bunch.zmax=3.5mm" "species bunch.rmax=0.4mm" "species bunch.ppc_theta=18" \
        "dielectric lining.r_in=0.7mm" "dielectric lining.r_out=1.6mm" "dielectric lining.z_start=5.8mm" "dielectric lining.z_end=100mm" \
        "dielectric lining.eps=2.1" output.openpmd_every=0 output.axis_every=0 output.every=50 output.dir=mpi_sqdiel_$np > /dev/null
done
compare mpi_sqdiel_1/diag.txt mpi_sqdiel_$NP/diag.txt " (RZ metal depending on phi + dielectric lining, window, $NP ranks along z: lines on the angles)" 1e-8
if [ -n "$CK" ]; then
    for np in $NP 3; do
        rm -rf mpi_rzrestart_$np; mkdir -p mpi_rzrestart_$np/checkpoints
        cp mpi_rzw_$NP/checkpoints/checkpoint_000300.h5 mpi_rzrestart_$np/checkpoints/
        cp mpi_rzw_$NP/diag.txt mpi_rzrestart_$np/
        $MPIRUN -np $np $EXE examples/rz_lwfa.deck $RZW output.dir=mpi_rzrestart_$np --restart latest > /dev/null
    done
    same_run mpi_rzw_$NP/diag.txt mpi_rzrestart_$NP/diag.txt " (RZ restart from step 300)"
    compare mpi_rzw_1/diag.txt mpi_rzrestart_3/diag.txt " (RZ restart of $NP ranks on 3)"
else
    echo "mpi (RZ restart): skipped (built without HDF5)"
fi
# warm non-neutral plasma entering with the filter (2 passes; decision 91): the entry columns frozen for 3 + 2 cells, the entered
# charge a background in gauss0 (checkpointed): binomial and projection, irip; the restart of NP ranks on NP and 3 ranks
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_lwfa.deck $RZW $CKZ numerics.solver=irip numerics.filter_passes=2 "species electrons.ppc_z=2" \
        "species protons.ppc_z=1" "species electrons.uth_z=0.05" "species electrons.uth_x=0.05" "species electrons.uth_y=0.05" \
        output.dir=mpi_rzwf_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_lwfa.deck $RZW numerics.solver=irip numerics.filter_passes=2 numerics.filter=projection \
        "species electrons.ppc_z=2" "species protons.ppc_z=1" "species electrons.uth_z=0.05" "species electrons.uth_x=0.05" \
        "species electrons.uth_y=0.05" output.dir=mpi_rzwfp_$np > /dev/null
done
compare mpi_rzwf_1/diag.txt mpi_rzwf_$NP/diag.txt " (RZ window, warm non-neutral entry, binomial filter, irip, $NP ranks along z)"
compare mpi_rzwfp_1/diag.txt mpi_rzwfp_$NP/diag.txt " (RZ window, warm non-neutral entry, projection filter, irip, $NP ranks along z)"
if [ -n "$CK" ]; then
    for np in $NP 3; do
        rm -rf mpi_rzwfrs_$np; mkdir -p mpi_rzwfrs_$np/checkpoints
        cp mpi_rzwf_$NP/checkpoints/checkpoint_000300.h5 mpi_rzwfrs_$np/checkpoints/
        cp mpi_rzwf_$NP/diag.txt mpi_rzwfrs_$np/
        $MPIRUN -np $np $EXE examples/rz_lwfa.deck $RZW numerics.solver=irip numerics.filter_passes=2 "species electrons.ppc_z=2" \
            "species protons.ppc_z=1" "species electrons.uth_z=0.05" "species electrons.uth_x=0.05" "species electrons.uth_y=0.05" \
            output.dir=mpi_rzwfrs_$np --restart latest > /dev/null
    done
    same_run mpi_rzwf_$NP/diag.txt mpi_rzwfrs_$NP/diag.txt " (RZ window with filter, restart from step 300)"
    compare mpi_rzwf_1/diag.txt mpi_rzwfrs_3/diag.txt " (RZ window with filter, restart of $NP ranks on 3)"
fi
# 2D Cartesian geometry (docs/specs/cart2d.md): the foil with a TFSF laser at 30 deg (exact plane waves), absorbing x; RIP-CC on
# NP ranks in x and y, iRIP on NP ranks along x, RIP-CC with CPML in y (incident grid); with HDF5 the restart of NP ranks on NP and 3.
# The deck's filter pass: the local filter where ranks split y or y has CPML (the projection needs y periodic and ranks along x
# only, decision 69), the projection with iRIP along x
# The CPML case runs with the default absorber_particles = conserve (decision 115); with remove its Gauss residual at the layer
# face is 6.6e-3 and more (the same on 1 and NP ranks), above the 1e-12 limit; the rank invariance of remove is checked above
F2="grid.t_end=30fs output.every=20"
CK2=""
[ -n "$CK" ] && CK2="checkpoint.every=200"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/foil_2d.deck $F2 $CK2 numerics.filter=local output.dir=mpi_2d_$np > /dev/null
    $MPIRUN -np $np $EXE examples/foil_2d.deck $F2 numerics.solver=irip grid.mpi_y=1 output.dir=mpi_i2d_$np > /dev/null
    $MPIRUN -np $np $EXE examples/foil_2d.deck $F2 numerics.filter=local grid.boundary_y=pml pml.cells=12 laser.tfsf_incident=grid laser.angle=0deg \
        output.dir=mpi_p2d_$np > /dev/null 2>&1
done
compare mpi_2d_1/diag.txt mpi_2d_$NP/diag.txt " (2D foil, rip_cc, TFSF modes 30 deg, $NP ranks in x, y)"
compare mpi_i2d_1/diag.txt mpi_i2d_$NP/diag.txt " (2D foil, irip, projection filter, $NP ranks along x)"
compare mpi_p2d_1/diag.txt mpi_p2d_$NP/diag.txt " (2D foil, rip_cc, CPML in y, incident grid)"
# O5 closure of iRIP (decision 72): the foil with open x and the laser through the left edge (O5 has no TFSF plane yet)
sed -e "/x_absorber_width/d" -e "s/boundary_x = absorbing/boundary_x = open/" -e "/tfsf_x/d" -e "/tfsf_incident/d" \
    examples/foil_2d.deck > mpi_o5.deck
for np in 1 $NP; do
    $MPIRUN -np $np $EXE mpi_o5.deck numerics.solver=irip numerics.closure=o5 grid.mpi_y=1 grid.t_end=30fs output.every=20 \
        output.dir=mpi_o5_$np > /dev/null
done
compare mpi_o5_1/diag.txt mpi_o5_$NP/diag.txt " (2D foil, irip, closure o5, open x, laser at 30 deg, $NP ranks along x)"
if [ -n "$CK" ]; then
    for np in $NP 3; do
        rm -rf mpi_2drs_$np; mkdir -p mpi_2drs_$np/checkpoints
        cp mpi_2d_$NP/checkpoints/checkpoint_000200.h5 mpi_2drs_$np/checkpoints/
        cp mpi_2d_$NP/diag.txt mpi_2drs_$np/
        $MPIRUN -np $np $EXE examples/foil_2d.deck $F2 numerics.filter=local output.dir=mpi_2drs_$np --restart latest > /dev/null
    done
    same_run mpi_2d_$NP/diag.txt mpi_2drs_$NP/diag.txt " (2D restart from step 200)"
    compare mpi_2d_1/diag.txt mpi_2drs_3/diag.txt " (2D restart of $NP ranks on 3)"
else
    echo "mpi (2D restart): skipped (built without HDF5)"
fi
# CHF diagnostic (harmonic_focus, docs/specs/harmonic_focus.md): the peak and fluence maps of 1 and NP ranks, assembled from
# the parts of the ranks; RZ laser through its focus (ranks along z), the 2D foil (ranks in x and y), band 0.5 .. 1.5
hcompare() {
python3 - "$1" "$2" "$3" <<'PY'
import array, glob, os, sys
def maps(d):
    v = {}
    for mp in glob.glob(os.path.join(d, "meta_r*.txt")):
        m = {l.split()[0]: l.split()[1:] for l in open(mp) if l.strip() and not l.startswith("#")}
        s = [int(x) for x in m["start"]]; n = [int(x) for x in m["size"]]; q = m["quantities"]
        a = array.array("d"); a.frombytes(open(mp.replace("meta_r", "r").replace(".txt", ".bin"), "rb").read())
        npt = n[0] * n[1] * n[2]
        for c in range(len(q)):
            if q[c].startswith("tmax"):
                continue                   # the time of a peak may jump between two equal maxima at round-off
            for p in range(npt):
                i, j, k = p % n[0], (p // n[0]) % n[1], p // (n[0] * n[1])
                v[(c, s[0] + i, s[1] + j, s[2] + k)] = a[c * npt + p]
    return v
a, b = maps(sys.argv[1]), maps(sys.argv[2])
assert set(a) == set(b) and a, "different nodes"
s = max(abs(x) for x in a.values())
m = max(abs(a[k] - b[k]) for k in a) / s
print("mpi%s: 1 vs %s ranks, max difference of the peak and fluence maps %.1e of the largest (%d values)" % (sys.argv[3], os.environ["NP"], m, len(a)))
sys.exit(0 if m < 1e-9 else 1)
PY
}
HF="harmonic_focus.n_min=0.5 harmonic_focus.n_max=1.5"
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_laser.deck output.every=40 $HF output.dir=mpi_rzhf_$np > /dev/null
    $MPIRUN -np $np $EXE examples/foil_2d.deck grid.t_end=30fs output.every=20 numerics.filter=local $HF harmonic_focus.xmin=2um harmonic_focus.xmax=6um \
        output.dir=mpi_2dhf_$np > /dev/null
done
hcompare mpi_rzhf_1/harmonic_focus mpi_rzhf_$NP/harmonic_focus " (harmonic_focus: RZ laser, $NP ranks along z)"
hcompare mpi_2dhf_1/harmonic_focus mpi_2dhf_$NP/harmonic_focus " (harmonic_focus: 2D foil, $NP ranks in x, y)"
# RZ TFSF plane with absorbing z layers (docs/specs/tfsf.md "RZ"): examples/rz_tfsf.deck on 1 and NP ranks along z, RIP-CC and
# iRIP, iRIP + o5 (decision 78) (4 and 8 ranks: the plane at cell 60 on a rank boundary); diag.txt and the reflected / transmitted energy of xflux.txt; 33 fs:
# later the hot electrons of the foil amplify round-off (1 rank with 1 or 4 threads differ by 1 % in W at 50 fs)
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/rz_tfsf.deck grid.t_end=33fs output.dir=mpi_rzts_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_tfsf.deck grid.t_end=33fs numerics.solver=irip output.dir=mpi_rzitf_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_tfsf.deck grid.t_end=33fs numerics.solver=irip numerics.closure=o5 output.dir=mpi_rzotf_$np > /dev/null
done
compare mpi_rzts_1/diag.txt mpi_rzts_$NP/diag.txt " (RZ TFSF plane, z layers, rip_cc, $NP ranks along z)"
compare mpi_rzitf_1/diag.txt mpi_rzitf_$NP/diag.txt " (RZ TFSF plane, z layers, irip, $NP ranks along z)"
compare mpi_rzotf_1/diag.txt mpi_rzotf_$NP/diag.txt " (RZ TFSF plane, z layers, irip, closure o5, $NP ranks along z)"
xcompare mpi_rzts_1/xflux.txt mpi_rzts_$NP/xflux.txt " (xflux: RZ TFSF plane, z layers)"
# O7 linear media (docs/specs/linear_medium.md, decision 84): the parabolic plasma mirror (2D, open x, CPML in y, NP ranks along x,
# the mirror crossing the rank boundaries), RZ (Lorentz glass and a collisional plasma, ranks along z) and the implicit mover with a
# medium (the streaming plasma through a Drude layer at omega_p dt = 4.5); diag.txt and medium.txt (K, dissipated) to 1e-11
# (medium_tol = 1e-13: the GMRES iteration count may differ by the summation order)
mcompare() {     # 1-rank medium.txt, N-rank medium.txt, label
python3 - "$1" "$2" "$3" <<'PY'
import os, sys
a=[l.split() for l in open(sys.argv[1]) if not l.startswith('#')]
b=[l.split() for l in open(sys.argv[2]) if not l.startswith('#')]
assert len(a)==len(b) and a, "different number of lines"
worst=0
for x,y in zip(a,b):
    for c in (2,3,4):
        worst=max(worst,abs(float(x[c])-float(y[c]))/max(abs(float(x[4])),1e-300))
print("mpi%s: 1 vs %s ranks, max difference of K, W_dissipated, the balance %.1e of the total; GMRES iterations %s / %s"
      % (sys.argv[3], os.environ["NP"], worst, a[-1][6], b[-1][6]))
sys.exit(0 if worst<1e-11 else 1)
PY
}
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/plasma_mirror_2d.deck grid.t_end=70fs numerics.medium_tol=1e-13 output.every=20 output.dir=mpi_med_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_medium.deck grid.t_end=70fs numerics.medium_tol=1e-13 output.every=40 output.dir=mpi_rzmed_$np > /dev/null
    $MPIRUN -np $np $EXE $D numerics.solver=irip numerics.mover=implicit numerics.shape=2 numerics.mover_tol=1e-11 numerics.medium_tol=1e-13 \
        grid.ny=8 grid.nz=8 grid.dy=0.25um grid.dz=0.25um grid.t_end=60fs output.every=6 "species electrons.uth_y=0.3" "species electrons.uth_z=0.3" \
        medium.density=0.5 "medium.profile=x > 20 um && x < 41 um" output.dir=mpi_imed_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rb_gas_2d.deck grid.t_end=40fs numerics.medium_tol=1e-13 numerics.medium_bound_tol=1e-11 output.every=20 output.dir=mpi_rb_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_rb.deck grid.t_end=30fs numerics.medium_tol=1e-13 numerics.medium_bound_tol=1e-11 output.every=40 output.dir=mpi_rzrb_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rb_gas_2d.deck grid.t_end=50fs window.start=24fs window.velocity=1 numerics.medium_tol=1e-13 numerics.medium_bound_tol=1e-11 output.every=20 output.dir=mpi_rbw_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_rb.deck grid.t_end=36fs window.start=20fs window.velocity=1 numerics.medium_tol=1e-13 numerics.medium_bound_tol=1e-11 output.every=40 output.dir=mpi_rzrbw_$np > /dev/null
done
compare mpi_med_1/diag.txt mpi_med_$NP/diag.txt " (O7 parabolic plasma mirror, 2D, CPML in y, $NP ranks along x)"
mcompare mpi_med_1/medium.txt mpi_med_$NP/medium.txt " (O7 parabolic plasma mirror: medium.txt)"
compare mpi_rzmed_1/diag.txt mpi_rzmed_$NP/diag.txt " (O7 RZ glass and plasma, $NP ranks along z)"
mcompare mpi_rzmed_1/medium.txt mpi_rzmed_$NP/medium.txt " (O7 RZ: medium.txt)"
compare mpi_imed_1/diag.txt mpi_imed_$NP/diag.txt " (O7 implicit mover with a Drude layer, $NP ranks along x)"
mcompare mpi_imed_1/medium.txt mpi_imed_$NP/medium.txt " (O7 implicit mover with a medium: medium.txt)"
compare mpi_rb_1/diag.txt mpi_rb_$NP/diag.txt " (O7 model = bound: Rb vapour, Maxwell-Bloch atoms, 2D, CPML in y, $NP ranks along x)"
mcompare mpi_rb_1/medium.txt mpi_rb_$NP/medium.txt " (O7 model = bound: medium.txt)"
compare mpi_rzrb_1/diag.txt mpi_rzrb_$NP/diag.txt " (O7 model = bound in RZ: Rb with ionization into species, $NP ranks along z)"
mcompare mpi_rzrb_1/medium.txt mpi_rzrb_$NP/medium.txt " (O7 model = bound in RZ: medium.txt)"
compare mpi_rbw_1/diag.txt mpi_rbw_$NP/diag.txt " (O7 model = bound, moving window: Rb, 2D, CPML in y, $NP ranks along x)"
mcompare mpi_rbw_1/medium.txt mpi_rbw_$NP/medium.txt " (O7 model = bound, moving window: medium.txt)"
compare mpi_rzrbw_1/diag.txt mpi_rzrbw_$NP/diag.txt " (O7 model = bound in RZ, moving window: Rb with ionization, $NP ranks along z)"
mcompare mpi_rzrbw_1/medium.txt mpi_rzrbw_$NP/medium.txt " (O7 model = bound in RZ, moving window: medium.txt)"
# Metal blocks in irip (decision 124): the conductor as a Drude medium with omega_p dt = 10. The streaming plasma through a metal
# slab (3D, NP ranks) and the laser of rz_medium.deck onto a stack of rings behind the glass (RZ, ranks along z).
for np in 1 $NP; do
    $MPIRUN -np $np $EXE $D numerics.solver=irip numerics.shape=2 numerics.medium_tol=1e-13 grid.ny=8 grid.nz=8 grid.dy=0.25um grid.dz=0.25um \
        grid.t_end=60fs output.every=6 "species electrons.uth_y=0.3" "species electrons.uth_z=0.3" \
        "metal.function=(x - 20 um) * (41 um - x)" output.dir=mpi_imet_$np > /dev/null
    $MPIRUN -np $np $EXE $D numerics.solver=irip numerics.shape=2 numerics.medium_tol=1e-13 grid.ny=8 grid.nz=8 grid.dy=0.25um grid.dz=0.25um \
        grid.t_end=60fs output.every=6 "species electrons.uth_y=0.3" "species electrons.uth_z=0.3" \
        "metal.function=(x - 20 um) * (41 um - x)" metal.particles=absorb output.dir=mpi_imeta_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_medium.deck grid.t_end=70fs numerics.medium_tol=1e-13 output.every=40 rings.r_in=1um rings.r_out=3um \
        rings.period=2um rings.gap=1um rings.z_start=10um rings.z_end=15um output.dir=mpi_rzmet_$np > /dev/null
done
compare mpi_imet_1/diag.txt mpi_imet_$NP/diag.txt " (metal slab in irip, streaming plasma through it, $NP ranks)"
mcompare mpi_imet_1/medium.txt mpi_imet_$NP/medium.txt " (metal slab in irip: medium.txt)"
compare mpi_imeta_1/diag.txt mpi_imeta_$NP/diag.txt " (metal slab in irip, particles = absorb: the stored charge across the ranks, $NP ranks)"
mcompare mpi_imeta_1/medium.txt mpi_imeta_$NP/medium.txt " (metal slab in irip, absorb: medium.txt)"
compare mpi_rzmet_1/diag.txt mpi_rzmet_$NP/diag.txt " (RZ rings in irip, $NP ranks along z)"
mcompare mpi_rzmet_1/medium.txt mpi_rzmet_$NP/medium.txt " (RZ rings in irip: medium.txt)"
# the absorbed charge in the checkpoint (vlpl_metal_absorbed): from step 9 on NP ranks exactly the uninterrupted run, on 1 and 3
# ranks to round-off (builds with HDF5)
MA="$D numerics.solver=irip numerics.shape=2 numerics.medium_tol=1e-13 grid.ny=8 grid.nz=8 grid.dy=0.25um grid.dz=0.25um grid.t_end=60fs output.every=3"
if $MPIRUN -np 1 $EXE $MA "metal.function=(x - 20 um) * (41 um - x)" metal.particles=absorb --check checkpoint.every=9 > /dev/null 2>&1; then
    $MPIRUN -np $NP $EXE $MA "species electrons.uth_y=0.3" "species electrons.uth_z=0.3" "metal.function=(x - 20 um) * (41 um - x)" \
        metal.particles=absorb checkpoint.every=9 output.dir=mpi_imetack_$NP > /dev/null
    for np in $NP 1 3; do
        rm -rf mpi_imetars_$np; mkdir -p mpi_imetars_$np/checkpoints
        cp mpi_imetack_$NP/checkpoints/checkpoint_000009.h5 mpi_imetars_$np/checkpoints/
        cp mpi_imetack_$NP/diag.txt mpi_imetack_$NP/medium.txt mpi_imetars_$np/
        $MPIRUN -np $np $EXE $MA "species electrons.uth_y=0.3" "species electrons.uth_z=0.3" "metal.function=(x - 20 um) * (41 um - x)" \
            metal.particles=absorb output.dir=mpi_imetars_$np --restart latest > /dev/null
    done
    same_run mpi_imetack_$NP/diag.txt mpi_imetars_$NP/diag.txt " (metal slab in irip, absorb: restart from step 9)"
    for np in 1 3; do
        compare mpi_imetack_$NP/diag.txt mpi_imetars_$np/diag.txt " (metal slab in irip, absorb: restart of $NP ranks on $np)"
    done
else
    echo "mpi (metal absorb in irip, restart): skipped (built without HDF5)"
fi
# end of the metal blocks in irip
# O7 restarts (HDF5): the media state in the checkpoint (linear J, P; the atoms' density matrix, bP, the last Et): the Rb deck and
# the RZ medium deck from step 200 (the window decks from a step after shifts) on NP ranks (exactly the uninterrupted run, diag.txt and medium.txt), on 1 and 3 ranks
o7_restart() {     # name, checkpoint step, deck and overrides
    nm=$1; ck=$2; shift 2
    $MPIRUN -np $NP $EXE "$@" checkpoint.every=$ck output.dir=mpi_${nm}ck_$NP > /dev/null
    for np in $NP 1 3; do
        rm -rf mpi_${nm}rs_$np; mkdir -p mpi_${nm}rs_$np/checkpoints
        cp mpi_${nm}ck_$NP/checkpoints/checkpoint_$(printf %06d $ck).h5 mpi_${nm}rs_$np/checkpoints/
        cp mpi_${nm}ck_$NP/diag.txt mpi_${nm}ck_$NP/medium.txt mpi_${nm}rs_$np/
        $MPIRUN -np $np $EXE "$@" output.dir=mpi_${nm}rs_$np --restart latest > /dev/null
    done
    same_run mpi_${nm}ck_$NP/diag.txt mpi_${nm}rs_$NP/diag.txt " (O7 $nm: restart from step $ck)"
    same_run mpi_${nm}ck_$NP/medium.txt mpi_${nm}rs_$NP/medium.txt " (O7 $nm: medium.txt after the restart)"
    for np in 1 3; do
        compare mpi_${nm}ck_$NP/diag.txt mpi_${nm}rs_$np/diag.txt " (O7 $nm: restart of $NP ranks on $np)"
        mcompare mpi_${nm}ck_$NP/medium.txt mpi_${nm}rs_$np/medium.txt " (O7 $nm: medium.txt, restart on $np)"
    done
}
if [ -n "$CK" ]; then
    o7_restart rb 200 examples/rb_gas_2d.deck grid.t_end=40fs numerics.medium_tol=1e-13 numerics.medium_bound_tol=1e-11 output.every=20
    o7_restart rzmed 200 examples/rz_medium.deck grid.t_end=70fs numerics.medium_tol=1e-13 output.every=40
    o7_restart rzrb 200 examples/rz_rb.deck grid.t_end=30fs numerics.medium_tol=1e-13 numerics.medium_bound_tol=1e-11 output.every=40
    o7_restart rbw 400 examples/rb_gas_2d.deck grid.t_end=50fs window.start=24fs window.velocity=1 numerics.medium_tol=1e-13 numerics.medium_bound_tol=1e-11 output.every=20
    o7_restart rzrbw 320 examples/rz_rb.deck grid.t_end=36fs window.start=20fs window.velocity=1 numerics.medium_tol=1e-13 numerics.medium_bound_tol=1e-11 output.every=40
else
    echo "mpi (O7 restarts): skipped (built without HDF5)"
fi
# Field ionization (docs/specs/field_ionization.md, decision 86): a laser into nitrogen, 2D (NP ranks along x) and RZ (ranks along z):
# diag.txt as above, ionization.txt (energies, events, charge states) to 1e-8 (the random numbers are keyed by the particle id)
icompare() {     # 1-rank ionization.txt, N-rank ionization.txt, label
python3 - "$1" "$2" "$3" <<'PY'
import os, sys
a=[l.split() for l in open(sys.argv[1]) if not l.startswith('#')]
b=[l.split() for l in open(sys.argv[2]) if not l.startswith('#')]
assert len(a)==len(b) and a, "different number of lines"
worst=0
for x,y in zip(a,b):
    for c in range(2,len(x)):
        worst=max(worst,abs(float(x[c])-float(y[c]))/max(abs(float(x[c])),1e-300))
print("mpi%s: 1 vs %s ranks, max relative difference in ionization.txt %.1e; events %s / %s, E_ion %s / %s"
      % (sys.argv[3], os.environ["NP"], worst, a[-1][6], b[-1][6], a[-1][2], b[-1][2]))
sys.exit(0 if worst<1e-8 else 1)
PY
}
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/ionization_2d.deck grid.t_end=24fs output.every=20 output.dir=mpi_ion_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_ionization.deck grid.t_end=24fs output.every=20 output.dir=mpi_rzion_$np > /dev/null
done
compare mpi_ion_1/diag.txt mpi_ion_$NP/diag.txt " (field ionization, 2D, $NP ranks along x)"
icompare mpi_ion_1/ionization.txt mpi_ion_$NP/ionization.txt " (field ionization, 2D: ionization.txt)"
compare mpi_rzion_1/diag.txt mpi_rzion_$NP/diag.txt " (field ionization, RZ, $NP ranks along z)"
icompare mpi_rzion_1/ionization.txt mpi_rzion_$NP/ionization.txt " (field ionization, RZ: ionization.txt)"
# Boosted frame (docs/specs/boosted_frame.md, decision 87): plasma streaming through the window at -beta c, injected 3 + filter reach cells
# behind the right edge, the laser at the boosted entrance plane, lab-frame snapshots of fields and particles; 3D RIP-CC
# (2 x 2 x 2 ranks) and RZ iRIP (ranks along z). diag.txt as above, lab_beams.txt and the lab field snapshots to 1e-8.
lcompare() {    # dir of 1 rank, dir of N ranks, label
python3 - "$1" "$2" "$3" <<'PY'
import os, sys, glob
from array import array          # no numpy: the default python3 of a compute node may not have it
def f64(path):                   # a lab_fields snapshot: raw float64 (boost_diag.cpp)
    a=array('d')
    with open(path,'rb') as fh: a.frombytes(fh.read())
    return a
d1, dn = sys.argv[1], sys.argv[2]
a=[l.split() for l in open(d1+"/lab_beams.txt") if not l.startswith('#')]
b=[l.split() for l in open(dn+"/lab_beams.txt") if not l.startswith('#')]
assert len(a)==len(b) and len(a)>0, "lab_beams.txt: different number of lines"
worst=0
for x,y in zip(a,b):
    assert x[:5]==y[:5], "lab_beams.txt: %s vs %s"%(x[:5],y[:5])
    for c in range(5,len(x)):
        worst=max(worst,abs(float(x[c])-float(y[c]))/max(abs(float(x[c])),1e-6))   # floor: eps of a flat beam is round-off
fw=0
files=sorted(glob.glob(d1+"/lab_fields/snap_*.bin"))
assert files, "no lab field snapshots"
for f in files:
    u=f64(f); v=f64(f.replace(d1,dn))
    assert len(u)==len(v), "snapshot sizes differ"
    fw=max(fw,max(abs(p-q) for p,q in zip(u,v))/max(max(abs(p) for p in u),1e-300))
print("mpi%s: 1 vs %s ranks, lab_beams.txt max relative difference %.1e, %d lab field snapshots max difference %.1e"
      % (sys.argv[3], os.environ["NP"], worst, len(files), fw))
sys.exit(0 if worst<1e-8 and fw<1e-8 else 1)
PY
}
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/lwfa_boost.deck grid.t_end=120fs output.dir=mpi_boost_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_lwfa_boost.deck grid.t_end=150fs output.dir=mpi_rzboost_$np > /dev/null
done
compare mpi_boost_1/diag.txt mpi_boost_$NP/diag.txt " (boosted frame, 3D RIP-CC, gamma 2)"
lcompare mpi_boost_1 mpi_boost_$NP " (boosted frame, 3D: lab-frame beams and fields)"
compare mpi_rzboost_1/diag.txt mpi_rzboost_$NP/diag.txt " (boosted frame, RZ iRIP, gamma 2, $NP ranks along z)"
lcompare mpi_rzboost_1 mpi_rzboost_$NP " (boosted frame, RZ: lab-frame beams and fields)"
# Lab snapshots across checkpoint restarts (decision 116): the RZ deck with RIP-CC, rz_deposit = chord, shape 2, filter 2 and the
# 3D deck on NP ranks, a checkpoint every 40 steps. Stopped runs (a shorter grid.t_end: they end with their open snapshots written
# as complete 0 and lab_beams.txt lines of them) are continued from their last checkpoint: once and twice on NP ranks, once on 3
# ranks. NP ranks: every lab file (fields, particles, lab_beams.txt) identical to the uninterrupted run (RESTART_EXACT=1); 3 ranks:
# lab_beams.txt and the lab fields as lcompare, the particle records (count, column sums) to 1e-8.
lrcompare() {   # uninterrupted dir, restarted dir, label, exact (1) or to round-off (0)
python3 - "$1" "$2" "$3" "$4" <<'PY'
import os, sys, glob
from array import array
def raw(p):
    with open(p,'rb') as fh: return fh.read()
def f64(b):
    a=array('d'); a.frombytes(b); return a
d1, d2, lab, exact = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]=="1"
files=["lab_beams.txt"]+sorted(os.path.relpath(f,d1) for f in glob.glob(d1+"/lab_fields/*")+glob.glob(d1+"/lab_particles/*"))
assert len(files)>3, "no lab files in "+d1
same=sum(raw(d1+"/"+f)==raw(d2+"/"+f) for f in files)
if exact:
    print("mpi%s: %d of %d lab files identical to the uninterrupted run (%d bytes)" % (lab, same, len(files), sum(len(raw(d1+"/"+f)) for f in files)))
    sys.exit(0 if same==len(files) else 1)
a=[l.split() for l in open(d1+"/lab_beams.txt") if not l.startswith('#')]
b=[l.split() for l in open(d2+"/lab_beams.txt") if not l.startswith('#')]
assert len(a)==len(b) and a, "lab_beams.txt: different number of lines"
worst=0
for x,y in zip(a,b):
    assert x[:5]==y[:5], "lab_beams.txt: %s vs %s"%(x[:5],y[:5])
    for c in range(5,len(x)): worst=max(worst,abs(float(x[c])-float(y[c]))/max(abs(float(x[c])),1e-6))
fw=0; pw=0; nrec=0
for f in files:
    if not f.endswith(".bin"): continue
    u=f64(raw(d1+"/"+f)); v=f64(raw(d2+"/"+f))
    assert len(u)==len(v), f+": sizes differ"
    if f.startswith("lab_fields"):
        fw=max(fw,max(abs(p-q) for p,q in zip(u,v))/max(max(abs(p) for p in u),1e-300))
    else:   # records in another order (the ranks): count and column sums, relative to the sum of |position|, |momentum|, w
        # (a transverse momentum without laser along it is round-off noise of zero mean); the record width from the .txt
        # (7; + 3 spin, + 6 Stokes vector and polarization axis of photons, decision 117), each extra column against sum |column|
        cols=[l for l in open(d1+"/"+f[:-4]+".txt") if l.startswith("columns")]
        W=len(cols[0].split("(")[0].split())-1 if cols else 7
        nrec+=len(u)//W
        for c in range(W):
            g=(0,1,2) if c<3 else (3,4,5) if c<6 else (6,) if c==6 else (c,)
            su=sum(u[c::W]); sv=sum(v[c::W]); sa=sum(abs(t) for k in g for t in u[k::W])
            pw=max(pw,abs(su-sv)/max(sa,1e-300))
print("mpi%s: %d of %d lab files identical; lab_beams.txt max relative difference %.1e, lab fields %.1e, %d particle records, column sums %.1e"
      % (lab, same, len(files), worst, fw, nrec, pw))
sys.exit(0 if worst<1e-8 and fw<1e-8 and pw<1e-8 else 1)
PY
}
if [ -n "$CK" ]; then
    LR="numerics.solver=rip_cc grid.dr=0.3um numerics.rz_deposit=chord numerics.shape=2 numerics.filter_passes=2"
    for c in rz 3d; do
        if [ $c = rz ]; then D="examples/rz_lwfa_boost.deck $LR"; T=150; T1=90; T2=120; else D=examples/lwfa_boost.deck; T=120; T1=70; T2=95; fi
        rm -rf mpi_lr${c}_a mpi_lr${c}_1 mpi_lr${c}_2 mpi_lr${c}_3
        lrun() { d=$1; n=$2; shift 2; $MPIRUN -np $n $EXE $D checkpoint.every=40 output.dir=$d "$@" > /dev/null; }
        lrun mpi_lr${c}_a $NP grid.t_end=${T}fs
        lrun mpi_lr${c}_1 $NP grid.t_end=${T1}fs
        lrun mpi_lr${c}_1 $NP grid.t_end=${T}fs --restart latest
        lrun mpi_lr${c}_2 $NP grid.t_end=${T1}fs
        lrun mpi_lr${c}_2 $NP grid.t_end=${T2}fs --restart latest
        lrun mpi_lr${c}_2 $NP grid.t_end=${T}fs --restart latest
        lrun mpi_lr${c}_3 $NP grid.t_end=${T2}fs
        lrun mpi_lr${c}_3 3 grid.t_end=${T}fs --restart latest
        lrcompare mpi_lr${c}_a mpi_lr${c}_1 " (lab snapshots, $c: restarted once on $NP ranks)" $RESTART_EXACT
        lrcompare mpi_lr${c}_a mpi_lr${c}_2 " (lab snapshots, $c: restarted twice on $NP ranks)" $RESTART_EXACT
        lrcompare mpi_lr${c}_a mpi_lr${c}_3 " (lab snapshots, $c: $NP ranks restarted on 3)" 0
    done
else
    echo "mpi (lab snapshots across restarts): skipped (built without HDF5)"
fi
# The same with the lab-fixed antenna (laser.injection = antenna, decision 89): the current sheet with its charge moves at
# -beta c across the ranks along x (z) and its transverse polarization Pi spans the rank edges in y and z (3D).
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/lwfa_boost.deck grid.t_end=120fs laser.injection=antenna output.dir=mpi_boostant_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_lwfa_boost.deck grid.t_end=150fs laser.injection=antenna output.dir=mpi_rzboostant_$np > /dev/null
done
compare mpi_boostant_1/diag.txt mpi_boostant_$NP/diag.txt " (boosted frame, antenna, 3D RIP-CC, gamma 2)"
lcompare mpi_boostant_1 mpi_boostant_$NP " (boosted frame, antenna, 3D: lab-frame beams and fields)"
compare mpi_rzboostant_1/diag.txt mpi_rzboostant_$NP/diag.txt " (boosted frame, antenna, RZ iRIP, $NP ranks along z)"
lcompare mpi_rzboostant_1 mpi_rzboostant_$NP " (boosted frame, antenna, RZ: lab-frame beams and fields)"
# Field ionization in the boosted frame (decision 90): nitrogen streaming at -beta c through the injection plane, the rate in the
# ion's rest frame; 2D (NP ranks along x) and RZ (ranks along z). The ids mix in the step of the first ionization pass, the same
# for any decomposition: ionization.txt to 1e-8 as above.
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/ionization_boost.deck output.dir=mpi_bion_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_ionization_boost.deck output.dir=mpi_rzbion_$np > /dev/null
done
compare mpi_bion_1/diag.txt mpi_bion_$NP/diag.txt " (field ionization, boosted frame, 2D, $NP ranks along x)"
icompare mpi_bion_1/ionization.txt mpi_bion_$NP/ionization.txt " (field ionization, boosted frame, 2D: ionization.txt)"
compare mpi_rzbion_1/diag.txt mpi_rzbion_$NP/diag.txt " (field ionization, boosted frame, RZ, $NP ranks along z)"
icompare mpi_rzbion_1/ionization.txt mpi_rzbion_$NP/ionization.txt " (field ionization, boosted frame, RZ: ionization.txt)"
# Spin-resolved QED (docs/specs/qed.md, decision 93): examples/qed_collision.deck, 1 GeV electrons with spin through an a0 = 100
# pulse (2 x 2 x 2 ranks: the beam crosses the rank faces along x), Monte-Carlo emission keyed by the particle id, photons
# above 1 MeV as a species: diag.txt as above (photon numbers equal), qed.txt (spins, chi, emitted energy, Stokes) to 1e-8
qcompare() {     # 1-rank qed.txt, N-rank qed.txt, label
python3 - "$1" "$2" "$3" <<'PY'
import os, sys
a=[l.split() for l in open(sys.argv[1]) if not l.startswith('#')]
b=[l.split() for l in open(sys.argv[2]) if not l.startswith('#')]
assert len(a)==len(b) and a, "different number of lines"
# relative to the largest |value| of each column over the run: the mean spin components and Stokes parameters near zero
# (sums of +- terms) are round-off there
n=len(a[0])
cm=[max(abs(float(x[c])) for x in a) for c in range(n)]
worst=0
for x,y in zip(a,b):
    for c in range(2,n):
        worst=max(worst,abs(float(x[c])-float(y[c]))/max(abs(float(x[c])),cm[c],1e-300))
m=max(range(len(a)),key=lambda i: float(a[i][7]))           # the row of the largest <chi>
print("mpi%s: 1 vs %s ranks, max difference in qed.txt %.1e (of each column's maximum); at the largest <chi> %s: <S2> %s / %s; "
      "E_rad %s / %s, most photons in the box %.0f / %.0f" % (sys.argv[3], os.environ["NP"], worst, a[m][7], a[m][5], b[m][5], a[-1][10],
      b[-1][10], max(float(x[-2]) for x in a), max(float(x[-2]) for x in b)))
sys.exit(0 if worst<1e-8 else 1)
PY
}
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/qed_collision.deck output.dir=mpi_qed_$np > /dev/null
done
compare mpi_qed_1/diag.txt mpi_qed_$NP/diag.txt " (QED: Monte-Carlo emission, spin, photon species)"
qcompare mpi_qed_1/qed.txt mpi_qed_$NP/qed.txt " (QED: qed.txt)"
# Binary collisions and impact ionization (docs/specs/collisions.md, decision 94): examples/collisions_2d.deck (dense Al plasma,
# e-e, e-i with impact ionization, i-i, automatic Coulomb log; NP ranks along x) and examples/rz_collisions.deck (hydrogen in RZ,
# ranks along z), mobile species (full PIC): diag.txt as above, collisions.txt (pairs, <s>, <lnL>, energies, events, <Z>,
# moments) to 1e-8 relative to each column's maximum (the pairing is keyed by the particle ids, the angles by the absolute cell)
ccompare() {     # 1-rank collisions.txt, N-rank collisions.txt, label
python3 - "$1" "$2" "$3" <<'PY'
import os, sys
a=[l.split() for l in open(sys.argv[1]) if not l.startswith('#')]
b=[l.split() for l in open(sys.argv[2]) if not l.startswith('#')]
assert len(a)==len(b) and a, "different number of lines"
n=len(a[0])
h=[l for l in open(sys.argv[1]) if l.startswith('#')][0].split()[1:n+1]
cm=[max(abs(float(x[c])) for x in a) for c in range(n)]
# dE of a block is the kinetic energy it exchanged; an intra-species block exchanges ~0 and its dE is rounding, summed in
# another order on N ranks: scale every dE column by the largest energy exchange of any block
de=[c for c in range(n) if h[c].startswith('dE_')]
for c in de: cm[c]=max(cm[k] for k in de)
worst=0
for x,y in zip(a,b):
    for c in range(2,n):
        worst=max(worst,abs(float(x[c])-float(y[c]))/max(cm[c],1e-300))
print("mpi%s: 1 vs %s ranks, max difference in collisions.txt %.1e (of each column's maximum); pairs %s / %s"
      % (sys.argv[3], os.environ["NP"], worst, a[-1][2], b[-1][2]))
sys.exit(0 if worst<1e-8 else 1)
PY
}
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/collisions_2d.deck grid.t_end=5fs output.every=25 output.dir=mpi_coll_$np > /dev/null
    $MPIRUN -np $np $EXE examples/rz_collisions.deck grid.t_end=5fs output.every=25 output.dir=mpi_rzcoll_$np > /dev/null
done
compare mpi_coll_1/diag.txt mpi_coll_$NP/diag.txt " (collisions + impact ionization, 2D, $NP ranks along x)"
ccompare mpi_coll_1/collisions.txt mpi_coll_$NP/collisions.txt " (collisions, 2D: collisions.txt)"
compare mpi_rzcoll_1/diag.txt mpi_rzcoll_$NP/diag.txt " (collisions, RZ, $NP ranks along z)"
ccompare mpi_rzcoll_1/collisions.txt mpi_rzcoll_$NP/collisions.txt " (collisions, RZ: collisions.txt)"
# Breit-Wheeler pairs (decision 96): examples/qed_cascade.deck in the small box of qed_collision (waist 1.5 um, 24 x 24 cells):
# 10 GeV electrons through an a0 = 80 pulse, photons decay into pairs that radiate again; decays keyed by the photon ids, the
# photons removed, the pairs appended: diag.txt as above (photon, electron and positron numbers equal), qed.txt to 1e-8
pcompare() {     # 1-rank qed.txt, N-rank qed.txt, label (the last 8 columns: the pairs)
python3 - "$1" "$2" "$3" <<'PY'
import os, sys
a=[l.split() for l in open(sys.argv[1]) if not l.startswith('#')]
b=[l.split() for l in open(sys.argv[2]) if not l.startswith('#')]
assert len(a)==len(b) and a, "different number of lines"
n=len(a[0])
cm=[max(abs(float(x[c])) for x in a) for c in range(n)]
worst=0
for x,y in zip(a,b):
    for c in range(2,n):
        worst=max(worst,abs(float(x[c])-float(y[c]))/max(abs(float(x[c])),cm[c],1e-300))
print("mpi%s: 1 vs %s ranks, max difference in qed.txt %.1e (of each column's maximum); pairs (sum w) %s / %s, their photons' "
      "energy %s / %s, <S+.e2> %s / %s" % (sys.argv[3], os.environ["NP"], worst, a[-1][-8], b[-1][-8], a[-1][-7], b[-1][-7], a[-1][-5], b[-1][-5]))
sys.exit(0 if worst<1e-8 and float(a[-1][-8])>0 else 1)
PY
}
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/qed_cascade.deck grid.ny=24 grid.nz=24 laser.waist=1.5um laser.waist_z=1.5um \
        "species electrons.ymin=1.42um" "species electrons.ymax=2.42um" "species electrons.zmin=1.42um" "species electrons.zmax=2.42um" \
        "species electrons.ppc_x=2" "species electrons.ppc_y=1" "species electrons.ppc_z=1" output.dir=mpi_pairs_$np > /dev/null
done
compare mpi_pairs_1/diag.txt mpi_pairs_$NP/diag.txt " (QED: Breit-Wheeler pairs, cascade)"
pcompare mpi_pairs_1/qed.txt mpi_pairs_$NP/qed.txt " (QED pairs: qed.txt)"
# The same cascade with iRIP at c dt = h_x = 0.16 um = h_perp, 4 times the RIP-CC step of the deck (RIP-CC rejects it:
# h_x^2 (1/h_y^2 + 1/h_z^2) = 2); ranks along x only (the partitioned solve), the beam keeps its macro-particle number
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/qed_cascade.deck numerics.solver=irip grid.nx=75 grid.dx=0.16um output.every=5 \
        grid.ny=24 grid.nz=24 laser.waist=1.5um laser.waist_z=1.5um \
        "species electrons.ymin=1.42um" "species electrons.ymax=2.42um" "species electrons.zmin=1.42um" "species electrons.zmax=2.42um" \
        "species electrons.ppc_x=8" "species electrons.ppc_y=1" "species electrons.ppc_z=1" output.dir=mpi_ipairs_$np > /dev/null
done
compare mpi_ipairs_1/diag.txt mpi_ipairs_$NP/diag.txt " (QED: Breit-Wheeler cascade with iRIP at 4 x the RIP-CC step, $NP ranks along x)"
pcompare mpi_ipairs_1/qed.txt mpi_ipairs_$NP/qed.txt " (QED pairs with iRIP: qed.txt)"
# QED in the boosted frame (decision 117): the same cascade at gamma 2 (iRIP, the lab-fixed antenna, ranks along x; ppc_x 28 keeps
# the beam's macro-particles: it is gamma^2 (1 + beta)^2 shorter in boosted cells), a lab snapshot of the fields and of all four
# species (electrons and pairs with their lab spin, photons with Stokes vector and polarization axis) at 29.35 fs: diag.txt,
# qed.txt, lab_beams.txt, the lab fields and the lab particle records (count, column sums) 1 vs NP ranks; then across restarts
# (a checkpoint every 40 steps, stopped at lab t_end = 24 fs): continued on NP ranks (every lab file identical with
# RESTART_EXACT=1) and on 3 ranks (to round-off).
qbrun() { d=$1; n=$2; shift 2
    $MPIRUN -np $n $EXE examples/qed_cascade.deck numerics.solver=irip laser.injection=antenna boost.gamma=2 \
        grid.ny=24 grid.nz=24 laser.waist=1.5um laser.waist_z=1.5um \
        "species electrons.ymin=1.42um" "species electrons.ymax=2.42um" "species electrons.zmin=1.42um" "species electrons.zmax=2.42um" \
        "species electrons.ppc_x=28" "species electrons.ppc_y=1" "species electrons.ppc_z=1" \
        lab_diagnostics.t_first=29.3537fs lab_diagnostics.count=1 lab_diagnostics.fields=true lab_diagnostics.gamma_min=1 \
        "lab_diagnostics.species=electrons photons positrons pair_electrons" lab_diagnostics.particles=true output.dir=$d "$@" > /dev/null; }
for np in 1 $NP; do qbrun mpi_qboost_$np $np; done
compare mpi_qboost_1/diag.txt mpi_qboost_$NP/diag.txt " (QED in the boosted frame, gamma 2, cascade, $NP ranks along x)"
pcompare mpi_qboost_1/qed.txt mpi_qboost_$NP/qed.txt " (QED in the boosted frame: qed.txt)"
lrcompare mpi_qboost_1 mpi_qboost_$NP " (QED in the boosted frame: lab snapshot of the fields and the four species, 1 vs $NP ranks)" 0
if [ -n "$CK" ]; then
    rm -rf mpi_qbr_a mpi_qbr_1 mpi_qbr_3
    qbrun mpi_qbr_a $NP checkpoint.every=40
    qbrun mpi_qbr_1 $NP checkpoint.every=40 grid.t_end=24fs
    qbrun mpi_qbr_1 $NP checkpoint.every=40 --restart latest
    qbrun mpi_qbr_3 $NP checkpoint.every=40 grid.t_end=24fs
    qbrun mpi_qbr_3 3 checkpoint.every=40 --restart latest
    lrcompare mpi_qbr_a mpi_qbr_1 " (QED in the boosted frame: lab snapshots restarted on $NP ranks)" $RESTART_EXACT
    lrcompare mpi_qbr_a mpi_qbr_3 " (QED in the boosted frame: lab snapshots, $NP ranks restarted on 3)" 0
else
    echo "mpi (QED in the boosted frame across restarts): skipped (built without HDF5)"
fi
# Structure runs (decision 102, docs/specs/structure3d.md): examples/structure_pipe.deck shortened (a corrugated pipe
# as embedded metal entering the moving window, two quadrupoles, a rigid self-field beam, CPML in y, z): diag.txt as
# above, and the wake potentials and the side flux to 1e-8 of their maxima
wcompare() {     # 1-rank file, N-rank file, label: all numeric columns of the data lines
python3 - "$1" "$2" "$3" <<'PY'
import os, sys
a=[l.split() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
b=[l.split() for l in open(sys.argv[2]) if l.strip() and not l.startswith('#')]
assert len(a)==len(b) and a, "different number of lines"
n=len(a[0])
cm=[max(abs(float(x[c])) for x in a) for c in range(n)]
worst=0
for x,y in zip(a,b):
    for c in range(1,n):
        worst=max(worst,abs(float(x[c])-float(y[c]))/max(cm[c],1e-300))
print("mpi%s: 1 vs %s ranks, max difference %.1e (of each column's maximum), %d lines" % (sys.argv[3], os.environ["NP"], worst, len(a)))
sys.exit(0 if worst<1e-8 else 1)
PY
}
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/structure_pipe.deck grid.t_end=2.5ps "wake axis.every=25" "wake offset.every=25" \
        "beam_slices drive.every=25" output.dir=mpi_struct_$np > /dev/null
done
compare mpi_struct_1/diag.txt mpi_struct_$NP/diag.txt " (structure: metal entering the window, quadrupoles, rigid beam)"
wcompare mpi_struct_1/wake_offset.txt mpi_struct_$NP/wake_offset.txt " (structure: wake potential at the offset)"
wcompare mpi_struct_1/side_flux.txt mpi_struct_$NP/side_flux.txt " (structure: side flux)"
wcompare mpi_struct_1/slices_drive.txt mpi_struct_$NP/slices_drive.txt " (structure: beam slices)"
# the same pipe with a dielectric liner across its corrugated wall (decision 100 in Cart3D, dielectric3d.hpp: the lines along x
# partitioned over the row of ranks along x, the entry of the window from div D, the front layer), the beam behind the layer
for np in 1 $NP; do
    $MPIRUN -np $np $EXE examples/structure_pipe.deck grid.t_end=2.5ps "wake axis.every=25" "wake offset.every=25" \
        "species drive.density_function=gauss(x - 0.3 mm, 40 um) * gauss(sqrt((y - 0.67 mm)^2 + (z - 0.64 mm)^2), 20 um) * (abs(x - 0.3 mm) < 160 um) * (sqrt((y - 0.67 mm)^2 + (z - 0.64 mm)^2) < 80 um)" \
        "dielectric liner.eps=x > 1.0 mm && sqrt((y - 0.64 mm)^2 + (z - 0.64 mm)^2) > 0.25 mm && sqrt((y - 0.64 mm)^2 + (z - 0.64 mm)^2) < 0.55 mm ? 2.1 : 1" \
        "beam_slices drive.every=25" output.dir=mpi_sdiel_$np > /dev/null
done
compare mpi_sdiel_1/diag.txt mpi_sdiel_$NP/diag.txt " (structure + dielectric liner, Cart3D window: lines along x over the ranks)" 1e-10
# Structure runs in RZ (decision 105, docs/specs/structure3d.md section 5): an axisymmetric corrugated pipe (metal block,
# spectral phi derivative) entering the moving window, an offset rigid bunch with its self-field (modes 0, 1), a witness
# through two quadrupoles (one with its axis off r = 0), wake probes on the axis and off it, beam slices: diag.txt and
# the probe and slice files of 1 and N ranks along z
cat > mpi_rzst.deck <<'DECK'
grid {
  geometry = rz
  modes = 2
  lambda0 = 100 um
  nz = 267
  nr = 100
  dz = 3 um
  dr = 5 um
  boundary_z = open
  boundary_r = absorbing
  r_absorber_cells = 16
  t_end = 4 ps
}
numerics {
  solver = rip_cc
  pusher = boris
}
window {
  start = 0 fs
  velocity = 1
}
metal {
  function = z >= 0.6 mm && (r > 0.35 mm || (r > 0.3 mm && (z - 40 um * floor(z / (40 um))) >= 20 um))
  phi_derivative = spectral
}
species bunch {
  charge = -1
  mass = 10000
  density = 1e-4
  density_function = gauss(z - 0.5 mm, 60 um) * gauss(sqrt((x - 40 um)^2 + y^2), 25 um)
  zmin = 0.25 mm
  zmax = 0.75 mm
  rmax = 0.15 mm
  ppc_theta = 8
  uz = 2000
  self_field = true
}
species witness {
  charge = -1
  mass = 1
  density = 1e-6
  density_function = gauss(z - 0.2 mm, 30 um) * gauss(sqrt((x - 30 um)^2 + (y + 20 um)^2), 20 um)
  zmin = 0.08 mm
  zmax = 0.32 mm
  rmax = 0.1 mm
  ppc_theta = 8
  uz = 200
}
quadrupole Q1 {
  z_start = 0.3 mm
  length = 0.4 mm
  gradient = 3000
}
quadrupole Q2 {
  z_start = 0.9 mm
  length = 0.3 mm
  gradient = -3000
  x_center = 10 um
}
wake axis {
  every = 25
}
wake off {
  x = 40 um
  every = 25
}
beam_slices w {
  species = witness
  dx = 10 um
  every = 25
}
output {
  every = 25
}
DECK
for np in 1 $NP; do
    $MPIRUN -np $np $EXE mpi_rzst.deck output.dir=mpi_rzst_$np > /dev/null
done
compare mpi_rzst_1/diag.txt mpi_rzst_$NP/diag.txt " (RZ structure: corrugated pipe in the window, self-field bunch, quadrupoles, $NP ranks along z)" 1e-8
wcompare mpi_rzst_1/wake_off.txt mpi_rzst_$NP/wake_off.txt " (RZ structure: wake probe off the axis)"
wcompare mpi_rzst_1/wake_axis.txt mpi_rzst_$NP/wake_axis.txt " (RZ structure: wake probe on the axis)"
wcompare mpi_rzst_1/slices_w.txt mpi_rzst_$NP/slices_w.txt " (RZ structure: slices of the witness through the quadrupoles)"
