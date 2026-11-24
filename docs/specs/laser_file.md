# Laser from a LASY file

Status: implemented (`src/laser_file.*`, decision 35). Needs a build with HDF5.

## Physics
[LASY](https://github.com/LASY-org/lasy) writes a laser pulse as the complex envelope of the
transverse electric field at a plane, versus time:

    E_{x,y}(x, y, t) = Re[ p_{x,y} env(x, y, t) e^{-i omega_L t} ]

The structure can be anything: chirps, spatio-temporal couplings, aberrations, Laguerre-Gauss modes,
axiparabola or flying-focus fields, a measured pulse. VLPL injects the laser as the inbound
transport vector through the open left x boundary (decision 5). A field at a plane versus time is
exactly what that boundary needs, so a file written at the entrance plane is used as it is.

A file at another plane x_f (e.g. at the focus inside the plasma) is first propagated to the entrance
plane x = 0. This is exact for every plane wave in vacuum, without the paraxial approximation. In the
retarded time u = t - x, each component (omega, k_y, k_z) gains the phase

    e^{i (k_x - omega) (0 - x_f)},   k_x = (omega^2 - k_y^2 - k_z^2)^{1/2}.

Evanescent and backward parts are dropped; the run reports their energy share.

## File format (LASY 0.7, openPMD)
- Mesh `laserEnvelope` of one iteration (`file_iteration`, default the last).
- The dataset is complex: a compound of `r` and `i`, as h5py writes `complex128`.
- Attributes:
  - `angularFrequency` ω_L in rad/s;
  - `polarization` (p_x, p_y), real or complex, normalised on reading;
  - `envelopeField`: `electric_field` (V/m) or `normalized_vector_potential`. For the latter,
    E = i ω_L a - da/dt (central differences in t).
- Geometry:
  - `cartesian`: axes (t, y, x).
  - `thetaMode`: components [C_0, C_1, S_1, ...] over (t, r), with
    env = C_0 + Σ_m C_m cos mθ + S_m sin mθ. It is resampled on a (y, z) grid with the radial
    spacing.
- Grids are in SI with `gridUnitSI`, `gridGlobalOffset`, `position`, and `dataOrder` C or F.
- A cartesian axis of length 1 makes the pulse uniform along it, for slab beams.
- Axes: VLPL propagates along x, so LASY's x is our y and LASY's y is our z (p_x → E_y,
  p_y → E_z). The file's transverse origin sits at (`y_center`, `z_center`), by default the centre
  of the box.

## Time
The file's time `t_center` passes the file plane at the simulation time `delay + file_x`. At the
entrance plane, the field at time t is the envelope at u = t - delay + t_center. The carrier keeps
the file's absolute phase e^{-i ω_L u} (its CEP).
- `t_center` defaults to the intensity centroid of the file.
- `delay` defaults to the value at which the file's first time enters at t = 0.

## Discretisation
- Every rank at the open left x edge (rank 0 among them) reads the file. Each such rank:
  1. resamples a thetaMode file;
  2. propagates if `file_x ≠ 0`, with FFTs in u, y, z on a grid padded to powers of two with twice
     the time window and twice the transverse extent, keeping the padded extent, since the beam may
     be wider at x = 0;
  3. keeps only the part of the grid its own cells need, uploaded to the device.
- Each step, the boundary hook evaluates the envelope trilinearly at the guard positions of E_y
  (j + ½, k) and E_z (j, k + ½), at t_n + Δt, and applies the carrier. Both components are injected:
  T+_y with E_y and B_z, T-_z with E_z and -B_y. This is the same as the built-in lasers, which are
  summed with it.
- Interpolating the slowly varying envelope, not the carrier, keeps the injection accurate on the
  file's grid. The error is the linear interpolation of the envelope, about (h²/8) |env''|.
- Warnings:
  - the envelope at the edge of the grid exceeds 1e-3 of the peak (a pulse cut by the file, or one
    the propagation wrapped around);
  - the window starts moving before the pulse has entered; injection stops at the first shift.

## Deck interface
```
laser <name> {                     # or the block `laser`
  file           = laser.h5        # relative to the working directory or to the deck
  file_mesh      = laserEnvelope
  file_iteration = -1              # the last
  file_x         = 0 um            # x of the file's plane (absolute); 0: the entrance plane
  delay          = 30 fs           # t_center passes x = file_x at delay + file_x (default: the file enters at t = 0)
  t_center       = 0 fs            # file time placed at delay (default: the intensity centroid)
  y_center       = 6.4 um          # where the file's transverse origin sits (default: the box centre)
  z_center       = 6.4 um
  a0             = 2               # optional: rescale to peak |env| = a0 omega_L
  enabled        = true
}
```
The keys of the built-in pulse (waist, duration, polarization, mode_*, focus, phase) are not allowed
with `file`. Making a file with LASY (the axes are LASY's: its z is our x):
```python
from lasy.laser import Laser
from lasy.profiles import GaussianProfile
p = GaussianProfile(wavelength=0.8e-6, pol=(1, 0), laser_energy=1.0, w0=5e-6, tau=10e-15, t_peak=0)
Laser("xyt", (-15e-6, -15e-6, -30e-15), (15e-6, 15e-6, 30e-15), (64, 64, 128), p).write_to_file("laser", "h5")
# -> diags/laser_00000.h5; laser { file = diags/laser_00000.h5  a0 = 2 }
```

## Tests
- Fast (tests/test_lasy.cpp). The test writes the files in LASY's layout, checked against files of
  LASY 0.7.0.
  - L4: cartesian file of the built-in Gaussian against `LaserConfig::field`: 9.6e-4 of a0 (file grid
    dt = 0.2, dy = 0.25). The other component is 0.
  - L5:
    - a thetaMode file with m = 1 (Laguerre-Gauss (0, 1)) against the built-in LG mode: 1.1e-3;
    - a normalized vector potential with circular polarisation (complex attribute) against
      E = Re[p (i ω a - ȧ) e^{-iωu}] written out: 3e-5.
  - L6: a short pulse (τ = 12, about 20 % bandwidth) at its focus at x_f = z_R (W = 24, uniform in z),
    propagated exactly to x = 0. The reference is independent: a spectrally resolved paraxial beam,
    where every frequency gets its own q = -x_f - i z_R(ω) and the spectrum is summed numerically.
    Difference 1.6e-3 of the peak, the paraxial error 1/(kW)² ≈ 1.7e-3. Against the monochromatic
    paraxial beam the difference is 3.5e-2: that is the bandwidth, which the file propagation keeps.
  - L7: the file injected in the PIC loop against the built-in laser, fields in the box: 1.7e-3.
    Also on 4 ranks, where each rank keeps its own part of the grid.
- By hand (2026-10-27): files written by LASY 0.7.0 itself (GaussianProfile, xyt and rt), injected
  in a 3D box, against the built-in Gaussian with phase 90° (LASY's cosine carrier): 1.2e-2 (xyt) and
  5e-3 (rt). That is the interpolation of LASY's coarse sample grid (1 µm for w0 = 5 µm).

## References
- LASY, https://github.com/LASY-org/lasy (openPMD LaserEnvelope extension).
- GEM-PIC-RZ `laser.file` (src/laser_file.*): the same format, there in the slice coordinates of GT2.

## RZ, 2D and the boosted frame (decision 97)
- **RZ** (`geometry = rz`, src/laser_file_rz.cpp, `RzFileLaser`): the file stays a set of azimuthal modes,
  env(u, r, phi) = sum_{k = -K..K} a_k(u, r) e^{-i k phi}, with phi of the particles: LASY's x is the particles' x (as
  the built-in `polarization = x`). A thetaMode file is used mode by mode (a_0 = C_0, a_{+-k} = (C_k +- i S_k) / 2); a
  cartesian file is resampled on polar nodes around its own origin (the axis) and projected on the modes. Only |k| < the
  simulation's modes are kept (the dropped energy is reported). Cell-centred files (r_0 > 0) are put on nodes j dr from
  the axis (a_0 constant, a_k ~ r inside r_0).
  - `file_x != 0`: exact vacuum propagation per mode, Hankel transform of order |k| in r and FFT in u, every
    (omega, kappa) with e^{i (k_z - omega)(0 - file_x)} at fixed retarded time. The forward transform is the quadrature of
    GEM-PIC-RZ (exact for an interpolant r^|k| g(r^2), g piecewise linear in r^2, so regular at the axis); the inverse
    transform is evaluated on the nodes j dr up to the radial grid's edge, on twice the file's time window.
  - Injection through the open left z edge as the built-in RZ laser (rz_laser.cpp): E_x, E_y of the file at the azimuthal
    quadrature points of each radial node, E_r at rh_j, E_phi at r_j, projected on the modes, set as the inbound T.
  - `y_center`, `z_center` are not used in RZ (the file's origin is the axis).
- **2D** (`geometry = cartesian2d`): slab files only (cartesian, LASY y axis of length 1); injection as in 3D.
- **Boosted frame**: every laser file is emitted by the lab-fixed antenna of decision 89 (`injection = antenna` is the
  only choice and the default for files): K = -2 E_perp of the file at the lab plane x = 0 (z = 0 in RZ), evaluated at the
  lab time on the antenna's world line with the same time pre-compensation as the built-in lasers. A file at the plane
  x = 0 is what the antenna needs, so no propagation is involved unless `file_x != 0`. The file lasers are built on every
  rank (the sheet crosses all of them). A moving window is required (the antenna leaves the box when it starts); the run
  warns if the file's time span ends after `window.start`.
  - Cost: the antenna evaluates the file at (2 + 1) times x (4 (modes + 8)) angles x 2 radii per radial node and mode in
    every step, also after it has left the box, and a restart re-sums the sheet from step 0 (as for the built-in lasers).

### Tests (decision 97, tests/test_lasy.cpp)
- L8: RZ view of a thetaMode Gaussian, a thetaMode Laguerre-Gauss (0, 1) and a cartesian file (polar resampling)
  against `LaserConfig::field`: 4.9e-4, 5.6e-4, 1.4e-3 of a0.
- L9: RZ propagation of a short round pulse (tau = 12, W = 24) at its focus x_f = z_R to z = 0 against the spectrally
  resolved paraxial beam (round beam q0/q e^{i omega r^2 / (2q)}): 2.2e-3 of the peak (paraxial error 1.7e-3).
- L10: PIC runs, file of the built-in Gaussian against the built-in laser on the same grid, fields in the box: RZ lab
  (2 modes, polarization y) 5.6e-4, RZ boosted (gamma 2, antenna, iRIP) 2.5e-3 with the Gauss law at 2e-16, 2D lab
  5.7e-4, 2D boosted (antenna, iRIP) 3.1e-4 (Gauss law 2.7e-16).
- test_boost "lab particle snapshots with select_uz, select_r": exactly the records of the unfiltered file that pass.
