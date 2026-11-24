# VLPL: the output and how to read it

Everything is in `output.dir` (default `out`). Text files start with `#` header lines that name their columns.
`python3 scripts/quick_look.py <output.dir> [--lambda0 <um>] [--density <cm^-3>]` prints the main numbers in
physical units (with the density: E_z on the axis also in units of E0).

## Code units

Unless a header gives another unit, numbers are in code units fixed by `grid.lambda0`:

| quantity | unit | value for lambda0 in um |
|---|---|---|
| length | 1/k0 = lambda0 / 2 pi | 0.15915 lambda0 um |
| time | 1/omega0 | 0.53088 lambda0 fs |
| electric field | m_e c omega0 / e | 3.2107e12 / lambda0 V/m (a0 = 1 is 1 in this unit) |
| magnetic field | m_e omega0 / e | 1.0710e4 / lambda0 T |
| density | n_c | 1.1149e21 / lambda0^2 cm^-3 |
| momentum | u = gamma v / c = p / (m c) | |
| energy (W_field, W_kin) | m_e c^2 n_c / k0^3 | 2D: per `grid.dz` of slab thickness |
| macro-particle weight w | n_c / k0^3 particles | |

Plasma numbers in these units: omega_p / omega0 = sqrt(n / n_c); the cold wave-breaking field E0 = m c omega_p / e
= sqrt(n / n_c); k_p = sqrt(n / n_c).

## diag.txt (every `output.every` steps)

`step  t  W_field  W_kin  W_total  max|dGauss|  max|rho|  max|divB|  N_<species> ...`

- **max|dGauss|** is the change of the Gauss-law residual since t = 0. The scheme conserves charge exactly, so
  it stays at round-off: about 1e-12 or below relative to max|rho|. A growing value means a problem of the setup
  (table in `debugging.md`), not noise.
- **max|divB|** stays at round-off (1e-14 of the field per cell).
- **W_field** contains the laser pulse once it has entered: with a laser it is large next to W_kin, and it falls
  when the pulse leaves or is depleted.
- **W_total** is conserved in a closed (periodic) box up to the particle noise heating; with open boundaries,
  lasers, absorbers and a moving window it changes by what enters and leaves (`xflux.txt` accounts for the x edges).
  The implicit mover conserves it to its tolerance.
- **N_<species>**: macro-particles; falls when particles leave or are absorbed, rises with a window or ionization.
- RZ runs also write the field energy and the Gauss residual per azimuthal mode (`rz_modes.txt`):
  energy growing in a mode the physics does not drive is noise or an instability.

## Fields

- **Cartesian slices:** `slices/<field>_<step>.f32`, float32, ny x nx values with x fastest, at the plane
  `output.slice_z` (default the middle); fields `output.slice_fields` from Ex Ey Ez Bx By Bz rho n_<species>
  (n in n_c). `slices/meta.txt` has nx, ny, dx, dy (code units). In python:
  `numpy.fromfile(f, dtype="f4").reshape(ny, nx)`. Movie: the repository's `tools/movie.py <dir> --fields Ez
  n_electrons` (matplotlib, ffmpeg).
- **RZ has no slices.** `output.axis_every` writes E_z on the axis (`plane/axis_r<rank>.bin`; read by
  `scripts/quick_look.py`), `output.plane_z` all components at a lab plane every step (`plane/plane_r<rank>.bin`,
  layout in `plane/meta.txt`).
- **openPMD** (`output.openpmd_every`, needs HDF5): `openpmd/data_<step>.h5`, fields `output.openpmd_fields`
  (E B rho J rho_<species>), particles `output.openpmd_species`. Values in code units with their `unitSI`; RZ in
  thetaMode layout. With openPMD-viewer:
  `ts = OpenPMDTimeSeries("<dir>/openpmd"); Ex, info = ts.get_field("E", "x", iteration=ts.iterations[-1])`.

## Beams and spectra (block `diagnostic <name>`)

- `beam_<name>.txt`: `step t[fs] N_macro Q[pC] E_mean[MeV] dE/E_rms E_max[MeV] eps_ny[um] eps_nz[um] x_mean[um]
  sigma_x[um] sigma_y[um] sigma_z[um]` for the particles of `diagnostic.species` with gamma >=
  `diagnostic.gamma_min`. In 2D the charge is per `grid.dz` of slab thickness.
- `spectrum_<name>.txt`: dQ/dE in pC/MeV on `diagnostic.bins` bins up to `diagnostic.e_max_mev`.
- `diagnostic.particles = true`: `particles/<name>_<step>_r<rank>.bin`, float64 columns x y z ux uy uz w.

## Physics modules

| file | written by | read it for |
|---|---|---|
| `ionization.txt` | field ionization | energy taken from the field, the conserved sum W_field + W_kin + E_cur, charge-state populations (sums of weights) |
| `collisions.txt` | `collisions` | per block: pairs, mean Coulomb logarithm, energy exchanged; impact-ionization events; per species drift and temperatures |
| `qed.txt` | `qed` | mean spin, mean and maximum chi, emitted energy, mean Stokes vector, pairs |
| `qed_spectrum.txt`, `qed_pairs.txt` | `qed` | photon spectrum dN/du and dN/d eps with polarization; pair spectra |
| `medium.txt` | `medium` | energy stored in the medium, dissipated energy, the energy balance, solver iterations |
| `xflux.txt` | `output.xflux = true` | power and accumulated energy through the left and right x edges: reflected and transmitted light |
| `xflux/` | `output.xflux_every` | time records of the outgoing field: spectra of the reflected light (repository `tools/xflux.py`) |
| `wake_<name>.txt`, `slices_<name>.txt` | `wake`, `beam_slices` | force on a test charge along the window and its integral (wake potential); slice current, centroid, size |
| `harmonic_focus*/` | `harmonic_focus` | peak and fluence of the band-passed field per node (repository `tools/hfocus.py`) |

## Boosted-frame runs

`diag.txt`, slices and `ionization.txt` are in the boosted frame and are not what the user wants to see. Use the
block `lab_diagnostics`:
- `lab_fields/snap_KKK.bin` + `.txt`: the six field components at the lab times; read with the repository's
  `tools/lab_fields.py` (in RZ the m >= 1 slots hold half the cos / sin amplitudes: use its `field_at_angle`);
- `lab_particles/<species>_KKK.bin`: x y z ux uy uz w in the lab frame;
- `lab_beams.txt`: charge, energy, spread, emittances at the lab times, as `beam_<name>.txt`.
A snapshot is complete when its lab time has left the box; snapshots still open at the end are marked incomplete.

## What to look at first

| run | number | expectation |
|---|---|---|
| any | max\|dGauss\| / max\|rho\| | round-off (<= 1e-12) |
| laser in vacuum | peak field at focus | a0 (3D, RZ); a0 (w0 / w(x))^(1/2) scaling in 2D |
| wakefield | E_z on the axis behind the pulse | a fraction of E0 = sqrt(n / n_c) for a0 <~ 1, ~ sqrt(a0) E0 in the blow-out regime; period lambda_p (longer when nonlinear) |
| wakefield | laser amplitude over time | self-focusing raises it when P > P_c; a fall within z_R means the pulse is not guided |
| solid target | reflected energy in `xflux.txt` | absorption = 1 - reflected / incident; a few percent to tens of percent, depends on the scale length |
| ionization | charge states | the level whose barrier-suppression field is below the peak field is fully ionized |
| collisions | temperatures | relax towards each other at the collision rate; total energy conserved |
| boosted | `lab_beams.txt`, lab fields | equal to the lab run of a short test case within a few percent |
