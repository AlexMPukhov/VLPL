# CHF: coherent harmonic focusing from a pre-curved overdense mirror

Track O benchmark 4 (docs/specs/overdense.md, Gate O in docs/ROADMAP.md): "CHF (dented or pre-curved target) in RZ and 3D:
focal intensity, spot size and position converge in dx and dr; energy balance (absorbed + reflected + transmitted) to
1e-3". Coherent harmonic focusing: Gordienko, Pukhov, Shorokhov, Baeva, PRL 94, 103903 (2005). Set up 2026-11-03 with
physics parameters approved by Alexander on 2026-11-03 (below; target shape and curvature, a0, density and the harmonic bands are
Alexander's choice); this README holds the code check: 2D (cylindrical mirror), RZ and 3D (paraboloid).

**Summary (2026-11-03).** 2D: both bands converge in position, spot and fluence at lambda0/512 (the high-band peak
still +15 % from /256); energy balance 1.5e-4 (target 1e-3). RZ (dz = lambda0/256, dr = lambda0/80): not converged (low
band peak +24 % from /128, focus 0.09 um closer to the target); balance 1.1e-3 up to 48 fs, then an excess growing to
6e-3 at 59 fs that does not fall with the cell size (open). 3D (lambda0/64): low band between RZ /128 and /256, balance
5.2e-3 like 2D at the same dx. The converged 3D point (lambda0/128) is prepared for JUPITER.

| | |
|---|---|
| laser | lambda0 = 0.8 um, a0 = 10, 5 fs FWHM of the intensity, waist 2 um focused on the vertex, normal incidence. 2D: E_y (in the plane) through a TFSF plane at x = 1.2 um with exact plane waves (decision 65), absorbing x (1 um layers), periodic y (8 um); RZ: polarised along x (modes 1), through a TFSF plane at z = 1.8 um with its incident grid (docs/specs/tfsf.md "RZ", decision 68), absorbing z layers of 1.6 um at both ends, radial absorber (0.5 um); the first RZ runs (below, "open edge") had the laser through the open left z edge, which returns part of the oblique light |
| target | 40 n_c electrons on fixed ions, parabolic mirror (focal length F = 3 um, sag r^2 / 4F; 2D a cylinder in y), vertex at x = 5 um (2D) / z = 6 um (RZ; open-edge runs 4 um), exponential pre-plasma 0.04 um cut at 0.2 um, 0.8 um thick; `density_function`, 2 x 2 (RZ 2 x 2 x (2M + 2)) particles per cell, shape 2, one filter pass |
| expected | the reflected harmonics focus at F = 3 um in front of the vertex (x = 2 um, z = 3 um; open edge z = 1 um), shifted towards the target by the motion of the surface during the pulse |
| measured | `harmonic_focus` (docs/specs/harmonic_focus.md): bands of orders 4.5 .. 10.5 ("low") and 10.5 .. 20.5 ("high"), 8-pole Butterworth edges; the peak of |E_band|^2 over the run and the fluence per node in a region around the focus; `output.xflux` for the energy balance |
| resolutions | 2D: dx = lambda0/64, /128, /256, /512 with dy = 1.1 dx (RIP-CC); RZ: dz = lambda0/64, /128, /256 with dr = 3.4 dz, 3.2 dz, 3.2 dz (dr = lambda0/19, /40, /80; RIP-CC with 4 modes needs dr >~ 3.1 dz), modes 0 .. 3; 3D: dx = lambda0/64, dy = dz = 1.55 dx = lambda0/41 (RIP-CC 3D stability), round paraboloid, the same laser polarised along y |

`series.py` writes the target deck and its vacuum twin for one resolution, `analyse.py` gives the focus of each band and
the energy balance. Runs in $WORK/chf on the CPU cluster (one node each, 24 threads).

    python3 bench/chf/series.py 2d 128 runs        # runs/t128/chf.deck, runs/v128/chf.deck
    (cd runs/t128 && build/vlpl chf.deck) ; (cd runs/v128 && build/vlpl chf.deck)
    python3 bench/chf/analyse.py runs/t128 runs/v128 --a0 10

## Energy balance
B = W_in + W_kin + Enet_left + Enet_right + E_abs_r + E_part_out (xflux.txt and diag.txt; W_in the field between the
measuring planes, E_abs_r the energy the radial absorber of RZ removed, E_part_out the kinetic energy of the particles
that left the box). It is compared with the incident energy that crossed the TFSF plane, the vacuum twin's B at its end
(2D: its Enet_right; RZ: 1e-3 below E_inj, the energy injected into the incident grid at z = 0, of which the low-frequency
part of the 5 fs pulse diffracts away before the plane); the open-edge RZ runs: the vacuum twin's B step by step. The
largest deviation after the pulse has entered:

| run | balance (max) | at the end | reflected | kinetic | transmitted | particles out | radial absorber |
|---|---|---|---|---|---|---|---|
| 2D lambda0/64 | 5.5e-3 | 5.5e-3 | 0.837 | 0.129 | 9.8e-3 | 2.3e-3 | - |
| 2D lambda0/128 | 1.55e-3 | 1.55e-3 | 0.845 | 0.126 | 6.3e-3 | 2.1e-3 | - |
| 2D lambda0/256 | 3.1e-4 | 2.5e-4 | 0.846 | 0.125 | 4.8e-3 | 2.9e-3 | - |
| 2D lambda0/512 | 1.5e-4 | -1.0e-4 | 0.843 | 0.127 | 4.6e-3 | 3.0e-3 | - |
| 3D lambda0/64, dy = dz = lambda0/41 | 5.2e-3 | 5.2e-3 | 0.866 | 0.108 | 6.2e-3 | 2.7e-3 | - |
| RZ lambda0/64, dr = lambda0/19, TFSF, 59 fs | 2.35e-2 | 2.35e-2 | 0.872 | 0.083 | 1.7e-2 | 1.1e-2 | 2.0e-2 |
| RZ lambda0/128, dr = lambda0/40, TFSF, 59 fs | 6.8e-3 (2.3e-3 to 48 fs) | 6.8e-3 | 0.853 | 0.087 | 9.0e-3 | 1.8e-2 | 2.9e-2 |
| RZ lambda0/256, dr = lambda0/80, TFSF, 59 fs | 6.0e-3 (1.1e-3 to 48 fs) | 6.0e-3 | 0.848 | 0.083 | 7.8e-3 | 2.4e-2 | 3.4e-2 |
| RZ lambda0/64, open edge, 48 fs | 1.45e-2 | 1.45e-2 | 0.858 | 0.092 | 1.4e-2 | 7.4e-3 | 1.2e-2 |
| RZ lambda0/128, open edge, 48 fs | 2.0e-3 | 1.1e-3 | 0.844 | 0.096 | 8.9e-3 | 1.2e-2 | 1.7e-2 |

The rest (1-3 %) is the field still in the box at the end (quasi-static fields of the target and late radiation). In 2D
the balance falls 3.5 and 4.9 times per halving of dx (second order) and is below 1e-3 at lambda0/256 and /512 (the
plasma heating there is below the 1e-4 noise of the measurement). 3D at lambda0/64 (5.2e-3, growing monotonically
+1.1e-3, +2.7e-3, +4.4e-3, +5.2e-3 at 27, 37, 43, 48 fs) is where 2D is at lambda0/64 (5.5e-3).

RZ, the balance against time (TFSF runs, 59 fs):

| dz, dr | 27 fs | 37 fs | 43 fs | 48 fs | 53 fs | 59 fs |
|---|---|---|---|---|---|---|
| lambda0/128, lambda0/40 | -4.0e-4 | -2.25e-3 | -1.56e-3 | -3e-6 | +2.21e-3 | +6.78e-3 |
| lambda0/256, lambda0/80 | -1.9e-4 | -1.08e-3 | -8.5e-4 | -2.0e-4 | +1.70e-3 | +5.95e-3 |

Up to 48 fs (the pulse in, the interaction, the reflected light out of the box) the error halves with dz and dr (first
order: 2.25e-3 -> 1.08e-3), so 1e-3 is reached there at about lambda0/256. **After 48 fs the energy grows by 6e-3 at both
resolutions** (-12 % from /128 to /256): this is not the grid heating, which would fall with the cell size. In that time
the hot electrons ejected by the surface spread out radially: the particles leaving the box carry 2.4 % of the laser
energy at /256 (1.8 % at /128) and the radial absorber takes 3.4 % (2.9 %). Checked with RZ lambda0/64 (2.35e-2 at 59
fs), one change at a time:
- E_abs_r counted the radial damping inside the z layers too, where Enet had already counted the light as gone (fixed
  2026-11-03, docs/specs/xflux.md, test R4): 2.35e-2 -> 1.99e-2. The table above is before this fix.
- Particles removed at the inner face of the radial absorber instead of the wall (an experimental build): 1.90e-2.
- A conducting radial wall instead of the absorber: 2.06e-2. So the radial absorber is not the source.
- The whole set-up 4 um further from the left z layer (the electrons thrown back cannot reach it in time): the excess
  is +1.46e-2 already while the reflected light is still in the box (W_in 0.82). So the electrons in the z layer are not
  the source either.
The excess grows while the reflected, diverging harmonic light crosses the box and falls with dz (9.5e-3, 2.2e-3, 9e-4
gained while the light leaves, 37 -> 48 fs, at /64, /128, /256), plus a later part that does not (1.4e-2, 6.8e-3, 6.2e-3
from 48 to 59 fs). A vacuum beam diverging at 37 deg (test R4, dz = lambda0/32) keeps the balance to 2.7e-4, so it is not
oblique propagation alone. Keeping the target out of the radial absorber does not change it (lambda0/64: 2.47e-2 instead of
2.35e-2).

(2026-11-06: correction - only part of the late excess is noise heating; the rest is an axis instability of the dense target
(decision 80, bench/noise_heating "The rest of the CHF excess"). The RZ high-band numbers of this README (modes 4 -> 6
+41 %, dr halved +39 %, iRIP against RIP-CC in the high band) were made with ppc_theta 10 and are noise-dominated: at
lambda0/64 the high-band fluence falls from 5.4 to 2.3 with ppc_theta 30; the low band is not (-0.4 %).
The noise part: bench/noise_heating/README.md and decision 79: the heating grows with the number
of azimuthal modes at fixed ppc_theta: use ppc_theta >= 2 (2M - 1); the transverse Gaussian shape removes it but damps
the harmonics.)

**The late excess is numerical heating of the laser-heated electrons** (2026-11-04, lambda0/64, dr = 3.4 dz, 4 modes, TFSF,
59 fs; runs in $WORK/chf/filt on the CPU cluster, ~1.5 core-h each). The balance against time, one change at a time:

| run | 27 fs | 37 fs | 48 fs | 59 fs | growth 37 -> 59 fs | kinetic at the end |
|---|---|---|---|---|---|---|
| no filter | +7.4e-4 | +1.13e-2 | +3.06e-2 | +5.49e-2 | 4.4e-2 | 0.100 |
| local filter, 1 pass (the series) | -8.9e-4 | -3.9e-3 | +5.3e-3 | +1.94e-2 | 2.3e-2 | 0.084 |
| projection filter, 1 pass | -9e-5 | +7.1e-3 | +1.87e-2 | +3.45e-2 | 2.7e-2 | 0.084 |
| local, 1 pass, 2 x ppc_theta | -8.8e-4 | -3.5e-3 | +2.5e-3 | +1.36e-2 | 1.7e-2 | 0.084 |
| local, 2 passes | -2.5e-3 | -1.17e-2 | -4.6e-3 | +6.9e-3 | 1.9e-2 | 0.081 |

- The energy appears as kinetic energy of the electrons and as the energy they carry out (radial absorber, box edges):
  without the filter W_kin is 0.100 of the incident energy at the end instead of 0.084. It is no bookkeeping error.
- It is not grid heating of the target: the same target without the laser (a0 = 0.001) gains 2e-10 (local filter) and
  5e-10 (no filter) of the a0 = 10 incident energy in the same 59 fs.
- It falls with the noise (growth 37 -> 59 fs): twice the particles -27 %, two passes instead of one -20 %, one local
  pass instead of none -47 %.
  It does not fall with the cell size at fixed particles per cell (late part 6.8e-3 at /128, 6.2e-3 at /256): the hot
  electrons spreading from the surface are sampled by few macro-particles whatever the cell, so their noise heating stays.
- The projection filter, which does not radiate, damps it less than the local filter.
For a converged energy balance in RZ more particles per cell (and filter passes) are needed, not finer cells; the focus
(the harmonic peak) is a separate convergence question. Without E_part_out the first RZ run
was off by -1.1 %: the electrons that leave the box (ejected by the surface) carry 1.2 % of the laser energy.

## The focus

2D (peak of |E_band|^2 in units of a0^2; position x of the maximum, its y within one cell of the axis y = 4.06 um;
FWHM across (y) and along (x) the axis; fluence in a0^2 / omega0):

| band | dx | peak / a0^2 | x (um) | FWHM y (um) | FWHM x (um) | fluence / a0^2 | x of fluence max (um) | FWHM y of fluence (um) |
|---|---|---|---|---|---|---|---|---|
| 4.5 .. 10.5 | lambda0/64 | 2.229 | 2.155 | 0.089 | 0.458 | 3.071 | 2.020 | 0.171 |
| | lambda0/128 | 2.384 | 2.242 | 0.092 | 0.423 | 3.310 | 2.075 | 0.165 |
| | lambda0/256 | 2.341 | 2.244 | 0.097 | 0.453 | 3.389 | 2.092 | 0.163 |
| | lambda0/512 | 2.323 | 2.255 | 0.097 | 0.464 | 3.284 | 2.095 | 0.164 |
| 10.5 .. 20.5 | lambda0/64 | 0.162 | 2.041 | 0.066 | 0.619 | 0.149 | 2.015 | 0.093 |
| | lambda0/128 | 0.745 | 2.192 | 0.049 | 0.248 | 0.367 | 2.160 | 0.066 |
| | lambda0/256 | 1.165 | 2.231 | 0.045 | 0.225 | 0.505 | 2.223 | 0.061 |
| | lambda0/512 | 1.336 | 2.246 | 0.046 | 0.232 | 0.521 | 2.236 | 0.062 |

- Orders 4.5 .. 10.5 converge: from lambda0/256 to /512 the peak changes by -0.8 % (from /128 to /256: -1.8 %), its
  position by 0.010 um, the spot not at all (0.097 um, lambda0 / 8), the fluence by -3.1 %. The focus lies 0.25 um behind
  the geometric one (x = 2 um), towards the target: the surface moves during the pulse.
- Orders 10.5 .. 20.5 converge more slowly: the peak changes +56 % from /128 to /256 and +15 % from /256 to /512, the
  fluence +38 % and +3.1 %, the position 0.039 and 0.015 um, the spot 0.045 -> 0.046 um. The fluence, the position and
  the spot are converged at lambda0/512 (3 % and 0.015 um); the peak is not yet (0.420, then 0.171 a0^2 per halving, a factor
  2.5: an estimated 1.45 a0^2 in the limit, 9 % above /512). As the risk in docs/ROADMAP.md says, CHF of N ~ 20 needs
  dx <~ lambda0/256 .. /512 (or the O5 closure).

RZ (phi = 0 is the polarisation plane; the maxima lie on the axis or one cell off it):

| band | dz, dr | laser | peak / a0^2 (phi = 0) | z - z_vertex (um) | FWHM r (um) | FWHM z (um) | peak / a0^2 (phi = 90) | fluence / a0^2 | z of fluence max - z_vertex (um) |
|---|---|---|---|---|---|---|---|---|---|
| 4.5 .. 10.5 | lambda0/64, lambda0/19 | TFSF | 14.26 | -3.357 | - | 0.818 | 9.54 | 24.82 | -3.265 |
| | lambda0/128, lambda0/40 | TFSF | 30.61 | -2.897 | 0.105 | 0.439 | 15.81 | 34.53 | -2.977 |
| | lambda0/256, lambda0/80 | TFSF | 37.85 | -2.810 | 0.098 | 0.330 | 22.66 | 38.39 | -2.889 |
| 10.5 .. 20.5 | lambda0/128, lambda0/40 | TFSF | 5.18 | -3.018 | 0.076 | 0.515 | 2.30 | 3.33 | -3.075 |
| | lambda0/256, lambda0/80 | TFSF | 11.43 | -2.905 | 0.059 | 0.257 | 4.61 | 6.04 | -2.889 |
| 4.5 .. 10.5 | lambda0/64, lambda0/19 | open edge | 14.41 | -3.355 | - | 0.834 | 9.82 | 23.23 | -3.244 |
| | lambda0/128, lambda0/40 | open edge | 28.71 | -2.885 | 0.109 | 0.470 | 15.20 | 33.29 | -2.975 |
| 10.5 .. 20.5 | lambda0/128, lambda0/40 | open edge | 5.68 | -3.012 | 0.075 | 0.502 | 1.60 | 3.29 | -3.042 |

- The paraboloid focuses in both transverse directions: orders 4.5 .. 10.5 reach 30.6 a0^2 at lambda0/128, thirteen times
  the 2D cylinder's 2.4 a0^2 and thirty times the laser's a0^2.
- TFSF plane with absorbing z layers against the open left edge (which returns part of the oblique light, and had the focus
  1.1 um in front of it): the same focus position (0.01 um), at lambda0/128 the peak +6.6 % (low band) and -9 % (high band),
  the fluence +3.7 %, the spot 0.105 against 0.109 um; at lambda0/64 -1 %. The TFSF laser also has the compact envelope of
  docs/specs/tfsf.md (zero beyond 3 sigma) and travels 6 instead of 4 um to the target, so not all of the difference is
  the edge.
- RZ at lambda0/64 (dr = lambda0/19) is far from it (the focus at z = 0.64 um instead of 1.1, half the peak): with
  RIP-CC and 4 modes dr >~ 3.1 dz, so dr = lambda0/19 has 2-3 points across the spot.
- **RZ is not converged at dz = lambda0/256, dr = lambda0/80.** From /128 to /256: low band peak +24 % (30.6 -> 37.9
  a0^2), fluence +11 %, the focus moves 0.09 um towards the target, the spot 0.105 -> 0.098 um, the axial FWHM
  0.44 -> 0.33 um; high band peak x2.2 (5.2 -> 11.4 a0^2), fluence x1.8, spot 0.076 -> 0.059 um. The fluence maxima of
  both bands now lie at the same point (z - z_vertex = -2.889 um). This is the step 2D made from /64 to /128 (peak +7 %,
  focus 0.09 um): in RZ the spot (~0.1 um FWHM) has 10 radial cells at dr = lambda0/80, and dr is tied to dz by the
  RIP-CC mode limit (dr >~ 3.1 dz). The next point, dz = lambda0/512 with dr = lambda0/160, costs ~16 times /256 on
  CPUs; iRIP with dr = dz has no mode limit on dr and would refine r alone.
- 3D at dx = lambda0/64, dy = dz = lambda0/41 (transverse cells between RZ /128 and /256): low band peak 33.80 a0^2 at
  x - x_vertex = -2.909 um, FWHM 0.101 (y, the polarisation) and 0.094 um (z), 0.392 um along x, fluence 44.26 a0^2 at
  -2.988 um; high band 3.60 a0^2 at -3.075 um, FWHM 0.072 / 0.069 um, fluence 3.11 at -3.138 um. The low band lies
  between RZ /128 and /256 (peak 30.6 / 37.9, position -2.897 / -2.810, spot 0.105 / 0.098 um): the 3D code and the RZ
  modes agree on the focusing paraboloid within the resolution change of RZ itself. The high band in 3D is low (3.6
  against 5.2 a0^2 in RZ /128): at dx = lambda0/64 the orders 10.5 .. 20.5 are 3-6 points per wavelength, as in 2D /64
  (0.16 a0^2, 7 times below 2D /512). 3D lambda0/128 (JUPITER, below) is the next 3D point.
- On the axis the field at phi = 0 and 90 deg differs: the polarisation plane focuses higher (30.6 against 15.8 a0^2);
  the fluence is the same on the axis (both angles see the same point).

### 3D lambda0/128 (JUPITER, Thomas Wilson, 2026-11-04)

16 GH200 GPUs (4 x 2 x 2 ranks, CUDA-aware MPI), 2494 steps in 531 s; the vacuum twin on 4 GPUs in 174 s. The local filter
(the run's decomposition is refused by the projection; series.py now sets it explicitly). x - x_vertex, x_vertex = 5 um:

| band | dx, dy = dz | peak / a0^2 | x - x_vertex (um) | FWHM y, z (um) | FWHM x (um) | fluence / a0^2 |
|---|---|---|---|---|---|---|
| 4.5 .. 10.5 | lambda0/64, lambda0/41 | 33.80 | -2.909 | 0.101, 0.094 | 0.392 | 44.26 |
| | lambda0/128, lambda0/83 | 53.68 | -2.824 | 0.093, 0.087 | 0.307 | 52.87 |
| 10.5 .. 20.5 | lambda0/64, lambda0/41 | 3.60 | -3.075 | 0.072, 0.069 | - | 3.11 |
| | lambda0/128, lambda0/83 | 23.00 | -2.899 | 0.056, 0.055 | 0.260 | 10.89 |

- Energy balance 1.78e-3 (lambda0/64: 5.2e-3, 2.9 times per halving); reflected 0.866, kinetic 0.105, transmitted 5.4e-3,
  particles out 2.9e-3, field left in the box 2.2e-2.
- **3D is not converged at lambda0/128:** low band peak +59 %, fluence +19 %, high band peak x6.4, fluence x3.5 from /64. It is
  now above RZ lambda0/256 (37.9 and 11.4 a0^2) at a similar transverse cell (dy = lambda0/83, RZ dr = lambda0/80).
- 89 % of the run is MPI wait (475 s): the target sits in one of the 4 x slabs, so 4 of the 16 GPUs do the particle work.
  Ranks only in y and z (`grid.mpi_x = 1`) put the target on every rank.

### RZ: radial cell, modes and solver at lambda0/128 (iRIP, 2026-11-04)

iRIP has no mode limit on dr, so dr and the number of modes can be changed at dz = lambda0/128 (TFSF, 59 fs, local filter,
ppc_theta = 2M + 2; $WORK/chf/irip on the CPU cluster, epyc, 32 threads). Peaks at phi = 0, z - z_vertex (z_vertex = 6 um):

| run | modes | low peak / a0^2 | low fluence | high peak / a0^2 | high fluence | low peak at (um) | balance max | core-h |
|---|---|---|---|---|---|---|---|---|
| RIP-CC, dr = lambda0/40 (above) | 4 | 30.61 | 34.53 | 5.18 | 3.33 | -2.897 | 6.8e-3 | 14 |
| iRIP, dr = lambda0/40 | 4 | 25.54 | 32.88 | 3.64 | 2.85 | -2.990 | 2.6e-3 | 15 |
| iRIP, dr = lambda0/80 | 4 | 31.89 | 35.26 | 5.07 | 4.13 | -2.921 | 1.3e-3 | 41 |
| iRIP, dr = lambda0/40 | 6 | 27.70 | 35.58 | 5.12 | 3.38 | -2.996 | 2.1e-3 | 26 |
| iRIP, dr = lambda0/40 | 8 | 29.13 | 37.12 | 5.58 | 3.77 | -2.999 | 1.8e-3 | 40 |

- **dr matters on its own:** at fixed dz, dr = lambda0/40 -> /80 raises the low peak 25 % and the high peak 39 %.
- **The modes matter:** 4 -> 6 -> 8 modes raise the low peak 8.5 % and 5.2 % (fluence 8.2 %, 4.3 %), the high peak 41 % and 9 %.
  So 4 modes (0 .. 3) cost the low band ~15-20 % in the limit and the high band more. Together with dr this accounts for part
  of the 3D-RZ gap (RZ /256 x ~1.2 ~ 45 against 3D /128 53.7 a0^2), not all of it; neither geometry is converged.
- **The solver matters at this resolution:** on the same grid iRIP gives a 17 % lower low peak than RIP-CC (25.5 against 30.6),
  0.09 um further from the target. Converged results must agree between the two; so far they do not.
- iRIP's energy balance stays within 2.6e-3 and ends near 0 (RIP-CC /128: 6.8e-3 at the end): the late noise heating
  (above) is much weaker with iRIP's dispersion.

### iRIP against RIP-CC: the dispersion of oblique harmonics

Both closures are exact along z (c dt = dz) but not for oblique waves, and differently (decision 61):
RIP-CC sin^2(w D/2) = sin^2(kz D/2) + (D^2 Lam/4) cos^2(kz D/2), iRIP tan^2(w D/2) = tan^2(kz D/2) + D^2 Lam/4, with Lam the
discrete transverse eigenvalue. At fixed frequency and k_perp the numerical kz is too large, by an amount that grows with
the angle; over the 3 um from the target to the focus the phase error (rad, `python3 bench/chf/dispersion.py`):

| dz, dr | order N, angle theta | RIP-CC | iRIP |
|---|---|---|---|
| lambda0/128, lambda0/40 | 7, 0.2 / 0.4 / 0.6 | 0.04 / 0.30 / 1.12 | 0.14 / 0.69 / 2.02 |
| | 15, 0.2 / 0.4 / 0.6 | 0.42 / 2.87 / 10.5 | 1.29 / 6.26 / 17.8 |
| lambda0/128, lambda0/80 | 7, 0.2 / 0.4 / 0.6 | 0.03 / 0.15 / 0.37 | 0.13 / 0.54 / 1.29 |
| lambda0/256, lambda0/80 | 7, 0.2 / 0.4 / 0.6 | 0.01 / 0.08 / 0.28 | 0.04 / 0.17 / 0.52 |
| | 15, 0.2 / 0.4 / 0.6 | 0.11 / 0.73 / 2.76 | 0.34 / 1.68 / 4.94 |

The light reflected by the paraboloid converges at angles up to ~0.6, so the angle-dependent error is a spherical aberration
of the harmonic focus. iRIP's is 2-3 times RIP-CC's at the same grid, which explains why its low-band peak is 17 % lower
(phase spread ~0.7 against ~0.3 rad at order 7). It also explains the slow convergence of the high band in both
geometries (order 15 at angle 0.4 still has 0.7 rad at lambda0/256), and why refining dr helps RIP-CC more than iRIP (iRIP's
error is mostly the tan form in dz). The cure is in the scheme, not in the resolution: a closure whose oblique dispersion is
corrected (O5 in docs/ROADMAP.md), checked against these formulas on a small vacuum case.

### Why iRIP's focus is 17 % below RIP-CC's: chromatic focal shift (2026-11-06)

Small cases, one step of the chain each, against an independent reference (scripts in research/chf_gap/, runs in
$WORK/chfgap on the CPU cluster, epyc; about 60 core-h in all). **Neither solver is wrong.** Both have the oblique
dispersion of the previous section, which focuses every harmonic order at a different z (longitudinal chromatic
aberration). The shift grows with the order, and iRIP's is about twice RIP-CC's. The focus of the band is the sum of
foci spread along z, so its peak is lower. O5 removes the shift.

1. **Reflection at the surface: same for all three.** Fresnel reflection off a 40 n_c step (fixed ions, a0 = 1e-3; the
   F1 set-up of docs/specs/fresnel.md). |r| = 1 to 1e-4 in all cases. arg(r / r_F) in rad, RIP-CC / iRIP / iRIP+O5:

   | case | lambda0/64 | lambda0/128 |
   |---|---|---|
   | 0 deg | -9.55e-3, all three equal | -2.59e-3, all three equal |
   | 30 deg, s | -6.82e-3 / -6.72e-3 / -6.86e-3 | -7.6e-4 / -7.3e-4 / -8.8e-4 |
   | 30 deg, p (surface between nodes) | -2.71e-3 / -2.71e-3 / -6.23e-3 | +1.02e-3 / +1.02e-3 / +1.6e-4 |

   The coupling of the field to the surface current is second order and does not differ between the solvers.
2. **Generation: the same harmonic energy.** In the runs of the previous sections (rz2/t128 and irip/rr3.2/t128), the
   low-band fluence integrated over each plane z = 2.0 .. 4.7 um agrees between iRIP and RIP-CC to 0.983 .. 1.015.
   The peak axial fluence is 9 % lower with iRIP, and the focus is 0.15 um further from the target. The high band carries
   3-10 % more energy with iRIP. The full relativistic surface rung (harmonic spectrum of a flat target) was replaced by this
   in-run check: the generated energy already agrees.
3. **Vacuum focus of one harmonic order, the independent reference.** A beam at lambda0/N, NA 0.5, focused 3 um from the
   open left edge on the CHF grid (dz = 0.00625 um, dr = 0.02 um = g1), and refined x2 (g2) and x4 (g4). No particles,
   2 modes, `harmonic_focus` band 0.5 .. 1.5 of its own frequency. Peak position z (um), and peak / reference peak:

   | order | g1 RIP-CC | g1 iRIP | g1 iRIP+O5 | g2 RIP-CC / iRIP / O5 | g4 RIP-CC |
   |---|---|---|---|---|---|
   | 5 | 2.652 | 2.697 | 2.604 | 2.621 / 2.632 / 2.597 | |
   | 7 | 2.748 | 2.842 | 2.658 | 2.663 / 2.684 / 2.641 | 2.656 |
   | 10 | 2.901 | 3.102 | 2.712 | 2.746 / 2.791 / 2.682 | |
   | 7, peak / g4 | 1.058 | 1.077 | 0.942 | 1.006 / 1.010 / 0.978 | 1 |

   - **The chromatic shift.** Relative to the converged focus (2.61 / 2.656 / 2.69 um for orders 5 / 7 / 10), the focus
     moves away from the source by the following amounts:
     - RIP-CC: +0.04 / +0.09 / +0.21 um;
     - iRIP: +0.09 / +0.19 / +0.41 um;
     - iRIP+O5: -0.01 / 0.00 / +0.02 um.

     Across orders 5-10, the foci therefore spread over 0.17 um (RIP-CC), 0.32 um (iRIP) and 0.03 um (O5). For comparison,
     the converged focal depth (FWHM along z) is about 0.70 / 0.58 / 0.46 um. The difference between iRIP's and RIP-CC's shift at order 7 is
     0.10 um. This is the offset of the two CHF low-band foci (0.09 um, previous section).
   - **The beam itself.** The rms radius of the fluence along z matches the g4 reference within 2 % for O5 on g1 at every z,
     before and behind the focus. RIP-CC's and iRIP's curves are shifted along z.
   - **One caveat for O5.** The single-order peak of O5 on g1 is 6 % low (order 7), with a 5 % wider spot. It converges
     (g2: -2.2 %). One harmonic alone does not show the gap: there iRIP's peak is not lower, only displaced.
4. **The CHF case itself at lambda0/128.** dr = lambda0/40, 4 modes, the approved parameters, with open z edges for all
   three solvers, because O5 is not yet available with the RZ TFSF plane (decision 74). The open left edge returns a few %
   of the oblique harmonics, so these runs are compared only with each other:

   | solver | low peak / a0^2 | low fluence | high peak / a0^2 | high fluence | core-h |
   |---|---|---|---|---|---|
   | RIP-CC | 28.78 | 34.24 | 5.84 | 3.51 | 8 |
   | iRIP | 24.49 | 32.43 | 3.70 | 2.98 | 10 |
   | iRIP+O5 | 35.32 | 35.31 | 9.62 | 4.45 | 16 |

   - iRIP is 15 % below RIP-CC, as with the TFSF plane (17 %).
   - O5 is 23 % above RIP-CC in the low band and 65 % above it in the high band. That is close to RIP-CC at lambda0/256 with the
     TFSF plane (37.9 and 11.4 a0^2, above). The O5 focus also lies nearer the target (0.15 um from RIP-CC's), and the low
     band's fwhm_z is 0.35 um against 0.46 (RIP-CC) and 0.53 (iRIP); RIP-CC at lambda0/256 has 0.33. With the foci of the
     orders no longer spread, the focus of the band is shorter.

Consequences:
- The RZ CHF results at lambda0/128 without O5 are limited by the chromatic shift of the scheme, not by the surface physics.
- O5 at lambda0/128 comes close to RIP-CC at lambda0/256, at about 2 times RIP-CC's /128 cost (16 against 8 core-h here).
- What was missing for the production set-up, O5 with the RZ TFSF plane, is in since decision 78 (2026-11-06; leakage at
  round-off, docs/specs/overdense.md "O5 with the RZ TFSF plane").

### Cost of 3D lambda0/256 (an estimate only: not planned, code development comes first, Alexander 2026-11-04)

1.75e9 particles (both species) at /128 become 1.4e10 at /256 (x8), the cells 6.9e9 (x8), the steps 5000 (x2): memory x8, ~2.4 TB
(not x16), work x16. On GH200 (96 GB each) that needs 32 GPUs at the edge, 64 GPUs (16 nodes) comfortably. With ranks only in
y, z (1 x 8 x 8, every rank holds part of the target) the /128 work (4 GPUs busy for 531 s, ~2100 GPU-s) times 16 is ~34000 GPU-s,
~10 min on 64 GPUs plus overhead: about 16 nodes x 0.5 h = 8 node-h, the twin ~2 node-h; the two harmonic_focus maps ~7 GB each.
With the 4 x 2 x 2 type of decomposition the imbalance would make it ~4 times longer.

## Cost (one cluster node, 24 threads)

| run | wall time | particles |
|---|---|---|
| 2D lambda0/64, /128, /256 | 34 s, 4 min, 34 min | 0.19 M, 0.74 M, 3.0 M |
| RZ lambda0/64, /128, open edge (5.5 um, 48 fs) | 2.6 min, 21 min (88 % particle push) | 0.36 M, 1.44 M |
| RZ lambda0/64, /128, TFSF (9 um, 59 fs) | 4.3 min, 34 min | 0.36 M, 1.44 M |

The 2026-11-03 points ran on 8 cluster nodes each (MPI, 4 ranks x 6 threads per node), their vacuum twins on 2:

| run | wall time | core-hours (target + twin) | note |
|---|---|---|---|
| RZ lambda0/256, dr = lambda0/80 | 2.0 h (twin 53 min) | 384 + 42 | 32 ranks along z: the target is in a few of them, the MPI wait is 88 % of the run (6288 s, shown as xflux, the first allreduce after the field step) |
| 2D lambda0/512 | 70 min (twin 87 min) | 225 + 69 | migrate 25 % |
| 3D lambda0/64 | 62 min (twin 49 min) | 200 + 40 | push 84 % |

The estimate from the one-node test steps was ~260 core-hours; the runs took 960, mostly because the work of the target
does not spread over 32 slabs along the laser axis (RZ: 6 times slower per core than the one-node estimate). On a few
nodes, or with a decomposition weighted by the particles, these points cost about a third.

## 3D lambda0/128 on JUPITER (for Thomas Wilson)

The converged 3D point is too large for the CPU cluster (~45 node-hours there): 1280 x 826 x 826 cells (dx = lambda0/128, dy = dz = 1.55
dx), ~2500 steps, ~1e9 particles per species, ~300 GB in all. `jupiter_3d128.sbatch` runs it on GH200 GPUs, one rank per GPU,
with the settings and the build of `bench/jupiter/` (README there):

    git fetch && git checkout chf                 # or main, once the chf pull request is merged
    sh bench/jupiter/build.sh                     # build-gpu-mpi (once; ACCOUNT in bench/jupiter/env.sh)
    sbatch --account=<project> bench/chf/jupiter_3d128.sbatch                                  # target: 4 nodes, 16 GPUs
    sbatch --account=<project> --nodes=1 --export=ALL,KIND=v bench/chf/jupiter_3d128.sbatch    # vacuum twin: 1 node

Result: above ("3D lambda0/128"). The job writes the decks with `series.py 3d 128` and runs in `bench/jupiter/runs/chf3d128.t` (target) and `.v` (twin), limit
3 h each (expected well below 1 h on 16 GPUs). Please send back `run.log`, `out/diag.txt`, `out/xflux.txt` of both and the two
folders `out/harmonic_focus_low`, `out/harmonic_focus_high` of the target (~0.9 GB each), or the output of
`python3 bench/chf/analyse.py bench/jupiter/runs/chf3d128.t bench/jupiter/runs/chf3d128.v --a0 10 --json chf3d128.json`
(python3 with numpy) together with the json.

## Proposed next runs (Alexander decides the physics first)
- Fix the physics: target shape and curvature (pre-curved paraboloid or a flat target dented by the laser), a0,
  density, pulse length, the harmonic bands.
- Done 2026-11-03: 2D lambda0/512, RZ lambda0/256 (dr = lambda0/80), 3D lambda0/64 (above).
- The late energy excess of RZ after 48 fs (6e-3 at both resolutions): check the bookkeeping of the electrons at the
  radial wall (a code check, small runs) before longer RZ runs.
- RZ convergence: dz = lambda0/512, dr = lambda0/160 (RIP-CC, ~16 times /256), or iRIP with dz = lambda0/256 and dr
  refined alone; a mode check (5 modes) at lambda0/128 (~30 min).
- 3D lambda0/128 on JUPITER (above), the converged 3D point.

Filter: all decks of `series.py` use `numerics.filter = local` (one pass), the filter the 2D, RZ and 3D lambda0/64 runs were made
with; the projection filters (decisions 69, 70) became the default later (decision 71).
