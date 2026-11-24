# Noise heating in RZ (2026-11-06, decision 79)

The late energy excess of the RZ CHF runs (bench/chf/README.md) is noise heating of the hot electrons. This benchmark
isolates it: a uniform hot electron column (n = n_c, u_th = 1 per component, radius 1.6 um, fixed ions), no laser, periodic
z, the conducting wall at 3.6 um (the sheath settles, few electrons reach the wall), the CHF cells dz = lambda0/128, dr =
lambda0/40, ppc 1 x 2 x 8, shape 2, one projection filter pass; 2D Cartesian: a slab of the same width with the same cells
and particles per cell. G = (W_total + E_part_out + Enet + E_abs_r) / W_total(0) - 1, its slope from 15 to 30 fs per 10 fs
(the first 15 fs: the sheath forms and the fastest electrons reach the wall). Runs in $WORK/noiseheat on the CPU cluster
(epyc, about 6 min each).

    python3 nh_make.py <dir> key=value ...     # geom=rz|2d, solver, closure, shape, fp, filt, ppc=z,r,theta, modes, dz, dr, ts, sw, gm
    python3 nh_ana.py <directory of runs>

## What it depends on (no transverse shape)

| case | G per 10 fs |
|---|---|
| 2D, 16 particles per cell / 2 x | 1.0e-4 / 5.6e-5 |
| RZ 1 / 2 / 3 / 4 modes (RIP-CC) | 0.8e-4 / 3.9e-4 / 6.9e-4 / 1.02e-3 |
| RZ 8 modes (iRIP) | 1.83e-3 |
| RZ 4 modes, ppc_theta 4 / 16 / 28 / 56 | 2.1e-3 / 4.8e-4 / 2.2e-4 / 4e-5 |
| RZ shape 1 | 5.2e-3 |
| RZ 0 / 2 projection passes, local filter | 1.1e-3 / 1.1e-3 / 2.4e-4 |
| RZ iRIP / iRIP+O5 | 9.0e-4 / 6.5e-4 |
| RZ dr = 0.025 / 0.032 um (0.60 / 0.36 of the RIP-CC limit) | 1.1e-3 / 1.4e-3 |
| 2D at 0.9 / 0.5 of its RIP-CC limit (dy = 0.0066 / 0.0088 um) | 5.2e-5 / 5.1e-5 |
| matched gather (rz_gather = matched) | 1.0e-3 (-2 %) |
| ppc_theta 56 only for r < 0.4 / 0.8 um | 1.01e-3 / 0.88e-3 |

- RZ with 1 mode heats like 2D; the rate grows with the modes, about with 2M - 1: a macro-particle is a truncated Fourier
  series in phi, and ppc_theta = 8 particles sample 2M - 1 = 7 azimuthal components (CHF: ppc_theta 10 with 4 modes).
- Not the cause: the distance to the RIP-CC stability limit (0.93 at 4 modes, dr = lambda0/40; further away it heats
  more, through fewer particles per volume), the mismatch of the Ruyten deposit and the plain gather, the filter passes.
- The local filter damps it (it smooths the m >= 1 noise field, -25..45 % of its energy; Gauss law at round-off), the
  projection filter does not (its change of J is a gradient).
- More particles near the axis only do not help: the heating is in the whole column, and hot electrons cross the axis
  region within a few fs (a boost at loading mixes away).

## The transverse Gaussian shape (rz_transverse_shape = gaussian, an option; the default is none with a ppc_theta warning)

| case (4 modes unless noted) | none | Gaussian 0.5 dr | 1 dr | 2 dr |
|---|---|---|---|---|
| RIP-CC | 1.02e-3 | 3.7e-4 | -2e-5 | -1.0e-4 |
| iRIP / iRIP+O5 | 9.0e-4 / 6.5e-4 | | -2e-5 / -3e-5 | |
| iRIP, 8 modes | 1.83e-3 | | 6e-5 | |
| 2 modes / 2 x ppc_theta | 3.9e-4 / 4.8e-4 | | -7e-5 / -6e-5 | |

Gauss law at round-off (7e-14), +30 % run time. Its price: deposit and gather at k_perp are scaled by exp(-k_perp^2 a^2/2),
at dr = lambda0/40:

| harmonic, angle | k_perp dr | f, a = 0.5 dr | f, a = 1 dr |
|---|---|---|---|
| 5, 30 deg | 0.39 | 0.98 | 0.93 |
| 10, 15 deg | 0.41 | 0.98 | 0.92 |
| 10, 30 deg | 0.79 | 0.93 | 0.73 |
| 20, 30 deg | 1.57 | 0.73 | 0.29 |

## The width against the CHF focus (lambda0/64, 2026-11-06)

The CHF deck of bench/chf at dz = lambda0/64, dr = 0.0425 um = lambda0/19 (twice the production dr), 4 modes, a0 = 10, TFSF,
projection filter, 60 fs ($WORK/noiseheat/chf64, epyc, 16 threads each; harmonic_focus peaks at phi = 0, the
fluences in the code's units):

| run | low peak / a0^2 | z (um) | fwhm_z (um) | low fluence | high peak | high fluence | balance 37 / 48 / 59 fs | time |
|---|---|---|---|---|---|---|---|---|
| none, ppc_theta 10 | 15.10 | 2.631 | 0.81 | 25.6 | 1.05 | 5.44 | 7.1e-3 / 1.9e-2 / 3.5e-2 | 276 s |
| none, ppc_theta 30 (reference) | 15.04 | 2.641 | 0.84 | 26.5 | 1.11 | 2.29 | 5.8e-3 / 1.2e-2 / 2.3e-2 | 519 s |
| gaussian 0.5 dr | 13.31 | 2.619 | 0.96 | 26.0 | 1.09 | 1.15 | 5.1e-3 / 9.9e-3 / 2.1e-2 | 326 s |
| gaussian 1 dr | 9.11 | 2.741 | 1.08 | 20.5 | 0.51 | 0.40 | 3.9e-3 / 5.8e-3 / 6.4e-3 | 325 s |

- More particles (ppc_theta 30) leave the low-band focus unchanged (-0.4 %) and cut the excess at 59 fs by a third; the
  high-band fluence falls from 5.4 to 2.3: at ppc_theta 10 most of the high-band fluence at its maximum is particle noise.
- The Gaussian damps the harmonics themselves: the low-band peak -11 % (0.5 dr) and -39 % (1 dr) against the reference,
  the focus longer (fwhm_z 0.96, 1.08 um against 0.84). At this dr harmonic 7 at 30 deg has k_perp dr = 0.97: the
  coupling exp(-k_perp^2 a^2/2) is 0.89 (0.5 dr) and 0.62 (1 dr) on each side.
- So for CHF (harmonics at large k_perp) the Gaussian of 1 dr is not acceptable at this resolution; at the production dr =
  lambda0/40 the same harmonic has half the k_perp dr (factor 0.97 at 0.5 dr, 0.89 at 1 dr). The accurate choice for CHF
  is `none` with ppc_theta about 2 (2M - 1) (1.9 times the run time here).

## The rest of the CHF excess: an axis instability of dense plasma (2026-11-06, decision 80)

The excess of the lambda0/64 CHF deck at 59 fs against ppc_theta (no transverse shape): 3.45e-2 (10), 2.26e-2 (30), 1.90e-2
(60) = a + b / ppc_theta with a floor a = 1.6e-2 that particle noise does not explain. It builds after the interaction
(ppc_theta 60: 5.7e-3 at 37 fs, 1.05e-2 at 48 fs, 1.9e-2 at 59 fs); in the balance it shows as field energy left in the
box (W_in at 59 fs 0.015 against 0.008 with the Gaussian) and more energy out through the rear edge and the absorbers.
Scripts: chf64_split.py (balance terms against time), chf64_where.py (electron kinetic energy inside / outside the target
from openPMD dumps), chf64_ana.py; runs in $WORK/noiseheat/chf64.

| ppc_theta 30, one change at a time | excess 37 / 48 / 59 fs |
|---|---|
| RIP-CC (base) | 5.8e-3 / 1.20e-2 / 2.26e-2 |
| iRIP / iRIP+O5 | 5.6e-3 / 1.32e-2 / 2.84e-2 ; 3.6e-3 / 8.9e-3 / 1.43e-2 |
| no current filter | 7.7e-3 / 1.67e-2 / 3.24e-2 |
| right z layer 3 um further from the target | 6.0e-3 / 1.22e-2 / 2.17e-2 |
| dr doubled (0.085 um), ppc_theta 30 / 60 | 6.9e-2 / 0.31 / 0.67 ; 6.7e-2 / 0.27 / 0.55 |
| rz_gather = matched | 4.5e-3 / 8.6e-3 / 1.22e-2 |
| rz_gather = energy | 1.9e-3 / 5.1e-3 / 7.0e-3 |
| Gaussian 1 dr (ppc_theta 60) | 4.2e-3 / 5.8e-3 / 6.2e-3 |
| 2D lambda0/64 (dy = 0.01375 um), 4 / 16 / 64 ppc | 4.1e-3 / 8.2e-3 / 1.09e-2 ; 3.5e-3 / 5.7e-3 / 7.8e-3 ; 3.4e-3 / 5.4e-3 / 7.8e-3 |

Not the cause: the z layers, the radial absorber (bench/chf: a wall instead changes nothing), numerical Cherenkov of
relativistic electrons (a neutral pair beam at gamma 5 drifting along z in vacuum keeps its energy to 1e-3 in RZ and 2D,
all solvers). The floor grows steeply with dr (dr doubled: unstable).

**Reproducer**: a uniform 40 n_c plasma at 1 keV, fixed ions, no laser, dz = 0.0125 um, dr = 0.0425 um (k_p dr = 2.1), 4
modes, periodic z (`nh_make.py n=40 uth=0.044 dz=0.0125 nz=64 dr=0.0425`). Physically nothing happens. W_kin / W_kin(0):

| case | 10 fs | 20 fs | 29 fs |
|---|---|---|---|
| 2D, dy = 0.0425 / 0.01375 um | 0.99 / 1.00 | 0.99 / 1.00 | 1.00 / 1.01 |
| RZ RIP-CC / iRIP (plain gather) | 1.8 / 2.4 | 9.7 / 12 | 18 / 21 |
| RZ, ppc_theta 30 | 1.2 | 8.3 | 16 |
| RZ, dr = 0.085 um (iRIP) | 3.5 | 61 | 142 |
| RZ, plasma only at r > 0.5 um | 0.93 | 0.95 (19 fs) | |
| RZ, 10 n_c (k_p dr = 1.05) | 0.95 | 0.95 (19 fs) | |
| RZ, 1 mode / 2 modes | 1.16 / 2.6 | 2.3 / 10 (19 fs) | |
| RZ, Gaussian 1 dr | 0.94 | 0.95 | 0.96 |
| RZ, rz_gather = matched | 0.94 | 1.26 | 2.1 |
| RZ, rz_gather = energy | 0.92 | 1.19 | 2.4 |

- It is the axis: without plasma next to the axis it is stable; E_r on the first half node grows first and stays ten
  times the bulk value; mode 0 alone grows too; it needs a dense plasma (k_p dr > ~1.5), cold or warm, and is
  independent of the particle number and the time step (dz / 2: still unstable).
- The plain gather (plain splines in r) differs from the deposit (the density-conserving Ruyten maps, decision 53) next to
  the axis: a particle feels a different shape of its own charge than it deposits. The matched gather (the deposit's map)
  or the energy-consistent gather (E_r with g_h = dn_h xi'(r) S_{O-1}(xi - h - 1/2), the work of the gathered E_r = that of
  the deposited J_r) remove most of it (W_kin at 29 fs 2.1 / 2.4 instead of 18); a slower growth remains after 20 fs (not
  explained yet; z and time centring next to the axis are the next suspects).
- The matched gather returns a uniform transverse field exactly (test R4, 4e-16). The energy gather does not next to the
  axis: its E_r weights sum to dr xi'(r), and the Ruyten map has xi' != 1/dr there (shape 2: E_x / E0 - 1 = +1.46 at r =
  0.05 dr, -0.62 at 0.8 dr, -0.05 at 2 dr). Energy consistency and an exact uniform field conflict as long as the density
  correction is a map of the particle position.
- CHF (lambda0/64, ppc_theta 30): matched halves the floor (2.26e-2 -> 1.22e-2) with the focus unchanged (low-band peak
  -1.4 %); energy brings it to the 2D level (7.0e-3) and the low band +3 %, high-band fluence 2.29 -> 1.57.
- Decided (Alexander, 2026-11-06): `rz_gather = matched` is the default (no cost for smooth fields); the remaining axis
  growth is the next code question.

### The residual axis growth (2026-11-06): the radial resolution of the skin depth at the axis

With the matched gather the reproducer still grows after 20 fs (2.1 at 29 fs). One change at a time (matched gather,
W_kin / W_kin(0) at 29 fs, e-folding time 20 -> 29 fs):

| case | 29 fs | e-fold |
|---|---|---|
| base (40 n_c, dr = 0.0425 um, k_p dr = 2.1, RIP-CC) | 2.09 | 18 fs |
| plasma only at r > 0.5 um | 0.98 | stable |
| 20 n_c (k_p dr = 1.5) | 0.98 | stable |
| dr = 0.02125 um (k_p dr = 1.05, iRIP) | 1.05 | stable |
| dz and dt halved | 2.53 | 13 fs |
| ppc_theta 40 | 2.36 | 15 fs |
| 1 mode / 2 modes | 2.42 / 4.25 | 23 / 9 fs |
| no filter / 2 passes / local filter | 8.3 / 1.79 / 2.53 | 11 / 24 / 13 fs |
| iRIP / iRIP+O5 | 3.63 / 1.65 | 10 / 20 fs |
| shape 1 | 4.89 | 17 fs |
| cold plasma (uth 0.005) | 8.3 | 9 fs |
| iRIP, dr refined at the axis only (0.02125 um for r < 0.3 um, x 1.1 per cell to 0.0425 um; 94 instead of 85 cells) | 1.02 | stable |

- Not the z or time centring (dz and dt halved: unchanged), not the particle number. It is the radial resolution of the
  skin depth in the cells next to the axis: stable for k_p dr <= 1.5 there. A radial grid refined only near the axis
  removes it at +10 % cells. A run warns when a mobile species has k_p dr > 1.5 at the axis.
- CHF: the production grid (lambda0/128, dr = lambda0/40) has k_p dr = 1.0 for the 40 n_c target; the lambda0/64 deck 2.1.
  On the lambda0/64 deck (iRIP, matched, ppc_theta 30) the axis-refined grid gives 1.08e-2 at 59 fs against 1.22e-2: with
  the matched gather the remaining excess is about the noise (b / 30 = 6e-3) plus a floor of the 2D size. The axis
  refinement raises the low-band peak by 18 % (13.5 -> 15.9 a0^2): the focus on the axis needs the near-axis resolution too.
