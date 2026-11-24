# Oblique-incidence high harmonics in 3D: VLPL against Smilei

Track O benchmark (docs/ROADMAP.md, Gate O: "oblique-incidence HHG in 3D vs a reference code, same setup"). Parameters
chosen by Alexander (2026-11-01): a p-polarised pulse at 45 deg to x enters through the TFSF plane and hits a foil whose
normal is tilted by 22.5 deg, so the incidence on the target is 22.5 deg and the specular reflection with its harmonics
leaves along -x, the dispersionless axis of RIP-CC. Production case: a0 = 10, 100 n_c. Run so far, at Alexander's
request: the **code check** at 6 n_c, a0 = 2 (S = n / (a0 n_c) = 3), lambda0/32 x lambda0/16 (omega_p dt = 0.48).

| | |
|---|---|
| laser | lambda0 = 0.8 um, a0 = 2 at the focus, 13 fs FWHM of the intensity, waist 2 um (round), p (E in x-y), focus at the foil hit point P |
| target | 6 n_c, 0.6 um along the normal (cos 22.5, sin 22.5, 0), exponential pre-plasma L = 0.08 um cut at 5 L, 8 um long in y, all z; electrons + protons (mobile), 2 x 2 x 1 per cell, shape / interpolation order 2 |
| box | y 28.8 um, z 10 um periodic; VLPL x 11 um with absorbing layers of 2 um, Smilei x 9 um with Silver-Mueller boundaries |
| grid | dx = 25 nm, dy = dz = 50 nm; RIP-CC c dt = dx (1319 steps to 110 fs), Smilei dt = 0.95 Courant (1778 steps to 115 fs) |
| reflected field | VLPL: the outbound components at the inner face of the left absorber (output.xflux_every = 1, tools/xflux.py); left of the TFSF plane there is only the scattered field. Smilei: probe plane 0.5 um inside xmin, the plasma run minus the same run without particles, outbound Ey - Bz, Ez + By. The same spectrum formula in both (S = pi dW/domega). |

Files: `rip.deck` (VLPL, the TFSF plane with exact plane waves, `laser.tfsf_incident = modes`, decision 65),
`rip_grid.deck` (the first version with the incident grid, see below), `smilei/hhg_oblique.py` (Smilei 5.1 namelist;
environment variables select resolution, vacuum and a0), `smilei/compare_smilei.py` (spectrum of the probe),
`smilei/check_vacuum.py`, `smilei/energy_balance.py`, `smilei/run_code.sbatch` (the CPU cluster), `compare.py` (the table below),
`angles.py` (angular distribution per order, both codes; `--runs` for VLPL variants), `rip_2d.deck` (the 2D small-box test).
Smilei is installed on the CPU cluster under $WORK/smilei (README.txt there); runs in $WORK/hhg_oblique.

    vlpl bench/hhg_oblique/rip.deck                    # then python tools/xflux.py out_32/xflux --edge left
    python bench/hhg_oblique/compare.py out_32/xflux/spectrum_left.txt smilei/spectrum_smilei.txt --a0 2 --waist 2 --tau 13

## Results (2026-11-01, the CPU cluster, 2 nodes x 24 cores each code)

Reflected energy per harmonic band [N - 1/2, N + 1/2] over the incident energy W0 = 2.526e4 (code units, analytic
Gaussian beam):

| N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VLPL | 0.666 | 8.28e-2 | 3.06e-2 | 1.23e-2 | 4.13e-3 | 2.48e-3 | 1.21e-3 | 8.0e-4 | 4.5e-4 | 2.7e-4 | 1.7e-4 | 9.4e-5 | 5.5e-5 | 3.1e-5 | 1.9e-5 |
| Smilei | 0.690 | 8.22e-2 | 3.07e-2 | 1.19e-2 | 3.88e-3 | 1.98e-3 | 7.0e-4 | 4.3e-4 | 1.9e-4 | 1.3e-4 | 9.3e-5 | 4.5e-5 | 1.4e-5 | 3.9e-6 | 1.9e-6 |
| RIP / Smilei | 0.97 | 1.01 | 1.00 | 1.03 | 1.07 | 1.25 | 1.72 | 1.87 | 2.40 | 2.11 | 1.80 | 2.08 | 3.9 | 8.0 | 9.9 |

- Reflected fraction 0.805 (VLPL) against 0.825 (Smilei; of its measured incident energy 0.798): 2.4 %. The
  fundamental and the harmonics 2-5 agree within 1-7 %.
- From N = 6 on VLPL is above Smilei by 1.25-2.4 (N = 6-12); from N = 13 Smilei reaches its noise floor (~1.3e-6 per
  band from N = 15 to 20), VLPL still falls smoothly (1.9e-5 at N = 15). At lambda0/32 these orders have 2-5 cells
  per wavelength; the generation at the surface is not resolved in either code, and Smilei's Yee scheme also disperses
  them on the way to its probe (RIP-CC transports them along x without dispersion). Two VLPL runs of the same physics
  with the two TFSF sources (z-uniform beam, the incident grid against the plane waves) differ by up to 1.7 at N = 5-9
  (1.001 at N = 1): the high orders of a single run scatter strongly in this regime. **The orders N >= 6 are not
  converged at this resolution.** This is a code check, and it is complete: where both codes resolve the physics
  (reflection, N = 1-5, energy balance) they agree; the high orders differ where the two schemes differ by design. A
  full-size lambda0/64 pair (~350 core-hours per code) would mostly compare the schemes; convergence is checked on
  small, fast configurations instead (Alexander, 2026-11-02). Then halved dx in both codes ("Convergence along x"
  below): the excess shrinks to 1.1-2.0 (N = 6-12), both codes still rise above N ~ 10.
- Energy balance of VLPL after the pulse: field + kinetic + reflected (Enet_left 2.030e4) + transmitted (429) =
  0.992 W0; the rest left with the particles that leave through x (71 000 of 5.66 M). Smilei: Utot drift 1.25 % of the
  incident energy (its scalar bookkeeping of the injected energy, 2.2 % at the peak of its vacuum run).
- Smilei's oblique injection at its xmin boundary gives 0.82 of the requested amplitude at 45 deg (1.632 at P for a0 = 2,
  0.989 at normal incidence; direction 44.95 deg, waist 2.006 / 2.002 um): the foil run used a0 = 2.4502, the vacuum run
  2 scaled by 1.2251 in the subtraction (exact by linearity). Its pulse reaches P ~1 fs later than VLPL's (not corrected).
- Cost: VLPL 1672 s (22 core-hours; push and deposit 62 %, field step 37 %), Smilei 1496 s (20 core-hours, 48 % of the
  loop "Sync Densities", the load imbalance of the foil).

## Convergence along x (2026-11-02)

Asked for by Alexander ("HHG convergence first"): is VLPL's excess over Smilei from N = 6 up resolved physics or
under-resolution? The code check repeated with dx halved in both codes (lambda0/64, dy = dz = lambda0/16 unchanged; RIP-CC
c dt = dx, omega_p dt = 0.24; Smilei dt = 0.95 Courant), and Smilei also at dx = lambda0/16 (dx = dy = dz). RIP-CC
cannot run dx = dy = dz = lambda0/16 (its stability needs dx^2 (1/dy^2 + 1/dz^2) + dx^2 omega_p^2 / 4 < 1; 2.2 there).
Same physics, decks and analysis as above (`rip.deck` with `grid.nx=880 grid.dx=0.0125um`; `smilei/hhg_oblique.py`
with `HHG_RES=x64 | x16`, foil a0 = 2.4502, vacuum a0 = 2 scaled by 1.2251). Table: `convergence.py`, figure
`convergence.png`.

    python bench/hhg_oblique/convergence.py --a0 2 --waist 2 --tau 13 --nmax 24 --png convergence.png \
        RIP/32=out_32/xflux/spectrum_left.txt RIP/64=out_64/xflux/spectrum_left.txt \
        Smilei/16=spectrum_smilei_x16.txt Smilei/32=spectrum_smilei.txt Smilei/64=spectrum_smilei_x64.txt

Reflected energy per band [N - 1/2, N + 1/2] over W0 (empty: above the Nyquist frequency of the run):

| N | RIP /32 | RIP /64 | Smilei /16 | Smilei /32 | Smilei /64 | RIP/Smilei at /64 |
|---|---|---|---|---|---|---|
| 1 | 0.666 | 0.662 | 0.687 | 0.690 | 0.689 | 0.96 |
| 2 | 8.28e-2 | 7.97e-2 | 8.78e-2 | 8.22e-2 | 8.27e-2 | 0.96 |
| 3 | 3.06e-2 | 3.01e-2 | 3.26e-2 | 3.07e-2 | 3.03e-2 | 0.99 |
| 4 | 1.23e-2 | 1.24e-2 | 7.6e-3 | 1.19e-2 | 1.24e-2 | 0.99 |
| 5 | 4.13e-3 | 4.55e-3 | 2.08e-3 | 3.88e-3 | 4.38e-3 | 1.04 |
| 6 | 2.48e-3 | 3.04e-3 | 3.3e-4 | 1.98e-3 | 2.75e-3 | 1.10 |
| 7 | 1.21e-3 | 1.31e-3 | 1.0e-4 | 7.0e-4 | 1.01e-3 | 1.30 |
| 8 | 8.0e-4 | 9.4e-4 | 2.5e-5 | 4.3e-4 | 7.1e-4 | 1.32 |
| 9 | 4.5e-4 | 4.7e-4 | 7.0e-6 | 1.9e-4 | 3.0e-4 | 1.56 |
| 10 | 2.7e-4 | 3.3e-4 | 3.6e-6 | 1.3e-4 | 2.1e-4 | 1.61 |
| 11 | 1.7e-4 | 2.2e-4 | 2.9e-6 | 9.3e-5 | 1.1e-4 | 1.91 |
| 12 | 9.4e-5 | 1.5e-4 | 2.7e-6 | 4.5e-5 | 7.4e-5 | 1.99 |
| 13 | 5.5e-5 | 1.05e-4 | 2.5e-6 | 1.4e-5 | 4.9e-5 | 2.15 |
| 14 | 3.1e-5 | 7.2e-5 | 2.4e-6 | 3.9e-6 | 3.2e-5 | 2.28 |
| 15 | 1.9e-5 | 5.3e-5 | | 1.9e-6 | 2.3e-5 | 2.34 |
| 16 | | 3.8e-5 | | 1.5e-6 | 1.5e-5 | 2.46 |
| 18 | | 2.0e-5 | | 1.3e-6 | 9.0e-6 | 2.20 |
| 20 | | 1.1e-5 | | 1.3e-6 | 6.5e-6 | 1.72 |
| 24 | | 4.5e-6 | | | 8.2e-6 | 0.54 |

- Reflected fraction (of W0): VLPL 0.805 (/32), 0.799 (/64); Smilei 0.821 (/16), 0.825 (/32), 0.827 (/64). Of its own
  incident energy (inbound through the probe of the vacuum run, 4.4 % above W0 at /64) Smilei reflects 0.794, 0.798,
  0.792: the reflection agrees with VLPL's to 1 % at /64. N = 1-5 agree within 4 % at /64.
- VLPL /32 -> /64: N = 6-11 rise by 1.04-1.29, N = 12-15 by 1.6-2.8.
- Smilei /16 -> /32 -> /64: N = 6-10 rise by 6-17 and then by 1.4-1.7; N = 13-15 by 3.5-12 from /32 to /64. Smilei at
  /64 is within 0.67-1.21 of VLPL at /32 for N = 6-15.
- VLPL's excess over Smilei shrinks with resolution: 1.25-2.4 (N = 6-12) and 4-10 (N >= 13) at /32, 1.1-2.0
  (N = 6-12) and 2.2-2.5 (N = 13-18) at /64. Both codes still rise from /32 to /64 above N ~ 10, Smilei faster; the
  orders above ~10 are not converged in either code at /64.
- Smilei's noise floor: 2.4e-6 per band at /16 (N = 10-14), 1.3e-6 at /32 (N = 15-20), 4.3e-7 at /64 (N = 28-32); at /64
  a plateau of 6-8e-6 at N = 20-25 (not identified; VLPL falls through it at N ~ 22).
- Cost: VLPL /64 3993 s on 6 nodes (160 core-hours; push and deposit 74 %, field step 24 %); Smilei /64 2368 s on
  6 nodes (95 core-hours) + vacuum 16, /16 6.8 + 1.6. A VLPL /128 point (x only) would cost ~4 times /64 (~650
  core-hours at this scaling) and was not run.

## Angular distribution of the harmonics (2026-11-03)

The two codes measure the reflected field on planes at different x, so the angular spectrum is compared, which does not
depend on where it is taken: `angles.py` Fourier transforms the outbound components in t, then for every frequency bin of
the band [N - 1/2, N + 1/2] in y and z, and bins |F|^2 in the direction sines s_y = k_y c / omega (plane of incidence)
and s_z. Smilei's probe has 0.1 um spacing (every second cell): it resolves |s| < s_max = 4 / N, VLPL's records (every
cell, 0.05 um) |s| < 8 / N. At lambda0/64 (`a64*.txt` in $WORK/hhg_oblique/angles):

    python3 bench/hhg_oblique/angles.py --rip conv/out_rip64/xflux --smilei conv/sm_foil64 conv/sm_vac64 1.2251 --out a64
    python3 bench/hhg_oblique/angles.py ... --rip-stride 2 --out a64s2     # VLPL sampled like Smilei's probe

| N | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| RIP / Smilei, band energy | 0.92 | 0.92 | 0.95 | 1.06 | 1.26 | 1.54 | 1.91 | 2.18 | 2.35 |
| VLPL's energy at \|s\| > 4 / N | 0 | 0 | 0 | 0.044 | 0.20 | 0.33 | 0.43 | 0.51 | 0.55 |
| RIP / Smilei inside \|s\| < 4 / N | 0.92 | 0.92 | 0.95 | 1.01 | 1.02 | 1.04 | 1.09 | 1.08 | 1.07 |
| centroid theta_y (deg) RIP, Smilei | +0.1, -0.7 | -0.5, -0.7 | -1.0, -1.2 | -0.6, -0.6 | -0.6, -0.4 | -0.7, -0.4 | -0.7, -0.3 | -0.6, -0.2 | -0.4, 0.0 |
| rms theta_y (deg) RIP, Smilei | 4.1, 6.4 | 4.7, 4.6 | 8.0, 7.3 | 7.2, 6.5 | 7.3, 6.4 | 7.7, 6.6 | 7.7, 6.6 | 7.3, 6.4 | 6.9, 6.1 |
| rms theta_z (deg) RIP, Smilei | 4.2, 4.2 | 5.3, 5.4 | 9.0, 7.8 | 8.2, 6.2 | 8.2, 5.9 | 8.2, 6.0 | 7.9, 6.0 | 7.3, 5.9 | 6.7, 5.6 |
| RIP / Smilei, VLPL sampled like the probe | 0.92 | 0.92 | 0.95 | 1.05 | 1.25 | 1.51 | 1.95 | 2.17 | 2.39 |

- The harmonic beams agree: inside the cone both planes resolve, VLPL / Smilei is 0.92-1.11 for every order up to 16
  (N = 1-4: 0.92-0.95, the 4 % lower reflection; N = 5-16: 0.98-1.11), the centroids agree within 0.8 deg (the specular
  direction, -0.4 .. -1 deg), the rms widths within 1.2 deg in theta_y (N >= 2; N = 1: Smilei 6.4, VLPL 4.1 deg) and
  2.3 deg in theta_z (VLPL slightly wider), the
  normalised distributions overlap to 0.98-0.999.
- VLPL's excess over Smilei from N = 6 on (README tables above) is entirely wide-angle content, |s| between 4 / N and
  8 / N: transverse wavelengths of 2-4 cells (0.1-0.2 um), 30-55 % of VLPL's band energy at N >= 9. Sampled on the
  probe's spacing it folds back into the cone and the excess returns (1.25-2.4): Smilei, on the same sampling, carries
  less of it. At 2-4 cells per transverse wavelength this content is not resolved in either code; in RIP-CC it travels
  along x without dispersion, in Yee it is slowed and dispersed. It is grid-scale content (likely radiated particle noise
  at the surface), not the harmonic beam; a test on a small box (more particles per cell, the current filter, dy = dz
  halved) would tell which. Diagnostics of high harmonics should keep |s| below the resolved cone (the band energy within
  sin theta < k_perp,max / (2 N k0) with a margin).
- The VLPL lambda0/32 run in out_rip32 had the earlier z-uniform beam (rms theta_z 0.05 deg): not comparable here.

### Small-box test of the wide-angle content (2D, 2026-11-03)

The same oblique case in 2D (`rip_2d.deck`: `rip.deck` with `grid.geometry = cartesian2d`, z-uniform beam and foil) at
dx = lambda0/64, dy = lambda0/16 as the 3D lambda0/64 run, 1-4 minutes on one cluster node each, and four variants: 4 x 4
instead of 2 x 2 particles per cell, one pass of the current filter, dy halved, dx halved. `angles.py --runs` gives per
order the band energy and the part at |s| > s_c = 4 / N (transverse wavelengths below 0.2 um, outside the cone of the
Smilei comparison):

    python3 bench/hhg_oblique/angles.py --runs base=base/out/xflux ppc4=ppc4/out/xflux filt=filt/out/xflux \
        dy32=dy32/out/xflux dx128=dx128/out/xflux --out small

| N | 4 | 6 | 8 | 9 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|
| base: part at \|s\| > 4 / N | 0 | 0.08 | 0.29 | 0.41 | 0.47 | 0.56 | 0.64 | 0.65 |
| ppc 4 x 4: band energy / base | 1.04 | 0.96 | 0.81 | 0.71 | 0.66 | 0.59 | 0.54 | 0.49 |
| ppc 4 x 4: part at \|s\| > 4 / N | 0 | 0.03 | 0.14 | 0.22 | 0.26 | 0.36 | 0.44 | 0.50 |
| ppc 4 x 4: energy at \|s\| > 4 / N / base | - | 0.36 | 0.39 | 0.38 | 0.37 | 0.38 | 0.37 | 0.38 |
| ppc 4 x 4: energy inside / base | 1.04 | 1.00 | 0.99 | 0.93 | 0.94 | 0.84 | 0.82 | 0.71 |
| filter: band energy / base | 1.12 | 1.44 | 2.06 | 2.68 | 2.92 | 3.53 | 3.75 | 3.66 |
| dy / 2: band energy / base | 1.08 | 0.84 | 0.76 | 0.74 | 0.74 | 1.41 | 1.87 | 2.56 |
| dx / 2: band energy / base | 1.07 | 1.02 | 0.97 | 0.86 | 0.87 | 0.90 | 0.97 | 1.04 |

- **The wide-angle content is mostly particle noise.** With four times the particles its energy falls to 0.36-0.39 of
  the base at every order from 6 to 16 (white noise: 0.25), while the harmonic beam inside the cone stays within 7 % up to
  N = 10. Above N ~ 10 the inside also falls (0.71-0.84 at N = 12-16): there the noise reaches into the cone.
- VLPL's excess over Smilei at high orders (above) is therefore noise radiated at the surface that RIP-CC carries to the
  plane without dispersion; with 2 x 2 particles per cell it dominates the orders above ~8 at 2-4 cells per transverse
  wavelength. The harmonic beams agree with Smilei (0.92-1.11 inside the cone up to N = 16).
- **One pass of the (then local) current filter raised the high orders** (2-3.7 times at N = 8-16, 11-28 % at N = 4-5;
  outside the cone 52-80 %). Explained and fixed (decision 69): the local filter changed one current component at a time,
  which adds a grid-scale transverse current that radiates. The projection filter (now the default) changes J by RIP-CC's
  gradient only: in the same run the orders are 1.03-1.20 of the unfiltered run (1.10 at N = 5, 1.07 at N = 8, 1.03 at
  N = 10, 1.20 at N = 16; the scatter of a single run), and as expected it does not remove the wide-angle noise, which is
  transverse current (more particles do).
- dy halved (lambda0/32) moves the plane's resolved cone out to 16 / N and adds content beyond the old cone at N >= 12
  (0.81-0.91 of the band outside at N = 14-16); dx halved changes the band energies by 0.86-1.07 up to N = 16, with less
  outside the cone (0.50 instead of 0.65 at N = 16).
- For the production case: more particles per cell (or a noise-reducing loading) and a diagnostic restricted to the
  resolved harmonic cone; the high-order band energies of a single run with 2 x 2 per cell are noise-dominated above
  N ~ 8.

## Azimuthal modes instead of 3D? (2026-11-02)

Alexander's question: RZ with a few azimuthal modes, the axis along the specular direction. Tested with Smilei's
AMcylindrical geometry (`smilei/hhg_oblique_am.py`: the axis through P along x; the laser is the boundary field of the 3D
run, By, Bz(y, z, t) at x = 0, projected on the modes, B_m = (1/pi) int B e^{i m theta} dtheta; the tilted foil as a
particle array with Smilei's AM weights, since its AM density profiles are functions of (x, r) only).
- Mode content of the 45-deg beam (Gaussian, w0 = 2 um, independent of where it enters): energy outside |m| < M
  18 % (M = 8), 5.5e-3 (16), 3e-5 (24), 4e-8 (32). The tilted foil surface (L = 0.08 um, 22.5 deg) at r = 2 um: 1.6e-3 of
  its azimuthal content above m = 16, 2.2e-4 above m = 32. So M ~ 24-32 modes.
- Vacuum run, M = 16: the incident field at P against the 3D run, max |B_z| 1.983 vs 1.995 (0.6 %), time-integrated
  E^2 0.8 % low (the missing 0.5 % of the beam energy): the projection works.
- Cost: Smilei's AM Yee scheme needs c dt ~ dr / (M - 1) (dt 6 times smaller than 3D at M = 16, 10983 steps), and the
  deposit runs over all modes. The foil run at M = 16 (1.1 M particles per species) ran at 2.9 s per step on 48 cores
  (430 core-hours projected; ~70 with perfect load balance), against 20 core-hours for the 3D /32 run; cancelled. With
  M = 24-32 it grows as M^2.
- VLPL's RZ (iRIP) has no mode-dependent time step (c dt = dz) and would pay only the linear cost in M, but has neither
  an oblique laser source (the RZ laser enters through the z edge along the axis) nor a density depending on the azimuth
  (`density_function = f(z, r)`). For this geometry (45 deg beam, 22.5 deg foil) the modes are not cheaper than 3D;
  they pay off near normal incidence (a few modes).

## Why the TFSF source with exact plane waves (decision 65)

The first VLPL run (`rip_grid.deck`, the incident grid of O1) needed the whole pulse between the left absorber and the
plane at t = 0: at 45 deg ~17 um in x and y (checked with the start warnings: below 1e-4 only with 16 um of room). Box
29.5 x 28.5 x 10 um, twice the field work for the incident grid: 7473 s on 48 cores (100 core-hours, field step 72 %),
4.5 times Smilei's. Alexander asked to reconsider decision 59; with the plane waves the plane sits 0.5 um after the left
layer and the pulse starts outside the box: the same physics in a box of 11 um, 22 core-hours. The grid run had a beam
uniform in z (no waist_z, unlike Smilei's round beam); against the plane-wave run with the same z-uniform beam the
reflected energy agrees to 0.9 % (7.62e4 / 7.69e4) and N = 1 to 0.1 %.

## Next
- Convergence on small, fast configurations (minutes to a few core-hours), not full-size pairs; code development
  first, the production case (100 n_c, a0 = 10, at its own resolution) after it (Alexander, 2026-11-02). The x series
  above (/32, /64): the orders above ~10 are not converged in either code; a VLPL /128 point or a dy, dz series is
  Alexander's call.
- Angular distribution and the small-box test: done (above). The wide-angle content of the high orders is mostly particle
  noise. The filter's rise of the high orders is fixed (projection filter, decision 69). Open: the particles per cell of
  the production case.
