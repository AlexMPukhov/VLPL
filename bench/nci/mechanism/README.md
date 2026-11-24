# Mechanism of the numerical Cherenkov instability in VLPL, and its cure (2026-11-17, decision 108)

Question (Alexander, 2026-11-17): "is there a hope to remove the NCI in VLPL or do we have to implement the galilean moving grid?",
then "go on with NCI research in VLPL". Method: short periodic streaming-plasma boxes (minutes each), fields dumped every step,
the unstable spectrum in (k_x, k_perp) and its frequency against the scheme's dispersion relation. Physics decisions (defaults,
the recommended setting) are Alexander's; everything below is provisional.

**Answer.** The NCI of VLPL is the classical spatial-alias Cherenkov resonance: the EM branch of the RIP / iRIP solver crosses
the first alias (m = +-1) of the streaming plasma's current. It is not a force mismatch of E and v x B (no growth off the
resonance curves), and the main (m = 0) resonance does not occur. A filter along the streaming axis removes it wherever the
crossing lies at large |k_x|; that is the case for gamma_b >= 10 on LWFA grids. For gamma_b ~ 5 at low density the crossing
reaches down to k_x D ~ pi/4, where no filter that is flat at the laser wavenumber can act fully; the remaining growth there is
slow (0.0045 omega_p). No Galilean / comoving grid is needed for the cases measured.

## 1. Setup (`scan.sh`, `nci_spec.py`, `run_rz.sh`, `rz_spec.py`)
Neutral e-p plasma streaming at -beta c through a periodic box (../boost_stability/lin_c2.deck): the grid of a boosted LWFA,
lab lambda0 = 1 um, dx_lab = lambda0 / 32 (dx' = gamma (1 + beta) dx_lab), n = gamma n0 (n0 = 0.004 n_c; 0.002 at gamma 20),
128 x 32 cells, 2 x 2 particles per cell, iRIP (RIP-CC where its Courant limit allows), shape 2 unless noted. Seed: a thermal
spread **isotropic in the plasma rest frame**, u_th = 1e-3 (boosted: u_th,x = gamma 1e-3, u_th,perp = 1e-3). The seed of
bench/nci/boost_stability (u_th = 1e-2 in every boosted component) is 1 / gamma^2 colder along x than across in the rest frame;
with it the low-k modes on the beam line (omega = beta c k_x) grow as well (a Weibel-type instability of that anisotropic
seed, not numerical); with the isotropic seed they do not (section 2).

`nci_spec.py` Fourier transforms all six field components over (x, y) at every step, takes the amplitude growth rate of every
mode between two steps of the exponential phase, its frequency from the phase advance per step, and overlays the resonance
curves of the vacuum (+ omega_p) dispersion of the scheme with the beam aliases. Its summary line names the alias closest to the
measured frequency of the strongest mode. RZ: `rz_spec.py` does the same along z (all radial rows, modes, components).

Figures (on the CPU cluster `$WORK/nci/s5*/*.png`, `$WORK/nci/rz/spec_f0.png`):
mode energy, growth rate, frequency in (k_x D, k_y h), and (right) the frequency offset of the strongest modes from the beam
line, (omega - |beta| c k) tau in units of the alias spacing 2 pi (1 - |beta|), with the EM branch for k_y h = 0 .. pi.

## 2. The mechanism
With c tau = D the beam's current sampled on the grid has the frequencies omega tau = beta (k_x D + 2 pi m) (mod 2 pi). The alias
m = -+1 (sign of beta) lies **above** the light line omega = |k_x| c by (1 - |beta|)(2 pi - |k_x| D) ~ (2 pi - |k_x| D) / (2 gamma^2):
the slip of 1 / (2 gamma^2) cell per step is exactly what puts it next to the light line. The EM branch of the scheme,
RIP-CC sin^2(omega tau / 2) = sin^2(k_x D / 2) + (D^2 Lambda / 4) cos^2(k_x D / 2), iRIP tan^2(omega tau / 2) = tan^2(k_x D / 2) + D^2 Lambda / 4
(Lambda = (2/h)^2 sin^2(k_y h / 2) + plasma), lies above the light line by ~ (D^2 Lambda / 4) cot(k_x D / 2) (iRIP: x cos^2(k_x D / 2)),
which vanishes at Nyquist for every k_perp. The two meet on a curve that runs from Nyquist (large k_perp) to smaller k_x at
small k_perp (the magenta / cyan curves in the figures); the lowest k_x is set by the plasma term and falls with gamma^2 (omega_p tau)^2
~ gamma^4 n0 (k0 dx_lab)^2, i.e. lower gamma and lower density move the resonance to smaller k_x. The m = 0 line lies below the EM
branch everywhere: no main Cherenkov resonance.

Evidence (2D, no filter, the strongest mode at the end of the exponential phase; omega tau / pi measured vs the EM branch at that
(k_x, k_y) and the nearest alias):

| run | k_x D / pi, k_y h / pi | omega tau / pi | EM branch | alias m = +-1 | m = 0 line | rate / omega_p (all modes) |
|---|---|---|---|---|---|---|
| iRIP gamma 10, h 0.25 um (c dt / h 2.5) | 0.906, 0.625 | 0.9140 | 0.9104 | 0.9117 | 0.9017 | 0.058 |
| iRIP gamma 10, h 0.25, shape 1 | 0.906, 0.625 | 0.9101 | 0.9104 | 0.9117 | 0.9017 | 0.068 |
| iRIP gamma 10, h 1 um (0.62) | 0.781, 0.750 | 0.7881 | 0.7856 | 0.7874 | 0.7773 | 0.020 |
| RIP-CC gamma 10, h 1 um | 0.828, 0.312 | 0.8359 | 0.8373 | 0.8340 | 0.8240 | 0.011 |
| iRIP gamma 5, h 0.25 (1.25) | 0.703, 0.562 | 0.7304 | 0.7290 | 0.7293 | 0.6889 | 0.048 |
| RZ LWFA grid: RIP-CC gamma 6, dz' 0.30163 um, dr 0.405 um | 0.875 (kz) | 0.8914 | 0.875 (k_perp = 0) | 0.8907 | 0.8628 | 0.052 |

The measured frequency equals the m = +-1 alias to 0.0003 - 0.002 pi and lies 0.01 - 0.04 pi off the m = 0 line. At gamma 10, 73 - 93 %
of the field energy is at |k_x| D > 0.75 pi; at gamma 5, 93 % at 0.25 - 0.75 pi (the crossing at 0.70 pi); RZ: 94 % at |kz dz| > 0.75 pi. Further (warm anisotropic seed, `list_mechanism.txt`): the rate
scales with omega_p (n0 = 0.002 / 0.004 / 0.008 n_c: 0.0365 / 0.0367 / 0.0356 omega_p at gamma 10), weakly with dx_lab (lambda / 64:
0.031, lambda / 32: 0.037), grows with gamma from 2 (none within 250 / omega_p) to 5 - 10, and drops at h > c dt (iRIP h = 1 um:
0.015 vs 0.037). With the filter the remaining growth stays on the same curve, at smaller k_x where the filter is weaker:
gamma 10, two passes: k_x D = 0.56 pi (alias 0.5697, measured 0.5695 pi); gamma 5, 16 passes + compensator: 0.25 pi (0.2854 / 0.2852 pi).

Candidates discussed before the runs: (1) "at k_x D -> pi all k_perp travel at c and resonate with the beam": right about where,
but the partner is the alias m = +-1, not the beam itself (the beam line is below the EM branch; the measured frequency is above it);
(2) residual of E + v x B (Godfrey-Vay non-resonant NCI): not seen. Off the alias curves nothing grows: with an x-only filter of
8 passes (response 1 at k_x D < 0.1 pi) the low-k modes stay at the noise level (5e-8 for 600 / omega_p, gamma 10), and the RZ
gathers plain and matched agree (decision 104). A matched E / B gather is therefore not needed.

## 3. Cures: filters along the streaming axis (`list_cures.txt`, decision 108 options)
New options (off by default; provisional): `numerics.filter_compensate = true` adds, after the N binomial passes, one step
J <- J + c (L J - J) with c = -N along x (z in RZ): the response T^N ((N + 1) - N T), T = cos^2(k_x D / 2), flat to O(k^4) at the laser,
zero at Nyquist (Lee and Vay). `numerics.filter_axes = x` makes the passes act along the streaming axis only (no transverse
smoothing). Both are polynomials in the per-axis operators of decision 91 (d_a P(S_a) = P(T_a) d_a), so the Gauss law stays exact:
tests S9b and R25 (Cartesian 4 - 7e-16, transfer function 3 - 4e-15; RZ to round-off); the runs below keep max|dGauss| <= 4e-15.

W_field / K_e at omega_p t = 600 (2D, seed isotropic in the rest frame; the noise level at 10 / omega_p is 1e-8 - 6e-8; the
residence of the plasma in the boosted box is omega_p t = k_p L_lab, e.g. 533 for the boosted RZ LWFA window):

| filter | loss at k0 (lambda / 32) | iRIP gamma 5 h 0.25 | iRIP gamma 10 h 0.25 | gamma 10 h 0.25 shape 1 | iRIP gamma 10 h 1 | RIP-CC gamma 10 h 1 | iRIP gamma 20 h 0.25 |
|---|---|---|---|---|---|---|---|
| none | 0 | 1.2e-2 | 4.8e-3 | - | - | - | - |
| 2 passes (current advice) | 1.9 % | 2.1e-4 | 3.2e-7 | 1.7e-4 | 2.3e-7 | 9.9e-8 | 3.2e-8 |
| 2 passes + compensator | 0.03 % | - | 1.7e-5 | - | - | - | - |
| 4 passes | 3.8 % | 4.0e-6 | - | - | - | - | - |
| 4 passes + compensator | 0.09 % | 2.9e-5 | 8.8e-8 | 3.9e-6 | 8.9e-8 | 7.3e-8 | 2.1e-8 |
| 4 passes, x only | 3.8 % | 1.0e-5 | 6.5e-8 | - | - | - | - |
| 4 passes + compensator, x only | 0.09 % | 1.3e-4 | 1.6e-7 | 1.3e-5 | 2.4e-7 | 1.4e-7 | 9.4e-8 |
| 8 passes, x only | 7.4 % | 1.8e-6 | - | - | - | - | - |
| 8 passes + compensator | 0.32 % | 1.3e-6 | - | - | - | - | - |
| **8 passes + compensator, x only** | **0.32 %** | **2.9e-6** | **5.4e-8** | - | - | - | - |
| 16 passes + compensator, x only | 1.2 % | 1.5e-6 | 4.0e-8 | - | - | - | - |

RZ, the boosted RZ LWFA grid (RIP-CC gamma 6, dz' 0.30163 um, dr 0.405 um, n0 3.118936e-4 n_c, 2 modes, isotropic seed u_th 1e-3), W / K_e at
omega_p t = 300 | 533 | 600 (`run_rz.sh`, `wk_rz.py`; the noise level at 10 / omega_p 0.5 - 2.5e-8; "z" = filter_axes = x, the z axis):

| filter | 300 | 533 | 600 |
|---|---|---|---|
| none | 1.7e-3 | 3.2e-3 | 3.8e-3 |
| 1 pass | 5.8e-6 | 2.4e-4 | 2.9e-4 |
| 2 passes (the f2_str run) | 3.4e-7 | 3.0e-5 | 7.6e-5 |
| 2 passes + compensator | 1.9e-6 | 1.8e-4 | 2.9e-4 |
| 2 passes, z only | 7.7e-7 | 7.2e-5 | 1.5e-4 |
| 4 passes | 2.5e-8 | 4.9e-7 | 1.2e-6 |
| 4 passes + compensator | 1.9e-7 | 1.5e-5 | 4.2e-5 |
| 4 passes, z only | 7.8e-8 | 2.1e-6 | 5.6e-6 |
| 4 passes + compensator, z only | 6.8e-7 | 7.4e-5 | 1.5e-4 |
| **8 passes + compensator, z only** | **3.8e-8** | **3.7e-7** | **8.1e-7** |

On this grid (low density, gamma 6) a residual grows at 0.005 - 0.01 omega_p (amplitude) in every filtered case; 8 + compensator along z and 4 plain
passes on both axes are 80 x lower than the two passes of the running boosted RZ LWFA check at its residence time 533 / omega_p.

Laser-plasma coupling (`run_em.sh`, `em_coupling.py`): a plasma at rest (0.1 n_c), the EM mode at k_x D = 2 pi / 32 from the thermal
noise; RIP is dispersion-free along x, so sin^2(omega tau / 2) - sin^2(k_x D / 2) is the plasma term alone, R(k0) (omega_p tau / 2)^2.
Relative to the compensated runs (which agree with each other to 0.1 %): two passes 0.981 (T^2 / T^2 (3 - 2T) = 0.981), four
passes 0.961 (0.963), one pass 0.995 (0.990); reproducible to 0.1 - 0.3 % between a 3000- and a 12000-step series. The column
"loss at k0" is the transfer function, 1 - T^N or 1 - T^N ((N + 1) - N T). Open: the absolute plasma term of the filtered runs is
~1 % below (omega_p tau / 2)^2 x the shape factor sinc^6(k D / 2); the unfiltered reference scatters by +-2 % (its noise is not filtered).

Findings:
- The compensator alone makes things worse at equal passes (gamma 10: 2 passes 3.2e-7, 2 + compensator 1.7e-5; RZ 4 passes 4.9e-7 at
  533 / omega_p, 4 + compensator 1.5e-5): it lifts the response at k_x D = 0.4 - 0.8 pi, where the resonance curve lies at small k_perp.
  Lee and Vay's 4 + compensator works for PSATD, whose unstable band is at Nyquist.
- Transverse passes help little: at equal x passes the all-axes filter is 1.5 - 4 x lower (gamma 10: 2 passes 3.2e-7 vs x only
  5.2e-7; gamma 5: 4 passes 4.0e-6 vs 1.0e-5, 8 + compensator 1.3e-6 vs 2.9e-6); the response in k_x decides.
- gamma >= 10: 4 plain x passes or 8 + compensator keep the plasma at the noise level for 600 / omega_p; 4 + compensator leaves
  x4 - x8 above it.
- gamma 5 at h < c dt is the hard case: the resonance reaches k_x D = 0.25 pi; every filter that keeps the laser leaves a growth
  of 0.0045 omega_p there (x30 - x50 over 600 / omega_p, 1.3 - 3e-6 of the drift energy at the end); two passes give 2.1e-4.
- gamma 3 is not stable without a filter on long times: 1.2e-3 at 600 / omega_p (3.2e-4 at 300), two passes 4.6e-6. The summary
  warning of decision 104 starts at gamma > 3; gamma 3 should warn as well.
- Cost: 2D (128 x 32, 2 x 2 ppc, 4 threads) push + deposit + filter 32.9 s without, +6.3 s with 4 passes (all axes), +7.0 s with
  8 + compensator x only, +19 s with 16 + compensator (~15 % of a step for 8 + compensator); RZ (128 x 16, 3 x 2 x 8 ppc): 2 and 4
  passes within 1 % (98 % of the time is the particle push).

## 4. Proposal (Alexander decides)
- For boosted runs: shape 2 and the filter along the streaming axis with 8 passes and the compensator
  (`filter_passes = 8`, `filter_compensate = true`, `filter_axes = x`): at the laser 0.3 % less plasma response than without a
  filter (two plain passes: 1.9 %, and they smooth the laser transversely); the NCI at the noise level for gamma >= 10, and
  x70 lower than two passes at gamma 5; on the boosted RZ LWFA grid (RZ) x80 lower than two passes at 533 / omega_p (3.7e-7 vs 3.0e-5).
  Four plain passes on all axes are as good in RZ (4.9e-7) but cost 3.8 % at k0. The present advice (shape 2, two passes) stays
  valid where 1.9 % at k0 and the growth to 3e-5 over 533 / omega_p are acceptable.
- Prefer gamma_b where the resonance sits at large k_x (gamma_b >= ~8 - 10 on lambda / 32 grids at 0.004 n_c); at gamma_b ~ 5 and
  below with h < c dt a slow residual (0.0045 omega_p) remains. This pulls against the accuracy window of decision 107
  (gamma_b <= gamma_g / 4): to be weighed per case.
- Not needed on this evidence: a matched E / B gather (no force mismatch growth) and a Galilean / comoving grid. A comoving grid would
  remove the alias resonance only by changing what the plasma samples per step; for RIP (c dt = dx, x-dispersionless) no such scheme
  exists yet, and the remaining gamma ~ 5 residual is small.
- Not changed: the defaults (filter_passes = 0, filter = binomial on all axes, no compensator) and the decision-104 warning.

## 5. Neutral boosted injection (decision 108)
Alexander: "the boosted injection has to be absolutely neutral. This is a precondition." BoostInject creates the entering plasma with
no creation current, so any net charge it creates stays at the injection plane and the Gauss law there drifts (the boosted RZ LWFA test decks: electron density 3.11893e-4 instead of 3.118936e-4, 2e-6 relative: residual 1.2e-3 of the electron
charge density and field energy 2e-3 after 2800 steps without a laser; exact density: 4e-13 and 1e-23).
- Config::check (boost_check_neutral, src/boost.cpp) refuses a boosted plasma unless all its species (at rest in the lab, nonzero
  density) share ppc, bounds and slab, and sum q Z0 n f(x) = 0 at every point of the lab path to 1e-12 (Z0: the initial charge state of
  an ionizable species). The loaders place particles on deterministic lattice points with weight density x profile, so the injection is
  then neutral at every particle position. The error names the mismatch and the neutral electron density (the RZ LWFA deck: 3.118936e-4).
  A deliberately non-neutral plasma, or electrons alone, is refused: in the boosted frame the missing ions would also miss their current.
  The alternative, a creation current (a non-neutral plasma entering as through an open boundary), is not implemented.
- The injection plane is 3 + R cells behind the right edge, R = filter reach (decision 91's frozen columns of the lab window): with
  two passes and the plane 3 cells behind, warm electrons left up to 3e-9 relative (decision 104's open item).
- Tests B7 (refusals, messages) and B8 (2D, 3D, RZ; e + He+ + N3+ ionizable; 20 um ramp; warm electrons; gamma 10; filter 0 - 2):
  max|dGauss| 3e-16 .. 5e-15. The NCI runs of sections 1 - 4 are periodic boxes without injection (equal e / p lattices).

## 6. Less filtering (decision 109; Alexander: "too much filtering will destroy physics at some point. any other solution?")
Runs: `list_flat.txt` (scan.sh, TW=600), `run_rz_flat.sh`, `force.py` (fields every step, gamma 10, no filter).
W_field / K_e at 533 / omega_p (noise level ~1e-7):

| cure | removed at k0 | 2D gamma 10 | 2D gamma 5 | RZ LWFA |
|---|---|---|---|---|
| none | 0 | 3.5e-3 | 1.2e-2 | 3.2e-3 |
| 2 binomial passes | 1.9e-2 | 1.9e-7 | 1.6e-4 | 3.0e-5 |
| 8 passes + compensator, x (section 4) | 3.2e-3 | 5.4e-8 | 1.3e-6 | 3.7e-7 |
| flat N=6 L=4 (`filter_passes=6 filter_flat=4 filter_axes=x`) | 1.0e-6 | 1.8e-7 | 3.2e-4 | 2.2e-4 |
| flat N=8 L=4 | 2.7e-6 | 1.0e-7 | 4.2e-5 | 6.0e-5 |
| flat N=12 L=6 | 8.9e-9 | 1.0e-7 | 4.2e-5 | 4.8e-5 |
| implicit mover (decision 82), no filter | 0 | 16 | 1.9 | - (RZ m >= 1 open) |

- Flat filters: noise level at gamma 10 with 1000 - 300000 times less loss at k0; at gamma 5 - 6 the band reaches ~0.3 pi (4th - 5th
  harmonic of a lambda / 32 laser) and passes. For gamma_b >= 10 only.
- Force (force.py): 98 % of |F|^2 of the unstable modes is E_x; |E_perp + v x B| / |E_perp| = 0.09. The gather is already matched
  (decision 4): a matched gather is ruled out.
- Implicit mover: energy exact, NCI stronger (the field draws on the drift energy). Ruled out.
- Keeping the physics means moving the band (gamma_b >= 10, finer grid), not filtering it.

## 6. Open
- gamma ~ 5 residual at k_x D ~ pi/4 (0.0045 omega_p): a sharper low-pass does not help without touching the laser harmonics
  (section 6); a different EM dispersion near the light line, a higher gamma_b or a finer grid would.
- 3D not run (the mechanism is the same: k_x alone selects); RZ runs with the laser (coupling, focusing) with the proposed filter:
  the next check, on the boosted RZ LWFA case, needs Alexander's OK.
- The ~1 % absolute offset of the coupling measurement (section 3).
