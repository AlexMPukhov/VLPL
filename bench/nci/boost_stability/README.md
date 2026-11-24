# Stability of boosted-frame runs: streaming plasma, linear and nonlinear (decision 104)

Question (Alexander, 2026-11-14): are boosted-frame runs stable, linearly (seeded noise) and nonlinearly (large-amplitude
laser), in VLPL-RIP, -iRIP, -RZ and -iRZ? Answer: **not without the current filter**. A plasma streaming at -beta c through
the boosted grid develops the numerical Cherenkov instability (NCI) on the grids of a real boosted LWFA (h_perp < c dt, i.e.
dx' = gamma (1 + beta) dx_lab > h_perp). Decision 87 found no NCI only for a cold plasma (u_th = 1e-4) on coarse transverse
cells (h_perp >= 3 c dt) over 280 / omega_p. **With the binomial filter (`numerics.filter_passes`, decision 91) and shape 2
every case measured here is stable.** The summary now warns when a boosted deck has numerics that were measured unstable, and
the filter is allowed with the moving window of a boosted 2D/3D run (it was refused before: a boosted Cartesian run had no remedy).

Settings (provisional, Alexander decides on defaults):

| boosted run | measured |
|---|---|
| `shape = 2`, `filter_passes = 2` (binomial) | stable everywhere: flat at <= 6e-6 of the drift energy for 300 / omega_p |
| `shape = 2`, `filter_passes = 1` | stable for ~100 / omega_p; where h_perp < c dt a slow growth afterwards (x10 - x100 by 300 / omega_p) |
| `shape = 2`, no filter | stable at gamma 2, 3; unstable from gamma 5 |
| `shape = 1` | unstable from gamma 3 without filter, from gamma 5 also with 1 and 2 passes |
| `filter = projection / local` | unstable for drifting plasma (decision 91) |

A plasma element stays in the boosted box for about its proper time omega_p T = k_p L_lab (L_lab the lab window length), e.g.
20 / omega_p for a 50 um window at 0.004 n_c: one pass suffices there; the warning asks for two when k_p L_lab > 100.
The RZ gathers `plain` and `matched` give the same numbers to two digits in every streaming case (the gather is not the issue).

## 1. Linear: streaming plasma with a seed, periodic box (`run_linear.sh`, `table.sh`)
Neutral e-p plasma (`lin_c2.deck`, `lin_c3.deck`, `lin_rz.deck`) streaming at -beta c through a periodic box on the grid of a
boosted LWFA: lab lambda0 = 1 um, dx_lab = lambda / 32, n0 = 0.004 n_c (gamma 5, 10) or 0.002 n_c (gamma 20); in the box
dx' = gamma (1 + beta) dx_lab = 0.31 / 0.62 / 1.25 um, n = gamma n0, u = -gamma beta. Transverse cell h_perp = 0.25 um
(c dt / 1.2, / 2.5, / 5) and 1 um (3.2, 1.6, 0.8 c dt). Seed: thermal spread u_th = 1e-2 (all components, about 50 eV) on the
regular lattice. 64 x 32 cells (2D), 64 x 32 x 32 (3D), 64 x 32 x 2 modes (RZ, wall at r = 8 / 32 um), 2 x 2 (x 2) or
2 x 2 x 8 particles per cell, 300 / omega_p. RIP-CC only where its Courant limit allows (2D dx' < h_perp, 3D dx' < h_perp / sqrt 2).
148 runs, about 20 core-hours on the cluster nodes.

W_field / K_e (K_e the electrons' drift energy) at 10 / omega_p (the noise level after the start) and at 300 / omega_p, the
relative change of the total energy at the end, max |dGauss| (iRIP unless noted; 2D):

| gamma | h_perp | filter 0, shape 1 | filter 0, shape 2 | filter 1, shape 1 | filter 1, shape 2 | filter 2, shape 1 | filter 2, shape 2 |
|---|---|---|---|---|---|---|---|
| 5 | 0.25 um | 4.5e-5 -> 1.1e-2 | 2.3e-5 -> 9.5e-3 | 5.8e-6 -> 2.8e-3 | 5.0e-6 -> 2.9e-4 | 3.0e-6 -> 2.9e-4 | 2.8e-6 -> 5.8e-6 |
| 5 | 1 um | 2.0e-5 -> 2.7e-3 | 1.2e-5 -> 4.1e-5 | 4.2e-6 -> 2.1e-4 | 3.6e-6 -> 3.4e-6 | 2.4e-6 -> 7.5e-6 | 2.2e-6 -> 1.6e-6 |
| 10 | 0.25 um | 2.2e-5 -> 9.8e-3 | 1.1e-5 -> 2.3e-3 | 2.8e-6 -> 1.1e-3 | 2.2e-6 -> 2.5e-4 | 1.3e-6 -> 1.5e-5 | 1.2e-6 -> 2.7e-6 |
| 10 | 1 um | 1.2e-5 -> 1.8e-2 | 6.1e-6 -> 5.1e-3 | 1.7e-6 -> 5.1e-4 | 1.4e-6 -> 1.2e-5 | 8.3e-7 -> 1.8e-5 | 7.2e-7 -> 2.6e-6 |
| 20 | 0.25 um | 3.4e-6 -> 5.5e-3 | 1.7e-6 -> 1.4e-3 | 4.0e-7 -> 4.1e-5 | 3.3e-7 -> 4.6e-6 | 2.0e-7 -> 6.3e-6 | 1.8e-7 -> 2.1e-6 |
| 20 | 1 um | 2.0e-6 -> 1.2e-2 | 1.0e-6 -> 2.3e-3 | 2.8e-7 -> 2.4e-5 | 2.3e-7 -> 3.5e-6 | 1.4e-7 -> 7.5e-6 | 1.2e-7 -> 1.3e-6 |

RZ (iRZ, gather plain; matched within 2 % of these in every case):

| gamma | dr | filter 0, shape 1 | filter 0, shape 2 | filter 1, shape 1 | filter 1, shape 2 | filter 2, shape 1 | filter 2, shape 2 |
|---|---|---|---|---|---|---|---|
| 5 | 0.25 um | 1.7e-5 -> 1.5e-2 | 8.0e-6 -> 8.2e-3 | 2.4e-6 -> 3.3e-3 | 2.0e-6 -> 1.2e-4 | 1.2e-6 -> 7.6e-3 | 1.1e-6 -> 4.7e-6 |
| 5 | 1 um | 7.7e-6 -> 3.1e-3 | 4.1e-6 -> 1.4e-5 | 1.8e-6 -> 2.0e-4 | 1.4e-6 -> 1.9e-6 | 9.6e-7 -> 1.2e-5 | 8.3e-7 -> 8.0e-7 |
| 10 | 0.25 um | 8.4e-6 -> 2.7e-2 | 3.6e-6 -> 4.9e-3 | 1.1e-6 -> 2.4e-3 | 8.6e-7 -> 5.0e-5 | 5.6e-7 -> 5.6e-6 | 4.8e-7 -> 1.5e-6 |
| 10 | 1 um | 4.6e-6 -> 1.0e-2 | 2.0e-6 -> 4.1e-3 | 7.4e-7 -> 1.9e-4 | 5.3e-7 -> 6.3e-6 | 3.5e-7 -> 8.5e-6 | 2.8e-7 -> 1.1e-6 |
| 20 | 0.25 um | 1.3e-6 -> 3.6e-2 | 5.9e-7 -> 2.2e-3 | 1.7e-7 -> 1.6e-5 | 1.3e-7 -> 2.7e-6 | 8.4e-8 -> 3.3e-6 | 7.2e-8 -> 1.4e-6 |
| 20 | 1 um | 7.5e-7 -> 7.0e-3 | 3.4e-7 -> 9.5e-4 | 1.2e-7 -> 8.8e-6 | 8.7e-8 -> 1.5e-6 | 5.6e-8 -> 2.9e-6 | 4.7e-8 -> 6.3e-7 |

RIP-CC (h_perp = 1 um; gamma 5, 10): the same picture (2D: 6.3e-3 / 1.4e-2 for filter 0 shape 1, 1.6e-4 / 6.6e-4 shape 2,
<= 5.7e-6 for shape 2 with a filter; RZ RIP-CC: 2.9e-3 / 1.0e-2, shape 2 with one pass 2.2e-6). 3D (subset): iRIP gamma 10,
h_perp 0.25 um: 2.0e-3 (f0 s1), 5.9e-4 (f0 s2), 8.5e-4 (f1 s1), 3.1e-5 (f1 s2); gamma 20: 1.4e-3, 3.2e-4, 5.0e-6, 1.1e-6;
RIP-CC h_perp 1 um, no filter: gamma 5 9.7e-3 / 2.1e-4, gamma 10 5.4e-3 / 1.9e-4 (shape 1 / 2).
gamma 2 and 3 (2D, h_perp 0.4 um, 100 / omega_p): no filter shape 1 4.8e-5 / 3.8e-4, shape 2 1.1e-5 / 1.5e-5 (flat), one pass
2.4e-6 / 2.9e-6.
Unstable cases heat the plasma (W_total + 1e-4 .. 2e-4; RZ shape 1 at gamma 5: -3.5e-2); the Gauss law stays <= 2e-14 in every
run (the instability is not a charge-conservation error). Box length (64 vs 512 cells) and the anisotropy of the seed (u_th,x =
0.1, the boosted lab temperature) change W_field / K_e by < 10 % (`warm` scan, 2D gamma 10, dy = 0.4 um, 100 / omega_p:
3.7e-3 / 1.2e-3 / 5.2e-6 / 2.3e-6 for f0 s1 / f0 s2 / f1 s2 / f2 s2).

## 2. Nonlinear: boosted LWFA vs the lab run (`run_nonlinear.sh`, `plot.py`, `noise.py`)
`nl_c2.deck`, `nl_rz.deck`: lambda0 = 1 um, n0 = 0.004 n_c (lambda_p = 16 um) after a linear 100 um up-ramp (from x = 60 um),
laser a0 = 2 and 4, w0 = 8 um, 15 fs, focus 100 um, through the antenna (`laser.injection = antenna`); lab grid lambda / 32 x
0.4 um, window 50 um, 0.6 mm of propagation (k_p L = 240); iRIP shape 2, RZ 2 modes with 2 x 2 x 6 particles per cell. The lab
run of the same deck is the reference; lab snapshots at 600, 1000, 1400 and 1800 fs. Boosted gamma 5 and 10 (1622 and 1486
steps against 19187 in the lab: 12x and 13x fewer). Cold plasma (no seed but the laser) and warm plasma (u_th = 1e-2).

**Noise in the plasma ahead of the laser** (boosted frame, boosted step 600, rms E_x in the plasma the laser has not reached):

| case | 2D gamma 5 | 2D gamma 10 | RZ gamma 10 (and n_e on the axis / n_e at r = 2-10 um) |
|---|---|---|---|
| cold, filter 0 | 1.3e-16 | 7.6e-16 | 3e-15 (1.002) |
| warm, filter 0 | 1.6e-3 | 1.2e-2 (density modulation 18 %) | 8.1e-3 (1.17; 1.36 at step 600 near the entrance, 1.48 at step 1000) |
| warm, filter 1 | 3.2e-4 | 2.5e-4 | 1.9e-4 (1.004), plain = matched |
| warm, filter 2 | 1.3e-4 | 6.4e-5 | 5.7e-5 (1.001) |

The noise grows along the plasma's path from the injection plane: 2D gamma 10, warm, no filter, rms E_x 3.5e-4 at entry ->
1.1e-2 after 750 cells (~9 / omega_p of the plasma's proper time), E_y ~ B_z, density modulation 3 % -> 20 %. In RZ without the
filter the noise peaks on the axis (E_z 1.3e-2 on the axis, 4e-3 off axis) with an electron density excess of 17 - 48 % there;
with one pass the axis density is 1.002 - 1.004 and the axis E_z noise 3e-4 (4x the off-axis 7e-5, 0.15 % of the wake).
A cold plasma has no seed (the lattice is regular) and stays at round-off until the laser comes; the wake then seeds it.

**Boosted vs lab** (lab-frame snapshot, rms difference of the on-axis E_x over [laser - 30 um, laser + 8 um] / rms of the lab
E_x there; laser peak a):

| case | 1000 fs | 1400 fs | 1800 fs | laser peak at 1800 fs (lab 3.745) |
|---|---|---|---|---|
| 2D a0 = 2, gamma 5 / 10 | | 2.8 % / 5.7 % | | |
| 2D a0 = 4, gamma 5 | 12 % | 13 % | 12 % | 3.751 |
| 2D a0 = 4, gamma 5, filter 1 / 2 (cold) | 15 % / 18 % | | 20 % / 26 % | 3.746 / 3.752 |
| 2D a0 = 4, gamma 10, filter 0 cold | 105 % | 83 % | 80 % | 3.758 |
| 2D a0 = 4, gamma 10, filter 1 / 2 cold | 26 % / 20 % | | 19 % / 26 % | 3.777 / 3.798 |
| 2D a0 = 4, gamma 10, warm: filter 0 / 1 / 2 | 52 % / 47 % / 14 % | 68 % / 23 % / | 87 % / 22 % / 25 % | 3.553 / 3.788 / 3.803 |

The laser (amplitude, position, envelope) agrees to <= 1.5 % in every filtered or cold case; the warm unfiltered gamma 10 run
loses 5 % of the laser amplitude and modulates the pulse. The a0 = 4 wakes differ at the back of the bubble, where electrons are
self-injected: the lab-frame difference is concentrated there (the bubble front and the laser region agree to a few %). The
unfiltered gamma 10 run (cold) injects a much stronger bunch at the bubble back (E_x spikes to 0.3, a radiation spike of
amplitude 1.3 behind the bubble) than the lab run; with the filter the bunch and the spike are lab-like.
How large is the bubble-back difference by itself? Two lab runs that differ only in a detail differ there as well: 2D cold vs
warm lab run 8.5 / 9.8 / 13 % at 1000 / 1400 / 1800 fs; RZ lab gather plain vs matched 14 / 23 % at 600 / 1000 fs (bubble
front 0.9 / 1.3 %). Self-injection at a0 = 4 is sensitive; the filtered boosted runs are at 1 - 2x that level.

**RZ** (iRZ, 2 modes; rms difference of the on-axis E_z over the bubble [laser - 30, laser + 8 um] / over its front
[laser - 12, laser + 8 um]; the lab RZ runs take ~0.8 s per step with 2.5 M particles, the 1400 and 1800 fs snapshots were not
waited for):

| case | 600 fs | 1000 fs | laser peak at 1000 fs (lab 4.455; a0 = 2: 1.898) |
|---|---|---|---|
| a0 = 2, gamma 5, filter 1 | 5.0 % / 3.8 % | 6.7 % / 9.2 % | 1.907 |
| a0 = 2, gamma 10, filter 1 | 23 % / 4.8 % | 9.3 % / 9.6 % | 1.932 |
| a0 = 4, gamma 5, filter 0 / 1 | 97 % / 2.8 %, 81 % / 3.8 % | 56 % / 6.7 %, 52 % / 7.8 % | 4.461 / 4.429 |
| a0 = 4, gamma 10, filter 0 cold | 92 % / 4.5 % | 222 % / 8.2 % (E_z spike 1.24) | 4.497 |
| a0 = 4, gamma 10, filter 1 cold, plain / matched | 84 % / 5.3 %, 84 % / 5.3 % | 77 % / 8.1 %, 75 % / 8.1 % | 4.497 / 4.499 |
| a0 = 4, gamma 10, warm, filter 0 | 81 % / 16 % | 112 % / 62 % | 4.490 |
| a0 = 4, gamma 10, warm, filter 1, plain / matched | 82 % / 5.6 %, 82 % / 5.6 % | 78 % / 9.0 %, 78 % / 9.0 % | 4.485 / 4.482 |

The RZ laser agrees to <= 1.8 %; the bubble front to 4 - 10 % when filtered, 62 % for warm unfiltered plasma. The whole-bubble
difference (50 - 84 %) is the injected bunch at the bubble back (the lab plain/matched pair: 23 %); unfiltered cold gamma 10
injects a far stronger bunch (E_z spike 1.24 against 0.34). Plain and matched gathers agree to 1 - 3 % in every boosted case.
Figures (2D and RZ, lab frame and boosted frame) are written by plot.py; no on-axis artefact in the filtered RZ runs (axis
density 1.001 - 1.004 of the off-axis value ahead of the laser).

Gauss law: <= 1.3e-13 in every boosted run with no filter or one pass (2D and RZ, cold and warm) and with two passes in cold
plasma (1.7e-14). Warm plasma with two passes: an offset of 3.4e-9 (2D gamma 5), 3.3e-10 (2D gamma 10), 2.3e-10 (RZ gamma 10)
appears at boosted step ~1450 in all three runs (at two interior cells, not at an edge; 1e-8 of the electron charge density);
cause not found (open). MPI with the filter (one pass, 1 vs 4 ranks): examples/lwfa_boost.deck (3D RIP-CC) energies 4e-13,
lab fields 7e-15; rz_lwfa_boost.deck 9e-14, 7e-15, lab_beams.txt identical; ionization_boost.deck (2D) identical; Gauss <= 8e-16.

## 2b. a0 = 5: well above the self-injection threshold (2026-11-14, `bunch.py`)
The decks of section 2 at a0 = 5 (Alexander: "a0 = 4 is a threshold case; what about a0 = 5?"). Lab runs: 2D cold and warm,
RZ with the plain and the matched gather (the lab-vs-lab reference). Boosted: gamma 5 and 10, shape 2, one and two binomial
passes, cold; warm (u_th = 1e-2) and matched as variants. The boosted runs record the lab-frame electrons with u_x >= 19
(`lab_diagnostics.select_uz = 19`); `bunch.py` gives charge, mean energy and rms spread of the electrons with gamma >= 20 for
both kinds of run (lab: openPMD particles at the same lab times). 18 runs, ~250 core-hours (the two RZ lab runs ~2.5 h each
on 24 threads).

**RZ, injected bunch (gamma >= 20)** at 1000 / 1400 / 1800 fs (290 / 410 / 530 um); Q [pC], E_mean [MeV], rms dE/E, macro-particles:

| run | 1000 fs | 1400 fs | 1800 fs |
|---|---|---|---|
| lab plain | 112, 49.0, 0.41 (29713) | 149, 93.7, 0.41 | 166, 140, 0.37 (35096) |
| lab matched | 118, 48.5, 0.40 | 151, 96.2, 0.37 | 171, 140, 0.39 |
| gamma 5, 1 pass | 96, 52.0, 0.28 (428) | 119, 98.6, 0.38 | 116, 162, 0.16 (115) |
| gamma 5, 2 passes | 102, 51.8, 0.28 | 102, 115, 0.17 | 104, 169, 0.13 |
| gamma 5, 1 pass, ppc_z 8 (4x particles) | 95, 51.8, 0.28 (1764) | 120, 99.0, 0.37 | 122, 160, 0.19 (481) |
| gamma 10, 1 pass | 183, 26.8, 0.47 (87) | 189, 75.8, 0.41 | 200, 121, 0.42 (131) |
| gamma 10, 2 passes | 200, 27.8, 0.47 | 195, 80.3, 0.27 | 190, 135, 0.16 |
| gamma 10, 1 pass, ppc_z 8 | 160, 29.7, 0.41 (276) | 173, 81.5, 0.38 | 200, 121, 0.46 (267) |
| gamma 10, 1 pass, matched / warm | 169 / 168, 29.7 / 33.8 | 175 / 193, 86 / 78 | 216 / 214, 119 / 118, 0.52 / 0.46 |

Lab vs lab: charge 3 %, energy 0.4 %. **The boosted bunch stays far off:** gamma 5 -30 % charge, +15 % energy, half the spread;
gamma 10 +20 - 30 % charge, -4 .. -15 % energy (+60 % / -40 % at 1000 fs). Four times the particles changes nothing beyond the
scatter (gamma 5: 116 -> 122 pC, gamma 10: 200 -> 200 pC): the macro-particle sampling (the boosted lattice is gamma^2 (1 + beta)
coarser in the lab, 50 - 130 macro-particles per bunch) is not the cause. The wake (on-axis E_z, whole bubble / its front):

| run | 600 fs | 1000 fs | 1400 fs | 1800 fs |
|---|---|---|---|---|
| lab plain vs matched | 11 % / 0.8 % | 18 % / 1.2 % | 22 % / 1.7 % | 26 % / 2.6 % |
| gamma 5, 1 / 2 passes | 86 / 82 % (front 3 %) | 36 / 40 % (7 %) | 25 / 30 % (7 %) | 21 / 28 % (10 %) |
| gamma 10, 1 / 2 passes | 86 / 86 % (5 %) | 58 / 46 % (7 %) | 49 / 54 % (7 - 8 %) | 41 / 59 % (10 - 11 %) |
| gamma 10 matched / warm (1 pass) | 86 / 87 % (5 %) | 57 / 62 % (7 - 8 %) | 52 / 61 % (7 - 11 %) | 43 / 67 % (10 - 15 %) |

The laser peak agrees to <= 2 % (lab 5.18 / 5.65 / 5.14 / 4.57). The bubble front differs by 3 - 15 %, 4 - 6 x the lab-vs-lab level.

**2D**: the lab run (cold and warm) has no electron above gamma 10 up to 530 um (the bubble-back electrons reach 4 - 5 MeV):
no self-injected bunch. Boosted gamma 10 with one pass makes one: 3.1 / 2.7 / 2.2 pC (5 - 7 macro-particles, 13 - 27 MeV), warm
9 / 19 / 23 pC up to 176 MeV; gamma 5 and gamma 10 with two passes none. Wake: whole bubble 16 - 19 % (gamma 5), 19 - 40 %
(gamma 10), front 2 - 5 %, against lab cold vs warm 5 - 8 % / 1.2 %; laser <= 1.5 %.

**Interpretation** (with the relay session; Lee and Vay, Comput. Phys. Commun. 238, 102 (2019)): dx' = gamma (1 + beta) dx keeps
the resolution of what moves at c (the laser), not of the bubble back, which moves at gamma_w (about gamma_p / sqrt 3 ~ 9 here,
gamma_p = 15.8): its resolution is lost by about 1 + (gamma / gamma_w)^2 = 1.3 (gamma 5), 2.2 (gamma 10), and plasma electrons
cross the sheath in 2 - 2.4x fewer steps. At gamma 10 the boost is at gamma_w, where trapping (stopping an electron from
u' = -10 within ~10 steps) is decided by the step error: more charge at lower energy. Lee and Vay (Warp, 2D, a0 = 5 at 1e19 cm^-3)
used gamma_b ~ gamma_g / 4 and needed N_z / lambda0 >= 64 for < 1 % in charge and energy (~10 % at 32; emittance 128), with the
same convergence rate in the lab and the boosted frames. Here gamma_g / 4 = 4: our gamma 5 is slightly, gamma 10 2.5x above, and
lambda / 32 is where they still see ~10 %. Proposed rule for self-injection (not yet tested here): gamma_b <= gamma_g / 4,
lab-equivalent resolution >= lambda / 64. A VLPL convergence series against Lee and Vay's case (gamma_b = 1, 2, 3 at lambda / 32
and 64) is the next step; not run (cost below).

## 2c. Convergence lab vs boosted: the case of Lee and Vay (2D, 1e19 cm^-3, a0 = 5; `lv.deck`, `run_lv.sh`, `lv_ana.py`)
Lee and Vay, Comput. Phys. Commun. 238, 102 (2019) (Warp, CK and PSATD) found charge and mean energy of the self-injected bunch
equal in the lab and in boosted frames (gamma_b = 1, 2, 3) to < 1 % at N_z / lambda0 >= 64, ~10 % at 32. Their 1e19 case in
VLPL 2D: a0 = 5, lambda0 = 0.8 um, waist 8.9 um (k_p sigma = 5.3), field exp(-zeta^2 / L^2) with k_p L = 2 (intensity FWHM
13.2 fs; their text does not define sigma and L further: an assumption), focus at the plasma entrance, 50 um linear ramp,
dy = 0.1 um, 2 x 2 particles per cell (theirs 4 x 4), shape 2 (theirs cubic), lab without filter (then refused with the lab window; section 2d),
boosted with one binomial pass. Bunch as theirs: electrons with gamma >= 50 (they also cut at the first bucket). 11 runs,
~250 core-hours (lab lambda / 64 ~5 h on one node).

Injected bunch at lab time 1000 fs (plasma depth z = 247 um, injection complete) and 800 fs (187 um); Q [pC per um of slab]:

| run | Q, E_mean [MeV], dE/E at 247 um | at 187 um | on-axis E_x vs lab iRIP lambda/64, bubble / front |
|---|---|---|---|
| lab iRIP lambda / 64 | 17.85, 39.3, 0.25 | 14.86, 31.5 | - |
| lab RIP-CC lambda / 64 | 17.77, 39.2, 0.24 | 15.11, 31.5 | 2.0 % / 0.7 % |
| lab iRIP lambda / 32 | 17.36, 39.6, 0.20 | 15.29, 31.6 | 7.0 % / 1.0 % |
| lab RIP-CC lambda / 32 | 17.46, 39.5, 0.20 | 15.24, 31.8 | 7.8 % / 1.0 % |
| gamma 2, iRIP lambda / 32 | 17.55, 40.9, 0.19 | 16.24, 32.5 | 7.2 % / 1.1 % |
| gamma 2, RIP-CC lambda / 32 | 17.40, 41.6, 0.21 | 16.04, 32.6 | 8.0 % / 1.0 % |
| gamma 3, iRIP lambda / 32 | **21.52, 52.3, 0.39** | 20.51, 35.4 | 21 % / 1.1 % |
| gamma 2, iRIP lambda / 64 | 17.92, 40.9, 0.21 | 16.02, 32.7 | 3.9 % / 0.7 % |
| gamma 2, RIP-CC lambda / 64 | 18.03, 40.6, 0.21 | 15.73, 32.7 | 4.0 % / 0.8 % |
| gamma 3, iRIP lambda / 64 | 18.08, 40.8, 0.19 | 16.25, 32.5 | 3.6 % / 0.7 % |
| gamma 3, RIP-CC lambda / 64 | 18.03, 41.5, 0.22 | 16.14, 32.5 | 5.4 % / 0.8 % |

- Lab frame: lambda / 32 -> 64 changes the charge by 2.2 - 2.8 % and the energy by 0.8 %; RIP-CC = iRIP within 0.5 % at either
  resolution. No sign that the box scheme makes the lab frame converge slowly for charge and energy (two resolutions only: no
  order measured; lambda / 128 not run).
- Boosted, lambda / 64: charge within +0.4 .. +1.3 % of the lab lambda / 64 (gamma 2 and 3, both solvers); mean energy +3.3 ..
  +5.6 %, energy spread 0.19 - 0.22 against 0.24 - 0.25. At 187 um: charge +6 .. +9 %, energy +3 .. +4 %.
- Boosted, lambda / 32: gamma 2 within -2.5 .. +0.5 % (charge) and +4 .. +6 % (energy); gamma 3 with iRIP +21 % / +33 % (the
  failure Lee and Vay see at low resolution). At lambda / 64 gamma 3 joins gamma 2.
- The energy offset of the boosted runs (+1.5 MeV, ~4 %) does not shrink from lambda / 32 to 64 at gamma 2 while the lab energy
  moves by 0.8 %: a systematic difference, not a resolution effect. It is the filter: the boosted runs had one pass, the lab
  runs none (section 2d). Lee and Vay filtered both frames.
- VLPL's threshold from these runs: gamma_b <= 2 (= gamma_g / 6.6) at lambda / 32, gamma_b <= 3 (gamma_g / 4.4) at lambda / 64,
  for charge to ~1 - 2 % and energy to ~5 %; Warp's: gamma_b <= 3 at lambda / 64 for < 1 % in both. The energy is the weaker
  quantity here.
- Against Lee and Vay's absolute numbers: the charge agrees (17.5 uC/m = 17.5 pC per um at 200 - 450 um; VLPL 17.4 - 17.9 at
  247 um, 14.4 - 16.3 to 487 um, lab lambda / 32), the mean energy does not (theirs 160 MeV at 450 um; VLPL 61 MeV at 427 um with
  E_max 207 MeV and an on-axis field of 14 - 16 GeV/cm against their mean 5.3 GeV/cm): our gamma >= 50 bunch holds a large,
  continuously injected tail at 25 - 70 MeV. A different setup (laser sigma and L) is the likely reason; not resolved.

The a0 = 5 case of section 2b (0.004 n_c, gamma_g = 15.8) used gamma 5 and 10 at lambda / 32: gamma_g / 3.2 and gamma_g / 1.6,
outside this window; that is where its 15 - 30 % come from.

## 2d. The energy offset of 2c is the current filter (2026-11-16, decision 107)
The lab window now accepts the filter (decision 107), so both frames can be run with the same numerics. New runs: the four lab
runs of 2c with one binomial pass, and boosted gamma 2 iRIP lambda / 32 without filter (stable over this interval: Gauss
1.9e-14). ~170 core-hours (cluster nodes, lab lambda / 64 5.0 - 5.3 h each on 24 cores). Bunch gamma >= 50 as in 2c:

| run | 1000 fs (247 um): Q [pC], E [MeV], dE/E | 800 fs (187 um): Q, E |
|---|---|---|
| lab iRIP lambda / 32, no filter / 1 pass | 17.36, 39.6, 0.20 / 17.79, 40.2, 0.19 | 15.29, 31.6 / 16.17, 32.5 |
| lab RIP-CC lambda / 32, no filter / 1 pass | 17.46, 39.5, 0.20 / 17.60, 40.3, 0.19 | 15.24, 31.8 / 16.14, 32.5 |
| lab iRIP lambda / 64, no filter / 1 pass | 17.85, 39.3, 0.25 / 18.31, 40.5, 0.21 | 14.86, 31.5 / 16.27, 32.6 |
| lab RIP-CC lambda / 64, no filter / 1 pass | 17.77, 39.2, 0.24 / 18.32, 40.4, 0.21 | 15.11, 31.5 / 16.35, 32.6 |
| gamma 2 iRIP lambda / 32, no filter / 1 pass | 17.33, 39.3, 0.22 / 17.55, 40.9, 0.19 | 14.68, 31.5 / 16.24, 32.5 |
| boosted lambda / 64, 1 pass (gamma 2, 3; iRIP, RIP-CC; from 2c) | 17.92 - 18.08, 40.6 - 41.5, 0.19 - 0.22 | 15.73 - 16.25, 32.5 - 32.7 |

- Same numerics in both frames, the offset is gone. No filter (gamma 2, lambda / 32): boosted - lab = -0.2 % in charge, -0.8 %
  in energy (1000 fs); -4 % / -0.3 % at 800 fs. One pass, lambda / 32 (gamma 2): -1.4 .. -1.1 % / +1.7 .. +3.2 % at 1000 fs,
  +0.4 .. -0.6 % / 0 .. +0.3 % at 800 fs. One pass, lambda / 64 (gamma 2, 3, both solvers, against the lab of the same
  solver): charge -1.3 .. -2.1 %, energy +0.2 .. +2.7 % (iRIP gamma 2 +1.0 %, gamma 3 +0.7 %; RIP-CC gamma 2 +0.5 %, gamma 3
  +2.7 %), energy spread 0.19 - 0.22 against 0.21; at 800 fs charge -0.1 .. -3.8 %, energy within 0.3 %.
- The +3 .. 6 % of 2c compared a filtered boosted run with an unfiltered lab run. One pass itself moves the lab bunch by +1.5 ..
  +3 % in energy and +1 .. +3 % in charge at 1000 fs (+6 .. +9 % in charge at 800 fs): the filter changes the injection, in
  either frame, by about as much as gamma_b does at lambda / 64.
- Lab with the filter, lambda / 32 -> 64: charge +2.9 / +4.1 % (iRIP / RIP-CC), energy +0.7 / +0.2 %. No box-scheme slowness
  either (RIP-CC = iRIP within 1 % at each resolution).
- VLPL against Warp's < 1 %: with the same filter in both frames gamma_b <= 3 at lambda / 64 gives ~1 % in energy (2.7 % for
  gamma 3 RIP-CC) and -1.3 .. -2.1 % in charge. Not explained: this remaining charge deficit of the boosted runs and the gamma 3
  RIP-CC energy. Not tested (no longer needed for the offset): the lab-frame particle snapshot; lambda / 128.

## 3. Code changes (decision 104; 107)
- `numerics.filter_passes` with the moving window in a boosted 2D/3D run (it was refused for every Cartesian window; the
  lab window followed in decision 107: W9 in tests/test_window.cpp, Gauss 5e-16 in 3D, 6e-17 in 2D, 1 and 2 passes). The Gauss check skips
  the first filter_passes cells at the open left edge in a boosted run with the filter, as RZ does (rz_gauss_istart): the
  plasma leaves there and the filter's closure differs from the interior, the offset cannot propagate. Without the skip:
  2.4e-3 when the plasma leaves (2D gamma 5, 2 passes).
- Summary warning (`boost_nci_risk`, src/boost.cpp) for a boosted run with a streaming plasma and numerics measured unstable:
  filter = projection / local; shape 1 at gamma > 2.5; no filter at gamma > 3; one pass when k_p L_lab > 100.
- Test B6 in tests/test_boost.cpp (`build/tests test_boost`; BOOST_NCI_ONLY=1 runs B6 alone, ~80 s on 8 threads):
  (a) the 2D and RZ periodic streaming plasma at gamma 10 (warm, 100 / omega_p): without filter W_field / K_e = 1.2e-3 (2D),
  6.3e-4 (RZ) against 4.6e-6, 2.2e-6 with one pass (the test asks > 10x and < 1e-5); (b) a boosted 2D run with one pass and
  warm plasma streaming out at the left edge: Gauss law 1.6e-15 over 171 steps (6.4e-4 without the left-edge skip); (c) the
  warning on and off.

## What stays open / untested
- Self-injection (a0 = 5, section 2b): the boosted bunch differs from the lab run by 15 - 30 % in charge and energy at
  lambda / 32 with gamma 5 and 10 (not a sampling effect). Section 2c: on the Lee and Vay case VLPL converges for gamma_b <= 3
  (gamma_g / 4.4) at lambda / 64; with the same filter in both frames (2d) energy ~1 % (2.7 % gamma 3 RIP-CC), charge -1.3 ..
  -2.1 %; lambda / 128 and the a0 = 5 case of 2b at gamma_b <= 4 not run.
- 3D boosted LWFA end to end (only the periodic 3D subset); shape 3 (refused with a window).
- Gamma 4 and gamma > 20; densities other than 0.004 / 0.002 n_c; RIP-CC below its Courant limit is impossible by construction.
- Two passes with warm plasma: the Gauss offset above (1e-10 .. 3e-9 from boosted step ~1450).
- The boosted a0 = 4 wake differs from the lab run by 12 - 25 % at the bubble back (self-injection) even when stable; whether
  finer boosted cells close this (the decision 87 resolution note: the plasma streams into the laser, (1 + beta) finer cells)
  was not measured.
- Physics defaults (filter_passes = 0, shape 1 in Cartesian decks) are unchanged; a boosted default of shape 2 + 2 passes is
  proposed to Alexander, not implemented.
