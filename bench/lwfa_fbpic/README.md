# LWFA benchmark: VLPL against FBPIC

Phase 1 exit criterion "LWFA bubble vs FBPIC" (docs/ROADMAP.md), first, mildly nonlinear version.

## Setup (the same in both codes, lab frame)
- Laser: λ0 = 0.8 µm, a0 = 1, linear polarisation, duration 10 fs (intensity FWHM; FBPIC
  τ = 10 fs / sqrt(2 ln 2)), waist 4 µm focused at x = 0. The centre crosses x = 0 at 20 fs.
  VLPL injects the pulse through the open left boundary; FBPIC initialises it in the box at
  z0 = -c 20 fs.
- Plasma: electrons at 0.01 n_c (1.74e19 cm^-3, λp ≈ 8 µm), a linear up-ramp from 13 to 20 µm,
  then flat. VLPL has mobile protons; FBPIC has a neutralising background.
- Moving window at c. From 40 fs on, both boxes are [c (t - 40 fs), c (t - 40 fs) + 12.8 µm].
- Grid: dx = dz = 0.05 µm (λ/16), c dt = dx.
  - VLPL: 3D, 256 × 72 × 72, dy = dz = 0.25 µm, periodic in y and z (18 µm).
  - FBPIC: RZ with 2 modes, Nr = 36, dr = 0.25 µm (r_max = 9 µm, reflective), PSATD of infinite
    order.
- Particles:
  - VLPL: 1 per cell per species on a regular lattice, shape 2.
  - FBPIC: 2 × 2 × 4 per cell, cubic shape.
- Output every 300 steps (50 fs) up to 1800 steps (300 fs).

## Run
```
build/vlpl bench/lwfa_fbpic/rip.deck                    # out_bench_rip/ (3D: ~1 h on 2 cores)
pip install fbpic pyfftw openpmd-viewer h5py scipy
python3 bench/lwfa_fbpic/fbpic_lwfa.py out_bench_fbpic    # RZ, CPU (numba); use_cuda=True on a GPU
python3 bench/lwfa_fbpic/compare.py out_bench_rip out_bench_fbpic [tolerance]
```
`compare.py` prints, per output time:
- the peak on-axis longitudinal field behind the pulse in both codes (units m c ω0 / e);
- the relative L2 difference of the on-axis Ex over the wake (from the window's left edge to 2 µm
  behind the laser peak);
- the peak laser field of both codes.

It exits 1 if the largest L2 difference exceeds the tolerance (default 0.10). Outputs with no wake yet
(pulse still in vacuum) are skipped.

## Results (2026-10-27, cloud session, 4 cores)
FBPIC 0.27.1 (CPU, numba): 15 min. VLPL: 30 min with 4 OpenMP threads, max|dGauss| 4.0e-14.

| t [fs] | on-axis Ex, VLPL | FBPIC | rel. L2 difference | peak laser field, VLPL | FBPIC |
|---|---|---|---|---|---|
| 100 | 0.0241 | 0.0232 | 0.039 | 0.931 | 0.940 |
| 150 | 0.0273 | 0.0261 | 0.046 | 0.888 | 0.874 |
| 200 | 0.0240 | 0.0232 | 0.040 | 0.824 | 0.815 |
| 250 | 0.0205 | 0.0201 | 0.034 | 0.739 | 0.746 |
| 300 | 0.0172 | 0.0166 | 0.041 | 0.673 | 0.660 |

The wake agrees to 3.4-4.6 % (relative L2) and the laser amplitude to 1-2 %. The VLPL wake peak is
3-4 % higher.

Lesson (README deck section): `laser.waist_z` defaults to 0, uniform in z. The
first run drove a sheet laser that diffracted in y only: amplitude ~ (1 + (x/zR)^2)^(-1/4), 0.83
instead of 0.66 at 300 fs, wake 20-48 % off. A 3D deck needs `waist_z = waist`.

## RZ version (2026-10-28, Phase 2 exit test "RZ vs 3D for LWFA")

Update (shape 2 in RZ, decision 50; `rip_rz.deck` now uses it, as the 3D run):

| t [fs] | wake, VLPL-RZ shape 2 | FBPIC | L2 (r <= 0.5 µm) | axis row | VLPL-RIP 3D | laser RZ | FBPIC |
|---|---|---|---|---|---|---|---|
| 100 | 0.0241 | 0.0232 | 0.036 | 0.046 | 0.0241 | 0.927 | 0.940 |
| 150 | 0.0273 | 0.0259 | 0.049 | 0.062 | 0.0273 | 0.886 | 0.874 |
| 200 | 0.0238 | 0.0230 | 0.037 | 0.047 | 0.0240 | 0.817 | 0.815 |
| 250 | 0.0204 | 0.0198 | 0.035 | 0.044 | 0.0205 | 0.736 | 0.746 |
| 300 | 0.0171 | 0.0165 | 0.043 | 0.051 | 0.0172 | 0.677 | 0.660 |

With shape 2 the RZ wake peak equals VLPL-RIP 3D's to 0-1 %, the near-axis L2 against FBPIC is 3.5-4.9 %
(3D: 3.4-4.6 %) and the axis row 4.4-6.2 % (shape 1: 7-11 %). Gauss law 2.5e-14. 220 s on 4 threads.
The shape-1 run below stays for reference.
`rip_rz.deck`: the same setup in VLPL's RZ geometry on FBPIC's grid (2 modes, Nr = 36, dr = 0.25 µm,
2 × 2 × 4 particles per cell, RIP-CC, linear shapes, a conducting wall at 9 µm). Both codes write openPMD
thetaMode, and `compare_rz.py` reads both with openpmd-viewer in the same way:
```
build/vlpl bench/lwfa_fbpic/rip_rz.deck                        # out_bench_rip_rz/ (2 min on 4 cores)
python3 bench/lwfa_fbpic/compare_rz.py out_bench_rip_rz out_bench_fbpic [tolerance]
```
The wake is compared as Ez of m = 0 averaged over r <= 0.5 µm (two radial cells). On the axis row alone
it is limited by particle noise with linear shapes: two VLPL runs that differ only in the sampling
(ppc_r 2 -> 4, ppc_theta 4 -> 8) differ by 5-14 % there, as much as VLPL and FBPIC do.

| t [fs] | wake, VLPL-RZ | FBPIC | rel. L2 difference (r <= 0.5 µm) | on axis row | VLPL-RIP 3D (above) | peak laser field, VLPL-RZ | FBPIC |
|---|---|---|---|---|---|---|---|
| 100 | 0.0238 | 0.0232 | 0.043 | 0.070 | 0.0241 | 0.938 | 0.940 |
| 150 | 0.0274 | 0.0259 | 0.055 | 0.072 | 0.0273 | 0.906 | 0.874 |
| 200 | 0.0244 | 0.0230 | 0.045 | 0.074 | 0.0240 | 0.826 | 0.815 |
| 250 | 0.0206 | 0.0198 | 0.046 | 0.088 | 0.0205 | 0.735 | 0.746 |
| 300 | 0.0180 | 0.0165 | 0.059 | 0.108 | 0.0172 | 0.685 | 0.660 |

VLPL-RZ agrees with FBPIC to 4.3-5.9 % (3D: 3.4-4.6 %), its wake 3-9 % higher as in 3D, and with
VLPL-RIP 3D to 0-5 % in the wake peak; the laser amplitude to 0.2-4 % of FBPIC. Gauss law per mode at
5e-14 throughout. VLPL-RZ: 120 s on 4 threads (3D: 30 min, FBPIC CPU: 35 min).
Variants (worst L2 on the axis row): 4x the particles 0.099, dr = 0.125 µm 0.086 (base 0.108). A tried
axis volume V_0 = pi dr^2/3 (uniform plasma deposits n on the axis instead of 4/3 n) made it worse
(0.121) and was dropped. Shape 2 near the axis (step 4) should lower the axis noise.
