# Track O benchmark 1: relativistic oscillating mirror harmonics (1D-like)

Gate O criterion (docs/ROADMAP.md, docs/specs/overdense.md): the reflected spectrum of a relativistic laser at normal
incidence on an overdense surface converges in dx and follows the relativistic-spikes power law I(omega) ~ omega^(-8/3)
(Baeva, Gordienko, Pukhov, PRE 74, 046404 (2006)) up to the resolved order.

`rom.deck`: a plane, linearly polarised pulse, a0 = 20, 8 fs, on 50 n_c (S = n / (a0 n_c) = 2.5) with an exponential
pre-plasma of L = 0.1 lambda0, immobile ions, iRIP, shape 2. Quasi-1D (y, z periodic, 4 cells). The reflected field is
recorded at x = 0 by `output.xflux_every = 1` (docs/specs/xflux.md); `spectrum.py` gives the spectrum, the fitted
exponent over the odd harmonics 5 .. 30 and the reflected energy fraction.

    build/vlpl bench/rom_harmonics/rom.deck grid.dx=0.002um grid.nx=2000 output.dir=out_400   # lambda0 / 400
    python bench/rom_harmonics/spectrum.py out_200 out_400 --png rom.png

## Results (2026-10-30, the CPU cluster, 24 threads; 2 and 7 min per run)

| target | cells per lambda0 | exponent (odd N = 5 .. 30) | R (energy) | I_11 / I_1 | I_31 / I_1 |
|---|---|---|---|---|---|
| 50 n_c, pre-plasma L = 0.1 lambda0, a0 = 20 (deck) | 200 | -2.79 | 0.80 | 1.9e-3 | 1.2e-4 |
| | 400 | -2.95 | 0.80 | 1.4e-3 | 1.4e-4 |
| 50 n_c step, a0 = 20 | 200 / 400 | -3.80 / -3.83 | 0.86 | 9.7e-4 / 5.1e-4 | 1.4e-5 / 1.7e-5 |
| 100 n_c step, a0 = 10 (S = 10) | 100 / 200 / 400 | -6.6 / -6.5 / -7.2 | 0.99 / 0.98 / 0.98 | 4e-5 .. 6e-5 | 1e-8 .. 2e-8 |

- With the pre-plasma the spectrum follows omega^(-8/3) (-2.79 / -2.95) and the two resolutions agree to the 60th
  harmonic within the spike-to-spike scatter. A sharp step and a stiffer surface (S = 10) give much steeper spectra,
  as expected outside the spikes regime; their low orders converge in dx as well.
- About 20 % of the energy is absorbed with the pre-plasma (R = 0.80), 14 % for the sharp 50 n_c step.
- Next: the transverse structure (RZ m = 1, finite waist) and oblique incidence in 3D (needs O1, the TFSF source).
