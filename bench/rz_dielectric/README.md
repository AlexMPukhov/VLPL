# RZ dielectric blocks: benchmarks (decision 100, docs/specs/rz_dielectric.md)

Runs on the CPU cluster (Ivy Bridge nodes, 24 threads), 2026-11-13. Units SI; c dt = dz.

## 1. Dielectric wakefield accelerator: `dwa.deck`, `dwa.py`
A metal tube b = 75 um (the grid wall) lined with eps for 50 um < r < b from z = 0.95 mm on; a rigid 100 pC bunch at gamma = 1000
(sigma_z 10 um); moving window 1 mm, 6 mm of propagation. dwa.py fits E_z on the axis behind the bunch by the first modes with
free wavenumbers and compares with the modal theory of the lined tube (TM_0n of a v = c bunch; the formulas of the public
GEM-PIC-RZ tests/dielectric.py).

| eps | mode | k (1/mm) theory | k / k_th - 1 | amplitude / theory - 1 |
|---|---|---|---|---|
| 3.8 | TM01 | 32.9076 | -1.6e-05 | +2.2e-04 |
| 3.8 | TM02 | 94.1031 | -3.3e-04 | +9.3e-04 |
| 3.8 | TM03 | 162.526 | -5.6e-04 | -5.0e-03 |
| 2.1 (PTFE) | TM01 | 45.9500 | -4.4e-05 | +1.1e-04 |
| 2.1 (PTFE) | TM02 | 140.103 | +8.2e-05 | -3.0e-04 |

Fit residual 1.2 % (both); Gauss law 1.8e-12 of max rho over the run. (Before decision 100's front layer the
eps = 3.8 run grew at the moving front edge after 2 mm: Gauss 0.6, 3.7 GV/m ahead of the bunch, TM02 off by 18 %.)

Thin wall, dr convergence: a = 60 um, b = 80 um (20 um lining, eps 3.8), dz = 0.8 dr, sigma_z 20 um:

| dr (cells across the wall) | TM01 k / k_th - 1 | amplitude - 1 | TM02 k - 1 | Gauss |
|---|---|---|---|---|
| 4 um (5) | -3.4e-03 | -4.9e-03 | -2.3e-02 | 6.5e-14 |
| 2 um (10) | -8.4e-04 | -1.2e-03 | -5.9e-03 | 1.8e-13 |
| 1 um (20) | -2.1e-04 | -1.6e-04 | -1.5e-03 | 5.0e-13 |

Second order. With the inner interface off the nodes (a = 59 um): -1.9e-3 (5 cells), +4.6e-5 (10 cells).
  build/vlpl bench/rz_dielectric/dwa.deck  &&  python3 bench/rz_dielectric/dwa.py out_dwa 3.8
  thin wall: ... grid.nr=40 grid.dr=2um grid.dz=1.6um grid.nz=625 "dielectric layer.r_in=60um" "dielectric layer.r_out=90um"
  (bunch at 0.6 mm: "species bunch.density=0.71093" ... gauss(z - 0.6 mm, 20 um)), python3 dwa.py <dir> 3.8 60 80 20 1000 18361526.7 0.6

## 2. Cherenkov cone: `cherenkov.deck`, `cherenkov.py`
The same bunch entering an open dielectric eps = 1.3 (r < 0.5 mm, radial absorber): at the lab plane 1.5 mm inside, the arrival
time of the B_phi peak against r (60 ... 360 um): c dt/dr = 0.54622 vs sqrt(eps beta^2 - 1) / beta = 0.54772 (-2.7e-3); angle
28.64 deg vs 28.71 deg; rms deviation from the line 0.08 um; Gauss 1.3e-12.

## 3. Thin pipe and metal stripes: `pipe.deck`
A free pipe (200 um, wall 20 um, eps 3.8 or 2.1, vacuum inside and outside), and with metal stripes on its outer surface
(rings.r_in = 220 um, r_out = 230 um, period 230 um, gap 115 um): 6 mm in a moving window, Gauss 1.8e-13, the field ahead of
the bunch < 0.02 MV/m (stable).

