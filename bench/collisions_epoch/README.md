# Solid-density electron-proton relaxation: VLPL against Smilei 5.1, EPOCH 4.20 and NRL

A track-O case (docs/ROADMAP.md): hydrogen at n_e = n_i = 100 n_c (lambda = 1 um: 1.1e23 cm^-3), T_e = 100 eV, T_i = 10 eV,
lnL = 5 for e-e, e-p and p-p, 0.5 ps (the e-p relaxation time ~0.2 ps). VLPL and Smilei with frozen species (only the
collisions act, c dt = 0.02 / k0, s ~ 0.02 per e-p pair at the thermal speed); EPOCH as full 1D PIC (dx = 0.25 nm = 1.1 Debye
lengths, cubic B-splines), with its Nanbu-Pérez scheme (use_nanbu = T), with the older Sentoku-Kemp scheme (use_nanbu = F) and
without collisions (the heating by the PIC noise).

```
EXE=../../build/vlpl sbatch run_vlpl_smilei.sbatch   # out_vlpl_solid, out_vlpl_solid_s5 (angle = smilei5), out_smilei_solid
EPOCH=<epoch1d> sbatch run_epoch.sbatch              # out_epoch_{nanbu,sk,nocoll}
. $WORK/smilei/smilei_env.sh; python3 compare.py   # EPOCH: needs the sdf module of EPOCH's SDF/utilities
```

Time at which T_e - T_i falls to 1/e of its initial 90.2 eV (1/omega, 942 = 0.5 ps; the CPU cluster, 2026-11-10):

| run | 1/e time | vs NRL |
|---|---|---|
| NRL (Spitzer, Maxwellians, lnL = 5) | 409.3 | 1 |
| VLPL (Nanbu, low_t_cap = true, default) | 487.8 | 1.19 |
| VLPL, dt / 4 | 487.0 | 1.19 |
| VLPL, angle = smilei5 | 503.2 | 1.23 |
| Smilei 5.1 | 512.8 | 1.25 |
| VLPL, low_t_cap = false | 455.0 | 1.11 |
| VLPL, low_t_cap = false, dt / 4 | 425.2 | 1.04 |
| EPOCH 4.20.1 Nanbu-Pérez (use_nanbu = T) | 486.9 | 1.19 |
| EPOCH 4.20.1 Nanbu-Pérez, dt / 4 | 487.6 | 1.19 |
| EPOCH 4.20.1 Sentoku-Kemp (use_nanbu = F) | 241.9 | 0.59 |
| EPOCH 4.20.1 without collisions | not reached (T_e - T_i 90.60 -> 90.59 eV) | |

- VLPL and EPOCH's Nanbu-Pérez scheme agree to 0.2 % (487.8 / 486.9); EPOCH is a full PIC run (fields, 500 particles per
  species and cell, its CFL step c dt = 0.95 dx, i.e. 1.5e-3 / omega, 13 times VLPL's) and is converged in dt (dt / 4: 487.6). EPOCH's
  scheme applies the same low-temperature limit, so it lands with VLPL and Smilei at 1.19 NRL, not at NRL. The PIC noise alone
  heats the electrons by 0.02 eV and the protons by 0.03 eV in 0.5 ps (no-collision run): negligible.
- EPOCH's older Sentoku-Kemp scheme relaxes twice as fast (0.59 NRL); not investigated (it is not the scheme VLPL follows).
- VLPL and Smilei agree (2-5 %; with the same angle law 1.9 %).
- Both are ~20 % slower than NRL because of Pérez's low-temperature limit s <= s' (the mean free path not below the interatomic
  distance): at this density and temperature it binds for the electron-proton pairs below ~0.66 of the thermal relative speed.
  Without it VLPL converges to NRL with dt (455 -> 425, +4 % at dt / 4; the rest is the time-step error of the slow pairs).
  Whether the limit should act here: decided in decision 94 (kept, see below).

## The physical Coulomb log (dense_plasma.py, run_vlpl_auto.sbatch)

lnL = 5 is not the log of this plasma.
- Plasma parameters: Gamma_ii = 1.1, T_e / T_F = 12 (not degenerate).
- The dense-plasma logs at t = 0: Lee-More 2.79, GMS6 2.56, BPS 2.34, Landau-Spitzer 2.74, NRL formulary 2.07.

`python3 dense_plasma.py` prints them, the 1/e times the Spitzer rate gives with them (lnL following T(t)), and how often the
cap acts. `sbatch run_vlpl_auto.sbatch` runs VLPL with the automatic lnL for 1.5 ps (cap on / off, dt and dt / 4); the
`out_vlpl_autoh*` runs used Smilei's quantum b_min h / 2p* (`quantum_bmin = h`).

| run | 1/e time (1/omega) |
|---|---|
| Spitzer rate with Lee-More / GMS6 / BPS lnL | 756 / 865 / 944 |
| Spitzer rate with Landau-Spitzer / NRL-formulary lnL | 826 / 1078 |
| VLPL automatic lnL (hbar / 2p*), cap on: dt, dt / 4 | 949.7, 937.2 |
| VLPL automatic lnL, cap off: dt, dt / 4 | 898.2, 885.0 |
| VLPL automatic lnL with h / 2p* (Smilei), cap on: dt, dt / 4 | 1137.3, 1135.9 |
| VLPL automatic lnL with h / 2p*, cap off: dt, dt / 4 | 1097.8, 1055.5 |

How much the cap removes from the e-p exchange (sum of s p*^2): 13 % at lnL = 5, 7 % with the automatic lnL.

With the physical log the cap changes the result by 6 %, inside the 9 % between GMS and BPS. VLPL with the cap sits at BPS,
without it near GMS. Smilei's h / 2p* puts every pair on the floor lnL = 2 (20-31 % too slow). Decision 94: the cap stays on
by default, and the automatic lnL uses hbar / 2p*.
- EPOCH build (the CPU cluster): Warwick-Plasma/epoch v4.20.1 in $WORK/epoch, `make COMPILER=gfortran DEFINE=-DPARTICLE_SHAPE_BSPLINE3`
  in epoch1d (12 MPI ranks per run, 30 min; dt / 4: 1 h 54 min); the sdf reader for compare.py: `make` in SDF/C, then
  `pip install --target <dir> .` in SDF/utilities and PYTHONPATH=<dir>. `RUNS="nanbu sk nocoll nanbu_dt4"` selects the runs.
