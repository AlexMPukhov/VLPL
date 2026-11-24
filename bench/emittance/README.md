# Emittance of a low-emittance beam: VLPL solvers against Yee and FBPIC

Phase 1 exit benchmark "NCI and emittance tests of the 2020 paper reproduced with RIP-CC vs Yee" (`docs/ROADMAP.md`;
the NCI half is `bench/nci/`). Decisions 98 and 99. Two cases, chosen with Alexander (2026-11-09):

- **A**, Fig. 4 of A. Pukhov, J. Comput. Phys. 418, 109622 (2020), Sec. 7.3, as published: a 10 GeV electron beam in
  vacuum, eps_n = 1 nm, every solver including Yee.
- **B**, a witness with eps_n = 0.1 um in a linear laser wake: VLPL-RIP, VLPL-iRIP, VLPL-RZ, VLPL-iRZ, FBPIC. Yee is
  left out: it runs only in periodic boxes (no laser, window or open x; `docs/specs/yee.md`). Yee in a wake comes after
  a Yee moving window.

The current filter is off in every run (`numerics.filter_passes = 0`, the default).

## Case A: Fig. 4 of the 2020 paper

Setup (`vacuum.deck`, `rz_vacuum.deck`, `run_vacuum.sh`, `fbpic_vacuum.py`):
- beam: 10 GeV (u = 19570), round Gaussian sigma_x = sigma_y = sigma_z = 1.41 um (cut at 4 sigma), eps_n,y = eps_n,z
  = 1 nm (Gaussian momenta, sigma_u = 7.09e-4), 6 nC (peak 0.761 n_c at lambda0 = 1 um) or 0.6 nC, with its steady
  self-field;
- grid: h_x = 0.1 um, h_y = h_z = 0.15 um (RZ: dz = 0.1, dr = 0.15 um, modes 0 and 1); 1 mm;
- box 25.6 x 19.2 x 19.2 um, periodic in y and z (RZ: wall at 9.6 um); shape 2, Boris;
- solvers:
  - Yee, c dt = 0.7 h_x as in the paper, in a periodic x box;
  - RIP-CC with periodic x (`ripp`) and in a window moving at c (`ripw`);
  - iRIP on the same grid (`irip`) and at h_x = 0.3 um, i.e. h_perp = c dt / 2 (`irip3`);
  - VLPL-RZ (`rz`) and VLPL-iRZ (`irz`, also with dr = 0.05 um = c dt / 2: `irz_dr05`);
  - FBPIC 0.27.1 on the RTX 5090: dz = 0.1, dr = 0.15 um, 2 modes, PSATD of infinite order, cubic shape, 4 M
    macro-particles, open r. Standard (`g0`) and Galilean with v_comoving = the beam velocity (`g1`).
- the self-field init: `numerics.beam_init = steady` (RIP-CC, decision 99, the default) or `gauss` (the Gauss-law
  field of decision 24; iRIP, Yee and RZ always).

What the runs needed in VLPL (decisions 98, 99; `docs/specs/beam_self_field.md`):
- the self-field init for periodic x (3D DFT) and for Yee (Yee's own Gauss law, B at x_{i+1/2});
- a co-moving neutralizing background in fully periodic boxes (`remove_mean_current`): without it, the beam's net
  current drives a uniform E_x = -<J_x> t that decelerates it;
- self_field with iRIP (RIP-CC's Gauss-law equilibrium, which iRIP conserves);
- the exact discrete steady-state init for RIP-CC (below).

```
sh bench/emittance/run_vacuum.sh build/vlpl [case ...]        # emA_<case>/beam_beam.txt (epyc: 4 min - 5 h per case)
python3 bench/emittance/fbpic_vacuum.py <Q_nC> <galilean 0|1> <out.txt> [length_mm]   # cuda: 24-33 min per run
python3 bench/emittance/table.py A <dir>                        # the table below
```
The steady runs: `numerics.beam_init=steady` (the default now); the `gauss` rows were run before it existed, with
`beam_init = gauss` behaviour. `_h2`: h_x, h_y, h_z halved (0.05 / 0.075 um, same box).

### Result (2026-11-10; VLPL branch emittance-bench)
eps_n,y in nm; the `_6` / `_06` suffix is the charge in nC (6 / 0.6). VLPL writes its first moments at step 200 (20 um).

| run | init | 0.02 mm | 0.1 mm | 0.5 mm | 1 mm |
|---|---|---|---|---|---|
| Yee, 6 nC (`yee_6`) | Gauss (Yee) | 26.4 | 155 | 792 | 2371 (0.99 mm) |
| Yee, 0.6 nC (`yee_06`) | Gauss (Yee) | 2.82 | 15.6 | 80.3 | 158 (0.99 mm) |
| RIP-CC periodic, 6 nC (`ripp_6`) | **steady** | 0.9982 | 0.9982 | 0.9982 | **0.9983** |
| RIP-CC periodic, 0.6 nC (`ripp_06`) | **steady** | 0.9982 | 0.9982 | 0.9982 | **0.9982** |
| RIP-CC periodic, 6 nC, h/2 (`ripp_6_h2`) | **steady** | 0.9988 | 0.9988 | 0.9988 | **0.9989** |
| RIP-CC periodic, 0.6 nC, h/2 (`ripp_06_h2`) | **steady** | 0.9988 | 0.9988 | 0.9988 | **0.9988** |
| RIP-CC window, 6 nC (`ripw_6`) | **steady** | 0.9982 | 0.9982 | 0.9982 | **0.9984** |
| RIP-CC window, 0.6 nC (`ripw_06`) | **steady** | 0.9982 | 0.9982 | 0.9982 | **0.9982** |
| RIP-CC periodic, 6 nC | Gauss | 12.27 | 10.63 | 10.98 | 11.19 |
| RIP-CC periodic, 0.6 nC | Gauss | 1.579 | 1.454 | 1.480 | 1.496 |
| RIP-CC window, 6 nC | Gauss | 12.27 | 11.70 | 11.71 | 11.71 |
| RIP-CC window, 0.6 nC | Gauss | 1.579 | 1.534 | 1.536 | 1.535 |
| iRIP window, 6 nC (`irip_6`) | Gauss | 0.9981 | 0.9981 | 0.9981 | 0.9983 |
| iRIP window, 0.6 nC (`irip_06`) | Gauss | 0.9982 | 0.9982 | 0.9981 | 0.9981 |
| iRIP, h_x = 0.3 um = 2 h_perp, 6 nC (`irip3_6`) | Gauss | - | - | - | 1.004 (0.96 mm) |
| RZ (RIP-CC), 6 nC (`rz_6`) | Gauss | 12.23 | 13.79 | 12.46 | 12.45 |
| RZ (RIP-CC), 0.6 nC (`rz_06`) | Gauss | 1.574 | 1.698 | 1.592 | 1.591 |
| iRZ, 6 nC (`irz_6`) | Gauss | 1.001 | 1.001 | 1.001 | 1.001 |
| iRZ, 0.6 nC (`irz_06`) | Gauss | 1.001 | 1.001 | 1.001 | 1.001 |
| iRZ, dr = 0.05 um, 6 nC (`irz_dr05_6`) | Gauss | 0.9988 | 0.9988 | 0.9988 | 0.9992 |
| FBPIC standard, 6 nC | | 11.46 | 54.4 | 275 | 574 |
| FBPIC standard, 0.6 nC | | 1.516 | 5.52 | 27.2 | 54.5 |
| FBPIC Galilean, 6 nC | | 0.9992 | 0.9992 | 0.9992 | 0.9993 |
| FBPIC Galilean, 0.6 nC | | 1.000 | 1.000 | 1.000 | 1.000 |

The initial eps_n,y of the loaded beam is 0.9982 nm (h) and 0.9988 nm (h/2) in 3D (sampling of 1 nm), 1.001 nm in RZ.

The table was run before decision 95 (main, 2026-11-10) was merged. With 95 the open-x Gauss-law field of explicit
RIP-CC is clamped per mode (1 / gamma^2 -> max(1 / gamma^2, h_x^2 K / 4)); at this gamma every mode is clamped, and the
clamped field is RIP-CC's local steady state. Reruns with the merged code (`emM_*`, `emN_*`, 2026-11-11):

| run | init | 0.2 mm | 1 mm |
|---|---|---|---|
| RIP-CC window, 6 nC | Gauss (clamped) | 0.9982 | - |
| RIP-CC window, 0.6 nC | Gauss (clamped) | 0.9982 | - |
| RZ (RIP-CC), 6 nC | Gauss (clamped) | 1.0006 | 1.0009 |
| RZ (RIP-CC), 0.6 nC | Gauss (clamped) | 1.0006 | 1.0006 |
| iRIP window, 6 nC, clamp applied | Gauss (clamped) | 11.73 | - |
| iRZ, 6 nC, clamp applied | Gauss (clamped) | 12.15 | 12.46 |
| iRIP window, 6 / 0.6 nC, unclamped (final) | Gauss | 0.9981 / 0.9982 | - |
| iRZ, 6 / 0.6 nC, unclamped (final) | Gauss | 1.0005 / 1.0005 | 1.0007 / 1.0005 |

So the clamp cures explicit RIP-CC at this gamma (window and RZ) and spoils the implicit solvers, whose steady state is
the unclamped field. The clamp is therefore applied for explicit RIP-CC only (Alexander, 2026-11-11). The steady init stays
the default: it is exact at every gamma and with periodic x, where nothing is clamped; at lower gamma the clamped Gauss-law
field still differs from the steady state (test B9, in a window: 1.0e-2 of max|F| at u = 5, 5.9e-3 at 20, 8.8e-4 at 60,
3.2e-12 at 1e6).

- **Yee** reproduces the paper: a fast jump from the staggered field interpolation, then growth as the bunch becomes
  unstable (numerical Cherenkov): 155 nm at 0.1 mm and 2.4 um at 1 mm for 6 nC, 10 times less for 0.6 nC.
- **RIP-CC with the steady init keeps the emittance**: 0.9983 nm (6 nC) and 0.9982 nm (0.6 nC) at 1 mm, from 0.9982 nm;
  at h/2 0.9989 and 0.9988 nm from 0.9988 nm. The change over 1 mm is 1e-4 of eps_n or less, periodic and in the window.
- **iRIP and iRZ keep the emittance** with the Gauss-law init: 0.998-1.004 nm at 1 mm, also at h_perp = c dt / 2.
- **RIP-CC with the Gauss-law init jumps once** in the first ~20 um, to 12.3 nm (6 nC) and 1.6 nm (0.6 nC), then stays
  flat (11.2-12.5 nm and 1.5-1.6 nm at 1 mm, 3D and RZ alike).
  - The jump scales with the charge and as h^2: at h_x = 0.05 um the excess was 2.3 nm instead of 11.3 nm (x 4.8).
  - Cause: the Gauss-law field (discrete Gauss law with B = beta x E) is not RIP-CC's discrete steady state; they differ
    by O(h^2) (B3: 1e-2 at h_x = 0.1). The mismatch is a small fraction of the beam's own E_r, which is not suppressed by
    1/gamma^2, so its kick is many times sigma_u = 7e-4. RIP-CC moves forward waves exactly one cell per step, so the
    mismatch travels with the beam and does not leave.
  - A rigid warm-up (`species.rigid_until = 100 fs`: momenta frozen, current deposited) only reduced the jump to
    1.2-4.4 nm, oscillating.
  - The cure is the steady init (decision 99): per Fourier mode, the field that one RIP-CC step leaves unchanged.
  - RZ has no steady init yet (its step differs per radial operator): VLPL-RZ starts with the Gauss-law field, which with
    decision 95's clamp is stationary at large gamma (1.0009 nm, table above). A steady init for RZ (and moderate gamma)
    and a tail-free init for iRIP / iRZ in a window from t = 0 are open items.
- **FBPIC**: Galilean with v_comoving = beam velocity keeps 0.9993 nm (6 nC). Standard PSATD grows linearly with
  distance and charge, to 574 nm (6 nC) and 54.5 nm (0.6 nC) at 1 mm.
- The beam in the moving window (`ripw`, `irip`, `irip3`) loses energy, 10 -> 9.1 GeV over 1 mm (6 nC; 9.9 GeV at
  0.6 nC). This is the sheet field behind a net charge in a periodic transverse box (`docs/specs/beam_self_field.md`):
  uniform across the slice, so it does not change eps_n. It is absent in the periodic-x boxes (neutralizing background)
  and in RZ (wall).

## Case B: 0.1 um witness in a linear wake

Setup (`wake.deck`, `rz_wake.deck`, `run_wake.sh`, `fbpic_wake.py`):
- laser a0 = 1, lambda0 = 0.8 um, 10 fs, waist 4 um, in a matched parabolic channel, n_e = 0.01 n_c on the axis
  (dn/n0 = 0.405 at r = w0; cut at r = 6 um in 3D), 7 um up-ramp, window at c from 45 fs;
- witness: 10 MeV, loaded at t = 150 fs at x = 34.3 um (4.5 um behind the pulse centre) with its self-field,
  sigma_z = 0.3 um, sigma_r = 0.30 um (matched to the focusing gradient, from a scan of 0.30 / 0.35 / 0.40 um),
  eps_n = 0.1 um, 0.94 pC (nominal 1 pC, cut at 3 sigma);
- 3D: 320 x 160 x 160 cells, h_x = 0.05 um, h_perp = 0.075 um, 12 um periodic transverse box (channel cut at 6 um);
  `irip_dx01` with h_x = 0.1 um;
- RZ: nz 320, nr 120, dz = 0.05, dr = 0.075 um, modes 0 and 1; `irz_dr025` with dr = 0.025 um;
- FBPIC: the same RZ grid, standard (`g0`, also dr = 0.025 um) and Galilean (`g1`, v_comoving = 0.99995 c).

```
sh bench/emittance/run_wake.sh build/vlpl [case ...]          # emB_<case>/beam_beam.txt
python3 bench/emittance/fbpic_wake.py <galilean 0|1> <dr_um> <out.txt>
python3 bench/emittance/table.py B <dir>
```

### Result (2026-11-10)
eps_n,r of the witness in um; z is the distance of the laser from the start of the run.

| run | 0.05 mm | 0.1 mm | 0.2 mm | 0.25 mm | 0.3 mm | 0.4 mm | E at 0.5 mm [MeV] |
|---|---|---|---|---|---|---|---|
| RIP 3D | 0.0977 | 0.1022 | 0.1081 | 0.1096 | 0.123 | 0.1418 | 21.8 |
| iRIP 3D | 0.0977 | 0.1018 | 0.1047 | 0.1064 | 0.1181 | 0.1258 | 21.5 |
| iRIP 3D, h_x = 0.1 um | 0.0982 | 0.1018 | 0.1015 | - | 0.1069 | 0.1059 | 21.0 |
| RZ | 0.0973 | 0.1031 | 0.1074 | 0.1085 | 0.1265 | 0.3434 | 21.1 |
| iRZ | 0.0973 | 0.1025 | 0.1055 | - | 0.1207 | 0.3328 | 20.6 |
| iRZ, dr = 0.025 um | 0.0976 | 0.1031 | 0.1063 | - | 0.1221 | 0.3383 | 20.6 |
| FBPIC standard | 0.1001 | 0.1062 | 0.1234 | - | 0.1615 | 0.3758 | 20.1 |
| FBPIC standard, dr = 0.025 um | 0.0999 | 0.1062 | 0.1231 | - | 0.161 | 0.3749 | 20.1 |
| FBPIC Galilean | 0.1002 | 0.1058 | 0.119 | - | 0.1502 | 0.3469 | 20.2 |

- The witness is accelerated from 10 MeV to 31.7 MeV at z = 0.32 mm (RZ; 30.6 MeV in 3D), then dephases and decelerates.
- **Focusing phase (z <= 0.25 mm)**: eps_n grows from 0.097 to 0.105-0.111 um in all VLPL solvers (RZ 0.1054 at
  0.23 mm, RIP 0.1106, iRIP 0.1066), converged in dr (iRZ, dr/3: within 1 %) and h_x (iRIP). FBPIC standard grows
  faster, 0.123 um at 0.2 mm (Galilean 0.119), and does not change with dr.
- **After dephasing (z > 0.3 mm)** the witness slides into the defocusing phase: in RZ and FBPIC eps_n blows up to
  0.33-0.38 um at 0.4 mm (0.78 um at 0.5 mm in RZ); in 3D it stays at 0.13-0.14 um. The charge is kept in all runs
  (0.944 pC). This difference is not investigated: it is physics of the defocusing phase in different transverse
  boxes (3D periodic 12 um box with the channel cut at 6 um vs RZ wall), not numerical emittance growth. Case B is
  reported for the focusing phase.
- In case B the witness carries 0.94 pC and its self-field is small next to the wake; the init (steady for RIP 3D,
  Gauss for the others) does not show.

## Regression (ctest-sized)
`build/tests beam`:
- B6: the case A beam scaled down (sigma = 1 um, 2 nC, 96 x 48 x 48 cells, 15 um, periodic box) for RIP-CC, iRIP and
  Yee. Measured growth of eps_n,y: RIP-CC (steady init) -3.0e-6, iRIP -2.0e-5, Yee +19.8 (with the Gauss-law init
  RIP-CC was +6.35). Passes if |RIP-CC| < 1e-2, |iRIP| < 1e-2 and Yee > 1.
- B7: the steady init is stationary under one real RIP-CC step (rigid beam, u = 1e6): 6.0e-14 of max|F|, periodic and
  in a window.
- B9: Gauss-law vs steady init in a window at u = 5, 20, 60, 1e6 (numbers above).
- B5 / M5 (decision 95) with iRIP / iRZ: the Gauss law over 12 window shifts at 2.6e-14 / 1.9e-14.
- B8: the self-field init with periodic x, iRIP and Yee: Gauss law at round-off, near field steady.
- RZ M6 (`build/tests rz_metal`): a witness loaded at t_start into an RZ window with plasma, RIP-CC and iRIP.

## Files
- `vacuum.deck`, `rz_vacuum.deck`, `run_vacuum.sh`, `fbpic_vacuum.py`: case A.
- `wake.deck`, `rz_wake.deck`, `run_wake.sh`, `fbpic_wake.py`: case B.
- `wake_probe.py`, `wake_time.py`: E_z and the focusing gradient of an RZ run on the axis.
- `table.py`: the tables.
- Raw data: the CPU cluster `$WORK/emruns` (emA_*, emS_* (steady), emB_*, fbpic/).
