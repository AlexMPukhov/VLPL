# Flying focus in vacuum: VLPL (side TFSF tube) against GEM-PIC-RZ and the analytic pulse

The luminal Bessel pulse of GEM-PIC-RZ's `flying_focus` block, fed into VLPL through the faces of the side TFSF tube
(`side.wave = flying_focus`, docs/specs/side_injection.md, decision 62). Scaled down from GEM-PIC-RZ
`benchmarks/flying_focus/vacuum.deck` for 3D: λ = 0.8 µm, a0 = 1 on the axis, cone 0.2 rad (core radius 1.5 µm,
transverse wavelength 4 µm), 20 fs (intensity FWHM; 1/e half length of the field 5.1 µm), 45 µm window, 0.3 mm.

| file | content |
|---|---|
| `rip_vacuum.deck` | VLPL: 1350 × 80 × 80 cells, dx = λ/24, dy = dz = 0.4 µm (λ⊥/10), CPML of 10 cells in y and z, the tube 2 cells inside, window at c from t = 0 |
| `gem_vacuum.deck` | GEM-PIC-RZ: m_max = 1, dr = 0.4 µm, r_max = 24 µm, absorber (feeding) layer 16 ... 24 µm, dζ = λ/24, dz = 0.5 µm |
| `check_rip.py` | on-axis E_y of VLPL against E_y = a0 exp(−ζ'²/L²) cos(cos θ k0 ζ' − κ ω0 t), ζ' = x − ct − x_center: the metrics of GEM-PIC-RZ's `check_axis.py` plus the fitted amplitude and carrier phase |

```
build/vlpl bench/flying_focus/rip_vacuum.deck && python3 bench/flying_focus/check_rip.py rip_ff_out
gempic_rz gem_vacuum.deck && python3 <GEM-PIC-RZ>/benchmarks/flying_focus/check_axis.py gem_ff_out 1 0.2 20 22.5
```

## Results (2026-10-31, the CPU cluster, 24 cores)

On the axis, against the analytic luminal pulse:

| z (mm) | GEM-PIC-RZ max\|E\| | GEM-PIC-RZ rms(E − ref)/a0 | GEM-PIC-RZ envelope shift (µm) | VLPL max\|E\| (ref) | VLPL rms(E − ref)/a0 | VLPL envelope shift (µm) |
|---|---|---|---|---|---|---|
| 0 | 1.0672 | 2.67e-2 | 0.000 | 1.00020 (0.99985) | 2.44e-4 | 0.0000 |
| 0.06 | 0.9968 | 0.142 | −0.046 | 1.00024 (0.99989) | 2.44e-4 | 0.0000 |
| 0.12 | 1.0141 | 0.237 | −0.132 | 0.99923 (0.99888) | 2.44e-4 | 0.0000 |
| 0.21 | 0.9991 | 0.231 | −0.139 | 0.99678 (0.99643) | 2.44e-4 | 0.0000 |
| 0.27 | 1.0037 | 0.232 | −0.137 | 0.99418 (0.99384) | 2.44e-4 | 0.0000 |

- VLPL: the incident field is the exact discrete solution, so the total field inside the tube is the incident pulse
  (test_side_ff: to 5e-15). On the axis it is the analytic pulse to 2.4e-4 of a0 at every z: fitted amplitude
  1.00036 a0, carrier phase 0 to 1e-5 rad, envelope centroid fixed to 1e-4 µm. The 2.4e-4 is the O(h²) difference of
  the discrete modes' polarisation from the continuum one used for the normalisation (largest 5.5e-3 per component).
  The field energy in the box is constant to 13 digits over 0.3 mm. (max|E| follows the analytic max, which changes
  as the carrier slides under the envelope by κ z.)
- GEM-PIC-RZ: amplitude within 1-3 %, the envelope moves 0.14 µm back in the first 0.12 mm and then stays; the
  carrier phase settles at an offset from the analytic one (rms 0.23 ≈ 0.23 rad; 0.08 rad at 50 mrad in its own
  vacuum benchmark): the numerical dispersion of the conical wave on its grid, as described there.
- The two codes therefore set up the same pulse (a0 on the axis, envelope, cone, carrier with the κ slip); VLPL holds
  it at the analytic shape to O(h²), GEM-PIC-RZ within its dispersion.
- Cost: VLPL 1795 s (8995 steps, 8.6e6 cells; field step 1218 s, of which the face corrections are a table lookup,
  setup of 10816 components = 169 frequencies × 64 azimuths); GEM-PIC-RZ 35 s.

# Flying focus in a preformed plasma: VLPL (side tube, cold-fluid shell) against GEM-PIC-RZ

The same kind of pulse in a uniform plasma of 1e18 cm⁻³ (λp = 33 µm), as GEM-PIC-RZ `benchmarks/flying_focus/plasma.deck`
scaled down for 3D: cone 0.2 rad, 30 fs (1/e half length 7.6 µm), 72 µm window with the pulse centre at 42 µm, 0.3 mm.
VLPL (decision 63): the incident pulse of the plasma through the faces of the side tube, kinetic electrons inside it
(2 × 2 per cell across, 1 along x; immobile ions), the linear cold fluid outside (`side.background_density`). GEM-PIC-RZ:
`flying_focus.background_density`, plasma in the whole box, m_max = 3 (a0 = 2) or 2 (a0 = 0.3), ppc_r × ppc_phi = 4 × 16 (a0 = 2).

| file | content |
|---|---|
| `rip_plasma.deck` | VLPL: 2160 × 80 × 80 cells (dx = λ/24, dy = dz = 0.4 µm), CPML of 10 cells, kinetic electrons 5.6 … 26 µm |
| `gem_plasma.deck` | GEM-PIC-RZ: dr = 0.4 µm, r_max = 24 µm, absorber (feeding) layer 16 … 24 µm, dζ = λ/24, dz = 0.5 µm |
| `check_plasma.py` | on the axis: the laser (fitted amplitude and carrier phase against the luminal pulse, which in the plasma is the same as in vacuum on the axis; VLPL at fixed t, GEM-PIC-RZ at fixed z) and the wake E_x / E_z smoothed over a laser period at ζ' = −6, −16, −30 µm |

```
build/vlpl bench/flying_focus/rip_plasma.deck numerics.deposit=global numerics.sort_every=0
gempic_rz gem_plasma.deck
python3 bench/flying_focus/check_plasma.py rip_ffp_out gem_ffp_out
# linear: side.a0=0.3 output.dir=rip_ffl_out; flying_focus.a0=0.3 grid.m_max=2 plasma.ppc_r=2 plasma.ppc_phi=8 output.dir=gem_ffl_out
python3 bench/flying_focus/check_plasma.py rip_ffl_out gem_ffl_out 0.3 0.2 30 42 40 0.8
```

## Results (2026-11-01, the CPU cluster, 24 cores)

Linear (a0 = 0.3; wake in m c ω0 / e, 1 = 4.0 TV/m):

| z (µm) | RIP amp / a0 | RIP phase | GEM amp / a0 | GEM phase | wake ζ' = −6 RIP / GEM | −16 RIP / GEM | −30 RIP / GEM |
|---|---|---|---|---|---|---|---|
| 30 | 1.0007 | 0.016 | 0.999 | 0.19 | 1.0e-4 / 1.3e-4 | −3.8e-4 / −3.4e-4 | 1.0e-4 / 3.4e-4 |
| 90 | 1.0012 | 0.022 | 1.031 | 0.56 | 1.0e-4 / 1.2e-4 | −3.7e-4 / −3.7e-4 | 3.6e-4 / 4.3e-4 |
| 150 | 1.0010 | 0.022 | 1.000 | 0.61 | 1.0e-4 / 1.3e-4 | −3.7e-4 / −3.2e-4 | 3.6e-4 / 3.2e-4 |
| 270 | 1.0011 | 0.022 | 1.007 | 0.61 | 1.0e-4 / 1.3e-4 | −3.7e-4 / −3.3e-4 | 3.6e-4 / 3.4e-4 |

Nonlinear (a0 = 2, the GEM-PIC-RZ plasma deck; the axis cavitates from ζ' = +6 to −12 µm in both codes):

| z (µm) | RIP amp / a0 | RIP phase | GEM amp / a0 | GEM phase | wake ζ' = −6 RIP / GEM | −16 RIP / GEM | −30 RIP / GEM |
|---|---|---|---|---|---|---|---|
| 30 | 1.0005 | 0.042 | 1.000 | 0.22 | 5.4e-4 / 5.5e-4 | −1.40e-3 / −2.18e-3 | 0.7e-3 / 2.0e-3 |
| 90 | 0.9993 | 0.081 | 1.031 | 0.64 | 4.8e-4 / 9.2e-4 | −1.35e-3 / −2.10e-3 | 1.04e-3 / 1.40e-3 |
| 150 | 0.9982 | 0.080 | 0.999 | 0.70 | 4.6e-4 / 6.4e-4 | −1.26e-3 / −1.87e-3 | 0.93e-3 / 1.07e-3 |
| 270 | 0.9984 | 0.079 | 1.003 | 0.71 | 4.6e-4 / 5.9e-4 | −1.27e-3 / −1.88e-3 | 0.97e-3 / 0.97e-3 |

- The injection: VLPL's laser on the axis is the analytic luminal pulse of the plasma (amplitude 1.001 a0, a constant
  carrier phase 0.02 rad in the linear case, the response of the kinetic plasma), and the wake is stationary from
  z = 60 µm on to two digits: the pulse enters through the faces in the plasma without a transient of its own. GEM-PIC-RZ
  keeps the amplitude within 3 % and settles at its numerical carrier phase offset (0.61 rad here, the vacuum benchmark
  above 0.23 rms at 0.2 rad).
- Linear wake: the codes agree within ~10 % (ζ' = −16 µm: −3.7e-4 against −3.2 … −3.7e-4; −30 µm 3.6e-4 against
  3.2 … 4.3e-4; GEM-PIC-RZ's axis values scatter by its particle noise); the rms difference of the wake profile over
  −40 … 10 µm is 0.18 … 0.21 of its peak (with that noise).
- Nonlinear: the same cavity and the same field inside it (ζ' = −6 µm 4.6e-4 / 5.9e-4 late in the run); behind the cavity
  VLPL's wake is ~35 % weaker (−1.27e-3 against −1.88e-3 at −16 µm), stationary in both. GEM-PIC-RZ with m_max = 1
  gave the same (−1.9e-3 … −2.1e-3), so the difference is not its mode truncation; it lies in the strongly nonlinear
  return of the electrons onto the axis (core radius 1.5 µm = 4 cells, 4 particles per cell in VLPL) and was not
  resolved further. The fluid shell is linear: at its faces (r ≈ 11 µm) the Bessel rings have a ≈ 0.4.
- Cost: VLPL 11607 s (a0 = 2) / 10058 s (a0 = 0.3) for 8995 steps of 1.4e7 cells and 4.5e7 particles (push and
  deposit 78 %, field step 17 %; `numerics.deposit = global` without sorting is 3.5 × faster here than the default tiled
  deposit, whose tile sort took 58 %); GEM-PIC-RZ 18 min (m_max = 3) / 7 min (m_max = 2).
