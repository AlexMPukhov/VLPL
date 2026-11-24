# Oblique Fresnel reflection (track O, after O1-O3)

Status: verified 2026-10-31 (decision 60), `build/tests test_fresnel` (tests/test_xflux.cpp, F1).

## Physics and equations
A plane wave at angle theta to x falls on a cold plasma step (electrons on immobile ions, a0 = 1e-3: linear response)
with eps = 1 - n / n_c. In the e^{-i omega t} convention, with k = sqrt(eps - sin^2 theta) (Im k >= 0):
- s (E along z, `polarization = z`): r = (cos theta - k) / (cos theta + k);
- p (E in the plane of incidence, `polarization = y`), as the ratio of the outbound E_y-type components that O2 records:
  r = (k - eps cos theta) / (k + eps cos theta).
Overdense (9 n_c): |r| = 1, only the phase is tested. Underdense (0.5 n_c at 0 deg, 0.25 n_c at 30 deg): |r| and phase.

## The plane wave of one k_y (`laser.waist = 0`)
A Fresnel coefficient belongs to one k_y, so a finite beam (a spectrum of angles) is not the clean test. With
`waist = 0` the TFSF laser (O1, docs/specs/tfsf.md) is a plane wave of k_y = k0 sin(angle):
a(t, x, y) = a0 g(t_e) sin(-(cos a (x - x_s) + sin a (y - y_c)) - t_c + phase), t_e = -(x - x_s) / cos a - t_c,
with g the envelope of the laser (Gaussian with the compact-support taper): the envelope moves along x at the group
velocity cos a, the phase fronts at the angle a. It needs periodic y with sin(angle) n_y dy = m lambda0 (checked; a
wave of one k_y cannot meet a CPML in y), a TFSF plane, no waist_z or Laguerre-Gauss mode, and like every TFSF laser
|angle| <= 45 deg (decision 60: steeper lasers are to enter from a side boundary).

## Measurement
- Absorbing x boundaries (O3, default 4 lambda0); O2 records the outbound components at the inner faces of the layers
  every step (`output.xflux_every = 1`). Left of the TFSF plane only the reflected wave is present.
- r(omega0) = A_left(plasma run) / A_right(the same run without plasma), A = sum over steps q of the record projected on
  exp(-i k_y y_j) times exp(i omega0 t_q). The surface lies half way between the two measuring planes, so the vacuum
  propagation phases of the two paths are equal and cancel exactly, whatever the numerical k_x.
- A wave of fixed k_y has a cutoff at omega = c k_y: its spectral components near the cutoff travel slowly along x and
  give a long tail (in an underdense slab also near the cutoff sqrt(omega_p^2 + c^2 k_y^2) of the transmitted wave). The
  pulse is therefore 12 fs (6 fs left 37 % of the peak in a 65 fs record at 45 deg, and at 0.5 n_c, 30 deg the cutoff
  of the transmitted wave, 0.87 omega0, is too close for any affordable record: that case was replaced by 0.25 n_c,
  cutoff 0.71 omega0). The records are 130 fs (overdense) and 160 fs (underdense); the test requires the last 40 steps
  below 2e-2 of the peak (seen: <= 1.1e-2).
- The underdense slab runs into the right absorber: no rear face, no echo in the record (with a rear face 6.6 um behind
  the surface the echo arrived 66 fs after the direct reflection at 0 deg).

## Results (RIP-CC, lambda0 = 0.8 um, dy = lambda0 / (32 sin theta), 12 fs)

| case | lambda0/32: d\|r\| | darg (rad) | lambda0/64: d\|r\| | darg (rad) |
|---|---|---|---|---|
| 9 n_c, 30 deg, s | -1.1e-4 | -1.30e-2 | -1.1e-4 | -1.13e-3 |
| 9 n_c, 45 deg, s | -8.1e-5 | -1.08e-2 | -7.9e-5 | -6.6e-4 |
| 9 n_c, 30 deg, p | +1.5e-4 | +4.88e-2 | +1.4e-4 | +3.04e-2 |
| 9 n_c, 45 deg, p (lambda0/32: 160 fs record) | +7.8e-4 | +1.29e-1 | +1.2e-4 | +7.37e-2 |
| 0.5 n_c, 0 deg (\|r\| 0.1716) | -1.2e-3 rel. | -8.7e-4 | -7.6e-4 rel. | -1.08e-3 |
| 0.25 n_c, 30 deg, s (\|r\| 0.1010) | -1.26e-2 rel. | -2.1e-3 | -1.32e-2 rel. | -2.2e-3 |
| 0.25 n_c, 30 deg, p (\|r\| 0.0424) | +1.9e-2 rel. | -1.31e-2 | +1.5e-2 rel. | -6.0e-3 |

- s, overdense: the phase error falls by 12-16 from lambda0/32 to lambda0/64 (second order or better); |r| = 1 to 1e-4.
- p converges only at first order in the phase (factor 1.6-1.8; at 45 deg the error is 0.13 rad at lambda0/32): the
  transition layer of the deposited step, next section. The roadmap (O7) expects second order only for s-pol.
- 0.25 n_c, 30 deg: the |r| deficit does not change with dx because it comes from dy, which the scan keeps at 32 cells
  per transverse period lambda0 / sin theta: d|r| = -2.23e-3, -1.27e-3, -5.8e-4 for 24, 32, 48 cells (second order
  in dy to 2 %; the transverse numerical dispersion shifts k_x differently in vacuum and plasma). Its phase error,
  -2.1e-3, is mostly the absorber reflecting the transmitted wave (at 45 deg in the plasma): with an 8 lambda0 layer
  +3e-4 (|r| unchanged, -1.20e-3).

## The p error: the transition layer of the deposited step (2026-11-01)
The deposition spreads the charge of the particles next to the surface over the neighbouring nodes, so the plasma that
the field sees is a layer of about one cell in which eps passes from 1 to 1 - n / n_c (through 0). For a thin layer the
reflection changes at first order in its thickness d for p, through the integral of (eps - 1)(eps - eps_2) / eps across
it, and only at second order for s when the reference surface is the middle of the layer (Drude; J. Lekner, Theory of
Reflection, ch. 3). The exact reflection of a linear ramp of width d (characteristic matrix integrated below the pole
eps = 0, i.e. the limit of vanishing collisions; `research/fresnel_layer/exact_ramp.py`), referred to its middle:

| 9 n_c, arg(r / r_step) | d = lambda0/32 | lambda0/64 | lambda0/128 |
|---|---|---|---|
| s, 30 / 45 deg | -1.49e-2 / -1.24e-2 | -3.9e-3 / -3.2e-3 | -1.0e-3 / -8e-4 |
| p, 30 / 45 deg | +4.13e-2 / +1.131e-1 | +2.63e-2 / +6.45e-2 | +1.46e-2 / +3.44e-2 |

The measured errors of the step (surface on a node, previous section) are of this size and sign. (An earlier estimate in
this spec, several times larger, came from layers that did not resolve the pole and is withdrawn.) The layer predicts a
small resonance absorption for p (|r| 3-6 % low at one cell) that the runs do not show (|r| = 1 to 1e-4): with one or two
nodes inside the layer no node sits at eps = 0, so only the principal-value part of the integral acts.

The layer depends on where the surface sits between the nodes. With the surface half way between nodes the deposited
profile is nearly symmetric about it and most of the first-order term cancels. RIP-CC, 9 n_c, dy refined with dx
(n_y = 32, 64, 128 cells per transverse period lambda0 / sin theta), arg(r / r_F) in rad, r referred to the surface:

| case | surface on a node: lambda0/32, /64, /128 | half way between nodes: lambda0/32, /64, /128 |
|---|---|---|
| s, 30 deg | -1.30e-2, -3.58e-3, - | -1.29e-2, -3.55e-3, - |
| s, 45 deg | -1.10e-2, -3.02e-3, - | -1.07e-2, -2.99e-3, - |
| p, 30 deg | +4.88e-2, +3.06e-2, +1.70e-2 | -3.7e-3, +3.2e-3, +3.0e-3 |
| p, 45 deg | +1.30e-1, +7.45e-2, +3.99e-2 | +1.42e-2, +1.34e-2, +8.6e-3 |

|r| = 1 to 1e-4 in all cases. s does not depend on the position. p on a node approaches first order (factor 1.8-1.9
between the last two); half way between nodes the error is 5-10 times smaller but not second order. A surface tilted
against the grid crosses all sub-cell positions, so its error lies between the two columns. (The s values at lambda0/64
differ from the previous section, where dy was kept at 32 cells per period.)

### A resolved ramp: resonance absorption and shape-1 grid heating
A ramp 0 -> 9 n_c over 0.2 um (lambda0/4), 45 deg and 30 deg: the exact |r| is 0.770 and 0.861 for p (resonance
absorption at eps = 0), 1 for s. s converges at second order (phase -7.1e-3, -1.8e-3 at lambda0/32, /64).
- p with shape 1 is unstable: after the reflected pulse the total energy grows exponentially (about 0.04 omega0, a factor
  600 at lambda0/32 and 6000 at lambda0/64 by 130 fs). Boris instead of Higuera-Cary: the same; iRIP: also unstable;
  ppc 8 x 4 instead of 2 x 1: slower (x 38); shape 2: stable (the energy decays). The resonance drives plasma oscillations
  that phase-mix and break in the gradient; the velocity spread with lambda_D << dx feeds the finite-grid instability of
  shape 1. Overdense targets with a pre-plasma need shape >= 2 (track O4).
- p with shape 2: |r| - |r_exact| = -0.13, +0.06, -0.03 and phase -0.45, -0.03, -0.05 rad at lambda0/32, /64, /128 (45
  deg; 30 deg similar). Not a convergence test yet: the phase-mixing resonance keeps radiating at omega0 at the end of the
  130 fs record (5-20 % of the peak), so the record needs to be much longer, or the plasma warm.

## Deck interface
```
laser { waist = 0 um, tfsf_x, angle (|angle| <= 45 deg), duration, delay, polarization = y|z }
grid  { boundary_x = absorbing, periodic y with sin(angle) ny dy = m lambda0 }
output { xflux_every = 1 }
```

## Tests
- Fast (`build/tests test_fresnel`, about 11 min on 24 cores): lambda0/32, 9 n_c at 45 deg s (|d|r|| < 1e-3,
  |darg| < 2e-2), 9 n_c at 30 deg p and at 45 deg p with the surface half way between nodes (< 1e-3; < 1e-2 and
  < 3e-2), 0.5 n_c at 0 deg (relative 3e-3, 5e-3), 0.25 n_c at 30 deg s (relative 2e-2, 1e-2), tails < 2e-2.
- `tests/mpi.sh`: the plane wave at 30 deg from examples/tfsf_oblique.deck onto the 4 n_c slab, 1 vs 8 ranks (fields
  and xflux records).

## References
Born and Wolf, Principles of Optics, sec. 1.5 and 1.6 (characteristic matrix); J. Lekner, Theory of Reflection
(Springer), thin-layer expansion; V. L. Ginzburg, The Propagation of Electromagnetic Waves in Plasmas (resonance
absorption); C. K. Birdsall and A. B. Langdon, Plasma Physics via Computer Simulation (finite-grid instability); A. Taflove and S. Hagness, Computational Electrodynamics (staircased
dielectric interfaces).
