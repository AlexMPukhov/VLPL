# O3: absorbing layers at the x ends (`grid.boundary_x = absorbing`)

Status: implemented 2026-10-30 (decision 59), roadmap track O, item O3 (`docs/specs/overdense.md`).

## Why
At an open x edge the RIP transport lets the outbound component T = E -+ B (one cell per step) out exactly, but
an oblique wave also has the inbound component, which the edge refills from the guard (zero): a wave at angle
theta to x is reflected with |r| ~ (1 - cos theta) / (1 + cos theta) (0.07 at 30 deg, 0.17 at 45 deg in field).
Oblique-incidence harmonics (track O) need the reflected beam to leave the box at any angle.

## Choice
Option B of the draft (Alexander): graded damping layers of L cells at both x ends, not a CPML along x.
- A CPML along x needs psi memories in the x transport of RIP-CC and in the implicit box solve of iRIP (its
  eigenbasis along x); the damping layer needs nothing in either solver: it is one multiplication after the step.
- The cost of the layers is their thickness. Damping E and B alike is matched only at normal incidence: an oblique
  wave sees an impedance step (the continuum layer has k_x' != n k_x for k_y != 0, which a PML avoids), and only a
  slow ramp keeps it small. The default thickness is therefore 4 lambda0 per side (a CPML would need 8-16 cells).
  Scan at 45 deg (RIP-CC, lambda0/32, R0 = 1e-4, reflected field over the peak field):

  | right x boundary | open edge | 32 cells (1 lambda0) | 64 cells (2 lambda0) | 128 cells (4 lambda0) |
  |---|---|---|---|---|
  | reflected field | 1.4e-1 | 9.5e-2 | 1.4e-2 | 5.6e-4 |

  With 32 cells: 1.1e-3 at 0 deg, 3.7e-2 at 30 deg (RIP-CC and iRIP the same to the printed digits).
- The laser enters through the TFSF plane (O1, `laser.tfsf_x`), at any angle: the incident grid of the TFSF
  source has the same layers, so the incident beam is absorbed in the same way left and right and the
  correction at the plane stays exact. A laser through the left edge (or a laser file, a moving window) is
  refused with absorbing x boundaries.

## Discretisation
After every field step (RIP-CC, iRIP; not Yee) all six field components are multiplied by
f(s) = exp(-kappa s^2), s = (L - i) / L for the global cell i < L, s = (i - (N - 1 - L)) / L for i > N - 1 - L
(0 inside). The factor depends only on the global x index, so the guard copies get the factor of their originals
(no exchange) and the result does not depend on the decomposition.

kappa: a wave crossing the layer at normal incidence spends a = L dx / dt steps in it and is damped by
exp(-kappa a / 3); in and out again: exp(-2 kappa a / 3) = R0, so kappa = 1.5 ln(1 / R0) / a
(`grid.x_absorber_R0`, default 1e-4, as the radial layer of RZ, decision 49). An oblique wave stays longer and is
damped more. The reflection from the gradient of the profile is the limit: it falls with L (smooth at s = 0).

The layers and one cell more are excluded from the Gauss and div B checks (the damping does not conserve charge
there; there is no plasma in them in the decks of track O). Particles are not touched: keep the plasma out of the
layers.

## O2 with layers
The measuring planes of `output.xflux` move to the inner faces of the layers (global cells L and N - 1 - L). Next
to the outbound energy E_left, E_right (exact for normal incidence) xflux.txt has Enet_left, Enet_right: outbound
minus the inbound energy of the cell beyond the plane, which enters during the next step. With the inbound part
coming back out of the layer (oblique waves) this is the Poynting flux through the plane, the right number for an
energy balance at any angle. Columns: `step t P_left P_right E_left E_right Enet_left Enet_right`; the meta file
of the records has a line `planes gL gR`.

## Deck interface
```
grid { boundary_x = absorbing   # open with damping layers at both ends (Cart3D; RZ: boundary_z, z_absorber_*)
       x_absorber_width = 4 lambda0   # thickness of each layer (default 4 lambda0; L = ceil(width / dx))
       x_absorber_cells = ...   # or L directly, in [2, nx / 3]
       x_absorber_R0 = 1e-4 }   # round-trip amplitude factor at normal incidence
laser { tfsf_x = ..., angle = ... }   # the only laser source with absorbing x
```
The laser must start between the left layer and the TFSF plane: a warning reports an initial incident field in
the layer (above 1e-4 of the peak).

## RZ (2026-11-03, decision 68)
`grid.boundary_z = absorbing` in geometry = rz: the same layers along z (`z_absorber_width` default 4 lambda0,
`z_absorber_cells`, `z_absorber_R0`), the profile and kappa of the x layers, applied to all six components of every mode at
the end of FieldsRZ::step (rz_absorb_z). The laser must come in through the RZ TFSF plane (docs/specs/tfsf.md "RZ"), whose
incident grid has the right layer only. xflux measures at the inner faces of the layers; the Gauss and div B checks leave
out the layers and one cell more. Test (`build/tests test_rz_tfsf`, R2): the focused beam of R1 (waist 1 um, 2 modes,
lambda0/32) into a 1 um layer (40 cells) against a box 10 um longer: reflected field 8.2e-4 of the peak (Cart3D at 0 deg with
1 lambda0: 1.1e-3).

## Tests (`tests/test_tfsf.cpp` T2 = `build/tests test_x_absorber`, `tests/mpi.sh`)
- T2: a focused beam (waist 1.5 um, 6 fs, lambda0 / 16, default layers of 4 lambda0 = 64 cells) from a TFSF plane at
  13 um into the right layer, RIP-CC at 0, 30, 45 deg and iRIP at 45 deg. The reflection is the largest difference of
  the fields to the same run in a box 15 um longer (whose layer is too far away for anything to come back in time),
  over the cells left of the layer and over time, relative to the peak field: below 2e-3. Also no leakage across the
  TFSF plane (1e-13 of the peak) and nothing out through the left measuring plane (|Enet_left| < 1e-12 Enet_right).
  Results: reflected field 3.9e-5 (0 deg), 1.3e-4 (30 deg), 4.3e-4 (45 deg) for RIP-CC, 4.4e-4 for iRIP at 45 deg;
  leakage 6e-16, Enet_left 4e-31 of Enet_right (12.6 min on 24 cores).
- mpi.sh: the TFSF slab deck at 30 deg with absorbing x (32 cells), diag.txt and xflux.txt of 1 and NP ranks.

The energy left in the incident grid is no measure of the layer: the focused beam as set up on the grid has a small
discrete div B, a static Bx that stays where the pulse started (left of the plane, it never reaches the main grid).

## References
Decision 49 (the radial absorbing layer of RZ, same profile); docs/specs/tfsf.md (O1); docs/specs/xflux.md (O2).
