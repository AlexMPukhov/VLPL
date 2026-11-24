# O2: the field leaving through the open x edges (`output.xflux`)

Status: implemented 2026-10-29 (decision 56), roadmap track O, item O2 (`docs/specs/overdense.md`).

## Idea
The RIP transport splits the transverse fields into the components moving along +x and -x:
T+_y = E_y + B_z, T-_y = E_y - B_z, and for the other pair E_z - B_y (+x) and E_z + B_y (-x) (RZ, per mode:
E_r +- B_phi, E_phi -+ B_r). A step moves each by exactly one cell. At an open left edge the component moving
along -x is the field reflected by the target, also while the laser enters there (the laser lives only in the
inbound component of the guard layer). At the right edge the component moving along +x is the transmitted field.

## What is written
- Every step, the energy each edge lets out: a pure outbound wave has E^2 + B^2 = T^2 / 2 per pair, so the
  energy of the edge cell, T^2 / 4 per point times the cell volume (Cart3D h_x h_y h_z; RZ dz 2 pi A_j or B_j with the
  mode weights 1, 2 of `FieldsRZ::energy`), leaves during the next step. `dir/xflux.txt` (rank 0, every
  `output.every` steps): step, t, P_left, P_right (energy per unit time) and E_left, E_right (the sums since the
  start). W_field + W_kin + E_left + E_right is the energy balance of Gate O (absorbed + reflected + transmitted).
- With `output.xflux_every = n` (n > 0) also the components themselves, every n steps after the field step:
  `dir/xflux/left_r<rank>.bin`, `right_r<rank>.bin` (float64 records [2][nz][ny] of the rank's part of the edge
  plane; RZ [2][slots][nr]) and `meta_r<rank>.txt` (layout, offsets, dt, the radii and the energy weights in RZ).
  `tools/xflux.py dir/xflux --edge left` assembles the plane and writes the spectrum dW/domega (harmonic order
  omega / omega0) and the transverse profile of each harmonic.

With absorbing x boundaries (O3, `docs/specs/x_absorber.md`) the measuring planes are the inner faces of the layers
(global cells L and N - 1 - L) instead of the edge cells, and two more columns are written: Enet_left, Enet_right,
the outbound energy minus the inbound energy of the cell beyond the plane (which enters during the next step), i.e.
the Poynting flux through the plane, exact also for oblique waves. Columns of xflux.txt (since 2026-11-03 also E_abs_r, W_in,
E_part_out, E_inj, see below): step, t, P_left, P_right,
E_left, E_right, Enet_left, Enet_right; the meta file of the records has `planes gL gR`.

Deck: `output { xflux = true }` (energies) or `output { xflux_every = 1 }` (energies and records).
Not with a moving window (refused): a wave along -x crosses two cells per step relative to the grid there.
After a restart E_left and E_right count from the restart; records are appended.

## Energy balance columns (2026-11-03, CHF benchmark, docs/specs/harmonic_focus.md)
Three more columns: E_abs_r (the energy the radial absorber of RZ removed between the measuring planes: W there before
minus after the damping, every step; since 2026-11-03 only there, see below), W_in (the field energy between the measuring
planes, i.e. without the x absorbers, whose content Enet has already counted as gone) and E_part_out (the kinetic energy w m (gamma - 1) of the particles removed at open edges or the RZ
wall, summed over the species in the migration). W_in + W_kin + Enet_left + Enet_right + E_abs_r + E_part_out is the
energy balance of Gate O.

The laser through an open left edge (2026-11-03, fixed): the inflow hooks write it into the guard layer inside the field step, and
after the step the guard no longer holds it, so Enet_left had counted no inbound energy (E_left = Enet_left exactly, Cart3D and
RZ). xflux now adds one more inflow hook, right after the laser hooks, that measures the inbound component in the guard (the
energy entering cell 0 during this step) and counts it as inbound at the left plane in the same step; column 12, E_inj, is its
sum: the laser energy. With an RZ TFSF plane the hook sits on the incident grid and E_inj is the energy injected there (the
left plane is then the face of the left layer). Test R3 (`build/tests test_rz_tfsf`): W_in + Enet_left + Enet_right (+ E_abs_r)
= 0 within 2.4e-4 of E_inj (RZ beam, O(dz^2)) and 3.4e-13 (Cart3D plane pulse) at every step.

E_abs_r in the corners (2026-11-03, fixed): the radial damping was summed over the whole z range, so light that had crossed a
measuring plane (counted in Enet) and was then damped by the radial absorber inside a z layer was counted a second time. Now
E_abs_r is W between the planes before minus after the damping (FieldsRZ::energy(i0, i1)). Test R4 (`build/tests
test_rz_tfsf`): a strongly diverging vacuum beam (waist 0.4 um) that reaches the corner of the right z layer and the radial
absorber: W_in + Enet_left + Enet_right + E_abs_r constant within 2.7e-4 after the pulse is in (before: +2.9e-2). R1 against
E_inj is -5.7e-4 now (before -4.2e-4, partly compensated by the corner): E_inj is measured where the laser enters the incident
grid, whose own radial absorber takes 3e-4 of it before the plane (-2.5e-4 with nr = 128 or 256).

## Exactness
- Plane waves (no transverse variation): the transport is an exact shift, the balance holds to round-off
  (4e-13, RIP-CC and iRIP).
- Transverse structure: the longitudinal fields and the transverse sources of the edge half cell are not in
  T^2 / 4; the balance holds to a discretisation term that falls as dz^2 (RZ Gaussian beam, waist 3 um:
  1.1e-4 .. 1.6e-4 at lambda0/16, 2-3e-5 at lambda0/32), well below the 1e-3 of the Gate O criterion.

## Tests (`tests/test_xflux.cpp`, `tests/mpi.sh`)
- X1 plane pulse in vacuum through the box: W + E_left + E_right constant to 4e-13 (RIP-CC, iRIP), nothing out
  through the left edge (1e-28).
- X2 RZ Gaussian beam: balance to 1.6e-4 (bound 3e-4).
- X3 normal incidence on a cold overdense step (9 n_c, a0 = 1e-3, electrons on immobile ions): reflected energy =
  incident (the right edge of the same run without plasma) to 1e-6, transmitted 6.5e-11, phase of the reflected
  field at omega0 -2.4640 against the Fresnel phase -2 atan(sqrt(n/n_c - 1)) = -2.4619 (25 n_c: -2.7422 against
  -2.7389); the difference is the half-cell position of the edge-cell field.
- X4 refused with a moving window.
- mpi.sh: E_left, E_right of 1 and 8 ranks (Cart3D, the edge plane over 4 ranks; RZ laser through open z).

## Next (track O)
O1 (TFSF, docs/specs/tfsf.md) and O3 (x absorbers, docs/specs/x_absorber.md) are done; at an open edge the
outbound component of an oblique wave is not the whole outgoing field: use the absorbing boundary and Enet there.
