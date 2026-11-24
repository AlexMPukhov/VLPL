# Embedded perfect conductors in RZ (block `rings`), the RZ self-field, the plane probe

Status: implemented (decision 83), RIP-CC only.

Motivation: full-Maxwell runs of relativistic bunches through corrugated and disk-loaded structures (metal rings,
optionally closed by an outer wall), including backward waves.

## Physics and equations
Perfect conductor: tangential E = 0 and normal B = 0 on the surface, all fields 0 inside; the normal E and the
tangential B on the surface are the surface charge and current. The rings are fixed in the lab frame (they move through
a moving window). Particles that enter the metal are absorbed.

## Discretisation
Staircase on the RIP-CC grid (all fields at integer z columns, rho at i + 1/2; docs/specs/rz_modes.md).
- **Geometry.** Ring k occupies the closed index rectangle [IA_k, IB_k] x [J, K]: J, K the radial nodes nearest to
  r_in, r_out (K = nr if r_out reaches the wall), IA_k = round((z_start + k p) / dz), IB_k = round((z_start + k p + p - g)
  / dz) (cut at z_end) in absolute cells (`Domain::cell0`, so with a moving window too). Rings without a vacuum column
  between them merge (gap = 0: one tube from z_start to z_end). A ring must be at least one cell thick.
- **Column types.** Vacuum, interior (both neighbours metal), left face (vacuum at I - 1), right face (vacuum at I + 1).
- **Points.** Node points (E_z, E_phi, B_r at r_j), half-node points (E_r, B_z, B_phi at r_{j+1/2}).
  - Interior columns, nodes J..K and half nodes J..K-1: all fields 0.
  - Face columns, the nodes J (if J > 0) and K: all 0 (they lie on the r surfaces, where E_z, E_phi are tangential and
    B_r normal).
  - Face columns, nodes J < j < K: E_phi = 0 (tangential), E_z from the corrector (the normal field: surface charge),
    B_r from the characteristic arriving from the vacuum side.
  - Face columns, half nodes J..K-1: E_r = 0, B_z = 0 (normal), B_phi from the characteristic.
- **Reflection at a z face.** The RIP transport moves T+ = E_r + B_phi, E_phi - B_r along +z and T- = E_r - B_phi,
  E_phi + B_r along -z by one cell per step. At a left face the arriving T+ (computed from the vacuum column I - 1 with
  its sources) sets B_phi = T+_y and B_r = -T-_z(E_phi - B_r) with E_r = E_phi = 0, so the outgoing T- = -T+: total
  reflection, exact for normal incidence. Right face: B_phi = -T-_y, B_r = T+_z.
- **Where.** Inside `FieldsRZ::pass`: the predictor E_z*, B_z* (zero where E_z or B_z is), the transport kernel (zeros
  and the face rule), the corrector (E_z, B_z). No extra pass, one branch per point when off. The column type is
  computed in the kernel from the sorted ring faces (binary search, device arrays uploaded once).
- **Charge conservation.** The RIP-CC Gauss identity holds for any field values that enter the step; overriding the
  results at metal points changes the Gauss residual only at the rho nodes whose stencil holds such a point: (i + 1/2,
  r_j) with column i or i + 1 metal and J <= j <= K. They carry the surface charge and are left out of the Gauss check
  (div B: J - 1 <= j <= K). Every other node keeps its residual to round-off.
- **Particles.** After the push, a particle whose shape lies entirely in a ring's closed index rectangle (shape 1: its
  position; shape 2: half a cell inside) is moved beyond the wall, so `migrate` absorbs it (its kinetic energy in
  E_part_out). Its current up to there was deposited, so the Gauss law at the checked nodes stays exact.
- **Stability.** The explicit limit dz^2 (lambda_max + wp^2) / 4 < 1 is unchanged (the metal only removes degrees of
  freedom, the face rule is a lossless reflection); for m = 0 on a uniform radial grid it means dz <~ dr.
- **iRIP: rejected.** iRIP's z lines need one radial operator for all columns (one eigenbasis per mode); the metal
  makes the radial operator depend on the column type, so the system does not separate.

## Jacket (`rings.jacket`, decision 83 amended 2026-11-12)
`jacket = t > 0` adds a solid tube r_out < r < r_out + t over [z_start, z_end] (the same absolute columns as the stack),
closing the gaps from outside: a closed corrugated pipe whose open end at z_end lets the field leave into vacuum. The
metal is then a union of rectangles, so the rules are evaluated per point instead of per column: a node (I, j) is metal
if it lies in a ring's [IA_k, IB_k] x [J, K] or in the jacket's [ta, tb] x [K, KJ]; a metal node is zero if both z
neighbours are metal (interior) or if it lies on an r surface (a radial neighbour in vacuum, j > 0); otherwise it is a
left / right z face (vacuum at I - 1 / I + 1) with the reflection rule above. A half node is metal if both its nodes are,
interior if both z neighbours are, else a face. The Gauss / div B points left out are those whose stencil touches a
metal node. Without the jacket these rules are identical to the column rules (bit-identical results). The jacket must
be at least one radial cell thick and r_out inside the grid.

## Self-field of a beam in RZ (`species.self_field`)
As in Cart3D (docs/specs/beam_self_field.md): phi at the rho nodes, Ez(i) = -(phi_{i+1/2} - phi_{i-1/2}) / (gamma^2 dz),
E_perp(i) = -G_m phibar(i), phibar the z mean of the two neighbours, B_phi = beta E_r, B_r = -beta E_phi. The RZ
Gauss law becomes (dzz / gamma^2 + mu_z^2 D_m G_m) phi = -rho; per radial eigenmode of the TM operator (rz_tm_basis,
lam > 0 with the wall) a tridiagonal line along z, s (phi_{I-1} + phi_{I+1}) + b phi_I = -dz^2 rho_I, s = 1/gamma^2 -
dz^2 lam / 4, b = -2/gamma^2 - dz^2 lam / 2 (diagonally dominant), phi = 0 just beyond the Gauss cells [0, N - 1).
Per eigenmode 1 / gamma^2 is clamped at dz^2 lam / 4 (s >= 0, decision 95; Ez from phi / gamma_eff^2 per mode): without it
the undamped zigzags of s < 0 carried a field tail ahead of an ultrarelativistic beam to the right edge, cut by the first
window shift (test M5: u = 1e4 in a window from t = 0, Gauss law 2.2e-14 of max rho).
Open z only, loaded at the start, beam along z. Host code at the setup, every rank solves the global problem (memory
8 N nr (2M - 1) bytes). The metal is ignored: start the beam in vacuum before the rings.

## Plane probe and axis output
- `output.plane_z`: every step (after the field step), the rank that owns the lab column round(plane_z / dz) appends
  [step, t, Ez Er Ephi Bz Br Bphi as [slots][nr]] (float64) to `dir/plane/plane_r<rank>.bin`; with a moving window the
  plane is recorded while it is inside. `tools/rz_plane.py` assembles the records: Poynting energy through the plane
  and, per frequency, the far field from the vector Hankel transform of E_r (radially polarized m = 0 field).
- `output.axis_every = n`: every n steps each rank appends [step, t, first absolute cell, cells, Ez(m = 0, r = 0)].
- `dir/plane/meta.txt`: the layout, lambda0, dz, dt, the plane cell and the radii (code units).

## Deck interface
```
rings {
  r_in = 1 mm;  r_out = 1.06 mm      # radial extent (snapped to the nearest radial nodes; r_out >= wall: to the wall)
  period = 230 um;  gap = 115 um     # metal thickness period - gap; gap = 0: a solid tube
  z_start = 1 mm;  z_end = 51 mm     # lab frame
  jacket = 60 um                     # optional: solid tube r_out .. r_out + jacket over [z_start, z_end] (closed pipe)
}
species beam { ...  uz = 9.95;  self_field = true }     # RZ: open z, beam along z
output { plane_z = 55 mm;  axis_every = 1000 }
```

## Tests (tests/test_rz_metal.cpp, `build/tests test_rz_metal`)
- M1 A tube r_in = R, r_out = wall (gap = 0) on a larger grid reproduces the conducting wall at R: TM01 and TE11
  standing modes equal to the wall run inside R to round-off.
- M2 Pillbox / periodic cavity: discs (r_in = 0) closing a cylinder; the TM_01p standing-mode frequency against the
  continuum sqrt((j01 / R)^2 + (p pi / l)^2) (radial discretisation error only); energy conserved.
- M3 Hot plasma between rings in a periodic tube: the Gauss law at round-off at every checked node, particles absorbed by
  the metal, div B unchanged; moving window and MPI along z in tests/mpi.sh.
- M7 Jacket: a tube R..R2 (gap = 0) with a jacket R2..R3 equals one tube R..R3 bit for bit after 300 steps (TM m = 0,
  TE m = 1); a jacket thinner than one radial cell is refused. M3 is repeated with a jacket (Gauss law, div B at
  round-off, fields in the interior zero); tests/mpi.sh: the stack with a jacket and its open end passing the moving
  window, 1 rank = N ranks.
- M4 Self-field: the Gauss law after the init at round-off; the steady field of a beam in vacuum stays steady.
