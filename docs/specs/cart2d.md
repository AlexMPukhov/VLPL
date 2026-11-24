# 2D Cartesian geometry (x, y)

Status: implemented (branch `geometry-2d`); decision 66 in `docs/decisions.md`.

## Physics and equations
Fields and particles independent of z: d/dz = 0 in Maxwell's equations, all six field components and all three
momentum components kept (both polarisations, J_z from v_z). The 2D problem of Smilei's `2Dcartesian` and of the
2D-like Cart3D decks with 3 periodic z cells (`bench/tnsa/rip.deck` before this change), at a fraction of their cost.

Units as in 3D. The run is a slab of thickness h_z = `grid.dz` (default 1 um): densities stay in n_c, a macro-particle
weight is n h_x h_y h_z / ppc, and every total (field and kinetic energies in diag.txt, charges and spectra of the
`diagnostic` blocks, xflux energies) is per slab of thickness dz, i.e. per dz of length along z. With dz equal to the
dz of a Cart3D run with z-uniform data, the 2D run equals one z column of it cell by cell.

## Discretisation
The Cart3D code on one z cell (decision 66, Alexander's choice 2026-11-02 over a separate Cart2D field geometry):
- `Domain::flat`: N_z = n_z = 1, z periodic, no z guards (`Domain::gw(2) = 0`), and every field array has the z
  stride 0 (`FV::sy = 0`, so the plane stride sx sy is 0; the index arithmetic of Cart3D is unchanged): f(i, j, k) is
  f(i, j, 0) for every k. Every Cart3D field kernel (RIP-CC, iRIP, CPML in y,
  TFSF plane, x absorber, current filter, Gauss residual, div B) then computes d/dz as the difference of a value with
  itself, exactly zero, without a 2D copy of the kernel. A field is one (x, y) plane: memory and halo traffic of 2D.
- iRIP: the z DFT has length 1 (eigenvalue 0), so the transverse eigenbasis is the 1D one in y; the tridiagonal
  systems along x are unchanged.
- Halo exchange skips z; loops over the guarded range use the per-axis width `Domain::gw(a)`, `Domain::ext(a)`.
- Particles (`Cart2D` in `src/geometry.hpp`): positions x, y, z and momenta ux, uy, uz as in 3D; z wraps over the
  slab on the device in the migration. Gather and charge deposit with the shape in (x, y) only. The Esirkepov current
  in (x, y) is the 2D Esirkepov sum (z factor 1); J_z = (q w / dt h_x h_y) (dz / h_z) W_xy with the Esirkepov weight
  W_xy = S0x S0y + DSx S0y / 2 + S0x DSy / 2 + DSx DSy / 3: the sum over z of the Cart3D J_z of the same move, so the
  discrete continuity equation holds as in 3D (its z term is identically zero). Moves of more than one cell along x
  or y are split as in Cart3D. The box deposit keeps one plane (B_z = 1).
- Stability: RIP-CC dx^2/dy^2 + dx^2 wp^2/4 < 1 (no z term); the guard width from dt/h_x, dt/h_y only.

## Deck interface
```
grid {
  geometry = cartesian2d   # 2D (x, y); cartesian (3D, default) | rz
  nx, ny, dx, dy, boundary_x, boundary_y, mpi_x, mpi_y   # as in 3D
  dz = 1 um                # slab thickness that totals refer to (default 1 um); no nz, mpi_z, boundary_z
}
```
Supported in the first version: rip_cc and irip, shapes 1-3, the current filter, periodic / open / absorbing x,
periodic or CPML y, lasers through the left edge or a TFSF plane (`tfsf_incident = grid | modes`), the moving window,
species with slabs, ramps and density functions (z ignored), `diagnostic` blocks, xflux, openPMD output and
checkpoints (as Cart3D with nz = 1), MPI along x and y (irip: along x only, as in 3D).
Refused with a message: solver = yee, block `side`, laser files, lasers with waist_z, mode_p or mode_l, self_field,
ppc_z, zmin, zmax, nz, mpi_z, boundary_z.

## Tests
- Fast (tests/test_cart2d.cpp):
  - continuity (rho^{n+1} - rho^n)/dt + div J = 0 for random moves, shapes 1-3, also over several cells: < 1e-13
    relative to |drho/dt|;
    J_z = q w v_z W_xy summed over the plane: the 2D total equals q w v_z / (h_x h_y h_z) per particle to 1e-14;
  - plasma (hot, drifting, neutral and not) with rip_cc and irip, shapes 1-3: Gauss law and div B < 1e-12;
  - vacuum dispersion of oblique modes (kx, ky): RIP-CC sin^2(w dt/2) = sin^2(kx dx/2) + dx^2 Lam/4 cos^2(kx dx/2),
    iRIP tan^2(w dt/2) = tan^2(kx dx/2) + dx^2 Lam/4, Lam = 4/dy^2 sin^2(ky dy/2): < 1e-11; waves along x one cell per step;
  - Langmuir frequency: sin(w dt/2) = wp dt/2 within 1 %;
  - equivalence with Cart3D (3 periodic z cells, z-uniform data, the same dz): a cold foil hit by a TFSF laser (incident
    grid) with absorbing x (rip_cc and irip, shape 2, filter), 14 fs: every z plane of the 3D fields = the 2D fields to
    1e-9 of the peak (measured 1.2e-11 rip_cc, 4e-11 irip: round-off amplified by the interaction, as between the 3D z
    planes themselves), the field and kinetic energies per z cell to 1e-11 (measured 7e-14 field, 5e-13 kinetic). With `tfsf_incident = modes`
    the two differ by ~5e-5 of a0: the frequency lattice of the plane waves follows the box diagonal, which has no z part
    in 2D (the same difference as between 3D boxes of other dz);
  - TFSF plane waves in 2D (rip_cc, irip, 0 and 30 deg): no field left of the plane (< 1e-13 of the peak);
  - moving window with a laser through the open left edge and a hot plasma entering: Gauss law < 1e-12.
- `tests/mpi.sh`: `examples/foil_2d.deck` on 1 vs NP ranks: rip_cc (x, y), irip (x), CPML in y; restart of NP on NP and 3.
- Benchmark: `bench/tnsa` code check in 2D against the Cart3D slab and Smilei 2D: the speed-up and the proton cutoff
  (README there).

## References
- T. Zh. Esirkepov, Comput. Phys. Commun. 135, 144 (2001) - charge-conserving deposition (2D form).
- A. Pukhov, J. Comput. Phys. 418, 109622 (2020) - RIP.
