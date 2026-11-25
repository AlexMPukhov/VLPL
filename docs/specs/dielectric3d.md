# Dielectrics eps(x, y, z) in the explicit RIP-CC step of Cart3D and 2D (blocks `dielectric`)

Status: implemented (branch `generic-dielectric`, 2026-11-24); decision 100 in `docs/decisions.md` (the RZ scheme, carried over).
Asked for by Alexander (2026-11-24): "dielectrics and metals have to be implemented in general". The choices are his: the
time-centred (Crank-Nicolson) polarization current of decision 100, lines along the transport axis x, the 32-cell front layer and
its warning in a moving window, the per-component sub-cell rule (Q2 (a)); dispersive media stay with iRIP's `medium` block.

## Physics and equations
Lossless, non-dispersive dielectric: D = eps E with eps(x, y, z) >= 1 fixed in the lab frame (it moves through a moving window),
chi = eps - 1, the polarization current J_P = d_t P, P = chi E. Several blocks add their chi.

## Discretisation
RIP-CC (src/fields.cpp, docs/specs/rip_cc.md): E_x at (i, j, k), E_y at (i, j+1/2, k), E_z at (i, j, k+1/2); J_x at E_x, J_y at
(i+1/2, j+1/2, k), J_z at (i+1/2, j, k+1/2). J_P is time-centred at each current point (the RZ scheme with z -> x, E_r -> E_y,
E_phi -> E_z, B_phi -> B_z, B_r -> B_y):
- E_x: (1 + chi_x)(E_x^{n+1} - E_x^n) = dt (curl B - J_x): the predictor and the corrector divide their increment by 1 + chi_x.
- E_y, E_z: y_a = dt J_P(a) = chi_a mu_x(E^{n+1} - E^n) at a = i + 1/2; through the characteristics (T+_y = E_y + B_z to +x,
  T-_z = E_z - B_y to +x) node i gets -(y_{i-1/2} + y_{i+1/2}) / 2 in E, -(y_{i-1/2} - y_{i+1/2}) / 2 in B_z (E_y lines),
  +(y_{i-1/2} - y_{i+1/2}) / 2 in B_y (E_z lines). Per (j, k, component) the tridiagonal line along x
    y_a (1 + chi_a (m_a + m_{a+1}) / 4) + chi_a / 4 (m_a y_{a-1} + m_{a+1} y_{a+1}) = chi_a / 2 (m_a dEv_a + m_{a+1} dEv_{a+1})
  (m = 0 at metal edges of the block metal and at the guard nodes beyond an open x end), solved by Thomas; with ranks along x
  the partition solve of the RZ code, one allgather per step within the row of ranks along x (`Comm::allgather_x`, an
  MPI_Cart_sub communicator; ranks along y and z are independent rows), cyclic for periodic x.
- Metal (block metal, metal3d.hpp): at an x face the B partner takes the characteristic of the vacuum side including y
  (left face: B_z -= y_{i-1/2}, B_y += y_{i-1/2}; right face: B_z += y_{i+1/2}, B_y -= y_{i+1/2}); E stays 0.
- Step: predictor (1 + chi_x) -> vacuum transport -> metal rules -> exchange -> line solves and corrections -> exchange ->
  corrector (1 + chi_x). The same in the CPML step (pml.cpp; the polarization current is not stretched, like J). Without
  dielectric blocks nothing changes (bit-identical).
- Exactly div(E + P) = rho with D at the Gauss point's neighbours ((1 + chi_x) E_x, (1 + chi_y) mu_x E_y, (1 + chi_z) mu_x E_z):
  `Fields::gauss_residual` returns div D - rho, `Fields::energy` adds E.P / 2.
- chi per component from 4 x 4 x 4 sub-samples of eps over its control volume (2D: 4 x 4), harmonic along the component's own
  direction, arithmetic across: E_x [x -+ h_x/2] x [y -+ h_y/2] x [z -+ h_z/2] harmonic in x; J_y [x, x + h_x] x [y, y + h_y] x
  [z -+ h_z/2] harmonic in y; J_z likewise in z. Exact for interfaces on cell faces, better than a staircase for tilted ones.
- Moving window: as RZ (decision 100) a 32-cell absorbing layer at the front edge (only with dielectric, a window and no x
  absorber), E_x of the entering column and of the layer from div D (`dielectric3_window_entry`), chi moved with the window and
  sampled in the entering columns (`Dielectric3::shift`); the Gauss check excludes the layer; charged particles in the layer are
  warned (rank 0; at the start and on entry, then every 2000 steps; the RZ watch in rz_sim.cpp, shared).
- 2D (grid.geometry = cartesian2d): the same code on the flat domain (z stride 0): lines per j; eps of x and y.
- Not: iRIP / Yee (rejected), the side tube (rejected), the boosted frame (rejected). The TFSF plane must lie in vacuum.

## Deck interface
```
dielectric [<name>] {   # several blocks; Cart3D / 2D, numerics.solver = rip_cc
  eps = 2.1                                              # everywhere, or an expression of x, y, z (lab frame, units):
  # eps = sqrt((y - 0.64 mm)^2 + (z - 0.64 mm)^2) > 0.25 mm && x > 1 mm ? 3.8 : 1
}
```
The RZ keys (r_in, r_out, z_start, z_end, ramp) are rejected in Cart3D / 2D. eps < 1 anywhere is an error.

## Tests (tests/test_dielectric3d.cpp)
- E1 smooth random fields (4 harmonics per component) in a periodic box with a sphere (eps 2.5) and a tilted slab (eps 3.8),
  400 steps: div D unchanged to 4.6e-15 (3D) / 2.9e-15 (2D) of max, W / W0 in [0.991, 1.005] / [0.998, 1.010].
  (Found with it: the guards of periodic axes must hold the chi of their partners: the sample positions are wrapped. Without
  the wrap a non-uniform eps broke div D by 0.1 - 0.3.)
- E2 normal incidence on eps 4 (3D and 2D, E_y and E_z): reflected -1/3 to 4e-16 (exact: the x transport is dispersionless
  and the interface on a node), transmitted 2/3 to the grid dispersion inside the dielectric: -4.1e-3 for a pulse of 20 cells
  (1/e half width), -1.5e-4 for 40 cells.
- E3 oblique incidence (2D, 30 degrees, eps 4, 20-25 cells per wavelength): p (with the normal E_x across the interface)
  |r| = 0.28523 vs 0.28286 (+8.4e-3), s 0.38173 vs 0.38197 (-6.1e-4).
- E4 2D moving window with a dielectric lining and a rigid bunch (3000 steps): Gauss 7e-13, W(3000) / W(1500) = 0.999, no
  warning; a bunch loaded in the front layer: 1 warning.
- tests/mpi.sh: the structure pipe of examples/structure_pipe.deck with a dielectric liner across its corrugated wall entering the
  window, 1 vs NP ranks (2 x 2 x 2).

## References
Decision 100 and docs/specs/rz_dielectric.md (the scheme, its von Neumann analysis, the moving-window front layer).
