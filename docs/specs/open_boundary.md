# Exact particle currents at an open boundary

Status: implemented (open x; y, z follow with their absorbing layers)

## Physics and equations
A particle leaves the box through an open boundary. The Gauss law dx Ex + mu_x dy Ey + mu_x dz Ez = rho
must stay at round-off in every cell where it is a dynamic constraint. That requires the discrete
continuity equation in those cells: the charge of the leaving particle must go out through the current
of its last steps, never by deleting it.

## Discretisation
RIP-CC (descriptor RipCC): rho sits at (i + 1/2) along x. The Gauss point of cell i uses Ex(i) and Ex(i+1).

- **Right open edge** (x = hi). The last cell n-1 is no Gauss point: Ex(n) lies beyond the edge, in a
  zero guard. It is skipped by `gauss_residual` and `max_div_b`. A particle removed at x >= hi has no
  charge weight in cells <= n-2: its charge node distance is >= 1.5 cells for shape 2 and >= 1 cell for
  shape 1. It also contributes no Jx at face n-1 afterwards. So deleting it at x >= hi is exact for
  every checked cell. Nothing changes here.
- **Left open edge** (x = lo, the global lower edge). Cell 0 is a Gauss point: Ex(0) and Ex(1) are both
  owned. A particle at lo - h < x < lo still has charge weight in cell 0, which is centred at
  lo + h/2: up to 0.5 for shape 1 and up to 0.5 for shape 2. Deleting it at x < lo violated the Gauss
  law there. Measured on main before the fix: 0.3-0.36 with a plasma at drift -3 (test 11b), and
  1.5e-2 on the open-x streaming deck.
- **Rule**: a rank keeps a particle while `x >= Domain::keep_lo(0)`. This is lo - h_x at an open lower
  global edge and lo otherwise. Such a particle is pushed and deposits as usual. Its Esirkepov current
  carries its charge out of cell 0, and it is deleted once x < lo - h_x. At that point its weight in
  cell 0 is zero for both shapes (distance >= 1.5 cells).
  - The guards hold the stencil: the cell is >= -1, the reach is >= -3, and ng = 3.
  - The particle's gather reads guard fields (zero, or the laser inflow layer). This only affects
    particles that have already left the physical box.
- **Tiles**. A kept particle is in cell -1, outside every tile, so the tiled and box deposits treat it
  as a stray (atomic path). The tile key clamps it to tile 0.
- **MPI**. Only ranks with off[0] = 0 keep particles below lo. Transverse migration of kept particles
  stays inside that x slab of ranks.
- **Diagnostics**. Particle counts and W_kin include the kept particles: one cell of transit, about
  one step for v ~ c.
- **Moving window (Phase 1)**. The window moves +x, and the plasma leaves through the left edge. The
  same rule applies with lo advancing by one cell per shift.

## Deck interface
None: always on for an open boundary (`grid.boundary_x = open`).

## Tests
- Fast (tests/tests.cpp, test 11b): hot plasma at drift -3 (leaving left) and +3 (leaving right),
  shapes 1 and 2, global and tiled deposit. The electrons leave and max|dGauss| < 1e-12 (measured
  <= 7e-15; it was 0.36 at the left edge before).
- Fast: registry test, attributes survive migration through the open x edge, with kept particles in
  [keep_lo, L).
- MPI (`tests/mpi.sh`, second case): streaming deck with `boundary_x = open` and transversely hot
  electrons. 1 vs 8 ranks give the same energies (1e-11) and particle counts, and the Gauss drift is
  3e-16 on 8 ranks.

## References
- T. Zh. Esirkepov, Comput. Phys. Commun. 135, 144 (2001).
