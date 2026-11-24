# Moving window

Status: implemented (window along +x at speed v <= c; exact Gauss law at v = c)

## Physics and equations
The box follows a laser pulse along +x. At v = c the window shifts by exactly one cell per step
(c dt = h_x). The X-dispersionless RIP-CC transport then makes a pulse in vacuum stand still in the
window, bit for bit. Plasma enters at the right edge and leaves at the left edge.

## Discretisation
- **Shift** (operator `moving_window`, stage Fields, after `rip_cc`): at t_{n+1}, the target number of
  shifts is floor((t_{n+1} - start) v / h_x). Each shift does the following:
  1. Moves E, B and the initial Gauss residual one cell towards -x in storage: f(i) = f(i+1). The last
     cell takes the right guard, which is the right neighbour's cell 0, or zero beyond the open edge.
  2. Refreshes the guards.
  3. Increments `Domain::wx`. `Domain::lo/hi` then give the absolute extent, `cell0(0) = off + wx`.
  4. Loads every species into the new last cell. Positions and momenta use the same per-cell
     generator as the initial load, so the result is independent of the decomposition and of when the
     cell entered.

  Particles keep their absolute positions. `migrate` then hands them to the ranks of their new window
  cell and deletes those behind the window (decision 21: kept until x < lo - h_x, exact currents).
- **Exact entry at v = c** (decision 22). No particle moves towards the right edge of the window, since
  v_x < c. In a step that ends with a shift, the particles of the last three columns, x >= hi - 3 h_x
  (`Domain::xfreeze`), are not pushed. These are exactly the cells loaded in the last three shifts.
  Then no current reaches the last cell:
  - A moving particle starts at x < hi - 3h and ends at x < hi - 2h, so its shape-2 charge never
    reaches the last node.
  - The last cell's fields at the start of the step are the zero guard of the previous step.
  - The fields beyond the edge would stay zero in an extended domain too.

  So the truncated update equals the extended one, and the Gauss law of the last cell, which becomes
  a Gauss point after the shift, stays at round-off. The price is that freshly loaded plasma is held
  for 3 steps (3 h_x of the plasma ahead of the pulse).

  Without the frozen columns, hot plasma (u_th = 0.01, 0.05) left a static residual of about rho u_th
  in every entering column (2e-4, 9e-4).
- **Current filter and the frozen depth** (decision 91 for RZ; Cart3D and 2D since decision 107, same code). With p filter passes the filter's closure at the
  right edge changes at every shift on the last p + 1 Gauss cells; the entering plasma is therefore frozen for
  F = 3 + p cells (`Simulation::window_freeze_cells`, 3 without the filter). Before, B7 with warm plasma had 3e-4
  of max rho in every mode.
- **Non-neutral entering columns** (decision 91). Two species on different lattices (electrons 2 x 2 x 2, protons
  1 per cell) put different charges on the nodes where the density is not uniform (a ramp, the front edge of the
  plasma). That charge appears with no current and no field, as a non-neutral initial load, which gauss0 holds as a
  background. So the charge of a column that leaves the frozen cells (window cell N - F - 1, still at its load
  positions) is subtracted from gauss0, filtered as rho (`window_fold_entered`), and the Gauss check adds the charge
  of the columns still frozen (`window_frozen_rho`). The fields are not changed. gauss0 is in the checkpoint, and the
  frozen part is recomputed after a restart. bench/lwfa_bubble had 1.95e-7 while its ramp entered. Off for v < c and
  for the boosted frame (no frozen columns there).
- **Before the window starts** the right edge is a plain open boundary, and its last cell is no Gauss
  point. Plasma currents there (hot plasma, laser-driven) leave a residual that becomes visible at
  the first shift (2.9e-2 with a hot plasma filling the box and start = 40 fs). The Gauss law is
  therefore exact if the window starts at t = 0 or the plasma begins ahead of the box (typical LWFA).
  A warning is printed otherwise. The same holds for fields: a pulse whose head crosses the right
  edge before the first shift leaves there the longitudinal field of a non-plane wave. A round
  1 um waist with c start > L gave 8.8e-2 at the first shift. The first shift comes after the step that ends at
  start + h_x / v, so keep c window.start + h_x <= L (decision 91: c start = L cut one cell of the head in
  bench/lwfa_bubble, 1.05e-4 in the Gauss law and div B), which
  together with start >= 2 delay needs L >= 2 c delay (warning).
- **v < c**: shifts are not every step. Particles faster than the window can reach the right edge, so
  there is no freezing (they would pile up). The right edge then acts as an open boundary: the Gauss law
  is exact only for cold plasma at rest near the edge.
- **Laser**. It enters through the left edge, which moves with the window. The inflow stops at the first
  shift, and `window.start >= 2 laser.delay` is required: the tail is cut at a(tc) as the head is at
  a(-tc).
- **Slices**. `slices/index.txt` has a third column with the absolute x of window cell 0.

## Cost of a shift (decision 36)
- The shift of a field is one flat copy of the whole array by one element. x is the fastest index,
  so this is f(i) = f(i + 1), with no index arithmetic.
- The y and z guards move with the data and stay valid, so only the x guards are exchanged
  afterwards (`Comm::exchange_x`). The result is bit-identical to the indexed copy plus a full
  exchange.
- A cold species is loaded into the entering column without a random generator.
- Effect: `examples/lwfa_window.deck` (256 x 64 x 64, 4 threads), the window operator went from
  6.9 s to 0.7 s (16 % to 2 % of the run). The indexed copy took 37 ms per shift; the exchange
  along y and z took 17 ms; seeding a Mersenne twister per cell for a cold plasma took 10 ms.
- Decision 55: the flat copy is now a move of the start of the array in its buffer
  (`Field::shift_x`, one plane of slack, a copy once the slack is used up), and a cold species enters
  through kernels (`src/load.*`, the same particles in the same order). `bench/lwfa_bubble` on the RTX
  5090: 5.3 ms -> 0.41 ms per shift (`docs/GPU_TESTING.md`, 2026-10-28).

## Deck interface
```
window {
  start    = 40 fs     # time of the first shift (default 0); >= 2 laser.delay with a laser
  velocity = 1         # units of c, 0 < v <= 1 (default 1); exact Gauss law at 1
}
```
Requires `grid.boundary_x = open`. Species limits (xmin, xmax, slab) are absolute coordinates.

## Tests
- Fast (tests/test_window.cpp):
  - W1: a plane pulse in vacuum, frozen in the window over 160 shifts. Field change 0, energy change 0.
  - W2: laser wake in a cold neutral plasma, shapes 1 and 2, global and tiled deposit. Gauss law at
    round-off (<= 2e-15). The ions grow by exactly one column (the one kept beyond the left edge).
  - W3: a slab ahead of the box enters at v = 1 and v = 0.5. The number of shifts is exact, and so is
    the particle count, the lattice points of the slab inside [keep_lo, hi).
  - W4: hot plasma (u_th = 0.05), laser wake with the plasma ahead of the box, and the box filled
    with the window moving from t = 0. Gauss law at round-off; 1.5e-3 with the freeze switched off.
  - W8: a warm non-neutral column entering (electrons 2 x 2 x 2, u_th = 0.05, protons 1 per cell, a ramp and a front
    edge), RIP-CC and iRIP: Gauss law 5.9e-16 / 6.4e-16 with a net entering charge of 6.3e-3 (decision 91).
  - R27 (tests/test_rz_grid.cpp): the same in RZ with 2 filter passes, iRIP + projection and RIP-CC + binomial: 3e-15.
- MPI: `mpi_winnn` (Cart3D, non-neutral entry), `mpi_rzwf`, `mpi_rzwfp` (RZ, warm non-neutral entry with the
  binomial / projection filter, restart of 8 ranks on 8 and 3).
- MPI (`tests/mpi.sh`, third case): small LWFA deck, hot plasma entering. 1 vs 8 ranks (2 x 2 x 2)
  give the same particle counts and energies; Gauss drift 2e-15.
- Regression: case `lwfa` (`examples/lwfa_window.deck`, small box, hot electrons).

## References
- A. Pukhov, J. Comput. Phys. 418, 109622 (2020): RIP solver, c dt = h_x.
