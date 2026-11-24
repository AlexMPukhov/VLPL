# O4: solid-density targets (tilted slab with pre-plasma, mobile ions, shape 3, current filter, omega_p dt)

Status: implemented 2026-11-01 (decision 64), roadmap track O, item O4 (`docs/specs/overdense.md`). Cart3D.

## Physics and equations
A foil whose normal lies in the x-y plane at `slab_angle` from x. With the normal coordinate
s = (x - x0) cos(angle) + (y - y0) sin(angle) the density is n(x, y, z) = density * p(s) * f(x, y, z), f the optional
`density_function` and
- p(s) = exp(s / L_f)          for -E_f <= s < 0     (front pre-plasma, scale length L_f, cut at the extent E_f)
- p(s) = 1                     for 0 <= s < T        (plateau, thickness T)
- p(s) = exp(-(s - T) / L_r)   for T <= s < T + E_r  (optional rear ramp)
- 0 elsewhere.
The areal density along the normal is integral p ds = L_f (1 - e^(-E_f/L_f)) + T + L_r (1 - e^(-E_r/L_r)).
Ions are ordinary species (`charge`, `mass`, `mobile = true` by default) with the same slab keys: with the same lattice
and Z n_i = n_e the target is exactly neutral at t = 0.

## Discretisation
1. **Exact loaded charge.** With a ramp (or `slab_exact = true`) a macro-particle at the midpoint of its sub-cell
   (h_a / ppc_a) gets the weight density * <p>_sub * f(r) * dV / ppc, <p>_sub the exact mean of p over the sub-cell.
   p depends on x, y only through s = c x + s_ y (c = cos, s_ = sin), so with F2'' = p:
   int int p(s_m + c u + s_ v) du dv over [-a, a] x [-b, b] = [F2(++) - F2(-+) - F2(+-) + F2(--)] / (c s_),
   F2(+-) = F2(s_m + c a - s_ b), F2 the second antiderivative of p taken from the midpoint value s_m (so its values
   are O(h^2) and the four-term difference loses no digits); piecewise closed forms of the exponentials, series for
   small b Delta. |c s_| < 1e-6: the one-dimensional formula along the larger component. Sub-cells cut by the slab
   faces or the ramp ends get their partial weight, so sum w = density * area * integral p ds / cos(angle) to round-off
   for any tilt, scale length and resolution. A plain slab without `slab_exact` keeps the point test (bit-identical).
2. **Shape 3** (cubic B-spline, `numerics.shape = 3`): the gather, the Esirkepov deposit and rho are generated from the
   spline (W = 4 nodes) like shapes 1 and 2, so continuity holds to round-off. Its support is 2 cells: particles are
   kept 2 cells below an open lower edge (shapes 1, 2: 1, decision 21) and 1 cell above an open upper edge (shapes 1,
   2: 0), so no Gauss point loses charge without a current; guards ng = 4 + ceil(dt / h_a) (5 for RIP-CC). The tile
   box [a - 2, b + 2] of the tiled and box deposits covers shape 3 too. Not in RZ, not with a moving window (refused).
3. **Current filter for Cart3D** (`numerics.filter_passes`, `numerics.filter = binomial | projection | local`). All make the
   continuity of the filtered current that of the filtered charge, D J' = T D J with T = T_z T_y T_x and
   T_a = 1 + (h_a^2/4) delta_a^2 the binomial (1, 2, 1)/4 along a over the Gauss nodes (Neumann at an open edge); the
   charge of the diagnostics is filtered by T, so the Gauss law dx Ex + mu_x (dy Ey + dz Ez) = T rho stays at round-off.
   - `binomial` (the default, decision 91): per pass and axis a, J_a <- J_a + (h_a/4)(w_c - w_c-1) with w = d_a J_a on the
     Gauss nodes (the step of `local`, with the a part of D J only), and J_b <- T_a J_b for b != a (the binomial along a over
     the Gauss range of a; J_b lies at the a-position of the nodes, so T_a commutes with its difference along b). In the
     interior every component gets T_x T_y T_z as rho: a drifting plasma keeps J = v rho at the grid scale. Any boundary and
     decomposition, no transform. Price: grid-scale transverse currents are smoothed too (harmonic N of a laser damped by
     cos^2(N k0 h / 2) per pass, as FBPIC's filter). Tests S9 (identity 4e-16, plane-wave transfer function 3e-15) and S10
     (streaming plasma: projection unstable, binomial 1e-10; bench/nci/README.md).
   - Why not the longitudinal filters by default (decision 91): they smooth rho but not the transverse current. For a
     relativistic drift the electric force of the smoothed charge then no longer cancels the magnetic force of the
     unsmoothed current, and the streaming-plasma test grows at 0.4 wp (gamma 10) to W_field ~ K_e. Grid heating is
     electrostatic aliasing, and the transverse surface current that radiates the harmonics is left alone by them, so
     they are kept for static targets (HHG, foils), set explicitly in those decks.
   - `projection` (the default from decision 69 to 91): per pass w = D J, r = (T - 1) w, L psi = r and J <- J + G psi, with
     G = (dx, mu_x^2 dy, mu_x^2 dz) the gradient of RIP-CC's Gauss law and L = D G = dxx + mu_x^2 (dyy + dzz) (the
     Laplacian of the beam self-field, docs/specs/beam_self_field.md). The change G psi only moves charge: fed alone into
     the field solver it leaves B = 0 (test S6, 2e-31), so the filter radiates nothing. L psi = r: DFT in y and z (mixed
     radix), per transverse mode a tridiagonal system along x (s = 1 - hx^2 lambda / 4 > 0 under RIP-CC's stability
     limit, b = -2 - hx^2 lambda / 2; open x: Neumann at node 0, psi_N-1 = psi_N-2 for the last node, which is no Gauss
     node; the mode lambda = 0 pinned, psi_0 = 0), the partitioned solve of iRIP with ranks along x (src/xlines.hpp).
     y and z periodic (no pml, no side tube), ranks along x only; otherwise the local filter is used with a warning
     (decision 71; a test in test_cart2d). Cost: two transforms and a line
     solve per pass, 2-4 times the RIP-CC field step at 2D sizes.
   - `local` (decision 64): per pass three steps a = x, y, z: w = D J (0 at nodes that are no Gauss points),
     J_a += (h_a / 4) (w_c - w_c-1) on the faces between two Gauss points; then D S_a = T_a D identically. Local, any
     boundary and decomposition. But a change of J_a alone is no gradient (its curl (0, dz dx w, -dy dx w) is not 0): it
     adds a grid-scale transverse current that radiates (test S6: 7 % of the field energy of the change ends in B, for a
     gradient 0). In a thermal plasma slab one pass raises the radiated noise at harmonic orders 7-16 by 1.1-1.8 (two
     passes 1.3-2.2), in the oblique HHG case (bench/hhg_oblique) the reflected orders 8-16 by 2-3.7; the projection
     leaves both at the unfiltered level (0.99-1.00, 1.03-1.20). Keep it for decompositions in y, z or non-periodic y, z.
   RIP-CC and iRIP; not Yee, not with a moving window (refused), except in a boosted frame (decision 104: the plasma enters
   3 cells behind the right edge and nothing is frozen; the Gauss check skips the first filter_passes cells at the left edge).
4. **omega_p dt at the peak density.** After the initial load the largest density of each species (density times the
   largest profile factor actually loaded, over all ranks) gives omega_p^2 = sum n_s q_s^2 / m_s; the run stops if
   omega_p dt >= 2 (iRIP, Yee) or dx^2 (1/dy^2 + 1/dz^2) + (omega_p dt)^2 / 4 >= 1 (RIP-CC), and warns above
   omega_p dt = 1 (the explicit current coupling oscillates at 2 asin(omega_p dt / 2) / dt, 4.7 % above omega_p).
   The summary prints omega_p dt and the skin depth in cells.
   A slab ramp with shape 1 gives a warning: resonance absorption in a resolved ramp seeds the shape-1 finite-grid
   instability (decision 64); use shape 2 or 3 for targets with pre-plasma.

## Deck interface
```
species electrons {
  density = 100                      # plateau, n_c
  slab_x0 = 6 um   slab_y0 = 8 um    slab_angle = 22.5 deg   slab_thickness = 1 um
  slab_ramp = 0.08 um                # front pre-plasma scale length L_f (0: none)
  slab_ramp_extent = 0.4 um          # E_f (default 5 L_f)
  slab_rear_ramp = 0 um              # L_r, slab_rear_ramp_extent (default 5 L_r)
  slab_exact = true                  # exact sub-cell weights (automatic with a ramp)
}
numerics { shape = 3   filter_passes = 1 }
```

## Tests (`tests/test_solid.cpp`, `build/tests test_solid`)
- S1 loaded charge of a tilted slab (0, 30 and 22.5 deg, front and rear ramps, scale lengths below and above the cell):
  sum w against the analytic areal density to 1e-12; the point-sampled slab for comparison.
- S2 neutral target: electrons and ions (Z = 1) on the same lattice, rho = 0 at t = 0; mobile ions, shape 3, filter,
  RIP-CC and iRIP with absorbing x: Gauss law and div B at round-off over 60 steps while the laser heats the surface.
- S3 shape 3: continuity at round-off (random moves), charge carried out through both open x edges (Gauss law while
  plasma drifts out), Langmuir frequency; tiled and box deposits = global.
- S4 filter: the transfer function cos^2(k h / 2) of rho for a charge wave along x, y, z (round-off), S J = J for a
  divergence-free J, Gauss law at round-off with open x and the filter on.
- S6 radiation of the filter's change dJ = S J - J (a random J on a periodic grid, one field step with dJ alone, then 40
  steps without current): B^2 / (E^2 + B^2) = 2.2e-31 (RIP-CC), 2.8e-31 (iRIP) for the projection, 7.3e-2 and 6.1e-2 for
  the local filter, 0.33 and 0.29 for the raw J.
- S7 projection: D J' = T D J to 2.8e-15 for a random J, periodic and open x, one and two passes; a discrete curl is
  unchanged (1.6e-15).
- S8 the DFT lines of src/xlines.hpp (radix 2 and mixed radix, n = 6, 7, 12, 15, 49, 64, 120, 576, both signs) against the
  direct sum: 3.4e-15.
- S5 omega_p dt: a deck whose density_function peak exceeds the limit is refused, the same with `density` alone passes.
- `tests/mpi.sh`: the ramped tilted slab with ions, shape 3, filter, 1 vs 8 ranks: the local filter on 2 x 2 x 2 ranks,
  the projection with RIP-CC and iRIP on 8 ranks along x.

## References
Esirkepov, CPC 135, 144 (2001) (any spline order); Birdsall and Langdon, Plasma Physics via Computer Simulation, ch. 8
(grid heating, filtering); Vay et al., JCP 230, 5908 (2011) (binomial filters); decision 54 (the RZ filter).
