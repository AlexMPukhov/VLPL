# VLPL: choosing the solver, the grid and the numerical parameters

Everything here is stated in the units of the deck. `h_perp` is the transverse cell (`grid.dy`, `grid.dz`; RZ:
`grid.dr`), `h_x` the cell along the transport axis (`grid.dx`; RZ: `grid.dz`). `python3 scripts/vlpl_numbers.py`
evaluates the rules for given numbers.

## 1. What makes VLPL different: the transport axis

The RIP solver moves the transverse fields exactly one cell per step along x (RZ: along z): **c dt = h_x always**,
set by the grid, not by the user. Waves along that axis have no numerical dispersion, so a laser or a
relativistic beam travelling along x keeps its phase velocity c exactly; there is no numerical Cherenkov radiation
from the solver for particles moving along x. Consequences for every deck:

- Put the laser axis, the beam axis and the long direction of the problem along x (RZ: z). Specular reflection
  from a tilted target can be turned so that it leaves along -x (`examples/solid_target.deck`).
- The time step follows from `grid.dx` (RZ: `grid.dz`): dt = dx / c. Number of steps = c t_end / dx.
- Waves at an angle to x do have numerical dispersion, of second order in the cells; it is corrected for the
  implicit solver by `numerics.closure = o5`.

## 2. Which variant: geometry and solver

| variant | deck | use it when |
|---|---|---|
| **VLPL-RIP** | `grid.geometry = cartesian` (default) or `cartesian2d`, `numerics.solver = rip_cc` (default) | 3D or 2D, and the transverse cells are coarser than the longitudinal one: the explicit condition below holds |
| **VLPL-iRIP** | Cartesian, `numerics.solver = irip` | 3D or 2D with fine transverse cells (h_perp < c dt): boosted frames with gamma >~ 2, tightly focused light, harmonics, dense targets with fine transverse structure |
| **VLPL-RZ** | `grid.geometry = rz`, `numerics.solver = rip_cc` | problems close to rotational symmetry (round laser pulse, beam on or near the axis) with `grid.dz` well below `grid.dr` and few modes |
| **VLPL-iRZ** | `grid.geometry = rz`, `numerics.solver = irip` | RZ with `grid.dz` comparable to or above `grid.dr`, many modes, fine radial cells, boosted frames |

**3D or RZ?** RZ with M azimuthal modes (`grid.modes` = M: m = 0 .. M - 1) costs about 2M - 1 two-dimensional
grids instead of a 3D grid. Use it when the driver is round:
- `grid.modes = 1` (m = 0 only): beam drivers and bunches on the axis, radially polarized or no laser;
- `grid.modes = 2`: a linearly or circularly polarized round laser pulse (the laser field is m = 1, the wake m = 0);
- `grid.modes = 3`: adds the m = 2 content of the ponderomotive force of a linearly polarized pulse and of
  elliptical spots; Laguerre-Gauss LG_p^l needs `grid.modes` >= |l| + 2; a laser or beam off the axis needs as many
  modes as its azimuthal content.
Use 3D (or 2D for a first look) for oblique incidence, tilted targets, strongly non-round beams, hosing far from
the axis, and whenever the result must not depend on a mode truncation. A 2D run (`grid.geometry = cartesian2d`)
is a slab: totals are per `grid.dz` of thickness, self-focusing and wake amplitudes differ from 3D.

**Explicit or implicit?** The explicit solver is stable only if (the code checks it and stops otherwise):

- Cartesian 3D: `dx^2 (1/dy^2 + 1/dz^2) + dx^2 omega_p^2 / (4 c^2) < 1`; 2D: without the dz term. For equal
  transverse cells in 3D this means h_perp > 1.42 dx (2D: h_perp > dx), a little more with plasma. The summary line
  `courant dx^2(1/dy^2+1/dz^2) = ...` prints the first term.
- RZ: `dz^2 (lambda_max + omega_p^2/c^2) / 4 < 1`, where lambda_max is the largest eigenvalue of the radial
  operators. It grows as 1/dr^2 and with the modes near the axis (about 4.5 x 4/dr^2 for 3 modes), so the explicit
  RZ solver needs `grid.dz` well below `grid.dr` (the examples use dz = dr / 4 with 2 modes). With many modes
  `numerics.rz_axis_cut` > 0 removes the growth with the mode number.
- The implicit solver (`irip`) has no such limit: only omega_p dt < 2 at the peak density. Its cost per step is
  higher (two passes of the field step plus a transverse transform; in Cartesian geometry the transform is fast
  only when `grid.ny` and `grid.nz` are powers of two, and an axis with CPML uses a dense transform), and it
  decomposes over MPI ranks along x (RZ: z) only. If the explicit condition holds with the cells the physics
  needs, take the explicit solver.

Rule of thumb: first choose the cells from the physics (section 3). If then h_perp >= 1.5 h_x (RZ: dr >= 4 dz with
2 modes): explicit. Otherwise: implicit. Do not coarsen a transverse grid the physics needs, or refine h_x beyond
need, only to satisfy the explicit condition.

**The plasma frequency and the step.** omega_p dt = (dx / (c/omega_0)) sqrt(n / n_c) = 2 pi (dx / lambda0)
sqrt(n / n_c) at the peak density. Explicit particles need omega_p dt < 2 in any case; the plasma oscillation then
runs at 2 asin(omega_p dt / 2) / dt (a warning gives the error in percent). Keep omega_p dt <= 0.2 for wakefields
(error of the plasma frequency < 0.2 %); solid targets run at 0.5 - 1 (30 n_c at lambda0 / 40: 0.86). Beyond
that: `numerics.mover = implicit` (with `irip`, `numerics.closure = box`; energy conserving, no limit on
omega_p dt) or a particle-free `medium` block.

## 3. Resolution, as numbers

| quantity | rule |
|---|---|
| laser along x, lab frame | `grid.dx` <= lambda0 / 16 for a first look (the examples), lambda0 / 32 for quantitative wakes and ionization; the phase velocity is exact at any dx, the resolution is needed for the particles' response |
| boosted frame | lab `grid.dx` <= lambda0 / 32: the plasma streams into the laser, so the same accuracy needs about (1 + beta) finer cells (measured at gamma = 2: wake 15 - 21 % too strong at lambda0 / 16, 0.5 - 1.9 % at lambda0 / 32) |
| transverse cell, underdense plasma | h_perp <= waist / 20 and <= (c / omega_p) / 5 |
| plasma wavelength along x | automatically resolved when the laser is; without a laser dx <= lambda_p / 30 and <= sigma_x / 5 of the shortest bunch |
| solid target | dx <= lambda0 / 40, cells below the skin depth c / omega_p, `numerics.shape = 3` (Cartesian 3D) or 2, `numerics.filter_passes = 1`; a density ramp needs shape 2 or 3 (shape 1 is unstable on a resolved ramp with p-polarized light: a warning) |
| harmonics up to order n | lambda0 / (n x 8) along x at least; off-axis harmonics: `numerics.closure = o5` with `irip` |
| particles per cell, Cartesian | `species.ppc_x`, `species.ppc_y`, `species.ppc_z`: 1 1 1 to 2 2 2 for cold underdense plasma with `numerics.shape = 2`; 2 - 4 per direction for dense or hot plasma and collisions |
| particles per cell, RZ | `species.ppc_theta` >= 2 (2 `grid.modes` - 1) (a warning below: noise heating), `species.ppc_r` >= 2, `species.ppc_z` 1 - 2; `numerics.shape = 2` (folded across the axis: less noise there) |
| laser duration and entry | `laser.duration` is the FWHM of the intensity; `laser.delay` (default 2 x duration) is the time of the peak at the entrance plane x = 0; the pulse is zero beyond 3 sigma |
| moving window | 2 x `laser.delay` <= `window.start` < box length / c. Below the lower limit: an error (the pulse is not in yet). Above the upper limit: a warning, the light that entered at t = 0 has reached the right edge, and max|dGauss| and max|divB| are no longer at round-off. So: box length > 2 c `laser.delay` + the plasma periods to be seen, and the peak sits c x (`window.start` - `laser.delay`) from the left edge of the window |
| plasma start | begin the plasma ahead of the box at the time the window starts (`species.xmin` beyond the right edge, as in the examples), or start the window at t = 0: then the Gauss law is exact while plasma enters |
| transverse box, 3D | periodic by default: wide enough that the laser at its widest is < 1 % at the edge, or `grid.boundary_y = pml`, `grid.boundary_z = pml` (16 cells each side by default; 32 with `numerics.closure = o5`) |
| radial box, RZ | `grid.boundary_r = wall` (default: a conducting wall at r = nr dr that reflects) or `absorbing` (`grid.r_absorber_cells`, default 32); a laser needs r_max >= 3 x the largest spot in the run, or the absorber |
| stretched radial grid | `grid.r_uniform`, `grid.stretch`, `grid.dr_max`: keep k_perp dr <~ 1 wherever radiation must propagate (cutoff k_perp = 2 / dr: a coarser cell reflects), stretch by a few percent per cell (the examples: 1.03) |

A matched spot or P / P_c near 1 does not guarantee guiding: over more than a Rayleigh length z_R = pi w0^2 /
lambda0 size the transverse box with the vacuum spot w0 sqrt(1 + (x - x_focus)^2 / z_R^2).

## 4. Noise and the current filter

- `numerics.shape`: 1 linear, 2 quadratic, 3 cubic (Cartesian 3D, not with a moving window). Use 2 unless there
  is a reason; the charge-conserving deposition keeps the Gauss law at round-off for every shape.
- `numerics.filter_passes` = N passes of a binomial filter on the current that keeps the Gauss law exact.
  `numerics.filter = binomial` (default) smooths every component and is the one for beams, wakefields, streaming
  and boosted plasma. `projection` filters only the longitudinal current (harmonics radiate unfiltered: for
  solid-target harmonics; needs periodic y, z and ranks along x) but makes a relativistically drifting plasma
  unstable: never for LWFA, beams or boosted frames. In Cartesian geometry the filter runs without a window, with
  a window at v = c, or in a boosted frame.
- RZ: 1 - 2 passes bring the noise on the axis down to the level of spectral RZ codes.
- Lab-frame wakefields: 0 passes are fine with shape 2; add 1 if the density on the axis is noisy.

## 5. Boosted frame (`boost { gamma = ... }`)

The deck stays in the lab frame; the code transforms it. With the window at c the number of steps falls by about
gamma^2 (1 + beta)^2 for interaction lengths much longer than the box.

1. **Choose gamma** below the Lorentz factor of the wake, gamma_wake ~ omega_0 / omega_p = sqrt(n_c / n), so
   that the wake still moves forward in the boosted frame. Start with gamma = 2 - 5 and compare one short case
   with the lab run before going higher.
2. **Solver:** the boosted cell is dx' = gamma (1 + beta) dx: for gamma >~ 2 the explicit condition fails
   (h_perp < c dt'): use `numerics.solver = irip`.
3. **Stability of the streaming plasma (numerical Cherenkov):** `numerics.shape = 2` and
   `numerics.filter_passes = 1` with the binomial filter; 2 passes when the plasma stays longer than 100 / omega_p
   in the box (k_p x lab window length > 100). The summary warns when a deck has numerics measured unstable. With
   gamma >= 10, `numerics.filter_axes = x` with 8 passes and `numerics.filter_compensate = true` filters only along
   the streaming axis.
4. **Neutral loading:** the injected plasma must be exactly neutral per lattice group: every ion species comes
   with an electron species of the same `ppc`, the same bounds and the matching density (sum of q Z n = 0); the run
   stops otherwise. `mobile = false` is refused (a lab background moves in the boosted frame): give the ions their
   mass instead.
5. **Beams** (species with `ux` != 0 or `self_field`): a beam moving against the boost is gamma^2 (1 + beta)^2
   shorter in boosted cells, so the loading lattice holds n = (xmax - xmin) a ppc_x / dx' planes of it, a = 1 /
   (gamma (1 - beta v)). **The loading checks:** n < 1 is an error (the message names the `species.ppc_x` needed),
   n < 20 a warning (charge quantized, error ~ 1/n), and after loading the loaded charge is printed against the
   deck's (a difference above 1 % is a warning). Raise `species.ppc_x` until there is no warning.
6. **Laser injection.** Default: at the boosted entrance plane, i.e. at lab events x in [X0, 0] with |X0| = gamma
   beta gamma (1 + beta) c `window.start`. A beam focused far downstream is wider there and must fit the
   transverse box (the summary warns when the box cuts it). `laser.injection = antenna` removes this: a current
   sheet fixed at the lab plane x = 0. **Near field of the antenna:** its field reaches the lab value only one
   boosted step after emission. Within

       d_1% = 1.1 (gamma (1 + beta))^2 h_lab        (h_lab = the lab `grid.dx`; RZ `grid.dz`)

   of lab distance downstream of x = 0 the field is weaker than in the lab run (0.6 - 0.66 of it at 0.1 d_1%, 0.78
   at 0.3, 0.9 at 0.5, 0.97 at 0.8), beyond it within 1 %. The summary prints d_1% per antenna laser. Place the
   plasma, the beam collision point and any ionization front further downstream than d_1%, or reduce h_lab.
   Example: gamma = 5, lambda0 / 20 = 0.04 um: d_1% = 1.1 x 9.9^2 x 0.04 um = 4.3 um.
7. **Times are lab times:** `grid.t_end`, `window.start`, `laser.delay`. The run ends when the lab time t_end
   has left the box. Output for the user comes from the block `lab_diagnostics` (lab-frame snapshots);
   `diag.txt` and `ionization.txt` are in the boosted frame.
8. **Not available in a boosted frame:** side injection, `medium` blocks, the Cartesian TFSF plane and oblique
   lasers, drifting plasmas, the implicit mover, embedded metal rings. Field ionization and QED work (rates in
   the rest frame; QED energies and spins are reported in the lab frame).

## 6. What VLPL is, and what to use instead

VLPL is a full electromagnetic PIC code: forward and backward waves, reflection, overdense plasma, any angle.
It is the right tool for laser-solid interaction, harmonics, plasma mirrors, strong-field QED, collisional dense
plasma, ionization physics, short wakefield stages including injection, and structures (metal, dielectric) with
backward waves.

It is the expensive tool for long underdense stages: steps = length / dx with dx ~ lambda0 / 20. For centimetres
to metres of wakefield acceleration, beam transport, undulators and FEL use a reduced model (GEM-PIC-RZ,
github.com/AlexMPukhov/GEM-PIC-RZ, a window moving at c with steps of micrometres; or the quasi-static QUARZ,
github.com/AlexMPukhov/QUARZ, for beam drivers), or a boosted frame here. Say so before writing a deck, with the
step count (`scripts/vlpl_numbers.py` prints it).
