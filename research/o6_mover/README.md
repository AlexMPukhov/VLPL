# O6: energy-conserving implicit particle mover with the iRIP box closure (prototype, 2026-11-06)

Spec and options: `docs/specs/implicit_mover.md`. Everything here is numpy (the CPU cluster: `. $WORK/smilei/smilei_env.sh`,
runs via `srun -p epyc`; scripts `run_*.sbatch`, outputs in `out/`, not committed).

## Files
- `pic2d.py`: 2D3V PIC (periodic x, y) on the box-iRIP step of `research/linear_medium/box2d.py` (exact FFT solve), with
  `step_explicit` (the code's leapfrog: Boris, E^n, B^n at x^n with the plain shape, Esirkepov deposit) and
  `step_implicit` (the O6 mover, below).
- `rz0.py`: RZ, mode m = 0, TM fields (Ez, Er, Bphi) on the finite-volume radial grid of decision 52, conducting wall,
  Cartesian particles (decision 44), 2D Esirkepov in (z, xi(r)) with the shape-2 Ruyten map (port of src/rz_ruyten.cpp)
  folded across the axis; explicit (plain or matched gather) and implicit movers. TE and J_phi are left out.
- `t0_identity.py`, `rz_t0.py`: energy identity of the box step with a current, energy and Gauss law with hot plasma.
- `t1_langmuir.py`: cold Langmuir frequency vs Crank-Nicolson / leapfrog theory, solver comparison.
- `t2_heating.py`: grid heating at unresolved Debye length (default: the 2D analogue of the decision-80 reproducer).
- `t3_emwave.py`: transverse wave in a cold plasma vs the O7 cold-medium dispersion relation.
- `axis_map.py`, `c1map.py`, `c1_density.py`: density-conserving radial maps with a bounded slope at the axis (exp) and with a
  continuous slope (C1, the one that converges); uniform-density check. `c1scan.py`: the C1 map with a free slope at node 1.
  `c1_loading.py`: the loading error per map and ppc_r. `c1core.py`: the map with a linear axis core (not used). `rz_conv.py`: RZ solve convergence per map.
- `rz_axis.py`: the RZ axis case of decision 80 (40 n_c, 1 keV, k_p dr = 2.1) in the m = 0 TM model.

## The mover (step_implicit)
    u1 = u0 + q/m dt (Ebar + vbar x Bbar),  vbar = (u0 + u1)/(g0 + g1),  x1 = x0 + dt vbar
    J = Esirkepov deposit of the straight path x0 -> x1   (unchanged: Gauss law exact)
    Ebar_a = q-normalised work of the box closure's work field Et along the path with the Esirkepov weights / dx_a
Et is the work field of the box closure (docs/specs/linear_medium.md): W^{n+1} - W^n = -dt sum J . Et. So
m (g1 - g0) = q Ebar . dx = the field work of the particle's own current, and W + K is conserved to the tolerance of the
nonlinear solve; Bbar (any interpolation of (B^n + B^{n+1})/2 at the path midpoint) does no work.
Nonlinear solve: fixed point in the grid current J; per iteration one field solve (exact, linear) -> Et, one particle pass
(update u1 from the current iterate, "inner = 1"), one deposit; residual preconditioned by the lumped cold response
P = 1 + dt^2/4 sum q^2/(m g) S / V (one scalar deposit per step) and accelerated by Anderson(5). The step is committed
with the deposit of the final paths (Gauss law exact whatever the residual).
RZ: the radial work is divided by r_mid . dx_perp (r_mid the unit vector at the path midpoint), so F . dx = work for the
straight Cartesian move.

## Results (shape 2 unless noted)
**Identities.** Box step with random fields and currents: W^{n+1} - W^n = -dt A sum J . Et to 2-5e-16 (2D) and 3-6e-15 (RZ m = 0).
Hot relativistic plasma (u_th = 0.5, wp dt = 3, dy = 0.4 dx, 40 steps): implicit |dW+K|/E 1.3e-13, Gauss law 2e-14;
explicit +13x (heats), Gauss law 3e-14.

**Langmuir (T1, 1D, k dx = 0.098, cold).** w dt measured / Crank-Nicolson 2 atan(W dt/2):
| wp dt | CN theory | implicit | |dE|/E | outer its (tol 1e-10) | leapfrog theory | explicit |
|---|---|---|---|---|---|---|
| 0.3 | 0.297544 | 0.297543 | 1e-12 | 8.9 | 0.300893 | 0.300831 |
| 1.0 | 0.926653 | 0.926648 | 2e-11 | 12.9 | 1.046271 | 1.046035 |
| 1.9 | 1.518723 | 1.518717 | 3e-10 | 16.0 | 2.501604 | 2.500366 |
| 3 | 1.964846 | 1.964840 | 7e-10 | 16.5 | unstable | blows up |
| 5 | 2.380026 | 2.380021 | 1e-8 | 15.9 | unstable | blows up |
| 10 | 2.746492 | 2.746490 | 6e-8 | 19.0 | unstable | blows up |
Gauss law 1e-14 in all. Shape 1: the same within 4e-4 (the shape factor of the estimate), 14-31 iterations.
Solvers (tol 1e-10): plain Picard 12 its at wp dt = 0.3, no convergence in 200 at 1.9, diverges at 5 (contraction
(wp dt/2)^2); Picard + P 8.8 / 14.8 / 24.5; Anderson + P 8.9 / 16.0 / 15.9.

**Transverse wave in a cold plasma (T3, 1D, wp dt = 0.3, 1).** Measured w dt against the discrete relation of the box transport
with the Crank-Nicolson cold response (`disp1d.py`, derived from the step): k dx = 0.098 / 0.39 / 0.79 at wp dt = 0.3:
0.313097 / 0.488338 / 0.830057, relative error -8e-8 / +3e-8 / +6e-7; wp dt = 1: 0.931144 (-2e-9). The explicit leapfrog gives
0.316447 / 0.491946 / 0.834562 / 1.050671 (another discrete relation). The tan^2 form of decision 76 (0.316273 at wp dt = 0.3)
belongs to a different cold-fluid model.

**Grid heating (T2, 2D analogue of the decision-80 reproducer: 40 n_c, 1 keV, dx = 0.0125 um, dy = 0.0425 um, wp dt = 0.62,
dx/lD = 14, dy/lD = 48, 16 ppc, fixed ions, no filter), 29 fs:**
| run | K/K0 | (W+K)/E0 - 1 | particle passes per step |
|---|---|---|---|
| explicit shape 1 | 2.07 | +1.4 | 1 |
| explicit shape 2 | 1.049 | +9.5e-2 | 1 |
| explicit shape 2, dy = dx (dy/lD = 14) | 1.31 | +0.42 | 1 |
| implicit, tol 1e-6 | 0.97 | -4.3e-9 | 5.0 |
| implicit, tol 1e-10 (5.8 fs) | 0.974 | -1e-13 | 7.8 |
| implicit, tol 1e-4 (5.8 fs) | 0.974 | +1e-8 (random sign) | 4.0 |
K/K0 = 0.97 is the initial exchange with the thermal field fluctuations (W/K0 = 0.03), then flat. Inner per-particle
iterations to 1e-10 ("inner = 50") give the same result with 21 passes per step: one pass per outer iteration is best.

**Large wp dt (same cell, density raised; 1 keV, 4 fs):** passes per step at tol 1e-6 / 1e-10: wp dt = 2: 9 / 15;
5: 18 / 34; 10: 31 / 63; energy error 1e-7 ... 6e-6 (tol 1e-6), 1e-11 ... 3e-10 (tol 1e-10); Gauss law 7e-15. Explicit at
wp dt = 2.0: +27 % of the energy in 4 fs.

**RZ axis case (rz_axis.py, m = 0 TM, 40 n_c, 1 keV, dz = 0.0125, dr = 0.0425 um, k_p dr = 2.1, nz = 32, ppc 1, 2, 8), 29 fs:**
| run | K/K0 | K/K0 of electrons at r < 3 dr | (W+K)/E0 - 1 | passes |
|---|---|---|---|---|
| explicit, plain gather | 1.071 | 1.48 | +0.13 | 1 |
| explicit, matched gather | 1.065 | 1.10 | +0.10 | 1 |
| implicit, tol 1e-8 | 0.977 | 0.83 (noisy: few particles) | +9e-9 | 8.7 |
| explicit, dr = 0.085 um | 1.029 | 1.05 | +0.068 | 1 |
| implicit, dr = 0.085 um | 0.974 | 1.06 | -1.5e-10 | 6.5 |
Gauss law per node 5e-15 ... 9e-15 in all. The m = 0 TM model does not reproduce the strong growth of the C++ code
(4 modes: x18; 1 mode: x2.3 at 19 fs): there the TE part (J_phi, E_phi), the modes m >= 1, the projection filter or the
Higuera-Cary pusher must take part. It does show the near-axis heating of the explicit push (plain 1.48, matched 1.10)
and its absence with the implicit mover, whose W + K cannot grow by construction.

**RZ solver convergence near the axis (rz_conv.py; nz = 16, nr = 32, dr = 0.1, plasma r < 0.6, 30 steps, tol 1e-9):**
| dz/dr | wp dt | u_th | radial map | outer its mean / max | |dE|/E |
|---|---|---|---|---|---|
| 1 | 0.55 | 0.05 | Ruyten | 7.2 / 10 | 3e-12 |
| 3 | 1.64 | 0.001 | Ruyten | 11.9 / 12 | 6e-12 |
| 3 | 0.52 | 0.05 | Ruyten | 17.3 / 200 | 5e-8 |
| 1 | 1.64 | 0.05 | Ruyten | 28.2 / 200 | 2e-8 |
| 3 | 1.64 | 0.05 | Ruyten | 64-75 / 200 | 2.4e-5 |
| 3 | 1.64 | 0.05 | plain splines | 14.1 / 15 | 1.5e-11 |
Hot electrons that pass close to the axis stall the iteration with the shape-2 Ruyten map: in the axis cell xi = u^p,
p = 0.328, so dxi/dr ~ r^(p-1) is unbounded, and the energy-consistent radial force (work / dr) inherits it. With plain splines
the same case converges like the Cartesian ones. The axis map must have a bounded slope for the implicit mover.

**A bounded-slope density-conserving axis map (axis_map.py; Alexander chose this option, 2026-11-06).** Shape 2, cells j >= 1 as
decision 53, the axis cell with a one-parameter map solved with all node shares (Newton, residual <= 1e-14):
| axis-cell map | parameter | slope dxi/du at the axis | max / min slope in the cell |
|---|---|---|---|
| u^p (decision 53) | p = 0.3283 | 3.6e5 at u = 1e-9 (unbounded) | - |
| u + b u (1 - u) | b = 1.233 | - | non-monotone (slope -0.23 at u = 1): not usable |
| ((u + e)^p - e^p)/((1 + e)^p - e^p), e = 0.01 / 0.1 / 0.3 / 1 / 20 | p = 0.163 / -0.283 / -0.991 / -3.19 / -60.1 | 14.2 / 5.72 / 4.30 / 3.58 / 3.17 | ratio 47 / 22 / 18 / 18 / 20 |
| limit e -> inf: (1 - e^(-k u)) / (1 - e^(-k)) ("exp") | k ~ 3 | ~3.2 | ~3.2 / ~0.16 |
The exponential map is the flattest of the family; the slope cannot stay near 1: node 0 must get A_0 = dr^2/8 instead of the
13/8 A_0 of the plain spline.

**RZ solve with the maps (rz_conv.py, fixed path segments n = ceil(c dt max(dxi/dr) / 0.95), radial force per segment,
30 steps, tol 1e-9):**
| map | dz/dr | wp dt | iterations mean / max | |dE|/E |
|---|---|---|---|---|
| plain splines | 3 | 1.64 | 14.0 / 15 | 1.5e-11 |
| exp | 1 | 0.55 | 7.3 / 10 | 4e-12 |
| exp | 3 | 0.52 | 41.6 / 200 | 8e-7 |
| exp | 1 | 1.64 | 31.4 / 200 | 7e-7 |
| exp | 3 | 1.64 | 62.2 / 200 | 2.6e-5 |
| u^p (decision 53), adaptive segments | 3 | 1.64 | 75 / 200 | 2.4e-5 |
Two prototype bugs found on the way: (1) the segment count must be fixed (as in C++); (2) in RZ the radial force must be built per
path segment, f = (1/n) sum_s w_s / (rhat_s . dx_s) rhat_s (the path of a particle passing the axis has a non-monotone r(t), and one
direction for the whole path divides a finite work by a vanishing dr; with it, plain splines and fixed segments converge).
The remaining stall with the exponential map: the energy-consistent radial force is proportional to dxi/dr, which varies 20x inside
the axis cell and jumps at every cell edge (1 - b_j -> 1 + b_{j+1} for the decision-53 cells, 0.16 -> 1.28 at r = dr). The fixed
point is not smooth where hot electrons cross these edges. A C1 map (cubic Hermite with xi(r_j) = j and the node slopes as the
unknowns) was tried: its share equations did not solve (slopes 10, 5, 4.5, ... instead of -> 1).

**The C1 map (c1map.py, rz0.py rmap = "c1", 2026-11-07).** In cell c, with u = r/dr - c,
xi = c + u + (s_c - 1)(u^3 - 2u^2 + u) + a_c u^2 (1 - u)^2: a cubic Hermite with xi(r_j) = j, node slope s_0 at the axis and 1 at
every node j >= 1, plus a bump a_c u^2 (1 - u)^2 that moves charge inside the cell and leaves the end values and end slopes alone.
dxi/dr is continuous everywhere (across the axis too: xi is odd). The unknowns are a_0 .. a_{K-1} (one per node share) at a given
s_0; Newton converges in 4-10 iterations to 1e-14 (the bumps fall like 5/(2j+1) far out, as the decision-53 b_j ~ 1/(2j+1)).
| s_0 | largest slope | smallest slope | monotone |
|---|---|---|---|
| 1.0 | 2.17 | -0.17 | no |
| 2.6 | 2.60 | -0.009 | no |
| 2.8 | 2.80 | 0.006 | yes |
| 3.3 (used) | 3.30 | 0.032 | yes |
| 4.0 | 4.00 | 0.034 | yes |
The smallest slope sits inside the axis cell. Uniform plasma (c1_density.py, 4000 samples per cell): rho = 1 at every node to
1.6e-8 (exp 4e-9: the sampling; plain splines 1.625 at the axis).

RZ solve with the C1 map, same cases (run_c1.sbatch):
| map | dz/dr | wp dt | iterations mean / max | |dE|/E |
|---|---|---|---|---|
| C1 | 1 | 0.55 | 7.0 / 7 | 9.4e-14 |
| C1 | 3 | 0.52 | 7.9 / 10 | 1.3e-11 |
| C1 | 1 | 1.64 | 12.1 / 13 | 4.9e-12 |
| C1 | 3 | 1.64 | 19.5 / 61 | 6.7e-11 |
All converge to the tolerance; the stall of the exponential map was the slope jump. Cost: the largest slope 3.3 sets the fixed
segment count (nseg 4 at dz/dr = 1, 11 at dz/dr = 3) and the deposit uses the C1 map instead of decision 53's (a model change in RZ).

**A free slope s1 at node 1 (c1scan.py).** With s1 = 1 the map runs from 3.3 down to 0.03 inside the axis cell (ratio ~100);
freeing s1 flattens it: ratio of the largest to the smallest dxi/dr
| s1 \ s0 | 3.0 | 3.5 | 4.0 |
|---|---|---|---|
| 1.0 | 164 | 93 | 119 |
| 0.6 | 26 | 23 | 24 |
| 0.4 | 21 | 17 (0.20 .. 3.5, used in C++) | 17 |
| 0.3 | 21 | 16 | 15 |
(monotone for s0 >= 2.5 when s1 <= 0.6). The flatter map did not by itself help the C++ solve; see below.

**Loading (c1_loading.py).** The code loads ppc_r Gauss-Legendre points per cell at physical r; the deposit sees them through the
map. rho of a uniform plasma, largest error over the nodes (the axis node):
| ppc_r | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| decision 53 (u^p) | 2.6e-3 | 3.5e-3 | 4.3e-4 | 8e-5 |
| C1, s1 = 1 | 9.3e-2 | 1.1e-2 | 7.4e-4 | 4e-5 |
The C1 map needs ppc_r >= 8 near the axis.

**C++ (decision 82, 2026-11-07).** Three more findings on the way to a converging RZ solve in C++ (hot column, omega_p dt = 2.5,
dz = 2 dr, ppc_r = 8, tol 1e-12):
1. J_phi of m = 0: r_j dphi jumps by 2 pi r_j wbar when a segment passes over the axis (dphi -> +-pi); replaced by wbar L / r_j with
   the angular momentum L = x0 y1 - y0 x1 of the move (smooth, no continuity constraint for m = 0). Stalled steps went from 1e-5 to
   1e-8 ... 3e-5.
2. The Picard step of one particle oscillates with period 2 when its path ends next to the axis (r1 = 0.004 / 0.007, force 5 %
   apart): the radial force is the field gradient times xi'^2 (~11), so |dG/du1| > 1 for that particle, and the global Anderson on J
   does not see it (u1 is not in its iterate). A Newton step (difference Jacobian) for the particles whose path comes within 2 dr of
   the axis: every step to 1e-12 in 21-24 iterations (m = 0, 9 steps; 30 steps: 23-26 on average, max 27 / 61).
3. m >= 1: deposit_seg's (Delta W) e_bar + W_bar Delta e has the cross terms S_r1 e0, S_r0 e1, which jump when an end of a segment
   passes next to the axis. With the split J_z from the folded complex profiles S_r e of each end, J_r with the phase
   ((z0 + z1)/(r0 + r1))^m and J_phi the rest, m = 0, 1 converged in 28 of 30 steps; 2 stalled at 5e-5 and 4e-4 (a path 0.006 dr from
   the axis; backtracking in the particle Newton step made it worse: 4 stalls with m = 0, 1, 1 with m = 0). Three modes (even m = 2,
   whose folded charge at the axis keeps the arbitrary phase) stall in most steps. Plasma kept at r >= 3 dr: 2 and 3 modes converge in
   ~20 iterations, so the trouble is only the axis.
4. What the stalled m = 1 steps cost (test M7 with modes = 2, 30 steps, tol 1e-12): |d(W + K + wall loss)|/E 7e-5 (uniform grid)
   and 1.3e-4 (stretched), Gauss 4e-14, 47 / 73 iterations per step on average (m = 0 alone: 7e-15, 23 / 26). Which steps stall
   depends on the OpenMP summation order: 10 repeats on the uniform grid (4 threads) gave 4.6e-5, 5.3e-4, 3.2e-3, 1.5e-4, 1.3e-4,
   4.8e-3, 5.0e-5, 7.2e-5, 8.0e-5, 9.0e-4 with 2, 3, 5, 3, 4, 3, 4, 2, 2, 4 unconverged steps. m = 1 accepted (decision 82).
5. A linear core xi = s0 u for u < uc in the axis cell (c1map.UC, c1core.py; idea: dxi/dr = xi / r there makes the m = 1 force
   isotropic on the axis, without it the radial factor xi' and the azimuthal factor xi / r part at first order in r). Monotone only
   for small cores (s0 = 2.5, s1 = 0.4, uc = 0.2: slopes 0.17 .. 2.5). In C++ it made m = 0 worse (3 of 30 steps stalled at up to
   7e-2; the same s0 = 2.5, s1 = 0.4 without the core: all 30 converged, 24 iterations) and did not remove the m = 1 stalls (2 of
   30). Not used; probably the large curvature where the core joins the Hermite.

## Findings that matter for the C++ design
- One particle pass per nonlinear iteration (joint fixed point of J and u1) with Anderson on J: 4-8 passes per step at the
  CHF/TNSA parameters (wp dt < 1), 9-31 at wp dt = 2-10 (tol 1e-6). Plain Picard fails above wp dt ~ 1.5.
- The per-particle force Ebar = work / displacement has a round-off floor ~1e-11 (cancellation for tiny displacements);
  harmless for the energy (the work is exact), but per-particle convergence tests must not ask for less.
- The transposed (energy) gather near the RZ axis sees the Ruyten map: it is the path version of rz_gather = energy
  (decision 80), with the same distortion of a uniform transverse field inside the first cell.
- Shape 1 works with the joint iteration (Langmuir 14-31 its) but its gather of the longitudinal field is piecewise
  constant along the path; with inner per-particle Picard it did not converge in a hot relativistic test (5e-2). Shape >= 2.
