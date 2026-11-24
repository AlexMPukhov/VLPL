# O6: energy-conserving implicit particle mover with the iRIP box closure

Status: option A chosen (Alexander, 2026-11-06); implemented for Cart3D and 2D (`numerics.mover = implicit`, src/implicit_mover.cpp,
decision 81, section "Implementation" below) and for RZ with m = 0 (decision 82, section "RZ"); m >= 1 in RZ open. Prototype and all
numbers: `research/o6_mover/` (README.md). Roadmap track O, item O6.

## Implementation (v1, 2026-11-06)
- `irip_trial` (src/irip.cpp): the iRIP step with the iterate J without changing the fields; returns the work field Et at the
  positions of Jx, Jy, Jz and (B^n + B^{n+1})/2. In 3D Et_y, Et_z carry the corner terms -D^2/4 delta_{y,z} delta_x mu_t Ex.
- `implicit_mover_step` (src/implicit_mover.cpp) replaces push_deposit + the field step: per iteration irip_trial, one particle pass
  (u1 from the path work of Et, then the Esirkepov deposit of the new path), the preconditioned residual and Anderson(5) on J (global
  norms); commit = the field step with the final deposit, then x, u from the final iterate (the stored-position rule of decision 58).
- Paths are cut into n = ceil(c dt / h_min) equal segments for every particle (fixed, decision 81): the adaptive split made the
  current discontinuous in the iterate and the solve stalled at 1e-3.
- Tests (tests/test_mover.cpp): M1 work-field identity 3D 0, 2D 9e-16; M2 hot plasma (omega_p dt = 3, u_th = 0.5, h_perp = c dt/2,
  3D/2D, shapes 2/3, tol 1e-12): |d(W + K)|/E 3e-14 ... 1.4e-13, Gauss 4e-14 ... 3e-13, div B < 1e-14, 24-34 iterations per step;
  M3 cold Langmuir omega_p dt = 0.5 / 3 / 10: w dt 0.488400 / 1.962527 / 2.745525 against Crank-Nicolson 0.488447 / 1.962619 /
  2.745564; M4 2D, 40 n_c, 1 keV, dx/lD = 14, dy = 3.4 dx, 300 steps, no filter: implicit dE/E 2.5e-14, dK/K -0.025 (the initial
  exchange with the field noise), explicit +4.2e-2; M5 config. tests/mpi.sh: the streaming plasma with the mover, periodic and open x,
  1 vs 8 ranks.

## Physics and equations
Goal: no grid heating at unresolved Debye lengths and no omega_p dt < 2 limit in solid-density plasma (CHF, TNSA), RZ and Cart3D,
keeping exact charge conservation (Esirkepov on the RipCC charge grid). With c dt = dx the time step is fixed by the grid:
at lambda0/64 ... /256 and 40-200 n_c omega_p dt = 0.3-1.4, so the gain in practice is energy conservation (no heating, no
numerical axis growth); omega_p dt > 2 matters only for coarse cells in thick targets.

Crank-Nicolson in the particles, coupled to the box closure (decision 17) through its **work field** Et (docs/specs/linear_medium.md:
Et_x = mu_t Ex, Et_perp = transport midpoints - dt^2/4 delta_perp delta_x mu_t Ex - dt/2 J_perp), for which exactly
W^{n+1} - W^n = -dt sum_cells V J . Et  (prototype: 2e-16 in 2D, 6e-15 in RZ):

    u1 = u0 + (q/m) dt (Ebar + vbar x Bbar),   vbar = (u0 + u1) / (g0 + g1),   x1 = x0 + dt vbar
    J  = Esirkepov deposit of the straight move x0 -> x1 (the existing deposit, segments as decision 28/44)
    Ebar_a = (sum over the deposit's weights of the a-current: q w_a Et_a) / (x1 - x0)_a   (work along the path / displacement)

Then m (g1 - g0) = vbar . (u1 - u0) m = q Ebar . (x1 - x0) = the field work of the particle's own current: W + K is conserved to
the tolerance of the nonlinear solve, for any B interpolation (vbar x Bbar does no work), relativistically, on any grid. The
gather is the transpose of the Esirkepov deposit (in 1D: the shape of one order lower averaged over the path; the cross terms
(S^n + dS/2) of Esirkepov included). Literature: Chen, Chacon & Barnes, JCP 230, 7018 (2011) (1D ES, JFNK, orbit averaging);
Markidis & Lapenta, JCP 230, 7037 (2011); Angus et al., JCP 456, 111030 (2022), 491, 112383 (2023) (EM, exact energy + charge,
dense plasmas; 2024 axisymmetric pusher), the WarpX theta-implicit solver (theta = 1/2).
**RZ:** Cartesian particles (decision 44); the radial work is divided by r_mid . dx_perp (r_mid: unit vector at the path midpoint),
so F . dx = work for the straight move. With the Ruyten maps (decision 53) this gather is the path version of
rz_gather = energy (decision 80): exact energy, and the same distortion of a uniform transverse field inside the first cell.

## Discretisation and solver
Unknown: the grid current J^{n+1/2}. Per nonlinear iteration: (1) the iRIP field step with J (unchanged, exact linear solve) gives
Et and B^{n+1}; (2) one particle pass: gather Et along the current path iterate with the deposit's weights, update u1 (stored per
particle, 3 doubles); (3) deposit J' from the new paths; residual r = P^{-1}(J' - J), P = 1 + (dt^2/4) sum q^2/(m g) S/V per current
component (lumped cold response, one scalar deposit per step); Anderson acceleration (depth 5) on J. Commit: field step with the
deposit of the final paths, so the Gauss law is exact whatever the residual; the energy error is set by the tolerance.
Measured (2D, shape 2): one particle pass per iteration beats converging each particle first (7.8 vs 21 passes per step).

| case (shape 2) | passes per step, tol 1e-4 / 1e-6 / 1e-10 | energy error |
|---|---|---|
| 40 n_c, 1 keV, CHF cell (omega_p dt = 0.62) | 4.0 / 5.0 / 7.8 | 1e-8 / 4e-9 (29 fs) / 1e-13 |
| omega_p dt = 2 / 5 / 10 (same cell, denser) | - / 9 / 18 / 31 ; tol 1e-10: 15 / 34 / 63 | <= 6e-6 (tol 1e-6) |
| cold Langmuir omega_p dt = 0.3 ... 10 (tol 1e-10) | 9 - 19 | <= 6e-8 |
Plain (unpreconditioned) Picard fails above omega_p dt ~ 1.5 (contraction (omega_p dt / 2)^2).

## Cost and GPU fit
A pass = Esirkepov weights of the path (once, reused for gather and deposit when fused) + gather of Et and B + deposit, plus one
iRIP field step per iteration. At CHF/TNSA parameters (tol 1e-6) a step costs about 5 explicit steps (RZ CHF: 88 % of the time is
the push, so ~4-5x wall time); at omega_p dt = 5-10 about 20-30x. Memory: u1 per particle, P and 2 x 5 Anderson vectors of the
current size. All work is the existing per-particle / per-cell kernel structure (par_for, deposit atomics as now), plus a few
par_sum dot products and a 5 x 5 least-squares solve per iteration; no mass matrices, no new communication pattern (the
iteration count is global; MPI: the field step's partition method and the current exchanges per iteration, norms by allreduce).
Cheaper start guesses (J extrapolated from the last two steps) should save 1-2 iterations.

## Options for Alexander
A. **This scheme (recommended)**: exact energy and exact charge, reuses deposit, field step and Esirkepov segments; cost 4-8x of
   the push at the CHF parameters, more at large omega_p dt (a better preconditioner, the O7 linear-medium solve with the local
   wp2, would cut the 18-31 iterations; later).
B. Semi-implicit mass-matrix scheme (Lapenta's ECsim, JCP 334, 349 (2017); RelSIM, Bacchini, ApJS 268, 60 (2023); WarpX
   semi_implicit): one linear solve, no nonlinear iteration, but the current is deposited at a fixed x^{n+1/2}: not Esirkepov,
   so the Gauss law is not exact (conflicts with the code's rule); mass matrices of 9 (2s+1)^d entries per node (3D shape 2: 1125)
   and, in RZ, mode-coupled ((2M-1)^2 blocks). Not recommended.
C. Explicit energy-conserving correction (Ricketson & Hu, JCP 537, 114098 (2025); relativistic version arXiv 2605.18542): explicit
   particles, a per-particle analytic energy correction, linearly implicit fields; cheap, but the correction can fail for single
   particles and charge conservation with Esirkepov is not shown. Worth a check if A is too slow.
D. Stay explicit: shape >= 2, filter, rz_gather = matched / energy (decisions 64, 79, 80). The prototype's 2D CHF cell heats
   +9.5 % of the energy in 29 fs without filter (shape 1: x2.4; square cells: +42 %).

## Tests (for the C++ implementation)
1. Box step identity W^{n+1} - W^n = -dt sum V J . Et with random fields and currents (Cart3D, 2D, RZ per mode): 1e-14.
2. Hot relativistic plasma, h_perp < c dt, moves over several cells: |d(W + K)| / E <= 10 x tol, Gauss law and div B at round-off.
3. Cold Langmuir: w dt = 2 atan(W dt / 2) to 1e-5 for omega_p dt = 0.3 ... 10 (prototype: 2e-6).
4. Transverse wave in a cold plasma: the discrete relation of the box transport with the Crank-Nicolson cold response
   (research/o6_mover/disp1d.py; the cold limit of the mover is the O7 Drude medium).
5. Grid heating: dx / lambda_D = 14 - 48, 29 fs: K constant to the tolerance after the initial exchange (prototype: -4e-9).
6. RZ axis case of decision 80 (40 n_c, 1 keV, 4 modes): bounded (W + K conserved); explicit for comparison.
7. MPI 1 vs 8 ranks to 1e-11 (the iteration count must not depend on the decomposition: global norms).

## RZ (2026-11-07, decision 82)
- Radial map: the C1 density-conserving map (rz_xi shape 4, rz_c1_coefficients): per cell a cubic Hermite with the node slopes
  dxi/dr = 3.5 / dr (axis), 0.4 x 2 / (dn_0 + dn_1) (node 1), 2 / (dn_{j-1} + dn_j) (beyond) and a bump ca_j u^2 (1 - u)^2 from the
  node shares; dxi/dr continuous, 0.20 ... 3.5 on a uniform grid. Used for the deposit and rho only when the mover is implicit.
- rz_irip_trial (src/rz_irip.cpp): the RZ box step with the iterate J per mode, without changing the fields: Et at the J_z, J_r,
  J_phi points (Et_z = Ez*, Et_r, Et_phi with the transport terms and the corner terms -D/4 d_r, +D/4 G_phi of the change of Ez*
  along z) and (B^n + B^{n+1})/2. W^{n+1} - W^n = -dt sum_m w_m dV J_m . Et_m exactly.
- Particle pass (src/rz_particles.cpp): paths in n = ceil(c dt max(1/dz, dxi/dr)) segments; the current of a segment from
  seg_currents (the deposit and the work share it: exact transpose). Charge-conserving per mode; J_phi of m = 0 from the angular
  momentum of the move (wbar L / r_j, L = x0 y1 - y0 x1); m >= 1 without the cross terms of the explicit deposit. Force per segment
  f_s = f_ref + (W_s - f_ref . ds) ds / |ds|^2 (f_ref: the exact derivative of the work at the midpoint), Ebar = the mean.
  Particles whose path comes within 2 dr of the axis take a Newton step for their u1 (the Picard step oscillates there: the radial
  force is the field gradient times xi'^2).
- Solve as in Cart3D: Anderson on (J_z, J_r, J_phi) of all modes, preconditioner 1 + D^2/4 wp2 of the m = 0 density.
- Tests: M6 work field (3 modes, stretched grid, dz = 2.5 dr): 0 / 2e-15; M7 hot column at omega_p dt = 2.5, m = 0, uniform and
  stretched grid (and m = 0, 1 on the uniform grid, rare unconverged steps allowed): W + K (+ wall loss) 8e-15, Gauss 5e-14, density 1e-3 (ppc_r = 8), 23-26 iterations per step; tests/mpi.sh: RZ
  mover 1 vs 8 ranks along z.

## Open
- **RZ m = 1** (accepted, decision 82): converges except for rare steps with a path within ~0.01 dr of the axis (2 of 30 steps of
  M7 stalled at 5e-5 and 4e-4); 10 repeats of M7 with m = 0, 1 (the stalled steps depend on the OpenMP summation order): |d(W + K)|/E 5e-5 ... 4.8e-3 over 30 steps (median ~1.4e-4), 2 ... 5 of 30 steps unconverged. The unconverged steps are counted in MoverStats,
  Gauss stays exact.
  A Cartesian treatment of the m = 1 current in the axis cells could remove them. **Even m >= 2:** no smooth deposit at the axis
  (decision 50's fold), rejected.
- **RZ m = 0 at c dt >~ 4 dr:** the tests/mpi.sh RZ deck (dz = 4 dr, stretched grid, omega_p dt = 2.5) has rare steps where the
  global iteration stalls (4 of 15 at maxit 100, 1 of 15 with Anderson depth 10: 1.9e-4), many slow near-axis particles changing
  together; the 1 vs 8 rank energies then differ by 6e-5 (tests/mpi.sh asks for 1e-8, so its RZ mover case runs at dz = 2 dr).
  Candidates: the C1 map's small effective cell next to the axis (dr / 3.5) seen by moves of
  4 dr, a preconditioner that knows the radial coupling. Open.
- Shape 1: converges with one pass per iteration, but its path gather is piecewise constant; restrict to shape >= 2 (shape 3 Cart3D).
- The m = 0 TM prototype does not reproduce the strong C++ axis growth (x18 with 4 modes; there x1.5 plain / x1.1 matched near the
  axis, +13 % / +10 % total energy, implicit 1e-8): TE / J_phi, m >= 1, the filter or the pusher must take part (a hint for the
  axis-instability work).
- Filters: a current filter between deposit and field step breaks the transpose unless the gather applies the adjoint filter
  (the projection filters of decisions 69/70 are symmetric in the energy norm: apply them to Et as well).
- Laser/TFSF/CPML/window: the work identity with sources and absorbers is W^{n+1} - W^n = -dt J . Et + boundary terms; the mover
  needs only Et, so these carry over, but each has to be checked.
