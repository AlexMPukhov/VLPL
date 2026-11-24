# Linear-response media in irip: cold Drude plasma and Lorentz dielectrics without particles (track O, O7)

Status: physics settled and prototyped in 2D (Alexander + cloud session, 2026-10-28); implemented in C++ 2026-11-07
(decision 84, section "C++ implementation" below).
Prototype: `research/linear_medium/` (2D: `box2d.py`, `setup.py`, `test_energy.py`, `conv2d.py`; the 1D scripts and
the movie scenarios stayed in the cloud session).

## Purpose
Model reflectors and optics as a linearly responding continuum instead of particles: plasma mirrors of any
shape (flat, tilted, parabolic for CHF), lossy plasmas, dielectric slabs, lenses and windows. There is no
omega_p dt limit (the scheme is unconditionally stable) and no particle cost in those regions. The medium
coexists with particles elsewhere: the Gauss law is unaffected (Theorem 1 holds for any current).

## Equations (per current component, at the cell and time centre like the RIP-CC currents)
  (P^{n+1} - P^n)/dt = mu_t J
  (J^{n+1} - J^n)/dt = wp2(x) Et - w02(x) mu_t P - nu(x) mu_t J
The current given to the field step is mu_t J. Drude: w02 = 0. Lorentz: eps(w) = 1 + wp2/(w02 - w^2 - i nu w).
**Et must be the work field of the box closure** (paper, Theorem 2):
  Et_x = mu_t Ex
  Et_y = (M+_y + M-_y)/2 - dt^2/4 delta_y delta_x mu_t Ex
  Et_z = (M+_z + M-_z)/2 - dt^2/4 delta_z delta_x mu_t Ex
where M+-: the characteristic midpoints of the transported T+-. Then exactly
  [W + K]^{n+1} - [W + K]^n = -dt sum nu (mu_t J)^2 / wp2 <= 0,   K = sum (J^2 + w02 P^2)/(2 wp2).
Any other drive field (predictor E, E at n, E at n+1) loses this identity. This is why earlier attempts with
explicit RIP were of poor quality.

## Discretisation
- M+-_y contains -dt/2 mu_t J_y (the medium current's own source in the transport). Eliminate it locally:
  u_y = (2 J + dt wp2 Et0_y - dt w02 P)/(2 + dt nu + dt^2 w02/2 + dt^2 wp2/2),
  with Et0 = Et without the medium current. For x: u_x = (2 J + dt wp2 mu_t Ex - dt w02 P)/(2 + dt nu + dt^2 w02/2).
- The implicit unknowns stay Ex^{n+1}, Bx^{n+1}. The medium makes the operator variable-coefficient. Solve with a
  Krylov method, preconditioned by the vacuum box solver (FFT/DST or Hankel + tridiagonal along x) after
  scaling the Ex equation by the local diagonal 1 + dt^2 wp2_x/(2(2 + dt nu + dt^2 w02/2)).
  Prototype: GMRES, 3 iterations (glass), 5-9 (9 n_c), 11-19 (100 n_c), 24 (1000 n_c, wp dt = 10) for tolerance 1e-9.
  The operator is the Crank-Nicolson form of a dissipative, energy-conserving system: try MINRES/CG on the
  symmetrised form in C++.
- Medium parameters per current component at its own staggered position, from the sub-cell area or volume
  fraction of the body (4x4 supersampling in the prototype). This smooths the staircase.
- Vacuum regions: the medium adds nothing (wp2 = 0); without any medium the direct vacuum solve is used.

## Verified in the prototype
1. Energy balance W + K + dissipated = const to 1e-14 ... 1e-11 (random fields, random media,
   wp dt up to 30, dy = 0.4 dt, 200 steps), and to 1e-12 in all 2D scenarios. Unconditionally stable.
2. 1D: time domain = exact discrete 2-port theory to 8e-6. Lossless overdense: |r| = 1 exactly for any
   wp dt. The phase error is a sub-cell shift of the surface (to dt/2 as wp dt -> infinity), second order.
   Absorption is second order and too low while the skin depth is unresolved (100 n_c, nu = 0.1 w0:
   1 - |r|^2 = 0.0167 at lambda/48 vs 0.0199 exact).
3. 2D movies (c dt = dx = lambda/20, dy = dx/2, 20 x 20 lambda, `run.py`; incidence 22.5 deg, specular along -x):
   - 100 n_c mirror (wp dt = 3.1): reflected beam along -x (0.3 deg p, -0.2 deg s); p-pol leaves about 2 %
     of the energy in surface waves along the staircased tilted surface (see 4).
   - Lossy plasma 9 n_c, nu = w0; Lorentz glass eps(w0) = 2.25 at 45 deg: refraction 27.7 deg vs Snell 28.1 deg
     (the measured index 1.518 vs 1.5 is the second-order box dispersion at 28 deg plus CN warping).
   - Parabolic mirror 1000 n_c (wp dt = 10), f = 4 lambda: focus at (10.90, 9.98) vs geometric (11, 10).
4. Resolution study (`conv2d.py`, 12 x 12 lambda, beam waist 1.5 lambda, Fresnel weighted over the beam spectrum),
   error at lambda/dx = 10 / 20 / 40:
   | case | 10 | 20 | 40 | order |
   |---|---|---|---|---|
   | glass, s, R (0.0964) | 1.9e-2 | 4.2e-3 | 1.0e-3 | 2 |
   | glass, p, R (0.0092), surface x-aligned | 1.1e-2 | 2.5e-3 | 8e-4 | 2 |
   | lossy 9 n_c, s, absorption (0.445) | 1.9e-2 | 5.5e-3 | 1.4e-3 | 2 |
   | lossy 9 n_c, p, absorption (0.509), tilted staircase | 8.4e-2 | 4.0e-2 | 1.9e-2 | 1 |
   | 100 n_c mirror, p, 1 - R (surface waves), tilted staircase | 7.0e-2 | 3.0e-2 | 1.7e-2 | 1 |
   | 100 n_c mirror, s, 1 - R | 5e-4 | 8e-4 | - | - |
   Energy error <= 6e-12 in all runs. GMRES iterations fall with resolution (wp dt smaller).
   Conclusion: the medium is second-order accurate; the only first-order error is p-polarisation on surfaces
   tilted against the grid (staircase -> surface charge on steps -> surface waves).

## Tests for the C++ implementation
1. Energy identity with random fields and media (Cart3D periodic, RZ m = 0, 1): drift <= solver tolerance.
2. 1D normal incidence on a Drude slab: |r| = 1 exactly; reflection phase equal to the discrete 2-port
   formula (per cell: z T_out = T_in - g (T+_in + T-_in)/2, g = (z+1)/2 wp2 dt^2 / ((z+1) c/2 - 2),
   c = 2 + wp2 dt^2/2 + nu dt, z = e^{i w dt}) to 1e-8.
3. Oblique-incidence Fresnel: lossy plasma and dielectric, s and p, against the spectrally weighted Fresnel
   formula; second-order convergence lambda/10 ... lambda/40.
4. Snell refraction angle and parabolic-mirror focus position (stored values with tolerances).
5. Gauss law with particles plus a medium region: round-off.
6. MPI: medium crossing rank boundaries, 1 vs N ranks to 1e-11.

## Risks / open
- p-pol surface-wave coupling on staircased tilted surfaces: 3 % at lambda/20, 1.7 % at lambda/40 (22.5 deg),
  first order. Remedies: tilt the grid so that the mirror surface
  is x-aligned where possible (normal incidence, RZ); a conformal (cut-cell) weighting of wp2 per face;
  a short density ramp (physical pre-plasma) for HHG targets.
- Unresolved skin depth: reflection exact, absorption too low. A sub-cell surface-impedance model is possible.
- Nonlinear extension (relativistic fluid, ponderomotive steepening) would lose the exact identity. Keep it
  linear; nonlinear regions use particles.

## C++ implementation (2026-11-07, decision 84)
- Deck: `medium [<name>] { model = drude|lorentz, density | permittivity, collision_frequency, resonance, profile, supersample }`,
  `numerics.medium_tol = 1e-10, medium_maxit = 200, medium_restart = 30`; output dir/medium.txt (K, W_dissipated, W + K + W_diss,
  GMRES statistics). Files: src/medium.{hpp,cpp} (profile build, Cart3D current, commit, charge, GMRES), src/medium_rz.cpp (per mode),
  src/medium_sim.cpp (checks, setup, output), the step in src/irip.cpp and src/rz_irip.cpp.
- Unknowns X = Ex^{n+1}, Y = Bx^{n+1} (RZ: Ez, Bz of all modes). One operator application = the box means of the iterate, Et0 (decision
  81's work field with the external current), the local u (the medium's own -dt/2 u in Et_y, Et_z eliminated), one pass with J + u.
  GMRES(30), right-preconditioned: the Ex part divided by its local diagonal, then the vacuum solve of each kind. Krylov vectors are
  applied as (G(0) - G(s v)) / s + v with s = |G(0)|. Start: E^n, or the last solution of the same step (the implicit mover calls the
  medium solve once per iteration through irip_trial; the particles then see Et including the medium's own source).
- Weighting: the profile's sub-cell volume fraction at each current position (supersample^3 points, RZ r-weighted); the MPI and
  GPU paths are the same kernels (par_for / for_owned).
- Measured (tests/test_medium.cpp, tests/test_medium2.cpp, CPU, 16 threads):
  - energy identity, random fields and media, 30 steps (tol 1e-13): 2D 3e-15 ... 8e-13, 3D 8e-15 ... 2.4e-13 for omega_p dt <= 3, 10,
    10 with nu dt = 1, Lorentz (w0 dt = 4, nu dt = 0.3) and 30; RZ (3 modes, stretched grid, dz = 2.5 dr, omega_p dt <= 10) 3.7e-14.
    GMRES 14-92 iterations per step for these grid-scale random fields; 2-7 for the optics cases below.
  - |r| = 1 at normal incidence on a lossless overdense slab (1D-like, lambda/20): |R - 1| 7e-12, 3e-12, 3e-12 at omega_p dt = 0.5, 3, 20.
  - oblique s-pol at 30 deg on a grid-aligned half space (dy = dx, the pulse a forward wave per x component; with one angle for all
    components it starts with 8e-4 of its energy moving backwards, which looked like a floor of the Fresnel error):
    | case | lambda/10 | lambda/20 | lambda/40 |
    |---|---|---|---|
    | glass eps = 2.25, R (ref 0.05828) error | 1.6e-2 | 3.5e-3 | 8.5e-4 |
    | glass, Snell angle (ref 19.460 deg) error | 0.285 deg | 0.056 deg | 0.002 deg |
    | 9 n_c, nu = omega0, absorption (ref 0.42327) error | 7.4e-2 | 2.3e-2 | 6.0e-3 |
    second order; the absorption converges more slowly at lambda/10 (skin depth lambda/19 unresolved, as in the prototype).
  - Gauss law with particles plus media (3D, a hot plasma half inside a 50 n_c layer and a Lorentz layer): 4e-15 explicit, 3.6e-15 with
    the implicit mover (W + K_particles + K + W_diss to 7e-16, 8 mover iterations per step); RZ m = 0, 1: 1.3e-14.
  - parabolic mirror (100 n_c, f = 4 lambda, waist 3 lambda, NA 0.75): focus -0.353, -0.223, -0.143 lambda from the geometric focus at
    lambda/12, 16, 24 (second order towards about -0.06 ... -0.08); physical optics with the propagated Gaussian beam: -0.037 (with
    the collimated amplitude alone +0.135). The remaining difference of the extrapolated value is within lambda/20.
  - MPI (tests/mpi.sh, 8 ranks): the 2D mirror with CPML in y (K 11.7, dissipated 1.8 by step 280) and the RZ glass + plasma deck
    (K 1110) equal to 1 rank in all printed digits with the same GMRES iteration counts; the implicit mover through a Drude layer
    (omega_p dt = 4.5) 9.4e-13.
  - GPU (RTX 5090, CUDA 13.0, sm_120; VLPL_MEDIUM_FAST: lambda/10, 20): every medium test passes with the CPU's numbers to the printed
    digits; sm_90 compiles.

## Bound electrons: model = bound (2026-11-08, decision 85)
An option of the `medium` block, not the default: the bound electrons of a gas as V-type Maxwell-Bloch atoms. Motivation: Rb vapour
near an 800 nm laser (D1 795 nm, D2 780 nm; detunings 0.6 % and 2.5 %), where the transitions saturate and Rabi-cycle at
ionizing intensities and a Lorentz oscillator does not apply. The physics is that of GEM-PIC-RZ's `bound_electrons` block.
- Deck: `medium <name> { model = bound, atom = Rb | wavelength1, strength1[, wavelength2, strength2], density (atoms, n_a / n_c or
  '<n> cm^-3'), polarization = x|y|z (the dipole axis, default z), profile, supersample }`; `numerics.medium_bound_tol = 1e-8`,
  `medium_bound_maxit = 30`. Effective dipoles from the oscillator strengths, d_j^2 = f_j hbar e^2 / (2 m omega_j): Rb D1 1.73,
  D2 2.44 e a0 (Steck: f = 0.342, 0.695). One `bound` block per run; Cart3D and cartesian2d.
- Per node (the positions of the current component of the dipole axis) the density matrix rho (9 reals) of ground 0 and excited
  1, 2; H(E) = diag(0, w1, w2) - E X, X_0j = a_j = d_j / (e a0 alpha); P = n_a sum_j cP_j Re rho_0j, cP_j = 2 (d_j / e a0) a0 k0.
  Weak field: chi = n_a sum_j f_j / (w_j^2 - omega^2). The carrier is resolved, no rotating-wave approximation.
- Step (src/bloch.hpp): rho -> U rho U^+, U = (1 + iK)^-1 (1 - iK), K = (dt/2) H'(Et) with Et the work field of the box closure.
  U commutes with H'(Et): tr(rho H'(Et)) is conserved, so K_b = ef n_a sum_j w_j' p_j changes by exactly Et u dt, the field's loss
  (ef = cP_j' / (2 a_j') must be the same for all lines). w' = (2/dt) tan(w dt/2) (exact free precession); a_j' = a_j / cos x_j
  (the Cayley step drives a resonant line with a' cos x: Rabi frequency exact), cP_j' = cP_j cos(xbar) / cos x_j (chi exact at the
  line up to O(x (x_j - xbar))), ef = alpha a0 k0 cos(xbar), x = w dt / 2. Needs w_j dt < 2 (checked).
- Coupling: u_b = n_a (P(rho^{n+1}) - P(rho^n)) / dt is nonlinear in Et. Inside the linear solve of decision 84 it is linearized at
  Et*: u_b = u* + s (Et - Et*), s = du_b / dEt by central differences of the node step; with a linear medium on the same component
  Et = (Et0 - kap dt/2 (a_l + a_b)) / (1 + kap dt/2 (b_l + s)) per node (u_l = a_l + b_l Et, u_b = a_b + s Et). After each GMRES
  solve Et* <- the Et of the solution until max |Et - Et*| <= medium_bound_tol max |Et| (Newton; warm starts of GMRES in the
  step; a new step starts from the last Et). Commit: rho with the final Et, bP += dt u_b (Gauss law to round-off with -div bP).
- medium.txt: K includes K_b; the columns newton_total, newton_max, newton_unconverged are appended.
- Relaxation (2026-11-08): `lifetime` (T1, excited -> ground) and `dephasing_time` (T2, pure dephasing), exact exponentials over
  dt/2 in a symmetric Strang splitting around the Cayley step; dissipated = dt Et u - Delta K_b per atom (exact bookkeeping).
- Ionization (`ionization = adk`): ADK (s-states, GEM-PIC-RZ form, constant beyond its maximum) per level at |Et| of the step: level k
  decays as exp(-w_k t), coherence kl as exp(-(w_k + w_l) t / 2), within the same splitting; the trace is the neutral fraction
  (Rb: 4.177 eV, excited 2.618 / 2.588 eV built in; else `ionization_potentials`). With `electrons`, `ions` (species of the deck,
  density 0 allowed) the ionized density per atom position becomes an electron and an ion at rest once it reaches
  n_a / `ionization_ppc` (weight = accumulated density x volume): the pair deposits no charge (Gauss law untouched), the electron is
  born at rest (tunnel ionization). The ionization energy and the removed internal energy are in the dissipated energy, so W +
  K_particles + K_b + dissipated is conserved (with the O6 mover exactly). medium.txt: ionized, emitted.
- Checkpoints with media (all models): J, P and the coefficients (wp2, w02, nu; in a moving window they belong to the shifted
  profile) of the linear media; the atoms' density matrix, n_a, bP, the last Et (the predictor),
  the ionization accumulator (RZ: the angle arrays as (n_phi, N_r, N_z) datasets, the lagged atom current, bP per mode); the
  dissipated, ionized, emitted energies and the solver statistics as attributes. The restart is bit-identical on the same ranks.
- RZ (`geometry = rz`, dipole along x, `n_phi` samples, default 4 x modes, at least 2 m_max + 1; src/medium_rz_bound.cpp): atoms
  at the Jr positions (z_{i+1/2}, rh_j). Et_x = Et_r cos - Et_phi sin from the modes, Et_phi interpolated linearly to rh_j, on the
  axis Et_phi,1(0) = i Et_r,1(rh_0) (regularity); the atom current back to (u_r, u_phi) per mode with the transpose of that map in
  the energy inner product of the RZ step (exact quadrature for n_phi >= 2 m_max + 1), so the energy exchange is exact. The
  atoms' own source couples neighbouring nodes: the iterate uses the atom current of the previous outer iteration (lag), the outer
  iteration converges both Et* and the lag. Atoms must stay clear of the last radial cells and the absorber.
- Measured (RZ, tests/test_bound.cpp): W + K + K_atoms with random fields in modes 0..2, Rb 5e-2 n_c on 12 angles and a Lorentz
  layer, stretched grid: 7.8e-16 (6 outer iterations per step: the lag converges more slowly than the Cart3D elimination); Rabi
  flopping in a uniform m = 1 pulse at 640 atoms on all angles and radii: grid against the node step in the analytic field 4e-8
  (the first half node too: the axis rule is exact), against Schroedinger 1.6e-5 ... 2.8e-5 (the node step's own error at
  lambda/32). On a stretched radial grid the uniform m = 1 field itself is not kept exactly (the stretched cells and the wall
  disturb it; 3.5e-4 in p1 there), a property of the field solve, not of the atoms. Deck with ionization into species (m = 0, 1):
  Gauss law 8e-18, charge bookkeeping 1.4e-15.
- Moving window (all media, Cart3D, 2D, RZ; Simulation::shift_window_medium): every array of the media state moves with the
  fields (coefficients, J, P, the atoms' density matrix, n_a, the last Et, bP, the ionization accumulator; RZ: the lagged atom current,
  bP per mode and the angle arrays, whose last column comes from the right neighbour by one message), the entering column is rebuilt
  from the profiles (medium_build with i0: the same sub-cell sampling as at the start; atoms in the ground state, linear media at
  rest). The iRIP entry cell includes the media charge -div P at its Gauss point. Window against a long fixed box
  (tests/test_bound.cpp B10, Rb 1e-3 n_c up to p1 = 0.33 and a Lorentz layer entering at the right edge): 2D, 270 shifts:
  fields 2.8e-17, atoms 4.4e-16, Gauss law 1.4e-20; RZ, 216 shifts: interior fields 4.9e-13, atoms 1.9e-12, glass 1.7e-13 (the
  cells at the edges 1e-4: the iRIP z-line solve differs there because the window box ends where the long box continues),
  Gauss law 4.6e-18.
- Not yet: several bound blocks; TFSF, side tube, o5 (as all media); O6 with media in RZ.
- Residual floor of the solve (all media): the GMRES tolerance is relative to max(|G(0)|, 1e-4 |F|), |F| the norm of the field
  state. With Ex, Bx ~ 0 (an s-polarised wave along x) |G(0)| is round-off of the transport and the relative residual stalled at
  ~2e-9 > tol, i.e. medium_maxit iterations every step; now 0.5 iterations per solve on such a deck.
- Measured (tests/test_bound.cpp, CPU, 16 threads):
  - node step (Rabi flopping of a resonant line, a E0 = 0.133, against the Schroedinger equation without RWA, RK4): error of the
    excited population after a 3 pi / 2 pulse 2.0e-4, 5.0e-5, 1.3e-5 at lambda/16, 32, 64 (pi / 2: 1.0e-5 ... 6.5e-7; second order);
    with the coupling and polarization both scaled by 1/sqrt(cos x) (the first try) 2.3e-2, 5.7e-3, 1.4e-3. Energy exchange
    sum E (P^{n+1} - P^n) = Delta K_b to 8.8e-14.
  - grid (thin layer, lambda/32, the pulse's analytic vacuum field at the atom as the reference): 0.499932 / 0.999648 / 0.500216
    against 0.499935 / 0.999653 / 0.500166 for areas pi / 2, pi, 3 pi / 2.
  - weak field, Rb D1 + D2, 1e-4 n_c, 60 lambda, pulse at 0.92 omega0 (envelope 12 lambda): spectral phase against (n - 1) omega L of
    the Lorentz chi over 0.90 ... 0.94 omega0, rms 2.7e-3 (lambda/20), 6.8e-4 (lambda/40). D2 alone: bound 6.8e-4, the same line as
    a linear `lorentz` medium (Crank-Nicolson without pre-warping) 7.5e-2 / 1.75e-2: near a line the pre-warped atom is ~25 times
    more accurate. (A first version with a 6 lambda pulse at 0.95 omega0 had a third of its spectral amplitude at D1: the undamped
    lines ring and leak into the band through the finite window, a 7 % offset for both models.)
  - saturated regime (Rb 1e-3 n_c, a0 1e-3 at 800 nm, 2D): W + K_b constant to 1.7e-15 while up to 99.6 % of the atoms are
    excited (K_b up to 1.4e-2 of the pulse energy), 2.8 outer iterations per step (max 3).
  - Gauss law with particles, atoms (dipole along y) and a Lorentz layer, 3D: 4.2e-15 explicit, 4.3e-15 implicit mover (W + K to
    4.4e-16).
