# Self-field of a relativistic beam at the start

Status: implemented (`src/beam_field.*`, `species.self_field`, `species.t_start`; decisions 24, 95, 98, 99)

## Physics and equations
A beam moving along x with velocity v = beta c (gamma = 1/sqrt(1 - beta^2)), rigid at the start, has
the steady fields of the Lorentz-boosted electrostatic problem (lab frame, code units):

    (dxx / gamma^2 + dyy + dzz) phi = -rho_b
    Ex = -dx phi / gamma^2,   Ey = -dy phi,   Ez = -dz phi,   Bz = beta Ey,   By = -beta Ez,   Bx = 0

so div E = rho_b. Without them the beam starts in zero fields: its field builds up from the current
and radiates a transient, and the initial Gauss residual is -rho_b.

## Discretisation
RIP-CC Gauss law at rho (i+1/2, j, k): dx Ex + mu_x dy Ey + mu_x dz Ez = rho. The potential phi sits
at the rho points.

- Ex(i) = -(phi(i) - phi(i-1)) / (gamma^2 h_x).
- Ey(i, j+1/2) = -(phibar(j+1) - phibar(j)) / h_y, with phibar(i) = (phi(i-1) + phi(i)) / 2 at the
  integer x of Ey. Ez is the same along z.
- Bz = beta Ey and By = -beta Ez are co-located with Ey and Ez (RIP-CC pairs). Bx = 0.

Then the Gauss law reads L phi = dxx phi / gamma^2 + mu_x^2 (dyy + dzz) phi = -rho. Solving it
exactly gives the Gauss law to round-off.

- **Solve**. y and z are periodic, so use a DFT in y and z. Per (k_y, k_z), with
  K = (4/h_y^2) sin^2(k_y h_y/2) + (4/h_z^2) sin^2(k_z h_z/2), the problem is tridiagonal along x:

      a (phi(i-1) + phi(i+1)) + b phi(i) = -rho(i),   a = 1/(gamma^2 h_x^2) - K/4,   b = -2/(gamma^2 h_x^2) - K/2

  - It is strictly diagonally dominant for K > 0. Use Thomas in complex arithmetic.
  - Dirichlet phi = 0 at i = -1 and i = N_x (just beyond both open x edges).
  - **k_perp = 0 apart**: the slice-mean charge (a charged sheet in the periodic y, z box) has
    phi ~ gamma^2 rhobar L^2, which cancels in E_perp only to round-off. Through phi the residual grew
    as gamma^2 (6e-9 at u = 2000). So it gives Ex alone, from the 1D Gauss law
    Ex0(i+1) - Ex0(i) = h_x rhobar(i). Its dynamics is dEx/dt = -Jx, so every constant is steady:
    - a moving beam gets Ex0 = 0 ahead of it (a sheet that came from -x; the moving window needs zero
      fields at its right edge) and the sheet field behind it;
    - a charge at rest gets the symmetric choice phi(-1) = phi(N_x) = 0.

    The field energy then grows as the beam moves (the region behind it grows). This is an artefact
    of the periodic transverse box with a net charge, not radiation.
  - **Clamped boost per mode (decision 95)**. For gamma h_x sqrt(K) > 2, a < 0 < |b|, and the
    homogeneous solutions are zigzags (-1)^i exp(-kappa i), kappa ~ 4 / (gamma h_x sqrt(K)) per cell;
    at large gamma (-1)^i (A + B i), no decay at all. With phi = 0 at both x ends they left a field
    tail ahead of the beam that fell only linearly to the right edge, where the first window shift cut
    it: a static Gauss residual of 1.8e-4 of max rho for a beam alone at u = 1e4 (bench/qed_pwfa 7e-5 ..
    1.2e-4 until the window had moved one box length). The continuum pancake there is thinner than
    h_x / 2. So per mode 1 / gamma^2 -> max(1 / gamma^2, h_x^2 K / 4), i.e. a >= 0:
    - at a = 0 the solve is local, phi = rho / K (the slice Poisson problem) and
      Ex = -h_x (rho(i) - rho(i-1)) / 4: no field ahead of or behind the beam;
    - Ex is formed per mode, Ex = -(G(i) - G(i-1)) / h_x with G = phi / gamma_eff^2 transformed back
      along with phi;
    - modes with gamma h_x sqrt(K) <= 2 are unchanged (beams with u <~ 10 unchanged);
    - the Gauss law stays exact. B3's near field became 8 times steadier (9.9e-3 -> 1.2e-3).
    - Explicit RIP-CC only (decision 99, Alexander 2026-11-11), Cart3D and RZ. The clamped field is the local steady
      state of the explicit step; iRIP and iRZ keep the unclamped field, their steady state (with the clamp the Fig. 4
      beam jumped 1 -> 11.7 nm with iRIP and 12.5 nm with iRZ; unclamped 1.00 nm). In a window from t = 0 the
      unclamped iRIP field has the zigzag tail up to the right edge (1.6e-4 of max|E| at u = 1e4), but the iRIP
      shifts keep the Gauss law (B5 2.6e-14, M5 1.9e-14 over 12 shifts). A tail-free implicit init is open.
- **Charge**. rho_b is the charge density of the species marked `self_field = true`. gamma and beta
  come from its mean momentum u0 (one solve per such species; the fields add). The other species are
  assumed neutral, as before; their residual stays in the reference residual.
- **Load time**. `t_start > 0` loads the species at the first step with t_n >= t_start (operator
  `delayed_load`, stage Inject), inside the moving window, with absolute coordinates.
  - The solve is linear, so the self-field adds to the fields present and the Gauss law of the
    other charges is kept. This is external injection of a witness beam behind a laser.
  - The window injects such a species into entering cells only after t_start, and those cells get
    no self-field.
- **MPI**. rho_b is summed into a global array (allreduce), and every rank solves the whole problem
  and keeps its block. Memory is 8 N_x N_y N_z bytes per rank, fine up to ~1e8 cells. A distributed
  solve can follow when needed.
- **FFT**: radix-2 for powers of two, a plain DFT otherwise (setup only). Host code.

## Periodic boxes, Yee, iRIP, RZ injection (decision 98)
For the emittance benchmark (`bench/emittance/`) the self-field also starts runs with the other solvers:
- **Periodic x** (any Cartesian solver): a 3D DFT. Per k, with K_a^2 = (4/h_a^2) sin^2(theta_a/2), W_a = exp(i theta_a):
  - RIP-CC, iRIP (phi at the rho point): L = K_x^2/gamma^2 + cos^2(theta_x/2) K_perp^2, Ex = -(1 - 1/W_x) phi/(gamma^2 h_x),
    Ey = -(W_y - 1)/h_y (1 + 1/W_x)/2 phi (phibar at integer x), Bz = beta Ey, By = -beta Ez: the same equations as the open
    x solve, cyclic along x.
  - Yee (phi at the node): L = K_x^2/gamma^2 + K_perp^2 (Yee's Gauss law), Ex = -(W_x - 1) phi/(gamma^2 h_x),
    Ey = -(W_y - 1)/h_y phi; Bz = beta Ey and By = -beta Ez at x_{i+1/2} by the factor exp(i theta_x/2) (B at integer time
    levels, the band-limited value of the moving field). This is the Gauss law of the scheme exactly, not its discrete steady
    state: Yee's waves are slower than the beam, which has no steady state on that grid.
  - phi = rho / L; the k_perp = 0 modes need no special treatment (E_perp is exactly 0 there). k = 0 is dropped: a net
    charge in a periodic box has no field without a neutralizing background.
- **The co-moving background** (`remove_mean_current`): the beam also carries a mean current, which would drive a uniform
  Ex = -<Jx> t and decelerate it (Yee: field energy doubled in 86 steps in the Fig. 4 setup). In a fully periodic Cartesian box
  with a self_field species the mean of Jx, Jy, Jz is removed every step: a uniform neutralizing background of charge
  -<rho> that moves with the mean current. div J is unchanged, so the Gauss law stays exact; its residual is the uniform
  -<rho> of the init, which the run's reference residual absorbs.
- **iRIP**: the RIP-CC solution. iRIP keeps the same discrete Gauss law (`docs/specs/implicit_rip.md`), so it holds to
  round-off; the near field is iRIP's steady state to O(h^2) as for RIP-CC (B8).
- **RZ with t_start**: the RZ solve (`rz_add_beam_field`) adds to the fields present and uses the global z index, so a
  witness can be injected into a moving window as in Cart3D (M6).
- **rigid_until** (`species.rigid_until`, any species, Cart3D and RZ): before that time the momenta are restored after
  every push, so the species moves ballistically and deposits its current while the fields relax. A diagnostic of the
  init transient (bench/emittance: with RIP-CC the co-moving part of the mismatch does not leave, so it is no cure).

## Steady init for RIP-CC (decision 99, default)
The Gauss-law field above is not the discrete steady state of the RIP-CC step: they differ by O(h^2) (B3: near-field change
1e-2 at h_x = 0.1). RIP-CC moves forward waves exactly one cell per step, so the difference travels with the beam and
keeps acting on it. For a 1 nm beam this was a jump to 12 nm (bench/emittance). `numerics.beam_init = steady` (the
default; `gauss` keeps the field above) therefore starts a RIP-CC run with the field the step leaves unchanged:
- The beam is rigid and moves beta cells per step (c dt = h_x). Its charge is rho^n = rho^ lambda^n per Fourier mode
  F(i, j, k) = F^ exp(i (tx i + ty j + tz k)) of the array index, with lambda = exp(-i beta tx) and the signed tx.
  From the discrete continuity equation J^{n+1/2} = lambda^n Jx^, with Jx^ = (1 - lambda) rho^ / (Sx - 1),
  S_a = exp(i t_a). Jx is at i and rho at i + 1/2; Jy = Jz = 0.
- One vacuum step of `Fields::step_core` is an affine map of a mode, F^ -> M F^ + c Jx^. `ModeStep` applies the
  kernel's formulas term by term, each index shift a factor S_a^(+-1). Probing it with the 6 unit vectors and the
  source gives M and c, and the steady field solves (M - lambda) F^ = -c Jx^ (6 x 6, pivoting).
- Closed forms where M - lambda is singular:
  - k_perp = 0: only the sheet field Ex^ = h_x rho^ / (Sx - 1). The TEM modes there have the eigenvalue 1/Sx, equal to
    lambda at beta = 1, and get no source.
  - tx = 0: the Gauss-law field. It is x independent, and the step leaves static E_perp with B = beta x E unchanged,
    because the corrector and the Gauss law use the same stencils.
  - k = 0 is dropped (as above).
- The Gauss law needs no extra condition: the step conserves G = Gauss residual, so G^{n+1} = G^n, while a steady state
  has G^{n+1} = lambda G^n. Hence G = 0 for lambda != 1.
- Periodic x: the 3D DFT of the box. Open x (window): the box is padded with zeros to a power of two >= 2 N_x. The
  field is contracted to ~ r / gamma along x, so the images do not overlap. The k_perp = 0 sheet field is the open box's
  Ex0, zero ahead of the beam. Memory: 16 bytes x padded cells per complex array, two arrays plus the 6-component
  modes. Every rank solves the global problem.
- Used for RIP-CC with beta > 0 and no CPML in y, z (its step differs). Yee, iRIP, RZ, CPML runs and beams at rest keep
  the Gauss-law field. RZ (its own step per radial operator) is the follow-up.
- With decision 95 (clamped boost per mode in the open-x Gauss-law solve): every mode with gamma h_x k_perp > 2 is the
  local slice solution, and that is stationary under the RIP-CC step as well. At large gamma the two inits then nearly
  coincide in a window (B7: both stationary to 6e-14 at u = 1e6). They differ in periodic-x boxes (the periodic
  Gauss-law solve has no clamp) and at moderate gamma (B9, window: 1.0e-2 of max|F| at u = 5, 5.9e-3 at 20, 8.8e-4 at
  60, 3.2e-12 at 1e6). The steady init stays the default (Alexander, 2026-11-11). The steady field has a small discrete tail ahead of the beam
  (B5: 8.5e-11 of max|E| at u = 1e4), which the first window shifts cut: max|dGauss| 2.7e-12 of max rho over 12 shifts
  (Gauss-law init: exactly zero ahead, 2.6e-14). B4 (beam loaded at t_start into a wake): 8.4e-12 (Gauss-law init
  8e-16).
- Test B7 (tests/test_beam.cpp): a rigid beam (u = 1e6), one real RIP-CC step:
  - the fields equal the initial ones shifted by one cell to 6e-14 of max|F|, periodic and in the moving window (near
    the beam);
  - the Gauss-law init with periodic x changes by 1.1e-3; in the window it is stationary too (decision 95, above);
  - Gauss law at round-off.

## Deck interface
```
species beam {
  charge = -1   mass = 1   density = 1e-3   ux = 400       # gamma ~ 400
  density_function = gauss(x - 10 um, 1 um) * gauss(sqrt(y^2 + z^2) ..., 0.5 um)
  self_field = true                                         # steady fields of the moving beam
  t_start    = 100 fs                                       # optional: load later (in the window)
  rigid_until = 100 fs                                      # optional: momenta frozen until then (decision 98)
}
numerics {
  beam_init = steady          # steady (default; RIP-CC: the discrete steady state, decision 99) | gauss (the Gauss-law field)
}
```

## Tests
- Fast (tests/test_beam.cpp):
  - B1: the Gauss law after the init is at round-off (< 1e-12 of max|rho|; measured 3e-15 to 7e-15)
    for a non-neutral beam alone, at u = 20, 2000 and 0 (Poisson). Without self_field the residual
    is -rho.
  - B2: in a long round Gaussian beam (gamma 10), E_r at r = 1.1 .. 3.1 sigma equals the line-charge
    field minus the uniform compensation of the periodic box to 0.7 %, and Ex does not vary across
    the slice.
  - B3: the beam in vacuum, gamma 50, 60 steps. Its near field moves with it and changes by
    O(h_x^2): 1.0e-2 at h_x = 0.1 and 1.6e-3 at 0.05 over the same distance, the offset between the
    continuous boosted solution and the discrete steady state. The Gauss law stays at round-off.
    Without self_field the change is 0.3.
  - B4: a beam loaded at t_start into a moving window with a laser wake. None before t_start, a
    constant count after, and max|dGauss| 8e-16 (1.1e-12 before decision 95; 8.4e-12 with the
    steady init of decision 99, the default).
  - B5: a beam alone at u = 1e4, 12 cells from the right edge, in a window moving from t = 0: the
    fields ahead of the beam exactly zero, max|dGauss| 2.6e-14 of max rho over 12 shifts (1.8e-4 before
    decision 95), with beam_init = gauss. With the steady init: 8.5e-11 ahead, 2.7e-12 over 12 shifts. iRIP
    (unclamped): 1.6e-4 ahead, 2.6e-14 over 12 shifts. RZ: M5 in tests/test_rz_metal.cpp, RIP-CC 2.2e-14, iRZ 1.9e-14.
  - B9 (decision 99): Gauss-law vs steady init in a window at u = 5, 20, 60 (differ by more than 1e-4 of max|F|) and
    u = 1e6 (equal to 1e-10).
  - B6 (decision 98): the emittance regression, Fig. 4 of the 2020 paper scaled down (bench/emittance/README.md):
    growth of eps_n,y over 15 um RIP-CC -3.0e-6 (steady init), iRIP -2.0e-5, Yee +19.8.
  - B7 (decision 99): the steady init is stationary under the RIP-CC step (above).
  - B8 (decision 98): the B3 beam with RIP-CC periodic, iRIP open and periodic, Yee periodic (dt = h_x/2): Gauss law
    at the init (periodic: up to the uniform -<rho>) and while the beam moves 60 cells, near field steady (1.4e-3,
    iRIP open x 9.7e-3, Yee 1.9e-2).
  - tests/test_rz_metal.cpp M6 (decision 98): a witness loaded at t_start into an RZ moving window with plasma, RIP-CC
    and iRIP: max|dGauss| 1.4e-13 of max rho.
- MPI (`tests/mpi.sh`, fourth case): `examples/beam_vacuum.deck`, 1 vs 8 ranks identical. Regression
  case `beam`.

## References
- Standard Lorentz-boosted Poisson initialisation (e.g. WarpX / FBPIC "space-charge field of a
  relativistic beam"); here in the discrete RIP-CC form, so the Gauss law holds exactly.
