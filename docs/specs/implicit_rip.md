# Implicit RIP (`irip`): space-time centred ("box") closure

Status: physics settled (Alexander + cloud session, 2026-10-26). Implemented for Cart3D, periodic boxes
(`src/irip.cpp`, `numerics.solver = irip`, decision 38, 2026-10-27), on one rank or with ranks along x
(partitioned tridiagonal solve, decision 39), x periodic or open with laser and moving window (decision 40), y and z periodic or with CPML layers
(decision 41); RZ follows. Phase 2.
Prototype and all numbers below: `research/irip_rz/` (`README.md`, `box_rz.py`, `box_plasma.py`,
`irip_rz.py`). Decision 17 in `docs/decisions.md`.

## Why an implicit solver
Explicit RIP-CC needs h_x^2 (1/h_y^2 + 1/h_z^2) < 1, i.e. h_perp >= c dt. Fine transverse grids
(RZ near the axis, many azimuthal modes, h_perp < c dt in 3D) need an implicit transverse coupling.
c dt = h_x stays, so the transport along x is still an exact one-cell shift.

## Do NOT implement the characteristic-trapezoid iRIP
The earlier iRIP idea (right movers use (F(i+1, n+1) + F(i, n))/2, left movers (F(i, n+1) + F(i+1, n))/2,
F = Ex, Bx) is neutrally stable per mode but **not energy conserving**: in the energy norm it amplifies
grid-scale transverse structure transiently by ~min(kappa, 2 kappa sin(kx D/2)), kappa = k_perp D/2
(measured: ~11 for m = 0, D/dr = 10; ~420 for m = 50, D/dr = 10; the same in Cartesian geometry). It also
conserves only a modified Gauss law. Truncated iterations towards a symmetric variant are unstable.

## The scheme (box / Crank-Nicolson closure)
Everything as RIP-CC (transport of T+-, corrector with mu_t of the transverse fields, currents at
n + 1/2, `docs/specs/architecture.md` and README "Field solver"), except: the longitudinal fields in the
transverse sources of **both** movers are the space-time cell average at (i + 1/2, n + 1/2)
  Ex_bar = mu_x mu_t Ex = [Ex(i,n) + Ex(i+1,n) + Ex(i,n+1) + Ex(i+1,n+1)] / 4,   same for Bx_bar,
replacing the predictor Ex*, Bx*. This makes the step implicit in Ex^{n+1}, Bx^{n+1} only.

Step (linear, exact):
1. **Explicit part.** Run the RIP-CC transport + corrector with Ex^{n+1} = Bx^{n+1} = 0 inside Ex_bar,
   Bx_bar (i.e. Ex_bar = mu_x Ex^n / 2), currents included. Result: R_E, R_B (the would-be Ex, Bx)
   and provisional transverse fields.
2. **Implicit solve**, TM and TE decoupled:
     (I - D^2/4 mu_x^2 L_TM) Ex^{n+1} = R_E,   L_TM = Psi_x o (Phi_y, Phi_z)   (discrete transverse Laplacian on Ex)
     (I - D^2/4 mu_x^2 L_TE) Bx^{n+1} = R_B,   L_TE = Phi_x o (Psi_y, Psi_z)
   with mu_x^2 = [1, 2, 1]/4 along x (centred on i). Verified to 1e-15 against the full step for
   m = 0, 1, 5 (RZ) on the dynamic DOFs.
3. **Explicit correction** of the transverse fields: add the transport of D times the sources built from
   mu_x Ex^{n+1}/2, mu_x Bx^{n+1}/2 (the same kernel as step 1 with only these sources), and the
   corresponding corrector term. Local, no further solve.

## Implementation so far (Cart3D, periodic; `src/irip.cpp`)
- Cartesian identity checked first: `research/irip_3d/identity.py` (numpy copy of the C++ step):
  dF/dX = D^2/4 mu_x^2 Lap_perp to 1.4e-15, TM/TE decoupled, energy (plain E^2 + B^2) conserved to 2e-14.
- Step 3 is done as a second full pass of the RIP-CC transport and corrector with
  Ex_bar = mu_x (Ex^n + X)/2, Bx_bar likewise (the same kernel as pass 1; simpler than the incremental
  correction and equal to it). The corrector's Ex, Bx are kept, not X, Y: they equal X, Y to the solver's
  round-off and keep the Gauss law exact.
- Solve: 2D DFT in y, z of every x slice (radix 2 for powers of two, a direct sum otherwise; the exact
  eigenbasis of the 5-point Lap_perp), per mode (qy, qz) the cyclic tridiagonal system along x by Thomas +
  Sherman-Morrison, then back. All `par_for` kernels (GPU build compiles; one thread per line or mode).
- Particles may move more than one transverse cell per step (h_perp < c dt). The Esirkepov deposit
  already splits such moves (decision 28); the guard width is now 2 + ceil(dt / h_min) (3 for RIP-CC and
  Yee, unchanged), and the tiled and box deposits are not used (they need dt <= h).
- Time step c dt = h_x; checked: omega_p dt < 2 only. Rejected so far: self_field (it solves RIP-CC's
  moving equilibrium), nx < 3, ranks along y or z.
- CPML layers in y, z (decision 41; `src/irip_pml.cpp`, prototype `research/irip_3d/pml_identity.py`):
  - The CPML derivative g + psi_new, psi_new = b psi + a g with a = b - 1, is linear in the current
    values with the factor b, fixed per location; the memory b psi_old goes into the right-hand side.
    So the solve operator stays separable: L_E = (Cn Dm Ch Dp)_y + (...)_z on Ex, L_B = (Ch Dp Cn Dm)_y +
    (...)_z on Bx, C = b at the derivative's location, walls (zero) behind the layers. Checked against
    the full step to 1e-14.
  - Each 1D operator C T (T symmetric tridiagonal) has the real eigenbasis V = C^(1/2) Q from
    S = C^(1/2) T C^(1/2) = Q Lambda Q^T (computed once, QL with implicit shifts). A pml axis uses V
    (dense transforms of the lines) instead of the DFT; the system along x per mode and the partition
    method are unchanged, with the eigenvalues from tables.
  - b is floored at 1e-4 for iRIP (sigma dt <= ln 1e4): with c dt >> h_perp, b reaches 1e-20 deep in a
    layer, C^(+-1/2) spans 1e20 and the solve degrades to 1e-6. With the floor: 2e-13.
  - psi is only read in pass 1 and committed in pass 2.
  - The energy E^2 + B^2 is bounded but not monotone with CPML (psi stores energy): random grid-scale
    fields gain up to ~20 % transiently (also at D/h_perp = 0.5), then decay.
  - Moving window: the entering cell's Ex, Bx are set from the constraints only outside the layers
    (where the Gauss law holds); inside them it stays 0 as with RIP-CC. Setting it there from the plain
    constraints fed back from shift to shift and grew without bound.
  - Cost: the dense transforms are O(n^2) per line (the DFT O(n log n)); on the GPU they are GEMMs
    (cuBLAS later), on the CPU a naive loop.
- Open x, laser, window (decision 40): an open edge has zero guards, so the system along x is truncated
  with zeros outside (Thomas without Sherman-Morrison, also in the partition method). The laser is
  RIP-CC's inbound T in the x guards (BoundaryHooks), set once before pass 1 and used by both passes.
  Moving window: the implicit solve is not causal along x. The last cells carry a small precursor of
  Ex, Bx, and the entering cell (the old zero guard) would break the Gauss law and div B at the point it
  closes (3e-5 in the window case of tests/mpi.sh). So at each shift the entering cell's Ex, Bx are set
  from the two constraints at that point; its transverse fields stay 0 and the entering plasma is
  neutral, the same condition as RIP-CC's exact entry. For RIP-CC this rule would give 0.
- MPI (decision 39): ranks along x only, so every rank holds whole y, z planes and the DFT stays local.
  Per mode the cyclic system along x is split with Sherman-Morrison into A' (tridiagonal, first and
  last diagonal modified) plus a rank-one term. A' is solved by the partition method: on rank p,
  x = y + R_{p-1} u + L_{p+1} v, with local Thomas solves T y = r, T u = -s e_0, T v = -s e_{n-1}, and the
  neighbours' edge values R_{p-1}, L_{p+1}. These solve a block-tridiagonal system of 2 x 2 blocks,
  unknowns (R_{q-1}, L_q) for the P - 1 interfaces. One allgather of 10 numbers per mode and rank
  (u, v, y at both edges, for Re r, Im r and the Sherman-Morrison vector); every rank then solves the
  small system of every mode. Exact, no overlap or truncation (the inverse decays slowly for
  kappa >> 1, so truncated overlaps would not be).

## Non-uniform transverse grids: reflection (prototype, `research/irip_3d/stretch.py`)
Cells much smaller than c dt near the axis, stretched outward (as in RZ), must not reflect radiation
back (the CLAUDE.md rule of GEM-PIC-RZ). The prototype is the box step on a 2D slice with one Fourier mode
along x and an arbitrary node grid in y (spacings of the nodes and of the dual cells in the
differences). A wave packet runs from the fine region (h = 0.1 c dt) into the stretch. Absorbers at both
ends measure the reflected and transmitted energy; the same run on the uniform fine grid removes the
backward part of the initial packet. Results (`research/irip_3d/stretch_results.txt`):
- Energy is conserved on the stretched grid to 2e-13 (h from 0.1 to 2 c dt): stable.
- Resolved waves, k_y h_max <~ 1.2, stretch 5 % per cell: reflected 1e-7 ... 2e-5 (measurement floor
  ~1e-7). An abrupt stretch (50 % per cell) at k_y h_max = 0.22: 2e-6 ... 2e-5.
- Near the grid cutoff k_y h_max = 2 (1.97): 45 ... 50 % reflected, also with the slow stretch; beyond it
  (2.96, 3.94): all of it. This is the cutoff and the numerical index of the stretched cells.
- The time step does not matter: at a fixed physical grid and wavelength, c dt / h_min = 10, 5, 2, 1
  gives the same reflection (12 h_max: 2e-6; 6 h_max: 1e-5; 3 h_max, near the cutoff: 45 ... 50 %). The
  implicit step adds no reflection; the rule is the spatial one: k_perp h <~ 1 wherever radiation must
  leave, and a slow stretch.
- Near the cutoff the waves are slow in y (vanishing group velocity), so the reflection comes late.

## Solving step 2
-L is non-negative (RZ: real spectrum for every m; similar to symmetric with the area weights). Per
transverse eigenvalue lambda of -L the system along x is tridiagonal with diag 1 + kappa^2/2,
off-diagonals kappa^2/4, kappa^2 = D^2 lambda / 4: strictly diagonally dominant, SPD.
- **RZ**: L_m is a fixed Nr x Nr matrix per mode (also with a stretched radial grid or a radial
  absorber): diagonalise once at setup; per step two dense transforms per slice and mode (GEMM,
  O(Nr^2), like FBPIC's Hankel transforms) + batched Thomas along z.
- **Cart3D**, periodic or conducting y, z: FFT / DST in y, z diagonalise L exactly; then batched Thomas
  along x. With PML or stretched y, z grids: preconditioned CG on the SPD operator.
- **MPI along x**: the decay of the inverse is slow for large kappa (|z| ~ 1 - 2/kappa), so overlapping
  truncation is not exact: use a partitioned (SPIKE-type) tridiagonal solve across ranks.

## Properties (verified in the prototype)
- Energy conserved exactly: sup_n ||M^n|| = 1.000000 in the area-weighted norm (RZ, m = 0, 1, 3, 20, 50,
  D/dr up to 10). Unconditionally stable.
- The **standard** RIP-CC Gauss law and div B are conserved to round-off (1e-16 ... 4e-13): any Ex_bar,
  Bx_bar shared by both movers keep them. Hence the existing Esirkepov deposition (`stagger.hpp`, RipCC
  descriptor) stays exactly charge conserving; `irip` uses the RipCC descriptor unchanged.
- Dispersion: tan^2(w dt/2) = tan^2(kx D/2) + D^2 lambda/4. Zero dispersion for kx-only waves
  (lambda = 0). Phase error second order in k_perp D (RZ guided modes, lambda_0 = 24 D, R = 5 lambda_0:
  m = 0: -6.6e-5, m = 5: -8.5e-4, m = 20: -5.9e-3, m = 50: -2.2e-2).
- Cold plasma: stable for omega_p dt < 2, independent of D/h_perp (tested D/h = 8).
- RZ axis: m = 0: Ephi = Br = 0 on axis, Psi_x(0) = 4 Bphi_{1/2}/dr; m >= 1: Ez = 0 on axis;
  Ephi, Br on the axis are never dynamic DOFs (for m = 1 they are only driven; letting them feed back
  breaks energy conservation: gain 1.58 seen before this rule).

## Tests (fast unless noted; 1-5 in `tests/test_irip.cpp`, results 2026-10-27)
Done: 1 (error 0, h_perp = h_x/5); 2 (periodic 3D, both polarisations, w dt to 6e-13, residual of the
mode recurrence 6e-13, D/h_perp = 4); 3 (500 steps, D/h_perp = 10: |W/W0 - 1| 3e-12, div B change 3e-16);
4 (h_x = 0.5, h_perp = 0.2, 0.25, |u_th| = 0.3, moves up to 2.5 cells: Gauss law <= 1e-14, div B 3e-16);
5 (omega_p dt = 1.9, h_perp = h_x/4: cos(w dt) = 1 - omega_p^2 dt^2/2 to 1 %, the shape factor at
k = 2 pi/L). 6 (tests/mpi.sh: the streaming plasma with h_perp = h_x/4, transversely hot, 1 vs 2, 3 and 8 ranks along
x: energies to 7e-12 ... 1.2e-11, the summation order of the deposition as in the other MPI cases;
Gauss law 3e-16). Open x, laser, window (2026-10-27): a wave along x leaves exactly (W = 0 after);
random fields at dx/h_perp = 10 in an open box never gain energy (gain per step exactly 0); the focused
laser of tests/test_laser.cpp L1 with h_y = 3 dx/8 has the paraxial widths to 0.5 % and fluence gain to
1.2 %; tests/mpi.sh: open x and the LWFA window with laser at h_perp = 0.8 dx, 1 vs 8 ranks, Gauss law
and div B at 1e-14. CPML (2026-10-28): reflection of a packet into a 16-cell y layer with h_y = dx/2,
2.6e-6 at normal and 1.1e-6 at 45 deg incidence (RIP-CC's layer: 5e-6, 2e-5); on grids where both run,
iRIP's layer reflects exactly as RIP-CC's (h_y = 2 dx: 6.3e-6 and 1.3e-4 for both); random fields with
y and z layers at dx/h_perp = 4: bounded (max 1.04 W0), absorbed; hot plasma through the layers: Gauss
law and div B 2e-15 outside them; tests/mpi.sh: the streaming plasma with layers and the LWFA window
with laser and layers, 1 vs 8 ranks. Note: a reflection test must keep the packet clear of the layer at
t = 0 (an initial overlap gave a false 1e-3). Open: 7 and the RZ parts.

1. Vacuum wave along x (k_perp = 0): exact one-cell shift per step (error 0), as RIP-CC.
2. Single transverse mode (periodic 3D, and RZ m = 0, 1, 3): measured frequency = tan^2 law to 1e-10.
3. Random vacuum fields, D/h_perp = 10, 2000 steps: field energy constant to the solver tolerance
   (machine precision with the direct solvers), Gauss law and div B unchanged to round-off.
4. Hot drifting plasma and non-neutral slab with irip, shapes 1 and 2: Gauss law to round-off.
5. Cold plasma omega_p dt = 1.9: bounded; Langmuir frequency vs the per-mode matrix of `research/irip_rz/box_plasma.py`.
6. MPI: 1 vs 8 ranks (partitioned tridiagonal) agree to 1e-11.
7. Python prototype numbers (`research/irip_rz/`) reproduced for one RZ case (amplification matrix
   of one slice and mode from the C++ kernels).

## Comparison with FBPIC (PSATD) and Gate B benchmark plan
FBPIC facts (docs, 2026-10): z-only decomposition (efficient to ~4-16 domains), n_order = -1 (exact) on one
device but finite (32 recommended) with MPI; curl-free current correction (default, not local) or
cross-deposition (local, polarisation not strictly kept); one-pass binomial filter of J, rho by default;
radial PML "significantly more costly"; Galilean PSATD via `v_comoving`.
Expected trade-off: PSATD is exact for transverse dispersion (box iRIP: second order in k_perp D, e.g. laser
group velocity error 1e-6 at 32 cells/lambda0, w0 = 20 um); iRIP has exact local charge conservation, exact
z decomposition, arbitrary radial grids and absorbers, and ~4x fewer dense transforms per mode (estimate).

Done (prototype, `research/irip_rz/README.md` section 6): NCI of a streaming plasma, 2D3V. On FBPIC-like grids
(h_perp > c dt) box iRIP is quiet without a filter, PSATD needs its filter; with h_perp = c dt/4 box iRIP needs
the one-pass binomial filter (then quiet, 1e-9 of the beam energy), PSATD with that filter is fully unstable.
=> irip ships with an optional binomial current filter (same stencil on J and rho, keeps the Gauss law).

Benchmarks for Gate B (automatic, stored curves; FBPIC runs on the GPU workstation / JUWELS by the Developer agent):
- B1 NCI: the 2D3V test in C++ (Cart3D) and RZ (m = 0, 1), gamma 10 and 50, h_perp/c dt = 3, 1, 1/4, with and
  without filter; plus FBPIC (standard and Galilean) on the RZ cases.
- B2 LWFA bubble in RZ (Nm = 2) against FBPIC on the same deck: a0(z), on-axis Ez, injected charge, spectrum.
- B3 relativistic beam in vacuum (gamma 100-1000): self-field balance and emittance growth vs FBPIC.
- B4 high-k_perp radiation (betatron / ICL-like emitter): angular spectrum vs FBPIC (PSATD exact reference).
- B5 MPI: Gauss-law error and dispersion vs number of ranks, iRIP (SPIKE) vs FBPIC n_order = 32.
- B6 performance on one GPU: field time per mode and step, time per particle and step (FP64 on A100/H100).

## References
- A. Pukhov, J. Comput. Phys. 418, 109622 (2020) - RIP.
- R. Lehe et al., Comput. Phys. Commun. 203, 66 (2016) - azimuthal modes, Hankel transforms (FBPIC).
- E. Polizzi, A. Sameh, Parallel Comput. 32, 177 (2006) - SPIKE.
- M. Kirchen et al., Phys. Plasmas 23, 100704 (2016); R. Lehe et al., Phys. Rev. E 94, 053305 (2016) - Galilean PSATD.
- S. Jalas et al., Phys. Plasmas 24, 033115 (2017) - finite-order PSATD with domain decomposition (FBPIC).
