# iRIP in quasi-3D RZ: stability and phase error per azimuthal mode (prototype, 2026-10-26)

`irip_rz.py`: one iRIP field step for a single azimuthal mode m, Fourier mode e^{i kx x} along x,
VLPL staggering mapped y -> r, z -> theta (Ex, Etheta, Br on nodes r_j = j dr; Er, Btheta, Bx on
r_{j+1/2}), axis rules for m = 0, 1, >= 2, conducting wall at R. The implicit Ex^{n+1}, Bx^{n+1}
(characteristic trapezoid in the transverse sources) are solved exactly by linear algebra, and the
one-step amplification matrix is built column by column.

## Results (Nr = 48, kx D = 0 ... 3.1)

1. **No stability limit in m.** All eigenvalues of the amplification matrix have |z| = 1 to 1e-14 ...
   3e-11 for m = 0, 1, 2, 3, 5, 10, 20, 30, 50 and D/dr = 0.1, 0.5, 2, 10. The discrete transverse
   operators -L_m (TM: Psi_x o Phi, TE: Phi_x o Psi) have real, non-negative spectra for every m.
2. **The dispersion is exactly the Cartesian iRIP law** with the discrete radial eigenvalue:
   sin^2(w tau/2) = [sin^2(kx D/2) + D^2 lambda/4] / (1 + D^2 lambda/4), lambda = eig(-L_m);
   matrix and formula agree to 1e-14 for all m (for m = 0 after removing the lambda = 0 TE entry,
   a uniform Bx that is not a wave but a static mode).
3. **Radial (near-axis) discretization is second order and uniform in m**: the relative error of
   k_perp of the lowest TM modes is -1.2e-4 (m = 0) ... -7.7e-4 (m = 50) at Nr = 48 and drops by
   exactly 4 when dr is halved. No special degradation near the axis for large m.
4. **The phase error is set by k_perp * D, not by m or dr** (implicit time discretization):
   laser carrier lambda = 24 D, guide radius R = 5 lambda, lowest guided mode of each m:
   m = 0: -1.7e-5, m = 1: -4.4e-5, m = 5: -2.4e-4, m = 20: -2.6e-3, m = 50: -1.6e-2,
   identical for D/dr = 0.25, 1, 4. Since k_perp >= j_{m,1}/R ~ m/R, high-m modes are slower only
   because their transverse wavenumber is larger; the accuracy criterion is k_perp D <~ 0.2 for
   dw/w <~ 3e-3.
5. **Transient amplification of iRIP (open point, now resolved).** The matrix powers are bounded
   (no secular growth), but in the area-weighted energy norm iRIP amplifies some field
   combinations transiently: ~11 (m = 0, D/dr = 10), ~39 (m = 20, D/dr = 2), ~420 (m = 50, D/dr = 10).
   This is not an RZ effect: the Cartesian iRIP shows the same, sup_n ||M^n|| ~ min(kappa, 2 kappa
   sin(kx D/2)) for kappa = k_perp D/2 >> 1, i.e. grid-scale transverse structure with
   kx != 0 (in RZ: high-m fields within r ~ m D of the axis, e.g. a near-axis Bx spike). The cause is
   the asymmetric characteristic trapezoid (right movers use (Ex(i+1,n+1) + Ex(i,n))/2, left movers
   (Ex(i,n+1) + Ex(i+1,n))/2): it conserves no energy that is uniformly equivalent to |E|^2 + |B|^2,
   and it conserves only a modified Gauss law. Truncated iterations towards a symmetric variant are
   unstable (Cartesian per-mode analysis).

## Fix: the space-time centred ("box") variant, `box_rz.py`

Both movers use the cell average mu_x mu_t of Ex, Bx at (i+1/2, n+1/2); everything else as RIP-CC
(corrector with mu_t B_perp, mu_t E_perp). Verified (`box_rz.py`, `box_plasma.py`, and per-mode checks of the implicit operator and the dispersion law):

- **Exact energy conservation:** sup_n ||M^n|| = 1.000000 in the area-weighted norm for m = 0, 1, 3,
  20, 50 and D/dr up to 10 (the m = 1 on-axis Etheta, Br are not dynamic DOFs: they were only driven,
  never fed back, which broke the symmetry).
- **Standard Gauss law and div B conserved to round-off** (1e-16 ... 4e-13), the same discrete laws as
  RIP-CC (the RIP-CC theorem: any Ex*, Bx* in the transverse sources conserve Gauss/div B exactly, as
  long as both movers use the same ones). The existing Esirkepov deposition (RZ: its per-mode analogue)
  stays exactly charge conserving.
- **Dispersion** tan^2(w tau/2) = tan^2(kx D/2) + D^2 lambda/4 (matrix vs formula to 1e-11): exact
  along x, unconditionally stable. Phase error of guided modes (lambda = 24 D, R = 5 lambda, D = dr):
  m = 0: -6.6e-5, m = 1: -1.7e-4, m = 5: -8.5e-4, m = 20: -5.9e-3, m = 50: -2.2e-2 (iRIP: -1.7e-5 ...
  -1.6e-2), second order.
- **Cold plasma:** stable for wp dt < 2, independent of D/h_perp (tested D/h = 8; |z| = 1.5 at wp dt = 2.05).
- **Implicit system:** exactly (I - D^2/4 mu_x^2 L_m) Ex^{n+1} = R_E and the same with the TE operator for
  Bx^{n+1}, decoupled (verified to 1e-16 on the dynamic DOFs), mu_x^2 = [1, 2, 1]/4 along x. It couples
  neighbouring x-slices (iRIP was per slice). Solution: L_m is a fixed radial matrix (also with a
  stretched grid or a radial absorber), so diagonalise it once per mode; then each radial eigenvalue
  gives an independent, strictly diagonally dominant tridiagonal system along x (diag 1 + kappa^2/2,
  off-diagonals kappa^2/4): batched Thomas on the GPU, partitioned tridiagonal across MPI ranks in x.
  Cost per step O(Nr^2) per slice and mode for the two transforms (dense GEMMs), the same order as the
  Hankel transforms of FBPIC; particles still dominate.

Conclusion: use the box variant for RZ (and for Cartesian runs with h_perp < c dt). It is
unconditionally stable for any number of modes, conserves energy and the standard Gauss law exactly,
and keeps zero dispersion along x; the number of modes is limited only by cost (particles, ~M^2) and
by accuracy in k_perp D.

## 6. Numerical Cherenkov: box iRIP vs FBPIC-like PSATD (`nci2d.py`, `nci_g50.py`, `nci_plot.py`)

2D3V periodic PIC in Python, neutral e-p plasma streaming along x (test of JCP 2020, Sec. 7.1), c dt = dx =
0.63 c/wp, 2 x 2 ppc, linear shapes, Boris, 444 steps (t = 280/wp). RIP solvers with Esirkepov on the charge
grid (Gauss drift 1e-13 for every run); PSATD as FBPIC: co-located nodes, J at x^{n+1/2}, curl-free current
correction, exact k (n_order = -1), optionally FBPIC's default one-pass binomial filter of J and rho in k space.
Solvers checked first: box step consistent to 2e-15, PSATD equal to the exact matrix exponential to 2e-17.
Field energy at t = 280/wp in units of the electron kinetic energy:

| case | RIP-CC | box iRIP | box + filter | PSATD | PSATD + filter |
|---|---|---|---|---|---|
| gamma 10, h_perp = 3 c dt (FBPIC-like grid) | 5.5e-9 | 1.2e-8 | 5.0e-10 | 3.7e-2 (NCI) | 2.9e-9 |
| gamma 10, h_perp = c dt / 4 (fine transverse) | not usable (h_perp < c dt) | 1.4e-3 (NCI) | 1.1e-9 | 6.1 (NCI) | 1.0 (NCI) |
| gamma 50, h_perp = 3 c dt | - | 8.3e-10 | 1.2e-10 | 4.7e-9 (growing) | 1.9e-10 |

- On the usual FBPIC grid (h_perp > c dt) RIP-CC and box iRIP need no filter; PSATD needs its default filter.
- With fine transverse cells (the regime iRIP is for) the unfiltered box iRIP has a slow aliased NCI, removed by
  the same one-pass binomial filter (charge conservation kept: the filter commutes with the discrete
  divergence).
- Not covered: Galilean PSATD (FBPIC's `v_comoving`), which removes NCI of a single uniform drift by
  construction; finite-order PSATD (n_order = 32, MPI); cubic shapes; 3D/RZ. Single runs, 4 ppc.
