# iRIP (box closure) in Cartesian 3D: the operator identity (check (a) of docs/specs/implicit_rip.md)

`identity.py`: a numpy copy of the RIP-CC step of `src/fields.cpp` (periodic box, with currents) in which
the predictor Ex*, Bx* is replaced by mu_t Ex = (Ex^n + X)/2, mu_t Bx = (Bx^n + Y)/2, X, Y the unknown
Ex^{n+1}, Bx^{n+1}. Grid 6 x 5 x 4, h = 0.4, 0.3, 0.25, c dt = h_x (D/h_perp = 1.6, explicit RIP-CC
would be unstable).

Result (2026-10-27):
- dF/dX = D^2/4 mu_x^2 Lap_perp (5-point transverse Laplacian at the Ex nodes) to 1.4e-15, dG/dY the
  same at the Bx positions to 1.5e-15; TM and TE decoupled (dF/dY, dG/dX at 1e-15).
- So (I - D^2/4 mu_x^2 Lap_perp) X = F(0, 0), the same for Y; with the solved X, Y the step is
  self-consistent, F(X, Y) = X to 3e-15.
- Vacuum, 300 steps: E^2 + B^2 conserved to 2.2e-14.

In a periodic box the 2D DFT in y, z diagonalises Lap_perp, eigenvalue -lambda,
lambda = (4/hy^2) sin^2(pi qy/ny) + (4/hz^2) sin^2(pi qz/nz); per mode the system along x is cyclic
tridiagonal with diagonal 1 + kappa^2/2 and off-diagonals kappa^2/4, kappa^2 = D^2 lambda/4.

## Non-uniform transverse grid: reflection (`stretch.py`, results in `stretch_results.txt`)
The box step on a 2D slice (one Fourier mode e^{i kx x}, no z dependence) with arbitrary y nodes: fine
cells h = 0.1 c dt (as near an RZ axis), stretched by a ratio per cell up to h_max. A wave packet runs
from the fine region into the stretch; absorbers at both ends record the reflected and transmitted
energy, and the same run on the uniform fine grid is subtracted. `python3 stretch.py` (about 20 min on
2 cores). Summary in docs/specs/implicit_rip.md: resolved waves (k_y h_max <~ 1.2) reflect 1e-7 ... 2e-5,
near the cutoff k_y h = 2 about half, beyond it all; independent of c dt / h_min = 1 ... 10.
