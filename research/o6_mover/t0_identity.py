"""T0: the identities the mover rests on.
(a) box step with a given current: W^{n+1} - W^n = -dt A sum J . Et  (Et = work field returned by field_solve);
(b) hot plasma, dy < dx, moves over several transverse cells, relativistic: implicit mover W + K conserved to the solve
    tolerance, Gauss law unchanged at round-off; explicit leapfrog: Gauss law too."""
import numpy as np
from pic2d import Sim, lattice

rng = np.random.default_rng(1)
nx, ny, D, dy = 24, 20, 1.0, 0.4
S = Sim(nx, ny, D, dy)
for trial in range(3):
    S.f = [rng.normal(size=(nx, ny)) for _ in range(6)]
    J = [rng.normal(size=(nx, ny)) for _ in range(3)]
    W0 = S.W()
    f1, Et = S.field_solve(J)
    S.f = f1
    lhs = S.W() - W0
    rhs = -D * S.A * sum((J[c] * Et[c]).sum() for c in range(3))
    print(f"(a) dW = {lhs:+.12e}  -dt A J.Et = {rhs:+.12e}  rel diff {abs(lhs - rhs) / abs(lhs):.1e}")

import sys
INNER = int(sys.argv[1]) if len(sys.argv) > 1 else 1
for shape in (1, 2):
    for mode in ("implicit", "explicit"):
        S = Sim(nx, ny, D, dy, shape)
        x, y = lattice(nx, ny, D, dy, 2, 2)
        u = rng.normal(scale=0.5, size=(len(x), 3))            # relativistic, moves up to several dy per step
        wpdt = 3.0
        w = wpdt ** 2 / (len(x) / (nx * ny * S.A))            # q^2/m * n = wp^2 with q = -w, m = w
        S.add_species(x, y, u, -w, w)
        S.neutralise()
        S.f = [1e-2 * rng.normal(size=(nx, ny)) for _ in range(6)]
        S.f[0] *= 0; S.f[1] *= 0                                  # Gauss: keep div E = 0 + no charge
        g0 = S.gauss()
        E0 = S.W() + S.K()
        worst = 0
        for n in range(40):
            if mode == "implicit":
                S.step_implicit(tol=1e-12, maxit=200, inner=INNER)
            else:
                S.step_explicit()
            worst = max(worst, abs(S.W() + S.K() - E0) / E0)
        dg = np.abs(S.gauss() - g0).max() / np.abs(S.rho_bg).max()
        its = np.mean([a for a, _, _ in S.stats]) if S.stats else 0; ps = np.mean([c for _, _, c in S.stats]) if S.stats else 0
        print(f"(b) shape {shape} {mode:8s} wp dt = {wpdt}: max |dE|/E = {worst:.1e}, Gauss change / rho = {dg:.1e}, "
              f"outer its {its:.1f}, particle passes {ps:.1f}")
