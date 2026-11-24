"""RZ m = 0 prototype checks: (a) energy identity of the box step with a current, (b) hot plasma near the axis: implicit
W + K to the solve tolerance, Gauss law per node at round-off (implicit and explicit)."""
import numpy as np
from rz0 import RZ0

rng = np.random.default_rng(2)
nz, nr, D, dr = 16, 32, 0.3, 0.1
S = RZ0(nz, nr, D, dr)
print("Ruyten b2[:5] =", S.b2[:5])
for t in range(2):
    S.f = [rng.normal(size=(nz, nr)) for _ in range(3)]
    J = [rng.normal(size=(nz, nr)) for _ in range(2)]
    W0 = S.W()
    f1, Et = S.field_solve(J)
    S.f = f1
    rhs = -D * 2 * np.pi * D * ((J[0] * Et[0] * S.A).sum() + (J[1] * Et[1] * S.Bw).sum())
    print(f"(a) dW {S.W() - W0:+.12e}  work {rhs:+.12e}  rel {abs(S.W() - W0 - rhs) / abs(rhs):.1e}")


def plasma(S, ppz, ppr, ppt, rmax, uth, n):
    zc = (np.arange(S.nz * ppz) + 0.5) * S.D / ppz
    gl = np.array([0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)]) if ppr == 2 else (np.arange(ppr) + 0.5) / ppr
    rr, ww = [], []
    for j in range(int(round(rmax / S.dr))):
        for g in gl:
            r = (j + g) * S.dr
            rr.append(r); ww.append(r * S.dr / ppr)            # integral of r dr over the cell
    rr, ww = np.array(rr), np.array(ww)
    Z, R, T = np.meshgrid(zc, rr, np.arange(ppt), indexing="ij")
    Wt = np.broadcast_to(ww[None, :, None], Z.shape)
    ph = (T + rng.random(T.shape)) * 2 * np.pi / ppt
    z, r, ph, wt = Z.ravel(), R.ravel(), ph.ravel(), Wt.ravel()
    q = -n * wt * 2 * np.pi / ppt * (S.D / ppz)                 # charge of a ring segment of density n
    u = rng.normal(scale=uth, size=(len(z), 3))
    S.add_species(z, r * np.cos(ph), r * np.sin(ph), u, q, -q)
    S.neutralise()


for mode in ("implicit", "explicit"):
    S = RZ0(nz, nr, D, dr)
    plasma(S, 2, 2, 8, 0.6, 0.05, 30.0)                         # wp dt = 1.6
    g0 = S.gauss(); E0 = S.W() + S.K(); worst = 0
    for n in range(30):
        S.step_implicit(tol=1e-9, maxit=200, inner=1) if mode == "implicit" else S.step_explicit()
        worst = max(worst, abs(S.W() + S.K() - E0) / E0)
    dg = np.abs(S.gauss() - g0).max() / np.abs(S.rho_bg).max()
    st = np.mean([s[0] for s in S.stats]) if S.stats else 0
    ps = np.mean([s[2] for s in S.stats]) if S.stats else 0
    print(f"(b) {mode}: max |dE|/E {worst:.1e}, Gauss change/rho {dg:.1e}, outer {st:.1f}, passes {ps:.1f}")
