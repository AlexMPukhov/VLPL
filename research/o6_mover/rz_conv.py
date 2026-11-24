"""RZ m = 0: convergence of the nonlinear solve against wp dt, dz/dr and temperature (plasma near the axis).
usage: python3 rz_conv.py D uth n [tol]"""
import sys
import numpy as np
from rz0 import RZ0
rng = np.random.default_rng(2)
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

D, uth, n = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
tol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-9
rmap = {"plain": False, "exp": "exp", "pow": "pow", "hermite": "hermite", "c1": "c1"}[sys.argv[5]] if len(sys.argv) > 5 else "pow"
fixed = len(sys.argv) <= 6 or sys.argv[6] != "adaptive"      # fixed path segments (default) or the adaptive count
S = RZ0(16, 32, D, 0.1, rmap=rmap)
if not fixed: S.nseg_fixed = None
if len(sys.argv) > 7 and sys.argv[7] == "midpoint": S.per_segment = False
plasma(S, 2, 2, 8, 0.6, uth, n)
E0 = S.W() + S.K(); worst = 0
for k in range(30):
    S.step_implicit(tol=tol, maxit=200, inner=1)
    worst = max(worst, abs(S.W() + S.K() - E0) / E0)
its = [s[0] for s in S.stats]
if S.b2 is not None: print("map slopes s_0..4", np.round(S.b2[:5], 4), "nseg", S.nseg_fixed)
print(f"D = {D} (dz/dr = {D / 0.1:.0f}), wp dt = {np.sqrt(n) * D:.2f}, uth = {uth}, map {rmap}: outer mean {np.mean(its):.1f} max {max(its)}, "
      f"final residual max {max(s[1] for s in S.stats):.1e}, |dE|/E {worst:.1e}", flush=True)

