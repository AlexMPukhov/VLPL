"""The C1 map with a free slope s1 at node 1 (c1map.S1): the range of dxi/dr (max / min) over (s0, s1). A flatter map makes the
radial force of the implicit mover less nonlinear in the axis cells (s1 = 1: min slope 0.03 at s0 = 3.3, ratio ~100).
usage: python3 c1scan.py"""
import numpy as np
import c1map
best = []
for s1 in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3):
    c1map.S1[0] = s1
    a = None
    for s0 in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        try:
            a, res, it, smax, smin = c1map.solve(s0, K=24, a0=a)
        except np.linalg.LinAlgError:
            print(f"s1 {s1} s0 {s0}: singular"); a = None; continue
        ok = res < 1e-12 and smin > 0
        print(f"s1 {s1} s0 {s0}: residual {res:.0e}, slope max {smax:.3f} min {smin:.4f} ratio {smax / max(smin, 1e-9):.1f}"
              f"{'' if ok else '  (rejected)'}, a_0..2 {np.round(a[:3], 3)}", flush=True)
        if ok: best.append((smax / smin, s0, s1, smax, smin))
        else: a = None
best.sort()
print("best ratios:", [(round(r, 1), s0, s1, round(mx, 2), round(mn, 3)) for r, s0, s1, mx, mn in best[:6]])
