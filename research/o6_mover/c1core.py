"""The C1 map with a linear core xi = s0 u (u < uc) in the axis cell (c1map.UC): monotone? slope range? Scan (s0, s1, uc); the core
makes the m = 1 force isotropic next to the axis (dxi/dr = xi / r there). usage: python3 c1core.py"""
import numpy as np
import c1map
out = []
for s1 in (1.0, 0.7, 0.4):
    c1map.S1[0] = s1
    for s0 in (2.0, 2.5, 3.0, 3.5):
        a = None
        for uc in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25):
            if s0 * uc >= 0.9: break
            c1map.UC[0] = uc
            try:
                a, res, it, smax, smin = c1map.solve(s0, K=24, a0=a)
            except np.linalg.LinAlgError:
                print(f"s1 {s1} s0 {s0} uc {uc}: singular"); break
            ok = res < 1e-12 and smin > 0
            print(f"s1 {s1} s0 {s0} uc {uc:.2f} (xi_core {s0 * uc:.2f}): residual {res:.0e}, slope {smin:.3f} .. {smax:.3f}"
                  f"{'' if ok else '  NOT monotone/converged'}", flush=True)
            if ok and uc > 0: out.append((smax / smin, s0, s1, uc, smin, smax))
            if not res < 1e-12: break
out.sort()
print("monotone with a core, best ratio:", [(round(r, 1), s0, s1, uc, round(mn, 3), round(mx, 2)) for r, s0, s1, uc, mn, mx in out[:8]])
