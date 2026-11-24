# Energy gain of the noise-heating runs: G(t) = (W_total + E_part_out + Enet_l + Enet_r + E_abs_r)(t) / W_total(0) - 1,
# its slope over [t0, t1] fs (per 10 fs), and the field energy fraction at the end.
import os, sys, glob
import numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else "s1"
t0, t1 = 15.0, 30.0
fs = 2 * np.pi * 2.99792458e8 * 1e-15 / 0.8e-6       # omega0 t per fs (lambda0 = 0.8 um)
print("%-14s %9s %9s %9s %9s %9s %8s %7s" % ("run", "G(10fs)", "G(20fs)", "G(30fs)", "slope/10fs", "Wf/Wk", "Nlost", "steps"))
for d in sorted(glob.glob(os.path.join(root, "*"))):
    try:
        a = np.loadtxt(os.path.join(d, "out", "diag.txt"))
        x = np.loadtxt(os.path.join(d, "out", "xflux.txt"))
    except Exception:
        print("%-14s (no data)" % os.path.basename(d)); continue
    if a.ndim < 2 or len(a) < 3: print("%-14s (running)" % os.path.basename(d)); continue
    n = min(len(a), len(x))
    a, x = a[:n], x[:n]
    t = a[:, 1] / fs
    E = a[:, 4] + x[:, 10] + x[:, 6] + x[:, 7] + x[:, 8]
    G = E / E[0] - 1
    def at(tt): return G[np.argmin(abs(t - tt))] if t[-1] >= tt - 0.5 else np.nan
    m = (t >= t0) & (t <= t1)
    sl = np.polyfit(t[m], G[m], 1)[0] * 10 if m.sum() > 2 else np.nan
    print("%-14s %9.2e %9.2e %9.2e %9.2e %9.2e %8d %7d" % (os.path.basename(d), at(10), at(20), at(30), sl, a[-1, 2] / a[-1, 3],
          int(a[0, 8] - a[-1, 8]), int(a[-1, 0])))
