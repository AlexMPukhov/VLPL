"""Generic boosted-frame drift boxes (paper irip, 2026-11-21): lambda0 = 0.8 um, lab n0 = 1e-3 n_c, lab lambda/32, L_w = 300 um,
gamma 5 / 10 / 20, 128 x 16 x 16, RIP-CC, shape 2, 12 ppc. W_field / K_e (K_e = W_kin / 1837) at the residence time
omega_p t = gamma k_p L_w (omega_p of the lab density, t the boosted time), max over the run, max dGauss. As nci/d3nci/d3long.py."""
import math, os, sys, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); N0 = 1e-3; KPLW = math.sqrt(N0) * 2 * math.pi / 0.8 * 300.0
for g in (5, 10, 15, 20):
    for f in ("f0", "f2", "f8cz"):
        p = os.path.join(D, "g%d_%s" % (g, f), "diag.txt")
        if not os.path.exists(p): print(g, f, "missing"); continue
        x = np.loadtxt(p); wt = x[:, 1] * math.sqrt(N0); r = x[:, 2] / (x[:, 3] / 1837.0); res = g * KPLW
        i = int(np.argmin(abs(wt - res))); k = int(np.argmax(r))
        print("%-4d %-5s res %7.1f  W/K at res %9.2e  max %9.2e at wpt %6.0f  dGauss %8.1e  steps %d" % (g, f, res, r[i], r[k], wt[k], x[:, 5].max(), int(x[-1, 0])))
