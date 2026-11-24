#!/usr/bin/env python3
"""W_field / K_e of the streaming-plasma runs at omega_p t = 10, 100, 300, 600 (bench/nci/mechanism). K_e = W_kin / 1837 (electrons
and protons drift with the same gamma; the seed's thermal energy is negligible). omega_p of the proper density n0.
  wk_table.py DIR [DIR ...]   (n0 read from the run name: _n002 / _n008, gamma 20 -> 0.002, else 0.004)"""
import sys, os, glob, numpy as np
for D in sys.argv[1:]:
    for d in sorted(glob.glob(os.path.join(D, "*", "diag.txt"))):
        name = os.path.basename(os.path.dirname(d))
        n0 = 0.002 if ("_n002" in name or "_g20_" in name) else 0.008 if "_n008" in name else 0.004
        a = np.loadtxt(d)
        if a.ndim < 2 or len(a) < 3: continue
        wt = a[:, 1] * np.sqrt(n0); r = a[:, 2] / (a[:, 3] / 1837.0 + 1e-300)
        vals = []
        for T in (10, 100, 300, 600):
            vals.append("%9.2e" % r[np.argmin(np.abs(wt - T))] if wt[-1] >= 0.98 * T else "        -")
        print("%-28s %s   max|dGauss| %.1e" % (name, " ".join(vals), a[:, 5].max()))
