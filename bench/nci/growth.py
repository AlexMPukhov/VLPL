#!/usr/bin/env python3
"""NCI table of bench/nci/run_boost.sh: W_field / W_kin(0) at omega_p t = 28, 56, 112, 282 and the e-folding rate of
the field energy over the second half of the run (units of omega_p), for every nci_<case>/diag.txt given.
    python3 bench/nci/growth.py nci_*        (n = 0.01 n_c: omega_p = 0.1 omega0)
"""
import sys
import numpy as np

wp = 0.1
print("| case | " + " | ".join("w_p t = %d" % t for t in (28, 56, 112, 282)) + " | rate (2nd half) [w_p] |")
print("|---|---|---|---|---|---|")
for d in sys.argv[1:]:
    a = np.loadtxt(d + "/diag.txt")
    t, w, k0 = a[:, 1] * wp, a[:, 2], a[0, 3]
    vals = []
    for tt in (28, 56, 112, 282):
        i = np.argmin(abs(t - tt))
        vals.append("%.1e" % (w[i] / k0))
    h = t > t[-1] / 2
    rate = np.polyfit(t[h], np.log(np.maximum(w[h], 1e-300)), 1)[0]
    print("| %s | %s | %+.4f |" % (d.split("nci_")[-1], " | ".join(vals), rate))
