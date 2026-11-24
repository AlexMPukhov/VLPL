import sys, glob, os, numpy as np
for d in sorted(glob.glob(sys.argv[1] + "/*/diag.txt")):
    n0 = float(sys.argv[2]); a = np.loadtxt(d)
    wt = a[:, 1] * np.sqrt(n0); r = a[:, 2] / (a[:, 3] / 1837.0)
    v = ["%9.2e" % r[np.argmin(np.abs(wt - T))] if wt[-1] >= 0.98 * T else "        -" for T in (10, 100, 213, 300, 533, 600)]
    print("%-10s %s  max|dGauss| %.1e" % (os.path.basename(os.path.dirname(d)), " ".join(v), a[:, 5].max()))
