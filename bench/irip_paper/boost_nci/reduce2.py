import math, os, sys, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); N0 = 1e-3; KPLW = math.sqrt(N0) * 2 * math.pi / 0.8 * 300.0
for run in sys.argv[1:]:
    g = int(run.split("_")[0][1:]); p = os.path.join(D, run, "diag.txt")
    if not os.path.exists(p): print(run, "missing"); continue
    x = np.loadtxt(p); wt = x[:, 1] * math.sqrt(N0); r = x[:, 2] / (x[:, 3] / 1837.0); res = g * KPLW
    i = int(np.argmin(abs(wt - res))); k = int(np.argmax(r))
    print("%-12s res %7.1f  W/K at res %9.2e  max %9.2e at wpt %6.0f  dGauss %8.1e  steps %d" % (run, res, r[i], r[k], wt[k], x[:, 5].max(), int(x[-1, 0])))
