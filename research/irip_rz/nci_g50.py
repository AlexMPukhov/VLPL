import numpy as np, sys
from nci2d import run
for s in sys.argv[1].split(","):
    h = run(s, 64, 32, 0.63, 1.88, -50.0, int(280 / 0.63), gauss=True)
    np.savetxt("nci_g50_%s.txt" % s, h)
    print(s, "W(t):", " ".join("%.2e" % w for w in h[::5, 1]), flush=True)
