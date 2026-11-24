import sys, os
import numpy as np
sys.path.insert(0, "$WORK/rip-pic-hhgang/tools")
import hfocus
root = "$WORK/chfgap/r3"
print("order grid solver  peak     z_um    fwhm_z  fwhm_r  fluence")
for N, pre in ((5, "n5_"), (7, ""), (10, "n10_")):
    for g in ("g1", "g2", "g4"):
        for s in ("cc", "ir", "o5"):
            d = os.path.join(root, "%s%s_%s" % (pre, g, s), "out", "harmonic_focus_h")
            try:
                geom, q, A, c, m = hfocus.load(d)
                r = hfocus.analyse(geom, q, A, c)
                p, f = r["peak_phi0"], r["fluence_phi0"]
                print("%3d   %s   %s   %8.4f %7.4f %7.4f %7.4f %8.3f" % (N, g, s, p["max"], p["z_um"], p["fwhm_z_um"], p["fwhm_r_um"], f["max"]))
            except BaseException:
                pass
for r in ("g1_cc", "g1_ir", "g1_o5"):
    a = np.loadtxt(os.path.join(root, "w_" + r, "out", "diag.txt"))
    W = a[:, 2]
    print("energy", r, "max W %.5f  at t %.1f  W(t) / max at 1/4,1/2,3/4 of run: %s" % (W.max(), a[W.argmax(), 1],
          " ".join("%.4f" % (W[int(len(W) * x)] / W.max()) for x in (0.25, 0.5, 0.75))))
