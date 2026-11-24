"""Compare the RZ run of VLPL (rip_rz.deck) with FBPIC (fbpic_lwfa.py): both written as openPMD
thetaMode and read with openpmd-viewer in the same way.

python3 bench/lwfa_fbpic/compare_rz.py out_bench_rip_rz out_bench_fbpic [tolerance]
Prints, per output time, the peak wake Ez (m = 0, averaged over r <= 0.5 um) behind the pulse of both
codes, their relative L2 difference there (and on the axis row alone, noisier), and the peak laser field
E_x (theta = 0); exits 1 if the near-axis L2 difference exceeds the tolerance (default 0.10).
"""
import os, sys
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

rip, fb = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
lam = 0.8e-6
E0 = 9.1093837015e-31 * 2.99792458e8 * (2 * np.pi * 2.99792458e8 / lam) / 1.602176634e-19   # m c w0 / e [V/m]
tr = OpenPMDTimeSeries(os.path.join(rip, "openpmd"), check_all_files=False)
tf = OpenPMDTimeSeries(os.path.join(fb, "hdf5"), check_all_files=False)

RAVG = 0.5e-6                      # the wake Ez (m = 0) averaged over r <= RAVG: two radial cells

def fields(ts, it):
    """Ez of m = 0 averaged over r <= RAVG and on the axis row, and the peak laser field E_x in the plane
    theta = 0 (all modes), units m c w0 / e. The axis row alone is dominated by particle noise with
    VLPL's linear shapes (two runs that differ only in the sampling differ by 5-14 % there)."""
    ez, info = ts.get_field("E", "z", iteration=it, m=0)
    r = np.abs(info.r)
    near = ez[r <= RAVG + 1e-12].mean(axis=0)
    axis = ez[r == r.min()].mean(axis=0)
    ex, _ = ts.get_field("E", "x", iteration=it, m="all", theta=0)
    return info.z, near / E0, axis / E0, np.max(np.abs(ex)) / E0

worst = 0.0
print("%8s %10s %10s %9s %9s %10s %10s" % ("t [fs]", "Ez_RIP", "Ez_FBPIC", "L2 r<0.5", "L2 axis", "E_las RIP", "E_las FB"))
for it in tr.iterations:
    if it == 0 or it not in tf.iterations:
        continue
    zr, ezr, axr, lr = fields(tr, it)
    zf, ezf, axf, lf = fields(tf, it)
    ezf_on, axf_on = np.interp(zr, zf, ezf), np.interp(zr, zf, axf)
    # behind the pulse: from the window's left edge to 2 um behind the laser centre (peak of E_x on axis)
    exa, info = tr.get_field("E", "x", iteration=it, m="all", theta=0, slice_across="r")
    zl = info.z[np.argmax(np.abs(exa))] - 2e-6
    m = zr < zl
    if m.sum() < 10 or np.max(np.abs(ezf_on[m])) < 5e-3:      # no wake yet
        continue
    l2 = lambda a, b: np.sqrt(np.sum((a[m] - b[m]) ** 2) / np.sum(b[m] ** 2))
    d, da = l2(ezr, ezf_on), l2(axr, axf_on)
    worst = max(worst, d)
    t = tr.t[list(tr.iterations).index(it)]
    print("%8.1f %10.4f %10.4f %9.3f %9.3f %10.4f %10.4f" % (t * 1e15, np.max(np.abs(ezr[m])), np.max(np.abs(ezf_on[m])), d, da, lr, lf))
print("largest relative L2 difference of the near-axis Ez (r <= 0.5 um) behind the pulse: %.3f (tolerance %.2f)" % (worst, tol))
sys.exit(0 if worst <= tol else 1)
