"""Compare VLPL (rip.deck) with FBPIC (fbpic_lwfa.py): on-axis longitudinal field and laser amplitude.

python3 bench/lwfa_fbpic/compare.py out_bench_rip out_bench_fbpic
Prints, per output time, the peak on-axis Ex of both codes, the relative L2 difference of the on-axis
Ex behind the pulse and the peak laser field; exits 1 if the L2 difference exceeds the tolerance.
"""
import glob, os, sys
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

rip, fb = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
lam = 0.8e-6
k0 = 2 * np.pi / lam
E0 = 9.1093837015e-31 * 2.99792458e8 * (2 * np.pi * 2.99792458e8 / lam) / 1.602176634e-19   # m c w0 / e [V/m]

meta = dict(l.split(None, 1) for l in open(os.path.join(rip, "slices", "meta.txt")))
nx, ny, dx = int(meta["nx"]), int(meta["ny"]), float(meta["dx"])
index = [l.split() for l in open(os.path.join(rip, "slices", "index.txt"))]

def rip_slice(name, step):
    return np.fromfile(os.path.join(rip, "slices", "%s_%06d.f32" % (name, step)), dtype=np.float32).reshape(ny, nx)

ts = OpenPMDTimeSeries(os.path.join(fb, "hdf5"), check_all_files=False)

def fbpic_fields(it):
    """on-axis Ez (all modes, theta = 0) and the peak of the laser field E_x in the plane theta = 0"""
    ez, info = ts.get_field("E", "z", iteration=it, m="all", theta=0, slice_across="r")
    exl, _ = ts.get_field("E", "x", iteration=it, m="all", theta=0)
    return info.z, ez / E0, np.max(np.abs(exl)) / E0

worst = 0.0
print("%8s %10s %10s %8s %10s %10s" % ("t [fs]", "Ex_RIP", "Ex_FBPIC", "L2 diff", "E_las RIP", "E_las FB"))
for step_s, t_s, x0_s in index:
    step = int(step_s)
    if step == 0 or step not in ts.iterations:
        continue
    ex = rip_slice("Ex", step)[ny // 2]             # Ex at (i, ny/2, nz/2): on axis
    ez = rip_slice("Ez", step)
    x = (float(x0_s) + dx * np.arange(nx)) / k0      # absolute x [m]
    zf, exf, lasf = fbpic_fields(step)
    exf_on = np.interp(x, zf, exf)
    # compare behind the pulse: from the window's left edge to 2 um behind the laser centre
    xl = x[np.argmax(np.abs(ez[ny // 2]))] - 2e-6
    m = x < xl
    if m.sum() < 10 or np.max(np.abs(exf_on[m])) < 5e-3:      # no wake yet (pulse still in vacuum)
        continue
    d = np.sqrt(np.sum((ex[m] - exf_on[m]) ** 2) / np.sum(exf_on[m] ** 2))
    worst = max(worst, d)
    print("%8.1f %10.4f %10.4f %8.3f %10.4f %10.4f" % (float(t_s) / (2 * np.pi * 2.99792458e8 / lam) * 1e15,
          np.max(np.abs(ex[m])), np.max(np.abs(exf_on[m])), d, np.max(np.abs(ez)), lasf))
print("largest relative L2 difference of the on-axis Ex behind the pulse: %.3f (tolerance %.2f)" % (worst, tol))
sys.exit(0 if worst <= tol else 1)
