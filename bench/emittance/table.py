"""eps_n,y of the emittance runs at a few propagation distances (bench/emittance/README.md).

python3 table.py A <run_dir>   # case A: emA_<case>/beam_beam.txt and fbpic/fbA_q<Q>_g<G>.txt, distances in mm
python3 table.py B <run_dir>   # case B: emB_<case>/beam_beam.txt and fbpic/fbB_g<G>_dr<dr>.txt, from the load
Prints one markdown row per run: eps_n,y [nm or um] at the distances, and the mean energy at the end.
"""
import glob
import os
import sys
import numpy as np

case, d = sys.argv[1], sys.argv[2]
c_um_fs = 0.299792458                      # um per fs


def vlpl(fn):
    a = np.loadtxt(fn, comments="#", ndmin=2)
    a = a[a[:, 2] > 0]                     # with particles
    return a[:, 1], a[:, 7], a[:, 4]       # t [fs], eps_ny [um], E_mean [MeV]


def fbpic(fn):
    a = np.loadtxt(fn, comments="#", ndmin=2)
    a = a[np.isfinite(a[:, 2])]
    return a[:, 1], a[:, 2], a[:, 5]


def at(t, e, tq):
    i = int(np.argmin(np.abs(t - tq)))
    return e[i] if abs(t[i] - tq) < 40 else np.nan


runs = []
if case == "A":
    for fn in sorted(glob.glob(d + "/emA_*/beam_beam.txt")):
        runs.append((os.path.basename(os.path.dirname(fn))[4:], *vlpl(fn)))
    for fn in sorted(glob.glob(d + "/fbpic/fbA_*.txt")):
        runs.append(("fbpic_" + os.path.basename(fn)[4:-4], *fbpic(fn)))
    dist = [0.0, 0.02, 0.1, 0.5, 1.0]      # mm
    print("| run | " + " | ".join("%g mm" % x for x in dist) + " | last [mm: nm] | E [MeV] |")
    print("|---|" + "---|" * (len(dist) + 2))
    for name, t, e, E in runs:
        cells = []
        for x in dist:
            v = at(t, e, x * 1e3 / c_um_fs)        # VLPL writes no step 0: "-"
            cells.append("-" if not np.isfinite(v) else "%.4g nm" % (v * 1e3))
        print("| %s | %s | %.3g: %.4g | %.0f |" % (name, " | ".join(cells), t[-1] * c_um_fs * 1e-3, e[-1] * 1e3, E[-1]))
else:
    for fn in sorted(glob.glob(d + "/emB_*/beam_beam.txt")):
        runs.append((os.path.basename(os.path.dirname(fn))[4:], *vlpl(fn)))
    for fn in sorted(glob.glob(d + "/fbpic/fbB_*.txt")):
        runs.append(("fbpic_" + os.path.basename(fn)[4:-4], *fbpic(fn)))
    dist = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # mm after the load (t = 150 fs)
    print("| run | " + " | ".join("%g mm" % x for x in dist) + " | E [MeV] |")
    print("|---|" + "---|" * (len(dist) + 1))
    for name, t, e, E in runs:
        t0 = t[0]
        cells = []
        for x in dist:
            v = at(t, e, t0 + x * 1e3 / c_um_fs)
            cells.append("-" if not np.isfinite(v) else "%.4g" % v)
        print("| %s | %s | %.1f |" % (name, " | ".join(cells), E[-1]))
