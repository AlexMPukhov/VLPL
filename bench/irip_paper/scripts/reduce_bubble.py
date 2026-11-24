"""Reduce the bubble benchmark with self-injection (VLPL bench/lwfa_bubble: a0 = 4, 0.005 n_c) to ../data:
  bubble_injected.txt   t [fs], charge [pC], mean and maximum energy [MeV] of the electrons with gamma >= 20, VLPL-RIP
                        (beam_injected.txt of the diagnostic block) and FBPIC (particle output, uz >= 15 written,
                        gamma >= 20 selected here)
  bubble_spectrum.txt   dQ/dE [pC/MeV] on VLPL's bins (0 .. 400 MeV, 200 bins) at the last common output, both codes
Run with h5py:  python reduce_bubble.py RIP_DIR FBPIC_DIR"""
import os, sys, glob
import numpy as np
import h5py

rip, fb = sys.argv[1], sys.argv[2]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
mec2, qe = 0.51099895, 1.602176634e-19
lam = 0.8e-6; w0 = 2 * np.pi * 2.99792458e8 / lam

# VLPL: beam_injected.txt columns: step t Q[pC] <E> ... (compare.py: r[3] charge, r[4] mean, r[6] max)
rows = {}
for l in open(os.path.join(rip, "beam_injected.txt")):
    if l.startswith("#") or not l.strip():
        continue
    v = [float(x) for x in l.split()]
    rows[int(v[0])] = v
sp = open(os.path.join(rip, "spectrum_injected.txt")).read().splitlines()
centres = np.array([float(x) for x in sp[0].lstrip("#").split()[-200:]])
spec = {}
for l in sp[1:]:
    v = [float(x) for x in l.split()]
    spec[int(v[0])] = np.array(v[-len(centres):])
dE = centres[1] - centres[0]


def fbpic(step):
    p = glob.glob(os.path.join(fb, "hdf5", "data%08d.h5" % step))
    if not p:
        return None
    with h5py.File(p[0], "r") as f:
        it = f["data"][str(step)]
        e = it["particles"]["electrons"]
        mom = e["momentum"]
        u = [mom[c][()] * mom[c].attrs["unitSI"] / (9.1093837015e-31 * 2.99792458e8) for c in ("x", "y", "z")]
        w = e["weighting"][()] * e["weighting"].attrs["unitSI"]
        t = it.attrs["time"] * it.attrs["timeUnitSI"]
    g = np.sqrt(1 + u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    m = g >= 20
    E = (g[m] - 1) * mec2
    return t, w[m] * qe * 1e12, E


tab, last = [], None
for step in sorted(rows):
    r = rows[step]
    f = fbpic(step)
    if f is None:
        continue
    t, q, E = f
    Q = q.sum()
    tab.append([t * 1e15, r[3], Q, r[4], (q * E).sum() / Q if Q > 0 else 0, r[6], E.max() if len(E) else 0])
    last = (step, t, q, E)
np.savetxt(os.path.join(out, "bubble_injected.txt"), np.array(tab), fmt="%.4g",
           header="electrons with gamma >= 20: t [fs], charge [pC] VLPL-RIP, FBPIC, mean energy [MeV] VLPL-RIP, FBPIC, "
                  "max energy [MeV] VLPL-RIP, FBPIC")
step, t, q, E = last
h, _ = np.histogram(E, bins=len(centres), range=(centres[0] - dE / 2, centres[-1] + dE / 2), weights=q)
np.savetxt(os.path.join(out, "bubble_spectrum.txt"), np.column_stack([centres, spec[step], h / dE]), fmt="%.4e",
           header="t = %.0f fs (step %d): E [MeV], dQ/dE [pC/MeV] VLPL-RIP, FBPIC" % (t * 1e15, step))
for r in tab:
    print(" ".join("%.4g" % v for v in r))
