"""Review points 3 and 4 (2026-11-12): bubble benchmark (VLPL bench/lwfa_bubble) in RZ on FBPIC's grid and with finer
radial cells. Charge, mean and maximum energy of the electrons with gamma >= 20 at every output (100 fs):
VLPL runs from beam_injected.txt, FBPIC runs from the particle output (as reduce_bubble.py). Run with h5py:
  python reduce_bubble_rz.py REVIEW_DIR OLD_RUNS_DIR
REVIEW_DIR: bub_{irz,rz}_h{5,2,1,05}, bub_irz_h5_f1 (VLPL, out/beam_injected.txt), fbb_nofilter, fbb_linear (FBPIC);
OLD_RUNS_DIR: bubble_rip (3D VLPL-RIP), bubble_fbpic (FBPIC reference). Writes ../data/bubble_rz.txt."""
import os, sys, glob
import numpy as np
import h5py

R, O = sys.argv[1], sys.argv[2]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
mec2, qe = 0.51099895, 1.602176634e-19


def vlpl(d):
    p = os.path.join(d, "beam_injected.txt")
    if not os.path.exists(p):
        p = os.path.join(d, "out", "beam_injected.txt")
    if not os.path.exists(p):
        return None
    res = {}
    for l in open(p):
        if l.startswith("#") or not l.strip():
            continue
        v = [float(x) for x in l.split()]
        res[int(round(v[1] / 100.0)) * 100] = (v[3], v[4], v[6])
    return res


def fbpic(d):
    res = {}
    for p in sorted(glob.glob(os.path.join(d, "hdf5", "data*.h5"))):
        with h5py.File(p, "r") as f:
            it = f["data"][list(f["data"].keys())[0]]
            t = it.attrs["time"] * it.attrs["timeUnitSI"] * 1e15
            if "particles" not in it or "electrons" not in it["particles"]:
                continue
            e = it["particles"]["electrons"]
            if "momentum" not in e:
                continue
            mom = e["momentum"]
            u = [mom[c][()] * mom[c].attrs["unitSI"] / (9.1093837015e-31 * 2.99792458e8) for c in ("x", "y", "z")]
            w = e["weighting"][()] * e["weighting"].attrs["unitSI"]
        g = np.sqrt(1 + u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        m = g >= 20
        q = w[m] * qe * 1e12; E = (g[m] - 1) * mec2
        Q = q.sum()
        res[int(round(t / 100.0)) * 100] = (Q, (q * E).sum() / Q if Q > 0 else 0, E.max() if len(E) else 0)
    return res


runs = [("VLPL-RIP 3D h5", vlpl(os.path.join(O, "bubble_rip"))), ("FBPIC", fbpic(os.path.join(O, "bubble_fbpic"))),
        ("FBPIC no filter", fbpic(os.path.join(R, "fbb_nofilter"))), ("FBPIC linear", fbpic(os.path.join(R, "fbb_linear")))]
for n, lab in (("bub_rz_h5", "VLPL-RZ h5"), ("bub_rz_h2", "VLPL-RZ h2"), ("bub_irz_h5", "VLPL-iRZ h5"),
               ("bub_irz_h5_f1", "VLPL-iRZ h5 filter1"), ("bub_irz_h2", "VLPL-iRZ h2"), ("bub_irz_h1", "VLPL-iRZ h1"),
               ("bub_irz_h05", "VLPL-iRZ h05")):
    runs.append((lab, vlpl(os.path.join(R, n))))
with open(os.path.join(out, "bubble_rz.txt"), "w") as f:
    f.write("# electrons with gamma >= 20: Q [pC] / mean E [MeV] / max E [MeV] at t = 400, 600, 900 fs; h = h_r / (c tau)\n")
    for lab, r in runs:
        if not r:
            continue
        cols = []
        for t in (400, 600, 900):
            v = r.get(t, (np.nan,) * 3)
            cols.append("%6.1f %5.1f %5.1f" % v)
        line = "%-22s %s" % (lab, " | ".join(cols))
        f.write(line + "\n"); print(line)
