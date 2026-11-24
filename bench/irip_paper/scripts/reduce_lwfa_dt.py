"""Review point 5 (2026-11-12): LWFA RZ benchmark (VLPL bench/lwfa_fbpic/rip_rz.deck) at c tau = h_x = lambda/16 (d1)
and lambda/32 (d2), VLPL-RZ (RIP-CC) and VLPL-iRZ (box iRIP), FBPIC (2 modes) at both resolutions. Near-axis E_z of
m = 0 (r <= 0.5 um), relative L2 behind the pulse (as reduce_lwfa.py). Run with h5py:
  python reduce_lwfa_dt.py REVIEW_DIR OLD_RUNS_DIR   (REVIEW_DIR/lw_{rz,irz}_{d1,d2}/out/openpmd, fb_d2/hdf5;
                                                     OLD_RUNS_DIR/fbpic_lwfa/hdf5 = FBPIC at lambda/16)
Writes ../data/lwfa_dt_l2.txt."""
import os, sys, glob
import numpy as np
import h5py

R, O = sys.argv[1], sys.argv[2]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
lam = 0.8e-6; c = 2.99792458e8
E0 = 9.1093837015e-31 * c * (2 * np.pi * c / lam) / 1.602176634e-19


def thetamode(path):
    with h5py.File(path, "r") as f:
        it = f["data"][list(f["data"].keys())[0]]
        mp = f.attrs["meshesPath"]; mp = (mp.decode() if isinstance(mp, bytes) else mp).strip("/")
        E = it[mp]["E"]
        sp = E.attrs["gridSpacing"] * E.attrs["gridUnitSI"]; off = E.attrs["gridGlobalOffset"] * E.attrs["gridUnitSI"]
        ez = E["z"][()] * E["z"].attrs["unitSI"]; pz = E["z"].attrs["position"]
        er = E["r"][()] * E["r"].attrs["unitSI"]
        z = off[1] + (np.arange(ez.shape[2]) + pz[1]) * sp[1]
        r = off[0] + (np.arange(ez.shape[1]) + pz[0]) * sp[0]
        las = np.abs(er[1]).max(axis=0)                           # m = 1 (laser) envelope along z
        return dict(z=z, r=r, ez=ez, zl=z[np.argmax(las)], t=it.attrs["time"] * it.attrs["timeUnitSI"])


def series(d):
    fs = glob.glob(os.path.join(d, "*.h5"))
    res = {}
    for p in fs:
        m = thetamode(p)
        t = m["t"] * 1e15
        if abs(t - round(t / 50) * 50) < 5:
            res[round(t / 50) * 50] = m
    return res


near = lambda d: d["ez"][0][d["r"] <= 0.5e-6 + 1e-12].mean(axis=0) / E0
runs = {k: series(os.path.join(R, k, "out", "openpmd")) for k in ("lw_rz_d1", "lw_irz_d1", "lw_rz_d2", "lw_irz_d2")}
fb = {"d1": series(os.path.join(O, "fbpic_lwfa", "hdf5")), "d2": series(os.path.join(R, "fb_d2", "hdf5"))}
rows = []
for t in (100, 150, 200, 250, 300):
    row = [t]
    for d in ("d1", "d2"):
        f, a, b = fb[d].get(t), runs["lw_rz_" + d].get(t), runs["lw_irz_" + d].get(t)
        if f is None or a is None or b is None:
            row += [np.nan] * 3; continue
        x = a["z"]; m = x < a["zl"] - 2e-6
        F = np.interp(x, f["z"], near(f)); A = near(a); B = np.interp(x, b["z"], near(b))
        l2 = lambda u, v: np.sqrt(np.sum((u[m] - v[m]) ** 2) / np.sum(v[m] ** 2))
        row += [l2(A, F), l2(B, F), l2(B, A)]
    # FBPIC d2 vs d1 (convergence of the reference)
    f1, f2 = fb["d1"].get(t), fb["d2"].get(t)
    if f1 is not None and f2 is not None:
        x = f2["z"]; m = x < f2["zl"] - 2e-6
        u, v = np.interp(x, f1["z"], near(f1)), near(f2)
        row.append(np.sqrt(np.sum((u[m] - v[m]) ** 2) / np.sum(v[m] ** 2)))
    else:
        row.append(np.nan)
    rows.append(row)
np.savetxt(os.path.join(out, "lwfa_dt_l2.txt"), np.array(rows), fmt="%.4g",
           header="t[fs]; c tau = lambda/16: L2 RZ vs FBPIC, iRZ vs FBPIC, iRZ vs RZ; lambda/32: the same (FBPIC at lambda/32); "
                  "FBPIC lambda/16 vs lambda/32")
for r in rows:
    print(" ".join("%.4g" % v for v in r))
