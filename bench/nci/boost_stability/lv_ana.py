#!/usr/bin/env python3
"""Lee-Vay 1e19 case (bench/nci/boost_stability/README.md, section 2c): injected bunch (gamma >= 50, as Lee and Vay) and on-axis
E_x of every run at the lab snapshots K = 0, 1, 2 (600, 800, 1000 fs; the long lab run also K = 3 .. 6 up to 1800 fs).
  lv_ana.py OUT [REF]   (REF: the reference lab run for the wake, default lab_irip_64)"""
import sys, os, glob, numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools"))
from lab_fields import read_snapshot
O = sys.argv[1]; REF = sys.argv[2] if len(sys.argv) > 2 else "lab_irip_64"
E_C, MEC2, UM, GMIN, X0 = 1.602176634e-19, 0.51099895, 0.8 / (2 * np.pi), 50.0, 45.0
def lab_file(run, K):
    every = 2400 if "_64" in run else 1200
    return os.path.join(O, run, "openpmd", "data_%06d.h5" % (every * (6 + 2 * K)))
def lab_data(run, K):
    h = h5py.File(lab_file(run, K)); g = h["data"][list(h["data"])[0]]
    E = g["meshes"]["E"]; ex = np.asarray(E["x"])[0]; ey = np.asarray(E["y"])[0]
    dx, x0 = E.attrs["gridSpacing"][-1], E.attrs["gridGlobalOffset"][-1]
    x = (x0 + dx * np.arange(ex.shape[-1])) * UM
    p = g["particles"]["electrons"]; mc = 9.1093837015e-31 * 299792458.0
    u = np.stack([np.asarray(p["momentum"][c]) * p["momentum"][c].attrs["unitSI"] / mc for c in ("x", "y", "z")], 1)
    xp = np.asarray(p["position"]["x"]) * p["position"]["x"].attrs["unitSI"] * 1e6
    w = np.asarray(p["weighting"]) * E_C * 1e12
    return x, ex[ex.shape[0] // 2], ey[ey.shape[0] // 2], xp, u, w, g.attrs["time"] * g.attrs["timeUnitSI"] * 1e15
def boost_data(run, K):
    s = read_snapshot(os.path.join(O, run, "lab_fields", "snap_%03d" % K)); f = s["fields"]
    x = s["x"] * UM; j = f["Ex"].shape[1] // 2
    b = os.path.join(O, run, "lab_particles", "electrons_%03d" % K)
    meta = dict(l.split(" ", 1) for l in open(b + ".txt").read().splitlines() if " " in l and not l.startswith("#"))
    a = np.fromfile(b + ".bin").reshape(-1, 7)
    return x, f["Ex"][0, j], f["Ey"][0, j], a[:, 0] * float(meta["length_um"]), a[:, 3:6], np.abs(a[:, 6] * float(meta["weight_pC"])), float(meta["t_lab_fs"])
def env(line):
    n = len(line); F = np.fft.fft(line); F[n // 2 + 1:] = 0; F[1:n // 2] *= 2
    return np.abs(np.fft.ifft(F))
runs = sorted(d for d in os.listdir(O) if os.path.isdir(os.path.join(O, d)) and (d.startswith("lab_") or d.startswith("b")))
for K in range(7):
    try: xr, exr, eyr, *_ = lab_data(REF, K)
    except Exception: xr = None
    lines = []
    for r in runs:
        try: x, ex, ey, xp, u, w, t = (lab_data if r.startswith("lab_") else boost_data)(r, K)
        except Exception: continue
        g = np.sqrt(1 + (u ** 2).sum(1)); s = (g >= GMIN) & (u[:, 0] > 0)
        e = env(ey); xl = x[np.argmax(e)]
        if s.any():
            E = (g[s] - 1) * MEC2; Em = np.average(E, weights=w[s]); dE = np.sqrt(np.average((E - Em) ** 2, weights=w[s]))
            b = "Q %7.2f pC  E %6.1f MeV  dE/E %.3f  N %6d" % (w[s].sum(), Em, dE / Em, s.sum())
        else: b = "no electrons above gamma 50"
        d = ""
        if xr is not None:
            Li = np.interp(xr, x, ex, left=np.nan, right=np.nan); ok = ~np.isnan(Li)
            for nm, lo, hi in (("bubble", 25, 6), ("front", 10, 6)):
                m = ok & (xr > xl - lo) & (xr < xl + hi)
                d += "  %s %.3f" % (nm, np.sqrt(np.mean((Li - exr)[m] ** 2) / np.mean(exr[m] ** 2)))
        lines.append("  %-18s t %6.1f fs z %5.1f um  a %.2f  %s%s" % (r, t, xl - X0, e.max(), b, d))
    if lines: print("K = %d" % K); print("\n".join(lines))
