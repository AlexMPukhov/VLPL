#!/usr/bin/env python3
"""Injected bunch in the lab frame (bench/nci/boost_stability/README.md): charge, mean energy, rms energy spread and
centre of the electrons with gamma >= GMIN at lab snapshot K.
  bunch.py OUT K GMIN RUN...   (boosted runs: lab_particles/electrons_KKK; lab runs (name contains _lab_): openPMD at
                                 step 1920 (3 + 2 K), the lab snapshot times of the boosted runs)"""
import sys, os, glob, numpy as np, h5py
O, K, GMIN = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
E_C, MEC2 = 1.602176634e-19, 0.51099895
def boosted(run):
    b = os.path.join(O, run, "lab_particles", "electrons_%03d" % K)
    meta = dict(l.split(" ", 1) for l in open(b + ".txt").read().splitlines() if " " in l and not l.startswith("#"))
    a = np.fromfile(b + ".bin").reshape(-1, 7)
    u = a[:, 3:6]; q = np.abs(a[:, 6] * float(meta["weight_pC"])); x = a[:, 0] * float(meta["length_um"])
    return x, u, q, float(meta["t_lab_fs"])
def lab(run):
    f = os.path.join(O, run, "openpmd", "data_%06d.h5" % (1920 * (3 + 2 * K)))
    h = h5py.File(f); it = list(h["data"])[0]; g = h["data"][it]; p = g["particles"]["electrons"]
    comps = [c for c in ("x", "y", "z") if c in p["momentum"]]
    mu = p["momentum"][comps[0]].attrs["unitSI"]; mc = 9.1093837015e-31 * 299792458.0
    u = np.stack([np.asarray(p["momentum"][c]) * mu / mc for c in comps], 1)
    lon = "z" if g["meshes"]["E"].attrs["geometry"] == b"thetaMode" else "x"
    x = np.asarray(p["position"][lon]) * p["position"][lon].attrs["unitSI"] * 1e6
    w = np.asarray(p["weighting"]) if "weighting" in p else np.full(len(x), p["weighting"].attrs["value"])
    t = g.attrs["time"] * g.attrs["timeUnitSI"] * 1e15
    if lon == "z": u = u[:, [2, 0, 1]]           # longitudinal first
    return x, u, w * E_C * 1e12, t
for run in sys.argv[4:]:
    try:
        x, u, q, t = (lab if "_lab_" in run else boosted)(run)
    except Exception as e:
        print("%-28s - (%s)" % (run, e)); continue
    if "_lab_" not in run:                         # longitudinal = the column with the largest mean |u| among the energetic
        gg = np.sqrt(1 + (u ** 2).sum(1)); s = gg >= GMIN
        if s.any(): c = int(np.argmax(np.abs(u[s]).mean(0))); u = u[:, [c] + [k for k in range(3) if k != c]]
    g = np.sqrt(1 + (u ** 2).sum(1)); s = (g >= GMIN) & (u[:, 0] > 0)
    Q = q[s].sum()
    if Q <= 0: print("%-28s t %.1f fs: no electrons above gamma %g" % (run, t, GMIN)); continue
    E = (g[s] - 1) * MEC2; Em = np.average(E, weights=q[s]); dE = np.sqrt(np.average((E - Em) ** 2, weights=q[s]))
    xm = np.average(x[s], weights=q[s])
    print("%-28s t %.1f fs: Q %.4g pC  E_mean %.1f MeV  dE/E %.3f  E_max %.0f MeV  x_mean %.1f um  N %d" %
          (run, t, Q, Em, dE / Em, E.max(), xm, s.sum()))
