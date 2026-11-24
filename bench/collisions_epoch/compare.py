#!/usr/bin/env python3
"""Solid-density electron-proton temperature relaxation: VLPL, Smilei 5.1, EPOCH 4.20 and NRL (bench/collisions_epoch/README.md).

  . $WORK/smilei/smilei_env.sh; python3 compare.py

VLPL: out_vlpl_solid[_s5]/collisions.txt; Smilei: out_smilei_solid/ParticleBinning*.h5 (frozen species, as VLPL); EPOCH:
out_epoch_{nanbu,sk,nocoll}/*.sdf (full PIC; the temperature per species and cell, read with the sdf module of EPOCH's
SDF/utilities if it is installed). nocoll: the heating by the PIC noise alone, subtracted from the EPOCH runs ("corrected").
NRL: dT_e/dt = -nu (T_e - T_i), nu = (8 sqrt(2 pi) / 3) n Z^2 e^4 lnL sqrt(m_e m_i) / ((4 pi eps0)^2 (m_e T_i + m_i T_e)^1.5).
"""
import glob
import math
import os
import numpy as np
import h5py

MC2 = 510998.95
KRE = 2 * math.pi * 2.8179403262e-15 / 1e-6
N, LNL, MI = 100.0, 5.0, 1836.15267
OMEGA = 2 * math.pi * 299792458.0 / 1e-6              # 1 / s


def vlpl(d):
    f = d + "/collisions.txt"
    head = [l for l in open(f) if l.startswith("#")][0].split()[1:]
    head = head[:head.index("(pairs,")]
    x = np.loadtxt(f)
    D = {k: x[:, i] for i, k in enumerate(head)}
    T = lambda s: MC2 * (D["Tx_" + s] + D["Ty_" + s] + D["Tz_" + s]) / 3
    return D["t"], T("electron"), T("proton")


def smilei(d):
    out = []
    for j, m in enumerate([1.0, MI]):
        mom = []
        for q in range(7):
            with h5py.File("%s/ParticleBinning%d.h5" % (d, 7 * j + q)) as h:
                keys = sorted(k for k in h.keys() if k.startswith("timestep"))
                mom.append(np.array([h[k][()].sum() for k in keys]))
                steps = np.array([int(k[8:]) for k in keys])
        w, px, vxpx, vypy, vzpz, ek, ch = mom
        g = 1 + ek / (w * m)
        out.append(MC2 * ((vxpx / w - m * (px / (w * m)) ** 2 / g) + vypy / w + vzpz / w) / 3)
    return steps * 0.02, out[0], out[1]


def epoch(d):
    try:
        import sdf
    except ImportError:
        return None
    t, te, ti = [], [], []
    for f in sorted(glob.glob(d + "/*.sdf")):
        s = sdf.read(f)
        t.append(s.Header["time"] * OMEGA)
        te.append(np.mean(s.Derived_Temperature_electron.data) * 8.617333262e-5)   # K -> eV
        ti.append(np.mean(s.Derived_Temperature_proton.data) * 8.617333262e-5)
    return np.array(t), np.array(te), np.array(ti)


def nrl(t, te, ti):
    y, T, res = [te / MC2, ti / MC2], [0.0], [[te, ti]]
    h = t[-1] / 20000
    for k in range(20000):
        r = (2.0 / 3.0) * math.sqrt(2 / math.pi) * KRE * N * LNL * math.sqrt(MI) / (y[1] + MI * y[0]) ** 1.5
        dd = r * (y[0] - y[1]) * h
        y = [y[0] - dd, y[1] + dd]
        T.append((k + 1) * h); res.append([MC2 * y[0], MC2 * y[1]])
    res = np.array(res)
    return np.interp(t, T, res[:, 0]), np.interp(t, T, res[:, 1])


runs = {}
for name, d, fn in [("VLPL", "out_vlpl_solid", vlpl), ("VLPL smilei5", "out_vlpl_solid_s5", vlpl), ("VLPL no cap", "out_vlpl_solid_nocap", vlpl), ("VLPL dt/4", "out_vlpl_solid_dt4", vlpl), ("VLPL nocap dt/4", "out_vlpl_solid_nocap_dt4", vlpl),
                    ("VLPL auto lnL", "out_vlpl_auto", vlpl), ("VLPL auto dt/4", "out_vlpl_auto_dt4", vlpl),
                    ("VLPL auto nocap", "out_vlpl_auto_nocap", vlpl), ("VLPL auto nocap dt/4", "out_vlpl_auto_nocap_dt4", vlpl),
                    ("VLPL auto h", "out_vlpl_autoh", vlpl), ("VLPL auto h dt/4", "out_vlpl_autoh_dt4", vlpl),
                    ("VLPL auto h nocap", "out_vlpl_autoh_nocap", vlpl), ("VLPL auto h nocap dt/4", "out_vlpl_autoh_nocap_dt4", vlpl),
                    ("Smilei 5.1", "out_smilei_solid", smilei),
                    ("EPOCH Nanbu", "out_epoch_nanbu", epoch), ("EPOCH Nanbu dt/4", "out_epoch_nanbu_dt4", epoch),("EPOCH Sentoku-Kemp", "out_epoch_sk", epoch), ("EPOCH no collisions", "out_epoch_nocoll", epoch)]:
    if os.path.isdir(d):
        r = fn(d)
        if r is not None: runs[name] = r
t0, te0, ti0 = runs["VLPL"]
tt = np.linspace(0, t0[-1], 6)
a, b = nrl(tt, te0[0], ti0[0])
print("T_e - T_i [eV] (initial %.2f eV; t in 1/omega, 942 = 0.5 ps)" % (te0[0] - ti0[0]))
names = list(runs)
print("  t       " + "  ".join("%12s" % n[:12] for n in names) + "         NRL")
heat = None
if "EPOCH no collisions" in runs:
    tn, en, inn = runs["EPOCH no collisions"]
    heat = (lambda t: np.interp(t, tn, en - en[0]), lambda t: np.interp(t, tn, inn - inn[0]))
for i, t in enumerate(tt):
    row = []
    for n in names:
        tv, e, p = runs[n]
        row.append(np.interp(t, tv, e) - np.interp(t, tv, p))
    print("  %6.0f  " % t + "  ".join("%12.3f" % x for x in row) + "  %10.3f" % (a[i] - b[i]))
print("time of (T_e - T_i) = 1/e of the initial (1/omega):")
for n in names + ["NRL"]:
    if n == "NRL":
        tv = np.linspace(0, t0[-1], 2001); e, p = nrl(tv, te0[0], ti0[0])
    else:
        tv, e, p = runs[n]
    d = (e - p) / (e[0] - p[0])
    k = np.argmax(d < math.exp(-1))
    print("  %-20s %s" % (n, "%.1f" % np.interp(math.exp(-1), [d[k], d[k - 1]], [tv[k], tv[k - 1]]) if d.min() < math.exp(-1) else "not reached"))
if heat is not None:
    print("EPOCH: the PIC noise alone heats electrons by %.2f eV and protons by %.2f eV in 0.5 ps (no collisions)" % (heat[0](t0[-1]), heat[1](t0[-1])))
