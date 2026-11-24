#!/usr/bin/env python3
"""VLPL against Smilei 5.1: binary collisions and impact ionization (bench/collisions_smilei/README.md).

  . $WORK/smilei/smilei_env.sh; python3 compare.py [cases]

Reads out_vlpl_<case>/collisions.txt and out_smilei_<case>/ParticleBinning*.h5 (smilei_coll.py: per species the moments
weight, px, vx px, vy py, vz pz, ekin, charge summed over the box), prints both on a common time grid and the largest
difference, plus the NRL rates where they apply (thermal: the e-i energy exchange; iso: the temperature isotropy).
Temperatures in eV (m_e c^2 = 511 keV), T_a = m <(u_a - <u_a>)^2 / gamma>.
"""
import math
import os
import sys
import numpy as np
import h5py

MC2 = 510998.95
KRE = 2 * math.pi * 2.8179403262e-15 / 1e-6          # k0 r_e at lambda0 = 1 um
CASES = sys.argv[1:] or ["thermal", "iso", "beam", "stop", "ion"]
SPECIES = {"thermal": ["electron", "ion"], "iso": ["electron"], "beam": ["electron", "ion"], "stop": ["beam0", "beam1", "beam2"],
           "ion": ["electron", "ion"]}
MASS = {"electron": 1.0, "ion": {"thermal": 10.0, "beam": 10.0, "ion": 1836.0 * 13}, "beam0": 1.0, "beam1": 1.0, "beam2": 1.0}
DT = {"thermal": 0.2, "iso": 0.01, "beam": 0.2, "stop": 1.0, "ion": 0.05}   # in L0 = 2 pi / omega


def vlpl(case):
    f = "out_vlpl_%s/collisions.txt" % case
    head = [l for l in open(f) if l.startswith("#")][0].split()[1:]
    head = head[:head.index("(pairs,")] if "(pairs," in head else head
    d = np.atleast_2d(np.loadtxt(f))
    return {k: d[:, i] for i, k in enumerate(head)}


def smilei(case):
    out = {}
    for j, nm in enumerate(SPECIES[case]):
        mom = []
        for q in range(7):
            with h5py.File("out_smilei_%s/ParticleBinning%d.h5" % (case, 7 * j + q)) as h:
                keys = sorted(k for k in h.keys() if k.startswith("timestep"))
                mom.append(np.array([h[k][()].sum() for k in keys]))
                steps = np.array([int(k[8:]) for k in keys])
        w, px, vxpx, vypy, vzpz, ek, ch = mom
        m = MASS[nm] if not isinstance(MASS[nm], dict) else MASS[nm][case]
        g = 1 + ek / (w * m)                            # Smilei: ekin = m (gamma - 1), p = m u (both in m_e c units)
        out["t"] = steps * DT[case] * 2 * math.pi
        out["<ux>_" + nm] = px / (w * m)
        out["Tx_" + nm] = vxpx / w - m * (px / (w * m)) ** 2 / g
        out["Ty_" + nm] = vypy / w
        out["Tz_" + nm] = vzpz / w
        out["<Ekin>_" + nm] = ek / w
        out["<Z>_" + nm] = ch / w
    return out


def at(t, tv, v):
    return np.interp(t, tv, v)


def nrl_thermal(t, te, ti, n=10.0, mi=10.0, lnL=5.0):
    y = [te, ti]
    T = [0.0]
    res = [list(y)]
    h = (t[-1]) / 4000
    for k in range(4000):
        r = (2.0 / 3.0) * math.sqrt(2 / math.pi) * KRE * n * lnL * math.sqrt(mi) / (y[1] + mi * y[0]) ** 1.5
        d = r * (y[0] - y[1]) * h
        y = [y[0] - d, y[1] + d]
        T.append((k + 1) * h); res.append(list(y))
    res = np.array(res)
    return np.interp(t, T, res[:, 0]), np.interp(t, T, res[:, 1])


def nrl_iso(t, tpar, tperp, n=10.0, lnL=2.0):
    y = [tpar, tperp]
    T = [0.0]
    res = [list(y)]
    h = t[-1] / 4000
    for k in range(4000):
        A = y[1] / y[0] - 1
        if abs(A) < 1e-6: f = 4.0 / 15
        elif A > 0: f = (-3 + (A + 3) * math.atan(math.sqrt(A)) / math.sqrt(A)) / A ** 2
        else: f = (-3 + (A + 3) * math.atanh(math.sqrt(-A)) / math.sqrt(-A)) / A ** 2
        nu = KRE * n * lnL / (2 * math.sqrt(math.pi) * y[0] ** 1.5) * f
        y = [y[0] + 2 * nu * (y[1] - y[0]) * h, y[1] - nu * (y[1] - y[0]) * h]
        T.append((k + 1) * h); res.append(list(y))
    res = np.array(res)
    return np.interp(t, T, res[:, 0]), np.interp(t, T, res[:, 1])


def key(case, D):
    """the quantity compared in the time table: thermal T_e - T_i [eV], iso (T_x - T_perp) / initial, beam <u_x> of the electrons"""
    if case == "thermal":
        return MC2 * ((D["Tx_electron"] + D["Ty_electron"] + D["Tz_electron"]) - (D["Tx_ion"] + D["Ty_ion"] + D["Tz_ion"])) / 3
    if case == "iso":
        d = D["Tx_electron"] - (D["Ty_electron"] + D["Tz_electron"]) / 2
        return d / d[0]
    return D["<ux>_electron"]


for case in CASES:
    V, S = vlpl(case), smilei(case)
    V5 = vlpl(case + "_s5") if os.path.exists("out_vlpl_%s_s5/collisions.txt" % case) else None
    tv, ts = V["t"], S["t"]
    tt = np.linspace(0, min(tv[-1], ts[-1]), 6)
    print("=== %s" % case)
    if case in ("thermal", "iso", "beam"):
        kv, ks = key(case, V), key(case, S)
        k5 = key(case, V5) if V5 is not None else None
        if case == "thermal":
            T0 = [(V["Tx_" + n][0] + V["Ty_" + n][0] + V["Tz_" + n][0]) / 3 for n in ("electron", "ion")]
            a, b = nrl_thermal(tt, T0[0], T0[1])
            th, label = MC2 * (a - b), "T_e - T_i [eV]"
        elif case == "iso":
            a, b = nrl_iso(tt, V["Tx_electron"][0], (V["Ty_electron"][0] + V["Tz_electron"][0]) / 2)
            th, label = (a - b) / (a[0] - b[0]), "(T_x - T_perp) / initial"
        else:
            th, label = None, "<u_x> of the beam electrons"
        print("  %s: t[1/w]  VLPL (Nanbu)  VLPL (smilei5)  Smilei 5.1%s" % (label, "  NRL" if th is not None else ""))
        for i, t in enumerate(tt):
            print("  %8.3f  %10.4f  %10s  %10.4f%s" % (t, at(t, tv, kv), "%.4f" % at(t, V5["t"], k5) if k5 is not None else "-", at(t, ts, ks),
                  "  %10.4f" % th[i] if th is not None else ""))
        ref = ks[0] if case != "thermal" else ks[0]
        print("  max |VLPL - Smilei| / initial: Nanbu %.4f%s" % (np.abs(np.interp(ts, tv, kv) - ks).max() / abs(ref),
              ", smilei5 %.4f" % (np.abs(np.interp(ts, V5["t"], k5) - ks).max() / abs(ref)) if k5 is not None else ""))
        if case == "beam":
            print("  <v_perp^2>^1/2 at the end: VLPL %.5f, Smilei %.5f; <u_x> of the ions: VLPL %.3e, Smilei %.3e" % (
                  math.sqrt(V["Ty_electron"][-1] + V["Tz_electron"][-1]), math.sqrt(np.interp(tv[-1], ts, S["Ty_electron"] + S["Tz_electron"])),
                  V["<ux>_ion"][-1], np.interp(tv[-1], ts, S["<ux>_ion"])))
    elif case == "stop":
        print("  t[1/w]   <Ekin> [keV] of the beams: VLPL / Smilei (v0 = 0.2, 0.27, 0.33)")
        for t in tt:
            print("  %7.1f  " % t + "  ".join("%7.3f / %7.3f" % (MC2 / 1e3 * at(t, tv, V["<Ekin>_beam%d" % k]), MC2 / 1e3 * at(t, ts, S["<Ekin>_beam%d" % k]))
                                             for k in range(3)))
        for k in range(3):
            lv = V["<Ekin>_beam%d" % k][0] - V["<Ekin>_beam%d" % k][-1]
            ls = S["<Ekin>_beam%d" % k][0] - np.interp(tv[-1], ts, S["<Ekin>_beam%d" % k])
            l5 = (V5["<Ekin>_beam%d" % k][0] - V5["<Ekin>_beam%d" % k][-1]) if V5 is not None else float("nan")
            print("  beam%d: energy lost VLPL %.4f keV (smilei5 %.4f), Smilei %.4f keV (ratios %.4f, %.4f)" % (k, MC2 / 1e3 * lv, MC2 / 1e3 * l5,
                  MC2 / 1e3 * ls, lv / ls, l5 / ls))
            # NRL test-particle drag on the Maxwellian background (theta = 0.01, n = 10, lnL = 3): d eps / dt = -2 (psi - psi') nu0 eps
            # with psi(x) = erf(sqrt x) - 2 sqrt(x / pi) e^-x, x = v^2 / (2 theta); the cold-target rate relativistic: 2 nu0 eps -> n k0 r_e lnL / beta
            v = [0.2, 0.27, 0.33][k]; x = v * v / 0.02
            psi = math.erf(math.sqrt(x)) - 2 * math.sqrt(x / math.pi) * math.exp(-x); dpsi = 2 * math.sqrt(x / math.pi) * math.exp(-x)
            lt = 10 * KRE * 3 / v * (psi - dpsi) * tv[-1]
            print("         NRL drag %.4f keV: VLPL / NRL %.4f, Smilei / NRL %.4f" % (MC2 / 1e3 * lt, lv / lt, ls / lt))
    elif case == "ion":
        z = [k for k in V if k.startswith("<Z>_")][0]
        print("  t[1/w]   <Z> VLPL   Smilei")
        for t in tt:
            print("  %7.1f  %8.5f  %8.5f" % (t, at(t, tv, V[z]), at(t, ts, S["<Z>_ion"])))
        dzv, dzs = V[z][-1] - V[z][0], np.interp(tv[-1], ts, S["<Z>_ion"]) - S["<Z>_ion"][0]
        print("  ionized per ion at the end: VLPL %.5f, Smilei %.5f (ratio %.4f)" % (dzv, dzs, dzv / dzs))
