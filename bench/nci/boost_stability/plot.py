#!/usr/bin/env python3
"""Snapshots of the nonlinear boosted-frame check (README.md here, run_nonlinear.sh).
  plot.py OUT lab GEO A0 K RUNS [LAB] -> lab-frame figure: lab run vs boosted gamma 5, 10 at lab snapshot K (fields E_y, E_x and
                                 the on-axis lines), and the numbers (on-axis E_x error, laser peak, noise ahead/behind)
  plot.py OUT boost RUN               -> boosted-frame figure of one boosted run (E_y', E_x', n_e' at three boosted steps)
Units: E in m c omega0 / e, lengths in um, n_e in n0 = 0.004 n_c (lambda0 = 1 um)."""
import sys, glob, os
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools"))
from lab_fields import read_snapshot
N0 = 0.004; UM = 1 / (2 * np.pi)          # 1/k0 in um
D = sys.argv[1]

def plane(a, rz, c1=1.0):                  # RZ: theta = 0 plane, mode 0 + c1 Re(mode 1); 2D: [0]
    return a[0] + c1 * a[1] if rz and a.shape[0] > 1 else a[0]   # openPMD stores 2 Re F_1 (c1 = 1), lab_fields F_1 (c1 = 2)

def opmd(path, comps):
    h = h5py.File(path); it = list(h["data"])[0]; g = h["data"][it]; m = g["meshes"]
    E = m["E"]; dx = E.attrs["gridSpacing"][-1]; x0 = E.attrs["gridGlobalOffset"][-1]
    rz = E.attrs["geometry"] == b"thetaMode"
    out = {}
    for c in comps:
        grp, sub = c.split("/") if "/" in c else (c, None)
        a = np.asarray(m[grp][sub] if sub else m[grp])
        out[c] = plane(a, rz)
    out["x"] = (x0 + dx * np.arange(a.shape[-1])) * UM
    out["d1"] = E.attrs["gridSpacing"][-2] * UM
    out["t"] = g.attrs["time"] * g.attrs.get("timeUnitSI", 1) if "time" in g.attrs else 0
    out["rz"] = rz
    return out

def lab_ref(geo, a0, K, lab=None):
    run = D + "/" + (lab or "%s_lab_a%s%s" % (geo, a0, "_plain" if geo == "rz" else ""))
    f = "%s/openpmd/data_%06d.h5" % (run, 1920 * (3 + 2 * K))
    lon, tr = ("E/z", "E/t") if geo == "rz" else ("E/x", "E/y")
    o = opmd(f, [lon, tr, "rho_electrons"])
    return o["x"], o["d1"], o[lon], o[tr], -o["rho_electrons"] / N0, o["rz"]

def lab_boost(run, K):
    run = D + "/" + run
    s = read_snapshot("%s/lab_fields/snap_%03d" % (run, K))
    rz = s["meta"]["geometry"] == "rz"
    f = s["fields"]; names = list(f)
    lon, tr = (names[0], names[2]) if rz else ("Ex", "Ey")     # RZ order: Ez Er Ephi; the laser (y-polarized) is E_phi in the theta = 0 plane
    return s["x"] * UM, float(s["meta"]["d1"]) * UM, plane(f[lon], rz, 2.0), plane(f[tr], rz, 2.0), rz, s["meta"]

def env(x, line):                           # laser envelope from the transverse field (Hilbert transform)
    n = len(line); F = np.fft.fft(line); F[n // 2 + 1:] = 0; F[1:n // 2] *= 2
    return np.abs(np.fft.ifft(F))

def lab_fig(geo, a0, K, runs, lab=None):
    xl, d1, ELl, ETl, nel, rz = lab_ref(geo, a0, K, lab)
    rows = [("lab", xl, ELl, ETl)]
    meta = None
    for rn in runs:
        try:
            x, d, EL, ET, _, meta = lab_boost(rn, K); rows.append((rn, x, EL, ET))
        except Exception as e:
            print("missing", rn, e)
    j0 = 0 if rz else ELl.shape[0] // 2
    fig, ax = plt.subplots(len(rows) + 1, 3, figsize=(15, 3.2 * (len(rows) + 1)), squeeze=False)
    ext = lambda x, n: [x[0], x[-1], 0 if rz else -n * d1 / 2, n * d1 if rz else n * d1 / 2]
    lw = np.abs(ETl).max(); lz = np.abs(ELl).max()
    for i, (lab, x, EL, ET) in enumerate(rows):
        ax[i, 0].imshow(ET, origin="lower", aspect="auto", extent=ext(x, ET.shape[0]), cmap="RdBu_r", vmin=-lw, vmax=lw)
        ax[i, 0].set_title("%s: E_%s (laser)" % (lab, "phi" if rz else "y"))
        ax[i, 1].imshow(EL, origin="lower", aspect="auto", extent=ext(x, EL.shape[0]), cmap="RdBu_r", vmin=-lz, vmax=lz)
        ax[i, 1].set_title("%s: E_%s (wake)" % (lab, "z" if rz else "x"))
        ax[i, 2].imshow(np.log10(np.abs(EL) + 1e-6), origin="lower", aspect="auto", extent=ext(x, EL.shape[0]), vmin=-5, vmax=np.log10(lz), cmap="viridis")
        ax[i, 2].set_title("%s: log10 |E_%s| (noise)" % (lab, "z" if rz else "x"))
    a = ax[-1, 0]; b = ax[-1, 1]; c = ax[-1, 2]
    c.imshow(nel, origin="lower", aspect="auto", extent=ext(xl, nel.shape[0]), vmin=0, vmax=3, cmap="Greys")
    c.set_title("lab run: n_e / n0")
    res = []
    for lab, x, EL, ET in rows:
        jj = 0 if rz else EL.shape[0] // 2
        a.plot(x, EL[jj], lw=1, label=lab)
        jt = 1 if rz else jj                 # RZ: E_phi of m = 1 vanishes in the axis row
        b.plot(x, env(x, ET[jt]), lw=1, label=lab)
        # numbers: on-axis E_x vs lab (interpolated onto the lab grid), laser peak, noise ahead of the laser
        Li = np.interp(xl, x, EL[jj], left=np.nan, right=np.nan)
        ok = ~np.isnan(Li)
        e = env(x, ET[jt]); xp = x[np.argmax(e)]
        ahead = x > xp + 6
        nz = np.sqrt(np.mean(EL[:, ahead] ** 2)) if ahead.any() else np.nan
        w = ok & (xl > xp - 30) & (xl < xp + 8); wf = ok & (xl > xp - 12) & (xl < xp + 8)   # whole bubble; its front half
        rd = lambda m: np.sqrt(np.mean((Li - ELl[j0])[m] ** 2) / np.mean(ELl[j0][m] ** 2))
        res.append((lab, rd(w), rd(wf), e.max(), xp, nz, np.abs(EL[jj]).max()))
    a.set_title("on-axis E_%s" % ("z" if rz else "x")); a.legend(fontsize=7)
    b.set_title("laser envelope on axis"); b.legend(fontsize=7)
    for q in ax.flat:
        q.set_xlabel("x [um]" if not rz else "z [um]"); q.set_xlim(xl[0], xl[-1])
    t = meta["t_lab_fs"] if meta else "?"
    fig.suptitle("%s a0 = %s, lab time %s fs" % (geo.upper(), a0, t))
    fig.tight_layout()
    out = "%s/fig_lab_%s_a%s_%d_%s.png" % (D, geo, a0, K, runs[-1]); fig.savefig(out, dpi=80)
    print(out, "t_lab", t)
    for r in res: print("  %-18s rms(dE_ax)/rms(E_lab) %.3f (front %.3f)  laser peak %.3f at %.1f um  rms E_long ahead %.2e  max|E_long| %.3f" % r)

def boost_fig(run):
    fs = sorted(glob.glob("%s/%s/openpmd/data_*.h5" % (D, run)))
    pick = [fs[len(fs) // 3], fs[2 * len(fs) // 3], fs[-1]]
    fig, ax = plt.subplots(3, 3, figsize=(15, 9.6))
    for i, f in enumerate(pick):
        h = h5py.File(f); rz = h["data"][list(h["data"])[0]]["meshes"]["E"].attrs["geometry"] == b"thetaMode"
        lon, tr = ("E/z", "E/t") if rz else ("E/x", "E/y")
        o = opmd(f, [lon, tr, "rho_electrons"]); x = o["x"]; n = o[lon].shape[0]
        ex = [x[0], x[-1], 0 if rz else -n * o["d1"] / 2, n * o["d1"] if rz else n * o["d1"] / 2]
        for k, (q, cm) in enumerate(((o[tr], "RdBu_r"), (o[lon], "RdBu_r"))):
            v = np.abs(q).max(); ax[i, k].imshow(q, origin="lower", aspect="auto", extent=ex, cmap=cm, vmin=-v, vmax=v)
        ne = -o["rho_electrons"] / N0
        ax[i, 2].imshow(ne, origin="lower", aspect="auto", extent=ex, cmap="Greys", vmin=0, vmax=np.percentile(ne, 99.5))
        st = os.path.basename(f)[5:11]
        ax[i, 0].set_title("step %s: E'_%s" % (st, "r" if rz else "y")); ax[i, 1].set_title("E'_%s" % ("z" if rz else "x"))
        ax[i, 2].set_title("n'_e / n0 (boosted frame)")
    for q in ax.flat: q.set_xlabel("x' [um]")
    fig.suptitle(run + " (boosted frame)"); fig.tight_layout()
    out = "%s/fig_boost_%s.png" % (D, run); fig.savefig(out, dpi=80); print(out)

if __name__ == "__main__":
    a = sys.argv[2:]
    if a[0] == "lab": lab_fig(a[1], a[2], int(a[3]), a[4].split(","), a[5] if len(a) > 5 else None)
    else: boost_fig(a[1])
