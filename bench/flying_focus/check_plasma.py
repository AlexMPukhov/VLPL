"""Flying focus in a preformed plasma: VLPL (bench/flying_focus/rip_plasma.deck, side tube with the cold-fluid shell)
against GEM-PIC-RZ (gem_plasma.deck) on the axis, every 30 um of propagation:
- the laser: fitted amplitude / a0 and carrier phase against the luminal pattern a0 exp(-zeta'^2/L^2)
  cos(cos(theta) k0 zeta' - kappa omega0 t), which holds in the plasma too (every component has w - k_x = kappa);
- the wake E_x (RIP; the mean of the two rows next to the axis) / E_z (GEM), smoothed over a laser period: its mean over
  zeta' = -6 +- 1, -16 +- 1 and -30 +- 1 um, the rms difference between the codes over -40 .. 10 um relative to the
  largest |E| of VLPL there (the on-axis density is not compared: with 4 particles per cell it is noise in both).
usage: python3 check_plasma.py rip_out gem_out [a0 theta[rad] tau_fwhm[fs] x_center[um] axis_j lambda_um]"""
import glob
import os
import re
import sys
import numpy as np

rip, gem = sys.argv[1], sys.argv[2]
a0, th, tau, xc, ja, lam = 2.0, 0.2, 30.0, 42.0, 40, 0.8
if len(sys.argv) > 3:
    a0, th, tau, xc = map(float, sys.argv[3:7])
    ja, lam = int(sys.argv[7]), float(sys.argv[8])
k0 = 2 * np.pi / lam
L = tau * 0.299792458 / np.sqrt(2 * np.log(2))    # um, 1/e half length of the field
kap = 1 - np.cos(th)
grid = np.arange(-40, 10, 0.05)


def metrics(zeta, laser, wake, z_um, fixed_z=False):
    # VLPL slices are at fixed t (carrier k_x = cos(theta) k0 along zeta), GEM-PIC-RZ axis files at fixed z (carrier
    # omega0 along zeta: k_x zeta - kappa t = k0 zeta - kappa z at fixed z)
    env = a0 * np.exp(-(zeta / L) ** 2)
    ph = (k0 if fixed_z else np.cos(th) * k0) * zeta - kap * k0 * z_um
    m = np.abs(zeta) < 2 * L
    M = np.vstack([env[m] * np.cos(ph[m]), env[m] * np.sin(ph[m])]).T
    (a, b), *_ = np.linalg.lstsq(M, laser[m], rcond=None)
    k = int(round(lam / (zeta[1] - zeta[0])))
    ws = np.convolve(wake, np.ones(k) / k, mode="same")
    pts = [ws[np.abs(zeta - p) < 1].mean() for p in (-6, -16, -30)]
    return [np.hypot(a, b), np.arctan2(-b, a)] + pts, np.interp(grid, zeta, ws)


rows = {}
meta = dict(l.split(None, 1) for l in open(os.path.join(rip, "slices", "meta.txt")) if l.strip() and not l.startswith("#"))
nx, ny = int(meta["nx"].split()[0]), int(meta["ny"].split()[0])
dx = float(meta["dx"].split()[0]) / k0 if "dx" in meta else float(meta["h"].split()[0]) / k0
for line in open(os.path.join(rip, "slices", "index.txt")):
    if line.startswith("#") or not line.strip():
        continue
    w = line.split()
    step, t = int(w[0]), float(w[1])
    rd = lambda n: np.fromfile(os.path.join(rip, "slices", "%s_%06d.f32" % (n, step)), dtype=np.float32).reshape(ny, nx).astype(float)
    ey, ex = rd("Ey")[ja], 0.5 * (rd("Ex")[ja] + rd("Ex")[ja + 1])
    zeta = np.arange(nx) * dx - xc
    z = t / k0                                         # um (c = 1)
    rows.setdefault(round(z / 30) * 30, {})["rip"] = metrics(zeta, ey, ex, z)
for f in sorted(glob.glob(os.path.join(gem, "axis_*.txt"))):
    z = float(re.search(r"z = ([0-9.eE+-]+) mm", open(f).readline()).group(1)) * 1e3
    A = np.loadtxt(f)
    rows.setdefault(round(z / 30) * 30, {})["gem"] = metrics(A[:, 0] - xc, A[:, 1], A[:, 2], z, True)
print("# z[um] code  laser amp/a0  phase[rad] | wake E (m c w0/e) at zeta' = -6, -16, -30 um"
      " | rms(wake RIP - GEM) / max|wake RIP| over -40..10 um")
for z in sorted(rows):
    for c in ("rip", "gem"):
        if c in rows[z]:
            r = rows[z][c][0]
            d = ""
            if c == "gem" and "rip" in rows[z]:
                wr, wg = rows[z]["rip"][1], rows[z]["gem"][1]
                d = f"{np.sqrt(np.mean((wr - wg) ** 2)) / np.abs(wr).max():7.3f}"
            print(f"{z:6.0f} {c}  {r[0]:7.4f} {r[1]:+8.4f} | {r[2]:+9.5f} {r[3]:+9.5f} {r[4]:+9.5f} | {d}")
