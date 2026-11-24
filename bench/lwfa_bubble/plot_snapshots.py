"""2D snapshots of n_e or Ex in the plane perpendicular to the polarisation, VLPL slices vs FBPIC, at 400, 700, 900 fs:
plot_snapshots.py rip_dir fbpic_dir out.png ne|ex"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries
rip, fb, out, qty = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
lam, cl = 0.8e-6, 2.99792458e8
k0 = 2 * np.pi / lam; w0 = k0 * cl
E0 = 9.1093837015e-31 * cl * w0 / 1.602176634e-19
n_c = 8.8541878128e-12 * 9.1093837015e-31 * w0**2 / 1.602176634e-19**2
n0 = 0.005
meta = dict(l.split(None, 1) for l in open(os.path.join(rip, "slices", "meta.txt")))
nx, ny, dx = int(meta["nx"]), int(meta["ny"]), float(meta["dx"])
dy = float(meta.get("dy", 0.25e-6 * k0))
index = {int(l.split()[0]): l.split() for l in open(os.path.join(rip, "slices", "index.txt"))}
ts = OpenPMDTimeSeries(os.path.join(fb, "hdf5"), check_all_files=False)
steps = [2400, 4200, 5400]
fig, ax = plt.subplots(len(steps), 2, figsize=(12, 8.5))
for row, st in enumerate(steps):
    _, t_s, x0_s = index[st]
    x0 = float(x0_s) / k0 * 1e6
    name = "n_electrons" if qty == "ne" else "Ex"
    a = np.fromfile(os.path.join(rip, "slices", "%s_%06d.f32" % (name, st)), dtype=np.float32).reshape(ny, nx)
    a = a / n0 if qty == "ne" else a
    ylen = ny * dy / k0 * 1e6
    ext_r = [x0, x0 + nx * dx / k0 * 1e6, -ylen / 2, ylen / 2]
    if qty == "ne":
        f, info = ts.get_field("rho_electrons", None, iteration=st, m="all", theta=np.pi/2)
        f = -f / 1.602176634e-19 / n_c / n0
        kw = dict(cmap="gray_r", vmin=0, vmax=3)
    else:
        f, info = ts.get_field("E", "z", iteration=st, m="all", theta=np.pi/2)
        f = f / E0
        kw = dict(cmap="RdBu_r", vmin=-0.15, vmax=0.15)
    ext_f = [info.zmin * 1e6, info.zmax * 1e6, info.rmin * 1e6, info.rmax * 1e6]
    t_fs = float(t_s) / w0 * 1e15
    for col, (img, ext, lab) in enumerate(((a, ext_r, "VLPL (3D, plane ⊥ polarisation)"), (f, ext_f, "FBPIC (RZ, plane ⊥ polarisation)"))):
        im = ax[row, col].imshow(img, origin="lower", extent=ext, aspect="auto", interpolation="nearest", **kw)
        ax[row, col].set_xlim(ext_r[0], ext_r[1]); ax[row, col].set_ylim(-15, 15)
        ax[row, col].set_title("%s, t = %.0f fs" % (lab, t_fs), fontsize=10)
        ax[row, col].set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax[row, :].tolist(), shrink=0.9, label="n_e / n_0" if qty == "ne" else "E_x [m c ω0 / e]")
for a_ in ax[-1]: a_.set_xlabel("x [µm]")
fig.savefig(out, dpi=110)
print("wrote", out)
