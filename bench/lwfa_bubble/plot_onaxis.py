"""On-axis Ex and n_e of VLPL and FBPIC at 400, 700, 900 fs (bench/lwfa_bubble): plot_onaxis.py rip_dir fbpic_dir out.png"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries

rip, fb, out = sys.argv[1], sys.argv[2], sys.argv[3]
lam, cl = 0.8e-6, 2.99792458e8
k0 = 2 * np.pi / lam; w0 = k0 * cl
E0 = 9.1093837015e-31 * cl * w0 / 1.602176634e-19
n_c = 8.8541878128e-12 * 9.1093837015e-31 * w0**2 / 1.602176634e-19**2
n0 = 0.005
meta = dict(l.split(None, 1) for l in open(os.path.join(rip, "slices", "meta.txt")))
nx, ny, dx = int(meta["nx"]), int(meta["ny"]), float(meta["dx"])
index = {int(l.split()[0]): l.split() for l in open(os.path.join(rip, "slices", "index.txt"))}
ts = OpenPMDTimeSeries(os.path.join(fb, "hdf5"), check_all_files=False)

def rip_axis(name, step):
    a = np.fromfile(os.path.join(rip, "slices", "%s_%06d.f32" % (name, step)), dtype=np.float32).reshape(ny, nx)
    return a[ny // 2].astype(float)

def fb_axis(field, comp, it):
    f, info = ts.get_field(field, comp, iteration=it, m="all", theta=0, slice_across="r")
    return info.z, f

steps = [2400, 4200, 5400]
fig, ax = plt.subplots(len(steps), 2, figsize=(11, 8), sharex="row")
for row, st in enumerate(steps):
    _, t_s, x0_s = index[st]
    x = (float(x0_s) + dx * np.arange(nx)) / k0
    zf, exf = fb_axis("E", "z", st)
    _, rhof = fb_axis("rho_electrons", None, st)
    t_fs = float(t_s) / w0 * 1e15
    a = ax[row, 0]
    a.plot(x * 1e6, rip_axis("Ex", st), lw=0.8, label="VLPL")
    a.plot(zf * 1e6, exf / E0, lw=0.8, ls="--", label="FBPIC")
    a.set_ylabel("E_x [m c ω0 / e]"); a.set_title("on-axis E_x, t = %.0f fs" % t_fs, fontsize=10)
    a = ax[row, 1]
    a.plot(x * 1e6, rip_axis("n_electrons", st) / n0, lw=0.8, label="VLPL")
    a.plot(zf * 1e6, -rhof / 1.602176634e-19 / n_c / n0, lw=0.8, ls="--", label="FBPIC")
    a.set_ylabel("n_e / n_0"); a.set_ylim(0, 6); a.set_title("on-axis n_e, t = %.0f fs" % t_fs, fontsize=10)
for a in ax[-1]: a.set_xlabel("x [µm]")
ax[0, 0].legend(fontsize=8); ax[0, 1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(out, dpi=130)
print("wrote", out)
