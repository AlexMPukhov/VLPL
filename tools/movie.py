#!/usr/bin/env python3
"""Movie of the x-y slices written by VLPL (output.slice_every > 0).

  python3 tools/movie.py out_reflect                       # Ez and n_electrons -> out_reflect/movie.mp4
  python3 tools/movie.py out_reflect --fields Ey Bz --fps 20 --out film.mp4

Field panels use a symmetric colour scale (max over all frames, clipped at --clip of it); density
panels (n_<species>) are shown in units of n_c with a contour at the critical density.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--fields", nargs="+", default=["Ez", "n_electrons"])
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--clip", type=float, default=0.8)
ap.add_argument("--out", default=None)
a = ap.parse_args()

sd = os.path.join(a.dir, "slices")
meta = {}
for line in open(os.path.join(sd, "meta.txt")):
    k, *v = line.split()
    meta[k] = v
nx, ny = int(meta["nx"][0]), int(meta["ny"][0])
dx, dy = float(meta["dx"][0]), float(meta["dy"][0])
lam = float(meta["lambda0_m"][0])
um = lam / (2 * np.pi) * 1e6                  # 1/k0 in um
fs = lam / (2 * np.pi * 2.99792458e8) * 1e15  # 1/omega0 in fs
frames = [(int(s), float(t)) for s, t in (l.split()[:2] for l in open(os.path.join(sd, "index.txt")))]

def load(name, step):
    return np.fromfile(os.path.join(sd, f"{name}_{step:06d}.f32"), dtype=np.float32).reshape(ny, nx)

scale = {}
for f in a.fields:
    if f.startswith("n_"):
        scale[f] = max(np.abs(load(f, s)).max() for s, _ in frames[:1]) * 2.0
    else:
        scale[f] = a.clip * max(np.abs(load(f, s)).max() for s, _ in frames)
ext = [0, nx * dx * um, 0, ny * dy * um]
dens = next((f for f in a.fields if f.startswith("n_")), None)

nf = len(a.fields)
fig, axs = plt.subplots(1, nf, figsize=(5.2 * nf, 4.8), constrained_layout=True, squeeze=False)
axs = axs[0]
ims, conts = [], [None] * nf
for ax, f in zip(axs, a.fields):
    d = load(f, frames[0][0])
    if f.startswith("n_"):
        im = ax.imshow(d, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=scale[f], aspect="equal")
        fig.colorbar(im, ax=ax, label=f"{f[2:]} density / $n_c$", shrink=0.85)
    else:
        im = ax.imshow(d, origin="lower", extent=ext, cmap="RdBu_r", vmin=-scale[f], vmax=scale[f], aspect="equal")
        fig.colorbar(im, ax=ax, label=f"{f}  [$m c\\omega_0/e$]", shrink=0.85)
    ax.set_xlabel("x [µm]"); ax.set_ylabel("y [µm]")
    ims.append(im)
title = fig.suptitle("")
xg = (np.arange(nx) + 0.5) * dx * um
yg = (np.arange(ny) + 0.5) * dy * um

out = a.out or os.path.join(a.dir, "movie.mp4")
w = FFMpegWriter(fps=a.fps, bitrate=4000)
with w.saving(fig, out, dpi=110):
    for s, t in frames:
        nd = load(dens, s) if dens else None
        for i, (ax, f) in enumerate(zip(axs, a.fields)):
            ims[i].set_data(load(f, s) if f != dens else nd)
            if conts[i] is not None:
                conts[i].remove()
            conts[i] = None
            if nd is not None and not f.startswith("n_") and nd.max() > 1:
                conts[i] = ax.contour(xg, yg, nd, levels=[1.0], colors="k", linewidths=0.8)
        title.set_text(f"t = {t * fs:6.1f} fs   (step {s})")
        w.grab_frame()
print("wrote", out, len(frames), "frames")
