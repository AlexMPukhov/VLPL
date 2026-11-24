"""Figs.: energy and Gauss law vs time for the box iRIP in VLPL, Cartesian 3D (energy3d) and RZ (energyrz).
Data: runs/series.cpp (built against VLPL) -> ../data/series_{cart_random,cart_hot,rz_random,rz_hot}.txt."""
import numpy as np, matplotlib, sys
matplotlib.use("pgf" if "pgf" in sys.argv else "Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "pgf.rcfonts": False, "axes.linewidth": 0.6})
EXT = "pgf" if "pgf" in sys.argv else "pdf"
D = "../data/series_"
fl = 1e-17
col = ["#2a78d6", "#eb6834", "#1baf7a"]


def clean(a):
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)


# Cartesian 3D
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
r = np.loadtxt(D + "cart_random.txt")
a = ax[0]
a.semilogy(r[1:, 0], np.maximum(np.abs(r[1:, 1]), fl), color=col[0], lw=1, label=r"$|W/W_0-1|$")
a.semilogy(r[1:, 0], np.maximum(r[1:, 2], fl), color=col[1], lw=1, label=r"Gauss: $\max|\Delta G|/\max|G_0|$")
a.semilogy(r[1:, 0], np.maximum(np.abs(r[1:, 3]), fl), color=col[2], lw=1, label=r"div $B$ change")
a.set_ylim(1e-17, 1e-9); a.set_xlabel("step"); a.set_title(r"random vacuum fields, $c\tau/h_\perp = 10$", fontsize=8)
a.legend(frameon=False, fontsize=6.5, loc="upper left"); clean(a)
h = np.loadtxt(D + "cart_hot.txt")
a = ax[1]
W0 = h[0, 1] + h[0, 2]
a.semilogy(h[:, 0], h[:, 1] / W0, color=col[0], lw=1, label=r"$W_{\rm field}/W_{\rm tot}(0)$")
a.semilogy(h[:, 0], np.maximum(np.abs(h[:, 3]), fl), color=col[2], lw=1, label=r"$|W_{\rm tot}/W_{\rm tot}(0)-1|$")
a.semilogy(h[1:, 0], np.maximum(h[1:, 4], fl), color=col[1], lw=1, label=r"Gauss: $\max|\Delta(G-\rho)|$")
a.set_xlabel("step"); a.set_title(r"hot plasma ($u_{\rm th}=0.3$), $c\tau/h_\perp = 10$", fontsize=8)
a.set_ylim(1e-17, 1); a.legend(frameon=False, fontsize=6.5, loc="center right"); clean(a)
fig.tight_layout(); fig.savefig("../figures/energy3d." + EXT)

# RZ
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
r = np.loadtxt(D + "rz_random.txt")
a = ax[0]
for m in range(3):
    a.semilogy(r[1:, 0], np.maximum(np.abs(r[1:, 1 + m]), fl), color=col[m], lw=1, label=r"$|W_%d/W_%d(0)-1|$" % (m, m))
    a.semilogy(r[1:, 0], np.maximum(r[1:, 4 + m], fl), "--", color=col[m], lw=0.8,
               label=r"Gauss (dashed), div $B$ (dotted)" if m == 0 else None)
    a.semilogy(r[1:, 0], np.maximum(r[1:, 7 + m], fl), ":", color=col[m], lw=0.8)
a.set_ylim(1e-17, 1e-9); a.set_xlabel("step"); a.set_title(r"random fields, modes 0--2, $c\tau/h_r = 4$", fontsize=8)
a.legend(frameon=False, fontsize=6, loc="upper left", ncol=2); clean(a)
h = np.loadtxt(D + "rz_hot.txt")
a = ax[1]
M = (h.shape[1] - 3) // 2
for m in range(M):
    a.semilogy(h[:, 0], h[:, 1 + m] / h[0, 1 + M], color=col[m], lw=1, label=r"$W_%d/K(0)$" % m)
    a.semilogy(h[1:, 0], np.maximum(h[1:, 2 + M + m], fl), "--", color=col[m], lw=0.8, label=r"Gauss $m=%d$" % m)
a.set_xlabel("step"); a.set_title(r"hot plasma column, modes 0--2, $c\tau/h_r = 4$", fontsize=8)
a.set_ylim(1e-17, 1); a.legend(frameon=False, fontsize=6, loc="center right", ncol=2); clean(a)
fig.tight_layout(); fig.savefig("../figures/energyrz." + EXT)
