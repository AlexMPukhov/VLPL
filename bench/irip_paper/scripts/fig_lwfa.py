"""Fig.: on-axis longitudinal field in the LWFA benchmark (a0 = 1, 0.01 n_c) at 150 and 300 fs: VLPL-RIP, VLPL-iRIP
(3D), VLPL-RZ, VLPL-iRZ (2 modes) and FBPIC (2 modes). Data: scripts/reduce_lwfa.py -> ../data/lwfa_axis_*fs.txt."""
import numpy as np, matplotlib, sys
matplotlib.use("pgf" if "pgf" in sys.argv else "Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "pgf.rcfonts": False, "axes.linewidth": 0.6})
EXT = "pgf" if "pgf" in sys.argv else "pdf"
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
cur = [(1, "VLPL-RIP (3D)", "#1baf7a", "-", 1.2), (2, "VLPL-iRIP (3D)", "#2a78d6", "-", 1.0),
       (3, "VLPL-RZ", "#1baf7a", ":", 1.4), (4, "VLPL-iRZ", "#2a78d6", ":", 1.4), (6, "FBPIC", "#eb6834", "--", 1.1)]
for a, t in zip(ax, ("150", "300")):
    d = np.loadtxt("../data/lwfa_axis_%sfs.txt" % t)
    for k, lab, col, ls, lw in cur:
        a.plot(d[:, 0], d[:, k], ls, color=col, lw=lw, label=lab)
    a.set_title(r"$t = %s$ fs" % t, fontsize=9); a.set_xlabel(r"$x$ [$\mu$m]")
    a.set_xlim(d[0, 0], d[-1, 0])
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
ax[0].set_ylabel(r"$E_x$ on axis [$m c\omega_0/e$]")
ax[1].legend(frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig("../figures/lwfa." + EXT)
