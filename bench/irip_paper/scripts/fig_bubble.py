"""Fig.: bubble regime with self-injection (VLPL bench/lwfa_bubble, a0 = 4, 0.005 n_c): injected charge vs time and the
spectrum at 900 fs, VLPL-RIP (3D, RIP-CC) and FBPIC (2 modes). Data: scripts/reduce_bubble.py -> ../data/bubble_*.txt."""
import numpy as np, matplotlib, sys
matplotlib.use("pgf" if "pgf" in sys.argv else "Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "pgf.rcfonts": False, "axes.linewidth": 0.6})
EXT = "pgf" if "pgf" in sys.argv else "pdf"
q = np.loadtxt("../data/bubble_injected.txt")
s = np.loadtxt("../data/bubble_spectrum.txt")
t_s = open("../data/bubble_spectrum.txt").readline().split("t = ")[1].split(" fs")[0]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
a = ax[0]
a.plot(q[:, 0], q[:, 1], "o-", color="#1baf7a", ms=3, lw=1.2, label="VLPL-RIP (3D)")
a.plot(q[:, 0], q[:, 2], "s--", color="#eb6834", ms=3, lw=1.2, label="FBPIC")
a.set_xlabel(r"$t$ [fs]"); a.set_ylabel(r"$Q\ (\gamma > 20)$ [pC]"); a.legend(frameon=False, fontsize=7)
a = ax[1]
a.step(s[:, 0], s[:, 1], where="mid", color="#1baf7a", lw=1.1, label="VLPL-RIP (3D)")
a.step(s[:, 0], s[:, 2], where="mid", color="#eb6834", lw=1.1, ls="--", label="FBPIC")
a.set_xlim(0, 250); a.set_xlabel(r"$E$ [MeV]"); a.set_ylabel(r"$dQ/dE$ [pC/MeV]")
a.set_title(r"$t = %s$ fs" % t_s, fontsize=8)
for a in ax:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("../figures/bubble." + EXT)
