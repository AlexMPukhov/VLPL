"""Fig. 4: numerical Cherenkov test (data from research/irip_rz/nci2d.py, nci_g50.py; copied to ../data)."""
import numpy as np, matplotlib, sys
matplotlib.use("pgf" if "pgf" in sys.argv else "Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "pgf.rcfonts": False, "axes.linewidth": 0.6})
EXT = "pgf" if "pgf" in sys.argv else "pdf"
cases = [("nci_paper_", r"$\gamma = 10$, $h_\perp = 3c\tau$", 64 * 0.63 * 32 * 1.88, np.sqrt(101) - 1),
         ("nci_fine_", r"$\gamma = 10$, $h_\perp = c\tau/4$", 64 * 0.63 * 128 * 0.1575, np.sqrt(101) - 1),
         ("nci_g50_", r"$\gamma = 50$, $h_\perp = 3c\tau$", 64 * 0.63 * 32 * 1.88, np.sqrt(2501) - 1)]
col = {"box": "#2a78d6", "psatd": "#eb6834", "ripcc": "#1baf7a"}
lab = {"box": "box iRIP", "psatd": "PSATD", "ripcc": "RIP-CC"}
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
for a, (pre, title, area, gm1) in zip(ax, cases):
    for s in ("ripcc", "box", "psatd"):
        for filt, ls in (("", "-"), ("_f", "--")):
            try:
                h = np.loadtxt("../data/" + pre + s + filt + ".txt")
            except OSError:
                continue
            a.semilogy(h[:, 0], h[:, 1] / (gm1 * area), ls, color=col[s], lw=1.3,
                       label=lab[s] + (" + filter" if filt else ""))
    a.set_title(title, fontsize=9); a.set_xlabel(r"$t\,\omega_p$")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
ax[0].set_ylabel(r"$W_{\rm field}/K_e$")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig("../figures/nci." + EXT)
