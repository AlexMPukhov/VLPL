"""Plot of the NCI comparison (run `python3 nci2d.py paper`, `python3 nci2d.py fine`,
`python3 nci_g50.py box,box_f,psatd,psatd_f` first). Writes nci_compare.png."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cases = [("nci_paper_", "γ = 10,  h⊥ = 3 c·dt", 64 * 0.63 * 32 * 1.88, np.sqrt(101) - 1),
         ("nci_fine_", "γ = 10,  h⊥ = c·dt / 4", 64 * 0.63 * 128 * 0.1575, np.sqrt(101) - 1),
         ("nci_g50_", "γ = 50,  h⊥ = 3 c·dt", 64 * 0.63 * 32 * 1.88, np.sqrt(2501) - 1)]
col = {"box": "#2a78d6", "psatd": "#eb6834", "ripcc": "#1baf7a"}
lab = {"box": "box iRIP", "psatd": "PSATD (FBPIC-like)", "ripcc": "RIP-CC (explicit)"}
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for a, (pre, title, area, gm1) in zip(ax, cases):
    for s in ("ripcc", "box", "psatd"):
        for filt, ls in (("", "-"), ("_f", "--")):
            try:
                h = np.loadtxt(pre + s + filt + ".txt")
            except OSError:
                continue
            a.semilogy(h[:, 0], h[:, 1] / (gm1 * area), ls, color=col[s], lw=2,
                       label=lab[s] + (" + binomial filter" if filt else ""))
    a.set_title(title, fontsize=11); a.set_xlabel("t ωp"); a.grid(alpha=.25, lw=.6)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
ax[0].set_ylabel("field energy / electron kinetic energy")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
fig.suptitle("Numerical Cherenkov test: e-p plasma streaming along x, 2D3V, c·dt = dx, linear shapes", fontsize=11.5)
fig.tight_layout(rect=(0, 0.1, 1, 0.95)); fig.savefig("nci_compare.png", dpi=130)
