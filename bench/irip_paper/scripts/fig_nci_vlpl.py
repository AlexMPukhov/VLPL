"""Fig.: NCI streaming-plasma test in VLPL (3D: VLPL-RIP, VLPL-iRIP; RZ, m = 0, 1: VLPL-RZ, VLPL-iRZ) and in FBPIC
(standard and Galilean PSATD, 2 modes). Data: runs/gpuq2.sh -> scripts/reduce_nci.py -> ../data/nciv_*.txt.
Solid: no filter; dashed: with the filter (VLPL: Gauss-exact binomial filter of all components, its default since
decision 91, one pass; FBPIC: its default binomial filter); dotted: VLPL with the projection filter (one pass, unstable). Data
../data/nciv_*_f1 (binomial), *_fp (projection), from VLPL bench/nci/run_filter.sh (branch filter-gauss) and runs/gpuq2.sh.
"""
import numpy as np, matplotlib, sys, os
matplotlib.use("pgf" if "pgf" in sys.argv else "Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "pgf.rcfonts": False, "axes.linewidth": 0.6})
EXT = "pgf" if "pgf" in sys.argv else "pdf"
D = "../data/nciv_"
cases = [("g10_h3", r"$\gamma = 10$, $h_\perp = 3c\tau$"), ("g10_hq", r"$\gamma = 10$, $h_\perp = c\tau/4$"),
         ("g50_h3", r"$\gamma = 50$, $h_\perp = 3c\tau$")]
rows = [[("c3_rip_cc_%s_f%d", "VLPL-RIP", "#1baf7a"), ("c3_irip_%s_f%d", "VLPL-iRIP", "#2a78d6")],
        [("rz_rip_cc_%s_f%d", "VLPL-RZ", "#1baf7a"), ("rz_irip_%s_f%d", "VLPL-iRZ", "#2a78d6"),
         ("fb_std_f%d_%s", "FBPIC", "#eb6834"), ("fb_gal_f%d_%s", "FBPIC Galilean", "#8a5bd6")]]
fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True, sharey=True)
for i, row in enumerate(rows):
    for j, (cs, title) in enumerate(cases):
        a = ax[i, j]
        for pat, lab, col in row:
            for f, ls in ((0, "-"), (1, "--"), ("p", ":")):
                if f == "p" and pat.startswith("fb"):
                    continue
                name = (pat.replace("%d", "%s")) % ((f, cs) if pat.startswith("fb") else (cs, f))
                p = D + name + ".txt"
                if not os.path.exists(p):
                    continue
                h = np.loadtxt(p)
                a.semilogy(h[:, 0], np.maximum(h[:, 1], 1e-14), ls, color=col, lw=1.1,
                           label=lab + ("" if f == 0 else " + projection filter" if f == "p" else " + filter"))
        if i == 0:
            a.set_title(title, fontsize=9)
        if i == 1:
            a.set_xlabel(r"$t\,\omega_p$")
        a.set_ylim(1e-12, 1e2)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[i, 0].set_ylabel((r"3D: " if i == 0 else r"RZ: ") + r"$W_{\rm field}/K_e$")
    h, l = ax[i, 0].get_legend_handles_labels()
    ax[i, 2].legend(h, l, frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout(); fig.savefig("../figures/nci_vlpl." + EXT)
