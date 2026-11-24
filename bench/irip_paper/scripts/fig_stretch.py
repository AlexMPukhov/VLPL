"""Fig.: reflection of a wave packet that runs from a fine transverse grid (h_min = 0.1 c dt) into a stretched one
(box iRIP). Data: VLPL research/irip_3d/stretch.py -> stretch_results.txt (commit ebe8483), copied to ../data."""
import numpy as np, matplotlib, sys, re
matplotlib.use("pgf" if "pgf" in sys.argv else "Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "pgf.rcfonts": False, "axes.linewidth": 0.6})
EXT = "pgf" if "pgf" in sys.argv else "pdf"
lines = open("../data/stretch_results.txt").read().splitlines()
main, scans, cur = [], {}, None
for ln in lines:
    m = re.match(r"\s*(\d+)\s+(\d+)\s+\(([\d.]+),\s*([\d.]+)\)\s+([\d.]+)\s+(\S+)\s+(\S+)\s+(\S+)", ln)
    if m:
        main.append([float(v) for v in m.groups()]); continue
    m = re.match(r"c dt scan.*wavelength (\d+) H", ln)
    if m:
        cur = int(m.group(1)); scans[cur] = []; continue
    m = re.match(r"\s+(\d+)\s+(\S+)\s+(\S+)\s+([\d.]+)\s+", ln)
    if m and cur is not None:
        scans[cur].append([float(v) for v in m.groups()])
main = np.array(main)
floor = 1e-8
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
a = ax[0]
for ratio, mk, lab in ((1.05, "o", "5 % per cell"), (1.50, "s", "50 % per cell")):
    s = main[np.isclose(main[:, 2], ratio)]
    a.semilogy(s[:, 4], np.maximum(np.abs(s[:, 5]), floor), mk, mfc="none", color="#2a78d6", ms=5, label=lab + ", TE")
    a.semilogy(s[:, 4], np.maximum(np.abs(s[:, 6]), floor), mk, color="#eb6834", ms=3, label=lab + ", TM")
a.axvline(2, color="gray", lw=0.6, ls=":"); a.text(2.05, 2e-8, r"cutoff $k_yh=2$", fontsize=7, color="gray")
a.set_xlabel(r"$k_y h_{\max}$"); a.set_ylabel("reflected energy / incident")
a.set_ylim(5e-9, 3); a.legend(frameon=False, fontsize=6.5, loc="upper left")
a = ax[1]
col = {12: "#1baf7a", 6: "#2a78d6", 3: "#eb6834"}
for wl, d in sorted(scans.items(), reverse=True):
    d = np.array(d)
    kh = 2 * np.pi / wl * np.sin(np.radians(70))       # k_y h_max for (h_min, h_max) = (0.1, 1) H
    a.loglog(d[:, 0], np.maximum(np.abs(d[:, 1]), floor), "o-", color=col[wl], ms=3, lw=1,
             label=r"$k_yh_{\max} = %.2f$" % kh)
    a.loglog(d[:, 0], np.maximum(np.abs(d[:, 2]), floor), "s--", color=col[wl], ms=3, lw=1, mfc="none")
a.set_xlabel(r"$c\tau/h_{\min}$"); a.set_ylim(5e-9, 3)
a.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.02, 0.86)); a.set_title("solid: TE, dashed: TM", fontsize=7)
for a in ax:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("../figures/stretch." + EXT)
