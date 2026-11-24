#!/usr/bin/env python3
"""Witness polarization in the PWFA demo (bench/qed_pwfa/README.md): <S_x>, <|S|>, <chi> and the radiated energy vs time
for the runs out_both, out_tbmt, out_rad (dir/qed.txt), as in Fig. 2 of arXiv:2605.19814.
   python3 bench/qed_pwfa/plot.py [run directory] [lambda0 in um, default 1]"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "."
lam = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
ps = lam * 1e-6 / (2 * np.pi * 299792458.0) * 1e12      # 1/omega0 in ps
runs = [("both", "T-BMT + radiation", "C3", "-"), ("tbmt", "T-BMT only", "k", "-."), ("rad", "radiation only", "C0", "--")]
fig, ax = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
for key, label, col, ls in runs:
    f = os.path.join(d, "out_" + key, "qed.txt")
    if not os.path.exists(f):
        continue
    q = np.loadtxt(f)
    t = q[:, 1] * ps
    ax[0].plot(t, q[:, 3], ls, color=col, label=label)                 # <S0> = <S_x>
    ax[1].plot(t, q[:, 6], ls, color=col, label=label)                 # <|S|>
    ax[2].plot(t, q[:, 7], ls, color=col, label=label + " <chi>")       # <chi>
    print("%-18s t = %6.2f ps: <S_x> %.4f, <S_y> %.4f, <|S|> %.4f, <chi> %.4f, chi_max %.3f, E_rad / E_0 %.4f"
          % (label, t[-1], q[-1, 3], q[-1, 4], q[-1, 6], q[-1, 7], q[-1, 8], q[-1, 10] / (q[0, 2] * 1956950.0)))
ax[0].set_ylabel(r"$\langle S_x\rangle$")
ax[1].set_ylabel(r"$\langle |S|\rangle$")
ax[2].set_ylabel(r"$\langle\chi\rangle$")
ax[2].set_xlabel("t (ps)")
ax[0].legend()
fig.tight_layout()
out = os.path.join(d, "pwfa_spin.png")
fig.savefig(out, dpi=120)
print("wrote", out)
