#!/usr/bin/env python3
"""Plots of examples/qed_cascade.deck (decision 96): pairs per beam electron vs time, the pair spectrum dN/dx with the
positron spin along e2, the lepton energy spectrum.   python3 bench/qed_cascade/plot.py <dir with qed.txt, qed_pairs.txt>"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "out_qed_cascade"
h = open(d + "/qed.txt").readline().split()[1:]
h = h[:h.index("<xi3>_decayed") + 1]
c = {k: i for i, k in enumerate(h)}
q = np.array([[float(v) for v in l.split()] for l in open(d + "/qed.txt") if not l.startswith("#")])
W0 = q[0, c["W_electrons"]]
fs = 0.8e-6 / (2 * np.pi * 2.99792458e8) * 1e15
p = np.loadtxt(d + "/qed_pairs.txt")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].plot(q[:, 1] * fs, q[:, c["N_pairs"]] / W0, "k-", label="pairs")
ax[0].plot(q[:, 1] * fs, q[:, c["W_positrons"]] / W0, "r--", label="positrons in the box")
ax[0].set_xlabel("t (fs)"); ax[0].set_ylabel("per beam electron"); ax[0].legend()
xm = 0.5 * (p[:, 0] + p[:, 1])
ax[1].plot(xm, p[:, 2] / p[:, 2].sum() / (p[:, 1] - p[:, 0]), "k-", label="dN/dx (normalized)")
ok = p[:, 2] > 0
ax[1].plot(xm[ok], p[ok, 4], "r.", label="<S+ . e2>")
ax[1].set_xlabel("x = eps+ / eps_gamma"); ax[1].legend()
em = np.sqrt(p[:, 6] * p[:, 7])
ok = p[:, 8] > 0
ax[2].loglog(em[ok], p[ok, 8] / (p[ok, 7] - p[ok, 6]) / W0, "k-")
ax[2].set_xlabel("eps (MeV)"); ax[2].set_ylabel("dN/d eps per beam electron (1/MeV), e+ and e-")
plt.tight_layout()
plt.savefig(d + "/pairs.png", dpi=110)
print("pairs per beam electron %.4f, <S+.e2> %.3f, <S-.e2-> %.3f, <xi3> of the decayed photons %.3f; wrote %s/pairs.png" % (
    q[-1, c["N_pairs"]] / W0, q[-1, c["<S+.e2>_pairs"]], q[-1, c["<S-.e2->_pairs"]], q[-1, c["<xi3>_decayed"]], d))
