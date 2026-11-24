# Field and total energy of the streaming plasma, RIP-CC against Yee: bench/nci/nci.png
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = os.path.dirname(os.path.abspath(__file__))
wp = 0.1                                    # omega_p / omega0 (n = 0.01 n_c)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for s, lab, c in [("rip_cc", "RIP-CC", "#1f77b4"), ("yee", "Yee", "#d62728")]:
    a = np.loadtxt(os.path.join(d, "diag_%s.txt" % s))
    t = a[:, 1] * wp
    ax[0].semilogy(t[1:], a[1:, 2] / a[0, 3], color=c, label=lab)
    ax[1].semilogy(t[1:], np.abs(a[1:, 4] / a[0, 4] - 1) + 1e-16, color=c, label=lab)
ax[0].set_xlabel(r"$\omega_p t$"); ax[0].set_ylabel(r"$W_{field} / W_{kin}(0)$"); ax[0].legend()
ax[1].set_xlabel(r"$\omega_p t$"); ax[1].set_ylabel(r"$|W_{total} / W_{total}(0) - 1|$"); ax[1].legend()
fig.suptitle("Plasma streaming at u = -10 through a periodic box (64 x 32 x 32)")
fig.tight_layout()
fig.savefig(os.path.join(d, "nci.png"), dpi=120)
