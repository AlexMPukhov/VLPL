"""Figures 2 and 3 of the iRIP paper (2D Cartesian, per Fourier mode).
fig_dispersion: relative phase-velocity error w/(c|k|) - 1 against the propagation angle for RIP-CC,
box iRIP and Yee.  fig_gain: transient amplification sup_n ||M^n|| (energy norm) of the TE subsystem
for the characteristic-trapezoid closure and the box closure, against kappa = k_perp D / 2.
Writes figures/*.pgf (text, for the LaTeX source) and figures/*.pdf."""
import numpy as np, matplotlib, sys
matplotlib.use("pgf" if "pgf" in sys.argv else "Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "pgf.rcfonts": False, "axes.linewidth": 0.6})
EXT = "pgf" if "pgf" in sys.argv else "pdf"
C = {"rip": "#1baf7a", "box": "#2a78d6", "yee": "#eb6834", "trap": "#eb6834"}

# ------------------------------------------------------------------ dispersion
D = 1.0
k = 2 * np.pi / (16 * D)
th = np.linspace(0, np.pi / 2, 181)
kx, ky = k * np.cos(th), k * np.sin(th)
def w_box(h):
    lam = (2 / h * np.sin(ky * h / 2)) ** 2
    return 2 / D * np.arctan(np.sqrt(np.tan(kx * D / 2) ** 2 + D * D * lam / 4))
def w_ripcc(h):
    q = (np.sin(ky * h / 2) / h) ** 2                       # lambda / 4
    s2 = D * D * q + np.sin(kx * D / 2) ** 2 * (1 - D * D * q)
    return 2 / D * np.arcsin(np.sqrt(s2))
def w_yee(dx, h, cfl):
    dt = cfl / np.sqrt(1 / dx ** 2 + 1 / h ** 2)
    s2 = dt * dt * ((np.sin(kx * dx / 2) / dx) ** 2 + (np.sin(ky * h / 2) / h) ** 2)
    return 2 / dt * np.arcsin(np.sqrt(s2))
fig, ax = plt.subplots(figsize=(3.4, 2.5))
deg = np.degrees(th)
ax.plot(deg, w_ripcc(2.0) / k - 1, color=C["rip"], lw=1.4, label=r"RIP-CC, $h_\perp = 2\Delta$")
ax.plot(deg, w_box(2.0) / k - 1, color=C["box"], lw=1.4, label=r"box iRIP, $h_\perp = 2\Delta$")
ax.plot(deg, w_box(0.25) / k - 1, color=C["box"], lw=1.4, ls="--", label=r"box iRIP, $h_\perp = \Delta/4$")
ax.plot(deg, w_yee(1.0, 2.0, 0.98) / k - 1, color=C["yee"], lw=1.4, label=r"Yee, $h_\perp = 2h_x$, CFL 0.98")
ax.axhline(0, color="0.6", lw=0.5)
ax.set_xlabel(r"angle to the $x$ axis (deg)"); ax.set_ylabel(r"$\omega/(c|k|) - 1$")
ax.set_xlim(0, 90); ax.legend(frameon=False, fontsize=7, loc="lower left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("../figures/dispersion." + EXT)

# ------------------------------------------------------------------ transient gain, TE per mode
def amp(kxD, kappa, closure):
    """One-step matrix of (Ex, Ey, Bz) for a Fourier mode; D = 1, g = i k_perp (any h)."""
    a = np.exp(1j * kxD); g = 2j * kappa                      # G = i k_perp, kappa = k_perp D/2
    def step(s, Ex1):
        Ex, Ey, Bz = s
        if closure == "box":
            ER = EL = 0.25 * (1 + a) * (Ex1 + Ex)
        else:                                                 # characteristic trapezoid
            ER, EL = 0.5 * (a * Ex1 + Ex), 0.5 * (Ex1 + a * Ex)
        Tp1 = (Ey + Bz + g * ER) / a
        Tm1 = (Ey - Bz) * a - g * EL
        Ey1, Bz1 = (Tp1 + Tm1) / 2, (Tp1 - Tm1) / 2
        return np.array([Ex + g * (Bz + Bz1) / 2, Ey1, Bz1])
    I = np.eye(3)
    c = np.array([step(I[q], 0)[0] for q in range(3)])
    e = step(np.zeros(3), 1)[0] - step(np.zeros(3), 0)[0]
    x = c / (1 - e)
    return np.array([step(I[q], x[q]) for q in range(3)]).T
def sup_norm(M, nmax=4096):
    P, s, n = M.copy(), 1.0, 1
    Q = np.eye(3)
    for n in range(1, nmax + 1):
        Q = Q @ M; s = max(s, np.linalg.norm(Q, 2))
    return s
kap = np.logspace(-1, 2, 40)
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for kxD, ls in ((np.pi / 2, "-"), (np.pi / 8, "--")):
    gt = [sup_norm(amp(kxD, q, "trap"), 1024) for q in kap]
    gb = [sup_norm(amp(kxD, q, "box"), 1024) for q in kap]
    ax.loglog(kap, gt, ls, color=C["trap"], lw=1.4, label=r"characteristic, $k_x\Delta = %s$" % ("\\pi/2" if kxD > 1 else "\\pi/8"))
    ax.loglog(kap, gb, ls, color=C["box"], lw=1.4, label=r"box, $k_x\Delta = %s$" % ("\\pi/2" if kxD > 1 else "\\pi/8"))
    print("kxD=%.3f  trap gain at kappa=10: %.2f, 100: %.2f; box max |gain-1| %.1e"
          % (kxD, gt[np.argmin(abs(kap - 10))], gt[-1], max(abs(np.array(gb) - 1))))
ax.set_xlabel(r"$\kappa = k_\perp\Delta/2$"); ax.set_ylabel(r"$\sup_n \|M^n\|$"); ax.set_ylim(0.7, 200)
ax.legend(frameon=False, fontsize=7, loc="upper left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("../figures/gain." + EXT)
