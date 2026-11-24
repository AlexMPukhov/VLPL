"""The axis case of decision 80 in the RZ m = 0 (TM) prototype: uniform 40 n_c, 1 keV electrons, fixed ions, no laser,
dz = 0.0125 um, dr = 0.0425 um (k_p dr = 2.1, omega_p dt = 0.62), plasma r < 1.6 um, wall at 3.6 um, periodic z, ppc
(z, r, theta) = (1, 2, 8), 29 fs. The C++ code (4 modes, filter) heats 18x (plain gather) / 2.1x (matched) by 29 fs; 1 mode:
2.3x at 19 fs (bench/noise_heating/README.md).
usage: python3 rz_axis.py explicit|matched|implicit [nz=32 t=29 tol=1e-8 ptol=1e-10 inner=50 seed=1 n=40 dr=0.0425]"""
import sys, time
import numpy as np
from rz0 import RZ0

mode = sys.argv[1]
P = dict(nz=32, t=29.0, tol=1e-8, ptol=1e-10, inner=50, seed=1, n=40.0, dr=0.0425, dz=0.0125, uth=0.044, rmax=1.6, R=3.6,
         ppz=1, ppr=2, ppt=8)
for a in sys.argv[2:]:
    k, v = a.split("=")
    P[k] = type(P[k])(v)
k0 = 2 * np.pi / 0.8
fs = 1e-15 * 2 * np.pi * 2.998e8 / 0.8e-6
D, dr = P["dz"] * k0, P["dr"] * k0
nr = int(round(P["R"] / P["dr"]))
S = RZ0(P["nz"], nr, D, dr)
S.ptol = P["ptol"]
rng = np.random.default_rng(P["seed"])
# loading: Gauss-Legendre points in r per cell (decision 53), ppt phases per ring with a random offset, z uniform
ppz, ppr, ppt = P["ppz"], P["ppr"], P["ppt"]
zc = (np.arange(P["nz"] * ppz) + 0.5) * D / ppz
gl = np.array([0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)]) if ppr == 2 else (np.arange(ppr) + 0.5) / ppr
rr = np.array([(j + g) * dr for j in range(int(round(P["rmax"] / P["dr"]))) for g in gl])
Z, R, T = np.meshgrid(zc, rr, np.arange(ppt), indexing="ij")
ph = (T + rng.random(T.shape)) * 2 * np.pi / ppt
z, r, ph = Z.ravel(), R.ravel(), ph.ravel()
q = -P["n"] * r * dr / ppr * 2 * np.pi / ppt * D / ppz       # ring segment of density n (r dr per GL point)
u = rng.normal(scale=P["uth"], size=(len(z), 3))
S.add_species(z, r * np.cos(ph), r * np.sin(ph), u, q, -q)
S.neutralise()
wp = np.sqrt(P["n"])
nsteps = int(P["t"] * fs / D)
print(f"# {mode}: wp dt = {wp * D:.3f}, kp dr = {wp * dr:.2f}, nz = {P['nz']}, nr = {nr}, N = {len(z)}, steps = {nsteps}",
      flush=True)
K0 = S.K(); E0 = K0 + S.W(); g0 = S.gauss()
t0 = time.time()
rax = r < 3 * dr


def kin_inner():
    s = S.sp[0]
    rr_ = np.hypot(s["x"], s["y"])
    u2 = (s["u"] ** 2).sum(1)
    k = s["m"] * u2 / (np.sqrt(1 + u2) + 1)
    m = rr_ < 3 * dr
    return k[m].sum() / max(s["m"][m].sum(), 1e-300)


ki0 = kin_inner()
for n in range(1, nsteps + 1):
    if mode == "implicit":
        S.step_implicit(tol=P["tol"], maxit=100, inner=P["inner"])
    else:
        S.step_explicit(matched=(mode == "matched"))
    if n % max(1, nsteps // 10) == 0 or n == nsteps:
        K, W = S.K(), S.W()
        st = S.stats[-max(1, nsteps // 10):]
        its = np.mean([s_[0] for s_ in st]) if st else 0
        ps = np.mean([s_[2] for s_ in st]) if st else 0
        print(f"t = {n * D / fs:6.2f} fs  K/K0 = {K / K0:.4f}  (r < 3 dr: {kin_inner() / ki0:.3f})  W/K0 = {W / K0:.2e}  "
              f"(W+K)/E0 - 1 = {(W + K) / E0 - 1:+.1e}  Er(axis half node) rms = {np.sqrt((S.f[1][:, 0] ** 2).mean()):.2e}  "
              f"outer {its:.1f} passes {ps:.1f}  ({time.time() - t0:.0f} s)", flush=True)
dg = np.abs(S.gauss() - g0).max() / np.abs(S.rho_bg).max()
print(f"# Gauss law change / rho = {dg:.1e}")
