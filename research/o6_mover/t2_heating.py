"""T2: grid heating of a warm dense plasma at unresolved Debye length, 2D, fixed ions, no filter.
Default = the 2D analogue of the RZ reproducer of decision 80 (bench/noise_heating): 40 n_c, 1 keV (u_th = 0.044),
dx = 0.0125 um, dy = 0.0425 um (lambda0 = 0.8 um; code units 1/k0): omega_p dt = 0.62, dx / lambda_D = 14, dy / lambda_D = 48.
usage: python3 t2_heating.py mode shape [n=40 uth=0.044 dx=0.0125 dy=0.0425 nx=32 ny=12 ppx=4 ppy=4 t=29 tol=1e-10]
prints K/K0, (W+K)/(W0+K0) - 1 and solver statistics against time (fs)."""
import sys, time
import numpy as np
from pic2d import Sim, lattice

mode, shape = sys.argv[1], int(sys.argv[2])
P = dict(n=40.0, uth=0.044, dx=0.0125, dy=0.0425, nx=32, ny=12, ppx=4, ppy=4, t=29.0, tol=1e-10, seed=1, inner=50,
         ptol=1e-10, maxit=100)
for a in sys.argv[3:]:
    k, v = a.split("=")
    P[k] = type(P[k])(v)
k0 = 2 * np.pi / 0.8                                  # 1/um
D, dy = P["dx"] * k0, P["dy"] * k0
fs = 1e-15 * 2 * np.pi * 2.998e8 / 0.8e-6            # omega0 * 1 fs
nx, ny = P["nx"], P["ny"]
S = Sim(nx, ny, D, dy, shape)
S.ptol = P["ptol"]
x, y = lattice(nx, ny, D, dy, P["ppx"], P["ppy"])
rng = np.random.default_rng(P["seed"])
u = rng.normal(scale=P["uth"], size=(len(x), 3))
w = P["n"] / (len(x) / (nx * ny * S.A))
S.add_species(x, y, u, -w, w)
S.neutralise()
wp = np.sqrt(P["n"])
print(f"# {mode} shape {shape}: wp dt = {wp * D:.3f}, dx/lD = {D * wp / P['uth']:.1f}, dy/lD = {dy * wp / P['uth']:.1f}, "
      f"N = {len(x)}, steps = {int(P['t'] * fs / D)}", flush=True)
K0 = S.K(); E0 = K0 + S.W(); g0 = S.gauss()
nsteps = int(P["t"] * fs / D)
t0 = time.time()
for n in range(1, nsteps + 1):
    if mode == "implicit":
        S.step_implicit(tol=P["tol"], maxit=P["maxit"], inner=P["inner"])
    else:
        S.step_explicit()
    if n % max(1, nsteps // 10) == 0 or n == nsteps:
        K = S.K(); W = S.W()
        if mode == "explicit":                         # u at n+1/2: report as is (heating is the point)
            pass
        st = S.stats[-max(1, nsteps // 10):] if S.stats else []
        its = np.mean([s[0] for s in st]) if st else 0
        ps = np.mean([s[2] for s in st]) if st else 0
        print(f"t = {n * D / fs:6.2f} fs  K/K0 = {K / K0:.4f}  W/K0 = {W / K0:.2e}  (W+K)/E0 - 1 = {(W + K) / E0 - 1:+.1e}  "
              f"outer {its:.1f} passes {ps:.1f}  ({time.time() - t0:.0f} s)", flush=True)
dg = np.abs(S.gauss() - g0).max() / np.abs(S.rho_bg).max()
print(f"# Gauss law change / rho = {dg:.1e}")
