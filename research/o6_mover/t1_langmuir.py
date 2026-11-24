"""T1: cold Langmuir oscillation (1D: ny = 1), omega_p = 1, k = 2 pi / (64 dx), omega_p dt = 0.3 ... 10.
Measured frequency against Crank-Nicolson, w dt = 2 atan(W dt / 2), and leapfrog, sin(w dt / 2) = W dt / 2 (unstable for W dt > 2),
W^2 = wp^2 sinc^(2s)(k dx / 2); energy W + K, Gauss law, outer iterations and particle passes per step."""
import sys
import numpy as np
from pic2d import Sim, lattice

shape = int(sys.argv[1]) if len(sys.argv) > 1 else 2
nx, ppx, nsteps, amp = 64, 8, 400, 1e-3


def run(wpdt, mode, accel="anderson", precond=True, tol=1e-10):
    D = wpdt                                  # omega_p = 1
    S = Sim(nx, 1, D, D, shape)
    S.ptol = 1e-10
    x, y = lattice(nx, 1, D, D, ppx, 1)
    k = 2 * np.pi / (nx * D)
    u = np.zeros((len(x), 3))
    u[:, 0] = amp * np.sin(k * x)
    w = 1.0 / (len(x) / (nx * S.A))
    S.add_species(x, y, u, -w, w)
    S.neutralise()
    xs = np.arange(nx) * D
    a, E = [], []
    E0 = S.W() + S.K()
    g0 = S.gauss()
    for n in range(nsteps):
        if mode == "implicit":
            S.step_implicit(tol=tol, maxit=200, inner=1, accel=accel, precond=precond)
        else:
            S.step_explicit()
        a.append((S.f[0][:, 0] * np.sin(k * xs)).sum())          # E_x ~ sin(k x) for u_x ~ sin(k x)
        E.append(S.W() + S.K())
        if not np.isfinite(E[-1]) or abs(E[-1]) > 1e6 * E0:
            break
    a = np.array(a)
    if len(a) < nsteps:
        return np.nan, np.inf, np.nan, (0, 0)
    sp = np.abs(np.fft.rfft((a - a.mean()) * np.hanning(len(a)), 64 * len(a)))
    i = np.argmax(sp[1:]) + 1
    if 1 <= i < len(sp) - 1:                  # parabolic peak
        y0, y1, y2 = np.log(sp[i - 1:i + 2] + 1e-300)
        i = i + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    wdt = 2 * np.pi * i / (64 * len(a))
    dE = np.max(np.abs(np.array(E) - E0)) / E0
    dg = np.abs(S.gauss() - g0).max() / np.abs(S.rho_bg).max()
    st = (np.mean([s[0] for s in S.stats]), np.mean([s[2] for s in S.stats])) if S.stats else (0, 0)
    return wdt, dE, dg, st


k = 2 * np.pi / nx                             # k dx
F = np.sinc(k / 2 / np.pi) ** (2 * shape)
print(f"shape {shape}, k dx = {k:.4f}, |W/wp|^2 = {F:.6f}")
print(flush=True)
print(" wp dt | CN theory  implicit (dE/E, Gauss, outer its, particle passes) | leapfrog theory  explicit (dE/E)")
for wpdt in (0.3, 1.0, 1.9, 3.0, 5.0, 10.0):
    W = np.sqrt(F) * wpdt
    cn = 2 * np.arctan(W / 2)
    lf = 2 * np.arcsin(W / 2) if W < 2 else np.nan
    wi, dEi, dgi, sti = run(wpdt, "implicit")
    we, dEe, _, _ = run(wpdt, "explicit")
    print(f"{wpdt:6.1f} | {cn:.6f}  {wi:.6f} ({dEi:.1e}, {dgi:.1e}, {sti[0]:.1f}, {sti[1]:.1f}) | {lf:.6f}  {we:.6f} ({dEe:.1e})", flush=True)
# solver comparison at wp dt = 5: plain Picard, preconditioned Picard, preconditioned Anderson
for lab, acc, pre in (("Picard", "none", False), ("Picard + P", "none", True), ("Anderson + P", "anderson", True),
                      ("Anderson", "anderson", False)):
    for wpdt in (0.3, 1.9, 5.0):
        wi, dEi, dgi, sti = run(wpdt, "implicit", accel=acc, precond=pre)
        print(f"solver {lab:13s} wp dt {wpdt}: w dt {wi:.6f}, dE/E {dEi:.1e}, outer its {sti[0]:.1f}, passes {sti[1]:.1f}", flush=True)
