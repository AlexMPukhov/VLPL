"""T3: transverse EM wave in a cold plasma along x (ny = 1), omega_p = 1, implicit mover.
Theory: the discrete relation of the box transport with the Crank-Nicolson cold response (the O7 Drude medium), derived
in disp1d.py from the step itself, with wpe^2 = wp^2 sinc^(2(s+1))(k D/2) (deposit and gather of Jy, Ety carry S_s along x).
(The tan^2 form of decision 76 is a different cold-fluid model: 0.316 instead of 0.313 at wp dt = 0.3.)"""
import sys
import numpy as np
from pic2d import Sim, lattice
from disp1d import solve

shape = 2
nx, ppx, nsteps, amp = 64, 8, 600, 1e-4


def theory(kD, D, wpe2):
    r, _ = solve(kD, D, wpe2)
    return [x for x in r if abs(x - kD) > 1e-6][0]


def run(wpdt, mk, mode):
    D = wpdt
    S = Sim(nx, 1, D, D, shape)
    S.ptol = 1e-10
    x, y = lattice(nx, 1, D, D, ppx, 1)
    w = 1.0 / (len(x) / (nx * S.A))
    S.add_species(x, y, np.zeros((len(x), 3)), -w, w)
    S.neutralise()
    k = 2 * np.pi * mk / (nx * D)
    xs = np.arange(nx) * D
    S.f[1][:, 0] = amp * np.cos(k * xs)                    # Ey at (i, j+1/2); standing wave start
    a = []
    for n in range(nsteps):
        S.step_implicit(tol=1e-10, maxit=200, inner=1) if mode == "implicit" else S.step_explicit()
        a.append((S.f[1][:, 0] * np.cos(k * xs)).sum())
    a = np.array(a)
    sp = np.abs(np.fft.rfft((a - a.mean()) * np.hanning(len(a)), 64 * len(a)))
    i = np.argmax(sp[1:]) + 1
    y0, y1, y2 = np.log(sp[i - 1:i + 2] + 1e-300)
    i = i + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return 2 * np.pi * i / (64 * len(a)), k * D


print("wp dt  mode  k dx  | theory w dt   implicit w dt   rel err | explicit w dt")
for wpdt in (0.3, 1.0, 3.0):
    for mk in (1, 4, 8):
        wi, kD = run(wpdt, mk, "implicit")
        we, _ = run(wpdt, mk, "explicit") if wpdt < 2 else (np.nan, 0)
        th = theory(kD, wpdt, np.sinc(kD / 2 / np.pi) ** (2 * (shape + 1)))
        print(f"{wpdt:5.1f} {mk:4d} {kD:6.3f} | {th:.6f}   {wi:.6f}   {wi / th - 1:+.1e} | {we:.6f}", flush=True)
