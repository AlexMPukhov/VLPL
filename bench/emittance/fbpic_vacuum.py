"""FBPIC reference for the vacuum case (bench/emittance/README.md, Fig. 4 of Pukhov 2020):
10 GeV, eps_n = 1 nm, sigma = 1.41 um, 0.6 or 6 nC, 1 mm, dz = 0.1 um, dr = 0.15 um, c dt = dz.

python3 fbpic_vacuum.py <charge_nC> <galilean 0|1> <out.txt> [length_mm] [cpu]
Writes "step t[fs] eps_ny[um] eps_nz[um] sigma_y[um] E_mean[MeV]" every 200 steps (the columns of VLPL's beam_*.txt
that matter here).
"""
import sys
import numpy as np
from scipy.constants import c, e, m_e
from fbpic.main import Simulation
from fbpic.lpa_utils.bunch import add_particle_bunch_gaussian

Q = float(sys.argv[1]) * 1e-9
gal = int(sys.argv[2])
out = sys.argv[3]
L = float(sys.argv[4]) * 1e-3 if len(sys.argv) > 4 else 1e-3
cuda = not (len(sys.argv) > 5 and sys.argv[5] == "cpu")

dz, dr = 0.1e-6, 0.15e-6
Nz, Nr, Nm = 256, 64, 2                     # 25.6 um x 9.6 um (VLPL: 19.2 um periodic square)
zmin, zmax = -12.8e-6, 12.8e-6
dt = dz / c
gamma0 = 1 + 10e9 / 0.51099895e6
sig = 1.41e-6
kw = dict(v_comoving=c * np.sqrt(1 - 1 / gamma0**2), use_galilean=True) if gal else {}
sim = Simulation(Nz, zmax, Nr, dr * Nr, Nm, dt, zmin=zmin, n_order=-1, use_cuda=cuda,
                 boundaries={'z': 'open', 'r': 'open'}, particle_shape='cubic', **kw)
sim.ptcl = []                               # no plasma
nmacro = 4_000_000
bunch = add_particle_bunch_gaussian(sim, -e, m_e, sig_r=sig, sig_z=sig, n_emit=1e-9, gamma0=gamma0, sig_gamma=0.0,
                                    n_physical_particles=Q / e, n_macroparticles=nmacro, zf=0.0, tf=0.0,
                                    initialize_self_field=True)
sim.set_moving_window(v=c)


def moments(sp):
    g = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
    x, y, ux, uy, uz, w = (g(sp.x), g(sp.y), g(sp.ux), g(sp.uy), g(sp.uz), g(sp.w))
    W = w.sum()
    def eps(a, u):
        da, du = a - (w * a).sum() / W, u - (w * u).sum() / W
        return np.sqrt((w * da * da).sum() / W * (w * du * du).sum() / W - ((w * da * du).sum() / W) ** 2)
    gam = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    sy = np.sqrt((w * (y - (w * y).sum() / W) ** 2).sum() / W)
    return eps(x, ux) * 1e6, eps(y, uy) * 1e6, sy * 1e6, (w * (gam - 1)).sum() / W * 0.51099895


nsteps = int(round(L / dz))
with open(out, "w") as f:
    f.write("# step t[fs] eps_ny[um] eps_nz[um] sigma_y[um] E_mean[MeV]  (FBPIC %s, Q = %g nC)\n"
            % ("Galilean" if gal else "standard", Q * 1e9))
    for n in range(0, nsteps + 1, 200):
        if n > 0:
            sim.step(200, show_progress=False)
        ex, ey, sy, E = moments(bunch)
        f.write("%d %.4f %.6e %.6e %.6e %.6e\n" % (n, n * dt * 1e15, ex, ey, sy, E))
        f.flush()
