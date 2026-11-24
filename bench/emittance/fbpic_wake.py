"""FBPIC reference for case B (bench/emittance/README.md, rz_wake.deck): a0 = 1, 10 fs, w0 = 4 um, 0.01 n_c in a
matched parabolic channel; witness 10 MeV, 1 pC, eps_n = 0.1 um, sigma_r = 0.30 um, sigma_z = 0.3 um loaded at 150 fs
4.5 um behind the pulse centre with its space-charge field; 0.5 mm. The same lab-frame setup as VLPL (the window
moves from t = 0, its box shifted back by c 45 fs: from 45 fs on it is VLPL's window [c (t - 45 fs), + 16 um]).

python3 fbpic_wake.py <galilean 0|1> <out.txt> [dr_um = 0.075] [length_mm = 0.5]
Writes "step t[fs] eps_ny[um] eps_nz[um] sigma_y[um] E_mean[MeV] Q[pC]" every 200 steps after the witness is loaded.
"""
import sys
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser
from fbpic.lpa_utils.bunch import add_particle_bunch_gaussian

gal = int(sys.argv[1])
out = sys.argv[2]
dr = float(sys.argv[3]) * 1e-6 if len(sys.argv) > 3 else 0.075e-6
L = float(sys.argv[4]) * 1e-3 if len(sys.argv) > 4 else 0.5e-3

lam = 0.8e-6
w0l = 2 * np.pi * c / lam
n_c = epsilon_0 * m_e * w0l**2 / e**2
n_e = 0.01 * n_c
dz = 0.05e-6
dt = dz / c
Nz = 320
zmax = 16e-6 - c * 45e-15                 # at t = 45 fs the box is [0, 16 um], as VLPL's window
zmin = zmax - Nz * dz
Nr = int(round(9e-6 / dr))
Nm = 2


def dens_func(z, r):
    """up-ramp 13 -> 20 um, flat after; channel 1 + 0.405 (r / 4 um)^2, cut at 9 um"""
    f = np.clip((z - 13e-6) / 7e-6, 0.0, 1.0)
    rr = np.minimum(r, 9e-6)
    return np.where(z >= 13e-6, f, 0.0) * (1 + 0.405 * (rr / 4e-6) ** 2)


kw = dict(v_comoving=0.99995 * c, use_galilean=True) if gal else {}     # gamma = 100
sim = Simulation(Nz, zmax, Nr, Nr * dr, Nm, dt, zmin=zmin, n_order=-1, use_cuda=True,
                 boundaries={'z': 'open', 'r': 'reflective'}, particle_shape='cubic', **kw)
sim.add_new_species(q=-e, m=m_e, n=n_e, dens_func=dens_func, p_nz=2, p_nr=2, p_nt=8, p_zmin=13e-6)
sim.add_new_species(q=e, m=1836 * m_e, n=n_e, dens_func=dens_func, p_nz=2, p_nr=2, p_nt=8, p_zmin=13e-6)
tau = 10e-15 / np.sqrt(2 * np.log(2))
add_laser_pulse(sim, GaussianLaser(a0=1.0, waist=4e-6, tau=tau, z0=-c * 20e-15, zf=0.0, lambda0=lam))
sim.set_moving_window(v=c)                  # from t = 0, the box shifted back by c 45 fs (as bench/lwfa_fbpic)
n_load = int(round(150e-15 / dt))
sim.step(n_load, show_progress=False)
gamma0 = 1 + 10e6 / 0.51099895e6
bunch = add_particle_bunch_gaussian(sim, -e, m_e, sig_r=0.30e-6, sig_z=0.3e-6, n_emit=0.1e-6, gamma0=gamma0,
                                    sig_gamma=0.0, n_physical_particles=1e-12 / e, n_macroparticles=400_000,
                                    zf=34.3e-6, tf=0.0, initialize_self_field=True)   # tf: offset from now


def moments(sp):
    g = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
    x, y, ux, uy, uz, w = (g(sp.x), g(sp.y), g(sp.ux), g(sp.uy), g(sp.uz), g(sp.w))
    W = w.sum()
    def eps(a, u):
        da, du = a - (w * a).sum() / W, u - (w * u).sum() / W
        return np.sqrt((w * da * da).sum() / W * (w * du * du).sum() / W - ((w * da * du).sum() / W) ** 2)
    gam = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    sy = np.sqrt((w * (y - (w * y).sum() / W) ** 2).sum() / W)
    return eps(x, ux) * 1e6, eps(y, uy) * 1e6, sy * 1e6, (w * (gam - 1)).sum() / W * 0.51099895, W * e * 1e12


nend = int(round((L / c + 20e-15) / dt))    # as VLPL's t_end = 1700 fs for 0.5 mm
with open(out, "w") as f:
    f.write("# step t[fs] eps_ny[um] eps_nz[um] sigma_y[um] E_mean[MeV] Q[pC]  (FBPIC %s, dr = %g um)\n"
            % ("Galilean" if gal else "standard", dr * 1e6))
    n = n_load
    while True:
        ex, ey, sy, E, Q = moments(bunch)
        f.write("%d %.4f %.6e %.6e %.6e %.6e %.4f\n" % (n, n * dt * 1e15, ex, ey, sy, E, Q))
        f.flush()
        if n >= nend:
            break
        k = min(200 - n % 200, nend - n)
        sim.step(k, show_progress=False)
        n += k
