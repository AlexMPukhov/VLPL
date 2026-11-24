"""NCI streaming-plasma test (paper Sec. 8.2, task 6) in FBPIC itself, standard or Galilean PSATD, 2 modes.
The VLPL-RZ deck runs/nci_rz.deck in FBPIC units: n = 0.01 n_c at 1 um (1.115e19 cm^-3), c/wp = 1.5915 um,
dz = c dt = 1.0027 um, periodic z (64 cells), reflective r, drift u_z = -gamma_beta along z, u_th = 1e-4
(electrons), protons drifting with them, linear shapes, 2 x 2 x 8 particles per cell, t = 280 / wp.
  python fbpic_nci.py --u 10 --drfac 3 [--galilean] [--no-filter] --out DIR
Writes DIR/energy.txt: step, t [s], field energy per mode W_0, W_1 [J] (every 7 steps, from the spectral grid
interpolated to the real grid; F = F_0 + 2 Re(F_1 e^{-i theta}): W_0 = 2 pi int, W_1 = 4 pi int of (eps0 |E|^2 + |B|^2/mu0)/2 r dr dz)."""
import argparse, os, time
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0, mu_0
from fbpic.main import Simulation

ap = argparse.ArgumentParser()
ap.add_argument("--u", type=float, default=10.0)
ap.add_argument("--drfac", type=float, default=3.0)
ap.add_argument("--galilean", action="store_true")
ap.add_argument("--no-filter", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

lam0 = 1e-6
nc = epsilon_0 * m_e * (2 * np.pi * c / lam0) ** 2 / e ** 2
n_e = 0.01 * nc
dz = 1.0027e-6
dr = a.drfac * dz
Nz = 64
Nr = 16 if a.drfac > 1 else 64
Nm = 2
dt = dz / c
wp = np.sqrt(n_e * e ** 2 / (epsilon_0 * m_e))
N_step = int(1.5e-12 / dt)
gamma = np.sqrt(1 + a.u ** 2)
v = -c * a.u / gamma
kw = dict(n_order=-1, use_cuda=True, boundaries={"z": "periodic", "r": "reflective"},
          filter_currents=not a.no_filter, particle_shape="linear")
if a.galilean:
    kw.update(v_comoving=v, use_galilean=True)
sim = Simulation(Nz, Nz * dz, Nr, Nr * dr, Nm, dt, zmin=0.0, **kw)
pe = sim.add_new_species(q=-e, m=m_e, n=n_e, dens_func=None, p_nz=2, p_nr=2, p_nt=8,
                         uz_m=-a.u, ux_th=1e-4, uy_th=1e-4, uz_th=1e-4)
pp = sim.add_new_species(q=e, m=1836 * m_e, n=n_e, dens_func=None, p_nz=2, p_nr=2, p_nt=8, uz_m=-a.u)
K_e = n_e * np.pi * (Nr * dr) ** 2 * Nz * dz * (gamma - 1) * m_e * c ** 2


def field_energy():
    W = np.zeros(Nm)
    for m in range(Nm):
        g = sim.fld.interp[m]
        r = g.r
        s = 0.0
        for name in ("Er", "Et", "Ez"):
            arr = getattr(g, name)
            arr = arr.get() if hasattr(arr, "get") else arr
            s += epsilon_0 * np.sum(np.abs(arr) ** 2 * r[None, :])
        for name in ("Br", "Bt", "Bz"):
            arr = getattr(g, name)
            arr = arr.get() if hasattr(arr, "get") else arr
            s += np.sum(np.abs(arr) ** 2 * r[None, :]) / mu_0
        W[m] = 0.5 * s * dr * dz * (2 * np.pi if m == 0 else 4 * np.pi)   # F = F_0 + 2 Re(F_m e^{-im theta})
    return W


rows = []
t0 = time.time()
for n in range(0, N_step + 1, 7):
    rows.append([n, sim.time] + list(field_energy()))
    if n < N_step:
        sim.step(min(7, N_step - n), show_progress=False)
np.savetxt(os.path.join(a.out, "energy.txt"), np.array(rows),
           header="FBPIC %s, u = %g, dr = %g dz, filter %s; K_e = %.6e J; wp = %.6e /s; wall %.1f s\nstep t[s] W_0 W_1 [J]"
           % ("Galilean" if a.galilean else "standard", a.u, a.drfac, not a.no_filter, K_e, wp, time.time() - t0))
print("done", a.out, time.time() - t0)
