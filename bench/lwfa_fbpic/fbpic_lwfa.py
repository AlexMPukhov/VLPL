"""FBPIC side of the LWFA benchmark (bench/lwfa_fbpic/README.md): the setup of rip.deck in RZ.

python3 bench/lwfa_fbpic/fbpic_lwfa.py [out_dir]      (CPU: about 10 min on 2 cores; GPU: use_cuda=True)
"""
import sys
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser
from fbpic.openpmd_diag import FieldDiagnostic

out = sys.argv[1] if len(sys.argv) > 1 else "out_bench_fbpic"
lam = 0.8e-6
w0 = 2 * np.pi * c / lam
n_c = epsilon_0 * m_e * w0**2 / e**2
n_e = 0.01 * n_c

dz = 0.05e-6                       # = RIP dx; dt = dz / c as RIP's c dt = dx
Nz, zmin, zmax = 256, -12.0e-6, 0.8e-6   # at t = 40 fs the box is [0, 12.8 um], as RIP's window
Nr, rmax = 36, 9.0e-6              # dr = 0.25 um = RIP dy
Nm = 2
dt = dz / c

def dens_func(z, r):
    """linear up-ramp 13 -> 20 um, then flat (absolute z, as RIP's density_function)"""
    f = np.clip((z - 13e-6) / 7e-6, 0.0, 1.0)
    return np.where(z >= 13e-6, f, 0.0)

sim = Simulation(Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin, n_order=-1, use_cuda=False,
                 boundaries={'z': 'open', 'r': 'reflective'}, particle_shape='cubic')
sim.add_new_species(q=-e, m=m_e, n=n_e, dens_func=dens_func, p_nz=2, p_nr=2, p_nt=4, p_zmin=13e-6)
# RIP: E = a0 exp(-2 ln2 (t - tc)^2 / 10 fs^2) sin(...), centre through x = 0 at tc = 20 fs, focus at 0
tau = 10e-15 / np.sqrt(2 * np.log(2))
add_laser_pulse(sim, GaussianLaser(a0=1.0, waist=4e-6, tau=tau, z0=-c * 20e-15, zf=0.0, lambda0=lam))
sim.set_moving_window(v=c)
sim.diags = [FieldDiagnostic(period=300, fldobject=sim.fld, comm=sim.comm, write_dir=out, fieldtypes=['E'])]
sim.step(1801)
