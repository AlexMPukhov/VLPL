"""FBPIC side of the bubble benchmark (bench/lwfa_bubble/README.md): the setup of rip.deck in RZ.

python3 bench/lwfa_bubble/fbpic_bubble.py [out_dir] [--cuda] [--steps N] [--period P] [--rmax R_um]
Mobile protons as in VLPL. --steps / --period / --rmax shrink the run for a smoke test.
"""
import argparse
import numpy as np
from scipy.constants import c, e, m_e, m_p, epsilon_0
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser
from fbpic.openpmd_diag import FieldDiagnostic, ParticleChargeDensityDiagnostic, ParticleDiagnostic

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="out_bubble_fbpic")
ap.add_argument("--cuda", action="store_true")
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--period", type=int, default=600)
ap.add_argument("--rmax", type=float, default=20.0, help="r_max in um (RIP: half the periodic width)")
ap.add_argument("--no-filter", action="store_true")
ap.add_argument("--shape", default="cubic")
a = ap.parse_args()

lam = 0.8e-6
w0 = 2 * np.pi * c / lam
n_c = epsilon_0 * m_e * w0**2 / e**2
n_e = 0.005 * n_c

dz = 0.05e-6                       # = RIP dx; dt = dz / c as RIP's c dt = dx
# laser centre 12 um = 3.3 sigma (field rms tau / sqrt 2 = 12 fs) behind the front edge; at t = 100 fs the box
# is [0, 30 um], as VLPL's window (window.start = 100 fs)
Nz, zmin, zmax = 600, -29.9792458e-6, 0.0207542e-6
dr = 0.25e-6                       # = RIP dy
rmax = a.rmax * 1e-6
Nr = int(round(rmax / dr))
Nm = 2
dt = dz / c

def dens_func(z, r):
    """linear up-ramp 32 -> 52 um, then flat (absolute z, as RIP's density_function)"""
    f = np.clip((z - 32e-6) / 20e-6, 0.0, 1.0)
    return np.where(z >= 32e-6, f, 0.0)

sim = Simulation(Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin, n_order=-1, use_cuda=a.cuda,
                 boundaries={'z': 'open', 'r': 'reflective'}, particle_shape=a.shape, filter_currents=not a.no_filter)
elec = sim.add_new_species(q=-e, m=m_e, n=n_e, dens_func=dens_func, p_nz=2, p_nr=2, p_nt=8, p_zmin=32e-6)
prot = sim.add_new_species(q=e, m=m_p, n=n_e, dens_func=dens_func, p_nz=2, p_nr=2, p_nt=8, p_zmin=32e-6)
# RIP: E = a0 exp(-2 ln2 (t - tc)^2 / 20 fs^2) sin(...), centre through x = 0 at tc = 40 fs, focus at 0
tau = 20e-15 / np.sqrt(2 * np.log(2))
add_laser_pulse(sim, GaussianLaser(a0=4.0, waist=7.2e-6, tau=tau, z0=-c * 40e-15, zf=0.0, lambda0=lam))
sim.set_moving_window(v=c)
sim.diags = [FieldDiagnostic(period=a.period, fldobject=sim.fld, comm=sim.comm, write_dir=a.out, fieldtypes=['E']),
             ParticleChargeDensityDiagnostic(period=a.period, sim=sim, species={"electrons": elec}, write_dir=a.out),
             # injected electrons (compare.py cuts at gamma >= 20, as VLPL's diagnostic block)
             ParticleDiagnostic(period=a.period, species={"electrons": elec}, select={"uz": [15., None]},
                                comm=sim.comm, write_dir=a.out, particle_data=["position", "momentum", "weighting"])]
sim.step(a.steps + 1)
