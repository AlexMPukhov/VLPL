# Smilei side of the TNSA benchmark against VLPL (bench/tnsa/README.md, rip.deck). Same physics as rip.deck:
#   lambda0 = 0.8 um, Gaussian pulse a0 = 10 at the focus, 30 fs FWHM of the intensity, w0 = 3 um, E_y (in the plane),
#   normal incidence, focus on the foil surface; C6+ foil n_e = 40 n_c, 1 um, exponential pre-plasma L = 0.08 um cut at
#   5 L; hydrogen contaminant (protons + electrons, 10 n_c) 0.1 um on the rear; cold, regular lattice, order 2.
# Smilei box = RIP box between its absorbers: x' = x - 2 um in [0, 26]; y periodic 40 um (the foil fills all y).
# Clock: Smilei t = RIP t (the pulse peak crosses x' = 0.5 at 60 fs and reaches the surface x' = 5.5 at 76.68 fs).
# Environment variables:
#   TNSA_RES   = code (dx = lambda/32, dy = lambda/16, 2 x 2 per cell) | bench (lambda/64 x lambda/32, 4 x 4) |
#                fine (lambda/128 x lambda/64, 4 x 4)
#   TNSA_T     = simulation time in fs (default 200)
#   TNSA_A0    = a0 parameter of LaserGaussian2D (default 10; Smilei's injection at normal incidence gives 0.989 of it at
#                the focus in 3D, bench/hhg_oblique/README.md: check with the vacuum run, TNSA_VACUUM = 1)
#   TNSA_VACUUM = 1: no particles
import os
from math import pi, sqrt

RES    = os.environ.get("TNSA_RES", "code")
TFS    = float(os.environ.get("TNSA_T", "200"))
A0     = float(os.environ.get("TNSA_A0", "10"))
VACUUM = os.environ.get("TNSA_VACUUM", "0") == "1"

lam = 0.8                         # um
um  = 2.*pi/lam                   # 1 um in units of c/omega0
fs  = 2.*pi*0.299792458/lam       # 1 fs in units of 1/omega0

if RES == "code":
    dx, dy, ppc, npatch = lam/32, lam/16, 2, [16, 32]
elif RES == "bench":
    dx, dy, ppc, npatch = lam/64, lam/32, 4, [32, 64]
elif RES == "fine":
    dx, dy, ppc, npatch = lam/128, lam/64, 4, [64, 64]
else:
    raise ValueError("TNSA_RES")
nx, ny = round(26.0/dx), round(40.0/dy)

x_surf = 5.5                      # um, front of the plateau (RIP x = 7.5)
d_foil, L, ext = 1.0, 0.08, 0.4   # plateau, pre-plasma scale length and extent (um)
d_cont = 0.1                      # contaminant layer (um)
n_e, n_h = 40., 10.               # n_c
t_surf = 60. + 5.0/0.299792458    # fs, peak on the surface
t_inj = t_surf - x_surf/0.299792458   # fs, peak crosses x' = 0
fwhm_field = 30.*sqrt(2.)         # fs, FWHM of the field envelope (30 fs of the intensity)

dt = 0.95/sqrt(1./(dx*um)**2 + 1./(dy*um)**2)
t_out = 50.*fs                    # particle dumps every 50 fs (as RIP: diagnostic protons.every)
every_out = max(1, round(t_out/dt))

Main(
    geometry = "2Dcartesian",
    interpolation_order = 2,
    cell_length = [dx*um, dy*um],
    grid_length = [nx*dx*um, ny*dy*um],
    number_of_patches = npatch,
    timestep = dt,
    simulation_time = every_out*dt*round(TFS*fs/(every_out*dt)) + 0.5*dt,   # ends on a particle dump
    EM_boundary_conditions = [["silver-muller"], ["periodic"]],
    print_every = 200,
    random_seed = 0,
)

LoadBalancing(initial_balance = True, every = 40)

LaserGaussian2D(
    box_side = "xmin",
    a0 = A0,
    omega = 1.,
    focus = [x_surf*um, 20.*um],
    waist = 3.*um,
    incidence_angle = 0.,
    polarization_phi = 0.,        # E along y
    ellipticity = 0.,
    time_envelope = tgaussian(start = 0., duration = 2.*t_inj*fs, fwhm = fwhm_field*fs, center = t_inj*fs),
)

import numpy as np
def foil(x, y):
    s = x/um - x_surf
    n = np.where(s < 0., n_e*np.exp(np.minimum(s, 0.)/L), n_e)
    return np.where((s >= -ext) & (s < d_foil), n, 0.)
def contaminant(x, y):
    s = x/um - x_surf - d_foil
    return np.where((s >= 0.) & (s < d_cont), n_h, 0.)

if not VACUUM:
    for name, q, m, prof, scale in (("electron", -1., 1., foil, 1.), ("carbon", 6., 22032., foil, 1./6.),
                                    ("h_electron", -1., 1., contaminant, 1.), ("proton", 1., 1836., contaminant, 1.)):
        Species(
            name = name,
            position_initialization = "regular",
            regular_number = [ppc, ppc],
            momentum_initialization = "cold",
            particles_per_cell = ppc*ppc,
            mass = m,
            charge = q,
            number_density = (lambda f, c: (lambda x, y: c*f(x, y)))(prof, scale),
            pusher = "higueracary",
            boundary_conditions = [["remove", "remove"], ["periodic", "periodic"]],
        )

    # every proton with its momentum (u = gamma v / c, Smilei's p / (m_s c)) and weight: spectrum.py
    DiagTrackParticles(species = "proton", every = every_out, attributes = ["x", "y", "px", "py", "pz", "w"])   # p in m_e c

DiagScalar(every = 50, vars = ["Utot", "Ukin", "Uelm", "Uexp", "Ubal", "Uelm_bnd", "Ukin_bnd",
                               "Ukin_proton", "Ukin_carbon", "Ukin_electron", "Ukin_h_electron"])
DiagProbe(every = 5, origin = [x_surf*um, 20.*um], fields = ["Ex", "Ey", "Bz"])   # incident field at the focus (vacuum run)
DiagProbe(every = every_out, origin = [0., 20.*um], corners = [[nx*dx*um, 20.*um]], number = [nx], fields = ["Ex", "Ey", "Bz"])
