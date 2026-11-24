# Smilei side of the 3D oblique-incidence HHG benchmark against VLPL (rip.deck in $WORK/hhg_oblique).
# Parameters chosen by Alexander (2026-11-01), identical to rip.deck:
#   lambda0 = 0.8 um, p-polarised Gaussian (E in x-y), a0 = 10 (production) or 2 (code check) at focus, 13 fs FWHM of the intensity, w0 = 2 um,
#   beam at +45 deg to +x (towards +y), focus at the foil hit point P; foil n0 (100 n_c production, 6 n_c code check), 0.6 um thick along the normal
#   n = (cos 22.5, sin 22.5, 0), exponential pre-plasma L = 0.08 um cut at 5 L, |y - yP| <= 4 um, all z;
#   electrons + protons (m = 1836), cold, 2x2x1 particles per cell on a regular lattice, interpolation order 2.
# Smilei box = RIP box cut to x in [18.5, 27.5] um (x' = x - 18.5); y periodic, z periodic.
# Clock: Smilei t = RIP t (the pulse peak reaches P at t_P = 43 fs + 5.657 um / c = 61.87 fs).
# Environment variables:
#   HHG_RES    = code (code check: dx = lambda/32, dy = dz = lambda/16) | x64, x16 (code with dx = lambda/64, lambda/16) | low (dx = lambda/64, dy = dz = lambda/32) | check (lambda/16 vacuum check, field probes near P)
#   HHG_VACUUM = 1: no particles (incident field only), 0: foil
#   HHG_A0     = laser a0 parameter of LaserGaussian3D (default 2)
#   HHG_N0     = foil density in n_c (default 6)
# Code check (Alexander, 2026-11-01): n0 = 6 n_c, a0 = 2, resolution code; the production case is n0 = 100, a0 = 10, low.
import os
from math import pi, cos, sin, sqrt, log
import numpy as np

RES    = os.environ.get("HHG_RES", "code")
VACUUM = os.environ.get("HHG_VACUUM", "0") == "1"
A0     = float(os.environ.get("HHG_A0", "2"))
N0     = float(os.environ.get("HHG_N0", "6"))

lam = 0.8                         # um
um  = 2.*pi/lam                   # 1 um in units of c/omega0
fs  = 2.*pi*0.299792458/lam       # 1 fs in units of 1/omega0

if RES == "low":
    dx, dy, dz = lam/64, lam/32, lam/32
    npatch = [16, 32, 16]
    T = 115.*fs
elif RES == "code":
    dx, dy, dz = lam/32, lam/16, lam/16
    npatch = [8, 16, 8]
    T = 115.*fs
elif RES == "x64":                # convergence along x (2026-11-02): dx = lambda/64, dy = dz = lambda/16
    dx, dy, dz = lam/64, lam/16, lam/16
    npatch = [16, 16, 8]
    T = 115.*fs
elif RES == "x16":                # convergence along x (2026-11-02): dx = dy = dz = lambda/16
    dx, dy, dz = lam/16, lam/16, lam/16
    npatch = [4, 16, 8]
    T = 115.*fs
elif RES == "check":
    dx = dy = dz = lam/16
    npatch = [4, 16, 8]
    T = 80.*fs
else:
    raise ValueError("HHG_RES")
nx, ny, nz = round(9.0/dx), round(28.8/dy), round(10.0/dz)
Ly = ny*dy                        # 28.8 um (RIP: 28.5 um; changed so that the patches divide the grid)

xP, yP, zP = 6.0, 21.5, 5.0       # um (RIP: 24.5, 21.5, 5)
tP = 43. + 5.657/0.299792458      # fs, peak at P
ang = pi/4                        # beam direction (cos 45, sin 45, 0)
dist = xP/cos(ang)                # path from x' = 0 to P along the axis, um
t_inj = tP - dist/0.299792458     # fs, peak crosses x' = 0 on the axis
fwhm_field = 13.*sqrt(2.)         # fs, FWHM of the field envelope (13 fs of the intensity)

Main(
    geometry = "3Dcartesian",
    interpolation_order = 2,
    cell_length = [dx*um, dy*um, dz*um],
    grid_length = [nx*dx*um, Ly*um, nz*dz*um],
    number_of_patches = npatch,
    timestep_over_CFL = 0.95,
    simulation_time = T,
    EM_boundary_conditions = [["silver-muller"], ["periodic"], ["periodic"]],
    print_every = 100,
    random_seed = 0,
)

LoadBalancing(initial_balance = True, every = 40)

LaserGaussian3D(
    box_side = "xmin",
    a0 = A0,
    omega = 1.,
    focus = [xP*um, yP*um, zP*um],
    waist = 2.*um,
    incidence_angle = [0., float(os.environ.get("HHG_ANGLE", str(ang)))],
    polarization_phi = float(os.environ.get("HHG_POLPHI", "0")),
    ellipticity = 0.,
    time_envelope = tgaussian(start = 0., duration = 2.*t_inj*fs, fwhm = fwhm_field*fs, center = t_inj*fs),
)

# ---- foil
cn, sn = cos(pi/8), sin(pi/8)
Lr, d = 0.08, 0.6
def density(x, y, z):
    xu, yu = x/um, y/um
    s = (xu - xP)*cn + (yu - yP)*sn
    n = np.where(s < 0., N0*np.exp(np.minimum(s, 0.)/Lr), N0)
    inside = (s >= -5.*Lr) & (s < d) & (np.abs(yu - yP) <= 4.)
    return np.where(inside, n, 0.)

if not VACUUM:
    for name, q, m in (("electron", -1., 1.), ("proton", 1., 1836.)):
        Species(
            name = name,
            position_initialization = "regular",
            regular_number = [2, 2, 1],
            momentum_initialization = "cold",
            particles_per_cell = 4,
            mass = m,
            charge = q,
            number_density = density,
            pusher = "higueracary",
            boundary_conditions = [["remove", "remove"], ["periodic", "periodic"], ["periodic", "periodic"]],
        )

# ---- diagnostics
DiagScalar(every = 50, vars = ["Utot", "Ukin", "Uelm", "Uexp", "Ubal", "Uelm_bnd", "Ukin_bnd"])

if RES in ("low", "code", "x64", "x16"):
    # plane x' = 0.5 um, every 2nd point in y and z (spacing 2 dy, 2 dz; periodic: last row not repeated)
    DiagProbe(
        every = 1,
        origin = [0.5*um, 0., 0.],
        corners = [[0.5*um, (Ly - 2*dy)*um, 0.], [0.5*um, 0., (nz*dz - 2*dz)*um]],
        number = [ny//2, nz//2],
        fields = ["Ey", "Ez", "By", "Bz"],
    )
    # point P (a0 and timing of the incident pulse in the vacuum run)
    DiagProbe(every = 1, origin = [xP*um, yP*um, zP*um], fields = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"])
else:
    F = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
    DiagProbe(every = 1, origin = [xP*um, yP*um, zP*um], fields = F)                     # 0: point P
    e = np.array([-sin(ang), cos(ang), 0.])                                              # transverse in x-y
    a, b = np.array([xP, yP, zP]) - 5*e, np.array([xP, yP, zP]) + 5*e
    DiagProbe(every = 1, origin = list(a*um), corners = [list(b*um)], number = [201], fields = F)   # 1
    DiagProbe(every = 1, origin = [xP*um, yP*um, 0.], corners = [[xP*um, yP*um, 10.*um]], number = [201], fields = F)  # 2
    DiagProbe(every = 20, origin = [0., 0., zP*um], corners = [[9.*um, 0., zP*um], [0., Ly*um, zP*um]],
              number = [181, 577], fields = F)                                            # 3: plane z = 5
