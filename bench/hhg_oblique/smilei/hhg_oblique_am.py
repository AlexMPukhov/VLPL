# Smilei AMcylindrical twin of hhg_oblique.py (3D): the oblique HHG code check (6 n_c, a0 = 2, lambda/32 x lambda/16) with
# azimuthal modes, to test how many modes the oblique geometry needs (Alexander, 2026-11-02). The axis x passes through
# the foil hit point P along the specular direction (-x), so the reflected beam and its harmonics travel along the axis;
# the incident beam (45 deg to the axis) and the tilted foil (normal 22.5 deg off the axis) need many modes.
# The laser at xmin is the boundary field of the 3D run (LaserGaussian3D with incidence_angle [0, 45 deg], the same
# By, Bz(y, z, t) on the plane x = 0), projected on the modes: B_m = (1/pi) int B e^{i m theta} dtheta (m >= 1, half
# of it for m = 0), Smilei's convention F = Re sum_m F_m e^{-i m theta}. The slowly varying envelope of each mode is
# tabulated at start (r every dr/2, t every 0.5 fs) and interpolated.
# Environment variables: HHG_MODES (number_of_AM, default 16), HHG_VACUUM (1: no particles), HHG_A0 (2.4502 as the 3D
# foil run), HHG_NTHETA (particles in theta per cell, default 4 HHG_MODES).
import os
from math import pi, cos, sin, sqrt, log, atan
import numpy as np

M      = int(os.environ.get("HHG_MODES", "16"))
VACUUM = os.environ.get("HHG_VACUUM", "0") == "1"
A0     = float(os.environ.get("HHG_A0", "2.4502"))
NTH    = int(os.environ.get("HHG_NTHETA", str(4*M)))
N0     = 6.

lam = 0.8
um  = 2.*pi/lam
fs  = 2.*pi*0.299792458/lam
dx, dr = lam/32, lam/16
nx, nr = round(9.0/dx), round(16.0/dr)       # x' in [0, 9] um as the 3D box, r up to 16 um (the beam at xmin to 1e-3)
T = 115.*fs

xP = 6.0                                      # um, P on the axis
tP = 43. + 5.657/0.299792458
ang = pi/4
dist = xP/cos(ang)
t_inj = tP - dist/0.299792458
fwhm_field = 13.*sqrt(2.)

Main(
    geometry = "AMcylindrical",
    number_of_AM = M,
    interpolation_order = 2,
    cell_length = [dx*um, dr*um],
    grid_length = [nx*dx*um, nr*dr*um],
    number_of_patches = [8, 16],
    timestep_over_CFL = 0.95,
    simulation_time = T,
    EM_boundary_conditions = [["silver-muller"], ["buneman"]],
    print_every = 100,
    random_seed = 0,
)
LoadBalancing(initial_balance = True, every = 40)

# ---- the boundary field of LaserGaussian3D (pyprofiles.py), focus at P = (xP, 0, 0) in the frame of the axis
omega = 1.
w0 = 2.*um
Zr = omega*w0**2/2.
f0 = xP*um
cz, sz = cos(ang), sin(ang)
ampY, ampZ = 0., A0*omega*cz                  # polarization_phi = 0: B along z, times cos(incidence) as in 3D
phase_offset = -(omega*dist*um - atan(dist*um/Zr))
sig_t = (0.5*fwhm_field*fs)**2/log(2.)
t_end_inj = 2.*t_inj*fs

def envelope(y, z, t):
    """complex envelopes S (B = Im(e^{i omega t} S)) of By, Bz at the plane x = 0; y, z, t arrays (code units)"""
    X = (-f0*cz + y*sz)/Zr
    Y = (f0*sz + y*cz)/w0
    Z = z/w0
    iw2 = 1./(1. + X**2)
    sp = np.sqrt(iw2)*np.exp(-(Y**2 + Z**2)*iw2)
    Yp, Zp = Y*w0/Zr, Z*w0/Zr
    ph = omega*Zr*X*(1. + 0.5*(Yp**2 + Zp**2)/(1. + X**2)) - np.arctan(X) - phase_offset
    tt = t - ph/omega
    g = np.where((tt > 0.) & (tt < t_end_inj), np.exp(-(tt - t_inj*fs)**2/sig_t), 0.)
    S = sp*g*np.exp(-1j*ph)
    return ampY*S, ampZ*S

NT = 256                                      # azimuthal quadrature
th = 2*pi*np.arange(NT)/NT
r_tab = 0.5*dr*um*np.arange(2*nr + 8)
t_tab = np.arange(0., t_end_inj + 1.*fs, 0.5*fs)
ms = np.arange(-(M - 1), M)
Gr = np.zeros((len(ms), len(r_tab), len(t_tab)), np.complex64)
Gt = np.zeros_like(Gr)
c, s = np.cos(th), np.sin(th)
for it, t in enumerate(t_tab):
    R = r_tab[:, None]
    Sy, Sz = envelope(R*c, R*s, t)
    Sr = Sy*c + Sz*s
    St = -Sy*s + Sz*c
    Fr = 2.*np.fft.ifft(Sr, axis=1)           # (1/pi) int S e^{i m theta} dtheta, index m mod NT
    Ft = 2.*np.fft.ifft(St, axis=1)
    Gr[:, :, it] = Fr[:, ms % NT].T
    Gt[:, :, it] = Ft[:, ms % NT].T
dr_tab, dt_tab = r_tab[1], t_tab[1]

def make(G, m):
    gp, gm = G[m + M - 1], G[-m + M - 1]      # G_m, G_{-m}
    half = 0.5 if m == 0 else 1.
    nr_t, nt_t = gp.shape
    def f(r, t):
        if t >= t_tab[-1] or t < 0.:
            return 0j
        a = r/dr_tab; i = int(a); a -= i
        b = t/dt_tab; k = int(b); b -= k
        if i >= nr_t - 1:
            return 0j
        P = (1-a)*((1-b)*gp[i, k] + b*gp[i, k+1]) + a*((1-b)*gp[i+1, k] + b*gp[i+1, k+1])
        Q = (1-a)*((1-b)*gm[i, k] + b*gm[i, k+1]) + a*((1-b)*gm[i+1, k] + b*gm[i+1, k+1])
        e = complex(cos(omega*t), sin(omega*t))
        return half*(e*complex(P) - e.conjugate()*complex(Q).conjugate())/2j
    return f

prof = []
for m in range(M):
    prof += [make(Gr, m), make(Gt, m)]
Laser(box_side = "xmin", space_time_profile_AM = prof)

# ---- foil, the 3D foil around the axis: |y| <= 4 um, |z| <= 5 um (the 3D box is 10 um periodic in z)
cn, sn = cos(pi/8), sin(pi/8)
Lr, d = 0.08, 0.6
def density(x, y, z):
    xu, yu, zu = x/um, y/um, z/um
    s_ = (xu - xP)*cn + yu*sn
    n = np.where(s_ < 0., N0*np.exp(np.minimum(s_, 0.)/Lr), N0)
    inside = (s_ >= -5.*Lr) & (s_ < d) & (np.abs(yu) <= 4.) & (np.abs(zu) <= 5.)
    return np.where(inside, n, 0.)

# Smilei's AM profiles depend on (x, r) only: the tilted foil is loaded as an array, the regular lattice of the 3D
# run's density times Smilei's own AM weights, w = n 2 pi dx dr r / (particles per cell) (ParticleCreator::createWeight)
def foil_particles():
    ppc = [2, 2, NTH]
    ix = np.arange(round(4.0/dx), round(8.5/dx))                  # x' in [4, 8.5] um holds the foil (|y| <= 4 um)
    jr = np.arange(0, round(6.5/dr))                              # r <= 6.5 um (|y| <= 4, |z| <= 5)
    a = (np.arange(ppc[0]) + 0.5)/ppc[0]; b = (np.arange(ppc[1]) + 0.5)/ppc[1]
    c = (np.arange(ppc[2]) + 0.5)/ppc[2]*2*pi
    xs = ((ix[:, None] + a[None, :])*dx).ravel()                  # um
    rs = ((jr[:, None] + b[None, :])*dr).ravel()
    X, Rr, Th = np.meshgrid(xs, rs, c, indexing="ij")
    Y, Z = Rr*np.cos(Th), Rr*np.sin(Th)
    n = density(X*um, Y*um, Z*um)
    k = n > 0
    w = n[k]*2*pi*(dx*um)*(dr*um)*(Rr[k]*um)/(ppc[0]*ppc[1]*ppc[2])
    return np.vstack([X[k]*um, Y[k]*um, Z[k]*um, w])

if not VACUUM:
    P0 = foil_particles()
    for name, q, m_ in (("electron", -1., 1.), ("proton", 1., 1836.)):
        Species(
            name = name,
            position_initialization = P0,
            momentum_initialization = "cold",
            mass = m_,
            charge = q,
            pusher = "higueracary",
            boundary_conditions = [["remove", "remove"], ["remove", "remove"]],
        )

DiagScalar(every = 50, vars = ["Utot", "Ukin", "Uelm", "Uexp", "Ubal", "Uelm_bnd", "Ukin_bnd"])
# plane x' = 0.5 um, |y|, |z| < 10 um every 0.1 um (2 dr), as the 3D probe (every 2nd point)
Lp, npnt = 10., 200
DiagProbe(every = 1, origin = [0.5*um, -Lp*um, -Lp*um],
          corners = [[0.5*um, (Lp - 2*Lp/npnt)*um, -Lp*um], [0.5*um, -Lp*um, (Lp - 2*Lp/npnt)*um]],
          number = [npnt, npnt], fields = ["Ey", "Ez", "By", "Bz"])
DiagProbe(every = 1, origin = [xP*um, 0., 0.], fields = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"])
