"""Beam initialisation and media geometry helpers (lengths in lambda0, c = 1, omega0 = 2 pi)."""
import numpy as np
from box2d import Box2D, Medium

W0 = 2 * np.pi            # laser frequency (lambda0 = 1, c = 1)


def grid(nx, ny, D, dy):
    x = np.arange(nx) * D
    y = np.arange(ny) * dy
    return np.meshgrid(x, y, indexing="ij")


def beam(s, pol, x0, y0, angle_deg, waist, tau, a0=1.0):
    """Gaussian beam pulse (focus at its centre, Gaussian envelope of length tau (1/e^2 half-length) along k).
    Only forward components; fields from E, B = k^ x E per Fourier mode; staggered positions via phase shifts."""
    nx, ny, D, dy = s.nx, s.ny, s.D, s.dy
    X, Y = grid(nx, ny, D, dy)
    th = np.deg2rad(angle_deg)
    kd = np.array([np.cos(th), np.sin(th)])
    Lx, Ly = nx * D, ny * dy
    dxp = (X - x0 + Lx / 2) % Lx - Lx / 2
    dyp_ = (Y - y0 + Ly / 2) % Ly - Ly / 2
    xl = dxp * kd[0] + dyp_ * kd[1]
    yt = -dxp * kd[1] + dyp_ * kd[0]
    g = a0 * np.exp(-(yt / waist) ** 2 - (xl / tau) ** 2 + 1j * W0 * xl)
    G = np.fft.fft2(g)
    kx = 2 * np.pi * np.fft.fftfreq(nx, D)[:, None] * np.ones((1, ny))
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)[None, :] * np.ones((nx, 1))
    k = np.sqrt(kx ** 2 + ky ** 2); k[k == 0] = 1
    G = np.where(kx * kd[0] + ky * kd[1] > 0, G, 0)
    sh = np.exp(1j * ky * dy / 2)                   # value at j + 1/2
    R = lambda H: np.fft.ifft2(H).real
    f = s.f
    if pol == "s":            # Ez, B = k^ x Ez z^ = (ky, -kx)/k Ez
        f[2] += R(G); f[3] += R(G * ky / k * sh); f[4] += R(-G * kx / k)
    else:                     # Bz, E = B x k^ = Bz (-ky, kx)/k
        f[5] += R(G * sh); f[0] += R(-G * ky / k); f[1] += R(G * kx / k * sh)


def fill(nx, ny, D, dy, inside, ox, oy, sub=4):
    """Area fraction of the region inside(x, y) for cells centred at ((i+ox) D, (j+oy) dy)."""
    X, Y = grid(nx, ny, D, dy)
    acc = np.zeros((nx, ny))
    for a in range(sub):
        for b in range(sub):
            acc += inside(X + (ox - 0.5 + (a + 0.5) / sub) * D, Y + (oy - 0.5 + (b + 0.5) / sub) * dy)
    return acc / sub ** 2


def set_medium(med, nx, ny, D, dy, inside, wp, w0=0.0, nu=0.0):
    """Lorentz/Drude medium with plasma frequency wp (density ~ wp^2 scaled by the area fraction)."""
    offs = [(0.0, 0.0), (0.5, 0.5), (0.5, 0.0)]      # Jx at (i, j); Jy (i+1/2, j+1/2); Jz (i+1/2, j)
    for c, (ox, oy) in enumerate(offs):
        fr = fill(nx, ny, D, dy, inside, ox, oy)
        med.wp2[c] = med.wp2[c] + wp ** 2 * fr
        med.w02[c] = np.where(fr > 0, w0 ** 2, med.w02[c])
        med.nu[c] = np.where(fr > 0, nu, med.nu[c])
