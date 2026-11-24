"""Absorbing layers at the y edges for RIP-CC: CPML against a matched-conductivity sponge (2D, vacuum).

Units c = 1. Grid as research/irip_rz/nci2d.py: Ex, Ez, By at (i, j); Ey, Bz, Bx at (i, j+1/2); D = dx = dt
(the transport moves one cell per step), periodic in x, layers of L cells at both y edges with zero fields
beyond (PEC behind the layer).

CPML (Roden & Gedney 2000): every transverse derivative of the step, d/dy F, becomes d/dy F / kappa + psi,
psi <- b psi + a d/dy F, with its own psi per call site (predictor: dy Bz, dy Ez; transport sources:
dy Ex*, dy Bx*; corrector: dy <Bz>, dy <Ez>), sigma, kappa, alpha graded in the layer and evaluated where
the derivative sits. Sponge: E and B both damped by exp(-sigma dt) per step in the layer (sigma = sigma*).

Reflection: a Gaussian wave packet (TE: Ey, Bz, Ex or TM: Ez, By, Bx) at angle phi from the x axis runs
into the upper layer; the field in the interior is compared with a reference run in a box tall enough that
nothing comes back in time. Stability: random fields over many steps, the energy must not grow.

python3 research/pml/pml2d.py [reflect|stable|all]
"""
import sys
import numpy as np


def dyp(F, dy):   # (F[j+1] - F[j]) / dy at j+1/2, zero beyond the upper edge
    G = np.zeros_like(F)
    G[:, :-1] = F[:, 1:] - F[:, :-1]
    G[:, -1] = -F[:, -1]
    return G / dy


def dym(F, dy):   # (F[j] - F[j-1]) / dy at j, zero beyond the lower edge
    G = np.zeros_like(F)
    G[:, 1:] = F[:, 1:] - F[:, :-1]
    G[:, 0] = F[:, 0]
    return G / dy


def mux(F):
    return 0.5 * (F + np.roll(F, -1, 0))


class Layer:
    """Graded CPML / sponge coefficients at the nodes (j) and the half nodes (j+1/2) of y."""

    def __init__(self, ny, dy, dt, L, kind, R0=1e-6, m=3, kmax=1.0, amax=0.0):
        self.kind = kind
        L = max(L, 1)                                      # L = 0 (no layer): coefficients unused
        Ly = L * dy
        smax = (m + 1) * np.log(1 / R0) / (2 * Ly)          # c = 1, eps0 = 1

        def prof(y):                                       # depth into the layer / L, 0 inside
            d = np.maximum(np.maximum(L * dy - y, y - (ny - L) * dy), 0) / Ly
            return d

        self.c = {}
        for key, y in (("n", np.arange(ny) * dy), ("h", (np.arange(ny) + 0.5) * dy)):
            d = prof(y)
            s = smax * d ** m
            k = 1 + (kmax - 1) * d ** m
            a = amax * (1 - d)
            b = np.exp(-(s / k + a) * dt)
            with np.errstate(invalid="ignore", divide="ignore"):
                aa = np.where(s > 0, s / (s * k + k * k * a) * (b - 1), 0.0)
            self.c[key] = (b[None, :], aa[None, :], 1 / k[None, :], np.exp(-s * dt)[None, :])
        self.psi = {}

    def d(self, name, G, where):
        """CPML derivative of call site `name`: G / kappa + psi, psi <- b psi + a G (G = plain derivative)."""
        if self.kind != "cpml":
            return G
        b, a, ik, _ = self.c[where]
        p = self.psi.get(name)
        p = b * (np.zeros_like(G) if p is None else p) + a * G
        self.psi[name] = p
        return G * ik + p


def step(f, D, dy, lay):
    Ex, Ey, Ez, Bx, By, Bz = f
    # Ex at j, Bz at j+1/2: dym(Bz) at j; Ez at j, Bx at j+1/2: dyp(Ez) at j+1/2
    Exs = Ex + 0.5 * D * lay.d("p_bz", dym(Bz, dy), "n")
    Bxs = Bx - 0.5 * D * lay.d("p_ez", dyp(Ez, dy), "h")
    dEx = lay.d("t_ex", dyp(mux(Exs), dy), "h")            # source of Ey, Bz (j+1/2)
    dBx = lay.d("t_bx", dym(mux(Bxs), dy), "n")            # source of Ez, By (j)
    Tp, Tm = Ey + Bz, Ey - Bz
    Tp1 = np.roll(Tp + D * dEx, 1, 0)
    Tm1 = np.roll(Tm, -1, 0) - D * dEx
    Up, Um = Ez - By, Ez + By
    Up1 = np.roll(Up - D * dBx, 1, 0)
    Um1 = np.roll(Um, -1, 0) - D * dBx
    Ey1, Bz1 = 0.5 * (Tp1 + Tm1), 0.5 * (Tp1 - Tm1)
    Ez1, By1 = 0.5 * (Up1 + Um1), 0.5 * (Um1 - Up1)
    Ex1 = Ex + D * lay.d("c_bz", dym(0.5 * (Bz + Bz1), dy), "n")
    Bx1 = Bx - D * lay.d("c_ez", dyp(0.5 * (Ez + Ez1), dy), "h")
    f1 = [Ex1, Ey1, Ez1, Bx1, By1, Bz1]
    if lay.kind == "sponge":
        en, eh = lay.c["n"][3], lay.c["h"][3]
        # Ex, Ez, By at nodes; Ey, Bz, Bx at half nodes
        f1 = [Ex1 * en, Ey1 * eh, Ez1 * en, Bx1 * eh, By1 * en, Bz1 * eh]
    return f1


def packet(nx, ny, dx, dy, phi, lam, w, y0, pol):
    """Gaussian wave packet with k = k0 (cos phi, sin phi), centred at (nx dx / 2, y0), width w."""
    k0 = 2 * np.pi / lam
    kx, ky = k0 * np.cos(phi), k0 * np.sin(phi)
    def g(xo, yo):
        x = (np.arange(nx)[:, None] + xo) * dx
        y = (np.arange(ny)[None, :] + yo) * dy
        X = x - nx * dx / 2
        Y = y - y0
        return np.exp(-(X ** 2 + Y ** 2) / w ** 2) * np.cos(kx * X + ky * Y)
    z = np.zeros((nx, ny))
    if pol == "TE":   # E in the plane: E = (-sin phi, cos phi) a, Bz = a
        Ex = -np.sin(phi) * g(0, 0)
        Ey = np.cos(phi) * g(0, 0.5)
        Bz = g(0, 0.5)
        return [Ex, Ey, z.copy(), z.copy(), z.copy(), Bz]
    # TM: Ez = a, B = k x E / k: Bx = sin phi a, By = -cos phi a
    Ez = g(0, 0)
    Bx = np.sin(phi) * g(0, 0.5)
    By = -np.cos(phi) * g(0, 0)
    return [z.copy(), z.copy(), Ez, Bx, By, z.copy()]


def reflection(kind, phi_deg, pol, dyr=2.0, L=24, **kw):
    lam = 16.0                         # dx = 1
    dx, dy = 1.0, dyr
    D = dx
    nx = 256
    ny_in = int(round(160 / dy))       # interior height 160 dx
    w = 16.0
    y0 = 0.5 * ny_in * dy              # packet in the middle of the interior
    phi = np.radians(phi_deg)
    # long enough to cross the layer to the wall and come back into the interior
    steps = int(1.1 * (0.5 * ny_in * dy + 2 * L * dy + 3 * w) / max(np.sin(phi), 0.25))
    steps = min(steps, 1200)
    # run with the layer
    ny = ny_in + 2 * L
    lay = Layer(ny, dy, D, L, kind, **kw)
    f = packet(nx, ny, dx, dy, phi, lam, w, y0 + L * dy, pol)
    # reference: a box so tall that nothing returns within `steps`
    Lr = int(steps * D / dy) + 8
    nyr = ny_in + 2 * Lr
    ref = Layer(nyr, dy, D, 0, "none")
    fr = packet(nx, nyr, dx, dy, phi, lam, w, y0 + Lr * dy, pol)
    amp = max(np.abs(a).max() for a in f)
    worst = 0.0
    for n in range(steps):
        f = step(f, D, dy, lay)
        fr = step(fr, D, dy, ref)
        if n % 10 == 0:
            e = max(np.abs(a[:, L:L + ny_in] - b[:, Lr:Lr + ny_in]).max() for a, b in zip(f, fr))
            worst = max(worst, e)
    return worst / amp


def stability(kind, steps=20000, dyr=2.0, L=24, **kw):
    rng = np.random.default_rng(3)
    nx, ny_in = 64, 64
    ny = ny_in + 2 * L
    dy = dyr
    lay = Layer(ny, dy, 1.0, L, kind, **kw)
    f = [1e-3 * rng.standard_normal((nx, ny)) for _ in range(6)]
    e0 = sum((a ** 2).sum() for a in f)
    emax, hist = 0.0, []
    for n in range(steps):
        f = step(f, 1.0, dy, lay)
        if n % 500 == 0:
            e = sum((a ** 2).sum() for a in f) / e0
            hist.append(e)
            emax = max(emax, e) if n > 0 else emax
    return hist


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("reflect", "all"):
        for dyr in (2.0, 5.0):
            print("dy = %g dx (lambda = 16 dx), layer 24 cells; max interior error / amplitude" % dyr)
            print("%8s %4s %10s %10s %10s" % ("phi", "pol", "cpml", "cpml k,a", "sponge"))
            for phi in (90, 60, 45, 30, 20):
                for pol in ("TE", "TM"):
                    r1 = reflection("cpml", phi, pol, dyr)
                    r2 = reflection("cpml", phi, pol, dyr, kmax=4.0, amax=0.05)
                    r3 = reflection("sponge", phi, pol, dyr)
                    print("%8d %4s %10.2e %10.2e %10.2e" % (phi, pol, r1, r2, r3), flush=True)
    if what in ("stable", "all"):
        for kind, kw in (("cpml", {}), ("cpml", {"kmax": 4.0, "amax": 0.05}), ("sponge", {})):
            h = stability(kind, **kw)
            print("stability %-6s %-28s energy / initial every 500 steps: %s" % (kind, kw,
                  " ".join("%.1e" % v for v in h[::4])), flush=True)
