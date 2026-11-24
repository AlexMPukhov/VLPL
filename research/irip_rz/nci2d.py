"""Numerical Cherenkov (NCI) comparison in 2D3V, periodic: RIP-CC (explicit), box iRIP, FBPIC-like PSATD.

Units c = 1, eps0 = 1, lengths in c/wp, times in 1/wp (wp of the electrons). Neutral e-p plasma streaming
along x (test of Pukhov, JCP 418, 109622, Sec. 7.1). Linear shapes for all solvers.

Grids
  RIP (rip, box): Ex, Ez, By at (i, j); Ey, Bz, Bx at (i, j+1/2); rho at (i+1/2, j);
                  Jx (i, j), Jy (i+1/2, j+1/2), Jz (i+1/2, j); Esirkepov deposition on the charge grid.
  PSATD (FBPIC-like): all fields at nodes (i, j); J at n+1/2 deposited directly at x^{n+1/2};
                  rho at n and n+1; curl-free current correction; exact k (infinite order).
"""
import numpy as np
import sys


class Plasma:
    def __init__(self, nx, ny, dx, dy, ppc, u0, uth, seed=1):
        rng = np.random.default_rng(seed)
        self.nx, self.ny, self.dx, self.dy = nx, ny, dx, dy
        px, py = ppc
        gx = (np.arange(nx * px) + 0.5) * dx / px
        gy = (np.arange(ny * py) + 0.5) * dy / py
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        n = X.size
        w = 1.0 * dx * dy / (px * py)                 # density 1 per species
        self.sp = []
        for q, m in ((-1.0, 1.0), (1.0, 1836.0)):
            u = np.zeros((n, 3)); u[:, 0] = u0
            if q < 0:
                u += uth * rng.standard_normal((n, 3))
            self.sp.append(dict(x=X.ravel().copy(), y=Y.ravel().copy(), u=u, q=q, m=m, w=w))


def lin_w(X):
    i0 = np.floor(X).astype(int); f = X - i0
    return i0, f


def gather(F, x, y, dx, dy, ox, oy):
    nx, ny = F.shape
    i0, fx = lin_w(x / dx - ox); j0, fy = lin_w(y / dy - oy)
    i0 %= nx; j0 %= ny; i1 = (i0 + 1) % nx; j1 = (j0 + 1) % ny
    return (F[i0, j0] * (1 - fx) * (1 - fy) + F[i1, j0] * fx * (1 - fy)
            + F[i0, j1] * (1 - fx) * fy + F[i1, j1] * fx * fy)


def boris(u, E, B, qm, dt):
    um = u + 0.5 * qm * dt * E
    g = np.sqrt(1 + (um ** 2).sum(1))[:, None]
    t = 0.5 * qm * dt * B / g
    s = 2 * t / (1 + (t ** 2).sum(1))[:, None]
    up = um + np.cross(um + np.cross(um, t), s)
    return up + 0.5 * qm * dt * E


def deposit_lin(G, x, y, val, dx, dy, ox, oy):
    nx, ny = G.shape
    i0, fx = lin_w(x / dx - ox); j0, fy = lin_w(y / dy - oy)
    i0 %= nx; j0 %= ny; i1 = (i0 + 1) % nx; j1 = (j0 + 1) % ny
    for ii, jj, ww in ((i0, j0, (1 - fx) * (1 - fy)), (i1, j0, fx * (1 - fy)), (i0, j1, (1 - fx) * fy), (i1, j1, fx * fy)):
        np.add.at(G, (ii, jj), val * ww)


def esirkepov(J, rho1, x0, y0, x1, y1, vz, qw, dx, dy, dt):
    """Linear-shape Esirkepov on the charge grid (nodes ((a+1/2) dx, b dy))."""
    Jx, Jy, Jz = J
    nx, ny = Jx.shape
    X0, X1 = x0 / dx - 0.5, x1 / dx - 0.5
    Y0, Y1 = y0 / dy, y1 / dy
    bx = np.floor(np.minimum(X0, X1)).astype(int)
    by = np.floor(np.minimum(Y0, Y1)).astype(int)
    k = np.arange(3)
    S = lambda X, b: np.maximum(0.0, 1 - np.abs(X[:, None] - (b[:, None] + k[None, :])))
    S0x, S1x, S0y, S1y = S(X0, bx), S(X1, bx), S(Y0, by), S(Y1, by)
    DSx, DSy = S1x - S0x, S1y - S0y
    Wx = DSx[:, :, None] * (S0y + 0.5 * DSy)[:, None, :]
    Wy = (S0x + 0.5 * DSx)[:, :, None] * DSy[:, None, :]
    Wz = (S0x[:, :, None] * S0y[:, None, :] + 0.5 * DSx[:, :, None] * S0y[:, None, :]
          + 0.5 * S0x[:, :, None] * DSy[:, None, :] + DSx[:, :, None] * DSy[:, None, :] / 3)
    jx = -(qw / (dt * dy))[:, None, None] * np.cumsum(Wx, axis=1)     # at charge (k+1/2, l) -> field i = bx+k+1
    jy = -(qw / (dt * dx))[:, None, None] * np.cumsum(Wy, axis=2)     # at charge (k, l+1/2) -> Jy[bx+k, by+l]
    jz = (qw * vz / (dx * dy))[:, None, None] * Wz
    I = (bx[:, None, None] + k[None, :, None]) % nx
    Jj = (by[:, None, None] + k[None, None, :]) % ny
    np.add.at(Jx, ((I + 1) % nx, Jj), jx)
    np.add.at(Jy, (I, Jj), jy)
    np.add.at(Jz, (I, Jj), jz)
    rho = (qw / (dx * dy))[:, None, None] * S1x[:, :, None] * S1y[:, None, :]
    np.add.at(rho1, (I, Jj), rho)


# ---------------------------------------------------------------- RIP field steps
def dyp(F, dy):   # (F[j+1] - F[j]) / dy
    return (np.roll(F, -1, 1) - F) / dy


def dym(F, dy):   # (F[j] - F[j-1]) / dy
    return (F - np.roll(F, 1, 1)) / dy


def rip_transport(f, Exbar, Bxbar, J, D, dy):
    Ex, Ey, Ez, Bx, By, Bz = f
    Jx, Jy, Jz = J
    dEx = dyp(Exbar, dy)                 # at (i+1/2, j+1/2)
    dBx = dym(Bxbar, dy)                 # at (i+1/2, j)
    Tp, Tm = Ey + Bz, Ey - Bz
    Tp1 = np.roll(Tp + D * (dEx - Jy), 1, 0)
    Tm1 = np.roll(Tm, -1, 0) + D * (-dEx - Jy)
    Up, Um = Ez - By, Ez + By            # Up moves +x, Um moves -x
    Up1 = np.roll(Up + D * (-dBx - Jz), 1, 0)
    Um1 = np.roll(Um, -1, 0) + D * (-dBx - Jz)
    Ey1, Bz1 = 0.5 * (Tp1 + Tm1), 0.5 * (Tp1 - Tm1)
    Ez1, By1 = 0.5 * (Up1 + Um1), 0.5 * (Um1 - Up1)
    Ex1 = Ex + D * (dym(0.5 * (Bz + Bz1), dy) - Jx)
    Bx1 = Bx - D * dyp(0.5 * (Ez + Ez1), dy)
    return [Ex1, Ey1, Ez1, Bx1, By1, Bz1]


def mux(F):       # value at i+1/2 from i, i+1
    return 0.5 * (F + np.roll(F, -1, 0))


def step_ripcc(f, J, D, dy):
    Ex, Ey, Ez, Bx, By, Bz = f
    Exs = Ex + 0.5 * D * (dym(Bz, dy) - J[0])
    Bxs = Bx - 0.5 * D * dyp(Ez, dy)
    return rip_transport(f, mux(Exs), mux(Bxs), J, D, dy)


def step_box(f, J, D, dy, check=False):
    Ex, Bx = f[0], f[3]
    R = rip_transport(f, 0.5 * mux(Ex), 0.5 * mux(Bx), J, D, dy)
    nx, ny = Ex.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, D)[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)[None, :]
    den = 1 + D ** 2 / 4 * np.cos(kx * D / 2) ** 2 * (2 / dy) ** 2 * np.sin(ky * dy / 2) ** 2
    Ex1 = np.fft.ifft2(np.fft.fft2(R[0]) / den).real
    Bx1 = np.fft.ifft2(np.fft.fft2(R[3]) / den).real
    out = rip_transport(f, 0.5 * mux(Ex + Ex1), 0.5 * mux(Bx + Bx1), J, D, dy)
    if check:
        return out, max(np.abs(out[0] - Ex1).max(), np.abs(out[3] - Bx1).max())
    return out


# ---------------------------------------------------------------- PSATD (FBPIC-like)
class PSATD:
    def __init__(self, nx, ny, dx, dy, dt, filt=False):
        kx = 2 * np.pi * np.fft.fftfreq(nx, dx)[:, None] * np.ones((1, ny))
        ky = 2 * np.pi * np.fft.fftfreq(ny, dy)[None, :] * np.ones((nx, 1))
        k = np.sqrt(kx ** 2 + ky ** 2); k0 = k == 0; ks = np.where(k0, 1.0, k)
        self.kx, self.ky, self.k, self.k0, self.dt = kx, ky, ks, k0, dt
        self.C = np.cos(k * dt); self.S = np.sin(k * dt)
        self.S_k = np.where(k0, dt, self.S / ks)
        self.omC_k = np.where(k0, 0.0, (1 - self.C) / ks)
        self.hx, self.hy = kx / ks, ky / ks
        # FBPIC default: filter_currents = True, one-pass binomial on J and rho in k space
        self.filt = (np.cos(kx * dx / 2) * np.cos(ky * dy / 2)) ** 2 if filt else 1.0

    def step(self, f, J, rho0, rho1):
        F = [np.fft.fft2(a) for a in f]
        Jh = [np.fft.fft2(a) * self.filt for a in J]
        r0, r1 = np.fft.fft2(rho0) * self.filt, np.fft.fft2(rho1) * self.filt
        hx, hy, k, dt = self.hx, self.hy, self.k, self.dt
        # curl-free correction: k^.J = i (rho1 - rho0) / (k dt)
        kJ = hx * Jh[0] + hy * Jh[1]
        kJn = np.where(self.k0, kJ, 1j * (r1 - r0) / (k * dt))
        Jh[0] = Jh[0] + hx * (kJn - kJ); Jh[1] = Jh[1] + hy * (kJn - kJ)
        Ex, Ey, Ez, Bx, By, Bz = F
        Jx, Jy, Jz = Jh
        cross = lambda ax, ay, az: (hy * az, -hx * az, hx * ay - hy * ax)   # k^ x a (kz = 0)
        C, S, Sk, oCk = self.C, self.S, self.S_k, self.omC_k
        kxB = cross(Bx, By, Bz); kxE = cross(Ex, Ey, Ez); kxJ = cross(Jx, Jy, Jz)
        kE = hx * Ex + hy * Ey; kJ = hx * Jx + hy * Jy
        E1 = []
        for a, kb, hh, jj in ((Ex, kxB[0], hx, Jx), (Ey, kxB[1], hy, Jy), (Ez, kxB[2], 0.0, Jz)):
            E1.append(C * a + 1j * S * kb - Sk * jj + (1 - C) * hh * kE + hh * kJ * (Sk - dt))
        kB = hx * Bx + hy * By
        B1 = [C * b - 1j * S * ke + 1j * oCk * kj + (1 - C) * hh * kB
              for b, ke, kj, hh in zip((Bx, By, Bz), kxE, kxJ, (hx, hy, 0.0))]
        return [np.fft.ifft2(a).real for a in E1 + B1]


# ---------------------------------------------------------------- driver
def energy(f, dx, dy):
    return 0.5 * sum((a ** 2).sum() for a in f) * dx * dy


def run(solver, nx, ny, dx, dy, u0, steps, ppc=(2, 2), uth=1e-4, every=10, gauss=False):
    D = dt = dx
    pl = Plasma(nx, ny, dx, dy, ppc, u0, uth)
    f = [np.zeros((nx, ny)) for _ in range(6)]
    # gather offsets (in cells) per component
    if solver.startswith("psatd"):
        off = [(0, 0)] * 6; ps = PSATD(nx, ny, dx, dy, dt, filt=solver == "psatd_f")
    else:
        off = [(0, 0), (0, 0.5), (0, 0), (0, 0.5), (0, 0), (0, 0.5)]
    rho_old = None
    hist = []
    for n in range(steps):
        J = [np.zeros((nx, ny)) for _ in range(3)]
        rho1 = np.zeros((nx, ny)); rho0 = np.zeros((nx, ny))
        for s in pl.sp:
            x, y, u = s["x"], s["y"], s["u"]
            E = np.stack([gather(f[c], x, y, dx, dy, *off[c]) for c in range(3)], 1)
            B = np.stack([gather(f[c + 3], x, y, dx, dy, *off[c + 3]) for c in range(3)], 1)
            u = boris(u, E, B, s["q"] / s["m"], dt)
            g = np.sqrt(1 + (u ** 2).sum(1))
            vx, vy, vz = u[:, 0] / g, u[:, 1] / g, u[:, 2] / g
            x1, y1 = x + vx * dt, y + vy * dt
            qw = np.full(x.size, s["q"] * s["w"])
            if solver.startswith("psatd"):
                xh, yh = x + 0.5 * vx * dt, y + 0.5 * vy * dt
                for c, v in enumerate((vx, vy, vz)):
                    deposit_lin(J[c], xh, yh, qw * v / (dx * dy), dx, dy, 0, 0)
                deposit_lin(rho0, x, y, qw / (dx * dy), dx, dy, 0, 0)
                deposit_lin(rho1, x1, y1, qw / (dx * dy), dx, dy, 0, 0)
            else:
                esirkepov(J, rho1, x, y, x1, y1, vz, qw, dx, dy, dt)
            s["x"], s["y"], s["u"] = x1 % (nx * dx), y1 % (ny * dy), u
        if solver.endswith("_f") and not solver.startswith("psatd"):   # same one-pass binomial, real space
            bino = lambda a: 0.25 * (np.roll(a, 1, 0) + 2 * a + np.roll(a, -1, 0))
            biny = lambda a: 0.25 * (np.roll(a, 1, 1) + 2 * a + np.roll(a, -1, 1))
            J = [biny(bino(a)) for a in J]; rho1 = biny(bino(rho1))
        if solver.startswith("psatd"):
            f = ps.step(f, J, rho0, rho1)
        elif solver.startswith("box"):
            f = step_box(f, J, D, dy)
        else:
            f = step_ripcc(f, J, D, dy)
        if gauss and not solver.startswith("psatd"):
            G = (np.roll(f[0], -1, 0) - f[0]) / dx + mux(dym(f[1], dy)) - rho1
            if rho_old is None:
                G0 = G.copy()
            rho_old = 1
        if n % every == 0 or n == steps - 1:
            rec = [n * dt, energy(f, dx, dy)]
            if gauss and not solver.startswith("psatd"):
                rec.append(np.abs(G - G0).max())
            hist.append(rec)
    return np.array(hist)


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "paper"
    if case == "paper":
        dx, dy, ny = 0.63, 1.88, 32
    else:
        dx, dy, ny = 0.63, 0.1575, 128
    nx, u0 = 64, -10.0
    steps = int(280 / dx)
    solvers = sys.argv[2].split(",") if len(sys.argv) > 2 else (("ripcc", "box", "psatd", "psatd_f") if case == "paper" else ("box", "psatd", "psatd_f"))
    for solver in solvers:
        h = run(solver, nx, ny, dx, dy, u0, steps, gauss=True)
        np.savetxt("nci_%s_%s.txt" % (case, solver), h)
        print(solver, "W(t):", " ".join("%.2e" % w for w in h[::5, 1]), flush=True)
