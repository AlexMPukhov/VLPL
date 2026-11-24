"""O6 prototype: 2D3V PIC on the box-iRIP field step (research/linear_medium/box2d.py, periodic x, y; c = 1, D = dx = dt)
with (a) the explicit leapfrog of the code (Boris, E^n and B^n gathered at x^n with the plain shape, Esirkepov deposit), and
(b) the energy-conserving implicit mover (Crank-Nicolson in the particles, the box closure in the fields):

    u1 = u0 + q/m dt (Ebar + vbar x Bbar),   vbar = (u0 + u1) / (g0 + g1),   x1 = x0 + dt vbar,
    J  = Esirkepov deposit of the straight path x0 -> x1 (unchanged: exact charge conservation),
    Ebar_a = (work of the work field Et along the path, with the Esirkepov weights) / (displacement along a),

Et the work field of the box closure (docs/specs/linear_medium.md): W^{n+1} - W^n = -dt A sum J . Et, so with this gather
m (g1 - g0) = q Ebar . (x1 - x0) = the field work of the particle's own deposited current: W + K is conserved exactly
(to the tolerance of the nonlinear solve). B does no work for any Bbar because the same vbar moves the particle.

Staggering (grid.hpp): Ex, Jx (i, j); Ey, Bz, Bx (i, j+1/2); Ez, By (i, j); rho (i+1/2, j); Jy (i+1/2, j+1/2); Jz (i+1/2, j).
Gauss law: dx Ex + mu_x dy Ey = rho.
"""
import numpy as np

OFF = np.arange(-2, 3)


def bspline(d, s):
    a = np.abs(d)
    if s == 0:
        return np.where(a < 0.5, 1.0, np.where(a == 0.5, 0.5, 0.0))
    if s == 1:
        return np.maximum(0.0, 1.0 - a)
    if s == 2:
        return np.where(a < 0.5, 0.75 - a * a, np.where(a < 1.5, 0.5 * (1.5 - a) ** 2, 0.0))
    raise ValueError(s)


def dyp(F, dy):
    return (np.roll(F, -1, 1) - F) / dy


def dym(F, dy):
    return (F - np.roll(F, 1, 1)) / dy


def mux(F):
    return 0.5 * (F + np.roll(F, -1, 0))


class Sim:
    def __init__(self, nx, ny, D, dy, shape=2):
        self.nx, self.ny, self.D, self.dy, self.s = nx, ny, D, dy, shape
        self.A = D * dy
        self.f = [np.zeros((nx, ny)) for _ in range(6)]           # Ex Ey Ez Bx By Bz
        kx = 2 * np.pi * np.fft.fftfreq(nx, D)[:, None]
        ky = 2 * np.pi * np.fft.fftfreq(ny, dy)[None, :]
        self.den = 1 + D ** 2 / 4 * np.cos(kx * D / 2) ** 2 * (2 / dy) ** 2 * np.sin(ky * dy / 2) ** 2
        self.sp = []                                                 # species: dicts x, y, u (N, 3), q, m (per macro)
        self.rho_bg = np.zeros((nx, ny))
        self.ptol = 1e-13                                            # inner (per-particle) Picard tolerance
        self.J = [np.zeros((nx, ny)) for _ in range(3)]
        self.t = 0.0
        self.stats = []                                              # per implicit step: iterations

    # ---------------- fields ----------------
    def _core(self, X, Y, J):
        D, dy = self.D, self.dy
        Ex, Ey, Ez, Bx, By, Bz = self.f
        Jx, Jy, Jz = J
        mtEx = 0.5 * (Ex + X)
        dEx = dyp(mux(mtEx), dy)
        dBx = dym(mux(0.5 * (Bx + Y)), dy)
        Tp, Tm = Ey + Bz, Ey - Bz
        Up, Um = Ez - By, Ez + By
        Tp1 = np.roll(Tp + D * (dEx - Jy), 1, 0)
        Tm1 = np.roll(Tm, -1, 0) + D * (-dEx - Jy)
        Up1 = np.roll(Up + D * (-dBx - Jz), 1, 0)
        Um1 = np.roll(Um, -1, 0) + D * (-dBx - Jz)
        Ey1, Bz1 = 0.5 * (Tp1 + Tm1), 0.5 * (Tp1 - Tm1)
        Ez1, By1 = 0.5 * (Up1 + Um1), 0.5 * (Um1 - Up1)
        Ex1 = Ex + D * (dym(0.5 * (Bz + Bz1), dy) - Jx)
        Bx1 = Bx - D * dyp(0.5 * (Ez + Ez1), dy)
        Et = [mtEx,
              0.5 * (Tp + np.roll(Tm, -1, 0)) - D ** 2 / 4 * dyp((np.roll(mtEx, -1, 0) - mtEx) / D, dy) - 0.5 * D * Jy,
              0.5 * (Up + np.roll(Um, -1, 0)) - 0.5 * D * dBx - 0.5 * D * Jz]
        return [Ex1, Ey1, Ez1, Bx1, By1, Bz1], Et

    def field_solve(self, J):
        """The box step with the current J (exact FFT solve): new fields and the work field Et."""
        Z = np.zeros((self.nx, self.ny))
        F0 = self._core(Z, Z, J)[0]
        X = np.fft.ifft2(np.fft.fft2(F0[0]) / self.den).real
        Y = np.fft.ifft2(np.fft.fft2(F0[3]) / self.den).real
        return self._core(X, Y, J)

    def W(self):
        return 0.5 * self.A * sum((a ** 2).sum() for a in self.f)

    # ---------------- particles ----------------
    def add_species(self, x, y, u, q, m, mobile=True):
        self.sp.append(dict(x=x.copy(), y=y.copy(), u=u.copy(), q=q, m=m, mobile=mobile))

    def neutralise(self):
        """Fixed ions: rho_bg = -rho of all species at t = 0 (exact Gauss law with E = 0)."""
        self.rho_bg = -self.rho()

    def K(self):
        k = 0.0
        for s in self.sp:
            if s["mobile"]:
                u2 = (s["u"] ** 2).sum(1)
                k += s["m"] * (u2 / (np.sqrt(1 + u2) + 1)).sum()
        return k

    def _stencil(self, xi):
        k = np.floor(xi + 0.5).astype(np.int64)
        return k, k[:, None] + OFF

    def rho(self, extra=None):
        out = np.zeros(self.nx * self.ny)
        for s in self.sp:
            w = s["q"] / self.A if extra is None else extra(s)
            if np.isscalar(w) and w == 0:
                continue
            xi, eta = s["x"] / self.D - 0.5, s["y"] / self.dy
            _, kx = self._stencil(xi)
            _, ky = self._stencil(eta)
            Sx = bspline(xi[:, None] - kx, self.s)
            Sy = bspline(eta[:, None] - ky, self.s)
            idx = (kx[:, :, None] % self.nx) * self.ny + ky[:, None, :] % self.ny
            ww = Sx[:, :, None] * Sy[:, None, :] * (w if np.isscalar(w) else w[:, None, None])
            out += np.bincount(idx.ravel(), ww.ravel(), self.nx * self.ny)
        return out.reshape(self.nx, self.ny)

    def gauss(self):
        Ex, Ey = self.f[0], self.f[1]
        div = (np.roll(Ex, -1, 0) - Ex) / self.D + mux(dym(Ey, self.dy))
        return div - self.rho() - self.rho_bg

    def gather_plain(self, F, x, y, ox, oy):
        xi, eta = x / self.D - ox, y / self.dy - oy
        _, kx = self._stencil(xi)
        _, ky = self._stencil(eta)
        Sx = bspline(xi[:, None] - kx, self.s)
        Sy = bspline(eta[:, None] - ky, self.s)
        return np.einsum("pa,pb,pab->p", Sx, Sy, F[kx[:, :, None] % self.nx, ky[:, None, :] % self.ny])

    def gather_EB_plain(self, F, x, y, Eskip=False):
        E = None if Eskip else np.stack([self.gather_plain(F[0], x, y, 0, 0), self.gather_plain(F[1], x, y, 0, 0.5),
                      self.gather_plain(F[2], x, y, 0, 0)], 1)
        B = np.stack([self.gather_plain(F[3], x, y, 0, 0.5), self.gather_plain(F[4], x, y, 0, 0),
                      self.gather_plain(F[5], x, y, 0, 0.5)], 1)
        return E, B

    def path(self, x0, y0, x1, y1, vz, q, Et=None, deposit=None):
        """Esirkepov over the straight path in nseg equal segments. With Et: returns the work per unit charge along x, y and
        Ebar_z (sum over the path); with deposit (list of 3 flat arrays): adds q's current."""
        D, dy, nx, ny, s = self.D, self.dy, self.nx, self.ny, self.s
        dxi, deta = (x1 - x0) / D, (y1 - y0) / dy
        nseg = max(1, int(np.ceil(max(np.abs(dxi).max(initial=0), np.abs(deta).max(initial=0)) / 0.95)))
        wx = np.zeros(len(x0)); wy = np.zeros(len(x0)); ez = np.zeros(len(x0))
        for g in range(nseg):
            xa = x0 / D - 0.5 + dxi * g / nseg; xb = x0 / D - 0.5 + dxi * (g + 1) / nseg
            ya = y0 / dy + deta * g / nseg; yb = y0 / dy + deta * (g + 1) / nseg
            _, kx = self._stencil(xa)
            _, ky = self._stencil(ya)
            Sx0, Sx1 = bspline(xa[:, None] - kx, s), bspline(xb[:, None] - kx, s)
            Sy0, Sy1 = bspline(ya[:, None] - ky, s), bspline(yb[:, None] - ky, s)
            dSx, dSy = Sx1 - Sx0, Sy1 - Sy0
            Cx = -np.cumsum(dSx, 1)                      # fraction of the charge through face (kx + a + 1/2) along +x
            Cy = -np.cumsum(dSy, 1)
            Fx = Cx[:, :, None] * (Sy0 + 0.5 * dSy)[:, None, :]
            Fy = (Sx0 + 0.5 * dSx)[:, :, None] * Cy[:, None, :]
            Wz = (Sx0[:, :, None] * Sy0[:, None, :] + 0.5 * (dSx[:, :, None] * Sy0[:, None, :] + Sx0[:, :, None] * dSy[:, None, :])
                  + dSx[:, :, None] * dSy[:, None, :] / 3)
            ix_c = kx[:, :, None] % nx; iy_c = ky[:, None, :] % ny
            ix_f = (kx[:, :, None] + 1) % nx            # Jx index of face kx + a + 1/2
            if Et is not None:
                wx += D * np.einsum("pab,pab->p", Fx, Et[0][ix_f, iy_c])
                wy += dy * np.einsum("pab,pab->p", Fy, Et[1][ix_c, iy_c])
                ez += np.einsum("pab,pab->p", Wz, Et[2][ix_c, iy_c]) / nseg
            if deposit is not None:
                fac = q if np.isscalar(q) else q[:, None, None]
                deposit[0] += np.bincount((ix_f * ny + iy_c).ravel(), (fac * Fx).ravel() / (self.D * dy), nx * ny)
                deposit[1] += np.bincount((ix_c * ny + iy_c).ravel(), (fac * Fy).ravel() / (self.D * D), nx * ny)
                deposit[2] += np.bincount((ix_c * ny + iy_c).ravel(), (fac * vz[:, None, None] * Wz).ravel() / (self.A * nseg), nx * ny)
        return wx, wy, ez

    def deposit(self, paths):
        J = [np.zeros(self.nx * self.ny) for _ in range(3)]
        for (s, x1, y1, vz) in paths:
            self.path(s["x"], s["y"], x1, y1, vz, s["q"], deposit=J)
        return [j.reshape(self.nx, self.ny) for j in J]

    # ---------------- explicit step (the code's leapfrog; u holds u^{n-1/2}) ----------------
    def step_explicit(self):
        paths = []
        for s in self.sp:
            if not s["mobile"]:
                continue
            E, B = self.gather_EB_plain(self.f, s["x"], s["y"])
            u = s["u"]; h = 0.5 * s["q"] / s["m"] * self.D
            um = u + h * E
            g = np.sqrt(1 + (um ** 2).sum(1))[:, None]
            t = h * B / g
            up = um + np.cross(um, t)
            u = um + 2 / (1 + (t ** 2).sum(1))[:, None] * np.cross(up, t) + h * E
            g = np.sqrt(1 + (u ** 2).sum(1))
            v = u / g[:, None]
            s["u"] = u
            paths.append((s, s["x"] + self.D * v[:, 0], s["y"] + self.D * v[:, 1], v[:, 2]))
        J = self.deposit(paths)
        self.f = self.field_solve(J)[0]
        for (s, x1, y1, _) in paths:
            s["x"], s["y"] = x1 % (self.nx * self.D), y1 % (self.ny * self.dy)
        self.J = J
        self.t += self.D

    # ---------------- implicit, energy-conserving step (u holds u^n) ----------------
    def wp2_faces(self):
        """Lumped linear response sum q^2 / (m g) S / A at the current positions (the preconditioner)."""
        c = self.rho(extra=lambda s: (s["q"] ** 2 / s["m"] / np.sqrt(1 + (s["u"] ** 2).sum(1)) / self.A) if s["mobile"] else 0.0)
        return [0.5 * (c + np.roll(c, 1, 0)), 0.5 * (c + np.roll(c, -1, 1)), c]

    def _push_cn(self, s, Et, Bn, Bn1, u1, inner):
        dt = self.D; qm = s["q"] / s["m"]
        u0 = s["u"]; g0 = np.sqrt(1 + (u0 ** 2).sum(1))
        x0, y0 = s["x"], s["y"]
        self.npass = getattr(self, "npass", 0)
        for _ in range(inner):
            self.npass += 1
            g1 =np.sqrt(1 + (u1 ** 2).sum(1))
            vb = (u0 + u1) / (g0 + g1)[:, None]
            x1, y1 = x0 + dt * vb[:, 0], y0 + dt * vb[:, 1]
            wx, wy, ez = self.path(x0, y0, x1, y1, vb[:, 2], s["q"], Et=Et)
            dx, dyy = dt * vb[:, 0], dt * vb[:, 1]
            # Ebar = work / displacement; the limit for a vanishing displacement (work 0 then) does not affect the energy
            ex = np.where(np.abs(dx) > 1e-14 * self.D, wx / np.where(dx == 0, 1, dx), 0.0)
            ey = np.where(np.abs(dyy) > 1e-14 * self.dy, wy / np.where(dyy == 0, 1, dyy), 0.0)
            small = (np.abs(dx) <= 1e-14 * self.D) | (np.abs(dyy) <= 1e-14 * self.dy)
            if small.any():
                Ep, _ = self.gather_EB_plain([Et[0], Et[1], Et[2], Et[0], Et[0], Et[0]], x0, y0)
                ex = np.where(np.abs(dx) <= 1e-14 * self.D, Ep[:, 0], ex)
                ey = np.where(np.abs(dyy) <= 1e-14 * self.dy, Ep[:, 1], ey)
            xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            _, B0 = self.gather_EB_plain([None] * 3 + list(Bn), xm, ym, Eskip=True)
            _, B1 = self.gather_EB_plain([None] * 3 + list(Bn1), xm, ym, Eskip=True)
            Bb = 0.5 * (B0 + B1)
            Eb = np.stack([ex, ey, ez], 1)
            un = u0 + qm * dt * (Eb + np.cross(vb, Bb))
            du = np.abs(un - u1).max()
            u1 = un
            if du < self.ptol * (1 + np.abs(u1).max()):
                break
        g1 = np.sqrt(1 + (u1 ** 2).sum(1))
        vb = (u0 + u1) / (g0 + g1)[:, None]
        return u1, vb

    def step_implicit(self, tol=1e-10, maxit=60, inner=1, accel="anderson", m_and=5, precond=True):
        mob = [s for s in self.sp if s["mobile"]]
        p0 = getattr(self, "npass", 0)
        if not hasattr(self, "u1"):
            self.u1 = [s["u"].copy() for s in mob]
        P = self.wp2_faces() if precond else None
        Pi = [1.0 / (1 + self.D ** 2 / 4 * p) for p in P] if precond else [1.0, 1.0, 1.0]
        J = [j.copy() for j in self.J]

        def G(Jv):
            fnew, Et = self.field_solve(Jv)
            paths = []
            for k, s in enumerate(mob):
                self.u1[k], vb = self._push_cn(s, Et, self.f[3:], fnew[3:], self.u1[k], inner)
                paths.append((s, s["x"] + self.D * vb[:, 0], s["y"] + self.D * vb[:, 1], vb[:, 2]))
            return self.deposit(paths), paths

        flat = lambda L: np.concatenate([a.ravel() for a in L])
        unflat = lambda v: [v[i * self.nx * self.ny:(i + 1) * self.nx * self.ny].reshape(self.nx, self.ny) for i in range(3)]
        Xh, Fh = [], []
        it = 0
        while True:
            Jd, paths = G(J)
            r = [Pi[c] * (Jd[c] - J[c]) for c in range(3)]
            x, fr = flat(J), flat(r)
            res = np.linalg.norm(fr) / max(np.linalg.norm(flat(Jd)), 1e-300)
            it += 1
            if res < tol or it >= maxit:
                break
            if accel == "anderson" and m_and > 0:
                Xh.append(x); Fh.append(fr)
                if len(Xh) > m_and + 1:
                    Xh.pop(0); Fh.pop(0)
                if len(Xh) > 1:
                    dF = np.stack([Fh[i + 1] - Fh[i] for i in range(len(Fh) - 1)], 1)
                    dX = np.stack([Xh[i + 1] - Xh[i] for i in range(len(Xh) - 1)], 1)
                    gam = np.linalg.lstsq(dF, fr, rcond=None)[0]
                    xn = x + fr - (dX + dF) @ gam
                else:
                    xn = x + fr
            else:
                xn = x + fr
            J = unflat(xn)
        # commit: the field step with the deposit of the final particle paths (exact Gauss law)
        self.f = self.field_solve(Jd)[0]
        for k, (s, x1, y1, _) in enumerate(paths):
            s["x"], s["y"] = x1 % (self.nx * self.D), y1 % (self.ny * self.dy)
            s["u"] = self.u1[k].copy()
        self.J = Jd
        self.t += self.D
        self.stats.append((it, res, self.npass - p0))


def lattice(nx, ny, D, dy, ppx, ppy, x0=0.0, x1=None):
    x1 = nx * D if x1 is None else x1
    xs = (np.arange(int(round((x1 - x0) / D)) * ppx) + 0.5) * D / ppx + x0
    ys = (np.arange(ny * ppy) + 0.5) * dy / ppy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return X.ravel(), Y.ravel()
