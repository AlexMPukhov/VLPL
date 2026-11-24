"""2D (x, y; d/dz = 0) box iRIP field solver with a linearly responding medium (Lorentz/Drude), periodic box.

Staggering as VLPL / nci2d.py: Ex, Ez, By at (i, j); Ey, Bz, Bx at (i, j+1/2); all fields at integer time.
Medium currents at (i+1/2 cell centre, n+1/2): Jy at (i+1/2, j+1/2), Jz at (i+1/2, j); Jx at (i, j) (with Ex).
Units c = 1. dt = D = dx.

Medium per component c (x, y, z): polarisation P, current J
   (P^{n+1} - P^n)/D = mu_t J
   (J^{n+1} - J^n)/D = wp2 * Et - w02 * mu_t P - nu * mu_t J
with Et the work field of the box closure (Theorem 2):
   Et_x = mu_t Ex
   Et_y = (M+_y + M-_y)/2 - D^2/4 delta_y delta_x mu_t Ex
   Et_z = (M+_z + M-_z)/2
Then  W + K,  K = sum (J^2 + w02 P^2) / (2 wp2),  decreases by D sum nu (mu_t J)^2 / wp2 (exact).
Drude: w02 = 0. Dielectric with eps(w) = 1 + wp2/(w02 - w^2 - i nu w).
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def dyp(F, dy):
    return (np.roll(F, -1, 1) - F) / dy


def dym(F, dy):
    return (F - np.roll(F, 1, 1)) / dy


def mux(F):
    return 0.5 * (F + np.roll(F, -1, 0))


class Medium:
    """Per-component arrays wp2, w02, nu (shape (nx, ny)) for x, y, z currents."""
    def __init__(self, shape):
        z = lambda: np.zeros(shape)
        self.wp2 = [z(), z(), z()]
        self.w02 = [z(), z(), z()]
        self.nu = [z(), z(), z()]
        self.J = [z(), z(), z()]
        self.P = [z(), z(), z()]
        self.diss = 0.0                     # energy dissipated so far (times cell area)

    def energy(self):
        K = 0.0
        for c in range(3):
            m = self.wp2[c] > 0
            K += 0.5 * ((self.J[c][m] ** 2 + self.w02[c][m] * self.P[c][m] ** 2) / self.wp2[c][m]).sum()
        return K


class Box2D:
    def __init__(self, nx, ny, D, dy, medium=None, tol=1e-10):
        self.nx, self.ny, self.D, self.dy = nx, ny, D, dy
        self.f = [np.zeros((nx, ny)) for _ in range(6)]     # Ex Ey Ez Bx By Bz
        self.med = medium
        self.tol = tol
        kx = 2 * np.pi * np.fft.fftfreq(nx, D)[:, None]
        ky = 2 * np.pi * np.fft.fftfreq(ny, dy)[None, :]
        self.den = 1 + D ** 2 / 4 * np.cos(kx * D / 2) ** 2 * (2 / dy) ** 2 * np.sin(ky * dy / 2) ** 2
        self.iters = []
        # preconditioner: vacuum box operator (Fourier) after scaling Ex by the local medium diagonal
        self.adiag = 1.0 if medium is None else 1 + D ** 2 * medium.wp2[0] / (2 * (2 + D * medium.nu[0] + D ** 2 * medium.w02[0] / 2))

    # one step with given guesses X = Ex^{n+1}, Y = Bx^{n+1}; returns new fields and new medium state
    def _step(self, X, Y, Jext=None):
        D, dy = self.D, self.dy
        Ex, Ey, Ez, Bx, By, Bz = self.f
        mtEx = 0.5 * (Ex + X)
        Exbar, Bxbar = mux(mtEx), mux(0.5 * (Bx + Y))
        dEx = dyp(Exbar, dy)
        dBx = dym(Bxbar, dy)
        Tp, Tm = Ey + Bz, Ey - Bz
        Up, Um = Ez - By, Ez + By
        Jx = np.zeros_like(Ex); Jy = np.zeros_like(Ex); Jz = np.zeros_like(Ex)
        if Jext is not None:
            Jx, Jy, Jz = [a.copy() for a in Jext]
        newJ, newP, u = None, None, None
        if self.med is not None:
            m = self.med
            # work fields without the medium's own current (local part eliminated)
            E0 = [mtEx,
                  0.5 * (Tp + np.roll(Tm, -1, 0)) - D ** 2 / 4 * dyp((np.roll(mtEx, -1, 0) - mtEx) / D, dy) - 0.5 * D * Jy,
                  0.5 * (Up + np.roll(Um, -1, 0)) - 0.5 * D * dBx - 0.5 * D * Jz]
            kap = (0.0, 1.0, 1.0)
            u, newJ, newP = [], [], []
            for c in range(3):
                wp2, w02, nu, J, P = m.wp2[c], m.w02[c], m.nu[c], m.J[c], m.P[c]
                uc = (2 * J + D * wp2 * E0[c] - D * w02 * P) / (2 + D * nu + D * D * w02 / 2 + kap[c] * D * D * wp2 / 2)
                u.append(uc); newJ.append(2 * uc - J); newP.append(P + D * uc)
            Jx, Jy, Jz = Jx + u[0], Jy + u[1], Jz + u[2]
        Tp1 = np.roll(Tp + D * (dEx - Jy), 1, 0)
        Tm1 = np.roll(Tm, -1, 0) + D * (-dEx - Jy)
        Up1 = np.roll(Up + D * (-dBx - Jz), 1, 0)
        Um1 = np.roll(Um, -1, 0) + D * (-dBx - Jz)
        Ey1, Bz1 = 0.5 * (Tp1 + Tm1), 0.5 * (Tp1 - Tm1)
        Ez1, By1 = 0.5 * (Up1 + Um1), 0.5 * (Um1 - Up1)
        Ex1 = Ex + D * (dym(0.5 * (Bz + Bz1), dy) - Jx)
        Bx1 = Bx - D * dyp(0.5 * (Ez + Ez1), dy)
        return [Ex1, Ey1, Ez1, Bx1, By1, Bz1], newJ, newP, u

    def _prec(self, v):
        n = self.nx * self.ny
        X = np.fft.ifft2(np.fft.fft2(v[:n].reshape(self.nx, self.ny) / self.adiag) / self.den).real
        Y = np.fft.ifft2(np.fft.fft2(v[n:].reshape(self.nx, self.ny)) / self.den).real
        return np.concatenate([X.ravel(), Y.ravel()])

    def step(self, Jext=None):
        nx, ny = self.nx, self.ny
        n = nx * ny
        Z = np.zeros((nx, ny))
        F0 = self._step(Z, Z, Jext)[0]
        b = np.concatenate([F0[0].ravel(), F0[3].ravel()])
        if self.med is None:
            X = np.fft.ifft2(np.fft.fft2(F0[0]) / self.den).real
            Y = np.fft.ifft2(np.fft.fft2(F0[3]) / self.den).real
            self.iters.append(0)
        else:
            def A(v):
                X, Y = v[:n].reshape(nx, ny), v[n:].reshape(nx, ny)
                F = self._step(X, Y, Jext)[0]
                return v - (np.concatenate([F[0].ravel(), F[3].ravel()]) - b)
            cnt = [0]
            def cb(_):
                cnt[0] += 1
            x0 = np.concatenate([self.f[0].ravel(), self.f[3].ravel()])
            sol, info = gmres(LinearOperator((2 * n, 2 * n), matvec=A), b, x0=x0, rtol=self.tol, atol=0, restart=60,
                              maxiter=50, M=LinearOperator((2 * n, 2 * n), matvec=self._prec), callback=cb,
                              callback_type="pr_norm")
            if info != 0:
                raise RuntimeError("gmres did not converge: %d" % info)
            X, Y = sol[:n].reshape(nx, ny), sol[n:].reshape(nx, ny)
            self.iters.append(cnt[0])
        f1, newJ, newP, u = self._step(X, Y, Jext)
        if self.med is not None:
            m = self.med
            for c in range(3):
                msk = m.wp2[c] > 0
                m.diss += self.D * (m.nu[c][msk] * u[c][msk] ** 2 / m.wp2[c][msk]).sum()
            m.J, m.P = newJ, newP
        self.f = f1
        return f1

    def energy(self):
        return 0.5 * sum((a ** 2).sum() for a in self.f)
