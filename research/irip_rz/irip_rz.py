"""iRIP field step in quasi-3D RZ geometry, one azimuthal mode m, Fourier mode e^{i kx x} along x.

Grid (Cartesian RIP-CC staggering with y -> r, z -> theta, d/dz -> i m / r):
  Ex, Etheta, Br  at nodes r_j = j dr            (j = 0..Nr-1, wall at r = Nr dr)
  Er, Btheta, Bx  at half nodes r_{j+1/2}        (j = 0..Nr-1)
Transport: T+ = Er+Btheta, Etheta-Br move +x; Er-Btheta, Etheta+Br move -x, one cell per step (c dt = D).
iRIP: the longitudinal fields in the transverse sources are the characteristic trapezoid
  right movers: (F(i+1, n+1) + F(i, n))/2,  left movers: (F(i, n+1) + F(i+1, n))/2,  F = Ex, Bx,
corrector Ex^{n+1} = Ex + D Psi_x(mu_t B_perp), Bx^{n+1} = Bx + D Phi_x(mu_t E_perp)  -> implicit in Ex^{n+1}, Bx^{n+1}.
Axis: m = 0: Etheta = Br = 0 on axis, Psi_x(0) = 4 Btheta_{1/2}/dr;  m >= 1: Ex = 0 on axis; Etheta, Br on the
axis are not dynamic (for m = 1 they can be reconstructed for the gather, they do not feed back).  Wall: Ex = Etheta = Br = 0 at r = Nr dr.
"""
import numpy as np


class RZ:
    def __init__(self, Nr, dr, D, m):
        self.N, self.dr, self.D, self.m = Nr, dr, D, m
        self.rn = np.arange(Nr) * dr               # nodes
        self.rh = (np.arange(Nr) + 0.5) * dr       # half nodes

    # --- transverse operators (mode m) -------------------------------------------------
    def psi_x(self, Bt, Br):                       # (1/r) d_r(r Btheta) - (i m/r) Br   at nodes
        N, dr, m, rn, rh = self.N, self.dr, self.m, self.rn, self.rh
        out = np.zeros(N, complex)
        j = np.arange(1, N)
        out[j] = (rh[j] * Bt[j] - rh[j - 1] * Bt[j - 1]) / (rn[j] * dr) - 1j * m / rn[j] * Br[j]
        if m == 0:
            out[0] = 4 * Bt[0] / dr
        return out

    def phi_x(self, Et, Er):                       # -[(1/r) d_r(r Etheta) - (i m/r) Er]   at half nodes
        N, dr, m, rn, rh = self.N, self.dr, self.m, self.rn, self.rh
        Etp = np.append(Et[1:], 0.0)                # Etheta_{j+1}, wall value 0
        rnp = np.append(rn[1:], N * dr)
        return -((rnp * Etp - rn * Et) / (rh * dr) - 1j * m / rh * Er)

    def psi_r(self, Bx):                           # (i m / r) Bx   at half nodes
        return 1j * self.m / self.rh * Bx

    def psi_t(self, Bx):                           # -d_r Bx   at nodes
        N, dr, m = self.N, self.dr, self.m
        out = np.zeros(N, complex)
        out[1:] = -(Bx[1:] - Bx[:-1]) / dr
        if m == 1:
            out[0] = -2 * Bx[0] / dr               # Bx(-dr/2) = -Bx(dr/2) for m = 1
        return out

    def phi_r(self, Ex):                           # -(i m / r) Ex   at nodes
        N, dr, m, rn = self.N, self.dr, self.m, self.rn
        out = np.zeros(N, complex)
        out[1:] = -1j * m / rn[1:] * Ex[1:]
        if m == 1:
            out[0] = -1j * Ex[1] / dr              # Ex ~ r near the axis
        return out

    def phi_t(self, Ex):                           # d_r Ex   at half nodes
        Exp = np.append(Ex[1:], 0.0)
        return (Exp - Ex) / self.dr

    # --- one step for given (guessed) Ex1, Bx1; linear in all arguments ----------------
    def step_given(self, S, Ex1, Bx1, a):
        Ex, Er, Et, Bx, Br, Bt = S
        D = self.D
        ExR, BxR = 0.5 * (a * Ex1 + Ex), 0.5 * (a * Bx1 + Bx)      # right movers, at i+1/2
        ExL, BxL = 0.5 * (Ex1 + a * Ex), 0.5 * (Bx1 + a * Bx)      # left movers
        Tpy = (Er + Bt) / a + D * (self.psi_r(BxR) + self.phi_t(ExR)) / a
        Tmy = (Er - Bt) * a + D * (self.psi_r(BxL) - self.phi_t(ExL))
        Tpz = (Et + Br) * a + D * (self.psi_t(BxL) + self.phi_r(ExL))
        Tmz = (Et - Br) / a + D * (self.psi_t(BxR) - self.phi_r(ExR)) / a
        Er1, Bt1 = 0.5 * (Tpy + Tmy), 0.5 * (Tpy - Tmy)
        Et1, Br1 = 0.5 * (Tpz + Tmz), 0.5 * (Tpz - Tmz)
        Exc = Ex + D * self.psi_x(0.5 * (Bt + Bt1), 0.5 * (Br + Br1))
        Bxc = Bx + D * self.phi_x(0.5 * (Et + Et1), 0.5 * (Er + Er1))
        return [Exc, Er1, Et1, Bxc, Br1, Bt1]

    # --- valid degrees of freedom (axis constraints) -----------------------------------
    def mask(self):
        N, m = self.N, self.m
        ok = [np.ones(N, bool) for _ in range(6)]   # Ex, Er, Et, Bx, Br, Bt
        if m >= 1:
            ok[0][0] = False                        # Ex on axis
        ok[2][0] = False; ok[4][0] = False          # Etheta, Br on axis: zero for m = 0 and m >= 2;
        # for m = 1 they are only driven (never feed back), i.e. diagnostic values: not dynamic DOFs
        return np.concatenate(ok)

    def amplification(self, kx):
        """Exact one-step matrix: the implicit Ex1, Bx1 solved by linear algebra."""
        N, a = self.N, np.exp(1j * kx * self.D)
        n = 6 * N
        unpack = lambda v: [v[k * N:(k + 1) * N] for k in range(6)]
        z = np.zeros(N, complex)
        # Ex1, Bx1 = C s + E [Ex1; Bx1]  ->  solve
        C = np.zeros((2 * N, n), complex); E = np.zeros((2 * N, 2 * N), complex)
        for q in range(n):
            e = np.zeros(n, complex); e[q] = 1
            out = self.step_given(unpack(e), z, z, a)
            C[:, q] = np.concatenate([out[0], out[3]])
        for q in range(2 * N):
            g = np.zeros(2 * N, complex); g[q] = 1
            zero = [z] * 6
            out = self.step_given(zero, g[:N], g[N:], a)
            E[:, q] = np.concatenate([out[0], out[3]])
        X = np.linalg.solve(np.eye(2 * N) - E, C)  # [Ex1; Bx1] = X s
        M = np.zeros((n, n), complex)
        for q in range(n):
            e = np.zeros(n, complex); e[q] = 1
            out = self.step_given(unpack(e), X[:N, q], X[N:, q], a)
            M[:, q] = np.concatenate(out)
        ok = self.mask()
        return M[np.ix_(ok, ok)], np.linalg.cond(np.eye(2 * N) - E)

    # --- transverse spectra for the dispersion formula ---------------------------------
    def lap_TM(self):       # L = Psi_x o (Phi_theta, Phi_r) acting on Ex
        N = self.N
        L = np.zeros((N, N), complex)
        for q in range(N):
            e = np.zeros(N, complex); e[q] = 1
            L[:, q] = self.psi_x(self.phi_t(e), self.phi_r(e))
        ok = np.ones(N, bool); ok[0] = self.m == 0
        return L[np.ix_(ok, ok)]

    def lap_TE(self):       # L = Phi_x o (Psi_theta, Psi_r) acting on Bx
        N = self.N
        L = np.zeros((N, N), complex)
        for q in range(N):
            e = np.zeros(N, complex); e[q] = 1
            L[:, q] = self.phi_x(self.psi_t(e), self.psi_r(e))
        return L
