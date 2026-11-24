# O5 prototype (docs/specs/overdense.md): the iRIP box step of research/irip_3d/identity.py in 2D (x, y; d/dz = 0) with a
# modified closure Ex_bar = C mu_t Ex, Bx_bar likewise (before mu_x), C = Cx Cy:
#   Cx = 1 - (cx / 3) h_x^2 delta_x^2            (cx = 1: the paraxial factor 1 + 4/3 sin^2(kx hx / 2) of the spec)
#   Cy = 1 - beta delta_y^2 h_y^2 ... written as 1 - b_y (f(j+1) - 2 f + f(j-1)), b_y = cy (D^2/6 + h_y^2/12) / h_y^2
# cx = cy = 0 is plain iRIP. Per Fourier mode (kx, ky) the step is linear in the 6 field amplitudes: the 6 x 6 amplification
# matrix G is built by stepping complex plane waves on a small periodic grid and reading the amplitude at the centre (the
# stencils reach ~4 cells; the roll at the far edge does not matter there). The implicit solve per mode: F(X) = F0 + a X.
import numpy as np

n = 15
c0 = n // 2

def make(D, hy, cx, cy):
    iy = 1 / hy
    by = cy * (D * D / 6 + hy * hy / 12) / (hy * hy)
    sh = lambda f, a, s: np.roll(f, -s, axis=a)
    Cx = lambda f: f - cx / 3 * (sh(f, 0, 1) - 2 * f + sh(f, 0, -1))
    Cy = lambda f: f - by * (sh(f, 1, 1) - 2 * f + sh(f, 1, -1))
    C = lambda f: Cx(Cy(f))

    def step(S, X, Y):
        Ex, Ey, Ez, Bx, By, Bz = S
        Exs, Bxs = C(0.5 * (Ex + X)), C(0.5 * (Bx + Y))
        mux = lambda f: 0.5 * (f + sh(f, 0, 1))
        psiy = 0 * Ex                                   # d/dz = 0
        psiz = -(mux(Bxs) - sh(mux(Bxs), 1, -1)) * iy
        phiy = 0 * Ex
        phiz = (sh(mux(Exs), 1, 1) - mux(Exs)) * iy
        m1 = lambda f: sh(f, 0, -1)
        tpy = m1(Ey) + m1(Bz) + D * (m1(psiy) + m1(phiz))
        tmy = sh(Ey, 0, 1) - sh(Bz, 0, 1) + D * (psiy - phiz)
        tpz = sh(Ez, 0, 1) + sh(By, 0, 1) + D * (psiz + phiy)
        tmz = m1(Ez) - m1(By) + D * (m1(psiz) - m1(phiy))
        nEy, nBz = 0.5 * (tpy + tmy), 0.5 * (tpy - tmy)
        nEz, nBy = 0.5 * (tpz + tmz), 0.5 * (tpz - tmz)
        bz0, bzm = 0.5 * (Bz + nBz), sh(0.5 * (Bz + nBz), 1, -1)
        ez0, ezp = 0.5 * (Ez + nEz), sh(0.5 * (Ez + nEz), 1, 1)
        nEx = Ex + D * ((bz0 - bzm) * iy)
        nBx = Bx - D * ((ezp - ez0) * iy)
        return [nEx, nEy, nEz, nBx, nBy, nBz]

    def G(kx, ky):
        x = np.arange(n)[:, None] - c0
        y = np.arange(n)[None, :] - c0
        w = np.exp(1j * (kx * D * x + ky * hy * y))        # mode on the grid (hx = D)
        Z = 0 * w
        M = np.zeros((6, 6), complex)
        for q in range(6):
            S = [w if p == q else Z for p in range(6)]
            F0 = step(S, Z, Z)
            a = step([Z] * 6, w, Z)[0][c0, c0]               # dF/dX per mode (Ex)
            b = step([Z] * 6, Z, w)[3][c0, c0]               # dG/dY (Bx)
            X, Yv = F0[0][c0, c0] / (1 - a), F0[3][c0, c0] / (1 - b)
            out = step(S, X * w, Yv * w)
            M[:, q] = [f[c0, c0] for f in out]
        return M
    return G

def omega(G, kx, ky, D):
    ev = np.linalg.eigvals(G(kx, ky))
    ph = np.abs(np.angle(ev)) / D
    k = np.hypot(kx, ky)
    j = np.argmin(np.abs(ph - k))
    return ph[j], np.abs(ev).max()

if __name__ == "__main__":
    lam0 = 1.0
    D = lam0 / 128
    for hy_over in (3.2, 1.0):
        hy = hy_over * D
        print("dx = lambda0/128, dy = %.1f dx" % hy_over)
        print("   N theta   exact w      iRIP: rel. phase err     O5 par     O5 full   | pred. iRIP  pred. full")
        Gs = {k: make(D, hy, *v) for k, v in {"irip": (0, 0), "par": (1, 0), "full": (1, 1)}.items()}
        for N in (7, 15):
            K = 2 * np.pi * N / lam0
            for th in (0.2, 0.4, 0.6):
                kx, ky = K * np.cos(th), K * np.sin(th)
                row = []
                for G in Gs.values():
                    w, _ = omega(G, kx, ky, D)
                    row.append(w / K - 1)
                # predictions from the dispersion relations
                b, q = kx * D / 2, D * D / 4 * (2 / hy * np.sin(ky * hy / 2)) ** 2
                a_ir = np.arctan(np.sqrt(np.tan(b) ** 2 + q)) * 2 / D / K - 1
                g = (1 + 4 / 3 * np.sin(b) ** 2) * (1 + (D * D / 6 + hy * hy / 12) * (2 / hy * np.sin(ky * hy / 2)) ** 2)
                a_fu = np.arctan(np.sqrt(np.tan(b) ** 2 + q * g)) * 2 / D / K - 1
                print("  %2d  %.1f  %9.3f   %+.2e   %+.2e   %+.2e  | %+.2e  %+.2e" % (N, th, K, *row, a_ir, a_fu))
