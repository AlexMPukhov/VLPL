# O5 in 3D (decision 73): per Fourier mode (kx, ky, kz) the 6 x 6 amplification matrix of the iRIP step (src/irip.cpp,
# box_pass) with the closure Ex_bar = c mu_t Ex, Bx_bar likewise, c = (1 + 4/3 sin^2(kx dx/2)) g_perp per mode,
# g_perp = 1 + D^2 lam/6 + (hy^2 ly^2 + hz^2 lz^2)/(12 lam) (c = 1: plain iRIP). Spectral radius and transient growth
# max_n ||G^n||_2 (n <= 400, plain E^2 + B^2) over the Brillouin zone; the dispersion of the matrix against the relation.
# o5 = 'cap': the time-step part D^2 lam/6 = (2/3) q capped at q = D^2 lam/4 <= 1 (decision 77).
# Run: python3 research/o5/stab3d.py (numpy)
import numpy as np


def G(kx, ky, kz, D, hy, hz, o5):
    ex, ey, ez = np.exp(1j * kx * D), np.exp(1j * ky * hy), np.exp(1j * kz * hz)
    iy, iz = 1 / hy, 1 / hz
    ly, lz = (2 / hy * np.sin(ky * hy / 2)) ** 2, (2 / hz * np.sin(kz * hz / 2)) ** 2
    lam = ly + lz
    c = 1.0
    if o5:
        gp = 1 + (2 / 3 * min(D * D * lam / 4, 1.0) if o5 == 'cap' else D * D * lam / 6) + (hy * hy * ly * ly + hz * hz * lz * lz) / (12 * lam) if lam > 0 else 1.0
        c = (1 + 4 / 3 * np.sin(kx * D / 2) ** 2) * gp

    def step(S, X, Y):
        Ex, Ey, Ez, Bx, By, Bz = S
        Exs, Bxs = c * 0.5 * (Ex + X), c * 0.5 * (Bx + Y)
        m = 0.5 * (1 + ex)
        psiy = m * (1 - 1 / ez) * iz * Bxs
        psiz = -m * (1 - 1 / ey) * iy * Bxs
        phiy = -m * (ez - 1) * iz * Exs
        phiz = m * (ey - 1) * iy * Exs
        tpy = (Ey + Bz) / ex + D * (psiy + phiz) / ex
        tmy = (Ey - Bz) * ex + D * (psiy - phiz)
        tpz = (Ez + By) * ex + D * (psiz + phiy)
        tmz = (Ez - By) / ex + D * (psiz - phiy) / ex
        nEy, nBz = 0.5 * (tpy + tmy), 0.5 * (tpy - tmy)
        nEz, nBy = 0.5 * (tpz + tmz), 0.5 * (tpz - tmz)
        bz0 = 0.5 * (Bz + nBz); by0 = 0.5 * (By + nBy); ez0 = 0.5 * (Ez + nEz); ey0 = 0.5 * (Ey + nEy)
        nEx = Ex + D * ((bz0 - bz0 / ey) * iy - (by0 - by0 / ez) * iz)
        nBx = Bx - D * ((ez0 * ey - ez0) * iy - (ey0 * ez - ey0) * iz)
        return np.array([nEx, nEy, nEz, nBx, nBy, nBz])

    M = np.zeros((6, 6), complex)
    Z = np.zeros(6, complex)
    a = step(Z, 1.0, 0.0)[0]
    b = step(Z, 0.0, 1.0)[3]
    for q in range(6):
        S = np.eye(6)[q].astype(complex)
        F0 = step(S, 0.0, 0.0)
        M[:, q] = step(S, F0[0] / (1 - a), F0[3] / (1 - b))
    return M


if __name__ == "__main__":
    D = 1.0
    # dispersion check against the relation (oblique, 45 deg azimuth)
    for hp in (0.25, 3.2):
        kx, ky, kz = 0.6, 0.5 / hp, 0.5 / hp
        w = np.abs(np.angle(np.linalg.eigvals(G(kx, ky, kz, D, hp, hp, True)))).max()
        ly = lz = (2 / hp * np.sin(0.25)) ** 2
        lam = ly + lz
        gp = 1 + lam / 6 + hp * hp * (ly * ly + lz * lz) / (12 * lam)
        pred = 2 * np.arctan(np.sqrt(np.tan(kx / 2) ** 2 + lam / 4 * (1 + 4 / 3 * np.sin(kx / 2) ** 2) * gp))
        print("h_perp = %.2f dx: w dt of the matrix %.12f, relation %.12f" % (hp, w, pred))
    nk = 13
    for hp in (3.2, 1.0, 0.5, 0.25, 0.1):
        for o5 in (False, True, 'cap'):
            rho, grow, worst = 0.0, 0.0, None
            for kx in np.linspace(0, np.pi, nk):
                for ky in np.linspace(0, np.pi / hp, nk):
                    for kz in np.linspace(0, np.pi / hp, nk):
                        M = G(kx, ky, kz, D, hp, hp, o5)
                        rho = max(rho, np.abs(np.linalg.eigvals(M)).max())
                        P, g = np.eye(6), 1.0
                        for _ in range(400):
                            P = M @ P
                            g = max(g, np.linalg.norm(P, 2))
                        if g > grow: grow, worst = g, (kx, ky * hp, kz * hp)
            print("h_perp = %.2f dx  %-4s  max spectral radius %.12f   max transient growth %.3f at (kx dx, ky dy, kz dz) = (%.2f, %.2f, %.2f)"
                  % (hp, {False: "box", True: "o5", "cap": "cap"}[o5], rho, grow, *worst), flush=True)
