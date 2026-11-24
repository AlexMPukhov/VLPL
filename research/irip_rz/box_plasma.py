import numpy as np
# Cartesian box (Crank-Nicolson) RIP with an explicit cold-fluid current (leapfrog), per Fourier mode.
def amp(kx, ky, kz, D, h, wp2):
    a = np.exp(1j * kx * D); dy = 2j * np.sin(ky * h / 2) / h; dz = 2j * np.sin(kz * h / 2) / h; mu = (1 + a) / 2
    def step(s, Ex1, Bx1):
        Ex, Ey, Ez, Bx, By, Bz, vx, vy, vz = s
        vx, vy, vz = vx + D * Ex, vy + D * Ey, vz + D * Ez           # push with E^n
        jx, jy, jz = wp2 * vx, wp2 * vy, wp2 * vz
        ExS, BxS = mu * 0.5 * (Ex1 + Ex), mu * 0.5 * (Bx1 + Bx)      # box average at (i+1/2, n+1/2)
        Jy, Jz = mu * jy, mu * jz
        Tpy = (Ey + Bz) / a + D * (dz * BxS - Jy + dy * ExS) / a
        Tmy = (Ey - Bz) * a + D * (dz * BxS - Jy - dy * ExS)
        Tpz = (Ez + By) * a + D * (-dy * BxS - Jz - dz * ExS)
        Tmz = (Ez - By) / a + D * (-dy * BxS - Jz + dz * ExS) / a
        Ey1, Bz1 = (Tpy + Tmy) / 2, (Tpy - Tmy) / 2
        Ez1, By1 = (Tpz + Tmz) / 2, (Tpz - Tmz) / 2
        Exc = Ex + D * (dy * (Bz + Bz1) / 2 - dz * (By + By1) / 2 - jx)
        Bxc = Bx - D * (dy * (Ez + Ez1) / 2 - dz * (Ey + Ey1) / 2)
        return np.array([Exc, Ey1, Ez1, Bxc, By1, Bz1, vx, vy, vz])
    I = np.eye(9)
    C = np.array([step(I[q], 0, 0)[[0, 3]] for q in range(9)]).T
    E = np.array([step(np.zeros(9), *g)[[0, 3]] - step(np.zeros(9), 0, 0)[[0, 3]] for g in np.eye(2)]).T
    X = np.linalg.solve(np.eye(2) - E, C)
    return np.array([step(I[q], X[0, q], X[1, q]) for q in range(9)]).T
D, h = 2.0, 0.25          # D/h = 8: far beyond the explicit RIP-CC limit
for wpD in [0.5, 1.0, 1.5, 1.9, 1.99, 2.05]:
    wp2 = (wpD / D) ** 2
    worst = 0
    for kx in np.linspace(0, np.pi / D, 9):
        for ky in np.linspace(0, np.pi / h, 9):
            for kz in np.linspace(0, np.pi / h, 5):
                worst = max(worst, np.abs(np.linalg.eigvals(amp(kx, ky, kz, D, h, wp2))).max())
    print("wp*dt = %.2f  (D/h = 8): max |z| = %.12f" % (wpD, worst))
