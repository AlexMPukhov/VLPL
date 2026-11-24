"""Check of the discrete Poynting theorem of the box closure in Cartesian 3D (paper, Theorem 2):
W^{n+1} - W^n = -D sum [Jx mu_t Ex + Jy Et_y + Jz Et_z],
Et_a = (M+_a + M-_a)/2 - D^2/4 delta_a delta_x mu_t Ex  (a = y, z), M+-: characteristic midpoints.
The step is the numpy copy of src/fields.cpp from VLPL research/irip_3d/identity.py."""
import numpy as np
rng = np.random.default_rng(7)
nx, ny, nz = 6, 5, 4
hx, hy, hz = 0.4, 0.3, 0.25
D, iy, iz = hx, 1 / hy, 1 / hz
sh = lambda f, a, s: np.roll(f, -s, axis=a)

def step(E, B, J, X, Y):
    Ex, Ey, Ez = E; Bx, By, Bz = B; Jx, Jy, Jz = J
    Exs, Bxs = 0.5 * (Ex + X), 0.5 * (Bx + Y)
    mux = lambda f: 0.5 * (f + sh(f, 0, 1))
    psiy = (mux(Bxs) - sh(mux(Bxs), 2, -1)) * iz - Jy
    psiz = -(mux(Bxs) - sh(mux(Bxs), 1, -1)) * iy - Jz
    phiy = -(sh(mux(Exs), 2, 1) - mux(Exs)) * iz
    phiz = (sh(mux(Exs), 1, 1) - mux(Exs)) * iy
    m1 = lambda f: sh(f, 0, -1)
    tpy = m1(Ey) + m1(Bz) + D * (m1(psiy) + m1(phiz))
    tmy = sh(Ey, 0, 1) - sh(Bz, 0, 1) + D * (psiy - phiz)
    tpz = sh(Ez, 0, 1) + sh(By, 0, 1) + D * (psiz + phiy)
    tmz = m1(Ez) - m1(By) + D * (m1(psiz) - m1(phiy))
    nEy, nBz = 0.5 * (tpy + tmy), 0.5 * (tpy - tmy)
    nEz, nBy = 0.5 * (tpz + tmz), 0.5 * (tpz - tmz)
    bz0, bzm = 0.5 * (Bz + nBz), sh(0.5 * (Bz + nBz), 1, -1)
    by0, bym = 0.5 * (By + nBy), sh(0.5 * (By + nBy), 2, -1)
    ez0, ezp = 0.5 * (Ez + nEz), sh(0.5 * (Ez + nEz), 1, 1)
    ey0, eyp = 0.5 * (Ey + nEy), sh(0.5 * (Ey + nEy), 2, 1)
    nEx = Ex + D * ((bz0 - bzm) * iy - (by0 - bym) * iz - Jx)
    nBx = Bx - D * ((ezp - ez0) * iy - (eyp - ey0) * iz)
    return nEx, nBx, (nEy, nEz, nBy, nBz)

N = nx * ny * nz
Z = np.zeros((nx, ny, nz))
def solve_step(E, B, J):
    F0, G0, _ = step(E, B, J, Z, Z)
    cols = []
    for q in range(N):
        e = np.zeros(N); e[q] = 1
        cols.append((step(E, B, J, e.reshape(nx, ny, nz), Z)[0] - F0).ravel())
    A = np.eye(N) - np.array(cols).T
    X = np.linalg.solve(A, F0.ravel()).reshape(nx, ny, nz)
    Y = np.linalg.solve(A, G0.ravel()).reshape(nx, ny, nz)   # same operator for Bx (verified in identity.py)
    return step(E, B, J, X, Y)

worst = 0
for trial in range(3):
    E = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
    B = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
    J = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
    nEx, nBx, (nEy, nEz, nBy, nBz) = solve_step(E, B, J)
    W = lambda fs: 0.5 * sum((f * f).sum() for f in fs)
    dW = W([nEx, nEy, nEz, nBx, nBy, nBz]) - W(E + B)
    Ex, Ey, Ez = E; Bx, By, Bz = B
    p1 = lambda f: sh(f, 0, 1)
    Mpy = 0.5 * (Ey + Bz + p1(nEy + nBz)); Mmy = 0.5 * (p1(Ey - Bz) + nEy - nBz)
    Mpz = 0.5 * (p1(Ez + By) + nEz + nBy); Mmz = 0.5 * (Ez - By + p1(nEz - nBy))
    mtEx = 0.5 * (Ex + nEx)
    dx = (p1(mtEx) - mtEx) / D
    Ety = 0.5 * (Mpy + Mmy) - D * D / 4 * (sh(dx, 1, 1) - dx) * iy
    Etz = 0.5 * (Mpz + Mmz) - D * D / 4 * (sh(dx, 2, 1) - dx) * iz
    work = -D * ((J[0] * mtEx).sum() + (J[1] * Ety).sum() + (J[2] * Etz).sum())
    worst = max(worst, abs(dW - work) / abs(dW))
    print("dW = %.12f   -D sum J.Et = %.12f" % (dW, work))
print("relative mismatch %.1e" % worst)
