"""Check of Theorem 1 (paper): with ANY longitudinal fields in the transverse sources (shared by both
members of each transport pair) and the time-centred corrector, the RIP Gauss law
G = delta_x Ex + mu_x(delta_y Ey + delta_z Ez) at (i+1/2, j, k) changes by -D div J, and the RIP div B
(at (i+1/2, j+1/2, k+1/2)) does not change. Step: the numpy copy of src/fields.cpp (poynting3d.py)."""
import numpy as np
from poynting3d import step, sh, nx, ny, nz, D, iy, iz
rng = np.random.default_rng(11)
p1 = lambda f, a: sh(f, a, 1)
mux = lambda f: 0.5 * (f + p1(f, 0))
def gauss(Ex, Ey, Ez):     # Ey at (i, j+1/2, k), Ez at (i, j, k+1/2) -> (i+1/2, j, k)
    return (p1(Ex, 0) - Ex) / D + mux((Ey - sh(Ey, 1, -1)) * iy + (Ez - sh(Ez, 2, -1)) * iz)
def divb(Bx, By, Bz):      # Bx (i, j+1/2, k+1/2), By (i, j, k+1/2), Bz (i, j+1/2, k) -> (i+1/2, j+1/2, k+1/2)
    return (p1(Bx, 0) - Bx) / D + mux((p1(By, 1) - By) * iy + (p1(Bz, 2) - Bz) * iz)
def divj(Jx, Jy, Jz):      # Jx (i, j, k), Jy (i+1/2, j+1/2, k), Jz (i+1/2, j, k+1/2) -> (i+1/2, j, k)
    return (p1(Jx, 0) - Jx) / D + (Jy - sh(Jy, 1, -1)) * iy + (Jz - sh(Jz, 2, -1)) * iz
worst = [0, 0]
for t in range(4):
    E = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
    B = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
    J = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
    X, Y = rng.normal(size=(nx, ny, nz)), rng.normal(size=(nx, ny, nz))     # arbitrary closure
    nEx, nBx, (nEy, nEz, nBy, nBz) = step(E, B, J, X, Y)
    dG = gauss(nEx, nEy, nEz) - gauss(*E) + D * divj(*J)
    dB = divb(nBx, nBy, nBz) - divb(*B)
    worst = [max(worst[0], np.abs(dG).max()), max(worst[1], np.abs(dB).max())]
print("Gauss law + D div J: %.1e   div B change: %.1e  (arbitrary Ex_bar, Bx_bar)" % tuple(worst))
