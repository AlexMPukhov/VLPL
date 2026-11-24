"""The work field of the 3D box step (numpy copy of src/irip.cpp, research/irip_3d/identity.py): find Et with
W^{n+1} - W^n = -D V sum J . Et for random fields and currents (periodic, h_perp < c dt)."""
import numpy as np
rng = np.random.default_rng(1)
nx, ny, nz = 6, 5, 4
hx, hy, hz = 0.4, 0.3, 0.25
D, iy, iz = hx, 1 / hy, 1 / hz
sh = lambda f, a, s: np.roll(f, -s, axis=a)
exec(open(__file__.replace("o6_mover/work3d.py", "irip_3d/identity.py")).read().split("E = [rng")[0].split("sh = lambda")[1].split("\n", 1)[1])   # def step(...)

def solve(E, B, J):
    Z = np.zeros((nx, ny, nz)); N = nx * ny * nz
    F0, G0, _ = step(E, B, J, Z, Z)
    def op(f):
        lap = (sh(f, 1, 1) - 2 * f + sh(f, 1, -1)) * iy * iy + (sh(f, 2, 1) - 2 * f + sh(f, 2, -1)) * iz * iz
        return D * D / 4 * 0.25 * (sh(lap, 0, -1) + 2 * lap + sh(lap, 0, 1))
    L = np.zeros((N, N))
    for q in range(N):
        e = np.zeros(N); e[q] = 1; L[:, q] = op(e.reshape(nx, ny, nz)).ravel()
    X = np.linalg.solve(np.eye(N) - L, F0.ravel()).reshape(nx, ny, nz)
    Y = np.linalg.solve(np.eye(N) - L, G0.ravel()).reshape(nx, ny, nz)
    nEx, nBx, (nEy, nEz, nBy, nBz) = step(E, B, J, X, Y)
    return X, Y, [nEx, nEy, nEz], [nBx, nBy, nBz]

def work_field(E, B, J, X, Y, cy, cz):
    Ex, Ey, Ez = E; Bx, By, Bz = B; Jx, Jy, Jz = J
    mt = 0.5 * (Ex + X); Bxs = 0.5 * (Bx + Y)
    mux = lambda f: 0.5 * (f + sh(f, 0, 1))
    psiy = (mux(Bxs) - sh(mux(Bxs), 2, -1)) * iz - Jy
    psiz = -(mux(Bxs) - sh(mux(Bxs), 1, -1)) * iy - Jz
    dxm = (sh(mt, 0, 1) - mt) / D                                      # at i + 1/2
    Ety = 0.5 * (Ey + Bz + sh(Ey - Bz, 0, 1)) + 0.5 * D * psiy + cy * D * D / 4 * (sh(dxm, 1, 1) - dxm) * iy
    Etz = 0.5 * (Ez - By + sh(Ez + By, 0, 1)) + 0.5 * D * psiz + cz * D * D / 4 * (sh(dxm, 2, 1) - dxm) * iz
    return [mt, Ety, Etz]

E = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
B = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
J = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
W = lambda E, B: 0.5 * sum((f * f).sum() for f in list(E) + list(B))
X, Y, E1, B1 = solve(E, B, J)
dW = W(E1, B1) - W(E, B)
for cy in (-1, 0, 1):
    for cz in (-1, 0, 1):
        Et = work_field(E, B, J, X, Y, cy, cz)
        rhs = -D * sum((J[c] * Et[c]).sum() for c in range(3))
        print(f"cy {cy:+d} cz {cz:+d}: dW {dW:+.10e}  -D J.Et {rhs:+.10e}  rel {abs(dW - rhs) / abs(dW):.1e}")
