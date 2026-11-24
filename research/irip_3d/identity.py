# Check (a) of docs/specs/implicit_rip.md in Cartesian 3D: the box step (RIP-CC with Ex*, Bx* replaced by
# mu_t Ex, mu_t Bx, i.e. (Ex^n + X)/2, (Bx^n + Y)/2) gives Ex^{n+1} = F(X, Y), Bx^{n+1} = G(X, Y), linear.
# Claim: dF/dX = D^2/4 mu_x^2 Lap_perp (on Ex nodes), dG/dY = D^2/4 mu_x^2 Lap_perp (on Bx), dF/dY = dG/dX = 0,
# so (I - D^2/4 mu_x^2 Lap) X = F(0, 0), and likewise for Y. Periodic box, numpy copy of src/fields.cpp.
import numpy as np
rng = np.random.default_rng(1)
nx, ny, nz = 6, 5, 4
hx, hy, hz = 0.4, 0.3, 0.25          # h_perp < c dt: the implicit regime
D, iy, iz = hx, 1 / hy, 1 / hz
sh = lambda f, a, s: np.roll(f, -s, axis=a)   # f(i + s) along axis a (0 x, 1 y, 2 z)

def step(E, B, J, X, Y):
    Ex, Ey, Ez = E; Bx, By, Bz = B; Jx, Jy, Jz = J
    Exs, Bxs = 0.5 * (Ex + X), 0.5 * (Bx + Y)                     # box: mu_t of the longitudinal fields
    mux = lambda f: 0.5 * (f + sh(f, 0, 1))                       # at i + 1/2
    # transport sources at (a + 1/2), as rip_cc (fields.cpp)
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

E = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
B = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
J = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
N = nx * ny * nz
Z = np.zeros((nx, ny, nz))
F0, G0, _ = step(E, B, J, Z, Z)
cols = lambda f: f.ravel()
dFdX = np.zeros((N, N)); dFdY = np.zeros((N, N)); dGdX = np.zeros((N, N)); dGdY = np.zeros((N, N))
for q in range(N):
    e = np.zeros(N); e[q] = 1; e = e.reshape(nx, ny, nz)
    F1, G1, _ = step(E, B, J, e, Z); F2, G2, _ = step(E, B, J, Z, e)
    dFdX[:, q] = cols(F1 - F0); dGdX[:, q] = cols(G1 - G0)
    dFdY[:, q] = cols(F2 - F0); dGdY[:, q] = cols(G2 - G0)
# claimed operator: D^2/4 mu_x^2 Lap_perp, mu_x^2 = [1, 2, 1]/4 along x, the 5-point transverse Laplacian
def op(f):
    lap = (sh(f, 1, 1) - 2 * f + sh(f, 1, -1)) * iy * iy + (sh(f, 2, 1) - 2 * f + sh(f, 2, -1)) * iz * iz
    return D * D / 4 * 0.25 * (sh(lap, 0, -1) + 2 * lap + sh(lap, 0, 1))
L = np.zeros((N, N))
for q in range(N):
    e = np.zeros(N); e[q] = 1
    L[:, q] = cols(op(e.reshape(nx, ny, nz)))
sc = np.abs(L).max()
print("dF/dX - op: %.1e   dG/dY - op: %.1e   dF/dY: %.1e   dG/dX: %.1e   (|op| %.2f)" % (
    np.abs(dFdX - L).max() / sc, np.abs(dGdY - L).max() / sc, np.abs(dFdY).max() / sc, np.abs(dGdX).max() / sc, sc))
# the solve and a consistency check: X = (I - op)^-1 F0 makes F(X, Y) = X
X = np.linalg.solve(np.eye(N) - L, cols(F0)).reshape(nx, ny, nz)
Y = np.linalg.solve(np.eye(N) - L, cols(G0)).reshape(nx, ny, nz)
F, G, T = step(E, B, J, X, Y)
print("self-consistency |F(X,Y) - X| %.1e, |G(X,Y) - Y| %.1e" % (np.abs(F - X).max(), np.abs(G - Y).max()))
# energy conservation in vacuum (J = 0) over many steps, D / h_perp > 1
J0 = [Z, Z, Z]
Ex, Bx = E[0].copy(), B[0].copy(); Ey, Ez = E[1].copy(), E[2].copy(); By, Bz = B[1].copy(), B[2].copy()
W = lambda: sum((f * f).sum() for f in (Ex, Ey, Ez, Bx, By, Bz))
w0 = W(); dev = 0
Minv = np.linalg.inv(np.eye(N) - L)
for n in range(300):
    F0, G0, _ = step([Ex, Ey, Ez], [Bx, By, Bz], J0, Z, Z)
    X = (Minv @ cols(F0)).reshape(nx, ny, nz); Y = (Minv @ cols(G0)).reshape(nx, ny, nz)
    nEx, nBx, (nEy, nEz, nBy, nBz) = step([Ex, Ey, Ez], [Bx, By, Bz], J0, X, Y)
    Ex, Ey, Ez, Bx, By, Bz = nEx, nEy, nEz, nBx, nBy, nBz
    dev = max(dev, abs(W() / w0 - 1))
print("vacuum, D/h_perp = %.2f, 300 steps: max |W/W0 - 1| = %.1e (plain E^2 + B^2)" % (D / min(hy, hz), dev))
