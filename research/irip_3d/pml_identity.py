# Box iRIP with CPML layers in y and z (docs/specs/implicit_rip.md, PML step): the identity behind the
# separable solve. numpy copy of the iRIP step with the CPML of src/pml.cpp: every transverse derivative
# g becomes g + psi_new, psi_new = b psi + a g (a = b - 1 in the layer, 0 outside), one psi per call site
# and location; zero fields behind the y and z edges (walls); x periodic. Pass 1 (X = Y = 0) only reads
# psi, pass 2 commits it.
# Claims checked:
#   (1) Ex^{n+1} = R_E + D^2/4 mu_x^2 L_E X with L_E = Cn_y Dm_y Ch_y Dp_y + (the same along z), C = b at
#       the derivative's location (node n or half node h), Dp: node -> half, Dm: half -> node; for Bx
#       L_B = Ch Dp Cn Dm; TM and TE decoupled. The psi memory only enters R.
#   (2) L_E, L_B per axis are similar to symmetric negative semi-definite matrices: real eigenbasis,
#       eigenvalues <= 0. The solve: dense eigen-transforms in y and z (once at setup), the tridiagonal
#       system along x per mode, back; X then satisfies the step to round-off.
#   (3) vacuum, random fields: the energy stays bounded. It is not monotone in E^2 + B^2 (psi stores
#       energy): random grid-scale fields gain up to ~20 % transiently, also at D/h_perp = 0.5.
# b is floored at 1e-4 (sigma dt <= ln 1e4) for iRIP: without it b reaches 1e-20 deep in a layer when
# c dt >> h_perp, the eigenbasis C^(1/2) Q is ill-conditioned and the solve is only good to 1e-6.
import numpy as np
rng = np.random.default_rng(4)
nx, ny, nz = 6, 10, 9
hx, hy, hz = 0.5, 0.1, 0.12
D, iy, iz = hx, 1 / hy, 1 / hz

def coeffs(n, h, L=3, R0=1e-6, m=3):
    smax = (m + 1) * np.log(1 / R0) / (2 * L * h)
    out = {}
    for half in (0, 1):
        y = (np.arange(n) + 0.5 * half) * h
        dep = np.maximum(np.maximum(L * h - y, y - (n - L) * h), 0) / (L * h)
        s = smax * dep ** m
        b = np.maximum(np.exp(-s * D), 1e-4)              # iRIP: sigma dt <= ln 1e4 (conditioning)
        a = np.where(s > 0, b - 1, 0.0)
        out['n' if half == 0 else 'h'] = (a, b)
    return out
cy, cz = coeffs(ny, hy), coeffs(nz, hz)

def shx(f, s): return np.roll(f, -s, axis=0)               # periodic x: f(i + s)
def sh(f, ax, s):                                          # zero beyond the y, z edges: f(j + s)
    g = np.zeros_like(f)
    n = f.shape[ax]
    src = [slice(None)] * 3; dst = [slice(None)] * 3
    if s > 0: src[ax] = slice(s, n); dst[ax] = slice(0, n - s)
    else: src[ax] = slice(0, n + s); dst[ax] = slice(-s, n)
    g[tuple(dst)] = f[tuple(src)]
    return g
def bc(c, ax):                                             # broadcast a per-axis coefficient
    shape = [1, 1, 1]; shape[ax] = -1
    return c.reshape(shape)

def cp(g, P, key, ax, kind, commit):
    a, b = (cy if ax == 1 else cz)[kind]
    psi_new = bc(b, ax) * P[key] + bc(a, ax) * g
    out = np.where(bc(a, ax) != 0, g + psi_new, g)
    if commit: P[key] = np.where(bc(a, ax) != 0, psi_new, P[key])
    return out

def box_pass(f, J, P, X, Y, commit):
    Ex, Ey, Ez, Bx, By, Bz = f; Jx, Jy, Jz = J
    Exs, Bxs = 0.5 * (Ex + X), 0.5 * (Bx + Y)
    mux = lambda g: 0.5 * (g + shx(g, 1))
    mB, mE = mux(Bxs), mux(Exs)
    # transport sources at (i + 1/2), once per location
    gbxz = cp((mB - sh(mB, 2, -1)) * iz, P, 't_bx_z', 2, 'n', commit)
    gbxy = cp((mB - sh(mB, 1, -1)) * iy, P, 't_bx_y', 1, 'n', commit)
    gexz = cp((sh(mE, 2, 1) - mE) * iz, P, 't_ex_z', 2, 'h', commit)
    gexy = cp((sh(mE, 1, 1) - mE) * iy, P, 't_ex_y', 1, 'h', commit)
    SPY, SPZ, SFY, SFZ = gbxz - Jy, -gbxy - Jz, -gexz, gexy
    m1 = lambda g: shx(g, -1)
    tpy = m1(Ey) + m1(Bz) + D * (m1(SPY) + m1(SFZ))
    tmy = shx(Ey, 1) - shx(Bz, 1) + D * (SPY - SFZ)
    tpz = shx(Ez, 1) + shx(By, 1) + D * (SPZ + SFY)
    tmz = m1(Ez) - m1(By) + D * (m1(SPZ) - m1(SFY))
    nEy, nBz = 0.5 * (tpy + tmy), 0.5 * (tpy - tmy)
    nEz, nBy = 0.5 * (tpz + tmz), 0.5 * (tpz - tmz)
    bz, by, ez, ey = 0.5 * (Bz + nBz), 0.5 * (By + nBy), 0.5 * (Ez + nEz), 0.5 * (Ey + nEy)
    gbz = cp((bz - sh(bz, 1, -1)) * iy, P, 'c_bz_y', 1, 'n', commit)
    gby = cp((by - sh(by, 2, -1)) * iz, P, 'c_by_z', 2, 'n', commit)
    gez = cp((sh(ez, 1, 1) - ez) * iy, P, 'c_ez_y', 1, 'h', commit)
    gey = cp((sh(ey, 2, 1) - ey) * iz, P, 'c_ey_z', 2, 'h', commit)
    nEx = Ex + D * (gbz - gby - Jx)
    nBx = Bx - D * (gez - gey)
    return nEx, nBx, [nEx, nEy, nEz, nBx, nBy, nBz]

keys = ['t_bx_z', 't_bx_y', 't_ex_z', 't_ex_y', 'c_bz_y', 'c_by_z', 'c_ez_y', 'c_ey_z']
f = [rng.normal(size=(nx, ny, nz)) for _ in range(6)]
J = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
P = {k: rng.normal(size=(nx, ny, nz)) for k in keys}
Z = np.zeros((nx, ny, nz)); N = nx * ny * nz

# (1) the derivative against the claimed operator
def ops(which):
    """1D operators along y (n1 = ny) and z: L_E = Cn Dm Ch Dp, L_B = Ch Dp Cn Dm (walls: zero beyond)."""
    out = []
    for n, h, c in ((ny, hy, cy), (nz, hz, cz)):
        Dp = (np.eye(n, k=1) - np.eye(n)) / h            # node j -> half j + 1/2 (f(j+1) - f(j)), zero beyond
        Dm = (np.eye(n) - np.eye(n, k=-1)) / h           # half -> node j (g(j) - g(j-1))
        Cn, Ch = np.diag(c['n'][1]), np.diag(c['h'][1])
        out.append(Cn @ Dm @ Ch @ Dp if which == 'E' else Ch @ Dp @ Cn @ Dm)
    return out
def apply_L(Ly, Lz, X):
    return np.einsum('ab,ibk->iak', Ly, X) + np.einsum('ab,ijb->ija', Lz, X)
LyE, LzE = ops('E'); LyB, LzB = ops('B')
mux2 = lambda g: 0.25 * (shx(g, -1) + 2 * g + shx(g, 1))
Pc = {k: v.copy() for k, v in P.items()}
F0, G0, _ = box_pass(f, J, Pc, Z, Z, False)
err = [0, 0, 0, 0]
for t in range(20):
    e = rng.normal(size=(nx, ny, nz))
    F1, G1, _ = box_pass(f, J, {k: v.copy() for k, v in P.items()}, e, Z, False)
    F2, G2, _ = box_pass(f, J, {k: v.copy() for k, v in P.items()}, Z, e, False)
    err[0] = max(err[0], np.abs(F1 - F0 - D * D / 4 * mux2(apply_L(LyE, LzE, e))).max())
    err[1] = max(err[1], np.abs(G2 - G0 - D * D / 4 * mux2(apply_L(LyB, LzB, e))).max())
    err[2] = max(err[2], np.abs(G1 - G0).max()); err[3] = max(err[3], np.abs(F2 - F0).max())
print("(1) dF/dX - op %.1e, dG/dY - op %.1e, dG/dX %.1e, dF/dY %.1e" % tuple(err))

# (2) eigenbasis per axis and the separable solve
def eig(L):
    w, V = np.linalg.eig(L)
    return w.real, V.real, np.abs(w.imag).max()
info = []
for L in (LyE, LzE, LyB, LzB):
    w, V, im = eig(L); info.append((w.max(), im))
print("(2) eigenvalues of L: max %s (<= 0), max |Im| %s" % (["%.1e" % i[0] for i in info], ["%.1e" % i[1] for i in info]))
def solve(R, Ly, Lz):
    wy, Vy, _ = eig(Ly); wz, Vz, _ = eig(Lz)
    Rh = np.einsum('ab,ibk->iak', np.linalg.inv(Vy), R); Rh = np.einsum('ab,ijb->ija', np.linalg.inv(Vz), Rh)
    lam = -(wy[:, None] + wz[None, :])                    # >= 0
    k2 = D * D * lam / 4
    b, s = 1 + 0.5 * k2, 0.25 * k2
    Xh = np.zeros_like(Rh)
    for j in range(ny):
        for k in range(nz):
            A = np.diag(np.full(nx, b[j, k])) + s[j, k] * (np.eye(nx, k=1) + np.eye(nx, k=-1))
            A[0, -1] += s[j, k]; A[-1, 0] += s[j, k]     # periodic x
            Xh[:, j, k] = np.linalg.solve(A, Rh[:, j, k])
    X = np.einsum('ab,ibk->iak', Vy, Xh); return np.einsum('ab,ijb->ija', Vz, X)
X, Y = solve(F0, LyE, LzE), solve(G0, LyB, LzB)
F, G, _ = box_pass(f, J, {k: v.copy() for k, v in P.items()}, X, Y, False)
print("    self-consistency |F(X,Y) - X| %.1e, |G(X,Y) - Y| %.1e" % (np.abs(F - X).max(), np.abs(G - Y).max()))

# (3) vacuum energy
f = [rng.normal(size=(nx, ny, nz)) for _ in range(6)]
P = {k: np.zeros((nx, ny, nz)) for k in keys}
J0 = [Z, Z, Z]
W = lambda f: sum((g * g).sum() for g in f)
w0 = W(f); w = w0; grow = 0
for n in range(400):
    R, S, _ = box_pass(f, J0, P, Z, Z, False)
    X, Y = solve(R, LyE, LzE), solve(S, LyB, LzB)
    _, _, f = box_pass(f, J0, P, X, Y, True)
    wn = W(f); grow = max(grow, (wn - w) / w0); w = wn
print("(3) vacuum, D/h_perp = %.1f, 400 steps: max growth per step %.1e W0, W/W0 at the end %.3f" % (D / hy, grow, w / w0))
