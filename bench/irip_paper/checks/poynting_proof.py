"""Step-by-step check of the proof of Theorem 2 (discrete Poynting theorem of the box closure), Appendix A.
One assert per equation of the appendix. Periodic 3D box, random fields and currents; the field step is the
numpy copy of VLPL src/fields.cpp used in poynting3d.py (index i of a cell-centred array = cell i+1/2).
Identities (A1)-(A5) hold for ANY closure (random X, Y in place of E_x^{n+1}, B_x^{n+1});
(A6)-(A9) use the box closure (X, Y from the implicit solve).
Run: python3 poynting_proof.py   (prints the largest residual of every identity)"""
import numpy as np

rng = np.random.default_rng(11)
nx, ny, nz = 6, 5, 4
hx, hy, hz = 0.4, 0.3, 0.25
D, iy, iz = hx, 1 / hy, 1 / hz
sh = lambda f, a, s: np.roll(f, -s, axis=a)          # sh(f, a, s)[i] = f[i + s]

# operators (n = integer position, c = half-integer position stored at the lower index)
mx_nc = lambda f: 0.5 * (f + sh(f, 0, 1))             # mu_x  node -> cell
mx_cn = lambda g: 0.5 * (g + sh(g, 0, -1))            # mu_x  cell -> node
dx_nc = lambda f: (sh(f, 0, 1) - f) / D               # delta_x node -> cell
dx_cn = lambda g: (g - sh(g, 0, -1)) / D              # delta_x cell -> node
dy_f = lambda f: (sh(f, 1, 1) - f) * iy               # delta_y  j -> j+1/2
dy_b = lambda g: (g - sh(g, 1, -1)) * iy              # delta_y  j+1/2 -> j
dz_f = lambda f: (sh(f, 2, 1) - f) * iz
dz_b = lambda g: (g - sh(g, 2, -1)) * iz


def step(E, B, J, X, Y):
    """RIP step with the closure Ebar = mu_x (E_x^n + X)/2, Bbar = mu_x (B_x^n + Y)/2 (box closure if X, Y = new
    E_x, B_x). Returns the new fields and the sources Pi_y, Pi_z, Omega_y, Omega_z at the cell centres."""
    Ex, Ey, Ez = E; Bx, By, Bz = B; Jx, Jy, Jz = J
    Eb, Bb = mx_nc(0.5 * (Ex + X)), mx_nc(0.5 * (Bx + Y))
    Piy = dz_b(Bb) - Jy; Piz = -dy_b(Bb) - Jz
    Omy = -dz_f(Eb); Omz = dy_f(Eb)
    m1 = lambda f: sh(f, 0, -1)
    tpy = m1(Ey + Bz) + D * m1(Piy + Omz)            # T+_y(i, n+1) = T+_y(i-1, n) + D S+_y(i-1/2)
    tmy = sh(Ey - Bz, 0, 1) + D * (Piy - Omz)        # T-_y(i, n+1) = T-_y(i+1, n) + D S-_y(i+1/2)
    tpz = sh(Ez + By, 0, 1) + D * (Piz + Omy)        # T+_z(i, n+1) = T+_z(i+1, n) + D S+_z(i+1/2)
    tmz = m1(Ez - By) + D * m1(Piz - Omy)            # T-_z(i, n+1) = T-_z(i-1, n) + D S-_z(i-1/2)
    nEy, nBz = 0.5 * (tpy + tmy), 0.5 * (tpy - tmy)
    nEz, nBy = 0.5 * (tpz + tmz), 0.5 * (tpz - tmz)
    nEx = Ex + D * (dy_b(0.5 * (Bz + nBz)) - dz_b(0.5 * (By + nBy)) - Jx)
    nBx = Bx - D * (dy_f(0.5 * (Ez + nEz)) - dz_f(0.5 * (Ey + nEy)))
    return (nEx, nEy, nEz), (nBx, nBy, nBz), (Piy, Piz, Omy, Omz)


N = nx * ny * nz
Z = np.zeros((nx, ny, nz))


def box_XY(E, B, J):
    F0 = step(E, B, J, Z, Z)
    cols = []
    for q in range(N):
        e = np.zeros(N); e[q] = 1
        cols.append((step(E, B, J, e.reshape(nx, ny, nz), Z)[0][0] - F0[0][0]).ravel())
    A = np.eye(N) - np.array(cols).T
    X = np.linalg.solve(A, F0[0][0].ravel()).reshape(nx, ny, nz)
    Y = np.linalg.solve(A, F0[1][0].ravel()).reshape(nx, ny, nz)
    return X, Y


S = lambda f: f.sum()
res = {}
def check(name, lhs, rhs, scale=None):
    lhs, rhs = np.asarray(lhs, float), np.asarray(rhs, float)
    sc = scale if scale is not None else max(np.abs(lhs).max(), np.abs(rhs).max(), 1e-300)
    r = np.abs(lhs - rhs).max() / sc
    res[name] = max(res.get(name, 0.0), r)
    assert r < 1e-12, (name, r)

for trial in range(4):
    E = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
    B = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
    J = [rng.normal(size=(nx, ny, nz)) for _ in range(3)]
    box = trial % 2 == 1
    if box:
        X, Y = box_XY(E, B, J)
    else:                                            # an arbitrary closure
        X, Y = rng.normal(size=(nx, ny, nz)), rng.normal(size=(nx, ny, nz))
    (nEx, nEy, nEz), (nBx, nBy, nBz), (Piy, Piz, Omy, Omz) = step(E, B, J, X, Y)
    Ex, Ey, Ez = E; Bx, By, Bz = B; Jx, Jy, Jz = J
    if box:
        check("box: corrector reproduces X, Y", np.concatenate([nEx, nBx]), np.concatenate([X, Y]))
    p1 = lambda f: sh(f, 0, 1)
    # transport components at the start and end of each characteristic in cell i+1/2
    Tpy0, Tpy1 = Ey + Bz, p1(nEy + nBz)               # T+_y: (i, n) -> (i+1, n+1)
    Tmy0, Tmy1 = p1(Ey - Bz), nEy - nBz               # T-_y: (i+1, n) -> (i, n+1)
    Tpz0, Tpz1 = p1(Ez + By), nEz + nBy               # T+_z: (i+1, n) -> (i, n+1)
    Tmz0, Tmz1 = Ez - By, p1(nEz - nBy)               # T-_z: (i, n) -> (i+1, n+1)
    Spy, Smy, Spz, Smz = Piy + Omz, Piy - Omz, Piz + Omy, Piz - Omy
    Mpy, Mmy, Mpz, Mmz = 0.5 * (Tpy0 + Tpy1), 0.5 * (Tmy0 + Tmy1), 0.5 * (Tpz0 + Tpz1), 0.5 * (Tmz0 + Tmz1)
    Py, Qz = 0.5 * (Mpy + Mmy), 0.5 * (Mpy - Mmy)     # E_y-like and B_z-like characteristic averages
    Pz, Qy = 0.5 * (Mpz + Mmz), 0.5 * (Mpz - Mmz)     # E_z-like and B_y-like
    # (A1) transport along each characteristic and its energy: |T|^2_end - |T|^2_start = 2 D M S
    for T0, T1, M, Sx in ((Tpy0, Tpy1, Mpy, Spy), (Tmy0, Tmy1, Mmy, Smy), (Tpz0, Tpz1, Mpz, Spz), (Tmz0, Tmz1, Mmz, Smz)):
        check("A1 transport T_end = T_start + D S", T1, T0 + D * Sx)
        check("A1 |T|^2_end - |T|^2_start = 2 D M S", T1 ** 2 - T0 ** 2, 2 * D * M * Sx)
    # (A2) energy of the transverse fields
    Wt = lambda Ey, Ez, By, Bz: 0.5 * S(Ey ** 2 + Ez ** 2 + By ** 2 + Bz ** 2)
    dWt = Wt(nEy, nEz, nBy, nBz) - Wt(Ey, Ez, By, Bz)
    rhsA2 = D * S(Py * Piy + Qz * Omz + Pz * Piz + Qy * Omy)
    check("A2 dW_perp = D sum (P_y Pi_y + Q_z Om_z + P_z Pi_z + Q_y Om_y)", dWt, rhsA2, abs(dWt) + abs(rhsA2))
    # (A3) energy of the longitudinal fields (corrector)
    mtEx, mtBx = 0.5 * (Ex + nEx), 0.5 * (Bx + nBx)
    mtEy, mtEz, mtBy, mtBz = 0.5 * (Ey + nEy), 0.5 * (Ez + nEz), 0.5 * (By + nBy), 0.5 * (Bz + nBz)
    dWx = 0.5 * S(nEx ** 2 - Ex ** 2 + nBx ** 2 - Bx ** 2)
    rhsA3 = D * S(mtEx * (dy_b(mtBz) - dz_b(mtBy) - Jx)) - D * S(mtBx * (dy_f(mtEz) - dz_f(mtEy)))
    check("A3 dW_x = D sum [mu_tEx (d_y mu_tBz - d_z mu_tBy - jx) - mu_tBx (d_y mu_tEz - d_z mu_tEy)]",
          dWx, rhsA3, abs(dWx) + abs(rhsA3))
    # (A4) the four node identities (key identity, Eq. keyid, and its companions), any closure
    check("A4 mu_t Bz = mu_x Q_z - D^2/4 d_x Pi_y", mtBz, mx_cn(Qz) - D * D / 4 * dx_cn(Piy))
    check("A4 mu_t By = mu_x Q_y + D^2/4 d_x Pi_z", mtBy, mx_cn(Qy) + D * D / 4 * dx_cn(Piz))
    check("A4 mu_t Ey = mu_x P_y - D^2/4 d_x Om_z", mtEy, mx_cn(Py) - D * D / 4 * dx_cn(Omz))
    check("A4 mu_t Ez = mu_x P_z + D^2/4 d_x Om_y", mtEz, mx_cn(Pz) + D * D / 4 * dx_cn(Omy))
    # (A5) cell-centre form: P_y = mu_x mu_t Ey + D^2/4 d_t d_x Bz etc. (the O(D^2) estimate)
    dtdx = lambda f0, f1: (p1(f1) - f1 - p1(f0) + f0) / (D * D)
    check("A5 P_y = mu_x mu_t Ey + D^2/4 d_t d_x Bz", Py, mx_nc(mtEy) + D * D / 4 * dtdx(Bz, nBz))
    check("A5 P_z = mu_x mu_t Ez - D^2/4 d_t d_x By", Pz, mx_nc(mtEz) - D * D / 4 * dtdx(By, nBy))
    # summation by parts used below (any arrays): sum_nodes f d_y mu_x g = - sum_cells g d_y mu_x f, etc.
    f, g = rng.normal(size=(nx, ny, nz)), rng.normal(size=(nx, ny, nz))
    check("SBP y", S(f * dy_b(mx_cn(g))), -S(g * dy_f(mx_nc(f))), 1)
    check("SBP z", S(f * dz_b(mx_cn(g))), -S(g * dz_f(mx_nc(f))), 1)
    check("SBP xy", S(f * dx_cn(dy_b(g))), S(g * dy_f(dx_nc(f))), 1)
    # (A6) divergence of the sources: B_bar drops out of d_y Pi_y + d_z Pi_z; d_y Om_y + d_z Om_z = 0 (any closure)
    check("A6 d_y Pi_y + d_z Pi_z = -(d_y jy + d_z jz)", dy_b(Piy) + dz_b(Piz), -(dy_b(Jy) + dz_b(Jz)))
    check("A6 d_y Om_y + d_z Om_z = 0", dy_f(Omy) + dz_f(Omz), 0 * Omy, np.abs(Omy).max())
    if not box:
        continue
    Eb, Bb = mx_nc(mtEx), mx_nc(mtBx)                 # box closure: Ebar = mu_x mu_t Ex, Bbar = mu_x mu_t Bx
    # (A7) cancellation of the Ebar terms: Q-terms of (A2) + curl terms of the Ex energy with mu_t B from (A4)
    t1 = D * S(Qz * Omz + Qy * Omy)
    t2 = D * S(mtEx * (dy_b(mx_cn(Qz)) - dz_b(mx_cn(Qy))))
    check("A7 D sum (Q_z Om_z + Q_y Om_y) + D sum mu_tEx (d_y mu_x Q_z - d_z mu_x Q_y) = 0", t1 + t2, 0, abs(t1))
    # (A8) cancellation of the Bbar terms; the remainder of the Bx energy vanishes by (A6)
    t3 = D * S(Py * dz_b(Bb) - Pz * dy_b(Bb))
    t4 = -D * S(mtBx * (dy_f(mx_cn(Pz)) - dz_f(mx_cn(Py))))
    check("A8 D sum (P_y d_z Bbar - P_z d_y Bbar) - D sum mu_tBx (d_y mu_x P_z - d_z mu_x P_y) = 0", t3 + t4, 0, abs(t3))
    t5 = -D ** 3 / 4 * S(mtBx * dx_cn(dy_f(Omy) + dz_f(Omz)))
    check("A8 remainder of the Bx energy = 0", t5, 0, abs(t3))
    # (A9) the theorem
    Ety = Py - D * D / 4 * dy_f(dx_nc(mtEx))
    Etz = Pz - D * D / 4 * dz_f(dx_nc(mtEx))
    dW = dWt + dWx
    work = -D * S(Jx * mtEx + Jy * Ety + Jz * Etz)
    check("A9 W^{n+1} - W^n = -D sum (jx mu_tEx + jy Et_y + jz Et_z)", dW, work, abs(dW))

w = max(len(k) for k in res)
for k, v in res.items():
    print("%-*s  %.1e" % (w, k, v))
print("all identities hold")
