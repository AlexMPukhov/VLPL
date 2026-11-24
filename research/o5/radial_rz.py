# O5 in RZ: the grid error of the discrete radial operator L_m (src/rz_irip.cpp basis()) per radial eigenmode.
# Wall at R: TM (Ez) J_m(k R) = 0, TE (Bz) J_m'(k R) = 0. Question: is k^2 - lambda = h^2 lambda^2 / 12 (as in Cartesian),
# and what is the effective h on a stretched grid?
import numpy as np
from scipy.special import jn_zeros, jnp_zeros

def grid(nr, dr, ju=None, stretch=1.0, drmax=1e300):
    n = nr + 5
    if ju is None or stretch == 1: ju = n
    r = np.zeros(n); h = dr
    for j in range(n - 1):
        if j < nr: h = dr if j < ju else min(dr * stretch ** (j - ju + 1), drmax)
        r[j + 1] = (j + 1) * dr if j < ju else r[j] + h
    dn = np.zeros(n); rh = np.zeros(n)
    for j in range(n - 1):
        dn[j] = r[j + 1] - r[j]; rh[j] = (j + .5) * dr if j < ju else .5 * (r[j] + r[j + 1])
    dn[-1] = dn[-2]; rh[-1] = r[-1] + .5 * dn[-1]
    A = np.array([.5 * rh[0] ** 2 if j == 0 else .5 * (rh[j] ** 2 - rh[j - 1] ** 2) for j in range(n)])
    B = rh * dn
    dd = np.array([dr if j == 0 else A[j] / r[j] for j in range(n)])
    return dict(nr=nr, r=r, rh=rh, A=A, B=B, dn=dn, dd=dd)

def L(g, m, kind):
    n, r, rh, A, B, dn, dd = g['nr'], g['r'], g['rh'], g['A'], g['B'], g['dn'], g['dd']
    at = lambda v, j: v[j] if 0 <= j < n else 0
    def psi_x(Bp, Br):
        o = np.zeros(n, complex); o[0] = rh[0] * Bp[0] / A[0] if m == 0 else 0
        for j in range(1, n): o[j] = (rh[j] * Bp[j] - rh[j - 1] * Bp[j - 1]) / A[j] - 1j * m / r[j] * Br[j]
        return o
    def phi_x(Ep, Er):
        return np.array([-(r[j + 1] * at(Ep, j + 1) - r[j] * Ep[j]) / B[j] + 1j * m / rh[j] * Er[j] for j in range(n)])
    psi_r = lambda Bz: np.array([1j * m / rh[j] * Bz[j] for j in range(n)])
    def psi_t(Bz):
        o = np.zeros(n, complex); o[0] = -Bz[0] / rh[0] if m == 1 else 0
        for j in range(1, n): o[j] = -(Bz[j] - Bz[j - 1]) / dd[j]
        return o
    def phi_r(Ez):
        o = np.zeros(n, complex); o[0] = -1j / r[1] * at(Ez, 1) if m == 1 else 0
        for j in range(1, n): o[j] = -1j * m / r[j] * Ez[j]
        return o
    phi_t = lambda Ez: np.array([(at(Ez, j + 1) - Ez[j]) / dn[j] for j in range(n)])
    j0 = 1 if kind == 0 and m >= 1 else 0
    M = np.zeros((n, n))
    for c in range(j0, n):
        e = np.zeros(n, complex); e[c] = 1
        o = psi_x(phi_t(e), phi_r(e)) if kind == 0 else phi_x(psi_t(e), psi_r(e))
        M[j0:, c] = o[j0:].real
    M = M[j0:, j0:]
    lam, V = np.linalg.eig(-M)
    i = np.argsort(lam.real)
    return lam.real[i], V[:, i].real, j0

def exact(m, kind, R, cnt):
    z = jn_zeros(m, cnt) if kind == 0 else (np.r_[0.0, jnp_zeros(0, cnt - 1)] if m == 0 else jnp_zeros(m, cnt))
    return (z / R) ** 2

def heff(g, V, j0, kind):
    # the mode-weighted local spacing squared: weights |V|^2 x the cell volume (A for nodes, B for half nodes)
    n = g['nr']
    h2 = (g['dd'][j0:n] ** 2 if kind == 0 else g['dn'][:n] ** 2)
    w = (g['A'][j0:n] if kind == 0 else g['B'][:n])
    # symmetrise the weight: L is self-adjoint in the volume weights; the energy of the mode ~ w |v|^2
    return np.array([np.sum(w * V[:, a] ** 2 * h2) / np.sum(w * V[:, a] ** 2) for a in range(V.shape[1])])

np.set_printoptions(linewidth=160)
for label, g, dr in [("uniform nr 64", grid(64, 1.0), 1.0),
                     ("stretched: 24 x dr then 1.04", grid(64, 1.0, ju=24, stretch=1.04), 1.0)]:
    R = g['r'][g['nr']]
    print(f"== {label}, R = {R:.2f} dr")
    for m in (0, 1, 2):
        for kind in (0, 1):
            lam, V, j0 = L(g, m, kind)
            cnt = 20
            k2 = exact(m, kind, R, cnt + 1)
            if kind == 1 and m == 0: lam_c, k2_c = lam[:cnt], k2[:cnt]
            else: lam_c, k2_c = lam[:cnt], k2[:cnt]
            h2 = heff(g, V, j0, kind)[:cnt]
            err = k2_c - lam_c
            pred_dr = dr * dr * lam_c ** 2 / 12
            pred_h = h2 * lam_c ** 2 / 12
            sel = [1, 4, 9, 14, 19]
            print(f" m {m} {'TM' if kind == 0 else 'TE'}: k dr     ", np.sqrt(k2_c[sel]).round(3) * dr)
            print("   (k^2 - lam)/(dr^2 lam^2/12)  ", (err[sel] / pred_dr[sel]).round(3))
            print("   (k^2 - lam)/(h_eff^2 lam^2/12)", (err[sel] / pred_h[sel]).round(3))
            # residual relative phase error left by the three choices, per radian of k: |k_corr - k| / k
            for nm, kc2 in (("none", lam_c), ("dr", lam_c + pred_dr), ("heff", lam_c + pred_h)):
                print(f"     rel. error of k, {nm:4s}", (np.abs(np.sqrt(np.maximum(kc2, 0)) - np.sqrt(k2_c)) / np.sqrt(k2_c))[sel].round(6))
