"""Cross-section eigenvalues (cutoff wavenumbers k_c, k_z = 0) of metal waveguides, for the tests of the RZ `metal` block
(docs/specs/rz_metal_phi.md).

Two independent calculations:
  discrete(...)  the RZ discretisation of VLPL with the metal on the 2M - 1 collocation angles, written from the operators
                 of src/rz_fields.cpp (node / half-node staggering in r, spectral derivative in phi, every field at a metal
                 point zero), as dense matrices. The code's RIP-CC step obeys sin(w dt / 2) = dt k_c / 2 for k_z = 0, so the
                 frequencies measured in tests/test_rz_metal_phi.cpp give these k_c to the accuracy of the frequency fit.
  polar_fd(...)  the continuum problem -Lap_perp u = k_c^2 u (TM: u = E_z = 0 on the metal; TE: u = B_z, du/dn = 0) on a polar
                 grid whose cell faces lie on the metal edges (r = const, phi = const), second-order finite differences; used
                 with Richardson extrapolation as the continuum reference for shapes made of r = const and phi = const pieces.
Square guide: analytic (TM11 k_c = sqrt(2) pi / a, TE10 pi / a, a the side).

Usage: python3 xsection.py   (prints the reference numbers used by the tests and the convergence tables of the spec)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def dphi_matrix(npts):
    """Spectral derivative on npts (odd) equidistant angles."""
    k = np.fft.fftfreq(npts, 1.0 / npts)
    F = np.fft.fft(np.eye(npts), axis=0)
    return np.real(np.fft.ifft(1j * k[:, None] * F, axis=0))


def discrete(metal, nr, dr, M, kind="TM", nev=6):
    """k_c of the lowest modes of the RZ collocation operator. metal(r, phi) -> bool (arrays). Uniform radial grid,
    wall at r = nr dr (node nr metal by the wall)."""
    Np = 2 * M - 1
    phi = 2 * np.pi * np.arange(Np) / Np
    rn = np.arange(nr + 1) * dr                      # nodes 0 .. nr
    rh = (np.arange(nr) + 0.5) * dr                  # half nodes 0 .. nr-1
    A = rn * dr
    A[0] = rh[0] ** 2 / 2
    Bw = rh * dr
    Mn = metal(rn[:, None], phi[None, :]).astype(bool)
    Mn[nr, :] = True
    Mh = Mn[:-1] & Mn[1:]
    D = dphi_matrix(Np)
    # index maps: nodes j = 0 .. nr-1 per angle; the axis node is one point (j = 0 uses only the mean)
    nn = nr * Np
    def ni(j, k): return j * Np + k
    if kind == "TM":
        # unknowns: Ez at nodes (nn), Bp at half nodes (nn), Br at nodes (nn)
        G = sp.lil_matrix((2 * nn, nn))              # (Bp, Br) <- Ez
        C = sp.lil_matrix((nn, 2 * nn))              # Ez <- (Bp, Br)
        for j in range(nr):
            for k in range(Np):
                # dBp_j/dt = (Ez_{j+1} - Ez_j) / dr   (Ez_nr = 0); the axis node: its mean (all angles equal)
                if j + 1 < nr: G[ni(j, k), ni(j + 1, k)] += 1 / dr
                if j == 0:
                    for l in range(Np): G[ni(0, k), ni(0, l)] -= 1 / dr / Np
                else: G[ni(j, k), ni(j, k)] -= 1 / dr
                if j >= 1:                           # dBr_j/dt = -(1/r) dphi Ez_j
                    for l in range(Np):
                        if D[k, l] != 0: G[nn + ni(j, k), ni(j, l)] -= D[k, l] / rn[j]
                # dEz_j/dt = (rh_j Bp_j - rh_{j-1} Bp_{j-1}) / A_j - (1/r) dphi Br_j; axis: rh_0 mean(Bp_0) / A_0
                if j == 0:
                    for l in range(Np): C[ni(0, k), ni(0, l)] += rh[0] / A[0] / Np
                else:
                    C[ni(j, k), ni(j, k)] += rh[j] / A[j]
                    C[ni(j, k), ni(j - 1, k)] -= rh[j - 1] / A[j]
                    for l in range(Np):
                        if D[k, l] != 0: C[ni(j, k), nn + ni(j, l)] -= D[k, l] / rn[j]
        pe = (~Mn[:nr]).ravel().astype(float)
        pb = np.concatenate([(~Mh).ravel(), (~Mn[:nr]).ravel() * (np.repeat(np.arange(nr), Np) > 0)]).astype(float)
    else:
        # unknowns: Bz at half nodes (nn), Er at half nodes (nn), Ep at nodes (nn)
        G = sp.lil_matrix((2 * nn, nn))              # (Er, Ep) <- Bz
        C = sp.lil_matrix((nn, 2 * nn))              # Bz <- (Er, Ep)
        for j in range(nr):
            for k in range(Np):
                for l in range(Np):                  # dEr/dt = (1/rh) dphi Bz
                    if D[k, l] != 0: G[ni(j, k), ni(j, l)] += D[k, l] / rh[j]
                if j >= 1:                           # dEp_j/dt = -(Bz_j - Bz_{j-1}) / dd_j, dd = A / r = dr
                    G[nn + ni(j, k), ni(j, k)] -= 1 / dr
                    G[nn + ni(j, k), ni(j - 1, k)] += 1 / dr
                # dBz_j/dt = -(r_{j+1} Ep_{j+1} - r_j Ep_j) / B_j + (1/rh) dphi Er_j   (Ep_nr = 0, Ep_0 = 0)
                if j + 1 < nr: C[ni(j, k), nn + ni(j + 1, k)] -= rn[j + 1] / Bw[j]
                if j >= 1: C[ni(j, k), nn + ni(j, k)] += rn[j] / Bw[j]
                for l in range(Np):
                    if D[k, l] != 0: C[ni(j, k), ni(j, l)] += D[k, l] / rh[j]
        pe = (~Mh).ravel().astype(float)
        pb = np.concatenate([(~Mh).ravel(), (~Mn[:nr]).ravel() * (np.repeat(np.arange(nr), Np) > 0)]).astype(float)
    Pe, Pb = sp.diags(pe), sp.diags(pb)
    L = (Pe @ C.tocsr() @ Pb @ G.tocsr() @ Pe).toarray()
    keep = pe > 0
    if kind == "TM":                                 # the axis node: one unknown (angle 0), the others are copies
        ax = np.zeros(nn, bool); ax[1:Np] = True
        keep &= ~ax
        L[:, 0] += L[:, 1:Np].sum(axis=1)
    Lr = L[np.ix_(keep, keep)]
    ev = np.sort(np.real(-np.linalg.eigvals(Lr)))
    ev = ev[ev > 1e-8 * max(ev.max(), 1)]
    return np.sqrt(ev[:nev])


def polar_fd(inside, R, nr, nphi, kind="TM", nev=6):
    """Continuum reference: 5-point Laplacian in polar coordinates on cell centres (r_i = (i + 1/2) h, phi_k = (k + 1/2) dphi),
    the metal as whole cells (inside(r, phi) False = metal); TM: u = 0 on metal faces (mirror -u), TE: du/dn = 0."""
    h = R / nr
    dp = 2 * np.pi / nphi
    r = (np.arange(nr) + 0.5) * h
    ph = (np.arange(nphi) + 0.5) * dp
    vac = inside(r[:, None], ph[None, :])
    idx = -np.ones((nr, nphi), int)
    idx[vac] = np.arange(vac.sum())
    n = vac.sum()
    rows, cols, vals = [], [], []
    def add(a, b, v): rows.append(a); cols.append(b); vals.append(v)
    sgn = -1.0 if kind == "TM" else 1.0              # the ghost value beyond a metal face
    for i in range(nr):
        for k in range(nphi):
            if not vac[i, k]: continue
            p = idx[i, k]
            diag = 0.0
            # radial: (1/r) d_r (r d_r u): faces at r_i +- h/2
            for di, rf in ((1, r[i] + h / 2), (-1, r[i] - h / 2)):
                c = rf / (r[i] * h * h)
                ii = i + di
                if ii < 0: continue                  # the axis: face of radius 0
                if ii >= nr or not vac[ii, k]: diag += c * (sgn - 1.0)   # metal face (wall included)
                else: add(p, idx[ii, k], c); diag -= c
            c = 1.0 / (r[i] * dp) ** 2
            for dk in (1, -1):
                kk = (k + dk) % nphi
                if not vac[i, kk]: diag += c * (sgn - 1.0)
                else: add(p, idx[i, kk], c); diag -= c
            add(p, p, diag)
    L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    ev = spl.eigs(-L, k=nev + 2, sigma=0, which="LM", return_eigenvectors=False)
    ev = np.sort(np.real(ev))
    ev = ev[ev > 1e-6]
    return np.sqrt(ev[:nev])


# --- the shapes of the tests (code units) ---
R = 1.0
B_SQ = 0.6                                           # square guide |x|, |y| < b
def square_metal(r, p):
    return (np.abs(r * np.cos(p)) > B_SQ) | (np.abs(r * np.sin(p)) > B_SQ) | (r > R)
R_IN, W_RIDGE = 0.6, np.pi / 16                    # four ridges r > R_IN, |phi - n pi/2| < W_RIDGE
def ridge_metal(r, p):
    q = np.mod(p + np.pi / 4, np.pi / 2) - np.pi / 4
    return ((r > R_IN) & (np.abs(q) < W_RIDGE)) | (r > R)


if __name__ == "__main__":
    a = 2 * B_SQ
    print("square a = %.3f: analytic TM11 %.6f, TE10 %.6f" % (a, np.sqrt(2) * np.pi / a, np.pi / a))
    for M in (4, 8, 12, 16):
        for nr in (40,):
            tm = discrete(square_metal, nr, R / nr, M, "TM", 3)
            te = discrete(square_metal, nr, R / nr, M, "TE", 3)
            print("square discrete M=%2d nr=%d: TM %s  TE %s" % (M, nr, np.array2string(tm, precision=6), np.array2string(te, precision=6)))
    # ridged pipe continuum: polar FD with nphi a multiple of 4 * the ridge cells; ridge edges on cell faces need
    # W_RIDGE / dphi integer: dphi = W_RIDGE / q
    print("ridged pipe R_in = %.2f, half width %.2f rad" % (R_IN, W_RIDGE))
    res = {}
    for kind in ("TM", "TE"):
        vals = []
        for s in (1, 2, 4):
            nr = 20 * s
            # angular cells: the ridge half width = q cells, the gap pi/4 - W = ... not commensurate in general: take
            # nphi = round(2 pi / dphi) with dphi = W / (2 s) and check the edge mismatch below
            dp = W_RIDGE / (2 * s)                   # the ridge edges and r = R_IN on cell faces
            nphi = 64 * s
            v = polar_fd(lambda r, p: ~ridge_metal(r, p), R, nr, nphi, kind, 4)
            vals.append(v)
            print("  polar FD %s nr=%d nphi=%d: %s" % (kind, nr, nphi, np.array2string(v, precision=6)))
        e = vals[-1] + (vals[-1] - vals[-2]) / 3
        print("  Richardson %s: %s" % (kind, np.array2string(e, precision=6)))
        res[kind] = e
    for M in (4, 8, 12, 16):
        tm = discrete(ridge_metal, 40, R / 40, M, "TM", 3)
        te = discrete(ridge_metal, 40, R / 40, M, "TE", 3)
        print("ridge discrete M=%2d nr=40: TM %s  TE %s" % (M, np.array2string(tm, precision=6), np.array2string(te, precision=6)))


def discrete_fd(metal, nr, dr, M, kind="TM", nev=6):
    """As discrete(), with the staggered angular grid of the metal block (decision 103): E_z, E_r, B_phi at phi_k,
    E_phi, B_r, B_z at phi_k + dphi/2, the phi derivative the local difference over dphi = 2 pi / (2M - 1) (in the modes:
    m -> m_eff = (2 / dphi) sin(m dphi / 2)). Masks from the node bits Mn(j, k) at the integer angles: half nodes in r
    metal if both nodes are, half angles if both angles are, B_z cells if all four corners are."""
    Np = 2 * M - 1
    dp = 2 * np.pi / Np
    phi = dp * np.arange(Np)
    rn = np.arange(nr + 1) * dr
    rh = (np.arange(nr) + 0.5) * dr
    A = rn * dr
    A[0] = rh[0] ** 2 / 2
    Bw = rh * dr
    Mn = metal(rn[:, None], phi[None, :]).astype(bool)
    Mn[nr, :] = True
    Mh = Mn[:-1] & Mn[1:]                            # (j + 1/2, k)
    Mp = Mn & np.roll(Mn, -1, axis=1)                # (j, k + 1/2)
    Mc = Mh & np.roll(Mh, -1, axis=1)                # (j + 1/2, k + 1/2)
    nn = nr * Np
    def ni(j, k): return j * Np + (k % Np)
    G = sp.lil_matrix((2 * nn, nn)); C = sp.lil_matrix((nn, 2 * nn))
    jj = np.repeat(np.arange(nr), Np)
    if kind == "TM":                                 # Ez (j,k); Bp (j+1/2,k) | Br (j,k+1/2)
        for j in range(nr):
            for k in range(Np):
                if j + 1 < nr: G[ni(j, k), ni(j + 1, k)] += 1 / dr
                if j == 0:
                    for l in range(Np): G[ni(0, k), ni(0, l)] -= 1 / dr / Np
                else: G[ni(j, k), ni(j, k)] -= 1 / dr
                if j >= 1:
                    G[nn + ni(j, k), ni(j, k + 1)] -= 1 / (rn[j] * dp)
                    G[nn + ni(j, k), ni(j, k)] += 1 / (rn[j] * dp)
                if j == 0:
                    for l in range(Np): C[ni(0, k), ni(0, l)] += rh[0] / A[0] / Np
                else:
                    C[ni(j, k), ni(j, k)] += rh[j] / A[j]
                    C[ni(j, k), ni(j - 1, k)] -= rh[j - 1] / A[j]
                    C[ni(j, k), nn + ni(j, k)] -= 1 / (rn[j] * dp)
                    C[ni(j, k), nn + ni(j, k - 1)] += 1 / (rn[j] * dp)
        pe = (~Mn[:nr]).ravel().astype(float)
        pb = np.concatenate([(~Mh).ravel(), (~Mp[:nr]).ravel() * (jj > 0)]).astype(float)
    else:                                            # Bz (j+1/2,k+1/2); Er (j+1/2,k) | Ep (j,k+1/2)
        for j in range(nr):
            for k in range(Np):
                G[ni(j, k), ni(j, k)] += 1 / (rh[j] * dp)
                G[ni(j, k), ni(j, k - 1)] -= 1 / (rh[j] * dp)
                if j >= 1:
                    G[nn + ni(j, k), ni(j, k)] -= 1 / dr
                    G[nn + ni(j, k), ni(j - 1, k)] += 1 / dr
                if j + 1 < nr: C[ni(j, k), nn + ni(j + 1, k)] -= rn[j + 1] / Bw[j]
                if j >= 1: C[ni(j, k), nn + ni(j, k)] += rn[j] / Bw[j]
                C[ni(j, k), ni(j, k + 1)] += 1 / (rh[j] * dp)
                C[ni(j, k), ni(j, k)] -= 1 / (rh[j] * dp)
        pe = (~Mc).ravel().astype(float)
        pb = np.concatenate([(~Mh).ravel(), (~Mp[:nr]).ravel() * (jj > 0)]).astype(float)
    Pe, Pb = sp.diags(pe), sp.diags(pb)
    L = (Pe @ C.tocsr() @ Pb @ G.tocsr() @ Pe).toarray()
    keep = pe > 0
    if kind == "TM":
        ax = np.zeros(nn, bool); ax[1:Np] = True
        keep &= ~ax
        L[:, 0] += L[:, 1:Np].sum(axis=1)
    Lr = L[np.ix_(keep, keep)]
    ev = np.sort(np.real(-np.linalg.eigvals(Lr)))
    ev = ev[ev > 1e-8 * max(ev.max(), 1)]
    return np.sqrt(ev[:nev])
