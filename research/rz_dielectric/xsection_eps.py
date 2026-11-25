#!/usr/bin/env python3
"""TM cutoffs (k_z = 0) of a round metal pipe (radius R) with eps(x, y): the continuum reference of test D11
(tests/test_rz_dielectric.cpp, eps depending on phi). Independent of VLPL: polar finite differences.

TM at k_z = 0 (c = 1): -lap E_z = w^2 eps E_z, E_z(R) = 0. Cell-centred polar grid r_j = (j - 1/2) h (no axis point; the
flux through r = 0 vanishes), phi_k = (k + 1/2) 2 pi / nphi; eps per cell from 8 x 8 sub-samples (arithmetic). The
symmetric generalized problem A e = w^2 B e (A: the flux form times r, B = diag(eps r)) by shift-invert eigsh; Richardson
over three grids (second order).

  python3 xsection_eps.py      # the half-filled pipe of test D11: eps = 2.1 for x < 0, R = 1
"""
import math
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def cutoffs(eps, R, N, nphi, nev=4, sigma=1.0):
    h = R / N
    dp = 2 * math.pi / nphi
    rj = (np.arange(N) + 0.5) * h
    S = 8
    E = np.zeros((N, nphi))
    for a in range(S):
        r = rj - 0.5 * h + (a + 0.5) * h / S
        for b in range(S):
            p = (np.arange(nphi) + 0.5) * dp - 0.5 * dp + (b + 0.5) * dp / S
            E += eps(np.outer(r, np.cos(p)), np.outer(r, np.sin(p)))
    E /= S * S
    n = N * nphi
    idx = lambda j, k: j * nphi + (k % nphi)
    rows, cols, vals = [], [], []
    for j in range(N):
        rp, rm = rj[j] + 0.5 * h, rj[j] - 0.5 * h
        for k in range(nphi):
            q = idx(j, k)
            diag = 0.0
            # radial: -(1/r) d/dr (r dE/dr), times r h^2 -> -(rp (E+ - E) - rm (E - E-))
            if j + 1 < N:
                rows += [q]; cols += [idx(j + 1, k)]; vals += [-rp / h**2]; diag += rp / h**2
            else:
                diag += 2 * rp / h**2          # Dirichlet at R: ghost E_N = -E_{N-1}
            if j > 0:
                rows += [q]; cols += [idx(j - 1, k)]; vals += [-rm / h**2]; diag += rm / h**2
            # azimuthal: -(1/r^2) d2E/dphi2, times r
            c = 1.0 / (rj[j] * dp**2)
            rows += [q, q]; cols += [idx(j, k + 1), idx(j, k - 1)]; vals += [-c, -c]; diag += 2 * c
            rows += [q]; cols += [q]; vals += [diag]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    B = sp.diags((E * rj[:, None]).ravel())
    w2 = sla.eigsh(A, k=nev, M=B, sigma=sigma**2, which="LM", return_eigenvectors=False)
    return np.sort(np.sqrt(w2))


if __name__ == "__main__":
    R = 1.0
    half = lambda x, y: np.where(x < 0, 2.1, 1.0)
    out = []
    for N in (100, 200, 400):
        w = cutoffs(half, R, N, 4 * N, nev=4, sigma=1.8)
        out.append(w)
        print("N = %d, nphi = %d:" % (N, 4 * N), " ".join("%.8f" % x for x in w))
    w1, w2, w3 = out
    rich = w3 + (w3 - w2) / 3
    print("Richardson (2nd order):", " ".join("%.8f" % x for x in rich), " order check:",
          " ".join("%.2f" % (math.log(abs((a - b) / (b - c))) / math.log(2)) for a, b, c in zip(w1, w2, w3)))
    # check: the empty pipe, TM01 = j01 / R
    print("vacuum check:", cutoffs(lambda x, y: np.ones_like(x), R, 200, 800, nev=1, sigma=2.4)[0], 2.404825557695773)
    sys.exit(0)
