"""A density-conserving shape-2 radial map with a bounded slope at the axis (O6 in RZ; decision 53 has xi = u^p, p = 0.328,
in the axis cell: dxi/dr ~ r^(p-1) unbounded, which stalls the implicit mover's solve near the axis).

Shares: node j must receive its finite-volume area A_j (A_0 = dr^2/8, A_j = j dr^2) from a uniform density, with S2 folded
across the axis. Cells j >= 1 keep xi = j + u + b_j u (1 - u). Candidate axis-cell maps (u = r/dr in [0, 1], xi(0) = 0, xi(1) = 1):
  pow   : xi = u^p                                   (decision 53; slope unbounded)
  quad  : xi = u + b u (1 - u)                       (slope 1 + b, |b| <= 1)
  shift : xi = ((u + e)^p - e^p) / ((1 + e)^p - e^p)  (slope at 0: p e^(p-1) / norm, bounded for e > 0; e sets the cap)
For shift, p is the unknown of node 0's equation at a fixed e; we report the slope at the axis, the largest slope, and whether
Newton finds all shares (to 1e-13).
"""
import numpy as np
from rz0 import bspline, GX, GW

dr = 1.0
NC = 40


def xi_cell(c, u, b, axis):
    if c == 0:
        return axis(u)
    return c + u + b[c] * u * (1 - u)


def shares(b, axis, K):
    out = np.zeros(K)
    for c in range(0, K + 2):
        # split each cell at the S2 kink xi = c + 1/2 (bisection) and integrate r dr with Gauss-Legendre
        lo, hi = 0.0, 1.0
        f = lambda u: xi_cell(c, u, b, axis) - (c + 0.5)
        for _ in range(80):
            m = 0.5 * (lo + hi)
            if f(m) < 0: lo = m
            else: hi = m
        uk = 0.5 * (lo + hi)
        for a, e in ((0.0, uk), (uk, 1.0)):
            u = 0.5 * (a + e) + GX * 0.5 * (e - a)
            w = GW * 0.5 * (e - a) * (c + u) * dr * dr
            xi = np.array([xi_cell(c, v, b, axis) for v in u])
            for j in range(max(0, c - 2), min(K, c + 3)):
                W = bspline(xi - j, 2) + (bspline(xi + j, 2) if j > 0 else 0)
                out[j] += (w * W).sum()
    return out


A = np.array([dr * dr / 8] + [j * dr * dr for j in range(1, NC + 4)])


def solve(axis_of, p0, K=NC):
    """unknowns: the axis parameter p (axis_of(p) -> map) and b_1 .. b_{K-1}"""
    b = np.zeros(K + 3)
    for j in range(1, K + 3): b[j] = (dr * dr / 12) / (j * dr * dr / 6 + dr * dr / 12)
    x = np.concatenate([[p0], b[1:K]])
    for it in range(40):
        def R(x):
            bb = b.copy(); bb[1:K] = x[1:]
            return shares(bb, axis_of(x[0]), K) - A[:K]
        r = R(x)
        if np.max(np.abs(r) / A[:K]) < 1e-13: break
        Jm = np.zeros((K, K))
        for c in range(K):
            dx = np.zeros(K); dx[c] = 1e-7
            Jm[:, c] = (R(x + dx) - r) / 1e-7
        x = x - np.linalg.solve(Jm, r)
    bb = b.copy(); bb[1:K] = x[1:]
    return x[0], bb, np.max(np.abs(R(x)) / A[:K]), it


def slope(axis, n=2001):
    u = np.linspace(1e-9, 1, n)
    return np.gradient(axis(u), u)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        p, b, res, it = solve(lambda p: (lambda u: u ** p), 0.33)
        print(f"pow  : p = {p:.6f}, residual {res:.1e} ({it} its), b_1..3 = {b[1:4]}, slope at u=1e-9: {p * 1e-9 ** (p - 1):.1e}")
        q, b, res, it = solve(lambda q: (lambda u: u + q * u * (1 - u)), 0.9)
        print(f"quad : b0 = {q:.6f} (needs |b0| <= 1 for a monotone map), residual {res:.1e}")
    es = [float(a) for a in sys.argv[1:]] or [0.3, 0.1, 0.03, 0.01]
    for e in es:
        m = lambda p, e=e: (lambda u: ((u + e) ** p - e ** p) / ((1 + e) ** p - e ** p))
        p, b, res, it = solve(m, 0.3)
        s = slope(m(p))
        print(f"shift e = {e}: p = {p:.6f}, residual {res:.1e} ({it} its), slope at axis {s[0]:.2f}, max slope {s.max():.2f}, "
              f"min slope {s.min():.3f}, b_1..3 = {np.round(b[1:4], 4)}")
