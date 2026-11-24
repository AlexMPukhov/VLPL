"""A C1 density-conserving shape-2 radial map (O6 in RZ). The exp map of axis_map.py conserves the density with a bounded
slope, but dxi/dr jumps at every cell edge (cells c >= 1: slope 1 + b_c at the left end, 1 - b_c at the right end), and the
energy-consistent radial force ~ dxi/dr jumps with it: the implicit solve stalls.

C1 family: in cell c (u = r/dr - c in [0, 1])
    xi = c + u + (s_c - 1)(u^3 - 2u^2 + u) + (s_{c+1} - 1)(u^3 - u^2) + a_c u^2 (1 - u)^2
with node slopes s_0 (axis, free) and s_c = 1 for c >= 1; the bump a_c u^2 (1 - u)^2 moves charge inside the cell and leaves the
end values and end slopes alone. dxi/dr is continuous everywhere (also across the axis: xi(-r) = -xi(r) is odd).
Unknowns a_0 .. a_{K-1} for a given s_0 (one per node share); scan s_0 for the smallest largest slope with a monotone map.
usage: python3 c1map.py [s0 ...]
"""
import sys
import numpy as np
from rz0 import bspline, GX, GW

dr = 1.0


S1 = [1.0]          # the slope at node 1 (1: the map of the C++ code; other values: c1scan.py)


def _ends(c, s0):
    sl = s0 if c == 0 else (S1[0] if c == 1 else 1.0)
    sr = S1[0] if c == 0 else 1.0
    return sl, sr


UC = [0.0]          # the linear core xi = s0 u, u < UC, of the axis cell (the C++ map, rz.hpp rz_c1_cell)


def _herm(v, sl, sr, a):
    w = 1 - v
    return v + (sl - 1) * v * w * w - (sr - 1) * v * v * w + a * v * v * w * w, \
        1 + (sl - 1) * w * (1 - 3 * v) - (sr - 1) * v * (2 - 3 * v) + a * 2 * v * w * (1 - 2 * v)


def _cell(c, u, a, s0):
    sl, sr = _ends(c, s0)
    uc = UC[0] if c == 0 else 0.0
    if uc == 0:
        h, d = _herm(u, sl, sr, a[c])
        return c + h, d
    x0 = s0 * uc
    v = np.clip((u - uc) / (1 - uc), 0, 1)
    f = (1 - uc) / (1 - x0)
    h, d = _herm(v, sl * f, sr * f, a[c])
    xi = np.where(u < uc, s0 * u, x0 + (1 - x0) * h)
    dd = np.where(u < uc, s0, (1 - x0) * d / (1 - uc))
    return xi, dd


def xi_c(c, u, a, s0):
    return _cell(c, u, a, s0)[0]


def dxi_c(c, u, a, s0):
    return _cell(c, u, a, s0)[1]


def share(a, s0, j):
    tot = 0.0
    for c in range(max(0, j - 2), j + 2):
        if c + 1 >= len(a): continue
        # break points: where xi crosses a half-integer (the S2 kinks) inside the cell
        uu = np.linspace(0, 1, 401); xx = xi_c(c, uu, a, s0)
        br = [0.0]
        for h in np.arange(np.floor(xx.min()) + 0.5, xx.max(), 1.0):
            k = np.nonzero((xx[:-1] - h) * (xx[1:] - h) <= 0)[0]
            for i in k:
                lo, hi = uu[i], uu[i + 1]
                for _ in range(60):
                    m = 0.5 * (lo + hi)
                    if (xi_c(c, m, a, s0) - h) * (xi_c(c, lo, a, s0) - h) <= 0: hi = m
                    else: lo = m
                br.append(0.5 * (lo + hi))
        br = sorted(set(br + [1.0]))
        for p in range(len(br) - 1):
            lo, hi = br[p], br[p + 1]
            u = 0.5 * (lo + hi) + GX * 0.5 * (hi - lo)
            xi = xi_c(c, u, a, s0)
            W = bspline(xi - j, 2) + (bspline(xi + j, 2) if j > 0 else 0)
            tot += (GW * 0.5 * (hi - lo) * W * (c + u) * dr * dr).sum()
    return tot


def solve(s0, K=36, a0=None):
    A = np.array([dr * dr / 8] + [j * dr * dr for j in range(1, K + 4)])
    a = np.zeros(K + 4)
    a[1:] = [4 / (2 * j + 1) for j in range(1, K + 4)] if a0 is None else a0[1:]
    if a0 is not None: a[0] = a0[0]
    for it in range(40):
        R = np.array([share(a, s0, j) - A[j] for j in range(K)])
        res = np.max(np.abs(R) / A[:K])
        if res < 1e-13: break
        Jm = np.zeros((K, K))
        for c in range(K):
            keep = a[c]; a[c] = keep + 1e-7
            for j in range(max(0, c - 3), min(K, c + 3)):
                Jm[j, c] = (share(a, s0, j) - A[j] - R[j]) / 1e-7
            a[c] = keep
        a[:K] += np.linalg.solve(Jm, -R)
        a[K:] = a[K - 1] * (2 * K - 1) / (2 * np.arange(K, len(a)) + 1)     # tail ~ 1/(2j+1)
    uu = np.linspace(0, 1, 1001)
    sl = np.array([dxi_c(c, uu, a, s0) for c in range(K)])
    return a, res, it, sl.max(), sl.min()


if __name__ == "__main__":
    a = None
    for s0 in [float(x) for x in sys.argv[1:]] or [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        a, res, it, smax, smin = solve(s0, a0=a)
        print(f"s0 = {s0}: residual {res:.1e} ({it} its), slope max {smax:.3f} min {smin:.3f} "
              f"{'monotone' if smin > 0 else 'NOT monotone'}, a_0..4 = {np.round(a[:5], 4)}, a_35 * 71 = {a[35] * 71:.4f}", flush=True)
