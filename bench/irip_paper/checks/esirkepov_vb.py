"""Esirkepov vs Villasenor-Buneman (linear shapes, 2D, unit cells, q = dt = 1).
VB = straight-path current, built as Esirkepov on the sub-segments between cell-boundary crossings
(inside one cell the two are identical). Prints max |J_E - J_VB| and the divergence of the difference."""
import numpy as np
N = 8
def S(X):   # linear shape on nodes 0..N-1
    k = np.arange(N); return np.maximum(0, 1 - np.abs(X - k))
def esir(x0, y0, x1, y1):
    S0x, S1x, S0y, S1y = S(x0), S(x1), S(y0), S(y1)
    DSx, DSy = S1x - S0x, S1y - S0y
    Wx = np.outer(DSx, S0y + 0.5 * DSy); Wy = np.outer(S0x + 0.5 * DSx, DSy)
    Jx = -np.cumsum(Wx, 0); Jy = -np.cumsum(Wy, 1)      # Jx at (k+1/2, l), Jy at (k, l+1/2)
    return Jx, Jy
def vb(x0, y0, x1, y1):
    ts = [0.0, 1.0]
    for a0, a1 in ((x0, x1), (y0, y1)):
        if a1 != a0:
            for g in range(int(np.floor(min(a0, a1))) + 1, int(np.floor(max(a0, a1))) + 1):
                ts.append((g - a0) / (a1 - a0))
    ts = sorted(set(ts)); Jx = 0; Jy = 0
    for ta, tb in zip(ts[:-1], ts[1:]):
        jx, jy = esir(x0 + ta * (x1 - x0), y0 + ta * (y1 - y0), x0 + tb * (x1 - x0), y0 + tb * (y1 - y0))
        Jx = Jx + jx; Jy = Jy + jy
    return Jx, Jy
def div(Jx, Jy):
    return (Jx - np.roll(Jx, 1, 0)) + (Jy - np.roll(Jy, 1, 1))
rng = np.random.default_rng(1)
for case, (x0, y0, x1, y1) in {"inside one cell": (3.2, 3.3, 3.7, 3.9),
                               "crosses y = 4": (3.2, 3.7, 3.6, 4.4),
                               "crosses x and y": (3.8, 3.7, 4.3, 4.4)}.items():
    JE, JV = esir(x0, y0, x1, y1), vb(x0, y0, x1, y1)
    d = (JE[0] - JV[0], JE[1] - JV[1])
    print("%-16s max|J_E - J_VB| = %.3f   max|div(J_E - J_VB)| = %.1e" % (case, max(np.abs(d[0]).max(), np.abs(d[1]).max()), np.abs(div(*d)).max()))
# curl of the difference: a current loop around the crossed node
x0, y0, x1, y1 = 3.2, 3.7, 3.6, 4.4
JE, JV = esir(x0, y0, x1, y1), vb(x0, y0, x1, y1)
print("J_E - J_VB, x-component (rows k, cols l):\n", np.round((JE[0] - JV[0])[2:5, 2:6], 4))
print("y-component:\n", np.round((JE[1] - JV[1])[2:6, 2:5], 4))
