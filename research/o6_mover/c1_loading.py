"""Loading error of a uniform density with ppc_r Gauss-Legendre points per radial cell (as Simulation::load_rz), deposited
with the radial map: the code loads at physical r, so the map's steep axis cell is sampled by few points.
usage: python3 c1_loading.py   (maps exp-free: pow (decision 53), c1)"""
import numpy as np
from rz0 import RZ0
for m in ("pow", "c1"):
    S = RZ0(4, 24, 0.1, 0.1, rmap=m)
    for ppr in (2, 4, 8, 16):
        x, w = np.polynomial.legendre.leggauss(ppr)
        u = 0.5 * (x + 1); gw = 0.5 * w
        r = np.concatenate([(j + u) * S.dr for j in range(16)])
        wt = np.concatenate([gw * (j + u) * S.dr * S.dr for j in range(16)])
        q = wt * 2 * np.pi * S.D
        rho = S._dep_charge(np.full_like(r, 0.2), r, q).sum(0)
        print(f"map {m}, ppc_r {ppr}: rho nodes 0..3 {np.round(rho[:4], 5)}, max |rho - 1| nodes 0..12 {np.abs(rho[:13] - 1).max():.1e}")
