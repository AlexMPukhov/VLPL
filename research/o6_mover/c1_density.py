"""Density check of a radial map: a finely sampled uniform plasma must deposit a uniform density (rz0 _dep_charge).
usage: python3 c1_density.py map   (plain | pow | exp | c1)"""
import sys
import numpy as np
from rz0 import RZ0
m = sys.argv[1]
S = RZ0(4, 24, 0.1, 0.1, rmap={"plain": False}.get(m, m))
nr_s = 4000                                   # radial samples per cell (midpoint rule in r, weight r dr)
r = (np.arange(16 * nr_s) + 0.5) * S.dr / nr_s
z = np.full_like(r, 0.2)
q = r * S.dr / nr_s * 2 * np.pi * S.D        # n = 1 times the ring volume per unit... (z-integrated by one cell)
rho = S._dep_charge(z, r, q).sum(0)
print(f"map {m}: rho at nodes 0..5 {np.round(rho[:6], 6)}, max |rho - 1| over nodes 0..12: {np.abs(rho[:13] - 1).max():.1e}")
