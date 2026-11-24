# O5 stability: per Fourier mode the spectral radius of G and the transient growth max_n ||G^n||_2 (n <= 400) in the plain
# field norm (E^2 + B^2 summed), over the whole Brillouin zone, for plain iRIP, the paraxial and the full closure.
import numpy as np
from closure2d import make

D = 1.0
nk = 33
for hy_over in (3.2, 1.0, 0.5, 0.2):
    hy = hy_over * D
    for name, (cx, cy) in {"irip": (0, 0), "par": (1, 0), "full": (1, 1)}.items():
        G = make(D, hy, cx, cy)
        rho, grow, worst = 0.0, 0.0, None
        for kx in np.linspace(0, np.pi / D, nk):
            for ky in np.linspace(0, np.pi / hy, nk):
                M = G(kx, ky)
                r = np.abs(np.linalg.eigvals(M)).max()
                P, g = np.eye(6), 1.0
                for _ in range(400):
                    P = M @ P
                    g = max(g, np.linalg.norm(P, 2))
                if r > rho: rho = r
                if g > grow: grow, worst = g, (kx * D, ky * hy)
        print("dy = %.1f dx  %-5s  max spectral radius %.12f   max transient growth %.3f at (kx dx, ky dy) = (%.2f, %.2f)"
              % (hy_over, name, rho, grow, *worst))
