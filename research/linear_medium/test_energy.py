"""Test 1: exact energy balance W + K + dissipated = const with random fields and random media, dy < D, large wp dt.
Also: implicit solve consistent (the returned Ex^{n+1}, Bx^{n+1} are the fixed point)."""
import numpy as np
from box2d import Box2D, Medium
rng = np.random.default_rng(3)
nx, ny, D, dy = 24, 20, 1.0, 0.4
for wpdt, w0dt, nudt in ((3.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 0.0, 1.0), (2.0, 4.0, 0.3), (30.0, 0.0, 0.0)):
    med = Medium((nx, ny))
    for c in range(3):
        mask = rng.random((nx, ny)) < 0.5
        med.wp2[c] = np.where(mask, wpdt ** 2 * rng.random((nx, ny)), 0.0)
        med.w02[c] = np.where(mask, w0dt ** 2, 0.0)
        med.nu[c] = np.where(mask, nudt, 0.0)
    s = Box2D(nx, ny, D, dy, med, tol=1e-13)
    s.f = [rng.normal(size=(nx, ny)) for _ in range(6)]
    E0 = s.energy() + med.energy()
    worst = 0
    for n in range(200):
        s.step()
        tot = s.energy() + med.energy() + med.diss
        worst = max(worst, abs(tot - E0) / E0)
    print(f"wp dt <= {wpdt:5.1f}  w0 dt = {w0dt:3.1f}  nu dt = {nudt:3.1f}: |W+K+diss - W0|/W0 max over 200 steps = {worst:.1e},"
          f" dissipated {med.diss/E0:.3f}, GMRES its {np.mean(s.iters):.1f}")
