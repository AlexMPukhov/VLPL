"""On-axis E_y of bench/flying_focus/rip_vacuum.deck against the analytic luminal Bessel pulse, in the metrics of
GEM-PIC-RZ's benchmarks/flying_focus/check_axis.py (max |E|, rms difference within 2 L of the centre, shift of the
envelope centroid), plus the fitted amplitude and carrier phase.
  E_y = a0 exp(-zeta'^2 / L^2) cos(cos(theta) k0 zeta' - kappa omega0 t + cep),  zeta' = x - c t - x_center
usage: python3 check_rip.py out_dir [a0 theta[rad] tau_fwhm[fs] x_center[um] axis_j lambda_um]"""
import os
import sys
import numpy as np

d = sys.argv[1]
a0, th, tau, xc, ja, lam = 1.0, 0.2, 20.0, 22.5, 40, 0.8
if len(sys.argv) > 2:
    a0, th, tau, xc = map(float, sys.argv[2:6])
    ja, lam = int(sys.argv[6]), float(sys.argv[7])
k0 = 2 * np.pi / lam
L = tau * 0.299792458 / np.sqrt(2 * np.log(2))    # um, 1/e half length of the field
kap = 1 - np.cos(th)
meta = dict(l.split(None, 1) for l in open(os.path.join(d, "slices", "meta.txt")) if l.strip() and not l.startswith("#"))
nx, ny = int(meta["nx"].split()[0]), int(meta["ny"].split()[0])
dx = float(meta["dx"].split()[0]) / k0 if "dx" in meta else float(meta["h"].split()[0]) / k0
print("# t[fs]  z[mm]  max|Ey|  max|ref|  rms(Ey - ref)/a0  centroid shift [um]  fitted amplitude / a0  phase [rad]")
for line in open(os.path.join(d, "slices", "index.txt")):
    if line.startswith("#") or not line.strip():
        continue
    w = line.split()
    step, t = int(w[0]), float(w[1])                   # t in 1/omega0
    f = os.path.join(d, "slices", "Ey_%06d.f32" % step)
    if not os.path.exists(f):
        f = os.path.join(d, "slices", "Ey_%d.f32" % step)
    A = np.fromfile(f, dtype=np.float32).reshape(ny, nx)
    ey = A[ja].astype(float)
    zeta = np.arange(nx) * dx - xc                      # zeta' (the window moves one cell per step from t = 0)
    tt = t / k0                                         # um / c
    env = a0 * np.exp(-(zeta / L) ** 2)
    ph = np.cos(th) * k0 * zeta - kap * k0 * tt
    ref = env * np.cos(ph)
    m = np.abs(zeta) < 2 * L
    shift = np.sum(zeta[m] * ey[m] ** 2) / np.sum(ey[m] ** 2)
    # least squares ey = a cos(ph) + b sin(ph) times the envelope: amplitude and phase relative to the analytic carrier
    M = np.vstack([env[m] * np.cos(ph[m]), env[m] * np.sin(ph[m])]).T
    (a, b), *_ = np.linalg.lstsq(M, ey[m], rcond=None)
    print(f"{t / k0 * 1e-6 / 299792458 * 1e15:7.1f} {tt * 1e-3:7.4f} {np.abs(ey).max():8.5f} {np.abs(ref).max():8.5f} "
          f"{np.sqrt(np.mean((ey - ref)[m] ** 2)) / a0:9.2e} {shift:8.4f} {np.hypot(a, b) / a0:9.5f} {np.arctan2(-b, a):9.5f}")
