"""Resolution study (small box 12 lambda): python3 conv2d.py <mirror_p|lossy_p|glass_p|...> <cells per lambda>"""
import numpy as np, sys, json, time
from box2d import Box2D, Medium
from setup import W0, beam, set_medium, grid


def fresnel(eps, cth, pol):
    q = np.sqrt(eps - (1 - cth ** 2) + 0j)
    q = np.where(q.imag < 0, -q, q)
    return (cth - q) / (cth + q) if pol == "s" else (eps * cth - q) / (eps * cth + q)


name, N = sys.argv[1], int(sys.argv[2])
D = 1.0 / N; DY = D / 2; L = 12.0
nx, ny = int(L / D), int(L / DY)
th = np.deg2rad(22.5)
n = np.array([-1 - np.cos(2 * th), -np.sin(2 * th)]); n /= np.linalg.norm(n)
c = np.array([6.0, 6.0])
kd = np.array([np.cos(2 * th), np.sin(2 * th)])
med = Medium((nx, ny))
pol = name[-1]
if name.startswith("glass"):
    inside = lambda x, y: (x > 6) & (x < 10)
    set_medium(med, nx, ny, D, DY, inside, wp=np.sqrt(10) * W0, w0=3 * W0)
    eps = lambda w: 1 + 10 * W0 ** 2 / (9 * W0 ** 2 - w ** 2)
    start, T, nrm = c - 4 * kd, 9.0, np.array([-1.0, 0.0])
else:
    inside = lambda x, y: (((x - c[0]) * n[0] + (y - c[1]) * n[1]) < 0) & (((x - c[0]) * n[0] + (y - c[1]) * n[1]) > -1.5)
    if name.startswith("mirror"):
        set_medium(med, nx, ny, D, DY, inside, wp=10 * W0); eps = lambda w: 1 - 100 * W0 ** 2 / w ** 2
    else:
        set_medium(med, nx, ny, D, DY, inside, wp=3 * W0, nu=W0); eps = lambda w: 1 - 9 * W0 ** 2 / (w * (w + 1j * W0))
    start, T, nrm = c - 5 * kd, 12.0, n
s = Box2D(nx, ny, D, DY, med, tol=1e-9)
beam(s, pol, start[0], start[1], 45, 1.5, 1.5)
comp = 5 if pol == "p" else 2
# Fresnel expectation, spectrally weighted
G = np.abs(np.fft.fft2(s.f[comp])) ** 2
kx = 2 * np.pi * np.fft.fftfreq(nx, D)[:, None] * np.ones((1, ny)); ky = 2 * np.pi * np.fft.fftfreq(ny, DY)[None, :] * np.ones((nx, 1))
k = np.hypot(kx, ky); k[k == 0] = 1
cth = -(kx * nrm[0] + ky * nrm[1]) / k
G = np.where(cth > 0, G, 0)
R = np.nan_to_num(np.abs(fresnel(eps(k), np.clip(cth, 1e-6, 1), pol)) ** 2)
R_th = float((G * R).sum() / G.sum())
W0e = s.energy()
t0 = time.time()
for it in range(int(round(T / D))):
    s.step()
X, Y = grid(nx, ny, D, DY)
e = 0.5 * sum(a ** 2 for a in s.f)
out = dict(name=name, N=N, R_fresnel=R_th, energy_err=abs(s.energy() + med.energy() + med.diss - W0e) / W0e,
           gmres=float(np.mean(s.iters)), cpu=time.time() - t0)
if name.startswith("glass"):
    out["R_num"] = float(e[X < 6].sum() / W0e)
else:
    dist = (X - c[0]) * n[0] + (Y - c[1]) * n[1]
    out["R_num"] = float(e[(dist > 1.5) | (dist < -2.0)].sum() / W0e)   # specular beam (wraps behind the slab)
    out["absorbed"] = float(med.diss / W0e)
    out["surface_waves"] = float((e[(dist < 1.5) & (dist > -2.0)].sum() + med.energy()) / W0e)
print(json.dumps(out), flush=True)
