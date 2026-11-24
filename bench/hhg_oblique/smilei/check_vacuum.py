# Vacuum check of the injected oblique beam (run dir = argv[1]): direction, polarisation, focus position, a0, timing.
import sys, h5py, numpy as np
from math import pi
d = sys.argv[1]
lam = 0.8; um = 2*pi/lam; fs = 2*pi*0.299792458/lam
xP, yP, zP = 6.0, 21.5, 5.0; tP = 43. + 5.657/0.299792458
def load(i):
    f = h5py.File(f"{d}/Probes{i}.h5", "r")
    ks = sorted(k for k in f.keys() if k.isdigit())
    pos = f["positions"][()] / um if "positions" in f else None
    return f, ks, pos
# timestep
import re
dt = float(re.search(r"timestep = ([0-9.]+)", open(f"{d}/smilei.log").read()).group(1))
# ---- point P
f, ks, _ = load(0)
A = np.array([f[k][()][:, 0] if f[k][()].ndim == 2 else f[k][()] for k in ks])   # (nt, 6)
t = np.array([int(k) for k in ks]) * dt / fs
E = A[:, :3]; Emag = np.sqrt((E**2).sum(1))
i = np.argmax(Emag)
u = (A[:, :3]**2 + A[:, 3:]**2).sum(1)
# smooth energy density over one period to get the envelope peak time
nper = int(round(2*pi/dt)); us = np.convolve(u, np.ones(nper)/nper, mode="same")
print(f"dt = {dt:.4f} /omega0 = {dt/fs:.4f} fs, steps {len(ks)}")
print(f"P: max |E| = {Emag[i]:.3f} at t = {t[i]:.2f} fs; envelope (period-averaged u) peak at t = {t[np.argmax(us)]:.2f} fs (target {tP:.2f})")
print(f"P: field at max |E|: Ex {A[i,0]:+.3f} Ey {A[i,1]:+.3f} Ez {A[i,2]:+.3f} | Bx {A[i,3]:+.3f} By {A[i,4]:+.3f} Bz {A[i,5]:+.3f}")
print(f"P: energy fractions over the pulse: Ex2 {np.sum(A[:,0]**2):.1f} Ey2 {np.sum(A[:,1]**2):.1f} Ez2 {np.sum(A[:,2]**2):.1f} Bx2 {np.sum(A[:,3]**2):.1f} By2 {np.sum(A[:,4]**2):.1f} Bz2 {np.sum(A[:,5]**2):.1f}")
# Poynting direction at P (time integrated)
S = np.cross(A[:, :3], A[:, 3:]).sum(0); S /= np.linalg.norm(S)
print(f"P: time-integrated Poynting direction ({S[0]:+.4f}, {S[1]:+.4f}, {S[2]:+.4f}); angle to +x in x-y: {np.degrees(np.arctan2(S[1], S[0])):+.2f} deg")
# ---- transverse lines: peak |E| vs position, Gaussian fit
for i, lab in ((1, "transverse in x-y (e = (-sin45, cos45, 0))"), (2, "along z")):
    f, ks, pos = load(i)
    M = np.zeros(len(pos))
    for k in ks:
        a = f[k][()]; M = np.maximum(M, np.sqrt((a[:3]**2).sum(0)))
    r = (pos[:, 2] - zP) if i == 2 else ((pos[:, 0] - xP)*(-np.sqrt(.5)) + (pos[:, 1] - yP)*np.sqrt(.5))
    sel = M > 0.05*M.max()
    c = np.polyfit(r[sel], np.log(M[sel]), 2)        # ln M = c0 r^2 + c1 r + c2
    w = np.sqrt(-1/c[0]); r0 = -c[1]/(2*c[0])
    print(f"line {lab}: max {M.max():.3f}, Gaussian fit w0 = {w:.3f} um (target 2), centre offset {r0:+.3f} um")
# ---- plane z = 5: energy centroid trajectory
f, ks, pos = load(3)
x, y = pos[:, 0], pos[:, 1]
print("plane z=5: energy centroid (x', y) vs t; expected on the line y - yP = x' - xP:")
tr = []
for k in ks:
    a = f[k][()]; w = (a**2).sum(0); W = w.sum()
    if W <= 0: continue
    tt = int(k)*dt/fs
    if w.max() < 1e-2*1e2: continue
    xc, yc = (w*x).sum()/W, (w*y).sum()/W
    tr.append((tt, xc, yc))
tr = np.array(tr)
for row in tr[::max(1, len(tr)//10)]:
    print(f"  t {row[0]:6.1f} fs  x' {row[1]:6.3f}  y {row[2]:7.3f}  (y - yP) - (x' - xP) = {(row[2]-yP)-(row[1]-xP):+.3f}")
sel = (tr[:, 1] > 1.5) & (tr[:, 1] < 7.5)
px = np.polyfit(tr[sel, 0], tr[sel, 1], 1); py = np.polyfit(tr[sel, 0], tr[sel, 2], 1)
v = np.hypot(px[0], py[0]) / 0.299792458
print(f"centroid velocity: angle {np.degrees(np.arctan2(py[0], px[0])):+.2f} deg to +x, |v| = {v:.4f} c; "
      f"centroid passes x' = {xP} at t = {(xP-px[1])/px[0]:.2f} fs, y = {np.polyval(py, (xP-px[1])/px[0]):.3f}")
