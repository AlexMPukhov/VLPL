#!/usr/bin/env python3
"""Residual force of the NCI modes on the streaming plasma (bench/nci/mechanism, candidate 1 'matched gather').
2D periodic run with E, B dumped every step (openPMD). The plasma streams with v_x = beta c (beta < 0 here). Per Fourier
mode (ky, kx) at step s2: F_y = E_y - v_x B_z, F_z = E_z + v_x B_y, F_x = E_x (cold beam). RIP-CC / iRIP place (E_y, B_z) and
(E_z, B_y) at the same points and the particles gather them with identical weights at the same time (decision 4), so the
ratio of the Fourier coefficients is the force the particles see (the common shape factor drops out).
Reference: a wave exp(i(k x - omega t)) of a continuum medium has B_z / E_y = c k_x / omega (TM part, Faraday), so
F_y / E_y = 1 - beta c k_x / omega: printed with the measured omega of the mode. Equal numbers mean the gather adds no mismatch.
  force.py RUN UX [s1 s2] [--n0 0.004] [--npz out.npz]"""
import sys, os, glob, numpy as np, h5py
a, opt, args = [], {}, sys.argv[1:]
while args:
    x = args.pop(0)
    if x.startswith("--"): opt[x[2:]] = args.pop(0)
    else: a.append(x)
run, ux = a[0], float(a[1]); n0 = float(opt.get("n0", "0.004"))
steps, F = [], []
for fn in sorted(glob.glob(os.path.join(run, "openpmd", "data_*.h5"))):
    with h5py.File(fn, "r") as h:
        it = list(h["data"])[0]; g = h["data"][it]; m = g["meshes"]
        if not steps:
            sp = m["E"].attrs["gridSpacing"]; D = float(sp[-1]); dt = float(g.attrs["dt"])
        if int(it) == 0: continue
        steps.append(int(it)); F.append(np.fft.fft2(np.stack([np.asarray(m[q][c])[0] for q in ("E", "B") for c in ("x", "y", "z")])))
steps = np.array(steps); F = np.array(F)                    # (t, 6, ky, kx): Ex Ey Ez Bx By Bz
W = (np.abs(F) ** 2).sum(1); Wt = W.sum((1, 2))
if len(a) > 3: s1, s2 = int(a[2]), int(a[3])
else:
    L = np.log(Wt); n = len(L); w = max(n // 5, 10); j0 = n // 8
    i0 = j0 + int(np.argmax([L[i + w] - L[i] for i in range(j0, n - w)])); s1, s2 = steps[i0], steps[i0 + w]
i1, i2 = int(np.searchsorted(steps, s1)), int(np.searchsorted(steps, s2))
beta = ux / np.sqrt(1 + ux * ux); g = np.sqrt(1 + ux * ux); vx = beta
ny, nx = W.shape[1:]
kx = np.fft.fftfreq(nx) * 2 * np.pi; ky = np.fft.fftfreq(ny) * 2 * np.pi
comp = np.argmax(np.abs(F[i2]), axis=0)
A = np.take_along_axis(F[i1:i2 + 1], comp[None, None], 1)[:, 0]
wt = -np.angle((A[1:] * np.conj(A[:-1])).sum(0)) / np.diff(steps[i1:i2 + 1]).mean()    # omega tau of each mode
Ex, Ey, Ez, Bx, By, Bz = F[i2]
Fy = Ey - vx * Bz; Fz = Ez + vx * By
Ep2 = np.abs(Ey) ** 2 + np.abs(Ez) ** 2
rperp = np.sqrt((np.abs(Fy) ** 2 + np.abs(Fz) ** 2) / (Ep2 + 1e-300))
rx = np.abs(Ex) / np.sqrt(Ep2 + 1e-300)
print("run %s: gamma %.2f, steps %d-%d, 1 - |beta| = %.4g, 1/(2 gamma^2) = %.4g" % (run, g, s1, s2, 1 - abs(beta), 0.5 / g ** 2))
print("  kx D/pi  ky h/pi   share   omega tau/pi  |F_perp|/|E_perp|  continuum 1 - beta c kx/omega  |E_x|/|E_perp|   F_x^2/|F|^2")
order = np.argsort(W[i2].ravel())[::-1][:12]
rows = []
for q in order:
    j, i = np.unravel_index(q, W[i2].shape)
    cont = abs(1 - beta * kx[i] / wt[j, i]) if wt[j, i] != 0 else np.nan     # c dt = D: c kx / omega = kx D / (omega tau)
    fx2 = np.abs(Ex[j, i]) ** 2; fp2 = np.abs(Fy[j, i]) ** 2 + np.abs(Fz[j, i]) ** 2
    rows.append((kx[i] / np.pi, ky[j] / np.pi, W[i2][j, i] / W[i2].sum(), wt[j, i] / np.pi, rperp[j, i], cont, rx[j, i], fx2 / (fx2 + fp2)))
    print("  %7.3f  %7.3f   %6.3f   %9.4f      %10.4g        %10.4g                  %8.3g        %8.3g" % rows[-1])
# energy-weighted over the strong modes (> 1e-2 of the peak)
sel = W[i2] > 1e-2 * W[i2].max(); w8 = W[i2][sel]
fx2 = np.abs(Ex[sel]) ** 2; fp2 = np.abs(Fy[sel]) ** 2 + np.abs(Fz[sel]) ** 2
print("SUMMARY strong modes %d: weighted |F_perp|/|E_perp| %.4g, weighted continuum %.4g, share of F_x in |F|^2 %.3f"
      % (sel.sum(), np.sqrt((w8 * rperp[sel] ** 2).sum() / w8.sum()),
         np.sqrt((w8 * np.where(np.isfinite(1 - beta * np.outer(np.ones(ny), kx)[sel] / wt[sel]), (1 - beta * np.outer(np.ones(ny), kx)[sel] / wt[sel]) ** 2, 0)).sum() / w8.sum()),
         (w8 * fx2 / (fx2 + fp2)).sum() / w8.sum()))
FX2 = np.abs(Ex) ** 2; FP2 = np.abs(Fy) ** 2 + np.abs(Fz) ** 2
KX = np.outer(np.ones(ny), kx); CONT = np.abs(1 - beta * KX / np.where(wt == 0, np.nan, wt))
if "npz" in opt: np.savez(opt["npz"], rows=np.array(rows), fxs=FX2 / (FX2 + FP2 + 1e-300), cont=CONT, kx=kx, ky=ky, W=W[i2], rperp=rperp, rx=rx, wt=wt, beta=beta, s1=s1, s2=s2)
