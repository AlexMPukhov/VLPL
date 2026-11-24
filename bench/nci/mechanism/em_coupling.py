#!/usr/bin/env python3
"""Laser-plasma coupling through the current filter (bench/nci/mechanism/README.md): a warm plasma at rest in a periodic 2D box
(thermal noise), fields dumped every step. The EM mode with kx = n 2 pi / L, ky = 0 (Ey, Bz) oscillates at the scheme's
frequency; RIP is dispersion-free along x, so sin^2(w tau / 2) - sin^2(k D / 2) is the plasma term alone, R(k) (w_p tau / 2)^2,
with R the filter's response at k. The ratio of a filtered to the unfiltered run is the measured coupling.
  em_coupling.py RUN [n=4]   -> prints k D, w tau, the plasma term
"""
import sys, os, glob, numpy as np, h5py
run = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
files = sorted(glob.glob(os.path.join(run, "openpmd", "data_*.h5")))
ser = []
for fn in files:
    with h5py.File(fn, "r") as h:
        it = list(h["data"])[0]
        if int(it) == 0: continue
        g = h["data"][it]; m = g["meshes"]
        ey = np.asarray(m["E"]["y"])[0].mean(0); bz = np.asarray(m["B"]["z"])[0].mean(0)   # ky = 0
        dt = float(g.attrs["dt"]); D = float(m["E"].attrs["gridSpacing"][-1])
        ser.append(np.fft.fft(ey)[n] + 1j * 0 * bz[0])
a = np.array(ser); N = len(a)
kD = 2 * np.pi * n / len(ey)
win = np.hanning(N); P = 64
S = np.abs(np.fft.fft(a * win, P * N)) ** 2
f = np.fft.fftfreq(P * N) * 2 * np.pi                 # omega tau per step
res = []
for sgn in (1, -1):                                    # the forward and backward wave: peaks at -w and +w
    sel = (np.sign(f) == sgn) & (np.abs(f) > 0.5 * kD) & (np.abs(f) < 2 * kD + 0.5)
    i = np.flatnonzero(sel)[np.argmax(S[sel])]
    y0, y1, y2 = np.log(S[i - 1]), np.log(S[i]), np.log(S[i + 1])   # parabolic peak interpolation
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    res.append(abs(f[i] + d * (f[1] - f[0])))
wt = np.mean(res)
term = np.sin(wt / 2) ** 2 - np.sin(kD / 2) ** 2
print("%s  kD/pi %.5f  omega tau %.6f (+w %.6f, -w %.6f)  plasma term %.6e  steps %d" % (os.path.basename(run.rstrip('/')), kD / np.pi, wt, res[1], res[0], term, N))
