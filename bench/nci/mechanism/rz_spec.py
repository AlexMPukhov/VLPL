#!/usr/bin/env python3
"""kz spectrum of the NCI in an RZ periodic streaming-plasma run (bench/nci/mechanism): openPMD thetaMode dumps every step,
FFT along z of every component, mode and radial row; per kz the energy (summed over r, modes, components), its growth rate
between s1 and s2 and the frequency (phase advance per step of the strongest coefficient), against the beam aliases.
  rz_spec.py RUN UZ N0 [s1 s2] [--png out.png]"""
import sys, os, glob, numpy as np, h5py
a, opt, args = [], {}, sys.argv[1:]
while args:
    x = args.pop(0)
    if x.startswith("--"): opt[x[2:]] = args.pop(0)
    else: a.append(x)
run, uz, n0 = a[0], float(a[1]), float(a[2])
steps, F = [], []
for fn in sorted(glob.glob(os.path.join(run, "openpmd", "data_*.h5"))):
    with h5py.File(fn, "r") as h:
        it = list(h["data"])[0]
        if int(it) == 0: continue
        g = h["data"][it]; m = g["meshes"]
        if not steps:
            dt = float(g.attrs["dt"]); sp = m["E"].attrs["gridSpacing"]; D, H = float(sp[-1]), float(sp[-2])
        arr = np.stack([np.asarray(m[q][c]) for q in ("E", "B") for c in ("r", "t", "z")])   # (6, 2M-1, nr, nz)
        steps.append(int(it)); F.append(np.fft.fft(arr, axis=-1))
steps = np.array(steps); F = np.array(F)                      # (t, 6, slots, nr, nz)
W = (np.abs(F) ** 2).sum((1, 2, 3))                          # (t, kz)
Wt = W.sum(1)
if len(a) > 4: s1, s2 = int(a[3]), int(a[4])
else:
    L = np.log(Wt); n = len(L); w = max(n // 5, 10); j0 = n // 8
    i0 = j0 + int(np.argmax([L[i + w] - L[i] for i in range(j0, n - w)])); s1, s2 = steps[i0], steps[i0 + w]
i1, i2 = int(np.searchsorted(steps, s1)), int(np.searchsorted(steps, s2))
wp = np.sqrt(n0); beta = uz / np.sqrt(1 + uz * uz); g = np.sqrt(1 + uz * uz)
nz = W.shape[1]; kz = np.fft.fftfreq(nz) * 2 * np.pi
gam = 0.5 * np.log((W[i2] + 1e-300) / (W[i1] + 1e-300)) / ((s2 - s1) * dt) / wp
flat = F.reshape(F.shape[0], -1, nz)
best = np.argmax(np.abs(flat[i2]), axis=0)
A = flat[i1:i2 + 1, best, np.arange(nz)]
wt = -np.angle((A[1:] * np.conj(A[:-1])).sum(0))
print("run %s: gamma %.2f, dz %.4g, dr %.4g (1/k0), c dt/dr %.3f, omega_p tau %.4f, window %d-%d, total x%.3g, rate %.4f omega_p"
      % (run, g, D, H, dt / H, wp * dt, s1, s2, Wt[i2] / Wt[i1], 0.5 * np.log(Wt[i2] / Wt[i1]) / ((s2 - s1) * dt) / wp))
print("  kz dz/pi   share   growth/wp   omega tau/pi   beam m=0  m=+1  m=-1 (/pi)   EM(k_perp=0) /pi")
for i in np.argsort(W[i2])[::-1][:10]:
    al = [np.angle(np.exp(1j * beta * (kz[i] + 2 * np.pi * mm))) / np.pi for mm in (0, 1, -1)]
    em = 2 * np.arcsin(np.sqrt(np.sin(kz[i] / 2) ** 2 + (wp * dt / 2) ** 2 * np.cos(kz[i] / 2) ** 2)) / np.pi
    print("   %7.3f  %6.3f   %8.4f   %8.4f     %s   %7.4f" % (kz[i] / np.pi, W[i2][i] / W[i2].sum(), gam[i], wt[i] / np.pi,
          " ".join("%7.4f" % x for x in al), em))
sh = lambda Wm: [Wm[(np.abs(kz) >= lo * np.pi) & (np.abs(kz) < hi * np.pi)].sum() / Wm.sum() for lo, hi in ((0, .25), (.25, .75), (.75, 1.01))]
print("SUMMARY share |kz dz|<.25,.25-.75,>.75 pi at s2: %s  end: %s" % (" ".join("%.2f" % x for x in sh(W[i2])), " ".join("%.2f" % x for x in sh(W[-1]))))
if "npz" in opt:
    np.savez(opt["npz"], kz=kz, W=W[i2], gam=gam, wt=wt, Wt=Wt, steps=steps, D=D, H=H, dt=dt, beta=beta, wp=wp, s1=s1, s2=s2)
if "png" in opt:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    o = np.argsort(kz)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].semilogy(kz[o] / np.pi, W[i2][o] / W[i2].max(), "k.-", ms=3); ax[0].set_ylim(1e-6, 1.5)
    ax[0].set_xlabel("kz dz / pi"); ax[0].set_title("mode energy at step %d (all r, modes, components)" % s2)
    ax[1].plot(kz[o] / np.pi, gam[o], "r.-", ms=3); ax[1].set_xlabel("kz dz / pi"); ax[1].set_title("growth rate / omega_p, steps %d-%d" % (s1, s2))
    unit = 2 * np.pi * (1 - abs(beta))
    fl = np.where(kz < 0, -1, 1); sel = W[i2] > 1e-3 * W[i2].max()
    off = np.angle(np.exp(1j * (fl * wt - beta * fl * kz))) / unit
    sc = ax[2].scatter(fl[sel] * kz[sel] / np.pi, off[sel], c=np.log10(W[i2][sel] / W[i2].max()), cmap="viridis", vmin=-3, vmax=0)
    for mm, c in ((0, "k"), (1, "m"), (-1, "c")): ax[2].axhline(mm, color=c, ls="--", lw=0.9, label="beam alias m=%d" % mm)
    fk = np.linspace(0.02, np.pi, 500)
    for s in (1, -1):
        e = 2 * np.arcsin(np.sqrt(np.sin(fk / 2) ** 2 + (wp * dt / 2) ** 2 * np.cos(fk / 2) ** 2))
        ax[2].plot(fk / np.pi, np.angle(np.exp(1j * (s * e - beta * fk))) / unit, "k-" if s > 0 else "k:", lw=0.6)
    fig.colorbar(sc, ax=ax[2]); ax[2].set_ylim(-2.5, 2.5); ax[2].set_xlim(0, 1); ax[2].legend(fontsize=7)
    ax[2].set_xlabel("|kz| dz / pi"); ax[2].set_ylabel("(omega - beta c kz) tau / (2 pi (1 - |beta|))")
    ax[2].set_title("modes > 1e-3 of peak; black: EM at k_perp = 0 (+w solid, -w dotted)", fontsize=8)
    fig.suptitle(os.path.basename(run.rstrip("/")) + "  RZ rip_cc gamma %.1f, c dt/dr %.2f" % (g, dt / H), fontsize=9)
    fig.tight_layout(); fig.savefig(opt["png"], dpi=110)
