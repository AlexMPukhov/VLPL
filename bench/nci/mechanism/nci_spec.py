#!/usr/bin/env python3
"""Spectrum of the numerical Cherenkov instability in a 2D periodic streaming-plasma run (bench/nci/mechanism/README.md).
Reads the openPMD field dumps (every step) of one run, Fourier transforms E, B over (x, y), measures per mode the amplitude
growth rate between steps s1 and s2 and the frequency (phase advance per step), and overlays the resonance curves of the
scheme's vacuum dispersion with the beam modes and their aliases.
  nci_spec.py RUN_DIR SOLVER(irip|rip_cc) UX [s1 s2] [--png out.png] [--n0 0.004]
"""
import sys, os, glob, numpy as np, h5py

def load(run):
    files = sorted(glob.glob(os.path.join(run, "openpmd", "data_*.h5")))
    steps, F = [], []; D_set = False
    for fn in files:
        with h5py.File(fn, "r") as h:
            it = list(h["data"])[0]; g = h["data"][it]
            m = g["meshes"]
            comps = [np.asarray(m[q][c])[0] for q in ("E", "B") for c in ("x", "y", "z")]
            if not D_set:
                sp = m["E"].attrs["gridSpacing"]; D, H = float(sp[-1]), float(sp[-2]); dt = float(g.attrs["dt"]); D_set = True
            if int(it) == 0: continue
            steps.append(int(it)); F.append(np.fft.fft2(np.stack(comps)))      # axes (y, x)
    return np.array(steps), np.array(F), D, H, dt

def em_branch(solver, kxD, Lp):
    """omega tau >= 0 of the vacuum (plus omega_p) mode of the scheme; Lp = D^2 Lambda / 4 (+ (omega_p tau / 2)^2)."""
    s2, c2 = np.sin(kxD / 2) ** 2, np.cos(kxD / 2) ** 2
    if solver == "rip_cc":
        return 2 * np.arcsin(np.sqrt(np.clip(s2 + Lp * c2, 0, 1)))
    return 2 * np.arctan(np.sqrt(np.tan(kxD / 2) ** 2 + Lp))

def main():
    a, opt, args = [], {}, sys.argv[1:]
    while args:
        x = args.pop(0)
        if x.startswith("--"): opt[x[2:]] = args.pop(0)
        else: a.append(x)
    run, solver, ux = a[0], a[1], float(a[2])
    n0 = float(opt.get("n0", "0.004"))
    steps, F, D, H, dt = load(run)
    W = (np.abs(F) ** 2).sum(1)                          # mode energy (arbitrary units), (t, ky, kx)
    Wt = W.sum((1, 2))
    if len(a) > 4:
        s1, s2 = int(a[3]), int(a[4])
    else:                                                # the window of fastest total growth, 1/4 of the run before saturation
        L = np.log(Wt); n = len(L); w = max(n // 5, 10); j0 = n // 8   # skip the start (the noise level builds up)
        r = [(L[i + w] - L[i]) for i in range(j0, n - w)]
        i0 = j0 + int(np.argmax(r)); s1, s2 = steps[i0], steps[i0 + w]
    i1, i2 = int(np.searchsorted(steps, s1)), int(np.searchsorted(steps, s2))
    gam = 0.5 * (np.log(W[i2] + 1e-300) - np.log(W[i1] + 1e-300)) / ((s2 - s1) * dt)   # amplitude growth / (1/omega0)
    wp = np.sqrt(n0); beta = ux / np.sqrt(1 + ux * ux); g = np.sqrt(1 + ux * ux)
    ny, nx = W.shape[1:]
    kx = np.fft.fftfreq(nx) * 2 * np.pi                  # kx D
    ky = np.fft.fftfreq(ny) * 2 * np.pi                  # ky H
    # frequency per mode: phase advance per step of the strongest component over [s1, s2]
    comp = np.argmax(np.abs(F[i2]), axis=0)
    A = np.take_along_axis(F[i1:i2 + 1], comp[None, None], 1)[:, 0]
    ratio = (A[1:] * np.conj(A[:-1])).sum(0)
    dsteps = np.diff(steps[i1:i2 + 1]).mean()
    wt = -np.angle(ratio) / dsteps                       # omega tau, signed: mode exp(i(k x - omega t))
    print("run %s: solver %s, gamma %.2f, D = %.4g, h = %.4g (1/k0), c dt / h = %.3f, omega_p tau = %.4f, window steps %d-%d" %
          (run, solver, g, D, H, dt / H, wp * dt, s1, s2))
    print("total field energy x%.3g over the window, rate %.4f omega_p (amplitude)" % (Wt[i2] / Wt[i1], 0.5 * np.log(Wt[i2] / Wt[i1]) / ((s2 - s1) * dt) / wp))
    order = np.argsort(W[i2].ravel())[::-1]
    print("  strongest modes at step %d:  kx D/pi  ky h/pi   share   growth/omega_p   omega tau/pi   EM+ /pi   beam m=0, +-1 /pi" % s2)
    seen = 0
    for q in order[:12]:
        j, i = np.unravel_index(q, W[i2].shape)
        Lp = (D / H) ** 2 * np.sin(ky[j] / 2) ** 2 + (wp * dt / 2) ** 2
        em = em_branch(solver, kx[i], Lp)
        bm = [beta * (kx[i] + 2 * np.pi * m) for m in (0, 1, -1)]
        bm = [np.angle(np.exp(1j * b)) for b in bm]
        print("    %7.3f  %7.3f  %6.3f   %8.4f   %8.4f   %7.4f   %s" % (kx[i] / np.pi, ky[j] / np.pi, W[i2].ravel()[q] / W[i2].sum(),
              gam[j, i] / wp, wt[j, i] / np.pi, em / np.pi, " ".join("%7.4f" % (b / np.pi) for b in bm)))
    # where the energy sits at the end of the window and at the end of the run; the dominant mode and its nearest beam alias
    KXa = np.abs(kx)[None, :] * np.ones((ny, 1))
    def share(Wm):
        return [Wm[(KXa >= lo * np.pi) & (KXa < hi * np.pi)].sum() / Wm.sum() for lo, hi in ((0, .25), (.25, .75), (.75, 1.01))]
    j, i = np.unravel_index(int(np.argmax(W[i2])), W[i2].shape)
    Lp = (D / H) ** 2 * np.sin(ky[j] / 2) ** 2 + (wp * dt / 2) ** 2
    em = em_branch(solver, kx[i], Lp) * np.sign(wt[j, i])
    al = {m: np.angle(np.exp(1j * beta * (kx[i] + 2 * np.pi * m))) for m in (-2, -1, 0, 1, 2)}
    mbest = min(al, key=lambda m: abs(np.angle(np.exp(1j * (al[m] - wt[j, i])))))
    rt = 0.5 * np.log(Wt[i2] / Wt[i1]) / ((s2 - s1) * dt) / wp
    print("SUMMARY rate %.4f  dom kxD/pi %.3f kyh/pi %.3f  wtau/pi %.4f  EM %.4f  alias m=%d %.4f  dom-rate %.4f  share |kxD|<.25,.25-.75,>.75 pi: %s  end: %s  W_end/W_s1 %.3g"
          % (rt, kx[i] / np.pi, ky[j] / np.pi, wt[j, i] / np.pi, em / np.pi, mbest, al[mbest] / np.pi, gam[j, i] / wp,
             " ".join("%.2f" % x for x in share(W[i2])), " ".join("%.2f" % x for x in share(W[-1])), Wt[-1] / Wt[i1]))
    if "npz" in opt:                                     # the data of the figures (plotted separately)
        np.savez(opt["npz"], kx=kx, ky=ky, W=W[i2], gam=gam / wp, wt=wt, Wt=Wt, steps=steps, D=D, H=H, dt=dt, beta=beta, wp=wp,
                 s1=s1, s2=s2, solver=solver)
    if "png" in opt:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        KX, KY = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky))
        G = np.fft.fftshift(gam) / wp; S = np.fft.fftshift(np.log10(W[i2] / W[i2].max() + 1e-30))
        fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
        # omega - k diagram: the modes within 1e-2 of the peak (colour: |ky h|), EM branches for a few ky, beam aliases
        sel = W[i2] > 1e-2 * W[i2].max()
        JJ, II = np.nonzero(sel)
        # frequency offset from the beam line, in units of the alias spacing 2 pi (1 - |beta|): beam alias m sits at m;
        # modes folded to kx >= 0 ((k, w) -> (-k, -w) for a real field); the beam streams along -x (beta < 0)
        # k along the beam: kb = kx sign(beta) (the beam moves at +|beta| c in kb); folded to kb >= 0 by (k, w) -> (-k, -w)
        bb = abs(beta); unit = 2 * np.pi * (1 - bb)
        def off(w, k): return np.angle(np.exp(1j * (w - bb * k))) / unit
        fk = np.linspace(0.02, np.pi, 1000)
        for kyh in (0.0, 0.125 * np.pi, 0.25 * np.pi, 0.5 * np.pi, np.pi):
            e = em_branch(solver, fk, (D / H) ** 2 * np.sin(kyh / 2) ** 2 + (wp * dt / 2) ** 2)
            y = off(e, fk); y[np.abs(np.diff(y, append=y[-1])) > 1] = np.nan
            ax[3].plot(fk / np.pi, y, "k-", lw=0.6)
        for m, c in ((0, "k"), (1, "m"), (-1, "c"), (2, "y"), (-2, "g")):
            ax[3].axhline(np.angle(np.exp(1j * 2 * np.pi * m * bb)) / unit, color=c, ls="--", lw=0.9, label="beam alias m=%d" % m)
        kb = np.sign(beta) * kx[II]; wb = wt[JJ, II].copy()
        fl = np.where(kb < 0, -1, 1)
        sc = ax[3].scatter(fl * kb / np.pi, off(fl * wb, fl * kb), c=np.abs(ky[JJ]) / np.pi,
                           s=10 + 60 * W[i2][sel] / W[i2].max(), cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(sc, ax=ax[3]); ax[3].set_xlabel("|k| D / pi along the beam"); ax[3].set_ylabel("(omega - |beta| c k) tau / (2 pi (1 - |beta|))")
        ax[3].set_title("modes > 1e-2 of peak (colour |ky h|/pi); black: EM branch, ky h = 0, pi/8, pi/4, pi/2, pi", fontsize=7)
        ax[3].legend(fontsize=6, loc="lower left"); ax[3].set_xlim(0, 1); ax[3].set_ylim(-2.5, 2.5)
        ax = ax[:3]
        im = ax[0].pcolormesh(KX / np.pi, KY / np.pi, S, vmin=-6, vmax=0, shading="nearest"); fig.colorbar(im, ax=ax[0])
        ax[0].set_title("log10 mode energy, step %d" % s2)
        v = np.nanmax(np.abs(G)); im = ax[1].pcolormesh(KX / np.pi, KY / np.pi, G, cmap="RdBu_r", vmin=-v, vmax=v, shading="nearest")
        fig.colorbar(im, ax=ax[1]); ax[1].set_title("growth rate / omega_p, steps %d-%d" % (s1, s2))
        Wm = np.fft.fftshift(wt) / np.pi
        im = ax[2].pcolormesh(KX / np.pi, KY / np.pi, np.where(S > -3, Wm, np.nan), cmap="twilight", vmin=-1, vmax=1, shading="nearest")
        fig.colorbar(im, ax=ax[2]); ax[2].set_title("omega tau / pi (modes within 1e-3 of the peak)")
        # resonance curves on a fine grid
        fx, fy = np.meshgrid(np.linspace(-np.pi, np.pi, 801), np.linspace(-np.pi, np.pi, 801))
        Lp = (D / H) ** 2 * np.sin(fy / 2) ** 2 + (wp * dt / 2) ** 2
        em = em_branch(solver, fx, Lp)
        cols = {0: "k", 1: "m", -1: "c", 2: "y", -2: "g"}
        for axx in ax:
            for m in (0, 1, -1, 2, -2):
                b = beta * (fx + 2 * np.pi * m)
                for s in (1, -1):
                    d = s * em - b
                    f = np.where(np.cos(d) > 0, np.sin(d), np.nan)
                    axx.contour(fx / np.pi, fy / np.pi, f, levels=[0], colors=cols[m], linewidths=0.8,
                                linestyles="-" if s > 0 else "--")
            axx.set_xlabel("kx D / pi"); axx.set_ylabel("ky h / pi")
        fig.suptitle("%s  %s gamma %.1f  c dt/h %.2f  (curves: EM branch = beam alias m; k 0, magenta +1, cyan -1, y +2, g -2; solid +omega, dashed -omega)"
                     % (os.path.basename(run.rstrip("/")), solver, g, dt / H, ), fontsize=9)
        fig.tight_layout(); fig.savefig(opt["png"], dpi=110)

if __name__ == "__main__":
    main()
