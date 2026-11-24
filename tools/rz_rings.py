#!/usr/bin/env python3
"""A bunch through embedded metal (block rings, docs/specs/rz_metal.md): wake, energy loss and the field at a lab plane.

  python3 tools/rz_rings.py <run dir> <vacuum run dir> --gamma G --mass M_me --zb Z0_mm --sigma S_um
                            [--xi A_mm B_mm] [--step N (axis record; default the last)] [--plot out.png]

The structure field = run - vacuum run (the same bunch without metal; linear fields). Prints
- the bunch charge Q (from W_kin at the start, mass M in electron masses, Lorentz factor G) and its energy loss (W_kin of
  diag.txt, run - vacuum), and the loss rate dW/d(ct) over the second half of the run;
- the wake on the axis (dir/plane/axis_r*.bin, the last record of both runs): a least-squares fit a cos(k xi) + b sin(k xi)
  in A < -xi < B mm behind the bunch centre z_b = Z0 + beta c t; its amplitude E1 and the loss factor kappa = E1 / (2 Q F),
  F = exp(-k^2 sigma^2 / 2) (Gaussian bunch);
- at the plane (dir/plane/plane_r*.bin): the Poynting energy int dt int 2 pi r dr E_r B_phi / mu0 of the structure field,
  and per frequency the far field from the vector Hankel transform E~(k_perp) = 2 pi int E_r J1(k_perp r) r dr with
  d2W / dOmega domega = 2 k^2 cos^2(theta) |E~|^2 / (4 pi^2 Z0 2 pi) (omega > 0, the formulas of the GEM-PIC-RZ analysis
  exit.py): peak angle, FWHM, 90 % cone, spectrum.
Units: SI from lambda0 in dir/plane/meta.txt (fields m c omega0 / e, densities n_c, lengths 1/k0, energies m c^2 n_c / k0^3).
"""
import glob
import sys

import numpy as np
from scipy import special as sp

C, ME, QE, EPS0 = 299792458.0, 9.1093837015e-31, 1.602176634e-19, 8.8541878128e-12
MU0 = 1.25663706212e-6
Z0 = MU0 * C


def meta(d):
    m = {}
    for line in open(d + "/plane/meta.txt"):
        w = line.split()
        m[w[0]] = w[1:]
    lam = float(m["lambda0"][0])
    k0 = 2 * np.pi / lam
    w0 = k0 * C
    u = dict(l=1 / k0, E=ME * C * w0 / QE, B=ME * w0 / QE, n=EPS0 * ME * w0**2 / QE**2)
    u["W"] = ME * C**2 * u["n"] * u["l"]**3
    u["Q"] = QE * u["n"] * u["l"]**3
    return dict(nr=int(m["nr"][0]), S=int(m["slots"][0]), dz=float(m["dz"][0]), dt=float(m["dt"][0]),
                cell=int(m["plane_cell"][0]), r=np.array(m["r"], float), rh=np.array(m["rh"], float), u=u)


def plane(d, M):
    n = 2 + 6 * M["S"] * M["nr"]
    a = np.concatenate([np.fromfile(f, dtype=np.float64).reshape(-1, n) for f in glob.glob(d + "/plane/plane_r*.bin")])
    a = a[np.argsort(a[:, 0])]
    return a[:, 0].astype(int), a[:, 2:].reshape(-1, 6, M["S"], M["nr"])


def axis(d):
    recs = {}
    for f in glob.glob(d + "/plane/axis_r*.bin"):
        a = np.fromfile(f, dtype=np.float64)
        p = 0
        while p < len(a):
            st, _, c0, n = a[p:p + 4]
            n = int(n)
            recs.setdefault(int(st), []).append((int(c0), a[p + 4:p + 4 + n]))
            p += 4 + n
    return {st: (min(c for c, _ in v), np.concatenate([x for _, x in sorted(v, key=lambda q: q[0])])) for st, v in recs.items()}


def main():
    args = sys.argv[1:]
    opt = {}
    for key, nv in (("--gamma", 1), ("--mass", 1), ("--zb", 1), ("--sigma", 1), ("--xi", 2), ("--plot", 1), ("--step", 1)):
        if key in args:
            i = args.index(key)
            opt[key] = args[i + 1:i + 1 + nv]
            del args[i:i + 1 + nv]
    run, vac = args[0], args[1]
    G, mass = float(opt["--gamma"][0]), float(opt["--mass"][0])
    beta = np.sqrt(1 - 1 / G**2)
    zb0, sig = float(opt["--zb"][0]) * 1e-3, float(opt["--sigma"][0]) * 1e-6
    xa, xb = (float(x) * 1e-3 for x in opt.get("--xi", ["1", "3"]))
    M = meta(run)
    u = M["u"]
    dzm = M["dz"] * u["l"]
    d1, d0 = np.loadtxt(run + "/diag.txt"), np.loadtxt(vac + "/diag.txt")
    nn = min(len(d1), len(d0))
    Q = d1[0, 3] / (mass * (G - 1)) * u["Q"]
    dW = ((d1[:nn, 3] - d1[0, 3]) - (d0[:nn, 3] - d0[0, 3])) * u["W"]
    ct = d1[:nn, 1] * u["l"]
    print("bunch charge Q = %.5g nC, gamma = %g" % (Q * 1e9, G))
    print("bunch energy loss by the structure (run - vacuum): %.5g mJ over c t = %.4g m (vacuum run alone: %.3g mJ)" %
          (-dW[-1] * 1e3, ct[-1], -(d0[nn - 1, 3] - d0[0, 3]) * u["W"] * 1e3))
    h = nn // 2
    rate = -np.polyfit(ct[h:], dW[h:], 1)[0]
    print("loss rate (second half) %.5g J/m = %.5g kV/pC/m x Q^2" % (rate, rate / (Q * 1e12)**2 * 1e9))
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        i0, i1 = int(frac * (nn - 1)) - 1, int(frac * (nn - 1)) + 1
        if i0 >= 0:
            print("    at c t = %.3f m: dW/d(ct) = %.4g kV/pC/m x Q^2" %
                  (ct[i0 + 1], -(dW[i1] - dW[i0]) / (ct[i1] - ct[i0]) / (Q * 1e12)**2 * 1e9))
    # wake on the axis
    A1, A0 = axis(run), axis(vac)
    st = int(opt["--step"][0]) if "--step" in opt else max(s for s in A1 if s in A0)
    (c1, e1), (c0, e0) = A1[st], A0[st]
    assert c1 == c0
    ez = (e1 - e0) * u["E"]
    z = (c1 + np.arange(len(ez))) * dzm
    zb = zb0 + beta * st * M["dt"] * u["l"]
    xi = z - zb
    s = (xi < -xa) & (xi > -xb)
    best = None
    for k in np.linspace(3e3, 20e3, 3401):
        A = np.stack([np.cos(k * xi[s]), np.sin(k * xi[s])], 1)
        c, res, *_ = np.linalg.lstsq(A, ez[s], rcond=None)
        r = np.sum((A @ c - ez[s])**2)
        if best is None or r < best[0]:
            best = (r, k, c)
    from scipy.optimize import minimize_scalar
    def rr(k):
        A = np.stack([np.cos(k * xi[s]), np.sin(k * xi[s])], 1)
        c = np.linalg.lstsq(A, ez[s], rcond=None)[0]
        return np.sum((A @ c - ez[s])**2)
    k = minimize_scalar(rr, bracket=(best[1] - 5, best[1], best[1] + 5)).x
    A = np.stack([np.cos(k * xi[s]), np.sin(k * xi[s])], 1)
    c = np.linalg.lstsq(A, ez[s], rcond=None)[0]
    E1 = np.hypot(*c)
    F = np.exp(-0.5 * (k * sig)**2)
    fitres = np.sqrt(rr(k) / np.sum(ez[s]**2))
    print("axis wake at step %d (bunch centre z = %.3f mm), %.1f-%.1f mm behind: k = %.4f /mm (lambda %.1f um), "
          "amplitude %.5g MV/m, kappa = E1 / (2 Q F) = %.4g kV/pC/m (F = %.4f), relative fit residual %.2e" %
          (st, zb * 1e3, xa * 1e3, xb * 1e3, k * 1e-3, 2 * np.pi / k * 1e6, E1 * 1e-6, E1 / (2 * Q * F) * 1e-15, F, fitres))
    # amplitude envelope along xi (the wake tail)
    for a in np.arange(0.0, 6.0, 0.5):
        q = (xi < -a * 1e-3) & (xi > -(a + 0.5) * 1e-3)
        if q.sum() > 10:
            print("    xi = -%.1f .. -%.1f mm: max |Ez| = %.4g MV/m" % (a, a + 0.5, np.abs(ez[q]).max() * 1e-6))
    # plane
    s1, F1 = plane(run, M)
    s0, F0 = plane(vac, M)
    com = np.intersect1d(s1, s0)
    F1 = F1[np.isin(s1, com)]
    F0 = F0[np.isin(s0, com)]
    print("plane z = %.3f mm: %d records (steps %d .. %d)" % (M["cell"] * dzm * 1e3, len(com), com[0], com[-1]))
    Er = (F1[:, 1, 0, :] - F0[:, 1, 0, :]) * u["E"]           # at the half nodes rh_j
    Bp = (F1[:, 5, 0, :] - F0[:, 5, 0, :]) * u["B"]
    rh = M["rh"] * u["l"]
    rn = np.append(M["r"], 2 * M["rh"][-1] - M["r"][-1]) * u["l"]
    area = np.pi * (rn[1:]**2 - rn[:-1]**2)                    # the ring r_j .. r_{j+1} around rh_j
    dt = M["dt"] * u["l"] / C
    S_t = np.sum(Er * Bp * area[None, :], axis=1) / MU0
    W_plane = S_t.sum() * dt
    S_vac = np.sum(F0[:, 1, 0, :] * F0[:, 5, 0, :] * u["E"] * u["B"] * area[None, :], axis=1).sum() / MU0 * dt
    S_run = np.sum(F1[:, 1, 0, :] * F1[:, 5, 0, :] * u["E"] * u["B"] * area[None, :], axis=1).sum() / MU0 * dt
    print("Poynting energy through the plane: structure field %.5g mJ (run %.5g, vacuum %.5g mJ)" %
          (W_plane * 1e3, S_run * 1e3, S_vac * 1e3))
    nt = len(com)
    nfft = 1 << int(np.ceil(np.log2(4 * nt)))
    Ew = np.fft.rfft(Er, n=nfft, axis=0) * dt
    om = 2 * np.pi * np.fft.rfftfreq(nfft, dt)
    th = np.linspace(0.0, 0.5 * np.pi, 721)[1:]
    sel = (om > 0) & (om < 2 * np.pi * 3e12)
    ww = rh * area / (2 * np.pi * rh)                             # r dr
    d2 = np.zeros((sel.sum(), len(th)))
    for j, (o, ew) in enumerate(zip(om[sel], Ew[sel])):
        kk = o / C
        Et = 2 * np.pi * (sp.j1(np.outer(kk * np.sin(th), rh)) @ (ew * ww))
        d2[j] = 2.0 * kk * kk * np.cos(th)**2 * np.abs(Et)**2 / (4 * np.pi**2 * Z0) / (2 * np.pi)
    omp = om[sel]
    dWdO = np.trapezoid(d2, omp, axis=0)
    dWdw = np.trapezoid(d2 * 2 * np.pi * np.sin(th)[None, :], th, axis=1)
    W_far = np.trapezoid(dWdw, omp)
    ip = np.argmax(dWdO)
    half = th[dWdO >= 0.5 * dWdO[ip]]
    jp = np.argmax(dWdw)
    fh = omp[dWdw >= 0.5 * dWdw[jp]] / (2 * np.pi)
    cum = np.cumsum(dWdO * 2 * np.pi * np.sin(th)) * (th[1] - th[0])
    th90 = th[np.searchsorted(cum, 0.9 * cum[-1])]
    print("far field: W = %.5g mJ; dW/dOmega peak %.4g mJ/sr at %.2f mrad, FWHM %.2f - %.2f mrad; 90 %% within %.1f mrad" %
          (W_far * 1e3, dWdO[ip] * 1e3, th[ip] * 1e3, half.min() * 1e3, half.max() * 1e3, th90 * 1e3))
    print("spectrum: peak %.1f GHz, FWHM %.1f - %.1f GHz (%.1f %%)" % (omp[jp] / (2 * np.pi) * 1e-9, fh.min() * 1e-9,
                                                                       fh.max() * 1e-9, 100 * (fh.max() - fh.min()) / (omp[jp] / (2 * np.pi))))
    if "--plot" in opt:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        ax[0].plot(xi * 1e3, ez * 1e-6)
        ax[0].set_xlabel("xi [mm]"); ax[0].set_ylabel("E_z on the axis [MV/m]")
        ax[1].plot(th * 1e3, dWdO * 1e3)
        ax[1].set_xlabel("theta [mrad]"); ax[1].set_ylabel("dW/dOmega [mJ/sr]"); ax[1].set_xlim(0, 4 * th90 * 1e3)
        ax[2].plot(omp / (2 * np.pi) * 1e-9, dWdw * 2 * np.pi * 1e9 * 1e3)
        ax[2].set_xlabel("f [GHz]"); ax[2].set_ylabel("dW/df [mJ/GHz]"); ax[2].set_xlim(0, 2 * omp[jp] / (2 * np.pi) * 1e-9)
        t = com * dt
        vm = np.abs(Er).max() * 1e-6
        ax[3].pcolormesh(t * 1e12, rh * 1e3, Er.T * 1e-6, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto", rasterized=True)
        ax[3].set_xlabel("t [ps]"); ax[3].set_ylabel("r [mm]"); ax[3].set_title("E_r at the plane [MV/m]")
        fig.tight_layout()
        fig.savefig(opt["--plot"][0], dpi=110)


if __name__ == "__main__":
    main()
