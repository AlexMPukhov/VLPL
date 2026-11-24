#!/usr/bin/env python3
"""Handover test 2: the on-axis wake of dwa.deck against the modal theory of a dielectric-lined metal tube (the formulas of
the public GEM-PIC-RZ tests/dielectric.py, Bessel functions from scipy).

Vacuum channel r < a, eps in a < r < b, conducting wall at b; a bunch at v = c excites the synchronous TM_0n modes:
  E_z = E0 in the channel, C F(r) in the dielectric, F = J0(q r) Y0(q b) - Y0(q r) J0(q b), q = k sqrt(eps - 1),
  matched at r = a:  eps F'(a) / (q^2 F(a)) = -a / 2.
Per mode the stored energy u per length, the power P, beta_g = P / (u c), the loss factor kappa = E0^2 / (4 u (1 - beta_g)).
The wake of a Gaussian bunch (charge Q, rms length sigma) behind it (in the filled region):
  E_z(xi) = -Q sum_n 2 kappa_n exp(-k_n^2 sigma^2 / 2) cos(k_n (xi - xi_b)).
usage: python3 dwa.py <run dir> <eps> [a_um b_um sigma_um gamma mass_me zb_mm]   (defaults: dwa.deck)
"""
import os
import sys

import numpy as np
from scipy import special as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import rz_rings as R  # noqa: E402

EPS0, C = 8.8541878128e-12, 299792458.0


def F(q, r, b):
    return sp.j0(q * r) * sp.y0(q * b) - sp.y0(q * r) * sp.j0(q * b)


def dF(q, r, b):
    return -q * (sp.j1(q * r) * sp.y0(q * b) - sp.y1(q * r) * sp.j0(q * b))


def disp(k, a, b, eps):
    q = k * np.sqrt(eps - 1.0)
    return eps * dF(q, a, b) + 0.5 * a * q * q * F(q, a, b)


def modes(a, b, eps, n=3):
    """the first n synchronous TM_0n modes: (k, beta_g, kappa [V/C/m]), lengths in m"""
    from scipy.optimize import brentq
    ks, k, dk = [], 0.01 / (b - a), 0.01 / (b - a)
    f0 = disp(k, a, b, eps)
    while len(ks) < n:
        f1 = disp(k + dk, a, b, eps)
        if f0 * f1 < 0 and abs(F((k + dk) * np.sqrt(eps - 1), a, b)) > 1e-9 and abs(F(k * np.sqrt(eps - 1), a, b)) > 1e-9:
            ks.append(brentq(disp, k, k + dk, args=(a, b, eps), xtol=1e-14))
        k, f0 = k + dk, f1
    out = []
    for k in ks:
        q = k * np.sqrt(eps - 1.0)
        Cc = 1.0 / F(q, a, b)
        rv = np.linspace(0.0, a, 4001)
        Ezv, Erv = np.ones_like(rv), k * rv / 2
        rd = np.linspace(a, b, 8001)
        Ezd, dFd = Cc * F(q, rd, b), Cc * dF(q, rd, b)
        Erd, Bd = k / q**2 * dFd, k * eps / q**2 * dFd
        u = EPS0 / 4 * (np.trapezoid((Ezv**2 + 2 * Erv**2) * 2 * np.pi * rv, rv) +
                        np.trapezoid((eps * (Ezd**2 + Erd**2) + Bd**2) * 2 * np.pi * rd, rd))
        P = EPS0 * C / 2 * (np.trapezoid(Erv**2 * 2 * np.pi * rv, rv) + np.trapezoid(Erd * Bd * 2 * np.pi * rd, rd))
        bg = P / (u * C)
        out.append((k, bg, 1.0 / (4 * u * (1 - bg))))
    return out


def fit_modes(xi, f, kv, iters=60):
    """least squares f = sum_j (c_j cos k_j xi + s_j sin k_j xi) with free k_j (Gauss-Newton on k)"""
    kv = np.array(kv, float)
    def lin(k):
        A = np.column_stack([fn(kk * xi) for kk in k for fn in (np.cos, np.sin)])
        c = np.linalg.lstsq(A, f, rcond=None)[0]
        return c, A @ c - f
    for _ in range(iters):
        c, r0 = lin(kv)
        J = []
        for j in range(len(kv)):
            dk = 1e-6 * kv[j]
            kk = kv.copy(); kk[j] += dk
            J.append((lin(kk)[1] - r0) / dk)
        J = np.column_stack(J)
        step = np.linalg.lstsq(J, -r0, rcond=None)[0]
        kv = kv + step
        if np.max(np.abs(step / kv)) < 1e-12:
            break
    c, r0 = lin(kv)
    return kv, c, np.sqrt(np.sum(r0**2) / np.sum(f**2))


def main():
    run, eps = sys.argv[1], float(sys.argv[2])
    p = [50, 75, 10, 1000, 1836.15267343 * 1e4, 0.85]          # the defaults of dwa.deck
    for i, x in enumerate(sys.argv[3:9]):
        p[i] = float(x)
    a, b, sig, G, mass, zb0 = p
    a, b, sig, zb0 = a * 1e-6, b * 1e-6, sig * 1e-6, zb0 * 1e-3
    th = modes(a, b, eps, 3)
    M = R.meta(run)
    u = M["u"]
    dzm = M["dz"] * u["l"]
    d = np.loadtxt(run + "/diag.txt")
    Q = d[0, 3] / (mass * (G - 1)) * u["Q"]
    A = R.axis(run)
    st = max(A)
    c0, e = A[st]
    ez = e * u["E"]
    z = (c0 + np.arange(len(ez))) * dzm
    zb = zb0 + np.sqrt(1 - 1 / G**2) * st * M["dt"] * u["l"]
    xi = z - zb
    s_in = zb - 0.95e-3                                  # distance inside the lined section
    print("bunch Q = %.4g pC, gamma %g, sigma %.1f um; step %d: bunch %.3f mm inside the lining" % (Q * 1e12, G, sig * 1e6, st, s_in * 1e3))
    for n, (k, bg, kap) in enumerate(th):
        print("  theory TM0%d: k = %.5f /mm (lambda %.2f um), beta_g = %.4f, kappa = %.4f kV/pC/m, filled to %.2f mm behind, "
              "amplitude 2 kappa Q F = %.4f MV/m" % (n + 1, k * 1e-3, 2 * np.pi / k * 1e6, bg, kap * 1e-15,
                                                     s_in * (1 / bg - 1) * 1e3, 2 * kap * Q * np.exp(-0.5 * (k * sig)**2) * 1e-6))
    # fit region: behind the bunch (5 sigma) to the window end or the filled length of the slowest mode in it
    lo = max(xi[0] + 5 * dzm, -0.95 * min(s_in * (1 / bg - 1) for _, bg, _ in th[:2]))
    sel = (xi < -5 * sig) & (xi > lo)
    nm = sum(1 for k, _, _ in th if k * sig < 2.5)       # the modes the bunch excites (form factor > 0.04)
    kv, c, res = fit_modes(xi[sel], ez[sel], [t[0] for t in th[:nm]])
    ok = True
    for n in range(nm):
        k, bg, kap = th[n]
        amp = np.hypot(c[2 * n], c[2 * n + 1])
        ath = 2 * kap * Q * np.exp(-0.5 * (k * sig)**2)
        ek, ea = kv[n] / k - 1, amp / ath - 1
        strong = ath > 0.1 * 2 * th[0][2] * Q * np.exp(-0.5 * (th[0][0] * sig)**2)   # modes above 10 % of TM01: 2e-3, 2 %; weaker 5e-3, 5 %
        ok = ok and abs(ek) < (2e-3 if strong else 5e-3) and abs(ea) < (0.02 if strong else 0.05)
        print("  fit TM0%d over xi = %.3f .. %.3f mm: k = %.5f /mm (k / k_theory - 1 = %+.2e), amplitude %.4f MV/m "
              "(/ theory - 1 = %+.2e), kappa = %.4f kV/pC/m" % (n + 1, lo * 1e3, -5 * sig * 1e3, kv[n] * 1e-3, ek, amp * 1e-6, ea,
                                                                amp / (2 * Q * np.exp(-0.5 * (kv[n] * sig)**2)) * 1e-15))
    print("  relative fit residual %.2e;  %s" % (res, "PASS (k to 2e-3, amplitude to 2 % for modes above 10 % of TM01; weaker 5e-3, 5 %)" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
