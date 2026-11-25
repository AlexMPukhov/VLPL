#!/usr/bin/env python3
"""TM_0n modes of a metal tube (radius R) filled with a graded dielectric eps(r): the continuum reference of test D8
(tests/test_rz_dielectric.cpp, eps from a deck function). Pure python (no numpy / scipy).

Fields ~ exp(i (k_z z - w t)), c = 1, m = 0 TM (E_z, E_r, B_phi). With u = r B_phi and v = u' / (eps r):
    u' = eps r v,    v' = -(w^2 - k_z^2 / eps) u / r,
regular at the axis (u ~ r^2, v -> 2 / eps(0)), E_z(R) = 0 <=> v(R) = 0. Classical RK4 with n and 2n steps (Richardson,
error ~ h^4), bisection on w.

  python3 grin_tm01.py            # the numbers of test D8: eps = 1 + 1.5 (1 - (r/R)^2), R = 1.6, k_z = 2 pi lz / 2.56
"""
import math


def v_end(w, kz, R, eps, n):
    r0 = 1e-9 * R
    h = (R - r0) / n
    u, v, r = r0 * r0, 2.0 / eps(r0), r0

    def f(r, u, v):
        e = eps(r)
        return e * r * v, -(w * w - kz * kz / e) * u / r

    for _ in range(n):
        a1, b1 = f(r, u, v)
        a2, b2 = f(r + h / 2, u + h / 2 * a1, v + h / 2 * b1)
        a3, b3 = f(r + h / 2, u + h / 2 * a2, v + h / 2 * b2)
        a4, b4 = f(r + h, u + h * a3, v + h * b3)
        u += h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        v += h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
        r += h
    return v / u                    # scale-free (u > 0 for the lowest modes near R)


def root(kz, R, eps, lo, hi, n):
    g = lambda w: v_end(w, kz, R, eps, n)
    flo = g(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = g(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tm01(kz, R, eps, n=4000):
    # bracket: scan for the first sign change of v(R) / u(R)
    ws = [0.02 * q for q in range(1, 400)]
    prev = v_end(ws[0], kz, R, eps, 400)
    for q in range(1, len(ws)):
        cur = v_end(ws[q], kz, R, eps, 400)
        if (cur > 0) != (prev > 0) and abs(cur - prev) < 1e3:
            a = root(kz, R, eps, ws[q - 1], ws[q], n)
            b = root(kz, R, eps, ws[q - 1], ws[q], 2 * n)
            return b + (b - a) / 15         # Richardson for RK4
        prev = cur
    raise RuntimeError("no root")


if __name__ == "__main__":
    R = 1.6
    eps = lambda r: 1 + 1.5 * (1 - (r / R) ** 2)
    for lz in (1, 4):
        kz = 2 * math.pi * lz / 2.56
        print("k_z %.10f: w(TM01) = %.12f" % (kz, tm01(kz, R, eps)))
    kz = 2 * math.pi / 2.56            # check: uniform eps 2.25, w = sqrt(k_z^2 + (j01/R)^2) / 1.5
    print("uniform eps 2.25: %.12f vs %.12f" % (tm01(kz, R, lambda r: 2.25), math.sqrt(kz * kz + (2.404825557695773 / R) ** 2) / 1.5))
