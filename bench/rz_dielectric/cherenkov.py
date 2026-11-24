#!/usr/bin/env python3
"""Handover test 3: the Cherenkov cone of cherenkov.deck at the lab plane. For every radius r (half nodes) the arrival time
of the peak of |B_phi(r, t)| at the plane; a straight-line fit t(r) = t0 + r s over r1 < r < r2 against the cone
s = sqrt(eps beta^2 - 1) / (beta c) (the front of a bunch at v > c / n: zeta = -r sqrt(n^2 beta^2 - 1)).
  python3 cherenkov.py <run dir> <eps> [gamma r1_um r2_um]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import rz_rings as R  # noqa: E402


def main():
    run, eps = sys.argv[1], float(sys.argv[2])
    G = float(sys.argv[3]) if len(sys.argv) > 3 else 1000.0
    r1, r2 = (float(sys.argv[4]) * 1e-6, float(sys.argv[5]) * 1e-6) if len(sys.argv) > 5 else (60e-6, 360e-6)
    beta = np.sqrt(1 - 1 / G**2)
    M = R.meta(run)
    u = M["u"]
    st, Fp = R.plane(run, M)
    t = st * M["dt"] * u["l"]                     # c t in m
    Bp = Fp[:, 5, 0, :]                          # B_phi at the half nodes
    rh = M["rh"] * u["l"]
    sel = (rh > r1) & (rh < r2)
    tp = []
    for j in np.where(sel)[0]:
        y = np.abs(Bp[:, j])
        i = int(np.argmax(y))
        if 0 < i < len(y) - 1:                    # parabolic interpolation of the peak
            a, b, c = y[i - 1], y[i], y[i + 1]
            tp.append(t[i] + 0.5 * (a - c) / (a - 2 * b + c) * (t[1] - t[0]))
        else:
            tp.append(t[i])
    tp = np.array(tp)
    s, t0 = np.polyfit(rh[sel], tp, 1)
    sth = np.sqrt(eps * beta**2 - 1) / beta
    res = np.std(tp - (t0 + s * rh[sel]))
    th, thth = np.degrees(np.arctan(1 / s)), np.degrees(np.arccos(1 / (np.sqrt(eps) * beta)))
    print("eps %.3f, gamma %g: c dt/dr of the B_phi peak over r = %.0f .. %.0f um: %.5f (cone %.5f, relative %+.2e); "
          "rms deviation from the line %.2f um; Cherenkov angle %.3f deg (theory %.3f deg)" %
          (eps, G, r1 * 1e6, r2 * 1e6, s, sth, s / sth - 1, res * 1e6, 90 - th, thth))
    ok = abs(s / sth - 1) < 5e-3
    print("PASS (slope to 5e-3)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
