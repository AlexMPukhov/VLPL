# bench/chf/README.md "iRIP against RIP-CC": the phase error of a harmonic of order N crossing at angle theta from the target to
# the focus (3 um), from the closed-form dispersion relations of the two closures (decision 61, docs/specs/side_injection.md).
# Run: python3 bench/chf/dispersion.py
import numpy as np
# Phase error of a wave of frequency N omega0 crossing at angle theta to z: numerical kz at fixed omega and k_perp
# RIP-CC: sin^2(w D/2) = sin^2(kz D/2) + D^2 Lam/4 cos^2(kz D/2)
# iRIP:   tan^2(w D/2) = tan^2(kz D/2) + D^2 Lam/4
# Lam = (2/dr sin(kp dr/2))^2 (uniform radial grid away from the axis), D = dz = c dt. Units: lambda0 = 1.
L = 3.0 / 0.8                                    # path from the target to the focus, in lambda0
for R, rr in ((128, 40), (128, 80), (256, 80)):
    D, dr = 1.0 / R, 1.0 / rr
    print("dz = lambda0/%d, dr = lambda0/%d: phase error over %.1f lambda0 (rad)" % (R, rr, L))
    print("   N  theta | RIP-CC   iRIP   | RIP-CC - iRIP")
    for N in (5, 7, 10, 15, 20):
        w = 2 * np.pi * N
        for th in (0.0, 0.2, 0.4, 0.6):
            kz0, kp = w * np.cos(th), w * np.sin(th)
            lam = (2 / dr * np.sin(kp * dr / 2)) ** 2
            q = D * D * lam / 4
            s2 = np.sin(w * D / 2) ** 2
            c_cc = (s2 - q) / (1 - q)                # sin^2(kz D/2)
            kcc = 2 / D * np.arcsin(np.sqrt(c_cc)) if 0 <= c_cc <= 1 else np.nan
            t2 = np.tan(w * D / 2) ** 2 - q
            kir = 2 / D * np.arctan(np.sqrt(t2)) if t2 >= 0 else np.nan
            e1, e2 = (kcc - kz0) * L, (kir - kz0) * L
            print("  %3d  %.1f   | %+7.2f %+7.2f | %+7.2f" % (N, th, e1, e2, e1 - e2))
