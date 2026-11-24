import sys, numpy as np
sys.path.insert(0, "$WORK/rip-pic-hhgang/tools")
import hfocus
runs = {"ripcc": "$WORK/chf/rz2/t128/out", "irip": "$WORK/chf/irip/rr3.2/t128/out"}
for band in ("low", "high"):
    print("band", band)
    D = {}
    for k, d in runs.items():
        g, q, A, c, m = hfocus.load(d + "/harmonic_focus_" + band)
        z, r = c
        F = 0.5 * (A[2] + A[5])            # fluence, mean of phi 0 and 90
        I = 0.5 * (A[0] + A[3])
        dr = r[1] - r[0]
        w = 2 * np.pi * r * dr; w[0] = np.pi * (dr / 2) ** 2
        D[k] = (z, r, (F * w[:, None]).sum(0), F[0], I.max(0), I[0])
    z = D["ripcc"][0]
    print(" z-6   Etot(cc)  Etot(irip) ratio | axisF cc irip | maxI cc irip | axisI cc irip")
    for iz in range(0, len(z), 24):
        a, b = D["ripcc"], D["irip"]
        print("%6.2f %9.3f %9.3f %6.3f | %7.2f %7.2f | %7.2f %7.2f | %7.2f %7.2f" % (z[iz] - 6, a[2][iz], b[2][iz], b[2][iz] / a[2][iz],
              a[3][iz], b[3][iz], a[4][iz], b[4][iz], a[5][iz], b[5][iz]))
