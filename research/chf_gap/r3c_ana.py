import sys, os
import numpy as np
sys.path.insert(0, "$WORK/rip-pic-hhgang/tools")
import hfocus
root = "$WORK/chfgap/r3"
# radially integrated fluence and rms radius of the fluence vs z (phi = 0 and 90 averaged), order 7 and 10
for pre, runs in (("", ("g1_cc", "g1_ir", "g1_o5", "g2_o5", "g4_cc")), ("n10_", ("g1_cc", "g1_ir", "g1_o5", "g2_cc", "g2_o5"))):
    print("order", 10 if pre else 7, "  z:", end="")
    out = {}
    for r in runs:
        geom, q, A, c, m = hfocus.load(os.path.join(root, pre + r, "out", "harmonic_focus_h"))
        z, rr = c
        F = 0.5 * (A[2] + A[5])
        dr = rr[1] - rr[0]
        w = 2 * np.pi * rr * dr; w[0] = np.pi * (dr / 2) ** 2
        E = (F * w[:, None]).sum(0)
        R = np.sqrt((F * (w * rr * rr)[:, None]).sum(0) / E)
        out[r] = (z, E, R)
    zs = [2.25, 2.4, 2.55, 2.65, 2.75, 3.0, 3.3]
    print("".join("%9.2f" % x for x in zs))
    for r, (z, E, R) in out.items():
        idx = [np.argmin(abs(z - x)) for x in zs]
        print("  %-6s E " % r + "".join("%9.3f" % E[i] for i in idx))
        print("  %-6s R " % r + "".join("%9.4f" % R[i] for i in idx))
