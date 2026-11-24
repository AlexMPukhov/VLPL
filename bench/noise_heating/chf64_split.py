# Balance split of CHF runs: B = W_in + W_kin + Enet_l + Enet_r + E_abs_r + E_part_out against the incident energy (v64)
import sys, os
sys.path.insert(0, "$WORK/rip-pic-hhgang/bench/chf")
from analyse import table
xv = table("$WORK/chf/filt/v64/out/xflux.txt", [1, 4, 5, 6, 7, 8, 9, 10, 11])
kv = max(xv)
inc = xv[kv][6] + xv[kv][3] + xv[kv][4] + xv[kv][5] + xv[kv][7]
fs = 0.8e-6 / 2.99792458e8 / (2 * 3.141592653589793) * 1e15
ts = [float(a) for a in os.environ.get("TS", "20 27 32 37 43 48 53 59").split()]
for d in sys.argv[1:]:
    out = os.path.join(d, "out")
    try:
        x = table(os.path.join(out, "xflux.txt"), [1, 4, 5, 6, 7, 8, 9, 10, 11])
        dg = table(os.path.join(out, "diag.txt"), [3, 1])
    except Exception as e:
        print(d, e); continue
    steps = sorted(k for k in x if k in dg)
    print("== %s   (units: incident energy %.4g)" % (d, inc))
    print("   t fs   W_in     W_kin    Enet_l   Enet_r   E_abs_r  E_pout   excess")
    for t in ts:
        k = min(steps, key=lambda s: abs(x[s][0] * fs - t))
        v = [x[k][6], dg[k][0], x[k][3], x[k][4], x[k][5], x[k][7]]
        print("  %5.1f " % (x[k][0] * fs) + " ".join("%8.5f" % (a / inc) for a in v) + "  %+.2e" % ((sum(v) - inc) / inc))
