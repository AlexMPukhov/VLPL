import sys, os
sys.path.insert(0, "$WORK/rip-pic-hhgang/tools")
sys.path.insert(0, "$WORK/rip-pic-hhgang/bench/chf")
import hfocus
from analyse import table
B0 = "$WORK/noiseheat/chf64"
runs = [("old binary (projection)", "$WORK/chf/filt/t64_projection"), ("none", B0 + "/none"),
        ("none, ppc_theta 30", B0 + "/none_pt30"), ("gaussian 0.5 dr", B0 + "/g05"), ("gaussian 1 dr", B0 + "/g1"), ("none, ppc_theta 60", B0 + "/pt60"), ("gaussian 1 dr, ppc_theta 60", B0 + "/g1_pt60"), ("iRIP, ppc 30", B0 + "/ir30"), ("iRIP+O5, ppc 30", B0 + "/o530"), ("no filter, ppc 30", B0 + "/fp030"), ("long box, ppc 30", B0 + "/long30"), ("matched gather, ppc 30", B0 + "/gm_matched"), ("energy gather, ppc 30", B0 + "/gm_energy")]
xv = table("$WORK/chf/filt/v64/out/xflux.txt", [1, 4, 5, 6, 7, 8, 9, 10, 11])
kv = max(xv)
inc = xv[kv][6] + xv[kv][3] + xv[kv][4] + xv[kv][5] + xv[kv][7]
fs = 0.8e-6 / 2.99792458e8 / (2 * 3.141592653589793) * 1e15
ts = [27, 37, 48, 59]
print("%-26s %8s %7s %7s %8s %8s %7s | balance at %s fs | kin" % ("run", "low pk", "z_um", "fwhm_z", "low flu", "high pk", "hi flu", ts))
for name, d in runs:
    out = os.path.join(d, "out")
    try:
        res = []
        for b in ("low", "high"):
            geom, q, A, c, m = hfocus.load(os.path.join(out, "harmonic_focus_" + b))
            r = hfocus.analyse(geom, q, A, c)
            res.append((r["peak_phi0"], r["fluence_phi0"]))
        x = table(os.path.join(out, "xflux.txt"), [1, 4, 5, 6, 7, 8, 9, 10, 11])
        dg = table(os.path.join(out, "diag.txt"), [3, 1])
        steps = sorted(k for k in x if k in dg)
        row = []
        for t in ts:
            k = min(steps, key=lambda s: abs(x[s][0] * fs - t))
            B = x[k][6] + dg[k][0] + x[k][3] + x[k][4] + x[k][5] + x[k][7]
            row.append("%+.2e" % ((B - inc) / inc))
        k = steps[-1]
        (pl, fl), (ph, fh) = res
        print("%-26s %8.2f %7.3f %7.3f %8.2f %8.2f %7.2f | %s | %.4f" % (name, pl["max"], pl["z_um"], pl["fwhm_z_um"], fl["max"],
              ph["max"], fh["max"], " ".join(row), dg[k][0] / inc))
    except Exception as e:
        print("%-26s (%s)" % (name, e))
