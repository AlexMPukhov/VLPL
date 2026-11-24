#!/usr/bin/env python3
"""CHF benchmark (bench/chf): the focus of each band and the energy balance of a run against its vacuum twin.

  python3 bench/chf/analyse.py RUN_DIR VACUUM_DIR [--a0 10] [--label L] [--json out.json]

RUN_DIR/out and VACUUM_DIR/out are output directories with xflux.txt, diag.txt and harmonic_focus[_<band>]/.
Energy balance (docs/specs/harmonic_focus.md): B(t) = W_in + W_kin + Enet_left + Enet_right + E_abs_r + E_part_out
of the run.
- xflux.txt with E_inj (column 12, since 2026-11-03): the laser through the open left edge is inbound in Enet_left, so B
  stays 0 (relative to E_inj); through an RZ TFSF plane E_inj is the energy injected into the incident grid at z = 0, and
  B is compared with the energy that crossed the plane, the vacuum twin's B at its end (E_inj without a twin).
- With a TFSF plane in Cart3D / 2D the incident energy enters through the plane: B is compared with the incident energy,
  Enet_right of the vacuum twin at its end.
- Older RZ runs without E_inj (laser through the open left edge): the injected energy up to each step is the vacuum twin's
  W_in + E_left + E_right + E_abs_r; B is compared with it step by step.
Reported: the largest |B - B_end_of_injection| / incident after the pulse has entered, and the value at the end; the
reflected, transmitted, absorbed (kinetic) fractions at the end.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import hfocus  # noqa: E402


def table(path, cols):
    rows = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        w = line.split()
        rows[int(w[0])] = [float(w[c]) if c < len(w) else 0.0 for c in cols]   # older files: fewer columns
    return rows


def balance(out, vac):
    # t, E_l, E_r, Enet_l, Enet_r, Eabs, W_in, E_part_out, E_inj
    x = table(os.path.join(out, "xflux.txt"), [1, 4, 5, 6, 7, 8, 9, 10, 11])
    dg = table(os.path.join(out, "diag.txt"), [3])                     # W_kin
    steps = sorted(k for k in x if k in dg)
    deck = open(os.path.join(os.path.dirname(out), "chf.deck")).read()
    edge = "tfsf_x" not in deck                                        # the laser through the left edge
    B = np.array([x[k][6] + dg[k][0] + x[k][3] + x[k][4] + x[k][5] + x[k][7] for k in steps])
    k = steps[-1]
    if x[k][8] > 0:                                                    # xflux.txt with E_inj (2026-11-03)
        inc = x[k][8]
        if edge:                                   # Enet_left counts the injection: B = 0 at all times
            ok = np.array([x[s][8] > (1 - 1e-6) * inc for s in steps])
            ref = 0.0
        else:                                      # RZ TFSF: E_inj went into the incident grid at z = 0
            W = np.array([x[s][6] + dg[s][0] for s in steps])
            ok = np.arange(len(steps)) >= int(np.argmax(W))
            vx = os.path.join(vac, "xflux.txt")
            if os.path.exists(vx):                 # the energy that crossed the plane: the vacuum twin's B at its end (the
                xv = table(vx, [1, 4, 5, 6, 7, 8, 9, 10, 11])   # incident grid loses a little before the plane)
                kv = max(xv)
                inc = xv[kv][6] + xv[kv][3] + xv[kv][4] + xv[kv][5] + xv[kv][7]
            ref = inc
        dev = (B[ok] - ref) / inc
        return {"incident": inc, "E_inj": x[k][8], "balance_max": float(np.max(np.abs(dev))), "balance_end": float(dev[-1]),
                "reflected": (x[k][1] if edge else x[k][3]) / inc,
                "transmitted": x[k][4] / inc, "kinetic": dg[k][0] / inc, "field_in_box": x[k][6] / inc,
                "absorbed_r": x[k][5] / inc, "particles_out": x[k][7] / inc, "t_end": x[k][0]}
    xv = table(os.path.join(vac, "xflux.txt"), [1, 4, 5, 6, 7, 8, 9, 10])
    if edge:
        # the injected energy up to step k: the vacuum twin's W_in + E_left + E_right + E_abs_r (its laser enters the same
        # way; the inflow sets the guard inside the field step, so xflux does not see it as inbound)
        injected = np.array([xv[k][6] + xv[k][1] + xv[k][2] + xv[k][5] for k in steps])
        inc = injected[-1]
        ok = injected > (1 - 1e-6) * inc                                 # after the pulse has entered
        B = B - injected + inc
        ref = inc
    else:
        inc = xv[max(xv)][4]                                             # the vacuum twin's transmitted energy
        W = np.array([x[k][6] + dg[k][0] for k in steps])
        ok = np.arange(len(steps)) >= int(np.argmax(W))                  # from the largest energy in the box on
        ref = inc
    dev = (B[ok] - ref) / inc
    k = steps[-1]
    return {"incident": inc, "balance_max": float(np.max(np.abs(dev))), "balance_end": float(dev[-1]),
            "reflected": (x[k][1] if edge else x[k][3]) / inc,
            "transmitted": x[k][4] / inc, "kinetic": dg[k][0] / inc, "field_in_box": x[k][6] / inc,
            "absorbed_r": x[k][5] / inc, "particles_out": x[k][7] / inc, "t_end": x[k][0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("vac")
    ap.add_argument("--a0", type=float, default=10)
    ap.add_argument("--label", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    out, vac = os.path.join(a.run, "out"), os.path.join(a.vac, "out")
    res = {"label": a.label, "balance": balance(out, vac), "bands": {}}
    for d in sorted(glob.glob(os.path.join(out, "harmonic_focus*"))):
        geom, q, A, coords, m = hfocus.load(d)
        r = hfocus.analyse(geom, q, A, coords)
        r["band"] = [float(v) for v in m["band"]]
        res["bands"][os.path.basename(d)] = r
    b = res["balance"]
    print("%s  energy balance: max %.2e, end %.2e of the incident %.6g; reflected %.4f, transmitted %.2e, kinetic %.4f, "
          "field %.2e, radial absorber %.2e, particles out %.2e (t_end %.1f)" % (a.label, b["balance_max"], b["balance_end"], b["incident"],
                                                              b["reflected"], b["transmitted"], b["kinetic"], b["field_in_box"],
                                                              b["absorbed_r"], b["particles_out"], b["t_end"]))
    for name, r in res["bands"].items():
        for k, e in r.items():
            if k == "band":
                continue
            s = "  ".join("%s %.5g" % (n, v) for n, v in e.items() if v is not None)
            print("  %-22s %-14s %s  max/a0^2 %.4g" % (name, k, s, e["max"] / a.a0 ** 2))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
