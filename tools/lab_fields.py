#!/usr/bin/env python3
"""Read the lab-frame snapshots of a boosted run (block lab_diagnostics, docs/specs/boosted_frame.md).

    from lab_fields import read_snapshot
    s = read_snapshot("out/lab_fields/snap_000")   # dict: meta, x (lab, 1/k0), fields {name: array [n2][n1][nx]}
    python3 tools/lab_fields.py out/lab_fields/snap_000 [component]   # prints the on-axis / mid-plane line

RZ: the arrays are the raw mode slots of the grid, slot 0 = F_0 and slots 2m-1, 2m = Re F_m, Im F_m, with the physical field
F(phi) = F_0 + 2 Re sum_m F_m e^{i m phi} (docs/specs/rz_modes.md). The m >= 1 slots are therefore half of the cos / sin amplitudes
(openPMD's C_m = 2 Re F_m, S_m = -2 Im F_m). Use field_at_angle for the physical field, e.g. E_x of a laser polarized along x on
the plane phi = 0 is field_at_angle(s, "Er", 0).
"""
import sys
import numpy as np


def read_snapshot(base):
    meta = {}
    with open(base + ".txt") as f:
        for line in f:
            if line.startswith("#") or " " not in line:
                continue
            k, v = line.rstrip("\n").split(" ", 1)
            meta[k] = v
    nx, n1, n2 = int(meta["nx"]), int(meta["n1"]), int(meta["n2"])
    a = np.fromfile(base + ".bin", dtype=np.float64).reshape(6, n2, n1, nx)
    names = meta["components"].split(" (")[0].split()
    x = float(meta["x0"]) + float(meta["dx"]) * np.arange(nx)
    return {"meta": meta, "x": x, "fields": {n: a[c] for c, n in enumerate(names)}}


def field_at_angle(s, comp, phi):
    """Physical field [n1][nx] of component comp of an RZ snapshot s (from read_snapshot) at the angle phi (rad):
    F_0 + 2 Re sum_m F_m e^{i m phi} = F_0 + 2 sum_m (Re F_m cos(m phi) - Im F_m sin(m phi)). Er, Ephi, Br, Bphi are the polar
    components at that angle (at phi = 0: Er = Ex, Ephi = Ey)."""
    f = s["fields"][comp]
    out = np.array(f[0], dtype=np.float64)
    for m in range(1, (f.shape[0] - 1) // 2 + 1):
        out += 2.0 * (f[2 * m - 1] * np.cos(m * phi) - f[2 * m] * np.sin(m * phi))
    return out


if __name__ == "__main__":
    s = read_snapshot(sys.argv[1])
    comp = sys.argv[2] if len(sys.argv) > 2 else list(s["fields"])[0]
    f = s["fields"][comp]
    rz = s["meta"]["geometry"] == "rz"
    line = f[0, 0] if rz else f[f.shape[0] // 2, f.shape[1] // 2]
    print("# lab time %s fs, complete %s; x[1/k0] %s" % (s["meta"]["t_lab_fs"], s["meta"]["complete"], comp))
    for xi, v in zip(s["x"], line):
        print("%.8g %.8g" % (xi, v))
