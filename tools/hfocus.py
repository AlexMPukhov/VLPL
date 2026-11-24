#!/usr/bin/env python3
"""Focus of the band-passed field (block harmonic_focus, docs/specs/harmonic_focus.md).

  python3 tools/hfocus.py out/harmonic_focus [--a0 A] [--json file]

Assembles the per-rank maps (peak of |E_band|^2 over the run, its time, the fluence) and reports, for the peak
intensity and for the fluence (RZ: at phi = 0 and phi = 90 deg):
  - the position of the maximum (x or z, and y, z or r, in um),
  - the maximum (I in units of (m c omega0 / e)^2, i.e. a^2; with --a0 also / a0^2),
  - the FWHM through the maximum along the axis and across it (RZ: radial full width 2 r_1/2, um),
interpolated quadratically around the grid maximum. Fluence in units of a^2 / omega0.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def read_meta(path):
    m = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        w = line.split()
        if w:
            m[w[0]] = w[1:]
    return m


def load(d):
    metas = sorted(glob.glob(os.path.join(d, "meta_r*.txt")))
    if not metas:
        sys.exit("no meta_r*.txt in " + d)
    parts = []
    for mp in metas:
        r = mp[mp.rfind("_r") + 2:-4]
        m = read_meta(mp)
        start = [int(v) for v in m["start"]]
        size = [int(v) for v in m["size"]]
        q = m["quantities"]
        a = np.fromfile(os.path.join(d, "r%s.bin" % r), dtype=np.float64)
        if m["geometry"][0] == "rz":
            a = a.reshape(len(q), size[1], size[0])            # [q][r][z]
        else:
            a = a.reshape(len(q), size[2], size[1], size[0])   # [q][z][y][x]
        parts.append((start, size, a, m))
    m = parts[0][3]
    geom = m["geometry"][0]
    q = m["quantities"]
    g0 = [min(p[0][a] for p in parts) for a in range(3)]
    g1 = [max(p[0][a] + p[1][a] for p in parts) for a in range(3)]
    h = [float(v) for v in m["h"]]
    lam = float(m["lambda0"][0])
    um = lam / (2 * np.pi) * 1e6                                # code length unit in um
    if geom == "rz":
        A = np.zeros((len(q), g1[1] - g0[1], g1[0] - g0[0]))
        for s, z, a, _ in parts:
            A[:, s[1] - g0[1]:s[1] - g0[1] + z[1], s[0] - g0[0]:s[0] - g0[0] + z[0]] = a
        r = np.array([float(v) for v in m["r_nodes"]])
        coords = [(g0[0] + np.arange(A.shape[2])) * h[0] * um, r[:A.shape[1]] * um]
    else:
        A = np.zeros((len(q), g1[2] - g0[2], g1[1] - g0[1], g1[0] - g0[0]))
        for s, z, a, _ in parts:
            A[:, s[2] - g0[2]:s[2] - g0[2] + z[2], s[1] - g0[1]:s[1] - g0[1] + z[1], s[0] - g0[0]:s[0] - g0[0] + z[0]] = a
        coords = [(g0[a] + np.arange(A.shape[3 - a])) * h[a] * um for a in range(3)]
    return geom, q, A, coords, m


def peak1d(x, f, k):
    """Quadratic interpolation of the maximum of f at index k: (position, value)."""
    if 0 < k < len(f) - 1:
        a, b, c = f[k - 1], f[k], f[k + 1]
        den = a - 2 * b + c
        if den < 0:
            u = 0.5 * (a - c) / den
            return x[k] + u * (x[k + 1] - x[k]), b - 0.25 * (a - c) * u
    return x[k], f[k]


def fwhm(x, f, k, half_axis=False):
    """Full width at half maximum of f around index k (linear interpolation); half_axis: f(x) on x >= 0 (radial),
    the full width 2 x_1/2 if the maximum is on the axis."""
    h = 0.5 * f[k]
    i = k
    while i < len(f) - 1 and f[i + 1] > h:
        i += 1
    if i == len(f) - 1:
        return float("nan")
    xr = x[i] + (f[i] - h) / (f[i] - f[i + 1]) * (x[i + 1] - x[i])
    if half_axis and k == 0:
        return 2 * xr
    j = k
    while j > 0 and f[j - 1] > h:
        j -= 1
    if j == 0:
        return float("nan")
    xl = x[j] - (f[j] - h) / (f[j] - f[j - 1]) * (x[j] - x[j - 1])
    return xr - xl


def analyse(geom, q, A, coords):
    res = {}
    names = ["phi0", "phi90"] if geom == "rz" else [""]
    for a, nm in enumerate(names):
        for kind, iq in (("peak", 3 * a), ("fluence", 3 * a + 2)):
            F = A[iq]
            key = kind + ("_" + nm if nm else "")
            idx = np.unravel_index(np.argmax(F), F.shape)
            if geom == "rz":
                jr, iz = idx
                z, v = peak1d(coords[0], F[jr], iz)
                e = {"z_um": z, "r_um": coords[1][jr], "max": v, "fwhm_z_um": fwhm(coords[0], F[jr], iz),
                     "fwhm_r_um": fwhm(coords[1], F[:, iz], jr, half_axis=True), "t_peak": float(A[iq + 1][jr, iz]) if kind == "peak" else None}
            else:
                kz, jy, ix = idx
                x, v = peak1d(coords[0], F[kz, jy], ix)
                e = {"x_um": x, "y_um": coords[1][jy], "z_um": coords[2][kz], "max": v,
                     "fwhm_x_um": fwhm(coords[0], F[kz, jy], ix), "fwhm_y_um": fwhm(coords[1], F[kz, :, ix], jy)}
                if F.shape[0] > 1:
                    e["fwhm_z_um"] = fwhm(coords[2], F[:, jy, ix], kz)
                if kind == "peak":
                    e["t_peak"] = float(A[iq + 1][kz, jy, ix])
            res[key] = e
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--a0", type=float, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    geom, q, A, coords, m = load(a.dir)
    res = analyse(geom, q, A, coords)
    res["band"] = [float(v) for v in m["band"]]
    for k, e in res.items():
        if k == "band":
            continue
        s = "  ".join("%s %.5g" % (n, v) for n, v in e.items() if v is not None)
        if a.a0 > 0:
            s += "  max/a0^2 %.5g" % (e["max"] / a.a0 ** 2)
        print("%-14s %s" % (k, s))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
