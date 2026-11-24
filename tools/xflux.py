"""Field records of output.xflux_every (docs/specs/xflux.md): the outbound transport components at the
open x edges, as a time series on the edge plane, and the harmonic spectrum of the reflected (left) or
transmitted (right) field.

    python tools/xflux.py out/xflux --edge left [--png spectrum.png] [--nmax 30]

Writes <dir>/spectrum_<edge>.txt: harmonic order omega / omega0 and dW/domega (summed over the plane, in the
energy weights of the code: Cart3D cell area, RZ 2 pi r dr per mode with w_0 = 1, w_m = 2), and for each
integer order N <= nmax the transverse profile of that harmonic (<dir>/profile_<edge>_N.txt).
Use as a module: load(dir, edge) -> t, T (records, [n_t][2][n_perp...]), meta.
"""
import argparse
import glob
import os
import numpy as np


def meta_of(path):
    m = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        w = line.split()
        if w:
            m[w[0]] = w[1:]
    return m


def load(d, edge):
    """t (code units 1/omega0), T [n_t][2][...] (Cart3D: [nz][ny] of the global plane; RZ: [slots][nr]), meta"""
    metas = sorted(glob.glob(os.path.join(d, "meta_r*.txt")))
    parts, meta, nt = [], None, None
    for mp in metas:
        m = meta_of(mp)
        if m.get(edge, ["0"])[0] != "1":
            continue
        r = mp.split("_r")[-1][:-4]
        n = [int(x) for x in m["n"]]
        off = [int(x) for x in m["off"]]
        a = np.fromfile(os.path.join(d, "%s_r%s.bin" % (edge, r)), dtype=np.float64)
        rec = 2 * n[1] * n[2]
        a = a[: (a.size // rec) * rec].reshape(-1, 2, n[2], n[1])
        nt = a.shape[0] if nt is None else min(nt, a.shape[0])
        parts.append((off, n, a))
        meta = m
    if not parts:
        raise SystemExit("no records of the %s edge in %s" % (edge, d))
    N = [int(x) for x in meta["N"]]
    rz = meta["geometry"][0] == "rz"
    if rz:
        T = parts[0][2][:nt]
    else:
        T = np.zeros((nt, 2, N[2], N[1]))
        for off, n, a in parts:
            T[:, :, off[2]:off[2] + n[2], off[1]:off[1] + n[1]] = a[:nt]
    every, dt = int(meta["every"][0]), float(meta["dt"][0])
    t = dt * every * (np.arange(nt) + 1)
    return t, T, meta


def weights(T, meta):
    """energy weight of each plane point (per component): T^2 / 4 * weight * c dt = energy per step"""
    h = [float(x) for x in meta["h"]]
    if meta["geometry"][0] == "rz":
        A = np.array([float(x) for x in meta["area_nodes"]])     # FieldsRZ::energy: E_phi, B_r at the nodes
        B = np.array([float(x) for x in meta["area_half"]])      # E_r, B_phi at the half nodes
        S = T.shape[1 + 1]
        wm = np.array([1.0] + [2.0] * (S - 1))[:, None]
        return 2 * np.pi * np.stack([wm * B[None, :], wm * A[None, :]])
    return np.full(T.shape[1:], h[1] * h[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--edge", default="left", choices=["left", "right"])
    ap.add_argument("--nmax", type=int, default=30)
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    t, T, meta = load(a.dir, a.edge)
    dtr = t[1] - t[0]
    F = np.fft.rfft(T, axis=0) * dtr
    w = np.fft.rfftfreq(T.shape[0], dtr) * 2 * np.pi            # omega / omega0 (time in 1/omega0)
    wt = weights(T, meta)
    spec = (np.abs(F) ** 2 * wt[None]).reshape(len(w), -1).sum(axis=1) / 4
    out = os.path.join(a.dir, "spectrum_%s.txt" % a.edge)
    np.savetxt(out, np.c_[w, spec], header="omega/omega0  dW/domega (arb.: |FFT T|^2 / 4 in the energy weights)")
    for N in range(1, a.nmax + 1):
        sel = (w > N - 0.5) & (w <= N + 0.5)
        if not sel.any():
            break
        prof = (np.abs(F[sel]) ** 2).sum(axis=0) / 4
        np.savetxt(os.path.join(a.dir, "profile_%s_%d.txt" % (a.edge, N)), prof.reshape(prof.shape[0], -1))
    print("%s: %d records, dt %.4g, spectrum to omega = %.1f omega0 -> %s" % (a.edge, len(t), dtr, w[-1], out))
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.semilogy(w, spec)
        plt.xlabel("omega / omega0")
        plt.ylabel("dW / domega")
        plt.xlim(0, min(w[-1], a.nmax))
        plt.title("%s edge" % a.edge)
        plt.savefig(a.png, dpi=120)


if __name__ == "__main__":
    main()
