#!/usr/bin/env python3
"""Angular distribution of the reflected harmonics, VLPL against Smilei (bench/hhg_oblique/README.md).

  python3 bench/hhg_oblique/angles.py --rip out_64/xflux --smilei sm_foil64 sm_vac64 1.2251 [--nmax 16] [--out angles]

The two codes record the reflected field on planes at different x (VLPL: the inner face of the left absorber;
Smilei: the probe 0.5 um inside xmin), so the near-field profiles differ by the propagation in between. The angular
spectrum does not: for each frequency bin of the band [N - 1/2, N + 1/2] the outbound components (VLPL: the xflux
records; Smilei: Ey - Bz, Ez + By of the plasma run minus scale x the vacuum run, as smilei/compare_smilei.py) are
Fourier transformed in y and z, and |F(omega, k_y, k_z)|^2 is binned in the direction sines s_y = k_y c / omega (in the
plane of incidence) and s_z = k_z c / omega. Reported per order: the band energy, the centroid angle theta_y (from the
specular direction -x) and the rms widths in theta_y and theta_z, each with |s| < s_max = the Nyquist sine of the
coarser plane (Smilei's probe spacing 0.1 um: s_max = 4 / N), the energy beyond s_max in both, and the overlap of the
two normalised distributions. Writes <out>_N.txt (s, dW/ds_y of both codes, dW/ds_z of both) and <out>.txt (the table).
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import xflux  # noqa: E402

SB = np.linspace(-1, 1, 401)            # bins of the direction sines


def rip_records(d):
    """t step, T [n_t][2][ny][nz], dy, dz (lengths in c / omega0, t in 1 / omega0)"""
    t, T, meta = xflux.load(d, "left")
    h = [float(x) for x in meta["h"]]
    return t[1] - t[0], np.ascontiguousarray(np.swapaxes(T, 2, 3)), h[1], h[2]


def smilei_records(pdir, vdir, scale):
    import h5py
    log = open(os.path.join(pdir, "smilei.log")).read()
    dt = float(re.search(r"timestep = ([0-9.]+)", log).group(1))
    fp, fv = h5py.File(os.path.join(pdir, "Probes0.h5"), "r"), h5py.File(os.path.join(vdir, "Probes0.h5"), "r")
    names = fp.attrs["fields"].decode().split(",")
    iEy, iEz, iBy, iBz = (names.index(n) for n in ("Ey", "Ez", "By", "Bz"))
    pos = fp["positions"][()]
    ys, zs = np.unique(np.round(pos[:, 1], 6)), np.unique(np.round(pos[:, 2], 6))
    iy = np.searchsorted(ys, np.round(pos[:, 1], 6))
    iz = np.searchsorted(zs, np.round(pos[:, 2], 6))
    ks = sorted(set(k for k in fp.keys() if k.isdigit()) & set(k for k in fv.keys() if k.isdigit()), key=int)
    T = np.zeros((len(ks), 2, len(ys), len(zs)), np.float32)
    for j, k in enumerate(ks):
        r = fp[k][()] - scale * fv[k][()]
        T[j, 0, iy, iz] = r[iEy] - r[iBz]
        T[j, 1, iy, iz] = r[iEz] + r[iBy]
    return dt, T, ys[1] - ys[0], zs[1] - zs[0]


def angular(dt, T, dy, dz, nmax):
    """per order N: (band energy, histogram over s_y, histogram over s_z) in the units of S = |rfft T dt|^2 dy dz / 4"""
    nt = T.shape[0]
    w = 2 * np.pi * np.fft.rfftfreq(nt, dt)
    F = np.fft.rfft(T, axis=0) * dt
    ny, nz = T.shape[2], T.shape[3]
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    kz = 2 * np.pi * np.fft.fftfreq(nz, dz)
    out = {}
    for N in range(1, nmax + 1):
        sel = np.where((w >= N - 0.5) & (w < N + 0.5))[0]
        if len(sel) == 0 or w[-1] < N + 0.5:
            break
        hy, hz, W = np.zeros(len(SB) - 1), np.zeros(len(SB) - 1), 0.0
        for b in sel:
            # Parseval in y, z: sum |fft2|^2 / (ny nz) = sum |F|^2 over the plane
            P = (np.abs(np.fft.fft2(F[b], axes=(-2, -1))) ** 2).sum(axis=0) * dy * dz / 4 / (ny * nz)
            W += P.sum()
            sy, sz = ky / w[b], kz / w[b]
            hy += np.histogram(np.broadcast_to(sy[:, None], P.shape).ravel(), SB, weights=P.ravel())[0]
            hz += np.histogram(np.broadcast_to(sz[None, :], P.shape).ravel(), SB, weights=P.ravel())[0]
        out[N] = (W, hy, hz)
    return out


def moments(h, smax):
    s = 0.5 * (SB[1:] + SB[:-1])
    m = np.abs(s) < smax
    p = h * m
    tot = p.sum()
    c = (p * s).sum() / tot
    rms = np.sqrt((p * (s - c) ** 2).sum() / tot)
    return np.degrees(np.arcsin(c)), np.degrees(rms), 1 - tot / h.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rip", required=True)
    ap.add_argument("--smilei", nargs=3, metavar=("PLASMA", "VACUUM", "SCALE"), required=True)
    ap.add_argument("--nmax", type=int, default=16)
    ap.add_argument("--out", default="angles")
    ap.add_argument("--rip-stride", type=int, default=1,
                    help="use every n-th point of the VLPL plane (2: Smilei's probe spacing, the same aliasing)")
    a = ap.parse_args()
    dr, Tr, dyr, dzr = rip_records(a.rip)
    if a.rip_stride > 1:
        Tr = np.ascontiguousarray(Tr[:, :, ::a.rip_stride, ::a.rip_stride])
        dyr, dzr = dyr * a.rip_stride, dzr * a.rip_stride
    print("VLPL: %d records, dt %.4g, plane %d x %d, dy %.4g dz %.4g" % (Tr.shape[0], dr, Tr.shape[2], Tr.shape[3], dyr, dzr))
    R = angular(dr, Tr, dyr, dzr, a.nmax)
    del Tr
    ds, Ts, dys, dzs = smilei_records(a.smilei[0], a.smilei[1], float(a.smilei[2]))
    print("Smilei: %d records, dt %.4g, plane %d x %d, dy %.4g dz %.4g" % (Ts.shape[0], ds, Ts.shape[2], Ts.shape[3], dys, dzs))
    S = angular(ds, Ts, dys, dzs, a.nmax)
    dmax = max(dyr, dzr, dys, dzs)
    s = 0.5 * (SB[1:] + SB[:-1])
    rows = []
    hdr = ("N  W_rip W_smilei ratio | theta_y centroid rip smilei (deg) | rms theta_y rip smilei | rms theta_z rip smilei | "
           "beyond s_max rip smilei | overlap y z | s_max | ratio inside s_max")
    print(hdr)
    for N in sorted(set(R) & set(S)):
        smax = min(1.0, np.pi / dmax / N)              # the Nyquist sine of the coarser plane at order N
        (Wr, hyr, hzr), (Ws, hys, hzs) = R[N], S[N]
        cr, wyr, br = moments(hyr, smax)
        cs, wys, bs = moments(hys, smax)
        _, wzr, _ = moments(hzr, smax)
        _, wzs, _ = moments(hzs, smax)
        m = np.abs(s) < smax
        ov = [np.sqrt((x[m] / x[m].sum() * y[m] / y[m].sum()).clip(0)).sum() for x, y in ((hyr, hys), (hzr, hzs))]
        rows.append((N, Wr, Ws, Wr / Ws, cr, cs, wyr, wys, wzr, wzs, br, bs, ov[0], ov[1], smax, Wr * (1 - br) / (Ws * (1 - bs))))
        print("%2d %.3e %.3e %5.2f | %+6.2f %+6.2f | %5.2f %5.2f | %5.2f %5.2f | %.1e %.1e | %.3f %.3f | %.2f | %5.2f" % rows[-1])
        db = SB[1] - SB[0]
        np.savetxt("%s_%d.txt" % (a.out, N), np.c_[s, hyr / db, hys / db, hzr / db, hzs / db],
                   header="order %d: s = sin(theta); dW/ds_y VLPL, Smilei; dW/ds_z VLPL, Smilei (|s| < %.3f resolved by both)"
                   % (N, smax))
    np.savetxt(a.out + ".txt", np.array(rows), fmt="%.6g", header=hdr)


def compare_runs(runs, nmax, cone_um, out):
    """VLPL runs against each other (e.g. 2D variants: particles per cell, filter, dy, dx): per order the band energy,
    the part at |s| > s_c = lambda0 / (2 N cone) (transverse wavelengths below 2 cone) and the energies relative to the
    first run, in all and inside s_c."""
    res = {}
    for lab, d in runs:
        dt, T, dy, dz = rip_records(d)
        lam = float(xflux.meta_of(os.path.join(d, "meta_r0.txt"))["lambda0"][0]) * 1e6    # um
        res[lab] = angular(dt, T, dy, dz, nmax)
        print("%s: %d records, dt %.4g, plane %d x %d, dy %.4g" % (lab, T.shape[0], dt, T.shape[2], T.shape[3], dy))
        del T
    cone = cone_um * 2 * np.pi / lam                   # in c / omega0
    labs = [l for l, _ in runs]
    hdr = "N s_c | " + " | ".join("%s: W, beyond s_c, W/W_%s, inside/inside_%s" % (l, labs[0], labs[0]) for l in labs)
    print(hdr)
    rows = []
    for N in sorted(set.intersection(*[set(res[l]) for l in labs])):
        sc = min(1.0, np.pi / cone / N)
        row, txt = [N, sc], "%2d %.2f" % (N, sc)
        W0, hy0, _ = res[labs[0]][N]
        in0 = W0 * (1 - moments(hy0, sc)[2])
        for l in labs:
            W, hy, _ = res[l][N]
            b = moments(hy, sc)[2]
            row += [W, b, W / W0, W * (1 - b) / in0]
            txt += " | %.3e %.2f %5.2f %5.2f" % (W, b, W / W0, W * (1 - b) / in0)
        rows.append(row)
        print(txt)
    np.savetxt(out + ".txt", np.array(rows), fmt="%.6g", header=hdr)


if __name__ == "__main__":
    if "--runs" in sys.argv:
        ap = argparse.ArgumentParser(description=compare_runs.__doc__)
        ap.add_argument("--runs", nargs="+", required=True, help="LABEL=XFLUX_DIR, the first is the reference")
        ap.add_argument("--nmax", type=int, default=16)
        ap.add_argument("--cone", type=float, default=0.1, help="spacing in um that defines s_c (default 0.1: Smilei's probe)")
        ap.add_argument("--out", default="runs")
        a = ap.parse_args()
        compare_runs([tuple(r.split("=", 1)) for r in a.runs], a.nmax, a.cone, a.out)
        sys.exit()
    main()
