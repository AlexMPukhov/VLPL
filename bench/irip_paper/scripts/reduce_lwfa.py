"""Reduce the LWFA benchmark (VLPL bench/lwfa_fbpic, a0 = 1, 0.01 n_c) to ../data/lwfa_axis_<t>fs.txt and
../data/lwfa_l2.txt: on-axis longitudinal field of VLPL-RIP (3D, RIP-CC), VLPL-iRIP (3D, box iRIP), VLPL-RZ (RIP-CC),
VLPL-iRZ (box iRIP) and FBPIC (2 modes), in m c omega0 / e. 3D: E_x on the axis from the x-y slices (as bench
compare.py); RZ and FBPIC: E_z of m = 0 averaged over r <= 0.5 um (as bench compare_rz.py), FBPIC also on its first
radial row (r = dr/2) for the 3D comparison. openPMD read with h5py (thetaMode, both codes). Run with h5py:
  python reduce_lwfa.py RUNS_DIR    (RUNS_DIR/lwfa3d_ripcc, lwfa3d_irip, lwfarz_ripcc, lwfarz_irip, fbpic_lwfa)"""
import os, sys, glob
import numpy as np
import h5py

R = sys.argv[1]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
lam = 0.8e-6; k0 = 2 * np.pi / lam; c = 2.99792458e8
E0 = 9.1093837015e-31 * c * (2 * np.pi * c / lam) / 1.602176634e-19


def slices(run):
    d = os.path.join(R, run, "slices")
    meta = dict(l.split(None, 1) for l in open(os.path.join(d, "meta.txt")))
    nx, ny, dx = int(meta["nx"]), int(meta["ny"]), float(meta["dx"])
    idx = {int(s): (float(t), float(x0)) for s, t, x0 in (l.split() for l in open(os.path.join(d, "index.txt")))}
    def get(name, step):
        return np.fromfile(os.path.join(d, "%s_%06d.f32" % (name, step)), dtype=np.float32).reshape(ny, nx)
    return nx, ny, dx, idx, get


def thetamode(path):
    """dict: z [m], r [m] of E_z, Ez (2M-1, nr, nz) [V/m], max |E_r(theta = 0)| (laser field) [V/m]"""
    with h5py.File(path, "r") as f:
        it = f["data"][list(f["data"].keys())[0]]
        mp = f.attrs["meshesPath"]; mp = (mp.decode() if isinstance(mp, bytes) else mp).strip("/")
        E = it[mp]["E"]
        sp = E.attrs["gridSpacing"] * E.attrs["gridUnitSI"]; off = E.attrs["gridGlobalOffset"] * E.attrs["gridUnitSI"]
        ez = E["z"][()] * E["z"].attrs["unitSI"]; pz = E["z"].attrs["position"]
        er = E["r"][()] * E["r"].attrs["unitSI"]
        z = off[1] + (np.arange(ez.shape[2]) + pz[1]) * sp[1]
        r = off[0] + (np.arange(ez.shape[1]) + pz[0]) * sp[0]
        M = (er.shape[0] + 1) // 2
        er0 = er[0] + sum(er[2 * m - 1] for m in range(1, M))
        return dict(z=z, r=r, ez=ez, las=np.abs(er0).max(), t=it.attrs["time"] * it.attrs["timeUnitSI"])


def files(run, sub, pat):
    return {int(os.path.basename(p).split("data")[1].strip("_").split(".")[0]): p
            for p in glob.glob(os.path.join(R, run, sub, pat))}


a3 = slices("lwfa3d_ripcc"); b3 = slices("lwfa3d_irip")
frc = files("lwfarz_ripcc", "openpmd", "data_*.h5"); fri = files("lwfarz_irip", "openpmd", "data_*.h5")
ffb = files("fbpic_lwfa", "hdf5", "data*.h5")
nx, ny, dx, idx, get = a3
rows = []
for step in sorted(idx):
    if step == 0 or step not in ffb or step not in frc or step not in fri:
        continue
    t, x0 = idx[step]
    x = (x0 + dx * np.arange(nx)) / k0
    ex_rc = get("Ex", step)[ny // 2]
    ex_ri = b3[4]("Ex", step)[ny // 2]
    las3 = np.max(np.abs(get("Ez", step)))
    fb, rc, ri = thetamode(ffb[step]), thetamode(frc[step]), thetamode(fri[step])
    near = lambda d: d["ez"][0][d["r"] <= 0.5e-6 + 1e-12].mean(axis=0) / E0
    I = lambda d, f: np.interp(x, d["z"], f)
    cols = np.column_stack([x * 1e6, ex_rc, ex_ri, I(rc, near(rc)), I(ri, near(ri)), I(fb, fb["ez"][0][0] / E0),
                            I(fb, near(fb))])
    tfs = t / (2 * np.pi * c / lam) * 1e15
    np.savetxt(os.path.join(out, "lwfa_axis_%03.0ffs.txt" % tfs), cols, fmt="%.5e",
               header="t = %.1f fs; x [um], 3D on-axis E_x: RIP-CC, box iRIP; RZ near-axis E_z (m=0, r<=0.5um): RIP-CC, "
                      "box iRIP; FBPIC first row (r = dr/2, m=0), FBPIC near-axis (m=0); units m c w0/e" % tfs)
    xl = x[np.argmax(np.abs(get("Ez", step)[ny // 2]))] - 2e-6       # behind the pulse, as the bench scripts
    m = x < xl
    if m.sum() < 10 or np.max(np.abs(cols[m, 5])) < 5e-3:
        continue
    l2 = lambda a, b: np.sqrt(np.sum((a[m] - b[m]) ** 2) / np.sum(b[m] ** 2))
    rows.append([tfs, l2(cols[:, 1], cols[:, 5]), l2(cols[:, 2], cols[:, 5]), l2(cols[:, 3], cols[:, 6]),
                 l2(cols[:, 4], cols[:, 6]), l2(cols[:, 2], cols[:, 1]), l2(cols[:, 4], cols[:, 3]),
                 np.max(np.abs(cols[m, 1])), np.max(np.abs(cols[m, 2])), np.max(np.abs(cols[m, 3])),
                 np.max(np.abs(cols[m, 4])), np.max(np.abs(cols[m, 6])),
                 las3, rc["las"] / E0, ri["las"] / E0, fb["las"] / E0])
np.savetxt(os.path.join(out, "lwfa_l2.txt"), np.array(rows), fmt="%.4g",
           header="t[fs]  L2: 3D RIP-CC vs FBPIC row, 3D iRIP vs FBPIC row, RZ RIP-CC vs FBPIC near-axis, RZ iRIP vs FBPIC "
                  "near-axis, 3D iRIP vs 3D RIP-CC, RZ iRIP vs RZ RIP-CC;  peak wake: 3D RIP-CC, 3D iRIP, RZ RIP-CC, RZ iRIP, "
                  "FBPIC near-axis;  peak laser field: 3D RIP-CC (E_z slice), RZ RIP-CC, RZ iRIP, FBPIC")
for r in rows:
    print(" ".join("%.4g" % v for v in r))
