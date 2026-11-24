"""E_z on the axis and the focusing gradient K = d(E_r - B_theta)/dr (m = 0, code units) at fixed distances from the back
edge of the moving window, against time: where a witness moving at ~c stays (bench/emittance/README.md, case B).

python3 wake_time.py <run_dir> [d1_um d2_um ...]       (default 2.0 2.5 3.0 3.5 4.0 4.5 5.0)
"""
import glob
import sys
import numpy as np
import h5py

run = sys.argv[1]
ds = [float(a) for a in sys.argv[2:]] or [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
lam = 0.8
k0 = 2 * np.pi / lam
E0 = 2 * np.pi * 2.99792458e8 / (lam * 1e-6) * 9.1093837e-31 * 2.99792458e8 / 1.602176634e-19
print("# t[fs]  " + "  ".join("Ez(%g) K(%g)" % (d, d) for d in ds) + "   (d: um from the back edge)")
for fn in sorted(glob.glob(run + "/openpmd/data_*.h5")):
    f = h5py.File(fn, "r")
    it = list(f["data"].keys())[0]
    m = f["data/%s/meshes" % it]
    E, B = m["E"], m["B"]
    dr, dz = E.attrs["gridSpacing"]
    t = f["data/" + it].attrs["time"] * f["data/" + it].attrs["timeUnitSI"] * 1e15
    Ez = E["z"][0] * E["z"].attrs["unitSI"] / E0
    Er = E["r"][0] * E["r"].attrs["unitSI"] / E0
    Bt = B["t"][0] * B["t"].attrs["unitSI"] / (E0 / 2.99792458e8)
    F = Er - 0.5 * (Bt + np.roll(Bt, 1, axis=1))
    K = (F[0] / (0.5 * dr) + F[1] / (1.5 * dr)) / 2
    row = []
    for d in ds:
        i = int(round(d * k0 / dz))
        row.append("%9.4f %9.2e" % (Ez[0, i], K[i]))
    print("%7.1f  %s" % (t, "  ".join(row)))
