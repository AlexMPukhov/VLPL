"""The wake of rz_wake.deck on the axis (bench/emittance/README.md, case B): for each openPMD file of a VLPL RZ run,
the laser peak position, and along z behind it E_z(r = 0) and the focusing gradient K = d(E_r - B_theta)/dr at the axis
(m = 0, code units m c omega0 / e per 1/k0; K > 0 focuses electrons, E_z < 0 accelerates them).

python3 wake_probe.py <run_dir> [lambda0_um = 0.8]
"""
import glob
import sys
import numpy as np
import h5py

run = sys.argv[1]
lam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
k0 = 2 * np.pi / lam                                     # 1/um
E0 = 2 * np.pi * 2.99792458e8 / (lam * 1e-6) * 9.1093837e-31 * 2.99792458e8 / 1.602176634e-19   # V/m
for fn in sorted(glob.glob(run + "/openpmd/data_*.h5")):
    f = h5py.File(fn, "r")
    it = list(f["data"].keys())[0]
    m = f["data/%s/meshes" % it]
    E, B = m["E"], m["B"]
    dr, dz = E.attrs["gridSpacing"]
    z0 = E.attrs["gridGlobalOffset"][1]
    u = lambda rec: rec[...] * rec.attrs["unitSI"] / (E0 if rec.parent.name.endswith("E") else E0 / 2.99792458e8)
    Ez, Er, Bt = u(E["z"]), u(E["r"]), u(B["t"])
    Ex1 = u(E["r"])[1]                                   # mode 1 (laser, x polarisation)
    nz = Ez.shape[2]
    z = (z0 + dz * np.arange(nz)) / k0                   # um
    zl = z[np.argmax(np.abs(Ex1[:4]).max(axis=0))]
    Btz = 0.5 * (Bt[0][:, :] + np.roll(Bt[0], 1, axis=1))  # B_theta at the z of E_r
    F = Er[0] - Btz                                       # at r = (j + 1/2) dr
    K = (F[0] / (0.5 * dr) + F[1] / (1.5 * dr)) / 2      # per 1/k0
    print("# %s t-step %s  laser peak z = %.2f um" % (fn.split("/")[-1], it, zl))
    print("#  z - z_laser[um]   Ez_axis   K[code]")
    for i in range(0, nz, 8):
        if z[i] < zl:
            print("%8.2f %11.4e %11.4e" % (z[i] - zl, Ez[0, 0, i], K[i]))
