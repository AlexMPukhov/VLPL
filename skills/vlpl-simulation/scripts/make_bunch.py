"""Gaussian electron bunch as an openPMD 1.1 particle file (for species { file = ... }, decision 105) and its moments.

  python3 make_bunch.py out.h5 [Q_pC] [n] [gamma] [sigma_z_um] [sigma_r_um] [eps_n_um] [x0_um]

Positions in um (unitSI 1e-6), z with a constant positionOffset of 1 mm, momenta macro-weighted (p * w in m_e c units times
m_e c), weighting, constant charge and mass records: the conventions the import must handle. The moments printed here are
the reference for the code's import line ("species ...: Q = ..., E = ..., centroid, sigma, eps_n"), computed independently
in numpy: Q [pC], <Ekin> [MeV], rms dE/E, centroid and rms sizes [um] along z, x, y, eps_n x y [um].
"""
import sys
import h5py
import numpy as np

QE, ME, CL = 1.602176634e-19, 9.1093837015e-31, 299792458.0


def moments(x, y, z, ux, uy, uz, w):
    W = w.sum()
    g = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    m = lambda a: (w * a).sum() / W
    gm = m(g)
    out = {"Q_pC": W * QE * 1e12, "Ekin_MeV": (gm - 1) * 0.51099895000, "dE_E": np.sqrt(m((g - gm) ** 2)) / gm}
    for n, a in (("z", z), ("x", x), ("y", y)):
        out["mean_" + n] = m(a)
        out["sigma_" + n] = np.sqrt(m((a - m(a)) ** 2))
    for n, a, u in (("x", x, ux), ("y", y, uy)):
        da, du = a - m(a), u - m(u)
        out["eps_" + n] = np.sqrt(m(da * da) * m(du * du) - m(da * du) ** 2)
    return out


def main():
    a = sys.argv[1:]
    path = a[0]
    Q = float(a[1]) * 1e-12 if len(a) > 1 else 100e-12
    n = int(a[2]) if len(a) > 2 else 200000
    gamma = float(a[3]) if len(a) > 3 else 2000.0
    sz = float(a[4]) if len(a) > 4 else 100.0
    sr = float(a[5]) if len(a) > 5 else 20.0
    eps = float(a[6]) if len(a) > 6 else 1.0
    x0 = float(a[7]) if len(a) > 7 else 0.0
    rng = np.random.default_rng(7)
    x = sr * rng.standard_normal(n) + x0
    y = sr * rng.standard_normal(n)
    z = sz * rng.standard_normal(n)
    su = eps / sr                                    # eps_n = sigma_r sigma_u (no correlation)
    ux = su * rng.standard_normal(n)
    uy = su * rng.standard_normal(n)
    uz = np.sqrt(gamma**2 - 1) * (1 + 1e-4 * rng.standard_normal(n))
    w = np.full(n, Q / QE / n)
    with h5py.File(path, "w") as f:
        f.attrs["openPMD"] = np.bytes_("1.1.0")
        f.attrs["openPMDextension"] = np.uint32(0)
        f.attrs["basePath"] = np.bytes_("/data/%T/")
        f.attrs["particlesPath"] = np.bytes_("particles/")
        f.attrs["meshesPath"] = np.bytes_("meshes/")
        g = f.create_group("/data/0/particles/bunch")
        for c, v in (("x", x), ("y", y), ("z", z)):
            d = g.create_dataset("position/" + c, data=v)
            d.attrs["unitSI"] = 1e-6
            o = g.create_group("positionOffset/" + c)
            o.attrs["value"] = 1e-3 if c == "z" else 0.0
            o.attrs["unitSI"] = 1.0
            o.attrs["shape"] = np.array([n], dtype=np.uint64)
        for c, v in (("x", ux), ("y", uy), ("z", uz)):
            d = g.create_dataset("momentum/" + c, data=v * w)
            d.attrs["unitSI"] = ME * CL
        g["momentum"].attrs["macroWeighted"] = np.uint32(1)
        g["momentum"].attrs["weightingPower"] = 1.0
        g.create_dataset("weighting", data=w).attrs["unitSI"] = 1.0
        for name, v in (("charge", -QE), ("mass", ME)):
            o = g.create_group(name)
            o.attrs["value"] = v
            o.attrs["unitSI"] = 1.0
            o.attrs["shape"] = np.array([n], dtype=np.uint64)
    M = moments(x, y, z + 1000.0, ux, uy, uz, w)
    print("python moments of %s: %d particles, Q = %.6g pC, E = %.6g MeV, dE/E = %.4g," % (path, n, M["Q_pC"], M["Ekin_MeV"], M["dE_E"]))
    print("    centroid z x y = %.6g %.6g %.6g um, sigma z x y = %.6g %.6g %.6g um, eps_n x y = %.6g %.6g um"
          % (M["mean_z"], M["mean_x"], M["mean_y"], M["sigma_z"], M["sigma_x"], M["sigma_y"], M["eps_x"], M["eps_y"]))


if __name__ == "__main__":
    main()
