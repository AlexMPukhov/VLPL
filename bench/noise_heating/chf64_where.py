# Where is the electron kinetic energy: inside the target (the curved slab of the deck) or outside, and bulk (< 50 keV) or
# hot, per openPMD dump. Units of the dump are inspected and printed once.
import sys, os, glob
import numpy as np
import h5py
def load(f):
    h = h5py.File(f, "r")
    it = list(h["data"].keys())[0]
    g = h["data"][it]
    t = g.attrs.get("time", 0.0) * g.attrs.get("timeUnitSI", 1.0)
    e = g["particles"]["electrons"]
    def comp(rec, c):
        d = e[rec][c]
        if isinstance(d, h5py.Dataset): return np.asarray(d[()], dtype=float) * d.attrs.get("unitSI", 1.0)
        return float(d.attrs["value"]) * d.attrs.get("unitSI", 1.0)          # a constant record
    x, y, z = [comp("position", a) + (comp("positionOffset", a) if a in e["positionOffset"] else 0) for a in "xyz"]
    px, py, pz = comp("momentum", "x"), comp("momentum", "y"), comp("momentum", "z")
    wd = e["weighting"]
    w = np.asarray(wd[()], dtype=float) * wd.attrs.get("unitSI", 1.0) if isinstance(wd, h5py.Dataset) else float(wd.attrs["value"])
    return t, x, y, z, px, py, pz, w
me, c, e0 = 9.1093837e-31, 2.99792458e8, 1.602176634e-19
for d in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(d, "out", "openpmd", "data_*.h5")), key=lambda f: int(f.split("_")[-1].split(".")[0]))
    print("== %s" % d)
    print("  t fs    KE target bulk  target hot   outside      total (J)      N_tgt  N_out")
    for f in files:
        t, x, y, z, px, py, pz, w = load(f)
        r = np.sqrt(x * x + y * y) * 1e6; zz = z * 1e6
        g = np.sqrt(1 + (px * px + py * py + pz * pz) / (me * c) ** 2)
        ke = (g - 1) * me * c * c * w
        tgt = (zz >= 6 - r * r / 12 - 0.2) & (zz < 6.8 - r * r / 12)
        hot = (g - 1) * 0.511e6 > 50e3
        print("  %5.1f  %.4e  %.4e  %.4e  %.4e  %7d %6d" % (t * 1e15, ke[tgt & ~hot].sum(), ke[tgt & hot].sum(), ke[~tgt].sum(), ke.sum(),
              tgt.sum(), (~tgt).sum()))
