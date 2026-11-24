# Noise of the streaming plasma ahead of the laser in boosted runs: rms E_long and rms dn/n in the oldest plasma that the laser
# has not reached, at boosted step ST, and in RZ the electron density in the axis cell over r = 2 .. 10 um.
#   python3 noise.py OUT ST RUN...   (bench/nci/boost_stability/README.md)
import sys, os, h5py, numpy as np
st = int(sys.argv[2])
for run in sys.argv[3:]:
    try: h = h5py.File(os.path.join(sys.argv[1], run, "openpmd", "data_%06d.h5" % st))
    except Exception: print("%-28s -" % run); continue
    g = h["data"][list(h["data"])[0]]["meshes"]
    rz = g["E"].attrs["geometry"] == b"thetaMode"
    el = np.asarray(g["E"]["z" if rz else "x"]); et = np.asarray(g["E"]["r" if rz else "y"]); r = -np.asarray(g["rho_electrons"])
    el0, r0 = el[0], r[0]
    prof = r0.mean(axis=0); i0 = int(np.argmax(prof > 0.3 * prof.max()))
    il = int(np.argmax(np.abs(et[0]).max(axis=0)))                 # laser peak
    a, b = i0, min(i0 + (r0.shape[1] - i0) // 8, r0.shape[1])
    if il >= a - 20: a = il + 40; b = a + (r0.shape[1] - a) // 8
    q = r0[: (r0.shape[0] if not rz else 30), a:b]
    ax = r0[0, a:b].mean() / r0[5:25, a:b].mean() if rz else np.nan
    print("%-28s laser %4d plasma %4d window %4d-%4d: rms E_long %.1e  dn/n %.1e  n_axis/n %.3f" % (run, il, i0, a, b, np.sqrt((el0[:, a:b] ** 2).mean()), q.std() / q.mean(), ax))
