"""Reduce the NCI streaming-plasma runs (runs/gpuq2.sh: VLPL 3D and RZ, FBPIC standard / Galilean) to
../data/nciv_<label>.txt: omega_p t and W_field / K_e (K_e: initial kinetic energy of the electrons).
VLPL: diag.txt (W_field, column 3; W_kin of electrons + protons with the same u, so K_e = W_kin(0) / (1 + 1836));
RZ also W_0, W_1 from rz_modes.txt. FBPIC: energy.txt of fbpic_nci.py (K_e in its header).
  python reduce_nci.py RUNS/nci"""
import os, sys, glob, re
import numpy as np

N = sys.argv[1]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
wp = 0.1                                            # omega_p / omega0 (n = 0.01 n_c)
summary = []
for d in sorted(glob.glob(os.path.join(N, "c3_*")) + glob.glob(os.path.join(N, "rz_*"))):
    if not os.path.isdir(d):
        continue
    lab = os.path.basename(d)
    a = np.loadtxt(os.path.join(d, "diag.txt"))
    Ke = a[0, 3] / 1837.0
    cols = [a[:, 1] * wp, a[:, 2] / Ke]
    hdr = "%s: omega_p t, W_field/K_e" % lab
    if lab.startswith("rz_"):
        m = np.loadtxt(os.path.join(d, "rz_modes.txt"))
        cols += [np.interp(a[:, 0], m[:, 0], m[:, 2]) / Ke, np.interp(a[:, 0], m[:, 0], m[:, 3]) / Ke]
        hdr += ", W_0/K_e, W_1/K_e"
    hdr += "; max|dGauss| over the run %.1e" % a[:, 5].max()
    np.savetxt(os.path.join(out, "nciv_%s.txt" % lab), np.column_stack(cols), fmt="%.4e", header=hdr)
    summary.append((lab, cols[1][-1], cols[0][-1], a[:, 5].max()))
for d in sorted(glob.glob(os.path.join(N, "fb_*"))):
    if not os.path.isdir(d):
        continue
    lab = os.path.basename(d)
    f = os.path.join(d, "energy.txt")
    if not os.path.exists(f):
        continue
    h = open(f).readline()
    Ke = float(re.search(r"K_e = (\S+) J", h).group(1)); wps = float(re.search(r"wp = (\S+) /s", h).group(1))
    a = np.loadtxt(f)
    cols = [a[:, 1] * wps, (a[:, 2] + a[:, 3]) / Ke, a[:, 2] / Ke, a[:, 3] / Ke]
    np.savetxt(os.path.join(out, "nciv_%s.txt" % lab), np.column_stack(cols), fmt="%.4e",
               header="%s: omega_p t, W_field/K_e, W_0/K_e, W_1/K_e (%s)" % (lab, h.strip("# \n")))
    summary.append((lab, cols[1][-1], cols[0][-1], float("nan")))
with open(os.path.join(out, "nciv_summary.txt"), "w") as s:
    s.write("# label  W_field/K_e at the end  omega_p t at the end  max|dGauss| (VLPL)\n")
    for r in summary:
        s.write("%-24s %.2e %.1f %.1e\n" % r)
print(open(os.path.join(out, "nciv_summary.txt")).read())
