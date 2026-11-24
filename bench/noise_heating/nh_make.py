# Noise heating reproducer: a uniform hot electron plasma (fixed ions), no laser, periodic z (RZ) / periodic x, y (2D).
# usage: python3 nh_make.py <dir> key=value ...   (geom=rz|2d, solver, closure, shape, fp, filt, ppc=z,r,theta, modes, n, uth,
#        dz, dr, nz, nr, t)
import os, sys
P = dict(geom="rz", solver="rip_cc", closure="box", shape="2", fp="1", filt="projection", ppc="1,2,8", modes="4", n="1",
         uth="1", dz="0.00625", dr="0.02", nz="128", nr="96", t="40", rmax="1.6", mobile_ions="false", pair="0", R="3.6", gm="plain", ts="", sw="")
d = sys.argv[1]
for a in sys.argv[2:]:
    k, v = a.split("=", 1)
    assert k in P, k
    P[k] = v
pz, pr, pt = P["ppc"].split(",")
P["nr"] = str(int(round(float(P["R"]) / float(P["dr"]))))
os.makedirs(d, exist_ok=True)
num = "numerics {\n  solver = %s\n  shape = %s\n  filter_passes = %s\n  filter = %s\n" % (P["solver"], P["shape"], P["fp"], P["filt"])
if P["ts"]: num += "  rz_transverse_shape = %s\n" % P["ts"]
if P["sw"]: num += "  rz_shape_width = %s\n" % P["sw"]
if P["gm"] != "plain": num += "  rz_gather = %s\n" % P["gm"]
if P["closure"] != "box": num += "  closure = %s\n" % P["closure"]
num += "}\n"
if P["geom"] == "rz":
    grid = """grid {
  geometry = rz
  modes = %s
  lambda0 = 0.8 um
  nz = %s
  nr = %s
  dz = %s um
  dr = %s um
  boundary_z = periodic
  t_end = %s fs
}
""" % (P["modes"], P["nz"], P["nr"], P["dz"], P["dr"], P["t"])
    sp = lambda name, q, m, mob, th: """species %s {
  charge = %s
  mass = %s
  density = %s
  rmax = %s um
  ppc_z = %s
  ppc_r = %s
  ppc_theta = %s
  mobile = %s
  uth_z = %s
  uth_x = %s
  uth_y = %s
}
""" % (name, q, m, P["n"], P["rmax"], pz, pr, pt, mob, th, th, th)
else:
    # 2D: x = z of RZ (periodic), y = r: the same cells; y periodic over the plasma width 2 rmax (no losses)
    ny = int(round(2 * float(P["R"]) / float(P["dr"])))
    y0, y1 = float(P["R"]) - float(P["rmax"]), float(P["R"]) + float(P["rmax"])
    grid = """grid {
  geometry = cartesian2d
  lambda0 = 0.8 um
  nx = %s
  ny = %d
  dx = %s um
  dy = %s um
  boundary_x = periodic
  t_end = %s fs
}
""" % (P["nz"], ny, P["dz"], P["dr"], P["t"])
    sp = lambda name, q, m, mob, th: """species %s {
  charge = %s
  mass = %s
  density = %s
  ymin = %g um
  ymax = %g um
  ppc_x = %s
  ppc_y = %s
  mobile = %s
  uth_x = %s
  uth_y = %s
  uth_z = %s
}
""" % (name, q, m, P["n"], y0, y1, pz, int(pr) * int(pt), mob, th, th, th)
out = "output {\n  dir = out\n  every = 50\n  xflux = true\n}\n"
if P["pair"] == "1": deck = grid + num + sp("electrons", -1, 1, "true", P["uth"]) + sp("positrons", 1, 1, "true", P["uth"]) + out
else: deck = grid + num + sp("electrons", -1, 1, "true", P["uth"]) + sp("ions", 1, 1836, P["mobile_ions"], 0) + out
open(os.path.join(d, "run.deck"), "w").write(deck)
open(os.path.join(d, "params.txt"), "w").write(repr(P) + "\n")
