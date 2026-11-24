#!/usr/bin/env python3
"""CHF benchmark decks (bench/chf/README.md): the target run and its vacuum twin for one resolution.

  python3 bench/chf/series.py 2d R DIR            # 2D, dx = lambda0 / R, dy = 1.1 dx (RIP-CC)
  python3 bench/chf/series.py rz R DIR [rr M]     # RZ, dz = lambda0 / R, dr = rr dz (default 3.2), M modes (4), RIP-CC
  python3 bench/chf/series.py 3d R DIR            # 3D, dx = lambda0 / R, dy = dz = 1.55 dx (RIP-CC 3D: dx^2 (2 / dy^2) +
                                                  # omega_p^2 dx^2 / 4 = 0.93 at 40 n_c, lambda0/64), round paraboloid

writes DIR/t<R>/chf.deck (target) and DIR/v<R>/chf.deck (vacuum twin: density_function = 0).
Physics (approved by Alexander 2026-11-03): a0 = 10, 5 fs, waist 2 um, lambda0 = 0.8 um, normal incidence onto a
40 n_c parabolic mirror (focal length F = 3 um, vertex at z = 6 um (RZ) / x = 5 um (2D), sag r^2 / 4F), exponential
pre-plasma 0.04 um cut at 0.2 um, 0.8 um thick, fixed ions; bands 4.5 .. 10.5 and 10.5 .. 20.5. Both geometries: the laser
through a TFSF plane, absorbing layers at both ends of the transport axis (RZ: 1.6 um; 2D: 1 um).
"""
import os
import sys

LAM = 0.8
COMMON_SPECIES = """species electrons {{
  charge  = -1
  mass    = 1
  density = 40
  density_function = {f}
{ppc}}}
species ions {{
  charge  = 1
  mass    = 3672
  density = 40
  density_function = {f}
{ppc}  mobile  = false
}}
"""
BANDS = """harmonic_focus low {{
  n_min = 4.5
  n_max = 10.5
{reg}}}
harmonic_focus high {{
  n_min = 10.5
  n_max = 20.5
{reg}}}
output {{
  dir   = out
  every = {every}
  xflux = true
}}
"""
T2D = """# CHF benchmark, 2D (bench/chf/README.md), lambda0 / {R}: a0 = 10, 5 fs, waist 2 um through a TFSF plane onto a 40 n_c
# parabolic mirror (F = 3 um, vertex x = 5 um); the harmonics focus near x = 2 um.
grid {{
  geometry   = cartesian2d
  lambda0    = 0.8 um
  nx         = {nx}
  ny         = {ny}
  dx         = {dx} um
  dy         = {dy} um
  boundary_x = absorbing
  x_absorber_width = 1 um
  t_end      = 52 fs
}}
numerics {{
  shape         = 2
  filter_passes = 1
  filter        = local     # the filter of the whole series (decisions 54, 64; the projection: 69, 70)
}}
laser {{
  a0            = 10
  polarization  = y
  duration      = 5 fs
  delay         = 13 fs
  waist         = 2 um
  focus         = 3.8 um
  tfsf_x        = 1.2 um
  tfsf_incident = modes
  y_center      = {yc} um
}}
"""
T3D = """# CHF benchmark, 3D (bench/chf/README.md), dx = lambda0 / {R}, dy = dz = 1.55 dx: a0 = 10, 5 fs, round waist 2 um through a
# TFSF plane at x = 1.2 um onto a 40 n_c paraboloid (F = 3 um, vertex x = 5 um); the harmonics focus near x = 2 um.
grid {{
  lambda0    = 0.8 um
  nx         = {nx}
  ny         = {ny}
  nz         = {ny}
  dx         = {dx} um
  dy         = {dy} um
  dz         = {dy} um
  boundary_x = absorbing
  x_absorber_width = 1 um
  t_end      = 52 fs
}}
numerics {{
  shape         = 2
  filter_passes = 1
  filter        = local     # the filter of the whole series (decisions 54, 64; the projection: 69, 70)
}}
laser {{
  a0            = 10
  polarization  = y
  duration      = 5 fs
  delay         = 13 fs
  waist         = 2 um
  waist_z       = 2 um
  focus         = 3.8 um
  tfsf_x        = 1.2 um
  tfsf_incident = modes
  y_center      = {yc} um
  z_center      = {yc} um
}}
"""
TRZ = """# CHF benchmark, RZ (bench/chf/README.md), dz = lambda0 / {R}, dr = {rr} dz, {M} modes: a0 = 10, 5 fs, waist 2 um,
# polarised along x, through a TFSF plane at z = 1.8 um (z layers of 1.6 um at both ends) onto a 40 n_c paraboloid
# (F = 3 um, vertex z = 6 um); the harmonics focus near z = 3 um.
grid {{
  geometry   = rz
  modes      = {M}
  lambda0    = 0.8 um
  nz         = {nz}
  nr         = {nr}
  dz         = {dz} um
  dr         = {dr} um
  boundary_z = absorbing
  z_absorber_width = 1.6 um
  boundary_r = absorbing
  r_absorber_cells = {nabs}
  t_end      = 60 fs
}}
numerics {{
  solver        = rip_cc
  rz_gather     = matched   # dense plasma at the axis (decision 80; the default is plain since decision 106)
  shape         = 2
  filter_passes = 1
  filter        = local     # the filter of the whole series (decisions 54, 64; the projection: 69, 70)
}}
laser {{
  a0            = 10
  polarization  = x
  duration      = 5 fs
  delay         = 10 fs
  waist         = 2 um
  focus         = 6 um
  tfsf_x        = 1.8 um
}}
"""


def main():
    geo, R, base = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    for kind in "tv":
        d = os.path.join(base, kind + str(R))
        os.makedirs(d, exist_ok=True)
        if geo == "2d":
            dx = LAM / R
            dy = 1.1 * dx
            ny = 2 * round(8.0 / dy / 2)
            yc = ny // 2 * dy
            s = "(y - %.10g um)^2 / (12 um)" % yc
            f = "x >= 5 um - {s} - 0.2 um && x < 5.8 um - {s} ? min(1, exp((x - 5 um + {s}) / (0.04 um))) : 0".format(s=s)
            text = T2D.format(R=R, nx=round(8.0 / dx), ny=ny, dx="%.10g" % dx, dy="%.10g" % dy, yc="%.10g" % yc)
            ppc = "  ppc_x   = 2\n  ppc_y   = 2\n"
            reg = "  xmin  = 1.3 um\n  xmax  = 3.7 um\n  ymin  = %.6g um\n  ymax  = %.6g um\n" % (yc - 1.5, yc + 1.5)
        elif geo == "3d":
            dx = LAM / R
            dy = 1.55 * dx
            ny = 2 * round(8.0 / dy / 2)
            yc = ny // 2 * dy
            s = "((y - {c:.10g} um)^2 + (z - {c:.10g} um)^2) / (12 um)".format(c=yc)
            f = "x >= 5 um - {s} - 0.2 um && x < 5.8 um - {s} ? min(1, exp((x - 5 um + {s}) / (0.04 um))) : 0".format(s=s)
            text = T3D.format(R=R, nx=round(8.0 / dx), ny=ny, dx="%.10g" % dx, dy="%.10g" % dy, yc="%.10g" % yc)
            ppc = "  ppc_x   = 2\n  ppc_y   = 2\n  ppc_z   = 2\n"
            reg = ("  xmin  = 1.3 um\n  xmax  = 3.7 um\n  ymin  = %.6g um\n  ymax  = %.6g um\n  zmin  = %.6g um\n  zmax  = %.6g um\n"
                   % (yc - 1.5, yc + 1.5, yc - 1.5, yc + 1.5))
        else:
            rr = float(sys.argv[4]) if len(sys.argv) > 4 else 3.2
            M = int(sys.argv[5]) if len(sys.argv) > 5 else 4
            dz = LAM / R
            dr = rr * dz
            s = "r^2 / (12 um)"
            f = "z >= 6 um - {s} - 0.2 um && z < 6.8 um - {s} ? min(1, exp((z - 6 um + {s}) / (0.04 um))) : 0".format(s=s)
            text = TRZ.format(R=R, rr=rr, M=M, nz=int(round(9.0 / dz)), nr=int(round(4.5 / dr)), dz="%.10g" % dz,
                              dr="%.10g" % dr, nabs=max(8, int(round(0.5 / dr))))
            ppc = "  ppc_z   = 2\n  ppc_r   = 2\n  ppc_theta = %d\n" % (2 * M + 2)
            reg = "  zmin  = 2.0 um\n  zmax  = 4.7 um\n  rmax  = 1.5 um\n"
        text += COMMON_SPECIES.format(f=f if kind == "t" else "0", ppc=ppc) + BANDS.format(reg=reg, every=2 * R)
        open(os.path.join(d, "chf.deck"), "w").write(text)
        print(os.path.join(d, "chf.deck"))


if __name__ == "__main__":
    main()
