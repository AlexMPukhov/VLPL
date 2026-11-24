# Rung 3 (vacuum part): a converging beam at the 7th-harmonic wavelength (lambda = 0.8/7 um) focused at NA ~0.5 over
# 3 um in RZ on the CHF grid (dz = 0.00625 um, dr = 0.02 um) and refined (x2, x4), RIP-CC / iRIP / iRIP+O5. No particles.
import os
root = "$WORK/chfgap/r3"
os.makedirs(root, exist_ok=True)
lam = 0.8 / 7
solvers = {"cc": "solver = rip_cc", "ir": "solver = irip", "o5": "solver = irip\n  closure = o5"}
# (tag, dz factor, dr factor)
grids = [("g1", 1, 1), ("g2", 2, 2), ("g4", 4, 4), ("r4", 1, 4)]
jobs = []
for gt, fz, fr in grids:
    dz, dr = 0.00625 / fz, 0.02 / fr
    nz = round(4.4 / dz)
    nr = round(3.0 / dr) + 25 * fr
    for sk, sv in solvers.items():
        if gt == "g4" and sk == "o5":
            pass
        name = "%s_%s" % (gt, sk)
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        dk = f"""grid {{
  geometry   = rz
  modes      = 2
  lambda0    = {lam:.10g} um
  nz         = {nz}
  nr         = {nr}
  dz         = {dz:.10g} um
  dr         = {dr:.10g} um
  boundary_z = open
  boundary_r = absorbing
  r_absorber_cells = {25 * fr}
  t_end      = 28 fs
}}
numerics {{
  {sv}
}}
laser {{
  a0            = 1
  polarization  = y
  duration      = 3 fs
  delay         = 6 fs
  waist         = {lam / (3.141592653589793 * 0.5):.10g} um
  focus         = 3.0 um
}}
harmonic_focus h {{
  n_min = 0.5
  n_max = 1.5
  zmin  = 2.2 um
  zmax  = 3.8 um
  rmax  = 0.6 um
}}
output {{
  dir   = out
  every = 100000
}}
"""
        open(os.path.join(d, "run.deck"), "w").write(dk)
        jobs.append(d)
open(os.path.join(root, "jobs.txt"), "w").write("\n".join(jobs) + "\n")
open(os.path.join(root, "run.sbatch"), "w").write("""#!/bin/bash
#SBATCH -p epyc
#SBATCH -c 32
#SBATCH -t 4:00:00
#SBATCH -J r3focus
export OMP_NUM_THREADS=8 OMP_PROC_BIND=false
cat $WORK/chfgap/r3/jobs.txt | xargs -P 4 -I{} sh -c 'cd {} && $WORK/chfgap/rippic run.deck > run.log 2>&1; echo done {}'
""")
print(len(jobs), "runs")
