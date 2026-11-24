# Rung 1: Fresnel reflection off a 40 n_c step (fixed ions, a0 = 1e-3), 2D-like Cart3D, rip_cc / irip / irip+o5,
# dx = lambda0/64, /128. Decks after tests/test_xflux.cpp fresnel_deck.
import math, os
root = "$WORK/chfgap/r1"
os.makedirs(root, exist_ok=True)
solvers = {"cc": "solver = rip_cc", "ir": "solver = irip", "o5": "solver = irip\n closure = o5"}
cases = [(0, "z", 0.0), (30, "z", 0.0), (30, "y", 0.5)]
jobs = []
for res in (64, 128):
    dx = 0.8 / res
    for ang, pol, sh in cases:
        s = math.sin(math.radians(ang))
        ny = 32 if s > 0 else 4
        dy = 0.8 / s / ny if s > 0 else 0.1
        xs, xp = 10.8, 10.3
        nx = 2 * round(xs / dx) + 1
        for sk, sv in solvers.items():
            for n in (40, 0):
                name = "%s_%d_%s%d_%s" % ("p" if n else "v", res, pol, ang, sk)
                d = os.path.join(root, name)
                os.makedirs(d, exist_ok=True)
                dk = f"""grid {{
 lambda0 = 0.8 um
 nx = {nx}
 ny = {ny}
 nz = 4
 dx = {dx:.10g} um
 dy = {dy:.10g} um
 dz = 0.1 um
 boundary_x = absorbing
 t_end = 130 fs
}}
numerics {{
 {sv}
}}
laser {{
 a0 = 0.001
 polarization = {pol}
 duration = 12 fs
 delay = 24 fs
 waist = 0 um
 tfsf_x = {xp} um
 angle = {ang} deg
}}
species electrons {{
 charge = -1
 mass = 1
 density = {n}
 xmin = {xs + sh * dx:.10g} um
 xmax = {xs + 1.5:.10g} um
 ppc_x = 2
}}
species ions {{
 charge = 1
 mass = 1836
 mobile = false
 density = {n}
 xmin = {xs + sh * dx:.10g} um
 xmax = {xs + 1.5:.10g} um
 ppc_x = 2
}}
output {{
 dir = out
 xflux_every = 1
}}
"""
                open(os.path.join(d, "run.deck"), "w").write(dk)
                jobs.append(d)
open(os.path.join(root, "jobs.txt"), "w").write("\n".join(jobs) + "\n")
open(os.path.join(root, "run.sbatch"), "w").write("""#!/bin/bash
#SBATCH -p epyc
#SBATCH -c 32
#SBATCH -t 4:00:00
#SBATCH -J r1fresnel
export OMP_NUM_THREADS=4 OMP_PROC_BIND=false
cat $WORK/chfgap/r1/jobs.txt | xargs -P 8 -I{} sh -c 'cd {} && $WORK/chfgap/rippic run.deck > run.log 2>&1; echo done {}'
""")
print(len(jobs), "runs")
