# energy balance from DiagScalar (scalars.txt) of a Smilei run: python3 energy_balance.py <run_dir>
import sys, numpy as np
from math import pi
fs = 2*pi*0.299792458/0.8
f = f"{sys.argv[1]}/scalars.txt"
cols = [l.split()[2] for l in open(f) if l.startswith("#") and len(l.split()) == 3 and l.split()[1].isdigit()]
a = np.loadtxt(f); c = {n: a[:, i] for i, n in enumerate(cols)}
U = c["Utot"].max()
print(sys.argv[1], "| max Utot %.5e" % U)
for tf in (40, 55, 70, 85, 100, 115):
    i = np.argmin(abs(c["time"]/fs - tf))
    print("  t %6.1f fs  Utot %.5e  Uelm %.5e  Ukin %.5e  Uelm_bnd %+.5e  Ukin_bnd %+.5e  Uexp %.5e  Ubal %+.4e  Ubal/Utot_max %+.2e"
          % (c["time"][i]/fs, c["Utot"][i], c["Uelm"][i], c["Ukin"][i], c["Uelm_bnd"][i], c["Ukin_bnd"][i], c["Uexp"][i], c["Ubal"][i], c["Ubal"][i]/U))
