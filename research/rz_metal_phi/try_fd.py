import numpy as np
from xsection import *
print("square analytic TM11 %.5f TE10 %.5f" % (np.sqrt(2) * np.pi / 1.2, np.pi / 1.2))
for M in (4, 8, 16, 32):
    for nr in (40,):
        print("square FD M=%2d: TM %s TE %s" % (M, discrete_fd(square_metal, nr, R / nr, M, "TM", 3), discrete_fd(square_metal, nr, R / nr, M, "TE", 3)), flush=True)
for M in (4, 8, 16, 32):
    print("ridge FD M=%2d: TM %s TE %s" % (M, discrete_fd(ridge_metal, 40, R / 40, M, "TM", 3), discrete_fd(ridge_metal, 40, R / 40, M, "TE", 3)), flush=True)
