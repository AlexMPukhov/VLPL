import sys, os, math, cmath, numpy as np
sys.path.insert(0, "$WORK/rip-pic-hhgang/tools")
import xflux
root = "$WORK/chfgap/r1"


def amp(d, edge, comp, ky):
    t, T, m = xflux.load(os.path.join(d, "out", "xflux"), edge)
    h = [float(x) for x in m["h"]]
    ny = T.shape[3]
    ph = np.exp(-1j * ky * np.arange(ny) * h[1])
    s = (T[:, comp] * ph[None, None, :]).mean(axis=(1, 2))
    return (s * np.exp(1j * t)).sum()


def fresnel(ang, pol, n):
    c, s, eps = math.cos(math.radians(ang)), math.sin(math.radians(ang)), 1 - n
    k = cmath.sqrt(eps - s * s)
    return (c - k) / (c + k) if pol == "z" else (k - eps * c) / (k + eps * c)


print("case            solver  |r|-|rF|    arg(r/rF) rad")
for res in (64, 128):
    dx = 0.8 / res
    for ang, pol, sh in ((0, "z", 0.0), (30, "z", 0.0), (30, "y", 0.5)):
        for sk in ("cc", "ir", "o5"):
            tag = "%d_%s%d_%s" % (res, pol, ang, sk)
            try:
                comp = 0 if pol == "y" else 1
                ky = math.sin(math.radians(ang))
                r = amp(os.path.join(root, "p_" + tag), "left", comp, ky) / amp(os.path.join(root, "v_" + tag), "right", comp, ky)
                k0dx = 2 * math.pi / 0.8 * dx   # dx in code units (1/k0)
                r *= cmath.exp(-2j * math.cos(math.radians(ang)) * sh * k0dx)
                f = fresnel(ang, pol, 40)
                print("lam/%-4d %s%-3d %s  %+.2e  %+.3e" % (res, pol, ang, sk, abs(r) - abs(f), cmath.phase(r / f)))
            except SystemExit as e:
                print(tag, "missing", e)
            except Exception as e:
                print(tag, "error", e)
