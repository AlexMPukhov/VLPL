# Closed forms of the plasma dispersion (fluid law A) and the polarisation of the modes
import numpy as np
exec(open("disp_fluid.py").read().split("wp2 = float")[0])
rng = np.random.default_rng(5)
worst = 0
for wp2 in (0.0006, 0.05, 0.3):
    for irip in (False, True):
        for t in range(40):
            k = np.array([rng.uniform(-3, 3), rng.uniform(-30, 30), rng.uniform(-30, 30)])
            ev, V = np.linalg.eig(G(k, wp2, irip))
            sx, cx = np.sin(k[0]*D/2), np.cos(k[0]*D/2)
            lam = 4/hy**2*np.sin(k[1]*hy/2)**2 + 4/hz**2*np.sin(k[2]*hz/2)**2
            if irip:
                R = np.tan(k[0]*D/2)**2 + D*D/4*lam; q = D*D/4*wp2
                T = (R + q)/(1 - q); wte = wtm = 2/D*np.arctan(np.sqrt(T))
            else:
                wte = 2/D*np.arcsin(np.sqrt(sx**2 + D*D/4*cx**2*(lam + wp2)))
                wtm = 2/D*np.arcsin(np.sqrt(sx**2 + D*D/4*cx**2*(lam + wp2*(1 - D*D/4*lam))))
            w = np.angle(ev)/(-D)
            for target, cls in ((wte, "TE"), (wtm, "TM")):
                i = np.argmin(abs(w - target)); worst = max(worst, abs(w[i] - target))
                if not irip and abs(wte - wtm) > 1e-9:
                    v = V[:, i]; e = np.abs(v[:3]); b = np.abs(v[3:6])
                    if cls == "TE" and e[0] > 1e-10*e.max(): print("TE with Ex", e)
                    if cls == "TM" and b[0] > 1e-10*b.max(): print("TM with Bx", b)
print("largest |omega - closed form|", worst)
