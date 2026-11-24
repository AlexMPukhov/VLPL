# One-step map of RIP-CC / iRIP with a linear cold-fluid current (stored-index Bloch mode), 9 x 9:
# J^{n+1/2} = J^{n-1/2} + D wp2 (Ex, mux Ey, mux Ez)^n; then the field step of src/fields.cpp with J^{n+1/2}.
import numpy as np, sys
D, hy, hz = 0.05, 0.1, 0.1
def step(k, v, wp2, irip):
    I = 1j; a = np.exp(I*k[0]*D)
    dyp = (np.exp(I*k[1]*hy)-1)/hy; dym = (1-np.exp(-I*k[1]*hy))/hy
    dzp = (np.exp(I*k[2]*hz)-1)/hz; dzm = (1-np.exp(-I*k[2]*hz))/hz
    m = 0.5*(1+a)
    E, B, J = v[0:3], v[3:6], v[6:9]
    Jn = J + D*wp2*np.array([E[0], m*E[1], m*E[2]])
    def pas(Exs, Bxs):
        o = np.zeros(6, complex)
        psiy = dzm*m*Bxs - Jn[1]; psiz = -dym*m*Bxs - Jn[2]; phiy = -dzp*m*Exs; phiz = dyp*m*Exs
        tpy = (E[1]+B[2]+D*(psiy+phiz))/a; tmy = a*(E[1]-B[2]) + D*(psiy-phiz)
        tpz = a*(E[2]+B[1]) + D*(psiz+phiy); tmz = (E[2]-B[1]+D*(psiz-phiy))/a
        o[1] = .5*(tpy+tmy); o[5] = .5*(tpy-tmy); o[2] = .5*(tpz+tmz); o[4] = .5*(tpz-tmz)
        bz = .5*(B[2]+o[5]); by = .5*(B[1]+o[4]); ez = .5*(E[2]+o[2]); ey = .5*(E[1]+o[1])
        o[0] = E[0] + D*(dym*bz - dzm*by - Jn[0]); o[3] = B[0] - D*(dyp*ez - dzp*ey)
        return o
    if not irip:
        pE = E[0] + .5*D*(dym*B[2] - dzm*B[1] - Jn[0]); pB = B[0] - .5*D*(dyp*E[2] - dzp*E[1])
        o = pas(pE, pB)
    else:
        r0 = pas(.5*E[0], .5*B[0]); rx = pas(.5*(E[0]+1), .5*B[0]); ry = pas(.5*E[0], .5*(B[0]+1))
        A = np.array([[1-(rx[0]-r0[0]), -(ry[0]-r0[0])], [-(rx[3]-r0[3]), 1-(ry[3]-r0[3])]])
        X, Y = np.linalg.solve(A, [r0[0], r0[3]])
        o = pas(.5*(E[0]+X), .5*(B[0]+Y))
    return np.concatenate([o, Jn])
def G(k, wp2, irip):
    return np.array([step(k, np.eye(9)[b], wp2, irip) for b in range(9)]).T
wp2 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
rng = np.random.default_rng(1)
for irip in (False, True):
    print("irip" if irip else "rip_cc")
    for t in range(6):
        k = np.array([rng.uniform(0.3, 3), rng.uniform(-8, 8), rng.uniform(-8, 8)])
        ev = np.linalg.eigvals(G(k, wp2, irip))
        w = np.sort(np.angle(ev)/(-D))
        sx, cx = np.sin(k[0]*D/2), np.cos(k[0]*D/2)
        lam = 4/hy**2*np.sin(k[1]*hy/2)**2 + 4/hz**2*np.sin(k[2]*hz/2)**2
        print(" k", np.round(k, 3), "|ev|-1", np.round(np.abs(ev)-1, 13).max(), "omega", np.round(w, 9))
