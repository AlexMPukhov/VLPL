import numpy as np
rng = np.random.default_rng(1)
def G_of(kx, ky, kz, D, hy, hz, irip=False):
    a = np.exp(1j*kx*D)
    dyp = (np.exp(1j*ky*hy)-1)/hy; dym = (1-np.exp(-1j*ky*hy))/hy
    dzp = (np.exp(1j*kz*hz)-1)/hz; dzm = (1-np.exp(-1j*kz*hz))/hz
    m = (1+a)/2
    def passes(v, Xs=None):
        Ex, Ey, Ez, Bx, By, Bz = v
        if Xs is None:
            Exs = Ex + D/2*(dym*Bz - dzm*By); Bxs = Bx - D/2*(dyp*Ez - dzp*Ey)
        else: Exs, Bxs = Xs
        psiy = dzm*m*Bxs; psiz = -dym*m*Bxs; phiy = -dzp*m*Exs; phiz = dyp*m*Exs
        tpy = (Ey+Bz + D*(psiy+phiz))/a; tmy = a*(Ey-Bz) + D*(psiy-phiz)
        tpz = a*(Ez+By) + D*(psiz+phiy); tmz = (Ez-By + D*(psiz-phiy))/a
        nEy, nBz = (tpy+tmy)/2, (tpy-tmy)/2; nEz, nBy = (tpz+tmz)/2, (tpz-tmz)/2
        bz, by, ez, ey = (Bz+nBz)/2, (By+nBy)/2, (Ez+nEz)/2, (Ey+nEy)/2
        nEx = Ex + D*(dym*bz - dzm*by); nBx = Bx - D*(dyp*ez - dzp*ey)
        return np.array([nEx, nEy, nEz, nBx, nBy, nBz])
    def step(v):
        if not irip: return passes(v)
        Ex, Bx = v[0], v[3]
        # X = Ex', Y = Bx' : pass with Exs=(Ex+X)/2 ; fixed point linear in X, Y
        f = lambda X, Y: passes(v, ((Ex+X)/2, (Bx+Y)/2))
        r0 = f(0, 0); rX = f(1, 0) - r0; rY = f(0, 1) - r0
        A = np.array([[1 - rX[0], -rY[0]], [-rX[3], 1 - rY[3]]]); X, Y = np.linalg.solve(A, [r0[0], r0[3]])
        return f(X, Y)
    G = np.zeros((6, 6), complex)
    for b in range(6):
        e = np.zeros(6, complex); e[b] = 1; G[:, b] = step(e)
    return G
for irip in (False, True):
    for t in range(6):
        D = 1/32; hy = D*rng.uniform(1.5, 3); hz = D*rng.uniform(1.5, 3)
        k = 2*np.pi*rng.uniform(0.5, 1.5); u = rng.normal(size=3); u /= np.linalg.norm(u)
        kx, ky, kz = k*u
        G = G_of(kx, ky, kz, D, hy, hz, irip)
        lam = np.linalg.eigvals(G); w = np.sort((1j*np.log(lam)/D).real)
        L = 4/hy**2*np.sin(ky*hy/2)**2 + 4/hz**2*np.sin(kz*hz/2)**2
        wa = 2/D*np.arcsin(np.sqrt(np.sin(kx*D/2)**2 + D*D*L/4*np.cos(kx*D/2)**2+0j)).real
        wb = 2/D*np.arctan(np.sqrt(np.tan(kx*D/2)**2 + D*D*L/4))
        print("irip" if irip else "ripcc", np.round(w, 6), " cand a", round(wa, 6), " cand b", round(wb, 6), " |lam|", np.round(np.abs(lam), 12).max())
