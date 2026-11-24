"""Prototype (docs/specs/side_injection.md): TFSF planes y = const in the RIP-CC step (vacuum, fields independent of z).

The step is the one of src/fields.cpp (predictor, x transport with the sources at i+1/2, corrector) with
z derivatives dropped. x is a Bloch phase exp(i kx dx) per cell (one plane-wave mode along x, any kx), y is a finite
line of cells with zero guards (far walls). Positions along y: Ex, Ez, By at nodes j; Ey, Bz, Bx at half nodes j+1/2.
The tube: total field at nodes j1..j2 and half nodes j1..j2-1 (j + 1/2), scattered field outside.

Every y difference whose two values lie on different sides of a plane gets the incident value of the neighbour
(+ if the output point is total and the neighbour scattered, - the other way). Incident Ex*, Bx* are the predictor
applied to the incident field; incident <F> in the corrector is the mean of t_n and t_n+1.

Incident fields: (a) the exact discrete plane wave of RIP-CC (eigenvector of the one-step matrix for (kx, ky)),
(b) the continuum Maxwell plane wave, omega = |k|. Leakage = max |field| in the scattered region / amplitude;
error = max |total - incident| in the tube / amplitude.
"""
import numpy as np, sys

def step(F, dx, dy, sx, sy_dn, sy_up, inc=None):
    """One RIP-CC step. F = dict of complex arrays. sx(f, s): f at i+s; sy_dn(f): f at j-1; sy_up(f): f at j+1.
    inc: None or (I0, I1, Tn, Th): incident fields at t_n, t_n+1 and the total masks of nodes / half nodes."""
    D, iy = dx, 1.0 / dy
    Ex, Ey, Ez, Bx, By, Bz = (F[c] for c in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"))

    def nb(f, finc, Tin, Tout, sh):  # neighbour value seen from an output point, with the TFSF correction
        if inc is None: return sh(f)
        return sh(f) + (Tout - sh(Tin)) * sh(finc)

    def pred(Ex, Bx, Ez, Bz, I=None):
        if I is None:   # plain (also used for the incident field)
            exs = Ex + 0.5 * D * (Bz - sy_dn(Bz)) * iy
            bxs = Bx - 0.5 * D * (sy_up(Ez) - Ez) * iy
            return exs, bxs
        I0, Tn, Th = I
        idf = lambda f: f
        exs = Ex + 0.5 * D * (nb(Bz, I0["Bz"], Th, Tn, idf) - nb(Bz, I0["Bz"], Th, Tn, sy_dn)) * iy
        bxs = Bx - 0.5 * D * (nb(Ez, I0["Ez"], Tn, Th, sy_up) - nb(Ez, I0["Ez"], Tn, Th, idf)) * iy
        return exs, bxs

    if inc is None:
        Exs, Bxs = pred(Ex, Bx, Ez, Bz)
    else:
        I0, I1, Tn, Th = inc
        Exs, Bxs = pred(Ex, Bx, Ez, Bz, (I0, Tn, Th))
        iExs, iBxs = pred(I0["Ex"], I0["Bx"], I0["Ez"], I0["Bz"])   # incident predictor

    mux = lambda f, s: 0.5 * (sx(f, s) + sx(f, s + 1))
    def phiz(s):   # at (i+s+1/2, j+1/2): output on half nodes, Exs on nodes
        a = mux(Exs, s)
        if inc is None: return (sy_up(a) - a) * iy
        ai = mux(iExs, s)
        return (nb(a, ai, Tn, Th, sy_up) - (a + (Th - Tn) * ai)) * iy
    def psiz(s):   # at (i+s+1/2, j): output on nodes, Bxs on half nodes
        a = mux(Bxs, s)
        if inc is None: return -(a - sy_dn(a)) * iy
        ai = mux(iBxs, s)
        return -((a + (Tn - Th) * ai) - nb(a, ai, Th, Tn, sy_dn)) * iy

    tpy = sx(Ey, -1) + sx(Bz, -1) + D * phiz(-1)
    tmy = sx(Ey, 1) - sx(Bz, 1) - D * phiz(0)
    tpz = sx(Ez, 1) + sx(By, 1) + D * psiz(0)
    tmz = sx(Ez, -1) - sx(By, -1) + D * psiz(-1)
    nEy, nBz = 0.5 * (tpy + tmy), 0.5 * (tpy - tmy)
    nEz, nBy = 0.5 * (tpz + tmz), 0.5 * (tpz - tmz)

    bz, ez = 0.5 * (Bz + nBz), 0.5 * (Ez + nEz)
    if inc is None:
        nEx = Ex + D * (bz - sy_dn(bz)) * iy
        nBx = Bx - D * (sy_up(ez) - ez) * iy
    else:
        ibz, iez = 0.5 * (I0["Bz"] + I1["Bz"]), 0.5 * (I0["Ez"] + I1["Ez"])
        nEx = Ex + D * ((bz + (Tn - Th) * ibz) - nb(bz, ibz, Th, Tn, sy_dn)) * iy
        nBx = Bx - D * (nb(ez, iez, Tn, Th, sy_up) - (ez + (Th - Tn) * iez)) * iy
    return dict(Ex=nEx, Ey=nEy, Ez=nEz, Bx=nBx, By=nBy, Bz=nBz)

COMPS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
HALF = {"Ex": 0, "Ez": 0, "By": 0, "Ey": 0.5, "Bz": 0.5, "Bx": 0.5}

def discrete_mode(kx, ky, dx, dy, pol):
    """Forward eigenmode of the one-step matrix (Bloch in x and y) in the subspace of the polarisation.
    The amplitudes refer to the stored index j (the half-node offset is in the phase of v)."""
    px, py = np.exp(1j * kx * dx), np.exp(1j * ky * dy)
    sx = lambda f, s: f * px ** s
    keep = ["Ex", "Ey", "Bz"] if pol == "p" else ["Ez", "Bx", "By"]
    G = np.zeros((3, 3), complex)
    for b, c in enumerate(keep):
        F = {q: np.zeros(1, complex) for q in COMPS}
        F[c][:] = 1
        out = step(F, dx, dy, sx, lambda f: f / py, lambda f: f * py)
        for a, q in enumerate(keep): G[a, b] = out[q][0]
        assert max(abs(out[q][0]) for q in COMPS if q not in keep) < 1e-14
    lam, V = np.linalg.eig(G)
    w = 1j * np.log(lam) / dx       # lam = exp(-i w dt)
    k = np.hypot(kx, ky)
    n = int(np.argmin(np.abs(w - k)))
    v = V[:, n]
    e = v[keep.index("Ez")] if pol == "s" else v[keep.index("Bz")]
    v = v / e
    out = {q: 0j for q in COMPS}
    out.update(dict(zip(keep, v)))
    return w[n], out

def continuum_mode(kx, ky, pol):
    k = np.hypot(kx, ky); c, s = kx / k, ky / k
    if pol == "p": v = dict(Ex=-s, Ey=c, Ez=0, Bx=0, By=0, Bz=1)
    else:          v = dict(Ex=0, Ey=0, Ez=1, Bx=s, By=-c, Bz=0)
    return k, {q: complex(v[q]) for q in COMPS}

def run(theta_deg, lam_dx, lam_dy, pol, kind, cone=False, nsteps=None, W=None, Tw=None):
    lam = 1.0; dx, dy = lam / lam_dx, lam / lam_dy
    assert (dx / dy) ** 2 < 1
    th = np.radians(theta_deg); k = 2 * np.pi / lam
    kx, ky = k * np.cos(th), k * np.sin(th)
    W = W or int(round(6 * lam / dy)); Tw = Tw or int(round(4 * lam / dy))   # scattered width, tube width (cells)
    ny = 2 * W + Tw; j1, j2 = W, W + Tw
    j = np.arange(ny)
    Tn = ((j >= j1) & (j <= j2)).astype(float); Th = ((j >= j1) & (j <= j2 - 1)).astype(float)
    nsteps = nsteps or int(1.5 * W * dy / dx)   # outgoing leakage reaches the wall at ~W dy / dt steps
    modes = []
    for sgn in ((1, -1) if cone else (1,)):
        w, v = (discrete_mode if kind == "discrete" else lambda a, b, c, d, p: continuum_mode(a, b, p))(kx, sgn * ky, dx, dy, pol)
        modes.append((w, sgn * ky, v))
    def incident(n):
        I = {}
        for q in COMPS:
            y = (j + (HALF[q] if kind == "continuum" else 0) - 0.5 * (j1 + j2)) * dy
            I[q] = sum(v[q] * np.exp(1j * (kyy * y - w * n * dx)) for w, kyy, v in modes)
        return I
    px = np.exp(1j * kx * dx)
    sx = lambda f, s: f * px ** s
    def sy_dn(f): g = np.zeros_like(f); g[1:] = f[:-1]; return g
    def sy_up(f): g = np.zeros_like(f); g[:-1] = f[1:]; return g
    I0 = incident(0)
    F = {q: I0[q] * (Tn if HALF[q] == 0 else Th) for q in COMPS}
    amp = max(np.abs(I0[q]).max() for q in COMPS)
    leak = err = 0.0
    lin = lout = ent = 0.0
    lo = (j < j1).astype(float); up = 1 - lo
    near = ((j >= j1) & (j < j1 + int(round(0.5 * lam / dy)))).astype(float)   # half a wavelength behind the entry
    for n in range(nsteps):
        I1 = incident(n + 1)
        F = step(F, dx, dy, sx, sy_dn, sy_up, (I0, I1, Tn, Th))
        I0 = I1
        for q in COMPS:
            T = Tn if HALF[q] == 0 else Th
            leak = max(leak, np.abs(F[q] * (1 - T)).max())
            lin = max(lin, np.abs(F[q] * (1 - T) * lo).max()); lout = max(lout, np.abs(F[q] * (1 - T) * up).max())
            ent = max(ent, np.abs((F[q] - I0[q]) * T * near).max())
            err = max(err, np.abs((F[q] - I0[q]) * T).max())
    return leak / amp, err / amp, nsteps, lin / amp, lout / amp, ent / amp

if __name__ == "__main__":
    print("# (1) exact discrete RIP-CC plane wave(s), dx = lambda/32: leakage and total - incident, rel. to the amplitude")
    print("# angle to x, lambda/dy, pol, one wave or two at +-angle (cone)")
    for cone in (False, True):
        for th in (5, 10, 20, 45, 60, 75, 90):
            for ldy in (8, 16, 24):
                for pol in ("p", "s"):
                    if cone and th == 90: continue
                    l, e = run(th, 32, ldy, pol, "discrete", cone)[:2]
                    print(f"{th:3d} {ldy:3d} {pol} {'cone' if cone else 'one ':4s}  leak {l:9.2e}  err {e:9.2e}")
    print()
    print("# (2) continuum plane wave entering through the lower plane: leakage at the entry side, at the exit side,")
    print("# total - incident within lambda/2 of the entry, and in the whole tube (4 lambda wide)")
    for ldx, ldys in ((32, (8, 16, 24)), (64, (32,)), (128, (32,))):
        for th in (5, 10, 20, 45, 60, 75, 90):
            for ldy in ldys:
                l, e, n, li, lo_, en = run(th, ldx, ldy, "p", "continuum")
                print(f"{th:3d} lambda/dx={ldx:3d} lambda/dy={ldy:3d}  entry-side {li:9.2e}  exit-side {lo_:9.2e}  "
                      f"near-entry err {en:9.2e}  tube err {e:9.2e}")
