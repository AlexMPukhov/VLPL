"""O6 prototype in RZ, mode m = 0, TM fields (Ez, Er, Bphi): the box-iRIP step on the finite-volume radial grid of
decision 52 (uniform here), periodic z, conducting wall at r = nr dr; Cartesian particles (z, x, y) as in the code
(decision 44); charge-conserving 2D Esirkepov in (z, xi(r)) with the shape-2 density-conserving (Ruyten) radial map of
decision 53, folded across the axis at charge level (decision 50).

Movers: explicit leapfrog (Boris; E^n, B^n gathered with the plain splines in r, or the deposit's map: rz_gather = plain |
matched), and the energy-conserving implicit mover of pic2d.py: the work of the work field Et along the path with the
Esirkepov weights (here in the mapped coordinate), force q (Ebar_z z + Ebar_r r_mid), Ebar_r = work_r / (r_mid . dx_perp).
TE (Bz, Br, Ephi) and J_phi are left out: the m = 0 TM subset conserves its own energy W_TM + K exactly.

Staggering (z = transport axis): Ez, Jz (i, j); Er, Bphi (i, j+1/2); rho (i+1/2, j); Jr (i+1/2, j+1/2); r_j = j dr.
"""
import numpy as np
from pic2d import bspline

OFF = np.arange(-2, 3)
G = 4                          # ghost cells (negative radial indices) of the extended arrays

# ---------------- shape-2 Ruyten coefficients (port of src/rz_ruyten.cpp, uniform grid) ----------------
_gx = np.array([0.0950125098376374, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
                0.7554044083550030, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499])
_gw = np.array([0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
                0.1246289712555339, 0.0951585116824928, 0.0622535239024133, 0.0271524594117541])
GX = np.concatenate([-_gx[::-1], _gx]); GW = np.concatenate([_gw[::-1], _gw])


C1_S0 = [3.3]           # axis slope of the C1 map ("c1", c1map.py; monotone for s0 >= 2.8)
AXIS = ["pow"]          # axis-cell map: "pow" xi = u^p (decision 53), "exp" xi = (1 - e^(-k u)) / (1 - e^(-k)) (bounded slope)


def axis0(u, k):
    if AXIS[0] == "exp":
        return (1 - np.exp(-k * u)) / (1 - np.exp(-k))
    return u ** k


def herm(c, u, s0, s1):
    """C1 map: cubic Hermite with xi(r_c) = c and slopes s0, s1 (in cells per cell) at the cell ends"""
    return c + u + (s0 - 1) * (u ** 3 - 2 * u ** 2 + u) + (s1 - 1) * (u ** 3 - u ** 2)


def xi2(b, c, u):
    if AXIS[0] == "hermite":
        return herm(c, u, b[c], b[c + 1])
    return axis0(u, b[0]) if c == 0 else c + u + b[c] * u * (1 - u)


def u_of(b, c, x):
    lo, hi = 0.0, 1.0
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if xi2(b, c, m) < x: lo = m
        else: hi = m
    return 0.5 * (lo + hi)


def share2(dr, b, j):
    tot = 0.0
    for c in range(j - 2, j + 2):
        if c < 0 or c + 1 >= len(b): continue
        br = [0.0]
        uu = u_of(b, c, c + 0.5)
        if 0 < uu < 1: br.append(uu)
        br.append(1.0)
        for p in range(len(br) - 1):
            a, e = br[p], br[p + 1]
            u = 0.5 * (a + e) + GX * 0.5 * (e - a)
            xi = np.array([xi2(b, c, v) for v in u])
            W = bspline(xi - j, 2) + (bspline(xi + j, 2) if j > 0 else 0)
            tot += (GW * 0.5 * (e - a) * W * (c * dr + u * dr) * dr).sum()
    return tot


def ruyten2(nr, dr):
    A = np.array([dr * dr / 8] + [j * dr * dr for j in range(1, nr + 4)])
    b = np.zeros(nr + 4); b[0] = 3.0 if AXIS[0] == "exp" else 1 / 3
    for j in range(1, len(b)): b[j] = (dr * dr / 12) / (j * dr * dr / 6 + dr * dr / 12)
    if AXIS[0] == "hermite": b[:] = 1.0
    K = len(b) - 3
    for it in range(30):
        R = np.array([share2(dr, b, j) - A[j] for j in range(K)])
        if np.max(np.abs(R) / A[:K]) < 1e-13: break
        Jm = np.zeros((K, K))
        for c in range(K):
            keep = b[c]; b[c] = keep + 1e-7
            for j in range(max(0, c - 3), min(K, c + 3)):
                Jm[j, c] = (share2(dr, b, j) - A[j] - R[j]) / 1e-7
            b[c] = keep
        b[:K] += np.linalg.solve(Jm, -R)
    return b


class RZ0:
    def __init__(self, nz, nr, D, dr, rmap=True):
        self.nz, self.nr, self.D, self.dr = nz, nr, D, dr
        self.r = np.arange(nr) * dr
        self.rh = (np.arange(nr) + 0.5) * dr
        self.A = np.array([dr * dr / 8] + [j * dr * dr for j in range(1, nr)])          # node areas (per radian)
        self.Bw = self.rh * dr                                                           # half-node weights
        self.f = [np.zeros((nz, nr)) for _ in range(3)]                                  # Ez Er Bphi
        self.J = [np.zeros((nz, nr)) for _ in range(2)]
        if rmap: AXIS[0] = rmap if rmap in ("exp", "hermite", "c1") else "pow"
        if rmap == "c1":                             # C1 map of c1map.py: Hermite cells, slope S0 at the axis, 1 at nodes >= 1, bumps a_c
            from c1map import solve
            self.b2 = solve(C1_S0[0], K=nr + 2)[0]
        else:
            self.b2 = ruyten2(nr, dr) if rmap else None
        # the largest slope dxi/dr * dr of the radial map: fixed segments of < 1 cell in z and in xi for |v| < c (implicit)
        if self.b2 is None: smax = 1.0
        elif AXIS[0] == "c1": smax = C1_S0[0]
        elif AXIS[0] == "exp": k = self.b2[0]; smax = max(k / (1 - np.exp(-k)), 1 + np.abs(self.b2[1:]).max())
        elif AXIS[0] == "hermite":
            uu = np.linspace(0, 1, 201)
            smax = max(np.gradient(herm(c, uu, self.b2[c], self.b2[c + 1]), uu).max() for c in range(len(self.b2) - 1))
        else: smax = None                            # u^p: unbounded slope, no fixed count possible
        self.nseg_fixed = None if smax is None else int(np.ceil(max(1.0, D * smax / dr) / 0.95))
        L = np.zeros((nr, nr))
        for j in range(nr):
            e = np.zeros((1, nr)); e[0, j] = 1
            L[:, j] = self.div(self.grad(e))[0]
        sa = np.sqrt(self.A)
        M = sa[:, None] * L / sa[None, :]
        lam, Q = np.linalg.eigh(0.5 * (M + M.T))
        self.lam, self.Q, self.sa = lam, Q, sa
        kz = 2 * np.pi * np.fft.fftfreq(nz, D)
        self.den = 1 - D ** 2 / 4 * np.cos(kz * D / 2)[:, None] ** 2 * lam[None, :]
        self.sp = []
        self.rho_bg = np.zeros((nz, nr))
        self.ptol = 1e-13
        self.per_segment = True                      # the radial force per path segment (path())
        self.stats = []
        self.npass = 0

    # radial operators: grad nodes -> half nodes (wall Ez(nr) = 0), div half nodes -> nodes
    def grad(self, F):
        Fp = np.concatenate([F[:, 1:], np.zeros((F.shape[0], 1))], 1)
        return (Fp - F) / self.dr

    def div(self, B):
        rb = self.rh * B
        rbm = np.concatenate([np.zeros((B.shape[0], 1)), rb[:, :-1]], 1)
        return (rb - rbm) / self.A

    def _core(self, X, J):
        D = self.D
        Ez, Er, Bp = self.f
        Jz, Jr = J
        mt = 0.5 * (Ez + X)
        dE = self.grad(0.5 * (mt + np.roll(mt, -1, 0)))
        Tp, Tm = Er + Bp, Er - Bp
        Tp1 = np.roll(Tp + D * (dE - Jr), 1, 0)
        Tm1 = np.roll(Tm, -1, 0) + D * (-dE - Jr)
        Er1, Bp1 = 0.5 * (Tp1 + Tm1), 0.5 * (Tp1 - Tm1)
        Ez1 = Ez + D * (self.div(0.5 * (Bp + Bp1)) - Jz)
        Et = [mt, 0.5 * (Tp + np.roll(Tm, -1, 0)) - D ** 2 / 4 * self.grad((np.roll(mt, -1, 0) - mt) / D) - 0.5 * D * Jr]
        return [Ez1, Er1, Bp1], Et

    def field_solve(self, J):
        b = self._core(np.zeros((self.nz, self.nr)), J)[0][0]
        bh = np.fft.fft(b, axis=0) * self.sa[None, :]
        c = (bh @ self.Q) / self.den
        X = np.fft.ifft((c @ self.Q.T) / self.sa[None, :], axis=0).real
        return self._core(X, J)

    def W(self):
        Ez, Er, Bp = self.f
        return np.pi * self.D * ((Ez ** 2 * self.A).sum() + ((Er ** 2 + Bp ** 2) * self.Bw).sum())   # 2 pi x 1/2

    # ---------------- particles ----------------
    def add_species(self, z, x, y, u, q, m):
        self.sp.append(dict(z=z.copy(), x=x.copy(), y=y.copy(), u=u.copy(), q=np.broadcast_to(q, z.shape).copy(), m=m))

    def K(self):
        k = 0.0
        for s in self.sp:
            u2 = (s["u"] ** 2).sum(1)
            k += (s["m"] * u2 / (np.sqrt(1 + u2) + 1)).sum()
        return k

    def xi(self, r):
        """deposit coordinate: the Ruyten map (shape 2), node j at xi = j"""
        if self.b2 is None:
            return r / self.dr
        c = np.minimum((r / self.dr).astype(np.int64), len(self.b2) - 1)
        u = r / self.dr - c
        b = self.b2[c]
        if AXIS[0] == "c1":
            s = np.where(c == 0, C1_S0[0], 1.0)
            return c + u + (s - 1) * (u ** 3 - 2 * u ** 2 + u) + b * u ** 2 * (1 - u) ** 2
        if AXIS[0] == "hermite":
            c2 = np.minimum(c, len(self.b2) - 2)
            return herm(c2, r / self.dr - c2, self.b2[c2], self.b2[c2 + 1])
        return np.where(c == 0, axis0(u, self.b2[0]), c + u + b * u * (1 - u))

    def ext(self, F, odd, half):
        """F (nz, nr) on nodes (half = False) or half nodes, extended to radial indices -G .. nr + G - 1."""
        nz = F.shape[0]
        out = np.zeros((nz, self.nr + 2 * G))
        out[:, G:G + self.nr] = F
        sg = -1.0 if odd else 1.0
        for k in range(1, G + 1):
            src = (k - 1) if half else k            # half: index -k <-> k - 1 ; nodes: -k <-> k
            out[:, G - k] = sg * F[:, src]
        return out

    def rho(self):
        out = np.zeros((self.nz, self.nr))
        for s in self.sp:
            out += self._dep_charge(s["z"], np.hypot(s["x"], s["y"]), s["q"])
        return out

    def _dep_charge(self, z, r, q):
        xz, xr = z / self.D - 0.5, self.xi(r)
        kz = np.floor(xz + 0.5).astype(np.int64)[:, None] + OFF
        kr = np.floor(xr + 0.5).astype(np.int64)[:, None] + OFF
        Sz, Sr = bspline(xz[:, None] - kz, 2), bspline(xr[:, None] - kr, 2)
        idx = (kz[:, :, None] % self.nz) * (self.nr + 2 * G) + (kr[:, None, :] + G)
        w = q[:, None, None] * Sz[:, :, None] * Sr[:, None, :]
        e = np.bincount(idx.ravel(), w.ravel(), self.nz * (self.nr + 2 * G)).reshape(self.nz, -1)
        out = e[:, G:G + self.nr].copy()
        for k in range(1, G + 1):
            out[:, k] += e[:, G - k]
        return out / (2 * np.pi * self.A[None, :] * self.D)

    def gauss(self):
        Ez, Er = self.f[0], self.f[1]
        dv = (np.roll(Ez, -1, 0) - Ez) / self.D + self.div(0.5 * (Er + np.roll(Er, -1, 0)))
        return dv - self.rho() - self.rho_bg

    def neutralise(self):
        self.rho_bg = -self.rho()

    def gather(self, Fe, z, r, oz, half, matched):
        """plain-spline gather of an extended array at (z, r); half: half-node array; matched: in the map coordinate"""
        xz = z / self.D - oz
        if matched and not half:
            xr = self.xi(r)
        elif matched:
            xr = self.xi(r) - 0.5
        else:
            xr = r / self.dr - (0.5 if half else 0.0)
        kz = np.floor(xz + 0.5).astype(np.int64)[:, None] + OFF
        kr = np.floor(xr + 0.5).astype(np.int64)[:, None] + OFF
        Sz, Sr = bspline(xz[:, None] - kz, 2), bspline(xr[:, None] - kr, 2)
        return np.einsum("pa,pb,pab->p", Sz, Sr, Fe[kz[:, :, None] % self.nz, kr[:, None, :] + G])

    def path(self, z0, x0, y0, z1, x1, y1, q, Et=None, dep=None):
        """Esirkepov in (z, xi(r)) over straight Cartesian segments; Et (extended: Etz even on nodes, Etr odd on half
        nodes) -> work per unit charge (z, r); dep -> adds the folded current (extended arrays)."""
        D, dr = self.D, self.dr
        r0, r1 = np.hypot(x0, y0), np.hypot(x1, y1)
        if self.nseg_fixed:          # implicit: a fixed count (a varying one makes J discontinuous in the iterate)
            nseg = self.nseg_fixed
        else:
            nseg = max(1, int(np.ceil(max(np.abs(z1 - z0).max(initial=0) / D, np.abs(r1 - r0).max(initial=0) / dr,
                                          np.hypot(x1 - x0, y1 - y0).max(initial=0) / dr) / 0.9)))
        wz = np.zeros(len(z0)); wr = np.zeros(len(z0))
        # per segment the radial force along the unit vector at the segment midpoint: f = (1/n) sum_s w_s / (rhat_s . dx_s) rhat_s,
        # so f . dx = sum_s w_s exactly (equal straight segments dx_s = dx / n); a path that passes the axis has r(t) not monotone,
        # and one direction for the whole path divides a finite work by a vanishing dr
        fx = np.zeros(len(z0)); fy = np.zeros(len(z0))
        self._seg_etr = None
        for g in range(nseg):
            fa, fb = g / nseg, (g + 1) / nseg
            za, zb = z0 + (z1 - z0) * fa, z0 + (z1 - z0) * fb
            ra = np.hypot(x0 + (x1 - x0) * fa, y0 + (y1 - y0) * fa)
            rb = np.hypot(x0 + (x1 - x0) * fb, y0 + (y1 - y0) * fb)
            xa, xb = za / D - 0.5, zb / D - 0.5
            ea, eb = self.xi(ra), self.xi(rb)
            kz = np.floor(xa + 0.5).astype(np.int64)[:, None] + OFF
            kr = np.floor(ea + 0.5).astype(np.int64)[:, None] + OFF
            Sz0, Sz1 = bspline(xa[:, None] - kz, 2), bspline(xb[:, None] - kz, 2)
            Sr0, Sr1 = bspline(ea[:, None] - kr, 2), bspline(eb[:, None] - kr, 2)
            dSz, dSr = Sz1 - Sz0, Sr1 - Sr0
            Fz = (-np.cumsum(dSz, 1))[:, :, None] * (Sr0 + 0.5 * dSr)[:, None, :]
            Fr = (Sz0 + 0.5 * dSz)[:, :, None] * (-np.cumsum(dSr, 1))[:, None, :]
            iz_f = (kz[:, :, None] + 1) % self.nz
            iz_c = kz[:, :, None] % self.nz
            ir = kr[:, None, :] + G
            if Et is not None:
                wz += D * np.einsum("pab,pab->p", Fz, Et[0][iz_f, ir])
                ws = dr * np.einsum("pab,pab->p", Fr, Et[1][iz_c, ir])
                wr += ws
                mx, my = x0 + (x1 - x0) * 0.5 * (fa + fb), y0 + (y1 - y0) * 0.5 * (fa + fb)
                mr = np.hypot(mx, my); mrs = np.where(mr > 0, mr, 1)
                drs = rb - ra
                ok = np.abs(drs) > 1e-13 * dr
                zm_ = 0.5 * (za + zb)
                lim = self.gather(Et[1], zm_, mr, 0.5, True, True)          # the limit dr_s -> 0 (does no work)
                es = np.where(ok, ws / np.where(ok, drs, 1), lim) * 2 * mr / np.maximum(ra + rb, 1e-300)
                fx += es * mx / mrs / nseg; fy += es * my / mrs / nseg
            if dep is not None:
                n = self.nr + 2 * G
                dep[0] += np.bincount((iz_f * n + ir).ravel(), (q[:, None, None] * Fz).ravel(), self.nz * n)
                dep[1] += np.bincount((iz_c * n + ir).ravel(), (q[:, None, None] * Fr).ravel(), self.nz * n)
        self._fperp = (fx, fy)
        return wz, wr

    def deposit(self, paths):
        n = self.nr + 2 * G
        dep = [np.zeros(self.nz * n), np.zeros(self.nz * n)]
        for (s, z1, x1, y1) in paths:
            self.path(s["z"], s["x"], s["y"], z1, x1, y1, s["q"], dep=dep)
        ez, er = dep[0].reshape(self.nz, n), dep[1].reshape(self.nz, n)
        Jz = ez[:, G:G + self.nr].copy(); Jr = er[:, G:G + self.nr].copy()
        for k in range(1, G + 1):
            Jz[:, k] += ez[:, G - k]             # node -k -> k
            Jr[:, k - 1] -= er[:, G - k]         # face -k + 1/2 -> k - 1/2 (index k - 1), odd
        Jz /= 2 * np.pi * self.A[None, :] * self.D               # q Fz / (dt * face area), dt = D
        Jr /= 2 * np.pi * self.rh[None, :] * self.D * self.D     # q Fr / (dt * rh dz)
        return [Jz, Jr]

    # ---------------- explicit ----------------
    def step_explicit(self, matched=False):
        self.nseg_fixed = None
        Eze = self.ext(self.f[0], False, False)
        Ere = self.ext(self.f[1], True, True)
        Bpe = self.ext(self.f[2], True, True)
        paths = []
        for s in self.sp:
            z, x, y = s["z"], s["x"], s["y"]
            r = np.hypot(x, y)
            rs = np.where(r > 0, r, 1)
            cx, cy = x / rs, y / rs
            ez = self.gather(Eze, z, r, 0, False, matched)
            er = self.gather(Ere, z, r, 0, True, matched)
            bp = self.gather(Bpe, z, r, 0, True, matched)
            E = np.stack([ez, er * cx, er * cy], 1)
            B = np.stack([0 * bp, -bp * cy, bp * cx], 1)
            u = s["u"]; h = 0.5 * (s["q"] / s["m"])[:, None] * self.D
            um = u + h * E
            g = np.sqrt(1 + (um ** 2).sum(1))[:, None]
            t = h * B / g
            up = um + np.cross(um, t)
            u = um + 2 / (1 + (t ** 2).sum(1))[:, None] * np.cross(up, t) + h * E
            v = u / np.sqrt(1 + (u ** 2).sum(1))[:, None]
            s["u"] = u
            paths.append((s, z + self.D * v[:, 0], x + self.D * v[:, 1], y + self.D * v[:, 2]))
        J = self.deposit(paths)
        self.f = self.field_solve(J)[0]
        for (s, z1, x1, y1) in paths:
            s["z"], s["x"], s["y"] = z1 % (self.nz * self.D), x1, y1
        self.J = J

    # ---------------- implicit, energy conserving ----------------
    def wp2(self):
        out = np.zeros((self.nz, self.nr))
        for s in self.sp:
            g = np.sqrt(1 + (s["u"] ** 2).sum(1))
            out += self._dep_charge(s["z"], np.hypot(s["x"], s["y"]), s["q"] ** 2 / s["m"] / g)
        out = np.abs(out)
        return [0.5 * (out + np.roll(out, 1, 0)), 0.5 * (out + np.concatenate([out[:, 1:], out[:, -1:]], 1))]

    def _push_cn(self, s, Et, Bn, Bn1, u1, inner):
        dt = self.D; qm = (s["q"] / s["m"])[:, None]
        u0 = s["u"]; g0 = np.sqrt(1 + (u0 ** 2).sum(1))
        z0, x0, y0 = s["z"], s["x"], s["y"]
        Etz, Etr = self.ext(Et[0], False, False), self.ext(Et[1], True, True)
        Be = 0.5 * (self.ext(Bn, True, True) + self.ext(Bn1, True, True))
        r0 = np.hypot(x0, y0)
        for _ in range(inner):
            self.npass += 1
            g1 = np.sqrt(1 + (u1 ** 2).sum(1))
            vb = (u0 + u1) / (g0 + g1)[:, None]
            z1, x1, y1 = z0 + dt * vb[:, 0], x0 + dt * vb[:, 1], y0 + dt * vb[:, 2]
            wz, wr = self.path(z0, x0, y0, z1, x1, y1, s["q"], Et=[Etz, Etr])
            dz = dt * vb[:, 0]
            r1 = np.hypot(x1, y1)
            xm, ym, zm = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)
            rm = np.hypot(xm, ym); rms = np.where(rm > 0, rm, 1)
            cx, cy = xm / rms, ym / rms
            drr = r1 - r0
            okz = np.abs(dz) > 1e-13 * self.D
            okr = np.abs(drr) > 1e-13 * self.dr
            ez = np.where(okz, wz / np.where(okz, dz, 1), self.gather(Etz, zm, rm, 0, False, True))
            wdr = np.where(okr, wr / np.where(okr, drr, 1), self.gather(Etr, zm, rm, 0.5, True, True))
            er = wdr * 2 * rm / np.maximum(r0 + r1, 1e-300)          # = work_r / (r_mid . dx_perp)
            bp = self.gather(Be, zm, rm, 0, True, False)
            if self.per_segment:
                Eb = np.stack([ez, self._fperp[0], self._fperp[1]], 1)
            else:
                Eb = np.stack([ez, er * cx, er * cy], 1)
            Bb = np.stack([0 * bp, -bp * cy, bp * cx], 1)
            un = u0 + qm * dt * (Eb + np.cross(vb, Bb))
            du = np.abs(un - u1).max()
            u1 = un
            if du < self.ptol * (1 + np.abs(u1).max()):
                break
        g1 = np.sqrt(1 + (u1 ** 2).sum(1))
        vb = (u0 + u1) / (g0 + g1)[:, None]
        return u1, vb

    def step_implicit(self, tol=1e-10, maxit=60, inner=50, m_and=5):
        p0 = self.npass
        if not hasattr(self, "u1"):
            self.u1 = [s["u"].copy() for s in self.sp]
        Pi = [1.0 / (1 + self.D ** 2 / 4 * p) for p in self.wp2()]
        J = [j.copy() for j in self.J]
        n = self.nz * self.nr

        def Gf(Jv):
            fnew, Et = self.field_solve(Jv)
            paths = []
            for k, s in enumerate(self.sp):
                self.u1[k], vb = self._push_cn(s, Et, self.f[2], fnew[2], self.u1[k], inner)
                paths.append((s, s["z"] + self.D * vb[:, 0], s["x"] + self.D * vb[:, 1], s["y"] + self.D * vb[:, 2]))
            return self.deposit(paths), paths

        Xh, Fh = [], []
        it = 0
        while True:
            Jd, paths = Gf(J)
            fr = np.concatenate([(Pi[c] * (Jd[c] - J[c])).ravel() for c in range(2)])
            x = np.concatenate([J[c].ravel() for c in range(2)])
            res = np.linalg.norm(fr) / max(np.linalg.norm(np.concatenate([a.ravel() for a in Jd])), 1e-300)
            it += 1
            if res < tol or it >= maxit:
                break
            Xh.append(x); Fh.append(fr)
            if len(Xh) > m_and + 1:
                Xh.pop(0); Fh.pop(0)
            if len(Xh) > 1:
                dF = np.stack([Fh[i + 1] - Fh[i] for i in range(len(Fh) - 1)], 1)
                dX = np.stack([Xh[i + 1] - Xh[i] for i in range(len(Xh) - 1)], 1)
                gam = np.linalg.lstsq(dF, fr, rcond=None)[0]
                xn = x + fr - (dX + dF) @ gam
            else:
                xn = x + fr
            J = [xn[:n].reshape(self.nz, self.nr), xn[n:].reshape(self.nz, self.nr)]
        self.f = self.field_solve(Jd)[0]
        for k, (s, z1, x1, y1) in enumerate(paths):
            s["z"], s["x"], s["y"] = z1 % (self.nz * self.D), x1, y1
            s["u"] = self.u1[k].copy()
        self.J = Jd
        self.stats.append((it, res, self.npass - p0))


