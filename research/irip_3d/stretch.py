# Box iRIP on a non-uniform transverse grid: reflection of radiation where fine cells (h << c dt, as near
# an RZ axis) stretch to coarse ones. 2D slice (no z dependence), one Fourier mode e^{i kx x} along x
# (x shifts are factors a = e^{i kx D}), y nodes y_j arbitrary and periodic; c dt = h_x = D = 1.
# The step is identity.py's RIP-CC step with the y differences divided by the local spacing:
#   node -> half node (j+1/2): / (y_{j+1} - y_j),  half node -> node j: / (yh_j - yh_{j-1}),
# and the box closure (both movers use mu_t of Ex, Bx; implicit solve (I - A) X = R, A tridiagonal in y).
# Fields: Ex, Ez, By at nodes; Ey, Bz, Bx at half nodes (RIP-CC staggering with z dropped).
# Checks: (1) energy (weights: dual cell for node fields, h_j for half-node fields) conserved on the
# stretched grid; (2) reflection of a wave packet running from the fine region into the stretch,
# against the same packet on a uniform fine grid (the floor of the measurement).
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import sys


class Grid:
    def __init__(self, y, L, kx):
        self.y = y; self.N = len(y); self.L = L
        yn = np.append(y[1:], y[0] + L)
        self.hH = yn - y                                   # spacing node j -> j+1 (weight of half nodes)
        yh = 0.5 * (y + yn)
        self.hN = yh - np.roll(yh, 1); self.hN[0] = yh[0] - (yh[-1] - L)   # dual cell of node j
        self.a = np.exp(1j * kx)
        self.yh = yh
        N = self.N
        # A = dR/dX (the same operator for Ex and Bx up to the staggering), probed with 3 combs
        self.M = {}
        for which in (0, 1):
            cols = []
            for c in range(3):
                e = np.zeros(N, complex); e[c::3] = 1
                f = [np.zeros(N, complex)] * 6
                if which == 0: X, Y = e, 0 * e
                else: X, Y = 0 * e, e
                F, G, _ = self.pass_(f, X, Y)
                cols.append(F if which == 0 else G)
            rows, colsi, vals = [], [], []
            for j in range(N):
                for dj in (-1, 0, 1):
                    q = (j + dj) % N
                    v = cols[q % 3][j]
                    if v != 0: rows.append(j); colsi.append(q); vals.append(v)
            A = sp.csc_matrix((vals, (rows, colsi)), shape=(N, N))
            self.M[which] = sla.splu(sp.identity(N, complex, format="csc") - A)

    def pass_(self, f, X, Y):
        """RIP-CC transport + corrector with Ex_bar = mu_x (Ex + X)/2, Bx_bar likewise (J = 0)."""
        Ex, Ey, Ez, Bx, By, Bz = f
        a = self.a
        mux = lambda g: 0.5 * (1 + a) * g
        Exs, Bxs = 0.5 * (Ex + X), 0.5 * (Bx + Y)
        up = lambda g: np.roll(g, -1)                      # g(j+1)
        dn = lambda g: np.roll(g, 1)                       # g(j-1)
        psiz = -(mux(Bxs) - dn(mux(Bxs))) / self.hN        # at node j (Ez, By)
        phiz = (up(mux(Exs)) - mux(Exs)) / self.hH         # at half node j+1/2 (Ey, Bz)
        # transport: right movers from i-1 (factor 1/a), left movers from i+1 (factor a); sources at i-1/2
        # (index i-1) and i+1/2 (index i)
        tpy = (Ey + Bz) / a + phiz / a
        tmy = a * (Ey - Bz) - phiz
        tpz = a * (Ez + By) + psiz
        tmz = (Ez - By) / a + psiz / a
        nEy, nBz = 0.5 * (tpy + tmy), 0.5 * (tpy - tmy)
        nEz, nBy = 0.5 * (tpz + tmz), 0.5 * (tpz - tmz)
        bz = 0.5 * (Bz + nBz)
        ez = 0.5 * (Ez + nEz)
        nEx = Ex + (bz - dn(bz)) / self.hN
        nBx = Bx - (up(ez) - ez) / self.hH
        return nEx, nBx, (nEy, nEz, nBy, nBz)

    def step(self, f):
        Z = np.zeros(self.N, complex)
        R, S, _ = self.pass_(f, Z, Z)
        X, Y = self.M[0].solve(R), self.M[1].solve(S)
        nEx, nBx, (nEy, nEz, nBy, nBz) = self.pass_(f, X, Y)
        return [nEx, nEy, nEz, nBx, nBy, nBz]

    def energy(self, f, mask_node=None, mask_half=None):
        wN, wH = self.hN.copy(), self.hH.copy()
        if mask_node is not None: wN = wN * mask_node; wH = wH * mask_half
        Ex, Ey, Ez, Bx, By, Bz = f
        return float(sum((abs(g) ** 2 * w).sum() for g, w in ((Ex, wN), (Ez, wN), (By, wN), (Ey, wH), (Bz, wH), (Bx, wH))))


def stretched(hf, Y1, ratio, hmax, Lc):
    """Nodes: uniform hf on [0, Y1), then each cell ratio times the previous up to hmax, then hmax for Lc."""
    y = list(np.arange(0, Y1 - 1e-9, hf))
    h = hf; yy = Y1
    while h < hmax:
        y.append(yy); h = min(h * ratio, hmax); yy += h
    end = yy + Lc
    while yy < end or len(y) % 3:                         # N multiple of 3: the comb probe of A
        y.append(yy); yy += hmax
    return np.array(y), yy


def energy_check():
    rng = np.random.default_rng(2)
    y, L = stretched(0.1, 6.0, 1.3, 2.0, 10.0)
    g = Grid(y, L, kx=0.7)
    f = [rng.normal(size=g.N) + 1j * rng.normal(size=g.N) for _ in range(6)]
    w0 = g.energy(f); dev = 0
    for n in range(400):
        f = g.step(f); dev = max(dev, abs(g.energy(f) / w0 - 1))
    print(f"energy check: N = {g.N}, h from 0.1 to 2.0 (D/h_min = 10), 400 steps: max |W/W0 - 1| = {dev:.1e}")


def packet_run(lam0, theta_deg, hf, ratio, hmax):
    """Packet of wavelength lam0 (units D) at angle theta to x in the fine region, moving +y towards the
    stretch. Absorbers at both ends (fine cells below y = 0, h_max cells beyond the stretch) record the
    energy they take; the run goes on until they take less than 1e-8 W0 in 200 steps (or 40 crossing
    times). "left" is mostly the static (zero-frequency) part of the continuum-initialised packet. Reflected = energy absorbed at the low end minus that of the same run on the uniform fine
    grid (the backward part of the continuum-initialised packet arrives there too, much earlier).
    Returns (reflected, transmitted, left) / W0 for both polarizations."""
    k = 2 * np.pi / lam0; th = np.radians(theta_deg)
    kx, ky = k * np.cos(th), k * np.sin(th)
    lamy = 2 * np.pi / ky
    sig = 2.0 * lamy
    y0 = 4 * sig; Y1 = 8 * sig
    La = 6 * lamy                                          # absorber length
    T0 = int((Y1 - y0 + 5 * sig) / np.sin(th)) + 10
    out = []
    grids = []
    for hm in (hmax, hf):
        yb, Lb = stretched(hf, Y1 + La, ratio, hm, La + 2 * hm)   # fine region extended into y < 0 below
        y = np.concatenate([np.arange(-La, 0, hf), yb])
        y = y[: len(y) - len(y) % 3]
        L = Lb + La
        g = Grid(y, L, kx)
        ylo, yhi = 0.0, y[-1] - La                         # absorbers: y < 0 and y > yhi
        def damp(q, ylo=ylo, yhi=yhi):
            r = np.where(q < ylo, (ylo - q) / La, np.where(q > yhi, (q - yhi) / La, 0.0))
            # strong enough for waves that cross it in a few steps; the outer tenth is zeroed, so
            # nothing wraps (periodic y) from one absorber into the other
            return np.where(r > 0.9, 0.0, np.exp(-2.0 * r * r))
        g.dN, g.dH = damp(g.y), damp(g.yh)
        g.loN, g.loH = (g.y < ylo).astype(float), (g.yh < ylo).astype(float)
        g.hiN, g.hiH = (g.y > yhi).astype(float), (g.yh > yhi).astype(float)
        grids.append(g)
    for pol in (1, 2):
        res = []
        for g in grids:
            env = lambda q: np.exp(-((q - y0) / sig) ** 2 / 2) * np.exp(1j * ky * q)
            yn, yh = g.y, g.yh
            Z = np.zeros(g.N, complex)
            if pol == 1: f = [Z, Z, env(yn), ky / k * env(yh), -kx / k * env(yn), Z]
            else: f = [-ky / k * env(yn), kx / k * env(yh), Z, Z, Z, env(yh)]
            w0 = g.energy(f); lo = hi = last = 0.0
            for n in range(40 * T0):
                f = g.step(f)
                w_lo, w_hi = g.energy(f, g.loN, g.loH), g.energy(f, g.hiN, g.hiH)
                f = [u * (g.dN if c in (0, 2, 4) else g.dH) for c, u in enumerate(f)]
                lo += w_lo - g.energy(f, g.loN, g.loH); hi += w_hi - g.energy(f, g.hiN, g.hiH)
                if n % 200 == 0:                           # stop when the absorbers take nothing new
                    if n > 2 * T0 and lo + hi - last < 1e-8 * w0: break
                    last = lo + hi
            res.append((lo / w0, hi / w0, g.energy(f) / w0, n + 1))
        (lo, hi, left, n), (lo_u, _, _, _) = res
        out.append((lo - lo_u, hi, left, n))
    return out, T0, grids[0].N


if __name__ == "__main__":
    energy_check()
    hf = 0.1
    print("energies / W0 at the end of the run; k_y: wavenumber in the fine region")
    print("lam0/D theta  stretch(ratio, h_max)  k_y h_max   reflected (TE, TM)     transmitted  left (TE)   steps")
    cases = [(20, 45, 1.05, 1.0), (20, 70, 1.05, 1.0), (20, 85, 1.05, 1.0), (20, 45, 1.5, 1.0), (20, 70, 1.05, 4.0),
             (6, 45, 1.05, 1.0), (6, 70, 1.05, 1.0), (6, 70, 1.05, 2.0), (3, 45, 1.05, 1.0), (3, 70, 1.05, 1.0),
             (3, 70, 1.05, 0.5), (3, 70, 1.5, 1.0), (3, 70, 1.05, 2.0), (6, 45, 1.05, 4.0)]
    if len(sys.argv) > 1: cases = cases[:int(sys.argv[1])]
    for lam0, th, r, hm in cases:
        R, T, N = packet_run(lam0, th, hf, r, hm)
        kp = 2 * np.pi / lam0 * np.sin(np.radians(th)) * hm
        print(f"{lam0:5g} {th:5g}   ({r:4.2f}, {hm:4.1f})          {kp:6.2f}     {R[0][0]:9.2e} {R[1][0]:9.2e}   {R[0][1]:8.4f}   {R[0][2]:8.1e}   {R[0][3]} (N {N})", flush=True)
    # c dt scan at a fixed physical grid and wavelength (lengths in units H): h_min = 0.1 H, h_max = H,
    # stretch 1.05, theta = 70 deg, wavelength 12 H, 6 H (k_y h_max = 0.98) and 3 H (1.97, near the cutoff 2);
    # D/h_min = 10, 5, 2, 1 (1: the explicit RIP-CC regime h_perp >= c dt in the fine region)
    for lamH in (12, 6, 3):
        print(f"c dt scan, fixed grid (h_min, h_max) = (0.1, 1) H, wavelength {lamH} H, theta 70 deg:")
        print("  D/h_min   reflected (TE, TM)     transmitted  left (TE)   steps")
        for q in (10, 5, 2, 1):
            D = q * 0.1                                    # in units H; the code units are D
            R, T, N = packet_run(lamH / D, 70, 0.1 / D, 1.05, 1 / D)
            print(f"  {q:6g}     {R[0][0]:9.2e} {R[1][0]:9.2e}   {R[0][1]:8.4f}   {R[0][2]:8.1e}   {R[0][3]} (N {N})", flush=True)
