# O5 limiter of the transverse term for dy << dx (docs/specs/overdense.md): per Fourier mode the transverse factor
#   g_perp = 1 + D^2 lam / 6 * f(q) + h^2 lam / 12,   q = D^2 lam / 4,   lam = (2/h sin(ky h/2))^2
# applied as a per-mode factor (as in 3D / RZ / CPML), with the time-step part limited by f:
#   none:   f = 1                     (the closure of decisions 72-76)
#   capQ:   D^2 lam/6 f = (2/3) min(q, Q)     (hard cap at q = Q)
#   soft:   (2/3) q / (1 + q / Q)              (smooth, -> (2/3) Q at the grid scale)
# Measured: spectral radius and transient growth max_n ||G^n|| (n <= 400) over the Brillouin zone, and the phase error of
# resolved modes (orders 7, 15 at 0.2-0.6 rad) against c |k|.
import sys
import numpy as np
sys.path.insert(0, ".")
import closure2d as c2

def make(D, hy, lim):
    G0 = c2.make(D, hy, 1, 0)        # Cx stencil, no transverse stencil
    def G(kx, ky):
        lam = (2 / hy * np.sin(ky * hy / 2)) ** 2
        q = D * D * lam / 4
        g = 1 + 2 / 3 * lim(q) + hy * hy * lam / 12
        # the per-mode factor: rescale the closure inside the step via a stencil-free hook
        c2._gfac = g
        return G0(kx, ky)
    return G

# patch closure2d.make so that C also multiplies by the per-mode factor c2._gfac
src = open("closure2d.py").read().replace("C = lambda f: Cx(Cy(f))", "C = lambda f: Cx(Cy(f)) * _gfac")
src = src.split('if __name__ == "__main__":')[0]
c2._gfac = 1.0
exec(compile(src, "closure2d_patched", "exec"), c2.__dict__)

lims = {
    "none": lambda q: q,
    "cap 1": lambda q: min(q, 1.0),
    "cap 2.47": lambda q: min(q, np.pi ** 2 / 4),
    "soft 1": lambda q: q / (1 + q),
    "soft 2.47": lambda q: q / (1 + q / (np.pi ** 2 / 4)),
}
mode = sys.argv[1] if len(sys.argv) > 1 else "stab"
D = 1.0
if mode == "stab":
    nk = 25
    for hyo in (3.2, 1.0, 0.5, 0.2, 0.1):
        hy = hyo * D
        for name, lim in lims.items():
            G = make(D, hy, lim)
            rho, grow, worst = 0.0, 0.0, None
            for kx in np.linspace(0, np.pi / D, nk):
                for ky in np.linspace(0, np.pi / hy, nk):
                    M = G(kx, ky)
                    r = np.abs(np.linalg.eigvals(M)).max()
                    P, g = np.eye(6), 1.0
                    for _ in range(400):
                        P = M @ P
                        g = max(g, np.linalg.norm(P, 2))
                    rho = max(rho, r)
                    if g > grow: grow, worst = g, (kx * D, ky * hy)
            print("dy = %.1f dx  %-9s  max |eig| - 1 = %+.1e   transient growth %6.3f at (kx dx, ky dy) = (%.2f, %.2f)"
                  % (hyo, name, rho - 1, grow, *worst), flush=True)
else:
    # phase error of resolved modes: dx = lambda0/128, dy = dx * hyo
    lam0 = 1.0
    Dp = lam0 / 128
    for hyo in (3.2, 1.0, 0.25):
        hy = hyo * Dp
        print("dx = lambda0/128, dy = %.2f dx: relative phase error w/(c k) - 1" % hyo)
        print("   N theta " + "".join("%11s" % k for k in lims))
        for N in (7, 15, 30):
            K = 2 * np.pi * N / lam0
            for th in (0.2, 0.4, 0.6):
                kx, ky = K * np.cos(th), K * np.sin(th)
                row = [c2.omega(make(Dp, hy, lim), kx, ky, Dp)[0] / K - 1 for lim in lims.values()]
                print("  %2d  %.1f " % (N, th) + "".join("%+11.2e" % e for e in row), flush=True)
