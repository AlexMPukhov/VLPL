import numpy as np
from irip_rz import RZ

class RZBox(RZ):
    """Space-time centred ('box', Crank-Nicolson) variant: both movers use mu_x mu_t of Ex, Bx."""
    def step_given(self, S, Ex1, Bx1, a):
        Ex, Er, Et, Bx, Br, Bt = S
        D = self.D
        ExR = ExL = 0.25 * (1 + a) * (Ex1 + Ex)
        BxR = BxL = 0.25 * (1 + a) * (Bx1 + Bx)
        Tpy = (Er + Bt) / a + D * (self.psi_r(BxR) + self.phi_t(ExR)) / a
        Tmy = (Er - Bt) * a + D * (self.psi_r(BxL) - self.phi_t(ExL))
        Tpz = (Et + Br) * a + D * (self.psi_t(BxL) + self.phi_r(ExL))
        Tmz = (Et - Br) / a + D * (self.psi_t(BxR) - self.phi_r(ExR)) / a
        Er1, Bt1 = 0.5 * (Tpy + Tmy), 0.5 * (Tpy - Tmy)
        Et1, Br1 = 0.5 * (Tpz + Tmz), 0.5 * (Tpz - Tmz)
        Exc = Ex + D * self.psi_x(0.5 * (Bt + Bt1), 0.5 * (Br + Br1))
        Bxc = Bx + D * self.phi_x(0.5 * (Et + Et1), 0.5 * (Er + Er1))
        return [Exc, Er1, Et1, Bxc, Br1, Bt1]

def weights(g):
    rn = g.rn.copy(); rn[0] = g.dr / 8
    return np.concatenate([rn, g.rh, rn, g.rh, rn, g.rh])[g.mask()]

def gauss_rows(g, kx):
    """Row vectors G_E, G_B: Gauss law at (i+1/2, r_j) and div B at (i+1/2, r_{j+1/2}) for Fourier mode kx."""
    N, D, a = g.N, g.D, np.exp(1j * kx * g.D)
    n = 6 * N; GE = np.zeros((N, n), complex); GB = np.zeros((N, n), complex)
    for q in range(n):
        e = np.zeros(n, complex); e[q] = 1
        Ex, Er, Et, Bx, Br, Bt = [e[k * N:(k + 1) * N] for k in range(6)]
        mu = 0.5 * (1 + a); dx = (a - 1) / D
        # div_perp E at nodes = psi_x(Btheta -> Er, Br -> -Etheta) structure
        GE[:, q] = dx * Ex + mu * g.psi_x(Er, -Et)
        GB[:, q] = dx * Bx - mu * g.phi_x(Br, -Bt)        # div_perp B at half nodes
    ok = g.mask()
    rows = slice(0 if g.m == 0 else 1, None)       # no Gauss row on the axis for m >= 1
    return GE[rows][:, ok], GB[:, ok]

if __name__ == "__main__":
    for m, D in [(0, 10.0), (1, 10.0), (3, 2.0), (20, 2.0), (50, 10.0)]:
        res = []
        for cls in (RZ, RZBox):
            g = cls(48, 1.0, D, m); s = np.sqrt(weights(g))
            sup, rad, gerr = 1.0, 0.0, 0.0
            for kxD in (0.26, 1.0, 3.0):
                M = g.amplification(kxD / D)[0]
                Mw = (s[:, None] * M) / s[None, :]
                P = Mw.copy()
                for k in range(12):
                    sup = max(sup, np.linalg.norm(P, 2)); P = P @ P
                rad = max(rad, np.abs(np.abs(np.linalg.eigvals(M)) - 1).max())
                GE, GB = gauss_rows(g, kxD / D)
                gerr = max(gerr, np.abs(GE @ M - GE).max() / np.abs(GE).max(), np.abs(GB @ M - GB).max() / np.abs(GB).max())
            res.append((sup, rad, gerr))
        print("m=%2d D/dr=%4.1f | iRIP: sup||M^n|| %7.2f  ||z|-1| %.0e  std Gauss/divB drift %.1e | box: sup %.6f  ||z|-1| %.0e  Gauss/divB %.1e"
              % ((m, D) + res[0] + res[1]))
