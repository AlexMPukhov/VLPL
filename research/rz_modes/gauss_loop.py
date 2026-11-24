# Full loop in quasi-3D RZ with azimuthal modes: the RIP-CC field step per mode on a (z, r) grid with the
# sources of the charge-conserving deposition of deposit.py, driven by ballistic particles. The per-mode
# Gauss residual  G_m = d_z Ez + mu_z [(1/r) d_r (r Er) + (i m / r) Ephi] - rho_m  at the rho nodes
# (i+1/2, r_j) must stay at its initial value (round-off), for the explicit RIP-CC step (predictor) and
# for the box-closure iRIP step (implicit Ez, Bz from a dense solve). Axis rules of research/irip_rz:
# m = 0: Ephi = Br = 0 on the axis, (1/r)d_r(r F)(0) = 4 F_{1/2}/dr; m >= 1: Ez = 0 on the axis,
# Ephi, Br on the axis not dynamic (m = 1: from the one-sided rules of irip_rz.py). Wall: zero beyond r_N.
import numpy as np
from deposit import Nz, Nr, M, dz, dr, deposit
rh = (np.arange(Nr) + 0.5) * dr
rng = np.random.default_rng(7)
D = dz                                   # c dt = dz
rn = np.arange(Nr) * dr
sh = lambda f, s: np.roll(f, -s, axis=0)  # periodic z: f(i + s)
def up(f):                               # f(j+1), zero beyond the wall
    g = np.zeros_like(f); g[:, :-1] = f[:, 1:]; return g
def dn(f):                               # f(j-1), zero below the axis (not used at j = 0 where rules apply)
    g = np.zeros_like(f); g[:, 1:] = f[:, :-1]; return g

def ops(m):
    rnz = rn.copy(); rnz[0] = 1.0        # avoid 1/0; axis rows are set by the rules
    def psi_x(Bt, Br):                   # (1/r) d_r(r Bphi) - (i m/r) Br at nodes
        o = (rh * Bt - dn(rh * Bt)) / (rnz * dr) - 1j * m / rnz * Br
        o[:, 0] = 4 * Bt[:, 0] / dr if m == 0 else 0
        return o
    def phi_x(Et, Er):                   # -[(1/r) d_r(r Ephi) - (i m/r) Er] at half nodes
        return -((up(rn * Et) - rn * Et) / (rh * dr) - 1j * m / rh * Er)
    def psi_r(Bx): return 1j * m / rh * Bx              # half nodes
    def psi_t(Bx):                                      # -d_r Bz at nodes
        o = -(Bx - dn(Bx)) / dr
        o[:, 0] = -2 * Bx[:, 0] / dr if m == 1 else 0
        return o
    def phi_r(Ex):                                      # -(i m/r) Ez at nodes
        o = -1j * m / rnz * Ex
        o[:, 0] = -1j * Ex[:, 1] / dr if m == 1 else 0
        return o
    def phi_t(Ex): return (up(Ex) - Ex) / dr            # half nodes
    return psi_x, phi_x, psi_r, psi_t, phi_r, phi_t

def step_given(m, S, J, Exs, Bxs):
    """RIP-CC step of mode m with the given longitudinal averages Exs (Ez*), Bxs (Bz*)."""
    psi_x, phi_x, psi_r, psi_t, phi_r, phi_t = ops(m)
    Ex, Er, Et, Bx, Br, Bt = S; Jz, Jr, Jp = J
    mux = lambda f: 0.5 * (f + sh(f, 1))
    PSr = psi_r(mux(Bxs)) - Jr; PSt = psi_t(mux(Bxs)) - Jp
    PHr = phi_r(mux(Exs)); PHt = phi_t(mux(Exs))
    m1 = lambda f: sh(f, -1)
    Tpy = m1(Er) + m1(Bt) + D * (m1(PSr) + m1(PHt))
    Tmy = sh(Er, 1) - sh(Bt, 1) + D * (PSr - PHt)
    Tpz = sh(Et, 1) + sh(Br, 1) + D * (PSt + PHr)
    Tmz = m1(Et) - m1(Br) + D * (m1(PSt) - m1(PHr))
    Er1, Bt1 = 0.5 * (Tpy + Tmy), 0.5 * (Tpy - Tmy)
    Et1, Br1 = 0.5 * (Tpz + Tmz), 0.5 * (Tpz - Tmz)
    Ex1 = Ex + D * (psi_x(0.5 * (Bt + Bt1), 0.5 * (Br + Br1)) - Jz)
    Bx1 = Bx + D * phi_x(0.5 * (Et + Et1), 0.5 * (Er + Er1))
    if m >= 1: Ex1[:, 0] = 0
    Et1[:, 0] = 0; Br1[:, 0] = 0          # not dynamic (m = 1: driven only; kept 0 here)
    return [Ex1, Er1, Et1, Bx1, Br1, Bt1]

def explicit(m, S, J):
    psi_x, phi_x, *_ = ops(m)
    Ex, Er, Et, Bx, Br, Bt = S
    Exs = Ex + 0.5 * D * (psi_x(Bt, Br) - J[0]); Bxs = Bx + 0.5 * D * phi_x(Et, Er)
    if m >= 1: Exs[:, 0] = 0
    return step_given(m, S, J, Exs, Bxs)

def box(m, S, J):
    Z = np.zeros((Nz, Nr), complex); n = Nz * Nr
    f = lambda X, Y: step_given(m, S, J, 0.5 * (S[0] + X), 0.5 * (S[3] + Y))
    o0 = f(Z, Z); A = np.zeros((2 * n, 2 * n), complex)
    for q in range(2 * n):
        e = np.zeros(2 * n, complex); e[q] = 1
        o = f(e[:n].reshape(Nz, Nr), e[n:].reshape(Nz, Nr))
        A[:, q] = np.concatenate([(o[0] - o0[0]).ravel(), (o[3] - o0[3]).ravel()])
    rhs = np.concatenate([o0[0].ravel(), o0[3].ravel()])
    XY = np.linalg.solve(np.eye(2 * n) - A, rhs)
    return f(XY[:n].reshape(Nz, Nr), XY[n:].reshape(Nz, Nr))

def gauss(m, S, rho):
    Ex, Er, Et = S[0], S[1], S[2]
    rnz = rn.copy(); rnz[0] = 1.0
    divT = (rh * Er - dn(rh * Er)) / (rnz * dr) + 1j * m / rnz * Et
    divT[:, 0] = 4 * Er[:, 0] / dr if m == 0 else 0
    return (sh(Ex, 1) - Ex) / dz + 0.5 * (divT + sh(divT, 1)) - rho

# particles: ballistic, |v_z| < 1 (c dt = dz), transverse moves < dr per step, kept off the wall
Np = 200
P = []
for _ in range(Np):
    r = rng.uniform(0, (Nr - 7) * dr); ph = rng.uniform(-np.pi, np.pi)
    P.append((np.array([rng.uniform(0, Nz * dz), r * np.cos(ph), r * np.sin(ph)]),
              np.array([rng.uniform(-0.9, 0.9), *rng.uniform(-0.15, 0.15, 2) * min(dr / D, 0.9)]), rng.choice([-1.0, 1.0])))
import os
runs = (("explicit RIP-CC", explicit),) if os.environ.get("RZ_STABLE") else (("explicit RIP-CC", explicit), ("box iRIP", box))
for name, stepper in runs:
    S = [[np.zeros((Nz, Nr), complex) for _ in range(6)] for m in range(M)]
    worst = np.zeros(M); G0 = None; pos = [p.copy() for p, v, q in P]
    for n in range(12):
        shape = (M, Nz, Nr + 2)
        rho0 = np.zeros(shape, complex); rho1 = np.zeros(shape, complex)
        Jz = np.zeros(shape, complex); Jr = np.zeros(shape, complex); Jp = np.zeros(shape, complex)
        for k, (p, v, q) in enumerate(P):
            p0 = pos[k]; p1 = p0 + v * D
            deposit(p0, p1, q, rho0, rho1, Jz, Jr, Jp)
            pos[k] = p1.copy(); pos[k][0] %= Nz * dz
        if G0 is None: G0 = [gauss(m, S[m], rho0[m, :, :Nr]) for m in range(M)]
        for m in range(M):
            S[m] = stepper(m, S[m], [Jz[m, :, :Nr], Jr[m, :, :Nr], Jp[m, :, :Nr]])
            G = gauss(m, S[m], rho1[m, :, :Nr])
            jlo = 0 if m == 0 else 1
            scale = max(np.abs(G0[m]).max(), np.abs(G + rho1[m, :, :Nr]).max())   # |rho| or |div E|
            worst[m] = max(worst[m], np.abs(G - G0[m])[:, jlo:Nr - 2].max() / scale)
    print("%-16s Gauss law change per mode (relative): %s" % (name, ["m=%d %.1e" % (m, w) for m, w in enumerate(worst)]))
