#!/usr/bin/env python3
# Usage: python3 tools/volkov_spin.py [checks|em_s|pairs|conv|emission]   (decision 96, docs/specs/qed.md; ~10 min per mode on 4 cores)
"""First-principles check of the transverse spin of Breit-Wheeler pairs (and of the Sokolov-Ternov sign of the emission)
from Volkov states and explicit Dirac spinors, with no formula from a paper.

Units m = c = hbar = 1, metric (+,-,-,-), Dirac representation, gamma5 = i g0 g1 g2 g3. The Dirac field has the electron charge
q = -1 in (i dslash - q Aslash - m) psi = 0 (force q (E + v x B) on the classical particle). Laser along -z: kappa = w0 (1,0,0,-1),
phi = kappa.x = w0 (t + z), A^mu = (0, a(phi), 0, 0), a(phi) = a0 (1 + tanh(phi / tau)): E = -w0 a' x, B = +w0 a' y with a' > 0
everywhere, so a particle moving along +z sees the rest-frame magnetic field B' = gamma (B - v x E) along +y (computed below).

Spinors: u(p, n) = (pslash + m)(chi, 0) / sqrt(E + m), sigma.n chi = +chi (pure boost of the rest-frame spin n).
Positron of physical spin n (hole theory): the negative-energy solution e^{+ipx} v with v = (-pslash + m)(0, eta) / sqrt(E + m),
sigma.n eta = -eta (the missing negative-energy electron has spin -n); cross-checked against charge conjugation v = i g2 u*.

Volkov solutions (checked numerically against the Dirac equation below):
  psi_p    = [1 + q kslash Aslash / (2 k.p)] u e^{-i p.x - i int (q p.A / k.p - q^2 A^2 / (2 k.p)) dphi}
  psi_{-p} = [1 - q kslash Aslash / (2 k.p)] v e^{+i p.x - i int (q p.A / k.p + q^2 A^2 / (2 k.p)) dphi}
Amplitudes (photon absorbed: eps e^{-ikx}; emitted: eps* e^{+ikx}); the phase integral over phi with analytic tails.
"""
import sys
import numpy as np
from scipy.special import kv
from scipy.integrate import quad

I2 = np.eye(2)
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)
Z2 = np.zeros((2, 2))
g0 = np.block([[I2, Z2], [Z2, -I2]]).astype(complex)
gi = [np.block([[Z2, s], [-s, Z2]]) for s in (sx, sy, sz)]
g5 = 1j * g0 @ gi[0] @ gi[1] @ gi[2]
SIG = [np.block([[s, Z2], [Z2, s]]) for s in (sx, sy, sz)]


def sl(a):
    return g0 * a[0] - gi[0] * a[1] - gi[1] * a[2] - gi[2] * a[3]


def dot(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def bar(s):
    return s.conj() @ g0


def two_spinor(n, sign):
    """eigenvector of sigma.n with eigenvalue sign"""
    M = n[0] * sx + n[1] * sy + n[2] * sz
    w, V = np.linalg.eigh(M)
    return V[:, np.argmin(np.abs(w - sign))]


def u_sp(p, n):
    E = p[0]
    chi = two_spinor(n, +1)
    return (sl(p) + np.eye(4)) @ np.concatenate([chi, [0, 0]]) / np.sqrt(E + 1)


def v_sp(p, n):            # positron of PHYSICAL spin n (rest frame reached by the pure boost)
    E = p[0]
    eta = two_spinor(n, -1)
    return (-sl(p) + np.eye(4)) @ np.concatenate([[0, 0], eta]) / np.sqrt(E + 1)


def lc(plus, px, py, m2=1.0):
    mi = (m2 + px * px + py * py) / plus
    return np.array([(plus + mi) / 2, px, py, (plus - mi) / 2])


W0 = 1e-4
KAP = W0 * np.array([1.0, 0, 0, -1.0])
EL = sl(np.array([0, 1.0, 0, 0]))          # Aslash = a EL
KS = sl(KAP)


def lncosh(y):
    y = np.abs(y)
    return y + np.log1p(np.exp(-2 * y)) - np.log(2)


class Field:
    def __init__(s, a0, tau):
        s.a0, s.tau = a0, tau

    def a(s, f):
        return s.a0 * (1 + np.tanh(f / s.tau))

    def da(s, f):
        return s.a0 / s.tau / np.cosh(f / s.tau) ** 2

    def A1(s, f):           # int a dphi
        return s.a0 * (f + s.tau * lncosh(f / s.tau))

    def A2(s, f):           # int a^2 dphi
        return s.a0 ** 2 * (2 * f + 2 * s.tau * lncosh(f / s.tau) - s.tau * np.tanh(f / s.tau))


def phase_integrals(F, lam, c1, c2, L, pts_per_rad=12.0):
    """I_n = int a^n e^{i Phi}, Phi = lam phi + c1 A1 + c2 A2 (Phi' = lam + c1 a + c2 a^2), tails analytic"""
    amax = 2 * F.a0
    dmax = max(abs(lam), abs(lam + c1 * amax + c2 * amax ** 2), abs(lam + c1 * F.a0 + c2 * F.a0 ** 2))
    n = int(2 * L * dmax * pts_per_rad) + 1
    n += n % 2 == 0
    f = np.linspace(-L, L, n)
    h = f[1] - f[0]
    ph = lam * f + c1 * F.A1(f) + c2 * F.A2(f)
    e = np.exp(1j * ph)
    a = F.a(f)
    w = np.full(n, 2.0); w[1::2] = 4.0; w[0] = w[-1] = 1.0; w *= h / 3      # Simpson
    I = [np.sum(w * e), np.sum(w * a * e), np.sum(w * a * a * e)]
    # tails: phi < -L (a = a(-L)), phi > L (a = a(L)), Phi linear
    for side in (-1, 1):
        fe = side * L
        ae = F.a(fe)
        dph = lam + c1 * ae + c2 * ae ** 2
        t = np.exp(1j * (lam * fe + c1 * F.A1(fe) + c2 * F.A2(fe))) / (1j * dph) * (-side)
        I[0] += t; I[1] += ae * t; I[2] += ae * ae * t
    return I


# ------------------------------------------------------------------ 0. checks of the building blocks
def check_dirac(q=-1.0):
    F = Field(3.0, 2.0)
    p = lc(5.0 / W0 * 1e-0, 0.7, -0.3)          # some momentum
    kp = dot(KAP, p)
    out = []
    for kind in ("electron", "positron"):
        sp = u_sp(p, np.array([0, 1.0, 0])) if kind == "electron" else v_sp(p, np.array([0, 1.0, 0]))
        sg = 1 if kind == "electron" else -1

        def Fphi(f):
            a = F.a(f)
            Asl = a * EL
            pre = (np.eye(4) + sg * q * KS @ Asl / (2 * kp)) @ sp
            # p.A = -a px, A^2 = -a^2
            if kind == "electron":
                ph = -(q * (-p[1]) * F.A1(f) / kp - q * q * (-F.A2(f)) / (2 * kp))
            else:
                ph = -(q * (-p[1]) * F.A1(f) / kp + q * q * (-F.A2(f)) / (2 * kp))
            return pre * np.exp(1j * ph)
        f0, h = 0.37, 1e-5
        Fv = Fphi(f0)
        dF = (Fphi(f0 + h) - Fphi(f0 - h)) / (2 * h)
        plow = np.array([p[0], -p[1], -p[2], -p[3]])
        klow = np.array([KAP[0], -KAP[1], -KAP[2], -KAP[3]])
        # psi = F(phi) e^{-i s p.x}: d_mu psi = (k_mu F' - i s p_mu F) e^{..}; i gamma^mu d_mu with lower-index vectors
        gl = lambda vlow: g0 * vlow[0] + gi[0] * vlow[1] + gi[1] * vlow[2] + gi[2] * vlow[3]
        dir_ = 1j * (gl(klow) @ dF - 1j * sg * gl(plow) @ Fv)
        res = dir_ - q * F.a(f0) * EL @ Fv - Fv
        out.append((kind, np.linalg.norm(res) / (np.linalg.norm(gl(plow) @ Fv) + 1e-300)))
    return out


def check_spinors():
    p = lc(3.0, 0.4, 0.9)
    res = {}
    for n in (np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])):
        u = u_sp(p, n)
        vh = v_sp(p, n)
        vc = 1j * gi[1] @ u.conj()           # charge conjugation (Dirac rep): v(p, s) = i gamma^2 u(p, s)*
        ov = abs(np.vdot(vh, vc)) / (np.linalg.norm(vh) * np.linalg.norm(vc))
        # which covariant spin s^mu reproduces v vbar = (pslash - 1)(1 + g5 sslash)/2 ?
        E = p[0]; pv = p[1:]
        s4 = np.concatenate([[pv @ n], n + pv * (pv @ n) / (E + 1)])
        P = np.outer(vh, bar(vh))
        Pp = (sl(p) - np.eye(4)) @ (np.eye(4) + g5 @ sl(s4)) / 2
        Pm = (sl(p) - np.eye(4)) @ (np.eye(4) - g5 @ sl(s4)) / 2
        U = np.outer(u, bar(u))
        Up = (sl(p) + np.eye(4)) @ (np.eye(4) + g5 @ sl(s4)) / 2
        # rest-frame spin of u: boost back is implicit; Sigma.n on (chi, 0) at rest = +1 by construction
        res[tuple(n)] = (ov, np.linalg.norm(P - Pp), np.linalg.norm(P - Pm), np.linalg.norm(U - Up))
    return res


def rest_B(p, f, F):
    """rest-frame magnetic field (pure boost) of a particle with 4-momentum p at phase f"""
    E = np.array([-W0 * F.da(f), 0, 0]); B = np.array([0, W0 * F.da(f), 0])
    v = p[1:] / p[0]; g = p[0]
    return g * (B - np.cross(v, E)) - g * g / (g + 1) * v * (v @ B)


# ------------------------------------------------------------------ 1. emission (Sokolov-Ternov orientation)
def emission(F, eta, q=-1.0, ns=24, nu=61, umax=6.0, L=None):
    """W(sigma_i -> sigma_f) along +y (rest frame) for an electron (q = -1) or positron (q = +1, the C-conjugate field) along
    +z emitting at the phase phi = 0 (k'_x fixed by collinearity); summed over the photon polarization, integrated over k'_y
    and the photon fraction s (Gauss-Legendre). Returns the flip rates and the equilibrium polarization along +y."""
    L = L or 9 * F.tau
    p = lc(eta / W0, 0.0, 0.0)
    kp = dot(KAP, p)
    xs, ws = np.polynomial.legendre.leggauss(ns)
    xs = 0.5 * (xs + 1); ws = 0.5 * ws
    ny = np.array([0, 1.0, 0])
    Wf = np.zeros((2, 2))              # [initial up/down][final up/down]
    piX = -q * F.a(0.0)                # kinetic pi_x = p_x - q a at phi = 0
    for s, wsg in zip(xs, ws):
        kx = s * piX
        us = np.linspace(-umax, umax, nu) * max(1.0, (eta * F.a0 / F.tau) ** (1 / 3))
        du = us[1] - us[0]
        for uu in us:
            ky = s * uu
            k = lc(s * p[0] * 2 if False else s * (p[0] + p[3]), kx, ky, 0.0)
            pp = lc((1 - s) * (p[0] + p[3]), -kx, -ky)
            kpp = dot(KAP, pp)
            D = pp + k - p
            lam = D[0] / KAP[0]
            # f_{p'} - f_p = -q a p'_x / k.p' + q^2 a^2 / (2 k.p') + q a p_x / k.p - q^2 a^2 / (2 k.p)
            c1 = -q * pp[1] / kpp + q * p[1] / kp
            c2 = q * q / (2 * kpp) - q * q / (2 * kp)
            I = phase_integrals(F, lam, c1, c2, L)
            kk = dot(KAP, k)
            amp = np.zeros((2, 2))
            for e in (np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0])):
                eps = e - dot(k, e) / kk * KAP          # Lambda_j, k.eps = 0
                es = sl(eps)
                G0 = es
                G1 = q * EL @ KS @ es / (2 * kpp) + q * es @ KS @ EL / (2 * kp)
                G2 = q * q * EL @ KS @ es @ KS @ EL / (4 * kp * kpp)
                G = G0 * I[0] + G1 * I[1] + G2 * I[2]
                for i, si in enumerate((1, -1)):
                    ui = u_sp(p, si * ny)
                    for j, sj in enumerate((1, -1)):
                        uf = u_sp(pp, sj * ny)
                        amp[i, j] += abs(bar(uf) @ G @ ui) ** 2
            Wf += wsg * amp * s * du / (s * (1 - s))      # d2k_perp dk+/(k+ p'+): dk_y = s du, dk_x/dphi ~ s
    up_down, down_up = Wf[0, 1], Wf[1, 0]
    return Wf, (down_up - up_down) / (down_up + up_down), rest_B(p, 0.0, F)


def emission_s(F, eta, s, q=-1.0, nu=81, umax=6.0, L=None):
    """the four spin channels W[i][f] (up, down along +y) at the photon fraction s, emitted at phi = 0, integrated over k'_y"""
    L = L or 9 * F.tau
    p = lc(eta / W0, 0.0, 0.0)
    kp = dot(KAP, p)
    ny = np.array([0, 1.0, 0])
    Wf = np.zeros((2, 2))
    kx = s * (-q * F.a(0.0))
    us = np.linspace(-umax, umax, nu) * max(1.0, (eta * F.a0 / F.tau) ** (1 / 3))
    for uu in us:
        ky = s * uu
        k = lc(s * (p[0] + p[3]), kx, ky, 0.0)
        pp = lc((1 - s) * (p[0] + p[3]), -kx, -ky)
        kpp = dot(KAP, pp)
        lam = (pp + k - p)[0] / KAP[0]
        c1 = -q * pp[1] / kpp + q * p[1] / kp
        c2 = q * q / (2 * kpp) - q * q / (2 * kp)
        I = phase_integrals(F, lam, c1, c2, L)
        kk = dot(KAP, k)
        for e in (np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0])):
            es = sl(e - dot(k, e) / kk * KAP)
            G = es * I[0] + (q * EL @ KS @ es / (2 * kpp) + q * es @ KS @ EL / (2 * kp)) * I[1] + \
                q * q * EL @ KS @ es @ KS @ EL / (4 * kp * kpp) * I[2]
            for i, si in enumerate((1, -1)):
                ui = u_sp(p, si * ny)
                for j, sj in enumerate((1, -1)):
                    Wf[i, j] += abs(bar(u_sp(pp, sj * ny)) @ G @ ui) ** 2
    return Wf, (Wf[1, 0] - Wf[0, 1]) / (Wf[1, 0] + Wf[0, 1]), rest_B(p, 0.0, F)


def vlpl_emission_peq(chi):
    """VLPL (decision 93): S_inf = -(K - G)/(Xi - W) along -e2 convention: returns S_inf (along e2)"""
    def y(u): return 2 * u / (3 * (1 - u) * chi)
    def ik13(z): return quad(lambda t: kv(1 / 3, t), z, np.inf, limit=200)[0]
    w = lambda u: (u * u - 2 * u + 2) / (1 - u) * kv(2 / 3, y(u)) - ik13(y(u))
    Xi = quad(lambda u: 2 * kv(2 / 3, y(u)) - ik13(y(u)), 0, 1, limit=200)[0]
    W = quad(w, 0, 1, limit=200)[0]
    K = quad(lambda u: -u / (1 - u) * kv(1 / 3, y(u)), 0, 1, limit=200)[0]
    G = quad(lambda u: -u * kv(1 / 3, y(u)), 0, 1, limit=200)[0]
    return -(K - G) / (Xi - W)


# ------------------------------------------------------------------ 2. Breit-Wheeler pairs
def pairs(F, eta, s, pol="unpol", nps=81, pmax=6.0, L=None, spin_axis="y", q=-1.0):
    """photon along +z (k+ = eta / w0), positron fraction s = k.p+ / k.k, created at phi = 0 (p+_x = a(0));
    returns <S+ . axis>, <S- . axis> (summed over the partner's spin, rest-frame axis), the rate (unnormalized), rest B'"""
    L = L or 9 * F.tau
    k = lc(eta / W0, 0.0, 0.0, 0.0)
    kk = dot(KAP, k)
    ac = F.a(0.0)
    if spin_axis == "y":
        nax = lambda p: np.array([0, 1.0, 0])
    else:                                # helicity: along the particle's momentum
        nax = lambda p: p[1:] / np.linalg.norm(p[1:])
    pols = {"unpol": [(np.array([0, 1.0, 0, 0]), 0.5), (np.array([0, 0, 1.0, 0]), 0.5)],
            "x": [(np.array([0, 1.0, 0, 0]), 1.0)], "y": [(np.array([0, 0, 1.0, 0]), 1.0)],
            "circ": [(np.array([0, 1.0, 1j, 0]) / np.sqrt(2), 1.0)]}[pol]
    acc = np.zeros((2, 2))
    pys = np.linspace(-pmax, pmax, nps)
    for py in pys:
        pp = lc(s * (k[0] + k[3]), ac, py)            # positron (canonical label: kinetic pi_x = p_x - a = 0 at phi = 0)
        pm = lc((1 - s) * (k[0] + k[3]), -ac, -py)    # electron
        kpp, kpm = dot(KAP, pp), dot(KAP, pm)
        D = pp + pm - k
        lam = D[0] / KAP[0]
        c1 = -q * pm[1] / kpm + q * pp[1] / kpp
        c2 = q * q / (2 * kpm) + q * q / (2 * kpp)
        I = phase_integrals(F, lam, c1, c2, L)
        for e, wt in pols:
            es = sl(e)
            G0 = es
            G1 = q * EL @ KS @ es / (2 * kpm) - q * es @ KS @ EL / (2 * kpp)
            G2 = -q * q * EL @ KS @ es @ KS @ EL / (4 * kpm * kpp)
            G = G0 * I[0] + G1 * I[1] + G2 * I[2]
            for i, sp_ in enumerate((1, -1)):
                v = v_sp(pp, sp_ * nax(pp))
                for j, sm_ in enumerate((1, -1)):
                    u = u_sp(pm, sm_ * nax(pm))
                    acc[i, j] += wt * abs(bar(u) @ G @ v) ** 2
    tot = acc.sum()
    Sp = (acc[0].sum() - acc[1].sum()) / tot
    Sm = (acc[:, 0].sum() - acc[:, 1].sum()) / tot
    pp0 = lc(s * (k[0] + k[3]), ac, 0.0); pm0 = lc((1 - s) * (k[0] + k[3]), -ac, 0.0)
    # kinetic momenta at phi = 0 (collinear with the photon) for the rest-frame field
    pip = pp0.copy(); pip[1] = 0; pip = lc(pip[0] + pip[3], 0.0, 0.0)
    return Sp, Sm, tot, rest_B(pip, 0.0, F)


def vlpl_pair(chi, x, xi3=0.0):
    """K13/x - xi3 K13/x_-, K13/x_- - xi3 K13/x, p0 at (chi, x)"""
    xm = 1 - x
    r = 2 / (3 * chi * x * xm)
    k13, k23 = kv(1 / 3, r), kv(2 / 3, r)
    ik = quad(lambda t: kv(1 / 3, t), r, np.inf)[0]
    p0 = (x * x + xm * xm) / (x * xm) * k23 + ik - xi3 * k23
    return (k13 / x - xi3 * k13 / xm) / p0, (k13 / xm - xi3 * k13 / x) / p0, p0


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    np.set_printoptions(precision=5, suppress=True)
    if what in ("all", "checks"):
        print("Dirac equation residual of the Volkov solutions (q = -1):", check_dirac())
        print("Dirac equation residual of the Volkov solutions (q = +1):", check_dirac(+1.0))
        for n, r in check_spinors().items():
            print("spin n = %s: |<v_hole, i g2 u*>| = %.6f; |v vbar - (p-m)(1+g5 s)/2| = %.2e, with -s: %.2e; "
                  "|u ubar - (p+m)(1+g5 s)/2| = %.2e" % (n, *r))
        sys.stdout.flush()
    a0, tau = 10.0, 5.0
    F = Field(a0, tau)
    chi = 0.5 * a0 / tau
    if what in ("all", "emission"):
        for q in (-1.0, +1.0):
            Wf, peq, Bp = emission(F, 0.5, q=q)
            print("emission q = %+g (chi = %.2f): W[i][f] (up, down along +y) =\n%s\n  equilibrium polarization along +y %.4f; "
                  "rest-frame B' = %s" % (q, chi, Wf / Wf.sum(), peq, Bp / np.linalg.norm(Bp)))
            sys.stdout.flush()
        print("VLPL (decision 93) S_inf along e2 at chi = %.2f: %.4f (e2 = +B' for an electron, -B' for a positron)" %
              (chi, vlpl_emission_peq(chi)))
        print("VLPL S_inf at chi = 0.01: %.4f (Sokolov-Ternov -8/(5 sqrt3) = %.4f)" % (vlpl_emission_peq(0.01), -8 / (5 * np.sqrt(3))))
        sys.stdout.flush()
    if what in ("all", "pairs"):
        for s in (0.2, 0.5, 0.8):
            Sp, Sm, tot, Bp = pairs(F, 0.5, s)
            bp, bm, p0 = vlpl_pair(chi, s)
            print("pairs unpolarized, chi = %.2f, x = %.1f: <S+.y> %+.4f, <S-.y> %+.4f (rest-frame B' along %s); "
                  "|K13/x|/p0 = %.4f, |K13/x_-|/p0 = %.4f" % (chi, s, Sp, Sm, Bp / np.linalg.norm(Bp), bp, bm))
            sys.stdout.flush()
        for pol, xi3 in (("x", 1.0), ("y", -1.0)):
            Sp, Sm, tot, Bp = pairs(F, 0.5, 0.5, pol=pol)
            print("pairs photon along %s (xi3 = %+g in e1 = -x), x = 0.5: rate %.5e, <S+.y> %+.4f, <S-.y> %+.4f" % (pol, xi3, tot, Sp, Sm))
            sys.stdout.flush()
        _, _, p0x = vlpl_pair(chi, 0.5, 1.0); _, _, p0y = vlpl_pair(chi, 0.5, -1.0)
        print("  theory rate ratio y / x = (B0 + K23)/(B0 - K23) = %.4f" % (p0y / p0x))
        for s in (0.2, 0.8):
            Sp, Sm, tot, Bp = pairs(F, 0.5, s, pol="circ", spin_axis="h")
            print("pairs circular photon (x + i y)/sqrt2 along +z (helicity +1), x = %.1f: positron helicity %+.4f, electron helicity %+.4f" %
                  (s, Sp, Sm))
            sys.stdout.flush()
    if what == "em_s":
        # spin-flip channels at fixed photon fractions (no s integral): electron (q = -1) and positron (q = +1, C-conjugate field)
        for q in (-1.0, +1.0):
            for s in (0.2, 0.5, 0.8):
                Wf, peq, Bp = emission_s(F, 0.5, s, q=q)
                y = 2 * s / (3 * (1 - s) * chi)
                k13, k23 = kv(1 / 3, y), kv(2 / 3, y)
                ik = quad(lambda t: kv(1 / 3, t), y, np.inf)[0]
                w = (s * s - 2 * s + 2) / (1 - s) * k23 - ik
                # VLPL (decision 93) along e2: W(si -> sf) ~ w - u K13 si - u/(1-u) K13 sf + si sf (2 K23 - IK13)
                Wv = np.array([[w - s * k13 * si - s / (1 - s) * k13 * sf + si * sf * (2 * k23 - ik) for sf in (1, -1)] for si in (1, -1)])
                e2sign = 1 if q < 0 else -1          # e2 = +B' for an electron, -B' for a positron (here B' = +y)
                Wv_y = Wv if e2sign > 0 else Wv[::-1, ::-1]
                print("emission q = %+g, chi = %.2f, u = %.1f: (W_dd-up - W_up-dn)/(sum) along +y (= B') direct %+.4f, VLPL %+.4f;"
                      " channels direct %s, VLPL %s" % (q, chi, s, peq, (Wv_y[1, 0] - Wv_y[0, 1]) / (Wv_y[1, 0] + Wv_y[0, 1]),
                                                       np.round(Wf.ravel() / Wf.sum(), 5), np.round(Wv_y.ravel() / Wv_y.sum(), 5)))
                sys.stdout.flush()
    if what == "conv":
        F2 = Field(a0, tau)
        for L, pr in ((9 * tau, 12.0), (11 * tau, 24.0)):
            k = lc(0.5 / W0, 0, 0, 0)
            print("convergence:", L, pr, pairs(F2, 0.5, 0.5, nps=21, L=L)[:2])
