#!/usr/bin/env python3
"""Solid-density benchmark (bench/collisions_epoch): plasma parameters, dense-plasma Coulomb logarithms, the e-p relaxation
time they give, and how often Perez's low-temperature limit s <= s' acts (decision 94, docs/specs/collisions.md).

  python3 dense_plasma.py          (numpy only)

Coulomb logarithms (SI; T in J):
  NRL     NRL formulary e-i, T_e > 10 Z^2 eV: 24 - ln(n_e[cm^-3]^1/2 / T_e[eV])
  LS      Landau-Spitzer ln(lambda_De / b_C), b_C = Z e^2 / (4 pi eps0 k T_e)
  LM      Lee & More, Phys. Fluids 27, 1273 (1984): max(2, 1/2 ln(1 + b_max^2 / b_min^2)), b_max = max(lambda_DH, a_i),
          lambda_DH^-2 = n_e e^2 / (eps0 k sqrt(T_e^2 + T_F^2)) + n_i Z^2 e^2 / (eps0 k T_i),
          b_min = max(Z e^2 / (4 pi eps0 3 k T_e), hbar / (2 sqrt(3 m_e k T_e)))
  GMS6    Gericke, Murillo, Schlanges, PRE 65, 036418 (2002), model 6 as quoted by Glosli et al., PRE 78, 025401 (2008),
          eq. 3: 1/2 ln(1 + (lambda_De^2 + a_i^2) / (Lambda^2 / (8 pi) + b_C^2)), Lambda = sqrt(2 pi hbar^2 / (m_e k T_e))
  BPS     Brown, Preston, Singleton (non-degenerate, as Glosli et al. eq. 4): ln(lambda_De / Lambda) + (ln(16 pi) - gamma - 1) / 2
Relaxation (Glosli et al. eq. 1, = compare.py's NRL): dT_e/dt = -(T_e - T_i) / tau,
  1/tau = (8 sqrt(2 pi) / 3) n Z^2 e^4 lnL / ((4 pi eps0)^2 m_e m_p c^3 (kT_e/m_e c^2 + kT_i/m_p c^2)^1.5)
with lnL re-evaluated from T_e(t), T_i(t) (T_e + T_i conserved); the time at which T_e - T_i falls to 1/e.
The cap: s and s' of every sampled pair evaluated exactly as src/collision_kernels.hpp (code units), for lnL = 5 (the
benchmark) and for the automatic per-pair log of VLPL (= the Lee-More form per pair: b_min = max(classical, quantum) of the
pair, b_max^2 = max(lambda_D^2, a_i^2) with lambda_D from all species, floor 2).
"""
import math
import numpy as np

e, me, eps0, hbar, kB, c = 1.602176634e-19, 9.1093837015e-31, 8.8541878128e-12, 1.054571817e-34, 1.380649e-23, 299792458.0
MI = 1836.15267
lam = 1e-6
omega = 2 * math.pi * c / lam
nc = eps0 * me * omega ** 2 / e ** 2
n = 100 * nc                                     # m^-3, n_e = n_i
Te0, Ti0 = 100.0, 10.0                          # eV
EV = e


def params(Te, Ti):
    TeJ, TiJ = Te * EV, Ti * EV
    lDe = math.sqrt(eps0 * TeJ / (n * e * e))
    lDi = math.sqrt(eps0 * TiJ / (n * e * e))
    lD = 1 / math.sqrt(1 / lDe ** 2 + 1 / lDi ** 2)
    ai = (3 / (4 * math.pi * n)) ** (1 / 3)
    EF = hbar ** 2 * (3 * math.pi ** 2 * n) ** (2 / 3) / (2 * me)
    return dict(lDe=lDe, lDi=lDi, lD=lD, ai=ai, EF=EF / EV, TeJ=TeJ, TiJ=TiJ)


def coulomb_logs(Te, Ti):
    p = params(Te, Ti)
    TeJ, TiJ, lDe, ai = p["TeJ"], p["TiJ"], p["lDe"], p["ai"]
    bC = e * e / (4 * math.pi * eps0 * TeJ)
    TF = p["EF"] * EV
    lDH = 1 / math.sqrt(n * e * e / (eps0 * math.sqrt(TeJ ** 2 + TF ** 2)) + n * e * e / (eps0 * TiJ))
    bmax = max(lDH, ai)
    bmin = max(e * e / (4 * math.pi * eps0 * 3 * TeJ), hbar / (2 * math.sqrt(3 * me * TeJ)))
    Lam = math.sqrt(2 * math.pi * hbar ** 2 / (me * TeJ))
    return {
        "NRL": 24 - math.log(math.sqrt(n * 1e-6) / Te),
        "LS": math.log(lDe / bC),
        "LM": max(2.0, 0.5 * math.log(1 + (bmax / bmin) ** 2)),
        "GMS6": 0.5 * math.log(1 + (lDe ** 2 + ai ** 2) / (Lam ** 2 / (8 * math.pi) + bC ** 2)),
        "BPS": math.log(lDe / Lam) + (math.log(16 * math.pi) - 0.5772156649 - 1) / 2,
    }


def relax_time(lnL_of):
    """1/e time (1/omega) of T_e - T_i; lnL_of(Te, Ti) in eV"""
    Te, Ti, t, d0 = Te0, Ti0, 0.0, Te0 - Ti0
    h = 0.05 / omega
    re = e * e / (4 * math.pi * eps0 * me * c * c)
    while Te - Ti > d0 / math.e:
        y = Te / 510998.95 + Ti / (510998.95 * MI)
        # 1/tau = (8 sqrt(2 pi) / 3) n r_e^2 c lnL (m_e / m_p) / y^1.5 (Glosli et al. eq. 1 with e^2 = 4 pi eps0 r_e m_e c^2)
        rate = (8 * math.sqrt(2 * math.pi) / 3) * n * re ** 2 * c * lnL_of(Te, Ti) / MI / y ** 1.5
        d = rate * (Te - Ti) * h                     # dT_e = -d, dT_i = +d (n_e = n_i)
        Te, Ti, t = Te - d, Ti + d, t + h
    return t * omega


def cap_stats(lnL_fixed, N=2_000_000, seed=1):
    """fraction of e-p, e-e, p-p pairs whose s is limited by s' (code units, as collision_kernels.hpp)"""
    rng = np.random.default_rng(seed)
    kre = 2 * math.pi * 2.8179403262e-15 / lam
    lt = (3 * kre) ** (-1 / 3)
    bq = 0.5 * (hbar / (me * c)) * (2 * math.pi / lam)       # lambda_C k0 / 2 (reduced Compton wavelength)
    ncode = 100.0
    ue, up = math.sqrt(Te0 / 510998.95), math.sqrt(Ti0 / 510998.95 / MI)
    out = {}
    for name, (m1, s1, m2, s2) in {"ep": (1.0, ue, MI, up), "ee": (1.0, ue, 1.0, ue), "pp": (MI, up, MI, up)}.items():
        u1 = rng.normal(0, s1, (N, 3)); u2 = rng.normal(0, s2, (N, 3))
        g1 = np.sqrt(1 + (u1 ** 2).sum(1)); g2 = np.sqrt(1 + (u2 ** 2).sum(1))
        E1, E2 = m1 * g1, m2 * g2; Et = E1 + E2
        p1, p2 = m1 * u1, m2 * u2
        vc = (p1 + p2) / Et[:, None]; vc2 = (vc ** 2).sum(1)
        gc = 1 / np.sqrt(1 - vc2); t1 = gc * gc / (gc + 1)
        vp1 = (vc * p1).sum(1); vp2 = (vc * p2).sum(1)
        E1s, E2s = gc * (E1 - vp1), gc * (E2 - vp2)
        t2 = t1 * vp1 - gc * E1
        ps = np.sqrt(((p1 + t2[:, None] * vc) ** 2).sum(1))
        X = E1s * E2s / ps ** 2 + 1
        if lnL_fixed > 0:
            L = np.full(N, lnL_fixed)
        else:
            b0 = kre * gc / Et * X; bqm = bq / ps
            bmin = np.maximum(b0, bqm)
            L = np.maximum(0.5 * np.log(1 + DEBYE2 / bmin ** 2), 2.0)
        s = kre * L * gc * ps * X * X / (m1 * m2 * Et * g1 * g2)
        vrel = ps * Et / (gc * E1s * E2s)
        n23 = ncode ** (2 / 3)
        smax = lt * (m1 + m2) * vrel / max(m1 * n23, m2 * n23)
        capped = s > smax
        vth = math.sqrt(s1 ** 2 + s2 ** 2)                  # rms relative speed per component (u ~ v here)
        thr = np.median(vrel[capped & (np.abs(s / smax - 1) < 0.02)]) / vth if (capped & (np.abs(s / smax - 1) < 0.02)).any() else float("nan")
        # weight of a pair in the relaxation: s p*^2 (for s << 1 a pair exchanges <1 - cos chi> p*^2 ~ s p*^2 of
        # momentum squared; the sum of s alone is dominated by the slowest pairs, s ~ v^-3)
        w, wc = s * ps ** 2, np.minimum(s, smax) * ps ** 2
        out[name] = dict(frac=capped.mean(), lost=1 - wc.sum() / w.sum(), thr=thr, lnL=float(np.mean(L)))
    return out


# Debye length^2 in 1/k0^2 for code units: lambda_D^2 = eps0 T / (n e^2); with n = N n_c, n_c = eps0 m omega^2 / e^2:
# lambda_D^2 k0^2 = (T / m c^2) / N (k0 = omega / c). Sum over species: 1 / sum(N_s / T_s); floor a_i^2 = (3 kre / N)^(2/3).
_kre = 2 * math.pi * 2.8179403262e-15 / lam
DEBYE2 = max(1 / (100 / (Te0 / 510998.95) + 100 / (Ti0 / 510998.95)), (3 * _kre / 100) ** (2 / 3))

if __name__ == "__main__":
    p = params(Te0, Ti0)
    ai, lDe, lDi, lD = p["ai"], p["lDe"], p["lDi"], p["lD"]
    print("n_e = n_i = %.3e cm^-3, T_e = %g eV, T_i = %g eV" % (n * 1e-6, Te0, Ti0))
    print("lambda_De %.3g nm, lambda_Di %.3g nm, lambda_D (e + i) %.3g nm, a_i %.3g nm" % (lDe * 1e9, lDi * 1e9, lD * 1e9, ai * 1e9))
    G = lambda T: e * e / (4 * math.pi * eps0 * ai * T * EV)
    print("coupling Gamma_ee = Gamma_ei = %.3f (at T_e), Gamma_ii = %.3f (at T_i); E_F = %.2f eV, T_e / T_F = %.1f"
          % (G(Te0), G(Ti0), p["EF"], Te0 / p["EF"]))
    print("electrons in the Debye sphere N_D = %.1f" % (4 * math.pi / 3 * n * lDe ** 3))
    L0 = coulomb_logs(Te0, Ti0)
    print("Coulomb logarithms at t = 0: " + ", ".join("%s %.2f" % kv for kv in L0.items()))
    print("1/e time of T_e - T_i (1/omega), lnL re-evaluated with T(t):")
    tref = {}
    for k in ["NRL", "LS", "LM", "GMS6", "BPS"]:
        tref[k] = relax_time(lambda Te, Ti, k=k: coulomb_logs(Te, Ti)[k])
        print("  %-5s %7.1f" % (k, tref[k]))
    print("  lnL = 5 (fixed, the benchmark) %7.1f" % relax_time(lambda Te, Ti: 5.0))
    print("Perez's limit s <= s' (pairs sampled from the initial Maxwellians, kernel formulas):")
    for lab, L in [("lnL = 5", 5.0), ("automatic lnL", 0.0)]:
        st = cap_stats(L)
        for b, v in st.items():
            print("  %-14s %s: capped %.3f of the pairs, %.3f of sum s p*^2 removed, <lnL> %.2f, cap sets in at v_rel = %.2f v_th,rel"
                  % (lab, b, v["frac"], v["lost"], v["lnL"], v["thr"]))
