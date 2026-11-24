#!/usr/bin/env python3
"""First-generation pair yield of examples/qed_cascade.deck (decision 96): an estimate independent of the code.

Each beam electron meets the pulse head-on (plane wave: the Rayleigh length 35 um is long against the overlap). It loses energy
deterministically, d gamma/dt = -P_alpha chi^2 g(chi) / (m c^2), and emits photons with the LCFA spectrum; a photon of energy
om emitted at t decays with the probability 1 - exp(-int_t^inf W_BW(om, chi_gamma(t')) dt'), W_BW = Cg (T0 - xi3 T3). Two
variants: unpolarized photons (xi3 = 0) and photons with the mean linear polarization of their emission, xi3 = K23(y) / w(u)
(along the field force; the axis stays the same in a linearly polarized pulse; no selection afterwards). Pairs per electron =
int dt int du dN/(du dt) P_decay. The beam average takes the transverse intensity profile of the waist over the 1 um x 1 um
beam. Not counted: pairs from photons of the pairs (later generations), the spread of the electron energy (straggling).

Rates in code units (omega0 = c = 1), as qed.hpp: chi_e = 2 gamma hw a |cos(2t)|, chi_gamma = 2 om hw a |cos(2t)|.
    python3 examples/qed_cascade_estimate.py [a0 [gamma [duration_fs [waist_um]]]]
"""
import sys
import numpy as np
from scipy.special import kv
from scipy.integrate import quad

ALPHA = 7.2973525693e-3
lam = 0.8e-6
hw = 2 * np.pi * 197.3269804e-15 / lam / 0.51099895000
a0 = float(sys.argv[1]) if len(sys.argv) > 1 else 80.0
g0 = float(sys.argv[2]) if len(sys.argv) > 2 else np.sqrt(1 + 19569.5 ** 2)
tau = (float(sys.argv[3]) if len(sys.argv) > 3 else 8.0) * 1e-15 * 2 * np.pi * 2.99792458e8 / lam   # intensity FWHM, 1/omega0
w0 = (float(sys.argv[4]) if len(sys.argv) > 4 else 3.0) * 1e-6


def ik13(y):
    return quad(lambda t: kv(1 / 3, t), y, np.inf, limit=200)[0]


def bw_T3(chi):                      # int_0^1 K23(rho) dx
    return 2 * quad(lambda x: kv(2 / 3, 2 / (3 * chi * x * (1 - x))), 1e-12, 0.5, limit=400)[0]


def bw_T0(chi):                      # int_0^1 [(x^2 + (1-x)^2)/(x(1-x)) K23(rho) + IK13(rho)] dx, rho = 2/(3 chi x (1-x))
    f = lambda x: ((x * x + (1 - x) ** 2) / (x * (1 - x)) * kv(2 / 3, 2 / (3 * chi * x * (1 - x))) + ik13(2 / (3 * chi * x * (1 - x))))
    return 2 * quad(f, 1e-12, 0.5, limit=400)[0]


# IK13 on a grid in ln y (interpolated; the quadrature per value would be slow)
ly = np.linspace(np.log(1e-9), np.log(600), 500)
IKt = np.log(np.array([ik13(np.exp(v)) for v in ly]))


def ik13v(y):
    return np.where(y < 600, np.exp(np.interp(np.log(np.clip(y, 1e-9, 600)), ly, IKt)), 0.0)


def em_w(u, chi):                    # (u^2 - 2u + 2)/(1 - u) K23(y) - IK13(y), y = 2u / (3 (1 - u) chi)
    y = 2 * u / (3 * (1 - u) * chi)
    i13 = ik13v(y)
    return np.where(y < 700, (u * u - 2 * u + 2) / (1 - u) * kv(2 / 3, np.minimum(y, 700)) - i13, 0.0)


# tables in chi: T0 (pairs), the power int u w du (energy loss)
lc = np.linspace(np.log(0.02), np.log(200), 90)
T0t = np.array([bw_T0(np.exp(l)) for l in lc])
T3t = np.array([bw_T3(np.exp(l)) for l in lc])
ug = np.concatenate([np.logspace(-7, np.log10(0.5), 120), 1 - np.logspace(np.log10(0.5), -9, 120)[1:]])
Pt = []
for l in lc:
    wv = em_w(ug, np.exp(l))
    Pt.append(np.trapezoid(ug * wv, ug))
Pt = np.array(Pt)


def interp_log(tab, chi):
    c = np.clip(np.log(np.maximum(chi, 1e-300)), lc[0], lc[-1])
    v = np.exp(np.interp(c, lc, np.log(np.maximum(tab, 1e-300))))
    return np.where(chi >= np.exp(lc[0]), v, 0.0)


def yield_at(a):
    """pairs per electron (unpolarized photons, polarized as emitted) and the final gamma for the peak amplitude a"""
    t = np.arange(-1.6 * tau, 1.6 * tau, 0.01)
    dt = t[1] - t[0]
    env = a * np.exp(-2 * np.log(2) * (2 * t / tau) ** 2)            # envelope at the electron (it meets the pulse at 2c)
    F = 2 * hw * env * np.abs(np.cos(2 * t))                          # chi / energy
    # photon decay: G(om, t) = int_t^inf W dt', W = alpha / (sqrt3 pi hw om) T0(om F)
    om = np.logspace(np.log10(2.0), np.log10(g0), 160)
    cg = ALPHA / (np.sqrt(3) * np.pi * hw * om[None, :])
    G0 = np.cumsum((cg * interp_log(T0t, om[None, :] * F[:, None]))[::-1], axis=0)[::-1] * dt
    G3 = np.cumsum((cg * interp_log(T3t, om[None, :] * F[:, None]))[::-1], axis=0)[::-1] * dt
    g = g0
    pairs = np.zeros(2)
    for k in range(len(t)):
        chi = g * F[k]
        if chi < 1e-3:
            continue
        C0 = ALPHA / (np.sqrt(3) * np.pi * hw * g)
        ok = ug * g > 2
        if ok.any():
            uu = ug[ok]
            wv = em_w(uu, chi)
            sp = C0 * wv                                              # dN / (du dt)
            y = 2 * uu / (3 * (1 - uu) * chi)
            xi3 = np.where(wv > 0, kv(2 / 3, np.minimum(y, 700)) / np.maximum(wv, 1e-300), 0.0)
            g0k = np.interp(np.log(uu * g), np.log(om), G0[k])
            g3k = np.interp(np.log(uu * g), np.log(om), G3[k])
            pairs[0] += np.trapezoid(sp * (1 - np.exp(-g0k)), uu) * dt
            pairs[1] += np.trapezoid(sp * (1 - np.exp(-(g0k - xi3 * g3k))), uu) * dt
        g -= g * C0 * interp_log(Pt, chi) * dt                        # d gamma/dt = -gamma C0 int u w du
        g = max(g, 1.0)
    return pairs[0], pairs[1], g


# beam average over the 1 um x 1 um cross-section in the waist w0 (a = a0 exp(-r^2 / w0^2))
r1 = np.linspace(-0.4e-6, 0.4e-6, 5)
res = []
for y in r1:
    for z in r1:
        res.append(yield_at(a0 * np.exp(-(y * y + z * z) / w0 ** 2)))
res = np.array(res)
p_axis, q_axis, g_axis = yield_at(a0)
print("a0 %g, gamma %g, duration %.2f fs, waist %.2f um: chi_e,max = %.2f" % (a0, g0, tau / (2 * np.pi * 2.99792458e8 / lam) * 1e15,
                                                                               w0 * 1e6, 2 * g0 * hw * a0))
print("first-generation pairs per electron: unpolarized photons: on the axis %.4f, beam average %.4f; photons polarized as "
      "emitted: on the axis %.4f, beam average %.4f" % (p_axis, res[:, 0].mean(), q_axis, res[:, 1].mean()))
print("final gamma (deterministic): on the axis %.0f, beam average %.0f (radiated %.3f of the energy)" % (g_axis, res[:, 2].mean(),
      1 - res[:, 2].mean() / g0))
