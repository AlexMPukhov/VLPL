#!/usr/bin/env python3
"""Independent references for field ionization (docs/specs/field_ionization.md, decision 86). Pure Python (no numpy).

The ADK rate is implemented here from the formula in atomic units (not from the C++ code), and the rate equations
dN_k/dt = w_{k-1} N_{k-1} - w_k N_k are integrated with RK4 on a fine time grid.

  python3 tools/ionization_ref.py rates            # rates of a few levels (SI, 1/s)
  python3 tools/ionization_ref.py pulse            # charge-state distribution after the Gaussian pulse of test I3
  python3 tools/ionization_ref.py pulse --discrete # the same sampled once per time step (the MC scheme's field)

The numbers printed by `pulse` are the reference stored in tests/test_ionization.cpp (test I3).
"""
import math
import os
import re
import sys

HARTREE_EV = 27.211386245988
E_AU = 5.14220674763e11          # V/m
OMEGA_AU = 4.1341373335e16       # 1/s
MC2_EV = 0.51099895000e6
C_SI = 299792458.0


def table(path=None):
    """{symbol: (Z, mass_u, [(I_eV, l), ...])} from data/ionization_energies.def"""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ionization_energies.def')
    t = {}
    for line in open(path):
        m = re.match(r'ELEMENT\((\w+),\s*(\d+),\s*([\d.]+)\)', line)
        if m:
            t[m.group(1)] = (int(m.group(2)), float(m.group(3)), [])
        m = re.match(r'LEVEL\((\w+),\s*(\d+),\s*([\d.]+),\s*(\d+)\)', line)
        if m:
            lev = t[m.group(1)][2]
            assert int(m.group(2)) == len(lev)
            lev.append((float(m.group(3)), int(m.group(4))))
    return t


def adk(ip_ev, zafter, l, E_vm, m=0, tong_lin=False, alpha=6.0):
    """ADK (l* = n* - 1) static rate [1/s]; the expression is kept at its maximum beyond its peak."""
    if E_vm <= 0:
        return 0.0
    I = ip_ev / HARTREE_EV
    ns = zafter / math.sqrt(2 * I)
    F0 = (2 * I) ** 1.5
    C2 = 2 ** (2 * ns) / (ns * math.gamma(2 * ns))
    f = (2 * l + 1) * math.factorial(l + m) / (2 ** m * math.factorial(m) * math.factorial(l - m))
    p = 2 * ns - m - 1
    tl = alpha * zafter ** 2 / (I * F0) if tong_lin else 0.0

    def w(F):
        return C2 * f * I * (2 * F0 / F) ** p * math.exp(-2 * F0 / (3 * F) - tl * F)
    # peak of the expression in F: tl F^2 + p F - 2 F0 / 3 = 0
    if tl > 0:
        Fp = (-p + math.sqrt(p * p + 4 * tl * 2 * F0 / 3)) / (2 * tl)
    else:
        Fp = 2 * F0 / (3 * p) if p > 0 else 1e300
    F = min(E_vm / E_AU, Fp)
    return OMEGA_AU * w(F)


def rk4_chain(N, rates, h):
    """one RK4 step of dN_k/dt = w_{k-1} N_{k-1} - w_k N_k (rates[k] = w_k, last state absorbing)"""
    def f(x):
        d = [0.0] * len(x)
        for k in range(len(x) - 1):
            r = rates[k] * x[k]
            d[k] -= r
            d[k + 1] += r
        return d
    k1 = f(N)
    k2 = f([a + 0.5 * h * b for a, b in zip(N, k1)])
    k3 = f([a + 0.5 * h * b for a, b in zip(N, k2)])
    k4 = f([a + h * b for a, b in zip(N, k3)])
    return [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(N, k1, k2, k3, k4)]


# Test I3: carbon from C0 in a linearly polarised Gaussian pulse E(t) = E0 exp(-((t - t0)/tau)^2) cos(omega0 (t - t0)),
# lambda0 = 0.8 um, E0 = 0.055 m c omega0 / e, tau = 4 cycles (code units: omega0 = 1, t0 = 3 tau), until t = 6 tau.
PULSE = dict(element='C', zmax=4, E0=0.055, tau=4 * 2 * math.pi, lam=0.8e-6)


def pulse(discrete_dt=None, P=PULSE, n=400000):
    el = table()[P['element']]
    lev = el[2]
    omega0 = 2 * math.pi * C_SI / P['lam']
    E0si = 2 * math.pi * MC2_EV / P['lam']          # m c omega0 / e [V/m]
    tau, t0, T = P['tau'], 3 * P['tau'], 6 * P['tau']
    zmax = P['zmax']
    N = [1.0] + [0.0] * zmax

    def rates(t):
        E = abs(P['E0'] * math.exp(-((t - t0) / tau) ** 2) * math.cos(t - t0)) * E0si
        return [adk(lev[k][0], k + 1, lev[k][1], E) / omega0 for k in range(zmax)]
    if discrete_dt:            # E held at E(t_n) over [t_n, t_n+1): the field the MC scheme samples
        n = int(round(T / discrete_dt))
        for i in range(n):
            w = rates(i * discrete_dt)
            sub = 50
            for _ in range(sub):
                N = rk4_chain(N, w, discrete_dt / sub)
        return N
    pass
    h = T / n
    for i in range(n):
        t = i * h
        wa, wm, wb = rates(t), rates(t + 0.5 * h), rates(t + h)
        # RK4 with time-dependent rates
        def f(x, w):
            d = [0.0] * len(x)
            for k in range(len(x) - 1):
                r = w[k] * x[k]
                d[k] -= r
                d[k + 1] += r
            return d
        k1 = f(N, wa)
        k2 = f([a + 0.5 * h * b for a, b in zip(N, k1)], wm)
        k3 = f([a + 0.5 * h * b for a, b in zip(N, k2)], wm)
        k4 = f([a + h * b for a, b in zip(N, k3)], wb)
        N = [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(N, k1, k2, k3, k4)]
    return N


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'rates'
    if what == 'rates':
        t = table()
        for sym, k, E in [('H', 0, 1e10), ('H', 0, 5e10), ('He', 0, 1e11), ('Ar', 0, 5e10), ('Ar', 7, 1e12), ('N', 5, 5e12)]:
            ip, l = t[sym][2][k]
            print('%-2s %2d  E = %.3e V/m  w = %.10e 1/s  (Tong-Lin: %.10e)' % (sym, k, E, adk(ip, k + 1, l, E),
                                                                                adk(ip, k + 1, l, E, tong_lin=True)))
    elif what == 'pulse':
        dt = None
        if '--discrete' in sys.argv:
            dt = float(sys.argv[sys.argv.index('--discrete') + 1])
        P = dict(PULSE)
        if '--E0' in sys.argv:
            P['E0'] = float(sys.argv[sys.argv.index('--E0') + 1])
        n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 400000
        N = pulse(dt, P, n)
        print('charge states 0..%d after the pulse: ' % (len(N) - 1) + ' '.join('%.6f' % x for x in N))
