"""Von Neumann analysis of the RIP-CC step with a uniform dielectric chi = eps - 1 (decision 100, docs/specs/rz_dielectric.md).

The 2D TM analogue of FieldsRZ::pass (c dt = dz = 1): E_z at nodes, E_r (E_y) and B_phi (B_x) at half nodes in r, the transverse
derivatives i K (K^2 = the radial eigenvalue of the mode, K = 2 sin(k dr / 2) / dr on a uniform Cartesian grid). E_z is local:
(1 + chi) in the predictor and the corrector. The options differ in the polarization current of E_r at i + 1/2:
  cn   : chi mu_z(E^{n+1} - E^n)                  (implemented: tridiagonal line along z)
  jacN : cn, N Jacobi sweeps from the lagged guess
  lag  : chi mu_z(E^n - E^{n-1})                  (explicit, first order)
  lag2 : chi mu_z(2 E^n - 3 E^{n-1} + E^{n-2})    (explicit, second order)
usage: python3 vonneumann.py [table | tm01]
  table: stability (max |g| - 1 over k_z dz in (0, pi], K dz in [0, 2]), Q and dv/v of the forward wave, stability thresholds
  tm01 : the prediction of test D2 (tests/test_rz_dielectric.cpp): w sqrt(eps) / w(vacuum) - 1 of TM01 in a filled tube
"""
import sys
import numpy as np


def amp(chi, tz, K, opt):
    """amplification matrix of the state (Ez, Er, Bp, Er^{n-1}, Er^{n-2}) for e^{i tz i}"""
    s = np.exp(1j * tz)
    mzm, mzp = (1 + 1 / s) / 2, (1 + s) / 2       # mu_z at i - 1/2, i + 1/2 relative to node i
    A = np.zeros((5, 5), complex)
    for c in range(5):
        x = np.zeros(5, complex)
        x[c] = 1
        Ez, Er, Bp, E1, E2 = x
        Ezs = Ez + 0.5 * (1j * K * Bp) / (1 + chi)
        tp = (Er + Bp) / s + (1j * K * Ezs) * mzm   # the vacuum transport (characteristics with their sources)
        tm = (Er - Bp) * s - (1j * K * Ezs) * mzp
        Erv, Bpv = 0.5 * (tp + tm), 0.5 * (tp - tm)
        aE, aB = 0.5 * chi * (mzm + mzp), 0.5 * chi * (mzm - mzp)   # node i gets -(y_- + y_+)/2, -(y_- - y_+)/2
        if opt == 'cn':
            JE = (Erv - Er) / (1 + aE)
        elif opt.startswith('jac'):
            JE, dg = Er - E1, 0.5 * chi
            for _ in range(int(opt[3:])):
                JE = (Erv - Er - (aE - dg) * JE) / (1 + dg)
        elif opt == 'lag':
            JE = Er - E1
        else:
            JE = 2 * Er - 3 * E1 + E2
        Ern, Bpn = Erv - aE * JE, Bpv - aB * JE
        Ezn = Ez + 1j * K * 0.5 * (Bp + Bpn) / (1 + chi)
        A[:, c] = [Ezn, Ern, Bpn, Er, E1]
    return A


def forward(chi, tz, K, opt):
    """(w dt, |g|) of the eigenvalue closest to the continuum w = sqrt(tz^2 + K^2) / sqrt(eps)"""
    w = np.sqrt(tz**2 + K**2) / np.sqrt(1 + chi)
    ev = [e for e in np.linalg.eigvals(amp(chi, tz, K, opt)) if abs(e) > 1e-6]
    e = min(ev, key=lambda e: abs(abs(np.angle(e)) - w))
    return abs(np.angle(e)), abs(e)


def gmax(chi, opt, nk=41, nt=200):
    return max(abs(np.linalg.eigvals(amp(chi, tz, K, opt))).max()
               for K in np.linspace(0, 2.0, nk) for tz in np.linspace(1e-3, np.pi, nt)) - 1


def table():
    for opt in ['cn', 'jac1', 'jac2', 'jac3', 'lag', 'lag2']:
        print(opt, 'max|g|-1 (chi .003 .05 .1 .3 1):', ' '.join('%9.1e' % gmax(c, opt) for c in [0.003, 0.05, 0.1, 0.3, 1.0]))
        for tz in [0.02, 0.1]:
            row = []
            for chi in [0.003, 0.05, 0.1, 0.3]:
                wn, g = forward(chi, tz, 0.0, opt)
                wd = tz / np.sqrt(1 + chi)
                Q = np.inf if g >= 1 - 1e-14 else wd / (2 * -np.log(g))
                row.append('Q=%9.3g dv/v=%9.2e' % (Q, wn / wd - 1))
            print('   kdz=%4.2f: ' % tz + ' | '.join(row))
    for opt, lo, hi in [('lag2', 0.3, 0.5), ('lag', 1.0, 3.0), ('jac2', 1.0, 3.0)]:
        for _ in range(14):
            c = 0.5 * (lo + hi)
            if gmax(c, opt) > 1e-12:
                hi = c
            else:
                lo = c
        print(opt, 'stable up to chi =', round(lo, 3))


def tm01():
    dz, K = 0.04, 2.404825557695773 / 1.6 * 0.04       # the grid of test D2: R = 32 x 0.05, dz = 0.04
    for tz in [2 * np.pi / 64, 2 * np.pi * 4 / 64]:
        w1 = forward(0.0, tz, K, 'cn')[0]
        print('kz dz %.3f' % tz, ' '.join('eps %.2f: %.3e' % (1 + c, forward(c, tz, K, 'cn')[0] * np.sqrt(1 + c) / w1 - 1)
                                          for c in [0.1, 1.25, 3.0]))


if __name__ == '__main__':
    (tm01 if len(sys.argv) > 1 and sys.argv[1] == 'tm01' else table)()
