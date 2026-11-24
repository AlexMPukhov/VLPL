"""1D transverse dispersion of the box step + Crank-Nicolson cold response (O7 medium), derived from the transport:
T+_{i+1}^{n+1} = T+_i^n - D J_{i+1/2},  T-_i^{n+1} = T-_{i+1}^n - D J_{i+1/2},  Et = (T+_i + T-_{i+1})/2 - D J/2,
J = (J^n + J^{n+1})/2,  J^{n+1} - J^n = D wp2 Et.  Plane wave s = e^{i k D}, z = e^{-i w D}."""
import numpy as np
from scipy.optimize import brentq


def F(th, ph, D, wp2):
    s, z = np.exp(1j * ph), np.exp(-1j * th)
    br = 1 / (s * z - 1) + s / (z - s)
    val = 1 + 1j * (D * D * wp2 / 4) / np.tan(th / 2) * (1 + br)
    return val


def solve(kD, D, wp2):
    f = lambda th: F(th, kD, D, wp2).real if abs(F(th, kD, D, wp2).imag) < 1e-6 * (1 + abs(F(th, kD, D, wp2).real)) else F(th, kD, D, wp2).real
    ths = np.linspace(1e-4, np.pi - 1e-4, 20001)
    v = np.array([F(t, kD, D, wp2) for t in ths])
    i = np.where(np.sign(v.real[:-1]) != np.sign(v.real[1:]))[0]
    return [brentq(lambda t: F(t, kD, D, wp2).real, ths[j], ths[j + 1]) for j in i], np.abs(v.imag).max()


if __name__ == "__main__":
  for D, kD in ((0.3, 2 * np.pi / 64), (0.3, 8 * np.pi / 64), (1.0, 2 * np.pi / 64)):
      sh = np.sinc(kD / 2 / np.pi) ** 6
      r, im = solve(kD, D, sh)
      g = lambda w: np.tan(w * D / 2) ** 2 - np.tan(kD / 2) ** 2 - D ** 2 / 4 * sh / np.cos(w * D / 2) ** 2
      d76 = brentq(g, 1e-9, np.pi / D * (1 - 1e-12)) * D
      print(f"D={D} kD={kD:.3f}: roots w dt {r} (max |Im F| {im:.1e}); decision-76 formula {d76:.6f}")
