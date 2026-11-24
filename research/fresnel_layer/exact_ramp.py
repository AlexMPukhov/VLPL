# exact reflection from a linear ramp eps = 1 - n x/L (0 < x < L), then eps = 1 - n: characteristic matrix
# dM/dx = M A(x), A = [[0, -i k/y], [-i k y, 0]] (p: k/y = eps, k y = (eps - s^2)/eps; s: 1, eps - s^2), integrated
# with RK4 along x = tau - i delta sin(pi tau / L): below the pole eps = 0, i.e. the limit of a vanishing collision
# rate nu -> 0+ (eps = 1 - n x/L (1 - i nu): the zero lies above the real axis). Units k0 = 1.
import numpy as np, math
def r_ramp(n, ang, pol, Lk, N=20000, delta=None):
    s = math.sin(math.radians(ang)); delta = 0.3 * Lk / n if delta is None else delta
    def A(tau):
        x = tau - 1j * delta * math.sin(math.pi * tau / Lk); dx = 1 - 1j * delta * math.pi / Lk * math.cos(math.pi * tau / Lk)
        e = 1 - n * x / Lk
        a, b = (e, (e - s * s) / e) if pol == "p" else (1, e - s * s)
        return np.array([[0, -1j * a], [-1j * b, 0]]) * dx
    M = np.eye(2, dtype=complex); h = Lk / N
    for i in range(N):
        t = i * h
        k1 = M @ A(t); k2 = (M + h / 2 * k1) @ A(t + h / 2); k3 = (M + h / 2 * k2) @ A(t + h / 2); k4 = (M + h * k3) @ A(t + h)
        M = M + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    eN = 1 - n; c = math.cos(math.radians(ang)); kN = np.sqrt(complex(eN - s * s)); kN = -kN if kN.imag < 0 else kN
    y0, yN = (c, kN / eN) if pol == "p" else (c, kN)
    B, C = M @ np.array([1, yN])
    return (y0 * B - C) / (y0 * B + C), (y0 - yN) / (y0 + yN)
if __name__ == "__main__":
    for N, d in ((5000, None), (20000, None), (20000, 0.1)):
        for ang in (30, 45):
            for pol in "ps":
                Lk = 2 * math.pi * 0.25
                r, r0 = r_ramp(9, ang, pol, Lk, N, None if d is None else d * Lk / 9)
                print(N, d, ang, pol, "|r| %.6f  arg(r/r_step) %+.6f" % (abs(r), np.angle(r / r0)))
