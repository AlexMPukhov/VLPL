"""Longitudinal and dipole wake of a rigid Gaussian bunch in a round corrugated pipe (examples/rz_corrugated_pipe.deck)
from the RZ wake probes (wake axis, wake offset), against Bane-Stupakov (leading order in h/a, p/a) and the modal
wavenumbers of the pipe with the slots as radial transmission lines.

  python3 corrugated_wake.py <output dir> [x0_um=50] [Q_pC=-100] [sigma_um=100] [xi_c_um=2500] [a h p g in um=1000 50 40 20]

Point-charge wakes (Bane and Stupakov, round pipe; s > 0 behind the source):
  w_z(s) = Z0 c / (pi a^2) cos(k0 s),   k0 = sqrt(2 p / (a h g))
  w_x(s) = 2 Z0 c / (pi a^4) sin(k1 s) / k1  (per unit offset; the universal slope 2 Z0 c / (pi a^4) at s -> 0)
The dipole wavenumber depends on the wall. An isotropic surface impedance (resistive wall, thin dielectric layer) gives
k1 = sqrt(4 p / (a h g)) = sqrt(2) k0 (printed as "iso"). A corrugated wall shorts E_phi (on the teeth, and in the
grooves E_phi is evanescent, cutoff pi / g): with E_z = A x inside at v = c (then H_z = -A y and a uniform E_perp part
fixed by E_phi(a) = 0), E_z / (Z0 H_phi) at r = a is 2 i / (k a) / (1 - 2 / (k a)^2), the m = 0 value up to
O(1 / (k a)^2), so k1^2 = 2 p / (a h g) + 2 / a^2 (my derivation, decision 105 notes; the default reference here).
Modal: the slots as radial lines shorted at b = a + h, averaged over the period (field in the slots, zero on the teeth):
(g / p) T(k) = 2 / (k a) for m = 0 and 2 / (k a) / (1 - 2 / (k a)^2) for m = 1, T = [J0(ka) Y0(kb) - Y0(ka) J0(kb)] /
[J1(ka) Y0(kb) - Y1(ka) J0(kb)] (T -> k h for small h).
Per output block of the probe files: the loss factor k = -(1/Q) int lambda E_z dxi and the kick factor
k_x = (1/(Q x0)) int lambda F_x dxi from the simulated fields and from the convolution of the point wakes with the bunch;
the wavenumber, amplitude and phase fitted to E_z, F_x behind the bunch (s > 5 sigma) against k0, k1 and the amplitudes
Z0 c / (pi a^2) exp(-k0^2 sigma^2 / 2), 2 Z0 c / (pi a^4 k1) exp(-k1^2 sigma^2 / 2).
"""
import os
import sys
import numpy as np
from scipy.optimize import brentq, curve_fit
from scipy.special import j0, j1, y0, y1

Z0, CL = 376.730313668, 299792458.0


def blocks(path):
    """the data blocks of a wake probe file: [(step, array)]"""
    out, cur, step = [], [], None
    for line in open(path):
        if line.startswith("# step"):
            if cur:
                out.append((step, np.array(cur)))
            cur, step = [], int(line.split()[2])
        elif line.strip() and not line.startswith("#"):
            cur.append([float(v) for v in line.split()])
    if cur:
        out.append((step, np.array(cur)))
    return out


def modal_k(m, a, h, p, g):
    b = a + h
    T = lambda k: (j0(k * a) * y0(k * b) - y0(k * a) * j0(k * b)) / (j1(k * a) * y0(k * b) - y1(k * a) * j0(k * b))
    f = lambda k: 2 / (k * a) / (1 - 2 / (k * a) ** 2 if m == 1 else 1.0) - g / p * T(k)
    kb = np.sqrt(2 * p / (a * h * g) + (2 / a**2 if m == 1 else 0.0))
    return brentq(f, 0.5 * kb, min(1.5 * kb, 0.99 * 2.404825557695773 / h))


def main():
    a_ = sys.argv[1:]
    d = a_[0]
    x0 = float(a_[1]) * 1e-6 if len(a_) > 1 else 50e-6
    Q = float(a_[2]) * 1e-12 if len(a_) > 2 else -100e-12
    sig = float(a_[3]) * 1e-6 if len(a_) > 3 else 100e-6
    xic = float(a_[4]) * 1e-6 if len(a_) > 4 else 2500e-6
    a, h, p, g = (float(v) * 1e-6 for v in a_[5:9]) if len(a_) > 8 else (1e-3, 50e-6, 40e-6, 20e-6)
    k0, k1 = np.sqrt(2 * p / (a * h * g)), np.sqrt(2 * p / (a * h * g) + 2 / a**2)
    k1iso = np.sqrt(4 * p / (a * h * g))
    k0m, k1m = modal_k(0, a, h, p, g), modal_k(1, a, h, p, g)
    Az, Ax = Z0 * CL / (np.pi * a**2), 2 * Z0 * CL / (np.pi * a**4)
    print("# a h p g = %.4g %.4g %.4g %.4g um; sigma_z %.4g um, x0 %.4g um, Q %.4g pC" % (a * 1e6, h * 1e6, p * 1e6, g * 1e6, sig * 1e6, x0 * 1e6, Q * 1e12))
    print("# Bane-Stupakov k0 = %.6g /mm (lambda %.4g um), k1 = %.6g /mm (lambda %.4g um; iso %.6g /mm); modal k0 = %.6g, k1 = %.6g /mm"
          % (k0 * 1e-3, 2 * np.pi / k0 * 1e6, k1 * 1e-3, 2 * np.pi / k1 * 1e6, k1iso * 1e-3, k0m * 1e-3, k1m * 1e-3))
    ax, off = blocks(os.path.join(d, "wake_axis.txt")), blocks(os.path.join(d, "wake_offset.txt"))
    print("# step  z_bunch[mm]  k_loss sim / BS / BS(modal k0) [kV/pC/m]  k_x sim / BS / BS(modal k1) / BS(iso k1) [kV/pC/m/mm]"
          "  fit behind: k0 [/mm], Ez amp / BS; k1 [/mm], Fx amp / BS")
    for (st, A), (_, B) in zip(ax, off):
        xi = A[:, 0] * 1e-6
        dxi = abs(xi[1] - xi[0])
        lam = np.exp(-0.5 * ((xi - xic) / sig) ** 2)
        lam /= lam.sum() * dxi
        Ez = A[:, 1]
        # the dipole: F_x on the axis (m = 0 has no transverse force there; the leading-order dipole force is uniform);
        # the offset probe also carries the m = 0 radial force (DIPOLE=offset uses it)
        Fx = B[:, 2] if os.environ.get("DIPOLE") == "offset" else A[:, 2]
        kl = -np.sum(lam * Ez) * dxi / Q
        kx = np.sum(lam * Fx) * dxi / (Q * x0)
        # the convolutions: field at xi from the sources ahead (xi' > xi), s = xi' - xi
        def conv(w):
            out = np.zeros_like(xi)
            for i in range(len(xi)):
                s = xi - xi[i]
                m = s > 0
                out[i] = np.sum(w(s[m]) * lam[m]) * dxi + 0.5 * w(np.array([0.0]))[0] * lam[i] * dxi
            return out
        res = []
        for kk, kkx in ((k0, k1), (k0m, k1m), (k0, k1iso)):
            Ezt = -Q * conv(lambda s: Az * np.cos(kk * s))
            Fxt = Q * x0 * conv(lambda s: Ax * np.sin(kkx * s) / kkx)
            res.append((-np.sum(lam * Ezt) * dxi / Q, np.sum(lam * Fxt) * dxi / (Q * x0)))
        s = xic - xi
        m = (s > 5 * sig) & (xi > xi.min() + 50e-6)
        fz, fx = (np.nan,) * 2, (np.nan,) * 2
        if m.sum() > 20:
            cz = lambda s, A, k, ph: A * np.cos(k * s + ph)
            pz, _ = curve_fit(cz, s[m], Ez[m] / (-Q), p0=[Az * np.exp(-0.5 * (k0 * sig) ** 2), k0, 0.0])
            px, _ = curve_fit(cz, s[m], Fx[m] / (Q * x0), p0=[Ax / k1 * np.exp(-0.5 * (k1 * sig) ** 2), k1, -np.pi / 2])
            if pz[0] < 0:
                pz[0], pz[2] = -pz[0], pz[2] + np.pi
            if px[0] < 0:
                px[0], px[2] = -px[0], px[2] + np.pi
            fz = (pz[1], pz[0] / (Az * np.exp(-0.5 * (k0 * sig) ** 2)))
            fx = (px[1], px[0] / (Ax / k1 * np.exp(-0.5 * (k1 * sig) ** 2)))
        zb = (xic + st * dxi) * 1e3        # c dt = dz: the bunch centre moves one cell per step
        print("%d %.3f  %.4f %.4f %.4f  %.4f %.4f %.4f %.4f  %.5g %.4f  %.5g %.4f" % (
            st, zb, kl * 1e-15, res[0][0] * 1e-15, res[1][0] * 1e-15, kx * 1e-18, res[0][1] * 1e-18, res[1][1] * 1e-18,
            res[2][1] * 1e-18,
            fz[0] * 1e-3, fz[1], fx[0] * 1e-3, fx[1]))
    if os.environ.get("PROFILE"):          # the last profiles: xi, s, E_z / (-Q) sim, BS; F_x / (Q x0) sim, BS
        st, A = ax[-1]
        _, B = off[-1]
        xi = A[:, 0] * 1e-6
        dxi = abs(xi[1] - xi[0])
        lam = np.exp(-0.5 * ((xi - xic) / sig) ** 2)
        lam /= lam.sum() * dxi
        for i in range(0, len(xi), 5):
            s = xi - xi[i]
            mm = s > 0
            wz = np.sum(Az * np.cos(k0 * s[mm]) * lam[mm]) * dxi
            wx = np.sum(Ax * np.sin(k1 * s[mm]) / k1 * lam[mm]) * dxi
            print("P %.2f %.2f %.6g %.6g %.6g %.6g" % (xi[i] * 1e6, (xic - xi[i]) * 1e6, A[i, 1] / (-Q) * 1e-15, wz * 1e-15,
                                                       A[i, 2] / (Q * x0) * 1e-18, wx * 1e-18))


if __name__ == "__main__":
    main()
