"""Oblique HHG benchmark: reflected spectra of VLPL and Smilei (bench/hhg_oblique/README.md).

    python bench/hhg_oblique/compare.py RIP_SPECTRUM SMILEI_SPECTRUM --a0 2 --waist 2 --tau 13 [--png cmp.png]

RIP_SPECTRUM: spectrum_left.txt of tools/xflux.py (omega/omega0 and S = sum over the plane of |dt rfft T|^2 h_y h_z / 4,
which is pi dW/domega by Parseval, of the field leaving through the left measuring plane, i.e. the reflected field:
left of the TFSF plane there is no incident field). SMILEI_SPECTRUM: the same
quantity from the Smilei probe (plasma run minus vacuum run, outbound components Ey - Bz, Ez + By). Both in code units
(c = 1, lengths 1/k0, fields m c omega0 / e). Each is divided by the incident pulse energy
W0 = (a0^2 / 2) (pi w0^2 / 2) tau sqrt(pi / (4 ln 2)) (Gaussian beam, field envelope exp(-2 ln2 t^2 / tau^2)), and the
energy in each harmonic band [N - 1/2, N + 1/2] is printed for both codes with their ratio.
"""
import argparse
import numpy as np


def bands(w, s, nmax):
    dw = np.gradient(w)
    return np.array([np.sum((s * dw)[(w >= n - 0.5) & (w < n + 0.5)]) for n in range(1, nmax + 1)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rip")
    p.add_argument("smilei")
    p.add_argument("--a0", type=float, required=True)
    p.add_argument("--waist", type=float, required=True, help="um")
    p.add_argument("--tau", type=float, required=True, help="fs, FWHM of the intensity")
    p.add_argument("--lambda0", type=float, default=0.8, help="um")
    p.add_argument("--nmax", type=int, default=15)
    p.add_argument("--png")
    a = p.parse_args()
    k0 = 2 * np.pi / a.lambda0                                    # 1/um
    w0, tau = a.waist * k0, a.tau * 1e-15 * 2.99792458e14 * k0     # code units
    W0 = 0.5 * a.a0 ** 2 * 0.5 * np.pi * w0 ** 2 * tau * np.sqrt(np.pi / (4 * np.log(2)))
    r = np.loadtxt(a.rip, comments="#")
    s = np.loadtxt(a.smilei, comments="#")
    r[:, 1] /= np.pi; s[:, 1] /= np.pi                          # S = pi dW/domega (rfft, Parseval)
    br, bs = bands(r[:, 0], r[:, 1], a.nmax) / W0, bands(s[:, 0], s[:, 1], a.nmax) / W0
    print("# incident energy W0 = %.6g (code units)" % W0)
    print("# reflected fraction: VLPL %.4f, Smilei %.4f" % (np.trapezoid(r[:, 1], r[:, 0]) / W0,
                                                               np.trapezoid(s[:, 1], s[:, 0]) / W0))
    print("# N   VLPL     Smilei      RIP/Smilei")
    for n in range(a.nmax):
        print("%3d  %.4e  %.4e  %.3f" % (n + 1, br[n], bs[n], br[n] / bs[n] if bs[n] > 0 else np.nan))
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.semilogy(r[:, 0], r[:, 1] / W0, label="VLPL", lw=1)
        ax.semilogy(s[:, 0], s[:, 1] / W0, label="Smilei", lw=1, alpha=0.8)
        ax.set_xlim(0, a.nmax + 0.5)
        ax.set_xlabel(r"$\omega/\omega_0$")
        ax.set_ylabel(r"$(dW/d\omega)/W_0$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.png, dpi=150)


if __name__ == "__main__":
    main()
