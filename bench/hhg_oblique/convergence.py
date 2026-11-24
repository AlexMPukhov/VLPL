"""Oblique HHG benchmark, resolution series: band energies of several reflected spectra (bench/hhg_oblique/README.md).

    python bench/hhg_oblique/convergence.py --a0 2 --waist 2 --tau 13 RIP/32=out_rip32/xflux/spectrum_left.txt \
        RIP/64=out_rip64/xflux/spectrum_left.txt Smilei/32=spectrum_smilei.txt ... [--nmax 20] [--png conv.png]

Each spectrum is a two-column file (omega/omega0, S = pi dW/domega) as written by tools/xflux.py (VLPL) or
smilei/compare_smilei.py (Smilei). The energy in each band [N - 1/2, N + 1/2] is divided by the incident energy W0 of
the analytic Gaussian beam (as compare.py). Bands above the Nyquist frequency of a run are left empty.
"""
import argparse
import numpy as np
from compare import bands


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spectra", nargs="+", help="label=file")
    p.add_argument("--a0", type=float, required=True)
    p.add_argument("--waist", type=float, required=True, help="um")
    p.add_argument("--tau", type=float, required=True, help="fs, FWHM of the intensity")
    p.add_argument("--lambda0", type=float, default=0.8, help="um")
    p.add_argument("--nmax", type=int, default=20)
    p.add_argument("--png")
    a = p.parse_args()
    k0 = 2 * np.pi / a.lambda0
    w0, tau = a.waist * k0, a.tau * 1e-15 * 2.99792458e14 * k0
    W0 = 0.5 * a.a0 ** 2 * 0.5 * np.pi * w0 ** 2 * tau * np.sqrt(np.pi / (4 * np.log(2)))
    runs = []
    for item in a.spectra:
        label, path = item.split("=", 1)
        d = np.loadtxt(path, comments="#")
        w, s = d[:, 0], d[:, 1] / np.pi
        b = bands(w, s, a.nmax) / W0
        b[np.arange(1, a.nmax + 1) + 0.5 > w[-1]] = np.nan
        runs.append((label, w, s, b, np.trapezoid(s, w) / W0))
    print("# incident energy W0 = %.6g (code units); band energies / W0" % W0)
    print("# reflected fraction: " + ", ".join("%s %.4f" % (r[0], r[4]) for r in runs))
    print("# N  " + "".join("%12s" % r[0] for r in runs))
    for n in range(a.nmax):
        print("%3d  " % (n + 1) + "".join("%12.3e" % r[3][n] for r in runs))
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        n = np.arange(1, a.nmax + 1)
        for r in runs:
            ax.semilogy(n, r[3], "o-" if r[0].startswith("RIP") else "s--", ms=4, label=r[0])
        ax.set_xlabel("harmonic order N")
        ax.set_ylabel(r"$W_N / W_0$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.png, dpi=150)


if __name__ == "__main__":
    main()
