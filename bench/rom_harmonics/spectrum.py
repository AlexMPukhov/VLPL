"""Track O benchmark 1: the spectrum of the reflected field of rom.deck runs (output.xflux_every records at x = 0).

    python bench/rom_harmonics/spectrum.py out_rom_100 out_rom_200 out_rom_400 [--png rom.png]

Per run: the plane mean of the reflected E_y (= (E_y - B_z) / 2 at x = 0), its power spectrum |E(omega)|^2 normalised to
the fundamental, the power-law exponent fitted to the odd harmonic peaks N = 5, 7, .. N_fit (log-log least squares; theory:
-8/3 for relativistic spikes), and the reflected energy fraction E_left / E_incident (E_incident: the total that
entered, from xflux.txt and diag.txt of the run: W_field + W_kin + E_left + E_right at the end).
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from xflux import load  # noqa: E402


def analyse(d, nfit):
    t, T, meta = load(os.path.join(d, "xflux"), "left")
    e = T[:, 0].mean(axis=(1, 2)) / 2                       # reflected E_y, plane mean
    dt = t[1] - t[0]
    w = np.fft.rfftfreq(len(e), dt) * 2 * np.pi              # omega / omega0
    s = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
    peak = lambda n: s[(w > n - 0.5) & (w <= n + 0.5)].max()
    s1 = peak(1)
    nmax = int(np.pi / dt) - 1                               # Nyquist of c dt = dx: k dx < pi
    ns = np.arange(5, min(nfit, nmax) + 1, 2)                 # odd orders (normal incidence, linear polarisation)
    p = np.array([peak(n) for n in ns]) / s1
    slope = np.polyfit(np.log(ns), np.log(p), 1)[0]
    x = np.loadtxt(os.path.join(d, "xflux.txt"))
    dg = np.loadtxt(os.path.join(d, "diag.txt"))
    total = dg[-1, 4] + x[-1, 4] + x[-1, 5]
    h = [float(v) for v in meta["h"]]
    return dict(w=w, s=s / s1, slope=slope, R=x[-1, 4] / total, ppl=2 * np.pi / h[0], nmax=nmax,
                harm={n: peak(n) / s1 for n in (3, 5, 11, 21, 31)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--nfit", type=int, default=30)
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    res = []
    for d in a.runs:
        r = analyse(d, a.nfit)
        res.append((d, r))
        hs = " ".join("I%d %.2e" % (n, v) for n, v in r["harm"].items())
        print("%-16s %4.0f cells/lambda: R %.4f, slope N = 5..%d %.2f (theory -2.67), %s" %
              (os.path.basename(d.rstrip("/")), r["ppl"], r["R"], min(a.nfit, r["nmax"]), r["slope"], hs))
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for d, r in res:
            plt.semilogy(r["w"], r["s"], lw=0.8, label="%.0f cells/lambda" % r["ppl"])
        n = np.linspace(1, 60, 100)
        plt.semilogy(n, n ** (-8 / 3), "k--", label="N^(-8/3)")
        plt.xlim(0, 60)
        plt.ylim(1e-9, 2)
        plt.xlabel("omega / omega0")
        plt.ylabel("|E_refl(omega)|^2 / fundamental")
        plt.legend()
        plt.savefig(a.png, dpi=120)


if __name__ == "__main__":
    main()
