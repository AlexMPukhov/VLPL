#!/usr/bin/env python3
"""Compare the proton cutoffs of two or more runs (outputs of spectrum.py), interpolated to common times.

    python compare.py RIP=tnsa_rip.txt Smilei=tnsa_smilei.txt [--png spectra.png] [--tol 0.10]

The first run is the reference of the ratios. Gate O criterion (docs/ROADMAP.md): E_cut(1e-3) of VLPL within 10 % of
Smilei's at the end of the run (the last common time). Exit status 1 if it is not met.
"""
import argparse, sys
import numpy as np

COLS = ["t", "N_macro", "N_left", "N>1MeV", "E_p", "E_max", "E_cut(1e-2)", "E_cut(1e-3)", "E_cut(1e-4)"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=file")
    ap.add_argument("--png", default=None)
    ap.add_argument("--tol", type=float, default=0.10)
    a = ap.parse_args()
    runs = [(s.split("=", 1)[0], np.atleast_2d(np.loadtxt(s.split("=", 1)[1]))) for s in a.runs]
    t0 = max(r[:, 0].min() for _, r in runs)
    t1 = min(r[:, 0].max() for _, r in runs)
    times = [t for t in runs[0][1][:, 0] if t0 - 1 <= t <= t1 + 1]

    def at(r, t, c):
        return np.interp(t, r[:, 0], r[:, COLS.index(c)])

    for c in ["E_cut(1e-2)", "E_cut(1e-3)", "E_cut(1e-4)", "E_max", "E_p", "N>1MeV", "N_left"]:
        print("\n%s" % c)
        print("t[fs]  " + "  ".join("%10s" % l for l, _ in runs) + "  " + "  ".join("%8s" % ("%s/%s" % (l, runs[0][0]))
                                                                                   for l, _ in runs[1:]))
        for t in times:
            v = [at(r, t, c) for _, r in runs]
            print("%5.0f  " % t + "  ".join("%10.4g" % x for x in v) + "  " +
                  "  ".join("%8.3f" % (x / v[0] if v[0] else np.nan) for x in v[1:]))

    ok = True
    t = times[-1]
    ref = at(runs[0][1], t, "E_cut(1e-3)")
    for l, r in runs[1:]:
        d = at(r, t, "E_cut(1e-3)") / ref - 1
        ok &= abs(d) <= a.tol
        print("\nt = %.0f fs: E_cut(1e-3) %s %.3f MeV, %s %.3f MeV, difference %+.1f %% (tolerance %.0f %%)"
              % (t, runs[0][0], ref, l, at(r, t, "E_cut(1e-3)"), 100 * d, 100 * a.tol))

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for l, f in [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in a.runs]:
            r = np.atleast_2d(np.loadtxt(f))
            ln, = ax[0].plot(r[:, 0], r[:, COLS.index("E_cut(1e-3)")], "o-", label=l + " E_cut(1e-3)")
            ax[0].plot(r[:, 0], r[:, COLS.index("E_max")], ":", color=ln.get_color(), label=l + " E_max")
            sp = np.atleast_2d(np.loadtxt(f + ".spec"))
            with open(f + ".spec") as h:
                h.readline()
                ec = np.array([float(x) for x in h.readline()[1:].split()])
            k = int(np.argmin(abs(sp[:, 0] - t)))
            ax[1].semilogy(ec, np.where(sp[k, 1:] > 0, sp[k, 1:], np.nan), color=ln.get_color(),
                           label="%s, t = %.0f fs" % (l, sp[k, 0]))
        ax[0].set_xlabel("t [fs]"); ax[0].set_ylabel("proton energy [MeV]"); ax[0].legend(fontsize=8)
        ax[1].set_xlabel("E [MeV]"); ax[1].set_ylabel("dN/dE per proton of the layer [1/MeV]"); ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.png, dpi=120)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
