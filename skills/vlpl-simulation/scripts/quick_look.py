#!/usr/bin/env python3
"""A first look at a VLPL run directory (output.dir), in physical units. Needs only the python standard library.

  python3 quick_look.py <output.dir> [--lambda0 0.8] [--density 1e18]
      --lambda0   lambda0 of the deck in um (default 0.8)
      --density   plasma density in cm^-3: E_z on the axis is then also given in units of E0 = m c omega_p / e

Prints
- diag.txt: energies of the first and last rows, the largest max|dGauss| and max|divB| (round-off in a healthy run),
  particle numbers; NaN or a growing Gauss error is flagged;
- beam_<name>.txt (blocks diagnostic): the last row (charge, energy, spread, emittances);
- RZ, output.axis_every: the extrema of E_z on the axis in the last record, in GV/m, with their position in the window;
- Cartesian, output.slice_every: the largest |value| of every slice field at the last slice step (code units; fields
  also in GV/m or T);
- ionization.txt, collisions.txt, qed.txt, medium.txt, xflux.txt, lab_beams.txt: that they exist, with their last row.
Code units: lengths 1/k0 = lambda0 / 2 pi, times 1/omega0, E in m_e c omega0 / e = 3.2107e12 V/m / lambda0[um],
B in m_e omega0 / e = 1.0710e4 T / lambda0[um], densities in n_c = 1.1149e21 cm^-3 / lambda0[um]^2.
"""
import array
import glob
import math
import os
import sys


def rows(path):
    out, head = [], ""
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                head = line[1:].strip()
            elif line.strip():
                out.append(line.split())
    return head, out


def num(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def main():
    argv, opt, args = sys.argv[1:], {"--lambda0": 0.8, "--density": 0.0}, []
    i = 0
    while i < len(argv):
        if argv[i] in opt and i + 1 < len(argv):
            opt[argv[i]] = float(argv[i + 1])
            i += 2
        elif argv[i].startswith("--"):
            sys.exit("unknown option %s\n%s" % (argv[i], __doc__))
        else:
            args.append(argv[i])
            i += 1
    if len(args) != 1:
        sys.exit("give one output directory\n%s" % __doc__)
    d = args[0]
    lam = opt["--lambda0"]
    e0 = 96.16 * math.sqrt(opt["--density"] / 1e18)      # cold wave-breaking field, GV/m
    e_unit = 3.2107e12 / lam          # V/m per code unit
    b_unit = 1.0710e4 / lam           # T
    t_unit = lam / (2 * math.pi * 0.299792458)     # fs per code unit
    l_unit = lam / (2 * math.pi)                   # um per code unit
    bad = 0

    p = os.path.join(d, "diag.txt")
    if not os.path.exists(p):
        sys.exit("no diag.txt in %s: not a VLPL output directory, or the run did not start" % d)
    head, r = rows(p)
    if not r:
        sys.exit("diag.txt is empty")
    first, last = r[0], r[-1]
    print("diag.txt: %d rows, steps %s .. %s, t = %.4g fs (%.4g um of light path)"
          % (len(r), first[0], last[0], num(last[1]) * t_unit, num(last[1]) * l_unit))
    print("  W_field %.6g -> %.6g, W_kin %.6g -> %.6g, W_total %.6g -> %.6g  (code units; open boundaries: not conserved)"
          % (num(first[2]), num(last[2]), num(first[3]), num(last[3]), num(first[4]), num(last[4])))
    print("  (W_field contains the laser pulse once it has entered: a large W_field next to a small W_kin is normal)")
    gauss = max(abs(num(x[5])) for x in r)
    divb = max(abs(num(x[7])) for x in r)
    rho = max(abs(num(x[6])) for x in r)
    print("  max|dGauss| = %.3g, max|divB| = %.3g, max|rho| = %.3g n_c" % (gauss, divb, rho))
    names = head.split()[8:]
    print("  particles: " + ", ".join("%s %s -> %s" % (n, first[8 + i], last[8 + i]) for i, n in enumerate(names)))
    if any(math.isnan(num(v)) or math.isinf(num(v)) for x in r for v in x):
        print("  PROBLEM: NaN or Inf in diag.txt (unstable run: see references/debugging.md)")
        bad = 1
    if gauss > 1e-8 * max(rho, 1e-30) and gauss > 1e-10:
        print("  CHECK: the Gauss law is not at round-off (max|dGauss| / max|rho| = %.2g): see references/debugging.md"
              % (gauss / max(rho, 1e-30)))

    for p in sorted(glob.glob(os.path.join(d, "beam_*.txt"))):
        head, r = rows(p)
        if r:
            print("%s, last row:\n  %s\n  %s" % (os.path.basename(p), head, " ".join(r[-1])))

    ax = sorted(glob.glob(os.path.join(d, "plane", "axis_r*.bin")))
    meta = os.path.join(d, "plane", "meta.txt")
    if ax and os.path.exists(meta):
        m = dict((l.split()[0], l.split()[1:]) for l in open(meta) if l.strip())
        dz = float(m["dz"][0])
        rec = {}                                  # step -> list of (first cell, values)
        for p in ax:
            a = array.array("d")
            with open(p, "rb") as f:
                a.frombytes(f.read())
            i = 0
            while i + 4 <= len(a):
                step, n = int(a[i]), int(a[i + 3])
                rec.setdefault(step, []).append((int(a[i + 2]), a[i + 4:i + 4 + n], a[i + 1]))
                i += 4 + n
        if rec:
            step = max(rec)
            cells = sorted(rec[step])
            c0 = cells[0][0]
            ez = [v for part in cells for v in part[1]]
            lo, hi = min(ez), max(ez)
            print("E_z on the axis (plane/axis_r*.bin), %d records, the last at step %d (t = %.4g fs), window from z = %.4g um:"
                  % (len(rec), step, cells[0][2] * t_unit, c0 * dz * l_unit))
            print("  min %.4g (%.4g GV/m) at %.4g um, max %.4g (%.4g GV/m) at %.4g um from the left edge of the window"
                  % (lo, lo * e_unit / 1e9, ez.index(lo) * dz * l_unit, hi, hi * e_unit / 1e9, ez.index(hi) * dz * l_unit))
            if e0 > 0:
                print("  in units of E0 = %.4g GV/m: min %.3g, max %.3g" % (e0, lo * e_unit / 1e9 / e0, hi * e_unit / 1e9 / e0))

    sl = os.path.join(d, "slices")
    if os.path.exists(os.path.join(sl, "meta.txt")):
        m = dict((l.split()[0], l.split()[1:]) for l in open(os.path.join(sl, "meta.txt")) if l.strip())
        nx, ny = int(m["nx"][0]), int(m["ny"][0])
        for name in m.get("fields", []):
            files = sorted(glob.glob(os.path.join(sl, name + "_*.f32")), key=lambda q: int(q.rsplit("_", 1)[1][:-4]))
            if not files:
                continue
            a = array.array("f")
            with open(files[-1], "rb") as f:
                a.frombytes(f.read())
            if len(a) != nx * ny:
                print("  slices/%s: unexpected size" % os.path.basename(files[-1]))
                continue
            big = max(abs(v) for v in a)
            k = max(range(len(a)), key=lambda q: abs(a[q]))
            unit = ""
            if name[0] == "E":
                unit = " = %.4g GV/m" % (big * e_unit / 1e9)
            elif name[0] == "B":
                unit = " = %.4g T" % (big * b_unit)
            elif name.startswith("n_") or name == "rho":
                unit = " n_c"
            print("slice %s (%d x %d, %d files), the last: max|%s| = %.4g%s at x = %.4g um, y = %.4g um"
                  % (os.path.basename(files[-1]), nx, ny, len(files), name, big, unit,
                     (k % nx) * float(m["dx"][0]) * l_unit, (k // nx) * float(m["dy"][0]) * l_unit))

    for name in ("ionization.txt", "collisions.txt", "qed.txt", "medium.txt", "xflux.txt", "lab_beams.txt"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            head, r = rows(p)
            print("%s: %d rows%s" % (name, len(r), "; last: " + " ".join(r[-1])[:200] if r else ""))
    n = len(glob.glob(os.path.join(d, "openpmd", "*.h5")))
    if n:
        print("openpmd/: %d files (openPMD-viewer: OpenPMDTimeSeries('%s'))" % (n, os.path.join(d, "openpmd")))
    sys.exit(bad)


if __name__ == "__main__":
    main()
