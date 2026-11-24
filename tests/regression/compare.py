#!/usr/bin/env python3
"""Compare a diag.txt with a stored reference (tolerances in docs/specs/regression.md).

    compare.py REF NEW        exit 0 if NEW matches REF
    compare.py --selftest REF exit 0 if a 1e-6 perturbation of REF is detected
"""
import sys

REL = (2, 3, 4, 6)          # W_field, W_kin, W_total, max|rho|
# absolute floor per column, relative to the column maximum: max|rho| of a neutral plasma is pure
# round-off at first (~1e-17) and differs between runs (order of the OpenMP / GPU atomics)
FLOOR = {2: 1e-12, 3: 1e-12, 4: 1e-12, 6: 1e-9}
ROUNDOFF = (5, 7)           # max|dGauss|, max|divB|
NAMES = ["step", "t", "W_field", "W_kin", "W_total", "max|dGauss|", "max|rho|", "max|divB|"]


def read(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        rows.append(line.split())
    return rows


def compare(ref, new):
    """List of messages, empty if new matches ref."""
    bad = []
    if len(ref) != len(new):
        return ["%d rows, reference has %d" % (len(new), len(ref))]
    scale = {c: max(abs(float(r[c])) for r in ref) for c in REL}
    for r, n in zip(ref, new):
        if r[0] != n[0]:
            bad.append("step %s, reference %s" % (n[0], r[0]))
            continue
        tr, tn = float(r[1]), float(n[1])
        if abs(tn - tr) > 1e-12 * abs(tr):
            bad.append("step %s: t %s, reference %s" % (n[0], n[1], r[1]))
        for c in REL:
            a, b = float(r[c]), float(n[c])
            if abs(b - a) > 1e-9 * abs(a) + FLOOR[c] * scale[c]:
                bad.append("step %s: %s %s, reference %s" % (n[0], NAMES[c], n[c], r[c]))
        for c in ROUNDOFF:
            a, b = float(r[c]), float(n[c])
            if b > max(1e-12, 10 * a):
                bad.append("step %s: %s %s, reference %s" % (n[0], NAMES[c], n[c], r[c]))
        if r[8:] != n[8:]:
            bad.append("step %s: particle numbers %s, reference %s" % (n[0], n[8:], r[8:]))
    return bad


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        ref = read(sys.argv[2])
        new = [list(r) for r in ref]
        k = len(new) // 2
        new[k][2] = repr(float(new[k][2]) * (1 + 1e-6) + 1e-6)
        ok = bool(compare(ref, new))
        print("regression self-test: a 1e-6 change of W_field %s" % ("is detected" if ok else "is NOT detected"))
        return 0 if ok else 1
    ref, new = read(sys.argv[1]), read(sys.argv[2])
    bad = compare(ref, new)
    for m in bad[:10]:
        print("   ", m)
    if len(bad) > 10:
        print("    ... %d more" % (len(bad) - 10))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
