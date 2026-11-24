#!/usr/bin/env python3
"""Proton spectra and cutoff energies of the TNSA benchmark, the same definition for VLPL and Smilei.

    python spectrum.py rip    <rip output dir>  [--name protons] -o tnsa_rip.txt
    python spectrum.py smilei <smilei run dir>  [--name proton]  -o tnsa_smilei.txt

Input: every proton of every dump (VLPL: diagnostic block with particles = true, particles/<name>_<step>_r*.bin;
Smilei: DiagTrackParticles, TrackParticlesDisordered_<name>.h5, momentum in m_e c, divided here by the proton mass),
u = gamma v / c and the weight w.
Kinetic energy E = (gamma - 1) m_p c^2 with m_p = 1836 m_e (the mass in both decks).
The weights are normalised by their sum in the first dump (the whole contaminant layer: the first dump is before the
pulse reaches the foil), so the two codes' weight units drop out and every number is per proton of the layer:
- E_max: the most energetic macro-particle (depends on the particle number; reported, not used for the criterion);
- E_cut(f): the energy above which the protons carry the fraction f of the layer, f = 1e-2, 1e-3, 1e-4
  (linear interpolation of the cumulative weight between macro-particles); E_cut(1e-3) is the cutoff of the benchmark;
- N_left: the fraction no longer in the box (left through x); N(>1 MeV); E_p: kinetic energy of all protons in MeV per
  proton of the layer;
- dN/dE per proton of the layer in 1/MeV on bins of 0.25 MeV (0-60 MeV) in <out>.spec.
"""
import argparse, glob, os
import numpy as np

MP_MEV = 1836.0 * 0.51099895
FRACS = (1e-2, 1e-3, 1e-4)
EDGES = np.arange(0.0, 60.0 + 1e-9, 0.25)


def rip_dumps(d, name):
    meta = dict(l.split(None, 1) for l in open(os.path.join(d, "particles", "meta.txt")) if " " in l)
    times = {}
    for l in open(os.path.join(d, "particles", name + ".txt")):
        if l.startswith("#") or not l.strip():
            continue
        v = l.split()
        times[int(v[0])] = float(v[1])
    for step in sorted(times):
        files = sorted(glob.glob(os.path.join(d, "particles", "%s_%06d_r*.bin" % (name, step))))
        a = np.concatenate([np.fromfile(f, dtype=np.float64) for f in files]) if files else np.zeros(0)
        a = a.reshape(-1, 7)
        yield times[step], a[:, 3:6], a[:, 6]
    assert float(meta["mass"]) == 1836.0, "proton mass in the deck: 1836"


def smilei_dumps(d, name):
    import h5py
    fn = os.path.join(d, "TrackParticlesDisordered_%s.h5" % name)
    with h5py.File(fn, "r") as f:
        fs = 2 * np.pi * 0.299792458 / 0.8          # 1 fs in 1/omega0 (lambda0 = 0.8 um)
        for k in sorted(f["data"].keys(), key=int):
            g = f["data"][k]
            p = g["particles"][name]
            u = np.stack([p["momentum"][c][()] for c in "xyz"], axis=1) / 1836.0   # Smilei: p in m_e c
            w = p["weight"][()]
            t = float(g.attrs["time"]) if "time" in g.attrs else int(k) * float(g.attrs["dt"])
            yield t / fs, u, w


def analyse(dumps):
    rows, specs, w0 = [], [], None
    for t, u, w in dumps:
        e = (np.sqrt(1.0 + np.sum(u * u, axis=1)) - 1.0) * MP_MEV
        if w0 is None:
            w0 = np.sum(w)
        o = np.argsort(e)[::-1]
        es, cum = e[o], np.cumsum(w[o]) / w0
        cuts = []
        for f in FRACS:
            if len(es) == 0 or cum[-1] < f:
                cuts.append(0.0)
                continue
            i = int(np.searchsorted(cum, f))
            if i == 0:
                cuts.append(es[0])
            else:            # cum[i-1] < f <= cum[i]
                a = (f - cum[i - 1]) / (cum[i] - cum[i - 1])
                cuts.append(es[i - 1] + a * (es[i] - es[i - 1]))
        rows.append([t, len(e), 1.0 - np.sum(w) / w0, np.sum(w[e > 1.0]) / w0, np.sum(w * e) / w0,
                     es[0] if len(es) else 0.0] + cuts)
        h, _ = np.histogram(e, bins=EDGES, weights=w / w0)
        specs.append([t] + list(h / np.diff(EDGES)))
    return np.array(rows), np.array(specs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("code", choices=["rip", "smilei"])
    ap.add_argument("dir")
    ap.add_argument("--name", default=None, help="species / diagnostic name (protons for VLPL, proton for Smilei)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    if a.code == "rip":
        rows, specs = analyse(rip_dumps(a.dir, a.name or "protons"))
    else:
        rows, specs = analyse(smilei_dumps(a.dir, a.name or "proton"))
    hdr = "t[fs] N_macro N_left N(>1MeV) E_p[MeV] E_max[MeV] " + " ".join("E_cut(%g)[MeV]" % f for f in FRACS)
    np.savetxt(a.out, rows, fmt="%.6g", header="%s %s\n%s" % (a.code, os.path.abspath(a.dir), hdr))
    np.savetxt(a.out + ".spec", specs, fmt="%.6g",
               header="dN/dE per proton of the layer [1/MeV]; columns t[fs], then bins centred at E[MeV] =\n" +
                      " ".join("%.4g" % x for x in 0.5 * (EDGES[1:] + EDGES[:-1])))
    for r in rows:
        print("t = %6.1f fs  N = %8d  left %.3f  >1 MeV %.3f  E_max %6.2f  E_cut(1e-2, 1e-3, 1e-4) = %6.2f %6.2f %6.2f MeV"
              % (r[0], r[1], r[2], r[3], r[5], r[6], r[7], r[8]))


if __name__ == "__main__":
    main()
