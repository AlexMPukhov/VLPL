# Breit-Wheeler pairs in a laser-electron collision (demo of decision 96)

`examples/qed_cascade.deck`: a 10 GeV electron beam (1 um x 1 um x 1.5 um, spin along z, 1e-4 n_c, 21600 macro-electrons)
collides head-on with an a0 = 80, 8 fs (intensity FWHM), y-polarized pulse focused to w0 = 3 um at 0.8 um: chi_e up to
2 gamma a0 hbar omega0 / m c^2 = 9.5. The electrons emit photons (spin- and polarization-resolved LCFA Monte Carlo), the photons
above 1 MeV are kept and decay into pairs (positron and electron spins, the photon Stokes vector, its selection without a decay),
the pairs radiate in turn (they are listed in qed.species).

```
build/vlpl examples/qed_cascade.deck                       # 300 steps, 12 s on 24 cores
python3 bench/qed_cascade/plot.py out_qed_cascade           # pairs.png, the numbers below
python3 examples/qed_cascade_estimate.py                    # the first-generation estimate (scipy; ~1 min)
```

## Results (the CPU cluster, 2026-11-10, branch breit-wheeler)

| quantity | simulation | estimate (first generation, deterministic energy loss) |
|---|---|---|
| pairs per beam electron | 0.317 (+- 0.004 statistical) | 0.322 photons polarized as emitted, 0.361 unpolarized photons |
| energy radiated by the beam | 0.727 | 0.731 |
| chi_e max | 9.26 | 9.50 (the peak field on the axis) |

- The estimate (`examples/qed_cascade_estimate.py`) is independent of the code: a plane wave, each electron loses energy by
  P_alpha chi^2 g(chi), emits the LCFA spectrum, each photon decays with 1 - exp(-int W_BW dt') along the rest of the pulse,
  averaged over the beam cross-section in the waist. With the mean polarization of the emission (xi3 = K23 / w along the
  field force) it agrees with the simulation to 1.6 %; the estimate for unpolarized photons is 14 % above the simulation. Left out:
  pairs of the second generation (more pairs), the energy spread of the electrons (straggling) and the selection of the
  photon polarization in flight (fewer pairs).
- The pairs radiate 2.5 % (positrons) and 2.5 % (electrons) of the beam energy; the decayed photons carried 13.6 % of it.
- Pair spins (local basis, e2 = n x e1, e1 along E + n x B, so e2 = -B'/|B'| with B' the rest-frame magnetic field):
  <S_+ . e2> = -0.380, <S_- . e2_-> = -0.377 (e2_- = -e2; rerun 2026-11-11 after the sign correction of decision 96: pairs per beam
  electron 0.3174); the decayed photons had <xi3> = 0.41. The soft lepton of a pair is polarized against its e2, i.e. the soft
  positron parallel and the soft electron antiparallel to B' (-0.98 at x -> 0, `pairs.png`), the hard one less and, for photons
  polarized along e1, the other way.
- No events with probability > 1 (sub-steps). The beam's spin and the photon-number columns are in qed.txt.
