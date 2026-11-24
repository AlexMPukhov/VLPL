# Field ionization: VLPL against Smilei 5.1

Phase 3 benchmark (docs/ROADMAP.md, Gate C: "Smilei/EPOCH ionization benchmarks"; decision 86, docs/specs/field_ionization.md).
Smilei's tunnel model (`ionization_model = "tunnel"`) uses the same ADK form (l* = n* - 1, l of the removed electron, m = 0) and
also an ionization current (Nuter et al. 2011), so both codes model the same physics.

## Setup
A plane-wave Gaussian pulse (lambda0 = 0.8 um, E_y, 10 fs intensity FWHM, peak at the left edge at 25 fs) through a 16 um
hydrogen slab (12 .. 28 um, 0.1 n_c) with a carbon tracer (1e-5 n_c: its electrons are negligible), ions immobile, box 44 um,
dx = lambda0 / 64, until 140 fs (the pulse has left the slab and not yet the box). VLPL: `ion.deck` (cartesian2d, 4 periodic
y cells, RIP-CC, dt = dx, shape 2; H 4 and C 64 particles per cell); Smilei: `smilei_ion.py` (1Dcartesian, Yee, dt = 0.95 dx,
order 2, the same particle numbers per cell length; VLPL's laser envelope a0 exp(-2 ln2 (t - tc)^2 / tau^2) sin(t - tc) as
the time profile at xmin).

    sbatch run_smilei.sbatch                                   # out_smilei/ (4 ranks x 6 threads, ~2 min)
    EXE=../../build/vlpl sbatch --export=ALL run_rip.sbatch  # out_rip/, and out_rip_nocur/ without the ionization current
    . $WORK/smilei/smilei_env.sh; python3 compare.py; python3 compare.py out_rip_nocur out_smilei

## Results (2026-11-08, the CPU cluster; VLPL branch field-ionization)
a0 = 0.03 (peak 0.23 atomic units): hydrogen and C0, C1+ ionize on the leading edge, C2+ -> C3+ partly near the peak.

| quantity (fractions of the pulse energy) | VLPL | Smilei | |
|---|---|---|---|
| hydrogen ionized | 1.00000 | 1.00000 | |
| carbon C2+ / C3+ | 0.98981 / 0.01019 | 0.98969 / 0.01031 | 0.3 sigma |
| field energy at the end | 0.96226 | 0.96342 | -1.2e-3 |
| electron kinetic energy | 0.00891 | 0.00797 | |
| ionization energy | 0.02966 (E_ion = E_cur) | 0.02860 (injected - field - kinetic) | n D I_p / U = 0.02967 |
| field energy at the end, no ionization current | 0.99302 | | |

- The ionization current takes n D I_p from the pulse in VLPL exactly (E_cur = E_ion = 0.02966, the analytic 0.02967); without
  it the transmitted energy is 3 % higher than Smilei's. With it VLPL and Smilei agree to 1.2e-3 of the pulse energy in the
  transmitted field.
- The rest of the absorption is the drift energy of the electrons born at a phase where the vector potential is not zero
  (ionization heating): 0.0089 against 0.0080 (12 % apart, 9e-4 of the pulse); Smilei's energy diagnostics carry an offset of
  ~1e-3 of the pulse (Ubal of the vacuum part, case A below).
- VLPL run: 95 s on 12 threads (2D, 4 cells across; ionization 11 % of the step); Smilei 1D on 24 cores ~2 min.

Case A (a0 = 0.1, 7.5e-3 n_c hydrogen, the first run): everything saturates (H fully ionized, carbon all C4+ in both codes;
Smilei 5e-5 of C5+ and C6+, which need fields 10 times the peak: a Smilei artefact). Electron kinetic energy 1.3e-4 (VLPL)
against 1.2e-4 (Smilei) of the pulse.
