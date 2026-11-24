# Scripts and input decks of the iRIP paper

A. Pukhov, T. C. Wilson, "Charge-conserving and implicit X-dispersionless Maxwell solvers for particle-in-cell
simulations of laser-plasma interaction" (J. Comput. Phys., in preparation). This directory holds the verification
scripts cited in the paper, the input decks that are not already in `bench/` or `examples/`, the run, reduction and
figure scripts. The data of the figures are in the supplementary material of the paper; the figure scripts read
them from `../data/` relative to `scripts/`.

| paper | decks / runs | reduction, figure |
|---|---|---|
| Sec. 3 (Esirkepov vs Villasenor-Buneman, Theorem 1, current split) | `checks/esirkepov_vb.py`, `checks/gauss3d.py`, `checks/current_split.py` | printed |
| Theorem 2 and Appendix A (discrete Poynting theorem) | `checks/poynting3d.py`, `checks/poynting_proof.py` | printed |
| Figs. dispersion, gain | analytic | `scripts/fig_dispersion_gain.py` |
| Fig. stretch | `research/irip_3d/stretch.py` | `scripts/fig_stretch.py` |
| Fig. nci (2D) | `research/irip_rz/nci2d.py`, `nci_g50.py` | `scripts/fig_nci.py` |
| Figs. energy3d, energyrz | `series/series.cpp`, `series/build_series.sh`, `decks/rz_hot.deck` | `scripts/fig_energy.py` |
| Table ncivlpl, Fig. nci_vlpl | 3D: `examples/streaming_plasma.deck`; RZ: `decks/nci_rz.deck` (`run_rz.sh`); FBPIC: `fbpic/fbpic_nci.py` | `scripts/reduce_nci.py`, `scripts/fig_nci_vlpl.py` |
| Boosted-frame grids | `boost_nci/box3d.deck`, `boost_nci/cases.txt`, `boost_nci/run.sh` | `boost_nci/reduce.py`, `reduce2.py`, `maps.py` |
| Fig. lwfa | `bench/lwfa_fbpic` (3D and RZ decks, FBPIC script), `run_rz.sh`; FBPIC at lambda/32: `fbpic/fbpic_lwfa_d2.py` | `scripts/reduce_lwfa.py`, `reduce_lwfa_dt.py`, `fig_lwfa.py` |
| Fig. bubble, Table bubblerz | `bench/lwfa_bubble` (3D deck, FBPIC script); RZ: `decks/bubble_rz.deck` (`run_rz.sh`); FBPIC variants: `fbpic/fbpic_bubble_v.py` | `scripts/reduce_bubble.py`, `reduce_bubble_rz.py`, `fig_bubble.py` |

The code tests quoted in the paper (vacuum, conservation, deposition, boundaries) are `build/tests` (`make test`).
Figure scripts: `cd scripts && for f in fig_*.py; do python3 $f pgf; done` writes `../figures/*.pgf`
(`matplotlibrc`: pdflatex).

Code version: the quasi-cylindrical results were rerun with VLPL main 2bb707f (2026-11-22,
`numerics.rz_axis_current = regular`, the default); the source changes up to 9b37dd5 touch only the boosted frame, QED,
the default `grid.absorber_particles = conserve` (no paper run has plasma reaching a PML) and openPMD output.
