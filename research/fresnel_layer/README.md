# Transition layer of a plasma step (docs/specs/fresnel.md, 2026-11-01)

`exact_ramp.py`: exact reflection of a plane wave (s or p) from a linear ramp eps = 1 - n x / L followed by eps = 1 - n,
the characteristic matrix integrated with RK4 along a path below the pole eps = 0 (the limit nu -> 0+ of a collision
rate; independent of the path and of the step count to all printed digits). `python3 exact_ramp.py` prints the ramp of
0.2 um at 9 n_c (|r| 0.770 at 45 deg p, 0.861 at 30 deg p). `r_ramp(n, angle, "p"|"s", k0 L)` returns r of the ramp
(referred to its start) and of the bare step.

The PIC runs of the study (decks, logs, records, analysis scripts ana*.py) are on the CPU cluster in $WORK/pp/.
