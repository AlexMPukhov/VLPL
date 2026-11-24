# RZ with azimuthal modes: charge-conserving deposition and the field step (docs/specs/rz_modes.md)

- `deposit.py`: the deposition for RIP-CC's staggering in (z, r) with modes: 2D Esirkepov in (z, r) with
  cylindrical cell volumes, times the mean phase e_bar, plus J_phi from the change of the phase.
  `python3 deposit.py`: random moves (300 particles, near the axis too, any azimuthal motion), the discrete
  continuity per mode: 6.7e-16 (m = 0), 5.9e-17, 9.3e-17, 4.7e-17 (m = 1, 2, 3) relative.
- `gauss_loop.py`: the RIP-CC step per mode on a (z, r) grid (operators and axis rules of
  research/irip_rz) with these sources, 200 ballistic particles, 12 steps; the per-mode Gauss law
  change relative to |rho| or |div E|:
  - box iRIP (dense solve, c dt = 1.25 h_r): 5.9e-16, 1.2e-16, 2.0e-16, 1.6e-16 (m = 0 .. 3);
  - explicit RIP-CC on a stable grid (`RZ_STABLE=1`, c dt = 0.2 h_r): 9.1e-16, 1.8e-16, 1.5e-16, 2.0e-16.
  (On the unstable grid the explicit step amplifies round-off, as expected.)
