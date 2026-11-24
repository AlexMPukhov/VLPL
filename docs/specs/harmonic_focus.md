# CHF diagnostic: the focus of the reflected harmonics (block `harmonic_focus`)

Status: implemented 2026-11-03 (decision 67); roadmap track O, Gate O benchmark 4 (`docs/specs/overdense.md`).
Cart3D, 2D (cartesian2d) and RZ; RIP-CC and iRIP; MPI (any decomposition). Not with a moving window or restarts.

## Why
Coherent harmonic focusing (Gordienko et al., PRL 94, 103903 (2005)): the harmonics reflected by a dented or pre-curved
overdense surface focus in front of it, to a spot ~ lambda0 / (2 N sin alpha), where the attosecond pulse of the high
orders reaches intensities far above the laser's. Gate O asks that the focal intensity, the spot size and the position of
the focus converge in dx and dr. Those are properties of the field inside the box, where the incident laser and the
reflected light overlap; the high orders carry a small part of the energy, so the field must be filtered in time before
its intensity is taken. Writing the field every step and filtering afterwards would need the whole time history of the
focal region (tens of GB); the filter runs on the fly instead, one state per node.

## What is computed
Every step, after the field step, at each node of a region:
- E at the node: Cart3D Ex there, Ey and Ez averaged from j -+ 1/2, k -+ 1/2 (2D: Ez itself); RZ (E_z, E_r, E_phi) of the
  modes at (i, r_j) with E_r averaged from the half nodes, reconstructed at phi = 0 and phi = pi/2,
  F(phi) = F_0 + 2 sum_m (Re F_m cos m phi - Im F_m sin m phi) (the polarisation plane and the plane across it for a laser
  polarised along x). On the axis E_r of mode m at -r_1/2 is (-1)^(m+1) E_r(r_1/2), and the transverse field is the
  Cartesian E_x = 2 Re E_r,1, E_y = -2 Im E_r,1 at both angles.
- Each component goes through a band-pass in time: Butterworth high-pass at n_min omega0 and low-pass at n_max omega0
  (`filter_order` poles each, default 8), by the bilinear transform prewarped at the edges (K = tan(omega_c dt / 2),
  sections Q_k = 1 / (2 cos(pi (2k + 1) / 2n))), as cascaded second-order sections in direct form II transposed.
  |H(omega)| = [1 + (K_lo / K)^2n]^-1/2 [1 + (K / K_hi)^2n]^-1/2, K = tan(omega dt / 2). With 8 poles the fundamental is
  suppressed by (1 / 4.5)^8 = 6e-6 in amplitude at n_min = 4.5; quasi-static fields (charge separation) are removed.
- Per node: the peak of |E_band|^2 over the run and its time, and the fluence int |E_band|^2 dt.

The fluence depends only on |H|^2 (Parseval), the peak also on the phase of H: a causal filter delays the edges of
the band against its centre and lowers the peak of a broadband pulse (a 6 fs pulse through 0.5 .. 1.5: 4 %). The peak
is therefore the peak of the filtered pulse: a resolution series compares it at the same band and filter order (the
bilinear filter at another dt differs by O((n_max omega0 dt)^2) near the upper edge), and the fluence is the measure
that does not depend on the phase of the filter.

Units: |E|^2 in (m c omega0 / e)^2, i.e. a^2 (a linearly polarised laser has the peak a0^2); times 1/omega0.

## Cost
6 signals (RZ) or 3 (Cart) times `filter_order` sections times 2 states per node of the region: RZ, 8 poles,
96 doubles per node. 5 % of the step in the vacuum test (RZ iRIP, 2 modes). Restrict the region to the focal volume.

## Deck
```
harmonic_focus {
  n_min        = 4.5      # band edges in harmonic order omega / omega0 (n_min = 0: no high-pass, n_max = 0: no low-pass)
  n_max        = 20       # below 0.9 of the Nyquist order pi / (omega0 dt)
  filter_order = 8        # poles per edge, even, 2 .. 16
  xmin = 1 um             # region (absolute coordinates; Cart3D xmin .. zmax, RZ zmin, zmax, rmax); default the box
  xmax = 5 um
  every = 0               # also write every n steps (0: at the end only)
}
```
Output `dir/harmonic_focus/`: per rank `r<rank>.bin` (float64 [quantities][z][y][x], RZ [quantities][r][z]) and
`meta_r<rank>.txt` (layout, band, dt, radii). Quantities: Cart `Imax tmax fluence`; RZ the same at phi = 0 and pi/2.
`tools/hfocus.py dir/harmonic_focus [--a0 A] [--json f]` assembles the ranks and reports for the peak and the fluence the
position of the maximum, its value (and / a0^2), and the FWHM along the axis and across it (RZ: the full radial width
2 r_1/2 when the maximum is on the axis), each interpolated around the grid maximum.

## Energy balance (with `output.xflux`)
xflux.txt has three more columns (docs/specs/xflux.md): E_abs_r, the energy removed by the radial absorber of RZ (W before
minus after the damping, summed over the steps); W_in, the field energy between the measuring planes (without the x
absorbers, whose content Enet already counts as gone); E_part_out, the kinetic energy of the particles that left the box.
W_in + W_kin + Enet_left + Enet_right + E_abs_r + E_part_out = the injected energy: with a TFSF plane the same sum of the
vacuum twin (the energy that crossed the plane; in RZ about E_inj, column 12), with the laser through an open left edge 0
(Enet_left counts the laser as inbound since 2026-11-03, docs/specs/xflux.md; bench/chf/analyse.py). CHF code check: 2D
5.5e-3, 1.55e-3, 3.1e-4, 1.5e-4 at lambda0/64, /128, /256, /512; 3D 5.2e-3 at lambda0/64; RZ TFSF 2.3e-3, 1.1e-3 at
lambda0/128, /256 up to 48 fs, later an excess of 6e-3 at both (open; bench/chf/README.md).

## Tests (`build/tests test_hfocus`)
- H1: the cascade of the coefficients on a sine (dt = 2 pi / 64, 4.5 .. 25, 8 poles): the steady amplitude against the
  analytic bilinear Butterworth 3.7e-13 at 9 frequencies from 0.5 to 28 omega0 (`hfocus_gain` 1.2e-15).
- H2: 2D plane pulse, band 0.5 .. 1.5: the maps at 6 nodes against a host replay of the grid's field through the same
  coefficients: equal (0); the fluence at all 484 nodes against a0^2 (tau / 2) sqrt(pi / 4 ln 2) within 3.1e-3 (|H|^2 over the
  pulse spectrum), the peak against a0^2 within 4.2 % (the filter's phase).
- H3: RZ Gaussian beam (2 modes), band 0.5 .. 1.5: the maps at phi = 0, pi/2 and on the axis against the host replay of the
  Cartesian field of `rz_gather_point`: 1.9e-14.
- Vacuum check (not in the fast tests): RZ, waist 1.5 um, focus 10 um, lambda0/32, iRIP: on-axis fluence FWHM 17.76 um
  (2 z_R = 17.67), radial FWHM of the peak at the focus 1.768 um (w0 sqrt(2 ln 2) = 1.766), peak 0.976 a0^2 (8 fs pulse),
  fluence maximum at z = 9.84 um (focus 10 um; the on-axis fluence is flat within 1e-3 over 0.4 um there).
