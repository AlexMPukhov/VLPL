# Embedded conductors that depend on the azimuth in RZ (block `metal`), the axis cut

Status: implemented (decision 103), RIP-CC only.

Motivation: structures with a discrete rotational symmetry, e.g. square or ridged guides (C4),
simulated with a few azimuthal modes instead of full 3D. The C4 metal couples mode m with m +- 4, so a
centred beam drives m = 0, 4, 8, .. and an offset beam m = 1, 3, 5, 7, ... The `rings` block (docs/specs/rz_metal.md) is the axisymmetric special case.

## Physics and equations
Perfect conductor: tangential E = 0, normal B = 0 on its surface, all fields 0 inside; normal E and tangential B on the
surface are the surface charge and current. The metal is given by a deck function, metal where f(x, y, z, r, phi) > 0
(lab frame, fixed; z the axis, x = r cos phi, y = r sin phi in the RZ deck axes). Particles cross the metal
(`metal.particles = pass`, the default) or are absorbed (`absorb`), decision 122.

## Discretisation
### Angular grid
M modes (m = 0 .. M - 1) are 2M - 1 real numbers per point and are represented exactly by the values on Np = 2M - 1
equidistant angles phi_k = k dphi, dphi = 2 pi / Np (and on the shifted angles phi_k + dphi/2): the transform is invertible
and Parseval holds, (1/Np) sum_k f_k^2 = F_0^2 + 2 sum |F_m|^2. The metal acts on these angles.

**The phi derivative (`metal.phi_derivative`).**
- `difference` (default): the RZ solver's i m is replaced by i m_eff, m_eff = (2 / dphi) sin(m dphi / 2): on the angles this is
  the local staggered difference (f(phi + dphi/2) - f(phi - dphi/2)) / dphi, still diagonal in the modes (each mode keeps its
  radial operators, the solver is unchanged otherwise). The fields then sit on a cylindrical Yee grid in phi:
  E_z, E_r, B_phi (and rho, J_z, J_r) at phi_k; E_phi, B_r, B_z (and J_phi) at phi_k + dphi/2, and the metal is a staircase as
  in a Cartesian Yee code. Every RZ operator reads m_eff (`RadialView::mf`: the field step, the divergences of the diagnostics,
  the J_phi of the charge-conserving deposition, i r W (e1 - e0) / m_eff, the radial bases, the self-field, the filters, the
  media): the Gauss law per mode stays exact. Without a metal block m_eff = m (bit-identical results).
- `spectral`: m unchanged, every component at phi_k. Correct only for axisymmetric masks (then it reproduces the rings block
  to round-off). With phi surfaces the zeros inside the metal enter the vacuum through the global spectral derivative (Gibbs)
  and B_z loses its Neumann condition there: the TE modes of a square guide come out far too low (0.81 .. 1.68 for M = 4 .. 16
  against TE10 = 2.62, research/rz_metal_phi/xsection.py); a run warns.

m_eff < m changes the dispersion of the high modes (m dphi / 2 = pi m / (2M - 1); m_eff / m = 0.99 at m = M/4, 0.90 at M/2,
0.64 at M - 1), as any finite difference does at the grid scale: the physics must live in the lower half of the modes (the
convergence tests below).

### Masks
Node bits Mn(I, j, k) = f(z_I, r_j, phi_k) > 0 (column I absolute, lab frame), j = 0 .. nr. Derived masks at the other
positions: Mh = Mn(j) Mn(j+1) at (j+1/2, k) (E_r, B_phi); Mp = Mn(k) Mn(k+1) at (j, k+1/2) (E_phi, B_r); Mc = Mh(k) Mh(k+1)
at (j+1/2, k+1/2) (B_z) (spectral: Mp = Mn, Mc = Mh). A ring buffer on the device holds the bits of the local columns plus
two guards each side; a moving window evaluates the function on the entering column only.

### Rules (after each pass of the RIP-CC step)
The fields of the pass (predictor E_z*, B_z*; transported E_r, E_phi, B_r, B_phi; corrected E_z, B_z) are taken to their
angles, the rules applied at the metal angles only, and the changes added back to the modes (cost ~ (metal angles) x M per
point and component; points without metal at any angle are skipped). Per point with its own mask X:
- not metal: free;
- both z neighbours metal (X(I-1) X(I+1)): zero (inside);
- else a z face (left: vacuum at I - 1), the rules of the rings block (decision 83): E_r = E_phi = B_z = 0; B_phi, B_r carry the
  characteristic arriving from the vacuum side (left face B_phi = T+ = E_r + B_phi, B_r = -T-_z = B_r - E_phi; right face
  B_phi - E_r, E_phi + B_r: the RIP transport is reflected exactly at normal incidence), unless the component is normal to a
  transverse metal surface through the point (B_phi: a B_z cell (j+1/2, k +- 1/2) in vacuum; B_r: a B_z cell (j +- 1/2, k+1/2)
  in vacuum, or the axis): then 0; E_z from the corrector (the normal E, surface charge) unless tangential to such a surface
  (one of its four B_z cells in vacuum): then 0.
For an axisymmetric mask these are the rules of the rings block with the jacket's per-point form (same at every angle).

### Charge conservation, diagnostics
The rules override field values only; the Gauss identity of RIP-CC holds for any fields entering a step, so the Gauss
residual changes only at the rho nodes whose stencil touches a metal node at some angle (the surface charge; in all modes,
as the metal couples them): (i+1/2, r_j) with node j of column i or i+1 metal at any angle. These and the div B points with a
metal corner are left out of the checks.

### Particles
After the push a particle whose shape (z cells, radial nodes as in rz_metal_remove) lies on metal nodes at its nearest
angle phi_k is moved beyond the wall; migrate() absorbs it (its energy in E_part_out). Its current up to there is deposited.

### Stability, energy
In vacuum the metal only removes degrees of freedom (zeros) and reflects losslessly at z faces; with the staggered angles
the scheme is a staircase Yee grid in (r, phi) with the RIP transport in z. The RIP-CC limit dz^2 (lambda_max + wp^2) / 4 < 1
is unchanged (m_eff <= m lowers lambda_max). Test P4: random fields in 7 modes in a square corrugated pipe stay bounded
over 3000 steps.

## Axis cut (`numerics.rz_axis_cut = kappa`, any RZ run, RIP-CC)
lambda_max of the radial operators grows like m^2 / r_1^2 next to the axis, so with many modes the explicit step needs
dz <~ 2 dr / m (dz = dr / 8 for 16 modes): the pole problem of polar grids. With kappa > 0 mode m >= 2 is held at zero on the
nodes r_j < m_eff dr / kappa and the half nodes inside (a perfectly conducting rod for that mode only; the zeros applied as
the metal rules of an interior point, after every pass). Where the mode lives m_eff / r <= kappa / dr, so lambda_max (computed
by rz_lambda_max with the rods) no longer grows with M. Physics: mode m varies as r^m near the axis; a field of mode m at the
structure radius a is reduced by (r_c / a)^m at the rod radius r_c = m dr / kappa, so the rods remove nothing that a wake at
the axis region could feel for m dr / kappa << a. They do remove the high-m near field of off-axis charges within r_c (an
offset beam at x0 keeps its modes m < kappa x0 / dr). Off by default. The Gauss / div B checks leave out the rod points of
each mode. The self-field solve (species.self_field) drops the rod eigenmodes of the radial basis (eigenvalue 0 up to
round-off) as it drops the axis node of m >= 1: kept, their near-singular lines along z amplified round-off (tests/mpi.sh
found 1 rank 2 % off N ranks before the fix; now equal to 4e-13).

## Deck interface
```
grid { geometry = rz; modes = 9; ... boundary_r = absorbing }
numerics { solver = rip_cc;  rz_axis_cut = 1 }               # optional
metal {
  # a square corrugated pipe (examples/rz_square_pipe.deck)
  function = "..."                                            # metal where > 0; variables x y z r phi (lab, code units after units)
  phi_derivative = difference                                 # default; spectral for axisymmetric masks only
  enabled = true                                              # the same block serves Cart3D (function of x, y, z there)
}
```
`examples/rz_square_pipe.deck` writes a square corrugated pipe as a function (max(|x|, |y|) against the local half-width).
Not with iRIP, a TFSF plane, the boosted frame or the rings block.

## Tests (tests/test_rz_metal_phi.cpp, `build/tests test_rz_metal_phi`)
- P1 An axisymmetric function equals the rings block (periodic stack with z faces, random fields in modes 0..2, 300 steps):
  spectral against rings as it is 4.8e-15, difference against rings with the same m_eff 4.7e-15 of max|F|.
- P2, P3 Cutoffs k_c (k_z = 0; from the dominant frequency of a z-uniform field, sin(w dt / 2) = dt k_c / 2) against the
  eigenvalues of the discrete operator written independently in numpy (research/rz_metal_phi/xsection.py, discrete_fd: dense
  matrices of the masked curl operators on the staggered angles): agreement 4e-9 .. 3e-7 in all 12 cases (the frequency fit).
  Against the continuum (nr = 40, R = 1):

  | case | continuum | M = 4 | M = 8 | M = 16 | M = 32 (numpy) |
  |---|---|---|---|---|---|
  | square |x|,|y| < 0.6, TM11 | 3.7024 (analytic) | -3.1 % | -2.9 % | -2.6 % | -2.3 % |
  | square, TE10 | 2.6180 (analytic) | -7.7 % | -9.9 % | -8.2 % | -5.5 % |
  | 4 ridges r > 0.6, |phi - n pi/2| < pi/16, TM | 3.3677 (polar FD, Richardson) | -23 % | -14 % | -5.4 % | -2.7 % |
  | 4 ridges, TE (dipole) | 1.6661 (polar FD, nr = 80) | +8.0 % | -5.2 % | -4.1 % | -2.0 % |

  The staircase error is first order in r dphi and dr (the square's edges are not aligned with r and phi; nr = 40 alone
  limits TM11 to ~2 %); the spectral derivative gives TE10 = 0.81, 1.16, 1.36, 1.68 for M = 4, 8, 12, 16 (spurious).
- A1 Axis cut, nr = 40, 32 modes: lambda_max dr^2 = 1610 without, 6.4 with kappa = 1; the square TM11 with 16 modes equal
  with and without the cut to 7 digits at a 7.8 times larger dz.
- P4 A square corrugated pipe (C4 with z faces), random fields in modes 0..6, 3000 steps: W / W0 in [0.94, 1.00] (no growth;
  the staircase edges damp grid-scale noise), Gauss law and div B off the metal unchanged to 2e-14, fields inside zero.
- P5 Deck path: hot plasma with corrugated ridges (5 modes, shapes 1, 2): Gauss law 5e-16 of max rho, div B 2e-17, 11-13 %
  of the particles absorbed; iRIP rejected.

## Limits
- Staircase in r and phi: first order in the wall position (the corner radius of a corrugation is not resolved below dr,
  r dphi). The odd number of angles is never C4 symmetric: degenerate pairs (x, y dipoles) split by the staircase, ~3 % for the
  ridged pipe at M = 16, falling with M.
- No iRIP (as rings: the radial operator would depend on the column), no TFSF plane, no boosted frame.
