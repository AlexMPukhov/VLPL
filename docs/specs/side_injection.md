# Laser injection through the side boundaries (y, z) of Cart3D

Status: first version implemented 2026-10-31 (decision 61): the TFSF tube (option S) with the exact discrete
plane wave (provider P1), RIP-CC and iRIP, CPML or periodic y, z, MPI, restarts. The flying focus (luminal Bessel
pulse, exact, tabulated in the window moving at c) 2026-10-31 (decision 62), stage A: vacuum or gas at the faces.
Stage B 2026-11-01 (decision 63): a uniform background plasma, the incident wave of that plasma and a linear cold fluid
outside the kinetic region (Alexander's choice of the cold-fluid shell over particles outside the tube). Next:
continuum providers (P2: rotated Gaussian, LASY).
Prototype and the evidence for the choice: `research/side_tfsf/` (proto.py, results.txt, disp3d.py).

## Why
- **Flying focus, luminal beatwaves, multi-mode LWFA in plasma channels** (Alexander, 2026-10-31; low plasma
  densities): a converging conical wave (axicon, axiparabola, luminal Bessel pulse) comes from far outside the box and
  its rays cross the transverse boundary over the whole length of the run, in a moving window. GEM-PIC-RZ does this
  with `flying_focus` and `laser { file; inject = boundary }` (total field / scattered field through its radial
  absorber, `docs/flying_focus_handoff.md` there). The cone angle is small: "side" is where the rays enter.
- **Steep oblique incidence (60-90 deg to x)**, track O: the x plane of O1 takes at most 45 deg (decision 60).
The x plane of O1 cannot do either: its incident grid must hold the beam before it crosses the plane, it is refused
with a moving window, and a cone entering along the whole box never crosses one x plane.

## Physics and equations
A total-field / scattered-field tube: faces y = const (global nodes lo_y, hi_y) and / or z = const. Inside, the
main grid holds the total field, outside it the scattered field (whatever the plasma reflects or radiates), which the
CPML layers absorb. The incident field is a vacuum solution (or that of a background plasma, below) and the currents
are the same in both representations,
so the scheme stays the plain scheme for (total inside, scattered outside) if every stencil that couples the two
sides adds the incident value of the far point.

The RIP-CC step in the transverse plane is a Yee-like half-staggered grid (Ex, Ez, By at y nodes, Ey, Bz, Bx at
y + 1/2; the same along z, src/stagger.hpp), and every y or z derivative is a two-point difference: the predictor
(Ex*, Bx*), the transport sources Psi_y, Psi_z, Phi_y, Phi_z at i + 1/2, the corrector (Ex, Bx). The x transport
never mixes y, z cells. A difference whose two values lie on different sides of a face gets the incident value of the
far point, + for a total-field output point, - for a scattered one; the incident Ex*, Bx* are the RIP-CC predictor of
the incident field (iRIP: the mean of t_n and t_n+1), the corrector takes the mean of t_n and t_n+1. On the closed
tube the corrections of the faces add (each difference runs along one axis).

**The incident field (P1)** is a sum of plane waves, each an exact discrete solution of the solver:

    RIP-CC  sin^2(w D/2) = sin^2(kx D/2) + D^2 Lam/4 cos^2(kx D/2)
    iRIP    tan^2(w D/2) = tan^2(kx D/2) + D^2 Lam/4,        Lam = 4/hy^2 sin^2(ky hy/2) + 4/hz^2 sin^2(kz hz/2)

(checked against the eigenvalues of the one-step matrix, research/side_tfsf/disp3d.py; both polarisations are
degenerate). The polarisation is the vector of the two-dimensional eigenspace of the one-step matrix whose E (at the
staggered points) is closest to the wanted s or p direction of that component (least squares). With such an incident
field the leakage is at round-off for any angle.
- Continuous wave: k = k0 n, w from the dispersion (|w - 1| = O(h^2)).
- Pulse: the envelope exp(-4 ln2 t^2 / tau^2) at the reference point as a Fourier sum, components w = 1 + nu with
  |nu| <= 12 / sqrt(4 ln2 / tau^2) (amplitude 2e-16 of the peak there), spacing 2 pi / T with T = 2 (run time + box
  diagonal + 12 sigma): no periodic image during the run. Open x: every component along n (|k| from the dispersion
  at w); periodic x: the components share k_x (on the lattice of the box) and the direction of k_perp, the natural
  pulse of a box periodic along x. Components below the cutoff of the family are dropped (an error if they carry more
  than 1e-6 of the central one).
- Direction n = (cos e cos a, cos e sin a, sin e) (`angle` a from x towards y, `elevation` e towards z);
  polarisation `z`: E along k x (z x k) (z for e = 0), `y`: along z x k (in the x-y plane for e = 0).

## The flying focus (`side.wave = flying_focus`, src/side_luminal.cpp)
The luminal Bessel pulse of GEM-PIC-RZ's `flying_focus` block: every component has w - k_x = kappa = omega0 (1 - cos
theta), so the pattern moves at c, and on the axis E = a0 exp(-zeta'^2 / L^2) cos(cos(theta) k0 zeta' - kappa omega0 t
+ cep), zeta' = x - c t - x_center.
- Frequencies w = 1 + nu, k_x = cos(theta) + nu, the envelope as a Fourier sum to e^{-36} of its centre (|nu| <= 12/L),
  spacing 2 pi / P with P = 2 (window + 12 L + R tan(theta)) (no periodic image in the window; R the farthest corner of
  the box from the axis). k_x > 0 for all of them: L > 12 / (k0 cos(theta)) (refused otherwise).
- Azimuths phi_l = 2 pi (l + 1/2) / N_phi of k_perp, N_phi = 1.25 k_perp,max R + 12 (even): the quadrature of the cone,
  exact for the Bessel orders that matter inside the box; |k_perp| from the discrete dispersion at each phi (the grid
  is not isotropic). E of a component along p - (p . k^) k^, p the polarisation on the axis; the amplitudes scaled so
  that the envelope peak on the axis is a0.
- In the window moving at c from t = 0 (required: window.velocity = 1, start = 0) the window offset is t, so the phase
  k . r - w t = k_x (local x) + k_perp . r_perp - kappa t: the corrections at the face columns are a table H(column, x)
  times e^{-i kappa t}, built once as a sum over the azimuths per frequency and column, then over the frequencies per
  x (8 complex per face cell: Ex*, Bx*, the four transported fields, Ex, Bx; for the solver of the run).
- The initial field fills the whole tube: per chunk of frequencies a sum over the azimuths at every transverse cell,
  then over the frequencies along x (cells x frequencies, not cells x components).
- The tube is open at the x ends of the window: the pattern must be negligible there (a warning above 1e-4 of a0; the
  rings at radius r run ahead of the axial focus by about r theta / 2). In test_side_ff a window 20 um long for
  L = 1.8 um left 1e-12 at the head (iRIP), 24 um gave 5e-15.

## Background plasma (`side.background_density`, decision 63)
Real hydrogen is ionised far outside the box (GEM-PIC-RZ found the same for its feeding layer), so the faces lie in
plasma, and the incident wave must be that of the plasma. In the scattered region the field update needs the plasma
current minus that of the incident wave. Particles there would have to gather scattered + incident fields and carry
j - j_inc (for the flying focus a table of E, B and j_inc over the whole shell, and still only as exact as the particle
response matches the linear one). Chosen instead (Alexander, 2026-11-01): **a linear cold fluid** outside the kinetic
region.
- The fluid current at the points of J: j^{n+1/2} = j^{n-1/2} + D wp^2 F (Ex, mux Ey, mux Ez)^n (Jy, Jz at i + 1/2 take
  the mean of Ey, Ez at i, i + 1, as the Gauss law), added to the deposited j before the field step
  (`SideTfsf::fluid_current`, after the guards are reduced). It is driven by the stored field: total in the tube,
  scattered outside. So the fluid outside carries exactly j - j_inc, and the incident wave is again an exact discrete
  solution: the plane waves of the one-step matrix with the fluid (9 x 9: E, B, j):

      RIP-CC  TE (Ex = 0): sin^2(w D/2) = sin^2(kx D/2) + D^2/4 cos^2(kx D/2) (Lam + wp^2)
              TM (Bx = 0): sin^2(w D/2) = sin^2(kx D/2) + D^2/4 cos^2(kx D/2) (Lam + wp^2 (1 - D^2 Lam/4))
      iRIP    both:        tan^2(w D/2) = tan^2(kx D/2) + D^2/4 (Lam + wp^2 / cos^2(w D/2))

  (research/side_tfsf/check_fluid.py: the eigenvalues of the one-step matrix to 2e-14 for wp^2 = 6e-4 .. 0.3; TE has
  Ex = 0, TM Bx = 0). With RIP-CC the two polarisations split, so every component of a provider becomes a TE and a TM
  wave of the same w with their own |k| (plane wave: the same direction; flying focus: the same k_x and azimuth, |k_perp|
  from each dispersion), fitted together to the wanted E at the reference point; iRIP stays degenerate. The eigenvector
  is the null space of the one-step matrix with a constraint row (Ex = 0 or Bx = 0), so a near degeneracy (the split
  is ~1e-10 at small cones) does not matter; the one-step residual of every component is checked (< 1e-10, ~5e-16).
- The fluid fraction F: 1 outside the kinetic region, 0 inside it and in the CPML, and at its edge (the particles end 2
  cells inside the faces) the complement of the coverage of the point by a uniform particle plasma ending there: the
  deposit weights a point with the shape of order p across the motion and p - 1 along it (Esirkepov: Jy at the half
  node in y), so particles + fluid give the uniform current. `fluid = all`: F = 1 everywhere (tests, a linear plasma).
- At t = 0 the fluid in the tube gets F j^{-D/2} of the incident wave (the eigenvector's j). The fluid current moves with
  the window (the entering column at rest) and is part of the checkpoint (`rippic_fluid_jx, jy, jz`).
- The kinetic plasma must reach the edge of the kinetic region with the background density (not checked); its
  particles respond kinetically, the fluid linearly: the difference (the PIC response vs the linear one, O(h^2) and
  the nonlinear and thermal part) radiates from the edge. The fluid holds where the field at the faces is weak:
  a Bessel ring at r has a ~ a0 (2 / (pi k_perp r))^{1/2} (0.4 at r = 11 um for a0 = 2, 0.2 rad).
- The Gauss law and div B are checked inside the kinetic region away from its edge (5 cells inside the faces): the
  fluid's charge is not part of rho. Flying focus: 2 kappa - kappa^2 > wp^2 (a conical wave at omega0, as GEM-PIC-RZ);
  frequencies of the spectrum below the plasma cutoff of the cone are dropped if they carry < 1e-6 of the peak.

## Discretisation
- `src/side.*`: `SideTfsf` holds the components (wave vector, w, the six complex amplitudes at the stored index, the
  RIP-CC predictor of the mode, (1 + e^{-iwD})/2). Three kernels add the corrections before the exchange that follows
  the predictor, the transport and the corrector, in all four step paths (fields.cpp, pml.cpp, irip.cpp box_pass,
  irip_pml.cpp). For iRIP the corrections are known values (the incident Ex, Bx of t_n+1 too), so they go into the
  right-hand side of both passes and the implicit operator is unchanged.
- The kernels run over the (j, k) columns next to a face (half node lo - 1, node lo, half node hi) times all x: a
  balanced flat loop. The phase is a product of tables e^{i kx D i}, e^{i ky hy j}, e^{i kz hz k} per component and a
  factor e^{i (kx x0 - w t)} uploaded per step (x0 the window offset): no sin, cos in the kernels, no exchange, any
  decomposition; restarts need nothing stored.
- At t = 0 the incident field is added to the total-field cells. Gauss law and div B are checked inside the tube,
  one cell away from its faces (the fields jump there by the incident field).
- The faces must lie at least one cell outside the CPML layers (default: `margin` = 2 cells), so no CPML memory is
  involved. The plasma must end 2 cells inside the faces (refused at the start): outside the tube particles would see
  only the scattered field (a background plasma continues there as the cold fluid). Particles that leave the kinetic
  region during the run see the total field up to the faces and only the scattered field beyond.
- The tube is open at its x ends: a warning if the incident pulse reaches an x end during the run (open x; for a
  continuous wave always). With periodic x there are no ends.
- Not with Yee, RZ or the TFSF x plane of O1.

## Deck interface
```
side {
  wave         = plane         # plane | flying_focus
  faces        = yz            # y, z or yz
  margin       = 2             # cells between the CPML layers (or the edges) and the faces
  y_lo, y_hi, z_lo, z_hi       # or the faces directly (lengths; nodes at or inside them)
  a0           = 1
  angle        = 75 deg        # from x towards y
  elevation    = 0 deg         # towards z
  polarization = z             # z: E along k x (z x k); y: along z x k
  duration     = 0 fs          # 0: continuous wave (fills the tube at t = 0); else as laser.duration
  delay        = 0 fs          # the peak passes (x_center, y_center, z_center) then
  x_center, y_center, z_center # default: the box centre
  phase        = 0
}
```
With periodic x, cos(elevation) cos(angle) nx dx must be a multiple of lambda0 (k_x is then snapped to the lattice).
The flying focus (window at c from t = 0, open x):
```
side { wave = flying_focus
       a0 = 1                  # the envelope peak on the axis
       cone_angle = 200 mrad   # theta at omega0
       length_1e = 5 um        # or duration (intensity FWHM on the axis), as GEM-PIC-RZ
       x_center = 22.5 um      # the envelope centre at t = 0 (absolute x)
       y_center, z_center      # the axis (default: the box centre)
       polarization = y        # E on the axis
       cep = 0
       faces, margin, y_lo ... # as above }
```
In a background plasma (either wave):
```
side { ...
       background_density = 5.741e-4   # n_c (1e18 cm^-3 at 0.8 um): the incident wave of that plasma
       fluid = shell                   # shell: the cold fluid outside the kinetic region (particles up to 2 cells
                                       # inside the faces, with this density at the edge); all: everywhere }
```

## Tests
- `build/tests test_side` (tests/test_side.cpp): vacuum, nx dx = 4 lambda0 periodic, CPML of 10 cells in y and z,
  a tube of 24 cells. Continuous waves at 60 and 75.5 deg (y faces) and at 30 deg elevation in the y-z plane (y and
  z faces and the corners), both polarisations, RIP-CC and iRIP, and pulses of 94 components at 60 deg (RIP-CC, iRIP):
  outside the tube 3e-15 .. 1e-14 of the peak, total - incident <= 1.5e-14 (which also checks the discrete dispersion
  and polarisation of the components against the solver), Gauss law <= 9e-15, div B <= 1.6e-14.
- `tests/mpi.sh`: `examples/side_wave.deck` (a pulse at 75.5 deg onto an underdense layer, CPML y, z) on 1 and NP
  ranks with RIP-CC and iRIP, and the restart from a checkpoint of NP ranks on NP (exact) and 3 ranks.
- Cost: the corrections cost components x face columns x nx per kernel. `examples/side_wave.deck` (128 x 64 x 64,
  150 components, 24 threads): field step 2.8 -> 7.6 s (RIP-CC), 12.2 -> 19.7 s (iRIP) over 240 steps; a
  continuous wave (1 component) costs nothing measurable.

- `build/tests test_side_ff`: the flying focus (cone 20 deg, L = 1.8 um, 5628 components = 201 frequencies x 28
  azimuths) in a vacuum window of 24 um moving at c, CPML y, z, RIP-CC and iRIP, against the incident field of the
  components over the whole box: outside the tube 1.5e-15 / 3.3e-15, total - incident 5e-15, Gauss law and div B 3e-15.
  The same in a plasma of 0.01 n_c with the fluid everywhere (11256 TE and TM waves with RIP-CC, 5628 with iRIP): outside
  1.4e-15 / 4.5e-15, total - incident 5e-15 (fields and fluid current), Gauss law 6e-15, div B 4e-15.
- `build/tests test_side_plasma`: (a) the plane waves of test_side in a plasma of 0.05 n_c (omega_p = 0.22 omega0),
  the fluid everywhere: outside 3e-15 .. 1e-14, total - incident (fields and fluid current) <= 1.3e-14, Gauss law
  <= 9e-15, div B <= 1.1e-14, one-step residual of the modes <= 6e-16. (b) cold kinetic electrons of 0.05 n_c (2 x 2 x 2
  per cell) in the kinetic region and the fluid shell, a weak pulse (a0 = 1e-3) at 60 deg: outside 1.4e-2 of the peak,
  against 2.5e-1 with the same particles and a vacuum shell (the stage A setup with the plasma at the faces).
- `tests/mpi.sh`: `examples/side_plasma.deck` (the flying focus in 0.01 n_c, kinetic electrons, fluid shell) on 1 and NP
  ranks, RIP-CC and iRIP, and the restart from step 30.
- `bench/flying_focus/rip_plasma.deck` against GEM-PIC-RZ (1e18 cm^-3, cone 0.2 rad, 30 fs, 0.3 mm; README there): the
  laser on the axis is the analytic luminal pulse of the plasma (1.001 a0, carrier phase constant), the wake stationary
  from 60 um on; a0 = 0.3: the wake within ~10 % of GEM-PIC-RZ; a0 = 2: the same cavity, behind it 35 % weaker.
- `bench/flying_focus/`: against GEM-PIC-RZ's flying_focus (the same pulse, cone 0.2 rad, 20 fs, 0.3 mm) and the
  analytic on-axis field: VLPL 2.4e-4 of a0 at every z (amplitude 1.00036 a0, phase and envelope fixed), GEM-PIC-RZ
  within 1-3 % in amplitude, 0.14 um envelope shift and its dispersion phase offset (rms 0.23).

## Options considered (2026-10-31, before the choice)
The prototype (`research/side_tfsf/proto.py`: the RIP-CC step 1D in y with one Bloch mode in x, two faces 4 lambda
apart, dx = lambda/32) confirmed the corrections. With the exact discrete plane wave the leakage is 5e-15 .. 2.5e-13
at 5 .. 90 deg, one wave or two at +-angle. With the continuum (exact Maxwell) plane wave the entry error is second
order in dx: 1e-2 at lambda/32 for steep waves (6.5e-4 at lambda/128), 8e-4 at 5 deg (5e-5 at lambda/128). The
total - incident difference across the tube is the y dispersion of RIP-CC itself (the scheme is free of dispersion only
along x; dt/dy < 1/sqrt(2) by stability): 0.13 rad after 4 lambda at 90 deg and lambda/16, so steep lasers need
dy, dz <~ lambda/24 .. lambda/32 whatever the injection.
- **S, a TFSF tube with a provider** (chosen): P1 discrete plane-wave sums (exact; for a luminal pattern in the moving
  window, omega - c k_x = kappa for every component, a table G(zeta, transverse) times e^{-i kappa t} per face, the 3D
  analogue of the G(r, zeta) table of the GEM-PIC-RZ flying focus) and P2 continuum formulas (any beam, O(dx^2)).
- **A, a y plane with an auxiliary incident grid** (O1 turned by 90 deg): exact, but the incident grid must contain the
  beam from its start up to the plane, a grid wider than the box; no moving window.
- **R, relaxation in a damping layer** (as GEM-PIC-RZ): F <- F_inc + d (F - F_inc) at the y, z faces. No stencil
  corrections, but the layer replaces the CPML (an oblique scattered wave needs ~4 lambda0 per face, O3), the incident
  field is needed in the whole layer volume, and it is never exact.
Choices of Alexander (2026-10-31): S with P1 + P2; faces y and z; the plasma ends at the faces; the incident field
must vanish at the x ends (a warning); the first provider the plane wave (P1).

## References
docs/specs/tfsf.md (O1), docs/specs/x_absorber.md (O3), docs/specs/pml.md; decisions 57, 59, 60, 61. GEM-PIC-RZ
docs/flying_focus_handoff.md and docs/input_deck.md (`flying_focus`, `laser.inject = boundary`). A. Taflove,
S. C. Hagness, Computational Electrodynamics, ch. 5 (TF/SF for staggered grids).
