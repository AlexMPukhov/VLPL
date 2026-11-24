# O1: total-field / scattered-field (TFSF) laser source (`laser.tfsf_x`, `laser.angle`)

Status: implemented 2026-10-30 (decision 57), exact plane waves 2026-11-01 (decision 65); roadmap track O, item O1. Cart3D, RIP-CC and
iRIP, periodic or CPML in y, z (plane waves: periodic). RZ at normal incidence since 2026-11-03 (decision 68, section "RZ"). Not
yet: a moving window; restarts only with the plane waves (Cart3D).

## Why
At oblique incidence (3D, target normal tilted so that the specular direction is x) the laser comes in at 2 theta from x.
The left-edge injection writes only the inbound transport component T+ into the guard layer; an oblique wave also has a
component along -x (E_y - B_z = E_y (1 - cos alpha) for the y pair) and longitudinal fields, so it cannot be injected
there exactly. A TFSF plane inside the box injects any incident field: the total field to its right, the scattered (here:
the reflected) field to its left, which leaves through the open left edge and is recorded by O2 (docs/specs/xflux.md)
without the incident part.

## The incident field: an auxiliary vacuum grid (variant B, Alexander 2026-10-30)
The incident field is the discrete solution of a second `Fields` on the same domain and decomposition, with the same
solver and no particles (src/tfsf.*). At t = 0 the beam is written into it as initial fields left of the plane: the
built-in laser (Gaussian, Laguerre-Gauss, focus anywhere) rotated by `angle` about z (from x towards y) through the point
(x_s, y_center, z_center), its peak there at t = delay, the field of the entrance-plane formula retarded along the beam
and with the focus distance shifted by it (paraxial). pol z: E = a z, B = a (sin a, -cos a, 0); pol y:
E = a (-sin a, cos a, 0), B = a z. Every step the grid is advanced first, so the incident fields of t_n and t_n+1 are known.
- The envelope of a TFSF laser has compact support (Alexander's suggestion): the Gaussian up to 2 sigma (sigma =
  tau / (2 sqrt(ln 2)) of the field), a C2 spline step 1 - (10 v^3 - 15 v^4 + 6 v^5), v = |t|/sigma - 2, to exactly 0 at
  3 sigma (`LaserConfig::taper`; the lasers at the left edge keep the plain Gaussian).
- The incident grid is set to zero from the plane on at t = 0: the scheme with the corrections is the plain scheme for
  (total right of the plane, scattered + incident left of it), which starts from the incident grid only if that is zero
  right of the plane; a part already across would be a free error wave, carried back across the plane by diffraction or
  obliqueness. With the cut envelope this is nothing at normal incidence; at oblique incidence the transverse wings of the
  beam may reach across (a warning above 1e-4 of the peak: increase `delay`).
- The longitudinal fields from the discrete constraints (decision 101, Alexander 2026-11-01): the beam formula gives the
  transverse fields well, but its Ex, Bx hold only the tilt (- sin a a, sin a a), not the longitudinal field of the
  focusing (~ a / (k w0)). The div E, div B of that start are conserved by the scheme and stayed as a static Ex, Bx where
  the pulse started (7e-2 of the peak at w0 = 1.5 um, 0 deg; 0.2-0.3 at 30 deg). After the cut, Ex and Bx are summed from
  the plane (where the field is 0) towards -x from the discrete Gauss law and div B of the scheme (no charge):
  Ex(i) = Ex(i + 1) + dx mu_x div_perp E, Bx(i) = Bx(i + 1) + dx mu_x div_perp B (Fields::gauss_residual, max_div_b), one
  pass per rank along x (the right neighbour's first cell in the guard). The incident grid then starts with div E =
  div B = 0 on the grid (1e-17); no static field remains. Only outside the CPML layers in y, z (the range of the Gauss law
  check, where these constraints hold): a pulse reaching into a layer or to the zero guard of an open edge would turn
  that cut into a large Ex, Bx; the layers keep the formula's longitudinal fields.

## Where the plane enters the RIP step
The plane lies between the cells i_s - 1 (scattered) and i_s (total), i_s = round(tfsf_x / dx). In the transport cell i
takes T+ from cell i - 1 and T- from cell i + 1, plus the sources psi, phi at i -+ 1/2, which average Ex*, Bx* of the two
neighbour cells (mu_x); predictor and corrector are local in x. Per transverse pair:
- cell i_s: + T+_inc(i_s - 1), and + half of the incident Ex*, Bx* of i_s - 1 in the sources at i_s - 1/2;
- cell i_s - 1: - T-_inc(i_s), and - half of the incident Ex*, Bx* of i_s.
Ex*, Bx* of the incident field: RIP-CC its predictor of t_n; iRIP the box mean of t_n and t_n+1 in both passes (in the
first pass the incident Ex, Bx of t_n+1 across the plane are known and go into the right-hand side, so the implicit solve
along x needs no change).

## Tests (`tests/test_tfsf.cpp`)
T1 vacuum, a plane wave at 0 deg and a focused Gaussian beam (waist 1.5 um, focus 1 um beyond the plane) at 0 and 30 deg,
both polarisations, RIP-CC and iRIP: the field left of the plane <= 1e-15 of the peak (no leakage), right of it equal to
the incident grid to 2e-15. (Before the cut at the plane, the leakage followed the initial overlap of the pulse with the
plane: 9e-8 for an overlap of 4e-6, 3e-5 for 1.4e-4.) The incident grid starts with div E, div B <= 1e-16 (dx / peak; before
decision 101: 1.4e-2 at 0 deg, 0.16 at 30 deg), and at 0 deg it holds 4e-3 of the peak after the pulse has left (waves
partly reflected at the open x edges, trapped in the periodic y box; before: a static Ex, Bx of 7e-2).

## Cost
One more field step per step on the whole domain (vacuum, no particles) and two more sets of fields. Later: restrict it
to the slab left of the plane plus a margin, and drop it once the pulse has passed.

## With CPML layers in y, z (2026-10-30)
The incident grid gets the same layers (its own `Pml`, the same coefficients and b floor), so it is the discrete solution
of the same scheme. Where the plane crosses a layer, the transport sources at the plane carry CPML memories
(psi <- b psi + a g); the map is linear, so the half incident source of each side goes through its own memory
(8 per transverse point: psi_y, psi_z, phi_y, phi_z for the total and the scattered side) with the coefficients of the
layers. RIP-CC takes the predictor Ex*, Bx* of the incident grid itself (in a layer it has CPML terms too); iRIP reads
the memories in the first pass and updates them in the second, as `irip_pml_pass`. Test: the focused beam at 30 deg with
12 CPML cells in y, both solvers and polarisations: leakage <= 1e-15, total - incident <= 2e-15, Gauss law <= 6e-15.

## Angles and the plane wave of one k_y (2026-10-31)
- |angle| <= 45 deg (Alexander, decision 60): a steeper laser is to enter from a side boundary, as in GEM-PIC-RZ (not
  yet implemented); larger angles are refused.
- `waist = 0`: a plane wave of k_y = k0 sin(angle), at the plane a(t - sin a (y - y_c)) with the envelope moving along x
  at cos a; needs periodic y with sin(angle) n_y dy = m lambda0. Used by the oblique Fresnel test
  (docs/specs/fresnel.md).

## Exact plane waves instead of the incident grid (`laser.tfsf_incident = modes`, 2026-11-01, decision 65)
Why (Alexander, 2026-11-01): the incident grid must hold the whole pulse between the left x absorber and the plane at
t = 0. At 45 deg a 13 fs pulse with a 2 um waist spans ~17 um in x and y there, so the box of the oblique HHG benchmark
was 3.3 times Smilei's (which injects at its x boundary) and the incident grid doubled the field work: 4.5 times the cost.

The corrections at the plane read the incident Ey, Ez, By, Bz and Ex*, Bx* only in the cells i_s - 1 and i_s. With
`modes` these are sums of plane waves, each an exact discrete solution of the solver (`plane_mode`, the eigenvector
builder of the side injection, decision 61: discrete dispersion and polarisation from the one-step matrix), so the
leakage stays at round-off, without an incident grid:
- The beam is its angular spectrum on the lattice of the periodic y, z box (k_y = 2 pi m / L_y, k_z likewise), times the
  Fourier sum of the envelope (field exp(-2 ln2 t^2 / tau^2), as LaserConfig; components w = 1 + nu, |nu| <= 12 sqrt(al),
  period of the sum longer than run time + box diagonal):
  A = i a0 e^{-i phase} (dnu / 2 pi) sqrt(pi / al) e^{-nu^2 / 4 al} (sqrt(pi) w0 J / L_y) e^{-q^2 w0^2 / 4} (sqrt(pi) w0z / L_z) e^{-k_z^2 w0z^2 / 4},
  q = -kx sin a + ky cos a (continuum kx), J = |dq/dk_y|; the focus r_f = plane point + focus n, the peak there at
  delay + focus, all components in phase there. waist = 0: one k_y = sin a; waist_z = 0: k_z = 0. Components below
  1e-12 of the central one are dropped; k_x of each from the discrete dispersion (bisection), so the discrete beam is the
  exact Gaussian focus in the continuum limit (no paraxial approximation).
- Per step: the sums over w per (k_y, k_z) on the host, then two device passes (over k_z, then k_y) for the plane
  points of the two cells (with one transverse guard): O(N_ky N_kz N_w + N_y N_z N_ky) per step, no exchange.
- The pulse may start anywhere (outside the box); the plane may sit next to the left x absorber; restarts need nothing.
  A warning if the pulse is already at the plane at t = 0. Not with CPML in y, z (the plane waves are those of the
  periodic box), not with Laguerre-Gauss modes.
- Deck: `laser { tfsf_x, angle, tfsf_incident = modes }` (default `grid`, the incident grid, unchanged).

Tests (`build/tests test_tfsf_modes`, tests/test_tfsf.cpp T3): vacuum, the T1 deck 22.5 um long (the pulse stays off the open
right edge until 40 fs): leakage 1.0e-15 .. 2.0e-15 of the peak at 0, 30, 45 deg, polarisations y and z, RIP-CC and iRIP,
a round beam (waist_z, 14567 plane waves); the plane 8 cells from the open left edge and from the left x layer (4 lambda0)
with the pulse starting outside the box (delay 40 fs): 1.4e-15, 1.6e-15. Against the incident grid at 0 deg (plane at
12 um, so the grid's pulse fits into the box): peak over the run 1.0035, Ey at 40 fs 4.5 % in L2 (paraxial entrance formula
with a cut envelope against the exact Gaussian focus). Gauss law <= 8e-15. About 2600 .. 3300 plane waves for the
z-uniform beam at lambda0/32.

Edge echoes: an open edge reflects an oblique wave by (1 - cos a) / (1 + cos a) and the x layers by ~5e-4 at 45 deg
(4 lambda0). The incident grid contains that echo of the incident beam (it cancels at the plane); with plane waves it is
scattered field and crosses into the record, including the echo of the low-level wide-angle content of the focus that
is everywhere in the box from t = 0 (~1e-6 off an edge 11 um away in the first test version). Keep the right layer
far enough and absorbing.

## RZ (2026-11-03, decision 68)
Why (Alexander, 2026-11-03): the open left z edge injects the laser exactly at normal incidence and lets the reflected
light out, but it returns (1 - cos a) / (1 + cos a) of a wave leaving at the angle a (3 % at 20 deg, 7 % at 30 deg in field):
the harmonics of CHF leave diverging from their focus. Absorbing z layers (docs/specs/x_absorber.md "RZ") would damp a laser
injected at the edge, so the laser comes in through a TFSF plane inside the box, as in Cart3D.
- The plane lies between the z cells i_s - 1 (scattered) and i_s (total), i_s = round(tfsf_x / dz); normal incidence only.
- The incident field is the discrete solution of an auxiliary vacuum FieldsRZ (src/rz_tfsf.cpp: the same domain, modes,
  radial grid and solver, no currents). The laser enters it through its open left edge exactly as rz_laser.cpp (focus and
  delay count from z = 0), so there is no initial overlap with the plane. It has the radial absorber and the right z layer of
  the main grid, the same scheme wherever the total field lives, but no left layer (it would damp the laser). Waves of the
  incident grid that cross the plane leftwards (the echo of its right layer) are removed from the scattered region by the
  correction itself.
- Per step: the incident grid of t_n is kept, then the grid is advanced (`RzTfsf::begin`, called at the start of
  FieldsRZ::step after the inflow hooks; for iRIP the box means (Ez, Bz)(t_n + t_n+1) / 2 are taken here, before the radial
  damping, which the main grid's implicit solve does not see either).
- Corrections in FieldsRZ::pass after the transport (`RzTfsf::correct`), per mode, the RZ form of the Cart3D ones: the total
  cell i_s takes T+ of i_s - 1 and the sources at i_s - 1/2, which average Ez*, Bz* of both cells (mu_z): + the incident
  (E_r + B_phi) and (E_phi - B_r) of i_s - 1 with half its Ez*, Bz* in Psi_r, Phi_phi, Psi_phi, Phi_r (axis rules of
  FieldsRZ::pass: m = 1 at j = 0); the scattered cell i_s - 1 takes T- of i_s: - the incident (E_r - B_phi), (E_phi + B_r)
  of i_s with half its Ez*, Bz*. E_phi, B_r on the axis are no DOFs (not corrected). Ez*, Bz* of the incident field: RIP-CC
  its own predictor of t_n, iRIP the box means in both passes.
- The Gauss and div B points on the plane are skipped in the diagnostics (they mix scattered and total fields).
- iRIP with `numerics.closure = o5` (decision 78): the incident grid runs o5, its box means get C_perp, then Cz, and the
  Cz stencil of the main grid across the plane gets the incident value of the far cell (docs/specs/overdense.md "O5 with
  the RZ TFSF plane").
- Not yet: restarts (the incident grid is not in the checkpoint), the moving window, oblique incidence (not representable in
  the azimuthal modes of one axis).

Tests (`build/tests test_rz_tfsf`, R1): a focused beam (waist 1 um, focus 3 um, 6 fs, 2 modes, lambda0/32, dr = 2 dz) through
the plane at 1.5 um, z layers of 1 um, radial absorber, vacuum: left of the plane 2.3e-15 (RIP-CC) and 1.9e-15 (iRIP) of the
peak field, total - incident right of it 4.1e-15, 3.9e-15 over the whole run; the energy W_in + Enet_left + Enet_right + E_abs_r
reaches the injected E_inj to -4.2e-4 (the O(dz^2) of xflux with transverse structure). iRIP with o5: 3.1e-15 and 4.1e-15. tests/mpi.sh: examples/rz_tfsf.deck
(a 10 n_c foil) on 1 and 8 ranks along z, RIP-CC and iRIP, the plane on a rank boundary.

## Next
The moving window with a TFSF plane; CPML in y, z with the plane waves (the beam must vanish in the layers).
