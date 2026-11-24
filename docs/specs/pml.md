# PML in y and z

Status: implemented (`src/pml.*`, `grid.boundary_y/z = pml`, block `pml`, decision 31). The 2D
prototype is in `research/pml/`.

## Physics and equations
Radiation and the wake fields that reach the transverse edges of the box should leave it instead of
coming back through the periodic boundary (and a beam should not see its periodic images). CPML,
convolutional perfectly matched layer (Roden & Gedney 2000): in a layer of L cells at an edge, every
transverse derivative ∂_y F of the field step becomes

    ∂̃_y F = ∂_y F / κ(y) + ψ,      ψ <- b ψ + a ∂_y F,
    b = exp(-(σ/κ + α) Δt),        a = σ (b - 1) / (σ κ + κ² α),

with σ = σ_max (d/L)^m graded with the depth d into the layer, σ_max = (m + 1) ln(1/R0) / (2 L h).
Behind the layer the fields are zero (a conducting wall). The same holds along z.

## Discretisation (RIP-CC)
- Every transverse derivative has its own ψ, one value per location, updated once per step. There
  are twelve call sites in the step, each with a y and a z term:
  - predictor: ∂ Bz, ∂ By, ∂ Ez, ∂ Ey at t_n;
  - transport sources: ∂ μ_x Bx*, ∂ μ_x Ex* at n + 1/2;
  - corrector: ∂ ⟨Bz⟩, ∂ ⟨By⟩, ∂ ⟨Ez⟩, ∂ ⟨Ey⟩.
  The y terms need ψ only in the y layers, the z terms only in the z layers.
- σ, κ, α and the coefficients a, b are evaluated where the derivative sits: the node j or the half
  node j + 1/2.
- The transport sources are computed once per x location (i + 1/2) into scratch arrays when a PML is
  on. Without a PML the step is unchanged and bit-identical.
- The Gauss law is not conserved inside the layer. It is checked outside the layers only.
- The moving window shifts the ψ arrays with the fields.
- `grid.absorber_particles` (decision 111, src/absorber_particles.cpp):
  - `remove` (Alexander: "keep the current deposition out of the PML"; the default until decision 115): after the deposit and the filter every
    current component at a point of positive depth is set to zero (J_y at the half nodes, J_x, J_z at the nodes along y; likewise
    along z), and a particle is removed once it is in a layer (y < L h or y > (N - L) h). The current of a particle stops at the
    face, so the charge continuity is broken there.
  - `conserve` (the default, decision 115): the current is deposited into the layer (charge conserving) and a particle is removed
    once its shape no longer touches a physical node, plus one cell; the Gauss law in the physical domain holds, the removed charge
    stays in the layer. Why the default: remove leaves the field of the vanished charge next to the faces inside the physical
    region (Gauss residual 2.6e-2 there in P5); conserve keeps the Gauss law at round-off and stops the streaming instability
    of decision 111 as well.
  - `keep`: the particles cross the layer and leave at the outer edge (the behaviour before decision 111).
  The removed number and charge per species are printed at the end of the run.
  Why: a plasma streaming through the window of a boosted run (u = -5.9, gamma 6) that fills the box into the layers drives a
  near-uniform transverse field that grows along the plasma's path (keep: E_phi m1 near the axis 3.1e8 V/m at 302 fs, 5.7e9 at
  453 fs); periodic y, z or a plasma column clear of the layers stay at 6.5e6 - 6.8e6. With remove or conserve the same run
  stays there (6.4e6 / 6.5e6 at 302 fs, 6.8e6 / 6.8e6 at 453 fs). The face error of remove (Gauss residual of the code's check,
  outside the layers + 1 cell): 1.4e-8 from the first steps, constant, down to round-off (7e-17) after ~400 fs; E_y next to the
  faces 4.9e6 against 5.8e6 inside, flat; mean rho next to the faces 2.2 C/m^3 (conserve -0.9) against the plasma's 5e5:
  no growing error (nci/pmlrm, 455 fs, 400 x 80 x 80, PML 16 cells).

## Prototype results (2D, vacuum, λ = 16 dx, L = 24, R0 = 1e-6, m = 3; research/pml/sweep.txt)
Maximum interior error of a Gaussian packet against a reference without walls, relative to the
amplitude. The incidence angle φ is measured from the x axis (90° = normal to the y edge).

| φ | dy = 2 dx: CPML | κ = 4, α = 0.05 | sponge | dy = 5 dx: CPML | κ = 4, α = 0.05 | sponge |
|---|---|---|---|---|---|---|
| 90 | 9e-7 | 4e-4 | 1.5e-4 | 9e-5 | 5e-2 | 9e-5 |
| 60 | 2e-6 | 2e-4 | 2e-3 | 4e-5 | 6e-2 | 1e-4 |
| 45 | 3e-5 | 6e-5 | 1.2e-2 | 6e-6 | 5e-2 | 7e-4 |
| 30 | 5e-4 | 3e-5 | 4.8e-2 | 1.8e-4 | 2.2e-2 | 8.9e-3 |
| 20 | 3.3e-3 | 6.6e-4 | 9.1e-2 | 2.4e-3 | 9e-3 | 3.3e-2 |

- Plain CPML (κ = 1, α = 0) is the choice. The graded κ and α help near grazing incidence on the
  fine grid but reflect badly on the coarse one, which is the typical LWFA case (h_⊥ ≈ λ/3).
- The matched sponge is fine only near normal incidence.
- 20,000 steps of random fields: the energy only decays, for all three.

## Deck interface
```
grid { boundary_y = pml   boundary_z = pml }     # periodic (default) or pml, per axis
pml  { cells = 16   R0 = 1e-6   order = 3 }      # defaults
```
- The layers lie inside the box: the last `cells` cells at each pml edge.
- Plasma may be loaded into them; with `grid.absorber_particles = conserve` (default) or `remove` its particles are removed
  (above), with `keep` they leave through the edge behind the layer.
- Requires n >= 2 cells + 4 along each pml axis.
- The beam self-field solve still assumes periodic y, z: keep beams far from the layers.

## Tests
- Fast (tests/test_pml.cpp):
  - P1: a packet into a 16-cell y layer (normal and 45°) and a z layer (45°), against a reference
    in a taller periodic box. Error 5.4e-6 at normal and 2.0e-5 at 45° incidence; z equals y.
  - P2: random fields with layers in y and z, 1000 steps: the energy never grows and falls to 0.43.
    The rest sits in modes along the periodic x.
  - P3: a hot plasma leaving through y and z: max|dGauss| 5e-16 outside the layers, 29 % of the
    electrons gone.
  - P5: grid.absorber_particles, the hot plasma of P3 (P3 itself runs with keep), 60 steps: remove: 112171 removed, none
    left in the layers, no current there, charge balance Q + removed = Q0 to 1.0e-12, Gauss residual 4.6e-16 more than
    3 cells from a face and 2.6e-2 next to the faces (the charge that vanished there; max |J| 3.5e-2); conserve: 94087
    removed, Gauss 4.9e-16 everywhere, balance 1.1e-12; keep: none removed, Gauss 4.9e-16.
  - P4: the moving window with layers equals the fixed box shifted by wx, inside the layers too:
    difference 0 over 60 shifts. Without the ψ shift it is 0.25.
- MPI (tests/mpi.sh): the streaming deck with layers in y and z on 1 and 8 ranks: energies agree to
  1.2e-11, Gauss 5e-16 outside the layers.
- Cost: the field step with layers takes 0.16 s against 0.15 s without (streaming deck, 4 threads).

## References
- J. A. Roden, S. D. Gedney, Microw. Opt. Technol. Lett. 27, 334 (2000): CPML.
- J.-P. Berenger, J. Comput. Phys. 114, 185 (1994).
