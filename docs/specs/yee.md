# Yee solver (reference)

Status: implemented for periodic boxes (`src/yee.cpp`, `numerics.solver = yee`, decision 37).

## Why
RIP-CC's claims are comparisons with the standard scheme: numerical Cherenkov radiation of
relativistic beams and plasma, and emittance growth (Pukhov, JCP 418, 109622 (2020)). The code paper
(Gate A) needs these comparisons on the same particles, deposition and diagnostics. The Yee solver is
therefore a second field solver of the same code, not a separate program.

## Scheme
Staggering (descriptor `Yee` in `src/stagger.hpp`, array index (i, j, k) = the lower corner):

| quantity | position |
|---|---|
| E_x, J_x | (i+½, j, k) |
| E_y, J_y | (i, j+½, k) |
| E_z, J_z | (i, j, k+½) |
| B_x | (i, j+½, k+½) |
| B_y | (i+½, j, k+½) |
| B_z | (i+½, j+½, k) |
| ρ | (i, j, k) |

- J_a = ρ + e_a/2: the Esirkepov deposition on the charge grid is the continuity equation of the
  scheme. So div E = ρ at the nodes and div B = 0 at the cell centres hold to round-off.
- One step n → n+1. B is kept at integer time levels with two half steps, so the push gathers E^n and
  B^n exactly as with RIP-CC:

      B^{n+1/2} = B^n − (Δt/2) ∇×E^n
      E^{n+1}   = E^n + Δt (∇×B^{n+1/2} − J^{n+1/2})
      B^{n+1}   = B^{n+1/2} − (Δt/2) ∇×E^{n+1}

  This is the standard leapfrog; B at half levels is the same scheme written differently.
- Δt = cfl / (1/h_x² + 1/h_y² + 1/h_z²)^{1/2}, with `numerics.cfl` (default 0.98, must be < 1).
  Also required: ω_p Δt < 2.
- Numerical dispersion: (2/Δt) sin(ωΔt/2) = |K|, with K_a = (2/h_a) sin(k_a h_a/2). Waves are slower
  than light, which is the source of the numerical Cherenkov instability of relativistic plasma.
- The gather, the deposition and the charge density are the same template kernels as for RIP-CC,
  instantiated with the `Yee` descriptor (`with_stagger` in `src/particles.cpp`). Coloured tiles,
  the box deposit, sorting, MPI and the GPU work unchanged.

## Scope so far
Periodic boxes only. The boundary physics of VLPL is RIP-CC's:
- the laser and the open x boundary use RIP-CC's transport vector;
- the CPML is written for its step;
- the moving window shifts one cell per step.

A Yee deck with an open x boundary, `pml`, a laser or the window is rejected with a message. The beam self-field
(`self_field`) works in the periodic box with Yee's own Gauss law and a co-moving neutralizing background (decision 98,
`docs/specs/beam_self_field.md`; used by `bench/emittance/`, Fig. 4 of the 2020 paper). The open boundaries come when a comparison needs them, e.g. Mur boundaries,
a Yee CPML and a window for the LWFA-with-Yee benchmark (Yee in a wake: deferred, bench/emittance/README.md).

## Deck interface
```
numerics {
  solver = yee        # rip_cc (default) | yee
  cfl    = 0.98       # yee only: c dt / (1/hx^2 + 1/hy^2 + 1/hz^2)^{-1/2}
}
```

## Tests
- Fast (tests/test_yee.cpp):
  - Y1, exact numerical dispersion. A standing wave with B = 0 evolves as E^n = cos(nθ) E^0,
    cos θ = 1 − Δt²K²/2, along x (E_y) and oblique (E_z). Error 9e-15 after 400 steps.
  - Y2, random vacuum fields, 2000 steps: no growth (the peak of W over the second 1000 steps is 0.96
    of the first), div B unchanged to 2e-13.
  - Y3, plasma with both shapes, neutral and drifting across the periodic boundary, and a
    non-neutral slab: Gauss law 1e-15 to 7e-15, div B < 1e-15.
  - Y4, Langmuir frequency: sin(ωΔt/2) = ω_pΔt/2 to 0.25 %.
- MPI (tests/mpi.sh): the streaming plasma with Yee, 1 vs 8 ranks.
- Benchmark (bench/nci/): numerical Cherenkov instability of the streaming plasma, RIP-CC against
  Yee (results in bench/nci/README.md).

## References
- K. S. Yee, IEEE Trans. Antennas Propag. 14, 302 (1966).
- B. B. Godfrey, J. Comput. Phys. 15, 504 (1974): numerical Cherenkov instability.
- A. Pukhov, J. Comput. Phys. 418, 109622 (2020), Sec. 7.
