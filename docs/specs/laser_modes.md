# Laser modes: focus position, Laguerre-Gauss, several lasers

Status: implemented (`src/laser.*`, blocks `laser` and `laser <name>`, decision 32). Laser from file
(LASY) follows with the openPMD output (needs HDF5).

## Physics and equations
The laser enters through the open left x boundary (decision 5). At the entrance plane x = 0, for a pulse
focused at x = x_f, the field is the paraxial Gaussian beam at the distance X = -x_f from its focus,
separable in y and z (waists W = w0 and w0z; w0z = 0: uniform in z), in code units (k = ω = 1):

    z_R = W²/2,   w = W sqrt(1 + X²/z_R²),   1/R = X / (X² + z_R²),   ψ = atan(X / z_R)
    E_pol = a0 Π_a sqrt(W/w) exp(-r_a²/w²) g(t') sin(t' + ψ/2 per axis + phase)
    t' = t - t_c - Σ_a r_a² / (2R),   g = exp(-2 ln2 t'²/τ²)

- a0 is the amplitude at the focus in vacuum.
- t_c is the time at which the pulse centre crosses x = 0.
- The curvature delays the envelope as well as the phase.

**Laguerre-Gauss** (round beam, w0z = w0): the factor (√u)^|l| L_p^|l|(u), with u = 2r²/w², the
phase l θ (θ = atan2(z, y)), and the Gouy phase (2p + |l| + 1) ψ. For p = l = 0 it reduces to the
Gaussian.

**Several lasers**: blocks `laser` and `laser <name>` add their fields at the entrance plane, each with
its own polarisation (y or z) and carrier phase. Two crossed lasers with a phase difference of π/2 give
circular polarisation.

## Discretisation
- Unchanged injection: the guard layer i = -1 at t_n holds the inbound transport vector T(t_n + dt).
  It is zeroed once per step, then every laser adds its field (Ey, Bz = E for y; Ez, -By = E for z).
- With x_f = 0, p = l = 0 and phase = 0, the original expression is used, bit-identical. For one such
  laser, 0 + a = a.

## Deck interface
```
laser {                      # and any number of `laser <name>` blocks, summed
  a0 = 2   duration = 30 fs   waist = 10 um   waist_z = 10 um   polarization = y
  focus  = 200 um            # focus position along x (default 0, the entrance plane)
  mode_p = 0   mode_l = 1    # Laguerre-Gauss (round beam: waist_z = waist)
  phase  = 90 deg            # carrier phase (default 0)
}
```

## Tests
- Fast (tests/test_laser.cpp):
  - L1: a beam focused in y only, W = 6 (≈ λ), focus at x_f = z_R. The width from the fluence,
    2 sqrt(⟨y²⟩), is 8.440 at the entrance (paraxial 8.485) and 6.026 at the focus (6.000). The
    fluence gain is 1.394 (w(0)/w(x_f) = 1.414).
  - L2: Laguerre-Gauss (0,0), (1,1), (2,-1) away from the focus against the complex q-parameter
    form with the explicit polynomials: difference 1.3e-15.
  - L3: two crossed lasers with phase π/2: Ey² + Ez² in a cell has no 2ω ripple. The change is
    2.9e-3, the envelope's; linear polarisation would swing by 100 %.
- Regression: all decks with a laser are bit-identical.

## References
- A. E. Siegman, Lasers (1986), ch. 16-17: Gaussian and Laguerre-Gauss beams.
