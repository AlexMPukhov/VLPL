# Density profiles (deck expressions)

Status: implemented

## Physics and equations
A species has the density n(x, y, z) = density * f(x, y, z) in n_c. The factor f is an expression of the
absolute coordinates; it is 1 if no function is given, and points with f <= 0 (or NaN) get no particle.
Examples: an up-ramp `x < 20 um ? (x - 13 um) / (7 um) : 1`, a channel `1 + 0.5 * (y^2 + z^2) / (10 um)^2`.

## Discretisation
- The particles keep the regular lattice of the uniform load: ppc points per cell at the midpoints
  of the sub-cells. Each has the weight density * f(r) * dV / ppc, so a profile changes the weights
  and not the positions. The regular lattice integrates a linear profile exactly (midpoint rule).
- f is evaluated on the host when a cell is loaded: at the start, and for each cell that enters the
  moving window. The window uses absolute x, so a ramp ahead of the box enters with it.
- Electrons and ions with the same function and density get the same weights at the same positions.
  The plasma is then exactly neutral, and the Gauss law of the entering plasma stays exact
  (moving window, decision 22).
- The stability check (Config::check) uses `density` alone. A warning is printed once if f exceeds 1.
- The expression parser is GEM-PIC-RZ's `src/expr.*`, taken over with VLPL's unit table
  (`unit_to_code` in `src/deck.cpp`: lengths, times, angles, densities in cm^-3 or m^-3 -> n_c).
  It provides operators, comparisons, `c ? a : b`, and the functions listed in `src/expr.hpp`.

## Deck interface
```
species electrons {
  density          = 0.01                                # n_c, reference density
  density_function = x < 20 um ? (x - 13 um) / (7 um) : 1   # factor of density, variables x, y, z
}
```
Quotes around the expression are optional. `#` starts a comment and cannot appear in it.

## Tests
- Fast (tests/test_profile.cpp):
  - P1: expression values (precedence, power, ternary, functions, units um, fs, cm^-3, %, pi) exact;
    five syntax errors are reported.
  - P2: a linear ramp is loaded exactly (sum of the weights = analytic integral, 1e-12), and a step
    keeps exactly the lattice points with f > 0.

## References
- GEM-PIC-RZ `src/expr.hpp` (same syntax).
