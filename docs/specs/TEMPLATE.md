# <Feature name>

Status: draft | implemented (PR #) | verified (benchmark)

## Physics and equations
What is modelled, in code units; key formulas with references.

## Discretisation
Where the quantities live on the grid / in the particle data, time levels, the algorithm per step,
how it interacts with charge conservation and the GPU kernels.

## Deck interface
```
block { key = value   # unit, default, meaning }
```

## Tests
- Fast (tests/tests.cpp): exact result or conservation law, tolerance.
- Benchmark (bench/): published result or other code, stored curve, tolerance.

## References
