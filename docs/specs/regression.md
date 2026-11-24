# Regression harness

Status: implemented (Phase 0 item; `tests/regression/`)

## Purpose
Catch any change of results that the fast tests do not see: whole runs of small versions of the
example decks are compared column by column with stored reference diagnostics (`diag.txt`). A
refactor that must not change physics passes untouched; a change that is meant to change results
re-blesses the references in the same pull request, so the diff of `tests/regression/ref/` shows the
change for review.

## Cases (`tests/regression/cases.txt`, about 25 s on 2 CPU cores)
| name | deck | what it covers |
|---|---|---|
| stream | streaming_plasma, 32 x 16 x 16, 0.6 ps | relativistic e-p beam, numerical Cerenkov growth, shape 1, Higuera-Cary, periodic |
| stream_s2 | as stream, 0.3 ps, shape 2, Boris | shape 2, Boris pusher |
| reflect | laser_reflection, 192 x 192 x 4, 25 fs, slab kept clear of the edges | laser inflow hook, open x boundary, overdense tilted slab, migration, tile sort |

## Comparison (`tests/regression/compare.py`, Python 3 standard library only)
Per row (same steps, particle counts exactly equal). The CPU/GPU and thread-count differences
measured so far are <= 1e-11 (`docs/GPU_TESTING.md`); the tolerances leave a factor 100 above that.
- W_field, W_kin, W_total: |new - ref| <= 1e-9 |ref| + 1e-12 max|ref column|.
- max|rho|: |new - ref| <= 1e-9 |ref| + 1e-9 max|ref column|: in a neutral plasma it starts as pure
  round-off (~1e-17) and differs between two runs of the same binary (order of the atomics).
- max|dGauss|, max|divB|: new <= max(1e-12, 10 ref) (conservation at round-off).
- t: relative 1e-12.
A self-test perturbs one reference value by 1e-6 and requires the comparison to fail.

## Use
```
make regression                          # all cases with build/vlpl
sh tests/regression/run.sh EXE           # any binary, e.g. build-gpu/vlpl on a GPU machine
sh tests/regression/run.sh EXE --bless   # rewrite the references (commit them with the change)
```
Runs with `OMP_NUM_THREADS` as set (the tolerances cover the thread-count differences).

## References
- `docs/GPU_TESTING.md` (CPU vs GPU differences of the diagnostics).
