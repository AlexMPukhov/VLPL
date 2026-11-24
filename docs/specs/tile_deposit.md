# Per-tile current deposition

Status: CPU colouring implemented (`deposit = tiled`, CPU default); GPU box version implemented
(`deposit = box`), verified on the CPU through the fallback of `par_blocks`, awaiting real-GPU tests

## Physics and equations
Unchanged: the Esirkepov current j^{n+1/2} of the move x^n -> x^{n+1} on the charge grid (decision 3,
`docs/specs/architecture.md` section 5). Only the order of the additions changes.

## Discretisation
The species is sorted by tile (`src/tiles.*`, tile_start ranges) after every migration; with
`numerics.sort_every = auto` (default, decision 20) only when the tiled or the box deposit is chosen.

Reach of one particle. A particle in cell c at t_n moves less than one cell per step: c dt = h_x, and
the explicit RIP-CC stability needs h_y, h_z > h_x. With the RipCC descriptor its Esirkepov stencil
then writes only the nodes [c - 2, c + 3] along every axis, for shape 1 and shape 2 (including the rho
offset 1/2 along x and the current shift of J_x). A tile with cells [a, b) therefore writes
[a - 2, b + 2]. The implicit `irip` (decision 17) allows h_perp < c dt: a particle can then cross more
than one cell per step, and `tiled_deposit_safe` (it checks dt <= h along every axis) selects the
atomic path. A halo of ceil(dt/h) cells would extend the colouring to irip later.

Colouring. Tiles get the colour (tx mod 2) + 2 (ty mod 2) + 4 (tz mod 2). Two tiles of the same colour
along an axis are separated by one full tile of width T, so their write regions are disjoint for
T >= 6 (T >= 5 would do). Writes beyond the owned box go to the guard layers, which are separate
memory and are added into the neighbours afterwards (`add_guards`), so the periodic wrap and MPI
neighbours cannot race either.

Algorithm per species and step (CPU, OpenMP):
1. For each colour: one thread per tile of that colour; it pushes its particles in their stored
   order and deposits with plain additions (no atomics). A particle whose cell at t_n is not in its
   tile (a stray: moved since the last sort, `sort_every > 1`) is skipped and flagged.
2. One pass over the flagged particles with the atomic deposit (as before).

Result: with no strays (sort every step) the current does not depend on the number of threads, so a
CPU run is bit-reproducible for any thread count.

Selection: `numerics.deposit = auto` (default) uses `tiled` in the CPU build when T >= 6 along every
axis with three or more tiles and every non-empty colour has at least as many tiles as there are
OpenMP threads (otherwise threads wait: 32 x 16 x 16 cells, 2 tiles per colour, 20 threads was 4x
slower than global, third GPU check); else `global` (the atomic deposit). The GPU build keeps `global`
until the box version below is timed on a real GPU: one thread per tile would serialise the GPU.

GPU version, `deposit = box` (decision 19). `par_blocks(nb, ns, body, flush)` in `portable.hpp`: one
CUDA thread block per tile with a zeroed scratch array of ns doubles in shared memory (opt-in above
48 KB), body for the threads of the block, a barrier, flush. CPU fallback: one OpenMP iteration per
tile with a thread-local array, so the same code is tested in every CPU build.
- Box: (T + 5) cells per axis and 3 components, origin a - 2: it covers the reach [a - 2, b + 2] of
  the tile [a, b) (measured: shape 2 reaches c - 2 and c + 3 along y and z, so the bound is tight).
  T = 8: 13^3 x 3 x 8 B = 52.7 KB (opt-in shared memory), T = 6: 32 KB.
- body: the threads take the particles of the tile in turn, gather from the global fields and
  deposit into the box with shared-memory atomics; the Esirkepov indices are shifted by the integer
  box origin (exact, the same weights as the global deposit). Strays deposit globally with atomics.
- Strays are listed (int atomic counter) and deposited after the box kernel by a separate atomic
  pass, as in the tiled path, so the box kernel holds one copy of the deposit.
- Occupancy (decision 27): `par_blocks<MinB>` sets `__launch_bounds__(256, MinB)`. The shape-1 box
  kernel takes 240 registers uncapped (ptxas, sm_120: one block per SM); `RIP_BOX_MINB1` = 2
  (default) caps it at 128 without spills, 3 at 80 with 56 bytes of spills. Shape 2 takes 254
  either way (as the global push); `RIP_BOX_MINB2` = 2 would spill 432 bytes, so the default is 1.
- Fixed point (decision 26). The box holds int64: FP64 (and int64) atomics in shared memory are CAS
  loops on sm_90 and sm_120, the 32-bit integer add is native. `block_add` adds an int64 as two
  32-bit atomics, the low word first; its carry comes from the returned old value, so the result is
  the exact 64-bit sum in any order. Scale: a particle adds at most |q w| |v_a| / dV <= |q w| / dV to
  a box cell (the cumulative Esirkepov sums are bounded by the move in cells), so the sums of a tile
  of N particles stay below B = N max|q w| / dV. With B < 2^e the scale is 2^(62 - e): sums
  < 2^62 + N / 2 < 2^63, conversion back exact (power of two), each addend rounded to within B 2^-62. The
  error is absolute, round-off of the heaviest content of the tile: a cell holding 1e-9 of it keeps
  ~1e-6 relative accuracy (test), a weight contrast of 1e3 inside a tile ~1e-12. The box is the same
  in every thread order (integer sums are associative).
- flush: one global atomic add per non-zero box cell (cells beyond a partial last tile are never
  written and are skipped). Global atomics per step: 3 (T + 5)^3 per tile instead of ~144 per particle
  (shape 2), i.e. about 50 x fewer for 8 particles per cell.
Requires sort_every >= 1 and dt <= h along every axis (else a particle can leave the box). Longer
moves (implicit solvers, c dt > h) take the global deposit, which splits them into segments of less
than one cell (decision 28).

## Measured (2 cloud cores, `deposit = global` vs `tiled`, time stepping)
| deck | 1 thread | 2 threads |
|---|---|---|
| streaming_plasma 200 fs (64 x 32 x 32, 1 M particles) | 26.0 -> 19.5 s (-25 %) | 15.0 -> 10.9 s (-27 %) |
| laser_reflection, regression size (225 steps) | 115.8 -> 85.2 s (-26 %) | 72.0 -> 64.4 s (-11 %) |

## Deck interface
```
numerics { deposit = auto     # auto | global | tiled | box; tiled: sort_every >= 1, tiles >= 6 cells;
                             # box: sort_every >= 1, dt <= h }
```

## Tests
- Fast: tiled vs global after 20 steps of a hot drifting plasma, shapes 1 and 2: field energy and
  currents equal to round-off; Gauss law and div B at round-off with the tiled deposit.
- Fast: tiled deposit with 1 thread and with 3 threads gives bit-identical currents and fields.
- Fast: sort_every = 3 produces strays, which take the atomic path; the Gauss law stays at round-off.
- Fast: the colouring visits every tile once, and no two tiles of a colour are neighbours.
- Regression harness: all cases match the references with the tiled deposit (CPU default).
- Fast: reach of the Esirkepov stencil within [c - 2, c + 3], both ends reached (shape 2, y and z).
- Fast: box = global to round-off with the Gauss law at round-off, shapes 1 and 2, full and partial
  tiles, transversely hot plasma (full reach); strays with sort_every = 3. On a GPU the same tests run
  the shared-memory kernel.
- Fast: weights over 8 decades inside a tile (density e^(-6 y)): box = global to 3.5e-15 of max|J|,
  Gauss at round-off; the per-cell error is reported (9.5e-7 at 1e-9 of max|J|).
- Fast: `block_add` of 20000 random int64 (both signs, carrying low words) into 4 shared cells by
  all threads of one block gives the exact 64-bit sums (concurrently on a GPU).

## References
- H. Vincenti et al., Comput. Phys. Commun. 210, 145 (2017) - tiled deposition.
- T. Zh. Esirkepov, Comput. Phys. Commun. 135, 144 (2001).
