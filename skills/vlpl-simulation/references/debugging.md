# VLPL: when a run stops or looks wrong

Errors are printed as `error (rank N): <message>` on stderr and the run exits with code 2; a deck error names the
file and line or the key. Warnings start with `warning:` (stderr): some come with `--check`, others when the run
starts or during it. Fix the deck or the run; do not modify the source. If the code itself is at fault (a crash
without a message, a wrong result of a shipped example), report it to the user and as an issue.

## Errors of the deck

| message (beginning) | meaning | what to do |
|---|---|---|
| `expected 'key = value'`, `expected 'block {'` | syntax: one `key = value` per line, no semicolons | split the line |
| `block '...' given twice` | two blocks with the same name | name them: `species a {`, `species b {` |
| unknown key / unit messages with the line number | the key does not exist in this block or geometry (RZ uses z, r names), a unit is missing or of the wrong kind | look the block up in `deck_reference.md` |
| `override '...' must read block.key=value` | command-line override | quote named blocks: `"species electrons.ppc_x=2"` |
| `unstable: dx^2 (1/dy^2 + 1/dz^2) + dx^2 wp^2/4 = ...` | explicit Cartesian solver beyond its limit | `numerics.solver = irip`, or coarser dx / finer is not possible: see `physics_rules.md` section 2 |
| `unstable: dz^2 (lambda_max + wp^2) / 4 = ...` | explicit RZ solver beyond its limit (dz too large for dr and the modes) | `numerics.solver = irip`; or fewer `grid.modes`, larger `grid.dr`, `numerics.rz_axis_cut` |
| `unstable at the peak density: omega_p dt = ...`, `unstable: omega_p dt >= 2` | the plasma frequency is not resolved in time | smaller dx (dt = dx / c); lower density; `numerics.mover = implicit`; or a `medium` block for an optically thick target |
| `the laser enters through an open x boundary` | laser without `grid.boundary_x = open` (RZ `grid.boundary_z`) | set it; with `grid.boundary_x = absorbing` use `laser.tfsf_x` |
| `window.start must be >= 2 laser.delay` | the window would move before the pulse is in | raise `window.start` (and lengthen the box so that the pulse is still inside) |
| `numerics.solver = irip: ranks along x only` | MPI decomposition across y or z with the implicit solver | `grid.mpi_y = 1`, `grid.mpi_z = 1` (or leave all three automatic) |
| `metal ...: numerics.solver = rip_cc only`, `rings ...: rip_cc only`, `dielectric: numerics.solver = rip_cc only` | embedded bodies need the explicit solver | `numerics.solver = rip_cc` with cells that satisfy its condition; for optical media with the implicit solver use `medium` |
| `output.openpmd_every ... need a build with HDF5`, `the checkpoint block needs a build with HDF5` | the executable was built without HDF5 | rebuild with HDF5 (skill vlpl-setup) or remove the key |
| `geometry = rz: no slices; field output: output.openpmd_every` | `output.slice_every` in RZ | use `output.openpmd_every`, `output.axis_every`, `output.plane_z` |
| `numerics.shape = 3: not with a moving window yet` | cubic shape with a window | `numerics.shape = 2` |
| `numerics.shape must be 1 or 2 in geometry = rz` | | `numerics.shape = 2` |
| `--restart latest: no checkpoint in ...` | no checkpoint was written | `--restart auto`, or check `output.dir` and `checkpoint.every` |
| boost: fewer than one lattice plane of a beam | a beam against the boost is gamma^2 (1 + beta)^2 shorter in boosted cells | raise `species.ppc_x` (RZ `species.ppc_z`) to the value the message names |
| boost: the injected plasma is not neutral | an ion species without an electron species of the same `ppc`, bounds and matching density | pair them |
| boost: `mobile = false`, media, side injection, TFSF, implicit mover | not available in a boosted frame | lab frame, or give the ions their mass |

## Warnings

| warning (beginning) | what to do |
|---|---|
| `omega_p dt = ... the plasma oscillation runs ... % fast` | accept (it states the error) or refine dx |
| `the laser head reaches the right edge before the window moves` | `window.start` must be below box length / c (and >= 2 x `laser.delay`): start the window earlier or lengthen the box; do not run with this warning |
| `plasma in the last 3 cells and window.start > 0` | begin the plasma ahead of the box (`species.xmin` beyond the right edge), or `window.start = 0` |
| `the envelope at the edge of its grid is ... of the peak` | the transverse box cuts the laser: widen it |
| `density_function reaches ... > 1` | the stability check used `species.density`: make `density` the peak and the function <= 1 |
| `ppc_theta = ... samples ... azimuthal components: noise heating` | `species.ppc_theta` >= 2 (2 `grid.modes` - 1) |
| `k_p dr = ... at the axis: a dense plasma next to the axis` | finer `grid.dr`, or `numerics.rz_gather = matched` |
| `a plasma streams through the grid ... with dr < ...` | add the current filter: `numerics.filter_passes = 1` |
| `a slab ramp with numerics.shape = 1` | `numerics.shape = 2` or 3 |
| `boost: ... the plasma streaming through the grid is unstable` | `numerics.shape = 2`, `numerics.filter_passes = 1` (2 for long residence), binomial filter |
| `boost: beam ...: the loaded charge differs from the deck's by ... %` | raise `species.ppc_x` until >= 20 lattice planes |
| `boost: the laser beam at its lab entrance events is ... wide` | the box cuts the converging beam: `laser.injection = antenna` (mind d_1%), or a wider box |
| particles `enter the window through its front edge` / in the absorbing layer of a dielectric run | keep beams out of the last cells of the window |

## Symptoms

| symptom | likely cause | fix |
|---|---|---|
| NaN in `diag.txt`, energy explodes within a few steps | an unstable setup that passed the checks: `species.density_function` above 1, omega_p dt near 2, shape 1 on a steep ramp | make `density` the peak; refine dx; `numerics.shape = 2` |
| W_field grows slowly in a quiet plasma, density ripples at the grid scale | numerical heating: too few particles, shape 1, RZ with few `ppc_theta` | `numerics.shape = 2`, more ppc, `numerics.filter_passes = 1` |
| a streaming or boosted plasma develops stripes, field noise x100 in 10 - 100 / omega_p | numerical Cherenkov instability of the drifting plasma | binomial filter with 1 - 2 passes, shape 2; never `numerics.filter = projection` for drifting plasma |
| max\|dGauss\| not at round-off | hot plasma touching an open edge before the window moves; plasma inside an absorber or a metal surface cell; an incomplete restart | start the plasma inside / ahead of the box; keep particles out of the layers (`grid.absorber_particles`); report it if none applies |
| laser weaker than a0 at focus | focus outside the validity of the paraxial entrance (very tight focus), the box cuts the beam, 2D geometry (amplitude scales as (w0 / w)^(1/2)), in a boosted frame: inside the antenna near field d_1% | check the summary warnings; wider box; finer dx in the boosted frame |
| light comes back from the side | periodic transverse box too narrow, or the RZ default wall | `grid.boundary_y = pml`, `grid.boundary_z = pml`; RZ `grid.boundary_r = absorbing`; widen the box |
| fringes of fixed radial wavenumber near the axis (stretched RZ grid) | the coarse outer cells reflect: k_perp dr > 1 | keep the grid uniform where light propagates (`grid.r_uniform`), stretch more slowly |
| noisy density or field on the RZ axis | few particles next to the axis | `numerics.shape = 2`, `species.ppc_r` >= 2, `numerics.filter_passes` 1 - 2 |
| wake too strong in a boosted run | lab cells too coarse (lambda0 / 16) | lambda0 / 32 |
| trapped charge differs between resolutions | injection is sensitive to noise and resolution | converge it (cells halved, particles doubled) before quoting a charge |
| run much slower than estimated | plasma filled the window after the smoke run; implicit solver with ny, nz not powers of two; openPMD output every few steps | estimate after filling; powers of two; fewer dumps |
| MPI run stops at the start with the implicit solver | ranks across y or z | `grid.mpi_y=1 grid.mpi_z=1` |
| results differ between two runs at the 1e-12 level | thread or rank order of the sums | expected (round-off); larger differences: report |
