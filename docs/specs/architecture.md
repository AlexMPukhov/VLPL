# Architecture: geometry, solver interface, species registry, tiles, step pipeline

Status: accepted (PR #1, 2026-10-26); D-1..D-6 = decisions 10-15 in `docs/decisions.md`.
Implemented: D-2 (staggering descriptor, `src/stagger.hpp`), D-3 (attribute registry, `src/species.hpp`),
operator pipeline (`src/pipeline.hpp`),
D-1 for Cart3D (`src/geometry.hpp`), D-5 tile sort (`src/tiles.*`; per-tile deposition open).

This spec fixes the five interfaces that the roadmap (`docs/ROADMAP.md`, section 2) calls expensive to
change later. Physics modules after Phase 0 plug into them without touching the core. The refactor
that introduces them must not change results (section 7). Proposed new conventions are marked
**D-n**; they are decisions 10-15 in `docs/decisions.md`.

## 1. Present state (what the interfaces must absorb)

- `Domain` + `Field` (`src/grid.hpp`): one real 3D array per component with `ng` guard cells; `FV`
  is the kernel view. The staggering is written down only as a comment and as hard-coded offsets in
  `fields_at` / `esirkepov` / `rho_impl` (`src/particles.cpp`).
- `Fields` (`src/fields.hpp`) is RIP-CC itself: storage, solver step and diagnostics in one class;
  `inflow` is the only hook.
- `Species` (`src/particles.hpp`): seven fixed `double` arrays (x, y, z, ux, uy, uz, w); migration
  copies exactly these seven.
- `Simulation::step` (`src/sim.cpp`): push+deposit per species, `add_guards`, field step, migrate.
  Push and deposit are one fused kernel (Esirkepov needs x^n and x^{n+1}). The Esirkepov kernel is
  valid for a move of at most one cell per axis; longer moves are split into straight segments
  (decision 28).

## 2. Geometry

The geometry is a compile-time parameter `G` of everything that touches the grid: field storage,
halo exchange, solvers, gather, deposition, field diagnostics, field I/O. Particles, species
storage, pushers, physics operators, particle I/O and the RNG are geometry-agnostic.

| | `Cart3D` | `RZModes` (Phase 2) |
|---|---|---|
| grid axes | (x, y, z): index (i, j, k) | (z, r): index (i, j); modes m = 0 .. M-1 |
| transport axis (c dt = h) | x | z |
| field component | one real array | m = 0 real, m >= 1 complex: 2M - 1 real arrays (layout [m][k][j][i] with k = re/im) |
| components | Ex Ey Ez Bx By Bz | Ez Er Ephi Bz Br Bphi |
| decomposition | x, y, z | z (r later, if needed) |
| cell volume | hx hy hz | 2 pi r_j hz hr (axis cell: pi (hr/2)^2 hz) |
| boundaries | periodic / open per axis | z periodic / open; r: axis (parity per m) and outer |

Field reconstruction in RZ: F(r, phi) = F_0(r) + 2 Re sum_{m>=1} F_m(r) e^{i m phi}, the
convention of GEM-PIC-RZ (`src/fields.hpp` there), so its mode machinery and diagnostics can be
reused.

**D-1. Particles are Cartesian in both geometries; particle axis 0 is the transport axis.** In
`Cart3D` the particle coordinates are (x, y, z); in `RZModes` they are (z, x, y): axis 0 = z, the
transverse pair (x, y) Cartesian, r and phi only computed inside gather/deposit. — Pushers, QED,
spin and collisions never see the geometry; the RZ singularity stays in two kernels. The deck and
I/O use the physical names of each geometry.

Interface (one struct per geometry, all members static or POD so kernels can capture them):
```cpp
struct Cart3D {
    static constexpr int grid_dims = 3, field_arrays = 1;      // real arrays per component
    using View = FV;                                           // FVm (mode view) for RZModes
    HD static void to_grid(const double r[3], double g[3]);    // particle -> grid coordinates (RZ: z, r, phi)
    template <class St, int O> HD static void gather(const FieldViews<View>&, const double r[3], ...,
                                                     double E[3], double B[3]);   // Cartesian E, B
    template <class St, int O> HD static void deposit(const CurrentViews<View>&, const double r0[3],
                                                      const double r1[3], double qw, ...);
};
```
`Domain` keeps its present role but takes `G::grid_dims`; `FieldSet<G>` owns the component arrays
(by enum, not by member name), so I/O and halo exchange can loop over components.

RZ charge conservation (open, Phase 2 spec `docs/specs/rz_modes.md`): the deposition must satisfy
the discrete continuity equation of the RZ solver per mode (target: the standard RIP-CC Gauss law per
mode to round-off, decision 17; it holds for rip_cc and for the box-closure irip). Candidates: an Esirkepov-type split in (z, r) with the phi current from the
azimuthal displacement (Davidson et al. 2015), or direct deposition plus a per-mode correction. The
interface above admits both; nothing else depends on the choice.

## 3. Field solver interface and staggering descriptor

**D-2. The staggering is a compile-time descriptor that the solver declares; gather and deposition
are generated from it, nowhere else are offsets written.**
```cpp
struct Loc { double o[3]; double t; };        // offset in cells per grid axis, time offset in steps
struct RipCC {                                 // Cart3D
    static constexpr Loc E[3] = {{{0, 0, 0}, 0}, {{0, .5, 0}, 0}, {{0, 0, .5}, 0}};
    static constexpr Loc B[3] = {{{0, .5, .5}, 0}, {{0, 0, .5}, 0}, {{0, .5, 0}, 0}};
    static constexpr Loc rho  = {{.5, 0, 0}, 0};
    // J_a = rho + e_a / 2 (mod 1), at t = 1/2 -- derived, not declared (rule below)
    static constexpr bool colocated_pairs = true;   // (Ey,Bz), (Ez,By) gathered with identical weights
};
struct Yee { E_a at edges, B_a at faces, rho at nodes, B at t = 1/2 ... };
```
Rules that the generated kernels rely on and a fast test checks for every registered solver:
- J_a is located at rho + e_a/2 along axis a (mod 1 cell): then the Esirkepov deposition on the
  charge grid is exactly the continuity equation of the solver (decision 3 in `docs/decisions.md`
  is the RIP-CC instance of this rule).
- The gather delivers E^n and B^n at x^n. A component declared at t = 1/2 (Yee B) is gathered as
  the mean of the stored levels n - 1/2 and n + 1/2; the solver keeps the old level.
- `colocated_pairs` makes the gather reuse the (Ey, Bz) and (Ez, By) weights (decision 4).

Runtime interface (one virtual call per step; no virtual calls in kernels):
```cpp
template <class G> class FieldSolver {
public:
    virtual const char* name() const = 0;
    virtual int guard_cells(int order) const = 0;
    virtual double stability(const Config&) const = 0;          // < 1 required; checked by Config::check
    virtual double dt(const Domain&) const = 0;                  // rip_cc: h_transport; yee: Courant
    virtual void step(FieldSet<G>&, CurrentSet<G>&, Comm&, const BoundaryHooks&) = 0;
    virtual void gauss_residual(const FieldSet<G>&, const Field& rho, Field& out) const = 0;
    virtual ~FieldSolver() = default;
};
```
Kernels that depend on the staggering (gather, deposit, rho, Gauss residual) are templates on the
descriptor and the shape order, dispatched once per step by a switch over the solvers compiled in
(`rip_cc`, `yee`; `irip` shares the RIP-CC descriptor). `BoundaryHooks` generalises today's
`inflow` (laser through the inbound transport vector, later PML and the moving window).

## 4. Species and the attribute registry

**D-3. Structure of arrays with named attributes registered at setup.** The core attributes
x0 x1 x2 u0 u1 u2 w have the fixed slots 0..6 (kernels index them by constant); every further
attribute is requested by the operator that needs it:
```cpp
int slot = species.require("chi", Attr::F64, /*init*/ 0.0);      // idempotent, before the first step
```
| attribute | type | requested by |
|---|---|---|
| id | I64 | any stochastic operator, tracking, ionization (parent id) |
| Z (charge state) | I32 | ion kind; q = Z e per particle in deposit and push |
| s0 s1 s2 (spin) | F64 | spin precession / spin-resolved QED |
| tau_em, tau_pair (optical depth) | F64 | QED emission, Breit-Wheeler |
| flags | I32 | tracking, diagnostics selections |

Types: F64, F32, I64, I32. Migration, sorting, checkpoint, particle I/O and `append` loop over the
registry, so a new attribute needs no change there. Kernels get a POD `SpeciesView` (pointer per
slot, at most 32 slots) and the operator captures the raw pointers of its slots.

Kinds (per species, deck `kind`): `particle` (fixed q, m), `ion` (q from Z, `Z_max`, element for the
ionization tables), `photon` (q = 0, m = 0, u = p / (m_e c), v = u / |u|: ballistic move, no
gather or deposit unless an operator asks; optional polarisation attributes). The kind selects the
push/move kernel variant at setup, not per particle.

**D-4. Random numbers are counter based**: Philox or splitmix64 of (seed, step, particle id, stream
of the operator). — The same physics on CPU, GPU and any rank count, as decision 8 does for the
initial momenta; it is why stochastic operators require `id`.

## 5. Tiles

**D-5. Each species stays one SoA, kept sorted by tile.** A tile is a block of cells (default
8 x 8 x 8 in Cart3D, 32 x 8 in (z, r); deck `numerics.tile`). The species holds
`tile_start[n_tiles + 1]`: the particles of tile t are [tile_start[t], tile_start[t+1]).
- Sorting: counting sort on the device (count per tile with atomics, exclusive scan, scatter of
  the permutation; then each registered attribute is gathered through the permutation with one
  scratch array). Every `numerics.sort_every` steps (default 1); particles are allowed to stray
  into the neighbouring tile between sorts (|dx| < 1 cell per step), so the per-tile deposit box
  is the tile plus guard of `sort_every + order` cells.
- Deposition (GPU): one thread block per tile accumulates into shared memory, then adds the tile
  box to the global array with atomics (no global atomics per particle). CPU: one thread per tile
  with a private box, merged in colour order (2^3 colours) without atomics.
- Per-cell order inside a tile (cell sort, `cell_start`) is a second, optional level for the
  binary collisions of Phase 3; it uses the same counting sort.
- Migration: the leaving particles are those outside the owned box; the sort moves them to the
  end, so the existing host path (`migrate`) takes a contiguous tail instead of a list.

## 6. Step pipeline of operators

One step n -> n+1, with the time levels at entry of each stage (positions at n, momenta at
n - 1/2, fields E^n, B^n):

| stage | operators | acts on | levels after |
|---|---|---|---|
| 1 inject | moving window, plasma injection, beam injection | species, fields | |
| 2 ionize | field ionization (ADK/PPT) | ion Z, new electrons at x^n | |
| 3 push | gather + Lorentz (+ radiation reaction in the same kernel) | u | u^{n+1/2} |
| 4 qed | photon emission, Breit-Wheeler, spin precession | u, new photons / pairs at x^n | |
| 5 move + deposit | x^n -> x^{n+1}, Esirkepov j^{n+1/2} | x, J | x^{n+1} |
| 6 fields | solver step, boundary hooks | E, B | E^{n+1}, B^{n+1} |
| 7 collisions | binary collisions, collisional ionization (implemented in stage 8 after the migration, decision 94) | u (per cell) | |
| 8 migrate | MPI particle exchange, tile sort | species | |
| 9 boundaries | absorbing / reflecting particle boundaries | species | |
| 10 diagnostics | output, probes, checkpoints | | |

```cpp
template <class G> struct Operator {
    virtual const char* name() const = 0;
    virtual Stage stage() const = 0;
    virtual void setup(Sim<G>&) {}                 // read the deck block, require attributes
    virtual void apply(Sim<G>&) = 0;               // may only launch kernels; no this-capture inside
    virtual ~Operator() = default;
};
```
Each physics module is one operator with its own deck block; a factory maps block names to
operators. Within a stage the operators run in deck order.

**D-6. Push and move are separate kernels only when stage 2 or 4 has an operator; otherwise the
fused push+deposit kernel of today runs.** — The move kernel reads x^n and u^{n+1/2} and needs no
stored old position; the split costs one extra pass over x and u and only in runs that need it.

Charge conservation of particle creation: a particle created at x^n together with a parent charge
change leaves rho^n unchanged (ionization: ion q -> q + 1, electron -1 at the same point; pairs:
+1 and -1), and both then move with the stage-5 kernel, so the Gauss law stays at round-off without
a correction. The energy-conserving ionization current (Phase 3) is an extra J term in stage 5.

## 7. Refactor plan (one pull request per interface, Phase 0)

1. Staggering descriptor; gather, Esirkepov and rho generated from it (the Gauss residual is the
   solver's own discrete divergence, `FieldSolver::gauss_residual`). Bit-identical. Done.
2. Attribute registry in `Species`; migration over the registry. Bit-identical. Done: storage and
   migration moved to `src/species.*`; migration exchanges packed rows of all attributes (slot order,
   bytes); `require()` belongs to setup, since it may reallocate the slot table.
3. Operator pipeline in `Simulation::step`; laser inflow as a boundary hook. Bit-identical. Done:
   `Pipeline` / `Operator` in `src/pipeline.*`; core operators push_deposit [push], rip_cc [fields],
   migrate [migrate] in `sim.cpp`; `BoundaryHooks` (fields.hpp) receive t_n explicitly; the laser is
   its own module (`src/laser.*`). Output diagnostics stay in the main loop for now.
4. Geometry parameter, `Cart3D` only (`RZModes` follows in Phase 2). Bit-identical. Done: the
   particle-grid kernels (field views, gather, Esirkepov, charge deposit) are `Cart3D` members in
   `src/geometry.hpp`, shapes in `src/shape.hpp`; push and charge kernels are templates on `G`.
   Templating the field storage (`FieldSet<G>`), halo exchange and `Simulation` on `G` waits for
   `RZModes`: with a single geometry that abstraction would have nothing to be tested against.
5. Tiles and sorting. Changes the summation order of the deposition: equal to round-off only;
   `numerics.sort_every = 0` switches it off and is bit-identical. Done: `src/tiles.*`, operator
   tile_sort [migrate] after the migration and once after loading; the sort is stable with the counting
   sort on the host for now (decision 16). Measured on one thread: example decks with sort_every = 0
   bit-identical; with sort_every = 1 energies within 5e-13, particle numbers and Gauss drift
   unchanged; cost +2 % (laser deck) and +9 % (streaming plasma) while the deposition does not yet
   use the tiles. Open: per-tile deposition (shared memory on the GPU, private boxes on the CPU) and a
   device sort, to be tested on a real GPU.

"Bit-identical" is checked on one OpenMP thread (`OMP_NUM_THREADS=1`), where the atomic order is
fixed: the 18 fast tests, `tests/mpi.sh`, and the `diag.txt` of `examples/` decks compared with the
regression harness (`tests/regression/`, Phase 0 item).

## Deck interface
```
grid      { geometry = cart3d   # cart3d | rz;  modes = 2 (rz only) }
numerics  { solver = rip_cc     # rip_cc | yee | irip
            tile = 8 8 8        # cells per tile;  sort_every = 1  (0: no sorting) }
species e { kind = particle     # particle | ion | photon }
<module>  { ... }               # one block per operator (ionization, collisions, qed, ...)
```

## Tests
- Fast: staggering rules (J_a = rho + e_a/2 for every solver) and the discrete continuity equation
  (rho^{n+1} - rho^n)/dt + div J = 0 for random single-particle moves of every shape order, for
  every descriptor, to 1e-14 relative (independent of any solver step).
- Fast: registry: an extra F64 and I64 attribute survive append, sort and migration unchanged
  (multiset of (id, value) pairs equal); `tests/mpi.sh` with 1 vs 8 ranks keeps all ids.
- Fast: tile sort: every particle lies in its tile range, the permutation preserves the multiset;
  the plasma-conservation tests pass with sorting on (Gauss law to round-off).
- Fast: counter-based RNG: a test operator drawing per particle gives the same values on 1 and 8
  ranks and before/after sorting.
- All present tests unchanged after each refactor step (table in section 7).

## References
- T. Zh. Esirkepov, Comput. Phys. Commun. 135, 144 (2001) - charge-conserving deposition.
- A. Pukhov, J. Comput. Phys. 418, 109622 (2020) - X-dispersionless solver (RIP).
- A. F. Lifschitz et al., J. Comput. Phys. 228, 1803 (2009); R. Lehe et al., Comput. Phys.
  Commun. 203, 66 (2016) - azimuthal-mode decomposition.
- A. Davidson et al., J. Comput. Phys. 281, 1063 (2015) - charge-conserving deposition in r-z modes.
- H. Vincenti et al., Comput. Phys. Commun. 210, 145 (2017) - tiled deposition.
- J. K. Salmon et al., SC'11 (2011) - counter-based random numbers (Philox).
