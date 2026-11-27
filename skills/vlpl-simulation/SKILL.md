---
name: vlpl-simulation
description: Choose the solver variant and write, check, run and debug input decks for the particle-in-cell plasma code VLPL (full electromagnetic PIC with the dispersion-free RIP solver - VLPL-RIP and VLPL-iRIP in 3D / 2D Cartesian geometry, VLPL-RZ and VLPL-iRZ in quasi-cylindrical geometry, explicit or implicit). Use for laser wakefield acceleration (lab or Lorentz-boosted frame), laser-solid interaction, plasma mirrors and harmonics, field and impact ionization, binary collisions, strong-field QED with spin and pair production, particle beams with self-fields, metal and dielectric structures and wakefields, restarts, MPI and GPU runs, and for choosing the resolution, reading the output (diag.txt, slices, openPMD, beam and lab-frame diagnostics) or finding out why a run stops or gives wrong or noisy results. If the vlpl executable is not built yet, use the skill vlpl-setup first.
license: BSD-3-Clause
compatibility: Needs the vlpl executable (skill vlpl-setup). The helper scripts need python3 (standard library only; make_bunch.py needs numpy and h5py). openPMD output, checkpoints and file input need a build with HDF5.
metadata:
  repository: https://github.com/AlexMPukhov/VLPL
---

# VLPL: solver choice and input decks

VLPL reads one ASCII deck: blocks `name { key = value ... }`, lengths, times and angles with a unit, densities in
units of the critical density n_c. The code checks the deck strictly (unknown keys, missing units, units of the
wrong kind and unstable grids are errors), so **let the code check your deck instead of guessing**:

```
vlpl my.deck --check        # reads the deck, prints the summary and the warnings of the setup, exits
```

## What this skill is for, and how to work with the user

The purpose is to help the user **run the code for their physics problem**: turn the problem into a deck, explain
the choices, run it and read the result. Work in this order.

1. **Talk about the physics first.** Before writing a deck make sure you know, and ask in one short message for
   what is missing (offer a default for each item, so that the user can answer "go ahead"):
   - the driver: laser (wavelength, a0 or intensity, spot, duration, polarization, focus, angle of incidence) or
     beam (species, energy, charge, sizes);
   - the target: density and profile (gas, ramp, foil with or without pre-plasma, structure), species, whether
     ionization, collisions or QED matter;
   - what is to be measured (wake field, beam energy and emittance, reflected light and harmonics, charge states,
     photon spectra, energy balance);
   - the propagation length or duration, and accuracy against cost (first look or production).
   If the user wants no questions, or only numerical choices are open, go on with stated assumptions.
2. **Say which variant you take and why** (table below), and discuss the choices in physics terms: geometry and
   number of modes, the cells along and across the axis and what they resolve, the box and its boundaries, the
   number of steps, which physics is on or off. Name the numbers (n / n_c, lambda_p, c / omega_p, omega_p dt, z_R).
3. **Then the deck,** with a short explanation per block, the estimated run time and the checks to make
   ("Verify before delivering").
4. **Run it if asked,** look at the result with the user and propose the next step.

**Hard rule: do not modify the VLPL source code.** Building and running the code are allowed; patching it is
not. If the problem needs something the code cannot do, say so, name the limit, and suggest an issue at
https://github.com/AlexMPukhov/VLPL/issues. Debugging means debugging the deck and the run.

**Ask before long runs.** Give the cost first (step 4 of "Verify before delivering"); start nothing above about
one hour of the user's machine without a clear yes.

## Which variant

c dt equals the cell along the transport axis (x; in RZ: z) in every variant: waves along that axis have no
numerical dispersion. Put the laser or beam along it.

| variant | deck | take it when |
|---|---|---|
| VLPL-RIP | Cartesian (3D default, or `grid.geometry = cartesian2d`), `numerics.solver = rip_cc` | transverse cells coarser than the longitudinal one: dx^2 (1/dy^2 + 1/dz^2) + (omega_p dt)^2 / 4 < 1 (3D: h_perp > 1.42 dx) |
| VLPL-iRIP | Cartesian, `numerics.solver = irip` | h_perp < c dt: boosted frame with gamma >~ 2, tight focus, harmonics, fine transverse structure; needs omega_p dt < 2 only; MPI ranks along x only |
| VLPL-RZ | `grid.geometry = rz`, `numerics.solver = rip_cc` | round driver, `grid.dz` well below `grid.dr` (dr >= 4 dz with 2 modes) |
| VLPL-iRZ | `grid.geometry = rz`, `numerics.solver = irip` | RZ with dz comparable to or above dr, many modes, boosted frame |

- **RZ or 3D:** RZ when the driver is round: `grid.modes = 1` for beams on the axis, `2` for a linearly polarized
  laser (the usual LWFA choice), more for off-axis or structured drivers. 3D (2D for a first look) for oblique
  incidence, tilted targets, non-round beams.
- **Explicit or implicit:** choose the cells from the physics first; if the explicit condition then holds, take the
  explicit solver (cheaper per step, any MPI decomposition), otherwise the implicit one. The code stops with
  `unstable: ...` when an explicit deck violates the condition.
- **Number of steps = length / dx** (dx ~ lambda0 / 16 ... 32). Long underdense stages are expensive in the lab
  frame: use the boosted frame (`boost` block), or say that a reduced model fits better
  (`references/physics_rules.md`, section 6).

The rules with their numbers (resolution, particles per cell, filter, boosted frame, antenna near field, what to
use instead): `references/physics_rules.md`. **Read it before you choose grid values for a new case.**

## Start here

Paths in this skill (`examples/`, `scripts/`, `references/`) are relative to the directory of this `SKILL.md`,
not to the working directory of the user. Copy decks into the working directory and call scripts with their full
path.

1. The executable exists (`build/vlpl` in the clone, or where the user built it)? If not: skill `vlpl-setup`.
2. **New simulation:** copy the closest deck from `examples/` (table below), change the physics, then follow
   "Verify before delivering". Do not write a deck from memory: keys that do not exist are errors.
   For a plain "LWFA, a0, density, length" request: `examples/lwfa_plain_rz.deck` with the numbers of
   `python3 scripts/vlpl_numbers.py --density <cm^-3> --a0 <a0> --length <mm>` (it prints n / n_c, lambda_p, the
   matched spot, cells, window, plasma start, t_end, the solver, and with `--boost <gamma>` the boosted-frame
   numbers). Replace in the deck every value the script prints: `grid.nz`, `grid.dz`, `grid.dr`, `grid.nr`,
   `grid.t_end`, the `laser` values, `window.start`, and for both species `species.density`, `species.zmin`,
   `species.rmax` and the ramp in `species.density_function`. Over more than a Rayleigh length take the box the
   script prints for the unguided pulse (`grid.nr`, `species.rmax`).
3. **Which keys exist, units and defaults:** `references/deck_reference.md` (the reference of the code, long:
   search it for the block or key).
4. **What the code can do, with the keys and the example of each capability:** `references/capabilities.md`.
5. **Reading the output:** `references/diagnostics.md`; `python3 scripts/quick_look.py <output.dir>`.
6. **A run stops or looks wrong:** `references/debugging.md`.

## Examples (all run as they are)

Times are for the deck as shipped; get the time of your own deck from a smoke run.

| deck | capability | case |
|---|---|---|
| `examples/lwfa_plain_rz.deck` | laser wakefield, RZ, lab frame | a0 = 2, matched spot, 1e18 cm^-3, 0.3 mm; all numbers from `scripts/vlpl_numbers.py`; 44 min on 24 cores (9833 steps), E_z on the axis -0.87 E0 |
| `examples/rz_lwfa.deck` | laser wakefield, RZ (small, fast) | a0 = 2.5, 10 fs, 0.01 n_c, moving window; about a minute on 2 cores |
| `examples/lwfa_window.deck` | laser wakefield, 3D | the 3D twin of rz_lwfa; slices and a movie |
| `examples/rz_lwfa_boost.deck` | boosted frame | rz_lwfa at gamma = 2 with `boost` and `lab_diagnostics`, implicit solver |
| `examples/rz_ionization.deck` | field ionization | a0 = 2 pulse ionizes nitrogen to N5+ / N6+ (ADK), electrons created with the ionization current |
| `examples/collisions_2d.deck` | collisions, impact ionization | dense Al plasma, hot electrons relax on the ions and ionize Al3+ |
| `examples/qed_collision.deck` | QED and spin | 1 GeV polarized electrons against an a0 = 100 pulse: photon emission, spin, Stokes vectors |
| `examples/solid_target.deck` | solid target | p-polarized pulse at 45 deg on a tilted 30 n_c foil with pre-plasma; reflected field recorded |
| `examples/plasma_mirror_2d.deck` | media: metal-like (Drude) | a curved plasma mirror at 100 n_c as a particle-free medium, implicit solver |
| `examples/rz_medium.deck` | media: dielectric (Lorentz) and lossy plasma | glass layer (eps = 2.25) and a collisional plasma as particle-free media, RZ |
| `examples/dwa.deck` | dielectric structure with a beam | bunch through a dielectric-lined tube (eps = 3.8): wake on the axis |
| `examples/rz_corrugated_pipe.deck` | metal structure with a beam | bunch from an openPMD file (`scripts/make_bunch.py`) through a corrugated metal pipe given by a function: wake probes |

Restarts, MPI and GPU are run options, not deck types: see "Running" below.

## The deck in one page

```deck
grid {
  geometry   = cartesian2d      # cartesian (3D, default) | cartesian2d | rz
  lambda0    = 0.8 um           # the unit: lengths 1/k0, times 1/omega0, densities n_c(lambda0)
  nx         = 256              # transport axis: c dt = dx
  dx         = 0.05 um          # lambda0 / 16
  ny         = 64               # transverse; periodic unless boundary_y = pml
  dy         = 0.2 um
  boundary_x = open             # periodic | open (laser, window) | absorbing
  t_end      = 60 fs
}
numerics {
  solver = rip_cc               # rip_cc (explicit) | irip (implicit)
  shape  = 2                    # particle shape 1, 2 (3: Cartesian 3D)
}
laser {                         # enters through the open edge x = 0
  a0           = 1.5
  polarization = y
  duration     = 10 fs          # FWHM of the intensity; laser.delay = 2 x duration by default
  waist        = 4 um
  focus        = 10 um
}
window {
  start    = 40 fs              # >= 2 x laser.delay
  velocity = 1                  # in c
}
species electrons {
  charge  = -1
  mass    = 1
  density = 0.01                # n_c
  xmin    = 13 um               # absolute x: ahead of the box, enters with the window
  ppc_x   = 1
  ppc_y   = 2
}
species ions {
  charge  = 1
  mass    = 1836
  density = 0.01
  xmin    = 13 um
  ppc_x   = 1
  ppc_y   = 2
}
output {
  dir          = run1
  every        = 20             # steps between the rows of diag.txt
  slice_every  = 200
  slice_fields = Ey Ex n_electrons
}
```

- **Syntax:** `key = value` statements, one per line or separated by `;`; `#` starts a comment; a block is `name {` ... `}`, also on one line. Keys do not depend on the case. A key given twice in a block is an error.
  `references/deck_reference.md` lists the keys of a block in a compact form (`grid { lambda0, nx ny nz, ... }`): that
  is a list of keys, not deck syntax. Units: nm um mm cm m (mm*mrad), as fs ps ns s, fC pC nC uC, eV keV MeV GeV TeV,
  rad mrad urad deg; densities bare (n_c) or with cm^-3, m^-3, n_c. Common names of the VLPL family are accepted:
  `grid.m_max`, laser `wavelength`, `tau_fwhm`, `z_focus`, and the block `bunch <name>` (a Gaussian bunch by charge,
  energy, sizes and emittances, no file).

- Blocks with a name (`species <name>`, `laser <name>`, `medium <name>`, `collisions <name>`,
  `diagnostic <name>`) may appear several times. Override on the command line: `vlpl my.deck grid.t_end=30fs
  "species electrons.density=0.02"`.
- RZ renames the axes: `grid.nz`, `grid.dz` (transport axis z), `grid.nr`, `grid.dr`, `grid.boundary_z`,
  `grid.boundary_r`, `grid.modes`; species keys `species.zmin`, `species.rmax`, `species.ppc_r`,
  `species.ppc_theta`, `species.uz`.
- Densities: n / n_c with n_c = 1.1149e21 cm^-3 / (lambda0 in um)^2; a `species.density_function` of the absolute
  coordinates multiplies `species.density` (ramps, Gaussians, any expression).
- Positions are absolute lab coordinates, also with the moving window and in a boosted frame.

## Rules that decide whether the result is right

Details and reasons: `references/physics_rules.md`.

1. **Transport axis:** laser and beam along x (RZ: z). The step is c dt = `grid.dx` (RZ `grid.dz`).
2. **Resolution:** `grid.dx` <= lambda0 / 16 (first look), lambda0 / 32 (quantitative, and always in a boosted
   frame); transverse cell <= waist / 20 and <= (c / omega_p) / 5; solid targets lambda0 / 40 with
   `numerics.shape = 3` (3D) or 2 and `numerics.filter_passes = 1`.
3. **Stability:** explicit solvers must satisfy the condition of the table above (checked by the code);
   omega_p dt = 2 pi (dx / lambda0) sqrt(n / n_c) < 2 always, <= 0.2 for wakefields. Beyond: `numerics.mover =
   implicit` or a `medium` block.
4. **RZ particles:** `species.ppc_theta` >= 2 (2 `grid.modes` - 1), `species.ppc_r` >= 2, `numerics.shape = 2`.
5. **Window and plasma:** 2 x `laser.delay` <= `window.start` < box length / c: the pulse has entered, and the
   light that entered at t = 0 has not reached the right edge (otherwise a warning, and the Gauss law and div B
   are no longer at round-off). So the box is longer than 2 c `laser.delay` plus the plasma periods to be seen.
   The plasma begins ahead of the box (`species.xmin` beyond the right edge when the window starts).
6. **Transverse box:** the light must not come back: wide enough for the unguided spot w0 sqrt(1 + (x / z_R)^2),
   or absorbing (`grid.boundary_y = pml`, RZ `grid.boundary_r = absorbing`). The default RZ boundary is a
   conducting wall.
7. **Boosted frame:** implicit solver for gamma >~ 2, `numerics.shape = 2` and `numerics.filter_passes = 1`
   (binomial), neutral plasma per species pair with the same `ppc`, lab `grid.dx` <= lambda0 / 32. **Loading
   checks of beams:** fewer than 1 lattice plane is an error, fewer than 20 a warning, and the loaded charge is
   printed against the deck's: raise `species.ppc_x` until the warning is gone. **Antenna**
   (`laser.injection = antenna`): its field is within 1 % of the lab field only beyond
   d_1% = 1.1 (gamma (1 + beta))^2 h_lab downstream of x = 0 (h_lab = the lab `grid.dx`; printed by the summary):
   put plasma, collision points and ionization fronts beyond it.
8. **Current filter:** `numerics.filter = binomial` (default) for everything that streams (beams, wakefields,
   boosted plasma); `projection` only for harmonics from solid targets.
9. **Units of the output are code units**; convert with `scripts/quick_look.py` or the table in
   `references/diagnostics.md`.

## Verify before delivering

A deck is done only when these hold. Do them in this order.

1. **Check:** `vlpl my.deck --check` exits without `error`. Read the summary: the variant in the first line, cells,
   `steps = N`, the Courant number, lasers and species are what the user meant.
2. **Warnings:** every `warning:` line is fixed or explained to the user. They go to stderr, and several are
   printed only when a run starts, not by `--check` (window timing, omega_p dt, RZ particle numbers): look for them
   at the top of the smoke run's output as well.
3. **Smoke run:** a short run in a scratch directory with the threads / ranks of the full run, long enough that
   the physics has started: with a moving window run to `window.start` plus about 150 fs, so that plasma has
   entered (a run that ends before `window.start` has no particles and tests nothing), e.g.
   `OMP_NUM_THREADS=8 vlpl my.deck grid.t_end=470fs output.dir=smoke output.every=50 > smoke.log 2>&1`. Let it end
   by itself (set `grid.t_end`, do not kill it): only then it prints `done: N steps, T s` and the time per
   operator. One run at a time per machine. `python3 scripts/quick_look.py smoke` must show finite energies and max|dGauss| at
   round-off (1e-12 or below relative to max|rho|).
4. **Cost:** T / N of the smoke run times `steps` of the summary is a lower limit of the run time: the steps get
   slower until the window is full of plasma (about one box length after `window.start`), and the time per step
   grows with the number of particles (N_<species> in `diag.txt`). Scale T / N with the particle number at full
   fill, or measure it on a smoke run that goes that far. Tell the user; ask before starting
   anything above about one hour.
5. **Full run and a look at the result:** `python3 scripts/quick_look.py <output.dir>`; compare with the
   estimates in `references/diagnostics.md` (wake field against E0, laser amplitude against a0, energy balance)
   and report numbers, not impressions.
6. **Convergence,** for any number the user will rely on: one more run with the cells along the axis halved (then
   the transverse cells, then twice the particles). For a first look it may be skipped; then say that it was not
   done.

## Running

```
OMP_NUM_THREADS=8 vlpl my.deck                      # threads
mpirun -np 8 vlpl my.deck                           # MPI build; implicit solver: ranks along the transport axis only
vlpl my.deck checkpoint.every=2000                  # checkpoints (HDF5 build); block checkpoint { every, keep, stop_after_hours }
vlpl my.deck --restart latest                       # continue; --restart auto: or start fresh
```

A GPU build runs the same deck with no change. Builds, GPU selection, cluster jobs: skill `vlpl-setup`.
