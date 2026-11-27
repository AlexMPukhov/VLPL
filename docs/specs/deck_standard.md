# The common input form of the VLPL family in VLPL

Status: implemented (`src/deck.*`, `Deck::apply_standard_names`, block `bunch`: `src/structure.cpp`,
`src/beam_import.*`; tests `tests/test_deck.cpp`; decision 125). Alexander accepted the common standard for VLPL,
GEM-PIC-RZ and QUARZ on 2026-11-26; QUARZ reads it on its branch `deck-standard`.

The aim: who can write a deck for one code of the family can write it for the others; a quantity all codes have
has one name, one unit rule and one meaning. Every deck that ran before runs unchanged.

## Syntax

- Blocks `name { key = value }`; a statement ends at the end of the line or at `;` (outside quotes); a block may
  stand on one line. `#` starts a comment.
- Keys and the kind of a block (`grid`, the `species` of `species Electrons`) do not depend on the case; the name
  given to a block does (it appears in file names).
- A key given twice in a block is an error (before: the last one won silently). An unknown key is an error, as before.
- Command line: `vlpl deck [block.key=value ...] [--check] [--restart file|latest|auto]`, unchanged.

## Units

Lengths `nm um mm cm m` (and `mm*mrad` for emittances), times `as fs ps ns s`, charges `fC pC nC uC`, energies
`eV keV MeV GeV TeV`, angles `rad mrad urad deg`. A density is a bare number in n_c, as always, or carries a unit:
`cm^-3`, `m^-3` (converted with `grid.lambda0`) or `n_c` (species and Drude media).

## Common names

| common name | VLPL key | remark |
|---|---|---|
| `grid.m_max` | `grid.modes` | highest azimuthal mode; VLPL counts the modes: `modes = m_max + 1` |
| `laser.wavelength` | `grid.lambda0` | VLPL has one reference wavelength: it sets `grid.lambda0`, or must equal it |
| `laser.tau_fwhm` | `laser.duration` | FWHM of the intensity |
| `laser.z_focus` | `laser.focus` | |
| `species <name>.charge` with a unit | `total_charge` | total charge of a beam from a file; a bare number stays the charge number |

Two names of one quantity in one deck are an error.

## Block `bunch <name>`

A Gaussian bunch without a file, in the coordinates of the beam whatever the geometry: z along the propagation
(the x of a Cartesian run), x and y across it.

| key | meaning |
|---|---|
| `species` | `electron` (default) or `positron` |
| `charge` | total charge, with its unit |
| `energy` or `gamma` | mean kinetic energy with its unit, or the mean Lorentz factor |
| `energy_spread` | relative rms spread of gamma (default 0) |
| `sigma_x`, `sigma_y`, `sigma_z` | rms sizes (`sigma_y` default `sigma_x`) |
| `emittance_x`, `emittance_y` | normalized rms emittances (`emittance_y` default `emittance_x`; default 0) |
| `z_center` | the charge centroid at t = 0, lab coordinate along the propagation |
| `x_offset`, `y_offset` | transverse offsets from the axis (r = 0 in RZ, the centre of the box in Cartesian runs) |
| `macroparticles` | default 100000, equal weights |
| `self_field`, `t_start`, `rigid_until`, `mobile` | as for a beam from a file |

The bunch is at its waist at t = 0 and uncorrelated: transverse momenta u = p / (m c) of rms eps_n / sigma,
gamma = gamma0 (1 + energy_spread N(0, 1)), u_z from gamma and the transverse momenta. The Gaussians are not truncated,
so the charge is `charge` (to the round-off of the sum of the weights, 4e-12 for 200000). It becomes a species of the name of the block (for `diagnostic`,
`output.openpmd_species`).

Memory: every rank draws the same sequence of particles in portions of 262144 and keeps those in its part of the box;
no rank holds the whole bunch. The moments printed at the start are summed in one pass about the nominal centre and
gamma. A beam from a file is read in portions of 262144 particles too (`BeamFileSource`; the file is passed up to three
times: total weight, charge centroid, delivery), with the same particles as the former whole-file import.

## Not in VLPL (differences that remain)

- `output.every` counts rows of diag.txt in VLPL and field files in GEM-PIC-RZ and QUARZ.
- Positions are absolute lab coordinates; there is no co-moving `zeta_center`.
- Protons and ions as a `bunch`: use a species block with `charge`, `mass` and a density function.
- An absorbing radial boundary takes energy from a bunch (docs/STATUS.md, caveat of 2026-11-26): for a run whose
  result is the energy of a bunch use `boundary_r = wall` or a wide box.
