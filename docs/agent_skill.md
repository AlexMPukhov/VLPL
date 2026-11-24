# Agent skills for VLPL

An *agent skill* is a folder with a `SKILL.md` file (and optional references, scripts and examples) that an AI
coding agent loads when a task calls for it. The repository ships two, so that an agent can build VLPL, choose the
solver variant, and write, check, run and debug input decks from a request in plain words. Decision 119.

## Format and where it works

The skills follow the open [Agent Skills specification](https://agentskills.io/specification):

- a directory named like the skill, with `SKILL.md`: YAML frontmatter with `name` (lower case, digits, hyphens,
  <= 64 characters, equal to the directory name) and `description` (<= 1024 characters: what the skill does and when
  to use it), then the instructions in Markdown;
- optional `references/` (read on demand), `scripts/` (run by the agent), and other files (`examples/` here);
- *progressive disclosure*: the agent always sees only name and description; it loads `SKILL.md` (kept below 500
  lines) when the task matches, and the other files only when `SKILL.md` sends it there.

The format is not tied to one vendor or model. It is read natively by Claude Code, OpenCode, Codex, Cursor,
Gemini CLI, Goose, Cline, Qwen Code, OpenHands and others. An agent without skill support can still use it: tell it
to read `skills/vlpl-simulation/SKILL.md` and follow it.

## Install

```
npx skills add AlexMPukhov/VLPL                       # asks which skills and which agents
npx skills add AlexMPukhov/VLPL -a opencode -g        # one agent, for all projects of the user
npx skills add AlexMPukhov/VLPL --skill vlpl-simulation
```

`skills` is the open-source installer of [skills.sh](https://skills.sh) (needs Node.js). It looks for
`skills/<name>/SKILL.md` in the repository. Without Node.js copy the folders:

| agent | project directory | user directory |
|---|---|---|
| Claude Code | `.claude/skills/` | `~/.claude/skills/` |
| OpenCode | `.agents/skills/` | `~/.config/opencode/skills/` |
| Codex, Cline | `.agents/skills/` | `~/.agents/skills/` |
| Cursor | `.agents/skills/` | `~/.cursor/skills/` |
| Gemini CLI | `.agents/skills/` | `~/.gemini/skills/` |
| Goose | `.goose/skills/` | `~/.config/goose/skills/` |

```
cp -r VLPL/skills/vlpl-* ~/.claude/skills/        # e.g. Claude Code, for all projects
```

## The two skills

```
skills/
  vlpl-setup/
    SKILL.md                      clone, make (OpenMP, MPI, CUDA, HDF5), tests, running (threads, ranks, GPUs,
                                  overrides, restarts, exit codes), output files, cluster jobs
  vlpl-simulation/
    SKILL.md                      dialogue first; which variant (VLPL-RIP, VLPL-iRIP, VLPL-RZ, VLPL-iRZ) and the
                                  conditions that decide it; table of examples; the deck in one page; 9 rules;
                                  "verify before delivering"; hard rule: the agent does not modify the source
    references/deck_reference.md  all blocks, keys, units, defaults      (generated from README "Units", "Deck")
    references/physics_rules.md   transport axis, variant choice, resolution as numbers, filter, boosted frame
                                  (loading checks, antenna near field d_1%), what to use instead
    references/capabilities.md    each capability: keys, example, where it works, what to check
    references/diagnostics.md     code units, diag.txt, fields, beams, module logs, lab-frame output
    references/debugging.md       errors, warnings, symptoms and fixes
    examples/*.deck               copies of examples/: rz_lwfa, lwfa_window, rz_lwfa_boost, rz_ionization,
                                  qed_collision, collisions_2d, plasma_mirror_2d, rz_medium, solid_target,
                                  rz_corrugated_pipe; of bench/rz_dielectric/dwa.deck;
                                  lwfa_plain_rz (the skill's own: all numbers from vlpl_numbers.py)
    scripts/vlpl_numbers.py       n / n_c, lambda_p, matched spot, P / P_c; cells, window, plasma start, t_end,
                                  solver; --boost: boosted cell, steps, filter passes, antenna d_1%
    scripts/quick_look.py         diag.txt, beams, E_z on the axis, slices in physical units (standard library)
    scripts/make_bunch.py         a Gaussian bunch as an openPMD file (copy of research/rz_structure/make_bunch.py)
```

Capabilities and their example in the skill: laser wakefield (RZ, 3D), boosted frame, field ionization, collisions
and impact ionization, QED with spin, solid targets, linear media without particles (Drude: metal-like and lossy
plasma; Lorentz: dielectric), dielectric bodies and metal bodies with a beam, beams with self-field and from files;
restarts, MPI and GPU as run options. Metals and dielectrics are presented as generic materials and shapes.

## Keeping the skills in step with the code

A skill is installed without the rest of the repository, so it carries copies. Two mechanisms keep them right:

- `sh tools/sync_skills.sh` regenerates `references/deck_reference.md` from the README sections "Units" and "Deck"
  (without the "Example(s): ..." paragraphs, pointers into `bench/`, and remarks of the form
  "(...: examples/<name>.deck)") and copies the decks and `make_bunch.py`. Run it after editing those.
- `tests/agent_skill.sh`:
  - `--static` (part of `make test`, no executable, a second):
    1. the format of the specification (names, description length, `SKILL.md` <= 500 lines, every file of the skill
       that a skill file names exists);
    2. the copies equal their sources (`tools/sync_skills.sh --check`);
    3. every deck key that the hand-written skill files name as `block.key` (148 keys) is read by the parser (a
       string literal in `src/*.cpp`, or built per axis as `"ppc_" + x`) and is in the deck reference.
  - `make agent_skill` (= `sh tests/agent_skill.sh build/vlpl [<mpi vlpl>]`, launcher from `MPIRUN`), in addition:
    4. every ```` ```deck ```` snippet of the skill files passes `vlpl --check`;
    5. every deck of the skills passes `--check` and runs 20 steps (exit 0, a `done:` line, finite numbers in
       `diag.txt`); the deck with a bunch file with the file written by the skill's script (skipped without HDF5 or
       h5py);
    6. a run stopped at a checkpoint and continued with `--restart latest` equals the uninterrupted run (HDF5);
    7. `vlpl_numbers.py` against the code's summary: the number of steps of `lwfa_plain_rz.deck`, its density in
       n_c, and the antenna near-field distance d_1% against the boost summary; `quick_look.py` on an output directory;
    8. with an MPI executable: 2 ranks = 1 rank (energies of `diag.txt` to 1e-9).

Rules for editing the skills: write deck keys as `block.key` (so that the test sees them); put a complete small
deck into a ```` ```deck ```` block if it should be checked; keep `SKILL.md` short and move details to
`references/`. **The skills are part of the public release: nothing private goes in** (no collaborators' cases,
no site paths or host names, no results that are not published in this repository).

## What the rules are based on

| in the skill | source |
|---|---|
| explicit stability conditions, omega_p dt limits, window and laser timing, RZ ppc_theta, filter choice, boosted-frame requirements, loading checks, antenna d_1% = 1.1 (gamma (1 + beta))^2 h_lab | the code's own checks and warnings, `docs/specs/`, decisions 79, 87, 89, 91, 104, 118 |
| resolution in a boosted frame (lambda0 / 32) | tests B1, B2 (`docs/specs/boosted_frame.md`) |
| lambda0 / 16 (first look), solid target numbers | the repository examples |
| transverse cell <= waist / 20 and <= (c / omega_p) / 5, ppc values for cold plasma, window length (pulse + 1.5 plasma periods), box = 3 x spot, omega_p dt <= 0.2 for wakefields | starting values proposed by the session that wrote the skill (common PIC practice), **not measured for VLPL**: to be confirmed or replaced by Alexander |

## Trial with a fresh agent

One fresh agent session (2026-11-24; a small Claude model run as a sub-agent) got the two skill folders, an existing
build reachable through a wrapper (one cluster node, 24 cores, 15 min per call) and the one-line request
"LWFA, a0 = 2, 800 nm, n = 1e18 cm^-3, 2 mm", with the instruction to use nothing but the skill files, to make
checks and smoke runs only, and without the possibility to ask questions.

| | result |
|---|---|
| first reply | five questions with defaults (laser, target, what to measure, "2 mm of plasma?", first look or production), the variant VLPL-RZ with 2 modes and the reason, the numbers (lambda_p 33.4 um, z_R 887 um, P = 1.0 P_c, omega_p dt 0.0094) |
| deck | `examples/lwfa_plain_rz.deck` with the numbers of `vlpl_numbers.py --length 2`; the box for the unguided pulse (2 mm = 2.3 z_R: plasma to 111 um, nr = 181) chosen without being told |
| `--check` attempts until clean | 1; no unknown key, no error of the code |
| smoke run | 2998 steps in 368 s (0.12 s per step at 1.1e6 particles per species); a warning at the start of the run and max\|the full run of the deck (9833 steps, 44 min on 24 cores): no warning, max|dGauss| 6e-15, E_z on the axis -0.87 E0 || = 3.8e-5, flagged by `quick_look.py`; after the fix 1.7e-15 |
| cost estimate | 43830 steps, 1.5 - 2.3 h on 24 cores; the agent did not start the run and listed the alternatives (1 mm, boosted frame, GEM-PIC-RZ) |

What the first version of the skill got wrong, by its report, and what was changed:

| finding | change |
|---|---|
| `vlpl_numbers.py` and the plain deck set `window.start` = 320 fs, 2.5 fs later than box length / c: the light that entered at t = 0 reached the right edge before the window moved; the code warns only when the run starts, and the Gauss law and div B were at 4e-5 instead of round-off | the script takes the latest start below box / c (315 fs); rule 5, the resolution table and the debugging table state 2 x delay <= `window.start` < box length / c; the full run of the deck: max\|dGauss\| 4e-15 |
| `quick_look.py --lambda0 0.8`, as documented, printed the usage text and stopped | option parsing fixed; the test calls it with the options |
| the smoke-run recipe (`grid.t_end=20fs`) ends before the window starts and before any particle enters | recipe: run to `window.start` + 150 fs; the script prints that value; the run must end by itself to print `done:` |
| several warnings come only at the start of a run, the checklist spoke of `--check` | step 2 says so |
| the cost from the first steps is too low: the steps slow down while the window fills | step 4: a lower limit, scale with the particle number at full fill |
| adapting the example to another length meant finding the keys to change | the recipe lists the keys; the script prints `grid.nr` and `species.rmax` for both boxes |
| the large W_field (the laser energy) next to a small W_kin was not explained; E_z was not compared with E0 | a note in `quick_look.py` and `diagnostics.md`; `quick_look.py --density` prints E_z in units of E0 |
| transverse cell: waist / 10 in one file, waist / 20 in another | waist / 20 everywhere |

Limits of this trial: one agent, one request, a sub-agent of the session that wrote the skill (with the instruction to
ignore everything else), not a separately installed agent; the full 2 mm run was not made; the `skills` installer
itself was not run.

## Left out

- Side injection (`side`, flying focus), LASY laser files, the harmonic-focus diagnostic, the implicit mover and
  the o5 closure are named with their keys but have no worked example in the skill.
- No plotting script beyond `quick_look.py` (the repository's `tools/` are referred to).
- Chains of codes (hand-over of beams between VLPL, GEM-PIC-RZ and QUARZ) belong to a later umbrella handbook.
- The `skills` installer itself was not run (the layout follows its documented discovery rule).
