#!/bin/sh
# Agent skills (skills/, see docs/agent_skill.md): the parts that are copies of repository files.
#   sh tools/sync_skills.sh           regenerate the copies (run it after editing the README sections "Units" and
#                                     "Deck" or one of the example decks listed below)
#   sh tools/sync_skills.sh --check   exit 1 if a copy differs from its source (tests/agent_skill.sh, make test)
# A skill is installed without the rest of the repository, so it carries its own copy of the deck reference and of
# the example decks; the check keeps them identical to the sources.
SRC=$(cd "$(dirname "$0")/.." && pwd)
SK="$SRC/skills/vlpl-simulation"
# examples/<name>.deck copied into the skill (the skill's own decks are not listed here)
DECKS="rz_lwfa lwfa_window rz_lwfa_boost rz_ionization qed_collision collisions_2d plasma_mirror_2d rz_medium solid_target rz_corrugated_pipe"
OTHER="bench/rz_dielectric/dwa.deck:examples/dwa.deck research/rz_structure/make_bunch.py:scripts/make_bunch.py"
MODE=$1
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# The deck reference: the README sections "Units" and "Deck" with
#   - the paragraphs "Example: ..." / "Examples: ..." left out (the skill has its own table of examples),
#   - pointers into bench/ left out (benchmarks are not part of an installed skill),
#   - remarks of the form "(<words>: examples/<name>.deck)" left out.
# Other paths (docs/specs/..., examples/..., tools/...) stay: they refer to the repository.
{
  cat << 'EOF'
<!-- Generated from README.md (sections "Units" and "Deck") by tools/sync_skills.sh. Do not edit this copy. -->
# VLPL deck reference

All blocks and keys of the input deck with their defaults, as in the README of the repository. Paths such as
`docs/specs/<name>.md`, `examples/<name>.deck` and `tools/<name>.py` refer to the VLPL repository; "decision N" is
an entry of its `docs/decisions.md`. The file is long: search it for the block or key you need.

Notation: `block { key, key = default, a|b ... }` lists the keys of a block with defaults and choices, and `nx ny nz`
stands for three keys. It is not deck syntax: in a deck write one `key = value` per line, lengths, times and angles
with a unit (`um`, `fs`, `deg`, ...), densities in n_c without a unit.

EOF
  awk '
    /^## / { on = ($0 == "## Units" || $0 == "## Deck"); if (on) { sub(/^## /, "## "); } }
    on { text = text $0 "\n" }
    END {
      gsub(/[,;] bench\/[A-Za-z0-9_\/.]*[A-Za-z0-9_\/]/, "", text)
      gsub(/ ?\([^()]*: examples\/[a-z0-9_]+\.deck\)/, "", text)
      n = split(text, line, "\n"); skip = 0
      for (i = 1; i < n; i++) {
        if (line[i] ~ /^Examples?: /) skip = 1
        else if (skip && (line[i] == "" || line[i] ~ /^```/ || line[i] ~ /^## /)) skip = 0
        if (!skip) print line[i]
      }
    }' "$SRC/README.md"
} > "$TMP/deck_reference.md"

rc=0
put() {  # <generated file> <destination>
  if [ "$MODE" = "--check" ]; then
    if cmp -s "$1" "$2"; then echo "  in sync: ${2#$SRC/}"
    else echo "  OUT OF SYNC: ${2#$SRC/}  (run: sh tools/sync_skills.sh)"; rc=1; fi
  else
    mkdir -p "$(dirname "$2")" && cp "$1" "$2" && echo "  written: ${2#$SRC/}"
  fi
}
put "$TMP/deck_reference.md" "$SK/references/deck_reference.md"
for d in $DECKS; do put "$SRC/examples/$d.deck" "$SK/examples/$d.deck"; done
# other copies: "<source>:<file in the skill>"
for p in $OTHER; do put "$SRC/${p%%:*}" "$SK/${p#*:}"; done
exit $rc
