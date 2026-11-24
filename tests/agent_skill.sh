#!/bin/sh
# Agent skills (skills/*/SKILL.md, docs/agent_skill.md) against the code, so that they cannot drift from it:
#   1. format of the open Agent Skills specification (agentskills.io): frontmatter name = directory name
#      (lower case, digits, hyphens, <= 64), description 1 ... 1024 characters, SKILL.md <= 500 lines; every file
#      of the skill that a skill file names (references/, scripts/, examples/) exists
#   2. the copies of the deck reference (README "Units", "Deck") and of the example decks equal their sources
#      (tools/sync_skills.sh --check)
#   3. every deck key the hand-written skill files name as block.key is a key the parser reads (a string literal
#      in src/*.cpp) and is in the deck reference
#   --- with an executable ---
#   4. every ```deck snippet of the skill files passes  vlpl --check
#   5. every deck shipped in a skill passes --check and runs STEPS steps (exit 0, a "done:" line, finite numbers in
#      diag.txt); the deck with a bunch file with the file written by the skill's own script (needs h5py and HDF5)
#   6. restart: a run stopped at a checkpoint and continued equals the uninterrupted run (needs HDF5)
#   7. the helper scripts: vlpl_numbers.py against the code's summary (steps, antenna near-field distance),
#      quick_look.py on an output directory
#   8. with an MPI executable: 2 ranks = 1 rank (diag.txt energies to 1e-9)
# usage: sh tests/agent_skill.sh --static                         (1 - 3: no executable; part of make test)
#        sh tests/agent_skill.sh <vlpl> [<mpi vlpl>]             (make agent_skill; MPIRUN = the launcher, default mpirun)
SRC=$(cd "$(dirname "$0")/.." && pwd)
abspath() { (cd "$(dirname "$1")" && echo "$(pwd)/$(basename "$1")"); }
STATIC=0; EXE=""; MPIEXE=""
if [ "$1" = "--static" ]; then STATIC=1
else
  [ -x "$1" ] || { echo "usage: sh tests/agent_skill.sh --static | <vlpl> [<mpi vlpl>]"; exit 2; }
  EXE=$(abspath "$1")
  [ -n "$2" ] && MPIEXE=$(abspath "$2")
fi
MPIRUN=${MPIRUN:-mpirun}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
fail=0
bad() { echo "  FAILED: $*"; fail=1; }
# ---- code-specific settings -----------------------------------------------------------------------------------
# blocks of the input deck: a word of the skill files written as block.key is taken as a deck key
BLOCKS="grid numerics species laser side window boost lab_diagnostics pml output checkpoint medium collisions qed diagnostic harmonic_focus rings dielectric metal quadrupole wake beam_slices side_flux"
REFERENCE="$SRC/skills/vlpl-simulation/references/deck_reference.md"   # the generated copy (checked in 2.)
MINSKILLS=2; MINKEYS=60; MINDECKS=10  # fewer found = the extraction is broken
STEPS=20
SIM="$SRC/skills/vlpl-simulation"
check_deck() { "$EXE" "$@" --check; }                       # <deck> [overrides]: parse and validate only
# <deck> <steps>: grid.t_end for that many steps, from the cell along the transport axis (dx; RZ: dz) in um
t_end() {
  awk -v n="$2" '
    /^[ \t]*geometry[ \t]*=[ \t]*rz/ { rz = 1 }
    /^[ \t]*dx[ \t]*=/ { dx = $3; ux = $4 }
    /^[ \t]*dz[ \t]*=/ { dz = $3; uz = $4 }
    END { h = rz ? dz : dx; u = rz ? uz : ux; if (u == "nm") h *= 1e-3; if (u == "mm") h *= 1e3
          if (h > 0) printf "%.6gfs", (n + 0.5) * h / 0.299792458 }' "$1"
}
# <deck> <output dir> <steps> [overrides]: a short run
short_run() {
  deck=$1; out=$2; n=$3; shift 3
  te=$(t_end "$deck" "$n")
  [ -n "$te" ] || { echo "no cell size along the transport axis in the deck"; return 1; }
  OMP_PROC_BIND=false "$EXE" "$deck" grid.t_end="$te" output.dir="$out" output.every=5 "$@"
}
finite() { ! grep -v '^#' "$1" | grep -qiE 'nan|inf'; }
# ---- end of the code-specific settings ------------------------------------------------------------------------
PY=""
python3 -c "import sys" 2>/dev/null && PY=python3

echo "1. format (agentskills.io specification)"
nskill=0
for dir in "$SRC"/skills/*/; do
  dir=${dir%/}; name=$(basename "$dir"); f="$dir/SKILL.md"; nskill=$((nskill + 1))
  [ -f "$f" ] || { bad "$name: no SKILL.md"; continue; }
  [ "$(head -1 "$f")" = "---" ] || bad "$name: SKILL.md must start with the YAML frontmatter (---)"
  n=$(awk '/^---$/ { c++; next } c == 1 && /^name:/ { sub(/^name:[ \t]*/, ""); print; exit }' "$f")
  d=$(awk '/^---$/ { c++; next } c == 1 && /^description:/ { sub(/^description:[ \t]*/, ""); print; exit }' "$f")
  [ "$n" = "$name" ] || bad "$name: frontmatter name '$n' differs from the directory name"
  echo "$n" | grep -Eq '^[a-z0-9]+(-[a-z0-9]+)*$' || bad "$name: name must be lower case letters, digits, single hyphens"
  [ ${#n} -le 64 ] || bad "$name: name longer than 64 characters"
  [ ${#d} -ge 1 ] && [ ${#d} -le 1024 ] || bad "$name: description has ${#d} characters (1 ... 1024)"
  l=$(wc -l < "$f"); [ "$l" -le 500 ] || bad "$name: SKILL.md has $l lines (<= 500)"
  for md in "$f" "$dir"/references/*.md; do
    [ -f "$md" ] || continue
    for ref in $(grep -oE '(references|scripts|examples)/[A-Za-z0-9_.-]+\.[a-z0-9]+' "$md" | sort -u); do
      case "$md:$ref" in */deck_reference.md:*) continue ;; esac      # the copy names repository paths
      [ -f "$dir/$ref" ] || bad "$name: $(basename "$md") names $ref, which is not in the skill"
    done
  done
  echo "  $name: name ok, description ${#d} characters, $l lines"
done
[ "$nskill" -ge $MINSKILLS ] || bad "expected at least $MINSKILLS skills in skills/, found $nskill"

echo "2. copies of repository files"
sh "$SRC/tools/sync_skills.sh" --check || fail=1

echo "3. deck keys named in the skill files"
# hand-written files only (references/deck_reference.md is the copy checked in 2.)
MD=$(ls "$SRC"/skills/*/SKILL.md "$SRC"/skills/*/references/*.md | grep -v '/deck_reference.md$')
alt=$(echo $BLOCKS | tr ' ' '|')
cat $MD | grep -oE "(^|[^A-Za-z0-9_.])($alt)\.[a-z_][a-z_0-9]*" |
  sed -E 's/^[^a-z]*//' | grep -vE '\.(deck|txt|h5|md|sh|py|cpp|hpp|bin)$' | sort -u > keys.txt
nkey=0
while IFS=. read -r block key; do
  nkey=$((nkey + 1))
  # a key is a string literal of the sources, or built per axis: "ppc_" + x, x + "min"
  stem=${key%[xyz]}; rest=${key#[xyz]}
  cat "$SRC"/src/*.cpp | grep -q -e "\"$key\"" -e "\"$stem\" + " -e " + \"$rest\"" ||
    bad "$block.$key: the parser (src/*.cpp) reads no key \"$key\""
  grep -qE "(^|[^A-Za-z0-9_])$key([^A-Za-z0-9_]|\$)" "$REFERENCE" || bad "$block.$key: not in ${REFERENCE#$SRC/}"
done < keys.txt
echo "  $nkey keys checked against src/*.cpp and ${REFERENCE#$SRC/}"
[ "$nkey" -ge $MINKEYS ] || bad "only $nkey keys found in the skill files: the extraction is broken"

if [ $STATIC = 1 ]; then
  if [ $fail = 0 ]; then echo "agent skill test (static part) passed"; else echo "agent skill test (static part) FAILED"; fi
  exit $fail
fi

HDF5=0
check_deck "$SIM/examples/rz_lwfa.deck" checkpoint.every=10 > hdf5.log 2>&1 && HDF5=1
H5PY=0
[ -n "$PY" ] && python3 -c "import numpy, h5py" 2>/dev/null && H5PY=1
echo "   (executable $EXE; HDF5 in the build: $HDF5; python3: ${PY:-no}; h5py: $H5PY)"

echo "4. deck snippets of the skill files"
nsnip=0
for f in $MD; do
  awk -v base="snip_$(basename "$(dirname "$f")")_$(basename "$f" .md)" '
    /^```deck[ \t]*$/ { n++; out = base "_" n ".deck"; next }
    /^```/ { out = ""; next }
    out != "" { print > out }' "$f"
done
for s in snip_*.deck; do
  [ -f "$s" ] || continue
  nsnip=$((nsnip + 1))
  if check_deck "$s" > "$s.log" 2>&1; then echo "  $s: ok"
  else bad "$s does not pass --check:"; tail -3 "$s.log"; fi
done
[ "$nsnip" -ge 1 ] || bad "no \`\`\`deck snippet found in the skill files"

echo "5. decks shipped in the skills: --check and $STEPS steps"
ndeck=0
for deck in "$SRC"/skills/*/examples/*.deck; do
  b=$(basename "$deck" .deck); ndeck=$((ndeck + 1))
  extra=""
  if grep -q '^[ \t]*file[ \t]*=' "$deck"; then          # a bunch file: written by the skill's script
    if [ $HDF5 != 1 ] || [ $H5PY != 1 ]; then echo "  $b: skipped (needs HDF5 and python3 with numpy and h5py)"; continue; fi
    python3 "$SIM/scripts/make_bunch.py" bunch.h5 100 20000 2000 100 20 1 50 > "$b.make.log" 2>&1 || { bad "$b: make_bunch.py"; continue; }
  fi
  check_deck "$deck" > "$b.check.log" 2>&1 || { bad "$b: --check"; tail -3 "$b.check.log"; continue; }
  if ! short_run "$deck" "out_$b" $STEPS > "$b.run.log" 2>&1; then
    bad "$b: the run stopped"; tail -5 "$b.run.log"; continue
  fi
  grep -q '^done:' "$b.run.log" || bad "$b: no 'done:' line"
  [ -s "out_$b/diag.txt" ] || { bad "$b: no diag.txt"; continue; }
  finite "out_$b/diag.txt" || bad "$b: NaN or Inf in diag.txt"
  echo "  $b: ok ($(grep '^done:' "$b.run.log"); $(head -1 "$b.check.log" | cut -c1-40))"
done
[ "$ndeck" -ge $MINDECKS ] || bad "expected at least $MINDECKS example decks in the skills, found $ndeck"

echo "6. restart from a checkpoint"
if [ $HDF5 = 1 ]; then
  D="$SIM/examples/rz_lwfa.deck"
  short_run "$D" r_full 40 > r_full.log 2>&1 || bad "restart: the uninterrupted run stopped"
  short_run "$D" r_part 20 checkpoint.every=20 > r_part.log 2>&1 || bad "restart: the first part stopped"
  short_run "$D" r_part 40 --restart latest > r_cont.log 2>&1 || bad "restart: the continuation stopped"
  grep -q 'restart from' r_cont.log || bad "restart: no 'restart from' line"
  tail -1 r_part/diag.txt > r_last.txt
  if [ "$(tail -1 r_full/diag.txt | awk '{ print $1, $3, $4 }')" = "$(awk '{ print $1, $3, $4 }' r_last.txt)" ]; then
    echo "  rz_lwfa: 20 steps + restart + 20 steps = 40 steps (step, W_field, W_kin identical)"
  else bad "restart: the continued run differs from the uninterrupted one"; tail -1 r_full/diag.txt; cat r_last.txt; fi
else
  echo "  skipped (build without HDF5)"
fi

echo "7. helper scripts"
if [ -n "$PY" ]; then
  for s in "$SIM"/scripts/*.py; do
    python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$s" || bad "$(basename "$s"): syntax"
  done
  # steps of the plain LWFA deck: vlpl_numbers.py against the code's summary
  python3 "$SIM/scripts/vlpl_numbers.py" --density 1e18 --a0 2 --length 0.3 > numbers.log 2>&1 || bad "vlpl_numbers.py stopped"
  ns=$(awk '/grid.t_end = .* of plasma/ { print $(NF - 1) }' numbers.log)
  nc=$(sed -n 's/.*steps = \([0-9]*\).*/\1/p' lwfa_plain_rz.check.log | head -1)
  if [ -n "$ns" ] && [ "$ns" = "$nc" ]; then echo "  vlpl_numbers.py: $ns steps = the summary of lwfa_plain_rz.deck  ok"
  else bad "vlpl_numbers.py: $ns steps, the code's summary of lwfa_plain_rz.deck has $nc"; fi
  # density in n_c: the script against the deck value that the code accepted
  dn=$(awk '/^plasma:/ { print $7 }' numbers.log)
  dd=$(awk '/^[ \t]*density[ \t]*=/ { print $3; exit }' "$SIM/examples/lwfa_plain_rz.deck")
  awk -v a="$dn" -v b="$dd" 'BEGIN { d = (a - b) / b; if (d < 0) d = -d; exit !(b > 0 && d < 1e-3) }' ||
    bad "vlpl_numbers.py: n = $dn n_c, lwfa_plain_rz.deck has $dd"
  # antenna near field d_1% = 1.1 (gamma (1 + beta))^2 h_lab: the script against the code's boost summary
  check_deck "$SIM/examples/rz_lwfa_boost.deck" grid.dz=0.025um laser.injection=antenna > antenna.log 2>&1 || bad "antenna --check"
  dc=$(sed -n 's/.*within 1 % only \([0-9.eE+-]*\) (1\/k0).*/\1/p' antenna.log | head -1)
  dp=$(python3 "$SIM/scripts/vlpl_numbers.py" --density 1.742e19 --a0 2.5 --cells 32 --boost 2 | sed -n 's/.*d_1% = .* = \([0-9.eE+-]*\) um.*/\1/p')
  if awk -v a="$dp" -v b="$dc" 'BEGIN { b = b * 0.8 / 6.283185307; d = (a - b) / (b > 0 ? b : 1); if (d < 0) d = -d; exit !(b > 0 && d < 5e-3) }'; then
    echo "  vlpl_numbers.py: antenna d_1% = $dp um (code: $dc / k0)  ok"
  else bad "vlpl_numbers.py: antenna d_1% = $dp um, the code prints $dc (1/k0)"; fi
  if python3 "$SIM/scripts/quick_look.py" out_lwfa_plain_rz --lambda0 0.8 --density 1e18 > look.log 2>&1 && grep -q 'max|dGauss|' look.log; then
    echo "  quick_look.py: ok ($(grep 'max|dGauss|' look.log | sed 's/^ *//'))"
  else bad "quick_look.py on out_lwfa_plain_rz"; tail -3 look.log; fi
else
  echo "  skipped (no python3)"
fi

echo "8. MPI: 2 ranks = 1 rank"
if [ -n "$MPIEXE" ]; then
  D="$SIM/examples/rz_lwfa.deck"; te=$(t_end "$D" 200)
  OMP_NUM_THREADS=1 "$MPIEXE" "$D" grid.t_end="$te" output.dir=m1 output.every=50 window.start=10fs laser.duration=2fs "species electrons.zmin=2um" "species protons.zmin=2um" > m1.log 2>&1 || bad "MPI executable on 1 rank"
  OMP_NUM_THREADS=1 $MPIRUN -np 2 "$MPIEXE" "$D" grid.t_end="$te" output.dir=m2 output.every=50 window.start=10fs laser.duration=2fs "species electrons.zmin=2um" "species protons.zmin=2um" > m2.log 2>&1 || { bad "$MPIRUN -np 2"; tail -3 m2.log; }
  if [ -s m1/diag.txt ] && [ -s m2/diag.txt ] && paste m1/diag.txt m2/diag.txt | awk '
      /^#/ { next } { n = NF / 2; for (i = 3; i <= 4; i++) { a = $i; b = $(i + n); d = a - b; if (d < 0) d = -d
        s = (a < 0 ? -a : a) + 1e-300; if (d / s > 1e-9) bad = 1 } rows++ } END { exit !(rows > 1 && !bad) }'; then
    echo "  rz_lwfa (200 steps, plasma in the box): W_field, W_kin of 2 ranks = 1 rank to 1e-9"
  else bad "MPI: diag.txt of 2 ranks differs from 1 rank"; fi
else
  echo "  skipped (no MPI executable given)"
fi

if [ $fail = 0 ]; then echo "agent skill test passed"; else echo "agent skill test FAILED"; fi
exit $fail
