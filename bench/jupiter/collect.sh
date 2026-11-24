#!/bin/sh
# Table of the timing series from the logs in $RUNS/scale_<N>gpu.<job>/*.log and $RUNS/cpu_<R>x<T>.<job>/*.log:
# time per step, speed-up and parallel efficiency against 1 GPU (strong: t1 / (N tN); weak: t1 / tN), and the
# time per step of every operator (rank 0; "other": diagnostics, output, start of the loop).
# With several jobs of the same case and size the latest counts. Markdown, to paste into docs/GPU_TESTING.md.
#   sh bench/jupiter/collect.sh [runs directory]
. "$(dirname "$0")/env.sh"
R=${1:-$RUNS}
ls -d "$R"/scale_*gpu.* "$R"/cpu_*.* 2> /dev/null | grep -q . || { echo "no runs in $R yet"; exit 1; }
python3 - "$R" <<'PY'
import glob, os, re, sys
rows = {}
for log in glob.glob(os.path.join(sys.argv[1], "*", "*.log")):
    d = os.path.basename(os.path.dirname(log))
    m = re.match(r"scale_(\d+)gpu(_ca)?(_noov)?\.(\d+)$", d) or re.match(r"cpu_(\d+x\d+)()()\.(\d+)$", d)
    if not m:
        continue
    gpus = int(m.group(1)) if d.startswith("scale_") else 0
    job, case = int(m.group(4)), os.path.basename(log)[:-4]
    txt = open(log, errors="replace").read()
    where = ("%d GPU" % gpus + (" CUDA-aware" if m.group(2) else "") + (" no overlap" if m.group(3) else "")) if gpus else "CPU " + m.group(1)
    r = dict(case=case, gpus=gpus, ca=bool(m.group(2)), noov=bool(m.group(3)), where=where, job=job, log=log)
    ex = re.search(r"exchanges \(max over ranks\): guard cells ([\d.]+) s in \d+ calls \(MPI wait ([\d.]+) s\), "
                   r"particle messages ([\d.]+) s", txt)
    r["ex"] = tuple(float(v) for v in ex.groups()) if ex else None
    dm = re.search(r"decomposition (\d+) x (\d+) x (\d+)", txt)
    r["decomp"] = "x".join(dm.groups()) if dm else "?"
    done = re.search(r"^done: (\d+) steps, ([\d.]+) s", txt, re.M)
    if done:
        r["steps"], r["sec"] = int(done.group(1)), float(done.group(2))
        ops = re.search(r"time per operator:(.*)", txt)
        r["ops"] = re.findall(r" (\S+) ([\d.]+) s \(\d+ %\)", ops.group(1)) if ops else []
        o = re.search(r"other ([\d.-]+) s", ops.group(1)) if ops else None
        if o:
            r["ops"].append(("other", o.group(1)))
    else:
        err = re.search(r"^error.*$", txt, re.M)
        r["fail"] = err.group(0) if err else "no 'done' line (still running, killed or out of time?)"
    pr = re.search(r"^  profile \((\d+) steps from step (\d+).*\n((?:    .*\n?)*)", txt, re.M)   # VLPL_PROFILE (src/prof.hpp)
    if pr:
        r["prof_steps"], r["prof_from"] = int(pr.group(1)), int(pr.group(2))
        r["prof"] = {mm.group(1).strip(): (float(mm.group(2)), float(mm.group(3)))
                     for mm in re.finditer(r"^    (\S.*?)\s+([\d.]+)\s+([\d.]+)\s*$", pr.group(3), re.M)}
    key = (case, r["where"])
    if key not in rows or rows[key]["job"] < job:
        rows[key] = r
base = {c: r["sec"] / r["steps"] for (c, w), r in rows.items() if r["gpus"] == 1 and not r["ca"] and not r["noov"] and "sec" in r}
print("| case | run | ranks (x y z) | steps | s/step | speed-up | efficiency | per operator, ms/step | "
      "exchanges, ms/step: guard cells (MPI wait), particles |")
print("|---|---|---|---|---|---|---|---|---|")
for (c, w), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[1]["gpus"] == 0, kv[1]["gpus"], kv[1]["ca"], kv[1]["noov"])):
    if "fail" in r:
        print("| %s | %s | %s | | | | | FAILED: %s (%s) | |" % (c, w, r["decomp"], r["fail"], r["log"]))
        continue
    exs = "%.2f (%.2f), %.2f" % tuple(1e3 * v / r["steps"] for v in r["ex"]) if r["ex"] else ""
    ps = r["sec"] / r["steps"]
    b = base.get(c.replace("_weak", "_strong") if c.endswith("_weak") else c)
    sp = eff = ""
    if b and r["gpus"] > 1:
        if c.endswith("_weak"):
            eff = "%.0f %% (weak)" % (100 * b / ps)
        else:
            sp, eff = "%.2f" % (b / ps), "%.0f %%" % (100 * b / ps / r["gpus"])
    elif b and r["gpus"] == 0:
        sp = "%.3f" % (b / ps)
    ops = ", ".join("%s %.2f" % (n, 1e3 * float(s) / r["steps"]) for n, s in r["ops"])
    print("| %s | %s | %s | %d | %.4f | %s | %s | %s | %s |" % (c, w, r["decomp"], r["steps"], ps, sp, eff, ops, exs))
print()
print("speed-up: t(1 GPU) / t; for the CPU rows t(1 GPU) / t(CPU node), i.e. < 1 when one GPU is faster than the node.")
print("exchanges: the slowest rank's time in the guard-cell exchanges (packing, copies and MPI; in brackets the")
print("MPI wait alone) and in the particle messages; empty for logs of versions before the exchange report.")
print("weak: bubble_weak on N GPUs has N times the cells and particles of bubble_strong on 1 GPU (efficiency t1 / tN).")
# the steady state from the step profile (VLPL_PROFILE, scaling.sbatch): the slowest rank's ms per step of the
# operators and of the exchange parts, over the counted steps (bubble: the box full of plasma)
cols = ["push_deposit", "push.add_guards J", "push.xchg J", "push.interior wait", "rip_cc", "moving_window", "migrate",
        "migrate.device", "migrate.route", "migrate.mpi", "migrate.append", "halo.mpi"]
prof = [(k, r) for k, r in rows.items() if r.get("prof")]
if prof:
    print()
    print("Steady state (step profile, max over ranks, ms per step):")
    print()
    print("| case | run | steps (from) | step | " + " | ".join(cols) + " |")
    print("|---|---|---|---|" + "---|" * len(cols))
    for (c, w), r in sorted(prof, key=lambda kv: (kv[0][0], kv[1]["gpus"] == 0, kv[1]["gpus"], kv[1]["ca"], kv[1]["noov"])):
        P = r["prof"]
        top = sum(v[0] for n, v in P.items() if "." not in n and n != "wait for ranks")
        print("| %s | %s | %d (%d) | %.2f | " % (c, w, r["prof_steps"], r["prof_from"], top) +
              " | ".join("%.2f" % P[n][0] if n in P else "" for n in cols) + " |")
    print()
    print("step: the sum of the operators (max over ranks each, so an upper bound of the time per step). push.add_guards J")
    print("and push.xchg J run while the interior particles are pushed (overlap); push.interior wait is what is left of")
    print("the interior push after them. migrate.device: flag, scan, pack, close the holes, copy to the host; route: the")
    print("host part with migrate.mpi (the messages, incl. waiting for the neighbours); append: the upload of the arrivals.")
PY
