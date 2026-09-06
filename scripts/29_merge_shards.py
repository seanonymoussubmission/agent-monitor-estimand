#!/usr/bin/env python3
"""Merge sharded probe outputs into the single .npz the analysis expects.

Shards partition runs by index, so no unit is split across shards in a way that
matters -- but this asserts that every (task, run, k) appears exactly once,
because a silently duplicated row would inflate a within-task AUROC.
"""
import argparse, glob, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--shards", required=True, help="glob, e.g. 'out/probe_cpu_shard*.npz'")
ap.add_argument("--out", required=True)
ap.add_argument("--expected-runs", default="",
                help="CSV task,n_runs. Units with fewer processed runs are DROPPED "
                     "to avoid length-biased within-unit subsets.")
a = ap.parse_args()

fs = sorted(glob.glob(a.shards))
if not fs: sys.exit(f"no shards matched {a.shards}")
print(f"[merge] {len(fs)} shards: {[f.split('/')[-1] for f in fs]}")

Z = [np.load(f, allow_pickle=False) for f in fs]
layers = [int(x) for x in Z[0]["layers"]]
for z, f in zip(Z, fs):
    assert [int(x) for x in z["layers"]] == layers, f"{f}: layer mismatch"
    assert [int(x) for x in z["turns"]] == [int(x) for x in Z[0]["turns"]], f"{f}: turn mismatch"

out = {}
for key in ("task", "run", "y", "k", "pos", "n_asst", "repo"):
    out[key] = np.concatenate([z[key] for z in Z])
for L in layers:
    out[f"H{L}"] = np.concatenate([z[f"H{L}"] for z in Z])
out["layers"] = np.array(layers)
out["turns"] = Z[0]["turns"]
out["n_units"] = np.array([len(set(out["task"]))])

ids = list(zip(out["task"], out["run"], out["k"]))
assert len(ids) == len(set(ids)), f"DUPLICATE rows: {len(ids)-len(set(ids))} repeats"

if a.expected_runs:
    # Within a shard, work is sorted by prefix length ASCENDING, so a partially
    # processed shard has finished the SHORT-prefix runs of its tasks. Keeping
    # such a unit would select runs on a variable that correlates with outcome --
    # outcome-dependent selection INSIDE the unit, which biases within-task AUROC.
    # Drop any unit that is not fully processed rather than keep a biased subset.
    exp = {}
    for line in open(a.expected_runs):
        t, n = line.rsplit(",", 1); exp[t.strip()] = int(n)
    kmax = len(set(out["k"].tolist()))
    have = {}
    for t, r in zip(out["task"], out["run"]): have.setdefault(t, set()).add(int(r))
    complete = {t for t, runs in have.items() if len(runs) >= exp.get(t, 10**9)}
    keep = np.array([t in complete for t in out["task"]])
    dropped = len(set(out["task"])) - len(complete)
    print(f"[merge] complete-units filter: keeping {len(complete)} fully processed "
          f"units, dropping {dropped} partial ones ({(~keep).sum()} rows)")
    for k2 in list(out):
        if k2 in ("layers", "turns", "n_units"): continue
        out[k2] = out[k2][keep]
    out["n_units"] = np.array([len(complete)])
n_units = len(set(out["task"]))
mixed = sum(1 for t in set(out["task"])
            if len(set(out["y"][out["task"] == t])) > 1)
print(f"[merge] {len(ids)} rows | {n_units} tasks | {mixed} mixed-outcome | "
      f"pass rate {out['y'].mean():.3f}")
for k in sorted(set(out["k"].tolist())):
    print(f"[merge]   k={k}: {(out['k']==k).sum()} rows")
np.savez_compressed(a.out, **out)
print(f"[merge] -> {a.out}")
