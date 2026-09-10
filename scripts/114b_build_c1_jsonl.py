#!/usr/bin/env python3
"""E6-C1: build the SWE-bench-verified-style JSONL EarlyEval's gold-answer enrichment
parses, for the C1 instance shard, from two public metadata sources (SWE-rebench and
SWE-bench). Instances present in neither are simply absent; the released code then
records gold_has_answer = 0 and __MISSING__ categoricals for them.

Coverage is reported and belongs in the paper: gold features are TASK-CONSTANT, so
partial coverage can inflate pooled discrimination but cannot contribute any
within-task signal, which is the estimand under test.
"""
import argparse, glob, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--meta-globs", nargs="+", required=True)
ap.add_argument("--instances", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

keep = set(l.strip() for l in open(a.instances) if l.strip())
frames = []
for g in a.meta_globs:
    for p in sorted(glob.glob(g)):
        frames.append(pd.read_parquet(p))
meta = pd.concat(frames, ignore_index=True)
meta = meta[meta.instance_id.isin(keep)].drop_duplicates("instance_id", keep="first")


def tolist(v):
    if isinstance(v, (list, np.ndarray)):
        return [str(x) for x in v]
    if isinstance(v, str):
        try:
            return [str(x) for x in json.loads(v)]
        except Exception:
            return [v] if v else []
    return []


def get(r, k):
    v = getattr(r, k, None)
    return "" if v is None else str(v)


n = 0
with open(a.out, "w") as f:
    for r in meta.itertuples(index=False):
        f.write(json.dumps({
            "instance_id": str(r.instance_id), "repo": get(r, "repo"),
            "version": get(r, "version"), "patch": get(r, "patch"),
            "test_patch": get(r, "test_patch"),
            "problem_statement": get(r, "problem_statement"),
            "hints_text": get(r, "hints_text"),
            "FAIL_TO_PASS": tolist(getattr(r, "FAIL_TO_PASS", None)),
            "PASS_TO_PASS": tolist(getattr(r, "PASS_TO_PASS", None)),
        }, ensure_ascii=False) + "\n")
        n += 1
print("[done] %d of %d shard instances covered (%.1f%%) -> %s"
      % (n, len(keep), 100.0 * n / max(len(keep), 1), a.out), flush=True)
