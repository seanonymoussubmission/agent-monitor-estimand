#!/usr/bin/env python3
"""E6: build the SWE-bench-verified-style JSONL EarlyEval's gold-answer enrichment
parses, from the public nebius/SWE-rebench instance metadata (HF parquet export),
restricted to the instances present in the C2 trajectory corpus. Field names match
answer_features._make_answer_rows exactly; `difficulty` is absent upstream and is
left out (the code substitutes "__MISSING__").
"""
import argparse, glob, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--meta-glob", default="e6/rebench_meta/*.parquet")
ap.add_argument("--traj", default="data/swe_rebench/trajectories.parquet")
ap.add_argument("--out", default="e6/verified_c2.jsonl")
a = ap.parse_args()

meta = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(a.meta_glob))],
                 ignore_index=True)
ours = set(pd.read_parquet(a.traj, columns=["instance_id"]).instance_id.unique())
meta = meta[meta.instance_id.isin(ours)].drop_duplicates("instance_id", keep="first")

def tolist(v):
    if isinstance(v, (list, np.ndarray)):
        return [str(x) for x in v]
    if isinstance(v, str):
        try:
            return [str(x) for x in json.loads(v)]
        except Exception:
            return [v] if v else []
    return []

n = 0
with open(a.out, "w") as f:
    for r in meta.itertuples(index=False):
        f.write(json.dumps({
            "instance_id": str(r.instance_id), "repo": str(r.repo or ""),
            "version": str(r.version or ""), "patch": str(r.patch or ""),
            "test_patch": str(r.test_patch or ""),
            "problem_statement": str(r.problem_statement or ""),
            "hints_text": str(r.hints_text or ""),
            "FAIL_TO_PASS": tolist(r.FAIL_TO_PASS),
            "PASS_TO_PASS": tolist(r.PASS_TO_PASS),
        }, ensure_ascii=False) + "\n")
        n += 1
print(f"[done] {n} instances -> {a.out}")
