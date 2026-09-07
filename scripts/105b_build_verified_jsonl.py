#!/usr/bin/env python3
"""E6: build the SWE-bench-verified-style JSONL that EarlyEval's released enrichment
(`answer_features.py`) parses, from the PUBLIC nebius/SWE-rebench instance metadata
(HF datasets-server parquet export), restricted to the instances present in the C2
trajectory corpus. Fields map 1:1 to the keys the released code reads:
instance_id, repo, version, patch, test_patch, problem_statement, hints_text,
FAIL_TO_PASS, PASS_TO_PASS. `difficulty` is absent upstream -> "__MISSING__" constant.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--meta-dir", default="e6/rebench_meta",
                help="dir of nebius/SWE-rebench parquet shards (HF parquet export)")
ap.add_argument("--traj", default="data/swe_rebench/trajectories.parquet")
ap.add_argument("--out", default="e6/verified_c2.jsonl")
a = ap.parse_args()

meta = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(a.meta_dir, "*.parquet")))],
                 ignore_index=True)
ours = set(pd.read_parquet(a.traj, columns=["instance_id"]).instance_id.unique())
meta = meta[meta.instance_id.isin(ours)].drop_duplicates("instance_id", keep="first")
missing = ours - set(meta.instance_id)
print(f"[cov] {len(ours)} corpus instances | {len(meta)} matched | {len(missing)} missing")

def tolist(v):
    if isinstance(v, (list, np.ndarray)): return [str(x) for x in v]
    if isinstance(v, str):
        try: return [str(x) for x in json.loads(v)]
        except Exception: return [v] if v else []
    return []

with open(a.out, "w") as f:
    for r in meta.itertuples(index=False):
        f.write(json.dumps({
            "instance_id": str(r.instance_id), "repo": str(r.repo or ""),
            "version": str(r.version or ""), "patch": str(r.patch or ""),
            "test_patch": str(r.test_patch or ""),
            "problem_statement": str(r.problem_statement or ""),
            "hints_text": str(r.hints_text or ""),
            "FAIL_TO_PASS": tolist(r.FAIL_TO_PASS), "PASS_TO_PASS": tolist(r.PASS_TO_PASS),
        }, ensure_ascii=False) + "\n")
print(f"[done] {len(meta)} instances -> {a.out}")
