#!/usr/bin/env python3
"""M-r (pre-registered amendment): standardized mixed-outcome rates at a common repeat
count, exact hypergeometric (no Monte Carlo).

For a unit with n runs of which s fail, the probability that an r-run subsample (without
replacement) contains both outcomes is 1 - C(s,r)/C(n,r) - C(n-s,r)/C(n,r).
We report the mean over units with n >= r, per corpus, at r = 3 and r = 5.
"""
import collections, csv, glob, os
import numpy as np, pandas as pd
from math import comb

def std_mixed(counts, r):
    vals = []
    for n, s in counts:
        if n < r: continue
        denom = comb(n, r)
        vals.append(1.0 - comb(s, r) / denom - comb(n - s, r) / denom)
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)

corpora = {}

# C1: SWE-agent parquet shards
cnt = collections.Counter(); fail = collections.Counter()
for s in range(12):
    p = f"data/swe_agent_traj/data/train-{s:05d}-of-00012.parquet"
    if not os.path.exists(p): continue
    df = pd.read_parquet(p, columns=["instance_id", "model_name", "target"])
    for iid, mdl, tgt in df.itertuples(index=False):
        u = f"{mdl}||{iid}"; cnt[u] += 1; fail[u] += int(not bool(tgt))
corpora["C1"] = [(cnt[u], fail[u]) for u in cnt]

# C2
df = pd.read_parquet("data/swe_rebench/trajectories.parquet",
                     columns=["instance_id", "resolved"])
g = df.assign(fail=1 - df.resolved.astype(int)).groupby("instance_id")["fail"]
corpora["C2"] = list(zip(g.size(), g.sum()))

# C3
df = pd.read_parquet("data/liveclaw/data/v0.2.1-00000-of-00001.parquet",
                     columns=["model_name", "case_id", "score"])
df["fail"] = (df.score != 1.0).astype(int)
g = df.groupby(["model_name", "case_id"])["fail"]
corpora["C3"] = list(zip(g.size(), g.sum()))

# C4-L / C4-Q
for name, path in [("C4-L", "data/runs_laguna.csv"), ("C4-Q", "data/runs_qwen.csv")]:
    c = collections.Counter(); f = collections.Counter()
    for row in csv.DictReader(open(path)):
        c[row["task"]] += 1; f[row["task"]] += 1 - int(row["success"])
    corpora[name] = [(c[u], f[u]) for u in c]

print(f"{'corpus':6s} {'units':>6s} {'raw mixed':>10s} {'std@r=3':>9s} {'(units)':>8s} "
      f"{'std@r=5':>9s} {'(units)':>8s}")
for name, counts in corpora.items():
    raw = np.mean([1.0 if 0 < s < n else 0.0 for n, s in counts])
    m3, n3 = std_mixed(counts, 3); m5, n5 = std_mixed(counts, 5)
    print(f"{name:6s} {len(counts):6d} {raw:10.3f} {m3:9.3f} {n3:8d} "
          f"{m5:9.3f} {n5:8d}")
