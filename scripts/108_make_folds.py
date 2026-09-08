#!/usr/bin/env python3
"""E6 amendment 3: partition the instances NOT in fold 1 (the registered seed-42 20%
shard) into folds 2-5 by a seeded shuffle (seed 43) over the sorted remainder, split
into four near-equal contiguous chunks. Writes fold2..fold5 instance lists.
"""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--traj", default="data/swe_rebench/trajectories.parquet")
ap.add_argument("--fold1", required=True)
ap.add_argument("--out-prefix", required=True, help="writes <prefix>fold{2..5}.txt")
ap.add_argument("--seed", type=int, default=43)
a = ap.parse_args()

allids = sorted(pd.read_parquet(a.traj, columns=["instance_id"]).instance_id.unique())
f1 = set(l.strip() for l in open(a.fold1) if l.strip())
rest = sorted(i for i in allids if i not in f1)
print(f"[universe] {len(allids)} | fold1 {len(f1)} | remainder {len(rest)}")

rng = np.random.default_rng(a.seed)
perm = list(rest); rng.shuffle(perm)
chunks = np.array_split(perm, 4)
for i, ch in enumerate(chunks, start=2):
    path = f"{a.out_prefix}fold{i}.txt"
    with open(path, "w") as f:
        f.write("\n".join(sorted(ch)) + "\n")
    print(f"[fold{i}] {len(ch)} instances -> {path}")
