#!/usr/bin/env python3
"""E6 amendment 2: constant-memory stream-filter of the released pipeline's own on-disk
prefix parts down to a registered random instance shard, written to the exact path the
released code reads back with --skip-prefix-table. No row is edited or reordered; the
result is byte-identical to what the released concat would produce for these instances.
"""
import argparse, glob, os
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--parts-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--instances-out", required=True)
ap.add_argument("--frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--instances-file", default=None,
                help="pre-registered instance list; skips sampling")
a = ap.parse_args()

parts = sorted(glob.glob(os.path.join(a.parts_dir, "prefix_table.part-*.parquet")))
print(f"[parts] {len(parts)}", flush=True)

if a.instances_file:
    keep = sorted(l.strip() for l in open(a.instances_file) if l.strip())
    print(f"[list] {len(keep)} instances from {a.instances_file}", flush=True)
else:
    ids = set()
    for p in parts:
        ids.update(pq.read_table(p, columns=["instance_id"])["instance_id"].to_pylist())
    ids = sorted(ids)
    rng = np.random.default_rng(a.seed)
    keep = sorted(rng.choice(ids, size=round(a.frac * len(ids)), replace=False))
    print(f"[sample] {len(keep)} of {len(ids)} instances (frac={a.frac}, seed={a.seed})",
          flush=True)
with open(a.instances_out, "w") as f:
    f.write("\n".join(keep) + "\n")

keep_set = set(keep)
writer, rows, trajs = None, 0, set()
for i, p in enumerate(parts):
    t = pq.read_table(p)
    mask = pc.is_in(t["instance_id"], value_set=__import__("pyarrow").array(list(keep_set)))
    t = t.filter(mask)
    if writer is None:
        writer = pq.ParquetWriter(a.out, t.schema)
    if t.num_rows:
        writer.write_table(t)
        rows += t.num_rows
        trajs.update(t["traj_id"].to_pylist())
    print(f"  part {i:03d}: kept {t.num_rows} (total {rows})", flush=True)
writer.close()
print(f"[done] {rows} prefix rows | {len(trajs)} trajectories -> {a.out}", flush=True)
