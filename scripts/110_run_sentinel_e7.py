#!/usr/bin/env python3
"""E7 driver: run agent-trajectory-sentinel's RELEASED monitors, unmodified, on the
converted C2 fold-1 shard, following the repo's own external-corpus study
(derail/experiments/run_atbench_study.py) step for step:

  Standardizer().fit(train healthy) -> make_hybrids(...) -> monitor.fit(train healthy)
  -> monitor.score_episode(test episode) per step.

Fit pool: healthy (resolved) episodes of fold-1 instances NOT in the E6 test set;
scored: every episode of the E6 test instances. hybrid_logistic is skipped for the
repo's own stated leak reason. Channels default to the repo's no-logprob precedent
("e","m","x"); --goal adds the text-derivable grounding+goal channels ("e","m","x",
"g","c") as a steelman config.

Output: a wide table in the E6 analysis format (106_earlyeval_within.py): one row per
(trajectory, step) with prob__<monitor> columns, label (1=failure), split="test".
Run from the sentinel repo root so `derail` imports resolve.
"""
import argparse, json
import numpy as np, pandas as pd

from derail.common import Standardizer
from derail.monitor.hybrid import make_hybrids
from derail.telemetry.adapter import episode_from_trace

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", required=True)
ap.add_argument("--test-instances", required=True)
ap.add_argument("--goal", action="store_true",
                help="use grounding+goal channels (telemetry v5) instead of e+m+x")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

test_ids = set(l.strip() for l in open(a.test_instances) if l.strip())
kw = dict(extended=True, grounding=a.goal, goal=a.goal)
channels = ("e", "m", "x", "g", "c") if a.goal else ("e", "m", "x")

train_eps, test_rows = [], []
n_in, n_bad = 0, 0
with open(a.episodes) as f:
    for line in f:
        r = json.loads(line)
        n_in += 1
        try:
            ep = episode_from_trace(r["steps"], r["episode_id"], **kw)
        except Exception:
            n_bad += 1
            continue
        if r["instance_id"] in test_ids:
            test_rows.append((r, ep))
        elif r["resolved"]:
            train_eps.append(ep)
print(f"[data] {n_in} episodes ({n_bad} failed conversion) | "
      f"fit pool (healthy, non-test): {len(train_eps)} | "
      f"test episodes: {len(test_rows)} over "
      f"{len(set(r['instance_id'] for r, _ in test_rows))} instances", flush=True)

std = Standardizer().fit(train_eps)
esn, maha, hybrids = make_hybrids(std, channels=channels, seed=1300 + a.seed)
monitors = [m for m in (esn, maha, *hybrids) if m.name != "hybrid_logistic"]
for m in monitors:
    m.fit(train_eps)
    print(f"[fit] {m.name}", flush=True)

recs = []
for r, ep in test_rows:
    scores = {}
    for m in monitors:
        s = np.asarray(m.score_episode(ep), dtype=float)
        scores[f"prob__{m.name}"] = s
    T = len(next(iter(scores.values())))
    for t in range(T):
        row = dict(traj_id=r["episode_id"], instance_id=r["instance_id"],
                   prefix_step_idx=t, label=int(not r["resolved"]), split="test")
        for k, v in scores.items():
            row[k] = float(v[t]) if t < len(v) else float(v[-1])
        recs.append(row)
df = pd.DataFrame(recs)
df.to_parquet(a.out, index=False)
print(f"[done] {len(df)} step rows, monitors: {[m.name for m in monitors]} -> {a.out}",
      flush=True)
