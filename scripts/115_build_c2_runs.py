#!/usr/bin/env python3
"""Build a compact runs table for the C2 deployment replay.

C2 records outcomes and trajectories but NOT per-run cost, which the replay needs in
order to charge a budget. We estimate it from trajectory text: tokens ~ characters / 4,
a standard approximation for code and terminal output, and apportion the total across
steps in proportion to per-message length -- the same apportionment the C4 harness
already uses for its per-step costs. Absolute scale is irrelevant to every comparison
here; only relative cost across runs matters, and character length tracks that closely.

Output columns: task, run, success, tokens, cum (per-assistant-step cumulative tokens).
"""
import argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--traj", default="data/swe_rebench/trajectories.parquet")
ap.add_argument("--tasks", default=None, help="optional file of task ids to keep")
ap.add_argument("--chars-per-token", type=float, default=4.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

keep = None
if a.tasks:
    keep = set(l.strip() for l in open(a.tasks) if l.strip())

df = pd.read_parquet(a.traj, columns=["instance_id", "resolved", "trajectory"])
if keep is not None:
    df = df[df.instance_id.isin(keep)]
print("[src] %d trajectories over %d tasks" % (len(df), df.instance_id.nunique()),
      flush=True)


def msg_len(m):
    n = 0
    for k in ("content", "text"):
        v = m.get(k)
        if isinstance(v, str):
            n += len(v)
    tcs = m.get("tool_calls")
    if tcs is not None:
        try:
            for tc in tcs:
                fn = (tc or {}).get("function") or {}
                args = fn.get("arguments")
                n += len(args) if isinstance(args, str) else len(str(args or ""))
        except TypeError:
            pass
    return n


rows = []
counter = {}
for iid, res, traj in df.itertuples(index=False):
    try:
        msgs = json.loads(traj) if isinstance(traj, str) else list(traj)
    except Exception:
        continue
    lens, step_lens, acc = [], [], 0
    for m in msgs:
        L = msg_len(m)
        lens.append(L)
        acc += L
        if (m.get("role") or "") == "assistant":
            step_lens.append(acc)     # cumulative chars through this assistant step
    total = max(sum(lens), 1)
    tok = total / a.chars_per_token
    cum = [float(s / a.chars_per_token) for s in step_lens] or [tok]
    r = counter.get(iid, 0)
    counter[iid] = r + 1
    rows.append(dict(task=str(iid), run=r, success=int(bool(res)),
                     tokens=float(tok), cum=cum))

out = pd.DataFrame(rows)
out.to_parquet(a.out, index=False)
print("[done] %d runs, %d tasks | median cost %.0f tok, median steps %.0f -> %s"
      % (len(out), out.task.nunique(), out.tokens.median(),
         np.median([len(c) for c in out["cum"]]), a.out), flush=True)
