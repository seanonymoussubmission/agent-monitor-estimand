#!/usr/bin/env python3
"""E6 adapter: convert the public SWE-rebench/OpenHands trajectory parquet (C2) into
the raw `tool-*.parquet` format consumed by EarlyEval's released pipeline
(github.com/inphotoo/earlyeval, MIT), whose step builder recognises assistant messages
with message_type == "action" plus an `action` field, and reads role=="tool" outputs.

Mapping (mechanical, content-preserving):
  system/user messages        -> passed through as {role, content}
  assistant with tool_calls   -> {role: assistant, message_type: "action",
                                  action: "<fn> <arguments>", thought: content,
                                  content: content, tool_calls: [...]}
  assistant without tool_calls-> {role: assistant, content}
  tool messages               -> {role: tool, content}
Output columns: traj_id, instance_id, model, model_id, resolved, messages (JSON string).
"""
import argparse, json, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/swe_rebench/trajectories.parquet")
ap.add_argument("--out", default="e6/toolpq")
ap.add_argument("--limit", type=int, default=0, help="0 = all trajectories")
ap.add_argument("--shard-size", type=int, default=8000)
ap.add_argument("--model-name", default="qwen3-coder-480b-a35b")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
df = pd.read_parquet(a.src, columns=["trajectory_id", "instance_id", "trajectory", "resolved"])
if a.limit:
    df = df.head(a.limit)
print(f"[src] {len(df)} trajectories", flush=True)

def get(m, k):
    v = m.get(k)
    return v if isinstance(v, str) else ("" if v is None else str(v))

def conv_msgs(traj):
    msgs = json.loads(traj) if isinstance(traj, str) else list(traj)
    out = []
    for m in msgs:
        role = m.get("role") or ""
        content = get(m, "content")
        tcs = m.get("tool_calls")
        if tcs is not None and not isinstance(tcs, list):
            try: tcs = list(tcs)
            except TypeError: tcs = None
        if role == "assistant" and tcs:
            names, parts, clean_tcs = [], [], []
            for tc in tcs:
                tc = tc if isinstance(tc, dict) else {}
                fn = tc.get("function") or {}
                if not isinstance(fn, dict): fn = {}
                name = get(fn, "name") or "unknown"
                args = fn.get("arguments")
                if isinstance(args, dict): args = json.dumps(args, ensure_ascii=False)
                args = args if isinstance(args, str) else ("" if args is None else str(args))
                names.append(name); parts.append(args)
                clean_tcs.append({"function": {"name": name, "arguments": args}})
            action = "\n".join(f"{n} {p}".strip() for n, p in zip(names, parts))
            out.append({"role": "assistant", "message_type": "action",
                        "action": action, "thought": content, "content": content,
                        "tool_calls": clean_tcs})
        else:
            out.append({"role": role, "content": content})
    return json.dumps(out, ensure_ascii=False)

rows, shard, wrote = [], 0, 0
for tid, iid, traj, res in df.itertuples(index=False):
    try:
        mj = conv_msgs(traj)
    except Exception:
        continue
    rows.append(dict(traj_id=str(tid), instance_id=str(iid), model=a.model_name,
                     model_id=a.model_name, resolved=bool(res), messages=mj))
    if len(rows) >= a.shard_size:
        pd.DataFrame(rows).to_parquet(f"{a.out}/tool-{shard:03d}.parquet", index=False)
        wrote += len(rows); print(f"[shard {shard}] {len(rows)} rows", flush=True)
        rows, shard = [], shard + 1
if rows:
    pd.DataFrame(rows).to_parquet(f"{a.out}/tool-{shard:03d}.parquet", index=False)
    wrote += len(rows); print(f"[shard {shard}] {len(rows)} rows", flush=True)
print(f"[done] {wrote} trajectories -> {a.out}", flush=True)
