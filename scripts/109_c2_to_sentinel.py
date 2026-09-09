#!/usr/bin/env python3
"""E7 adapter: convert C2 (SWE-rebench/OpenHands) trajectories, restricted to the E6
fold-1 instance shard, into agent-trajectory-sentinel's step-record format
(derail/telemetry/adapter.py, validate_steps / episode_from_trace).

Per assistant turn one step dict:
  text  = assistant content, with each tool call+result appended in the repo's v2
          textual event format "[name({args}) -> result]" (content preserved verbatim)
  error = True if the folded tool results match common error signatures
  logprobs_available = False   (C2 carries no logprobs; their native missing path)
  task  = the first user message (issue text), on step 0 only
Latency, output_tokens and token_logprobs are OMITTED, never fabricated.
Output JSONL: {"episode_id", "instance_id", "resolved", "steps": [...]}.
"""
import argparse, json
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--traj", default="data/swe_rebench/trajectories.parquet")
ap.add_argument("--instances", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

ERR = ("error", "traceback", "exception", "failed", "not found", "no such file")

keep = set(l.strip() for l in open(a.instances) if l.strip())
df = pd.read_parquet(a.traj, columns=["trajectory_id", "instance_id", "trajectory",
                                      "resolved"])
df = df[df.instance_id.isin(keep)]
print(f"[src] {len(df)} trajectories over {df.instance_id.nunique()} instances", flush=True)

def txt(v):
    return v if isinstance(v, str) else ("" if v is None else str(v))

def steps_of(traj):
    msgs = json.loads(traj) if isinstance(traj, str) else list(traj)
    task = next((txt(m.get("content")) for m in msgs if (m.get("role") == "user")), "")
    steps, pending = [], None
    for m in msgs:
        role = m.get("role") or ""
        if role == "assistant":
            if pending is not None:
                steps.append(pending)
            content = txt(m.get("content"))
            calls = []
            tcs = m.get("tool_calls")
            if tcs is not None and not isinstance(tcs, list):
                try: tcs = list(tcs)
                except TypeError: tcs = None
            for tc in (tcs or []):
                tc = tc if isinstance(tc, dict) else {}
                fn = tc.get("function") or {}
                if not isinstance(fn, dict): fn = {}
                args = fn.get("arguments")
                if isinstance(args, dict): args = json.dumps(args, ensure_ascii=False)
                calls.append((txt(fn.get("name")) or "unknown", txt(args)))
            pending = {"text": content, "calls": calls, "results": []}
        elif role == "tool" and pending is not None:
            pending["results"].append(txt(m.get("content")))
    if pending is not None:
        steps.append(pending)
    out = []
    for i, s in enumerate(steps):
        text = s["text"]
        for j, (name, args) in enumerate(s["calls"]):
            res = s["results"][j] if j < len(s["results"]) else ""
            text += f"\n[{name}({args}) -> {res}]"
        joined = " ".join(s["results"]).lower()
        d = {"text": text,
             "error": any(p in joined for p in ERR),
             "logprobs_available": False}
        if i == 0:
            d["task"] = task
        out.append(d)
    return out

n = 0
with open(a.out, "w") as f:
    for tid, iid, traj, res in df.itertuples(index=False):
        try:
            st = steps_of(traj)
        except Exception:
            continue
        if not st:
            continue
        f.write(json.dumps({"episode_id": str(tid), "instance_id": str(iid),
                            "resolved": bool(res), "steps": st},
                           ensure_ascii=False) + "\n")
        n += 1
print(f"[done] {n} episodes -> {a.out}", flush=True)
