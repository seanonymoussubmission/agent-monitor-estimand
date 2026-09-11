#!/usr/bin/env python3
"""E6-C1 adapter: convert the public SWE-agent trajectory corpus (C1) into the raw
`tool-*.parquet` format EarlyEval's released pipeline consumes.

C1 messages carry {role in (system,user,ai), text}. A SWE-agent assistant turn is
reasoning prose followed by a fenced command block; the following user turn is that
command's observation. The mapping is mechanical and content-preserving:

  ai turn    -> {role: assistant, message_type: "action",
                 action: "<last fenced block>", thought: "<prose before first fence>",
                 content: <full text>, tool_calls: [{function:{name,arguments}}]}
  FIRST user turn -> {role: user, content: text}   (the issue statement / task prompt)
  later user turns -> {role: tool, content: text}  (command observations)
  system     -> {role: system, content: text}

The first/later split matters: in SWE-agent the opening user turn is the issue text,
not an observation, and the released step builder reads the task prompt from a user
turn. Mapping every user turn to `tool` leaves task_prompt_text empty and the
pipeline's first TF-IDF vectorizer fails with an empty vocabulary.

instance_id keeps the real SWE-bench id so the pipeline's gold-answer lookup and its
instance-holdout split work; the acting model is carried in `model`/`model_id`, and the
within-task UNIT is (instance_id, model) -- pass --unit-cols instance_id model to the
analysis script, since C1 has three acting models.
"""
import argparse, glob, json, os, re
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src-glob", default="data/swe_agent_traj/data/train-*.parquet")
ap.add_argument("--out", required=True)
ap.add_argument("--instances", default=None,
                help="file of instance_ids to keep (one per line)")
ap.add_argument("--shard-size", type=int, default=8000)
a = ap.parse_args()

FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.S)
os.makedirs(a.out, exist_ok=True)

keep = None
if a.instances:
    keep = set(l.strip() for l in open(a.instances) if l.strip())

def txt(v):
    return v if isinstance(v, str) else ("" if v is None else str(v))

def conv(traj):
    msgs = json.loads(traj) if isinstance(traj, str) else list(traj)
    out = []
    seen_user = False
    for m in msgs:
        role = m.get("role") or ""
        body = txt(m.get("text"))
        if role == "ai":
            blocks = FENCE.findall(body)
            if blocks:
                action = blocks[-1].strip()
                thought = body[:body.find("```")].strip()
                head = action.split(None, 1)
                name = head[0] if head else "unknown"
                args = head[1] if len(head) > 1 else ""
                out.append({"role": "assistant", "message_type": "action",
                            "action": action, "thought": thought, "content": body,
                            "tool_calls": [{"function": {"name": name,
                                                         "arguments": args}}]})
            else:
                out.append({"role": "assistant", "content": body})
        elif role == "user":
            if not seen_user:
                seen_user = True
                out.append({"role": "user", "content": body})
            else:
                out.append({"role": "tool", "content": body})
        else:
            out.append({"role": "system", "content": body})
    return out

rows, shard, wrote, skipped = [], 0, 0, 0
for p in sorted(glob.glob(a.src_glob)):
    df = pd.read_parquet(p, columns=["instance_id", "model_name", "target", "trajectory"])
    if keep is not None:
        df = df[df.instance_id.isin(keep)]
    for i, (iid, mdl, tgt, traj) in enumerate(df.itertuples(index=False)):
        try:
            msgs = conv(traj)
        except Exception:
            skipped += 1
            continue
        if not any(m.get("message_type") == "action" for m in msgs):
            skipped += 1
            continue
        rows.append(dict(traj_id="%s::%s::%s::%d" % (mdl, iid, os.path.basename(p), i),
                         instance_id=str(iid), model=str(mdl), model_id=str(mdl),
                         resolved=bool(tgt),
                         messages=json.dumps(msgs, ensure_ascii=False)))
        if len(rows) >= a.shard_size:
            pd.DataFrame(rows).to_parquet("%s/tool-%03d.parquet" % (a.out, shard),
                                          index=False)
            wrote += len(rows)
            print("[shard %d] %d rows (total %d)" % (shard, len(rows), wrote), flush=True)
            rows, shard = [], shard + 1
if rows:
    pd.DataFrame(rows).to_parquet("%s/tool-%03d.parquet" % (a.out, shard), index=False)
    wrote += len(rows)
    print("[shard %d] %d rows (total %d)" % (shard, len(rows), wrote), flush=True)
print("[done] %d trajectories -> %s (%d skipped, no parseable action)"
      % (wrote, a.out, skipped), flush=True)
