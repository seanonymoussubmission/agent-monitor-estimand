#!/usr/bin/env python3
"""Re-evaluate the Automata failure predictor under pooled and within-task AUROC.

Automata (arXiv 2608.23670) reports held-out failure AUROC up to 0.941 and an online
monitor at fractional trajectory checkpoints. We reimplement its method from the paper
-- we do not have their score files, and their evaluation corpora mostly lack the
repeated runs the within-task estimand needs -- and run it on C1, which has 80k
trajectories over 3.6k tasks with repeated runs per (model, task).

Method, following the paper:
  1. abstract each step to an *activity* (the command verb the agent issues)
  2. build the FSM by Last-Activity Merge: the congruence q ~ q' iff the incoming
     activity matches, giving |A|+1 states
  3. rare-transition filtering: drop transitions seen exactly once unless a state's
     only continuation
  4. per-state behavioural features + cross-entropy anomaly features
  5. gradient boosting, 200 trees, depth 3

The transition model is fit on TRAINING traces only; test traces are scored under it.
Fitting it on all traces would leak the test outcomes into the surprise features.
"""
import argparse, glob, json, math, os, re, sys, collections
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/swe_agent_traj/data")
ap.add_argument("--shards", type=int, default=12)
ap.add_argument("--prefix-turns", type=int, default=0,
                help="0 = full trajectory; k>0 = first k assistant turns")
ap.add_argument("--prefix-frac", type=float, default=0.0,
                help=">0 = first fraction of the trajectory (Automata's own protocol)")
ap.add_argument("--rare-min", type=int, default=2)
ap.add_argument("--out", default=None)
a = ap.parse_args()

CMD = re.compile(r"```(?:bash|sh)?\s*\n?\s*([A-Za-z_][\w\-.]*)", re.M)
# SWE-agent's own command vocabulary. Automata abstracts a step to a tool/command
# token via "dataset-specific rules"; for this scaffold that is the interface command.
# Everything else is bucketed, which keeps the alphabet near the 7-43 states the paper
# reports rather than exploding into one state per shell binary.
SWE_CMDS = {
    "open", "goto", "scroll_down", "scroll_up", "create", "edit", "insert", "append",
    "search_dir", "search_file", "find_file", "submit", "python", "python3", "pytest",
    "ls", "cd", "cat", "grep", "find", "git", "pip", "echo", "rm", "mv", "cp", "mkdir",
    "touch", "chmod", "which", "head", "tail", "sed", "awk", "diff", "make", "bash",
}
def abstract(tok):
    t = (tok or "").lower()
    return t if t in SWE_CMDS else ("no_command" if t == "no_command" else "other_cmd")
ERR = re.compile(r"traceback|error|not found|no such file|command not found|failed|"
                 r"exception|invalid|denied", re.I)

def activities(traj):
    """(activity sequence, per-step observation length, per-step error flag)."""
    acts, obs_len, errs = [], [], []
    pending = None
    for st in traj:
        role = st.get("role"); txt = st.get("text") or ""
        if role == "ai":
            m = CMD.search(txt)
            pending = abstract(m.group(1)) if m else "no_command"
            acts.append(pending); obs_len.append(len(txt)); errs.append(0)
        elif role == "user" and acts:
            obs_len[-1] += len(txt)
            errs[-1] = int(bool(ERR.search(txt)))
    return acts, obs_len, errs

def cut(acts, ol, er):
    if a.prefix_frac > 0:
        n = max(1, int(round(len(acts) * a.prefix_frac)))
    elif a.prefix_turns > 0:
        n = a.prefix_turns
    else:
        return acts, ol, er
    return acts[:n], ol[:n], er[:n]

# ---------------------------------------------------------------- load
rows = []
for s in range(a.shards):
    p = os.path.join(a.data, f"train-{s:05d}-of-00012.parquet")
    if not os.path.exists(p): continue
    df = pd.read_parquet(p, columns=["instance_id", "model_name", "target", "trajectory"])
    for iid, mdl, tgt, tr in df.itertuples(index=False):
        try: acts, ol, er = activities(list(tr))
        except Exception: continue
        if len(acts) < 2: continue
        rows.append((f"{mdl}||{iid}", iid, int(not bool(tgt)), acts, ol, er))
    del df
    print(f"[load] shard {s}: {len(rows)} traces", flush=True)

unit  = np.array([r[0] for r in rows])
group = np.array([r[1] for r in rows])
y     = np.array([r[2] for r in rows])
seqs  = [r[3] for r in rows]; olens = [r[4] for r in rows]; errs = [r[5] for r in rows]
print(f"[data] {len(y)} traces | {len(set(unit))} units | {len(set(group))} tasks | "
      f"fail rate {y.mean():.3f}", flush=True)

# ---------------------------------------------------------------- FSM + features
def fit_fsm(idx):
    """Last-Activity Merge FSM with rare-transition filtering, from TRAIN traces only."""
    trans = collections.defaultdict(collections.Counter)
    for i in idx:
        s, seq = "<init>", cut(seqs[i], olens[i], errs[i])[0]
        for act in seq:
            trans[s][act] += 1; s = act
    kept = {}
    for s, c in trans.items():
        keep = {act: n for act, n in c.items() if n >= a.rare_min or len(c) == 1}
        if not keep: keep = dict(c)
        tot = sum(keep.values())
        kept[s] = {act: n / tot for act, n in keep.items()}
    alpha = set(); [alpha.update(c) for c in trans.values()]
    return kept, sorted(alpha)

def featurise(i, fsm, alpha, aidx):
    seq, ol, er = cut(seqs[i], olens[i], errs[i])
    n = len(seq)
    # per-state behavioural features
    visits = np.zeros(len(alpha))
    for act in seq:
        j = aidx.get(act)
        if j is not None: visits[j] += 1
    visits /= max(n, 1)
    # cross-entropy anomaly features under the fitted transition model
    FLOOR = 1e-4
    sur, s = [], "<init>"
    for act in seq:
        p = fsm.get(s, {}).get(act, FLOOR)
        sur.append(-math.log(max(p, FLOOR))); s = act
    sur = np.array(sur) if sur else np.array([0.0])
    h = max(1, len(sur) // 2)
    ce, mx = float(sur.mean()), float(sur.max())
    drift = float(abs(sur[:h].mean() - sur[h:].mean())) if len(sur) > 1 else 0.0
    minp = float(np.exp(-mx))
    hi = float((sur > 3.0).mean())
    uniq = len(set(seq))
    rep = 1.0 - uniq / max(n, 1)                     # cycle / repetition rate
    ol = np.array(ol, float) if ol else np.array([0.0])
    er = np.array(er, float) if er else np.array([0.0])
    return np.concatenate([visits, [
        ce, mx, drift, minp, hi,                      # CE anomaly block
        n, uniq, rep,                                 # structural
        ol.mean(), ol.max(), ol.std(),                # message length
        er.mean(), er.sum(),                          # error rate
        float(np.mean(np.diff(sur))) if len(sur) > 1 else 0.0,  # temporal
    ]])

# ---------------------------------------------------------------- cross-fitted eval
oof = np.zeros(len(y))
for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(y)), y, group):
    fsm, alpha = fit_fsm(tr)
    aidx = {act: j for j, act in enumerate(alpha)}
    Xtr = np.vstack([featurise(i, fsm, alpha, aidx) for i in tr])
    Xte = np.vstack([featurise(i, fsm, alpha, aidx) for i in te])
    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, random_state=0)
    clf.fit(Xtr, y[tr]); oof[te] = clf.predict_proba(Xte)[:, 1]
    print(f"[fold] {len(tr)} train / {len(te)} test | {len(alpha)} activities, "
          f"{len(fsm)} states", flush=True)

def within(u, yy, s):
    num = den = 0.0; nu = 0
    for g in set(u):
        m = u == g; P, N = s[m & (yy == 1)], s[m & (yy == 0)]
        if not len(P) or not len(N): continue
        nu += 1
        num += sum((p > q) + 0.5 * (p == q) for p in P for q in N); den += len(P) * len(N)
    return (num / den if den else float("nan")), den, nu

pool = roc_auc_score(y, oof)
w, npair, nun = within(unit, y, oof)
tag = (f"frac={a.prefix_frac}" if a.prefix_frac > 0 else
       f"turns={a.prefix_turns}" if a.prefix_turns > 0 else "full")
print(f"\n=== AUTOMATA REIMPLEMENTATION on C1 ({tag}) ===")
print(f"  pooled  AUROC = {pool:.4f}")
print(f"  within  AUROC = {w:.4f}   ({npair} within-unit pairs, {nun} mixed units)")
print(f"  gap           = {pool - w:+.4f}")
if a.out:
    json.dump(dict(setting=tag, pooled=pool, within=w, gap=pool - w,
                   pairs=int(npair), units=nun, n=len(y)), open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}")
