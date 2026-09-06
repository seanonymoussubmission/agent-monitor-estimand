#!/usr/bin/env python3
"""At equal budget, is it better to abort doomed runs or to reallocate the next attempt?

The paper argues for moving from "which run should I abort?" to "where should I spend the
next attempt?", but has not put the two policies side by side under one budget. Reviewers
will reach for the obvious baseline -- cap the turns, or abort on a monitor's verdict -- so
we implement those and compare directly.

Replay semantics: every pull draws a real recorded run of that task. Aborting at turn k
costs only the tokens actually consumed through turn k and returns no success, which is the
favourable accounting for aborting -- a real abort might also forgo a late success, and we
charge nothing for that beyond losing the run.

Policies, all at the same token budget:
  uniform            round-robin over unsolved tasks, run to completion
  turn-cap N         round-robin, every run truncated at N turns
  abort@k            round-robin, prefix scored at turn k, abort the worst-scoring share
  allocate           budgeted Thompson sampling over tasks, runs go to completion
  hybrid             allocate, and additionally abort at turn k

Prefix scores come from the same cross-fitted automaton predictor used elsewhere, so the
abort policies are given a monitor of exactly the quality this paper measures.
"""
import argparse, collections, csv, glob, json, math, os, re
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--traj", default="data/lph/swebench/laguna_xs2_full")
ap.add_argument("--runs", default="data/runs_laguna.csv")
ap.add_argument("--budgets", type=float, nargs="+", default=[2e6, 5e6, 1e7, 2e7])
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--trials", type=int, default=200)
ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)

VOCAB = {"cd","cat","sed","grep","python","python3","find","echo","ls","awk","head","tail",
         "git","pytest","mkdir","touch","rm","mv","cp","chmod","which","diff","make","pip"}
abstract = lambda t: (t if t in VOCAB else "other_cmd")

out = {}
for r in csv.DictReader(open(a.runs)):
    out[(r["task"], int(r["run"]))] = (int(r["success"]), int(r["tokens"]))

RE = re.compile(r"^(.*)_run(\d+)\.json$")
runs = collections.defaultdict(list)      # task -> list of dicts
for fp in sorted(glob.glob(os.path.join(a.traj, "*_run*.json"))):
    m = RE.match(os.path.basename(fp))
    if not m: continue
    task, rn = m.group(1), int(m.group(2))
    key = (task, rn) if (task, rn) in out else (task, rn-1)
    if key not in out: continue
    try: d = json.load(open(fp))
    except Exception: continue
    ch = d.get("command_history") or []
    acts, sizes = [], []
    for c in ch:
        t = c if isinstance(c, str) else (c.get("command") if isinstance(c, dict) else None)
        if not t: continue
        acts.append(abstract(str(t).strip().split()[0].lower() if str(t).strip() else ""))
        sizes.append(len(str(t)))
    if len(acts) < 2: continue
    succ, tok = out[key]
    # per-turn cost, scaled so the recorded total matches the run-level token count
    tot = max(sum(sizes), 1)
    cum = np.cumsum([s * tok / tot for s in sizes])
    runs[task].append(dict(succ=succ, tok=float(tok), acts=acts, cum=cum))
tasks = sorted(runs)
print(f"[data] {len(tasks)} tasks | {sum(len(v) for v in runs.values())} runs")

# ---- prefix monitor at turn k, cross-fitted by task
flat = [(t, i) for t in tasks for i in range(len(runs[t]))]
y = np.array([1 - runs[t][i]["succ"] for t, i in flat])
grp = np.array([t for t, _ in flat])
def fsm_of(idx, k):
    tr = collections.defaultdict(collections.Counter)
    for j in idx:
        t, i = flat[j]; s = "<init>"
        for x in runs[t][i]["acts"][:k]: tr[s][x] += 1; s = x
    o = {}
    for s, c in tr.items():
        keep = {x: n for x, n in c.items() if n >= 2 or len(c) == 1} or dict(c)
        tt = sum(keep.values()); o[s] = {x: n/tt for x, n in keep.items()}
    al = set(); [al.update(c) for c in tr.values()]
    return o, sorted(al)
def feat(j, k, fs, al, ai):
    t, i = flat[j]; seq = runs[t][i]["acts"][:k]; n = len(seq)
    v = np.zeros(len(al))
    for x in seq:
        q = ai.get(x)
        if q is not None: v[q] += 1
    v /= max(n, 1)
    FL = 1e-4; sur, s = [], "<init>"
    for x in seq: sur.append(-math.log(max(fs.get(s, {}).get(x, FL), FL))); s = x
    sur = np.array(sur) if sur else np.array([0.0])
    return np.concatenate([v, [sur.mean(), sur.max(), n, len(set(seq)),
                               1.0 - len(set(seq))/max(n,1)]])
score = np.zeros(len(flat))
for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(flat)), y, grp):
    fs, al = fsm_of(tr, a.k); ai = {x: q for q, x in enumerate(al)}
    mk = lambda ii: np.vstack([feat(j, a.k, fs, al, ai) for j in ii])
    c = HistGradientBoostingClassifier(max_iter=150, max_depth=3, random_state=0)
    c.fit(mk(tr), y[tr]); score[te] = c.predict_proba(mk(te))[:, 1]
SC = collections.defaultdict(dict)
for q, (t, i) in enumerate(flat): SC[t][i] = float(score[q])
print(f"[monitor] prefix predictor fitted at k={a.k}")

def cost_at(r, k):
    return float(r["cum"][min(k, len(r["cum"])) - 1]) if len(r["cum"]) else r["tok"]

def simulate(policy, budget, rg, thresh=None, cap=None, prior=None):
    alive = set(tasks); spent = 0.0; solved = 0
    ab = {t: [1.0, 1.0] for t in tasks}
    if prior is not None:
        for t in tasks:
            ab[t] = [1.0 + prior[t][0], 1.0 + prior[t][1]]
    cmean = {t: float(np.mean([r["tok"] for r in runs[t]])) for t in tasks}
    order = list(tasks); rg.shuffle(order); ptr = 0
    while alive and spent < budget:
        if policy in ("allocate", "hybrid"):
            best, bv = None, -1.0
            for t in alive:
                v = rg.beta(ab[t][0], ab[t][1]) / max(cmean[t], 1.0)
                if v > bv: bv, best = v, t
            cand = best
        else:
            cand = None
            for _ in range(len(order)):
                c = order[ptr % len(order)]; ptr += 1
                if c in alive: cand = c; break
            if cand is None: break
        i = rg.integers(len(runs[cand])); r = runs[cand][i]
        if cap is not None:                              # fixed turn cap
            spent += cost_at(r, cap)
            got = r["succ"] and len(r["acts"]) <= cap
        elif policy in ("abort", "hybrid") and thresh is not None and SC[cand][i] >= thresh:
            spent += cost_at(r, a.k); got = 0            # aborted at turn k
        else:
            spent += r["tok"]; got = r["succ"]
        if got: solved += 1; alive.discard(cand)
        else: ab[cand][1] += 1.0
    return solved, spent

allsc = np.array([SC[t][i] for t in tasks for i in range(len(runs[t]))])
THRESH = float(np.quantile(allsc, 0.5))
prior = {t: (sum(r["succ"] for r in runs[t]), sum(1-r["succ"] for r in runs[t])) for t in tasks}

print(f"\n{'budget':>8} | " + " | ".join(f"{n:>16}" for n in
      ["uniform","turn-cap 10","abort@k","allocate","hybrid"]))
res = {}
for B in a.budgets:
    line = []
    for name, kw in [("uniform", {}), ("turn-cap 10", {"cap": 10}),
                     ("abort", {"thresh": THRESH}), ("allocate", {"prior": prior}),
                     ("hybrid", {"thresh": THRESH, "prior": prior})]:
        pol = name if name in ("allocate", "hybrid", "abort") else "uniform"
        vals = [simulate(pol, B, np.random.default_rng(7000+s), **kw)[0]
                for s in range(a.trials)]
        line.append(f"{np.mean(vals):7.1f}+/-{np.std(vals):4.1f}")
        res[f"{int(B)}_{name}"] = [float(np.mean(vals)), float(np.std(vals))]
    print(f"{B/1e6:6.0f}M | " + " | ".join(f"{x:>16}" for x in line))
print(f"\n  abort threshold = median prefix score ({THRESH:.3f}); "
      f"aborting charges only tokens through turn {a.k}")
if a.out: json.dump(res, open(a.out,"w"), indent=2); print(f"[saved] {a.out}")
