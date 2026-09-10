#!/usr/bin/env python3
"""Deployment replay on a SECOND corpus (C2), with a cross-corpus transferred prior,
and a cost-side reading of the same replay.

Why this run exists: the paper's deployment findings previously rested on C4 alone,
while its measurement findings rest on four corpora. Here the target is C2
(SWE-rebench / OpenHands / Qwen3-Coder-480B) and the prior is transferred from C1
(SWE-agent / Llama-3) on the 1,203 tasks the two corpora share -- so the prior crosses
model family, scaffold AND corpus, a harder transfer than C4's within-corpus one.

Policies (all replay real recorded runs, charge their estimated cost, retire a task on
success): round-robin; static transferred ranking walked in fixed priority order;
budgeted Thompson sampling with the transferred prior; the same with an uninformative
prior (isolates what the prior contributes).

Two readings of the same simulation:
  (a) tasks solved at a fixed budget      -- the paper's existing framing;
  (b) budget needed to reach a fixed number of tasks solved -- the cost-side framing,
      reported as tokens and as a saving against round-robin.
"""
import argparse, collections, glob, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True, help="C2 runs table from script 115")
ap.add_argument("--c1-glob", default="data/swe_agent_traj/data/train-*.parquet")
ap.add_argument("--budgets", type=float, nargs="+",
                default=[2e6, 5e6, 1e7, 2.5e7, 5e7])
ap.add_argument("--targets", type=int, nargs="+", default=[100, 200, 300, 400])
ap.add_argument("--trials", type=int, default=150)
ap.add_argument("--lam", type=float, default=0.5, help="transferred-prior weight")
ap.add_argument("--out", required=True)
a = ap.parse_args()

# ---------------------------------------------------------------- target corpus
rt = pd.read_parquet(a.runs)
runs = collections.defaultdict(list)
for r in rt.itertuples(index=False):
    runs[r.task].append(dict(succ=int(r.success), tok=float(r.tokens),
                             cum=np.asarray(r.cum, dtype=float)))

# ---------------------------------------------------------------- transferred prior
prior = collections.defaultdict(lambda: [0, 0])   # task -> [successes, attempts]
for p in sorted(glob.glob(a.c1_glob)):
    d = pd.read_parquet(p, columns=["instance_id", "target"])
    for iid, tgt in d.itertuples(index=False):
        s = prior[str(iid)]
        s[0] += int(bool(tgt))
        s[1] += 1

tasks = sorted(t for t in runs if t in prior and len(runs[t]) >= 2)
print("[data] %d target tasks with a transferred prior | %d runs | mean %.1f runs/task"
      % (len(tasks), sum(len(runs[t]) for t in tasks),
         np.mean([len(runs[t]) for t in tasks])), flush=True)
solvable = sum(1 for t in tasks if any(r["succ"] for r in runs[t]))
print("[data] tasks with at least one recorded success: %d" % solvable, flush=True)

cmean = {t: float(np.mean([r["tok"] for r in runs[t]])) for t in tasks}


def ab(t, warm):
    if not warm:
        return [1.0, 1.0]
    s, n = prior[t]
    return [1.0 + a.lam * s, 1.0 + a.lam * (n - s)]


def prior_rate(t):
    s, n = prior[t]
    return (s + 0.5) / (n + 1.0)


RANK = sorted(tasks, key=lambda t: -(prior_rate(t) / max(cmean[t], 1.0)))


def simulate(policy, budget, rg, record=False):
    """Returns solved count at budget; if record, also the (spend, solved) curve."""
    alive = set(tasks)
    spent = 0.0
    solved = 0
    curve = []
    warm = policy in ("ts_warm",)
    st = {t: ab(t, warm) for t in tasks}
    if policy == "uniform":
        order = list(tasks)
        rg.shuffle(order)
    elif policy == "static_rank":
        order = list(RANK)
    else:
        order = None
    ptr = 0
    while alive and spent < budget:
        if order is not None:
            cand = None
            for _ in range(len(order)):
                c = order[ptr % len(order)]
                ptr += 1
                if c in alive:
                    cand = c
                    break
            if cand is None:
                break
        else:
            best, bv = None, -1.0
            for t in alive:
                v = rg.beta(st[t][0], st[t][1]) / max(cmean[t], 1.0)
                if v > bv:
                    bv, best = v, t
            cand = best
        r = runs[cand][rg.integers(len(runs[cand]))]
        spent += r["tok"]
        if r["succ"]:
            solved += 1
            alive.discard(cand)
        else:
            st[cand][1] += 1.0
        if record:
            curve.append((spent, solved))
    return (solved, curve) if record else solved


POLICIES = ["uniform", "static_rank", "ts_warm", "ts_cold"]
LABEL = {"uniform": "round-robin", "static_rank": "static transferred ranking",
         "ts_warm": "Thompson + transferred prior", "ts_cold": "Thompson, no prior"}

# ---- (a) tasks solved at fixed budget
res = {"solved_at_budget": {}, "budget_to_reach": {}}
print("\n=== tasks solved at a fixed budget (mean over %d paired trials) ===" % a.trials,
      flush=True)
hdr = "%-30s" % "policy" + "".join("%12s" % ("%.0fM" % (b / 1e6)) for b in a.budgets)
print(hdr, flush=True)
for pol in POLICIES:
    vals = []
    for b in a.budgets:
        v = [simulate(pol, b, np.random.default_rng(7000 + s)) for s in range(a.trials)]
        vals.append(float(np.mean(v)))
        res["solved_at_budget"].setdefault(pol, {})[str(int(b))] = \
            [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    print("%-30s" % LABEL[pol] + "".join("%12.1f" % v for v in vals), flush=True)

# ---- (b) budget needed to reach a fixed number solved
BIG = max(a.budgets) * 4
curves = {}
for pol in POLICIES:
    cs = []
    for s in range(a.trials):
        _, c = simulate(pol, BIG, np.random.default_rng(7000 + s), record=True)
        cs.append(c)
    curves[pol] = cs


def budget_for(curve, k):
    for spend, sol in curve:
        if sol >= k:
            return spend
    return float("nan")


print("\n=== budget needed to reach a target number of tasks solved ===", flush=True)
print("%-30s" % "policy" + "".join("%14s" % ("%d solved" % k) for k in a.targets),
      flush=True)
base = {}
for pol in POLICIES:
    cells = []
    for k in a.targets:
        vals = [budget_for(c, k) for c in curves[pol]]
        vals = [v for v in vals if np.isfinite(v)]
        m = float(np.mean(vals)) if len(vals) >= 0.5 * a.trials else float("nan")
        res["budget_to_reach"].setdefault(pol, {})[str(k)] = m
        if pol == "uniform":
            base[k] = m
        cells.append(m)
    print("%-30s" % LABEL[pol] +
          "".join("%14s" % ("%.1fM" % (c / 1e6) if np.isfinite(c) else "n/r")
                  for c in cells), flush=True)

print("\n=== token saving against round-robin at the same number solved ===", flush=True)
print("%-30s" % "policy" + "".join("%14s" % ("%d solved" % k) for k in a.targets),
      flush=True)
for pol in POLICIES:
    if pol == "uniform":
        continue
    cells = []
    for k in a.targets:
        m = res["budget_to_reach"][pol][str(k)]
        b = base.get(k, float("nan"))
        cells.append(100.0 * (1.0 - m / b) if np.isfinite(m) and np.isfinite(b) and b else float("nan"))
    print("%-30s" % LABEL[pol] +
          "".join("%13s%%" % ("%.0f" % c if np.isfinite(c) else "n/r") for c in cells),
          flush=True)

json.dump(res, open(a.out, "w"), indent=2)
print("\n[saved] %s" % a.out, flush=True)
