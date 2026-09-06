#!/usr/bin/env python3
"""Can pre-execution features seed the bandit prior on genuinely novel tasks?

Our proposal warm-starts each arm from another model's observed outcomes, which requires
that somebody has already attempted the task. We called the remaining case -- a task no
model has seen -- closed. Script 67 shows that was wrong: released pre-execution features
predict our observed difficulty at Spearman ~0.53 held-out.

So we test the constructive version. Fit difficulty from pre-execution features on a
training half of the tasks, predict it for the held-out half, and seed those arms'
Beta priors from the prediction. The arms are then warm-started using only information
available before any attempt. The comparison is against an uninformative prior (the
honest floor for a novel task) and against the transferred prior (which needs rollouts
and is therefore unavailable in this setting -- it is an upper reference, not a rival).
"""
import argparse, csv, glob, json, os
import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--target", default="data/runs_qwen.csv")
ap.add_argument("--prior",  default="data/runs_laguna.csv")
ap.add_argument("--budget", type=float, default=5.0e6)
ap.add_argument("--trials", type=int, default=300)
ap.add_argument("--splits", type=int, default=10)
ap.add_argument("--pseudo", type=float, nargs="+", default=[0.25,0.5,1.0,2.0,4.0,8.0],
                help="pseudo-run weights to sweep for the seeded prior")
ap.add_argument("--out", default=None)
a = ap.parse_args()

def load(p):
    by = {}
    for r in csv.DictReader(open(p)):
        by.setdefault(r["task"], []).append((int(r["success"]), int(r["tokens"])))
    return by
T, P = load(a.target), load(a.prior)

def read_rubric(path):
    d = {}
    with open(path) as f:
        rd = csv.DictReader(f); cols = [c for c in rd.fieldnames if c != "instance_id"]
        for r in rd:
            try: d[r["instance_id"]] = [float(r[c]) for c in cols]
            except (ValueError, TypeError): pass
    return d
RUB = read_rubric(f"{a.repo}/llm_judge_features/information_ablation/swebench_verified/1_problem_15.csv")
EMB_ids = EMB_X = None
for f in sorted(glob.glob(f"{a.repo}/embeddings/*.npz")):
    z = np.load(f, allow_pickle=True)
    if "verified" in str(z.get("dataset_name", "")).lower() and "task_ids" in z:
        if EMB_X is None or z["X"].shape[1] > EMB_X.shape[1]:
            EMB_ids, EMB_X = [str(x) for x in z["task_ids"]], z["X"]
eidx = {t: i for i, t in enumerate(EMB_ids)}

tasks = sorted(set(T) & set(P) & set(RUB) & set(eidx))
print(f"[data] {len(tasks)} tasks with runs, rubric and embeddings")
X = np.array([RUB[t] + list(EMB_X[eidx[t]]) for t in tasks])
p_true = np.array([np.mean([s for s, _ in T[t]]) for t in tasks])       # P(success)
c_mean = np.array([float(np.mean([c for _, c in T[t]])) for t in tasks])
TR = {t: T[t] for t in tasks}

def ts(sub, ab, rng, budget):
    alive = set(sub); spent = solved = 0
    A = {i: list(ab[i]) for i in sub}
    while alive and spent < budget:
        best, bv = None, -1.0
        for i in alive:
            v = rng.beta(A[i][0], A[i][1]) / max(c_mean[i], 1.0)
            if v > bv: bv, best = v, i
        runs = TR[tasks[best]]
        s, c = runs[rng.integers(len(runs))]
        spent += c
        if s: solved += 1; alive.discard(best)
        else: A[best][1] += 1.0
    return solved

res = {}
for sp in range(a.splits):
    rng = np.random.default_rng(500 + sp)
    perm = rng.permutation(len(tasks)); half = len(tasks) // 2
    tr, te = perm[:half], perm[half:]
    # fit difficulty on TRAIN tasks only, predict for the held-out (novel) tasks
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 25)))
    m.fit(X[tr], p_true[tr])
    pred = np.clip(m.predict(X[te]), 0.01, 0.99)
    rho = stats.spearmanr(pred, p_true[te]).statistic
    budget = a.budget * len(te) / len(tasks)

    ab_un   = {i: [1.0, 1.0] for i in te}
    PRE = {f"pre-execution seed (w={w})":
           {i: [1.0 + w * pred[j], 1.0 + w * (1 - pred[j])] for j, i in enumerate(te)}
           for w in a.pseudo}
    ab_tr   = {}
    for i in te:
        s = sum(x for x, _ in P[tasks[i]]); n = len(P[tasks[i]])
        ab_tr[i] = [1.0 + 1.0 * s, 1.0 + 1.0 * (n - s)]
    ab_or   = {i: [1.0 + 50 * p_true[i], 1.0 + 50 * (1 - p_true[i])] for i in te}

    ORDER = ([("uninformative", ab_un)] + sorted(PRE.items()) +
             [("transferred (needs rollouts)", ab_tr), ("oracle", ab_or)])
    for name, ab in ORDER:
        res.setdefault(name, [])
        r2 = np.random.default_rng(900 + sp)
        res[name].append(np.mean([ts(list(te), ab, r2, budget) for _ in range(a.trials)]))
    print(f"  split {sp}: held-out rho={rho:+.3f} | "
          f"unif={res['uninformative'][-1]:.1f} | "
          f"transferred={res['transferred (needs rollouts)'][-1]:.1f}", flush=True)

print(f"\n=== COLD START ON NOVEL TASKS (budget {a.budget/1e6:.1f}M scaled to half) ===")
base = float(np.mean(res["uninformative"]))
summary = {}
for k, v in res.items():
    v = np.array(v)
    print(f"  {k:32s} {v.mean():7.2f} +/- {v.std():5.2f}   "
          f"{100*(v.mean()-base)/base:+6.1f}% vs uninformative")
    summary[k] = [float(v.mean()), float(v.std())]
if a.out: json.dump(summary, open(a.out, "w"), indent=2); print(f"[saved] {a.out}")
