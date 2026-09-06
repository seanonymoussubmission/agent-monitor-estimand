#!/usr/bin/env python3
"""Is the within-task null an artifact of how within-unit pairs are weighted?

We report \within as a pair-weighted quantity: every discordant within-unit pair counts
once, so units contributing more pairs influence the estimate more. That is a choice, and a
reviewer is entitled to ask whether a different one would change the answer. We recompute
under four alternatives.

  PAIR-WEIGHTED   what we report (micro-average over pairs)
  UNIT-UNIFORM    macro-average: each unit's own AUROC counts equally regardless of size
  MIN-REPEAT      restrict to units with at least m recorded runs
  TASK-DEMEANED   subtract each unit's mean score before pooling -- the conditional-logit
                  analogue, which removes the unit effect rather than conditioning on it

Run on C4-L (the corpus carrying our headline null) and C1 at k=10 (where signal exists),
so the checks are exercised on both a null and a non-null.
"""
import argparse, json
import numpy as np
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--probe", default="FINAL/merged.npz")
ap.add_argument("--k", type=int, default=1)
ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--scores", default=None, help="optional npz with task,y,score")
ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

Z = np.load(a.probe, allow_pickle=False)
sel = Z["k"] == a.k
task, y = Z["task"][sel], Z["y"][sel].astype(int)
X = Z[f"H{a.layer}"][sel].astype(float)
GRID = [1e-3, 1e-2, 1e-1, 1.0]
s = np.zeros(len(y))
for tr, te in GroupKFold(n_splits=5).split(X, y, task):
    best, bs, gi = GRID[0], -np.inf, task[tr]
    if len(set(gi)) >= 3:
        for c in GRID:
            sc = []
            for itr, ite in GroupKFold(n_splits=3).split(X[tr], y[tr], gi):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
                try:
                    m.fit(X[tr][itr], y[tr][itr])
                    sc.append(roc_auc_score(y[tr][ite], m.predict_proba(X[tr][ite])[:,1]))
                except Exception: sc.append(-np.inf)
            if np.mean(sc) > bs: bs, best = np.mean(sc), c
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=3000))
    m.fit(X[tr], y[tr]); s[te] = m.predict_proba(X[te])[:,1]
print(f"[data] k={a.k} layer={a.layer} n={len(y)} tasks={len(set(task))}")

def pair_weighted(t, yy, sc):
    num = den = 0.0
    for u in set(t):
        m = t == u; P, N = sc[m & (yy==1)], sc[m & (yy==0)]
        if not len(P) or not len(N): continue
        num += sum((p>q)+0.5*(p==q) for p in P for q in N); den += len(P)*len(N)
    return num/den if den else np.nan

def unit_uniform(t, yy, sc):
    vals = []
    for u in set(t):
        m = t == u; P, N = sc[m & (yy==1)], sc[m & (yy==0)]
        if not len(P) or not len(N): continue
        vals.append(sum((p>q)+0.5*(p==q) for p in P for q in N)/(len(P)*len(N)))
    return float(np.mean(vals)) if vals else np.nan

def task_demeaned(t, yy, sc):
    d = sc.astype(float).copy()
    for u in set(t):
        m = t == u
        if len(set(yy[m].tolist())) > 1: d[m] -= d[m].mean()
        else: d[m] = np.nan
    ok = ~np.isnan(d)
    if len(set(yy[ok].tolist())) < 2: return np.nan
    return float(roc_auc_score(yy[ok], d[ok]))

def boot_ci(fn, t, yy, sc, nb):
    units = np.array(sorted(set(t))); loc = {u: np.where(t==u)[0] for u in units}
    vals = []
    for _ in range(nb):
        pick = rng.choice(len(units), len(units), replace=True)
        r = np.concatenate([loc[units[i]] for i in pick])
        t2 = np.concatenate([np.full(len(loc[units[i]]), f"b{n}") for n,i in enumerate(pick)])
        v = fn(t2, yy[r], sc[r])
        if not np.isnan(v): vals.append(v)
    return (np.percentile(vals,2.5), np.percentile(vals,97.5)) if len(vals)>20 else (np.nan,np.nan)

res = {}
print("\n=== WEIGHTING ===")
for name, fn in [("pair-weighted (reported)", pair_weighted),
                 ("unit-uniform macro", unit_uniform),
                 ("task-demeaned (cond. logit)", task_demeaned)]:
    v = fn(task, y, s); lo, hi = boot_ci(fn, task, y, s, a.boot)
    print(f"  {name:30s} {v:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    res[name] = [float(v), float(lo), float(hi)]

print("\n=== MINIMUM RUNS PER UNIT ===")
cnt = {u: int((task==u).sum()) for u in set(task)}
for m_ in (2, 3, 5, 8, 10):
    keep = np.array([cnt[t] >= m_ for t in task])
    if keep.sum() < 100: continue
    nu = sum(1 for u in set(task[keep]) if len(set(y[keep][task[keep]==u].tolist()))>1)
    v = pair_weighted(task[keep], y[keep], s[keep])
    lo, hi = boot_ci(pair_weighted, task[keep], y[keep], s[keep], max(400, a.boot//3))
    print(f"  >= {m_:2d} runs: {keep.sum():5d} rows, {nu:3d} mixed units  "
          f"within = {v:.4f}  [{lo:.4f}, {hi:.4f}]")
    res[f"min{m_}"] = [float(v), float(lo), float(hi), nu]
if a.out: json.dump(res, open(a.out,"w"), indent=2); print(f"\n[saved] {a.out}")
