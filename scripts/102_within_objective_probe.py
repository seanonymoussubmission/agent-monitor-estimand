#!/usr/bin/env python3
"""E1-P (pre-registered amendment): the within-unit training objective applied to the
C4-L hidden-state probes.

For each prefix k and layer: (a) pooled-objective logistic probe, (b) same-unit pairwise
Bradley-Terry ranker on activation differences (the conditional-logit estimator for
pairs), on identical GroupKFold-by-task splits and standardized 2048-d activations.
Within-task AUROC pair-weighted over mixed units; shuffled-label control.
Data: the merged C4-L probe archive (task, run, k rows; unit = task, single model).
"""
import argparse, collections, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--layers", type=int, nargs="+", default=[10, 20, 30, 40])
ap.add_argument("--out", default=None)
a = ap.parse_args()

z = np.load(a.npz, allow_pickle=True)
task = np.asarray(z["task"]); run = np.asarray(z["run"])
y = np.asarray(z["y"]).astype(int); kk = np.asarray(z["k"]).astype(int)
print(f"[data] {len(y)} rows | {len(set(task))} tasks | fail-ish mean {y.mean():.3f}",
      flush=True)

def within_auc(u, yy, s):
    num = den = 0.0; nu = 0
    for g in set(u):
        m = u == g; P, N = s[m & (yy == 1)], s[m & (yy == 0)]
        if not len(P) or not len(N): continue
        nu += 1
        num += sum((p > q) + 0.5 * (p == q) for p in P for q in N); den += len(P) * len(N)
    return (num / den if den else float("nan")), nu

def fit_pair(X, yv, u, Xte):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Z, Zt = (X - mu) / sd, (Xte - mu) / sd
    pool = LogisticRegression(C=1.0, max_iter=2000).fit(Z, yv)
    diffs = []
    for g in set(u):
        m = u == g
        if yv[m].min() == yv[m].max(): continue
        Zf, Zs = Z[m & (yv == 1)], Z[m & (yv == 0)]
        for zf in Zf:
            for zs in Zs: diffs.append(zf - zs)
    D = np.vstack(diffs); Xp = np.vstack([D, -D])
    yp = np.r_[np.ones(len(D)), np.zeros(len(D))]
    rk = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(Xp, yp)
    return pool.predict_proba(Zt)[:, 1], Zt @ rk.coef_[0]

res = {}
rng = np.random.default_rng(0)
for k in sorted(set(kk)):
    mk = kk == k
    tk, uk, yk = task[mk], task[mk], y[mk]
    for L in a.layers:
        X = np.asarray(z[f"H{L}"])[mk].astype(np.float64)
        op = np.zeros(mk.sum()); orr = np.zeros(mk.sum())
        for tr, te in GroupKFold(n_splits=5).split(X, yk, tk):
            sp, sr = fit_pair(X[tr], yk[tr], uk[tr], X[te])
            op[te], orr[te] = sp, sr
        wp, nu = within_auc(uk, yk, op); wr, _ = within_auc(uk, yk, orr)
        pp, pr = roc_auc_score(yk, op), roc_auc_score(yk, orr)
        res[f"k{k}_L{L}"] = dict(pooled_obj=[float(pp), float(wp)],
                                 ranker=[float(pr), float(wr)], units=nu)
        print(f"  k={k:2d} L{L}: pooled-obj pooled={pp:.3f} within={wp:.3f} | "
              f"ranker pooled={pr:.3f} within={wr:.3f}  ({nu} mixed units)", flush=True)

# control: shuffled labels within units at k=5, L=20, ranker
mk = kk == 5
tk, yk = task[mk], y[mk].copy()
X = np.asarray(z["H20"])[mk].astype(np.float64)
tr, te = next(GroupKFold(n_splits=5).split(X, yk, tk))
ysh = yk.copy()
for g in set(tk[tr]):
    idx = np.where(tk == g)[0]; idx = idx[np.isin(idx, tr)]
    ysh[idx] = rng.permutation(ysh[idx])
_, sr = fit_pair(X[tr], ysh[tr], tk[tr], X[te])
wv, _ = within_auc(tk[te], yk[te], sr)
print(f"\n[control] shuffled-label ranker within = {wv:.3f} (expect ~0.5)", flush=True)
if a.out: json.dump(res, open(a.out, "w"), indent=2); print(f"[saved] {a.out}")
