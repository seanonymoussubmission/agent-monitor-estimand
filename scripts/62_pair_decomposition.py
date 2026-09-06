#!/usr/bin/env python3
"""Exact decomposition of pooled AUROC into within-unit and cross-unit parts.

The reviewer is right that A_pool - A_within is not a decomposition: the two are
different pairwise estimands and their difference cannot be *assigned* to task
difficulty. But an exact decomposition does exist, because pooled AUROC is a
probability over positive-negative pairs and those pairs partition cleanly:

    A_pool = w * A_within + (1 - w) * A_cross

where w is the share of discordant pairs drawn from the same unit. This is an
identity, not a diagnostic -- it holds by construction for any scoring function.
It lets us say exactly how much of pooled discrimination comes from ranking runs
of the SAME task against each other, and how much from ranking ACROSS tasks.
"""
import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--probe", required=True)
ap.add_argument("--k", type=int, default=1)
ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)
GRID = [1e-3, 1e-2, 1e-1, 1.0]

Z = np.load(a.probe, allow_pickle=False)
sel = Z["k"] == a.k
task, y = Z["task"][sel], Z["y"][sel].astype(int)
H = Z[f"H{a.layer}"][sel].astype(float)
pos = Z["pos"][sel].astype(float).reshape(-1, 1)

def oof(X, yy, g):
    p = np.zeros(len(yy))
    for tr, te in GroupKFold(n_splits=5).split(X, yy, g):
        best, bs, gi = GRID[0], -np.inf, g[tr]
        if len(set(gi)) >= 3:
            for c in GRID:
                sc = []
                for itr, ite in GroupKFold(n_splits=3).split(X[tr], yy[tr], gi):
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
                    try:
                        m.fit(X[tr][itr], yy[tr][itr])
                        sc.append(roc_auc_score(yy[tr][ite], m.predict_proba(X[tr][ite])[:, 1]))
                    except Exception: sc.append(-np.inf)
                if np.mean(sc) > bs: bs, best = np.mean(sc), c
        m = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=3000))
        m.fit(X[tr], yy[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p

def decompose(t, yy, s):
    """Partition every positive-negative pair by same-unit vs cross-unit."""
    P, N = np.where(yy == 1)[0], np.where(yy == 0)[0]
    conc = lambda i, j: (s[i] > s[j]) + 0.5 * (s[i] == s[j])
    win_n = win_d = cro_n = cro_d = 0.0
    for i in P:
        same = t[N] == t[i]
        c = conc(i, N)
        win_n += c[same].sum();  win_d += same.sum()
        cro_n += c[~same].sum(); cro_d += (~same).sum()
    tot = win_d + cro_d
    return dict(
        A_pool   = (win_n + cro_n) / tot,
        A_within = win_n / win_d if win_d else float("nan"),
        A_cross  = cro_n / cro_d if cro_d else float("nan"),
        w        = win_d / tot,
        n_within_pairs = int(win_d), n_cross_pairs = int(cro_d))

def report(name, s):
    d = decompose(task, y, s)
    lhs = d["A_pool"]
    rhs = d["w"] * d["A_within"] + (1 - d["w"]) * d["A_cross"]
    # cluster bootstrap over units for the two components
    units = np.array(sorted(set(task)))
    idx = {u: np.where(task == u)[0] for u in units}
    bw, bc, bp = [], [], []
    for _ in range(a.boot):
        pick = rng.choice(len(units), len(units), replace=True)
        rows = np.concatenate([idx[units[i]] for i in pick])
        tt = np.concatenate([np.full(len(idx[units[i]]), f"b{n}") for n, i in enumerate(pick)])
        try:
            b = decompose(tt, y[rows], s[rows])
            if not np.isnan(b["A_within"]): bw.append(b["A_within"])
            if not np.isnan(b["A_cross"]):  bc.append(b["A_cross"])
            bp.append(b["A_pool"])
        except Exception: pass
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 20 else (float("nan"),)*2
    print(f"\n--- {name} ---")
    print(f"  A_pool   = {d['A_pool']:.4f}   95% CI [{ci(bp)[0]:.4f}, {ci(bp)[1]:.4f}]")
    print(f"  A_within = {d['A_within']:.4f}   95% CI [{ci(bw)[0]:.4f}, {ci(bw)[1]:.4f}]   ({d['n_within_pairs']} pairs, w={d['w']:.4f})")
    print(f"  A_cross  = {d['A_cross']:.4f}   95% CI [{ci(bc)[0]:.4f}, {ci(bc)[1]:.4f}]   ({d['n_cross_pairs']} pairs)")
    print(f"  identity check: {d['w']:.4f}*{d['A_within']:.4f} + {1-d['w']:.4f}*{d['A_cross']:.4f} = {rhs:.6f}  vs A_pool {lhs:.6f}")
    assert abs(lhs - rhs) < 1e-9, "DECOMPOSITION IDENTITY VIOLATED"
    print("  [exact to 1e-9]")
    d.update({f"ci_{n}": ci(v) for n, v in [("within", bw), ("cross", bc), ("pool", bp)]})
    return d

print(f"[data] k={a.k} layer={a.layer}  n={len(y)}  units={len(set(task))}  pass={y.mean():.3f}")
print(f"[note] only {sum(len(set(y[task==u]))>1 for u in set(task))} units are mixed; "
      f"cross-unit pairs are {100*(1-decompose(task,y,pos.ravel())['w']):.1f}% of all pairs")
out = {"k": a.k, "layer": a.layer,
       "activations": report(f"activations L{a.layer}", oof(H, y, task)),
       "length_only": report("length only", oof(pos, y, task))}
if a.out:
    json.dump(out, open(a.out, "w"), indent=2); print(f"\n[saved] {a.out}")
