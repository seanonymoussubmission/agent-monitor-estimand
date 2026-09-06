#!/usr/bin/env python3
"""
Pooled vs within-task AUROC for the LPH hidden-state probe.

The paper's whole claim is that pooled AUROC is inflated by between-unit
variation (task difficulty, model capability) and that within-task AUROC --
which holds both fixed by construction -- is the honest number. This script
tests that on real acting-model activations rather than hand-built features.

Guards, each of which has caught a bug at least once in this project:
  * GroupKFold by task, so a probe cannot memorise task identity.
  * Nested C selection, so the regulariser is not cherry-picked on test.
  * Two positive controls; if they fail, extraction is broken and every
    AUROC below is meaningless.
  * A length-only baseline, because trajectory length leaked 0.074 AUROC once.
"""
import argparse, json, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, r2_score

ap = argparse.ArgumentParser()
ap.add_argument("--probe", required=True)
ap.add_argument("--out",   required=True)
ap.add_argument("--boot",  type=int, default=1000)
ap.add_argument("--perm",  type=int, default=1000)
ap.add_argument("--seed",  type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

Z = np.load(args.probe, allow_pickle=False)
LAYERS = [int(x) for x in Z["layers"]]
rows = [dict(task=str(Z["task"][i]), run=int(Z["run"][i]), y=int(Z["y"][i]),
             k=int(Z["k"][i]), pos=int(Z["pos"][i]), repo=str(Z["repo"][i]),
             n_asst=int(Z["n_asst"][i])) for i in range(len(Z["y"]))]
H = {L: Z[f"H{L}"] for L in LAYERS}
for i, r in enumerate(rows): r["_i"] = i
D = {"layers": LAYERS, "turns": [int(x) for x in Z["turns"]]}
print(f"[load] {len(rows)} rows | layers {LAYERS} | turns {D['turns']}")

def within_auc(task, y, s):
    """AUROC over (fail, success) pairs drawn from the SAME unit.

    Task difficulty and model capability are constant inside a unit, so this
    isolates run-level discrimination -- what a deployed monitor actually needs.
    """
    num = den = 0.0
    per_unit = []
    for t in set(task):
        m = task == t
        pos, neg = s[m & (y == 1)], s[m & (y == 0)]
        if not len(pos) or not len(neg): continue
        c = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
        n = len(pos) * len(neg)
        num += c; den += n; per_unit.append(c / n)
    return (num / den if den else float("nan"),
            float(np.mean(per_unit)) if per_unit else float("nan"),
            len(per_unit))

def fit_predict(X, y, groups, kind="clf"):
    """Out-of-fold predictions, grouped by task, with nested hyperparameter choice."""
    oof = np.zeros(len(y), float)
    ng = len(set(groups))
    outer = GroupKFold(n_splits=min(5, ng))
    grid = [1e-3, 1e-2, 1e-1, 1.0] if kind == "clf" else [1.0, 10.0, 100.0, 1000.0]
    for tr, te in outer.split(X, y, groups):
        gi = groups[tr]
        best, best_s = grid[0], -np.inf
        if len(set(gi)) >= 3:
            inner = GroupKFold(n_splits=min(3, len(set(gi))))
            for c in grid:
                sc = []
                for itr, ite in inner.split(X[tr], y[tr], gi):
                    m = (make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
                         if kind == "clf" else make_pipeline(StandardScaler(), Ridge(alpha=c)))
                    try:
                        m.fit(X[tr][itr], y[tr][itr])
                        p = (m.predict_proba(X[tr][ite])[:, 1] if kind == "clf"
                             else m.predict(X[tr][ite]))
                        sc.append(roc_auc_score(y[tr][ite], p) if kind == "clf"
                                  else r2_score(y[tr][ite], p))
                    except Exception: sc.append(-np.inf)
                if np.mean(sc) > best_s: best_s, best = np.mean(sc), c
        m = (make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=3000))
             if kind == "clf" else make_pipeline(StandardScaler(), Ridge(alpha=best)))
        m.fit(X[tr], y[tr])
        oof[te] = (m.predict_proba(X[te])[:, 1] if kind == "clf" else m.predict(X[te]))
    return oof

results, SCORES = {}, {}
for k in sorted({r["k"] for r in rows}):
    sub = [r for r in rows if r["k"] == k]
    if len(sub) < 30: continue
    task = np.array([r["task"] for r in sub])
    y    = np.array([r["y"] for r in sub])
    pos  = np.array([r["pos"] for r in sub], float)
    repo = np.array([r["repo"] for r in sub])
    # a unit is only informative if its runs disagree
    keep = np.array([len(set(y[task == t])) > 1 for t in task])
    print(f"\n=== k={k}  n={len(sub)}  informative rows={keep.sum()}  "
          f"units={len(set(task[keep]))}  pass={y.mean():.3f}")

    # ---- positive control 1: is prefix length decodable from the activation?
    for L in LAYERS:
        X = H[L][[r["_i"] for r in sub]].astype(float)
        ctl_r2 = r2_score(np.log1p(pos), fit_predict(X, np.log1p(pos), task, "reg"))
        rep = (repo == repo[0]).astype(int)
        ctl_auc = (roc_auc_score(rep, fit_predict(X, rep, task, "clf"))
                   if 0 < rep.mean() < 1 else float("nan"))
        s = fit_predict(X, y, task, "clf")
        pooled = roc_auc_score(y, s) if 0 < y.mean() < 1 else float("nan")
        wp, wu, nu = within_auc(task[keep], y[keep], s[keep])
        results[f"k{k}_L{L}"] = dict(k=k, layer=L, n=len(sub), pooled=pooled,
                                     within_pair=wp, within_unit=wu, n_units=nu,
                                     ctl_len_r2=ctl_r2, ctl_repo_auc=ctl_auc)
        SCORES[f"k{k}_L{L}"] = (task[keep].copy(), y[keep].copy(), s[keep].copy())
        print(f"  L{L:>3}  pooled={pooled:.4f}  within={wp:.4f} (unit-avg {wu:.4f}, "
              f"{nu} units) | CTRL len-R2={ctl_r2:.3f} repo-AUC={ctl_auc:.3f}")

    # ---- length-only baseline: the confound that leaked 0.074 AUROC before
    Xl = np.c_[pos, np.log1p(pos), [r.get("n_asst", 0) for r in sub]]
    sl = fit_predict(Xl, y, task, "clf")
    pl = roc_auc_score(y, sl) if 0 < y.mean() < 1 else float("nan")
    wlp, wlu, _ = within_auc(task[keep], y[keep], sl[keep])
    results[f"k{k}_LENGTH"] = dict(k=k, layer="length-only", pooled=pl,
                                   within_pair=wlp, within_unit=wlu)
    print(f"  LEN  pooled={pl:.4f}  within={wlp:.4f}   <- no activations at all")

# ---------------- inference on EVERY cell, then Holm-Bonferroni
# Picking the max-AUROC cell and testing only that one is the winner's curse:
# across 16 cells a nominal p=0.0055 is ~0.09 corrected. Test them all.
print(f"\n=== permutation test on all {len(SCORES)} cells (Holm-corrected) ===")
cells = []
for name, (tk_, yk_, sk_) in SCORES.items():
    obs, _, nu = within_auc(tk_, yk_, sk_)
    if obs != obs: continue
    null = []
    for _ in range(args.perm):
        yp = yk_.copy()
        for t in set(tk_):
            m = tk_ == t; yp[m] = rng.permutation(yp[m])
        null.append(within_auc(tk_, yp, sk_)[0])
    null = np.array(null)
    praw = (1 + np.sum(np.abs(null - 0.5) >= abs(obs - 0.5))) / (args.perm + 1)
    cells.append(dict(cell=name, within=obs, p_raw=float(praw), n_units=nu,
                      k=results[name]["k"], layer=results[name]["layer"],
                      pooled=results[name]["pooled"]))

cells.sort(key=lambda c: c["p_raw"])
m = len(cells); prev = 0.0
for i, c in enumerate(cells):                       # Holm step-down
    prev = c["p_holm"] = max(prev, min(1.0, (m - i) * c["p_raw"]))
    c["sig"] = c["p_holm"] < 0.05

print(f"{'cell':>12} {'within':>8} {'pooled':>8} {'p_raw':>8} {'p_holm':>8}  sig")
for c in cells:
    print(f"{c['cell']:>12} {c['within']:>8.4f} {c['pooled']:>8.4f} "
          f"{c['p_raw']:>8.4f} {c['p_holm']:>8.4f}  {'YES' if c['sig'] else '-'}")

W = np.array([c["within"] for c in cells]); nsig = sum(c["sig"] for c in cells)
print(f"\nwithin-task AUROC over {m} cells: mean={W.mean():.4f} median={np.median(W):.4f} "
      f"min={W.min():.4f} max={W.max():.4f}")
print(f"cells surviving Holm at 0.05: {nsig}/{m}")
print("VERDICT: " + ("NO cell survives correction -- consistent with within-task AUROC = 0.5"
                    if nsig == 0 else f"{nsig} cell(s) show real within-task signal"))

bc = max(cells, key=lambda c: c["within"])
tk_, yk_, sk_ = SCORES[bc["cell"]]
units = sorted(set(tk_)); bs = []
for _ in range(args.boot):
    pick = rng.choice(units, len(units), replace=True)
    idx = np.concatenate([np.where(tk_ == u)[0] for u in pick])
    lab = np.concatenate([[f"{u}#{i}"] * (tk_ == u).sum() for i, u in enumerate(pick)])
    v = within_auc(lab, yk_[idx], sk_[idx])[0]
    if v == v: bs.append(v)
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\nmax-AUROC cell {bc['cell']}: within={bc['within']:.4f} "
      f"95% CI [{lo:.4f}, {hi:.4f}]  p_holm={bc['p_holm']:.4f}")
print("  (this CI is DESCRIPTIVE only -- the cell was chosen for being the max,\n"
      "   so the interval is optimistically biased. Holm is the inference.)")
best = bc

json.dump(dict(results=results, cells=cells, best=best, ci=[lo, hi],
               n_sig_after_holm=nsig, within_mean=float(W.mean()),
               within_median=float(np.median(W))), open(args.out, "w"), indent=2)
print(f"\n[saved] {args.out}")
