#!/usr/bin/env python3
"""Calibrated label-noise check: what noise rate would hide an effect of the size that matters?

Script 80 showed the analytic attenuation formula understates real damage, because it
assumes a fixed scorer whereas a deployed pipeline also *trains* on the corrupted labels.
So we answer the question empirically and in the right regime: construct a synthetic feature
whose clean within-task AUROC is close to a target effect size, corrupt the success labels
at rate alpha, refit, and find the alpha at which the observed value falls to what we
actually measure on C4-L.
"""
import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--probe", default="FINAL/merged.npz")
ap.add_argument("--k", type=int, default=1)
ap.add_argument("--targets", type=float, nargs="+", default=[0.55, 0.60, 0.65])
ap.add_argument("--alphas", type=float, nargs="+",
                default=[0.0,0.05,0.10,0.15,0.20,0.30,0.40,0.50])
ap.add_argument("--reps", type=int, default=25)
ap.add_argument("--observed", type=float, default=0.5241)
ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)

Z = np.load(a.probe, allow_pickle=False)
sel = Z["k"] == a.k
task, y0 = Z["task"][sel], Z["y"][sel].astype(int)
print(f"[data] {len(y0)} rows | {len(set(task))} tasks")

def within(t, yy, sc):
    num=den=0.0
    for u in set(t):
        m=t==u; P,N=sc[m&(yy==1)], sc[m&(yy==0)]
        if not len(P) or not len(N): continue
        num+=sum((p>q)+0.5*(p==q) for p in P for q in N); den+=len(P)*len(N)
    return num/den if den else np.nan

def fit_score(yy, F):
    s=np.zeros(len(yy))
    for tr,te in GroupKFold(n_splits=5).split(F, yy, task):
        if len(set(yy[tr].tolist()))<2: continue
        m=make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
        m.fit(F[tr], yy[tr]); s[te]=m.predict_proba(F[te])[:,1]
    return s

def make_feature(sigma):
    return np.c_[y0.astype(float) + rng.normal(0, sigma, len(y0))]

# calibrate sigma so the clean within-task AUROC lands near each target
print("\n=== calibrating synthetic signal strength ===")
cal = {}
for tgt in a.targets:
    lo, hi = 0.3, 30.0
    for _ in range(18):
        mid = (lo+hi)/2
        v = np.mean([within(task, y0, fit_score(y0, make_feature(mid))) for _ in range(3)])
        if v > tgt: lo = mid
        else: hi = mid
    cal[tgt] = (lo+hi)/2
    got = np.mean([within(task, y0, fit_score(y0, make_feature(cal[tgt]))) for _ in range(5)])
    print(f"  target {tgt:.2f} -> sigma {cal[tgt]:6.2f}  (clean within = {got:.4f})")

print(f"\n=== attenuation under label noise (observed on C4-L = {a.observed:.4f}) ===")
print(f"  {'alpha':>6} " + " ".join(f"{('true '+format(t,'.2f')):>12}" for t in a.targets))
res = {}
for al in a.alphas:
    row = []
    for tgt in a.targets:
        vals = []
        for _ in range(a.reps):
            yy = y0.copy(); succ = np.where(yy==1)[0]
            flip = rng.choice(succ, int(round(al*len(succ))), replace=False)
            yy[flip] = 0
            if len(set(yy.tolist())) < 2: continue
            v = within(task, yy, fit_score(yy, make_feature(cal[tgt])))
            if not np.isnan(v): vals.append(v)
        row.append(float(np.mean(vals)))
    res[al] = row
    print(f"  {al:6.2f} " + " ".join(f"{v:12.4f}" for v in row))

print("\n=== the alpha at which each true effect would look like our observation ===")
for j, tgt in enumerate(a.targets):
    hit = None
    prev_a, prev_v = None, None
    for al in a.alphas:
        v = res[al][j]
        if v <= a.observed and prev_v is not None:
            frac = (prev_v - a.observed) / max(prev_v - v, 1e-9)
            hit = prev_a + frac*(al-prev_a); break
        prev_a, prev_v = al, v
    if hit is None:
        print(f"  true {tgt:.2f}: never falls to {a.observed:.4f} within alpha <= {max(a.alphas):.2f}")
    else:
        print(f"  true {tgt:.2f}: would require alpha = {hit:.2f} "
              f"({100*hit:.0f}% of successes mislabelled)")
if a.out: json.dump({str(k): v for k, v in res.items()}, open(a.out,"w"), indent=2)
