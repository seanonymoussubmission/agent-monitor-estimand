#!/usr/bin/env python3
"""How much label noise would be required to hide a real run-level effect?

Our labels are the benchmarks' own oracles, and a fraction of SWE-bench patches marked
solved are semantically wrong. Label noise attenuates AUROC toward 0.5, so a genuine effect
could in principle be flattened into our null. We bound that analytically elsewhere; here we
check the bound empirically and invert it into the question a reviewer actually cares
about.

Two directions.

  FORWARD.  Take a condition where signal demonstrably exists, corrupt a known fraction
            alpha of the success labels into failures (the documented direction of the
            error), and watch the observed within-task AUROC decay. This tests whether the
            attenuation formula A_obs = 0.5 + (A_true - 0.5)(1 - alpha) describes the data
            rather than merely being asserted.

  INVERSE.  Given what we actually observe, ask what alpha would be needed for the true
            effect to be a specified size. If hiding a useful effect requires a noise rate
            far beyond anything documented, the null is not a noise artifact.
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
ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--alphas", type=float, nargs="+", default=[0.0,0.08,0.20,0.30,0.45])
ap.add_argument("--reps", type=int, default=25)
ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)

Z = np.load(a.probe, allow_pickle=False)
sel = Z["k"] == a.k
task, y0 = Z["task"][sel], Z["y"][sel].astype(int)
X = Z[f"H{a.layer}"][sel].astype(float)
pos = Z["pos"][sel].astype(float).reshape(-1,1)
print(f"[data] {len(y0)} rows | {len(set(task))} tasks | pass rate {y0.mean():.3f}")

def within(t, yy, sc):
    num=den=0.0
    for u in set(t):
        m=t==u; P,N=sc[m&(yy==1)], sc[m&(yy==0)]
        if not len(P) or not len(N): continue
        num+=sum((p>q)+0.5*(p==q) for p in P for q in N); den+=len(P)*len(N)
    return num/den if den else np.nan

def fit(yy, F):
    s=np.zeros(len(yy))
    for tr,te in GroupKFold(n_splits=5).split(F, yy, task):
        if len(set(yy[tr].tolist()))<2: continue
        m=make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
        m.fit(F[tr], yy[tr]); s[te]=m.predict_proba(F[te])[:,1]
    return s

# A condition with demonstrable signal: an oracle feature, attenuated the same way.
ORACLE = np.c_[y0.astype(float) + rng.normal(0, 0.55, len(y0))]
print("\n=== FORWARD: corrupting a known-signal condition ===")
print(f"  {'alpha':>6} {'observed':>10} {'predicted by formula':>22}")
rows=[]
base=None
for al in a.alphas:
    vals=[]
    for _ in range(a.reps):
        yy=y0.copy()
        succ=np.where(yy==1)[0]
        flip=rng.choice(succ, int(round(al*len(succ))), replace=False)
        yy[flip]=0
        if len(set(yy.tolist()))<2: continue
        v=within(task, yy, fit(yy, ORACLE))
        if not np.isnan(v): vals.append(v)
    obs=float(np.mean(vals))
    if al==0.0: base=obs
    pred=0.5+(base-0.5)*(1-al) if base is not None else np.nan
    print(f"  {al:6.2f} {obs:10.4f} {pred:22.4f}")
    rows.append(dict(alpha=al, observed=obs, formula=float(pred)))

print("\n=== INVERSE: what alpha would hide a real effect, given what we observe? ===")
obs_real = 0.5241     # C4-L, k=1, layer 20
print(f"  observed within-task AUROC = {obs_real:.4f}")
for target in (0.55, 0.60, 0.65, 0.70):
    need = 1 - (obs_real-0.5)/(target-0.5)
    print(f"  to be truly {target:.2f}, the success class would have to be "
          f"{100*need:5.1f}% mislabelled")
print("\n  Documented SWE-bench oracle error is on the order of 20%~\\cite{solved}.")
if a.out:
    json.dump(dict(forward=rows, observed=obs_real), open(a.out,"w"), indent=2)
    print(f"[saved] {a.out}")
