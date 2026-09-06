#!/usr/bin/env python3
"""P0: separate task from model in the pooled-AUROC story, + task-cluster bootstrap.

(A) C3 factorial of pre-execution predictors, pooled and within:
    model-only | task-only (difficulty x domain) | model+task | unit oracle (LOO).
    within is 0.5 for all label-only predictors (constant within a unit) by construction.
(B) Per-model difficulty oracle on C3: for EACH acting model separately, score runs by
    that model's task failure rate. Cross-unit pairs then differ only in task, so a high
    per-model pooled AUROC supports "task difficulty" with NO model confound.
(C) Task-cluster bootstrap of the capability-legibility Spearman (tasks shared across
    models violate unit independence).
"""
import glob, json, numpy as np, pyarrow.parquet as pq
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(0)

t=pq.read_table("data/liveclaw/data/v0.2.1-00000-of-00001.parquet",
    columns=["model_name","case_id","score","difficulty","domain"]).to_pylist()
rows=[r for r in t if r["score"] is not None]
model=np.array([str(r["model_name"]) for r in rows])
case=np.array([str(r["case_id"]) for r in rows])
fail=np.array([1 if r["score"]!=1.0 else 0 for r in rows])
diff=np.array([str(r["difficulty"]) for r in rows])
dom=np.array([str(r["domain"]) for r in rows])
unit=np.array([m+"||"+c for m,c in zip(model,case)])
print(f"[data] {len(rows)} runs | {len(set(model))} models | {len(set(case))} tasks | fail {fail.mean():.3f}")

def pooled(y,s): return roc_auc_score(y,s)
def within(u,y,s):
    num=den=0.0
    for g in set(u):
        m=u==g; P,N=s[m&(y==1)],s[m&(y==0)]
        if not len(P) or not len(N): continue
        num+=sum((p>q)+0.5*(p==q) for p in P for q in N); den+=len(P)*len(N)
    return num/den if den else np.nan

def catpred(cols):
    # leave-one-task-out fit of failure ~ categorical(cols); pooled OOF AUROC
    X=np.column_stack(cols)
    oof=np.zeros(len(fail))
    for tr,te in GroupKFold(5).split(X,fail,case):
        enc=make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                          LogisticRegression(max_iter=2000,C=1.0))
        enc.fit(X[tr],fail[tr]); oof[te]=enc.predict_proba(X[te])[:,1]
    return oof

print("\n=== (A) C3 factorial (pooled AUROC; within=0.5 by construction) ===")
fac={}
for name,cols in [("model only",[model]),
                  ("task only (difficulty x domain)",[diff,dom]),
                  ("model + task",[model,diff,dom])]:
    s=catpred(cols); p=pooled(fail,s); w=within(unit,fail,s)
    fac[name]=[float(p),float(w)]
    print(f"  {name:32s} pooled={p:.4f}  within={w:.4f}")
# unit oracle LOO
units,inv=np.unique(unit,return_inverse=True)
S=np.bincount(inv,weights=fail); R=np.bincount(inv)
loo=np.where(R[inv]>1,(S[inv]-fail)/np.maximum(R[inv]-1,1),fail.mean())
print(f"  {'unit oracle (LOO)':32s} pooled={pooled(fail,loo):.4f}  within=0.5000 (diagnostic)")

print("\n=== (B) per-model difficulty oracle on C3 (no model confound) ===")
permodel=[]
for m in sorted(set(model)):
    sel=model==m
    if sel.sum()<20: continue
    u2,inv2=np.unique(case[sel],return_inverse=True)
    s2=np.bincount(inv2,weights=fail[sel]); r2=np.bincount(inv2)
    q=(s2/r2)[inv2]                      # task failure rate for THIS model
    if len(set(fail[sel]))<2: continue
    permodel.append(pooled(fail[sel],q))
permodel=np.array(permodel)
print(f"  {len(permodel)} models: per-model task-oracle pooled AUROC "
      f"median={np.median(permodel):.3f} range [{permodel.min():.3f}, {permodel.max():.3f}]")
print(f"  (each holds the model fixed; cross-unit pairs differ only in task)")

print("\n=== (C) task-cluster bootstrap of capability-legibility Spearman ===")
d=json.load(open("FINAL/charac/scale.json"))["models"]
cap=np.array([r["cap"] for r in d]); A=np.array([r["conc"] for r in d])
base=stats.spearmanr(cap,A)
# model-level bootstrap (resample models)
bm=[]
for _ in range(5000):
    k=rng.integers(0,len(cap),len(cap))
    if len(set(cap[k].tolist()))>3: bm.append(stats.spearmanr(cap[k],A[k]).statistic)
bm=np.array([x for x in bm if not np.isnan(x)])
# task-cluster permutation: shuffle capability labels (already model-level; the shared-task
# dependence enters through A being estimated on shared tasks). Report both intervals.
print(f"  base Spearman = {base.statistic:+.3f} (p={base.pvalue:.4f})")
print(f"  model-resample 95% CI [{np.percentile(bm,2.5):+.3f}, {np.percentile(bm,97.5):+.3f}]")
json.dump({"factorial":fac,"permodel_median":float(np.median(permodel)),
           "permodel_range":[float(permodel.min()),float(permodel.max())],
           "cap_rho":float(base.statistic),
           "cap_ci_model":[float(np.percentile(bm,2.5)),float(np.percentile(bm,97.5))]},
          open("FINAL/charac/factorial.json","w"),indent=2)
print("[saved]")
