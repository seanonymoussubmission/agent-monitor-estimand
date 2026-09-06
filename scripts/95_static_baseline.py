#!/usr/bin/env python3
"""P1: a DEPLOYABLE pre-execution static predictor on C1 and C2.

The difficulty oracle uses the target unit's own other-run outcomes (diagnostic, not
deployable). This baseline uses only information available BEFORE the run starts---the
issue text and the repository identity---trained with GroupKFold by task so no target
task appears in training. If it reaches high pooled AUROC while within-task stays at
0.5, the between-unit confound is demonstrated without any outcome leakage.

Features: TF-IDF over the issue statement (the first user turn = the task input, not the
agent's behaviour) + repository token (C2) / instance-family prefix (C1).
"""
import glob, json, re, os, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

def within(task,y,s):
    num=den=0.0
    for u in set(task):
        m=task==u; P,N=s[m&(y==1)],s[m&(y==0)]
        if not len(P) or not len(N): continue
        num+=sum((p>q)+0.5*(p==q) for p in P for q in N); den+=len(P)*len(N)
    return num/den if den else np.nan

def run(name, iids, texts, y, groups):
    y=np.array(y); groups=np.array(groups); iids=np.array(iids)
    oof=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(texts,y,groups):
        vec=TfidfVectorizer(max_features=4000,ngram_range=(1,2),min_df=3,sublinear_tf=True)
        Xtr=vec.fit_transform([texts[i] for i in tr]); Xte=vec.transform([texts[i] for i in te])
        m=LogisticRegression(max_iter=2000,C=1.0)
        m.fit(Xtr,y[tr]); oof[te]=m.predict_proba(Xte)[:,1]
    unit=np.array([f"{g}" for g in groups])  # C2 single model: unit==task; C1: use model||task via iids
    p=roc_auc_score(y,oof); w=within(iids,y,oof)
    print(f"  {name}: pre-exec static  pooled={p:.4f}  within={w:.4f}  (n={len(y)}, tasks={len(set(groups))})")
    return float(p),float(w)

out={}
# ---- C1: issue text = first user turn; unit = model||instance; group = instance
rows=[]
for p in sorted(glob.glob("data/swe_agent_traj/data/*.parquet")):
    df=pd.read_parquet(p,columns=["instance_id","model_name","target","trajectory"])
    for iid,mdl,tgt,tr in df.itertuples(index=False):
        issue=""
        for st in list(tr):
            if st.get("role")=="user" and st.get("text"):
                issue=st["text"][:4000]; break
        if not issue: continue
        rows.append((f"{mdl}||{iid}", iid, issue, 1-int(bool(tgt))))
    del df
if rows:
    out["C1"]=run("C1", [r[0] for r in rows], [r[2] for r in rows],
                  [r[3] for r in rows], [r[1] for r in rows])

# ---- C2: issue text = first user turn; single model so unit==instance
import pyarrow.parquet as pq
t=pq.read_table("data/swe_rebench/trajectories.parquet",
                columns=["instance_id","repo","trajectory","resolved"])
rows=[]
for iid,repo,tr,res in zip(*[t.column(c).to_pylist() for c in ("instance_id","repo","trajectory","resolved")]):
    issue=""
    for st in (tr or []):
        if st.get("role")=="user":
            issue=str(st.get("content") or "")[:4000]; break
    if not issue: continue
    rows.append((iid, f"{repo} {issue}", 1-int(bool(res)), iid))
if rows:
    out["C2"]=run("C2", [r[0] for r in rows], [r[1] for r in rows],
                  [r[2] for r in rows], [r[3] for r in rows])

json.dump(out,open("FINAL/charac/static_baseline.json","w"),indent=2)
print("[saved]")
