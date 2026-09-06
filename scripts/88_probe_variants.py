#!/usr/bin/env python3
"""Nonlinear (MLP) and mean-pooled probe variants, so 'you only tried linear last-token
probes' is not an escape hatch. C4-L, k=1 and k=10, layer 20."""
import numpy as np, json, argparse
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
ap=argparse.ArgumentParser()
ap.add_argument("--probe", default="FINAL/merged.npz")
ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--out", default=None)
a=ap.parse_args()
Z=np.load(a.probe,allow_pickle=False)
def within(t,yy,sc):
    num=den=0.0
    for u in set(t):
        m=t==u;P,N=sc[m&(yy==1)],sc[m&(yy==0)]
        if not len(P) or not len(N): continue
        num+=sum((p>q)+0.5*(p==q) for p in P for q in N); den+=len(P)*len(N)
    return num/den if den else float("nan")
res={}
for k in (1,10):
    sel=Z["k"]==k; task,y=Z["task"][sel],Z["y"][sel].astype(int)
    X=Z[f"H{a.layer}"][sel].astype(float)
    # mean-pool proxy: average hidden state over available cuts <= k for same run
    runs=Z["run"][sel]
    for name,mk in [("linear last-token", lambda: make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.01,max_iter=3000))),
                    ("MLP (64) last-token", lambda: make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(64,),max_iter=400,
                                      alpha=1e-2,random_state=0)))]:
        s=np.zeros(len(y))
        for tr,te in GroupKFold(n_splits=5).split(X,y,task):
            m=mk(); m.fit(X[tr],y[tr]); s[te]=m.predict_proba(X[te])[:,1]
        w=within(task,y,s); res[f"k{k} {name}"]=float(w)
        print(f"  k={k:2d} {name:22s} within={w:.4f}")
    # mean-pooled across turns 1..k (only meaningful for k=10)
    if k==10:
        key=[(t,r) for t,r in zip(task,runs)]
        selall=np.isin(Z["k"],[1,3,5,10])
        ta,ra,Ha=Z["task"][selall],Z["run"][selall],Z[f"H{a.layer}"][selall].astype(float)
        import collections
        acc=collections.defaultdict(list)
        for i in range(len(ta)): acc[(ta[i],ra[i])].append(Ha[i])
        Xp=np.vstack([np.mean(acc[kk],axis=0) for kk in key])
        s=np.zeros(len(y))
        for tr,te in GroupKFold(n_splits=5).split(Xp,y,task):
            m=make_pipeline(StandardScaler(),LogisticRegression(C=0.01,max_iter=3000))
            m.fit(Xp[tr],y[tr]); s[te]=m.predict_proba(Xp[te])[:,1]
        w=within(task,y,s); res["k10 mean-pooled linear"]=float(w)
        print(f"  k=10 {'mean-pooled linear':22s} within={w:.4f}")
if a.out: json.dump(res,open(a.out,"w"),indent=2)
