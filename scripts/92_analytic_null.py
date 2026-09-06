#!/usr/bin/env python3
"""T2: closed-form permutation-null SD for within-task AUROC, validated per corpus.

Under H0 the within-unit Mann-Whitney statistic for a unit with P failures and N
successes has exact null variance (P+N+1)/(12 P N); the pair-weighted average therefore
has null SD  sqrt(sum (P_u N_u)^2 Var_u) / sum P_u N_u = sqrt(sum P_u N_u (P_u+N_u+1)/12)
/ sum P_u N_u -- computable from the outcome table, no permutations needed. Compare with
the empirically reported permutation SDs.

T4-light: precision-weighted capability slope. Per-model within-unit concordance A_m has
approximate sampling variance (P+N+1)-style over its pair count; regress A_m on
capability with inverse-variance weights and report the weighted slope +/- SE.
"""
import glob, csv, json
import numpy as np, pyarrow.parquet as pq

def null_sd(unit_ids, fails):
    unit_ids=np.asarray(unit_ids); fails=np.asarray(fails,int)
    units,inv=np.unique(unit_ids,return_inverse=True)
    out=[]
    for u in range(len(units)):
        m=inv==u; P=int(fails[m].sum()); N=int((~fails[m].astype(bool)).sum())
        if P>0 and N>0: out.append((P,N))
    num=sum(P*N*(P+N+1)/12 for P,N in out)
    den=sum(P*N for P,N in out)
    return np.sqrt(num)/den, len(out), den

# C2
t=pq.read_table("data/swe_rebench/trajectories.parquet",columns=["instance_id","resolved"])
sd,nu,pairs=null_sd([str(x) for x in t.column("instance_id").to_pylist()],
                    [1-int(bool(x)) for x in t.column("resolved").to_pylist()])
print(f"C2   analytic null SD = {sd:.4f}  (reported permutation: 0.0057)  units={nu} pairs={pairs}")
# C3
t=pq.read_table("data/liveclaw/data/v0.2.1-00000-of-00001.parquet",
                columns=["model_name","case_id","score"])
u=[str(a)+"||"+str(b) for a,b in zip(t.column("model_name").to_pylist(),t.column("case_id").to_pylist())]
f=[1 if (x is None or x!=1.0) else 0 for x in t.column("score").to_pylist()]
sd,nu,pairs=null_sd(u,f)
print(f"C3   analytic null SD = {sd:.4f}  (reported permutation: 0.0177-0.0190) units={nu} pairs={pairs}")
# C4 splits
for nm,path,emp in (("C4-L","data/runs_laguna.csv","0.021"),("C4-Q","data/runs_qwen.csv","~0.018")):
    u,f=[],[]
    for r_ in csv.DictReader(open(path)):
        u.append(r_["task"]); f.append(1-int(r_["success"]))
    sd,nu,pairs=null_sd(u,f)
    print(f"{nm} analytic null SD = {sd:.4f}  (reported permutation: {emp}) units={nu} pairs={pairs}")

# ---- T4-light: precision-weighted capability slope on C3 per-model concordances
d=json.load(open("FINAL/charac/scale.json"))["models"]
cap=np.array([r["cap"] for r in d]); A=np.array([r["conc"] for r in d])
npair=np.array([r["pairs"] for r in d],float)
var=(A*(1-A))/npair + 1e-9          # binomial-style concordance variance proxy
w=1/var
b=np.polyfit(cap,A,1,w=w)[0]
# SE via weighted least squares formula
X=np.vstack([cap,np.ones_like(cap)]).T
XtWX=X.T@(w[:,None]*X); cov=np.linalg.inv(XtWX)
resid=A-X@np.linalg.solve(XtWX,X.T@(w*A))
s2=np.sum(w*resid**2)/(len(A)-2)
se=np.sqrt(s2*cov[0,0])
print(f"\nT4  precision-weighted slope d(within)/d(cap) = {b:+.3f} +/- {se:.3f}  "
      f"(z={b/se:+.2f})")
