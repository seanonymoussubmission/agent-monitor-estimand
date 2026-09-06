#!/usr/bin/env python3
"""Predict each benchmark's difficulty-oracle pooled AUROC from its outcome table alone.

Theory: score runs by their unit's true failure rate q_u. Failures are drawn with weight
q_u, successes with weight (1-q_u). With run weights r_u and mu = the run-weighted mean
failure rate, the oracle's pooled AUROC (same-unit ties counted half) is the plug-in

  A_pred = [ sum_{u,v} r_u r_v q_u (1-q_v) (1{q_u>q_v} + .5*1{q_u=q_v}) ]
           / [ (sum_u r_u q_u)(sum_v r_v (1-q_v)) ]

whose continuous simplification is  A ~= 1/2 + MD / (4 mu (1-mu)),  MD the (weighted)
Gini mean difference of the unit failure rates; and the same-unit weight satisfies
  w = E[q(1-q)] / (n_eff mu (1-mu)).

Validation: compare A_pred (computed from unit rates only) against the MEASURED
leave-one-out oracle pooled AUROC computed from the raw runs, on all five corpora/splits.
"""
import glob, json, csv
import numpy as np, pyarrow.parquet as pq

def analyse(name, unit_ids, fails):
    unit_ids=np.asarray(unit_ids); fails=np.asarray(fails,float)
    units, inv = np.unique(unit_ids, return_inverse=True)
    r = np.bincount(inv).astype(float)
    s = np.bincount(inv, weights=fails)
    q = s/r
    mu = fails.mean()
    # ---- plug-in prediction from the outcome table only
    W = r
    order = np.argsort(q, kind="mergesort")
    qs, Ws = q[order], W[order]
    fw, sw = Ws*qs, Ws*(1-qs)           # failure / success mass per unit
    F_tot, S_tot = fw.sum(), sw.sum()
    # sum over pairs q_u > q_v of fw_u*sw_v  (+ half for ties incl. same unit)
    csw = np.cumsum(sw)                  # success mass at strictly smaller-or-equal index
    num = 0.0
    i = 0
    while i < len(qs):
        j = i
        while j < len(qs) and qs[j]==qs[i]: j += 1
        below = csw[i-1] if i>0 else 0.0            # success mass with strictly smaller q
        tie_s = csw[j-1]-below                       # success mass in the tie block
        blk_f = fw[i:j].sum()
        num += blk_f*below + 0.5*blk_f*tie_s
        i = j
    A_pred = num/(F_tot*S_tot)
    # ---- closed-form pieces
    md = 0.0                                        # weighted Gini mean difference
    cW = np.cumsum(Ws); cQW = np.cumsum(qs*Ws); T=cW[-1]
    for i in range(len(qs)):                        # sum_{j<i} W_i W_j (q_i - q_j)
        if i>0: md += Ws[i]*(qs[i]*cW[i-1] - cQW[i-1])
    MD = 2*md/(T*T)
    A_md = 0.5 + MD/(4*mu*(1-mu))
    w_pred = float(np.sum(W*q*(1-q))/T) / (len(units)* (mu*(1-mu))) * len(units)/ ( ( ( ( (1) ) ) ) )
    # cleaner: w = sum r^2 q(1-q) / (F_tot*S_tot); with r runs each pair counted r_u^2
    w_exact = float(np.sum((r*q)*(r*(1-q))))/(F_tot*S_tot)
    # ---- measured LOO oracle from raw runs
    loo = np.where(r[inv]>1, (s[inv]-fails)/(r[inv]-1), mu)
    P, N = loo[fails==1], loo[fails==0]
    # measured pooled AUROC via rank method
    allv = np.concatenate([P,N]); rk = allv.argsort(kind="mergesort").argsort().astype(float)
    # handle ties by average ranks
    import scipy.stats as st
    rk = st.rankdata(allv)
    A_meas = (rk[:len(P)].sum() - len(P)*(len(P)+1)/2)/(len(P)*len(N))
    print(f"{name:8s} units={len(units):5d} mu={mu:.3f} MD={MD:.3f} | "
          f"pred(plug-in)={A_pred:.4f} pred(MD form)={A_md:.4f} | measured LOO={A_meas:.4f} "
          f"| w_pred={w_exact:.2e}")
    return dict(name=name, units=len(units), mu=float(mu), MD=float(MD),
                pred=float(A_pred), pred_md=float(A_md), meas=float(A_meas),
                w=float(w_exact))

out=[]
# C1
u,f=[],[]
for p in sorted(glob.glob("data/swe_agent_traj/data/*.parquet")):
    t=pq.read_table(p,columns=["instance_id","model_name","target"])
    for iid,m,tg in zip(*[t.column(c).to_pylist() for c in ("instance_id","model_name","target")]):
        u.append(f"{m}||{iid}"); f.append(1-int(bool(tg)))
out.append(analyse("C1",u,f))
# C2
t=pq.read_table("data/swe_rebench/trajectories.parquet",columns=["instance_id","resolved"])
u=[x for x in t.column("instance_id").to_pylist()]
f=[1-int(bool(x)) for x in t.column("resolved").to_pylist()]
out.append(analyse("C2",u,f))
# C3
t=pq.read_table("data/liveclaw/data/v0.2.1-00000-of-00001.parquet",
                columns=["model_name","case_id","score"])
u,f=[],[]
for m,c,sc in zip(*[t.column(x).to_pylist() for x in ("model_name","case_id","score")]):
    u.append(f"{m}||{c}"); f.append(1 if (sc is None or sc!=1.0) else 0)
out.append(analyse("C3",u,f))
# C4 splits
for nm,path in (("C4-L","data/runs_laguna.csv"),("C4-Q","data/runs_qwen.csv")):
    u,f=[],[]
    for r_ in csv.DictReader(open(path)):
        u.append(r_["task"]); f.append(1-int(r_["success"]))
    out.append(analyse(nm,u,f))
json.dump(out,open("FINAL/charac/predict_inflation.json","w"),indent=2)
print("[saved]")
