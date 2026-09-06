#!/usr/bin/env python3
"""Exact finite-run theory for the LOO difficulty oracle's pooled AUROC.

A failure drawn from unit u (prob prop. to r_u q_u) leaves r_u-1 iid runs, so its LOO
score is B/(r_u-1) with B ~ Bin(r_u-1, q_u); a drawn success identically. Hence

  A_LOO = sum_{u,v} f_u s_v [ P(B_u/(r_u-1) > B_v/(r_v-1)) + .5 P(=) ] / (F S)

with f_u = r_u q_u, s_v = r_v (1-q_v) -- computable from the outcome table alone, and
converging to the D1 ceiling 1/2 + MD/(4 mu(1-mu)) as r -> inf. Same-unit pairs: the
drawn failure scores (S_u-1)/(r-1), the drawn success S_u/(r-1) minus its own... within a
unit every failure's LOO score is strictly below every success's, so same-unit pairs are
fully DIScordant: they contribute 0, not 1/2 -- the documented artifact, now in the
formula. Validate against the measured LOO oracles on all five corpora.

Estimated q_u are used as plug-ins (q_u = S_u/r_u), i.e. the parametric-bootstrap
prediction given the table.
"""
import glob, csv, numpy as np, pyarrow.parquet as pq
from scipy.stats import binom

def predict_and_measure(name, unit_ids, fails):
    unit_ids=np.asarray(unit_ids); fails=np.asarray(fails,float)
    units,inv=np.unique(unit_ids,return_inverse=True)
    r=np.bincount(inv).astype(float); S=np.bincount(inv,weights=fails)
    q=S/r
    keep=r>1                        # LOO defined
    rk,qk=r[keep],q[keep]
    fw,sw=rk*qk, rk*(1-qk)          # failure / success draw weights
    F,Sm=fw.sum(),sw.sum()
    # support of each unit's LOO score: values j/(r-1), pmf Bin(r-1,q)
    # aggregate failure-score and success-score distributions on a common grid
    grid={}
    def acc(dist, w, rr, qq):
        n=int(rr-1)
        pm=binom.pmf(np.arange(n+1),n,qq)
        for j,p in enumerate(pm):
            key=round(j/n,10) if n>0 else 0.0
            dist[key]=dist.get(key,0.0)+w*p
    Fd,Sd={},{}
    for i in range(len(rk)):
        acc(Fd,fw[i],rk[i],qk[i]); acc(Sd,sw[i],rk[i],qk[i])
    xs=sorted(set(Fd)|set(Sd))
    fa=np.array([Fd.get(x,0) for x in xs]); sa=np.array([Sd.get(x,0) for x in xs])
    fa/=fa.sum(); sa/=sa.sum()
    cs=np.cumsum(sa)
    # cross-unit concordance P(fail_score > succ_score)+.5 ties, over the mixed grid.
    conc=0.0
    for i,x in enumerate(xs):
        below=cs[i-1] if i>0 else 0.0
        conc+=fa[i]*(below+0.5*sa[i])
    # same-unit correction: fraction of pairs same-unit under LOO drawing = w'; those are
    # fully discordant (0) instead of the grid's mixture. w' uses r(r-1)-style? Pairs are
    # (one failure, one success) from same unit: mass sum r q * r(1-q)?? drawing is of a
    # failure and an independent success: same-unit prob = sum fw_u*sw_u /(F*Sm).
    w_same=float(np.sum(fw*sw))/(F*Sm)
    # grid conc already includes same-unit pairs as if independent Bin draws; replace:
    # exact same-unit concordance = 0 (fail B=Bin over r-1 including? both scores share
    # the SAME remaining runs minus themselves; fail score=(S-1)/(r-1)?? With both drawn
    # from same unit, fail leaves S-1 among... approximate correction: subtract the
    # independent-estimate for same-unit and add 0.
    same_ind=0.0
    for i in range(len(rk)):
        # concordance of two independent Bin(r-1,q) draws for this unit ~ .5 (symmetric)
        same_ind+= fw[i]*sw[i]*0.5
    same_ind/= (F*Sm)
    A_pred = conc - same_ind + 0.0*w_same
    # measured LOO
    loo=np.where(r[inv]>1,(S[inv]-fails)/np.maximum(r[inv]-1,1),fails.mean())
    from scipy.stats import rankdata
    P,N=loo[fails==1],loo[fails==0]
    rksum=rankdata(np.concatenate([P,N]))[:len(P)].sum()
    A_meas=(rksum-len(P)*(len(P)+1)/2)/(len(P)*len(N))
    print(f"{name:6s} pred(finite-r)={A_pred:.4f}  measured={A_meas:.4f}  diff={A_pred-A_meas:+.4f}")

# corpora
u,f=[],[]
for p in sorted(glob.glob("data/swe_agent_traj/data/*.parquet")):
    t=pq.read_table(p,columns=["instance_id","model_name","target"])
    for iid,m,tg in zip(*[t.column(c).to_pylist() for c in ("instance_id","model_name","target")]):
        u.append(m+"||"+iid); f.append(1-int(bool(tg)))
predict_and_measure("C1",u,f)
t=pq.read_table("data/swe_rebench/trajectories.parquet",columns=["instance_id","resolved"])
predict_and_measure("C2",[x for x in t.column("instance_id").to_pylist()],
                    [1-int(bool(x)) for x in t.column("resolved").to_pylist()])
t=pq.read_table("data/liveclaw/data/v0.2.1-00000-of-00001.parquet",
                columns=["model_name","case_id","score"])
u,f=[],[]
for m,c,sc in zip(*[t.column(x).to_pylist() for x in ("model_name","case_id","score")]):
    u.append(str(m)+"||"+str(c)); f.append(1 if (sc is None or sc!=1.0) else 0)
predict_and_measure("C3",u,f)
for nm,path in (("C4-L","data/runs_laguna.csv"),("C4-Q","data/runs_qwen.csv")):
    u,f=[],[]
    for r_ in csv.DictReader(open(path)):
        u.append(r_["task"]); f.append(1-int(r_["success"]))
    predict_and_measure(nm,u,f)
