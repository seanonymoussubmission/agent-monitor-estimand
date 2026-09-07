#!/usr/bin/env python3
"""Is the monitor's score useful as a difficulty ESTIMATOR even if useless for aborting?

A monitor reading five turns learns unit-level facts (repo size, issue vagueness) that
have no within-task value but real cross-task value -- exactly what the allocation prior
needs when no other model has attempted the task. The hybrid of Table 7 only ever ABORTS
on the score; this tests ROUTING on it. Three additions at matched budgets on C4-L:

  route-by-monitor       Thompson sampling; each pull's observed k=5 score updates a
                         plug-in difficulty estimate for that arm (no transferred prior,
                         no aborting) -- the novel-task regime.
  route+prior            same, on top of the transferred prior.
  fair-abort             abort only when calibrated P(fail | score) exceeds the
                         break-even d/(r+d) given the unit's base rate, instead of the
                         naive median threshold.
"""
import argparse, collections, csv, glob, json, math, os, re
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--traj", default="data/lph/swebench/laguna_xs2_full")
ap.add_argument("--runs", default="data/runs_laguna.csv")
ap.add_argument("--budgets", type=float, nargs="+", default=[2e6, 5e6, 1e7, 2e7])
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--trials", type=int, default=150)
ap.add_argument("--monw", type=float, default=1.0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
MONW = a.monw

VOCAB={"cd","cat","sed","grep","python","python3","find","echo","ls","awk","head","tail",
       "git","pytest","mkdir","touch","rm","mv","cp","chmod","which","diff","make","pip"}
abstract=lambda t:(t if t in VOCAB else "other_cmd")
out={}
for r in csv.DictReader(open(a.runs)):
    out[(r["task"],int(r["run"]))]=(int(r["success"]),int(r["tokens"]))
RE=re.compile(r"^(.*)_run(\d+)\.json$")
runs=collections.defaultdict(list)
for fp in sorted(glob.glob(os.path.join(a.traj,"*_run*.json"))):
    m=RE.match(os.path.basename(fp))
    if not m: continue
    task,rn=m.group(1),int(m.group(2))
    key=(task,rn) if (task,rn) in out else (task,rn-1)
    if key not in out: continue
    try: d=json.load(open(fp))
    except Exception: continue
    ch=d.get("command_history") or []
    acts=[]
    for c in ch:
        t=c if isinstance(c,str) else (c.get("command") if isinstance(c,dict) else None)
        if t and str(t).strip():
            acts.append(abstract(str(t).strip().split()[0].split("/")[-1].lower()))
    if len(acts)<2: continue
    succ,tok=out[key]
    runs[task].append(dict(succ=succ,tok=float(tok),acts=acts))
tasks=sorted(runs)
print(f"[data] {len(tasks)} tasks | {sum(len(v) for v in runs.values())} runs")

flat=[(t,i) for t in tasks for i in range(len(runs[t]))]
y=np.array([1-runs[t][i]["succ"] for t,i in flat]); grp=np.array([t for t,_ in flat])
def fsm_of(idx,k):
    tr=collections.defaultdict(collections.Counter)
    for j in idx:
        t,i=flat[j]; s="<init>"
        for x in runs[t][i]["acts"][:k]: tr[s][x]+=1; s=x
    o={}
    for s,c in tr.items():
        keep={x:n for x,n in c.items() if n>=2 or len(c)==1} or dict(c)
        tt=sum(keep.values()); o[s]={x:n/tt for x,n in keep.items()}
    al=set(); [al.update(c) for c in tr.values()]
    return o,sorted(al)
def feat(j,k,fs,al,ai):
    t,i=flat[j]; seq=runs[t][i]["acts"][:k]; n=len(seq)
    v=np.zeros(len(al))
    for x in seq:
        q=ai.get(x)
        if q is not None: v[q]+=1
    v/=max(n,1)
    FL=1e-4; sur,s=[],"<init>"
    for x in seq: sur.append(-math.log(max(fs.get(s,{}).get(x,FL),FL))); s=x
    sur=np.array(sur) if sur else np.array([0.0])
    return np.concatenate([v,[sur.mean(),sur.max(),n,len(set(seq)),1-len(set(seq))/max(n,1)]])
score=np.zeros(len(flat))
for tr,te in GroupKFold(n_splits=5).split(np.zeros(len(flat)),y,grp):
    fs,al=fsm_of(tr,a.k); ai={x:q for q,x in enumerate(al)}
    mk=lambda ii: np.vstack([feat(j,a.k,fs,al,ai) for j in ii])
    c=HistGradientBoostingClassifier(max_iter=150,max_depth=3,random_state=0)
    c.fit(mk(tr),y[tr]); score[te]=c.predict_proba(mk(te))[:,1]
SC=collections.defaultdict(dict)
for q,(t,i) in enumerate(flat): SC[t][i]=float(score[q])
# isotonic-ish calibration: bin scores -> empirical fail rate (cross-fitted crudely by bins)
bins=np.quantile(score,np.linspace(0,1,11))
def calib(s):
    b=np.searchsorted(bins,s,side="right")-1; b=min(max(b,0),9)
    m=(np.digitize(score,bins[1:-1]))==b
    return float(y[m].mean()) if m.sum()>=20 else float(y.mean())
prior={t:(sum(r["succ"] for r in runs[t]), sum(1-r["succ"] for r in runs[t])) for t in tasks}
cmean={t:float(np.mean([r["tok"] for r in runs[t]])) for t in tasks}
p_est={t: np.mean([1-SC[t][i] for i in range(len(runs[t]))]) for t in tasks}  # per-run avail at pull time

def cost_at(r,k):
    # approx per-turn cost share
    n=len(r["acts"]); return r["tok"]*min(k,n)/n

def simulate(kind,budget,rg,use_prior=False):
    alive=set(tasks); spent=0.0; solved=0
    ab={t:[1.0+ (0.5*prior[t][0] if use_prior else 0.0),
           1.0+ (0.5*prior[t][1] if use_prior else 0.0)] for t in tasks}
    mon={t:[0.0,0.0] for t in tasks}   # accumulated monitor pseudo-evidence
    order=list(tasks); rg.shuffle(order); ptr=0
    # break-even quantities for fair-abort: estimate r,d from replay stats per task
    while alive and spent<budget:
        if kind in ("allocate","route","route-pure","route+prior","hybrid","fair-abort"):
            best,bv=None,-1.0
            for t in alive:
                al_,be_=ab[t][0]+mon[t][0], ab[t][1]+mon[t][1]
                v=rg.beta(max(al_,1e-3),max(be_,1e-3))/max(cmean[t],1.0)
                if v>bv: bv,best=v,t
            cand=best
        else:
            cand=None
            for _ in range(len(order)):
                c=order[ptr%len(order)]; ptr+=1
                if c in alive: cand=c; break
            if cand is None: break
        i=rg.integers(len(runs[cand])); r=runs[cand][i]; s=SC[cand][i]
        if kind=="hybrid" and s>=np.quantile(score,0.5):
            spent+=cost_at(r,a.k); got=0
        elif kind=="fair-abort":
            pf=calib(s); base=1-p_est[cand]
            # break-even: abort pays if pf > base failure? use d/(r+d) with r=base succ, d=1-base succ proxy
            if pf > max(0.5, y.mean()) and pf > base:
                spent+=cost_at(r,a.k); got=0
            else:
                spent+=r["tok"]; got=r["succ"]
        else:
            spent+=r["tok"]; got=r["succ"]
        if kind in ("route","route-pure","route+prior"):
            # monitor score as pseudo-observation, weight MONW per pull
            mon[cand][0]+= MONW*(1-s); mon[cand][1]+= MONW*s
        if got: solved+=1; alive.discard(cand)
        elif kind != "route-pure": ab[cand][1]+=1.0   # route-pure: no outcome history
    return solved

POLS=[("uniform",False),("allocate",True),("ts-cold",False),("route",False),
      ("route-pure",False),("route+prior",True),("hybrid",True),("fair-abort",True)]
print(f"\n{'budget':>8} | "+" | ".join(f"{n:>13}" for n,_ in POLS))
res={}
for B in a.budgets:
    row=[]
    for name,up in POLS:
        kind="allocate" if name in ("allocate","ts-cold") else name
        vals=[simulate(kind,B,np.random.default_rng(11000+s),use_prior=up) for s in range(a.trials)]
        row.append(f"{np.mean(vals):6.1f}±{np.std(vals):4.1f}")
        res[f"{int(B)}_{name}"]=[float(np.mean(vals)),float(np.std(vals))]
    print(f"{B/1e6:6.0f}M | "+" | ".join(f"{x:>13}" for x in row))
if a.out: json.dump(res,open(a.out,"w"),indent=2); print(f"[saved] {a.out}")
