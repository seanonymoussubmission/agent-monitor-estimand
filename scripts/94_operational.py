#!/usr/bin/env python3
"""P0: decision-centric monitor evaluation on C4-L.

A within-task AUROC is a ranking metric; deployment needs an operating point under
asymmetric costs. For the cross-fitted k=5 automaton monitor we sweep the abort
threshold and report, per success-retention level, the compute it saves and the
failures it catches -- the curve a practitioner actually reads.

abort rule: at turn 5, abort runs whose monitor failure-score >= tau.
  success retention = P(not aborted | would-succeed)
  false-abort rate  = 1 - retention
  failure recall    = P(aborted | would-fail)
  compute saved     = tokens after turn 5 in aborted runs / total tokens
"""
import glob, json, math, os, re, csv, collections
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

VOCAB={"cd","cat","sed","grep","python","python3","find","echo","ls","awk","head","tail",
       "git","pytest","mkdir","touch","rm","mv","cp","chmod","which","diff","make","pip"}
abst=lambda t:(t if t in VOCAB else "other_cmd")
out={}
for r in csv.DictReader(open("data/runs_laguna.csv")):
    out[(r["task"],int(r["run"]))]=(int(r["success"]),int(r["tokens"]))
RE=re.compile(r"^(.*)_run(\d+)\.json$")
rows=[]
for fp in sorted(glob.glob("data/lph/swebench/laguna_xs2_full/*_run*.json")):
    m=RE.match(os.path.basename(fp))
    if not m: continue
    task,rn=m.group(1),int(m.group(2))
    key=(task,rn) if (task,rn) in out else (task,rn-1)
    if key not in out: continue
    try: d=json.load(open(fp))
    except Exception: continue
    ch=d.get("command_history") or []
    acts,sizes=[],[]
    for c in ch:
        tt=c if isinstance(c,str) else (c.get("command") if isinstance(c,dict) else None)
        if tt and str(tt).strip():
            acts.append(abst(str(tt).strip().split()[0].split("/")[-1].lower())); sizes.append(len(str(tt)))
    if len(acts)<2: continue
    succ,tok=out[key]
    tot=max(sum(sizes),1); cum5=sum(sizes[:5])*tok/tot
    rows.append(dict(task=task,succ=succ,tok=float(tok),cost5=float(cum5),acts=acts))
tasks=np.array([r["task"] for r in rows]); y=np.array([1-r["succ"] for r in rows])
print(f"[data] {len(rows)} runs | {len(set(tasks))} tasks | fail {y.mean():.3f}")

# cross-fitted k=5 monitor score
def fsm(idx):
    tr=collections.defaultdict(collections.Counter)
    for i in idx:
        s="<init>"
        for x in rows[i]["acts"][:5]: tr[s][x]+=1; s=x
    o={}
    for s,c in tr.items():
        keep={x:n for x,n in c.items() if n>=2 or len(c)==1} or dict(c)
        z=sum(keep.values()); o[s]={x:n/z for x,n in keep.items()}
    al=set(); [al.update(c) for c in tr.values()]
    return o,sorted(al)
def feat(i,fs,al,ai):
    seq=rows[i]["acts"][:5]; n=len(seq); v=np.zeros(len(al))
    for x in seq:
        q=ai.get(x)
        if q is not None: v[q]+=1
    v/=max(n,1)
    FL=1e-4; sur,s=[],"<init>"
    for x in seq: sur.append(-math.log(max(fs.get(s,{}).get(x,FL),FL))); s=x
    sur=np.array(sur) if sur else np.array([0.0])
    return np.concatenate([v,[sur.mean(),sur.max(),n,len(set(seq)),1-len(set(seq))/max(n,1)]])
score=np.zeros(len(rows))
for tr,te in GroupKFold(5).split(np.zeros(len(rows)),y,tasks):
    fs,al=fsm(tr); ai={x:q for q,x in enumerate(al)}
    c=HistGradientBoostingClassifier(max_iter=150,max_depth=3,random_state=0)
    c.fit(np.vstack([feat(i,fs,al,ai) for i in tr]),y[tr])
    score[te]=c.predict_proba(np.vstack([feat(i,fs,al,ai) for i in te]))[:,1]

tok=np.array([r["tok"] for r in rows]); cost5=np.array([r["cost5"] for r in rows])
succ=(y==0); fail=(y==1); T=tok.sum()
print("\n=== operating curve: abort at turn 5 when score >= tau ===")
print(f"  {'retention':>10} {'false-abort':>12} {'fail-recall':>12} {'compute-saved':>14}")
res=[]
for ret in (0.99,0.97,0.95,0.90):
    # threshold so that P(not aborted | success) = ret  -> abort the (1-ret) highest-scoring successes
    thr=np.quantile(score[succ], ret)      # scores above thr among successes = 1-ret fraction
    aborted=score>=thr
    retention=1-(aborted&succ).sum()/succ.sum()
    frecall=(aborted&fail).sum()/fail.sum()
    saved=(cost5[aborted]).sum()  # tokens NOT spent after turn5... approx: full tok minus cost5 saved
    saved_tok=(tok[aborted]-cost5[aborted]).sum()/T
    res.append(dict(ret=float(retention),fa=float(1-retention),recall=float(frecall),saved=float(saved_tok)))
    print(f"  {retention:10.3f} {1-retention:12.3f} {frecall:12.3f} {saved_tok:14.3f}")
# failure recall at fixed 5% false-abort
thr=np.quantile(score[succ],0.95)
ab=score>=thr
print(f"\n  at 5% false-abort: failure recall = {(ab&fail).sum()/fail.sum():.3f}, "
      f"compute saved = {(tok[ab]-cost5[ab]).sum()/T:.3f}")
json.dump(res,open("FINAL/charac/operational.json","w"),indent=2); print("[saved]")
