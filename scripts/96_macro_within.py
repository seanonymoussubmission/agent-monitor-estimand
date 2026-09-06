#!/usr/bin/env python3
"""P1: pair-weighted vs unit-uniform (macro) vs deployment-weighted within-task on C1/C2.

The main tables use pair-weighted within-task AUROC. The review asks whether the
weighting drives the result. We recompute at k=10 with the automaton featuriser under
three weightings:
  pair       : each success-failure pair counts once (reported)
  macro      : each mixed unit's own AUROC counts equally
  deployment : each TASK counts equally (average over models within a task, then tasks)
"""
import argparse, glob, json, math, os, re, collections
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
ap=argparse.ArgumentParser(); ap.add_argument("--corpus",required=True); a=ap.parse_args()
CMD=re.compile(r"```(?:bash|sh)?\s*\n?\s*([A-Za-z_][\w\-.]*)",re.M)
ERR=re.compile(r"traceback|error|not found|failed|exception|denied",re.I)
SWE={"open","goto","scroll_down","scroll_up","create","edit","insert","append","search_dir",
     "search_file","find_file","submit","python","python3","pytest","ls","cd","cat","grep",
     "find","git","pip","echo","rm","mv","cp","chmod","which","head","tail","sed","awk","diff","make","bash"}
def c1acts(tr):
    a_=[];
    for st in tr:
        if st.get("role")=="ai":
            m=CMD.search(st.get("text") or ""); t=(m.group(1).lower() if m else "no_command")
            a_.append(t if t in SWE else ("no_command" if t=="no_command" else "other_cmd"))
    return a_
def c2acts(tr):
    a_=[]
    for st in tr:
        if st.get("role")=="assistant":
            tc=st.get("tool_calls") or []
            nm=(tc[0].get("function") or {}).get("name") if (tc and isinstance(tc[0],dict) and isinstance(tc[0].get("function"),dict)) else (tc[0].get("name") if tc and isinstance(tc[0],dict) else "no_tool")
            a_.append(str(nm or "no_tool"))
    return a_
rows=[]
if a.corpus=="c1":
    for p in sorted(glob.glob("data/swe_agent_traj/data/*.parquet")):
        df=pd.read_parquet(p,columns=["instance_id","model_name","target","trajectory"])
        for iid,mdl,tgt,tr in df.itertuples(index=False):
            ac=c1acts(list(tr))
            if len(ac)>=10: rows.append((f"{mdl}||{iid}", iid, int(not bool(tgt)), ac[:10]))
        del df
else:
    import pyarrow.parquet as pq
    t=pq.read_table("data/swe_rebench/trajectories.parquet",columns=["instance_id","trajectory","resolved"])
    for iid,tr,res in zip(*[t.column(c).to_pylist() for c in ("instance_id","trajectory","resolved")]):
        ac=c2acts(tr or [])
        if len(ac)>=10: rows.append((iid, iid, int(not bool(res)), ac[:10]))
unit=np.array([r[0] for r in rows]); task=np.array([r[1] for r in rows]); y=np.array([r[2] for r in rows])
SEQ=[r[3] for r in rows]
print(f"[{a.corpus}] {len(rows)} traces >=10 turns | {len(set(unit))} units")
def fsm(idx):
    tr=collections.defaultdict(collections.Counter)
    for i in idx:
        s="<init>"
        for x in SEQ[i]: tr[s][x]+=1; s=x
    o={}
    for s,c in tr.items():
        k={x:n for x,n in c.items() if n>=2 or len(c)==1} or dict(c); z=sum(k.values()); o[s]={x:n/z for x,n in k.items()}
    al=set(); [al.update(c) for c in tr.values()]; return o,sorted(al)
def feat(i,fs,al,ai):
    seq=SEQ[i]; n=len(seq); v=np.zeros(len(al))
    for x in seq:
        q=ai.get(x)
        if q is not None: v[q]+=1
    v/=max(n,1)
    FL=1e-4; sur,s=[],"<init>"
    for x in seq: sur.append(-math.log(max(fs.get(s,{}).get(x,FL),FL))); s=x
    sur=np.array(sur) if sur else np.array([0.0])
    return np.concatenate([v,[sur.mean(),sur.max(),n,len(set(seq)),1-len(set(seq))/max(n,1)]])
oof=np.zeros(len(rows))
for tr,te in GroupKFold(5).split(np.zeros(len(rows)),y,task):
    fs,al=fsm(tr); ai={x:q for q,x in enumerate(al)}
    c=HistGradientBoostingClassifier(max_iter=150,max_depth=3,random_state=0)
    c.fit(np.vstack([feat(i,fs,al,ai) for i in tr]),y[tr])
    oof[te]=c.predict_proba(np.vstack([feat(i,fs,al,ai) for i in te]))[:,1]
def uauc(mask_units):
    vals=[]; sizes=[]
    for u in mask_units:
        m=unit==u; P,N=oof[m&(y==1)],oof[m&(y==0)]
        if not len(P) or not len(N): continue
        au=sum((p>q)+0.5*(p==q) for p in P for q in N)/(len(P)*len(N)); vals.append(au); sizes.append(len(P)*len(N))
    return np.array(vals),np.array(sizes)
mixed=[u for u in set(unit) if len(set(y[unit==u]))>1]
v,sz=uauc(mixed)
pair=np.sum(v*sz)/np.sum(sz); macro=np.mean(v)
# deployment: average unit AUCs within a task, then over tasks
bytask=collections.defaultdict(list)
for u,val in zip(mixed,v): bytask[u.split("||")[-1]].append(val)
deploy=np.mean([np.mean(x) for x in bytask.values()])
print(f"  within-task k=10: pair={pair:.4f}  macro={macro:.4f}  deployment={deploy:.4f}  ({len(mixed)} units)")
json.dump({"pair":float(pair),"macro":float(macro),"deployment":float(deploy)},
          open(f"FINAL/charac/macro_{a.corpus}.json","w"))
