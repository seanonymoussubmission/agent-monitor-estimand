#!/usr/bin/env python3
"""An external capability axis for the capability-legibility trend.

Capability has so far been each model's mean score on the same corpus in which legibility
is measured. We have already split by task (script 71, check 4), but tasks within a corpus
share a scaffold and a scoring rubric, so a sceptic can still call the two axes the same
measurement. C3 carries a domain label over ten application areas, which lets us do
something stronger: rank models by how well they do in one set of *domains* and measure
legibility of their failures in a disjoint set. A model's competence in, say, research
tasks is then being used to predict how readable its failures are in unrelated ones.
"""
import argparse, collections, json, math, re
import numpy as np, pyarrow.parquet as pq
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default="data/liveclaw/data/v0.2.1-00000-of-00001.parquet")
ap.add_argument("--splits", type=int, default=300)
ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)
ERRRE = re.compile(r"error|failed|not found|cannot|unable|exception|denied", re.I)

t = pq.read_table(a.parquet, columns=["model_name","case_id","score","trajectory","domain"]).to_pylist()
rows=[]
for d in t:
    if d["score"] is None: continue
    try: steps=json.loads(d["trajectory"]).get("steps",[])
    except Exception: continue
    acts,ol,er=[],[],[]
    for st in steps:
        if st.get("source")!="agent": continue
        tc=st.get("tool_calls") or []
        nm=str(tc[0].get("function_name") or "no_tool") if (isinstance(tc,list) and tc
            and isinstance(tc[0],dict)) else "no_tool"
        acts.append(nm); ol.append(len(str(st.get("message") or "")))
        er.append(int(bool(ERRRE.search(str(st.get("observation") or "")))))
    if len(acts)>=2: rows.append((d["model_name"], d["case_id"], float(d["score"]),
                                  str(d.get("domain") or "?"), acts, ol, er))
model=np.array([r[0] for r in rows]); case=np.array([r[1] for r in rows])
score=np.array([r[2] for r in rows]); dom=np.array([r[3] for r in rows])
SEQ=[r[4] for r in rows]; OL=[r[5] for r in rows]; ER=[r[6] for r in rows]
unit=np.array([f"{m}||{c}" for m,c in zip(model,case)])
MODELS=sorted(set(model.tolist())); DOMS=sorted(set(dom.tolist()))
print(f"[data] {len(rows)} traces | {len(MODELS)} models | {len(DOMS)} domains")
for d_ in DOMS: print(f"    {d_:28s} {(dom==d_).sum():5d} traces")

def fsm_of(idx):
    tr=collections.defaultdict(collections.Counter)
    for i in idx:
        s="<init>"
        for x in SEQ[i]: tr[s][x]+=1; s=x
    o={}
    for s,c in tr.items():
        keep={x:n for x,n in c.items() if n>=2 or len(c)==1} or dict(c)
        tt=sum(keep.values()); o[s]={x:n/tt for x,n in keep.items()}
    al=set(); [al.update(c) for c in tr.values()]
    return o, sorted(al)
def feat(i,fs,al,ai):
    seq,ol,er=SEQ[i],OL[i],ER[i]; n=len(seq)
    v=np.zeros(len(al))
    for x in seq:
        j=ai.get(x)
        if j is not None: v[j]+=1
    v/=max(n,1)
    FL=1e-4; sur,s=[],"<init>"
    for x in seq: sur.append(-math.log(max(fs.get(s,{}).get(x,FL),FL))); s=x
    sur=np.array(sur) if sur else np.array([0.0]); h=max(1,len(sur)//2)
    ol=np.array(ol,float) if ol else np.array([0.0]); er=np.array(er,float) if er else np.array([0.0])
    return np.concatenate([v,[sur.mean(),sur.max(),float((sur>3).mean()),
        abs(sur[:h].mean()-sur[h:].mean()) if len(sur)>1 else 0.0,
        n,len(set(seq)),1.0-len(set(seq))/max(n,1),
        ol.mean(),ol.max(),ol.std(),er.mean(),er.sum()]])

oof=np.zeros(len(rows))
for tr,te in GroupKFold(n_splits=5).split(np.zeros(len(rows)), score, case):
    fs,al=fsm_of(tr); ai={x:j for j,x in enumerate(al)}
    m=HistGradientBoostingRegressor(max_iter=200,max_depth=3,random_state=0)
    m.fit(np.vstack([feat(i,fs,al,ai) for i in tr]), score[tr])
    oof[te]=m.predict(np.vstack([feat(i,fs,al,ai) for i in te]))

byu=collections.defaultdict(list)
for i,u in enumerate(unit): byu[u].append(i)
def legibility(mdl, doms):
    num=den=0.0
    for u,ix in byu.items():
        if model[ix[0]]!=mdl or dom[ix[0]] not in doms: continue
        for i in ix:
            for j in ix:
                if score[i]>score[j]:
                    num+=(oof[i]>oof[j])+0.5*(oof[i]==oof[j]); den+=1
    return (num/den if den else np.nan)
def capability(mdl, doms):
    m=(model==mdl)&np.isin(dom,list(doms))
    return float(score[m].mean()) if m.sum() else np.nan

print(f"\n=== capability and legibility measured on DISJOINT domains ===")
rs=[]
for s_ in range(a.splits):
    r=np.random.default_rng(3000+s_)
    perm=list(DOMS); r.shuffle(perm); h=len(perm)//2
    A,B=set(perm[:h]), set(perm[h:])
    cap=np.array([capability(m,A) for m in MODELS])
    leg=np.array([legibility(m,B) for m in MODELS])
    ok=~np.isnan(cap)&~np.isnan(leg)
    if ok.sum()>=10: rs.append(stats.spearmanr(cap[ok],leg[ok]).statistic)
rs=np.array([x for x in rs if not np.isnan(x)])
print(f"  {len(rs)} domain splits")
print(f"  Spearman(capability in domains A, legibility in domains B) = {rs.mean():+.3f}")
print(f"    95% interval [{np.percentile(rs,2.5):+.3f}, {np.percentile(rs,97.5):+.3f}]")
print(f"    negative in {100*(rs<0).mean():.1f}% of splits")
if a.out:
    json.dump(dict(mean=float(rs.mean()), lo=float(np.percentile(rs,2.5)),
                   hi=float(np.percentile(rs,97.5)), frac_neg=float((rs<0).mean()),
                   n_splits=int(len(rs))), open(a.out,"w"), indent=2)
    print(f"[saved] {a.out}")
