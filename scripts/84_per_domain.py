#!/usr/bin/env python3
"""Does cross-task domination of pooled AUROC occur outside coding tasks?

Every corpus with our richest repeated-run structure is SWE-bench-adjacent, and a reviewer
will ask whether the whole phenomenon is a coding artifact. C3 already contains the answer:
its 134 tasks span ten application domains, only one of which is software development. We
compute pooled and within-task discrimination separately per domain, with the predictor
cross-fitted by task exactly as elsewhere. If the pooled-within gap appears in e-commerce,
health and social-media tasks, the phenomenon is not about code.
"""
import argparse, collections, json, math, re
import numpy as np, pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default="data/liveclaw/data/v0.2.1-00000-of-00001.parquet")
ap.add_argument("--boot", type=int, default=1000)
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
    if len(acts)>=2:
        rows.append((d["model_name"], d["case_id"], float(d["score"]),
                     str(d.get("domain") or "?"), acts, ol, er))
model=np.array([r[0] for r in rows]); case=np.array([r[1] for r in rows])
score=np.array([r[2] for r in rows]); dom=np.array([r[3] for r in rows])
SEQ=[r[4] for r in rows]; OL=[r[5] for r in rows]; ER=[r[6] for r in rows]
unit=np.array([f"{m}||{c}" for m,c in zip(model,case)])
fail=(score != 1.0).astype(int)
print(f"[data] {len(rows)} runs | {len(set(dom))} domains")

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
pred_fail = -oof   # higher predicted score = less likely to fail

def pooled_auc(mask):
    P=pred_fail[mask&(fail==1)]; N=pred_fail[mask&(fail==0)]
    if not len(P) or not len(N): return np.nan
    return float(np.mean([[ (p>q)+0.5*(p==q) for q in N] for p in P]))
def within_auc(mask):
    num=den=0.0; nu=0
    for u in set(unit[mask]):
        m=mask&(unit==u); P=pred_fail[m&(fail==1)]; N=pred_fail[m&(fail==0)]
        if not len(P) or not len(N): continue
        nu+=1; num+=sum((p>q)+0.5*(p==q) for p in P for q in N); den+=len(P)*len(N)
    return (num/den if den else np.nan), nu

print(f"\n{'domain':28s} {'runs':>5} {'pooled':>8} {'within':>8} {'gap':>7} {'mixed':>6}")
res=[]
for d_ in sorted(set(dom.tolist())):
    m=dom==d_
    po=pooled_auc(m); wi,nu=within_auc(m)
    coding = "coding" in d_.lower() or "devops" in d_.lower()
    res.append(dict(domain=d_, runs=int(m.sum()), pooled=po, within=wi, units=nu,
                    coding=coding))
    print(f"  {d_:26s} {m.sum():5d} {po:8.4f} {wi:8.4f} {po-wi:+7.4f} {nu:6d}")
cod=[r for r in res if r["coding"]]; non=[r for r in res if not r["coding"]]
print(f"\n  coding/devops domains ({len(cod)}):  mean pooled {np.mean([r['pooled'] for r in cod]):.4f}  "
      f"mean within {np.mean([r['within'] for r in cod]):.4f}")
print(f"  non-coding domains  ({len(non)}):  mean pooled {np.mean([r['pooled'] for r in non]):.4f}  "
      f"mean within {np.mean([r['within'] for r in non]):.4f}")
print(f"  gap present in {sum(1 for r in res if r['pooled']-r['within']>0.02)}/{len(res)} domains")
if a.out: json.dump(res, open(a.out,"w"), indent=2); print(f"[saved] {a.out}")
