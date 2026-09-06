#!/usr/bin/env python3
"""What does reading the trajectory add over simply knowing which task it is?

We have been answering RQ2 partly through the variance decomposition, saying that only the
within-unit share of outcome variance is available to a prefix predictor. That is an
over-reading: an ICC bounds variance, not AUROC, and it says nothing directly about what a
predictor gains. The direct measurement is an information increment, so we compute one.

Three nested predictors, all cross-fitted:
  M0  TASK ONLY     the unit's failure rate estimated from its OTHER runs (leave-one-out).
                    This is the label-only oracle: it knows exactly how hard the task is
                    for this model and nothing whatever about the run in front of it.
  M1  PREFIX ONLY   trajectory features, no unit information.
  M2  TASK + PREFIX both.

The quantity of interest is M2 - M0: what the trajectory buys a monitor that already knows
the task. We report it under a discrimination measure (AUROC) and two proper scoring rules
(Brier, log loss), because a predictor can shift ranking without improving calibration and
proper scores are what a deployed policy actually pays for.
"""
import argparse, collections, json, math, re
import numpy as np, pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default="data/liveclaw/data/v0.2.1-00000-of-00001.parquet")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)
ERRRE = re.compile(r"error|failed|not found|cannot|unable|exception|denied", re.I)

t = pq.read_table(a.parquet, columns=["model_name","case_id","score","trajectory"]).to_pylist()
rows = []
for d in t:
    if d["score"] is None: continue
    try: steps = json.loads(d["trajectory"]).get("steps", [])
    except Exception: continue
    acts, ol, er = [], [], []
    for st in steps:
        if st.get("source") != "agent": continue
        tc = st.get("tool_calls") or []
        nm = str(tc[0].get("function_name") or "no_tool") if (isinstance(tc,list) and tc
             and isinstance(tc[0],dict)) else "no_tool"
        acts.append(nm); ol.append(len(str(st.get("message") or "")))
        er.append(int(bool(ERRRE.search(str(st.get("observation") or "")))))
    if len(acts) >= 2: rows.append((d["model_name"], d["case_id"], float(d["score"]), acts, ol, er))

model=np.array([r[0] for r in rows]); case=np.array([r[1] for r in rows])
fail=(np.array([r[2] for r in rows]) != 1.0).astype(int)
SEQ=[r[3] for r in rows]; OL=[r[4] for r in rows]; ER=[r[5] for r in rows]
unit=np.array([f"{m}||{c}" for m,c in zip(model,case)])
print(f"[data] {len(rows)} runs | {len(set(unit))} units | failure rate {fail.mean():.3f}")

# ---- M0: leave-one-out unit failure rate (task-only oracle)
loo = np.zeros(len(rows))
for u in set(unit):
    ix = np.where(unit == u)[0]
    tot = fail[ix].sum(); n = len(ix)
    for i in ix:
        loo[i] = (tot - fail[i]) / max(n - 1, 1) if n > 1 else fail.mean()
print(f"[M0] leave-one-out unit rate computed (no run uses its own label)")

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

def feat(i, fs, al, ai):
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

def crossfit(include_task, include_prefix):
    p=np.zeros(len(rows))
    for tr,te in GroupKFold(n_splits=5).split(np.zeros(len(rows)), fail, case):
        fs,al=fsm_of(tr); ai={x:j for j,x in enumerate(al)}
        def mk(ii):
            cols=[]
            if include_prefix: cols.append(np.vstack([feat(i,fs,al,ai) for i in ii]))
            if include_task:   cols.append(loo[ii].reshape(-1,1))
            return np.hstack(cols)
        m=HistGradientBoostingClassifier(max_iter=200,max_depth=3,random_state=0)
        m.fit(mk(tr), fail[tr]); p[te]=m.predict_proba(mk(te))[:,1]
    return np.clip(p, 1e-6, 1-1e-6)

M0=crossfit(True,False); M1=crossfit(False,True); M2=crossfit(True,True)

def scores(p):
    return (roc_auc_score(fail,p), brier_score_loss(fail,p), log_loss(fail,p))

print(f"\n{'model':22s} {'AUROC':>8} {'Brier':>8} {'logloss':>9}")
out={}
for nm,p in [("M0 task only",M0), ("M1 prefix only",M1), ("M2 task + prefix",M2)]:
    au,br,ll = scores(p); out[nm]=[au,br,ll]
    print(f"  {nm:20s} {au:8.4f} {br:8.4f} {ll:9.4f}")

def ci(fn):
    us=np.array(sorted(set(unit))); loc={u:np.where(unit==u)[0] for u in us}; v=[]
    for _ in range(a.boot):
        pick=rng.choice(len(us),len(us),replace=True)
        r=np.concatenate([loc[us[i]] for i in pick])
        if len(set(fail[r].tolist()))<2: continue
        try: v.append(fn(r))
        except Exception: pass
    return (np.percentile(v,2.5), np.percentile(v,97.5)) if len(v)>20 else (np.nan,np.nan)

print(f"\n=== INCREMENT FROM READING THE TRAJECTORY (M2 - M0) ===")
dA = scores(M2)[0]-scores(M0)[0]; dB = scores(M0)[1]-scores(M2)[1]; dL = scores(M0)[2]-scores(M2)[2]
loA,hiA = ci(lambda r: roc_auc_score(fail[r],M2[r])-roc_auc_score(fail[r],M0[r]))
loB,hiB = ci(lambda r: brier_score_loss(fail[r],M0[r])-brier_score_loss(fail[r],M2[r]))
loL,hiL = ci(lambda r: log_loss(fail[r],M0[r])-log_loss(fail[r],M2[r]))
print(f"  AUROC gain        {dA:+.4f}  95% CI [{loA:+.4f}, {hiA:+.4f}]")
print(f"  Brier reduction   {dB:+.4f}  95% CI [{loB:+.4f}, {hiB:+.4f}]")
print(f"  log-loss reduction{dL:+.4f}  95% CI [{loL:+.4f}, {hiL:+.4f}]")
out["increment"]=dict(auroc=[dA,loA,hiA], brier=[dB,loB,hiB], logloss=[dL,loL,hiL])
print("  -> " + ("the trajectory adds measurable information beyond task identity"
      if loA>0 and loB>0 else "no reliable gain over knowing the task alone"))
if a.out: json.dump(out, open(a.out,"w"), indent=2); print(f"[saved] {a.out}")

# --- appended: within-task AUROC per predictor (script 86 follow-up) ---
def _within(u, yy, sc):
    num=den=0.0
    for g in set(u):
        m=u==g; P,N=sc[m&(yy==1)], sc[m&(yy==0)]
        if not len(P) or not len(N): continue
        num+=sum((p>q)+0.5*(p==q) for p in P for q in N); den+=len(P)*len(N)
    return num/den if den else float("nan")
print("\n=== WITHIN-TASK AUROC (same predictions) ===")
for nm,p in [("M0 task only",M0),("M1 prefix only",M1),("M2 task + prefix",M2)]:
    print(f"  {nm:20s} within = {_within(unit, fail, p):.4f}")
