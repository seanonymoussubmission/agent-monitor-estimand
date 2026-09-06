#!/usr/bin/env python3
"""Part (b) of the leakage study: is the fractional protocol's signal length or content?

The matched control shows a fractional checkpoint outscores a fixed-turn protocol that sees
more of the trajectory, and the length-only oracle shows total length T is worth 0.614
within-task on its own. What neither shows is whether the fractional protocol's advantage
*is* that length, as opposed to some genuine content signal that happens to accompany it.
We separate them four ways on the same features, under the 25% protocol:

  ALL          every feature
  LENGTH ONLY  only the features that encode how much was observed, which under a
               fractional cut is a deterministic function of T
  CONTENT ONLY length features removed
  CONTENT PERMUTED  content features shuffled among runs of the same unit while the length
               features stay attached to their own run. Any surviving within-task
               discrimination cannot come from content, because content has been decoupled
               from the outcome within the unit.
"""
import argparse, collections, json, math, os, re
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/swe_agent_traj/data")
ap.add_argument("--shards", type=int, default=12)
ap.add_argument("--frac", type=float, default=0.25)
ap.add_argument("--boot", type=int, default=400)
ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)

CMD = re.compile(r"```(?:bash|sh)?\s*\n?\s*([A-Za-z_][\w\-.]*)", re.M)
ERR = re.compile(r"traceback|error|not found|no such file|command not found|failed|"
                 r"exception|invalid|denied", re.I)
SWE = {"open","goto","scroll_down","scroll_up","create","edit","insert","append",
       "search_dir","search_file","find_file","submit","python","python3","pytest",
       "ls","cd","cat","grep","find","git","pip","echo","rm","mv","cp","mkdir",
       "touch","chmod","which","head","tail","sed","awk","diff","make","bash"}

def parse(traj):
    acts, ol, er = [], [], []
    for st in traj:
        role, txt = st.get("role"), (st.get("text") or "")
        if role == "ai":
            m = CMD.search(txt); tok = (m.group(1).lower() if m else "no_command")
            acts.append(tok if tok in SWE else ("no_command" if tok=="no_command" else "other_cmd"))
            ol.append(len(txt)); er.append(0)
        elif role == "user" and acts:
            ol[-1] += len(txt); er[-1] = int(bool(ERR.search(txt)))
    return acts, ol, er

rows=[]
for s in range(a.shards):
    p=os.path.join(a.data, f"train-{s:05d}-of-00012.parquet")
    if not os.path.exists(p): continue
    df=pd.read_parquet(p, columns=["instance_id","model_name","target","trajectory"])
    for iid,mdl,tgt,tr in df.itertuples(index=False):
        try: A,L,E=parse(list(tr))
        except Exception: continue
        if len(A)>=2: rows.append((f"{mdl}||{iid}", iid, int(not bool(tgt)), A, L, E))
    del df
unit=np.array([r[0] for r in rows]); group=np.array([r[1] for r in rows])
y=np.array([r[2] for r in rows]); SEQ=[r[3] for r in rows]; OL=[r[4] for r in rows]; ER=[r[5] for r in rows]
print(f"[data] {len(rows)} traces, fraction protocol at {a.frac}")

CUT=[(SEQ[i][:max(1,int(round(len(SEQ[i])*a.frac)))],
      OL[i][:max(1,int(round(len(SEQ[i])*a.frac)))],
      ER[i][:max(1,int(round(len(SEQ[i])*a.frac)))]) for i in range(len(rows))]

def fsm_of(idx):
    tr=collections.defaultdict(collections.Counter)
    for i in idx:
        s="<init>"
        for x in CUT[i][0]: tr[s][x]+=1; s=x
    o={}
    for s,c in tr.items():
        keep={x:n for x,n in c.items() if n>=2 or len(c)==1} or dict(c)
        tt=sum(keep.values()); o[s]={x:n/tt for x,n in keep.items()}
    al=set(); [al.update(c) for c in tr.values()]
    return o, sorted(al)

def blocks(i, fs, al, ai):
    seq,ol,er=CUT[i]; n=len(seq)
    v=np.zeros(len(al))
    for x in seq:
        j=ai.get(x)
        if j is not None: v[j]+=1
    v/=max(n,1)
    FL=1e-4; sur,s=[],"<init>"
    for x in seq: sur.append(-math.log(max(fs.get(s,{}).get(x,FL),FL))); s=x
    sur=np.array(sur) if sur else np.array([0.0]); h=max(1,len(sur)//2)
    olv=np.array(ol,float) if ol else np.array([0.0]); erv=np.array(er,float) if er else np.array([0.0])
    LENGTH  = np.array([n, float(np.sum(olv)), float(olv.mean())])   # how much was observed
    CONTENT = np.concatenate([v, [sur.mean(), sur.max(), float((sur>3).mean()),
        abs(sur[:h].mean()-sur[h:].mean()) if len(sur)>1 else 0.0,
        len(set(seq))/max(n,1), 1.0-len(set(seq))/max(n,1),
        olv.std(), erv.mean()]])                                     # what was done
    return LENGTH, CONTENT

def within(u, yy, sc):
    num=den=0.0
    for g in set(u):
        m=u==g; P,N=sc[m&(yy==1)], sc[m&(yy==0)]
        if not len(P) or not len(N): continue
        num+=sum((p>q)+0.5*(p==q) for p in P for q in N); den+=len(P)*len(N)
    return num/den if den else np.nan

def boot(u,yy,sc,nb):
    us=np.array(sorted(set(u))); loc={x:np.where(u==x)[0] for x in us}; v=[]
    for _ in range(nb):
        pick=rng.choice(len(us),len(us),replace=True)
        r=np.concatenate([loc[us[i]] for i in pick])
        t2=np.concatenate([np.full(len(loc[us[i]]),f"b{n}") for n,i in enumerate(pick)])
        w=within(t2, yy[r], sc[r])
        if not np.isnan(w): v.append(w)
    return (np.percentile(v,2.5), np.percentile(v,97.5)) if len(v)>20 else (np.nan,np.nan)

def run(mode):
    oof=np.zeros(len(rows))
    for tr,te in GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, group):
        fs,al=fsm_of(tr); ai={x:j for j,x in enumerate(al)}
        B=[blocks(i,fs,al,ai) for i in range(len(rows))]
        if mode=="permuted":
            C=[b[1] for b in B]
            perm=list(range(len(rows)))
            byu=collections.defaultdict(list)
            for i in range(len(rows)): byu[unit[i]].append(i)
            for u,ix in byu.items():
                sh=list(ix); rng.shuffle(sh)
                for src,dst in zip(ix,sh): perm[src]=dst
            B=[(B[i][0], C[perm[i]]) for i in range(len(rows))]
        def mk(ii):
            if mode=="length":  return np.vstack([B[i][0] for i in ii])
            if mode=="content": return np.vstack([B[i][1] for i in ii])
            return np.vstack([np.concatenate(B[i]) for i in ii])
        c=HistGradientBoostingClassifier(max_iter=200,max_depth=3,random_state=0)
        c.fit(mk(tr), y[tr]); oof[te]=c.predict_proba(mk(te))[:,1]
    w=within(unit,y,oof); lo,hi=boot(unit,y,oof,a.boot)
    return float(w), float(lo), float(hi)

res={}
print(f"\n=== FRACTION {int(a.frac*100)}% PROTOCOL, feature decomposition (within-task) ===")
for mode,label in [("all","all features"),("length","length only (= T under a fractional cut)"),
                   ("content","content only (length removed)"),
                   ("permuted","content permuted within unit, length kept")]:
    w,lo,hi=run(mode); res[mode]=[w,lo,hi]
    print(f"  {label:44s} {w:.4f}  [{lo:.4f}, {hi:.4f}]")
print(f"\n  If 'content permuted' stays near 'all', the fractional protocol's apparent")
print(f"  run-level discrimination is the leaked length rather than anything the agent did.")
if a.out: json.dump(res, open(a.out,"w"), indent=2); print(f"[saved] {a.out}")
