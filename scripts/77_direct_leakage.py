#!/usr/bin/env python3
"""How much of a fractional-checkpoint monitor's accuracy is leaked trajectory length?

Our estimate of the leak so far is an interpolation: the 25% checkpoint observes a mean of
6.7 assistant turns yet scores above a fixed-10-turn protocol, and reading the fixed-turn
curve back to 6.7 turns puts the leak near 0.07. A reviewer can object that interpolating a
curve is not measuring a leak. This measures it three ways.

  (a) LENGTH-ONLY ORACLE. Total trajectory length T is not knowable while a run is in
      progress, but a fractional protocol reveals it exactly: observing the first 25% of a
      trajectory tells you T = 4 x (what you saw). So the within-task AUROC obtainable from
      T alone is the ceiling on what the leak can be worth.

  (b) MATCHED FIXED-TURN CONTROL. Rather than interpolating, we run a fixed-turn protocol
      at the depth the fractional protocol actually observes on average, so the two see the
      same amount of trajectory and differ only in whether T is recoverable.

  (c) FIXED-BUDGET PROTOCOL. Cutting at a fixed token budget is the leak-free alternative
      a practitioner can actually deploy, since it needs no knowledge of the future. We
      report it as the recommended protocol rather than only criticising the other two.

All three on C1, using the automaton featuriser.
"""
import argparse, collections, glob, json, math, os, re
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/swe_agent_traj/data")
ap.add_argument("--shards", type=int, default=12)
ap.add_argument("--boot", type=int, default=600)
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

rows = []
for s in range(a.shards):
    p = os.path.join(a.data, f"train-{s:05d}-of-00012.parquet")
    if not os.path.exists(p): continue
    df = pd.read_parquet(p, columns=["instance_id","model_name","target","trajectory"])
    for iid, mdl, tgt, tr in df.itertuples(index=False):
        try: A, L, E = parse(list(tr))
        except Exception: continue
        if len(A) >= 2: rows.append((f"{mdl}||{iid}", iid, int(not bool(tgt)), A, L, E))
    del df
unit = np.array([r[0] for r in rows]); group = np.array([r[1] for r in rows])
y = np.array([r[2] for r in rows])
SEQ=[r[3] for r in rows]; OL=[r[4] for r in rows]; ER=[r[5] for r in rows]
T   = np.array([len(s) for s in SEQ])                 # total turns  (not knowable live)
TOK = np.array([sum(o) for o in OL])                  # total chars  (not knowable live)
print(f"[data] {len(rows)} traces | mean T={T.mean():.1f} median={np.median(T):.0f}")

def within(u, yy, sc):
    num=den=0.0; nu=0
    for g in set(u):
        m=u==g; P,N=sc[m&(yy==1)], sc[m&(yy==0)]
        if not len(P) or not len(N): continue
        nu+=1; num+=sum((p>q)+0.5*(p==q) for p in P for q in N); den+=len(P)*len(N)
    return (num/den if den else float("nan")), den, nu

def boot(u, yy, sc, nb):
    us=np.array(sorted(set(u))); loc={x:np.where(u==x)[0] for x in us}; v=[]
    for _ in range(nb):
        pick=rng.choice(len(us),len(us),replace=True)
        r=np.concatenate([loc[us[i]] for i in pick])
        t2=np.concatenate([np.full(len(loc[us[i]]),f"b{n}") for n,i in enumerate(pick)])
        w,_,_=within(t2, yy[r], sc[r])
        if not np.isnan(w): v.append(w)
    return (np.percentile(v,2.5), np.percentile(v,97.5)) if len(v)>20 else (np.nan,np.nan)

# ---------- (a) the ceiling: what is T alone worth, within task?
wT,_,nuT = within(unit, y, T.astype(float))
loT,hiT = boot(unit, y, T.astype(float), a.boot)
print(f"\n=== (a) LENGTH-ONLY ORACLE (total turns T, which a fractional cut reveals) ===")
print(f"  within-task AUROC from T alone = {wT:.4f}  95% CI [{loT:.4f}, {hiT:.4f}]  ({nuT} units)")
wTok,_,_ = within(unit, y, TOK.astype(float)); loK,hiK = boot(unit, y, TOK.astype(float), a.boot)
print(f"  within-task AUROC from total tokens = {wTok:.4f}  [{loK:.4f}, {hiK:.4f}]")

def cut_turns(i,k): return SEQ[i][:k], OL[i][:k], ER[i][:k]
def cut_frac(i,f):
    n=max(1,int(round(len(SEQ[i])*f))); return SEQ[i][:n], OL[i][:n], ER[i][:n]
def cut_budget(i,b):
    c=0
    for n,o in enumerate(OL[i],1):
        c+=o
        if c>=b: return SEQ[i][:n], OL[i][:n], ER[i][:n]
    return SEQ[i], OL[i], ER[i]

def run(cutter, arg, tag):
    C=[cutter(i,arg) for i in range(len(rows))]
    obs=np.array([len(c[0]) for c in C])
    def fsm(idx):
        tr=collections.defaultdict(collections.Counter)
        for i in idx:
            s="<init>"
            for x in C[i][0]: tr[s][x]+=1; s=x
        o={}
        for s,c in tr.items():
            keep={x:n for x,n in c.items() if n>=2 or len(c)==1} or dict(c)
            tt=sum(keep.values()); o[s]={x:n/tt for x,n in keep.items()}
        al=set(); [al.update(c) for c in tr.values()]
        return o, sorted(al)
    def feat(i,fs,al,ai):
        seq,ol,er=C[i]; n=len(seq)
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
    for tr,te in GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, group):
        fs,al=fsm(tr); ai={x:j for j,x in enumerate(al)}
        mk=lambda ii: np.vstack([feat(i,fs,al,ai) for i in ii])
        c=HistGradientBoostingClassifier(max_iter=200,max_depth=3,random_state=0)
        c.fit(mk(tr), y[tr]); oof[te]=c.predict_proba(mk(te))[:,1]
    w,_,nu=within(unit,y,oof); lo,hi=boot(unit,y,oof,a.boot)
    print(f"  {tag:34s} obs={obs.mean():5.1f} turns  within={w:.4f} [{lo:.4f},{hi:.4f}]  "
          f"pooled={roc_auc_score(y,oof):.4f}")
    return dict(tag=tag, obs=float(obs.mean()), within=float(w), ci=[float(lo),float(hi)],
                pooled=float(roc_auc_score(y,oof)))

res={"T_only":[float(wT),float(loT),float(hiT)], "tok_only":[float(wTok),float(loK),float(hiK)]}
print(f"\n=== (b) MATCHED COMPARISON: fraction vs fixed-turn at the SAME observed depth ===")
r_f25 = run(cut_frac, 0.25, "fraction 25%")
m = int(round(r_f25["obs"]))
r_fx  = run(cut_turns, m, f"fixed {m} turns (matched)")
res["frac25"]=r_f25; res["fixed_matched"]=r_fx
leak = r_f25["within"] - r_fx["within"]
print(f"  --> LEAK at matched depth = {leak:+.4f} AUROC")
res["leak_matched"]=float(leak)

print(f"\n=== (c) FIXED TOKEN BUDGET (leak-free, deployable) ===")
med_chars = float(np.median([sum(OL[i][:m]) for i in range(len(rows))]))
r_b = run(cut_budget, med_chars, f"budget {int(med_chars)} chars")
res["budget"]=r_b
print(f"\n=== SUMMARY ===")
print(f"  ceiling obtainable from T alone      {wT:.4f}")
print(f"  fraction 25%  ({r_f25['obs']:.1f} turns)          {r_f25['within']:.4f}")
print(f"  fixed {m} turns (same depth)          {r_fx['within']:.4f}")
print(f"  fixed token budget ({int(med_chars)} chars)  {r_b['within']:.4f}")
print(f"  measured leak (fraction - matched)   {leak:+.4f}")
if a.out: json.dump(res, open(a.out,"w"), indent=2); print(f"[saved] {a.out}")
