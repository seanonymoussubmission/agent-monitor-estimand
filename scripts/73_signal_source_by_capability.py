#!/usr/bin/env python3
"""Does the SOURCE of run-level signal shift as models get more capable?

Behavioural dispersion falls with capability but does not mediate the legibility decline
(script 72), and fail-vs-success separation in action distributions shows no capability
trend at all. What that leaves is a change in kind rather than degree: weak models may fail
*procedurally*, in which actions they take, while strong models execute a clean-looking
procedure and fail on the content they produce -- invisible at the action level.

That predicts the composition of the signal changes with capability, not just its size. We
test it by splitting the 17 C3 models into capability terciles and running the same
feature-family ablation within each, so we can see which family carries the signal for weak
models and whether any family carries it for strong ones.

Predictor is fitted per tercile and cross-fitted by task; legibility is within-unit
concordance on the continuous score, as elsewhere.
"""
import argparse, collections, json, math, re
import numpy as np, pyarrow.parquet as pq
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default="data/liveclaw/data/v0.2.1-00000-of-00001.parquet")
ap.add_argument("--boot", type=int, default=1000)
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
        nm = str(tc[0].get("function_name") or "no_tool") if (isinstance(tc, list) and tc
             and isinstance(tc[0], dict)) else "no_tool"
        acts.append(nm); ol.append(len(str(st.get("message") or "")))
        er.append(int(bool(ERRRE.search(str(st.get("observation") or "")))))
    if len(acts) >= 2: rows.append((d["model_name"], d["case_id"], float(d["score"]), acts, ol, er))

model = np.array([r[0] for r in rows]); case = np.array([r[1] for r in rows])
score = np.array([r[2] for r in rows])
SEQ = [r[3] for r in rows]; OL = [r[4] for r in rows]; ER = [r[5] for r in rows]
unit = np.array([f"{m}||{c}" for m, c in zip(model, case)])
CAP = {m: float(score[model == m].mean()) for m in sorted(set(model.tolist()))}
order = sorted(CAP, key=CAP.get)
T = len(order) // 3
GROUPS = [("weak",   order[:T+1]),
          ("middle", order[T+1:2*T+1]),
          ("strong", order[2*T+1:])]
print(f"[data] {len(rows)} traces | {len(order)} models")
for nm, ms in GROUPS:
    print(f"  {nm:7s} n={len(ms):2d} cap {CAP[ms[0]]:.3f}-{CAP[ms[-1]]:.3f}: {', '.join(ms)}")

def fsm_of(idx):
    tr = collections.defaultdict(collections.Counter)
    for i in idx:
        s = "<init>"
        for x in SEQ[i]: tr[s][x] += 1; s = x
    out = {}
    for s, c in tr.items():
        keep = {x: n for x, n in c.items() if n >= 2 or len(c) == 1} or dict(c)
        tt = sum(keep.values()); out[s] = {x: n/tt for x, n in keep.items()}
    al = set(); [al.update(c) for c in tr.values()]
    return out, sorted(al)

def fam(i, fsm, alpha, aidx):
    seq, ol, er = SEQ[i], OL[i], ER[i]; n = len(seq)
    v = np.zeros(len(alpha))
    for x in seq:
        j = aidx.get(x)
        if j is not None: v[j] += 1
    v /= max(n, 1)
    FL = 1e-4; sur, s = [], "<init>"
    for x in seq:
        sur.append(-math.log(max(fsm.get(s, {}).get(x, FL), FL))); s = x
    sur = np.array(sur) if sur else np.array([0.0]); h = max(1, len(sur)//2)
    ol = np.array(ol, float) if ol else np.array([0.0])
    er = np.array(er, float) if er else np.array([0.0])
    return {
      "which actions": v,
      "surprise":      np.array([sur.mean(), sur.max(), float((sur > 3.0).mean()),
                                 abs(sur[:h].mean()-sur[h:].mean()) if len(sur) > 1 else 0.0]),
      "repetition":    np.array([n, len(set(seq)), 1.0-len(set(seq))/max(n,1)]),
      "output size":   np.array([ol.mean(), ol.max(), ol.std()]),
      "errors seen":   np.array([er.mean(), er.sum()]),
    }
FAMS = ["which actions", "surprise", "repetition", "output size", "errors seen"]

def conc(idx, pred):
    loc = {i: k for k, i in enumerate(idx)}
    byu = collections.defaultdict(list)
    for i in idx: byu[unit[i]].append(i)
    num = den = 0.0; pl = []
    for u, ix in byu.items():
        for i in ix:
            for j in ix:
                if score[i] > score[j]:
                    c = (pred[loc[i]] > pred[loc[j]]) + 0.5*(pred[loc[i]] == pred[loc[j]])
                    num += c; den += 1; pl.append((u, c))
    return (num/den if den else np.nan), pl

def evaluate(idx, use):
    idx = np.asarray(idx)
    pred = np.zeros(len(idx))
    for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(idx)), score[idx], case[idx]):
        fsm, alpha = fsm_of(idx[tr]); aidx = {x: j for j, x in enumerate(alpha)}
        mk = lambda ii: np.vstack([np.concatenate([fam(i,fsm,alpha,aidx)[f] for f in use]) for i in idx[ii]])
        m = HistGradientBoostingRegressor(max_iter=200, max_depth=3, random_state=0)
        m.fit(mk(tr), score[idx[tr]]); pred[te] = m.predict(mk(te))
    c, pl = conc(idx, pred)
    us = sorted({u for u, _ in pl}); loc = collections.defaultdict(list)
    for u, v in pl: loc[u].append(v)
    bs = []
    for _ in range(a.boot):
        pick = rng.choice(len(us), len(us), replace=True)
        vals = [v for k in pick for v in loc[us[k]]]
        if vals: bs.append(float(np.mean(vals)))
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) > 20 else (np.nan, np.nan))
    return float(c), float(lo), float(hi), len(pl)

res = {}
for nm, ms in GROUPS:
    idx = np.where(np.isin(model, ms))[0]
    print(f"\n=== {nm.upper()} models (n={len(idx)} traces) ===")
    full = evaluate(idx, FAMS); res[f"{nm}_all"] = full
    print(f"  {'ALL FAMILIES':16s} {full[0]:.4f} [{full[1]:.4f},{full[2]:.4f}]  ({full[3]} pairs)")
    for f in FAMS:
        r = evaluate(idx, [f]); res[f"{nm}_{f}"] = r
        print(f"  {f:16s} {r[0]:.4f} [{r[1]:.4f},{r[2]:.4f}]")
if a.out: json.dump(res, open(a.out,"w"), indent=2); print(f"\n[saved] {a.out}")
