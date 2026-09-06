#!/usr/bin/env python3
"""Does within-task legibility of failure depend on the acting model?

C1 hints that weaker models fail more legibly -- within-task AUROC at ten turns is 0.636
for Llama-8B, 0.581 for 70B and 0.569 for 405B -- but two of those rest on 74 and 44
mixed units. C3 is the corpus that can actually test it: 17 models over 134 tasks, three
runs of every (model, task) pair, one scaffold throughout.

Two design choices matter.

First, C3 scores runs on a continuous [0,1] scale, and the rest of this paper binarises at
score = 1. Binarising here would discard most of the available comparisons: it leaves
about a thousand within-unit pairs. Ranking on the raw score instead uses all three runs
of every unit -- roughly 6,800 ordered pairs -- and within-unit concordance on a
continuous outcome is the direct generalisation of within-task AUROC. We report the
binarised version too, for comparability.

Second, one predictor is trained across all models (cross-fitted by task) and then
evaluated separately per model, rather than fitting seventeen small models. Differences
then reflect how legible each model's failures are under a common reader, not how much
training data each model happened to have.
"""
import argparse, collections, json, math
import numpy as np, pyarrow.parquet as pq
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default="data/liveclaw/data/v0.2.1-00000-of-00001.parquet")
ap.add_argument("--prefix", type=int, default=0, help="0 = full trajectory, else first k steps")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)
ERR = __import__("re").compile(r"error|failed|not found|cannot|unable|exception|denied", __import__("re").I)

t = pq.read_table(a.parquet, columns=["model_name","case_id","score","trajectory",
                                      "difficulty","domain"]).to_pylist()
rows = []
for d in t:
    if d["score"] is None: continue
    try: steps = json.loads(d["trajectory"]).get("steps", [])
    except Exception: continue
    acts, ol, er = [], [], []
    for st in steps:
        if st.get("source") != "agent": continue
        tc = st.get("tool_calls") or []
        nm = "no_tool"
        if isinstance(tc, list) and tc and isinstance(tc[0], dict):
            nm = str(tc[0].get("function_name") or "no_tool")
        acts.append(nm); ol.append(len(str(st.get("message") or "")))
        er.append(int(bool(ERR.search(str(st.get("observation") or "")))))
    if len(acts) >= 2:
        rows.append((d["model_name"], d["case_id"], float(d["score"]), acts, ol, er))

model = np.array([r[0] for r in rows]); case = np.array([r[1] for r in rows])
score = np.array([r[2] for r in rows])
SEQ = [r[3] for r in rows]; OL = [r[4] for r in rows]; ER = [r[5] for r in rows]
unit = np.array([f"{m}||{c}" for m, c in zip(model, case)])
print(f"[data] {len(rows)} traces | {len(set(model))} models | {len(set(case))} tasks | "
      f"{len(set(unit))} units | mean score {score.mean():.3f}")

def fsm_of(idx, k):
    tr = collections.defaultdict(collections.Counter)
    for i in idx:
        s = "<init>"
        for x in (SEQ[i][:k] if k else SEQ[i]): tr[s][x] += 1; s = x
    out = {}
    for s, c in tr.items():
        keep = {x: n for x, n in c.items() if n >= 2 or len(c) == 1} or dict(c)
        tt = sum(keep.values()); out[s] = {x: n/tt for x, n in keep.items()}
    al = set(); [al.update(c) for c in tr.values()]
    return out, sorted(al)

def feat(i, k, fsm, alpha, aidx):
    seq = SEQ[i][:k] if k else SEQ[i]
    ol = (OL[i][:k] if k else OL[i]); er = (ER[i][:k] if k else ER[i])
    n = len(seq); v = np.zeros(len(alpha))
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
    return np.concatenate([v, [sur.mean(), sur.max(), float((sur > 3.0).mean()),
        abs(sur[:h].mean()-sur[h:].mean()) if len(sur) > 1 else 0.0,
        n, len(set(seq)), 1.0-len(set(seq))/max(n,1),
        ol.mean(), ol.max(), ol.std(), er.mean(), er.sum()]])

oof = np.zeros(len(rows))
for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(rows)), score, case):
    fsm, alpha = fsm_of(tr, a.prefix); aidx = {x: j for j, x in enumerate(alpha)}
    Xtr = np.vstack([feat(i, a.prefix, fsm, alpha, aidx) for i in tr])
    Xte = np.vstack([feat(i, a.prefix, fsm, alpha, aidx) for i in te])
    m = HistGradientBoostingRegressor(max_iter=200, max_depth=3, random_state=0)
    m.fit(Xtr, score[tr]); oof[te] = m.predict(Xte)
print(f"[fit] cross-fitted by task, {len(set(case))} task groups\n")

def conc_cont(mask):
    """P(prediction ranks the better run above the worse | same unit), continuous score."""
    num = den = 0.0
    for u in set(unit[mask]):
        ix = np.where(mask & (unit == u))[0]
        for i in ix:
            for j in ix:
                if score[i] > score[j]:
                    num += (oof[i] > oof[j]) + 0.5*(oof[i] == oof[j]); den += 1
    return (num/den if den else float("nan")), den

def conc_bin(mask):
    fail = (score != 1.0).astype(int)
    num = den = 0.0
    for u in set(unit[mask]):
        ix = np.where(mask & (unit == u))[0]
        P = oof[ix][fail[ix] == 0]; N = oof[ix][fail[ix] == 1]
        num += sum((p > q) + 0.5*(p == q) for p in P for q in N); den += len(P)*len(N)
    return (num/den if den else float("nan")), den

allm = np.ones(len(rows), bool)
c_all, n_all = conc_cont(allm); b_all, nb_all = conc_bin(allm)
print(f"=== POOLED OVER MODELS ===")
print(f"  continuous within-unit concordance = {c_all:.4f}  ({int(n_all)} ordered pairs)")
print(f"  binarised within-task AUROC        = {b_all:.4f}  ({int(nb_all)} pairs)\n")

print(f"=== PER MODEL (capability = mean score) ===")
recs = []
for m in sorted(set(model.tolist())):
    msk = model == m
    c, n = conc_cont(msk); b, nb = conc_bin(msk)
    recs.append(dict(model=m, cap=float(score[msk].mean()), conc=float(c),
                     pairs=int(n), binauc=float(b), binpairs=int(nb)))
recs.sort(key=lambda r: r["cap"])
for r in recs:
    print(f"  {r['model']:20s} cap={r['cap']:.3f}  within(cont)={r['conc']:.4f} "
          f"({r['pairs']:4d} pairs)  within(bin)={r['binauc']:.4f}")

cap = np.array([r["cap"] for r in recs]); wc = np.array([r["conc"] for r in recs])
rho = stats.spearmanr(cap, wc)
bs = [stats.spearmanr(cap[k], wc[k]).statistic
      for k in (rng.integers(0, len(cap), len(cap)) for _ in range(a.boot))]
bs = [x for x in bs if not np.isnan(x)]
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\n=== TREND: does legibility fall as the model gets stronger? ===")
print(f"  Spearman(capability, within-unit concordance) over {len(cap)} models")
print(f"    rho = {rho.statistic:+.3f}  p = {rho.pvalue:.4f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
print("  -> " + ("weaker models fail MORE legibly" if rho.statistic < 0 and rho.pvalue < .05
                 else "no significant capability trend"))
if a.out:
    json.dump(dict(pooled_cont=c_all, pooled_bin=b_all, models=recs,
                   rho=float(rho.statistic), p=float(rho.pvalue), ci=[float(lo), float(hi)]),
              open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}")
