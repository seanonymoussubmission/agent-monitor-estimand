#!/usr/bin/env python3
"""Is the capability-legibility trend an artifact of which units are measurable?

Concordance is estimable only on units whose runs differ in score. Stronger models produce
more runs that all score 1.0, so their measurable stratum is smaller and may be atypical --
the principal-stratum problem this paper raises elsewhere, now sitting underneath our own
headline. Four checks, each removing a different explanation.

  1. PERMUTATION. Shuffle the capability labels across models and recompute the trend.
     Gives an exact null that does not assume anything about pair counts.
  2. COMMON TASKS. Restrict every model to the tasks on which many models are measurable,
     so no model is scored on its own idiosyncratic subset.
  3. MATCHED PAIRS. Subsample every model to the same number of ordered pairs, removing
     any dependence of the estimate's behaviour on how many comparisons it rests on.
  4. OUT-OF-SAMPLE CAPABILITY. Measure capability on one half of tasks and concordance on
     the other, so the x-axis is not computed from the same data as the y-axis.

If the trend survives all four it is not a composition effect.
"""
import argparse, collections, json, math, re
import numpy as np, pyarrow.parquet as pq
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default="data/liveclaw/data/v0.2.1-00000-of-00001.parquet")
ap.add_argument("--perm", type=int, default=20000)
ap.add_argument("--reps", type=int, default=400)
ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)
ERR = re.compile(r"error|failed|not found|cannot|unable|exception|denied", re.I)

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
        er.append(int(bool(ERR.search(str(st.get("observation") or "")))))
    if len(acts) >= 2: rows.append((d["model_name"], d["case_id"], float(d["score"]), acts, ol, er))

model = np.array([r[0] for r in rows]); case = np.array([r[1] for r in rows])
score = np.array([r[2] for r in rows])
SEQ = [r[3] for r in rows]; OL = [r[4] for r in rows]; ER = [r[5] for r in rows]
unit = np.array([f"{m}||{c}" for m, c in zip(model, case)])
MODELS = sorted(set(model.tolist()))
print(f"[data] {len(rows)} traces | {len(MODELS)} models | {len(set(case))} tasks")

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

def feat(i, fsm, alpha, aidx):
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
    return np.concatenate([v, [sur.mean(), sur.max(), float((sur > 3.0).mean()),
        abs(sur[:h].mean()-sur[h:].mean()) if len(sur) > 1 else 0.0,
        n, len(set(seq)), 1.0-len(set(seq))/max(n,1),
        ol.mean(), ol.max(), ol.std(), er.mean(), er.sum()]])

oof = np.zeros(len(rows))
for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(rows)), score, case):
    fsm, alpha = fsm_of(tr); aidx = {x: j for j, x in enumerate(alpha)}
    m = HistGradientBoostingRegressor(max_iter=200, max_depth=3, random_state=0)
    m.fit(np.vstack([feat(i, fsm, alpha, aidx) for i in tr]), score[tr])
    oof[te] = m.predict(np.vstack([feat(i, fsm, alpha, aidx) for i in te]))

# ordered pairs, precomputed once
PAIRS = collections.defaultdict(list)          # model -> list of (case, concordant?)
for u in set(unit):
    ix = np.where(unit == u)[0]
    m = model[ix[0]]; c = case[ix[0]]
    for i in ix:
        for j in ix:
            if score[i] > score[j]:
                PAIRS[m].append((c, (oof[i] > oof[j]) + 0.5*(oof[i] == oof[j])))
CAP = {m: float(score[model == m].mean()) for m in MODELS}
conc = lambda pl: float(np.mean([v for _, v in pl])) if pl else np.nan

base = np.array([conc(PAIRS[m]) for m in MODELS])
cap  = np.array([CAP[m] for m in MODELS])
r0 = stats.spearmanr(cap, base)
print(f"\n[base] rho = {r0.statistic:+.3f} (p={r0.pvalue:.4f})   "
      f"pairs/model {min(len(PAIRS[m]) for m in MODELS)}-{max(len(PAIRS[m]) for m in MODELS)}")

out = {"base_rho": float(r0.statistic), "base_p": float(r0.pvalue)}

# ---- 1. permutation on capability labels
print("\n=== 1. PERMUTATION (shuffle capability across models) ===")
null = [stats.spearmanr(rng.permutation(cap), base).statistic for _ in range(a.perm)]
null = np.array([x for x in null if not np.isnan(x)])
p = (1 + (np.abs(null) >= abs(r0.statistic)).sum()) / (len(null) + 1)
print(f"  null mean {null.mean():+.4f} sd {null.std():.4f} | exact p = {p:.5f}")
out["perm_p"] = float(p)

# ---- 2. common tasks
print("\n=== 2. COMMON TASKS (tasks where >= T models are measurable) ===")
bycase = collections.defaultdict(set)
for m in MODELS:
    for c, _ in PAIRS[m]: bycase[c].add(m)
for T in (8, 10, 12):
    keep = {c for c, ms in bycase.items() if len(ms) >= T}
    v = np.array([conc([(c, x) for c, x in PAIRS[m] if c in keep]) for m in MODELS])
    ok = ~np.isnan(v)
    if ok.sum() < 8: print(f"  T={T}: too few models"); continue
    r = stats.spearmanr(cap[ok], v[ok])
    print(f"  T={T:2d}: {len(keep):3d} tasks, {ok.sum()} models  rho={r.statistic:+.3f} p={r.pvalue:.4f}")
    out[f"common_T{T}"] = [float(r.statistic), float(r.pvalue), len(keep)]

# ---- 3. matched pair counts
print("\n=== 3. MATCHED PAIR COUNTS (subsample all models to the minimum) ===")
nmin = min(len(PAIRS[m]) for m in MODELS)
rs = []
for _ in range(a.reps):
    v = np.array([conc([PAIRS[m][k] for k in rng.choice(len(PAIRS[m]), nmin, replace=False)])
                  for m in MODELS])
    rs.append(stats.spearmanr(cap, v).statistic)
rs = np.array([x for x in rs if not np.isnan(x)])
print(f"  n={nmin} pairs each, {len(rs)} draws: rho = {rs.mean():+.3f} "
      f"[{np.percentile(rs,2.5):+.3f}, {np.percentile(rs,97.5):+.3f}]")
out["matched_rho"] = [float(rs.mean()), float(np.percentile(rs,2.5)), float(np.percentile(rs,97.5))]

# ---- 4. out-of-sample capability
print("\n=== 4. OUT-OF-SAMPLE CAPABILITY (capability and concordance on disjoint tasks) ===")
cases = np.array(sorted(set(case.tolist()))); rs2 = []
for _ in range(a.reps):
    pm = rng.permutation(len(cases)); h = len(cases)//2
    A = set(cases[pm[:h]]); B = set(cases[pm[h:]])
    capB = np.array([score[(model == m) & np.isin(case, list(A))].mean() for m in MODELS])
    vB = np.array([conc([(c, x) for c, x in PAIRS[m] if c in B]) for m in MODELS])
    ok = ~np.isnan(capB) & ~np.isnan(vB)
    if ok.sum() >= 8: rs2.append(stats.spearmanr(capB[ok], vB[ok]).statistic)
rs2 = np.array([x for x in rs2 if not np.isnan(x)])
print(f"  {len(rs2)} splits: rho = {rs2.mean():+.3f} "
      f"[{np.percentile(rs2,2.5):+.3f}, {np.percentile(rs2,97.5):+.3f}]  "
      f"P(rho<0) = {(rs2<0).mean():.3f}")
out["oos_rho"] = [float(rs2.mean()), float(np.percentile(rs2,2.5)), float(np.percentile(rs2,97.5)),
                  float((rs2<0).mean())]
if a.out: json.dump(out, open(a.out,"w"), indent=2); print(f"\n[saved] {a.out}")
