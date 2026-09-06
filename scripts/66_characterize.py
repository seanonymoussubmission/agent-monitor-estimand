#!/usr/bin/env python3
"""When does a trajectory become predictive of failure, and does that depend on cohort?

Two questions, and the first one is a check on our own claim.

(1) We report that within-task AUROC rises with prefix depth. But the sample changes
    with depth: at turn 10 only runs that survived to turn 10 are scored. If short runs
    are disproportionately successes, the rise could be composition rather than signal.
    We therefore compute the curve twice -- once on the varying sample, and once on a
    FIXED COHORT of traces long enough to appear at every depth. Only the fixed-cohort
    curve licenses a statement about signal emerging over time.

(2) Given a trustworthy curve, does it differ by scaffold (C1 vs C2) and by acting model
    (C1 has three model scales on one scaffold and benchmark, so that comparison holds
    scaffold, benchmark and featuriser constant)?

Featurisation follows the Automata reimplementation in 63_automata_reeval.py so the two
sets of numbers are comparable.
"""
import argparse, collections, glob, json, math, os, re
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", choices=["c1", "c2"], default="c1")
ap.add_argument("--depths", type=int, nargs="+", default=[1, 3, 5, 10])
ap.add_argument("--min-turns", type=int, default=10, help="fixed-cohort threshold")
ap.add_argument("--by-model", action="store_true")
ap.add_argument("--shards", type=int, default=12)
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)

CMD = re.compile(r"```(?:bash|sh)?\s*\n?\s*([A-Za-z_][\w\-.]*)", re.M)
ERR = re.compile(r"traceback|error|not found|no such file|command not found|failed|"
                 r"exception|invalid|denied", re.I)
SWE_CMDS = {"open","goto","scroll_down","scroll_up","create","edit","insert","append",
            "search_dir","search_file","find_file","submit","python","python3","pytest",
            "ls","cd","cat","grep","find","git","pip","echo","rm","mv","cp","mkdir",
            "touch","chmod","which","head","tail","sed","awk","diff","make","bash"}

def acts_c1(traj):
    acts, ol, er = [], [], []
    for st in traj:
        role, txt = st.get("role"), (st.get("text") or "")
        if role == "ai":
            m = CMD.search(txt); tok = (m.group(1).lower() if m else "no_command")
            acts.append(tok if tok in SWE_CMDS else
                        ("no_command" if tok == "no_command" else "other_cmd"))
            ol.append(len(txt)); er.append(0)
        elif role == "user" and acts:
            ol[-1] += len(txt); er[-1] = int(bool(ERR.search(txt)))
    return acts, ol, er

def acts_c2(traj):
    """OpenHands: the activity is the tool function name."""
    acts, ol, er = [], [], []
    for st in traj:
        role = st.get("role"); txt = str(st.get("content") or "")
        if role == "assistant":
            tc = st.get("tool_calls") or []
            name = "no_tool"
            if len(tc):
                f = tc[0]
                if isinstance(f, dict):
                    fn = f.get("function") or {}
                    name = (fn.get("name") if isinstance(fn, dict) else None) or f.get("name") or "no_tool"
            acts.append(str(name)); ol.append(len(txt)); er.append(0)
        elif role in ("tool", "user") and acts:
            ol[-1] += len(txt); er[-1] = int(bool(ERR.search(txt)))
    return acts, ol, er

# ---------------------------------------------------------------- load
rows = []
if a.corpus == "c1":
    for s in range(a.shards):
        p = f"data/swe_agent_traj/data/train-{s:05d}-of-00012.parquet"
        if not os.path.exists(p): continue
        df = pd.read_parquet(p, columns=["instance_id","model_name","target","trajectory"])
        for iid, mdl, tgt, tr in df.itertuples(index=False):
            try: A, L, E = acts_c1(list(tr))
            except Exception: continue
            if len(A) >= 2: rows.append((f"{mdl}||{iid}", iid, mdl, int(not bool(tgt)), A, L, E))
        del df
        print(f"[load] shard {s}: {len(rows)}", flush=True)
else:
    df = pd.read_parquet("data/swe_rebench/trajectories.parquet",
                         columns=["instance_id","repo","trajectory","resolved"])
    for iid, repo, tr, res in df.itertuples(index=False):
        try: A, L, E = acts_c2(list(tr))
        except Exception: continue
        if len(A) >= 2: rows.append((iid, iid, "openhands", int(not bool(res)), A, L, E))
    del df
    print(f"[load] c2: {len(rows)}", flush=True)

unit  = np.array([r[0] for r in rows]); group = np.array([r[1] for r in rows])
model = np.array([r[2] for r in rows]); y = np.array([r[3] for r in rows])
SEQ = [r[4] for r in rows]; OL = [r[5] for r in rows]; ER = [r[6] for r in rows]
nturn = np.array([len(s) for s in SEQ])
print(f"[data] {len(y)} traces | {len(set(unit))} units | {len(set(group))} tasks | "
      f"fail {y.mean():.3f} | median turns {np.median(nturn):.0f}", flush=True)

def fit_fsm(idx, k):
    trans = collections.defaultdict(collections.Counter)
    for i in idx:
        s = "<init>"
        for act in SEQ[i][:k]:
            trans[s][act] += 1; s = act
    kept = {}
    for s, c in trans.items():
        keep = {x: n for x, n in c.items() if n >= 2 or len(c) == 1} or dict(c)
        tot = sum(keep.values()); kept[s] = {x: n/tot for x, n in keep.items()}
    alpha = set(); [alpha.update(c) for c in trans.values()]
    return kept, sorted(alpha)

def feat(i, k, fsm, alpha, aidx):
    seq, ol, er = SEQ[i][:k], OL[i][:k], ER[i][:k]
    n = len(seq)
    v = np.zeros(len(alpha))
    for act in seq:
        j = aidx.get(act)
        if j is not None: v[j] += 1
    v /= max(n, 1)
    FL = 1e-4; sur, s = [], "<init>"
    for act in seq:
        sur.append(-math.log(max(fsm.get(s, {}).get(act, FL), FL))); s = act
    sur = np.array(sur) if sur else np.array([0.0]); h = max(1, len(sur)//2)
    ol = np.array(ol, float) if ol else np.array([0.0])
    er = np.array(er, float) if er else np.array([0.0])
    return np.concatenate([v, [
        sur.mean(), sur.max(),
        abs(sur[:h].mean()-sur[h:].mean()) if len(sur) > 1 else 0.0,
        float(np.exp(-sur.max())), float((sur > 3.0).mean()),
        n, len(set(seq)), 1.0 - len(set(seq))/max(n,1),
        ol.mean(), ol.max(), ol.std(), er.mean(), er.sum(),
        float(np.mean(np.diff(sur))) if len(sur) > 1 else 0.0]])

def within(u, yy, s):
    num = den = 0.0; nu = 0
    for g in set(u):
        m = u == g; P, N = s[m & (yy == 1)], s[m & (yy == 0)]
        if not len(P) or not len(N): continue
        nu += 1; num += sum((p > q) + 0.5*(p == q) for p in P for q in N); den += len(P)*len(N)
    return (num/den if den else float("nan")), den, nu

def evaluate(mask, k, label):
    idx = np.where(mask)[0]
    if len(idx) < 200: return None
    yy, gg, uu = y[idx], group[idx], unit[idx]
    if len(set(yy)) < 2: return None
    oof = np.zeros(len(idx))
    for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(idx)), yy, gg):
        fsm, alpha = fit_fsm(idx[tr], k)
        aidx = {x: j for j, x in enumerate(alpha)}
        Xtr = np.vstack([feat(idx[i], k, fsm, alpha, aidx) for i in tr])
        Xte = np.vstack([feat(idx[i], k, fsm, alpha, aidx) for i in te])
        c = HistGradientBoostingClassifier(max_iter=200, max_depth=3, random_state=0)
        c.fit(Xtr, yy[tr]); oof[te] = c.predict_proba(Xte)[:, 1]
    pool = roc_auc_score(yy, oof); w, npair, nu = within(uu, yy, oof)
    # cluster bootstrap over units for the within estimate
    units = np.array(sorted(set(uu))); loc = {u: np.where(uu == u)[0] for u in units}
    bs = []
    for _ in range(a.boot):
        pick = rng.choice(len(units), len(units), replace=True)
        r = np.concatenate([loc[units[i]] for i in pick])
        t2 = np.concatenate([np.full(len(loc[units[i]]), f"b{n}") for n, i in enumerate(pick)])
        v, d, _ = within(t2, yy[r], oof[r])
        if not np.isnan(v): bs.append(v)
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) > 20 else (np.nan, np.nan))
    print(f"  {label:26s} k={k:<3} n={len(idx):6d} pooled={pool:.4f} "
          f"within={w:.4f} [{lo:.4f},{hi:.4f}] units={nu}", flush=True)
    return dict(label=label, k=k, n=len(idx), pooled=float(pool), within=float(w),
                ci=[float(lo), float(hi)], units=nu, pairs=int(npair))

res = []
print(f"\n=== VARYING SAMPLE (what a naive depth sweep reports) ===")
for k in a.depths:
    res.append(evaluate(nturn >= k, k, "varying"))
    print(f"     (traces with >= {k} turns: {(nturn>=k).mean():.3f})")

print(f"\n=== FIXED COHORT (traces with >= {a.min_turns} turns, held constant) ===")
coh = nturn >= a.min_turns
print(f"     cohort = {coh.sum()} traces ({coh.mean():.3f} of corpus), "
      f"fail rate {y[coh].mean():.3f} vs {y.mean():.3f} overall")
for k in a.depths:
    res.append(evaluate(coh, k, "fixed-cohort"))

if a.by_model:
    print(f"\n=== BY ACTING MODEL, fixed cohort (scaffold + benchmark held constant) ===")
    for m in sorted(set(model.tolist())):
        for k in a.depths:
            res.append(evaluate(coh & (model == m), k, m[:24]))

if a.out:
    json.dump([r for r in res if r], open(a.out, "w"), indent=2)
    print(f"\n[saved] {a.out}")
