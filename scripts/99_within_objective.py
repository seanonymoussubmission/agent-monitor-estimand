#!/usr/bin/env python3
"""E1 (pre-registered): does a WITHIN-UNIT training objective recover early run-level
signal that the pooled objective misses?

Every predictor in the paper is fit with a pooled loss, which is maximised by learning
task difficulty. This script trains, on the SAME features and the SAME GroupKFold-by-task
folds:
  (a) pooled     - standard logistic regression on all training runs
  (b) ranker     - same-unit pairwise Bradley-Terry: logistic on within-unit feature
                   differences (fail - success), no intercept; the conditional-logit
                   estimator for pairs. Sees no between-unit contrasts at all.
  (c) demeaned   - features demeaned within unit (no labels used), pooled logistic.
and evaluates all three on identical held-out runs: pooled AUROC, pair-weighted
within-task AUROC over mixed units, within-unit permutation p, Holm across depths.
Controls: injected outcome feature -> within ~1.0; within-unit label shuffle -> ~0.5.

Data: C1 (SWE-agent trajectories), loader and featuriser identical to
63_automata_reeval.py. Decision rule pre-registered in PREREGISTRATION.md.
"""
import argparse, collections, glob, json, math, os, re
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/swe_agent_traj/data")
ap.add_argument("--shards", type=int, default=12)
ap.add_argument("--depths", type=int, nargs="+", default=[1, 3, 5, 10])
ap.add_argument("--perms", type=int, default=1000)
ap.add_argument("--rare-min", type=int, default=2)
ap.add_argument("--out", default=None)
a = ap.parse_args()

CMD = re.compile(r"```(?:bash|sh)?\s*\n?\s*([A-Za-z_][\w\-.]*)", re.M)
SWE_CMDS = {
    "open", "goto", "scroll_down", "scroll_up", "create", "edit", "insert", "append",
    "search_dir", "search_file", "find_file", "submit", "python", "python3", "pytest",
    "ls", "cd", "cat", "grep", "find", "git", "pip", "echo", "rm", "mv", "cp", "mkdir",
    "touch", "chmod", "which", "head", "tail", "sed", "awk", "diff", "make", "bash",
}
def abstract(tok):
    t = (tok or "").lower()
    return t if t in SWE_CMDS else ("no_command" if t == "no_command" else "other_cmd")
ERR = re.compile(r"traceback|error|not found|no such file|command not found|failed|"
                 r"exception|invalid|denied", re.I)

def activities(traj):
    acts, obs_len, errs = [], [], []
    for st in traj:
        role = st.get("role"); txt = st.get("text") or ""
        if role == "ai":
            m = CMD.search(txt)
            acts.append(abstract(m.group(1)) if m else "no_command")
            obs_len.append(len(txt)); errs.append(0)
        elif role == "user" and acts:
            obs_len[-1] += len(txt)
            errs[-1] = int(bool(ERR.search(txt)))
    return acts, obs_len, errs

rows = []
for s in range(a.shards):
    p = os.path.join(a.data, f"train-{s:05d}-of-00012.parquet")
    if not os.path.exists(p): continue
    df = pd.read_parquet(p, columns=["instance_id", "model_name", "target", "trajectory"])
    for iid, mdl, tgt, tr in df.itertuples(index=False):
        try: acts, ol, er = activities(list(tr))
        except Exception: continue
        if len(acts) < 2: continue
        rows.append((f"{mdl}||{iid}", iid, int(not bool(tgt)), acts, ol, er))
    del df
    print(f"[load] shard {s}: {len(rows)} traces", flush=True)

unit  = np.array([r[0] for r in rows]); group = np.array([r[1] for r in rows])
y     = np.array([r[2] for r in rows])
seqs  = [r[3] for r in rows]; olens = [r[4] for r in rows]; errs = [r[5] for r in rows]
print(f"[data] {len(y)} traces | {len(set(unit))} units | {len(set(group))} tasks | "
      f"fail {y.mean():.3f}", flush=True)

def fit_fsm(idx, k):
    trans = collections.defaultdict(collections.Counter)
    for i in idx:
        s = "<init>"
        for act in seqs[i][:k]: trans[s][act] += 1; s = act
    kept = {}
    for s, c in trans.items():
        keep = {act: n for act, n in c.items() if n >= a.rare_min or len(c) == 1}
        if not keep: keep = dict(c)
        tot = sum(keep.values()); kept[s] = {act: n / tot for act, n in keep.items()}
    alpha = set(); [alpha.update(c) for c in trans.values()]
    return kept, sorted(alpha)

def featurise(i, k, fsm, alpha, aidx):
    seq, ol, er = seqs[i][:k], olens[i][:k], errs[i][:k]
    n = len(seq)
    visits = np.zeros(len(alpha))
    for act in seq:
        j = aidx.get(act)
        if j is not None: visits[j] += 1
    visits /= max(n, 1)
    FLOOR = 1e-4; sur, s = [], "<init>"
    for act in seq:
        sur.append(-math.log(max(fsm.get(s, {}).get(act, FLOOR), FLOOR))); s = act
    sur = np.array(sur) if sur else np.array([0.0])
    h = max(1, len(sur) // 2)
    ol = np.array(ol, float) if ol else np.array([0.0])
    er = np.array(er, float) if er else np.array([0.0])
    return np.concatenate([visits, [
        sur.mean(), sur.max(),
        float(abs(sur[:h].mean() - sur[h:].mean())) if len(sur) > 1 else 0.0,
        float((sur > 3.0).mean()), n, len(set(seq)),
        1.0 - len(set(seq)) / max(n, 1),
        ol.mean(), ol.max(), ol.std(), er.mean(), er.sum(),
    ]])

def within_auc(u, yy, s):
    num = den = 0.0; nu = 0
    for g in set(u):
        m = u == g; P, N = s[m & (yy == 1)], s[m & (yy == 0)]
        if not len(P) or not len(N): continue
        nu += 1
        num += sum((p > q) + 0.5 * (p == q) for p in P for q in N); den += len(P) * len(N)
    return (num / den if den else float("nan")), den, nu

def perm_p(u, yy, s, B, rng):
    obs, _, _ = within_auc(u, yy, s)
    per = []
    units = {}
    for g in set(u):
        m = u == g
        if yy[m].min() != yy[m].max(): units[g] = (s[m], yy[m].sum(), (~yy[m].astype(bool)).sum())
    null = []
    for _ in range(B):
        num = den = 0.0
        for g, (sc, P, N) in units.items():
            lab = np.zeros(len(sc), int); lab[rng.choice(len(sc), int(P), replace=False)] = 1
            Ps, Ns = sc[lab == 1], sc[lab == 0]
            num += sum((p > q) + 0.5 * (p == q) for p in Ps for q in Ns); den += len(Ps) * len(Ns)
        null.append(num / den)
    null = np.array(null)
    p = (1 + np.sum(np.abs(null - 0.5) >= abs(obs - 0.5))) / (1 + B)
    return obs, float(p), float(null.std())

def std_fit(Xtr):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
    return mu, sd

def train_objectives(Xtr, ytr, utr):
    """Return scoring functions for the three objectives."""
    mu, sd = std_fit(Xtr); Z = (Xtr - mu) / sd
    out = {}
    m = LogisticRegression(C=1.0, max_iter=3000).fit(Z, ytr)
    out["pooled"] = lambda X: m.predict_proba((X - mu) / sd)[:, 1]
    # pairwise ranker on within-unit differences (mixed training units only)
    diffs = []
    for g in set(utr):
        msk = utr == g
        if ytr[msk].min() == ytr[msk].max(): continue
        Zf, Zs = Z[msk & (ytr == 1)], Z[msk & (ytr == 0)]
        for zf in Zf:
            for zs in Zs: diffs.append(zf - zs)
    D = np.vstack(diffs)
    Xp = np.vstack([D, -D]); yp = np.r_[np.ones(len(D)), np.zeros(len(D))]
    r = LogisticRegression(C=1.0, fit_intercept=False, max_iter=3000).fit(Xp, yp)
    wvec = r.coef_[0]
    out["ranker"] = lambda X: ((X - mu) / sd) @ wvec
    # within-unit demeaned features, pooled logistic (demeaning uses no labels)
    def demean(X, u):
        Xd = np.array(X, float)
        for g in set(u):
            msk = u == g; Xd[msk] -= Xd[msk].mean(0)
        return Xd
    Zd = demean(Z, utr)
    dm = LogisticRegression(C=1.0, max_iter=3000).fit(Zd, ytr)
    out["demeaned"] = ("demean", dm, mu, sd, demean)
    return out

res = {}
rng = np.random.default_rng(0)
for k in a.depths:
    print(f"\n===== depth k={k} =====", flush=True)
    oof = {o: np.zeros(len(y)) for o in ("pooled", "ranker", "demeaned")}
    for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(y)), y, group):
        fsm, alpha = fit_fsm(tr, k); aidx = {act: j for j, act in enumerate(alpha)}
        Xtr = np.vstack([featurise(i, k, fsm, alpha, aidx) for i in tr])
        Xte = np.vstack([featurise(i, k, fsm, alpha, aidx) for i in te])
        obj = train_objectives(Xtr, y[tr], unit[tr])
        oof["pooled"][te] = obj["pooled"](Xte)
        oof["ranker"][te] = obj["ranker"](Xte)
        _, dm, mu, sd, demean = obj["demeaned"]
        oof["demeaned"][te] = dm.predict_proba(demean((Xte - mu) / sd, unit[te]))[:, 1]
        print("  [fold done]", flush=True)
    for o in ("pooled", "ranker", "demeaned"):
        pool = roc_auc_score(y, oof[o])
        wv, pairs, nu = within_auc(unit, y, oof[o])
        _, p, sd0 = perm_p(unit, y, oof[o], a.perms, rng)
        res[f"k{k}|{o}"] = dict(pooled=float(pool), within=float(wv), p=p, null_sd=sd0,
                                pairs=int(pairs), units=nu)
        print(f"  {o:9s} pooled={pool:.4f}  within={wv:.4f}  perm-p={p:.4f} "
              f"(null sd {sd0:.4f}; {nu} mixed units)", flush=True)

# ---- control battery on the ranker pipeline at k=5
print("\n===== controls (ranker pipeline, k=5) =====", flush=True)
k = 5
tr, te = next(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, group))
fsm, alpha = fit_fsm(tr, k); aidx = {act: j for j, act in enumerate(alpha)}
Xtr = np.vstack([featurise(i, k, fsm, alpha, aidx) for i in tr])
Xte = np.vstack([featurise(i, k, fsm, alpha, aidx) for i in te])
# (i) injected outcome-carrying feature
Xi_tr = np.c_[Xtr, y[tr] + 0.1 * rng.standard_normal(len(tr))]
Xi_te = np.c_[Xte, y[te] + 0.1 * rng.standard_normal(len(te))]
obj = train_objectives(Xi_tr, y[tr], unit[tr])
wv, _, _ = within_auc(unit[te], y[te], obj["ranker"](Xi_te))
print(f"  injected outcome feature: within={wv:.3f} (expect ~1.0)", flush=True)
# (ii) within-unit label shuffle
ysh = y.copy()
for g in set(unit[tr]):
    msk = np.where(unit == g)[0]; msk = msk[np.isin(msk, tr)]
    ysh[msk] = rng.permutation(ysh[msk])
obj = train_objectives(Xtr, ysh[tr], unit[tr])
wv, _, _ = within_auc(unit[te], y[te], obj["ranker"](Xte))
print(f"  shuffled training labels: within={wv:.3f} (expect ~0.5)", flush=True)

if a.out:
    json.dump(res, open(a.out, "w"), indent=2); print(f"[saved] {a.out}")
