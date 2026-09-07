#!/usr/bin/env python3
"""E1-C2 (pre-registered amendment): the within-unit training objective on C2
(SWE-rebench / OpenHands), behavioural-feature instrument, k in {2,5,10,20}.
Objectives as in 99: pooled logistic, same-unit pairwise ranker, within-demeaned.
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/swe_rebench/trajectories.parquet")
ap.add_argument("--ks", type=int, nargs="+", default=[2, 5, 10, 20])
ap.add_argument("--perms", type=int, default=400)
ap.add_argument("--out", default=None)
a = ap.parse_args()

ERR = ("error", "traceback", "exception", "failed", "not found", "no such file")

def msg_text(m):
    for k in ("text", "content"):
        v = m.get(k)
        if isinstance(v, str): return v
    return ""

def msg_role(m):
    r = m.get("role") or ""
    if r in ("ai", "assistant"): return "ai"
    if r in ("user", "tool", "observation"): return "user"
    return r

def featurize(traj, k):
    msgs = [m for m in traj if msg_role(m) in ("ai", "user")]
    ai_idx = [i for i, m in enumerate(msgs) if msg_role(m) == "ai"]
    if len(ai_idx) < k: return None
    pre = msgs[:ai_idx[k - 1] + 2]
    ai = [msg_text(m) for m in pre if msg_role(m) == "ai"]
    ob = [msg_text(m) for m in pre if msg_role(m) == "user"]
    obs = ob[1:] if len(ob) > 1 else []
    joined = " ".join(obs).lower()
    errs = sum(joined.count(p) for p in ERR)
    return [float(np.mean([len(t) for t in ai])) if ai else 0.0,
            float(np.std([len(t) for t in ai])) if len(ai) > 1 else 0.0,
            float(np.mean([len(t) for t in obs])) if obs else 0.0,
            float(np.std([len(t) for t in obs])) if len(obs) > 1 else 0.0,
            float(len(obs[-1])) if obs else 0.0,
            float(errs), float(errs / max(len(obs), 1)),
            float(sum(1 for t in obs if len(t) < 20) / max(len(obs), 1))]

df = pd.read_parquet(a.data, columns=["instance_id", "resolved", "trajectory"])
df["fail"] = 1 - df["resolved"].astype(int)
print(f"[data] {len(df)} trajectories | {df.instance_id.nunique()} instances | "
      f"fail {df.fail.mean():.3f}", flush=True)

def within_auc(u, yy, s):
    num = den = 0.0; nu = 0
    for g in set(u):
        m = u == g; P, N = s[m & (yy == 1)], s[m & (yy == 0)]
        if not len(P) or not len(N): continue
        nu += 1
        num += sum((p > q) + 0.5 * (p == q) for p in P for q in N); den += len(P) * len(N)
    return (num / den if den else float("nan")), nu

def perm_p(u, yy, s, B, rng):
    obs, _ = within_auc(u, yy, s)
    units = {}
    for g in set(u):
        m = u == g
        if yy[m].min() != yy[m].max(): units[g] = (s[m], int(yy[m].sum()))
    null = []
    for _ in range(B):
        num = den = 0.0
        for g, (sc, P) in units.items():
            lab = np.zeros(len(sc), int)
            lab[rng.choice(len(sc), P, replace=False)] = 1
            Ps, Ns = sc[lab == 1], sc[lab == 0]
            num += sum((p > q) + 0.5 * (p == q) for p in Ps for q in Ns)
            den += len(Ps) * len(Ns)
        null.append(num / den)
    null = np.array(null)
    return obs, float((1 + np.sum(np.abs(null - 0.5) >= abs(obs - 0.5))) / (1 + B)), float(null.std())

res = {}
rng = np.random.default_rng(0)
for k in a.ks:
    feats, keep = [], []
    for idx, tr in zip(df.index, df["trajectory"]):
        f = featurize(tr, k)
        if f is not None: feats.append(f); keep.append(idx)
    sub = df.loc[keep]
    X = np.array(feats); y = sub["fail"].values.astype(int)
    u = sub["instance_id"].values
    print(f"\n== k={k}: {len(sub)} rows, {len(set(u))} units ==", flush=True)
    oof = {o: np.zeros(len(sub)) for o in ("pooled", "ranker", "demeaned")}
    for tr_, te in GroupKFold(n_splits=5).split(X, y, u):
        mu, sd = X[tr_].mean(0), X[tr_].std(0); sd[sd == 0] = 1.0
        Z, Zt = (X[tr_] - mu) / sd, (X[te] - mu) / sd
        m1 = LogisticRegression(C=1.0, max_iter=3000).fit(Z, y[tr_])
        oof["pooled"][te] = m1.predict_proba(Zt)[:, 1]
        diffs = []
        for g in set(u[tr_]):
            msk = u[tr_] == g
            if y[tr_][msk].min() == y[tr_][msk].max(): continue
            Zf, Zs = Z[msk & (y[tr_] == 1)], Z[msk & (y[tr_] == 0)]
            for zf in Zf:
                for zs in Zs: diffs.append(zf - zs)
        D = np.vstack(diffs); Xp = np.vstack([D, -D])
        yp = np.r_[np.ones(len(D)), np.zeros(len(D))]
        r2 = LogisticRegression(C=1.0, fit_intercept=False, max_iter=3000).fit(Xp, yp)
        oof["ranker"][te] = Zt @ r2.coef_[0]
        def dem(Zx, ux):
            Zd = Zx.copy()
            for g in set(ux):
                msk = ux == g; Zd[msk] -= Zd[msk].mean(0)
            return Zd
        m3 = LogisticRegression(C=1.0, max_iter=3000).fit(dem(Z, u[tr_]), y[tr_])
        oof["demeaned"][te] = m3.predict_proba(dem(Zt, u[te]))[:, 1]
    for o in ("pooled", "ranker", "demeaned"):
        pool = roc_auc_score(y, oof[o])
        wv, p, sd0 = perm_p(u, y, oof[o], a.perms, rng)
        _, nu = within_auc(u, y, oof[o])
        res[f"k{k}|{o}"] = dict(pooled=float(pool), within=float(wv), p=p, units=nu)
        print(f"  {o:9s} pooled={pool:.4f} within={wv:.4f} p={p:.4f} "
              f"(null sd {sd0:.4f}; {nu} mixed units)", flush=True)
if a.out: json.dump(res, open(a.out, "w"), indent=2); print(f"[saved] {a.out}")
