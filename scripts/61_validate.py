#!/usr/bin/env python3
"""Verification suite for the final results.

Every headline number rests on three things: the data being what we think, the
within-task AUROC being implemented correctly, and the cross-validation not
leaking. This checks all three by making the pipeline recover answers we already
know -- a feature that IS the outcome must score 1.0, random noise must score
0.5, and shuffled labels must score 0.5. A pipeline that cannot reproduce those
cannot be trusted on a real one.
"""
import argparse, collections, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--probe", required=True)
ap.add_argument("--k", type=int, default=1)
ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--nsim", type=int, default=400)
a = ap.parse_args()
rng = np.random.default_rng(0)
FAIL = []

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok: FAIL.append(name)

Z = np.load(a.probe, allow_pickle=False)
task, run, y, k, pos = Z["task"], Z["run"], Z["y"].astype(int), Z["k"], Z["pos"]
print(f"[data] {len(y)} rows | {len(set(task))} tasks | k values {sorted(set(k.tolist()))}\n")

# ---------------------------------------------------------------- 1. integrity
print("=== 1. DATA INTEGRITY ===")
ids = list(zip(task, run, k))
check("no duplicate (task, run, k) rows", len(ids) == len(set(ids)),
      f"{len(ids)-len(set(ids))} dupes")
lab = collections.defaultdict(set)
for t, r, yy in zip(task, run, y): lab[(t, r)].add(yy)
check("each run has ONE label across all k", all(len(v) == 1 for v in lab.values()),
      f"{sum(1 for v in lab.values() if len(v)>1)} inconsistent")
check("labels are binary", set(np.unique(y).tolist()) <= {0, 1})
check("prefix positions positive", (pos > 0).all())
# pos must increase with k within a run
bad = 0
byrun = collections.defaultdict(list)
for t, r, kk, p in zip(task, run, k, pos): byrun[(t, r)].append((int(kk), int(p)))
for v in byrun.values():
    v = sorted(v)
    if any(v[i][1] > v[i+1][1] for i in range(len(v)-1)): bad += 1
check("prefix length non-decreasing in k", bad == 0, f"{bad} runs violate")

sel = k == a.k
task_s, y_s, pos_s = task[sel], y[sel], Z[f"H{a.layer}"][sel]
X = pos_s.astype(float)
mixed = np.array([len(set(y_s[task_s == t])) > 1 for t in task_s])
check("mixed units have both outcomes",
      all(len(set(y_s[task_s == t])) == 2 for t in set(task_s[mixed])))
print(f"  ({mixed.sum()} rows in {len(set(task_s[mixed]))} mixed units)\n")

# ---------------------------------------------------------------- 2. estimator
print("=== 2. WITHIN-TASK AUROC ESTIMATOR ===")
def wauc(t, yy, s):
    num = den = 0.0
    for u in set(t):
        m = t == u; P, N = s[m & (yy == 1)], s[m & (yy == 0)]
        if not len(P) or not len(N): continue
        num += sum((p > n) + 0.5 * (p == n) for p in P for n in N); den += len(P)*len(N)
    return num / den if den else np.nan

# independent reimplementation: sklearn per unit, weighted by pair count
def wauc_sklearn(t, yy, s):
    num = den = 0.0
    for u in set(t):
        m = t == u
        if len(set(yy[m])) < 2: continue
        npair = (yy[m] == 1).sum() * (yy[m] == 0).sum()
        num += roc_auc_score(yy[m], s[m]) * npair; den += npair
    return num / den if den else np.nan

tk, yk = task_s[mixed], y_s[mixed]
probe = rng.normal(size=mixed.sum())
check("matches an independent sklearn implementation",
      abs(wauc(tk, yk, probe) - wauc_sklearn(tk, yk, probe)) < 1e-9,
      f"{wauc(tk,yk,probe):.6f} vs {wauc_sklearn(tk,yk,probe):.6f}")
check("perfect separator scores 1.0", abs(wauc(tk, yk, yk.astype(float)) - 1.0) < 1e-9)
check("inverted separator scores 0.0", abs(wauc(tk, yk, -yk.astype(float)) - 0.0) < 1e-9)
check("constant score is exactly 0.5",
      abs(wauc(tk, yk, np.ones(mixed.sum())) - 0.5) < 1e-9)
sims = [wauc(tk, yk, rng.normal(size=mixed.sum())) for _ in range(a.nsim)]
m_, sd_ = float(np.mean(sims)), float(np.std(sims))
check("random scores centre on 0.5", abs(m_ - 0.5) < 3*sd_/np.sqrt(a.nsim),
      f"mean {m_:.4f} sd {sd_:.4f} over {a.nsim} draws")

# ---------------------------------------------------------------- 3. pipeline
print("\n=== 3. FULL PIPELINE ON KNOWN ANSWERS ===")
GRID = [1e-3, 1e-2, 1e-1, 1.0]
def oof(Xf, yy, g):
    p = np.zeros(len(yy))
    for tr, te in GroupKFold(n_splits=5).split(Xf, yy, g):
        assert not (set(g[tr]) & set(g[te])), "GROUP LEAK"
        best, bs = GRID[0], -np.inf
        gi = g[tr]
        if len(set(gi)) >= 3:
            for c in GRID:
                sc = []
                for itr, ite in GroupKFold(n_splits=3).split(Xf[tr], yy[tr], gi):
                    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
                    try:
                        mdl.fit(Xf[tr][itr], yy[tr][itr])
                        sc.append(roc_auc_score(yy[tr][ite], mdl.predict_proba(Xf[tr][ite])[:,1]))
                    except Exception: sc.append(-np.inf)
                if np.mean(sc) > bs: bs, best = np.mean(sc), c
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=3000))
        mdl.fit(Xf[tr], yy[tr]); p[te] = mdl.predict_proba(Xf[te])[:,1]
    return p

n = len(y_s)
Xoracle = np.c_[y_s.astype(float) + rng.normal(0, .01, n)]          # the answer, noised
Xnoise  = rng.normal(size=(n, 32))                                   # pure noise
try:
    p_or = oof(Xoracle, y_s, task_s); leak_ok = True
except AssertionError: leak_ok = False
check("GroupKFold never shares a task between folds", leak_ok)
check("oracle feature recovers within-task ~1.0",
      wauc(tk, yk, p_or[mixed]) > 0.97, f"{wauc(tk, yk, p_or[mixed]):.4f}")
p_no = oof(Xnoise, y_s, task_s)
w_no = wauc(tk, yk, p_no[mixed])
check("32 noise features give within-task ~0.5", abs(w_no - 0.5) < 0.06, f"{w_no:.4f}")
ysh = y_s.copy(); rng.shuffle(ysh)
mixed_sh = np.array([len(set(ysh[task_s == t])) > 1 for t in task_s])
p_sh = oof(Z[f"H{a.layer}"][sel].astype(float), ysh, task_s)
w_sh = wauc(task_s[mixed_sh], ysh[mixed_sh], p_sh[mixed_sh])
check("real activations + SHUFFLED labels give ~0.5", abs(w_sh - 0.5) < 0.06, f"{w_sh:.4f}")

# ---------------------------------------------------------------- 4. permutation null
print("\n=== 4. PERMUTATION NULL ===")
s_real = oof(Z[f"H{a.layer}"][sel].astype(float), y_s, task_s)
null = []
for _ in range(a.nsim):
    yp = yk.copy()
    for u in set(tk):
        m = tk == u; yp[m] = rng.permutation(yp[m])
    null.append(wauc(tk, yp, s_real[mixed]))
null = np.array(null)
check("within-unit permutation null centres on 0.5",
      abs(null.mean() - 0.5) < 3*null.std()/np.sqrt(a.nsim),
      f"mean {null.mean():.4f} sd {null.std():.4f}")
obs = wauc(tk, yk, s_real[mixed])
p = (1 + (np.abs(null-0.5) >= abs(obs-0.5)).sum()) / (a.nsim + 1)
print(f"  observed within-task = {obs:.4f}, permutation p = {p:.4f}")

print("\n" + ("="*46))
print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
