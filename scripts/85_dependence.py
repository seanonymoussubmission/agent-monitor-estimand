#!/usr/bin/env python3
"""Dependence diagnostics for the exchangeability approximation.

The allocation model treats a unit's repeated outcomes as exchangeable. The paper already
checks order effects and lag-1 autocorrelation; a reviewer can reasonably ask about
higher-order structure. Three additional diagnostics, none of which can prove
independence -- the point is to show the approximation is not contradicted by anything we
can measure.

  lag-k autocorrelation (k = 1..3), pooled across units with >= k+1 runs
  Wald-Wolfowitz runs test within unit, combined by Stouffer
  first-half vs second-half success-rate difference within unit
"""
import argparse, csv, math
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["data/runs_qwen.csv","data/runs_laguna.csv"])
a = ap.parse_args()

for path in a.runs:
    by = {}
    for r in csv.DictReader(open(path)):
        by.setdefault(r["task"], []).append((int(r["run"]), int(r["success"])))
    seqs = [np.array([s for _, s in sorted(v)]) for v in by.values()]
    seqs = [s for s in seqs if len(s) >= 4 and 0 < s.mean() < 1]
    print(f"\n=== {path}: {len(seqs)} mixed units with >=4 ordered runs ===")
    # Demeaned autocorrelation on short sequences is biased by about -1/(n-1) even for
    # i.i.d. data, so a zero-centred null would flag pure noise as negative dependence.
    # A within-unit permutation null carries the bias and gives an honest reference.
    rng = np.random.default_rng(0)
    def lagcorr(seqlist, k):
        num = den = 0.0
        for s in seqlist:
            if len(s) <= k: continue
            x, y_ = s[:-k] - s.mean(), s[k:] - s.mean()
            num += float(np.sum(x[:len(y_)] * y_)); den += float(np.sum((s - s.mean())**2))
        return num / den if den else float("nan")
    for k in (1, 2, 3):
        obs = lagcorr(seqs, k)
        null = [lagcorr([rng.permutation(s) for s in seqs], k) for _ in range(2000)]
        null = np.array(null)
        pperm = (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / 2001
        print(f"  lag-{k}: observed {obs:+.4f} | permutation null "
              f"{null.mean():+.4f} +/- {null.std():.4f} | p = {pperm:.4f}")
    zs = []
    for s in seqs:
        n1, n0 = int(s.sum()), int(len(s)-s.sum())
        runs = 1 + int(np.sum(s[1:] != s[:-1]))
        mu = 1 + 2*n1*n0/(n1+n0)
        var = 2*n1*n0*(2*n1*n0-n1-n0)/(((n1+n0)**2)*(n1+n0-1))
        if var > 0: zs.append((runs-mu)/math.sqrt(var))
    Z = float(np.sum(zs)/math.sqrt(len(zs)))
    print(f"  runs test (Stouffer over {len(zs)} units): Z = {Z:+.3f}  "
          f"p = {2*stats.norm.sf(abs(Z)):.4f}")
    d = [s[:len(s)//2].mean()-s[len(s)//2:].mean() for s in seqs]
    t_, p_ = stats.ttest_1samp(d, 0.0)
    print(f"  first-half minus second-half success rate = {np.mean(d):+.4f}  "
          f"(t={t_:+.2f}, p={p_:.4f})")
