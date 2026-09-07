#!/usr/bin/env python3
"""
Finite-repeat-corrected ICC for the outcome variance decomposition (review P0 #2).

The naive ICC = Var(unit means)/Var(Y) treats the observed unit mean as the latent
failure probability q_u. But Var(ybar_u) = Var(q_u) + E[q_u(1-q_u)/r_u], so with few
runs per unit (C3 has ~3) the naive number overstates the between-unit share.

We report three estimators and a by-unit bootstrap CI:
  (1) naive           between/tot                     (what the paper currently reports)
  (2) ANOVA ICC(1)    (MSB-MSW)/(MSB+(n0-1)MSW)       classic finite-sample correction
  (3) beta-binom MoM  Var(q)/(mu(1-mu)) with E[q(1-q)] from unbiased per-unit estimates
Latent between-unit variance uses the unbiased per-unit estimator of q(1-q):
      v_u = s_u (r_u - s_u) / (r_u (r_u - 1)),   E[v_u] = q_u(1-q_u).
"""
import argparse, numpy as np, pandas as pd

def unit_table(df, group_cols, fail_col):
    g = df.groupby(group_cols)[fail_col]
    r = g.size().to_numpy().astype(float)          # runs per unit
    s = g.sum().to_numpy().astype(float)           # failures per unit
    return r, s

def naive_icc(y_all, unit_mean_all):
    tot = float(np.var(y_all))                      # = mu(1-mu) for binary, ddof=0
    within = float(np.mean((y_all - unit_mean_all) ** 2))
    between = tot - within
    return between / tot, tot, within

def anova_icc1(r, s):
    # one-way random-effects ICC(1) with unequal group sizes
    k = len(r); N = r.sum()
    ybar_u = s / r
    grand = s.sum() / N
    ssb = float(np.sum(r * (ybar_u - grand) ** 2))
    # within SS for binary: within a unit, sum (y-ybar)^2 = s - s^2/r = s*(r-s)/r
    ssw = float(np.sum(s * (r - s) / r))
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    n0 = (N - np.sum(r ** 2) / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return (msb - msw) / denom if denom else float("nan")

def betabinom_mom(r, s, mu):
    # unbiased per-unit q(1-q); need r>=2
    m = r >= 2
    v = (s[m] * (r[m] - s[m])) / (r[m] * (r[m] - 1))   # unbiased q(1-q)
    Eqq_unit = float(np.mean(v))                        # unit-weighted
    Eqq_run  = float(np.sum(r[m] * v) / np.sum(r[m]))   # run-weighted
    tot = mu * (1 - mu)
    varq_unit = tot - Eqq_unit
    varq_run  = tot - Eqq_run
    return varq_unit / tot, varq_run / tot, Eqq_unit, tot

def compute_all(df, group_cols, fail_col):
    y = df[fail_col].to_numpy().astype(float)
    unit_mean = df.groupby(group_cols)[fail_col].transform("mean").to_numpy()
    r, s = unit_table(df, group_cols, fail_col)
    mu = float(y.mean())
    naive, tot, within = naive_icc(y, unit_mean)
    icc1 = anova_icc1(r, s)
    mom_unit, mom_run, Eqq, _ = betabinom_mom(r, s, mu)
    return dict(naive=naive, icc1=icc1, mom_unit=mom_unit, mom_run=mom_run,
                mu=mu, tot=tot, Eqq=Eqq, k=len(r), N=int(r.sum()),
                rbar=float(r.mean()), r_med=float(np.median(r)))

def bootstrap(df, group_cols, fail_col, B, seed):
    # resample units with replacement, keep each unit's runs together
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(group_cols)]
    k = len(groups)
    out = {"naive": [], "icc1": [], "mom_unit": []}
    for _ in range(B):
        idx = rng.integers(0, k, size=k)
        boot = pd.concat([groups[i] for i in idx], ignore_index=True)
        # relabel units so identical resampled units stay separate
        boot["_u"] = np.repeat(np.arange(k), [len(groups[i]) for i in idx])
        y = boot[fail_col].to_numpy().astype(float)
        um = boot.groupby("_u")[fail_col].transform("mean").to_numpy()
        r = boot.groupby("_u")[fail_col].size().to_numpy().astype(float)
        s = boot.groupby("_u")[fail_col].sum().to_numpy().astype(float)
        mu = float(y.mean())
        out["naive"].append(naive_icc(y, um)[0])
        out["icc1"].append(anova_icc1(r, s))
        out["mom_unit"].append(betabinom_mom(r, s, mu)[0])
    ci = {k2: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
          for k2, v in out.items()}
    return ci

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/liveclaw/data/v0.2.1-00000-of-00001.parquet")
    ap.add_argument("--group", nargs="+", default=["model_name", "case_id"])
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_parquet(a.data)
    df["fail"] = (df[a.score_col] != 1.0).astype(int)
    print(f"[data] {len(df)} records | units by {a.group} "
          f"| fail rate {df.fail.mean():.3f}", flush=True)

    res = compute_all(df, a.group, "fail")
    print(f"[units] k={res['k']} units, N={res['N']} records, "
          f"mean r={res['rbar']:.2f}, median r={res['r_med']:.0f}")
    print(f"[var]   mu={res['mu']:.4f}  tot=mu(1-mu)={res['tot']:.4f}  "
          f"E[q(1-q)]hat={res['Eqq']:.4f}")
    print()
    print("  estimator                     ICC     within-share")
    print(f"  (1) naive between/tot        {res['naive']:.4f}   {1-res['naive']:.4f}")
    print(f"  (2) ANOVA ICC(1) corrected   {res['icc1']:.4f}   {1-res['icc1']:.4f}")
    print(f"  (3) beta-binom MoM (unit-wt) {res['mom_unit']:.4f}   {1-res['mom_unit']:.4f}")
    print(f"      beta-binom MoM (run-wt)  {res['mom_run']:.4f}   {1-res['mom_run']:.4f}")
    print(f"\n[bootstrap {a.boot} unit-resamples]", flush=True)
    ci = bootstrap(df, a.group, "fail", a.boot, a.seed)
    for name, lab in [("naive", "naive"), ("icc1", "ANOVA ICC(1)"),
                      ("mom_unit", "beta-binom MoM")]:
        lo, hi = ci[name]
        print(f"  {lab:16s} 95% CI [{lo:.4f}, {hi:.4f}]")

if __name__ == "__main__":
    main()
