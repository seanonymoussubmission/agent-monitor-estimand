#!/usr/bin/env python3
"""E10: is the within-task estimate an artifact of conditioning on observed-mixed units?

Within-task AUROC is identifiable only where a unit shows both outcomes, and with few
repeats a moderately stochastic unit enters that set more readily than a near-
deterministic one. Four weightings of the SAME held-out scores are compared:

  pair      pair-weighted over observed mixed units (the paper's primary)
  macro     each observed mixed unit weighted equally
  uniform   task-uniform among units with identifiable pairs (= macro when one model
            per task; reported separately when units are model x task)
  postpred  posterior-predictive over ALL units: latent q_u fitted by beta-binomial
            method of moments (shrunk toward the pooled mean), each unit weighted by
            2 q_u (1 - q_u), its probability of yielding a future discordant pair.
            This does not condition on what the finite sample happened to show.

A unit with no observed discordant pair contributes no measurable A_u, so postpred
weights its CONTRIBUTION by 2q(1-q) while its A_u is imputed from the fitted model as
0.5 (a task-constant scorer's value) -- stated plainly rather than dropped, since the
question is precisely what those units would contribute.

Also simulates observation at r in {3,5,10,20} by subsampling runs within units, to show
whether the observed-mixed estimator drifts with repeat count.
"""
import argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--preds", nargs="+", required=True)
ap.add_argument("--model", default="prob__K_LightGBM_Dense_Full")
ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 3, 5, 10])
ap.add_argument("--unit-cols", nargs="+", default=["instance_id"])
ap.add_argument("--repeats", type=int, nargs="+", default=[3, 5, 10, 20])
ap.add_argument("--draws", type=int, default=300)
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

frames = []
for i, p in enumerate(a.preds):
    cols = ["split", "traj_id", "prefix_step_idx", "label", a.model] + a.unit_cols
    df = pd.read_parquet(p, columns=list(dict.fromkeys(cols)))
    df = df[df["split"] == "test"].copy()
    key = df[a.unit_cols].astype(str).agg("||".join, axis=1) if len(a.unit_cols) > 1 \
        else df[a.unit_cols[0]].astype(str)
    df["unit"] = "f%d::" % i + key
    frames.append(df)
allp = pd.concat(frames, ignore_index=True)
print("[data] %d folds | %d trajs | %d units | model %s"
      % (len(a.preds), allp.traj_id.nunique(), allp.unit.nunique(), a.model), flush=True)


def auc_unit(s, y):
    P, N = s[y == 1], s[y == 0]
    if not len(P) or not len(N):
        return None, 0
    n = (P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()
    return n / (len(P) * len(N)), len(P) * len(N)


def beta_binom_mom(counts):
    """counts: list of (successes, trials). Returns shrunk per-unit rates."""
    s = np.array([c[0] for c in counts], float)
    n = np.array([c[1] for c in counts], float)
    p = s / n
    mu = s.sum() / n.sum()
    # unbiased per-unit variance of q via method of moments
    ex_var = np.mean([si * (ni - si) / (ni * (ni - 1)) if ni > 1 else 0.0
                      for si, ni in zip(s, n)])
    var_q = max(mu * (1 - mu) - ex_var, 1e-6)
    if var_q <= 0 or var_q >= mu * (1 - mu):
        k = 1.0
    else:
        k = mu * (1 - mu) / var_q - 1.0
        k = max(k, 1e-3)
    alpha, beta = mu * k, (1 - mu) * k
    return (s + alpha) / (n + alpha + beta), mu, k


res = {}
rng = np.random.default_rng(a.seed)
for k in a.ks:
    sub = allp[allp["prefix_step_idx"] == k]
    if not len(sub):
        continue
    units, counts, aucs, pairs = [], [], [], []
    for u, g in sub.groupby("unit"):
        y = g["label"].values.astype(int)
        s = g[a.model].values.astype(float)
        units.append(u)
        counts.append((int(y.sum()), len(y)))          # failures, trials
        au, pr = auc_unit(s, y)
        aucs.append(au)
        pairs.append(pr)
    aucs = np.array([np.nan if x is None else x for x in aucs], float)
    pairs = np.array(pairs, float)
    mixed = ~np.isnan(aucs)

    pair_w = float(np.nansum(aucs[mixed] * pairs[mixed]) / pairs[mixed].sum())
    macro = float(np.nanmean(aucs[mixed]))

    q, mu, kappa = beta_binom_mom(counts)
    w_pp = 2.0 * q * (1.0 - q)
    a_imp = np.where(mixed, np.nan_to_num(aucs, nan=0.5), 0.5)
    postpred = float(np.sum(a_imp * w_pp) / np.sum(w_pp))

    idx = np.where(mixed)[0]
    bs = np.empty(a.boot)
    for b in range(a.boot):
        pick = rng.choice(idx, len(idx), replace=True)
        bs[b] = np.sum(aucs[pick] * pairs[pick]) / pairs[pick].sum()
    lo, hi = np.percentile(bs, [2.5, 97.5])

    tag = "prerun" if k == 0 else "obs%d" % k
    res[tag] = dict(pair=pair_w, pair_ci=[float(lo), float(hi)], macro=macro,
                    postpred=postpred, mixed_units=int(mixed.sum()),
                    all_units=len(units), beta_mu=float(mu), beta_kappa=float(kappa),
                    delta_postpred_pair=float(postpred - pair_w))
    print("  %-7s pair=%.4f [%.3f,%.3f]  macro=%.4f  post-pred=%.4f  (delta %+.4f) "
          "| %d/%d units mixed | beta mu=%.3f kappa=%.1f"
          % (tag, pair_w, lo, hi, macro, postpred, postpred - pair_w,
             mixed.sum(), len(units), mu, kappa), flush=True)

# ---- drift with repeat count
print("\n=== observed-mixed pair-weighted estimate vs repeats per unit ===", flush=True)
sub = allp[allp["prefix_step_idx"] == (a.ks[1] if len(a.ks) > 1 else a.ks[0])]
byu = {u: g for u, g in sub.groupby("unit")}
drift = {}
for r in a.repeats:
    vals = []
    for _ in range(a.draws):
        num = den = 0.0
        for u, g in byu.items():
            if len(g) < r:
                continue
            take = rng.choice(len(g), r, replace=False)
            y = g["label"].values[take].astype(int)
            s = g[a.model].values[take].astype(float)
            au, pr = auc_unit(s, y)
            if au is None:
                continue
            num += au * pr
            den += pr
        if den:
            vals.append(num / den)
    if vals:
        drift[str(r)] = [float(np.mean(vals)), float(np.std(vals))]
        print("  r=%-3d  %.4f +/- %.4f  (%d draws with >=1 mixed unit)"
              % (r, np.mean(vals), np.std(vals), len(vals)), flush=True)
res["drift_by_repeats"] = drift

json.dump(res, open(a.out, "w"), indent=2)
print("\n[saved] %s" % a.out, flush=True)
