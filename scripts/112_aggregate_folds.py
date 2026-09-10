#!/usr/bin/env python3
"""E6 five-fold aggregation, exactly as pre-registered (amendment 3):

  - Within-task AUROC: same-unit discordant-pair NUMERATORS and DENOMINATORS are summed
    across folds, then divided. Comparisons never cross folds or models. The within-unit
    permutation test aggregates the same way (labels shuffled inside each unit, units
    pooled across folds).
  - Pooled AUROC: computed PER FOLD (scores from different trained models are never
    compared) and reported as per-fold values plus mean +/- sd.
  - Per-fold within values are also reported individually as replications.

Emits one row per (checkpoint, model) with the aggregate and the per-fold detail, plus a
compact LaTeX-ready summary for the strongest configurations.
"""
import argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--preds", nargs="+", required=True, help="per-fold prediction parquets")
ap.add_argument("--ks", type=int, nargs="+",
                default=[0, 1, 2, 3, 5, 10, 20, 30, 50],
                help="steps observed; 0 = pre-run")
ap.add_argument("--final", action="store_true", default=True)
ap.add_argument("--perms", type=int, default=600)
ap.add_argument("--perm-models", nargs="+", default=[
    "prob__K_LightGBM_Dense_Full", "prob__N_LightGBM_TfIdf_Full",
    "prob__H_LightGBM_Dense", "prob__I_LightGBM_Dense_AF",
    "prob__D_Dense_Full_LR", "prob__G_TfIdf_Full_LR"])
ap.add_argument("--out", required=True)
a = ap.parse_args()

folds = []
for i, p in enumerate(a.preds):
    df = pd.read_parquet(p)
    df = df[df["split"] == "test"].copy()
    folds.append(df)
    print(f"[fold {i+1}] {df.traj_id.nunique()} trajs, {df.instance_id.nunique()} instances",
          flush=True)
models = [c for c in folds[0].columns if c.startswith("prob__")]
print(f"[models] {len(models)}", flush=True)


def unit_pairs(sub, col):
    """[(scores, labels)] for units with both outcomes."""
    out = []
    for _, g in sub.groupby("instance_id"):
        y = g["label"].values.astype(int)
        if y.min() == y.max():
            continue
        out.append((g[col].values.astype(float), y))
    return out


def num_den(units):
    num = den = 0.0
    for s, y in units:
        P, N = s[y == 1], s[y == 0]
        num += (P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()
        den += len(P) * len(N)
    return num, den


def pooled(sub, col):
    s = sub[col].values.astype(float)
    y = sub["label"].values.astype(int)
    P, N = s[y == 1], s[y == 0]
    if not len(P) or not len(N):
        return float("nan")
    rng = np.random.default_rng(1)
    ii = rng.integers(0, len(P), 400_000)
    jj = rng.integers(0, len(N), 400_000)
    return float(((P[ii] > N[jj]) + 0.5 * (P[ii] == N[jj])).mean())


def perm_p(all_units, obs, B, rng):
    null = np.empty(B)
    for b in range(B):
        num = den = 0.0
        for s, y in all_units:
            lab = np.zeros(len(s), int)
            lab[rng.choice(len(s), int(y.sum()), replace=False)] = 1
            P, N = s[lab == 1], s[lab == 0]
            num += (P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()
            den += len(P) * len(N)
        null[b] = num / den
    return float((1 + np.sum(np.abs(null - 0.5) >= abs(obs - 0.5))) / (1 + B))


def checkpoint_slices(k):
    if k == "final":
        return [f.loc[f.groupby("traj_id")["prefix_step_idx"].idxmax()] for f in folds]
    return [f[f["prefix_step_idx"] == k] for f in folds]


rng = np.random.default_rng(0)
rows = []
keys = list(a.ks) + (["final"] if a.final else [])
for k in keys:
    subs = checkpoint_slices(k)
    if any(len(s) == 0 for s in subs):
        print(f"  [skip {k}] a fold has no rows at this checkpoint", flush=True)
        continue
    for col in models:
        per_fold_within, per_fold_pooled, all_units = [], [], []
        tot_num = tot_den = 0.0
        nunits = 0
        for sub in subs:
            units = unit_pairs(sub, col)
            n, d = num_den(units)
            per_fold_within.append(n / d if d else float("nan"))
            per_fold_pooled.append(pooled(sub, col))
            tot_num += n
            tot_den += d
            nunits += len(units)
            all_units.extend(units)
        agg_within = tot_num / tot_den if tot_den else float("nan")
        row = dict(checkpoint=str(k), model=col[6:],
                   within_agg=float(agg_within), units=nunits, pairs=float(tot_den),
                   within_per_fold=[float(v) for v in per_fold_within],
                   pooled_per_fold=[float(v) for v in per_fold_pooled],
                   pooled_mean=float(np.nanmean(per_fold_pooled)),
                   pooled_sd=float(np.nanstd(per_fold_pooled, ddof=1)))
        if col in a.perm_models and np.isfinite(agg_within):
            row["perm_p"] = perm_p(all_units, agg_within, a.perms, rng)
        rows.append(row)
        pp = f" p={row['perm_p']:.4f}" if "perm_p" in row else ""
        print(f"  {str(k):>5s} {col[6:]:34s} within={agg_within:.4f} "
              f"({nunits} units, {tot_den:.0f} pairs) | pooled "
              f"{row['pooled_mean']:.3f}+/-{row['pooled_sd']:.3f}{pp}", flush=True)

json.dump(rows, open(a.out, "w"), indent=2)
print(f"\n[saved] {a.out}", flush=True)

# compact summary for the paper
print("\n=== headline (aggregate within, pooled mean+/-sd across folds) ===", flush=True)
df = pd.DataFrame(rows)
for k in ["0", "1", "5", "final"]:
    d = df[df.checkpoint == k]
    if not len(d):
        continue
    print(f"  step {k}: within {d.within_agg.min():.3f}-{d.within_agg.max():.3f} "
          f"across {len(d)} variants | pooled up to {d.pooled_mean.max():.3f} | "
          f"exactly-0.500 variants: {(d.within_agg == 0.5).sum()}/{len(d)}", flush=True)
