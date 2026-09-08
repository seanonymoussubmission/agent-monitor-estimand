#!/usr/bin/env python3
"""E6 analysis: pooled vs pair-weighted within-task AUROC of EarlyEval's released
pipeline's own test predictions (runs/<run>/reports/test_predictions_all_models.parquet).

Rows are (trajectory, prefix_step_idx). At a fixed checkpoint k each surviving
trajectory contributes exactly one prediction, so no reweighting is needed. Units are
SWE instances; within-task AUROC is pair-weighted over units with both outcomes among
trajectories alive at k; w is the same-unit share of label-discordant pairs. The
permutation test shuffles labels within units (B draws), two-sided on |A_w - 0.5|.
Controls: label-as-score (expect within = 1.0) and within-unit score shuffle (~0.5).
"""
import argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 30, 50])
ap.add_argument("--final", action="store_true", default=True,
                help="also evaluate each trajectory's LAST prediction (end of run)")
ap.add_argument("--perms", type=int, default=600)
ap.add_argument("--perm-models", nargs="+", default=[
    "prob__N_LightGBM_TfIdf_Full", "prob__K_LightGBM_Dense_Full",
    "prob__I_LightGBM_Dense_AF", "prob__H_LightGBM_Dense",
    "prob__D_Dense_Full_LR", "prob__G_TfIdf_Full_LR",
    "prob__Abl_ProcessOnly_LightGBM", "prob__Abl_ProcessOnly_LR"])
ap.add_argument("--out", default=None)
a = ap.parse_args()

df = pd.read_parquet(a.pred)
df = df[df["split"] == "test"].copy()
models = [c for c in df.columns if c.startswith("prob__")]
print(f"[data] {len(df)} test rows | {df.traj_id.nunique()} trajs | "
      f"{df.instance_id.nunique()} instances | {len(models)} model columns", flush=True)

def within_pooled(u, y, s):
    """pooled AUROC, pair-weighted within AUROC, w, mixed-unit count."""
    P, N = s[y == 1], s[y == 0]
    pooled = float("nan")
    if len(P) and len(N):
        pooled = (( P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()) / (len(P) * len(N))
    num = den = 0.0; nu = 0
    for g in pd.unique(u):
        m = u == g
        Pp, Nn = s[m & (y == 1)], s[m & (y == 0)]
        if not len(Pp) or not len(Nn): continue
        nu += 1
        num += (Pp[:, None] > Nn[None, :]).sum() + 0.5 * (Pp[:, None] == Nn[None, :]).sum()
        den += len(Pp) * len(Nn)
    w = den / (len(P) * len(N)) if len(P) and len(N) else float("nan")
    return float(pooled), (num / den if den else float("nan")), float(w), nu

def perm_p(u, y, s, B, rng):
    obs = within_pooled(u, y, s)[1]
    units = {}
    for g in pd.unique(u):
        m = u == g
        if y[m].min() != y[m].max(): units[g] = (s[m], int(y[m].sum()))
    if not units or not np.isfinite(obs): return obs, float("nan")
    null = np.empty(B)
    for b in range(B):
        num = den = 0.0
        for sc, P in units.values():
            lab = np.zeros(len(sc), int)
            lab[rng.choice(len(sc), P, replace=False)] = 1
            Pp, Nn = sc[lab == 1], sc[lab == 0]
            num += (Pp[:, None] > Nn[None, :]).sum() + 0.5 * (Pp[:, None] == Nn[None, :]).sum()
            den += len(Pp) * len(Nn)
        null[b] = num / den
    return obs, float((1 + np.sum(np.abs(null - 0.5) >= abs(obs - 0.5))) / (1 + B))

def eval_at(sub, tag, res, rng):
    u = sub["instance_id"].values
    y = sub["label"].values.astype(int)
    if y.min() == y.max():
        print(f"  [{tag}] degenerate labels, skipped", flush=True); return
    # controls (metric machinery)
    _, w_lab, _, nu0 = within_pooled(u, y, y.astype(float))
    ssh = np.empty(len(y))
    for g in pd.unique(u):
        m = np.where(u == g)[0]
        ssh[m] = rng.permutation(y[m].astype(float))  # scores carry no info beyond unit
    _, w_shuf, _, _ = within_pooled(u, y, ssh)
    res[f"{tag}|__controls__"] = dict(label_as_score_within=w_lab, shuffled_within=w_shuf,
                                      mixed_units=nu0, trajs=int(len(y)))
    print(f"  [{tag}] {len(y)} trajs, {nu0} mixed units | controls: label-as-score "
          f"within={w_lab:.3f}, shuffled={w_shuf:.3f}", flush=True)
    for c in models:
        s = sub[c].values.astype(float)
        pooled, wv, w, nu = within_pooled(u, y, s)
        entry = dict(pooled=pooled, within=wv, w=w, mixed_units=nu, trajs=int(len(y)))
        if c in a.perm_models and np.isfinite(wv):
            _, p = perm_p(u, y, s, a.perms, rng)
            entry["perm_p"] = p
        res[f"{tag}|{c}"] = entry
        star = " *" if c in a.perm_models else ""
        pp = f" p={entry.get('perm_p'):.4f}" if "perm_p" in entry else ""
        print(f"    {c[6:]:38s} pooled={pooled:.3f} within={wv if np.isfinite(wv) else float('nan'):.3f} "
              f"w={w:.4f} ({nu} units){pp}{star}", flush=True)

res = {}
rng = np.random.default_rng(0)
for k in a.ks:
    sub = df[df["prefix_step_idx"] == k - 1]
    if not len(sub): continue
    eval_at(sub, f"step{k}", res, rng)
if a.final:
    sub = df.loc[df.groupby("traj_id")["prefix_step_idx"].idxmax()]
    eval_at(sub, "final", res, rng)

if a.out:
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}", flush=True)
