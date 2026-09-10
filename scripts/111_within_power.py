#!/usr/bin/env python3
"""Power and equivalence analysis for the within-task estimand (pre-registered
amendment): answers "would you have seen run-level signal if it were there?" without
new pipeline runs.

Three quantities, per checkpoint, on the REAL unit structure of the given prediction
tables (one or more folds, pooled by summing same-unit pair counts):

  (a) Observed pair-weighted within-task AUROC with a unit bootstrap 95% CI.
  (b) Empirical minimum detectable effect: synthetic scores with a controlled
      within-unit d' are injected into the same units (score = (y-0.5)*d' + N(0,1),
      d' = sqrt(2)*Phi^-1(target), as in the pre-registered E2 sweep), and the same
      two-sided test is applied. The MDE is the smallest target detected in >= 80%
      of draws.
  (c) TOST equivalence: the smallest delta for which |A_within - 0.5| < delta is
      established at 95% (one-sided bootstrap bound), i.e. a positive statement of
      the null within a tolerance.

The test statistic uses the analytic within-unit permutation null SD
(sqrt(sum_u P_u N_u (P_u+N_u+1)/12) / sum_u P_u N_u), validated against empirical
permutation nulls in the supplement; a permutation cross-check at the MDE boundary is
reported for the checkpoints named by --perm-check.
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import norm

ap = argparse.ArgumentParser()
ap.add_argument("--preds", nargs="+", required=True,
                help="one or more test_predictions parquet files (folds)")
ap.add_argument("--model", default="prob__K_LightGBM_Dense_Full",
                help="score column used for the observed value")
ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 3, 5, 10, 20, 30],
                help="steps observed; 0 = pre-run")
ap.add_argument("--targets", type=float, nargs="+",
                default=[0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70])
ap.add_argument("--draws", type=int, default=2000)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--perms", type=int, default=600)
ap.add_argument("--perm-check", type=int, nargs="*", default=[0, 1, 5])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()


def unit_arrays(sub):
    """[(scores, labels)] per unit with both outcomes present."""
    out = []
    for _, g in sub.groupby("instance_id"):
        y = g["label"].values.astype(int)
        if y.min() == y.max():
            continue
        out.append((g[a.model].values.astype(float), y))
    return out


def within_from_units(units):
    num = den = 0.0
    for s, y in units:
        P, N = s[y == 1], s[y == 0]
        num += (P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()
        den += len(P) * len(N)
    return (num / den) if den else float("nan"), den


def analytic_null_sd(units):
    num = den = 0.0
    for _, y in units:
        P, N = int((y == 1).sum()), int((y == 0).sum())
        num += P * N * (P + N + 1) / 12.0
        den += P * N
    return np.sqrt(num) / den if den else float("nan")


def synth_within(units, target, rng):
    """Scores with controlled within-unit separation on the same unit structure."""
    dprime = np.sqrt(2.0) * norm.ppf(target)
    out = []
    for _, y in units:
        s = (y - 0.5) * dprime + rng.standard_normal(len(y))
        out.append((s, y))
    return out


def perm_p(units, obs, B, rng):
    null = np.empty(B)
    for b in range(B):
        num = den = 0.0
        for s, y in units:
            lab = np.zeros(len(s), int)
            lab[rng.choice(len(s), int(y.sum()), replace=False)] = 1
            P, N = s[lab == 1], s[lab == 0]
            num += (P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()
            den += len(P) * len(N)
        null[b] = num / den
    return float((1 + np.sum(np.abs(null - 0.5) >= abs(obs - 0.5))) / (1 + B)), float(null.std())


frames = []
for i, p in enumerate(a.preds):
    df = pd.read_parquet(p, columns=["split", "instance_id", "traj_id",
                                     "prefix_step_idx", "label", a.model])
    df = df[df["split"] == "test"].copy()
    df["instance_id"] = f"f{i}::" + df["instance_id"].astype(str)  # folds are disjoint
    frames.append(df)
allp = pd.concat(frames, ignore_index=True)
print(f"[data] {len(a.preds)} fold(s) | {allp.traj_id.nunique()} trajs | "
      f"{allp.instance_id.nunique()} instances | model {a.model}", flush=True)

rng = np.random.default_rng(a.seed)
res = {}
for k in a.ks:
    sub = allp[allp["prefix_step_idx"] == k]
    if not len(sub):
        continue
    units = unit_arrays(sub)
    if not units:
        continue
    obs, pairs = within_from_units(units)
    sd = analytic_null_sd(units)
    thr = 1.96 * sd

    # (a) unit bootstrap CI
    idx = np.arange(len(units))
    boots = np.empty(a.boot)
    for b in range(a.boot):
        pick = rng.choice(idx, len(idx), replace=True)
        boots[b] = within_from_units([units[j] for j in pick])[0]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    # (b) empirical MDE
    power = {}
    for t in a.targets:
        hits = 0
        for _ in range(a.draws):
            sy = synth_within(units, t, rng)
            w, _ = within_from_units(sy)
            hits += abs(w - 0.5) > thr
        power[t] = hits / a.draws
    mde = next((t for t in a.targets if power[t] >= 0.80), None)

    # (c) TOST-style equivalence bound: smallest delta with 95% bootstrap support
    delta = float(np.percentile(np.abs(boots - 0.5), 95))

    entry = dict(within=float(obs), ci=[float(lo), float(hi)], units=len(units),
                 pairs=float(pairs), null_sd=float(sd), mde80=mde,
                 equiv_delta95=delta, power={str(t): power[t] for t in a.targets})
    if k in (a.perm_check or []):
        pv, esd = perm_p(units, obs, a.perms, rng)
        entry["perm_p"] = pv
        entry["perm_sd"] = esd
    res["prerun" if k == 0 else f"obs{k}"] = entry
    extra = (f" | perm p={entry['perm_p']:.3f} (null sd {entry['perm_sd']:.4f} vs "
             f"analytic {sd:.4f})") if "perm_p" in entry else ""
    print(f"  {('prerun' if k == 0 else f'obs{k}'):<7s} within={obs:.3f} CI[{lo:.3f},{hi:.3f}] "
          f"{len(units)} units, {pairs:.0f} pairs | null sd {sd:.4f} | "
          f"MDE(80%)={mde if mde else '>max'} | equivalent within +/-{delta:.3f}{extra}",
          flush=True)

if a.out:
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}", flush=True)
