#!/usr/bin/env python3
"""E11: turn control as a deployment baseline.

Turn control (Gao & Peng, ICSE 2026) is the strongest current SE-specific compute-control
method for coding agents: cap a run's length rather than predict its outcome. We cite it
but had not run it, so this adds it to the replay on the same data, splits, seeds and
budgets as every other policy.

A cap of C steps is charged faithfully: a run that would have succeeded only after more
than C steps is charged its cost through step C and returns failure; a run that succeeds
within C steps is charged and credited normally. Caps are set at percentiles of
SUCCESSFUL-run length measured on the tuning split only, so no cap sees the evaluation
tasks.

Policies compared: round-robin; round-robin + cap; allocation (Thompson, uninformative
prior - the strongest baseline on C2 per E9); allocation + cap. The last isolates
whether capping adds anything ON TOP of allocating well, which is the question the paper
actually turns on.

The dynamic variant of turn control needs per-step progress signals (newly passing
tests, files touched) that our stored cost tables do not retain; we report the fixed-cap
family and say so, noting that their reported dynamic gain is a further 12-24% over
fixed.
"""
import argparse, collections, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--budgets", type=float, nargs="+", default=[2e6, 5e6, 1e7, 2.5e7])
ap.add_argument("--pctiles", type=float, nargs="+", default=[50, 75, 90])
ap.add_argument("--trials", type=int, default=120)
ap.add_argument("--split-seed", type=int, default=42)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rt = pd.read_parquet(a.runs)
runs = collections.defaultdict(list)
for r in rt.itertuples(index=False):
    cum = np.asarray(r.cum, dtype=float)
    runs[r.task].append(dict(succ=int(r.success), tok=float(r.tokens),
                             cum=cum, steps=len(cum)))
tasks = sorted(t for t in runs if len(runs[t]) >= 2)
print("[data] %d tasks | %d runs" % (len(tasks), sum(len(runs[t]) for t in tasks)),
      flush=True)

sp = np.random.default_rng(a.split_seed)
perm = list(tasks)
sp.shuffle(perm)
TUNE = sorted(perm[:len(perm) // 2])
EVAL = sorted(perm[len(perm) // 2:])

succ_len = [r["steps"] for t in TUNE for r in runs[t] if r["succ"]]
CAPS = {p: int(np.percentile(succ_len, p)) for p in a.pctiles}
print("[caps] from %d successful TUNE runs: %s" % (len(succ_len), CAPS), flush=True)
cmean = {t: float(np.mean([r["tok"] for r in runs[t]])) for t in tasks}


def draw(r, cap):
    """Charge and credit a run under a step cap."""
    if cap is None or r["steps"] <= cap:
        return r["tok"], r["succ"]
    c = r["cum"]
    return float(c[min(cap, len(c)) - 1]), 0      # cut short: cost paid, no credit


def simulate(policy, pool, budget, rg, cap=None):
    alive = set(pool)
    spent = 0.0
    solved = 0
    st = {t: [1.0, 1.0] for t in pool}
    order = list(pool)
    rg.shuffle(order)
    ptr = 0
    while alive and spent < budget:
        if policy == "alloc":
            best, bv = None, -1.0
            for t in alive:
                v = rg.beta(st[t][0], st[t][1]) / max(cmean[t], 1.0)
                if v > bv:
                    bv, best = v, t
            cand = best
        else:
            cand = None
            for _ in range(len(order)):
                c = order[ptr % len(order)]
                ptr += 1
                if c in alive:
                    cand = c
                    break
            if cand is None:
                break
        r = runs[cand][rg.integers(len(runs[cand]))]
        cost, got = draw(r, cap)
        spent += cost
        if got:
            solved += 1
            alive.discard(cand)
        else:
            st[cand][1] += 1.0
    return solved


POLICIES = [("uniform", None, "round-robin")] + \
           [("uniform", p, "round-robin + cap p%d" % p) for p in a.pctiles] + \
           [("alloc", None, "allocation")] + \
           [("alloc", p, "allocation + cap p%d" % p) for p in a.pctiles]

res = {}
print("\n%-26s" % "policy" + "".join("%12s" % ("%.0fM" % (b / 1e6)) for b in a.budgets),
      flush=True)
for pol, p, label in POLICIES:
    cap = CAPS[p] if p is not None else None
    row = []
    for b in a.budgets:
        v = [simulate(pol, EVAL, b, np.random.default_rng(4400 + s), cap=cap)
             for s in range(a.trials)]
        row.append(float(np.mean(v)))
        res.setdefault(label, {})[str(int(b))] = [
            float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    print("%-26s" % label + "".join("%12.1f" % x for x in row), flush=True)

print("\n=== does capping add anything on top of allocating well? ===", flush=True)
for p in a.pctiles:
    lab = "allocation + cap p%d" % p
    row = []
    for b in a.budgets:
        d = res[lab][str(int(b))][0] - res["allocation"][str(int(b))][0]
        row.append(d)
    print("%-26s" % ("cap p%d minus allocation" % p) +
          "".join("%+12.1f" % x for x in row), flush=True)

json.dump(res, open(a.out, "w"), indent=2)
print("\n[saved] %s" % a.out, flush=True)
