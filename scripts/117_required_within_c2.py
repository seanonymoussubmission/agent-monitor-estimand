#!/usr/bin/env python3
"""E2-E: the deployment requirement curve estimated on a SECOND corpus.

The crossing point -- the within-task AUROC at which aborting begins to add throughput
over allocation -- previously came from C4-L alone, and it is the number that converts a
measured 0.52 into "an order of magnitude short". Here it is re-estimated on C2
(SWE-rebench / OpenHands / Qwen), using the cost table built for E9 and the same
protocol: 50/50 TUNE/EVAL task split, abort thresholds tuned per budget on TUNE only,
paired seeds, evaluation touching EVAL once.

The allocation baseline is the STRONGEST allocation policy available on this corpus.
E9 found that on C2 a transferred prior from C1 hurts (the source model is far weaker),
so the baseline here is Thompson sampling with an uninformative prior; using the weaker
warm-start variant would flatter the monitor.

Monitor families, as pre-registered: A (unit-demeaned, zero between-unit component) and
D (per-task noise sd scaled by task difficulty, d' bisected so achieved within-task
AUROC still hits each target).
"""
import argparse, collections, glob, json, math
import numpy as np, pandas as pd
from scipy.stats import norm

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--c1-glob", default="data/swe_agent_traj/data/train-*.parquet")
ap.add_argument("--budgets", type=float, nargs="+", default=[2e6, 5e6, 1e7, 2.5e7])
ap.add_argument("--targets", type=float, nargs="+",
                default=[0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00])
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--trials", type=int, default=120)
ap.add_argument("--tune-trials", type=int, default=25)
ap.add_argument("--quantiles", type=float, nargs="+",
                default=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
ap.add_argument("--corr", type=float, default=0.0, help=">0 = family D")
ap.add_argument("--split-seed", type=int, default=42)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rt = pd.read_parquet(a.runs)
runs = collections.defaultdict(list)
for r in rt.itertuples(index=False):
    runs[r.task].append(dict(succ=int(r.success), tok=float(r.tokens),
                             cum=np.asarray(r.cum, dtype=float)))

# difficulty proxy for family D: C1's failure rate on the same task where available
c1 = collections.defaultdict(lambda: [0, 0])
for p in sorted(glob.glob(a.c1_glob)):
    d = pd.read_parquet(p, columns=["instance_id", "target"])
    for iid, tgt in d.itertuples(index=False):
        s = c1[str(iid)]
        s[0] += int(bool(tgt))
        s[1] += 1

tasks_all = sorted(t for t in runs if len(runs[t]) >= 2)
print("[data] %d tasks | %d runs | mean %.1f runs/task"
      % (len(tasks_all), sum(len(runs[t]) for t in tasks_all),
         np.mean([len(runs[t]) for t in tasks_all])), flush=True)

sp = np.random.default_rng(a.split_seed)
perm = list(tasks_all)
sp.shuffle(perm)
TUNE = sorted(perm[:len(perm) // 2])
EVAL = sorted(perm[len(perm) // 2:])
print("[split] %d TUNE / %d EVAL (seed %d)" % (len(TUNE), len(EVAL), a.split_seed),
      flush=True)

cmean = {t: float(np.mean([r["tok"] for r in runs[t]])) for t in tasks_all}


def cost_at(r, k):
    c = r["cum"]
    return float(c[min(k, len(c)) - 1]) if len(c) else r["tok"]


def difficulty_z():
    q = []
    for t in tasks_all:
        s, n = c1[t]
        q.append(1.0 - (s + 0.5) / (n + 1.0) if n else 0.5)
    q = np.asarray(q)
    return dict(zip(tasks_all, (q - q.mean()) / (q.std() + 1e-9)))


ZD = difficulty_z() if a.corr > 0 else {}


def build(dprime, sigma, rng):
    SC = {}
    for t in tasks_all:
        ys = np.array([1 - r["succ"] for r in runs[t]], float)
        if dprime is None:
            s = ys * 4.0
        else:
            s = (ys - 0.5) * dprime + sigma.get(t, 1.0) * rng.standard_normal(len(ys))
        s -= s.mean()
        SC[t] = {i: float(v) for i, v in enumerate(s)}
    return SC


def within_auc(SC, pool):
    num = den = 0.0
    for t in pool:
        ys = np.array([1 - r["succ"] for r in runs[t]])
        if ys.min() == ys.max():
            continue
        s = np.array([SC[t][i] for i in range(len(ys))])
        P, N = s[ys == 1], s[ys == 0]
        num += sum((p > q) + 0.5 * (p == q) for p in P for q in N)
        den += len(P) * len(N)
    return num / den if den else float("nan")


def make_scores(target, seed):
    rng = np.random.default_rng(seed)
    if target >= 0.999:
        return build(None, {}, rng)
    base = math.sqrt(2.0) * norm.ppf(target)
    if a.corr <= 0:
        return build(base, {}, rng)
    sigma = {t: float(max(0.2, 1.0 + a.corr * ZD.get(t, 0.0))) for t in tasks_all}
    lo, hi = base, base * 5.0
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if within_auc(build(mid, sigma, np.random.default_rng(seed)), tasks_all) < target:
            lo = mid
        else:
            hi = mid
    return build(0.5 * (lo + hi), sigma, np.random.default_rng(seed))


def simulate(policy, pool, budget, rg, SC=None, thresh=None):
    alive = set(pool)
    spent = 0.0
    solved = 0
    st = {t: [1.0, 1.0] for t in pool}          # uninformative: the best baseline on C2
    order = list(pool)
    rg.shuffle(order)
    ptr = 0
    while alive and spent < budget:
        if policy in ("alloc", "hybrid"):
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
        i = rg.integers(len(runs[cand]))
        r = runs[cand][i]
        if policy in ("abort", "hybrid") and thresh is not None and SC[cand][i] >= thresh:
            spent += cost_at(r, a.k)
            got = 0
        else:
            spent += r["tok"]
            got = r["succ"]
        if got:
            solved += 1
            alive.discard(cand)
        else:
            st[cand][1] += 1.0
    return solved


base = {}
for B in a.budgets:
    base[("alloc", B)] = np.array([simulate("alloc", EVAL, B, np.random.default_rng(900 + s))
                                   for s in range(a.trials)], float)
    base[("uniform", B)] = np.array([simulate("uniform", EVAL, B, np.random.default_rng(900 + s))
                                     for s in range(a.trials)], float)
    print("[base] %5.1fM  allocate=%.1f  round-robin=%.1f"
          % (B / 1e6, base[("alloc", B)].mean(), base[("uniform", B)].mean()), flush=True)

res = {}
for li, target in enumerate(a.targets):
    SC = make_scores(target, 2000 + li)
    aw = within_auc(SC, EVAL)
    print("\n=== target within %.2f | achieved EVAL within %.3f ===" % (target, aw),
          flush=True)
    tune_scores = np.array([SC[t][i] for t in TUNE for i in range(len(runs[t]))])
    CAND = {q: float(np.quantile(tune_scores, q)) for q in a.quantiles}
    for B in a.budgets:
        tuned = {}
        for pol in ("abort", "hybrid"):
            bq, bv = None, -1.0
            for q, th in CAND.items():
                v = float(np.mean([simulate(pol, TUNE, B, np.random.default_rng(5000 + s),
                                            SC=SC, thresh=th)
                                   for s in range(a.tune_trials)]))
                if v > bv:
                    bv, bq = v, q
            tuned[pol] = CAND[bq]
        hyb = np.array([simulate("hybrid", EVAL, B, np.random.default_rng(900 + s),
                                 SC=SC, thresh=tuned["hybrid"]) for s in range(a.trials)], float)
        abo = np.array([simulate("abort", EVAL, B, np.random.default_rng(900 + s),
                                 SC=SC, thresh=tuned["abort"]) for s in range(a.trials)], float)
        d1 = hyb - base[("alloc", B)]
        d2 = abo - base[("uniform", B)]
        lo1, hi1 = np.percentile(d1, [2.5, 97.5])
        lo2, hi2 = np.percentile(d2, [2.5, 97.5])
        res["%.2f|%d" % (target, int(B))] = dict(
            achieved_within=float(aw),
            hybrid_minus_alloc=[float(d1.mean()), float(lo1), float(hi1)],
            abort_minus_rr=[float(d2.mean()), float(lo2), float(hi2)])
        print("  %5.1fM  hybrid-allocate %+6.1f [%+6.1f,%+6.1f]   abort-RR %+6.1f [%+6.1f,%+6.1f]"
              % (B / 1e6, d1.mean(), lo1, hi1, d2.mean(), lo2, hi2), flush=True)

json.dump(res, open(a.out, "w"), indent=2)
print("\n[saved] %s" % a.out, flush=True)
