#!/usr/bin/env python3
"""Does treating retry-allocation as a budgeted bandit beat the alternatives?

Our finding — within-unit outcomes are Bernoulli(p_u) with no readable run-level
signal — means each TASK is exactly a Bernoulli arm with a random token cost.
That licenses a budgeted multi-armed bandit, and gives us something no one else
has for the cold-start problem: an informative prior transferred from a DIFFERENT
model (per-task failure rates correlate at r=0.863 across our two acting models).

Objective: maximise the number of DISTINCT tasks solved within a token budget.
A solved task is removed -- you get no credit for solving the same issue twice.

Replay protocol: pulling task u draws one of its recorded runs at random (with
replacement) and returns that run's real outcome and real token count. No
simulated agents; every number came off a real trajectory.
"""
import argparse, csv, collections
import numpy as np
from scipy.stats import beta as beta_dist

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="data/runs_qwen.csv", help="model we deploy on")
ap.add_argument("--prior",  default="data/runs_laguna.csv", help="model we warm-start from")
ap.add_argument("--budget", type=float, default=2.0e7)
ap.add_argument("--trials", type=int, default=200)
ap.add_argument("--w", type=float, default=1.0, help="prior weight (pseudo-runs)")
ap.add_argument("--give-up", type=float, default=0.05,
                help="retire a task when P(success rate > this) < 0.05")
a = ap.parse_args()

def load(p):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(p)):
        by[r["task"]].append((int(r["success"]), int(r["tokens"])))
    return by

T, P = load(a.target), load(a.prior)
tasks = sorted(set(T) & set(P))
print(f"[data] {len(tasks)} shared tasks | budget {a.budget/1e6:.0f}M tokens | {a.trials} trials")

# ground truth for the oracle, and mean cost per task
p_true = {t: np.mean([s for s, _ in T[t]]) for t in tasks}          # P(success)
c_mean = {t: float(np.mean([c for _, c in T[t]])) for t in tasks}
solvable = sum(1 for t in tasks if p_true[t] > 0)
print(f"[data] {solvable} tasks are solvable at all (p_success > 0)\n")

def transferred_prior(t, w):
    """Beta prior from the OTHER model's outcomes on this task."""
    s = sum(x for x, _ in P[t]); n = len(P[t])
    return 1.0 + w * s, 1.0 + w * (n - s)

def run_policy(kind, rng, w=1.0, give_up=None, cost_sample=False):
    """Return number of distinct tasks solved before the budget runs out."""
    alive = set(tasks)
    if kind == "ts_warm":
        ab = {t: list(transferred_prior(t, w)) for t in tasks}
    else:
        ab = {t: [1.0, 1.0] for t in tasks}
    spent, solved = 0.0, 0
    order = list(tasks); rng.shuffle(order); ptr = 0
    costs = {t: [c for _, c in T[t]] for t in tasks}
    seen = {t: 0 for t in tasks}; wins = {t: 0 for t in tasks}
    stat_score = {t: (sum(x for x, _ in P[t]) / len(P[t])) / c_mean[t] for t in tasks}
    stat_order = sorted(tasks, key=lambda t: -stat_score[t]); sptr = 0
    while alive and spent < a.budget:
        if kind == "uniform":
            for _ in range(len(order)):
                cand = order[ptr % len(order)]; ptr += 1
                if cand in alive: break
        elif kind == "oracle":
            cand = max(alive, key=lambda t: p_true[t] / c_mean[t])
        elif kind == "static":
            # rank by transferred difficulty, never update -- but walk the ranked
            # list in fixed priority order (a failed task waits for the next
            # pass) rather than retrying the top task forever.
            cand = None
            for _ in range(len(stat_order)):
                c = stat_order[sptr % len(stat_order)]; sptr += 1
                if c in alive: cand = c; break
            if cand is None: break
        elif kind == "own_hist":
            # try everything once, then exploit what worked -- what a competent
            # engineer builds without any of our machinery
            unseen = [t for t in alive if seen[t] == 0]
            if unseen: cand = unseen[0]
            else: cand = max(alive, key=lambda t: (wins[t] / max(seen[t], 1)) / c_mean[t])
        else:                                    # thompson sampling
            best, bv = None, -1.0
            for t in alive:
                th = rng.beta(ab[t][0], ab[t][1])
                # P7: cost is a random variable, not its mean. Budgeted TS samples
                # BOTH reward and cost posteriors; using c_mean throws away the fact
                # that some tasks are cheap on a good day.
                c = (costs[t][rng.integers(len(costs[t]))] if cost_sample else c_mean[t])
                v = th / max(c, 1.0)
                if v > bv: bv, best = v, t
            cand = best
        s, c = T[cand][rng.integers(len(T[cand]))]     # replay a real run
        spent += c; seen[cand] += 1; wins[cand] += s
        if s:
            solved += 1; alive.discard(cand)
        else:
            ab[cand][1] += 1.0
            # P6: give up entirely. The bandit already de-prioritises a failing
            # task, but never removes it, so it keeps drawing occasional pulls.
            # Retire a task once the posterior says it is very unlikely to clear
            # the success rate that would justify its cost.
            if give_up is not None:
                al, be = ab[cand]
                if beta_dist.sf(give_up, al, be) < 0.05:
                    alive.discard(cand)
    return solved

rng = np.random.default_rng(0)
EXTRA = {}
policies = [("uniform (round-robin)", "uniform", None),
            ("greedy on own history", "own_hist", None),
            ("STATIC transferred prior (our §7)", "static", None),
            ("bandit, no prior", "ts_cold", None),
            ("bandit + transferred prior", "ts_warm", a.w),
            ("  + P6 give-up rule", "ts_warm", a.w),
            ("  + P7 cost sampling", "ts_warm", a.w),
            ("  + P6 and P7 together", "ts_warm", a.w),
            ("oracle (knows true p_u)", "oracle", None)]
EXTRA = {"  + P6 give-up rule":   dict(give_up=a.give_up),
         "  + P7 cost sampling":  dict(cost_sample=True),
         "  + P6 and P7 together": dict(give_up=a.give_up, cost_sample=True)}
print(f"  {'policy':32s} {'tasks solved':>14} {'vs uniform':>12}")
base = None
for name, kind, w in policies:
    kw = EXTRA.get(name, {})
    vals = [run_policy(kind, np.random.default_rng(1000 + i), w or 1.0, **kw)
            for i in range(a.trials if kind != "oracle" else max(a.trials // 4, 20))]
    m, sd = float(np.mean(vals)), float(np.std(vals))
    if base is None: base = m
    print(f"  {name:32s} {m:>8.1f} ± {sd:<4.1f} {100*(m-base)/base:>+11.1f}%")
