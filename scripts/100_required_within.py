#!/usr/bin/env python3
"""E2 (pre-registered): how much within-task AUROC would a monitor need before aborting
adds throughput to the paper's allocation policy?

The fair replay harness of 98_deployment_fair.py, unchanged (50/50 TUNE/EVAL task split,
paired seeds, transferred prior from C4-Q, abort thresholds tuned per budget on TUNE
only), with the real monitor replaced by SYNTHETIC scores of controlled within-unit
discrimination and zero between-unit component:

    score(run) = +d'/2 if fail else -d'/2, plus N(0,1) noise, then demeaned within unit,
    d' chosen so within-task AUROC hits each target; achieved value verified per table.

Reported per budget: paired tasks-solved gain of (allocation+tuned monitor) over
allocation, and of (tuned abort-only) over round-robin, versus target within-task AUROC,
with 95% percentile CIs over paired trials; the crossing level is the smallest target
whose CI excludes zero. Decision rule pre-registered in PREREGISTRATION.md.
"""
import argparse, collections, csv, glob, json, math, os, re
import numpy as np
from scipy.stats import norm

ap = argparse.ArgumentParser()
ap.add_argument("--traj", default="data/lph/swebench/laguna_xs2_full")
ap.add_argument("--runs", default="data/runs_laguna.csv")
ap.add_argument("--runs-prior", default="data/runs_qwen.csv")
ap.add_argument("--budgets", type=float, nargs="+", default=[1e6, 2.5e6, 5e6, 1e7])
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--targets", type=float, nargs="+",
                default=[0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00])
ap.add_argument("--trials", type=int, default=150)
ap.add_argument("--tune-trials", type=int, default=30)
ap.add_argument("--quantiles", type=float, nargs="+",
                default=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
ap.add_argument("--split-seed", type=int, default=42)
ap.add_argument("--task-weight", type=float, default=0.0,
                help=">0: family B - add task-level component (transferred-prior fail "
                     "rate x weight), no unit demeaning")
ap.add_argument("--out", default=None)
a = ap.parse_args()

out = {}
for r in csv.DictReader(open(a.runs)):
    out[(r["task"], int(r["run"]))] = (int(r["success"]), int(r["tokens"]))
qwen = collections.defaultdict(list)
for r in csv.DictReader(open(getattr(a, "runs_prior"))):
    qwen[r["task"]].append(int(r["success"]))

RE = re.compile(r"^(.*)_run(\d+)\.json$")
runs = collections.defaultdict(list)
for fp in sorted(glob.glob(os.path.join(a.traj, "*_run*.json"))):
    m = RE.match(os.path.basename(fp))
    if not m: continue
    task, rn = m.group(1), int(m.group(2))
    key = (task, rn) if (task, rn) in out else (task, rn - 1)
    if key not in out: continue
    try: d = json.load(open(fp))
    except Exception: continue
    ch = d.get("command_history") or []
    sizes = [len(str(c if isinstance(c, str) else (c.get("command") if isinstance(c, dict) else ""))) for c in ch]
    sizes = [s for s in sizes if s]
    if len(sizes) < 2: continue
    succ, tok = out[key]
    tot = max(sum(sizes), 1)
    cum = np.cumsum([s * tok / tot for s in sizes])
    runs[task].append(dict(succ=succ, tok=float(tok), cum=cum))
tasks_all = sorted(t for t in runs if t in qwen)
print(f"[data] {len(tasks_all)} tasks | {sum(len(runs[t]) for t in tasks_all)} runs", flush=True)

sp = np.random.default_rng(a.split_seed)
perm = list(tasks_all); sp.shuffle(perm)
TUNE = sorted(perm[: len(perm) // 2]); EVAL = sorted(perm[len(perm) // 2:])
print(f"[split] {len(TUNE)} TUNE / {len(EVAL)} EVAL (seed {a.split_seed})", flush=True)

def cost_at(r, k):
    return float(r["cum"][min(k, len(r["cum"])) - 1]) if len(r["cum"]) else r["tok"]

def q_prior(t, w=1.0):
    s = sum(qwen[t]); n = len(qwen[t])
    return 1.0 + w * s, 1.0 + w * (n - s)

def make_scores(target, seed, tw=0.0):
    """Fixed synthetic score table: within-unit separation d'.
    tw=0 (family A): unit-demeaned, zero between-unit component.
    tw>0 (family B): plus tw * transferred-prior failure rate, NOT demeaned."""
    rng = np.random.default_rng(seed)
    if target >= 0.999:
        dprime = None                       # deterministic
    else:
        dprime = math.sqrt(2.0) * norm.ppf(target)
    SC = {}
    for t in tasks_all:
        ys = np.array([1 - r["succ"] for r in runs[t]], float)   # 1 = fail
        if dprime is None:
            s = ys * 4.0
        else:
            s = (ys - 0.5) * dprime + rng.standard_normal(len(ys))
        if tw > 0:
            qhat = 1.0 - sum(qwen[t]) / len(qwen[t])   # transferred fail rate
            s = s + tw * qhat
        else:
            s -= s.mean()                   # kill the between-unit component
        SC[t] = {i: float(v) for i, v in enumerate(s)}
    return SC

def pooled_auc(SC, pool):
    ss, yy = [], []
    for t in pool:
        for i, r in enumerate(runs[t]):
            ss.append(SC[t][i]); yy.append(1 - r["succ"])
    ss = np.array(ss); yy = np.array(yy)
    P, N = ss[yy == 1], ss[yy == 0]
    rng2 = np.random.default_rng(1)
    ii = rng2.integers(0, len(P), 200_000); jj = rng2.integers(0, len(N), 200_000)
    return float(((P[ii] > N[jj]) + 0.5 * (P[ii] == N[jj])).mean())

def within_auc(SC, pool):
    num = den = 0.0
    for t in pool:
        ys = np.array([1 - r["succ"] for r in runs[t]])
        if ys.min() == ys.max(): continue
        s = np.array([SC[t][i] for i in range(len(ys))])
        P, N = s[ys == 1], s[ys == 0]
        num += sum((p > q) + 0.5 * (p == q) for p in P for q in N)
        den += len(P) * len(N)
    return num / den if den else float("nan")

def simulate(policy, pool, budget, rg, SC=None, thresh=None):
    alive = set(pool); spent = 0.0; solved = 0
    warm = policy in ("ts_warm", "hybrid")
    ab = {t: (list(q_prior(t)) if warm else [1.0, 1.0]) for t in pool}
    cmean = {t: float(np.mean([r["tok"] for r in runs[t]])) for t in pool}
    order = list(pool); rg.shuffle(order); ptr = 0
    while alive and spent < budget:
        if policy in ("ts_warm", "hybrid"):
            best, bv = None, -1.0
            for t in alive:
                v = rg.beta(ab[t][0], ab[t][1]) / max(cmean[t], 1.0)
                if v > bv: bv, best = v, t
            cand = best
        else:
            cand = None
            for _ in range(len(order)):
                c = order[ptr % len(order)]; ptr += 1
                if c in alive: cand = c; break
            if cand is None: break
        i = rg.integers(len(runs[cand])); r = runs[cand][i]
        if policy in ("abort", "hybrid") and thresh is not None and SC[cand][i] >= thresh:
            spent += cost_at(r, a.k); got = 0
        else:
            spent += r["tok"]; got = r["succ"]
        if got: solved += 1; alive.discard(cand)
        else: ab[cand][1] += 1.0
    return solved

# ---- level-independent baselines, paired seeds
res = {}
base = {}
for B in a.budgets:
    base[("ts_warm", B)] = np.array([simulate("ts_warm", EVAL, B,
                            np.random.default_rng(9000 + s)) for s in range(a.trials)], float)
    base[("uniform", B)] = np.array([simulate("uniform", EVAL, B,
                            np.random.default_rng(9000 + s)) for s in range(a.trials)], float)
    print(f"[base] {B/1e6:.1f}M  allocate={base[('ts_warm',B)].mean():.1f}  "
          f"round-robin={base[('uniform',B)].mean():.1f}", flush=True)

for li, target in enumerate(a.targets):
    SC = make_scores(target, 1000 + li, tw=getattr(a, "task_weight"))
    aw_eval = within_auc(SC, EVAL); aw_tune = within_auc(SC, TUNE)
    pa = pooled_auc(SC, EVAL)
    print(f"\n=== target within {target:.2f} | achieved EVAL within {aw_eval:.3f} "
          f"pooled {pa:.3f} | TUNE {aw_tune:.3f} ===", flush=True)
    tune_scores = np.array([SC[t][i] for t in TUNE for i in range(len(runs[t]))])
    CAND = {q: float(np.quantile(tune_scores, q)) for q in a.quantiles}
    for B in a.budgets:
        tuned = {}
        for pol in ("abort", "hybrid"):
            bq, bv = None, -1.0
            for q, th in CAND.items():
                v = float(np.mean([simulate(pol, TUNE, B, np.random.default_rng(50000 + s),
                                            SC=SC, thresh=th) for s in range(a.tune_trials)]))
                if v > bv: bv, bq = v, q
            tuned[pol] = CAND[bq]
        hyb = np.array([simulate("hybrid", EVAL, B, np.random.default_rng(9000 + s),
                                 SC=SC, thresh=tuned["hybrid"]) for s in range(a.trials)], float)
        abo = np.array([simulate("abort", EVAL, B, np.random.default_rng(9000 + s),
                                 SC=SC, thresh=tuned["abort"]) for s in range(a.trials)], float)
        d1 = hyb - base[("ts_warm", B)]; d2 = abo - base[("uniform", B)]
        lo1, hi1 = np.percentile(d1, [2.5, 97.5]); lo2, hi2 = np.percentile(d2, [2.5, 97.5])
        res[f"{target:.2f}|{int(B)}"] = dict(
            achieved_within=float(aw_eval),
            hybrid_minus_allocate=[float(d1.mean()), float(lo1), float(hi1)],
            abort_minus_rr=[float(d2.mean()), float(lo2), float(hi2)])
        print(f"  {B/1e6:4.1f}M  hybrid-allocate {d1.mean():+6.1f} [{lo1:+6.1f},{hi1:+6.1f}]"
              f"   abort-RR {d2.mean():+6.1f} [{lo2:+6.1f},{hi2:+6.1f}]", flush=True)

if a.out:
    json.dump(res, open(a.out, "w"), indent=2); print(f"\n[saved] {a.out}")
