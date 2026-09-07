#!/usr/bin/env python3
"""Fair deployment comparison (review P0 Phase 3, sections 7/8/19/20).

Restructures the main deployment table so the incremental sources of value are
explicit, with no strawman baselines:

  round-robin                     no information
  static transferred ranking      cross-model prior, no online correction
  Thompson, uninformative prior   online adaptation alone
  Thompson + transferred prior    prior + online adaptation
  abort @ median (untuned)        the old baseline, kept for reference
  tuned abort only                threshold chosen on held-out TUNE tasks
  Thompson+transferred + tuned    does the monitor add anything to allocation?

Fairness protocol:
  * tasks are split 50/50 into TUNE and EVAL (fixed seed);
  * abort thresholds are chosen per budget on TUNE replay to maximise
    distinct tasks solved under budget, separately for the abort-only and the
    hybrid policy; evaluation touches EVAL tasks only, once;
  * all policies are evaluated on the same EVAL pool with paired seeds
    (trial s uses the same rng stream for every policy);
  * we report mean +/- 95% percentile CI over trials and the paired
    differences  hybrid - allocate  and  tuned-abort - allocate.

Replay semantics (documented for review section 21): each pull draws one of the
task's recorded runs uniformly WITH replacement (the posterior approximates the
per-task success probability, not a finite population); cost and outcome stay
coupled because they come from the same recorded run; a task retires on first
success; aborting charges only the tokens consumed through turn k. The
transferred prior comes from the OTHER acting model's recorded outcomes
(C4-Q for this C4-L replay), so prior and replay outcomes are disjoint.
"""
import argparse, collections, csv, glob, json, math, os, re
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--traj", default="data/lph/swebench/laguna_xs2_full")
ap.add_argument("--runs", default="data/runs_laguna.csv")
ap.add_argument("--runs-prior", default="data/runs_qwen.csv",
                help="other acting model's outcomes -> transferred prior")
ap.add_argument("--budgets", type=float, nargs="+", default=[1e6, 2.5e6, 5e6, 1e7])
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--trials", type=int, default=150)
ap.add_argument("--tune-trials", type=int, default=30)
ap.add_argument("--quantiles", type=float, nargs="+",
                default=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
ap.add_argument("--split-seed", type=int, default=42)
ap.add_argument("--out", default=None)
a = ap.parse_args()

VOCAB = {"cd","cat","sed","grep","python","python3","find","echo","ls","awk","head","tail",
         "git","pytest","mkdir","touch","rm","mv","cp","chmod","which","diff","make","pip"}
abstract = lambda t: (t if t in VOCAB else "other_cmd")

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
    key = (task, rn) if (task, rn) in out else (task, rn-1)
    if key not in out: continue
    try: d = json.load(open(fp))
    except Exception: continue
    ch = d.get("command_history") or []
    acts, sizes = [], []
    for c in ch:
        t = c if isinstance(c, str) else (c.get("command") if isinstance(c, dict) else None)
        if not t: continue
        acts.append(abstract(str(t).strip().split()[0].lower() if str(t).strip() else ""))
        sizes.append(len(str(t)))
    if len(acts) < 2: continue
    succ, tok = out[key]
    tot = max(sum(sizes), 1)
    cum = np.cumsum([s * tok / tot for s in sizes])
    runs[task].append(dict(succ=succ, tok=float(tok), acts=acts, cum=cum))

tasks_all = sorted(t for t in runs if t in qwen)   # need a transferred prior
print(f"[data] {len(tasks_all)} tasks with C4-Q prior | "
      f"{sum(len(runs[t]) for t in tasks_all)} runs", flush=True)

# ---- prefix monitor at turn k, cross-fitted by task (identical to the paper's)
flat = [(t, i) for t in tasks_all for i in range(len(runs[t]))]
y = np.array([1 - runs[t][i]["succ"] for t, i in flat])
grp = np.array([t for t, _ in flat])
def fsm_of(idx, k):
    tr = collections.defaultdict(collections.Counter)
    for j in idx:
        t, i = flat[j]; s = "<init>"
        for x in runs[t][i]["acts"][:k]: tr[s][x] += 1; s = x
    o = {}
    for s, c in tr.items():
        keep = {x: n for x, n in c.items() if n >= 2 or len(c) == 1} or dict(c)
        tt = sum(keep.values()); o[s] = {x: n/tt for x, n in keep.items()}
    al = set(); [al.update(c) for c in tr.values()]
    return o, sorted(al)
def feat(j, k, fs, al, ai):
    t, i = flat[j]; seq = runs[t][i]["acts"][:k]; n = len(seq)
    v = np.zeros(len(al))
    for x in seq:
        q = ai.get(x)
        if q is not None: v[q] += 1
    v /= max(n, 1)
    FL = 1e-4; sur, s = [], "<init>"
    for x in seq: sur.append(-math.log(max(fs.get(s, {}).get(x, FL), FL))); s = x
    sur = np.array(sur) if sur else np.array([0.0])
    return np.concatenate([v, [sur.mean(), sur.max(), n, len(set(seq)),
                               1.0 - len(set(seq))/max(n,1)]])
score = np.zeros(len(flat))
for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(flat)), y, grp):
    fs, al = fsm_of(tr, a.k); ai = {x: q for q, x in enumerate(al)}
    mk = lambda ii: np.vstack([feat(j, a.k, fs, al, ai) for j in ii])
    c = HistGradientBoostingClassifier(max_iter=150, max_depth=3, random_state=0)
    c.fit(mk(tr), y[tr]); score[te] = c.predict_proba(mk(te))[:, 1]
SC = collections.defaultdict(dict)
for q, (t, i) in enumerate(flat): SC[t][i] = float(score[q])
print(f"[monitor] cross-fitted prefix predictor at k={a.k}", flush=True)

# ---- TUNE / EVAL split over tasks
sp = np.random.default_rng(a.split_seed)
perm = list(tasks_all); sp.shuffle(perm)
TUNE = sorted(perm[: len(perm)//2]); EVAL = sorted(perm[len(perm)//2:])
print(f"[split] {len(TUNE)} TUNE tasks / {len(EVAL)} EVAL tasks (seed {a.split_seed})",
      flush=True)

def cost_at(r, k):
    return float(r["cum"][min(k, len(r["cum"])) - 1]) if len(r["cum"]) else r["tok"]

def q_prior(t, w=1.0):
    s = sum(qwen[t]); n = len(qwen[t])
    return 1.0 + w * s, 1.0 + w * (n - s)

def simulate(policy, pool, budget, rg, thresh=None):
    """policy in {uniform, static, ts_cold, ts_warm, abort, hybrid}.
    abort  = uniform + abort@k;  hybrid = ts_warm + abort@k."""
    alive = set(pool); spent = 0.0; solved = 0
    warm = policy in ("ts_warm", "hybrid")
    ab = {t: (list(q_prior(t)) if warm else [1.0, 1.0]) for t in pool}
    cmean = {t: float(np.mean([r["tok"] for r in runs[t]])) for t in pool}
    stat = {t: (sum(qwen[t]) / len(qwen[t])) / max(cmean[t], 1.0) for t in pool}
    sorder = sorted(pool, key=lambda t: -stat[t]); sptr = 0
    order = list(pool); rg.shuffle(order); ptr = 0
    while alive and spent < budget:
        if policy in ("ts_cold", "ts_warm", "hybrid"):
            best, bv = None, -1.0
            for t in alive:
                v = rg.beta(ab[t][0], ab[t][1]) / max(cmean[t], 1.0)
                if v > bv: bv, best = v, t
            cand = best
        elif policy == "static":
            # fixed priority order from the transferred prior, no online update,
            # but no sticking: a failed task waits for the next pass through the
            # list rather than being retried immediately.
            cand = None
            for _ in range(len(sorder)):
                c = sorder[sptr % len(sorder)]; sptr += 1
                if c in alive: cand = c; break
            if cand is None: break
        else:                                   # uniform / abort
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

# ---- tune abort thresholds on TUNE tasks, per budget, per policy
tune_scores = np.array([SC[t][i] for t in TUNE for i in range(len(runs[t]))])
CAND = {q: float(np.quantile(tune_scores, q)) for q in a.quantiles}
tuned = {}
print("\n[tuning on TUNE tasks only]", flush=True)
for B in a.budgets:
    for pol in ("abort", "hybrid"):
        best_q, best_v = None, -1.0
        for q, th in CAND.items():
            vals = [simulate(pol, TUNE, B, np.random.default_rng(50_000 + s), thresh=th)
                    for s in range(a.tune_trials)]
            v = float(np.mean(vals))
            if v > best_v: best_v, best_q = v, q
        tuned[(B, pol)] = CAND[best_q]
        print(f"  budget {B/1e6:4.1f}M {pol:6s}: quantile {best_q:.2f} "
              f"(thr {CAND[best_q]:.3f}) -> {best_v:.1f} tune-solved", flush=True)

# untuned reference: median on TUNE distribution
TH_MED = float(np.quantile(tune_scores, 0.5))

# ---- evaluate all policies on EVAL with paired seeds
POLS = [("round-robin",              "uniform", None),
        ("static transferred rank",  "static",  None),
        ("Thompson, uninformative",  "ts_cold", None),
        ("Thompson + transferred",   "ts_warm", None),
        ("abort @ median (untuned)", "abort",   "median"),
        ("tuned abort only",         "abort",   "tuned"),
        ("Thompson+transf + tuned",  "hybrid",  "tuned")]

res = {}
print(f"\n[evaluation on EVAL tasks, {a.trials} paired trials]", flush=True)
for B in a.budgets:
    per = {}
    for label, pol, mode in POLS:
        th = (TH_MED if mode == "median"
              else tuned[(B, pol)] if mode == "tuned" else None)
        vals = np.array([simulate(pol, EVAL, B, np.random.default_rng(9_000 + s),
                                  thresh=th) for s in range(a.trials)], float)
        per[label] = vals
        lo, hi = np.percentile(vals, [2.5, 97.5])
        res[f"{int(B)}|{label}"] = dict(mean=float(vals.mean()),
                                        ci=[float(lo), float(hi)])
    print(f"\n  budget {B/1e6:.1f}M")
    for label, _, _ in POLS:
        v = per[label]; lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"    {label:26s} {v.mean():7.1f}  [{lo:6.1f}, {hi:6.1f}]")
    base = per["Thompson + transferred"]
    for other in ("Thompson+transf + tuned", "tuned abort only"):
        d = per[other] - base
        lo, hi = np.percentile(d, [2.5, 97.5])
        res[f"{int(B)}|diff|{other}"] = dict(mean=float(d.mean()),
                                             ci=[float(lo), float(hi)])
        print(f"    paired diff  {other:26s} - allocate = "
              f"{d.mean():+6.1f}  [{lo:+6.1f}, {hi:+6.1f}]")

if a.out:
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\n[saved] {a.out}")
