#!/usr/bin/env python3
"""E12: difficulty-only ceilings and same-task weights outside coding, from outcome
tables alone.

Pre-registered (PREREGISTRATION.md, E12) before any download. No trajectories, no
predictor, no model inference: every quantity here is a function of which runs of which
units passed.

Per benchmark-domain we report
  - the observed mixed-outcome rate (units with >=1 success and >=1 failure),
  - the same-task pair weight w,
  - the difficulty-only pooled AUROC ceiling (plug-in and the MD closed form) and the
    measured leave-one-out oracle,
  - the analytic permutation-null SD of within-task AUROC, hence the minimum detectable
    within-task effect at 80% power,
all via `analyse` imported unchanged from 90_predict_inflation.py.

Go/no-go, registered in advance: a domain with fewer than 30 mixed-outcome units is
reported as a count only; no ceiling is estimated from it.

Usage:
  120_e12_outcome_tables.py --tau2-results <dir> [--tbench <parquet-or-dir>] --out e12.json
"""
import argparse, glob, importlib.util, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

MIN_MIXED = 30          # registered go/no-go threshold


def load_analyse():
    """Import `analyse` from 90_predict_inflation.py without running its __main__ body."""
    path = os.path.join(HERE, "90_predict_inflation.py")
    src = open(path).read()
    cut = src.index("\nout=[]")          # everything after this is that script's own main
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("_p90", loader=None))
    exec(compile(src[:cut], path, "exec"), mod.__dict__)
    return mod.analyse


def null_sd(unit_ids, fails):
    """Analytic permutation-null SD of pair-weighted within-task AUROC, and the MDE.

    SD = sqrt(sum_u P_u N_u (P_u+N_u+1)/12) / sum_u P_u N_u   over mixed units only.
    """
    units, inv = np.unique(np.asarray(unit_ids), return_inverse=True)
    fails = np.asarray(fails, float)
    num = den = 0.0
    mixed = 0
    for i in range(len(units)):
        m = inv == i
        P = float(fails[m].sum())            # failures
        N = float(m.sum() - P)               # successes
        if P == 0 or N == 0:
            continue
        mixed += 1
        num += P * N * (P + N + 1) / 12.0
        den += P * N
    if den == 0:
        return None, None, mixed, 0.0
    sd = float(np.sqrt(num) / den)
    return sd, 0.5 + 2.80 * sd, mixed, den      # 2.80 = z(.975)+z(.80)


# ---------------------------------------------------------------- tau2-bench
def tau2_tables(results_dir):
    """Released baselines: <model>_<domain>_<config>_<usersim>_4trials.json.

    A unit is (model, domain, config, task); a run is one trial; reward 1.0 = success.
    Domains are kept separate, as registered.
    """
    by_domain = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        base = os.path.basename(p)[:-5]
        parts = base.split("_")
        try:
            model, domain, config = parts[0], parts[1], parts[2]
        except IndexError:
            print("  [skip] unparsable name:", base)
            continue
        try:
            d = json.load(open(p))
        except Exception as e:
            print("  [skip] %s: %s" % (base, e))
            continue
        sims = d.get("simulations", [])
        u, f = by_domain.setdefault(domain, ([], []))
        n = 0
        for s in sims:
            ri = s.get("reward_info") or {}
            rw = ri.get("reward")
            if rw is None:
                continue
            u.append("%s|%s|%s|t%s" % (model, domain, config, s.get("task_id")))
            f.append(1 - int(round(float(rw))))      # 1 = failure
            n += 1
        print("  [tau2] %-58s domain=%-16s runs=%d" % (base[:58], domain, n))
    return by_domain


# ------------------------------------------------------- Terminal-Bench 2.0
def tbench_tables(path):
    """Leaderboard records: one row per (agent, task, trial) with a pass/fail.

    Accepts a parquet file, a directory of parquet, or a JSON/JSONL export. Column names
    vary across exports, so we search for the first plausible triple.
    """
    import pandas as pd
    files = []
    if os.path.isdir(path):
        for ext in ("*.parquet", "*.json", "*.jsonl", "*.csv"):
            files += sorted(glob.glob(os.path.join(path, "**", ext), recursive=True))
    else:
        files = [path]
    frames = []
    for p in files:
        try:
            if p.endswith(".parquet"):
                frames.append(pd.read_parquet(p))
            elif p.endswith(".csv"):
                frames.append(pd.read_csv(p))
            else:
                frames.append(pd.read_json(p, lines=p.endswith(".jsonl")))
        except Exception as e:
            print("  [skip] %s: %s" % (os.path.basename(p), e))
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    print("  [tbench] %d rows, columns: %s" % (len(df), list(df.columns)[:18]))

    def pick(cands):
        for c in cands:
            for col in df.columns:
                if col.lower() == c:
                    return col
        for c in cands:
            for col in df.columns:
                if c in col.lower():
                    return col
        return None

    agent = pick(["agent", "agent_name", "submission", "model", "run_name"])
    task = pick(["task_id", "task", "instance_id", "task_name"])
    ok = pick(["resolved", "passed", "success", "is_resolved", "reward", "score", "result"])
    trial = pick(["trial", "attempt", "run_idx", "epoch", "seed"])
    print("  [tbench] using agent=%s task=%s outcome=%s trial=%s"
          % (agent, task, ok, trial))
    if not (agent and task and ok):
        print("  [tbench] could not identify required columns; skipping")
        return {}
    u, f = [], []
    for a, t, o in zip(df[agent], df[task], df[ok]):
        if o is None or (isinstance(o, float) and np.isnan(o)):
            continue
        if isinstance(o, str):
            val = 1 if o.strip().lower() in ("true", "pass", "passed", "resolved", "1") else 0
        else:
            val = 1 if float(o) >= 0.5 else 0
        u.append("%s|%s" % (a, t))
        f.append(1 - val)
    return {"terminal-bench-2.0": (u, f)} if u else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau2-results", default=None)
    ap.add_argument("--tbench", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    analyse = load_analyse()
    tables = {}
    if a.tau2_results:
        print("[tau2-bench]")
        for k, v in tau2_tables(a.tau2_results).items():
            tables["tau2-" + k] = v
    if a.tbench:
        print("[terminal-bench]")
        tables.update(tbench_tables(a.tbench))

    # Per-cell (one acting model and configuration) ceilings. Pooling several models
    # inside a domain lets model capability, not task difficulty, widen the spread; the
    # per-cell figures hold the model fixed, as we do for C3 in the paper.
    cells = {}
    for name, (u, f) in tables.items():
        for uid, fail in zip(u, f):
            parts = uid.split("|")
            if len(parts) < 4:
                continue
            key = (name, parts[0] + "|" + parts[2])       # model | config
            cu, cf = cells.setdefault(key, ([], []))
            cu.append(uid)
            cf.append(fail)

    rows = []
    print("\n%-26s %7s %7s %7s %8s %9s %9s %9s %9s" %
          ("domain", "units", "runs", "mixed", "mixed%", "ceiling", "measured", "w", "MDE"))
    for name, (u, f) in sorted(tables.items()):
        if not u:
            continue
        n_units = len(set(u))
        sd, mde, mixed, pairs = null_sd(u, f)
        mixed_pct = 100.0 * mixed / n_units if n_units else 0.0
        rec = dict(domain=name, units=n_units, runs=len(u), mixed_units=mixed,
                   mixed_pct=mixed_pct, within_pairs=pairs,
                   null_sd=sd, mde=mde, reported=mixed >= MIN_MIXED)
        if mixed >= MIN_MIXED:
            r = analyse(name[:8], u, f)
            rec.update(ceiling_plugin=r["pred"], ceiling_md=r["pred_md"],
                       measured_loo=r["meas"], w=r["w"], mu=r["mu"], MD=r["MD"])
            print("%-26s %7d %7d %7d %7.1f%% %9.4f %9.4f %9.2e %9.3f" %
                  (name, n_units, len(u), mixed, mixed_pct, r["pred_md"], r["meas"],
                   r["w"], mde))
        else:
            print("%-26s %7d %7d %7d %7.1f%%  [below registered go/no-go of %d mixed units]"
                  % (name, n_units, len(u), mixed, mixed_pct, MIN_MIXED))
        rows.append(rec)

    # ---- per-cell: one acting model, one configuration, model capability held fixed
    percell = []
    print("\n%-26s %-44s %6s %6s %9s %9s" %
          ("domain", "model|config", "units", "mixed", "ceiling", "measured"))
    for (dom, key), (cu, cf) in sorted(cells.items()):
        sd, mde, mixed, pairs = null_sd(cu, cf)
        if mixed < MIN_MIXED:
            print("%-26s %-44s %6d %6d   [below go/no-go]" %
                  (dom, key[:44], len(set(cu)), mixed))
            percell.append(dict(domain=dom, cell=key, units=len(set(cu)),
                                mixed_units=mixed, reported=False))
            continue
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):     # keep analyse's line quiet
            r = analyse(dom[:8], cu, cf)
        print("%-26s %-44s %6d %6d %9.4f %9.4f" %
              (dom, key[:44], len(set(cu)), mixed, r["pred_md"], r["meas"]))
        percell.append(dict(domain=dom, cell=key, units=len(set(cu)), mixed_units=mixed,
                            ceiling_md=r["pred_md"], measured_loo=r["meas"], w=r["w"],
                            mu=r["mu"], reported=True))

    rep = [c["ceiling_md"] for c in percell if c.get("reported")]
    mea = [c["measured_loo"] for c in percell if c.get("reported")]
    if rep:
        rep_s, mea_s = sorted(rep), sorted(mea)
        print("\nper-cell ceilings:  n=%d  median=%.4f  range %.4f-%.4f" %
              (len(rep_s), rep_s[len(rep_s) // 2], rep_s[0], rep_s[-1]))
        print("per-cell measured:  n=%d  median=%.4f  range %.4f-%.4f" %
              (len(mea_s), mea_s[len(mea_s) // 2], mea_s[0], mea_s[-1]))

    json.dump(dict(domains=rows, cells=percell), open(a.out, "w"), indent=2)
    print("\n[saved] %s" % a.out)


if __name__ == "__main__":
    main()
