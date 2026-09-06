#!/usr/bin/env python3
"""Is the capability-legibility trend a capability effect or a model-family effect?

The cross-model correlation compares seventeen models from eight developers, so capability
is confounded with architecture, tokenizer, training corpus and how well a model happens to
suit this scaffold. C3 contains same-family pairs at different capability levels
(qwen3.5/3.6, glm-5.1/5.2, minimax-m2.7/m3, kimi-k2.6/k2.7, deepseek flash/pro), which hold
lineage fixed. If the trend is real it should appear within families too; if it is a family
effect it will not.

Three analyses: paired differences within family, a sign/Wilcoxon test over those pairs, and
a family-demeaned correlation that removes each family's mean capability and legibility
before correlating -- the fixed-effects analogue for this design.
"""
import argparse, collections, json, itertools, re
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--scale", default="FINAL/scale.json")
ap.add_argument("--boot", type=int, default=20000)
a = ap.parse_args()
rng = np.random.default_rng(0)

d = json.load(open(a.scale))["models"]
def family(name):
    n = name.lower()
    for f in ("qwen", "glm", "minimax", "kimi", "deepseek", "mimo", "opus", "gpt"):
        if n.startswith(f): return f
    return n.split("-")[0]
for r in d: r["fam"] = family(r["model"])

byfam = collections.defaultdict(list)
for r in d: byfam[r["fam"]].append(r)
print(f"[data] {len(d)} models in {len(byfam)} families")
for f, ms in sorted(byfam.items()):
    print(f"  {f:9s} n={len(ms)}: " + ", ".join(f"{m['model']}(cap {m['cap']:.2f}, leg {m['conc']:.3f})" for m in sorted(ms, key=lambda x: x['cap'])))

# ---- 1. all within-family ordered pairs
print("\n=== 1. WITHIN-FAMILY PAIRS (stronger minus weaker) ===")
diffs = []
for f, ms in sorted(byfam.items()):
    for x, y in itertools.combinations(sorted(ms, key=lambda r: r["cap"]), 2):
        dcap = y["cap"] - x["cap"]; dleg = y["conc"] - x["conc"]
        if dcap <= 0: continue
        diffs.append((f, x["model"], y["model"], dcap, dleg))
        print(f"  {f:9s} {x['model']:18s} -> {y['model']:18s}  "
              f"dcap={dcap:+.3f}  dlegibility={dleg:+.4f}  {'DOWN' if dleg<0 else 'up'}")
dl = np.array([x[4] for x in diffs]); dc = np.array([x[3] for x in diffs])
neg = int((dl < 0).sum())
print(f"\n  {len(dl)} pairs | {neg} show lower legibility for the stronger model "
      f"({100*neg/len(dl):.0f}%)")
print(f"  mean change in legibility = {dl.mean():+.4f}  (sd {dl.std():.4f})")
try:
    w = stats.wilcoxon(dl); print(f"  Wilcoxon signed-rank: statistic={w.statistic:.1f} p={w.pvalue:.4f}")
except Exception as e: print("  Wilcoxon unavailable:", e)
sg = stats.binomtest(neg, len(dl), 0.5)
print(f"  sign test: p = {sg.pvalue:.4f}")
bs = [np.mean(rng.choice(dl, len(dl), replace=True)) for _ in range(a.boot)]
print(f"  bootstrap 95% CI on the mean change: "
      f"[{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]")
r = stats.spearmanr(dc, dl)
print(f"  Spearman(capability gain, legibility change) = {r.statistic:+.3f} p={r.pvalue:.4f}")

# ---- 2. family-demeaned correlation (fixed-effects analogue)
print("\n=== 2. FAMILY-DEMEANED (each family centred, removing family means) ===")
cap = np.array([r["cap"] for r in d]); leg = np.array([r["conc"] for r in d])
fam = np.array([r["fam"] for r in d])
capd = cap.copy(); legd = leg.copy()
for f in set(fam.tolist()):
    m = fam == f
    if m.sum() >= 2:
        capd[m] -= cap[m].mean(); legd[m] -= leg[m].mean()
    else:
        capd[m] = np.nan; legd[m] = np.nan
ok = ~np.isnan(capd)
print(f"  {ok.sum()} models in multi-model families")
rd = stats.spearmanr(capd[ok], legd[ok])
rp = stats.pearsonr(capd[ok], legd[ok])
print(f"  Spearman = {rd.statistic:+.3f} p={rd.pvalue:.4f} | Pearson = {rp.statistic:+.3f} p={rp.pvalue:.4f}")

# ---- 3. between-family: does family mean capability predict family mean legibility?
print("\n=== 3. BETWEEN-FAMILY (family means only) ===")
fm = [(f, np.mean([m["cap"] for m in ms]), np.mean([m["conc"] for m in ms]))
      for f, ms in byfam.items()]
rc = stats.spearmanr([x[1] for x in fm], [x[2] for x in fm])
print(f"  {len(fm)} families: Spearman = {rc.statistic:+.3f} p={rc.pvalue:.4f}")
print("\n  -> The cross-model trend decomposes into a between-family and a within-family")
print("     part. Both matter for interpretation: only the within-family part isolates")
print("     capability from lineage.")
