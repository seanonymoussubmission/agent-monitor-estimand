#!/usr/bin/env python3
"""Why do stronger models fail invisibly? A behavioural-dispersion account.

We find that within-task legibility of failure declines with model capability. This asks
what produces that. The hypothesis is Gould's account of the disappearance of the .400
hitter: as mean skill rises, variance around it shrinks, and extreme observations vanish
not because play worsened but because it became uniform. Transposed: as an agent gets more
capable its behaviour across repeated attempts at one task becomes more consistent, so a
failing run resembles a succeeding one more closely, and there is less for any monitor to
read.

The measurement has to avoid a circularity. "Are failed runs similar to successful runs"
is what within-task AUROC already measures, so correlating separability against AUROC
would correlate a quantity with itself. The two mechanism measures here therefore use NO
outcome labels at all:

  DISPERSION   mean pairwise Jensen-Shannon divergence between the activity distributions
               of runs sharing a unit. This is the quantity Anderson's PERMDISP tests --
               homogeneity of multivariate dispersion -- computed per model. JSD is
               bounded in [0,1] and defined on distributions, so it is inherently
               length-normalised, which matters because trajectory length varies by model.

  REPEATABILITY  R = between-unit variance / total variance over trajectory features, the
               intraclass correlation used as the standard index of behavioural
               consistency in animal behaviour (Nakagawa and Schielzeth 2010). High R means
               a model does the same thing whenever it sees the same task.

A third, PERMANOVA pseudo-F on the fail/success split within units (Anderson 2001), does
use labels; we report it as confirmatory rather than as the mechanism.

The decisive analysis is mediation. If capability acts on legibility THROUGH behavioural
uniformity, then partialling dispersion out of the capability-legibility correlation
should attenuate it substantially.
"""
import argparse, collections, json, math, re
import numpy as np, pyarrow.parquet as pq
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default="data/liveclaw/data/v0.2.1-00000-of-00001.parquet")
ap.add_argument("--perm", type=int, default=20000)
ap.add_argument("--reps", type=int, default=400)
ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)
ERRRE = re.compile(r"error|failed|not found|cannot|unable|exception|denied", re.I)

t = pq.read_table(a.parquet, columns=["model_name","case_id","score","trajectory"]).to_pylist()
rows = []
for d in t:
    if d["score"] is None: continue
    try: steps = json.loads(d["trajectory"]).get("steps", [])
    except Exception: continue
    acts, ol, er = [], [], []
    for st in steps:
        if st.get("source") != "agent": continue
        tc = st.get("tool_calls") or []
        nm = str(tc[0].get("function_name") or "no_tool") if (isinstance(tc, list) and tc
             and isinstance(tc[0], dict)) else "no_tool"
        acts.append(nm); ol.append(len(str(st.get("message") or "")))
        er.append(int(bool(ERRRE.search(str(st.get("observation") or "")))))
    if len(acts) >= 2: rows.append((d["model_name"], d["case_id"], float(d["score"]), acts, ol, er))

model = np.array([r[0] for r in rows]); case = np.array([r[1] for r in rows])
score = np.array([r[2] for r in rows])
SEQ = [r[3] for r in rows]; OL = [r[4] for r in rows]; ER = [r[5] for r in rows]
unit = np.array([f"{m}||{c}" for m, c in zip(model, case)])
MODELS = sorted(set(model.tolist()))
VOCAB = sorted({x for s in SEQ for x in s}); VI = {v: i for i, v in enumerate(VOCAB)}
print(f"[data] {len(rows)} traces | {len(MODELS)} models | {len(set(case))} tasks | "
      f"{len(VOCAB)} distinct activities")

def actdist(i):
    v = np.zeros(len(VOCAB))
    for x in SEQ[i]: v[VI[x]] += 1
    return v / max(v.sum(), 1)

DIST = np.vstack([actdist(i) for i in range(len(rows))])

def jsd(p, q):
    m = 0.5 * (p + q)
    def kl(x, y):
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / np.maximum(y[nz], 1e-12))))
    return max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))

# ---- feature matrix for repeatability (outcome-blind trajectory descriptors)
def desc(i):
    n = len(SEQ[i]); ol = np.array(OL[i], float); er = np.array(ER[i], float)
    return np.array([n, len(set(SEQ[i])), 1.0 - len(set(SEQ[i]))/max(n,1),
                     ol.mean(), ol.std(), er.mean()])
DESC = np.vstack([desc(i) for i in range(len(rows))])
DESCZ = (DESC - DESC.mean(0)) / (DESC.std(0) + 1e-9)

units_of = collections.defaultdict(list)
for i, u in enumerate(unit): units_of[u].append(i)

def dispersion(mdl, cases=None):
    """Mean pairwise JSD within units -- outcome-blind."""
    vals = []
    for u, ix in units_of.items():
        if model[ix[0]] != mdl: continue
        if cases is not None and case[ix[0]] not in cases: continue
        for x in range(len(ix)):
            for y in range(x+1, len(ix)):
                vals.append(jsd(DIST[ix[x]], DIST[ix[y]]))
    return float(np.mean(vals)) if vals else np.nan

def repeatability(mdl):
    """R = between-unit variance / total, averaged over descriptors (ICC form)."""
    sel = model == mdl
    Rs = []
    for c in range(DESCZ.shape[1]):
        vals = DESCZ[sel, c]; us = unit[sel]
        gm = vals.mean(); means = {}
        for u in set(us): means[u] = vals[us == u].mean()
        btw = np.mean([(means[u] - gm)**2 for u in set(us)])
        wth = np.mean([(v - means[u])**2 for v, u in zip(vals, us)])
        if btw + wth > 0: Rs.append(btw / (btw + wth))
    return float(np.mean(Rs)) if Rs else np.nan

def permanova(mdl):
    """Pseudo-F for the fail/success split within units (uses labels; confirmatory)."""
    fail = (score != 1.0).astype(int)
    num = den = 0.0
    for u, ix in units_of.items():
        if model[ix[0]] != mdl: continue
        g = [i for i in ix if fail[i] == 1]; b = [i for i in ix if fail[i] == 0]
        if not g or not b: continue
        between = np.mean([jsd(DIST[i], DIST[j]) for i in g for j in b])
        within = []
        for grp in (g, b):
            for x in range(len(grp)):
                for y in range(x+1, len(grp)): within.append(jsd(DIST[grp[x]], DIST[grp[y]]))
        if within: num += between; den += float(np.mean(within))
    return float(num / den) if den > 0 else np.nan

# ---- legibility: reuse the cross-fitted predictor from the scale experiment
def fsm_of(idx):
    tr = collections.defaultdict(collections.Counter)
    for i in idx:
        s = "<init>"
        for x in SEQ[i]: tr[s][x] += 1; s = x
    out = {}
    for s, c in tr.items():
        keep = {x: n for x, n in c.items() if n >= 2 or len(c) == 1} or dict(c)
        tt = sum(keep.values()); out[s] = {x: n/tt for x, n in keep.items()}
    al = set(); [al.update(c) for c in tr.values()]
    return out, sorted(al)

def feat(i, fsm, alpha, aidx):
    seq, ol, er = SEQ[i], OL[i], ER[i]; n = len(seq)
    v = np.zeros(len(alpha))
    for x in seq:
        j = aidx.get(x)
        if j is not None: v[j] += 1
    v /= max(n, 1)
    FL = 1e-4; sur, s = [], "<init>"
    for x in seq:
        sur.append(-math.log(max(fsm.get(s, {}).get(x, FL), FL))); s = x
    sur = np.array(sur) if sur else np.array([0.0]); h = max(1, len(sur)//2)
    ol = np.array(ol, float) if ol else np.array([0.0])
    er = np.array(er, float) if er else np.array([0.0])
    return np.concatenate([v, [sur.mean(), sur.max(), float((sur > 3.0).mean()),
        abs(sur[:h].mean()-sur[h:].mean()) if len(sur) > 1 else 0.0,
        n, len(set(seq)), 1.0-len(set(seq))/max(n,1),
        ol.mean(), ol.max(), ol.std(), er.mean(), er.sum()]])

oof = np.zeros(len(rows))
for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(rows)), score, case):
    fsm, alpha = fsm_of(tr); aidx = {x: j for j, x in enumerate(alpha)}
    m = HistGradientBoostingRegressor(max_iter=200, max_depth=3, random_state=0)
    m.fit(np.vstack([feat(i, fsm, alpha, aidx) for i in tr]), score[tr])
    oof[te] = m.predict(np.vstack([feat(i, fsm, alpha, aidx) for i in te]))

def legibility(mdl):
    num = den = 0.0
    for u, ix in units_of.items():
        if model[ix[0]] != mdl: continue
        for i in ix:
            for j in ix:
                if score[i] > score[j]:
                    num += (oof[i] > oof[j]) + 0.5*(oof[i] == oof[j]); den += 1
    return float(num/den) if den else np.nan

REC = []
for m in MODELS:
    REC.append(dict(model=m, cap=float(score[model == m].mean()),
                    disp=dispersion(m), rep=repeatability(m),
                    pF=permanova(m), leg=legibility(m),
                    length=float(np.mean([len(SEQ[i]) for i in range(len(rows)) if model[i]==m]))))
REC.sort(key=lambda r: r["cap"])
print(f"\n{'model':20s} {'cap':>6} {'disp':>7} {'repeat':>7} {'pseudoF':>8} {'legible':>8} {'len':>6}")
for r in REC:
    print(f"  {r['model']:18s} {r['cap']:.3f} {r['disp']:7.4f} {r['rep']:7.4f} "
          f"{r['pF']:8.3f} {r['leg']:8.4f} {r['length']:6.1f}")

cap = np.array([r["cap"] for r in REC]); disp = np.array([r["disp"] for r in REC])
rep = np.array([r["rep"] for r in REC]); leg = np.array([r["leg"] for r in REC])
ln  = np.array([r["length"] for r in REC])

def sp(x, y): 
    r = stats.spearmanr(x, y); return r.statistic, r.pvalue
print("\n=== H1: does behavioural dispersion fall as capability rises? (outcome-blind) ===")
r1, p1 = sp(cap, disp); print(f"  Spearman(capability, dispersion) = {r1:+.3f}  p = {p1:.4f}")
null = [stats.spearmanr(rng.permutation(cap), disp).statistic for _ in range(a.perm)]
pe = (1 + (np.abs(null) >= abs(r1)).sum()) / (len(null) + 1)
print(f"  exact permutation p = {pe:.5f}")

print("\n=== H2: does behavioural repeatability rise with capability? (outcome-blind) ===")
r2, p2 = sp(cap, rep); print(f"  Spearman(capability, repeatability) = {r2:+.3f}  p = {p2:.4f}")

print("\n=== confirmatory: PERMANOVA pseudo-F on the fail/success split ===")
r3, p3 = sp(cap, np.array([r["pF"] for r in REC]))
print(f"  Spearman(capability, pseudo-F) = {r3:+.3f}  p = {p3:.4f}")

print("\n=== control: is this just trajectory length? ===")
rl, pl = sp(cap, ln); print(f"  Spearman(capability, mean length) = {rl:+.3f}  p = {pl:.4f}")
def partial(x, y, z):
    rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
    rxy = np.corrcoef(rx, ry)[0,1]; rxz = np.corrcoef(rx, rz)[0,1]; ryz = np.corrcoef(ry, rz)[0,1]
    d = math.sqrt(max(1e-12, (1-rxz**2)*(1-ryz**2)))
    return (rxy - rxz*ryz) / d
print(f"  partial Spearman(capability, dispersion | length) = {partial(cap, disp, ln):+.3f}")

print("\n=== MEDIATION: does dispersion explain the capability effect on legibility? ===")
r0, p0 = sp(cap, leg)
pc = partial(cap, leg, disp)
print(f"  capability -> legibility, raw       = {r0:+.3f} (p={p0:.4f})")
print(f"  capability -> legibility | dispersion = {pc:+.3f}")
att = 100*(1 - abs(pc)/max(abs(r0), 1e-9))
print(f"  attenuation = {att:+.1f}%")
bs = []
for _ in range(a.reps):
    k = rng.integers(0, len(cap), len(cap))
    if len(set(cap[k].tolist())) < 5: continue
    try: bs.append(partial(cap[k], leg[k], disp[k]))
    except Exception: pass
if len(bs) > 20:
    print(f"  bootstrap 95% CI on the partial: [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
print("  -> " + ("dispersion MEDIATES the capability effect" if att > 40 else
                 "dispersion does not account for the capability effect"))
if a.out:
    json.dump(dict(models=REC, h1=[r1,p1,float(pe)], h2=[r2,p2], permanova=[r3,p3],
                   length=[rl,pl], mediation=[float(r0), float(pc), float(att)]),
              open(a.out,"w"), indent=2)
    print(f"\n[saved] {a.out}")
