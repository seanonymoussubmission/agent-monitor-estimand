#!/usr/bin/env python3
"""Numbers audit: extract every numeric claim from the paper source and try to trace each
one to a produced artifact (experiment log, result JSON, or CSV) in the run archive.

Numbers that appear verbatim in an artifact are TRACED. Numbers that do not are listed
with their surrounding context for manual adjudication - they are not necessarily wrong
(many are derived, rounded, or arithmetic on other numbers), but each one has to be
accounted for by a human read.

Usage:
  113_numbers_audit.py --tex main.tex supplement.tex --search ~/csc791/logs ~/csc791/e6 ...
"""
import argparse, glob, json, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--tex", nargs="+", required=True)
ap.add_argument("--search", nargs="+", required=True, help="dirs to scan for artifacts")
ap.add_argument("--exts", nargs="+", default=[".log", ".json", ".csv", ".txt", ".out"])
ap.add_argument("--context", type=int, default=90)
ap.add_argument("--out", default=None)
a = ap.parse_args()

# ---- load the artifact corpus once
blobs = []
nfiles = 0
for d in a.search:
    for root, _, files in os.walk(os.path.expanduser(d)):
        for fn in files:
            if not any(fn.endswith(e) for e in a.exts):
                continue
            p = os.path.join(root, fn)
            try:
                if os.path.getsize(p) > 80_000_000:
                    continue
                with open(p, "r", errors="ignore") as f:
                    blobs.append((p, f.read()))
                nfiles += 1
            except Exception:
                pass
corpus = "\n".join(b for _, b in blobs)
print(f"[corpus] {nfiles} artifact files, {len(corpus)/1e6:.1f} MB", flush=True)

# ---- numbers that carry no evidential weight
SKIP_EXACT = {"0", "1", "2", "3", "4", "5", "10", "100", "0.5", "1.0", "0.05", "95",
              "2026", "2027", "12", "16", "50"}
NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:[,{}]?\d{3})*(?:\.\d+)?|\d+\.\d+)(?![\w])")

def variants(tok):
    """Literal forms an artifact might use for this paper token."""
    raw = tok.replace(",", "").replace("{", "").replace("}", "")
    out = {raw, tok}
    if "." in raw:
        f = float(raw)
        out.add(f"{f:.4f}")
        out.add(f"{f:.3f}")
        out.add(f"{f:.2f}")
        # paper often rounds a 4dp artifact value to 3dp
        out.add(raw.rstrip("0"))
    else:
        try:
            out.add(f"{int(raw):,}")
        except ValueError:
            pass
    return {v for v in out if v}

results = {"traced": [], "untraced": []}
seen = set()
for tex in a.tex:
    src = open(tex, errors="ignore").read()
    # strip comments so commented-out numbers are not audited
    src = "\n".join(l.split("%")[0] if not l.strip().startswith("%") else ""
                    for l in src.splitlines())
    for m in NUM.finditer(src):
        tok = m.group(1)
        key = (os.path.basename(tex), tok)
        if tok in SKIP_EXACT or key in seen:
            continue
        seen.add(key)
        ctx = " ".join(src[max(0, m.start() - a.context):
                           m.end() + a.context].split())
        hit = None
        for v in variants(tok):
            if v in corpus:
                for p, b in blobs:
                    if v in b:
                        hit = (v, os.path.basename(p))
                        break
            if hit:
                break
        rec = dict(file=os.path.basename(tex), value=tok, context=ctx)
        if hit:
            rec["found_as"], rec["artifact"] = hit
            results["traced"].append(rec)
        else:
            results["untraced"].append(rec)

t, u = len(results["traced"]), len(results["untraced"])
print(f"[audit] {t} traced, {u} untraced ({t/(t+u)*100:.0f}% traced)", flush=True)
print("\n=== UNTRACED (need manual adjudication) ===", flush=True)
for r in results["untraced"]:
    print(f"  [{r['file']}] {r['value']}\n      ...{r['context']}...", flush=True)

if a.out:
    json.dump(results, open(a.out, "w"), indent=2)
    print(f"\n[saved] {a.out}", flush=True)
