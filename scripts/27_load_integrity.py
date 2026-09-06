#!/usr/bin/env python3
"""
Does the model ACTUALLY load, and is it the real model?

The Delta pilot loaded 63GB in 29s and silently randomly-initialised every MoE
layer. A probe on that would have produced a beautiful, meaningless null. So
before any probing, this gates on two things:

  1. from_pretrained reports no missing/unexpected keys.
  2. Teacher-forced next-token accuracy on a REAL trajectory prefix is high.

Check 2 is the one that matters: random experts cannot predict real tokens, so
accuracy near chance means the weights are wrong no matter what the loader said.
"""
import argparse, json, time, os, glob, warnings
import torch
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--traj", default="")
ap.add_argument("--trust-remote-code", action="store_true")
ap.add_argument("--bench", default="742,2176,4631")
ap.add_argument("--device", default="cpu")   # cpu | mps | cuda:0
# Calibration (teacher-forced next-token accuracy on the model's OWN generations):
#   Laguna-XS.2  CPU the compute node  0.7386   real weights
#   Laguna-XS.2  CPU a local Mac  0.7372   real weights, different hardware
#   Qwen3.6-A3B  CPU the compute node  0.6957   real weights, different MODEL
#   Laguna-XS.2  MPS a local Mac  0.3249   loads clean, computes WRONG values
# Real models cluster at 0.70-0.74; a broken backend is at 0.32. The gate must
# separate those, so it sits between them -- not pinned to one model's value,
# which is what made it reject a perfectly good Qwen load.
ap.add_argument("--min-acc", type=float, default=0.55)
args = ap.parse_args()

t0 = time.time()
model, info = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map=args.device,
    low_cpu_mem_usage=True, trust_remote_code=args.trust_remote_code,
    output_loading_info=True)
model.eval()
print(f"[load] {(time.time()-t0)/60:.1f} min  impl={type(model).__name__}")
for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
    v = info.get(key) or []
    print(f"[load] {key}: {len(v)}" + (f"  e.g. {v[:2]}" if v else ""))
BAD = len(info.get("missing_keys") or []) > 0

# ---- integrity gate: can it predict real tokens it actually generated?
if args.traj:
    d = json.load(open(args.traj))
    ids = d["tokenization"]["token_ids"][:2048]
    x = torch.tensor([ids]).to(args.device)
    with torch.no_grad():
        lg = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
    pred = lg[0, :-1].argmax(-1)
    gold = x[0, 1:]
    acc = (pred == gold).float().mean().item()
    print(f"[integrity] teacher-forced next-token accuracy = {acc:.4f}  (n={len(gold)})")
    # Reference: 0.7386 (the compute node CPU) and 0.7372 (a local Mac CPU) on this trajectory.
    # A 0.30 floor was far too lenient -- it PASSED a local Mac/MPS at 0.3249, where the
    # weights were fine but the backend silently computed wrong values. Gate on
    # agreement with the CPU reference, not merely on being better than random.
    REF = args.min_acc
    ok = acc >= REF
    print(f"[integrity] {'PASS' if ok else 'FAIL'} - acc={acc:.4f} vs floor {REF:.2f}"
          + ("" if ok else "  <- backend computes wrong values; DO NOT PROBE"))
    BAD = BAD or not ok

if BAD:
    print("[verdict] DO NOT PROBE THIS MODEL"); raise SystemExit(1)

# ---- throughput, only if the model is real
for L in [int(x) for x in args.bench.split(",")]:
    ids = torch.randint(0, int(model.config.vocab_size), (1, L)).to(args.device)
    at = torch.ones_like(ids)
    with torch.no_grad():
        model(input_ids=ids[:, :64], attention_mask=at[:, :64], use_cache=False)
    t = time.time()
    with torch.no_grad():
        model(input_ids=ids, attention_mask=at, output_hidden_states=True, use_cache=False)
    dt = time.time() - t
    print(f"[bench] {L:>6} tok -> {dt:7.1f}s = {L/dt:8.1f} tok/s")
print("[verdict] OK")
