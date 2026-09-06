#!/usr/bin/env python3
"""
Hidden-state probe on latent-programming-horizons (LPH), Laguna-XS.2 SWE-bench.

This is the reproduction the whole paper hinges on. Unlike the a GPU node attempt,
the ACTING model is public and named in every record (metadata.run_config.model_id
== poolside/Laguna-XS.2), and tokenization.token_ids is the exact token sequence
the model saw -- so teacher-forced replay is exact, not a chat-template guess.

Emits, per (instance_id, run_idx, turn k), the hidden state at the last token of
assistant turn k, for several layers. Analysis (pooled vs within-task AUROC) is
script 24; this one only extracts, so a crash never costs more than the shard.
"""
import argparse, json, os, time, glob
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoConfig

ap = argparse.ArgumentParser()
ap.add_argument("--data",   required=True)
ap.add_argument("--model",  required=True)
ap.add_argument("--out",    required=True)
ap.add_argument("--turns",  type=str, default="1,3,5,10")
ap.add_argument("--layer-fracs", type=str, default="0.25,0.5,0.75,1.0")
ap.add_argument("--max-prefix", type=int, default=41000)
ap.add_argument("--tok-budget", type=int, default=24576)  # padded tokens per batch
ap.add_argument("--limit",  type=int, default=0)
ap.add_argument("--shard",  type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--all-units", action="store_true")
ap.add_argument("--max-runs", type=int, default=0)   # 0 = no cap
ap.add_argument("--multi-run-only", action="store_true")  # identical task set across hosts
ap.add_argument("--resume", action="store_true")   # continue from .partial checkpoint
ap.add_argument("--max-seq", type=int, default=16000,
                help="refuse any prefix longer than this; empirically calibrated")
ap.add_argument("--mem-floor", type=float, default=150.0,
                help="GB of MemAvailable to keep free; batches that would breach it are skipped")
ap.add_argument("--device", default="cuda:0")   # cuda:0 | cpu  (NOT mps: silently wrong)
args = ap.parse_args()

TURNS = [int(x) for x in args.turns.split(",")]
KMAX  = max(TURNS)

# ---------------------------------------------------------------- index pass
# Only tasks whose runs DISAGREE matter: a unit where every run succeeded (or
# every run failed) contributes nothing to a within-task AUROC. Building the
# index first means we never load a model to process data we cannot use.
files = sorted(glob.glob(os.path.join(args.data, "*.json")))
print(f"[index] {len(files)} files", flush=True)

recs, by_task = [], {}
for fp in files:
    try:
        with open(fp) as fh: d = json.load(fh)
        md = d["metadata"]; tk = d.get("tokenization") or {}
        ids, segs = tk.get("token_ids"), tk.get("segments")
        if not ids or not segs: continue
        asst = [s for s in segs if s["role"] == "assistant"]
        if len(asst) < min(TURNS): continue
        # Cut points now, so work can be sorted by prefix length: batching a
        # 700-token prefix beside a 40k one would pad away most of the compute.
        pts = {}
        for k in TURNS:
            if k <= len(asst):
                e = min(asst[k-1]["end_token"], len(ids)) - 1
                if 0 <= e < args.max_prefix: pts[k] = e
        if not pts: continue
        recs.append(dict(fp=fp, task=md["instance_id"], run=md.get("run_idx", 0),
                         y=int(bool(md["outcome"])), n_asst=len(asst), pts=pts,
                         maxpos=max(pts.values()),
                         n_tok=len(ids), repo=md.get("repo", "?")))
        by_task.setdefault(md["instance_id"], []).append(recs[-1])
    except Exception:
        continue

if args.multi_run_only:
    # Hosts may hold different file subsets (one staged the multi-run pool, the
    # other the full split). Sharding by task index over DIFFERENT task lists
    # does not partition -- it duplicates and misaligns. Restricting to tasks
    # with >=2 runs makes the task list identical everywhere.
    by_task = {t: v for t, v in by_task.items() if len(v) >= 2}
    print(f"[index] --multi-run-only: {len(by_task)} tasks with >=2 runs", flush=True)
mixed = {t: v for t, v in by_task.items()
         if len(v) >= 2 and 0 < sum(r["y"] for r in v) < len(v)}
_n_mixed = len(mixed)
if args.all_units:
    # Restricting extraction to mixed units removes between-unit variance by
    # construction, which makes a corpus-wide POOLED auroc unmeasurable -- the
    # flaw that stopped the Qwen run from testing the paper's central claim.
    print(f"[index] --all-units: keeping all {len(by_task)} tasks "
          f"({len(mixed)} of them mixed) for a corpus-wide pooled baseline", flush=True)
    mixed = by_task
print(f"[index] {len(recs)} usable runs | {len(by_task)} tasks | "
      f"{_n_mixed} MIXED-OUTCOME tasks ({100*_n_mixed/max(len(by_task),1):.1f}%)", flush=True)
print(f"[index] runs inside mixed units: {sum(len(v) for v in mixed.values())}", flush=True)

_tasks = sorted(mixed)
_mine = {t for i, t in enumerate(_tasks) if i % args.nshard == args.shard}
work = [r for t in _tasks if t in _mine for r in mixed[t]]
if args.max_runs and len(work) > args.max_runs:
    # Keep every run from a MIXED unit -- those are the only ones that can form
    # a within-task pair -- and subsample the rest, which exist purely to restore
    # the between-unit variance that pooled AUROC needs.
    _mx = {t for t in _mine if len(by_task[t]) >= 2
           and 0 < sum(r["y"] for r in by_task[t]) < len(by_task[t])}
    keep_all = [r for r in work if r["task"] in _mx]
    rest = [r for r in work if r["task"] not in _mx]
    room = max(0, args.max_runs - len(keep_all))
    rs = np.random.default_rng(0)
    if room < len(rest):
        rest = [rest[i] for i in sorted(rs.choice(len(rest), room, replace=False))]
    work = keep_all + rest
    print(f"[index] capped to {len(work)} runs: {len(keep_all)} from mixed units "
          f"(all kept) + {len(rest)} sampled for pooled baseline", flush=True)
print(f"[index] sharding by TASK: {len(_mine)} tasks in this shard "
      f"(run-index sharding splits a unit across shards and destroys its pairs)",
      flush=True)
if args.limit: work = work[:args.limit]
work.sort(key=lambda r: r["maxpos"])          # homogeneous batches
drop = sum(1 for t in mixed for r in mixed[t] if len(r["pts"]) < len(TURNS))
print(f"[index] runs missing >=1 cut point (prefix over {args.max_prefix} or too few turns): {drop}", flush=True)
print(f"[index] shard {args.shard}/{args.nshard}: {len(work)} runs to embed", flush=True)
if not work:
    json.dump({"error": "no mixed-outcome units"}, open(args.out, "w")); raise SystemExit(0)

# ---------------------------------------------------------------- model
t0 = time.time()
# transformers 5.x registers `laguna` NATIVELY, and that native class wins over
# the checkpoint's own auto_map. The native layout uses fused experts
# (experts.gate_up_proj); this checkpoint stores them per-expert and unfused, so
# AutoModel silently randomly-initialises 39 layers of MoE. Force the repo code.
# Load via the NATIVE transformers implementation, not the repo's bundled
# modeling_laguna.py. The custom-code path expects a different expert layout
# than this checkpoint ships, so it silently randomly-initialises all 39 MoE
# layers while still reporting success -- that is what made the first pilot's
# numbers meaningless. Verified on the compute node (transformers 5.14.1): the native path
# gives missing_keys=0 and 0.739 teacher-forced next-token accuracy.
model, info = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map=args.device,
    attn_implementation="sdpa", output_loading_info=True)
model.eval()

# GATE 1: nothing may be randomly initialised. A missing expert weight is exactly
# the failure that made the first pilot's numbers meaningless.
miss = [k for k in info.get("missing_keys", []) if "rotary" not in k and "inv_freq" not in k]
if miss:
    raise SystemExit(f"[FATAL] {len(miss)} weights missing from checkpoint, model "
                     f"is partly random. First: {miss[:5]}")
print(f"[model] class={type(model).__name__} | no missing weights | "
      f"unexpected={len(info.get('unexpected_keys', []))}", flush=True)
NL = model.config.num_hidden_layers
LAYERS = sorted({max(1, int(round(f * NL))) for f in
                 (float(x) for x in args.layer_fracs.split(","))})
print(f"[model] loaded in {time.time()-t0:.0f}s | {NL} layers | probing {LAYERS} "
      f"| hidden={model.config.hidden_size}", flush=True)

# Capture ONLY the probed layers. output_hidden_states=True materialises all 41
# hidden-state tensors; at a 40k prefix that is ~6.7GB per sequence, and it is
# what OOM-killed three shards simultaneously when they reached the long tail.
CAP = {}
def _mk_hook(L):
    def _h(mod, inp, out):
        CAP[L] = (out[0] if isinstance(out, tuple) else out).detach()
    return _h
_dec = model.model.layers
_handles = []
for L in LAYERS:
    # hidden_states[i] is the INPUT to layer i, i.e. the output of layer i-1;
    # the final entry is post-final-norm. Hook accordingly so the captured
    # tensors are identical to the old indexing.
    _handles.append((model.model.norm if L >= NL else _dec[L-1]).register_forward_hook(_mk_hook(L)))

# Equivalence check: hooks must reproduce output_hidden_states exactly, or every
# number downstream is silently different from the pilot's.
with torch.no_grad():
    _t = torch.arange(16, dtype=torch.long)[None].to(args.device)
    CAP.clear()
    _ref = model(input_ids=_t, output_hidden_states=True, use_cache=False).hidden_states
    for L in LAYERS:
        if not torch.allclose(CAP[L].float(), _ref[L].float(), atol=1e-3, rtol=1e-3):
            raise SystemExit(f"[FATAL] hook for layer {L} != hidden_states[{L}]")
    print(f"[hooks] verified identical to output_hidden_states for {LAYERS}", flush=True)
CAP.clear()

# GATE 2: end-to-end faithfulness. These trajectories ARE this model's own
# sampled generations, so under correct weights + correct tokenisation it must
# assign them low perplexity. A correctly-keyed but wrongly-replayed setup would
# still pass gate 1; this catches that too.
def replay_ppl(r):
    with open(r["fp"]) as fh: d = json.load(fh)
    ids  = d["tokenization"]["token_ids"]
    asst = [x for x in d["tokenization"]["segments"] if x["role"] == "assistant"]
    end  = min(asst[min(2, len(asst)-1)]["end_token"], len(ids), 4096)
    t = torch.tensor(ids[:end])[None].to(args.device)
    with torch.no_grad():
        lg = model(input_ids=t, use_cache=False).logits.float()
    tgt, mask = t[0, 1:], torch.zeros(end - 1, dtype=torch.bool)
    for a in asst:                       # score ONLY model-generated tokens
        lo, hi = max(a["start_token"] - 1, 0), min(a["end_token"] - 1, end - 1)
        if lo < hi: mask[lo:hi] = True
    if mask.sum() < 20: return float("nan")
    nll = torch.nn.functional.cross_entropy(lg[0, :-1][mask.to(args.device)], tgt[mask.to(args.device)])
    return float(torch.exp(nll))

ppls = [v for v in (replay_ppl(r) for r in work[:6]) if v == v]
med = float(np.median(ppls)) if ppls else float("nan")
print(f"[GATE 2] replay perplexity on model's own generations: "
      f"median={med:.2f}  all={[round(x,2) for x in ppls]}", flush=True)
if not (med < 25):
    raise SystemExit(f"[FATAL] replay perplexity {med:.1f} is far too high for a "
                     f"model scoring its own outputs -- weights or tokenisation wrong.")
if args.device.startswith("cuda"): torch.cuda.empty_cache()

# ---------------------------------------------------------------- extract
def load_prefix(r):
    """Token prefix up to the deepest requested turn (cut points fixed at index time)."""
    with open(r["fp"]) as fh: d = json.load(fh)
    return d["tokenization"]["token_ids"][:r["maxpos"] + 1]

def _save(path, rows, layers, turns, n_units):
    """Write rows to .npz. Used for both checkpoints and the final output, so a
    partial file has exactly the same schema the merge/analysis expect."""
    if not rows: return
    m = dict(task=np.array([r["task"] for r in rows]),
             run=np.array([r["run"] for r in rows], np.int32),
             y=np.array([r["y"] for r in rows], np.int8),
             k=np.array([r["k"] for r in rows], np.int16),
             pos=np.array([r["pos"] for r in rows], np.int32),
             n_asst=np.array([r["n_asst"] for r in rows], np.int32),
             repo=np.array([r["repo"] for r in rows]),
             layers=np.array(layers), turns=np.array(turns),
             n_units=np.array([n_units]))
    for L in layers:
        m[f"H{L}"] = np.stack([np.asarray(r["h"][str(L)], np.float32) for r in rows])
    tmp = path + ".tmp.npz"          # atomic: never leave a half-written file
    np.savez_compressed(tmp, **m)
    os.replace(tmp, path if path.endswith(".npz") else path + ".npz")

out, done, t0, _last_ckpt = [], 0, time.time(), 0

if args.resume:
    # Checkpoints were useless without this: a killed shard restarted from zero.
    _pf = args.out + ".partial.npz"
    if os.path.exists(_pf):
        _Z = np.load(_pf, allow_pickle=False)
        _L = [int(x) for x in _Z["layers"]]
        for _i in range(len(_Z["y"])):
            out.append(dict(task=str(_Z["task"][_i]), run=int(_Z["run"][_i]),
                            y=int(_Z["y"][_i]), k=int(_Z["k"][_i]),
                            pos=int(_Z["pos"][_i]), repo=str(_Z["repo"][_i]),
                            n_asst=int(_Z["n_asst"][_i]), n_tok=0,
                            h={str(L): _Z[f"H{L}"][_i] for L in _L}))
        _seen = {(r["task"], r["run"]) for r in out}
        _before = len(work)
        work = [r for r in work if (r["task"], r["run"]) not in _seen]
        done = _last_ckpt = len(_seen)
        print(f"[resume] recovered {len(out)} rows / {len(_seen)} runs from checkpoint; "
              f"{_before} -> {len(work)} runs still to do", flush=True)
    else:
        print("[resume] no checkpoint found, starting fresh", flush=True)
batch = []

def mem_avail_gb():
    """MemAvailable in GB. Returns a huge number off-Linux so the guard is inert."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1048576.0
    except Exception:
        pass
    return 1e9

SKIPPED = []

def flush(batch):
    global done
    if not batch: return
    # MEMORY GUARD. Attention is O(seq^2) on CPU, so a single 40k-token prefix
    # can demand hundreds of GB. Four such processes took two machines down.
    # Estimate the need, refuse the batch if it would breach the floor, and
    # record the skip so the complete-units filter drops those units rather
    # than admitting a length-biased subset of their runs.
    L_ = max(len(b["ids"]) for b in batch)
    # A HARD LENGTH CAP, not an estimate. The analytic estimate was wrong by an
    # order of magnitude: it predicted ~86GB for a 41k-token forward pass and the
    # real consumption exhausted a 1.1TB machine. Empirically, prefixes up to
    # ~17-28k tokens completed and everything above that killed the process, so we
    # refuse by length. Skipped runs are recorded and their units are dropped by
    # the complete-units filter rather than contributing a length-biased subset.
    n_heads = getattr(model.config, "num_attention_heads", 16)
    est_gb = (n_heads * (L_ ** 2) * 2 * len(batch)) / 1073741824.0 * 1.6
    avail = mem_avail_gb()
    if L_ > args.max_seq or avail - est_gb < args.mem_floor:
        for b in batch:
            SKIPPED.append((b["r"]["task"], b["r"]["run"], len(b["ids"])))
        print(f"   [guard] SKIP batch of {len(batch)} at seq={L_}: "
              f"est {est_gb:.0f}GB vs {avail:.0f}GB avail, floor {args.mem_floor:.0f}GB",
              flush=True)
        return
    L = max(len(b["ids"]) for b in batch)
    pad = model.config.eos_token_id
    if isinstance(pad, (list, tuple)): pad = pad[0]      # this model lists several EOS ids
    if pad is None: pad = 0
    # Right padding. With a causal mask, positions after the true end cannot
    # influence earlier hidden states, so indexing at the true index is exact.
    # (The a GPU node run used left padding with a right-padding index formula --
    # a bug the positive control caught only after the fact.)
    inp = torch.full((len(batch), L), pad, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["ids"]); inp[i, :n] = torch.tensor(b["ids"]); att[i, :n] = 1
    CAP.clear()
    with torch.no_grad():
        model(input_ids=inp.to(args.device), attention_mask=att.to(args.device),
              use_cache=False)
    hs = CAP
    for i, b in enumerate(batch):
        for k, pos in b["pts"].items():
            vec = {str(l): hs[l][i, pos].float().cpu().numpy() for l in LAYERS}
            out.append(dict(task=b["r"]["task"], run=b["r"]["run"], y=b["r"]["y"],
                            k=k, pos=int(pos), repo=b["r"]["repo"], n_asst=b["r"]["n_asst"],
                            n_tok=b["r"]["n_tok"], h=vec))   # keep float32 arrays, not lists
        done += 1
    CAP.clear()
    if args.device.startswith("cuda"): torch.cuda.empty_cache()

for r in work:
    ids = load_prefix(r)
    batch.append(dict(ids=ids, pts=r["pts"], r=r))
    # cost is padded, not summed: a batch costs len(batch) * longest sequence
    if len(batch) * max(len(b["ids"]) for b in batch) >= args.tok_budget or len(batch) >= 8:
        flush(batch); batch = []
        if done - _last_ckpt >= 40:      # never lose more than ~40 runs to a crash
            _save(args.out + ".partial", out, LAYERS, TURNS, len(mixed))
            _last_ckpt = done
            print(f"   [ckpt] {done} runs saved", flush=True)
        if done % 50 < 8:
            el = time.time() - t0
            print(f"   {done}/{len(work)}  {el/60:.1f} min  "
                  f"~{(len(work)-done)*el/max(done,1)/60:.1f} min left", flush=True)
flush(batch)

# .npz, not JSON: at ~105KB/row the full set would be ~850MB of JSON text,
# which is slow to parse and needs several GB to load. Arrays keep it compact
# and let the analysis memory-map what it needs.
_save(args.out, out, LAYERS, TURNS, len(mixed))
if SKIPPED:
    print(f"[guard] skipped {len(SKIPPED)} runs that would have breached the memory floor; "
          f"longest {max(x[2] for x in SKIPPED)} tokens", flush=True)
    import csv as _csv
    with open(args.out + ".skipped.csv", "w", newline="") as fh:
        _w = _csv.writer(fh); _w.writerow(["task","run","prefix_tokens"]); _w.writerows(SKIPPED)
print(f"[done] {done} runs -> {len(out)} rows in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)
