# Pre-registered decision rules for supplementary experiments

Written and committed BEFORE the experiments below were run. Shared standards: leak-free
absolute-index prefixes; GroupKFold by task; report (pooled AUROC, within-task AUROC, w,
gap); 95% unit bootstrap + within-unit conditional permutation tests, Holm across depths;
control battery (injected outcome feature -> within 1.000, Gaussian noise -> ~0.5, real
inputs with shuffled labels -> ~0.5) rerun on every new pipeline.

## E1 - Within-unit training objective (script 99_within_objective.py)
Question: is the early within-task null an artifact of training predictors with a pooled
objective (which is maximised by learning task difficulty)?
Method: on the same features and folds, train (a) pooled logistic regression,
(b) a same-unit pairwise ranker (Bradley-Terry on within-unit feature differences,
no intercept - the conditional-logit estimator for pairs), (c) pooled logistic on
within-unit demeaned features. Evaluate all on identical held-out runs.
Decision rule, stated in advance:
- If within-objective within-task AUROC stays within resolution of 0.5 at k <= 5, the
  early null is objective-independent and the paper's claim is strengthened.
- If it rises clearly above the pooled-objective value early (>= 0.58 at k = 5), we
  report it as a positive finding and a technique, feed it into the replay, and soften
  the early-null language.
- The within-objective model's POOLED AUROC is expected to drop (it no longer learns
  difficulty); we will report that drop explicitly whichever way the primary outcome goes.

## E2 - Required within-task AUROC (script 100_required_within.py)
Question: how much within-task discrimination would a monitor need before aborting adds
throughput to the paper's allocation policy, per budget?
Method: the fair replay harness of 98_deployment_fair.py unchanged (50/50 tune/eval task
split, paired seeds, transferred prior, per-budget threshold tuning on the tuning split
only), with the real monitor replaced by synthetic scores of controlled within-unit
discrimination d' and zero between-unit component (unit-demeaned), at target within-task
AUROC in {0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00}; achieved within-task AUROC verified
per draw.
Decision rule, stated in advance:
- We report the smallest within-task AUROC at which the paired tasks-solved gain of
  hybrid over allocation excludes zero, per budget, and place the measured real-monitor
  values (~0.50-0.55 early, ~0.65-0.69 late) on the curve. We commit to reporting the
  crossing point wherever it falls, including if it falls low enough to make the measured
  early monitors nearly useful.

## E3 - Difficulty-only ceiling on published benchmarks' outcome tables (conditional)
Runs only if public per-task, per-trial outcome tables exist (tau2-bench telecom
leaderboard submissions, or equivalents). Formulas of main S2.3 applied verbatim.
Decision rule: ceiling >= published AUROC is reported as CONSISTENCY with a
difficulty-only scorer (identifiability, not proof); ceiling < published is reported
plainly as evidence the predictor carries information beyond single-model difficulty.

## Deferred with reasons (decided at gate-check, before any partial run)
- E4 (replicate published probe on its own corpus): the original authors' code/rollouts
  are not yet released; regeneration requires GPU-days. Held for the revision round
  rather than submitting a partial version.
- E5 (token-level uncertainty on C4): cached probe archives contain hidden states but
  not logits; recomputation requires full teacher-forced replay. Deferred.

## Amendment (registered before running, after E1/E2/E3 completed)

### E2-B - Mixed-signal synthetic monitors (script 100, --task-weight)
Same harness and targets as E2, but the synthetic score adds a task-level component
(the unit's transferred-prior failure rate, scaled) to the run-level part, WITHOUT unit
demeaning, so the monitor also carries realistic between-unit signal (achieved pooled and
within verified per table). Decision rule: we report whether the hybrid-minus-allocation
crossing moves relative to family A. The paper's prediction, stated in advance: it does
not move materially, because allocation already holds the task-level information.

### E1-P - Within-objective probes (script 102)
The E1 comparison applied to the C4-L hidden-state probes: pooled-objective logistic vs
same-unit pairwise ranker on the (standardized) 2048-d activations, per layer and prefix,
same GroupKFold-by-task splits, shuffled-label control. Decision rule as in E1: if the
within-trained probe stays near the pooled-trained values (well under 0.58 early), the
probe null is objective-independent; a clear early gain is reported as a positive finding.

### E1-C2 - Within-objective on C2 (script 103)
E1's three objectives on C2 (SWE-rebench, 1,741 mixed units) with C2's behavioural
features at k in {2,5,10,20}. Decision rule as in E1; C2's pooled-objective within-task
values are ~0.50 flat, so the question is whether a within objective finds anything the
flat null hides.

### M-r - Standardized mixed-outcome rates (script 104)
Subsample every corpus outcome table to a common r (3; and 5 where supported) many times;
report the expected observed mixed fraction with CIs. Descriptive; no decision rule
beyond reporting the standardized numbers next to the raw ones.

## E6 - A published system's own released pipeline under the within-task estimand
Registered before running. EarlyEval (arXiv:2609.02783) releases its complete pipeline
(MIT; feature engineering, dual-head LightGBM, evaluation) but not its trajectory data.
We run the RELEASED PIPELINE UNCHANGED on public repeated-run SWE corpora (C1
SWE-agent-trajectories; C2 SWE-rebench if the adapter generalizes), using the paper's own
leave-one-agent-out protocol where the corpus supports it (C1: three acting models),
plus a GroupKFold-by-task robustness variant. We report pooled AUROC/accuracy alongside
pair-weighted within-task AUROC, the same-unit weight w, and within-unit permutation
tests, at the pipeline's own step checkpoints; controls: within-unit label shuffle
(expect ~0.5) and an injected outcome feature (expect ~1.0).
Decision rule, stated in advance:
- If the released pipeline shows high pooled discrimination but within-task near the
  corpus baselines (~0.5 early, rising only late), the paper's central claim is
  demonstrated for the first time on a published system's own implementation, with no
  reimplementation step, and we will report it as such.
- If within-task is substantially high at early steps, we will report that plainly as a
  positive capability of their feature family, note it does not contradict their own
  benchmark-efficiency (Goal-B) objective, and weaken our early-null scope accordingly.
- Adapter failures or schema mismatches will be reported as such, not silently dropped.
Only public artifacts are used; the authors are not contacted.

### E6 amendment (registered after the 2,000-trajectory smoke run, before the full run)
Facts found during the smoke run, recorded per the rule above:
1. C1 (SWE-agent-trajectories) does not carry the action-typed message schema the
   released step builder requires; C1 is excluded as an adapter incompatibility and
   reported as such. E6 runs on C2 (SWE-rebench/OpenHands, 67,074 trajectories over
   6,306 instances).
2. The released reference-free flag (`--disable-answer-features`) crashes as shipped:
   `feature_engineer.py` concatenates the gold-answer feature names into
   NUMERIC/BOOL/CATEGORICAL_FEATURES unconditionally, so `fit()` raises
   `KeyError: 'gold_repo'` whenever the enrichment step that creates those columns is
   skipped. We do not modify the released code. We instead run the pipeline's DEFAULT
   full-featured path (gold-answer features enabled), supplying `--verified-jsonl`
   built mechanically from the public nebius/SWE-rebench instance metadata (100%
   coverage of the 6,306 instances; fields mapped 1:1 to the SWE-bench-verified schema
   the code parses; `difficulty` is absent upstream and becomes the constant
   `__MISSING__`). This strengthens, not weakens, the system under test: gold-answer
   features are constant within an instance, so they can raise pooled discrimination
   but cannot contribute within-task signal.
3. Executed invocation (full run): `SWE_PREFIX_SKIP_INSTANCE_DEDUP=1 python run_all.py
   --data-dir <converted C2> --split-by instance --verified-jsonl <public metadata
   jsonl> --skip-ablation --no-gpu-lgbm --run-name e6_full`. The `--split-by instance`
   task-holdout split is the registered by-task robustness variant; the paper's
   leave-one-agent-out protocol is unavailable on C2 (a single acting model).
The decision rule above is UNCHANGED and was fixed before any predictions existed;
no model had been trained when this amendment was registered.

### E6 amendment 2 (registered after the full-corpus attempt was OOM-killed, before any
### model training on C2 at scale)
The full-corpus run completed Phases 1-2 (all 67,074 trajectories; 4,383,830 prefix
samples streamed to 88 on-disk parts over 17.4 h) and was then SIGKILLed (rc=137) by
the OOM killer inside the released code's part-concatenation step. Measured cause: one
50,003-row part occupies 14.13 GB as a pandas DataFrame, so the full prefix table is
~1.24 TB in RAM - larger than the largest available machine (1.13 TB, shared) - before
any downstream phase's working copies. The full corpus is therefore infeasible through
the released monolithic pipeline on available hardware; this is reported as a scale
finding, not silently reduced.
Registered reduction: a uniform random 20% of the 6,306 instances (1,261 instances,
numpy default_rng seed 42 over the sorted instance list; the drawn list is committed in
the artifact), keeping ALL trajectories of every sampled instance (instance-level
sampling preserves the within-task repeat structure; expected ~13.4k trajectories).
Execution: the existing on-disk parts - produced by the released code, untouched - are
stream-filtered (constant-memory pyarrow pass, no row edited or reordered) to the
sampled instances and written to the path the released code itself reads
(`data/prefix_table.parquet`); the pipeline is then resumed with its own
`--skip-step-table --skip-prefix-table` flags, all other flags as in amendment 1.
This byte-preserving filter-and-resume is the released code's own resume path; the
sampling is in the data fed to it, not in its logic.
The decision rule of the original registration is UNCHANGED. No model has been trained
on C2 at this scale when this amendment is registered; the only trained models to date
are the 2,000-trajectory smoke run's, whose test split contained 6 mixed units and
decided nothing.

### E6 amendment 3 (registered while fold 1 was mid-pipeline, before any fold-1
### prediction, training completion, or evaluation output existed)
The amendment-2 shard is hereby fold 1 of a full grouped 5-fold protocol. The
remaining 5,045 instances are partitioned into folds 2-5 by a seeded shuffle (numpy
default_rng seed 43) over the sorted remainder, split into four near-equal chunks
(1262/1261/1261/1261). All five instance lists are committed in
`preregistered/e6_folds/fold{1..5}.txt`; the five folds are disjoint and cover all
6,306 instances. Each fold runs the identical released-pipeline invocation of
amendment 2 (own --skip flags; per-fold stream-filtered prefix table; per-fold
--run-name), strictly sequentially (memory forbids overlap).
Aggregation, fixed in advance:
- Within-task AUROC: same-unit discordant-pair counts (numerator and denominator)
  are SUMMED across folds; within-task comparisons never cross folds or models.
  The within-unit permutation test aggregates the same way.
- Pooled AUROC: computed per fold (scores from different trained models are never
  compared); reported as per-fold values and mean +/- sd across folds.
- Per-fold results are reported individually as five replications alongside the
  aggregate; a fold whose pipeline fails is reported as failed, not resampled.
The decision rule of the original registration applies to the aggregate and is
UNCHANGED. Fold 1's result will be read only after this amendment is pushed.

## E7 - A second released system, opposite training paradigm
Registered after E6 fold 1's results were known and before any E7 execution; E7's
decision rule is fixed here, in advance, and E6's outcome does not alter it.
System: the released real-time failure-detection monitor of arXiv:2608.02464
(github.com/sunnydubey1111/agent-trajectory-sentinel, Apache-2.0): a one-class
echo-state-network ensemble with CUSUM alarms, trained ONLY on healthy (successful)
episodes - the opposite training paradigm to E6's supervised LightGBM, and by
construction immune to any under-training objection about failure labels.
Protocol:
- Corpus: the E6 fold-1 instance shard of C2, converted to the system's trace format
  through its own real-trace import path (derail/telemetry/adapter.py,
  episode_from_trace); token-logprob and latency channels are absent from C2 and are
  declared missing via the system's native missing-channel mechanism
  (logprobs_available=false, sentinel values), never fabricated.
- Split: the identical instance-level train/test partition E6 fold 1 used (read from
  its emitted predictions table), so both released systems are scored on the same 190
  held-out instances.
- Training: the system's own deployment recipe - fit on healthy episodes from TRAIN
  instances only, thresholds calibrated per its released calibration code; no
  modification to the released code.
- Evaluation: the monitor's per-step alarm/health score as the predictor; pooled and
  pair-weighted within-task AUROC at the same step checkpoints as E6 plus end of run;
  within-unit permutation tests; label-as-score and within-unit shuffle controls.
Decision rule (fixed in advance, mirroring E6):
- High pooled discrimination with early within-task near baseline: the central claim
  is demonstrated on a SECOND published system's own implementation, under a training
  paradigm disjoint from E6's, and will be reported as such.
- Substantially high early within-task: reported plainly as a positive capability of
  one-class telemetry monitoring, and the paper's early-null scope weakened
  accordingly.
- Adapter failures, schema mismatches, or missing-channel degradation are reported as
  such, not silently dropped.
Only public artifacts are used; the authors are not contacted.

## E6-P - Power and equivalence for the within-task null (registered before running)
Registered as a re-analysis of predictions already produced by E6; no new pipeline runs
and no new model fits. The concern addressed is stated plainly: within-task AUROC is
estimated on mixed-outcome units only (tens per fold), while pooled AUROC uses all
pairs, so an early within-task value near 0.5 could in principle reflect low power
rather than absent signal.
For each checkpoint, on the real unit structure of the E6 test splits (folds pooled by
summing same-unit pair counts, folds being disjoint in instances):
- Observed pair-weighted within-task AUROC with a 95% unit-bootstrap CI.
- Empirical minimum detectable effect: synthetic scores of controlled within-unit
  separation (d' = sqrt(2) Phi^-1(target), the E2 construction) injected into the same
  units, scored by the same two-sided test against the analytic permutation null SD;
  MDE is the smallest target detected in >= 80% of 2000 draws.
- An equivalence bound: the smallest delta for which |A_within - 0.5| < delta holds at
  95% bootstrap support.
Reporting rule, fixed in advance: the MDE and the equivalence bound are reported
whatever they are, including if the MDE turns out to lie ABOVE values we would have
wished to exclude - in which case the early null is reported as underpowered at that
checkpoint rather than as evidence of absence. The step-1 result is exempt from this
analysis and reported as exact: when every run of a unit receives an identical score,
within-task AUROC is 0.5 by definition, with no sampling error.
Script: 111_within_power.py.

## E2-C / E2-D - Robustness of the deployment requirement curve
Registered before running. The requirement curve of E2 (the within-task AUROC at which
aborting begins to add throughput over allocation) currently rests on ONE target corpus
(C4-L) and on synthetic monitors whose errors are Gaussian and independent of task
identity. Both are narrow for a number that carries the paper's operational claim, so we
test the curve's stability in two directions:
- E2-C (second target): identical harness, targets C4-Q with the prior transferred from
  C4-L - the reverse of the direction already reported - so the crossing point is
  estimated on a second acting model.
- E2-D (correlated errors): monitors whose noise is correlated with task difficulty
  rather than white, at correlation rho in {0.3, 0.6}, holding the achieved within-task
  AUROC at each target by construction and verifying it per draw. This is a
  deliberately less favourable error structure than family A.
Decision rule, fixed in advance:
- If the crossing point on the second target and under correlated errors stays within
  about 0.05 of the reported 0.84-0.93 band, we report the requirement as stable and
  keep the "order of magnitude short" framing.
- If it moves materially DOWN, we report the lower value, widen the stated band, and
  soften that framing accordingly - including if it moves far enough to place measured
  monitors near usefulness.
- If it moves UP, we report that too and note the requirement is, if anything,
  understated.
Script: 100_required_within.py with new --corr flag; target corpus by existing flags.

## E6-C1 - The released pipeline on a multi-model, second-scaffold corpus
Registered before running. The E6 result rests on C2, which carries a SINGLE acting
model (Qwen3-Coder-480B) and a single scaffold (OpenHands); a within-task null on one
model cannot exclude that a different acting model is more legible early - and our own
capability finding (rho = -0.66) predicts weaker models would be MORE legible, so the
single strong model is the conservative case rather than a favourable one. We therefore
run the released pipeline, again unmodified, on C1 (SWE-agent trajectories: THREE acting
models, Llama-3 8B/70B/405B, and a different scaffold).
C1 was excluded at the original E6 registration as an adapter incompatibility. That
finding is superseded and we say so: C1's assistant turns carry a fenced command block
in 46/46 sampled messages (a recognisable SWE-agent command token in 45/46), with the
observation in the following user turn, which maps onto the released step schema the
same way C2's tool calls did. The adapter is mechanical and content-preserving.
Protocol: instance-holdout split as in E6, on a seeded random instance shard sized to
available memory (the shard fraction and seed are committed before running); all runs of
a sampled instance are kept. Reporting: pooled and pair-weighted within-task AUROC at
steps observed, PER ACTING MODEL as well as pooled over models, with the same controls
and permutation tests.
Decision rule: unchanged from E6, applied per acting model. Additionally, if the three
models differ materially in early within-task AUROC, we report that difference as a
finding rather than averaging it away - it would bear directly on the capability trend.
Scripts: 114_c1_to_earlyeval.py, then the E6 pipeline invocation and 106.

## E9 - Deployment replay on a second corpus, and the cost-side reading
Registered before running. The paper's deployment findings rest on C4 alone while its
measurement findings rest on four corpora; C4 is also the only corpus carrying a
MEASURED per-run cost, which is why. We therefore replay on C2 with an ESTIMATED cost
(tokens ~ characters/4, apportioned across steps by message length - the same
apportionment the C4 harness already uses per step), disclosed as an estimate. Absolute
token figures are therefore approximate; every claim we draw is a RELATIVE comparison
between policies replayed on identical runs, which the estimate supports.
Design: target C2 (SWE-rebench / OpenHands / Qwen3-Coder-480B); prior transferred from
C1 (SWE-agent / Llama-3) over the 1,203 tasks the corpora share, so the transfer crosses
model family, scaffold and corpus - harder than C4's within-corpus transfer. Policies:
round-robin; static transferred ranking in fixed priority order; budgeted Thompson
sampling with the transferred prior; and the same with an uninformative prior, which
isolates the prior's contribution.
Two readings of the same simulation, both reported whatever they show:
  (a) tasks solved at a fixed budget;
  (b) budget needed to reach a fixed number solved, and the implied saving against
      round-robin - the cost-side statement practitioners actually act on.
Decision rule, fixed in advance:
- If ranking/allocation beats round-robin on C2 as it does on C4, we report the
  deployment finding as generalizing across corpus, scaffold and model family.
- If it does NOT, we report that plainly and scope the deployment claim to C4,
  keeping the measurement claims untouched.
- If the uninformative-prior variant matches the transferred-prior one, we report that
  the prior contributes nothing here and say so, even though the transferred prior is
  our own proposed ingredient.
Scripts: 115_build_c2_runs.py, 116_deploy_c2.py.

## E2-E - The requirement curve on a second corpus
Registered before running. The crossing point (the within-task AUROC at which aborting
begins to add throughput over allocation) is the number that converts a measured 0.52
into "an order of magnitude short", and it has rested on C4-L alone. E2-D varied the
monitor's ERROR STRUCTURE and E9 gave a second corpus for the deployment REPLAY; neither
re-estimated the crossing itself. Here it is re-estimated on C2 using the E9 cost table
and the identical protocol (50/50 TUNE/EVAL task split, thresholds tuned per budget on
TUNE only, paired seeds, EVAL touched once), for monitor families A and D.
The allocation baseline is deliberately the STRONGEST available on this corpus:
Thompson sampling with an uninformative prior, because E9 showed the C1-transferred
prior hurts here. Using the weaker warm-start variant would flatter the monitor.
Decision rule, fixed in advance:
- If the C2 crossing lands near C4-L's 0.84-0.93 (family A) / 0.90-1.00 (family D), we
  report the requirement as corpus-stable and keep the framing.
- If it lands materially LOWER, we report the lower value, widen the stated band, and
  weaken "an order of magnitude short" accordingly - including if it falls near the late
  within-task values we measure on weaker acting models (C1: 0.59-0.67), in which case
  we will say plainly that late monitoring on weak models may be near useful.
- If it lands HIGHER, we report that the requirement is understated.
Script: 117_required_within_c2.py.

