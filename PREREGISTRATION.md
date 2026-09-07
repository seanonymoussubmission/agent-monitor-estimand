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
