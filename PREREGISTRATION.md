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
