# Replication package — *When Agent Failure Prediction Measures Task Difficulty Rather Than the Run*

Anonymous artifact for double-blind review. Everything here regenerates the paper's
headline results; `make test` needs no data at all.

## Layout
- `tests/test_metrics.py` — unit tests A–E for the core estimators (no data needed)
- `scripts/`               — numbered analysis scripts (inventory below)
- `results/`               — stored JSON outputs of the supplementary experiments
                             (`e6`, `e7`, `e9`, `e10`, `e11`, `e2`, `e2e`, `deploy`),
                             so every reported number can be checked without re-running
- `preregistered/`         — the registered five-fold instance partition
                             (`e6_folds/fold1..5.txt`, 6,306 instances, disjoint) and an
                             OpenTimestamps Bitcoin attestation of the manifest
- `PREREGISTRATION.md`     — decision rules for every supplementary experiment, each
                             naming the outcome that would have counted against us
- `paper/`                 — the submission PDF and its supplement
- `Makefile`               — `make test` and `make reproduce-main`

## Environment
Python ≥ 3.10; `pip install numpy scipy scikit-learn pandas pyarrow`.
All randomness is seeded (`np.random.default_rng(0)`); GroupKFold splits are deterministic
given the data.

## Quick start
```
make test            # metric unit tests (seconds, no data)
make reproduce-main  # headline numbers, after data download (hours)
```
`make test` is the fastest way to see the paper's central claim: test A builds a scorer
that knows only each task's difficulty and shows it scoring high pooled and exactly
$0.500$ within task.

`make reproduce-main` assumes the corpora are in place (see **Data**). Its first two
steps additionally need the merged probe file `FINAL/merged.npz`, which is not shipped
(it is multi-GB): build it by running the C4 probe extraction (`23_lph_probe.py`, sharded)
and then `29_merge_shards.py`. The remaining steps run from the corpora alone.

## Data
Public corpora (see the paper's Data Availability statement):
`nebius/SWE-agent-trajectories` (C1), `nebius/SWE-rebench-openhands-trajectories` (C2),
`Mosi-AI/LiveClawbench-trajectories` (C3), and the Latent Programming Horizons corpus (C4).
Scripts expect them under `data/`; see `data/README.md`.

## Script inventory (paper item → script)
| Result | Script |
|---|---|
| exact pooled/within decomposition | `62_pair_decomposition.py` |
| difficulty-oracle ceiling + finite-run theory | `90_predict_inflation.py`, `91_finite_r_theory.py` |
| analytic permutation-null SD | `92_analytic_null.py` |
| C3 factorial (model vs task) + task-cluster bootstrap | `93_factorial_and_taskboot.py` |
| deployable static baseline (C1/C2) | `95_static_baseline.py` |
| weighting robustness | `96_macro_within.py` |
| activation probes + recovery controls | `23,24,27,61` |
| Automata reimplementation | `63_automata_reeval.py` |
| fixed-cohort depth / capability | `66,70,71,72,73,74` |
| direct leakage study | `77,82,89` |
| within-task weighting sensitivity | `76_within_sensitivity.py` |
| increment (Brier/log-loss) | `78_prefix_increment.py` |
| label-noise calibration | `80,81` |
| bandit + cold-start + routing | `54,68,87` |
| abort vs allocate; operational curve | `79,94` |
| dependence diagnostics | `85_dependence.py` |
| non-coding generality; external capability | `84,83` |
| probe variants (MLP, mean-pool) | `88_probe_variants.py` |
| finite-repeat corrected ICC | `97_icc_corrected.py` |
| fair deployment comparison (tuned abort, CIs) | `98_deployment_fair.py` |
| E1: within-unit training objective (pre-registered) | `99_within_objective.py` |
| E2: required within-task AUROC sweep (pre-registered) | `100_required_within.py` |
| E3: leaderboard-moment difficulty ceilings | `101_leaderboard_ceiling.py` |
| E1-P: within-objective probes (pre-registered) | `102_within_objective_probe.py` |
| E1-C2: within-objective on C2 (pre-registered) | `103_within_objective_c2.py` |
| M-r: standardized mixed-outcome rates | `104_standardized_mixedness.py` |
| probe shard merge (builds `FINAL/merged.npz`) | `29_merge_shards.py` |

### Running published systems' released code (E6–E11)

These produce the paper's §5.5 results. They do not reimplement anything: they adapt our
corpora into the input format each released pipeline expects, run that pipeline
unmodified, and then score its own output predictions under both estimands.

| Result | Script |
|---|---|
| E6: C2 → EarlyEval input format (adapter) | `105_c2_to_earlyeval.py` |
| E6: gold-answer JSONL from public metadata | `105b_build_verified_jsonl.py` |
| E6: pooled vs within-task on its own predictions | `106_earlyeval_within.py` |
| E6: constant-memory prefix stream-filter (amendment 2) | `107_filter_prefix_shard.py` |
| E6: registered five-fold instance partition (amendment 3) | `108_make_folds.py` |
| E6: five-fold aggregation by summed pair counts | `112_aggregate_folds.py` |
| E6-P: power, MDE and equivalence for the null | `111_within_power.py` |
| E6-C1: C1 → EarlyEval adapter + gold JSONL | `114_c1_to_earlyeval.py`, `114b_build_c1_jsonl.py` |
| E7: C2 → second released system (adapter, driver) | `109_c2_to_sentinel.py`, `110_run_sentinel_e7.py` |
| E9: deployment replay on C2, cost-side reading | `115_build_c2_runs.py`, `116_deploy_c2.py` |
| E2-E: requirement curve on a second corpus | `117_required_within_c2.py` |
| E10: estimand sensitivity (four weightings, drift) | `118_estimand_sensitivity.py` |
| E11: turn control as a deployment baseline | `119_turn_control.py` |
| numbers audit (every numeral → produced artifact) | `113_numbers_audit.py` |
| E12: difficulty-only ceilings outside coding (outcome tables only) | `120_e12_outcome_tables.py` |

**Prefix convention.** `prefix_step_idx = k` means *k steps have been observed*, so index
`0` is a pre-run prediction with no run content. Scripts `106`, `110`, `111`, `112` and
`118` all share this convention; anyone extending them should keep it, since an off-by-one
here silently turns a pre-run score into an "early monitoring" score.

## Expected headline outputs
- test A: pooled > 0.65 with within = 0.500 exactly
- decomposition identity exact to 1e-12
- C4-L within-task mean ~0.51-0.52, 0/16 Holm survivors
- Automata reproduction: pooled 0.805-0.807 (published 0.799)
- matched-depth fractional leak: +0.060 to +0.065
- C3 factorial: task-only 0.825-0.827 and per-model oracle 0.962-0.998 (tight);
  model-only 0.55-0.57 (noisy over 17 values)
- E6 (released pipeline, 5 folds): within-task exactly 0.500 in 70/70 model-fold
  combinations before the agent acts, pooled up to 0.72 at the same checkpoint;
  within-task 0.49-0.52 at one observed step across all 14 variants
Stochastic steps (bootstraps, replay, cross-fitting) reproduce within these ranges;
deterministic identities (decomposition, ceiling formulas) reproduce exactly.

## Reusing this on your own data

The two reusable pieces are independent of our corpora.

1. **The estimand check.** `tests/test_metrics.py` contains standalone implementations of
   pooled AUROC, pair-weighted within-task AUROC and the exact decomposition, in about
   thirty lines with no dependencies beyond NumPy. Point them at your own
   `(unit, label, score)` triples to get $(A_{pool}, A_{within}, w)$ and the gap. If your
   benchmark repeats tasks, this is a few minutes of work and tells you whether your
   monitor reads runs or reads task identity.
2. **The feasibility check, before building a monitor.** `90_predict_inflation.py` and
   `91_finite_r_theory.py` compute the difficulty-only ceiling and the same-task weight
   from an *outcome table alone* — no predictor, no trajectories, just which runs of which
   tasks passed. `92_analytic_null.py` gives the permutation-null SD in closed form, so
   the power of any within-task test is predictable before running it. Together these say,
   in advance, what a pooled number on your benchmark could mean and how large an effect
   you could detect.

`100_required_within.py` (and `117` for a second corpus) answers the deployment question:
given your budget and costs, how good would a monitor have to be before aborting beats
simply allocating compute to easier tasks.
