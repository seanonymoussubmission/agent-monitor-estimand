# Replication package — *When Agent Failure Prediction Measures Task Difficulty Rather Than the Run*

Anonymous artifact for double-blind review. Everything here regenerates the paper's
headline results; `make test` needs no data at all.

## Layout
- `tests/test_metrics.py` — unit tests A–E for the core estimators (no data needed)
- `scripts/`               — numbered analysis scripts (inventory below)
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

## Expected headline outputs
- test A: pooled > 0.65 with within = 0.500 exactly
- decomposition identity exact to 1e-12
- C4-L within-task mean 0.513, 0/16 Holm survivors
- Automata reproduction: pooled 0.806 (published 0.799)
- matched-depth fractional leak: +0.064
- C3 factorial: model-only 0.568, task-only 0.826, per-model oracle 0.962–0.998
