# Running the main experiments

Run these commands from the repository root, after setup and DATA.md.
The commands below launch real training/evaluation; smoke tests do not.
The default is CPU. Preserve the original data schema, source identifiers,
row order, time split and feature order.

First verify data and dependencies:

```bash
python scripts/preflight.py --require-data
```

## 1. Contextual Feature Utility

```bash
python experiments/temporal_main_experiment_2026_09_14/run_classical_baselines.py
python experiments/temporal_main_experiment_2026_09_14/run_sensitivity_analyses.py
```

The first command selects on 2024, refits, and evaluates naive/linear/tree
models, including the feature-set comparisons. The second exports the
XGBoost predictions required by the final RA comparison (and produces
additional diagnostics). Results are under the main experiment's
`results/classical/` and `results/sensitivity/`.

## 2. Shared dependencies: original direct-price and energy Transformers

```bash
python experiments/temporal_main_experiment_2026_09_14/run_temporal_transformer.py
python experiments/temporal_main_experiment_2026_09_14/run_extended_robustness.py
```

This stage creates the energy-only controls reused by the final residual
transfer experiment, and the earlier price/shared embeddings required by
the historical retrieval benchmark. It also trains original direct-price
controls needed by retained analyses. It is **not** the final MT-Residual
experiment.

The robustness script writes the feature-comparison CIs to
`results/robustness/spatial_block_confidence_intervals.csv`. It also
contains older non-residual transfer analyses; do not substitute those
rows for the final residual Multi-Task Transfer results.

## 3. Multi-Task Transfer

```bash
python experiments/task2_residual_main_2026_09_26/run_task2_residual.py
```

Requires stage 2's energy-only controls and original Transformer outputs.
Primary summary and uncertainty:

- `experiments/task2_residual_main_2026_09_26/results/TASK2_RESIDUAL_RESULTS.csv`
- `experiments/task2_residual_main_2026_09_26/results/PAIRED_SPATIAL_BLOCK_UNCERTAINTY.csv`

The main summary is the mean and sample SD of per-seed metrics, not
prediction-ensemble performance.

## 4. Retrieval-Assisted Valuation

Historical benchmark (requires stage 2):

```bash
python experiments/temporal_main_experiment_2026_09_14/run_temporal_retrieval.py
```

Final price-only extension (requires stage 1's XGBoost prediction export,
but does not depend on the fitted shared residual model):

```bash
python experiments/temporal_retrieval_aware_residual_fusion_2026_09_24/run_experiment.py --phase all
```

The final runner builds retrieval-training tuples, selects the retrieval
weight on 2024, tunes neighbour fusion on 2024, refits the five models
and evaluates on 2025. Its `--phase build`, `screen`, and `full` switches
allow sequential continuation, in that order, on unchanged inputs.

Final outputs are in that folder's `results/`:
`test_2025_comparison.csv`, `paired_spatial_block_uncertainty.json`,
`FUSION_SELECTION.json`, `GAMMA_SELECTION.json` and seed summaries.
Generated per-record outputs contain sensitive property-level information;
they are ignored by Git and are not release artifacts.

## Helper modules are not extra required experiment stages

The folders `temporal_residual_screen_2026_09_23`,
`temporal_residual_full_2026_09_23`,
`temporal_retrieval_aware_2026_09_24` and
`temporal_residual_retrieval_fusion_2026_09_24` are retained because the
main runners import their market-index, fitting, metric-loss, neighbour
and bootstrap functions. Do not run every Python file simply because it
exists: their standalone entry points belong to earlier experiments.

Do not edit archived results to match a rerun. Keep new outputs separate,
record environment/data hashes, and investigate numerical differences.
