# Contextual Features, Multi-Task Transfer and Retrieval-Assisted Property Valuation: A Lyon Case Study

Research code accompanying Haoran Liu's COMP30013 project.

This first release covers the three main experimental questions. It preserves the
original numerical implementations and dated experiment folders so that imports
and provenance remain traceable. It is not a new benchmark run or a general-purpose library.

## Research questions

| Question | Controlled comparison | Entry point under `experiments/` |
| --- | --- | --- |
| Contextual Feature Utility | Separate XGBoost models, changing the feature set | `temporal_main_experiment_2026_09_14/run_classical_baselines.py` |
| Multi-Task Transfer | Matched single-task and shared residual Transformers | `task2_residual_main_2026_09_26/run_task2_residual.py` |
| Retrieval-Assisted Valuation | Earlier retrieval benchmark, then RA-Direct versus RA-Fusion | `temporal_main_experiment_2026_09_14/run_temporal_retrieval.py`; `temporal_retrieval_aware_residual_fusion_2026_09_24/run_experiment.py` |

The earlier shared retrieval benchmark uses a grouped Transformer in the retained
temporal experiment. It must not be confused with earlier spatial MLP experiments,
or with the final shared residual model.

**RA-Direct and RA-Fusion are two inference modes of the same five fitted,
price-only retrieval-aware residual models.** They are not two neural architectures.
Neither mode is the shared price–energy model.

## Main findings

The archived experiments show:

- Richer contextual inputs improve both task-specific XGBoost baselines.
- Hard sharing improves energy Macro-F1 but worsens price Log-MAE.
- Historical-neighbour correction improves the matched residual neural predictor.

Selected archived values (not regenerated during release preparation):

| Comparison | Price Log-MAE ↓ | Energy Macro-F1 ↑ |
| --- | ---: | ---: |
| Basic-input XGBoost | 0.2657 | 0.1635 |
| Full-input XGBoost | 0.1840 | 0.2693 |
| Single-task controls, mean over five seeds | 0.1969 | 0.2594 |
| Shared residual Transformer, mean over five seeds | 0.2035 | 0.2724 |
| RA-Direct, prediction ensemble | 0.188496 | — |
| RA-Fusion, prediction/distance ensembles | 0.185655 | — |

RA-Fusion improves RA-Direct by about 0.002840 Log-MAE, with paired spatial
95% CI [0.002112, 0.003617]. XGBoost has a slightly lower point error than
RA-Fusion; their paired interval includes zero. This does **not** establish
equivalence or an overall winner. The gain over geographic matching compares
complete pipelines, not the isolated effect of the learned selector.

## Evaluation

- Development: train on 2021–2023; select settings on 2024.
- Refit on 2021–2024; evaluate on 2025.
- Five neural seeds: 42, 2026, 314, 7, 99.
- Price target: `log1p(sale_price_eur)`; energy: seven classes A–G.
- Paired bootstrap resamples whole 3 km spatial blocks with replacement.
  Both methods use the same sampled blocks; metrics use saved predictions,
  without retraining. The 2.5th and 97.5th percentiles define the interval.

Earlier exploration had already inspected 2025. These results are therefore
**retrospective temporal evaluation**, not an untouched prospective test.
Multi-Task Transfer averages per-seed metric gains within each resample;
Retrieval-Assisted Valuation evaluates its specified ensembles before resampling.
Training seeds are not themselves bootstrapped.

## Setup and lightweight checks

Python 3.11 is the release-check environment. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/preflight.py
python -m unittest discover -s tests -v
```

These checks do not download data or train full models. Dependencies are described
in [ENVIRONMENT.md](docs/ENVIRONMENT.md); the original complete environment lock
was not recorded. See [RELEASE_CHECKS.md](docs/RELEASE_CHECKS.md) for what was
actually tested, rather than interpreting these checks as a full reproduction.

## Data and running the experiments

Raw/processed property records, per-record predictions, embeddings and fitted
weights are intentionally **not distributed**. See [DATA.md](docs/DATA.md).
After preparing the data, run the stages in [RUNNING.md](docs/RUNNING.md).
Full training can be expensive; none is triggered by installation or tests.

`configs/` records the feature schema, search spaces and saved selections.
These JSON files document the experiment; the original runners retain their
in-code constants and reselect on 2024. They do not automatically load every
saved selection as a fixed configuration.

`reference_results/` contains small archived aggregate tables, separate from
the working output directories. See its [README](reference_results/README.md).
Some retained utilities also produce ancillary diagnostics, but appendix
experiment suites are outside this release.

## Repository layout

```text
configs/             Feature schema, search spaces, archived selections
docs/                Data preparation, run order, provenance, environment
experiments/         Original experiment code and necessary local helpers
reference_results/   Archived aggregate results only
scripts/             Non-training preflight checks
tests/               Synthetic unit/smoke tests
```

Portability edits are listed in [RELEASE_NOTES.md](docs/RELEASE_NOTES.md).
Historical `PROTOCOL.md` files are retained as development records; their
initial plans do not supersede the final five-seed design, log1p target or
retrospective evaluation qualification.

## Availability and reuse

Repository: https://github.com/6hliu/multi-scale-property-representations

No software licence has been selected for this release yet. Public availability
alone does not grant an open-source reuse licence. Data remain subject to their
respective providers' terms; no redistribution rights are asserted here.
