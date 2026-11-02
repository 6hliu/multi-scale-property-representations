# Predeclared temporal main experiment

Status: frozen before running the new temporal 2025 evaluation.

## Purpose

This experiment is the temporal main analysis for the property-representation
project. It replaces the earlier spatial-split development results as the main
test of future-year generalisation. Earlier project work has already used 2025
records, so 2025 must not be described as globally unseen. The defensible claim
is that the protocol and model-selection rules below were frozen before this
new temporal rerun.

## Cohort and split

- Study area: Metropole de Lyon.
- DVF: clean ordinary single-residence sales from 2021--2025.
- DPE: clean dwelling diagnostics dated 2021--2025; 2026 is excluded.
- Development training: 2021--2023.
- Validation/model selection: calendar year 2024 only.
- Final refit: 2021--2024 with the selected configuration.
- Test: calendar year 2025, evaluated once after refitting.

All imputers, scalers, class weights and price centres are estimated from the
corresponding training period. No target-derived statistic from 2025 may be
used for model choice.

## Inputs

The full representation uses 55 shared inputs:

1. property type and log floor area;
2. longitude and latitude;
3. ten OSM object counts at each of 250 m, 500 m and 1,000 m;
4. six accepted BDNB structural values and six missingness indicators;
5. six nearest-OSM distances;
6. months since 2021-01-01 plus sine and cosine of calendar month.

Room count is a DVF-only auxiliary input to the price head. It is not part of
the shared embedding. Building age is always recomputed as observation year
minus construction year. Ambiguous, conflicting or low-confidence BDNB links
supply missing structural values.

OSM and BDNB are single later snapshots. They are treated as approximately
time-invariant retrospective context, not as historically exact covariates.
An intrinsic + coordinates + time sensitivity analysis is therefore required.

## Models and validation-only selection

Price baselines: training median, Ridge and XGBoost. Energy baselines: training
majority class, multinomial logistic regression and XGBoost. Linear and tree
hyperparameters are selected on 2024 only.

The neural representation model is the carried-forward `wide64` grouped
Transformer: 64 dimensions, two Transformer layers, four heads, feed-forward
width 128 and dropout 0.10. A ninth temporal token is added to the previous
eight feature-group tokens. Price-only, energy-only and shared versions use the
same architecture. The shared loss is

`price Huber loss + lambda_energy * weighted energy cross-entropy`.

Every shared update contains one DVF batch and one DPE batch. Energy uses
square-root inverse-frequency class weights. `lambda_energy` is screened from
0.03, 0.10, 0.20 and 0.30 using seed 42 and 2024 validation only. Final matched
comparisons use seeds 42, 2026 and 314. The validation-selected training step is
reused for the 2021--2024 refit.

Checkpoint scoring uses every 2024 DVF sale and a fixed class-stratified sample
of 30,000 2024 DPE records to keep repeated full-cohort Transformer inference
computationally bounded. Each selected checkpoint is additionally evaluated on
the complete 2024 DPE cohort for reporting. The subset is never used for model
fitting.

Neural updates use batches of 1,024. Architecture/lambda screens run for 256
updates and final development runs for 512 updates. This preserves the same
number of sampled training examples as 1,024 and 2,048 updates respectively
with a batch size of 256, while reducing CPU overhead.

Primary price metric: Log-MAE. Secondary: Euro MAE, Log-RMSE and log R-squared.
Primary energy metric: Macro-F1. Secondary: balanced accuracy and accuracy.

## Retrieval

For 2024 validation queries, candidates must be dated before the query and come
from 2021--2023. For 2025 test queries, candidates must be dated before the
query and come from 2021--2024. Candidates must have the same property type;
the query parcel is excluded. Shortlist size, neighbour count and blending are
selected on 2024 and frozen for 2025.

Compared methods are geographic distance, intrinsic similarity,
geographic + intrinsic similarity, full raw-feature similarity, price-only
Transformer embedding, shared Transformer embedding and a validation-selected
ensemble. Retrieval reports the same price metrics as direct prediction.

## Sensitivity and integrity checks

- full 55 inputs versus intrinsic + coordinates + time;
- all records versus parcels not observed in an earlier year;
- all DPE records versus entity-equal weighting using an accepted BDNB/RNB/BAN
  cluster proxy;
- source and year counts, missingness, class balance, duplicate keys and
  target leakage audit;
- three neural seeds with mean, standard deviation and paired differences.
