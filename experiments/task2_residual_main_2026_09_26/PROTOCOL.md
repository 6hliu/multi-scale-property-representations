# Frozen protocol: Task 2 with a local-market residual price target

## Status

This protocol was frozen on 26 September 2026 before the fresh run in this
directory. The broader project has already inspected 2025, so this is a
protocol-frozen temporal rerun rather than a globally untouched test.

## Research question

After removing the preceding-year local market level from the price target,
does joint DVF--DPE training improve or harm either task relative to matched
single-task training?

## Fixed data and split

- Shared inputs: the frozen 55-feature common schema.
- Development training: 2021--2023.
- Development validation and checkpoint selection: 2024 only.
- Refit: 2021--2024.
- Protocol-frozen temporal evaluation: 2025.
- Seeds: 42, 2026, 314, 7 and 99.

## Price target

The local market index is estimated on 2023 DVF for 2024 selection and on 2024
DVF for 2025 evaluation. It is hierarchical by property type, commune and
postcode, with the previously frozen shrinkage constants 50 and 30.

For property `i`:

`residual_i = log(price_i) - [log(area_i) + market_index_i]`

The evaluated log-price prediction adds the market baseline back:

`predicted_log_price_i = predicted_residual_i + log(area_i) + market_index_i`

No 2025 sale label contributes to the 2025 market index.

## Matched models

1. Single-task Residual Transformer: residual-price Huber loss only.
2. Shared Residual Transformer: the same price model and encoder jointly
   trained with the DPE energy task.
3. Energy-only Transformer: the exact frozen single-task energy baseline is
   reused because the price-target change cannot affect this model.

The shared objective is fixed as:

`L = L_residual_price + 0.20 * L_energy`.

Both residual models use the same two-layer, four-head, 64-dimensional Grouped
Transformer, room-count price auxiliary, optimiser, batch size and maximum 512
updates. No retrieval loss, neighbour correction, prediction fusion or Market
Token is used. Price-only checkpoints minimise 2024 reconstructed-price
Log-MAE. Shared checkpoints maximise the worst relative 2024 gain against the
matched residual price-only and frozen energy-only references.

## Metrics and decision rule

- Price primary metric: Log-MAE (lower is better).
- Energy primary metric: Macro-F1 (higher is better).
- Secondary metrics: Log-RMSE, log-R2, Euro MAE, balanced accuracy and accuracy.
- Uncertainty: paired 3 km spatial-block bootstrap with 20,000 repetitions.

The shared-versus-single result is reported regardless of its direction. The
2025 evaluation is not used to select checkpoints, loss weights or reporting.

