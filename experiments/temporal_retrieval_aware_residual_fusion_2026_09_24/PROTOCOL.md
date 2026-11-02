# Frozen protocol: retrieval-aware Residual Transformer + neighbour residual prediction

## Question

Can residual-comparability supervision improve the price-only Residual Grouped
Transformer, and does combining its learned representation with historical
neighbour residuals close or reverse the gap to full-55 XGBoost?

## Temporal protocol

- Development training: DVF 2021--2023.
- Development selection: DVF 2024 only.
- Refit: DVF 2021--2024.
- Protocol-frozen extension test: DVF 2025.
- Candidate comparables must be strictly earlier than the anchor/query, have
  the same property type, and have a different parcel identifier.
- No 2025 label is used for loss-weight, checkpoint or fusion selection.

The broader project has already inspected 2025.  This workflow is therefore a
protocol-frozen methodological extension, not a globally untouched test.

## Residual target

`r_i = log(price_i) - log(area_i) - lagged_local_market_index_i`.

Development uses a 2023-only market index and refit/test uses a 2024-only
market index.  The model predicts `r_i`; the area and market baseline are then
added back.

## Retrieval-aware loss

The price-only Grouped Transformer is trained with

`L = L_residual_price + gamma * L_residual_retrieval`.

The retrieval term is InfoNCE with temperature 0.10.  For each eligible anchor,
the candidate pool contains earlier, same-type, different-parcel geographic
neighbours.  The positive has a small residual gap while remaining structurally
plausible; four hard negatives are geographically/structurally plausible but
have large residual gaps.

Gamma candidates are 0.003, 0.01 and 0.03.  Seed 42 and 2024 validation select
gamma.  Each seed's checkpoint is selected independently by 2024 direct
residual-price Log-MAE.  No test metric participates.

## Neighbour-residual prediction

After the gamma screen, five development seeds are combined.  The 2024-only
fusion grid is identical to the preceding residual-retrieval experiment:

- geographic coefficient: 0, 0.10, 0.25;
- intrinsic coefficient: 0, 0.25, 0.50;
- neighbours: 5, 10, 20;
- median or softmax aggregation;
- softmax temperatures: 0.10, 0.25, 0.50;
- parametric-model weight: 0, 0.25, 0.50, 0.75, 1.

The final configuration minimises 2024 Log-MAE and is applied unchanged to
2025 after five-seed refitting.

## Comparisons

- original Residual Transformer ensemble;
- original Residual Transformer + post-hoc retrieval prediction;
- retrieval-aware Residual Transformer before neighbour fusion;
- retrieval-aware Residual Transformer + neighbour residual prediction;
- full-55 XGBoost.

Paired uncertainty uses 20,000 resamples of 3 km spatial blocks.  Positive
gain means the new fused predictor has lower absolute log error.

