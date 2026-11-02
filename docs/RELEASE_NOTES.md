# First main-experiments release

Scope: the three main report questions, their local helper dependencies,
data preparation code, configurations and aggregate reference results.
No distillation, asymmetric-sharing, partial-sharing or other appendix
experiment suites are included.

## Changes from the local research scripts

1. BDNB defaults now resolve within this repository instead of a personal
   Downloads/project path. Pass `--bdnb-dir` to use your own source directory.
2. The temporal DVF preparer's default Week 4 helper path points to the
   included `experiments/week4_data_pipeline.py`.
3. The shared Transformer module defaults to CPU, as recorded by the
   archived main-run manifest. `PROPERTY_DEVICE` allows an explicit override.
4. The final retrieval-aware runner no longer requires the appendix-only
   Prior-Fusion prediction file. `--include-prior-comparator` restores that
   optional comparison if the file exists. Primary training, selection,
   neighbour estimation and RA-Direct/RA-Fusion comparison are unchanged.
5. Its newly generated manifest explicitly describes 2025 as retrospective
   evaluation. Validation selection within this script still uses 2024.

All other retained source files are preserved. Original/release SHA-256
hashes are recorded in `source_manifest.json`.
Historical protocol files are archived verbatim, not rewritten as if they
had been written after the final experiment.

## Interpretation details retained

- The residual baseline uses `log1p(price) - log(area)`, not literally
  `log(price / area)`. It uses a 2023 index in development and a 2024
  index in refit/evaluation, fixed within each phase.
- Shrinkage pseudo-counts 50 (commune) and 30 (postcode) are fixed
  implementation choices, not selected by the retained validation search.
- Index-source training rows remove their own contributions at the type,
  commune and postcode levels. The exceptional global fallback is not
  leave-one-out. Postcodes are not assumed strictly nested in communes.
- Later contextual snapshots and recurring buildings mean that a temporal
  split alone does not remove every form of dependence or historical mismatch.
- Euro-MAE retains the original clipped exponential conversion. The saved
  target is log1p(price), so this is not exactly the inverse expm1 transform.
  The primary Log-MAE comparison does not depend on this euro conversion.
- A better complete retrieval pipeline does not isolate the effect of
  retrieval-aware training. RA-Direct versus RA-Fusion isolates the
  inference-time neighbour correction with fitted models held fixed.
