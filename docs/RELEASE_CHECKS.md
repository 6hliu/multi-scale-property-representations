# Release checks — 2 November 2026

## Verified

- All included Python files parse.
- All 18 retained experiment modules import without launching their main routines.
- The feature schema partitions 55 unique inputs into nine groups.
- Synthetic Transformer forward passes have the expected token, embedding,
  price and seven-class energy output shapes.
- Price/energy joint loss and the retrieval contrastive loss backpropagate
  finite gradients on small synthetic examples.
- Tiny two-tree XGBoost regression/classification fits run on synthetic data
  in a separate process, matching the separate training-stage commands.
- The market-index test excludes evaluation labels and removes a source
  row's own contribution in the tested ordinary group case.
- The implemented fusion matches baseline + 0.75 direct residual +
  0.25 neighbour residual.
- A paired spatial-block bootstrap example has the expected gain sign
  and percentile bounds (40 repetitions for this unit test only).
- Original retained source-file hashes match the originals; changes to
  release copies are documented.
- A filename/content scan found no personal machine paths, common private-key
  or access-token patterns, large binary artifacts, or property-record columns
  in the published aggregate CSV files.

The release has eight synthetic unit/smoke tests. The environment is
recorded in `release_environment.json`. Pattern scans are a practical
check, not a proof that every possible secret format is detectable.

## Observed local environment limitation

An initial combined test process crashed when fitting XGBoost after
PyTorch forward/backward tests. The XGBoost fits passed in a fresh process.
The smoke suite therefore uses a subprocess for this test; it does not
assert that mixed-framework training in one process works in this environment.
The underlying native-runtime cause was not established, and the environment
was not modified to work around it. Run the documented training stages as
separate commands, not by importing all runners into one interactive session.

## Not performed

- No real-data training, full benchmark rerun or 20,000-repetition
  uncertainty rerun during release preparation.
- No fresh-environment installation or cross-platform/device validation.
- No fresh DVF/DPE/BDNB/OSM acquisition, verification of current external
  API availability, or redistribution of historical snapshots.
- No GitHub upload, commit or push.

Archived reference scores are supplied for comparison, not asserted to be
newly reproduced. Full reproduction needs the original data snapshots,
execution conditions and a compatible numerical environment.
