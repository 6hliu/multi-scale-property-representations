# Archived reference results

These small aggregate files were copied from completed local experiments.
They are not outputs of release smoke tests and are not a claim that
the complete pipelines have been rerun from this release.

- `contextual_features/`: classical model/feature comparisons, validation
  tuning and the four full-versus-intrinsic XGBoost uncertainty rows.
- `multi_task_transfer/`: final residual matched comparison, per-seed
  metrics and paired uncertainty (also retaining secondary comparisons).
- `earlier_retrieval/`: original temporal shared-Transformer retrieval
  benchmark. Distinct from the final residual models.
- `residual_retrieval/`: final price-only retrieval-aware model.
  The archived comparison/CI files also retain an explicitly named
  "Original residual Transformer + retrieval" row for provenance; that
  earlier Prior-Fusion comparator is not required or rerun in this release.

RA-Direct and RA-Fusion use the same five fitted models. Their ensemble
metrics should not be substituted for Multi-Task Transfer's mean per-seed
metrics. The final direct-versus-fusion CI belongs only to that paired
comparison, not to the whole gap from geographic matching.

Exact source locations and checksums are in `docs/source_manifest.json`.
No record IDs, individual prices, coordinates or predictions are included
in these aggregate tables. The commune list elsewhere is geographic
reference metadata, not property-level observations.
