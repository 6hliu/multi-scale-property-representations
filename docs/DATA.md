# Data preparation

Property records are not included. Do not commit raw or processed records,
addresses/IDs, individual predictions, embedding arrays or fitted models.
The supplied commune-code list is the one geographic lookup needed by the
cleaning scripts.

## Route A: use the retained model-ready snapshot

For reproduction with the original author's locally retained data, place:

```text
experiments/temporal_main_experiment_2026_09_14/data/
    temporal_model_ready_55.csv.gz
    feature_manifest.json
```

Both files are ignored by Git. The public `configs/feature_manifest.json`
contains only the feature names/groups and can be used as the minimal
manifest for the training scripts; retain the full original data manifest
locally when available.

The expected historical compressed-file SHA-256 and aggregate counts are
recorded in `configs/data_reference.json`. Run:

```bash
python scripts/preflight.py --require-data --hash-data
```

A different gzip encoding can change a file hash even with identical rows.
Conversely, matching only the row count does not establish identical data.
The schema is listed in `configs/data_columns.json`.

The saved target `log_sale_price` is log1p(price). `property_type_is_house`
and `log_floor_area` are computed by the model feature-preparation code,
not required as stored columns. Room count is a separate price-only input.

## Route B: rebuild from raw sources

Use the original snapshots where possible. Downloading today's data or
map context will not necessarily reproduce the reported cohort or scores.
Data-provider terms and availability must be checked before acquisition
or redistribution. The acquisition scripts are historical implementations,
not a guarantee that external services retain the same schema.

Required sources:

- DVF: geocoded département 69 transaction files for 2021–2025, in the
  schema used by the retained cleaner. Name them `69_2021.csv.gz` through
  `69_2025.csv.gz`.
- DPE: the ADEME existing-dwellings dataset `dpe03existant`. The Week 4
  script declares the API endpoint, selected fields and commune filters.
- OSM: the retained Overpass caches for neighbourhood objects and roads.
  A fresh download changes the contextual snapshot.
- BDNB: the retained département 69 release, originally 2026-02-a.
  Building context is not reconstructed as an as-of-sale-time snapshot.

### 1. Clean the DVF temporal cohort

Place the five raw DVF files in
`experiments/experiments_2026_09_10/temporal_dvf/raw/`, then:

```bash
python experiments/experiments_2026_09_10/temporal_dvf/prepare_temporal_dvf.py
```

This writes `data/lyon_dvf_2021_2025_clean.csv` in that folder.

### 2. Clean the retained DPE cache

Place the historical DPE cache at
`experiments/week4_output_corrected/raw/lyon_dpe_selected_fields.jsonl`.
Use one raw DVF file for the Week 4 wrapper's required input; its
single-year DVF output does not replace step 1's temporal cohort.

```bash
python experiments/week4_data_pipeline.py \
  --dvf experiments/experiments_2026_09_10/temporal_dvf/raw/69_2025.csv.gz \
  --communes experiments/inputs/metropole_de_lyon_communes.csv \
  --output experiments/week4_output_corrected --skip-download
```

Without `--skip-download`, this contacts ADEME and creates a new snapshot.
The later common-table builder restricts observation years to 2021–2025.
It is not sufficient to recreate historical availability by filtering dates
in a newly downloaded dataset.

### 3. Build the common temporal table

```bash
python experiments/temporal_main_experiment_2026_09_14/build_temporal_common.py
```

### 4. Attach BDNB structural attributes

Extract the retained BDNB CSVs into `experiments/data_private/bdnb/csv/`.
The required tables are:
`rel_batiment_groupe_dpe_logement`, `rel_batiment_construction_rnb`,
`rel_batiment_groupe_adresse`, `rel_batiment_groupe_parcelle`,
`batiment_construction`, `batiment_groupe`,
`batiment_groupe_bdtopo_bat`, `batiment_groupe_ffo_bat`,
`batiment_groupe_synthese_propriete_usage`, and
`metadonnees_bdnb_version` (all `.csv`).

```bash
python experiments/experiments_2026_09_10/bdnb/enrich_with_bdnb.py \
  --dvf experiments/experiments_2026_09_10/temporal_dvf/data/lyon_dvf_2021_2025_clean.csv \
  --dpe experiments/week4_output_corrected/lyon_dpe_clean.csv \
  --bdnb-dir experiments/data_private/bdnb/csv \
  --output-dir experiments/temporal_main_experiment_2026_09_14/data/bdnb \
  --sample-size 0
```

### 5. Prepare OSM caches and the final 55-input table

For a retained complete cache, place its JSON files under
`experiments/week5_output/raw_osm/`. The final feature builder loads
this directory directly:

```bash
python experiments/temporal_main_experiment_2026_09_14/build_temporal_features.py
```

If caches are unavailable, the historical acquisition script can obtain a
new snapshot (network access, potentially substantial time):

```bash
python experiments/week5_osm_feature_pipeline.py \
  --common experiments/temporal_main_experiment_2026_09_14/data/temporal_common_core.csv.gz \
  --output experiments/week5_output
```

Then run the final feature builder. The Week 5 wrapper also emits older
spatial-split diagnostics; the main experiments use the temporal table
from `build_temporal_features.py`, not that older model-ready output.

## Reproduction boundary

This release supplies the transformation/training code and saved settings,
but not raw snapshots or trained weights. Exact numerical reproduction
requires the corresponding original data, compatible dependencies and
matching execution conditions. Smoke tests verify code paths on synthetic
inputs; they do not establish end-to-end reproduction of the report.
