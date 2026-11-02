#!/usr/bin/env python3
"""Build a frozen 2021--2025 Lyon DVF temporal cohort.

The cleaning operation is deliberately delegated to the corrected Week 4
pipeline so that the temporal experiment does not silently introduce a second
definition of a valid residential transaction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


YEARS = (2021, 2022, 2023, 2024, 2025)
PROPERTY_KEY = [
    "id_mutation",
    "id_parcelle",
    "adresse_numero",
    "adresse_nom_voie",
    "code_commune",
    "type_local",
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "longitude",
    "latitude",
]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=here / "raw")
    parser.add_argument("--output-dir", type=Path, default=here / "data")
    parser.add_argument(
        "--communes",
        type=Path,
        default=root / "inputs" / "metropole_de_lyon_communes.csv",
    )
    parser.add_argument(
        "--week4-pipeline",
        type=Path,
        default=root / "week4_data_pipeline.py",
    )
    return parser.parse_args()


def load_week4_module(path: Path):
    spec = importlib.util.spec_from_file_location("corrected_week4_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import corrected Week 4 pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def mutation_risk_audit(path: Path, metro_codes: set[str], clean_code) -> dict:
    usecols = list(dict.fromkeys(PROPERTY_KEY + ["nature_mutation", "code_postal"]))
    frame = pd.read_csv(
        path,
        usecols=usecols,
        dtype={
            "id_mutation": "string",
            "id_parcelle": "string",
            "code_commune": "string",
            "code_postal": "string",
        },
        low_memory=False,
    )
    frame["code_commune"] = clean_code(frame["code_commune"])
    metro = frame.loc[frame["code_commune"].isin(metro_codes)]
    residential = metro.loc[
        metro["nature_mutation"].astype("string").str.strip().eq("Vente")
        & metro["type_local"].isin(["Maison", "Appartement"])
    ].copy()
    unique = residential.drop_duplicates(PROPERTY_KEY)
    sizes = unique.groupby("id_mutation", dropna=False).size()
    multi = sizes.loc[sizes > 1]
    return {
        "residential_source_rows": int(len(residential)),
        "unique_residential_property_rows": int(len(unique)),
        "unique_residential_mutations": int(len(sizes)),
        "single_residence_mutations": int((sizes == 1).sum()),
        "multi_residence_mutations_excluded": int(len(multi)),
        "residential_property_rows_in_excluded_multi_mutations": int(multi.sum()),
        "max_residential_properties_in_one_mutation": int(sizes.max()) if len(sizes) else 0,
        "policy": (
            "Keep one deduplicated residential property row only when a mutation "
            "contains exactly one distinct Maison/Appartement. This avoids assigning "
            "one mutation-level valeur_fonciere to multiple dwellings."
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_year_dir = args.output_dir / "by_year"
    by_year_dir.mkdir(parents=True, exist_ok=True)

    week4 = load_week4_module(args.week4_pipeline)
    metro_codes, _ = week4.load_metro_codes(args.communes)

    yearly_frames: list[pd.DataFrame] = []
    audits: dict[str, dict] = {}
    sources: list[dict] = []

    for year in YEARS:
        raw_path = args.raw_dir / f"69_{year}.csv.gz"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        year_dir = by_year_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        clean, audit = week4.build_dvf(raw_path, metro_codes, year_dir)
        observed_years = set(pd.to_datetime(clean["observation_date"]).dt.year.unique())
        if observed_years != {year}:
            raise ValueError(
                f"Expected only {year} observations in {raw_path}, got {observed_years}"
            )

        clean = clean.copy()
        clean.insert(0, "source_year", year)
        clean.insert(1, "source_id_mutation", clean["record_id"].astype("string"))
        clean["record_id"] = (
            clean["source_year"].astype("string")
            + "::"
            + clean["source_id_mutation"].astype("string")
        )
        clean["temporal_split"] = (
            "train" if year <= 2023 else "validation" if year == 2024 else "test"
        )
        clean.to_csv(year_dir / "lyon_dvf_clean.csv", index=False)
        yearly_frames.append(clean)

        audits[str(year)] = {
            **{key: json_value(value) for key, value in audit.items()},
            "mutation_multiplicity": mutation_risk_audit(
                raw_path, metro_codes, week4.clean_code
            ),
        }
        sources.append(
            {
                "year": year,
                "url": (
                    "https://files.data.gouv.fr/geo-dvf/latest/csv/"
                    f"{year}/departements/69.csv.gz"
                ),
                "local_file": str(raw_path),
                "bytes": raw_path.stat().st_size,
                "sha256": sha256(raw_path),
            }
        )

    combined = pd.concat(yearly_frames, ignore_index=True)
    combined["observation_date"] = pd.to_datetime(combined["observation_date"])
    combined = combined.sort_values(
        ["observation_date", "source_year", "source_id_mutation"]
    ).reset_index(drop=True)

    if combined["record_id"].duplicated().any():
        raise ValueError("year::mutation record_id is not unique")
    if not combined["observation_date"].dt.year.eq(combined["source_year"]).all():
        raise ValueError("source_year does not match observation_date")

    combined_path = args.output_dir / "lyon_dvf_2021_2025_clean.csv"
    combined.to_csv(combined_path, index=False, date_format="%Y-%m-%d")

    assignments = combined[
        [
            "record_id",
            "source_id_mutation",
            "parcel_id",
            "source_year",
            "observation_date",
            "temporal_split",
        ]
    ].copy()

    first_parcel_year = combined.groupby("parcel_id")["source_year"].transform("min")
    assignments["parcel_first_year"] = first_parcel_year
    assignments["parcel_seen_in_earlier_year"] = (
        assignments["source_year"] > assignments["parcel_first_year"]
    )
    assignments["unseen_parcel_sensitivity_keep"] = ~assignments[
        "parcel_seen_in_earlier_year"
    ]
    assignments.to_csv(
        args.output_dir / "frozen_temporal_assignments.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    parcel_years = combined.groupby("parcel_id")["source_year"].nunique()
    split_counts = combined["temporal_split"].value_counts().to_dict()
    unseen_counts = assignments.loc[
        assignments["unseen_parcel_sensitivity_keep"], "temporal_split"
    ].value_counts().to_dict()

    audit = {
        "release": {
            "name": "Demandes de valeurs foncieres geolocalisees",
            "release_family": "https://files.data.gouv.fr/geo-dvf/latest/csv/",
            "years": list(YEARS),
            "department": "69 (Rhone)",
            "downloaded_utc_note": "Downloaded together for this frozen experiment.",
            "year_2026_policy": "Not used: 2026 is incomplete and is outside the prespecified 2021-2025 design.",
        },
        "cleaning_implementation": str(args.week4_pipeline),
        "record_key": "source_year::source_id_mutation",
        "parcel_key": "parcel_id (cadastral parcel proxy; not guaranteed to be a building ID)",
        "per_year": audits,
        "combined": {
            "rows": int(len(combined)),
            "date_min": combined["observation_date"].min().isoformat(),
            "date_max": combined["observation_date"].max().isoformat(),
            "record_id_duplicates": int(combined["record_id"].duplicated().sum()),
            "split_rows": {key: int(value) for key, value in split_counts.items()},
            "unique_parcels": int(combined["parcel_id"].nunique()),
            "parcels_seen_in_multiple_years": int((parcel_years > 1).sum()),
            "rows_on_parcels_seen_in_earlier_year": int(
                assignments["parcel_seen_in_earlier_year"].sum()
            ),
            "unseen_parcel_sensitivity_rows": {
                key: int(value) for key, value in unseen_counts.items()
            },
        },
        "temporal_design": {
            "train": "2021-2023",
            "validation": "2024",
            "test": "2025",
            "main_population": "Repeated parcels across years are allowed and explicitly reported.",
            "sensitivity": (
                "For validation/test, unseen_parcel_sensitivity_keep retains only "
                "records whose cadastral parcel was not observed in an earlier year. "
                "This is a parcel-level proxy, not a proven distinct-building split."
            ),
        },
    }
    with (args.output_dir / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    pd.DataFrame(sources).to_csv(args.output_dir / "source_manifest.csv", index=False)

    print(f"Clean combined data: {combined_path}")
    print(json.dumps(audit["combined"], indent=2))


if __name__ == "__main__":
    main()
