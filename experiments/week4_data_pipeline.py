#!/usr/bin/env python3
"""Week 4 pipeline: acquire, clean, harmonise, and audit Lyon DVF + DPE.

Core outputs:
  lyon_dvf_clean.csv
  lyon_dpe_clean.csv
  lyon_common_schema.csv
  audit_summary.json
  distributions/*.csv

The two sources do not need record-level linkage for partially-labelled
multi-task learning. They are harmonised to a shared feature/label schema.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests


DPE_API = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"

DPE_FIELDS = [
    "numero_dpe",
    "date_etablissement_dpe",
    "date_derniere_modification_dpe",
    "etiquette_dpe",
    "etiquette_ges",
    "type_batiment",
    "surface_habitable_logement",
    "annee_construction",
    "periode_construction",
    "code_departement_ban",
    "code_insee_ban",
    "nom_commune_ban",
    "code_postal_ban",
    "adresse_ban",
    "identifiant_ban",
    "id_rnb",
    "statut_geocodage",
    "_geopoint",
]

LYON_ARRONDISSEMENT_CODES = {f"6938{i}" for i in range(1, 10)}
LYON_MAIN_CODE = "69123"

COMMON_COLUMNS = [
    "record_id",
    "source",
    "property_type",
    "floor_area_m2",
    "longitude",
    "latitude",
    "commune_code",
    "postcode",
    "observation_date",
    "sale_price_eur",
    "log_sale_price",
    "energy_class",
    "price_mask",
    "energy_mask",
]


def clean_code(series: pd.Series, width: int = 5) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(width)
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def load_metro_codes(commune_file: Path) -> tuple[set[str], set[str]]:
    communes = pd.read_csv(commune_file, dtype="string")
    if "code_commune" not in communes:
        raise ValueError(f"Missing code_commune in {commune_file}")
    dvf_codes = set(clean_code(communes["code_commune"]).dropna())
    # ADEME currently contains a large number of Lyon DPE records under the
    # nine municipal arrondissement codes (69381--69389). Keep those codes in
    # the API query and include 69123 as a defensive fallback for records that
    # use the city-wide Lyon code.
    dpe_codes = set(dvf_codes)
    dpe_codes.add(LYON_MAIN_CODE)
    missing_arrondissements = LYON_ARRONDISSEMENT_CODES - dpe_codes
    if missing_arrondissements:
        raise ValueError(
            "The commune input is missing Lyon arrondissement codes: "
            + ", ".join(sorted(missing_arrondissements))
        )
    return dvf_codes, dpe_codes


def build_dvf(raw_path: Path, dvf_codes: set[str], output_dir: Path) -> tuple[pd.DataFrame, dict]:
    dtype = {
        "id_mutation": "string",
        "id_parcelle": "string",
        "code_commune": "string",
        "code_postal": "string",
        "code_departement": "string",
    }
    raw = pd.read_csv(raw_path, dtype=dtype, low_memory=False)
    raw_rows = len(raw)
    raw["code_commune"] = clean_code(raw["code_commune"])
    raw["code_postal"] = clean_code(raw["code_postal"])
    raw["date_mutation"] = pd.to_datetime(raw["date_mutation"], errors="coerce")
    for col in [
        "valeur_fonciere",
        "surface_reelle_bati",
        "nombre_pieces_principales",
        "longitude",
        "latitude",
    ]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    metro = raw[raw["code_commune"].isin(dvf_codes)].copy()
    sales = metro[
        metro["nature_mutation"].astype("string").str.strip().eq("Vente")
    ].copy()
    residential = sales[sales["type_local"].isin(["Maison", "Appartement"])].copy()

    property_key = [
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
    unique = residential.drop_duplicates(property_key).copy()
    count_by_mutation = unique.groupby("id_mutation").size()
    simple_ids = count_by_mutation[count_by_mutation == 1].index
    simple = unique[unique["id_mutation"].isin(simple_ids)].copy()

    simple["price_per_sqm"] = simple["valeur_fonciere"] / simple["surface_reelle_bati"]
    valid = simple[
        simple["valeur_fonciere"].between(10_000, 20_000_000)
        & simple["surface_reelle_bati"].between(9, 500)
        & simple["nombre_pieces_principales"].between(1, 20)
        & simple["longitude"].between(-180, 180)
        & simple["latitude"].between(-90, 90)
        & simple["date_mutation"].notna()
        & simple["price_per_sqm"].between(200, 30_000)
    ].copy()

    valid["property_type"] = valid["type_local"].map(
        {"Maison": "House", "Appartement": "Apartment"}
    )
    valid["source_commune_code"] = valid["code_commune"]
    valid["canonical_commune_code"] = valid["code_commune"].replace(
        {code: LYON_MAIN_CODE for code in LYON_ARRONDISSEMENT_CODES}
    )
    valid["log_sale_price"] = np.log1p(valid["valeur_fonciere"])
    valid = valid.rename(
        columns={
            "id_mutation": "record_id",
            "surface_reelle_bati": "floor_area_m2",
            "canonical_commune_code": "commune_code",
            "code_postal": "postcode",
            "date_mutation": "observation_date",
            "valeur_fonciere": "sale_price_eur",
            "nombre_pieces_principales": "room_count",
            "id_parcelle": "parcel_id",
        }
    )
    valid["source"] = "DVF"
    valid["energy_class"] = pd.NA
    valid["price_mask"] = 1
    valid["energy_mask"] = 0

    keep = COMMON_COLUMNS + [
        "room_count",
        "parcel_id",
        "source_commune_code",
        "nom_commune",
        "price_per_sqm",
    ]
    clean = valid[keep].sort_values(["observation_date", "record_id"]).reset_index(drop=True)
    clean.to_csv(output_dir / "lyon_dvf_clean.csv", index=False)

    audit = {
        "raw_rows": raw_rows,
        "metro_rows": len(metro),
        "ordinary_sale_rows": len(sales),
        "residential_rows": len(residential),
        "unique_residential_rows": len(unique),
        "simple_single_residence_rows": len(simple),
        "clean_rows": len(clean),
        "removed_by_final_quality_rules": len(simple) - len(clean),
        "coordinate_completeness_before_final_filter": float(
            (simple["longitude"].notna() & simple["latitude"].notna()).mean()
        ),
        "date_min": clean["observation_date"].min(),
        "date_max": clean["observation_date"].max(),
    }
    return clean, audit


def dpe_query(codes: set[str]) -> str:
    values = " OR ".join(sorted(codes))
    return f"code_insee_ban:({values})"


def download_dpe(dpe_codes: set[str], cache_file: Path, page_size: int = 10_000) -> dict:
    """Download selected DPE fields as newline-delimited JSON with resume support."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    state_file = cache_file.with_suffix(".state.json")
    if state_file.exists():
        state = json.loads(state_file.read_text())
        next_url = state.get("next_url")
        downloaded = int(state.get("downloaded", 0))
        total = int(state.get("total", 0))
        mode = "a"
        print(f"Resuming DPE download at {downloaded:,}/{total:,}")
    else:
        params = {
            "size": page_size,
            "select": ",".join(DPE_FIELDS),
            "qs": dpe_query(dpe_codes),
        }
        prepared = requests.Request("GET", DPE_API, params=params).prepare()
        next_url = prepared.url
        downloaded = 0
        total = 0
        mode = "w"

    session = requests.Session()
    session.headers.update({"User-Agent": "unimelb-student-property-research/1.0"})
    with cache_file.open(mode, encoding="utf-8") as stream:
        while next_url:
            error = None
            for attempt in range(1, 6):
                try:
                    response = session.get(next_url, timeout=180)
                    response.raise_for_status()
                    payload = response.json()
                    break
                except (requests.RequestException, ValueError) as exc:
                    error = exc
                    wait = min(60, 5 * 2 ** (attempt - 1))
                    print(f"DPE request attempt {attempt}/5 failed: {exc}; retrying in {wait}s")
                    time.sleep(wait)
            else:
                raise RuntimeError(f"DPE download failed after retries: {error}")

            total = int(payload.get("total", total))
            rows = payload.get("results", [])
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            downloaded += len(rows)
            next_url = payload.get("next")
            state_file.write_text(
                json.dumps(
                    {"next_url": next_url, "downloaded": downloaded, "total": total},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print(f"Downloaded DPE: {downloaded:,}/{total:,} ({downloaded/max(total,1):.1%})")
            if not rows:
                break
            time.sleep(0.15)

    if state_file.exists() and not next_url:
        state_file.unlink()
    return {"downloaded": downloaded, "api_total": total}


def parse_geopoint(value: Any) -> tuple[float, float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan, np.nan
    try:
        lat_text, lon_text = str(value).split(",", maxsplit=1)
        return float(lon_text.strip()), float(lat_text.strip())
    except (ValueError, TypeError):
        return np.nan, np.nan


def build_dpe(
    cache_file: Path,
    dpe_codes: set[str],
    output_dir: Path,
    cutoff_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    with cache_file.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    raw = pd.DataFrame(rows)
    raw_rows = len(raw)
    if raw.empty:
        raise RuntimeError("DPE cache is empty")

    raw["code_insee_ban"] = clean_code(raw["code_insee_ban"])
    raw["code_postal_ban"] = clean_code(raw["code_postal_ban"])
    raw = raw[raw["code_insee_ban"].isin(dpe_codes)].copy()
    requested_code_rows = len(raw)
    raw["surface_habitable_logement"] = pd.to_numeric(
        raw["surface_habitable_logement"], errors="coerce"
    )
    raw["date_etablissement_dpe"] = pd.to_datetime(
        raw["date_etablissement_dpe"], errors="coerce"
    )
    raw["date_derniere_modification_dpe"] = pd.to_datetime(
        raw["date_derniere_modification_dpe"], errors="coerce"
    )
    if cutoff_date is not None:
        raw = raw[
            raw["date_etablissement_dpe"].notna()
            & raw["date_etablissement_dpe"].le(cutoff_date)
        ].copy()
    rows_after_cutoff = len(raw)

    coords = raw["_geopoint"].apply(parse_geopoint)
    raw["longitude"] = coords.str[0]
    raw["latitude"] = coords.str[1]
    raw["type_normalised"] = raw["type_batiment"].astype("string").str.lower().map(
        {"maison": "House", "appartement": "Apartment"}
    )
    raw["etiquette_dpe"] = raw["etiquette_dpe"].astype("string").str.upper().str.strip()

    # numero_dpe is the diagnostic identifier. Keep the most recently modified
    # copy if the API snapshot happens to contain duplicates.
    raw = raw.sort_values("date_derniere_modification_dpe").drop_duplicates(
        "numero_dpe", keep="last"
    )
    valid = raw[
        raw["type_normalised"].isin(["House", "Apartment"])
        & raw["surface_habitable_logement"].between(9, 500)
        & raw["etiquette_dpe"].isin(list("ABCDEFG"))
        & raw["longitude"].between(-180, 180)
        & raw["latitude"].between(-90, 90)
        & raw["date_etablissement_dpe"].notna()
    ].copy()

    valid = valid.rename(
        columns={
            "numero_dpe": "record_id",
            "type_normalised": "property_type",
            "surface_habitable_logement": "floor_area_m2",
            "code_insee_ban": "commune_code",
            "code_postal_ban": "postcode",
            "date_etablissement_dpe": "observation_date",
            "etiquette_dpe": "energy_class",
        }
    )
    valid["source_commune_code"] = valid["commune_code"]
    valid["commune_code"] = valid["commune_code"].replace(
        {code: LYON_MAIN_CODE for code in LYON_ARRONDISSEMENT_CODES}
    )
    valid["source"] = "DPE"
    valid["sale_price_eur"] = np.nan
    valid["log_sale_price"] = np.nan
    valid["price_mask"] = 0
    valid["energy_mask"] = 1

    keep = COMMON_COLUMNS + [
        "nom_commune_ban",
        "source_commune_code",
        "adresse_ban",
        "identifiant_ban",
        "id_rnb",
        "annee_construction",
        "periode_construction",
        "etiquette_ges",
        "statut_geocodage",
    ]
    clean = valid[keep].sort_values(["observation_date", "record_id"]).reset_index(drop=True)
    clean.to_csv(output_dir / "lyon_dpe_clean.csv", index=False)

    audit = {
        "downloaded_rows": raw_rows,
        "rows_matching_requested_commune_codes": requested_code_rows,
        "rows_after_cutoff": rows_after_cutoff,
        "cutoff_date": cutoff_date,
        "unique_diagnostics_after_deduplication": len(raw),
        "clean_rows": len(clean),
        "removed_by_final_quality_rules": len(raw) - len(clean),
        "coordinate_completeness_before_final_filter": float(
            (raw["longitude"].notna() & raw["latitude"].notna()).mean()
        ),
        "area_completeness_before_final_filter": float(
            raw["surface_habitable_logement"].notna().mean()
        ),
        "energy_label_completeness_before_final_filter": float(
            raw["etiquette_dpe"].isin(list("ABCDEFG")).mean()
        ),
        "date_min": clean["observation_date"].min(),
        "date_max": clean["observation_date"].max(),
        "rnb_id_completeness_clean": float(clean["id_rnb"].notna().mean()),
        "lyon_proper_clean_rows": int(
            clean["source_commune_code"].isin(
                LYON_ARRONDISSEMENT_CODES | {LYON_MAIN_CODE}
            ).sum()
        ),
    }
    return clean, audit


def write_distribution(series: pd.Series, path: Path, name: str) -> None:
    out = series.value_counts(dropna=False).rename_axis(name).reset_index(name="count")
    out["proportion"] = out["count"] / out["count"].sum()
    out.to_csv(path, index=False)


def harmonise_and_audit(
    dvf: pd.DataFrame,
    dpe: pd.DataFrame,
    dvf_audit: dict,
    dpe_audit: dict,
    output_dir: Path,
) -> None:
    distributions = output_dir / "distributions"
    distributions.mkdir(exist_ok=True)

    common = pd.concat(
        [dvf[COMMON_COLUMNS], dpe[COMMON_COLUMNS]], ignore_index=True
    ).sort_values(["source", "observation_date", "record_id"])
    common.to_csv(output_dir / "lyon_common_schema.csv", index=False)

    write_distribution(dvf["property_type"], distributions / "dvf_property_types.csv", "property_type")
    write_distribution(dpe["property_type"], distributions / "dpe_property_types.csv", "property_type")
    write_distribution(dpe["energy_class"], distributions / "dpe_energy_classes.csv", "energy_class")
    write_distribution(dvf["commune_code"], distributions / "dvf_communes.csv", "commune_code")
    write_distribution(dpe["commune_code"], distributions / "dpe_communes.csv", "commune_code")

    area_summary = pd.concat(
        [
            dvf["floor_area_m2"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).rename("DVF"),
            dpe["floor_area_m2"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).rename("DPE"),
        ],
        axis=1,
    )
    area_summary.to_csv(distributions / "floor_area_comparison.csv")

    summary = {
        "dvf": dvf_audit,
        "dpe": dpe_audit,
        "common_schema": {
            "total_rows": len(common),
            "dvf_rows": int((common["source"] == "DVF").sum()),
            "dpe_rows": int((common["source"] == "DPE").sum()),
            "columns": COMMON_COLUMNS,
            "shared_encoder_inputs_before_osm": [
                "property_type",
                "floor_area_m2",
                "longitude",
                "latitude",
            ],
            "labels_and_masks": [
                "log_sale_price",
                "energy_class",
                "price_mask",
                "energy_mask",
            ],
            "important_limitation": (
                "DVF surface_reelle_bati and DPE surface_habitable_logement are "
                "not identical definitions; compare distributions and run sensitivity checks."
            ),
        },
    }
    summary = json.loads(json.dumps(summary, default=json_safe))
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# Week 4 Lyon DVF + DPE outputs

## Clean datasets
- `lyon_dvf_clean.csv`: {len(dvf):,} sale-labelled DVF properties.
- `lyon_dpe_clean.csv`: {len(dpe):,} energy-labelled DPE properties.
- `lyon_common_schema.csv`: stacked partially-labelled multi-task table.

## Shared encoder inputs available before Week 5
- `property_type`
- `floor_area_m2`
- `longitude`, `latitude`

OSM neighbourhood features will be derived consistently for both sources in Week 5.

## Labels
- DVF: `log_sale_price`, with `price_mask=1`.
- DPE: `energy_class`, with `energy_mask=1`.

## Important interpretation
Rows from DVF and DPE are not assumed to describe the same dwelling. The shared
encoder can be trained with alternating batches or masked losses. Only separately
verified high-confidence links should be treated as dual-labelled property records.

`surface_reelle_bati` (DVF) and `surface_habitable_logement` (DPE) are related but
not identical area definitions. Their distributions are reported separately.

DVF and DPE Lyon arrondissement codes (69381--69389) are harmonised to BAN/INSEE
Lyon commune code 69123 in `commune_code`; `source_commune_code` preserves the
original. The DPE API query includes all nine arrondissement codes plus 69123.

The DPE dataset is much larger than DVF. Training must use task-balanced batches
or explicit sampling rather than letting each DPE row dominate an epoch.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvf", required=True, type=Path)
    parser.add_argument("--communes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--page-size", type=int, default=10_000)
    parser.add_argument(
        "--dpe-cutoff",
        type=str,
        default=None,
        help="Optional inclusive date cutoff (YYYY-MM-DD) for DPE establishment dates.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dvf_codes, dpe_codes = load_metro_codes(args.communes)
    print(f"DVF commune codes: {len(dvf_codes)}; DPE/BAN commune codes: {len(dpe_codes)}")

    print("\n[1/4] Cleaning DVF")
    dvf, dvf_audit = build_dvf(args.dvf, dvf_codes, args.output)
    print(f"Clean DVF rows: {len(dvf):,}")

    cache_file = args.output / "raw" / "lyon_dpe_selected_fields.jsonl"
    if not args.skip_download:
        print("\n[2/4] Downloading DPE from official ADEME API")
        download_dpe(dpe_codes, cache_file, page_size=args.page_size)
    elif not cache_file.exists():
        raise FileNotFoundError(f"--skip-download was set but cache does not exist: {cache_file}")

    print("\n[3/4] Cleaning DPE")
    cutoff_date = pd.Timestamp(args.dpe_cutoff) if args.dpe_cutoff else None
    dpe, dpe_audit = build_dpe(
        cache_file, dpe_codes, args.output, cutoff_date=cutoff_date
    )
    dvf_lyon_rows = int(
        dvf["source_commune_code"].isin(LYON_ARRONDISSEMENT_CODES).sum()
    )
    dpe_lyon_rows = int(
        dpe["source_commune_code"].isin(
            LYON_ARRONDISSEMENT_CODES | {LYON_MAIN_CODE}
        ).sum()
    )
    if dvf_lyon_rows >= 100 and dpe_lyon_rows < 1_000:
        raise RuntimeError(
            "DPE Lyon-proper coverage check failed: "
            f"{dvf_lyon_rows:,} Lyon DVF rows but only {dpe_lyon_rows:,} Lyon DPE rows. "
            "Check that 69381--69389 were included in the DPE API query."
        )
    print(f"Clean DPE rows: {len(dpe):,}")

    print("\n[4/4] Harmonising and auditing")
    harmonise_and_audit(dvf, dpe, dvf_audit, dpe_audit, args.output)
    print(f"\nComplete. Outputs written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
