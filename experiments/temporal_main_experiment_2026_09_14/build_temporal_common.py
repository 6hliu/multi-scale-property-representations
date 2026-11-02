#!/usr/bin/env python3
"""Construct the 2021--2025 partially labelled temporal common table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


LYON_BBOX = (45.545, 4.680, 45.952, 5.075)
CORE = [
    "record_id", "source", "property_type", "floor_area_m2", "longitude",
    "latitude", "commune_code", "postcode", "observation_date",
    "sale_price_eur", "log_sale_price", "energy_class", "price_mask",
    "energy_mask", "room_count", "parcel_id", "identifiant_ban", "id_rnb",
    "annee_construction", "periode_construction", "source_commune_code",
]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def split_for_year(year: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [year.le(2023), year.eq(2024), year.eq(2025)],
            ["train", "validation", "test"],
            default="excluded",
        ),
        index=year.index,
    )


def add_missing_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in CORE:
        if column not in result:
            result[column] = np.nan
    return result[CORE]


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dvf",
        type=Path,
        default=root / "experiments_2026_09_10/temporal_dvf/data/lyon_dvf_2021_2025_clean.csv",
    )
    parser.add_argument(
        "--dpe", type=Path, default=root / "week4_output_corrected/lyon_dpe_clean.csv"
    )
    parser.add_argument("--output", type=Path, default=here / "data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dvf = pd.read_csv(args.dvf, low_memory=False, dtype={"record_id": "string"})
    dpe = pd.read_csv(args.dpe, low_memory=False, dtype={"record_id": "string"})
    dvf["observation_date"] = pd.to_datetime(dvf["observation_date"], errors="coerce")
    dpe["observation_date"] = pd.to_datetime(dpe["observation_date"], errors="coerce")

    south, west, north, east = LYON_BBOX
    dpe_spatial = dpe.latitude.between(south, north) & dpe.longitude.between(west, east)
    dpe_year = dpe.observation_date.dt.year.between(2021, 2025)
    dpe = dpe.loc[dpe_spatial & dpe_year].copy()
    dvf = dvf.loc[dvf.observation_date.dt.year.between(2021, 2025)].copy()

    dvf = add_missing_columns(dvf)
    dpe = add_missing_columns(dpe)
    data = pd.concat([dvf, dpe], ignore_index=True, sort=False)
    year = data.observation_date.dt.year
    data["temporal_split"] = split_for_year(year)
    data["observation_year"] = year.astype("int16")
    elapsed_months = (
        (year - 2021) * 12 + data.observation_date.dt.month - 1
        + (data.observation_date.dt.day - 1) / data.observation_date.dt.days_in_month
    )
    angle = 2 * np.pi * (data.observation_date.dt.month - 1) / 12
    data["months_since_2021"] = elapsed_months.astype("float32")
    data["month_sin"] = np.sin(angle).astype("float32")
    data["month_cos"] = np.cos(angle).astype("float32")

    if data.duplicated(["source", "record_id"]).any():
        raise AssertionError("source + record_id must be unique")
    if set(data.temporal_split) != {"train", "validation", "test"}:
        raise AssertionError(data.temporal_split.value_counts().to_dict())
    if not data.loc[data.source.eq("DVF"), "price_mask"].eq(1).all():
        raise AssertionError("DVF price mask failure")
    if not data.loc[data.source.eq("DPE"), "energy_mask"].eq(1).all():
        raise AssertionError("DPE energy mask failure")

    output = args.output / "temporal_common_core.csv.gz"
    data.to_csv(output, index=False, date_format="%Y-%m-%d")

    counts = (
        data.groupby(["source", "observation_year", "temporal_split"])
        .size().rename("rows").reset_index()
    )
    counts.to_csv(args.output / "source_year_counts.csv", index=False)
    energy_counts = (
        data.loc[data.source.eq("DPE")]
        .groupby(["observation_year", "energy_class"])
        .size().rename("rows").reset_index()
    )
    energy_counts.to_csv(args.output / "energy_year_class_counts.csv", index=False)
    audit = {
        "status": "PASS",
        "inputs": {
            "dvf": str(args.dvf), "dvf_sha256": sha256(args.dvf),
            "dpe": str(args.dpe), "dpe_sha256": sha256(args.dpe),
        },
        "output": str(output),
        "rows": int(len(data)),
        "source_rows": {str(k): int(v) for k, v in data.source.value_counts().items()},
        "split_rows": {str(k): int(v) for k, v in data.temporal_split.value_counts().items()},
        "year_rows": {str(k): int(v) for k, v in data.observation_year.value_counts().sort_index().items()},
        "date_range": [str(data.observation_date.min().date()), str(data.observation_date.max().date())],
        "duplicate_source_record_keys": int(data.duplicated(["source", "record_id"]).sum()),
        "dpe_2026_excluded": int((pd.to_datetime(pd.read_csv(args.dpe, usecols=["observation_date"])["observation_date"]).dt.year == 2026).sum()),
        "temporal_features": ["months_since_2021", "month_sin", "month_cos"],
        "lyon_bbox": list(LYON_BBOX),
        "notes": [
            "DVF cleaning is inherited unchanged from the corrected Week 4 pipeline.",
            "DPE is restricted to the same broad Lyon spatial window and calendar years 2021-2025.",
            "No DVF-DPE row-level match is required; labels remain source-masked.",
        ],
    }
    (args.output / "common_core_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: audit[k] for k in ["status", "rows", "source_rows", "split_rows", "year_rows", "date_range"]}, indent=2))


if __name__ == "__main__":
    main()
