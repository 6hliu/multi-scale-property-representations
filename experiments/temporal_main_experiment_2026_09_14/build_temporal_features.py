#!/usr/bin/env python3
"""Attach cached OSM and leakage-screened BDNB inputs to the temporal cohort."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


EARTH_RADIUS_M = 6_371_008.8
RADII = (250, 500, 1000)
POI_CATEGORIES = (
    "transport", "education", "healthcare", "retail", "food", "green_space",
    "landuse_residential", "landuse_commercial", "landuse_industrial",
)
NEAREST_CATEGORIES = (
    "transport", "education", "healthcare", "retail", "food", "green_space",
)
STRUCTURAL = (
    "bdnb_group_footprint_m2", "bdnb_component_count", "bdnb_height_mean_m",
    "bdnb_floor_count", "bdnb_building_age_years", "bdnb_dwelling_count",
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_week5(path: Path):
    spec = importlib.util.spec_from_file_location("week5_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nearest_distance(properties: np.ndarray, objects: np.ndarray) -> np.ndarray:
    tree = BallTree(np.radians(objects), metric="haversine")
    distance, _ = tree.query(np.radians(properties), k=1)
    return (distance[:, 0] * EARTH_RADIUS_M).astype("float32")


def entity_keys(data: pd.DataFrame) -> pd.Series:
    keys = pd.Series(pd.NA, index=data.index, dtype="string")
    accepted = data["building_features_accepted"]
    keys.loc[accepted] = "BDNB::" + data.loc[accepted, "accepted_building_group_id"].astype("string")
    dvf = data.source.eq("DVF") & keys.isna() & data.parcel_id.notna()
    keys.loc[dvf] = "PARCEL::" + data.loc[dvf, "parcel_id"].astype("string")
    rnb = data.source.eq("DPE") & keys.isna() & data.id_rnb.notna()
    keys.loc[rnb] = "RNB::" + data.loc[rnb, "id_rnb"].astype("string")
    ban = data.source.eq("DPE") & keys.isna() & data.identifiant_ban.notna()
    keys.loc[ban] = "BAN::" + data.loc[ban, "identifiant_ban"].astype("string")
    keys = keys.fillna(data.source.astype("string") + "::RECORD::" + data.record_id.astype("string"))
    return keys


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--common", type=Path, default=here / "data/temporal_common_core.csv.gz")
    parser.add_argument("--enrichment", type=Path, default=here / "data/bdnb/bdnb_building_enrichment.csv")
    parser.add_argument("--osm-cache", type=Path, default=root / "week5_output/raw_osm")
    parser.add_argument("--week5-script", type=Path, default=root / "week5_osm_feature_pipeline.py")
    parser.add_argument("--output", type=Path, default=here / "data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    week5 = load_week5(args.week5_script)
    data = pd.read_csv(args.common, low_memory=False, dtype={"record_id": "string"})
    enrichment = pd.read_csv(
        args.enrichment, low_memory=False,
        dtype={"record_id": "string", "bdnb_building_group_id": "string"},
    )
    keep = [
        "source", "record_id", "bdnb_building_group_id", "join_confidence",
        "evidence_conflict", "join_ambiguous", *STRUCTURAL,
    ]
    data = data.merge(enrichment[keep], on=["source", "record_id"], how="left", validate="one_to_one")
    accepted = (
        data.bdnb_building_group_id.notna()
        & data.join_confidence.isin(["HIGH", "MEDIUM"])
        & ~data.evidence_conflict.fillna(True).astype(bool)
        & ~data.join_ambiguous.fillna(True).astype(bool)
    )
    data["building_features_accepted"] = accepted
    data["accepted_building_group_id"] = data.bdnb_building_group_id.where(accepted)
    data.loc[~accepted, list(STRUCTURAL)] = np.nan

    coordinates = data[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    property_points = coordinates[["latitude", "longitude"]].to_numpy(float)
    object_audit = {}
    for category in POI_CATEGORIES:
        cache = args.osm_cache / f"{category}.json"
        objects = week5.load_poi_points(cache)
        counts = week5.radius_counts(property_points, objects)
        for radius in RADII:
            coordinates[f"{category}_count_{radius}m"] = counts[radius]
        object_audit[category] = {
            "objects": int(len(objects)), "cache": str(cache), "sha256": sha256(cache)
        }
        if category in NEAREST_CATEGORIES:
            coordinates[f"{category}_nearest_distance_m"] = nearest_distance(property_points, objects)

    road_cache = args.osm_cache / "selected_roads.json"
    road_objects = week5.load_road_junctions(road_cache)
    road_counts = week5.radius_counts(property_points, road_objects)
    for radius in RADII:
        coordinates[f"road_junction_count_{radius}m"] = road_counts[radius]
    object_audit["road_junction"] = {
        "objects": int(len(road_objects)), "cache": str(road_cache), "sha256": sha256(road_cache)
    }

    data = data.merge(coordinates, on=["latitude", "longitude"], how="left", validate="many_to_one")
    data["entity_id"] = entity_keys(data)
    data["entity_first_year"] = data.groupby("entity_id")["observation_year"].transform("min")
    data["unseen_entity_in_year"] = data.observation_year.eq(data.entity_first_year)
    data["entity_record_count"] = data.groupby("entity_id")["record_id"].transform("size")

    counts = sorted(column for column in data if "_count_" in column)
    nearest = [f"{category}_nearest_distance_m" for category in NEAREST_CATEGORIES]
    structural_with_missing = []
    for column in STRUCTURAL:
        structural_with_missing.extend([column, f"{column}_missing"])
        data[f"{column}_missing"] = data[column].isna().astype("int8")
    feature_names = [
        "property_type_is_house", "log_floor_area", "longitude", "latitude",
        *counts, *structural_with_missing, *nearest,
        "months_since_2021", "month_sin", "month_cos",
    ]
    if len(feature_names) != 55 or len(set(feature_names)) != 55:
        raise AssertionError((len(feature_names), feature_names))
    if any(target in feature_names for target in ["sale_price_eur", "log_sale_price", "energy_class", "etiquette_ges"]):
        raise AssertionError("Target leakage in feature manifest")
    context = counts + nearest
    if data[context].isna().any().any() or not np.isfinite(data[context].to_numpy()).all():
        raise AssertionError("OSM features incomplete")
    for prefix in sorted({column.rsplit("_", 1)[0] for column in counts}):
        columns = [f"{prefix}_{radius}m" for radius in RADII]
        if all(column in data for column in columns):
            if (np.diff(data[columns].to_numpy(), axis=1) < 0).any():
                raise AssertionError(f"Non-monotone counts: {prefix}")

    output = args.output / "temporal_model_ready_55.csv.gz"
    data.to_csv(output, index=False, date_format="%Y-%m-%d")
    coordinates.to_csv(args.output / "temporal_unique_coordinate_osm_features.csv.gz", index=False)

    entity_overlap = (
        data.groupby(["source", "temporal_split"])
        .agg(
            rows=("record_id", "size"), unique_entities=("entity_id", "nunique"),
            unseen_entity_rows=("unseen_entity_in_year", "sum"),
        ).reset_index()
    )
    entity_overlap.to_csv(args.output / "entity_overlap_summary.csv", index=False)
    coverage = (
        data.groupby(["source", "observation_year"])[list(STRUCTURAL)]
        .agg(lambda values: values.notna().mean()).reset_index()
    )
    coverage.to_csv(args.output / "structural_feature_coverage.csv", index=False)
    manifest = {
        "status": "PASS",
        "input": str(args.common), "input_sha256": sha256(args.common),
        "enrichment": str(args.enrichment), "enrichment_sha256": sha256(args.enrichment),
        "output": str(output), "output_sha256": sha256(output),
        "rows": int(len(data)), "unique_coordinates": int(len(coordinates)),
        "source_rows": {str(k): int(v) for k, v in data.source.value_counts().items()},
        "split_rows": {str(k): int(v) for k, v in data.temporal_split.value_counts().items()},
        "feature_count": len(feature_names), "feature_names": feature_names,
        "feature_groups": {
            "intrinsic": feature_names[0:2], "coordinates": feature_names[2:4],
            "osm_250m": [name for name in counts if "_250m" in name],
            "osm_500m": [name for name in counts if "_500m" in name],
            "osm_1000m": [name for name in counts if "_1000m" in name],
            "bdnb_values": list(STRUCTURAL),
            "bdnb_missing": [f"{name}_missing" for name in STRUCTURAL],
            "nearest_osm": nearest,
            "temporal": ["months_since_2021", "month_sin", "month_cos"],
        },
        "bdnb_acceptance": "Unique resolved HIGH/MEDIUM, no evidence conflict and no ambiguous match",
        "accepted_building_rate": {
            source: float(data.loc[data.source.eq(source), "building_features_accepted"].mean())
            for source in ["DVF", "DPE"]
        },
        "building_age_policy": "observation year minus construction year, computed by the enrichment pipeline",
        "distance_method": "Haversine BallTree; radius counts and nearest distances in metres",
        "osm_snapshot_caveat": "One cached later OSM snapshot is reused as approximately time-invariant retrospective context.",
        "bdnb_snapshot_caveat": "BDNB 2026-02-a is a later static snapshot and is not point-in-time historical data.",
        "osm_objects": object_audit,
        "target_leakage_check": True,
    }
    (args.output / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ["status", "rows", "unique_coordinates", "source_rows", "split_rows", "feature_count", "accepted_building_rate"]}, indent=2))


if __name__ == "__main__":
    main()
