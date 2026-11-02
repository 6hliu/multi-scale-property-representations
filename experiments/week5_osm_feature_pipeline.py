#!/usr/bin/env python3
"""Week 5: construct Lyon OSM context features and spatial data splits.

The script deliberately uses only pandas/numpy/scipy/sklearn plus the Python
standard library so it is easy to run on the student's current environment.
It downloads OSM objects once from Overpass, caches the JSON responses, counts
objects around unique property coordinates, joins features back to the common
schema, creates spatial-block splits, and writes an audit report.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


RADII_M = (250, 500, 1000)
EARTH_RADIUS_M = 6_371_008.8

# Deliberately broad enough to retain all valid Metro records plus a 1 km
# context margin, but narrow enough to reject the known invalid ADEME point.
LYON_BBOX = (45.545, 4.680, 45.952, 5.075)  # south, west, north, east

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

POI_QUERIES = {
    "transport": [
        'nwr["highway"="bus_stop"]',
        'nwr["public_transport"~"^(station|stop_position|platform)$"]',
        'nwr["railway"~"^(station|halt|tram_stop|subway_entrance)$"]',
    ],
    "education": [
        'nwr["amenity"~"^(school|kindergarten|college|university|childcare)$"]',
    ],
    "healthcare": [
        'nwr["amenity"~"^(hospital|clinic|doctors|pharmacy|dentist)$"]',
    ],
    "retail": [
        'nwr["shop"]',
        'nwr["amenity"="marketplace"]',
    ],
    "food": [
        'nwr["amenity"~"^(restaurant|cafe|fast_food|bar|pub)$"]',
    ],
    "green_space": [
        'nwr["leisure"~"^(park|garden|nature_reserve|recreation_ground)$"]',
        'nwr["landuse"~"^(grass|forest|meadow|recreation_ground)$"]',
    ],
    "landuse_residential": ['nwr["landuse"="residential"]'],
    "landuse_commercial": ['nwr["landuse"~"^(commercial|retail)$"]'],
    "landuse_industrial": ['nwr["landuse"="industrial"]'],
}

ROAD_QUERY = [
    'way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|living_street|unclassified)$"]'
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common", default="week4_output/lyon_common_schema.csv")
    parser.add_argument("--output", default="week5_output")
    parser.add_argument(
        "--osm-cache",
        default=None,
        help="Optional existing raw_osm directory. When supplied it must be complete.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-km", type=float, default=3.0)
    parser.add_argument("--skip-roads", action="store_true")
    return parser.parse_args()


def overpass_query(selectors: list[str], bbox: tuple[float, ...], *, geometry: bool = False) -> str:
    bbox_text = ",".join(str(x) for x in bbox)
    body = "\n".join(f"  {selector}({bbox_text});" for selector in selectors)
    output = "out body geom;" if geometry else "out center tags;"
    return f"[out:json][timeout:900];\n(\n{body}\n);\n{output}"


def download_overpass(query: str, target: Path, retries: int = 8) -> None:
    if target.exists() and target.stat().st_size > 100:
        print(f"Using cached {target}")
        return
    last_error = None
    for attempt in range(retries):
        endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        partial = target.with_suffix(target.suffix + ".part")
        try:
            print(f"Downloading {target.name} from {endpoint} (attempt {attempt + 1})")
            subprocess.run(
                [
                    "curl", "-L", "--fail", "--silent", "--show-error",
                    "--max-time", "900", "--get", "--data-urlencode",
                    f"data={query}", endpoint, "-o", str(partial),
                ],
                check=True,
            )
            content = partial.read_bytes()
            parsed = json.loads(content)
            if "elements" not in parsed:
                raise ValueError("Overpass response has no elements")
            partial.replace(target)
            print(f"Saved {target} ({len(content) / 1_000_000:.1f} MB)")
            time.sleep(3)
            return
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            print(f"Download attempt failed: {exc}")
            time.sleep(min(5 * (attempt + 1), 30))
    raise RuntimeError(f"Could not download {target.name}: {last_error}")


def split_bbox(bbox: tuple[float, ...], rows: int, columns: int) -> list[tuple[float, ...]]:
    south, west, north, east = bbox
    lat_edges = np.linspace(south, north, rows + 1)
    lon_edges = np.linspace(west, east, columns + 1)
    return [
        (
            float(lat_edges[row]),
            float(lon_edges[column]),
            float(lat_edges[row + 1]),
            float(lon_edges[column + 1]),
        )
        for row in range(rows)
        for column in range(columns)
    ]


def download_overpass_tiled(
    selectors: list[str],
    target: Path,
    *,
    geometry: bool = False,
    grid: int = 2,
) -> None:
    """Download small, resumable tiles and merge unique OSM elements."""
    if target.exists() and target.stat().st_size > 100:
        print(f"Using cached {target}")
        return
    tile_dir = target.parent / f"{target.stem}_tiles_{grid}x{grid}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_paths = []
    for tile_index, tile_bbox in enumerate(split_bbox(LYON_BBOX, grid, grid)):
        tile_path = tile_dir / f"tile_{tile_index:02d}.json"
        download_overpass(
            overpass_query(selectors, tile_bbox, geometry=geometry), tile_path
        )
        tile_paths.append(tile_path)

    unique_elements = {}
    metadata = None
    for tile_path in tile_paths:
        payload = json.loads(tile_path.read_text())
        if metadata is None:
            metadata = {key: value for key, value in payload.items() if key != "elements"}
        for element in payload["elements"]:
            unique_elements[(element.get("type"), element.get("id"))] = element
    merged = metadata or {"version": 0.6}
    merged["elements"] = list(unique_elements.values())
    target.write_text(json.dumps(merged, ensure_ascii=False))
    print(f"Merged {len(unique_elements):,} unique elements into {target}")


def element_point(element: dict) -> tuple[float, float] | None:
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    center = element.get("center")
    if center and "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    geometry = element.get("geometry")
    if geometry:
        return (
            float(np.mean([point["lat"] for point in geometry])),
            float(np.mean([point["lon"] for point in geometry])),
        )
    return None


def load_poi_points(path: Path) -> np.ndarray:
    elements = json.loads(path.read_text())["elements"]
    seen = set()
    points = []
    for element in elements:
        key = (element.get("type"), element.get("id"))
        if key in seen:
            continue
        seen.add(key)
        point = element_point(element)
        if point is not None:
            points.append(point)
    return np.asarray(points, dtype=float).reshape(-1, 2)


def load_road_junctions(path: Path) -> np.ndarray:
    """Return coordinates of OSM nodes shared by at least two selected ways."""
    elements = json.loads(path.read_text())["elements"]
    occurrence: Counter[int] = Counter()
    coordinates: dict[int, tuple[float, float]] = {}
    for way in elements:
        nodes = way.get("nodes", [])
        geometry = way.get("geometry", [])
        if len(nodes) != len(geometry):
            continue
        # Count each node at most once per way.
        for node_id, point in zip(dict.fromkeys(nodes), geometry):
            occurrence[node_id] += 1
            coordinates[node_id] = (float(point["lat"]), float(point["lon"]))
    return np.asarray(
        [coordinates[node_id] for node_id, count in occurrence.items() if count >= 2],
        dtype=float,
    ).reshape(-1, 2)


def project_to_meters(lat_lon: np.ndarray, reference_lat: float) -> np.ndarray:
    lat = np.radians(lat_lon[:, 0])
    lon = np.radians(lat_lon[:, 1])
    x = EARTH_RADIUS_M * lon * math.cos(math.radians(reference_lat))
    y = EARTH_RADIUS_M * lat
    return np.column_stack((x, y))


def radius_counts(property_lat_lon: np.ndarray, object_lat_lon: np.ndarray) -> dict[int, np.ndarray]:
    """Count OSM objects with exact spherical Haversine radius queries."""
    if len(object_lat_lon) == 0:
        return {radius: np.zeros(len(property_lat_lon), dtype=np.int32) for radius in RADII_M}
    property_radians = np.radians(property_lat_lon)
    object_radians = np.radians(object_lat_lon)
    tree = BallTree(object_radians, metric="haversine")
    return {
        radius: tree.query_radius(
            property_radians,
            r=radius / EARTH_RADIUS_M,
            count_only=True,
        ).astype(np.int32)
        for radius in RADII_M
    }


def assign_spatial_splits(data: pd.DataFrame, grid_km: float, seed: int) -> pd.Series:
    lat_lon = data[["latitude", "longitude"]].to_numpy(float)
    xy = project_to_meters(lat_lon, float(np.median(lat_lon[:, 0])))
    grid_m = grid_km * 1000
    grid_x = np.floor(xy[:, 0] / grid_m).astype(int)
    grid_y = np.floor(xy[:, 1] / grid_m).astype(int)
    groups = pd.Series(grid_x.astype(str) + "_" + grid_y.astype(str), index=data.index)

    # GroupShuffleSplit balances the number of blocks, not the number of rows.
    # Lyon's central blocks are much denser, so allocate whole blocks with a
    # deterministic weighted greedy objective targeting row and DVF counts.
    ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
    split_names = list(ratios)
    block_stats = pd.DataFrame({"block": groups, "is_dvf": data["source"].eq("DVF").astype(int)})
    block_stats = block_stats.groupby("block").agg(rows=("block", "size"), dvf=("is_dvf", "sum"))
    rng = np.random.default_rng(seed)
    block_stats["tie_breaker"] = rng.random(len(block_stats))
    block_stats = block_stats.sort_values(["rows", "tie_breaker"], ascending=[False, True])

    totals = {metric: float(block_stats[metric].sum()) for metric in ["rows", "dvf"]}
    targets = {
        split_name: {metric: ratios[split_name] * totals[metric] for metric in totals}
        for split_name in split_names
    }
    current = {split_name: {metric: 0.0 for metric in totals} for split_name in split_names}
    allocation = {}
    metric_weights = {"rows": 1.0, "dvf": 0.5}

    for block_id, row in block_stats.iterrows():
        candidate_scores = {}
        for candidate in split_names:
            score = 0.0
            for split_name in split_names:
                for metric, weight in metric_weights.items():
                    value = current[split_name][metric]
                    if split_name == candidate:
                        value += float(row[metric])
                    target = max(targets[split_name][metric], 1.0)
                    score += weight * ((value - target) / target) ** 2
            candidate_scores[candidate] = score
        chosen = min(candidate_scores, key=candidate_scores.get)
        allocation[block_id] = chosen
        for metric in totals:
            current[chosen][metric] += float(row[metric])

    split = groups.map(allocation).rename("split")
    return split, groups.rename("spatial_block_id")


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    raw_dir = Path(args.osm_cache) if args.osm_cache else output / "raw_osm"
    distributions = output / "distributions"
    if args.osm_cache:
        required_cache_files = [f"{category}.json" for category in POI_QUERIES]
        if not args.skip_roads:
            required_cache_files.append("selected_roads.json")
        missing = [name for name in required_cache_files if not (raw_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                "The supplied OSM cache is incomplete: " + ", ".join(missing)
            )
    else:
        raw_dir.mkdir(parents=True, exist_ok=True)
    distributions.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.common)
    initial_rows = len(data)
    south, west, north, east = LYON_BBOX
    spatially_valid = data["latitude"].between(south, north) & data["longitude"].between(west, east)
    excluded = data.loc[~spatially_valid].copy()
    excluded.to_csv(output / "excluded_invalid_spatial_records.csv", index=False)
    data = data.loc[spatially_valid].copy().reset_index(drop=True)

    coord_cols = ["latitude", "longitude"]
    unique_coordinates = data[coord_cols].drop_duplicates().reset_index(drop=True)
    property_lat_lon = unique_coordinates[coord_cols].to_numpy(float)

    object_counts = {}
    feature_table = unique_coordinates.copy()
    for category, selectors in POI_QUERIES.items():
        raw_path = raw_dir / f"{category}.json"
        grid = 4 if category in {"retail", "green_space"} else 2
        download_overpass_tiled(selectors, raw_path, grid=grid)
        object_points = load_poi_points(raw_path)
        object_counts[category] = int(len(object_points))
        counts = radius_counts(property_lat_lon, object_points)
        for radius, values in counts.items():
            feature_table[f"{category}_count_{radius}m"] = values

    if not args.skip_roads:
        road_path = raw_dir / "selected_roads.json"
        download_overpass_tiled(ROAD_QUERY, road_path, geometry=True, grid=4)
        junction_points = load_road_junctions(road_path)
        object_counts["road_junction"] = int(len(junction_points))
        counts = radius_counts(property_lat_lon, junction_points)
        for radius, values in counts.items():
            feature_table[f"road_junction_count_{radius}m"] = values

    model_data = data.merge(feature_table, on=coord_cols, how="left", validate="many_to_one")
    split, blocks = assign_spatial_splits(model_data, args.grid_km, args.seed)
    model_data["split"] = split
    model_data["spatial_block_id"] = blocks

    feature_columns = [column for column in model_data if "_count_" in column]
    missing_feature_values = int(model_data[feature_columns].isna().sum().sum())
    negative_feature_values = int((model_data[feature_columns] < 0).sum().sum())
    monotonic_violations = {}
    for prefix in sorted({column.rsplit("_", 1)[0] for column in feature_columns}):
        columns = [f"{prefix}_{radius}m" for radius in RADII_M]
        if all(column in model_data for column in columns):
            violation = (model_data[columns[0]] > model_data[columns[1]]) | (
                model_data[columns[1]] > model_data[columns[2]]
            )
            monotonic_violations[prefix] = int(violation.sum())

    split_summary = (
        model_data.groupby(["split", "source"]).size().rename("rows").reset_index()
    )
    split_summary.to_csv(distributions / "split_source_counts.csv", index=False)
    model_data.groupby(["split", "property_type"]).size().rename("rows").reset_index().to_csv(
        distributions / "split_property_type_counts.csv", index=False
    )
    model_data.loc[model_data["source"].eq("DPE")].groupby(
        ["split", "energy_class"]
    ).size().rename("rows").reset_index().to_csv(
        distributions / "split_energy_class_counts.csv", index=False
    )
    model_data[feature_columns].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).T.to_csv(
        distributions / "osm_feature_summary.csv"
    )
    feature_table.to_csv(output / "unique_coordinate_osm_features.csv", index=False)
    model_data.to_csv(output / "lyon_model_ready_with_osm_and_splits.csv", index=False)

    block_sets = {
        split_name: set(group["spatial_block_id"])
        for split_name, group in model_data.groupby("split")
    }
    block_overlap = {
        "train_validation": len(block_sets["train"] & block_sets["validation"]),
        "train_test": len(block_sets["train"] & block_sets["test"]),
        "validation_test": len(block_sets["validation"] & block_sets["test"]),
    }
    coordinate_split_violations = int(
        model_data.groupby(["longitude", "latitude"])["split"].nunique().gt(1).sum()
    )
    split_rows = model_data["split"].value_counts().to_dict()

    audit = {
        "osm_cache_dir": str(raw_dir.resolve()),
        "input_rows": initial_rows,
        "excluded_spatial_rows": int(len(excluded)),
        "model_ready_rows": int(len(model_data)),
        "unique_property_coordinates": int(len(unique_coordinates)),
        "lyon_bbox_south_west_north_east": list(LYON_BBOX),
        "radii_m": list(RADII_M),
        "distance_method": {
            "name": "Haversine BallTree",
            "coordinate_input": "latitude/longitude converted from degrees to radians",
            "balltree_metric": "haversine",
            "angular_query_radius": "radius_metres / 6371008.8",
        },
        "osm_object_counts": object_counts,
        "osm_feature_columns": feature_columns,
        "missing_osm_feature_values": missing_feature_values,
        "negative_osm_feature_values": negative_feature_values,
        "monotonic_violations": monotonic_violations,
        "osm_feature_zero_fraction": {
            column: round(float(model_data[column].eq(0).mean()), 6)
            for column in feature_columns
        },
        "spatial_split": {
            "method": f"{args.grid_km:g} km spatial blocks with weighted 70/15/15 whole-block allocation",
            "grid_km": args.grid_km,
            "seed": args.seed,
            "unique_blocks": int(model_data["spatial_block_id"].nunique()),
            "rows": split_rows,
            "row_fraction": {
                split_name: round(row_count / len(model_data), 6)
                for split_name, row_count in split_rows.items()
            },
            "blocks": model_data.groupby("split")["spatial_block_id"].nunique().to_dict(),
            "block_overlap": block_overlap,
            "coordinates_assigned_to_multiple_splits": coordinate_split_violations,
            "source_rows": {
                split_name: {
                    source: int(count)
                    for source, count in group["source"].value_counts().items()
                }
                for split_name, group in model_data.groupby("split")
            },
            "dpe_energy_class_rows": {
                split_name: {
                    energy_class: int(count)
                    for energy_class, count in group.loc[
                        group["source"].eq("DPE"), "energy_class"
                    ].value_counts().sort_index().items()
                }
                for split_name, group in model_data.groupby("split")
            },
        },
    }
    (output / "audit_summary.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
