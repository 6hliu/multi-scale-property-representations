#!/usr/bin/env python3
"""Attach leakage-safe BDNB building attributes to clean Lyon DVF and DPE rows.

The output is an enrichment table keyed by (source, record_id).  It does not
replace any canonical Week 4--8 file.  Building resolution deliberately keeps
all candidate building-group IDs and leaves structural attributes missing when
the available evidence remains ambiguous.

Matching routes
---------------
DPE:
  1. ADEME DPE ID -> rel_batiment_groupe_dpe_logement
  2. RNB ID -> rel_batiment_construction_rnb -> batiment_construction
  3. BAN interoperable address key -> rel_batiment_groupe_adresse

DVF:
  cadastral parcel ID -> rel_batiment_groupe_parcelle

Only non-target structural tables are read.  In particular, no BDNB DPE
label/statistic/simulation and no BDNB DVF price/statistic table is loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_BDNB = DEFAULT_PROJECT / "data_private/bdnb/csv"

STRUCTURAL_COLUMNS = [
    "bdnb_group_footprint_m2",
    "bdnb_component_count",
    "bdnb_component_footprint_sum_m2",
    "bdnb_height_mean_m",
    "bdnb_height_max_m",
    "bdnb_altitude_mean_m",
    "bdnb_floor_count",
    "bdnb_construction_year",
    "bdnb_building_age_years",
    "bdnb_dwelling_count",
    "bdnb_usage",
    "bdnb_wall_material",
    "bdnb_roof_material",
    "bdnb_iris_code",
    "bdnb_commune_code",
    "bdnb_group_geometry_fictive",
    "bdnb_component_geometry_fictive_share",
    "bdnb_component_height_fictive_share",
]


def clean_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return text


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def read_bdnb_filtered(
    path: Path,
    key: str,
    values: set[str],
    usecols: list[str],
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Read a semicolon-delimited BDNB table while retaining selected keys."""
    if not values:
        return pd.DataFrame(columns=usecols)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        sep=";",
        usecols=usecols,
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    ):
        normalized = chunk[key].astype("string").str.strip()
        keep = normalized.isin(values)
        if keep.any():
            selected = chunk.loc[keep].copy()
            selected[key] = normalized.loc[keep]
            chunks.append(selected)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def values_by_key(frame: pd.DataFrame, key: str, value: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    if frame.empty:
        return result
    for left, right in frame[[key, value]].itertuples(index=False, name=None):
        left_id = clean_id(left)
        right_id = clean_id(right)
        if left_id and right_id:
            result[left_id].add(right_id)
    return result


def strings_by_key(frame: pd.DataFrame, key: str, value: str) -> dict[str, set[str]]:
    return values_by_key(frame, key, value)


def numeric_max_by_key(frame: pd.DataFrame, key: str, value: str) -> dict[str, float]:
    if frame.empty:
        return {}
    work = frame[[key, value]].copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    return work.groupby(key, dropna=True)[value].max().to_dict()


def parse_match_inputs(dvf_path: Path, dpe_path: Path, sample_size: int, seed: int):
    dvf = pd.read_csv(dvf_path, low_memory=False)
    dpe = pd.read_csv(dpe_path, low_memory=False)
    require_columns(dvf, ["record_id", "source", "parcel_id", "observation_date"], "DVF")
    require_columns(
        dpe,
        ["record_id", "source", "identifiant_ban", "id_rnb", "observation_date"],
        "DPE",
    )
    if sample_size > 0:
        dvf = dvf.sample(n=min(sample_size, len(dvf)), random_state=seed).copy()
        dpe = dpe.sample(n=min(sample_size, len(dpe)), random_state=seed).copy()
    return dvf.reset_index(drop=True), dpe.reset_index(drop=True)


def load_dpe_evidence(dpe: pd.DataFrame, bdnb: Path) -> dict[str, object]:
    dpe_ids = {x for x in map(clean_id, dpe["record_id"]) if x}
    rnb_ids = {x for x in map(clean_id, dpe["id_rnb"]) if x}
    ban_ids = {x for x in map(clean_id, dpe["identifiant_ban"]) if x}

    direct = read_bdnb_filtered(
        bdnb / "rel_batiment_groupe_dpe_logement.csv",
        "identifiant_dpe",
        dpe_ids,
        [
            "batiment_groupe_id",
            "identifiant_dpe",
            "cle_interop_adr",
            "fiabilite_geocodage",
            "type_appariement",
        ],
    )
    direct_groups = values_by_key(direct, "identifiant_dpe", "batiment_groupe_id")
    direct_types = strings_by_key(direct, "identifiant_dpe", "type_appariement")
    direct_reliability = strings_by_key(direct, "identifiant_dpe", "fiabilite_geocodage")

    rnb_rel = read_bdnb_filtered(
        bdnb / "rel_batiment_construction_rnb.csv",
        "rnb_id",
        rnb_ids,
        ["rnb_id", "batiment_construction_id", "type_appariement"],
    )
    construction_ids = {
        x for x in map(clean_id, rnb_rel.get("batiment_construction_id", [])) if x
    }
    rnb_construction = read_bdnb_filtered(
        bdnb / "batiment_construction.csv",
        "batiment_construction_id",
        construction_ids,
        ["batiment_construction_id", "batiment_groupe_id"],
    )
    rnb_join = rnb_rel.merge(rnb_construction, on="batiment_construction_id", how="left")
    rnb_groups = values_by_key(rnb_join, "rnb_id", "batiment_groupe_id")
    rnb_types = strings_by_key(rnb_join, "rnb_id", "type_appariement")

    ban_rel = read_bdnb_filtered(
        bdnb / "rel_batiment_groupe_adresse.csv",
        "cle_interop_adr",
        ban_ids,
        ["batiment_groupe_id", "cle_interop_adr", "origine", "fiabilite"],
    )
    ban_groups = values_by_key(ban_rel, "cle_interop_adr", "batiment_groupe_id")
    ban_reliability = numeric_max_by_key(ban_rel, "cle_interop_adr", "fiabilite")

    return {
        "direct_groups": direct_groups,
        "direct_types": direct_types,
        "direct_reliability": direct_reliability,
        "rnb_groups": rnb_groups,
        "rnb_types": rnb_types,
        "ban_groups": ban_groups,
        "ban_reliability": ban_reliability,
        "raw_counts": {
            "direct_relation_rows": int(len(direct)),
            "rnb_relation_rows": int(len(rnb_rel)),
            "ban_relation_rows": int(len(ban_rel)),
        },
    }


def resolve_dpe(dpe: pd.DataFrame, evidence: dict[str, object]) -> pd.DataFrame:
    direct_groups = evidence["direct_groups"]
    direct_types = evidence["direct_types"]
    direct_reliability = evidence["direct_reliability"]
    rnb_groups = evidence["rnb_groups"]
    rnb_types = evidence["rnb_types"]
    ban_groups = evidence["ban_groups"]
    ban_reliability = evidence["ban_reliability"]

    rows = []
    for row in dpe.itertuples(index=False):
        record_id = clean_id(row.record_id)
        rnb_id = clean_id(row.id_rnb)
        ban_id = clean_id(row.identifiant_ban)
        direct = set(direct_groups.get(record_id, set()))
        rnb = set(rnb_groups.get(rnb_id, set())) if rnb_id else set()
        ban = set(ban_groups.get(ban_id, set())) if ban_id else set()
        methods = [name for name, group_set in (("DPE_REL", direct), ("RNB", rnb), ("BAN", ban)) if group_set]
        candidates = direct | rnb | ban
        selected = None
        confidence = "NONE"
        conflict = False

        if len(direct) == 1:
            proposed = next(iter(direct))
            conflicting_sets = [s for s in (rnb, ban) if s and proposed not in s]
            conflict = bool(conflicting_sets)
            if not conflict:
                selected = proposed
                direct_type_text = " ".join(direct_types.get(record_id, set())).lower()
                independent_support = (rnb and proposed in rnb) or (ban and proposed in ban)
                confidence = "HIGH" if independent_support or "+" in direct_type_text else "MEDIUM"
        elif len(direct) > 1:
            scores = Counter()
            for group in direct:
                scores[group] += 1
            for group in rnb:
                scores[group] += 1
            for group in ban:
                scores[group] += 1
            if scores:
                top_score = max(scores.values())
                leaders = [group for group, score in scores.items() if score == top_score]
                if top_score >= 2 and len(leaders) == 1:
                    selected = leaders[0]
                    confidence = "HIGH" if top_score == 3 else "MEDIUM"
                else:
                    conflict = bool((rnb or ban) and len(candidates) > 1)
        elif rnb and ban:
            overlap = rnb & ban
            if len(overlap) == 1:
                selected = next(iter(overlap))
                confidence = "HIGH"
            else:
                conflict = not bool(overlap)
        elif len(rnb) == 1:
            selected = next(iter(rnb))
            rnb_type_text = " ".join(rnb_types.get(rnb_id, set())).lower()
            confidence = "HIGH" if "1 bc = 1 rnb" in rnb_type_text else "MEDIUM"
        elif len(ban) == 1:
            selected = next(iter(ban))
            reliability = ban_reliability.get(ban_id, math.nan)
            confidence = "MEDIUM" if pd.notna(reliability) and reliability >= 12 else "LOW"

        ambiguous = bool(candidates) and selected is None
        rows.append(
            {
                "source": "DPE",
                "record_id": record_id,
                "observation_date": row.observation_date,
                "bdnb_building_group_id": selected,
                "join_method": "+".join(methods) if methods else "UNMATCHED",
                "join_confidence": confidence,
                "join_ambiguous": int(ambiguous),
                "evidence_conflict": int(conflict),
                "candidate_group_count": len(candidates),
                "candidate_building_group_ids": "|".join(sorted(candidates)),
                "dpe_direct_group_count": len(direct),
                "rnb_group_count": len(rnb),
                "ban_group_count": len(ban),
                "bdnb_dpe_match_types": "|".join(sorted(direct_types.get(record_id, set()))),
                "bdnb_dpe_geocode_reliability": "|".join(
                    sorted(direct_reliability.get(record_id, set()))
                ),
                "bdnb_ban_reliability_score": ban_reliability.get(ban_id, np.nan),
            }
        )
    return pd.DataFrame(rows)


def load_and_resolve_dvf(dvf: pd.DataFrame, bdnb: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    parcels = {x for x in map(clean_id, dvf["parcel_id"]) if x}
    relation = read_bdnb_filtered(
        bdnb / "rel_batiment_groupe_parcelle.csv",
        "parcelle_id",
        parcels,
        ["batiment_groupe_id", "parcelle_id", "parcelle_principale"],
    )
    all_groups = values_by_key(relation, "parcelle_id", "batiment_groupe_id")
    primary_relation = relation.loc[
        pd.to_numeric(relation["parcelle_principale"], errors="coerce").fillna(0).eq(1)
    ]
    primary_groups = values_by_key(primary_relation, "parcelle_id", "batiment_groupe_id")

    rows = []
    for row in dvf.itertuples(index=False):
        record_id = clean_id(row.record_id)
        parcel_id = clean_id(row.parcel_id)
        candidates = set(all_groups.get(parcel_id, set())) if parcel_id else set()
        primary = set(primary_groups.get(parcel_id, set())) if parcel_id else set()
        selected = next(iter(candidates)) if len(candidates) == 1 else None
        rows.append(
            {
                "source": "DVF",
                "record_id": record_id,
                "observation_date": row.observation_date,
                "bdnb_building_group_id": selected,
                "join_method": (
                    "PARCEL_UNIQUE"
                    if selected
                    else "PARCEL_AMBIGUOUS"
                    if candidates
                    else "UNMATCHED"
                ),
                "join_confidence": "MEDIUM" if selected else "NONE",
                "join_ambiguous": int(bool(candidates) and selected is None),
                "evidence_conflict": 0,
                "candidate_group_count": len(candidates),
                "candidate_building_group_ids": "|".join(sorted(candidates)),
                "dpe_direct_group_count": 0,
                "rnb_group_count": 0,
                "ban_group_count": 0,
                "bdnb_dpe_match_types": "",
                "bdnb_dpe_geocode_reliability": "",
                "bdnb_ban_reliability_score": np.nan,
                "parcel_primary_candidate_count": len(primary),
            }
        )
    return pd.DataFrame(rows), {"parcel_relation_rows": int(len(relation))}


def load_structural_features(groups: set[str], bdnb: Path) -> pd.DataFrame:
    if not groups:
        return pd.DataFrame(columns=["bdnb_building_group_id"] + STRUCTURAL_COLUMNS)

    group = read_bdnb_filtered(
        bdnb / "batiment_groupe.csv",
        "batiment_groupe_id",
        groups,
        [
            "batiment_groupe_id",
            "s_geom_groupe",
            "code_iris",
            "code_commune_insee",
            "contient_fictive_geom_groupe",
        ],
    ).rename(
        columns={
            "batiment_groupe_id": "bdnb_building_group_id",
            "s_geom_groupe": "bdnb_group_footprint_m2",
            "code_iris": "bdnb_iris_code",
            "code_commune_insee": "bdnb_commune_code",
            "contient_fictive_geom_groupe": "bdnb_group_geometry_fictive",
        }
    )

    bdtopo = read_bdnb_filtered(
        bdnb / "batiment_groupe_bdtopo_bat.csv",
        "batiment_groupe_id",
        groups,
        [
            "batiment_groupe_id",
            "hauteur_mean",
            "max_hauteur",
            "altitude_sol_mean",
        ],
    ).rename(
        columns={
            "batiment_groupe_id": "bdnb_building_group_id",
            "hauteur_mean": "bdnb_height_mean_m",
            "max_hauteur": "bdnb_height_max_m",
            "altitude_sol_mean": "bdnb_altitude_mean_m",
        }
    )

    ffo = read_bdnb_filtered(
        bdnb / "batiment_groupe_ffo_bat.csv",
        "batiment_groupe_id",
        groups,
        [
            "batiment_groupe_id",
            "nb_niveau",
            "annee_construction",
            "usage_niveau_1_txt",
            "mat_mur_txt",
            "mat_toit_txt",
            "nb_log",
        ],
    ).rename(
        columns={
            "batiment_groupe_id": "bdnb_building_group_id",
            "nb_niveau": "bdnb_floor_count",
            "annee_construction": "bdnb_construction_year",
            "usage_niveau_1_txt": "bdnb_usage_ffo",
            "mat_mur_txt": "bdnb_wall_material",
            "mat_toit_txt": "bdnb_roof_material",
            "nb_log": "bdnb_dwelling_count",
        }
    )

    usage = read_bdnb_filtered(
        bdnb / "batiment_groupe_synthese_propriete_usage.csv",
        "batiment_groupe_id",
        groups,
        ["batiment_groupe_id", "usage_principal_bdnb_open"],
    ).rename(
        columns={
            "batiment_groupe_id": "bdnb_building_group_id",
            "usage_principal_bdnb_open": "bdnb_usage",
        }
    )

    components = read_bdnb_filtered(
        bdnb / "batiment_construction.csv",
        "batiment_groupe_id",
        groups,
        [
            "batiment_groupe_id",
            "batiment_construction_id",
            "s_geom_cstr",
            "hauteur",
            "altitude_sol",
            "fictive_hauteur_cstr",
            "fictive_geom_cstr",
        ],
    )
    if components.empty:
        component_features = pd.DataFrame(columns=["bdnb_building_group_id"])
    else:
        for column in [
            "s_geom_cstr",
            "hauteur",
            "altitude_sol",
            "fictive_hauteur_cstr",
            "fictive_geom_cstr",
        ]:
            components[column] = pd.to_numeric(components[column], errors="coerce")
        component_features = (
            components.groupby("batiment_groupe_id", dropna=True)
            .agg(
                bdnb_component_count=("batiment_construction_id", "nunique"),
                bdnb_component_footprint_sum_m2=("s_geom_cstr", "sum"),
                bdnb_component_height_mean_fallback_m=("hauteur", "mean"),
                bdnb_component_height_max_fallback_m=("hauteur", "max"),
                bdnb_component_altitude_mean_fallback_m=("altitude_sol", "mean"),
                bdnb_component_geometry_fictive_share=("fictive_geom_cstr", "mean"),
                bdnb_component_height_fictive_share=("fictive_hauteur_cstr", "mean"),
            )
            .reset_index()
            .rename(columns={"batiment_groupe_id": "bdnb_building_group_id"})
        )

    features = group
    for other in (bdtopo, ffo, usage, component_features):
        features = features.merge(other, on="bdnb_building_group_id", how="outer")

    features["bdnb_usage"] = features.get("bdnb_usage").fillna(features.get("bdnb_usage_ffo"))
    features["bdnb_height_mean_m"] = pd.to_numeric(
        features.get("bdnb_height_mean_m"), errors="coerce"
    ).fillna(pd.to_numeric(features.get("bdnb_component_height_mean_fallback_m"), errors="coerce"))
    features["bdnb_height_max_m"] = pd.to_numeric(
        features.get("bdnb_height_max_m"), errors="coerce"
    ).fillna(pd.to_numeric(features.get("bdnb_component_height_max_fallback_m"), errors="coerce"))
    features["bdnb_altitude_mean_m"] = pd.to_numeric(
        features.get("bdnb_altitude_mean_m"), errors="coerce"
    ).fillna(pd.to_numeric(features.get("bdnb_component_altitude_mean_fallback_m"), errors="coerce"))

    numeric = [
        "bdnb_group_footprint_m2",
        "bdnb_component_count",
        "bdnb_component_footprint_sum_m2",
        "bdnb_height_mean_m",
        "bdnb_height_max_m",
        "bdnb_altitude_mean_m",
        "bdnb_floor_count",
        "bdnb_construction_year",
        "bdnb_dwelling_count",
        "bdnb_group_geometry_fictive",
        "bdnb_component_geometry_fictive_share",
        "bdnb_component_height_fictive_share",
    ]
    for column in numeric:
        if column in features:
            features[column] = pd.to_numeric(features[column], errors="coerce")

    drop_helpers = [
        "bdnb_usage_ffo",
        "bdnb_component_height_mean_fallback_m",
        "bdnb_component_height_max_fallback_m",
        "bdnb_component_altitude_mean_fallback_m",
    ]
    return features.drop(columns=[c for c in drop_helpers if c in features.columns])


def attach_features(matches: pd.DataFrame, structural: pd.DataFrame) -> pd.DataFrame:
    enriched = matches.merge(structural, on="bdnb_building_group_id", how="left")
    observed_year = pd.to_datetime(enriched["observation_date"], errors="coerce").dt.year
    construction_year = pd.to_numeric(enriched.get("bdnb_construction_year"), errors="coerce")
    valid = construction_year.between(1700, observed_year, inclusive="both")
    enriched["bdnb_building_age_years"] = (observed_year - construction_year).where(valid)
    return enriched


def source_audit(frame: pd.DataFrame) -> dict[str, object]:
    total = len(frame)
    resolved = frame["bdnb_building_group_id"].notna()
    any_candidate = frame["candidate_group_count"].gt(0)
    audit = {
        "records": int(total),
        "with_any_candidate": int(any_candidate.sum()),
        "candidate_rate": float(any_candidate.mean()) if total else None,
        "resolved_records": int(resolved.sum()),
        "resolved_rate": float(resolved.mean()) if total else None,
        "ambiguous_records": int(frame["join_ambiguous"].sum()),
        "ambiguous_rate": float(frame["join_ambiguous"].mean()) if total else None,
        "evidence_conflicts": int(frame["evidence_conflict"].sum()),
        "unique_selected_buildings": int(frame.loc[resolved, "bdnb_building_group_id"].nunique()),
        "confidence": {str(k): int(v) for k, v in frame["join_confidence"].value_counts().items()},
        "join_method": {str(k): int(v) for k, v in frame["join_method"].value_counts().items()},
        "structural_completeness_among_resolved": {},
    }
    for column in STRUCTURAL_COLUMNS:
        if column in frame:
            denominator = int(resolved.sum())
            rate = frame.loc[resolved, column].notna().mean() if denominator else np.nan
            audit["structural_completeness_among_resolved"][column] = (
                None if pd.isna(rate) else float(rate)
            )
    return audit


def markdown_report(audit: dict[str, object], output_csv: Path) -> str:
    lines = [
        "# BDNB building enrichment audit",
        "",
        f"Generated: {audit['generated_at']}",
        f"Mode: {audit['mode']}",
        f"BDNB: {audit['bdnb_release']['millesime']} / {audit['bdnb_release']['version']}",
        "",
        "## Outcome",
        "",
        "| Source | Records | Any candidate | Resolved unique building | Ambiguous |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in ["DVF", "DPE"]:
        s = audit["sources"][source]
        lines.append(
            f"| {source} | {s['records']:,} | {s['with_any_candidate']:,} "
            f"({s['candidate_rate']:.1%}) | {s['resolved_records']:,} "
            f"({s['resolved_rate']:.1%}) | {s['ambiguous_records']:,} "
            f"({s['ambiguous_rate']:.1%}) |"
        )
    lines.extend(
        [
            "",
            "`Resolved` means one defensible BDNB building-group ID was selected. "
            "Ambiguous rows retain all candidate IDs but receive no structural attributes.",
            "",
            "## Matching logic",
            "",
            "- DPE combines the official BDNB DPE relation, RNB relation and BAN address relation.",
            "- DVF uses its 14-character cadastral parcel ID. A building is selected only when "
            "the parcel maps to exactly one BDNB building group.",
            "- Every candidate list, evidence conflict and confidence label is retained.",
            "",
            "## Leakage controls",
            "",
            "Only building geometry, height, floor count, construction year, dwelling count, "
            "usage and material fields are attached. No BDNB DPE outcome/aggregate/simulation, "
            "DVF price statistic, energy-consumption statistic or predicted value is read.",
            "",
            "## Output",
            "",
            f"- Enrichment table: `{output_csv.name}`",
            "- Key: `(source, record_id)`",
            "- Merge the table back to canonical data only after inspecting this audit.",
            "",
            "## Important limitation",
            "",
            "The resolved entity is a building group, not necessarily an individual apartment. "
            "All apartments in one building can therefore share these structural attributes.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvf", type=Path, default=DEFAULT_PROJECT / "week4_output/lyon_dvf_clean.csv")
    parser.add_argument("--dpe", type=Path, default=DEFAULT_PROJECT / "week4_output/lyon_dpe_clean.csv")
    parser.add_argument("--bdnb-dir", type=Path, default=DEFAULT_BDNB)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000, help="Per-source sample; 0 means full")
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required_bdnb = [
        "rel_batiment_groupe_dpe_logement.csv",
        "rel_batiment_construction_rnb.csv",
        "rel_batiment_groupe_adresse.csv",
        "rel_batiment_groupe_parcelle.csv",
        "batiment_construction.csv",
        "batiment_groupe.csv",
        "batiment_groupe_bdtopo_bat.csv",
        "batiment_groupe_ffo_bat.csv",
        "batiment_groupe_synthese_propriete_usage.csv",
        "metadonnees_bdnb_version.csv",
    ]
    missing = [name for name in required_bdnb if not (args.bdnb_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"BDNB directory is incomplete; missing: {missing}")

    dvf, dpe = parse_match_inputs(args.dvf, args.dpe, args.sample_size, args.seed)
    print(f"Loaded DVF={len(dvf):,}, DPE={len(dpe):,}")

    print("Loading DPE match evidence...")
    dpe_evidence = load_dpe_evidence(dpe, args.bdnb_dir)
    dpe_matches = resolve_dpe(dpe, dpe_evidence)
    print("Loading DVF parcel evidence...")
    dvf_matches, dvf_raw_counts = load_and_resolve_dvf(dvf, args.bdnb_dir)
    matches = pd.concat([dvf_matches, dpe_matches], ignore_index=True, sort=False)

    selected_groups = {
        x for x in map(clean_id, matches["bdnb_building_group_id"]) if x
    }
    print(f"Loading structural features for {len(selected_groups):,} selected buildings...")
    structural = load_structural_features(selected_groups, args.bdnb_dir)
    enriched = attach_features(matches, structural)

    # Stable, merge-friendly column order.
    metadata_columns = [
        "source",
        "record_id",
        "observation_date",
        "bdnb_building_group_id",
        "join_method",
        "join_confidence",
        "join_ambiguous",
        "evidence_conflict",
        "candidate_group_count",
        "candidate_building_group_ids",
        "dpe_direct_group_count",
        "rnb_group_count",
        "ban_group_count",
        "parcel_primary_candidate_count",
        "bdnb_dpe_match_types",
        "bdnb_dpe_geocode_reliability",
        "bdnb_ban_reliability_score",
    ]
    final_columns = [c for c in metadata_columns + STRUCTURAL_COLUMNS if c in enriched.columns]
    enriched = enriched[final_columns].sort_values(["source", "record_id"]).reset_index(drop=True)

    mode = "full" if args.sample_size == 0 else f"pilot_{args.sample_size}_per_source"
    output_csv = args.output_dir / "bdnb_building_enrichment.csv"
    enriched.to_csv(output_csv, index=False)

    version = pd.read_csv(args.bdnb_dir / "metadonnees_bdnb_version.csv", sep=";")
    current = version.iloc[0]
    audit = {
        "generated_at": pd.Timestamp.now(tz="Australia/Melbourne").isoformat(),
        "mode": mode,
        "inputs": {
            "dvf": str(args.dvf),
            "dpe": str(args.dpe),
            "bdnb_dir": str(args.bdnb_dir),
        },
        "bdnb_release": {
            "millesime": str(current["millesime"]),
            "version": str(current["version"]),
            "date_publication": str(current["date_publication"]),
        },
        "matching_relation_rows_loaded": {
            **dpe_evidence["raw_counts"],
            **dvf_raw_counts,
        },
        "selected_building_groups": int(len(selected_groups)),
        "sources": {
            "DVF": source_audit(enriched.loc[enriched["source"].eq("DVF")]),
            "DPE": source_audit(enriched.loc[enriched["source"].eq("DPE")]),
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "leakage_exclusions": [
            "batiment_groupe_dpe_*",
            "dpe_logement",
            "batiment_groupe_dvf_open_*",
            "energy consumption tables",
            "simulated DPE/value tables",
        ],
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(
        markdown_report(audit, output_csv), encoding="utf-8"
    )
    print(json.dumps(audit["sources"], indent=2, ensure_ascii=False))
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
