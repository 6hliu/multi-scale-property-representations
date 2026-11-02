#!/usr/bin/env python3
"""Validation-selected residual retrieval fusion for property-price prediction."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN = ROOT / "temporal_main_experiment_2026_09_14"
RESIDUAL = ROOT / "temporal_residual_full_2026_09_23"
SCREEN = ROOT / "temporal_residual_screen_2026_09_23"
RETRIEVAL_AWARE = ROOT / "temporal_retrieval_aware_2026_09_24"
OUT = HERE / "results"

sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(SCREEN))
sys.path.insert(0, str(RESIDUAL))
sys.path.insert(0, str(MAIN / "vendor"))

import run_temporal_retrieval as rr  # noqa: E402
import run_temporal_transformer as tt  # noqa: E402
from run_residual_screen import market_index  # noqa: E402
from run_full_experiment import residual_lookup  # noqa: E402


DATA = MAIN / "data/temporal_model_ready_55.csv.gz"
MANIFEST = MAIN / "data/feature_manifest.json"
SEEDS = tuple(tt.SEEDS)
EARTH_RADIUS_M = 6_371_008.8
REPETITIONS = 20_000
BOOTSTRAP_SEED = 20260924

GEO_BETAS = (0.0, 0.10, 0.25)
INTRINSIC_BETAS = (0.0, 0.25, 0.50)
KS = (5, 10, 20)
AGGREGATIONS = ("median", "softmax")
TEMPERATURES = (0.10, 0.25, 0.50)
MODEL_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)
SHORTLIST = 300


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_model(path: Path, groups: dict):
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    model = tt.TemporalGroupedTransformer(groups).to(tt.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_direct_predictions(phase: str, queries: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
    values = []
    for seed in SEEDS:
        if phase == "development":
            path = RESIDUAL / "results/development" / f"price_seed{seed}_lambda0" / "validation_price_predictions.csv.gz"
        else:
            path = RESIDUAL / "results/refit" / f"price_seed{seed}_lambda0" / "predictions_price.csv.gz"
        frame = pd.read_csv(path, dtype={"record_id": "string"})
        if not np.array_equal(frame.record_id.astype(str), queries.record_id.astype(str)):
            raise AssertionError(f"direct prediction alignment failed: {phase}, seed {seed}")
        values.append(frame.prediction.to_numpy(float))
    return np.mean(values, axis=0), values


def load_embeddings(
    phase: str,
    groups: dict,
    all_dvf_x: np.ndarray,
    dvf: pd.DataFrame,
) -> list[np.ndarray]:
    result = []
    for seed in SEEDS:
        if phase == "development":
            path = RESIDUAL / "results/development" / f"price_seed{seed}_lambda0"
            model = load_model(path, groups)
            embedding = tt.predict(model, all_dvf_x, "price", embeddings=True).astype(np.float32)
        else:
            path = RESIDUAL / "results/refit" / f"price_seed{seed}_lambda0"
            index = pd.read_csv(path / "dvf_embedding_index.csv.gz", dtype={"record_id": "string"})
            if not np.array_equal(index.record_id.astype(str), dvf.record_id.astype(str)):
                raise AssertionError(f"embedding index alignment failed: seed {seed}")
            embedding = np.load(path / "dvf_price_embeddings.npy").astype(np.float32)
        result.append(embedding)
    return result


def residual_slopes(candidates: pd.DataFrame, residual: np.ndarray) -> dict[str, float]:
    output = {}
    for property_type in ("House", "Apartment"):
        mask = candidates.property_type.eq(property_type).to_numpy()
        output[property_type] = float(np.polyfit(
            candidates.loc[mask, "months_since_2021"].to_numpy(float),
            residual[mask],
            1,
        )[0])
    return output


def intrinsic_distance(candidates: pd.DataFrame, queries: pd.DataFrame, shortlist: np.ndarray) -> np.ndarray:
    candidate = np.column_stack([
        np.log(candidates.floor_area_m2.to_numpy(float)),
        np.log1p(candidates.room_count.to_numpy(float)),
    ])
    query = np.column_stack([
        np.log(queries.floor_area_m2.to_numpy(float)),
        np.log1p(queries.room_count.to_numpy(float)),
    ])
    mean = candidate.mean(axis=0)
    std = candidate.std(axis=0)
    std[std < 1e-8] = 1.0
    candidate = (candidate - mean) / std
    query = (query - mean) / std
    return np.sqrt(((candidate[shortlist] - query[:, None, :]) ** 2).sum(axis=2)).astype(np.float32)


def order_and_scores(score: np.ndarray, indices: np.ndarray, maximum_k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    chosen = np.argpartition(score, kth=maximum_k - 1, axis=1)[:, :maximum_k]
    chosen_score = np.take_along_axis(score, chosen, axis=1)
    local_order = np.argsort(chosen_score, axis=1, kind="stable")
    return (
        np.take_along_axis(indices, np.take_along_axis(chosen, local_order, axis=1), axis=1),
        np.take_along_axis(chosen_score, local_order, axis=1),
    )


def phase_context(
    phase: str,
    data: pd.DataFrame,
    matrix: np.ndarray,
    groups: dict,
    market: np.ndarray,
) -> dict:
    dvf_mask = data.source.eq("DVF").to_numpy()
    dvf = data.loc[dvf_mask].reset_index(drop=True)
    if phase == "development":
        candidate_mask = dvf.temporal_split.eq("train").to_numpy()
        query_mask = dvf.temporal_split.eq("validation").to_numpy()
    else:
        candidate_mask = dvf.temporal_split.isin(["train", "validation"]).to_numpy()
        query_mask = dvf.temporal_split.eq("test").to_numpy()
    candidates = dvf.loc[candidate_mask].reset_index(drop=True)
    queries = dvf.loc[query_mask].reset_index(drop=True)
    if candidates.observation_date.max() >= queries.observation_date.min():
        raise AssertionError(f"candidate/query temporal overlap: {phase}")

    residual, baseline = residual_lookup(dvf, market)
    residual_values = residual.loc[dvf.record_id.astype(str)].to_numpy(float)
    baseline_values = baseline.loc[dvf.record_id.astype(str)].to_numpy(float)
    candidate_residual = residual_values[candidate_mask]
    query_baseline = baseline_values[query_mask]

    embeddings = load_embeddings(phase, groups, matrix[dvf_mask], dvf)
    geo_index, geo_distance = rr.geo_shortlist(candidates, queries, SHORTLIST)
    if not (candidates.observation_date.to_numpy()[geo_index] < queries.observation_date.to_numpy()[:, None]).all():
        raise AssertionError("retrieval includes a non-earlier sale")
    if not (candidates.property_type.to_numpy()[geo_index] == queries.property_type.to_numpy()[:, None]).all():
        raise AssertionError("retrieval property type mismatch")
    if (candidates.parcel_id.astype(str).to_numpy()[geo_index] == queries.parcel_id.astype(str).to_numpy()[:, None]).any():
        raise AssertionError("retrieval parcel leakage")

    embedding_score = np.mean([
        rr.embedding_distance(value[candidate_mask], value[query_mask], geo_index)
        for value in embeddings
    ], axis=0)
    geo_score = geo_distance / np.maximum(np.median(geo_distance, axis=1, keepdims=True), 1.0)
    intrinsic_score = intrinsic_distance(candidates, queries, geo_index)
    direct_prediction, seed_predictions = load_direct_predictions(phase, queries)
    return {
        "dvf": dvf,
        "candidates": candidates,
        "queries": queries,
        "candidate_residual": candidate_residual,
        "query_baseline": query_baseline,
        "direct_prediction": direct_prediction,
        "seed_direct_predictions": seed_predictions,
        "geo_index": geo_index,
        "geo_score": geo_score,
        "intrinsic_score": intrinsic_score,
        "embedding_score": embedding_score,
        "slopes": residual_slopes(candidates, candidate_residual),
    }


def neighbour_residual(
    context: dict,
    geo_beta: float,
    intrinsic_beta: float,
    k: int,
    aggregation: str,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = (
        context["embedding_score"]
        + geo_beta * context["geo_score"]
        + intrinsic_beta * context["intrinsic_score"]
    )
    order, ordered_score = order_and_scores(score, context["geo_index"], max(KS))
    selected = order[:, :k]
    selected_score = ordered_score[:, :k]
    values = context["candidate_residual"][selected].copy()
    candidate_month = context["candidates"].months_since_2021.to_numpy(float)[selected]
    query_month = context["queries"].months_since_2021.to_numpy(float)[:, None]
    slope = context["queries"].property_type.map(context["slopes"]).to_numpy(float)[:, None]
    values += slope * (query_month - candidate_month)
    if aggregation == "median":
        prediction = np.median(values, axis=1)
        effective_n = np.full(len(values), float(k))
    elif aggregation == "softmax":
        shifted = selected_score - selected_score.min(axis=1, keepdims=True)
        weights = np.exp(-shifted / temperature)
        weights /= weights.sum(axis=1, keepdims=True)
        prediction = (weights * values).sum(axis=1)
        effective_n = 1.0 / np.maximum((weights * weights).sum(axis=1), 1e-12)
    else:
        raise ValueError(aggregation)
    return prediction, order, effective_n


def tune(context: dict) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    truth = context["queries"].log_sale_price.to_numpy(float)
    direct_residual = context["direct_prediction"] - context["query_baseline"]
    rows = []
    predictions = {}
    neighbour_cache = {}
    for geo_beta in GEO_BETAS:
        for intrinsic_beta in INTRINSIC_BETAS:
            for k in KS:
                for aggregation in AGGREGATIONS:
                    temperatures = (0.25,) if aggregation == "median" else TEMPERATURES
                    for temperature in temperatures:
                        key = (geo_beta, intrinsic_beta, k, aggregation, temperature)
                        neighbour, _, effective_n = neighbour_residual(
                            context, geo_beta, intrinsic_beta, k, aggregation, temperature,
                        )
                        neighbour_cache[key] = neighbour
                        for model_weight in MODEL_WEIGHTS:
                            fused_residual = model_weight * direct_residual + (1.0 - model_weight) * neighbour
                            prediction = context["query_baseline"] + fused_residual
                            metric = tt.price_metrics(
                                truth,
                                prediction,
                                context["queries"].sale_price_eur.to_numpy(float),
                            )
                            row = {
                                "geo_beta": geo_beta,
                                "intrinsic_beta": intrinsic_beta,
                                "k": k,
                                "aggregation": aggregation,
                                "temperature": temperature,
                                "model_weight": model_weight,
                                "mean_effective_neighbours": float(effective_n.mean()),
                                **metric,
                            }
                            rows.append(row)
                            predictions[(geo_beta, intrinsic_beta, k, aggregation, temperature, model_weight)] = prediction
    frame = pd.DataFrame(rows)
    frame["complexity"] = (
        frame.geo_beta.ne(0).astype(int)
        + frame.intrinsic_beta.ne(0).astype(int)
        + frame.aggregation.eq("softmax").astype(int)
    )
    selected = frame.sort_values(
        ["log_mae", "model_weight", "k", "complexity", "temperature"],
        ascending=[True, False, True, True, False],
    ).iloc[0]
    config = {
        "geo_beta": float(selected.geo_beta),
        "intrinsic_beta": float(selected.intrinsic_beta),
        "k": int(selected.k),
        "aggregation": str(selected.aggregation),
        "temperature": float(selected.temperature),
        "model_weight": float(selected.model_weight),
    }
    key = (
        config["geo_beta"], config["intrinsic_beta"], config["k"],
        config["aggregation"], config["temperature"], config["model_weight"],
    )
    neighbour_key = key[:-1]
    return frame, config, predictions[key], neighbour_cache[neighbour_key]


def apply(context: dict, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbour, order, effective_n = neighbour_residual(
        context,
        config["geo_beta"], config["intrinsic_beta"], config["k"],
        config["aggregation"], config["temperature"],
    )
    direct_residual = context["direct_prediction"] - context["query_baseline"]
    fused = config["model_weight"] * direct_residual + (1.0 - config["model_weight"]) * neighbour
    return context["query_baseline"] + fused, order, effective_n


def blocks(frame: pd.DataFrame) -> np.ndarray:
    reference = math.radians(float(frame.latitude.median()))
    x = EARTH_RADIUS_M * np.radians(frame.longitude.to_numpy(float)) * math.cos(reference)
    y = EARTH_RADIUS_M * np.radians(frame.latitude.to_numpy(float))
    return np.char.add(
        np.floor(x / 3000).astype(int).astype(str),
        np.char.add("_", np.floor(y / 3000).astype(int).astype(str)),
    )


def bootstrap(frame: pd.DataFrame, reference: np.ndarray, candidate: np.ndarray, label: str) -> dict:
    truth = frame.log_sale_price.to_numpy(float)
    difference = np.abs(truth - reference) - np.abs(truth - candidate)
    block = blocks(frame)
    unique, codes = np.unique(block, return_inverse=True)
    sums = np.bincount(codes, weights=difference, minlength=len(unique))
    counts = np.bincount(codes, minlength=len(unique))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(REPETITIONS, float)
    for index in range(REPETITIONS):
        sampled = rng.integers(0, len(unique), len(unique))
        draws[index] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "comparison": label,
        "point_gain": float(difference.mean()),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "spatial_blocks": int(len(unique)),
        "bootstrap_repetitions": REPETITIONS,
        "positive_favours": "Residual Transformer + retrieval prediction",
    }


def external_references(queries: pd.DataFrame) -> dict[str, np.ndarray]:
    references = {}
    main_xgb = pd.read_csv(
        MAIN / "results/sensitivity/xgboost_price_test_predictions.csv.gz",
        dtype={"record_id": "string"},
    )
    if not np.array_equal(main_xgb.record_id.astype(str), queries.record_id.astype(str)):
        raise AssertionError("XGBoost alignment")
    references["Full-55 XGBoost"] = main_xgb.prediction.to_numpy(float)

    aware = pd.read_csv(RETRIEVAL_AWARE / "results/retrieval_test_2025_predictions.csv.gz", dtype={"record_id": "string"})
    if not np.array_equal(aware.record_id.astype(str), queries.record_id.astype(str)):
        raise AssertionError("retrieval-aware alignment")
    references["Retrieval-aware shared embedding"] = aware.retrieval_aware_shared.to_numpy(float)
    return references


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data = pd.read_csv(DATA, low_memory=False, dtype={"record_id": "string"})
    data.observation_date = pd.to_datetime(data.observation_date)
    manifest = json.loads(MANIFEST.read_text())
    groups = tt.build_groups(manifest)
    raw = tt.raw_feature_frame(data, manifest["feature_names"])
    dvf = data.loc[data.source.eq("DVF")].reset_index(drop=True)

    dev_train = data.temporal_split.eq("train").to_numpy()
    dev_validation = data.temporal_split.eq("validation").to_numpy()
    dev = tt.prepare(data, raw, dev_train, dev_validation)
    dev_market, dev_audit = market_index(
        dvf, dvf.temporal_split.eq("train").to_numpy(), pd.Timestamp("2023-01-01"),
    )
    development = phase_context("development", data, dev["matrix"], groups, dev_market)
    tuning, config, dev_prediction, dev_neighbour = tune(development)
    tuning.to_csv(OUT / "validation_2024_tuning.csv", index=False)
    dump(OUT / "SELECTION.json", {
        "selected": config,
        "selection_population": "2024 validation only",
        "test_used_for_selection": False,
        "validation_metrics": tt.price_metrics(
            development["queries"].log_sale_price.to_numpy(float),
            dev_prediction,
            development["queries"].sale_price_eur.to_numpy(float),
        ),
    })
    dev_output = development["queries"][[
        "record_id", "parcel_id", "observation_date", "longitude", "latitude",
        "sale_price_eur", "log_sale_price",
    ]].copy()
    dev_output["residual_transformer"] = development["direct_prediction"]
    dev_output["retrieved_residual_prediction"] = development["query_baseline"] + dev_neighbour
    dev_output["residual_retrieval_fusion"] = dev_prediction
    dev_output.to_csv(OUT / "validation_2024_predictions.csv.gz", index=False)

    refit_train = data.temporal_split.isin(["train", "validation"]).to_numpy()
    test = data.temporal_split.eq("test").to_numpy()
    refit = tt.prepare(data, raw, refit_train, test)
    refit_market, refit_audit = market_index(
        dvf, dvf.temporal_split.isin(["train", "validation"]).to_numpy(), pd.Timestamp("2024-01-01"),
    )
    evaluation = phase_context("refit", data, refit["matrix"], groups, refit_market)
    prediction, order, effective_n = apply(evaluation, config)
    truth = evaluation["queries"].log_sale_price.to_numpy(float)
    euro = evaluation["queries"].sale_price_eur.to_numpy(float)

    references = {
        "Residual price-only Transformer": evaluation["direct_prediction"],
        **external_references(evaluation["queries"]),
    }
    rows = [{
        "method": "Residual Transformer + retrieval prediction",
        **tt.price_metrics(truth, prediction, euro),
    }]
    for name, value in references.items():
        rows.append({"method": name, **tt.price_metrics(truth, value, euro)})
    comparison = pd.DataFrame(rows).sort_values("log_mae")
    comparison.to_csv(OUT / "test_2025_comparison.csv", index=False)

    uncertainty = {
        name: bootstrap(evaluation["queries"], value, prediction, f"candidate vs {name}")
        for name, value in references.items()
    }
    dump(OUT / "paired_spatial_block_uncertainty.json", uncertainty)

    output = evaluation["queries"][[
        "record_id", "parcel_id", "observation_date", "property_type",
        "floor_area_m2", "room_count", "longitude", "latitude",
        "sale_price_eur", "log_sale_price",
    ]].copy()
    output["residual_transformer"] = evaluation["direct_prediction"]
    output["retrieved_residual_prediction"] = evaluation["query_baseline"] + neighbour_residual(
        evaluation, config["geo_beta"], config["intrinsic_beta"], config["k"],
        config["aggregation"], config["temperature"],
    )[0]
    output["residual_retrieval_fusion"] = prediction
    output["effective_neighbours"] = effective_n
    output.to_csv(OUT / "test_2025_predictions.csv.gz", index=False)

    integrity = {
        "status": "PASS",
        "development_candidate_max_date": str(development["candidates"].observation_date.max().date()),
        "development_query_min_date": str(development["queries"].observation_date.min().date()),
        "test_candidate_max_date": str(evaluation["candidates"].observation_date.max().date()),
        "test_query_min_date": str(evaluation["queries"].observation_date.min().date()),
        "same_type_rate": 1.0,
        "different_parcel_rate": 1.0,
        "strictly_earlier_rate": 1.0,
        "n_development_queries": len(development["queries"]),
        "n_test_queries": len(evaluation["queries"]),
        "mean_effective_neighbours_test": float(effective_n.mean()),
    }
    dump(OUT / "integrity_audit.json", integrity)
    dump(OUT / "market_index_audit.json", {"development": dev_audit, "refit": refit_audit})
    dump(OUT / "run_manifest.json", {
        "status": "PASS",
        "protocol": str(HERE / "PROTOCOL.md"),
        "protocol_sha256": digest(HERE / "PROTOCOL.md"),
        "data": str(DATA),
        "data_sha256": digest(DATA),
        "seeds": list(SEEDS),
        "selected_configuration": config,
        "development": "2021-2023 candidates; 2024 validation selection",
        "extension_test": "2021-2024 candidates; 2025 protocol-frozen extension test",
        "test_used_for_selection": False,
    })
    print("\nSELECTED CONFIGURATION\n", json.dumps(config, indent=2), flush=True)
    print("\n2025 COMPARISON\n", comparison.to_string(index=False), flush=True)
    print("\nUNCERTAINTY\n", json.dumps(uncertainty, indent=2), flush=True)


if __name__ == "__main__":
    main()
