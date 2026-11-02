#!/usr/bin/env python3
"""Retrieval-aware residual training followed by neighbour-residual fusion."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import BallTree
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN = ROOT / "temporal_main_experiment_2026_09_14"
SCREEN = ROOT / "temporal_residual_screen_2026_09_23"
RESIDUAL = ROOT / "temporal_residual_full_2026_09_23"
OLD_FUSION = ROOT / "temporal_residual_retrieval_fusion_2026_09_24"
RETRIEVAL_AWARE = ROOT / "temporal_retrieval_aware_2026_09_24"
OUT = HERE / "results"
TUPLES = HERE / "triplets"

sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(SCREEN))
sys.path.insert(0, str(RESIDUAL))
sys.path.insert(0, str(OLD_FUSION))
sys.path.insert(0, str(RETRIEVAL_AWARE))
sys.path.insert(0, str(MAIN / "vendor"))

import run_temporal_retrieval as rr  # noqa: E402
import run_temporal_transformer as tt  # noqa: E402
from run_residual_screen import market_index  # noqa: E402
from run_full_experiment import residual_lookup  # noqa: E402
import run_retrieval_aware as metric  # noqa: E402


_fusion_spec = importlib.util.spec_from_file_location(
    "residual_retrieval_fusion_helpers", OLD_FUSION / "run_experiment.py"
)
if _fusion_spec is None or _fusion_spec.loader is None:
    raise ImportError("Could not load residual retrieval fusion helpers")
fusion = importlib.util.module_from_spec(_fusion_spec)
_fusion_spec.loader.exec_module(fusion)


DATA = MAIN / "data/temporal_model_ready_55.csv.gz"
MANIFEST = MAIN / "data/feature_manifest.json"
SEEDS = tuple(tt.SEEDS)
GAMMAS = (0.003, 0.01, 0.03)
STEPS = 512
INTERVAL = 128
BATCH = 1024
NEIGHBOURS = 768
NEGATIVES = 4
EARTH_RADIUS_M = 6_371_008.8


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def gamma_tag(gamma: float) -> str:
    return f"gamma_{gamma:g}"


def model_dir(phase: str, gamma: float, seed: int) -> Path:
    return OUT / phase / gamma_tag(gamma) / f"seed_{seed}"


def values_for(frame: pd.DataFrame, lookup: pd.Series) -> np.ndarray:
    return lookup.loc[frame.record_id.astype(str)].to_numpy(np.float32)


def add_all_dvf(arrays: dict, data: pd.DataFrame) -> None:
    mask = data.source.eq("DVF").to_numpy()
    arrays["all_dvf_x"] = arrays["matrix"][mask]
    arrays["all_dvf_f"] = data.loc[mask].reset_index(drop=True)


def build_residual_tuples(frame: pd.DataFrame, residual: pd.Series, label: str) -> pd.DataFrame:
    output = TUPLES / f"{label}_tuples.csv.gz"
    audit_path = TUPLES / f"{label}_audit.json"
    if output.exists() and audit_path.exists():
        return pd.read_csv(output)

    frame = frame.copy().reset_index(drop=True)
    frame.observation_date = pd.to_datetime(frame.observation_date)
    frame["target_residual"] = values_for(frame, residual)
    frame["day"] = frame.observation_date.to_numpy(dtype="datetime64[D]").astype(np.int64)
    frame["log_area"] = np.log(frame.floor_area_m2.to_numpy(float))
    frame["log_rooms"] = np.log1p(frame.room_count.to_numpy(float))
    required = [
        "record_id", "property_type", "parcel_id", "latitude", "longitude",
        "floor_area_m2", "room_count", "target_residual", "day",
    ]
    if frame[required].isna().any().any():
        raise AssertionError(f"missing tuple fields in {label}")

    radians = np.radians(frame[["latitude", "longitude"]].to_numpy(float))
    parcel = frame.parcel_id.astype(str).to_numpy()
    day = frame.day.to_numpy(np.int64)
    target = frame.target_residual.to_numpy(float)
    log_area = frame.log_area.to_numpy(float)
    log_rooms = frame.log_rooms.to_numpy(float)
    rows = []
    for property_type in sorted(frame.property_type.unique()):
        global_index = np.flatnonzero(frame.property_type.eq(property_type).to_numpy())
        tree = BallTree(radians[global_index], metric="haversine")
        request = min(NEIGHBOURS + 1, len(global_index))
        eligible_before = len(rows)
        for start in range(0, len(global_index), 1024):
            anchors = global_index[start:start + 1024]
            angular, local = tree.query(radians[anchors], k=request)
            candidate_global = global_index[local]
            distance_m = angular * EARTH_RADIUS_M
            for offset, anchor in enumerate(anchors):
                candidates = candidate_global[offset]
                distances = distance_m[offset]
                keep = (day[candidates] < day[anchor]) & (parcel[candidates] != parcel[anchor])
                candidates = candidates[keep]
                distances = distances[keep]
                if len(candidates) < 16:
                    continue
                area_gap = np.abs(log_area[candidates] - log_area[anchor])
                room_gap = np.abs(log_rooms[candidates] - log_rooms[anchor])
                teacher = distances / 1000.0 + 2.0 * area_gap + 0.5 * room_gap
                order = np.argsort(teacher, kind="stable")
                pool_n = min(128, len(order))
                pool = candidates[order[:pool_n]]
                pool_distance = distances[order[:pool_n]]
                pool_area = area_gap[order[:pool_n]]
                pool_room = room_gap[order[:pool_n]]
                gap = np.abs(target[pool] - target[anchor])
                positive_local = int(np.argmin(gap + 0.01 * teacher[order[:pool_n]]))
                positive = int(pool[positive_local])
                available = np.flatnonzero(pool != positive)
                hard = available[np.argsort(gap[available], kind="stable")[::-1]][:NEGATIVES]
                if len(hard) < NEGATIVES:
                    continue
                row = {
                    "anchor_index": int(anchor),
                    "positive_index": positive,
                    "anchor_record_id": str(frame.record_id.iloc[anchor]),
                    "positive_record_id": str(frame.record_id.iloc[positive]),
                    "positive_price_gap": float(gap[positive_local]),
                    "positive_distance_m": float(pool_distance[positive_local]),
                    "positive_area_gap": float(pool_area[positive_local]),
                    "positive_room_gap": float(pool_room[positive_local]),
                    "anchor_date": str(frame.observation_date.iloc[anchor].date()),
                    "positive_date": str(frame.observation_date.iloc[positive].date()),
                }
                for number, local_index in enumerate(hard, start=1):
                    negative = int(pool[local_index])
                    row[f"negative_{number}_index"] = negative
                    row[f"negative_{number}_record_id"] = str(frame.record_id.iloc[negative])
                    row[f"negative_{number}_price_gap"] = float(gap[local_index])
                rows.append(row)
        print(f"TUPLES {label} {property_type}: {len(rows) - eligible_before}", flush=True)

    tuples = pd.DataFrame(rows).sort_values("anchor_index").reset_index(drop=True)
    if tuples.anchor_index.duplicated().any():
        raise AssertionError("duplicate tuple anchor")
    anchor = frame.iloc[tuples.anchor_index.to_numpy(int)].reset_index(drop=True)
    positive = frame.iloc[tuples.positive_index.to_numpy(int)].reset_index(drop=True)
    negative_columns = [f"negative_{number}_index" for number in range(1, NEGATIVES + 1)]
    integrity = (
        (positive.observation_date.to_numpy() < anchor.observation_date.to_numpy()).all()
        and (positive.property_type.to_numpy() == anchor.property_type.to_numpy()).all()
        and (positive.parcel_id.astype(str).to_numpy() != anchor.parcel_id.astype(str).to_numpy()).all()
    )
    for column in negative_columns:
        negative = frame.iloc[tuples[column].to_numpy(int)].reset_index(drop=True)
        integrity = integrity and (
            (negative.observation_date.to_numpy() < anchor.observation_date.to_numpy()).all()
            and (negative.property_type.to_numpy() == anchor.property_type.to_numpy()).all()
            and (negative.parcel_id.astype(str).to_numpy() != anchor.parcel_id.astype(str).to_numpy()).all()
        )
    if not integrity:
        raise AssertionError("tuple integrity failure")
    negative_gap = tuples[[f"negative_{n}_price_gap" for n in range(1, NEGATIVES + 1)]].min(axis=1)
    separation = float((tuples.positive_price_gap < negative_gap).mean())
    if separation < 0.95:
        raise AssertionError(f"weak residual separation {separation}")
    TUPLES.mkdir(parents=True, exist_ok=True)
    tuples.to_csv(output, index=False)
    dump(audit_path, {
        "status": "PASS", "label": label, "population_rows": len(frame),
        "eligible_anchor_rows": len(tuples), "eligible_rate": len(tuples) / len(frame),
        "strictly_earlier_rate": 1.0, "same_type_rate": 1.0,
        "different_parcel_rate": 1.0,
        "positive_better_than_all_negatives_rate": separation,
        "positive_residual_gap_median": float(tuples.positive_price_gap.median()),
        "minimum_negative_residual_gap_median": float(negative_gap.median()),
        "positive_distance_m_median": float(tuples.positive_distance_m.median()),
        "target": "lagged-local-market residual", "training_labels_only": True,
    })
    return tuples


def direct_evaluation(model, arrays: dict, residual: pd.Series, baseline: pd.Series, center: float):
    residual_prediction = tt.predict(
        model, arrays["price_evaluation_x"], "price", arrays["price_evaluation_room"],
    ) + center
    prediction = residual_prediction + values_for(arrays["price_evaluation_f"], baseline)
    result = tt.price_metrics(
        arrays["price_evaluation_y"], prediction,
        arrays["price_evaluation_f"].sale_price_eur.to_numpy(float),
    )
    return result, prediction


def fit_development(
    arrays: dict,
    groups: dict,
    tuples: pd.DataFrame,
    residual: pd.Series,
    baseline: pd.Series,
    gamma: float,
    seed: int,
) -> dict:
    output = model_dir("development", gamma, seed)
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = tt.TemporalGroupedTransformer(groups).to(tt.DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    huber = nn.HuberLoss()
    train_residual = values_for(arrays["price_train_f"], residual)
    center = float(np.median(train_residual))
    px = torch.from_numpy(arrays["price_train_x"]).to(tt.DEVICE)
    room = torch.from_numpy(arrays["price_train_room"]).to(tt.DEVICE)
    target = torch.from_numpy(train_residual).to(tt.DEVICE)
    rng_p = np.random.default_rng(seed + 10007)
    rng_m = np.random.default_rng(seed + 30013)
    states = {}
    history = []
    started = time.time()
    for step in range(1, STEPS + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        batch = torch.as_tensor(rng_p.integers(len(target), size=BATCH), device=tt.DEVICE)
        price_loss = huber(model.price(px[batch], room[batch]), target[batch] - center)
        retrieval_loss, positive, negative, retrieval_accuracy = metric.metric_loss(model, px, tuples, rng_m)
        total = price_loss + gamma * retrieval_loss
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        if step % INTERVAL == 0:
            result, _ = direct_evaluation(model, arrays, residual, baseline, center)
            states[step] = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            history.append({
                "step": step, "gamma_retrieval": gamma,
                "direct_log_mae": result["log_mae"],
                "direct_log_rmse": result["log_rmse"],
                "batch_price_loss": float(price_loss.detach().cpu()),
                "batch_retrieval_loss": float(retrieval_loss.detach().cpu()),
                "batch_positive_similarity": float(positive.detach().cpu()),
                "batch_negative_similarity": float(negative.detach().cpu()),
                "batch_retrieval_accuracy": float(retrieval_accuracy.detach().cpu()),
            })
            print(
                f"DEV gamma={gamma:g} seed={seed} step={step} direct={result['log_mae']:.6f} "
                f"metric_loss={float(retrieval_loss.detach().cpu()):.4f}", flush=True,
            )
    curve = pd.DataFrame(history)
    chosen = curve.sort_values(["direct_log_mae", "step"]).iloc[0]
    selected_step = int(chosen.step)
    model.load_state_dict(states[selected_step])
    result, prediction = direct_evaluation(model, arrays, residual, baseline, center)
    embedding = tt.predict(model, arrays["all_dvf_x"], "price", embeddings=True).astype(np.float32)
    torch.save({"state_dict": states[selected_step]}, output / "model.pt")
    np.save(output / "dvf_price_embeddings.npy", embedding)
    arrays["all_dvf_f"][["record_id", "parcel_id", "observation_date", "temporal_split"]].to_csv(
        output / "dvf_embedding_index.csv.gz", index=False,
    )
    pred = arrays["price_evaluation_f"][[
        "record_id", "parcel_id", "observation_date", "longitude", "latitude",
        "sale_price_eur", "log_sale_price",
    ]].copy()
    pred["prediction"] = prediction
    pred.to_csv(output / "validation_predictions.csv.gz", index=False)
    curve.to_csv(output / "validation_curve.csv", index=False)
    metadata = {
        "phase": "development", "seed": seed, "gamma_retrieval": gamma,
        "selected_step": selected_step, "price_center": center,
        "validation_price": result, "elapsed_seconds": time.time() - started,
        "test_used_for_selection": False,
    }
    dump(metadata_path, metadata)
    return metadata


def fit_refit(
    arrays: dict,
    groups: dict,
    tuples: pd.DataFrame,
    residual: pd.Series,
    baseline: pd.Series,
    gamma: float,
    seed: int,
    steps: int,
) -> dict:
    output = model_dir("refit", gamma, seed)
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = tt.TemporalGroupedTransformer(groups).to(tt.DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    huber = nn.HuberLoss()
    train_residual = values_for(arrays["price_train_f"], residual)
    center = float(np.median(train_residual))
    px = torch.from_numpy(arrays["price_train_x"]).to(tt.DEVICE)
    room = torch.from_numpy(arrays["price_train_room"]).to(tt.DEVICE)
    target = torch.from_numpy(train_residual).to(tt.DEVICE)
    rng_p = np.random.default_rng(seed + 10007)
    rng_m = np.random.default_rng(seed + 30013)
    started = time.time()
    for _step in range(1, steps + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        batch = torch.as_tensor(rng_p.integers(len(target), size=BATCH), device=tt.DEVICE)
        price_loss = huber(model.price(px[batch], room[batch]), target[batch] - center)
        retrieval_loss, _, _, _ = metric.metric_loss(model, px, tuples, rng_m)
        (price_loss + gamma * retrieval_loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
    result, prediction = direct_evaluation(model, arrays, residual, baseline, center)
    embedding = tt.predict(model, arrays["all_dvf_x"], "price", embeddings=True).astype(np.float32)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"state_dict": state}, output / "model.pt")
    np.save(output / "dvf_price_embeddings.npy", embedding)
    arrays["all_dvf_f"][["record_id", "parcel_id", "observation_date", "temporal_split"]].to_csv(
        output / "dvf_embedding_index.csv.gz", index=False,
    )
    pred = arrays["price_evaluation_f"][[
        "record_id", "parcel_id", "observation_date", "longitude", "latitude",
        "sale_price_eur", "log_sale_price",
    ]].copy()
    pred["prediction"] = prediction
    pred.to_csv(output / "test_predictions.csv.gz", index=False)
    metadata = {
        "phase": "refit", "seed": seed, "gamma_retrieval": gamma,
        "training_steps": steps, "price_center": center, "test_price": result,
        "elapsed_seconds": time.time() - started,
    }
    dump(metadata_path, metadata)
    return metadata


def make_context(
    arrays: dict,
    residual: pd.Series,
    baseline: pd.Series,
    embeddings: list[np.ndarray],
    direct_predictions: list[np.ndarray],
    development: bool,
) -> dict:
    dvf = arrays["all_dvf_f"]
    if development:
        candidate_mask = dvf.temporal_split.eq("train").to_numpy()
        query_mask = dvf.temporal_split.eq("validation").to_numpy()
    else:
        candidate_mask = dvf.temporal_split.isin(["train", "validation"]).to_numpy()
        query_mask = dvf.temporal_split.eq("test").to_numpy()
    candidates = dvf.loc[candidate_mask].reset_index(drop=True)
    queries = dvf.loc[query_mask].reset_index(drop=True)
    if candidates.observation_date.max() >= queries.observation_date.min():
        raise AssertionError("temporal overlap")
    geo_index, geo_distance = rr.geo_shortlist(candidates, queries, fusion.SHORTLIST)
    embedding_score = np.mean([
        rr.embedding_distance(value[candidate_mask], value[query_mask], geo_index)
        for value in embeddings
    ], axis=0)
    geo_score = geo_distance / np.maximum(np.median(geo_distance, axis=1, keepdims=True), 1.0)
    residual_values = residual.loc[dvf.record_id.astype(str)].to_numpy(float)
    baseline_values = baseline.loc[dvf.record_id.astype(str)].to_numpy(float)
    candidate_residual = residual_values[candidate_mask]
    return {
        "candidates": candidates,
        "queries": queries,
        "candidate_residual": candidate_residual,
        "query_baseline": baseline_values[query_mask],
        "direct_prediction": np.mean(direct_predictions, axis=0),
        "seed_direct_predictions": direct_predictions,
        "geo_index": geo_index,
        "geo_score": geo_score,
        "intrinsic_score": fusion.intrinsic_distance(candidates, queries, geo_index),
        "embedding_score": embedding_score,
        "slopes": fusion.residual_slopes(candidates, candidate_residual),
    }


def load_outputs(phase: str, gamma: float, seeds: tuple[int, ...], dvf: pd.DataFrame):
    embeddings, predictions = [], []
    for seed in seeds:
        path = model_dir(phase, gamma, seed)
        index = pd.read_csv(path / "dvf_embedding_index.csv.gz", dtype={"record_id": "string"})
        if not np.array_equal(index.record_id.astype(str), dvf.record_id.astype(str)):
            raise AssertionError(f"embedding alignment {phase} {seed}")
        embeddings.append(np.load(path / "dvf_price_embeddings.npy"))
        pred_name = "validation_predictions.csv.gz" if phase == "development" else "test_predictions.csv.gz"
        predictions.append(pd.read_csv(path / pred_name).prediction.to_numpy(float))
    return embeddings, predictions


def prepare():
    manifest = json.loads(MANIFEST.read_text())
    groups = tt.build_groups(manifest)
    data = pd.read_csv(DATA, low_memory=False, dtype={"record_id": "string"})
    data.observation_date = pd.to_datetime(data.observation_date)
    raw = tt.raw_feature_frame(data, manifest["feature_names"])
    dvf = data.loc[data.source.eq("DVF")].reset_index(drop=True)

    dev = tt.prepare(
        data, raw, data.temporal_split.eq("train").to_numpy(),
        data.temporal_split.eq("validation").to_numpy(),
    )
    add_all_dvf(dev, data)
    dev_market, dev_audit = market_index(
        dvf, dvf.temporal_split.eq("train").to_numpy(), pd.Timestamp("2023-01-01"),
    )
    dev_residual, dev_baseline = residual_lookup(dvf, dev_market)

    refit = tt.prepare(
        data, raw, data.temporal_split.isin(["train", "validation"]).to_numpy(),
        data.temporal_split.eq("test").to_numpy(),
    )
    add_all_dvf(refit, data)
    refit_market, refit_audit = market_index(
        dvf, dvf.temporal_split.isin(["train", "validation"]).to_numpy(), pd.Timestamp("2024-01-01"),
    )
    refit_residual, refit_baseline = residual_lookup(dvf, refit_market)
    return (
        data, groups, dev, dev_residual, dev_baseline, dev_audit,
        refit, refit_residual, refit_baseline, refit_audit,
    )


def build_phase(dev, dev_residual, refit, refit_residual):
    build_residual_tuples(dev["price_train_f"], dev_residual, "development_2021_2023")
    build_residual_tuples(refit["price_train_f"], refit_residual, "refit_2021_2024")


def screen_phase(groups, dev, residual, baseline):
    tuples = pd.read_csv(TUPLES / "development_2021_2023_tuples.csv.gz")
    rows = []
    for gamma in GAMMAS:
        meta = fit_development(dev, groups, tuples, residual, baseline, gamma, 42)
        embeddings, predictions = load_outputs("development", gamma, (42,), dev["all_dvf_f"])
        context = make_context(dev, residual, baseline, embeddings, predictions, True)
        _, config, fused, _ = fusion.tune(context)
        fused_metric = tt.price_metrics(
            context["queries"].log_sale_price.to_numpy(float), fused,
            context["queries"].sale_price_eur.to_numpy(float),
        )
        rows.append({
            "gamma_retrieval": gamma,
            "selected_step": meta["selected_step"],
            "direct_log_mae": meta["validation_price"]["log_mae"],
            "fused_log_mae": fused_metric["log_mae"],
            "fusion_config": json.dumps(config, sort_keys=True),
        })
    frame = pd.DataFrame(rows).sort_values(["fused_log_mae", "gamma_retrieval"])
    chosen = frame.iloc[0]
    frame.to_csv(OUT / "gamma_screen_2024.csv", index=False)
    dump(OUT / "GAMMA_SELECTION.json", {
        "selected_gamma_retrieval": float(chosen.gamma_retrieval),
        "selection_seed": 42, "selection_population": "2024 validation only",
        "criterion": "minimum validation fused Log-MAE",
        "test_used_for_selection": False,
    })
    print("\nGAMMA SCREEN\n", frame.to_string(index=False), flush=True)


def full_phase(data, groups, dev, dev_residual, dev_baseline, refit, refit_residual, refit_baseline,
               include_prior_comparator=False):
    gamma = float(json.loads((OUT / "GAMMA_SELECTION.json").read_text())["selected_gamma_retrieval"])
    dev_tuples = pd.read_csv(TUPLES / "development_2021_2023_tuples.csv.gz")
    development_rows = []
    for seed in SEEDS:
        meta = fit_development(dev, groups, dev_tuples, dev_residual, dev_baseline, gamma, seed)
        development_rows.append({
            "seed": seed, "gamma_retrieval": gamma,
            "selected_step": meta["selected_step"],
            **meta["validation_price"],
        })
    pd.DataFrame(development_rows).to_csv(OUT / "development_2024_by_seed.csv", index=False)

    dev_embeddings, dev_predictions = load_outputs("development", gamma, SEEDS, dev["all_dvf_f"])
    dev_context = make_context(dev, dev_residual, dev_baseline, dev_embeddings, dev_predictions, True)
    tuning, config, dev_fused, dev_neighbour = fusion.tune(dev_context)
    tuning.to_csv(OUT / "fusion_validation_2024_tuning.csv", index=False)
    dump(OUT / "FUSION_SELECTION.json", {
        "selected": config, "selection_population": "2024 validation only",
        "test_used_for_selection": False,
        "validation_metrics": tt.price_metrics(
            dev_context["queries"].log_sale_price.to_numpy(float), dev_fused,
            dev_context["queries"].sale_price_eur.to_numpy(float),
        ),
    })

    refit_tuples = pd.read_csv(TUPLES / "refit_2021_2024_tuples.csv.gz")
    test_rows = []
    for row in development_rows:
        seed = int(row["seed"])
        meta = fit_refit(
            refit, groups, refit_tuples, refit_residual, refit_baseline,
            gamma, seed, int(row["selected_step"]),
        )
        test_rows.append({"seed": seed, **meta["test_price"]})
    pd.DataFrame(test_rows).to_csv(OUT / "direct_test_2025_by_seed.csv", index=False)

    test_embeddings, test_predictions = load_outputs("refit", gamma, SEEDS, refit["all_dvf_f"])
    context = make_context(refit, refit_residual, refit_baseline, test_embeddings, test_predictions, False)
    prediction, order, effective_n = fusion.apply(context, config)
    neighbour = fusion.neighbour_residual(
        context, config["geo_beta"], config["intrinsic_beta"], config["k"],
        config["aggregation"], config["temperature"],
    )[0]
    truth = context["queries"].log_sale_price.to_numpy(float)
    euro = context["queries"].sale_price_eur.to_numpy(float)

    xgb = pd.read_csv(MAIN / "results/sensitivity/xgboost_price_test_predictions.csv.gz", dtype={"record_id": "string"})
    if not np.array_equal(xgb.record_id.astype(str), context["queries"].record_id.astype(str)):
        raise AssertionError("XGBoost alignment")

    references = {
        "Retrieval-aware residual Transformer (direct)": context["direct_prediction"],
        "Full-55 XGBoost": xgb.prediction.to_numpy(float),
    }
    # The appendix-only Prior-Fusion comparison is not a prerequisite for
    # the main RA-Direct / RA-Fusion experiment.
    if include_prior_comparator:
        old_fusion = pd.read_csv(
            OLD_FUSION / "results/test_2025_predictions.csv.gz",
            dtype={"record_id": "string"},
        )
        if not np.array_equal(old_fusion.record_id.astype(str), context["queries"].record_id.astype(str)):
            raise AssertionError("old fusion alignment")
        references["Original residual Transformer + retrieval"] = old_fusion.residual_retrieval_fusion.to_numpy(float)
    rows = [{
        "method": "Retrieval-aware residual Transformer + retrieval prediction",
        **tt.price_metrics(truth, prediction, euro),
    }]
    for name, value in references.items():
        rows.append({"method": name, **tt.price_metrics(truth, value, euro)})
    comparison = pd.DataFrame(rows).sort_values("log_mae")
    comparison.to_csv(OUT / "test_2025_comparison.csv", index=False)

    uncertainty = {
        name: fusion.bootstrap(context["queries"], value, prediction, f"candidate vs {name}")
        for name, value in references.items()
    }
    dump(OUT / "paired_spatial_block_uncertainty.json", uncertainty)

    output = context["queries"][[
        "record_id", "parcel_id", "observation_date", "property_type",
        "floor_area_m2", "room_count", "longitude", "latitude",
        "sale_price_eur", "log_sale_price",
    ]].copy()
    output["retrieval_aware_residual_direct"] = context["direct_prediction"]
    output["retrieved_residual_prediction"] = context["query_baseline"] + neighbour
    output["retrieval_aware_residual_fusion"] = prediction
    output["effective_neighbours"] = effective_n
    output.to_csv(OUT / "test_2025_predictions.csv.gz", index=False)

    stability = []
    weight = float(config["model_weight"])
    neighbour_price = context["query_baseline"] + neighbour
    for seed, direct in zip(SEEDS, test_predictions):
        fused = weight * direct + (1.0 - weight) * neighbour_price
        direct_metric = tt.price_metrics(truth, direct, euro)
        fused_metric = tt.price_metrics(truth, fused, euro)
        stability.append({
            "seed": seed, "direct_log_mae": direct_metric["log_mae"],
            "fused_log_mae": fused_metric["log_mae"],
            "log_mae_gain": direct_metric["log_mae"] - fused_metric["log_mae"],
        })
    pd.DataFrame(stability).to_csv(OUT / "test_2025_seed_stability.csv", index=False)

    dump(OUT / "run_manifest.json", {
        "status": "PASS", "protocol": str(HERE / "PROTOCOL.md"),
        "protocol_sha256": digest(HERE / "PROTOCOL.md"),
        "data": str(DATA), "data_sha256": digest(DATA),
        "selected_gamma_retrieval": gamma, "selected_fusion": config,
        "seeds": list(SEEDS), "development": "2021-2023 train; 2024 selection",
        "extension_test": "2021-2024 refit; retrospective 2025 temporal evaluation",
        "selection_in_this_script_uses_2024_only": True,
        "development_history_note": "Earlier exploration had inspected 2025; this is not an untouched test.",
        "include_prior_comparator": include_prior_comparator,
    })
    print("\nFUSION SELECTION\n", json.dumps(config, indent=2), flush=True)
    print("\n2025 COMPARISON\n", comparison.to_string(index=False), flush=True)
    print("\nUNCERTAINTY\n", json.dumps(uncertainty, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["build", "screen", "full", "all"], default="all")
    parser.add_argument("--include-prior-comparator", action="store_true",
                        help="Also compare an existing appendix Prior-Fusion prediction file.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    (
        data, groups, dev, dev_residual, dev_baseline, dev_audit,
        refit, refit_residual, refit_baseline, refit_audit,
    ) = prepare()
    dump(OUT / "market_index_audit.json", {"development": dev_audit, "refit": refit_audit})
    if args.phase in {"build", "all"}:
        build_phase(dev, dev_residual, refit, refit_residual)
    if args.phase in {"screen", "all"}:
        screen_phase(groups, dev, dev_residual, dev_baseline)
    if args.phase in {"full", "all"}:
        full_phase(
            data, groups, dev, dev_residual, dev_baseline,
            refit, refit_residual, refit_baseline,
            include_prior_comparator=args.include_prior_comparator,
        )


if __name__ == "__main__":
    main()
