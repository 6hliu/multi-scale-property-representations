#!/usr/bin/env python3
"""Train and evaluate a retrieval-aware temporal shared Transformer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import BallTree
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN = ROOT / "temporal_main_experiment_2026_09_14"
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN / "vendor"))
import run_temporal_transformer as tt  # noqa: E402
import run_temporal_retrieval as rr  # noqa: E402


DATA = MAIN / "data/temporal_model_ready_55.csv.gz"
MANIFEST = MAIN / "data/feature_manifest.json"
BASELINE = MAIN / "results/transformer"
BASELINE_RETRIEVAL = MAIN / "results/retrieval"
OUT = HERE / "results"
TRIPLETS = HERE / "triplets"
SEEDS = tuple(tt.SEEDS)
GAMMAS = (0.01, 0.03, 0.10)
LAMBDA_ENERGY = 0.20
STEPS = 512
INTERVAL = 128
BATCH = 1024
METRIC_BATCH = 256
TEMPERATURE = 0.10
NEGATIVES = 4
NEIGHBOURS = 768
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


def model_path(phase: str, seed: int, gamma: float) -> Path:
    return OUT / phase / gamma_tag(gamma) / f"seed_{seed}"


def build_tuples(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Build strictly earlier, same-type comparable tuples in frame order."""
    output = TRIPLETS / f"{label}_tuples.csv.gz"
    audit_path = TRIPLETS / f"{label}_audit.json"
    if output.exists() and audit_path.exists():
        return pd.read_csv(output)

    required = [
        "record_id", "property_type", "parcel_id", "observation_date",
        "latitude", "longitude", "floor_area_m2", "room_count",
        "log_sale_price", "months_since_2021",
    ]
    if frame[required].isna().any().any():
        raise AssertionError(f"missing tuple input in {label}")
    frame = frame.copy().reset_index(drop=True)
    frame.observation_date = pd.to_datetime(frame.observation_date)
    frame["day"] = frame.observation_date.to_numpy(dtype="datetime64[D]").astype(np.int64)
    frame["log_area"] = np.log(frame.floor_area_m2.to_numpy(float))
    frame["log_rooms"] = np.log1p(frame.room_count.to_numpy(float))
    radians = np.radians(frame[["latitude", "longitude"]].to_numpy(float))
    slopes = {}
    for property_type, part in frame.groupby("property_type"):
        slopes[property_type] = float(np.polyfit(part.months_since_2021, part.log_sale_price, 1)[0])

    rows: list[dict] = []
    parcel = frame.parcel_id.astype(str).to_numpy()
    day = frame.day.to_numpy(np.int64)
    log_price = frame.log_sale_price.to_numpy(float)
    month = frame.months_since_2021.to_numpy(float)
    log_area = frame.log_area.to_numpy(float)
    log_rooms = frame.log_rooms.to_numpy(float)

    for property_type in sorted(frame.property_type.unique()):
        global_indices = np.flatnonzero(frame.property_type.eq(property_type).to_numpy())
        tree = BallTree(radians[global_indices], metric="haversine")
        k = min(NEIGHBOURS + 1, len(global_indices))
        for start in range(0, len(global_indices), 1024):
            anchors = global_indices[start:start + 1024]
            distance_rad, local = tree.query(radians[anchors], k=k)
            neighbour_global = global_indices[local]
            distance_m = distance_rad * EARTH_RADIUS_M
            for offset, anchor in enumerate(anchors):
                candidates = neighbour_global[offset]
                distances = distance_m[offset]
                keep = (
                    (day[candidates] < day[anchor])
                    & (parcel[candidates] != parcel[anchor])
                )
                candidates = candidates[keep]
                distances = distances[keep]
                if len(candidates) < 16:
                    continue
                area_gap = np.abs(log_area[candidates] - log_area[anchor])
                room_gap = np.abs(log_rooms[candidates] - log_rooms[anchor])
                teacher = distances / 1000.0 + 2.0 * area_gap + 0.5 * room_gap
                order = np.argsort(teacher, kind="stable")
                candidates = candidates[order]
                distances = distances[order]
                area_gap = area_gap[order]
                room_gap = room_gap[order]
                teacher = teacher[order]
                pool_n = min(128, len(candidates))
                pool = candidates[:pool_n]
                adjusted = log_price[pool] + slopes[property_type] * (month[anchor] - month[pool])
                price_gap = np.abs(adjusted - log_price[anchor])
                close = np.flatnonzero(price_gap <= 0.15)
                positive_local = int(close[0]) if len(close) else int(np.argmin(price_gap + 0.02 * teacher[:pool_n]))
                positive = int(pool[positive_local])
                available = np.flatnonzero(pool != positive)
                hard_order = available[np.argsort(price_gap[available], kind="stable")[::-1]]
                if len(hard_order) < NEGATIVES:
                    continue
                negative_local = hard_order[:NEGATIVES]
                negatives = pool[negative_local].astype(int)
                row = {
                    "anchor_index": int(anchor),
                    "positive_index": positive,
                    "anchor_record_id": str(frame.record_id.iloc[anchor]),
                    "positive_record_id": str(frame.record_id.iloc[positive]),
                    "positive_price_gap": float(price_gap[positive_local]),
                    "positive_distance_m": float(distances[positive_local]),
                    "positive_area_gap": float(area_gap[positive_local]),
                    "positive_room_gap": float(room_gap[positive_local]),
                    "anchor_date": str(frame.observation_date.iloc[anchor].date()),
                    "positive_date": str(frame.observation_date.iloc[positive].date()),
                }
                for number, (negative, local_index) in enumerate(zip(negatives, negative_local), start=1):
                    row[f"negative_{number}_index"] = int(negative)
                    row[f"negative_{number}_record_id"] = str(frame.record_id.iloc[negative])
                    row[f"negative_{number}_price_gap"] = float(price_gap[local_index])
                rows.append(row)
        print(f"TUPLES {label} {property_type}: {sum(row['anchor_record_id'] in set(frame.loc[frame.property_type.eq(property_type), 'record_id'].astype(str)) for row in rows)}", flush=True)

    tuples = pd.DataFrame(rows).sort_values("anchor_index").reset_index(drop=True)
    if tuples.anchor_index.duplicated().any():
        raise AssertionError("duplicate tuple anchor")
    referenced = frame.iloc[tuples.positive_index.to_numpy(int)]
    anchors = frame.iloc[tuples.anchor_index.to_numpy(int)]
    if not (referenced.observation_date.to_numpy() < anchors.observation_date.to_numpy()).all():
        raise AssertionError("positive is not strictly earlier")
    if not (referenced.property_type.to_numpy() == anchors.property_type.to_numpy()).all():
        raise AssertionError("positive property type mismatch")
    if not (referenced.parcel_id.astype(str).to_numpy() != anchors.parcel_id.astype(str).to_numpy()).all():
        raise AssertionError("positive parcel mismatch")
    negative_columns = [f"negative_{number}_index" for number in range(1, NEGATIVES + 1)]
    for column in negative_columns:
        referenced = frame.iloc[tuples[column].to_numpy(int)]
        if not (referenced.observation_date.to_numpy() < anchors.observation_date.to_numpy()).all():
            raise AssertionError(f"{column} is not strictly earlier")
        if not (referenced.property_type.to_numpy() == anchors.property_type.to_numpy()).all():
            raise AssertionError(f"{column} property type mismatch")
        if not (referenced.parcel_id.astype(str).to_numpy() != anchors.parcel_id.astype(str).to_numpy()).all():
            raise AssertionError(f"{column} parcel mismatch")
    negative_gap = tuples[[f"negative_{number}_price_gap" for number in range(1, NEGATIVES + 1)]].min(axis=1)
    separation = float((tuples.positive_price_gap < negative_gap).mean())
    if separation < 0.90:
        raise AssertionError(f"weak tuple separation: {separation}")
    TRIPLETS.mkdir(parents=True, exist_ok=True)
    tuples.to_csv(output, index=False)
    dump(audit_path, {
        "status": "PASS",
        "label": label,
        "population_rows": len(frame),
        "eligible_anchor_rows": len(tuples),
        "eligible_rate": len(tuples) / len(frame),
        "strictly_earlier_positive_rate": 1.0,
        "strictly_earlier_negative_rate": 1.0,
        "different_parcel_rate": 1.0,
        "same_property_type_rate": 1.0,
        "positive_better_than_all_negatives_rate": separation,
        "positive_price_gap_median": float(tuples.positive_price_gap.median()),
        "minimum_negative_price_gap_median": float(negative_gap.median()),
        "positive_distance_m_median": float(tuples.positive_distance_m.median()),
        "training_labels_only": True,
        "candidate_neighbours": NEIGHBOURS,
        "negative_count": NEGATIVES,
        "temporal_slopes": slopes,
    })
    return tuples


def metric_loss(model, x: torch.Tensor, tuples: pd.DataFrame, rng: np.random.Generator):
    rows = rng.integers(len(tuples), size=METRIC_BATCH)
    anchor = torch.as_tensor(tuples.anchor_index.to_numpy(np.int64)[rows], device=tt.DEVICE)
    positive = torch.as_tensor(tuples.positive_index.to_numpy(np.int64)[rows], device=tt.DEVICE)
    negative_np = np.column_stack([
        tuples[f"negative_{number}_index"].to_numpy(np.int64)[rows]
        for number in range(1, NEGATIVES + 1)
    ])
    negative = torch.as_tensor(negative_np, device=tt.DEVICE)
    za = F.normalize(model.embed(x[anchor], "price"), dim=1)
    zp = F.normalize(model.embed(x[positive], "price"), dim=1)
    zn = F.normalize(model.embed(x[negative.reshape(-1)], "price"), dim=1).reshape(METRIC_BATCH, NEGATIVES, -1)
    positive_similarity = (za * zp).sum(1, keepdim=True)
    negative_similarity = torch.einsum("bd,bnd->bn", za, zn)
    logits = torch.cat([positive_similarity, negative_similarity], dim=1) / TEMPERATURE
    target = torch.zeros(METRIC_BATCH, dtype=torch.long, device=tt.DEVICE)
    loss = F.cross_entropy(logits, target)
    accuracy = (logits.argmax(1) == 0).float().mean()
    return loss, positive_similarity.mean(), negative_similarity.mean(), accuracy


def add_all_dvf(arrays: dict, data: pd.DataFrame) -> None:
    mask = data.source.eq("DVF").to_numpy()
    arrays["all_dvf_x"] = arrays["matrix"][mask]
    arrays["all_dvf_f"] = data.loc[mask].reset_index(drop=True)


def retrieval_context(arrays: dict, development: bool) -> dict:
    dvf = arrays["all_dvf_f"]
    if development:
        candidate_mask = dvf.temporal_split.eq("train").to_numpy()
        query_mask = dvf.temporal_split.eq("validation").to_numpy()
    else:
        candidate_mask = dvf.temporal_split.isin(["train", "validation"]).to_numpy()
        query_mask = dvf.temporal_split.eq("test").to_numpy()
    candidates = dvf.loc[candidate_mask].reset_index(drop=True)
    queries = dvf.loc[query_mask].reset_index(drop=True)
    geo_index, geo_distance = rr.geo_shortlist(candidates, queries, 300)
    return {
        "candidate_mask": candidate_mask,
        "query_mask": query_mask,
        "candidates": candidates,
        "queries": queries,
        "geo_index": geo_index,
        "geo_distance": geo_distance,
        "slopes": rr.temporal_slopes(candidates),
    }


def retrieval_prediction(embedding: np.ndarray, context: dict, shortlist=300, beta=0.0, k=10, adjust=True):
    ce = embedding[context["candidate_mask"]]
    qe = embedding[context["query_mask"]]
    gi = context["geo_index"][:, :shortlist]
    gd = context["geo_distance"][:, :shortlist]
    gd = gd / np.maximum(np.median(gd, axis=1, keepdims=True), 1.0)
    score = rr.embedding_distance(ce, qe, gi) + beta * gd
    order = rr.top_order(score, gi)
    prediction = rr.prediction_from_order(order, context["candidates"], context["queries"], k, adjust, context["slopes"])
    return prediction, order


def fixed_retrieval_metrics(model, arrays: dict, context: dict) -> dict:
    embedding = tt.predict(model, arrays["all_dvf_x"], "price", embeddings=True).astype(np.float32)
    prediction, _ = retrieval_prediction(embedding, context)
    return rr.metrics(context["queries"], prediction)


def baseline_reference(seed: int) -> dict:
    path = BASELINE / "development" / f"multi_seed{seed}_lambda{LAMBDA_ENERGY:g}" / "metadata.json"
    value = json.loads(path.read_text())
    return {
        "price_log_mae": float(value["validation_price"]["log_mae"]),
        "energy_macro_f1": float(value["validation_energy"]["macro_f1"]),
    }


def eligible(price: dict, energy: dict, reference: dict) -> bool:
    return (
        price["log_mae"] <= 1.03 * reference["price_log_mae"]
        and energy["macro_f1"] >= 0.97 * reference["energy_macro_f1"]
    )


def fit_development(arrays: dict, groups: dict, tuples: pd.DataFrame, context: dict, seed: int, gamma: float) -> dict:
    output = model_path("development", seed, gamma)
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    model = tt.TemporalGroupedTransformer(groups).to(tt.DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    huber = nn.HuberLoss()
    cross_entropy = nn.CrossEntropyLoss(weight=arrays["energy_class_weight"].to(tt.DEVICE))
    center = float(np.median(arrays["price_train_y"]))
    px = torch.from_numpy(arrays["price_train_x"]).to(tt.DEVICE)
    py = torch.from_numpy(arrays["price_train_y"]).to(tt.DEVICE)
    pr = torch.from_numpy(arrays["price_train_room"]).to(tt.DEVICE)
    ex = torch.from_numpy(arrays["energy_train_x"]).to(tt.DEVICE)
    ey = torch.from_numpy(arrays["energy_train_y"]).to(tt.DEVICE)
    rng_p = np.random.default_rng(seed + 10007)
    rng_e = np.random.default_rng(seed + 20011)
    rng_m = np.random.default_rng(seed + 30013)
    reference = baseline_reference(seed)
    history = []
    states = {}
    started = time.time()
    for step in range(1, STEPS + 1):
        model.train(); optimiser.zero_grad(set_to_none=True)
        pi = torch.as_tensor(rng_p.integers(len(py), size=BATCH), device=tt.DEVICE)
        ei = torch.as_tensor(rng_e.integers(len(ey), size=BATCH), device=tt.DEVICE)
        price_loss = huber(model.price(px[pi], pr[pi]), py[pi] - center)
        energy_loss = cross_entropy(model.energy(ex[ei]), ey[ei])
        comparable_loss, pos_sim, neg_sim, metric_acc = metric_loss(model, px, tuples, rng_m)
        total = price_loss + LAMBDA_ENERGY * energy_loss + gamma * comparable_loss
        total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimiser.step()
        if step % INTERVAL == 0:
            price = tt.validation_price(model, arrays, center)
            energy = tt.validation_energy(model, arrays, complete=False)
            retrieval = fixed_retrieval_metrics(model, arrays, context)
            passes = eligible(price, energy, reference)
            history.append({
                "step": step, "gamma_retrieval": gamma,
                "price_log_mae": price["log_mae"],
                "energy_macro_f1": energy["macro_f1"],
                "retrieval_log_mae": retrieval["log_mae"],
                "passes_noninferiority": passes,
                "batch_price_loss": float(price_loss.detach().cpu()),
                "batch_energy_loss": float(energy_loss.detach().cpu()),
                "batch_retrieval_loss": float(comparable_loss.detach().cpu()),
                "batch_positive_similarity": float(pos_sim.detach().cpu()),
                "batch_negative_similarity": float(neg_sim.detach().cpu()),
                "batch_retrieval_accuracy": float(metric_acc.detach().cpu()),
            })
            states[step] = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            print(
                f"DEV gamma={gamma:g} seed={seed} step={step} "
                f"price={price['log_mae']:.4f} energy={energy['macro_f1']:.4f} "
                f"retrieval={retrieval['log_mae']:.4f} pass={passes}", flush=True,
            )
    curve = pd.DataFrame(history)
    passing = curve.loc[curve.passes_noninferiority]
    fallback = passing.empty
    chosen = (curve if fallback else passing).sort_values(["retrieval_log_mae", "step"]).iloc[0]
    selected_step = int(chosen.step)
    model.load_state_dict(states[selected_step])
    reported_price = tt.validation_price(model, arrays, center)
    reported_energy = tt.validation_energy(model, arrays, complete=True)
    reported_retrieval = fixed_retrieval_metrics(model, arrays, context)
    embedding = tt.predict(model, arrays["all_dvf_x"], "price", embeddings=True).astype(np.float32)
    np.save(output / "dvf_price_embeddings.npy", embedding)
    arrays["all_dvf_f"][["record_id", "parcel_id", "observation_date", "temporal_split"]].to_csv(output / "dvf_embedding_index.csv.gz", index=False)
    torch.save({"state_dict": states[selected_step], "metadata": {"selected_step": selected_step}}, output / "model.pt")
    curve.to_csv(output / "validation_curve.csv", index=False)
    metadata = {
        "phase": "development", "seed": seed, "gamma_retrieval": gamma,
        "selected_step": selected_step, "fallback_used": bool(fallback),
        "passes_noninferiority": bool(chosen.passes_noninferiority),
        "validation_price": reported_price,
        "validation_energy": reported_energy,
        "validation_retrieval": reported_retrieval,
        "reference_validation": reference,
        "price_center": center,
        "elapsed_seconds": time.time() - started,
        "test_used_for_selection": False,
    }
    dump(metadata_path, metadata)
    return metadata


def fit_refit(arrays: dict, groups: dict, tuples: pd.DataFrame, seed: int, gamma: float, steps: int) -> dict:
    output = model_path("refit", seed, gamma)
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    model = tt.TemporalGroupedTransformer(groups).to(tt.DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    huber = nn.HuberLoss()
    cross_entropy = nn.CrossEntropyLoss(weight=arrays["energy_class_weight"].to(tt.DEVICE))
    center = float(np.median(arrays["price_train_y"]))
    px = torch.from_numpy(arrays["price_train_x"]).to(tt.DEVICE)
    py = torch.from_numpy(arrays["price_train_y"]).to(tt.DEVICE)
    pr = torch.from_numpy(arrays["price_train_room"]).to(tt.DEVICE)
    ex = torch.from_numpy(arrays["energy_train_x"]).to(tt.DEVICE)
    ey = torch.from_numpy(arrays["energy_train_y"]).to(tt.DEVICE)
    rng_p = np.random.default_rng(seed + 10007)
    rng_e = np.random.default_rng(seed + 20011)
    rng_m = np.random.default_rng(seed + 30013)
    started = time.time()
    for step in range(1, steps + 1):
        model.train(); optimiser.zero_grad(set_to_none=True)
        pi = torch.as_tensor(rng_p.integers(len(py), size=BATCH), device=tt.DEVICE)
        ei = torch.as_tensor(rng_e.integers(len(ey), size=BATCH), device=tt.DEVICE)
        price_loss = huber(model.price(px[pi], pr[pi]), py[pi] - center)
        energy_loss = cross_entropy(model.energy(ex[ei]), ey[ei])
        comparable_loss, _, _, _ = metric_loss(model, px, tuples, rng_m)
        total = price_loss + LAMBDA_ENERGY * energy_loss + gamma * comparable_loss
        total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimiser.step()
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"state_dict": state, "metadata": {"price_center": center}}, output / "model.pt")
    metadata = {
        "phase": "refit", "seed": seed, "gamma_retrieval": gamma,
        "training_steps": steps, "price_center": center,
        "elapsed_seconds": time.time() - started,
    }
    dump(metadata_path, metadata)
    return metadata


def load_model(path: Path, groups: dict):
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    model = tt.TemporalGroupedTransformer(groups).to(tt.DEVICE)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    return model


def evaluate_refit(arrays: dict, groups: dict, seed: int, gamma: float) -> dict:
    output = model_path("refit", seed, gamma)
    path = output / "test_metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    model = load_model(output, groups)
    price_prediction = tt.predict(model, arrays["price_evaluation_x"], "price", arrays["price_evaluation_room"]) + metadata["price_center"]
    energy_prediction = tt.predict(model, arrays["energy_evaluation_x"], "energy")
    price = tt.price_metrics(
        arrays["price_evaluation_y"], price_prediction,
        arrays["price_evaluation_f"].sale_price_eur.to_numpy(float),
    )
    energy = tt.energy_metrics(arrays["energy_evaluation_y"], energy_prediction)
    price_frame = arrays["price_evaluation_f"][["record_id", "observation_date", "parcel_id", "log_sale_price", "sale_price_eur"]].copy()
    price_frame["prediction"] = price_prediction
    price_frame.to_csv(output / "predictions_price.csv.gz", index=False)
    energy_frame = arrays["energy_evaluation_f"][["record_id", "observation_date", "energy_class"]].copy()
    energy_frame["truth"] = arrays["energy_evaluation_y"]
    energy_frame["prediction"] = energy_prediction
    energy_frame.to_csv(output / "predictions_energy.csv.gz", index=False)
    embedding = tt.predict(model, arrays["all_dvf_x"], "price", embeddings=True).astype(np.float32)
    np.save(output / "dvf_price_embeddings.npy", embedding)
    arrays["all_dvf_f"][["record_id", "parcel_id", "observation_date", "temporal_split"]].to_csv(output / "dvf_embedding_index.csv.gz", index=False)
    result = {"seed": seed, "gamma_retrieval": gamma, "price": price, "energy": energy}
    dump(path, result)
    return result


def load_embeddings(phase: str, gamma: float, dvf: pd.DataFrame) -> list[np.ndarray]:
    values = []
    for seed in SEEDS:
        path = model_path(phase, seed, gamma)
        index = pd.read_csv(path / "dvf_embedding_index.csv.gz", dtype={"record_id": "string"})
        if not np.array_equal(index.record_id.astype(str), dvf.record_id.astype(str)):
            raise AssertionError(f"embedding alignment {phase} {seed}")
        values.append(np.load(path / "dvf_price_embeddings.npy"))
    return values


def tune_retrieval(embeddings: list[np.ndarray], context: dict) -> tuple[pd.DataFrame, dict, np.ndarray]:
    rows = []
    predictions = {}
    for shortlist in (100, 300):
        for beta in (0.0, 0.25):
            orders = []
            for embedding in embeddings:
                ce = embedding[context["candidate_mask"]]
                qe = embedding[context["query_mask"]]
                gi = context["geo_index"][:, :shortlist]
                gd = context["geo_distance"][:, :shortlist]
                gd = gd / np.maximum(np.median(gd, axis=1, keepdims=True), 1.0)
                score = rr.embedding_distance(ce, qe, gi) + beta * gd
                orders.append(rr.top_order(score, gi))
            for k in (5, 10, 20):
                for adjust in (False, True):
                    prediction = np.mean([
                        rr.prediction_from_order(order, context["candidates"], context["queries"], k, adjust, context["slopes"])
                        for order in orders
                    ], axis=0)
                    metric = rr.metrics(context["queries"], prediction)
                    config = {"shortlist": shortlist, "beta": beta, "k": k, "temporal_adjustment": adjust}
                    rows.append({**config, **metric})
                    predictions[(shortlist, beta, k, adjust)] = prediction
    frame = pd.DataFrame(rows).sort_values(["log_mae", "shortlist", "beta", "k"])
    best = frame.iloc[0].to_dict()
    key = (int(best["shortlist"]), float(best["beta"]), int(best["k"]), bool(best["temporal_adjustment"]))
    return frame, best, predictions[key]


def apply_retrieval(embeddings: list[np.ndarray], context: dict, config: dict) -> np.ndarray:
    predictions = []
    for embedding in embeddings:
        prediction, _ = retrieval_prediction(
            embedding, context,
            shortlist=int(config["shortlist"]), beta=float(config["beta"]),
            k=int(config["k"]), adjust=bool(config["temporal_adjustment"]),
        )
        predictions.append(prediction)
    return np.mean(predictions, axis=0)


def spatial_bootstrap(frame: pd.DataFrame, reference: np.ndarray, candidate: np.ndarray, repetitions=20_000) -> dict:
    difference = np.abs(frame.log_sale_price.to_numpy(float) - reference) - np.abs(frame.log_sale_price.to_numpy(float) - candidate)
    block = rr.spatial_block(frame)
    unique = np.unique(block)
    by_block = {value: difference[block == value] for value in unique}
    rng = np.random.default_rng(20260924)
    draws = np.empty(repetitions)
    for index in range(repetitions):
        sample = rng.choice(unique, size=len(unique), replace=True)
        draws[index] = np.concatenate([by_block[value] for value in sample]).mean()
    return {
        "definition": "original shared absolute log error minus retrieval-aware shared absolute log error",
        "point_gain": float(difference.mean()),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "spatial_blocks": len(unique), "bootstrap_repetitions": repetitions,
        "positive_favours": "retrieval-aware shared representation",
    }


def prepare_all():
    manifest = json.loads(MANIFEST.read_text())
    groups = tt.build_groups(manifest)
    data = pd.read_csv(DATA, low_memory=False, dtype={"record_id": "string"})
    data.observation_date = pd.to_datetime(data.observation_date)
    raw = tt.raw_feature_frame(data, manifest["feature_names"])
    train = data.temporal_split.eq("train").to_numpy()
    validation = data.temporal_split.eq("validation").to_numpy()
    dev = tt.prepare(data, raw, train, validation)
    add_all_dvf(dev, data)
    refit_mask = data.temporal_split.isin(["train", "validation"]).to_numpy()
    test = data.temporal_split.eq("test").to_numpy()
    refit = tt.prepare(data, raw, refit_mask, test)
    add_all_dvf(refit, data)
    return data, groups, dev, refit


def build_phase(data: pd.DataFrame):
    development = data.loc[data.source.eq("DVF") & data.temporal_split.eq("train")].reset_index(drop=True)
    refit = data.loc[data.source.eq("DVF") & data.temporal_split.isin(["train", "validation"])].reset_index(drop=True)
    build_tuples(development, "development_2021_2023")
    build_tuples(refit, "refit_2021_2024")


def screen_phase(data, groups, dev):
    tuples = pd.read_csv(TRIPLETS / "development_2021_2023_tuples.csv.gz")
    context = retrieval_context(dev, development=True)
    rows = []
    for gamma in GAMMAS:
        result = fit_development(dev, groups, tuples, context, 42, gamma)
        rows.append({
            "gamma_retrieval": gamma,
            "selected_step": result["selected_step"],
            "passes_noninferiority": result["passes_noninferiority"],
            "fallback_used": result["fallback_used"],
            "validation_price_log_mae": result["validation_price"]["log_mae"],
            "validation_energy_macro_f1": result["validation_energy"]["macro_f1"],
            "validation_retrieval_log_mae": result["validation_retrieval"]["log_mae"],
        })
    frame = pd.DataFrame(rows)
    passing = frame.loc[frame.passes_noninferiority]
    fallback = passing.empty
    chosen = (frame if fallback else passing).sort_values(["validation_retrieval_log_mae", "gamma_retrieval"]).iloc[0]
    frame.to_csv(OUT / "gamma_screen_2024.csv", index=False)
    dump(OUT / "SELECTION.json", {
        "selected_gamma_retrieval": float(chosen.gamma_retrieval),
        "selection_seed": 42,
        "selection_population": "2024 validation only",
        "criterion": "minimum fixed-protocol retrieval Log-MAE subject to 3% price and energy non-inferiority",
        "fallback_used": bool(fallback),
        "test_used_for_selection": False,
    })
    print(frame.to_string(index=False), flush=True)
    print("SELECTED", float(chosen.gamma_retrieval), "fallback", fallback, flush=True)


def full_phase(data, groups, dev, refit):
    selection = json.loads((OUT / "SELECTION.json").read_text())
    gamma = float(selection["selected_gamma_retrieval"])
    dev_tuples = pd.read_csv(TRIPLETS / "development_2021_2023_tuples.csv.gz")
    refit_tuples = pd.read_csv(TRIPLETS / "refit_2021_2024_tuples.csv.gz")
    dev_context = retrieval_context(dev, development=True)
    development = {}
    for seed in SEEDS:
        development[seed] = fit_development(dev, groups, dev_tuples, dev_context, seed, gamma)
    dev_rows = [{
        "seed": seed, "gamma_retrieval": gamma,
        "selected_step": value["selected_step"],
        "passes_noninferiority": value["passes_noninferiority"],
        "fallback_used": value["fallback_used"],
        "price_log_mae": value["validation_price"]["log_mae"],
        "energy_macro_f1": value["validation_energy"]["macro_f1"],
        "retrieval_log_mae": value["validation_retrieval"]["log_mae"],
    } for seed, value in development.items()]
    pd.DataFrame(dev_rows).to_csv(OUT / "development_by_seed.csv", index=False)

    test_rows = []
    for seed in SEEDS:
        fit_refit(refit, groups, refit_tuples, seed, gamma, int(development[seed]["selected_step"]))
        result = evaluate_refit(refit, groups, seed, gamma)
        test_rows.append({
            "seed": seed,
            **{f"price_{key}": value for key, value in result["price"].items()},
            **{f"energy_{key}": value for key, value in result["energy"].items()},
        })
    test = pd.DataFrame(test_rows)
    test.to_csv(OUT / "direct_test_2025_by_seed.csv", index=False)
    summary = test.drop(columns="seed").agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    summary.to_csv(OUT / "direct_test_2025_summary.csv", index=False)

    dvf = dev["all_dvf_f"]
    dev_embeddings = load_embeddings("development", gamma, dvf)
    tuning, best, _ = tune_retrieval(dev_embeddings, dev_context)
    tuning.to_csv(OUT / "retrieval_validation_tuning.csv", index=False)
    dump(OUT / "RETRIEVAL_SELECTION.json", {
        "selected": best,
        "selection_population": "2024 validation only",
        "test_used_for_selection": False,
    })

    test_context = retrieval_context(refit, development=False)
    refit_embeddings = load_embeddings("refit", gamma, refit["all_dvf_f"])
    new_prediction = apply_retrieval(refit_embeddings, test_context, best)
    new_metrics = rr.metrics(test_context["queries"], new_prediction)
    old = pd.read_csv(BASELINE_RETRIEVAL / "test_predictions.csv.gz", dtype={"record_id": "string"})
    if not np.array_equal(old.record_id.astype(str), test_context["queries"].record_id.astype(str)):
        raise AssertionError("baseline retrieval alignment")
    comparison_rows = [
        {"method": "Retrieval-aware shared Transformer", **new_metrics},
        {"method": "Original shared Transformer", **rr.metrics(test_context["queries"], old["Multi Transformer embedding"].to_numpy(float))},
        {"method": "Geographic + intrinsic", **rr.metrics(test_context["queries"], old["Geographic + intrinsic"].to_numpy(float))},
        {"method": "Original validation ensemble", **rr.metrics(test_context["queries"], old["Validation ensemble"].to_numpy(float))},
    ]
    pd.DataFrame(comparison_rows).sort_values("log_mae").to_csv(OUT / "retrieval_test_2025_comparison.csv", index=False)
    predictions = test_context["queries"][["record_id", "parcel_id", "observation_date", "sale_price_eur", "log_sale_price"]].copy()
    predictions["retrieval_aware_shared"] = new_prediction
    predictions["original_shared"] = old["Multi Transformer embedding"].to_numpy(float)
    predictions["geographic_intrinsic"] = old["Geographic + intrinsic"].to_numpy(float)
    predictions.to_csv(OUT / "retrieval_test_2025_predictions.csv.gz", index=False)
    bootstrap = spatial_bootstrap(
        test_context["queries"],
        old["Multi Transformer embedding"].to_numpy(float), new_prediction,
    )
    dump(OUT / "paired_spatial_block_bootstrap.json", bootstrap)
    dump(OUT / "run_manifest.json", {
        "status": "PASS", "protocol": str(HERE / "PROTOCOL.md"),
        "protocol_sha256": digest(HERE / "PROTOCOL.md"),
        "data": str(DATA), "data_sha256": digest(DATA),
        "selected_gamma_retrieval": gamma, "lambda_energy": LAMBDA_ENERGY,
        "seeds": list(SEEDS), "temperature": TEMPERATURE,
        "negative_count": NEGATIVES,
        "development": "2021-2023 train; 2024 validation",
        "refit_test": "2021-2024 refit; protocol-frozen extension 2025",
        "test_used_for_selection": False,
    })
    print("\nDIRECT TEST\n", test.to_string(index=False), flush=True)
    print("\nRETRIEVAL TEST\n", pd.DataFrame(comparison_rows).sort_values("log_mae").to_string(index=False), flush=True)
    print("\nBOOTSTRAP\n", json.dumps(bootstrap, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["build", "screen", "full", "all"], default="all")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    data, groups, dev, refit = prepare_all()
    if args.phase in {"build", "all"}:
        build_phase(data)
    if args.phase in {"screen", "all"}:
        screen_phase(data, groups, dev)
    if args.phase in {"full", "all"}:
        full_phase(data, groups, dev, refit)


if __name__ == "__main__":
    main()
