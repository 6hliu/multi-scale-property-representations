#!/usr/bin/env python3
"""Validation-selected comparable-property retrieval for the temporal split."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent / "vendor"))
import torch
from sklearn.neighbors import BallTree


HERE = Path(__file__).resolve().parent
DATA = HERE / "data/temporal_model_ready_55.csv.gz"
TRANSFORMER = HERE / "results/transformer"
OUT = HERE / "results/retrieval"
EARTH_RADIUS_M = 6_371_008.8
SHORTLISTS = (100, 300)
KS = (5, 10, 20)
ALPHAS = (0.5, 1.0, 2.0)
BETAS = (0.0, 0.25)

sys.path.insert(0, str(HERE))
import run_temporal_transformer as tt  # noqa: E402
SEEDS = tt.SEEDS


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    y = frame.log_sale_price.to_numpy(float)
    euro = frame.sale_price_eur.to_numpy(float)
    return tt.price_metrics(y, prediction, euro)


def transform_from_json(raw: pd.DataFrame, path: Path) -> np.ndarray:
    meta = json.loads(path.read_text())
    median = pd.Series(meta["median"]); mean = pd.Series(meta["mean"]); std = pd.Series(meta["std"])
    matrix = ((raw[meta["features"]].fillna(median) - mean) / std).to_numpy(np.float32)
    if not np.isfinite(matrix).all(): raise AssertionError(path)
    return matrix


def load_transformer_model(path: Path, groups):
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    model = tt.TemporalGroupedTransformer(groups).to(tt.DEVICE)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    return model


def embedding_ensemble(data: pd.DataFrame, matrix: np.ndarray, groups, phase: str, mode: str, lam: float) -> list[np.ndarray]:
    dvf = data.source.eq("DVF").to_numpy(); result = []
    for seed in SEEDS:
        tag = f"{mode}_seed{seed}" if phase == "development" and mode == "price" else (
            f"multi_seed{seed}_lambda{lam:g}" if phase == "development" else f"{mode}_seed{seed}_lambda{(0.0 if mode == 'price' else lam):g}"
        )
        path = TRANSFORMER / phase / tag
        model = load_transformer_model(path, groups)
        result.append(tt.predict(model, matrix[dvf], "price", embeddings=True).astype(np.float32))
    return result


def geo_shortlist(candidates: pd.DataFrame, queries: pd.DataFrame, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.full((len(queries), maximum), -1, dtype=np.int32)
    distance = np.full((len(queries), maximum), np.inf, dtype=np.float32)
    for property_type in ("House", "Apartment"):
        ci = np.flatnonzero(candidates.property_type.eq(property_type).to_numpy())
        qi = np.flatnonzero(queries.property_type.eq(property_type).to_numpy())
        if not len(ci) or not len(qi): continue
        tree = BallTree(np.radians(candidates.iloc[ci][["latitude", "longitude"]].to_numpy(float)), metric="haversine")
        # Large apartment parcels can contain many historical sales. Request a
        # generous reserve before excluding the query parcel.
        request = min(maximum + 500, len(ci))
        angular, local = tree.query(np.radians(queries.iloc[qi][["latitude", "longitude"]].to_numpy(float)), k=request)
        mapped = ci[local]; metres = angular * EARTH_RADIUS_M
        candidate_parcels = candidates.parcel_id.astype(str).to_numpy()
        query_parcels = queries.parcel_id.astype(str).to_numpy()
        for row, global_query in enumerate(qi):
            keep = candidate_parcels[mapped[row]] != query_parcels[global_query]
            values = mapped[row][keep][:maximum]; values_distance = metres[row][keep][:maximum]
            index[global_query, :len(values)] = values
            distance[global_query, :len(values)] = values_distance
    if (index < 0).any(): raise AssertionError("insufficient eligible geographic candidates")
    return index, distance


def intrinsic_shortlist(candidates: pd.DataFrame, queries: pd.DataFrame, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    c = np.column_stack([np.log(candidates.floor_area_m2), np.log1p(candidates.room_count)])
    q = np.column_stack([np.log(queries.floor_area_m2), np.log1p(queries.room_count)])
    mean = c.mean(0); std = c.std(0); std[std < 1e-8] = 1
    c = (c - mean) / std; q = (q - mean) / std
    index = np.full((len(queries), maximum), -1, dtype=np.int32)
    distance = np.full((len(queries), maximum), np.inf, dtype=np.float32)
    for property_type in ("House", "Apartment"):
        ci = np.flatnonzero(candidates.property_type.eq(property_type).to_numpy())
        qi = np.flatnonzero(queries.property_type.eq(property_type).to_numpy())
        tree = BallTree(c[ci], metric="euclidean")
        request = min(maximum + 500, len(ci)); dist, local = tree.query(q[qi], k=request); mapped = ci[local]
        candidate_parcels = candidates.parcel_id.astype(str).to_numpy(); query_parcels = queries.parcel_id.astype(str).to_numpy()
        for row, global_query in enumerate(qi):
            keep = candidate_parcels[mapped[row]] != query_parcels[global_query]
            values = mapped[row][keep][:maximum]; values_distance = dist[row][keep][:maximum]
            index[global_query, :len(values)] = values; distance[global_query, :len(values)] = values_distance
    if (index < 0).any(): raise AssertionError("insufficient eligible intrinsic candidates")
    return index, distance


def temporal_slopes(candidates: pd.DataFrame) -> dict[str, float]:
    result = {}
    for property_type in ("House", "Apartment"):
        frame = candidates.loc[candidates.property_type.eq(property_type)]
        x = frame.months_since_2021.to_numpy(float); y = frame.log_sale_price.to_numpy(float)
        result[property_type] = float(np.polyfit(x, y, 1)[0])
    return result


def top_order(score: np.ndarray, indices: np.ndarray, maximum_k: int = 20) -> np.ndarray:
    chosen = np.argpartition(score, kth=maximum_k - 1, axis=1)[:, :maximum_k]
    chosen_score = np.take_along_axis(score, chosen, axis=1)
    order = np.argsort(chosen_score, axis=1)
    return np.take_along_axis(indices, np.take_along_axis(chosen, order, axis=1), axis=1)


def prediction_from_order(order: np.ndarray, candidates: pd.DataFrame, queries: pd.DataFrame, k: int, adjust: bool, slopes: dict[str, float]) -> np.ndarray:
    selected = order[:, :k]
    values = candidates.log_sale_price.to_numpy(float)[selected]
    if adjust:
        candidate_month = candidates.months_since_2021.to_numpy(float)[selected]
        query_month = queries.months_since_2021.to_numpy(float)[:, None]
        slope = queries.property_type.map(slopes).to_numpy(float)[:, None]
        values = values + slope * (query_month - candidate_month)
    return np.median(values, axis=1)


def intrinsic_distance(candidates: pd.DataFrame, queries: pd.DataFrame, shortlist: np.ndarray) -> np.ndarray:
    c = np.column_stack([np.log(candidates.floor_area_m2), np.log1p(candidates.room_count)])
    q = np.column_stack([np.log(queries.floor_area_m2), np.log1p(queries.room_count)])
    mean = c.mean(0); std = c.std(0); std[std < 1e-8] = 1
    c = (c - mean) / std; q = (q - mean) / std
    return np.sqrt(((c[shortlist] - q[:, None, :]) ** 2).sum(2)).astype(np.float32)


def raw_distance(candidate_matrix: np.ndarray, query_matrix: np.ndarray, shortlist: np.ndarray) -> np.ndarray:
    output = np.empty(shortlist.shape, dtype=np.float32)
    for start in range(0, len(shortlist), 512):
        end = min(start + 512, len(shortlist))
        difference = candidate_matrix[shortlist[start:end]] - query_matrix[start:end, None, :]
        output[start:end] = np.sqrt(np.mean(difference * difference, axis=2))
    return output


def embedding_distance(candidate_embedding: np.ndarray, query_embedding: np.ndarray, shortlist: np.ndarray) -> np.ndarray:
    c = candidate_embedding / np.maximum(np.linalg.norm(candidate_embedding, axis=1, keepdims=True), 1e-8)
    q = query_embedding / np.maximum(np.linalg.norm(query_embedding, axis=1, keepdims=True), 1e-8)
    output = np.empty(shortlist.shape, dtype=np.float32)
    for start in range(0, len(shortlist), 1024):
        end = min(start + 1024, len(shortlist))
        output[start:end] = 1 - np.einsum("qkd,qd->qk", c[shortlist[start:end]], q[start:end])
    return output


def select_methods(candidates, queries, candidate_matrix, query_matrix, candidate_embeddings, query_embeddings):
    maximum = max(SHORTLISTS)
    geo_index, geo_distance = geo_shortlist(candidates, queries, maximum)
    intrinsic_index, intrinsic_score = intrinsic_shortlist(candidates, queries, maximum)
    intrinsic_on_geo = intrinsic_distance(candidates, queries, geo_index)
    slopes = temporal_slopes(candidates); rows = []; predictions = {}

    method_configs = []
    method_configs.append(("Geographic", {}, top_order(geo_distance, geo_index)))
    method_configs.append(("Intrinsic", {}, top_order(intrinsic_score, intrinsic_index)))
    for shortlist in SHORTLISTS:
        gi = geo_index[:, :shortlist]; gd = geo_distance[:, :shortlist]
        gd = gd / np.maximum(np.median(gd, axis=1, keepdims=True), 1.0)
        ins = intrinsic_on_geo[:, :shortlist]
        for alpha in ALPHAS:
            method_configs.append(("Geographic + intrinsic", {"shortlist": shortlist, "alpha": alpha}, top_order(gd + alpha * ins, gi)))
        method_configs.append(("Raw full features", {"shortlist": shortlist}, top_order(raw_distance(candidate_matrix, query_matrix, gi), gi)))
        for mode in ("price", "multi"):
            seed_orders = []
            for candidate_embedding, query_embedding in zip(candidate_embeddings[mode], query_embeddings[mode]):
                ed = embedding_distance(candidate_embedding, query_embedding, gi)
                for beta in BETAS:
                    seed_orders.append((beta, top_order(ed + beta * gd, gi)))
            for beta in BETAS:
                orders = [order for value, order in seed_orders if value == beta]
                method_configs.append((f"{mode.title()} Transformer embedding", {"shortlist": shortlist, "beta": beta}, orders))

    for method, config, order_or_orders in method_configs:
        for k in KS:
            for adjust in (False, True):
                if isinstance(order_or_orders, list):
                    prediction = np.mean([prediction_from_order(order, candidates, queries, k, adjust, slopes) for order in order_or_orders], axis=0)
                else:
                    prediction = prediction_from_order(order_or_orders, candidates, queries, k, adjust, slopes)
                metric = metrics(queries, prediction)
                key = (method, tuple(sorted(config.items())), k, adjust)
                predictions[key] = prediction
                rows.append({"method": method, **config, "k": k, "temporal_adjustment": adjust, **metric})
    frame = pd.DataFrame(rows)
    selected = frame.sort_values(["log_mae", "method"]).groupby("method", as_index=False).first()

    # A late ensemble may combine only two already-selected validation methods.
    geo_row = selected.loc[selected.method.eq("Geographic + intrinsic")].iloc[0]
    shared_row = selected.loc[selected.method.eq("Multi Transformer embedding")].iloc[0]
    def key_from_row(row):
        config = {key: row[key] for key in ("shortlist", "alpha", "beta") if key in row and pd.notna(row[key])}
        return (row.method, tuple(sorted(config.items())), int(row.k), bool(row.temporal_adjustment))
    geo_prediction = predictions[key_from_row(geo_row)]; shared_prediction = predictions[key_from_row(shared_row)]
    ensemble_rows = []
    for weight in (0.25, 0.50, 0.75):
        prediction = (1 - weight) * geo_prediction + weight * shared_prediction
        ensemble_rows.append({"method": "Validation ensemble", "transformer_weight": weight, "k": np.nan, "temporal_adjustment": np.nan, **metrics(queries, prediction)})
    ensemble_frame = pd.DataFrame(ensemble_rows)
    best_ensemble = ensemble_frame.sort_values("log_mae").iloc[[0]]
    return pd.concat([frame, ensemble_frame], ignore_index=True), pd.concat([selected, best_ensemble], ignore_index=True)


def apply_selected(selected, candidates, queries, candidate_matrix, query_matrix, candidate_embeddings, query_embeddings, return_orders=False):
    maximum = max(SHORTLISTS); geo_index, geo_distance = geo_shortlist(candidates, queries, maximum)
    intrinsic_index, intrinsic_score = intrinsic_shortlist(candidates, queries, maximum)
    intrinsic_on_geo = intrinsic_distance(candidates, queries, geo_index); slopes = temporal_slopes(candidates)
    output = {}; retrieval_orders = {}; method_k = {}; selected_map = {row.method: row for _, row in selected.iterrows()}
    for method, row in selected_map.items():
        if method == "Validation ensemble": continue
        k = int(row.k); adjust = bool(row.temporal_adjustment)
        if method == "Geographic": order = top_order(geo_distance, geo_index)
        elif method == "Intrinsic": order = top_order(intrinsic_score, intrinsic_index)
        else:
            shortlist = int(row.shortlist); gi = geo_index[:, :shortlist]; gd = geo_distance[:, :shortlist]
            gd = gd / np.maximum(np.median(gd, axis=1, keepdims=True), 1.0)
            if method == "Geographic + intrinsic": order = top_order(gd + float(row.alpha) * intrinsic_on_geo[:, :shortlist], gi)
            elif method == "Raw full features": order = top_order(raw_distance(candidate_matrix, query_matrix, gi), gi)
            else:
                mode = "price" if method.startswith("Price") else "multi"; orders = []
                for ce, qe in zip(candidate_embeddings[mode], query_embeddings[mode]):
                    score = embedding_distance(ce, qe, gi) + float(row.beta) * gd
                    orders.append(top_order(score, gi))
                output[method] = np.mean([prediction_from_order(value, candidates, queries, k, adjust, slopes) for value in orders], axis=0)
                retrieval_orders[method] = orders
                method_k[method] = k
                continue
        output[method] = prediction_from_order(order, candidates, queries, k, adjust, slopes)
        retrieval_orders[method] = [order]
        method_k[method] = k
    ensemble = selected_map["Validation ensemble"]
    weight = float(ensemble.transformer_weight)
    output["Validation ensemble"] = (1 - weight) * output["Geographic + intrinsic"] + weight * output["Multi Transformer embedding"]
    if return_orders:
        return output, retrieval_orders, method_k
    return output


def spatial_block(frame: pd.DataFrame) -> np.ndarray:
    reference = math.radians(float(frame.latitude.median()))
    x = EARTH_RADIUS_M * np.radians(frame.longitude.to_numpy()) * math.cos(reference)
    y = EARTH_RADIUS_M * np.radians(frame.latitude.to_numpy())
    return np.char.add(np.floor(x / 3000).astype(int).astype(str), np.char.add("_", np.floor(y / 3000).astype(int).astype(str)))


def paired_bootstrap(frame, a, b, repetitions=2000):
    difference = np.abs(frame.log_sale_price.to_numpy() - a) - np.abs(frame.log_sale_price.to_numpy() - b)
    block = spatial_block(frame); unique = np.unique(block); by_block = {value: difference[block == value] for value in unique}
    rng = np.random.default_rng(20260914); draws = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws.append(float(np.concatenate([by_block[value] for value in sampled]).mean()))
    return {"definition": "geographic+intrinsic absolute log error minus shared-Transformer absolute log error", "point_difference": float(difference.mean()), "ci95_lower": float(np.quantile(draws, 0.025)), "ci95_upper": float(np.quantile(draws, 0.975)), "spatial_blocks": int(len(unique)), "bootstrap_repetitions": repetitions}


def main():
    torch.set_num_threads(4); OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "data/feature_manifest.json").read_text()); groups = tt.build_groups(manifest)
    selection = json.loads((TRANSFORMER / "SELECTION.json").read_text()); lam = float(selection["selected_lambda_energy"])
    data = pd.read_csv(DATA, low_memory=False, dtype={"record_id": "string"}); data.observation_date = pd.to_datetime(data.observation_date)
    raw = tt.raw_feature_frame(data, manifest["feature_names"]); dvf = data.loc[data.source.eq("DVF")].reset_index(drop=True); dvf_raw = raw.loc[data.source.eq("DVF")].reset_index(drop=True)

    dev_matrix = transform_from_json(dvf_raw, TRANSFORMER / "development_preprocessing.json")
    dev_embeddings = {mode: embedding_ensemble(data, transform_from_json(raw, TRANSFORMER / "development_preprocessing.json"), groups, "development", mode, lam) for mode in ("price", "multi")}
    dev_candidate_mask = dvf.temporal_split.eq("train").to_numpy(); dev_query_mask = dvf.temporal_split.eq("validation").to_numpy()
    tuning, selected = select_methods(
        dvf.loc[dev_candidate_mask].reset_index(drop=True), dvf.loc[dev_query_mask].reset_index(drop=True),
        dev_matrix[dev_candidate_mask], dev_matrix[dev_query_mask],
        {mode: [value[dev_candidate_mask] for value in dev_embeddings[mode]] for mode in dev_embeddings},
        {mode: [value[dev_query_mask] for value in dev_embeddings[mode]] for mode in dev_embeddings},
    )
    tuning.to_csv(OUT / "validation_tuning.csv", index=False); selected.to_csv(OUT / "RETRIEVAL_SELECTION.csv", index=False)
    tt.dump(OUT / "SELECTION.json", {"selection_population": "2024 validation only", "candidate_period": "2021-2023", "criterion": "minimum validation Log-MAE independently per method; ensemble weight selected last", "test_used": False, "selected": selected.to_dict("records")})

    # Only after selection has been persisted do we construct the 2025 test evaluation.
    refit_matrix = transform_from_json(dvf_raw, TRANSFORMER / "refit_preprocessing.json")
    refit_embeddings = {}
    for mode in ("price", "multi"):
        values = []
        for seed in SEEDS:
            tag = f"{mode}_seed{seed}_lambda{(0.0 if mode == 'price' else lam):g}"
            value = np.load(TRANSFORMER / "refit" / tag / "dvf_price_embeddings.npy")
            index = pd.read_csv(TRANSFORMER / "refit" / tag / "dvf_embedding_index.csv.gz", dtype={"record_id": "string"})
            if not np.array_equal(index.record_id.to_numpy(), dvf.record_id.to_numpy()): raise AssertionError(tag)
            values.append(value)
        refit_embeddings[mode] = values
    candidate_mask = dvf.temporal_split.isin(["train", "validation"]).to_numpy(); query_mask = dvf.temporal_split.eq("test").to_numpy()
    candidates = dvf.loc[candidate_mask].reset_index(drop=True); queries = dvf.loc[query_mask].reset_index(drop=True)
    predictions = apply_selected(
        selected, candidates, queries, refit_matrix[candidate_mask], refit_matrix[query_mask],
        {mode: [value[candidate_mask] for value in refit_embeddings[mode]] for mode in refit_embeddings},
        {mode: [value[query_mask] for value in refit_embeddings[mode]] for mode in refit_embeddings},
    )
    rows = []
    output = queries[["record_id", "parcel_id", "observation_date", "sale_price_eur", "log_sale_price"]].copy()
    for method, prediction in predictions.items():
        rows.append({"method": method, "n_test": len(queries), **metrics(queries, prediction)})
        output[method] = prediction
    pd.DataFrame(rows).sort_values("log_mae").to_csv(OUT / "test_results.csv", index=False)
    output.to_csv(OUT / "test_predictions.csv.gz", index=False)
    bootstrap = paired_bootstrap(queries, predictions["Geographic + intrinsic"], predictions["Multi Transformer embedding"])
    tt.dump(OUT / "paired_spatial_block_bootstrap.json", bootstrap)
    tt.dump(OUT / "run_manifest.json", {
        "status": "PASS", "development_candidates": "DVF 2021-2023", "development_queries": "DVF 2024",
        "refit_candidates": "DVF 2021-2024", "test_queries": "DVF 2025", "same_property_type": True,
        "same_parcel_excluded": True, "future_candidate_excluded": True, "price_aggregation": "median of k comparable log prices",
        "temporal_adjustment": "optional property-type-specific linear log-price slope; selected on validation",
        "transformer_seed_ensemble": list(SEEDS), "test_used_for_selection": False,
    })
    print(pd.DataFrame(rows).sort_values("log_mae").to_string(index=False)); print(json.dumps(bootstrap, indent=2))


if __name__ == "__main__":
    main()
