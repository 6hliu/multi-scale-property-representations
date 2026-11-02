#!/usr/bin/env python3
"""Five-seed residual Transformer refit, 2025 evaluation and RQ3 retrieval."""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN = ROOT / "temporal_main_experiment_2026_09_14"
SCREEN = ROOT / "temporal_residual_screen_2026_09_23"
OUT = HERE / "results"
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(SCREEN))
sys.path.insert(0, str(MAIN / "vendor"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn
from xgboost import XGBRegressor

import run_temporal_retrieval as tr
import run_temporal_transformer as tt
from run_residual_screen import market_index


DATA = MAIN / "data/temporal_model_ready_55.csv.gz"
MANIFEST = MAIN / "data/feature_manifest.json"
SEEDS = tt.SEEDS
LAMBDA_ENERGY = 0.20
STEPS = 512
INTERVAL = 128
BATCH = 1024
EARTH_RADIUS_M = 6_371_008.8
BOOTSTRAP_REPETITIONS = 20_000
BOOTSTRAP_SEED = 20260923


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def price_metric(y, prediction, euro) -> dict:
    return tt.price_metrics(np.asarray(y), np.asarray(prediction), np.asarray(euro))


def energy_metric(y, prediction) -> dict:
    return {
        "macro_f1": float(f1_score(y, prediction, labels=list(range(7)), average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(y, prediction, labels=list(range(7)), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
    }


def values_for(frame: pd.DataFrame, lookup: pd.Series) -> np.ndarray:
    return lookup.loc[frame.record_id.astype(str)].to_numpy(np.float32)


def residual_lookup(price_frame: pd.DataFrame, index: np.ndarray) -> tuple[pd.Series, pd.Series]:
    baseline = index + np.log(price_frame.floor_area_m2.to_numpy(float))
    residual = price_frame.log_sale_price.to_numpy(float) - baseline
    key = price_frame.record_id.astype(str)
    return pd.Series(residual, index=key), pd.Series(baseline, index=key)


def evaluate_price(model, arrays, residual: pd.Series, baseline: pd.Series, center: float) -> tuple[dict, np.ndarray]:
    residual_prediction = (
        tt.predict(model, arrays["price_evaluation_x"], "price", arrays["price_evaluation_room"])
        + center
    )
    prediction = residual_prediction + values_for(arrays["price_evaluation_f"], baseline)
    metric = price_metric(
        arrays["price_evaluation_y"],
        prediction,
        arrays["price_evaluation_f"].sale_price_eur.to_numpy(float),
    )
    return metric, prediction


def fit_development(
    arrays: dict,
    groups: dict[str, list[int]],
    residual: pd.Series,
    baseline: pd.Series,
    mode: str,
    seed: int,
    price_reference: float | None,
    energy_reference: float | None,
) -> dict:
    tag = f"{mode}_seed{seed}_lambda{(LAMBDA_ENERGY if mode == 'multi' else 0):g}"
    output = OUT / "development" / tag
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = tt.TemporalGroupedTransformer(groups).to(tt.DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    huber = nn.HuberLoss()
    cross_entropy = nn.CrossEntropyLoss(weight=arrays["energy_class_weight"].to(tt.DEVICE))

    train_residual = values_for(arrays["price_train_f"], residual)
    center = float(np.median(train_residual))
    px = torch.from_numpy(arrays["price_train_x"]).to(tt.DEVICE)
    pr = torch.from_numpy(arrays["price_train_room"]).to(tt.DEVICE)
    py = torch.from_numpy(train_residual).to(tt.DEVICE)
    ex = torch.from_numpy(arrays["energy_train_x"]).to(tt.DEVICE)
    ey = torch.from_numpy(arrays["energy_train_y"]).to(tt.DEVICE)
    rng_p = np.random.default_rng(seed + 10007)
    rng_e = np.random.default_rng(seed + 20011)
    best_score = -np.inf
    best_state = None
    best = None
    history = []
    started = time.time()
    for step in range(1, STEPS + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        p_index = torch.from_numpy(rng_p.integers(len(py), size=BATCH)).to(tt.DEVICE)
        price_loss = huber(model.price(px[p_index], pr[p_index]), py[p_index] - center)
        if mode == "price":
            total = price_loss
        else:
            e_index = torch.from_numpy(rng_e.integers(len(ey), size=BATCH)).to(tt.DEVICE)
            energy_loss = cross_entropy(model.energy(ex[e_index]), ey[e_index])
            total = price_loss + LAMBDA_ENERGY * energy_loss
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()

        if step % INTERVAL == 0:
            pm, _ = evaluate_price(model, arrays, residual, baseline, center)
            em = tt.validation_energy(model, arrays, complete=False) if mode == "multi" else None
            if mode == "price":
                score = -pm["log_mae"]
            else:
                price_gain = (price_reference - pm["log_mae"]) / price_reference
                energy_gain = (em["macro_f1"] - energy_reference) / max(energy_reference, 1e-8)
                score = float(min(price_gain, energy_gain))
            row = {"step": step, "selection_score": score, **{f"price_{k}": v for k, v in pm.items()}}
            if em is not None:
                row.update({f"energy_{k}": v for k, v in em.items()})
            history.append(row)
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                best = {"selected_step": step, "validation_price": pm, "selection_energy": em}
            print(
                f"DEV {tag} step={step} price={pm['log_mae']:.6f}"
                + (f" energy={em['macro_f1']:.6f}" if em else ""),
                flush=True,
            )
    if best_state is None or best is None:
        raise AssertionError(tag)
    model.load_state_dict(best_state)
    reported_price, prediction = evaluate_price(model, arrays, residual, baseline, center)
    reported_energy = tt.validation_energy(model, arrays, complete=True) if mode == "multi" else None
    metadata = {
        "mode": mode,
        "seed": seed,
        "lambda_energy": LAMBDA_ENERGY if mode == "multi" else 0.0,
        "selected_step": best["selected_step"],
        "selection_score": best_score,
        "validation_price": reported_price,
        "validation_energy": reported_energy,
        "selection_energy": best["selection_energy"],
        "price_center": center,
        "elapsed_seconds": time.time() - started,
    }
    torch.save({"state_dict": best_state, "metadata": metadata}, output / "model.pt")
    pd.DataFrame(history).to_csv(output / "validation_curve.csv", index=False)
    pred = arrays["price_evaluation_f"][["record_id", "observation_date", "longitude", "latitude", "sale_price_eur", "log_sale_price"]].copy()
    pred["prediction"] = prediction
    pred.to_csv(output / "validation_price_predictions.csv.gz", index=False)
    dump(metadata_path, metadata)
    return metadata


def fit_refit(
    arrays: dict,
    groups: dict[str, list[int]],
    residual: pd.Series,
    baseline: pd.Series,
    mode: str,
    seed: int,
    steps: int,
) -> dict:
    tag = f"{mode}_seed{seed}_lambda{(LAMBDA_ENERGY if mode == 'multi' else 0):g}"
    output = OUT / "refit" / tag
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = tt.TemporalGroupedTransformer(groups).to(tt.DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    huber = nn.HuberLoss()
    cross_entropy = nn.CrossEntropyLoss(weight=arrays["energy_class_weight"].to(tt.DEVICE))
    train_residual = values_for(arrays["price_train_f"], residual)
    center = float(np.median(train_residual))
    px = torch.from_numpy(arrays["price_train_x"]).to(tt.DEVICE)
    pr = torch.from_numpy(arrays["price_train_room"]).to(tt.DEVICE)
    py = torch.from_numpy(train_residual).to(tt.DEVICE)
    ex = torch.from_numpy(arrays["energy_train_x"]).to(tt.DEVICE)
    ey = torch.from_numpy(arrays["energy_train_y"]).to(tt.DEVICE)
    rng_p = np.random.default_rng(seed + 10007)
    rng_e = np.random.default_rng(seed + 20011)
    started = time.time()
    for step in range(1, steps + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        p_index = torch.from_numpy(rng_p.integers(len(py), size=BATCH)).to(tt.DEVICE)
        price_loss = huber(model.price(px[p_index], pr[p_index]), py[p_index] - center)
        if mode == "price":
            total = price_loss
        else:
            e_index = torch.from_numpy(rng_e.integers(len(ey), size=BATCH)).to(tt.DEVICE)
            energy_loss = cross_entropy(model.energy(ex[e_index]), ey[e_index])
            total = price_loss + LAMBDA_ENERGY * energy_loss
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
    price_result, price_prediction = evaluate_price(model, arrays, residual, baseline, center)
    energy_prediction = None
    energy_result = None
    if mode == "multi":
        energy_prediction = tt.predict(model, arrays["energy_evaluation_x"], "energy")
        energy_result = energy_metric(arrays["energy_evaluation_y"], energy_prediction)
    metadata = {
        "mode": mode,
        "seed": seed,
        "steps": steps,
        "lambda_energy": LAMBDA_ENERGY if mode == "multi" else 0.0,
        "price_center": center,
        "test_price": price_result,
        "test_energy": energy_result,
        "elapsed_seconds": time.time() - started,
    }
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "metadata": metadata}, output / "model.pt")
    price_frame = arrays["price_evaluation_f"][["record_id", "entity_id", "observation_date", "longitude", "latitude", "sale_price_eur", "log_sale_price"]].copy()
    price_frame["prediction"] = price_prediction
    price_frame.to_csv(output / "predictions_price.csv.gz", index=False)
    if mode == "multi":
        energy_frame = arrays["energy_evaluation_f"][["record_id", "entity_id", "observation_date", "longitude", "latitude", "energy_class"]].copy()
        energy_frame["truth"] = arrays["energy_evaluation_y"]
        energy_frame["prediction"] = energy_prediction
        energy_frame.to_csv(output / "predictions_energy.csv.gz", index=False)
    embedding = tt.predict(model, arrays["all_dvf_x"], "price", embeddings=True)
    np.save(output / "dvf_price_embeddings.npy", embedding.astype(np.float32))
    arrays["all_dvf_f"][["record_id", "parcel_id", "observation_date", "temporal_split"]].to_csv(output / "dvf_embedding_index.csv.gz", index=False)
    dump(metadata_path, metadata)
    return metadata


def spatial_blocks(frame: pd.DataFrame) -> np.ndarray:
    reference = math.radians(float(frame.latitude.median()))
    x = EARTH_RADIUS_M * np.radians(frame.longitude.to_numpy(float)) * math.cos(reference)
    y = EARTH_RADIUS_M * np.radians(frame.latitude.to_numpy(float))
    return np.char.add(np.floor(x / 3000).astype(int).astype(str), np.char.add("_", np.floor(y / 3000).astype(int).astype(str)))


def bootstrap(difference: np.ndarray, blocks: np.ndarray) -> dict:
    unique, codes = np.unique(blocks, return_inverse=True)
    sums = np.bincount(codes, weights=difference, minlength=len(unique))
    counts = np.bincount(codes, minlength=len(unique))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPETITIONS, float)
    for i in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.integers(0, len(unique), len(unique))
        draws[i] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "point_gain": float(difference.mean()),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "blocks": int(len(unique)),
        "repetitions": BOOTSTRAP_REPETITIONS,
    }


def load_original_predictions(mode: str, seed: int, task: str) -> pd.DataFrame:
    lam = 0.0 if mode == "price" else LAMBDA_ENERGY
    path = MAIN / "results/transformer/refit" / f"{mode}_seed{seed}_lambda{lam:g}" / f"predictions_{task}.csv.gz"
    return pd.read_csv(path, dtype={"record_id": "string"})


def run_xgboost(
    data: pd.DataFrame,
    feature_names: list[str],
    residual: pd.Series,
    baseline: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    raw = tt.raw_feature_frame(data, feature_names)
    raw["log1p_room_count"] = np.log1p(pd.to_numeric(data.room_count, errors="coerce"))
    price = data.source.eq("DVF")
    refit = price & data.temporal_split.isin(["train", "validation"])
    test = price & data.temporal_split.eq("test")
    train_frame = data.loc[refit]
    test_frame = data.loc[test]
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_estimators=446,
        random_state=42,
        n_jobs=4,
        tree_method="hist",
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=5,
    )
    model.fit(raw.loc[refit], values_for(train_frame, residual), verbose=False)
    prediction = model.predict(raw.loc[test]) + values_for(test_frame, baseline)
    result = price_metric(test_frame.log_sale_price, prediction, test_frame.sale_price_eur)
    output = test_frame[["record_id", "observation_date", "longitude", "latitude", "sale_price_eur", "log_sale_price"]].copy()
    output["prediction"] = prediction
    output.to_csv(OUT / "xgboost_residual_predictions_2025.csv.gz", index=False)
    model.get_booster().save_model(OUT / "xgboost_residual_full55.ubj")
    return result, output


def run_retrieval(data: pd.DataFrame, embeddings: list[np.ndarray]) -> tuple[dict, pd.DataFrame, dict]:
    dvf = data.loc[data.source.eq("DVF")].reset_index(drop=True)
    candidates = dvf.loc[dvf.temporal_split.isin(["train", "validation"])].reset_index(drop=True)
    queries = dvf.loc[dvf.temporal_split.eq("test")].reset_index(drop=True)
    candidate_mask = dvf.temporal_split.isin(["train", "validation"]).to_numpy()
    query_mask = dvf.temporal_split.eq("test").to_numpy()
    geo_index, _ = tr.geo_shortlist(candidates, queries, 300)
    distances = [tr.embedding_distance(value[candidate_mask], value[query_mask], geo_index) for value in embeddings]
    order = tr.top_order(np.mean(distances, axis=0), geo_index, maximum_k=10)
    slopes = tr.temporal_slopes(candidates)
    prediction = tr.prediction_from_order(order, candidates, queries, 10, True, slopes)
    result = price_metric(queries.log_sale_price, prediction, queries.sale_price_eur)
    output = queries[["record_id", "parcel_id", "observation_date", "longitude", "latitude", "sale_price_eur", "log_sale_price"]].copy()
    output["prediction"] = prediction
    output.to_csv(OUT / "residual_shared_retrieval_predictions_2025.csv.gz", index=False)

    original = pd.read_csv(MAIN / "results/retrieval/test_predictions.csv.gz", dtype={"record_id": "string"})
    if not np.array_equal(original.record_id.astype(str), output.record_id.astype(str)):
        raise AssertionError("retrieval query mismatch")
    direct_prediction = original["Multi Transformer embedding"].to_numpy(float)
    difference = np.abs(original.log_sale_price - direct_prediction) - np.abs(original.log_sale_price - prediction)
    uncertainty = bootstrap(difference.to_numpy(float), spatial_blocks(output))
    uncertainty["positive_favours"] = "residual shared retrieval"
    uncertainty["direct_shared_log_mae"] = float(np.mean(np.abs(original.log_sale_price - direct_prediction)))
    uncertainty["residual_shared_log_mae"] = result["log_mae"]
    return result, output, uncertainty


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    started = time.time()
    data = pd.read_csv(DATA, low_memory=False, dtype={"record_id": "string"})
    data.observation_date = pd.to_datetime(data.observation_date)
    manifest = json.loads(MANIFEST.read_text())
    feature_names = manifest["feature_names"]
    groups = tt.build_groups(manifest)
    raw = tt.raw_feature_frame(data, feature_names)
    dvf = data.loc[data.source.eq("DVF")].reset_index(drop=True)

    # Development: index frozen from 2023 and no 2025 use.
    dev_train_global = data.temporal_split.eq("train").to_numpy()
    validation_global = data.temporal_split.eq("validation").to_numpy()
    dev = tt.prepare(data, raw, dev_train_global, validation_global)
    dvf_train = dvf.temporal_split.eq("train").to_numpy()
    dev_index, dev_index_audit = market_index(dvf, dvf_train, pd.Timestamp("2023-01-01"))
    dev_residual, dev_baseline = residual_lookup(dvf, dev_index)

    development = []
    for seed in SEEDS:
        price_meta = fit_development(dev, groups, dev_residual, dev_baseline, "price", seed, None, None)
        original_energy_meta = json.loads((MAIN / "results/transformer/development" / f"energy_seed{seed}" / "metadata.json").read_text())
        energy_reference = float(original_energy_meta["validation_energy"]["macro_f1"])
        multi_meta = fit_development(
            dev,
            groups,
            dev_residual,
            dev_baseline,
            "multi",
            seed,
            float(price_meta["validation_price"]["log_mae"]),
            energy_reference,
        )
        for meta in (price_meta, multi_meta):
            development.append({
                "mode": meta["mode"],
                "seed": seed,
                "selected_step": meta["selected_step"],
                "price_log_mae": meta["validation_price"]["log_mae"],
                "energy_macro_f1": (meta.get("validation_energy") or {}).get("macro_f1"),
                "energy_balanced_accuracy": (meta.get("validation_energy") or {}).get("balanced_accuracy"),
                "energy_accuracy": (meta.get("validation_energy") or {}).get("accuracy"),
            })
    development_frame = pd.DataFrame(development)
    development_frame.to_csv(OUT / "development_2024_by_seed.csv", index=False)

    # Frozen refit: index uses 2024 only, never 2025 labels.
    refit_global = data.temporal_split.isin(["train", "validation"]).to_numpy()
    test_global = data.temporal_split.eq("test").to_numpy()
    refit = tt.prepare(data, raw, refit_global, test_global)
    dvf_mask = data.source.eq("DVF").to_numpy()
    refit["all_dvf_x"] = refit["matrix"][dvf_mask]
    refit["all_dvf_f"] = data.loc[dvf_mask].reset_index(drop=True)
    dvf_refit = dvf.temporal_split.isin(["train", "validation"]).to_numpy()
    refit_index, refit_index_audit = market_index(dvf, dvf_refit, pd.Timestamp("2024-01-01"))
    refit_residual, refit_baseline = residual_lookup(dvf, refit_index)

    test_rows = []
    embeddings = []
    for seed in SEEDS:
        for mode in ("price", "multi"):
            dev_meta = json.loads((OUT / "development" / f"{mode}_seed{seed}_lambda{(LAMBDA_ENERGY if mode == 'multi' else 0):g}" / "metadata.json").read_text())
            meta = fit_refit(refit, groups, refit_residual, refit_baseline, mode, seed, int(dev_meta["selected_step"]))
            test_rows.append({
                "mode": mode,
                "seed": seed,
                **{f"price_{k}": v for k, v in meta["test_price"].items()},
                **{f"energy_{k}": v for k, v in (meta.get("test_energy") or {}).items()},
            })
            if mode == "multi":
                embeddings.append(np.load(OUT / "refit" / f"multi_seed{seed}_lambda{LAMBDA_ENERGY:g}" / "dvf_price_embeddings.npy"))
    test_frame = pd.DataFrame(test_rows)
    test_frame.to_csv(OUT / "test_2025_by_seed.csv", index=False)

    xgb_result, _ = run_xgboost(data, feature_names, refit_residual, refit_baseline)
    retrieval_result, _, retrieval_uncertainty = run_retrieval(data, embeddings)

    # Direct paired price uncertainty, using the seed-mean absolute error per query.
    direct_rows = []
    residual_rows = []
    base_frame = None
    for seed in SEEDS:
        direct = load_original_predictions("price", seed, "price")
        residual_pred = pd.read_csv(OUT / "refit" / f"price_seed{seed}_lambda0" / "predictions_price.csv.gz", dtype={"record_id": "string"})
        if not np.array_equal(direct.record_id.astype(str), residual_pred.record_id.astype(str)):
            raise AssertionError("direct price query mismatch")
        truth = direct.truth.to_numpy(float)
        direct_rows.append(np.abs(truth - direct.prediction.to_numpy(float)))
        residual_rows.append(np.abs(truth - residual_pred.prediction.to_numpy(float)))
        base_frame = residual_pred
    direct_error = np.mean(direct_rows, axis=0)
    residual_error = np.mean(residual_rows, axis=0)
    direct_uncertainty = bootstrap(direct_error - residual_error, spatial_blocks(base_frame))
    direct_uncertainty["positive_favours"] = "residual price-only Transformer"
    direct_uncertainty["direct_seed_mean_log_mae"] = float(direct_error.mean())
    direct_uncertainty["residual_seed_mean_log_mae"] = float(residual_error.mean())

    # Compact comparisons with the original frozen results.
    original = pd.read_csv(MAIN / "results/final/RQ2_SINGLE_VS_SHARED.csv")
    summaries = []
    for mode in ("price", "multi"):
        subset = test_frame.loc[test_frame["mode"].eq(mode)]
        summaries.append({
            "family": "Residual Transformer",
            "mode": mode,
            "task": "price",
            "metric": "Log-MAE",
            "mean": float(subset.price_log_mae.mean()),
            "std": float(subset.price_log_mae.std(ddof=1)),
        })
        if mode == "multi":
            for metric in ("macro_f1", "balanced_accuracy", "accuracy"):
                summaries.append({
                    "family": "Residual Transformer",
                    "mode": mode,
                    "task": "energy",
                    "metric": metric,
                    "mean": float(subset[f"energy_{metric}"].mean()),
                    "std": float(subset[f"energy_{metric}"].std(ddof=1)),
                })
    summaries.extend([
        {"family": "Residual XGBoost", "mode": "price", "task": "price", "metric": "Log-MAE", "mean": xgb_result["log_mae"], "std": None},
        {"family": "Residual shared retrieval", "mode": "multi", "task": "retrieval", "metric": "Log-MAE", "mean": retrieval_result["log_mae"], "std": None},
    ])
    pd.DataFrame(summaries).to_csv(OUT / "summary_results.csv", index=False)
    dump(OUT / "uncertainty.json", {"direct_price": direct_uncertainty, "retrieval": retrieval_uncertainty})
    dump(OUT / "market_index_audit.json", {"development_2023": dev_index_audit, "refit_2024": refit_index_audit})
    dump(OUT / "run_manifest.json", {
        "status": "PASS",
        "protocol": str(HERE / "PROTOCOL.md"),
        "seeds": list(SEEDS),
        "lambda_energy": LAMBDA_ENERGY,
        "development": "train 2021-2023; validate 2024; preceding-year index=2023",
        "refit_test": "refit 2021-2024; protocol-frozen test 2025; preceding-year index=2024",
        "test_2025_used_for_selection": False,
        "elapsed_seconds": time.time() - started,
        "device": str(tt.DEVICE),
    })
    print("\nDEVELOPMENT\n", development_frame.to_string(index=False), flush=True)
    print("\nTEST\n", test_frame.to_string(index=False), flush=True)
    print("\nXGBOOST\n", json.dumps(xgb_result, indent=2), flush=True)
    print("\nRETRIEVAL\n", json.dumps({"metrics": retrieval_result, "uncertainty": retrieval_uncertainty}, indent=2), flush=True)
    print("\nDIRECT UNCERTAINTY\n", json.dumps(direct_uncertainty, indent=2), flush=True)


if __name__ == "__main__":
    main()
