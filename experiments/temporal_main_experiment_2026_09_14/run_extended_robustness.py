#!/usr/bin/env python3
"""Spatial-block confidence intervals and formal unseen-entity evaluation."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE / "vendor"))
from xgboost import XGBClassifier, XGBRegressor

import run_classical_baselines as cb
import run_temporal_transformer as tt


DATA = HERE / "data/temporal_model_ready_55.csv.gz"
MANIFEST = HERE / "data/feature_manifest.json"
OUT = HERE / "results/robustness"
EARTH_RADIUS_M = 6_371_008.8
REPETITIONS = 2_000
RNG_SEED = 20260915


def spatial_blocks(frame: pd.DataFrame) -> np.ndarray:
    reference = math.radians(float(frame.latitude.median()))
    x = EARTH_RADIUS_M * np.radians(frame.longitude.to_numpy(float)) * math.cos(reference)
    y = EARTH_RADIUS_M * np.radians(frame.latitude.to_numpy(float))
    return np.char.add(np.floor(x / 3000).astype(int).astype(str), np.char.add("_", np.floor(y / 3000).astype(int).astype(str)))


def bootstrap_mean_difference(difference: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, int]:
    unique, codes = np.unique(blocks, return_inverse=True)
    sums = np.bincount(codes, weights=difference, minlength=len(unique))
    counts = np.bincount(codes, minlength=len(unique))
    draws = np.empty(REPETITIONS, float)
    for i in range(REPETITIONS):
        sampled = rng.integers(0, len(unique), len(unique))
        draws[i] = sums[sampled].sum() / counts[sampled].sum()
    return float(difference.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), len(unique)


def block_confusions(y: np.ndarray, prediction: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, codes = np.unique(blocks, return_inverse=True)
    matrices = np.zeros((len(unique), 7, 7), dtype=np.int64)
    for index in range(len(unique)):
        mask = codes == index
        matrices[index] = confusion_matrix(y[mask], prediction[mask], labels=list(range(7)))
    return unique, matrices


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    matrix = matrix.astype(float)
    true_count = matrix.sum(1)
    predicted_count = matrix.sum(0)
    tp = np.diag(matrix)
    recall = np.divide(tp, true_count, out=np.zeros(7), where=true_count > 0)
    precision = np.divide(tp, predicted_count, out=np.zeros(7), where=predicted_count > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(7), where=(precision + recall) > 0)
    return {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "accuracy": float(tp.sum() / max(matrix.sum(), 1)),
    }


def bootstrap_confusion_difference(
    y: np.ndarray,
    prediction_a: list[np.ndarray],
    prediction_b: list[np.ndarray],
    blocks: np.ndarray,
    metric: str,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    unique = np.unique(blocks)
    matrices_a = [block_confusions(y, value, blocks)[1] for value in prediction_a]
    matrices_b = [block_confusions(y, value, blocks)[1] for value in prediction_b]
    point = float(np.mean([
        metrics_from_confusion(a.sum(0))[metric] - metrics_from_confusion(b.sum(0))[metric]
        for a, b in zip(matrices_a, matrices_b)
    ]))
    draws = np.empty(REPETITIONS, float)
    for i in range(REPETITIONS):
        sampled = rng.integers(0, len(unique), len(unique))
        draws[i] = np.mean([
            metrics_from_confusion(a[sampled].sum(0))[metric] - metrics_from_confusion(b[sampled].sum(0))[metric]
            for a, b in zip(matrices_a, matrices_b)
        ])
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), len(unique)


def add_ci(rows: list[dict], comparison: str, task: str, metric: str, direction: str, values: tuple[float, float, float, int]) -> None:
    point, lower, upper, block_count = values
    rows.append({
        "comparison": comparison,
        "task": task,
        "metric": metric,
        "positive_means": direction,
        "point_difference": point,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "spatial_blocks": block_count,
        "bootstrap_repetitions": REPETITIONS,
        "ci_excludes_zero": bool(lower > 0 or upper < 0),
    })


def xgb_models_and_predictions(data: pd.DataFrame, manifest: dict):
    sets = cb.feature_sets(manifest)
    test = data.temporal_split.eq("test")
    price_mask = test & data.source.eq("DVF")
    energy_mask = test & data.source.eq("DPE")
    py = data.loc[price_mask, "log_sale_price"].to_numpy(float)
    euro = data.loc[price_mask, "sale_price_eur"].to_numpy(float)
    ey = data.loc[energy_mask, "energy_class"].map(cb.LABELS).to_numpy(int)
    output = {"price_frame": data.loc[price_mask].reset_index(drop=True), "energy_frame": data.loc[energy_mask].reset_index(drop=True), "price_y": py, "price_euro": euro, "energy_y": ey}
    for set_name in ("intrinsic_only", "full_55"):
        matrix = cb.feature_frame(data, sets[set_name])
        price_matrix = cb.add_room(matrix, data)
        regressor = XGBRegressor(); regressor.load_model(HERE / f"results/classical/xgboost_price_{set_name}.ubj")
        classifier = XGBClassifier(); classifier.load_model(HERE / f"results/classical/xgboost_energy_{set_name}.ubj")
        output[f"price_{set_name}"] = regressor.predict(price_matrix.loc[price_mask])
        output[f"energy_{set_name}"] = classifier.predict(matrix.loc[energy_mask]).astype(int)
    return output


def transformer_predictions(task: str, training: str, seed: int, lam: float) -> pd.DataFrame:
    tag = f"{training}_seed{seed}_lambda{lam:g}"
    return pd.read_csv(HERE / "results/transformer/refit" / tag / f"predictions_{task}.csv.gz", dtype={"record_id": "string"})


def formal_entity_rows(xgb: dict, transformer_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in ("price", "energy"):
        frame = xgb[f"{task}_frame"]
        unseen = frame.unseen_entity_in_year.to_numpy(bool)
        for population, mask in (("all_2025", np.ones(len(frame), bool)), ("repeated_entity_2025", ~unseen), ("unseen_entity_2025", unseen)):
            if task == "price":
                metrics = cb.price_metrics(xgb["price_y"][mask], xgb["price_full_55"][mask], xgb["price_euro"][mask])
            else:
                metrics = cb.energy_metrics(xgb["energy_y"][mask], xgb["energy_full_55"][mask])
            rows.append({"architecture": "XGBoost", "training": "single-task", "task": task, "population": population, "n": int(mask.sum()), **metrics})
    wanted = transformer_summary[transformer_summary.population.isin(["all_2025", "repeated_entity_2025", "unseen_entity_2025"])].copy()
    for _, row in wanted.iterrows():
        result = {"architecture": row.architecture, "training": row.training, "task": row.task, "population": row.population}
        count_col = "log_mae_count" if row.task == "price" else "macro_f1_count"
        result["n"] = None
        result["seeds"] = int(row[count_col])
        for metric in ("log_mae", "log_rmse", "log_r2", "euro_mae", "macro_f1", "balanced_accuracy", "accuracy"):
            result[f"{metric}_mean"] = row.get(f"{metric}_mean")
            result[f"{metric}_std"] = row.get(f"{metric}_std")
        rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA, low_memory=False, dtype={"record_id": "string"})
    manifest = json.loads(MANIFEST.read_text())
    xgb = xgb_models_and_predictions(data, manifest)
    rng = np.random.default_rng(RNG_SEED)
    ci_rows = []

    # RQ1 full feature set versus intrinsic-only XGBoost.
    price_blocks = spatial_blocks(xgb["price_frame"])
    rq1_price_difference = np.abs(xgb["price_y"] - xgb["price_intrinsic_only"]) - np.abs(xgb["price_y"] - xgb["price_full_55"])
    add_ci(ci_rows, "XGBoost full_55 vs intrinsic_only", "price", "Log-MAE", "full_55 improves", bootstrap_mean_difference(rq1_price_difference, price_blocks, rng))
    energy_blocks = spatial_blocks(xgb["energy_frame"])
    for metric in ("macro_f1", "balanced_accuracy", "accuracy"):
        values = bootstrap_confusion_difference(
            xgb["energy_y"], [xgb["energy_full_55"]], [xgb["energy_intrinsic_only"]], energy_blocks, metric, rng
        )
        add_ci(ci_rows, "XGBoost full_55 vs intrinsic_only", "energy", metric, "full_55 improves", values)

    # RQ2 shared versus task-specific Transformer, averaged across five fixed seeds.
    lambda_energy = float(json.loads((HERE / "results/transformer/SELECTION.json").read_text())["selected_lambda_energy"])
    price_single_predictions, price_shared_predictions = [], []
    energy_single_predictions, energy_shared_predictions = [], []
    paired_seed_rows = []
    price_truth = None; energy_truth = None
    for seed in tt.SEEDS:
        ps = transformer_predictions("price", "price", seed, 0.0)
        pm = transformer_predictions("price", "multi", seed, lambda_energy)
        es = transformer_predictions("energy", "energy", seed, 0.0)
        em = transformer_predictions("energy", "multi", seed, lambda_energy)
        for left, right in ((ps, pm), (es, em)):
            if not np.array_equal(left.record_id.to_numpy(), right.record_id.to_numpy()): raise AssertionError(f"prediction alignment seed {seed}")
        price_truth = ps.truth.to_numpy(float); energy_truth = es.truth.to_numpy(int)
        p_single = ps.prediction.to_numpy(float); p_shared = pm.prediction.to_numpy(float)
        e_single = es.prediction.to_numpy(int); e_shared = em.prediction.to_numpy(int)
        price_single_predictions.append(p_single); price_shared_predictions.append(p_shared)
        energy_single_predictions.append(e_single); energy_shared_predictions.append(e_shared)
        p_delta = float(np.abs(price_truth - p_single).mean() - np.abs(price_truth - p_shared).mean())
        e_single_metric = tt.energy_metrics(energy_truth, e_single); e_shared_metric = tt.energy_metrics(energy_truth, e_shared)
        paired_seed_rows.append({
            "seed": seed,
            "price_log_mae_single_minus_shared": p_delta,
            "energy_macro_f1_shared_minus_single": e_shared_metric["macro_f1"] - e_single_metric["macro_f1"],
            "energy_balanced_accuracy_shared_minus_single": e_shared_metric["balanced_accuracy"] - e_single_metric["balanced_accuracy"],
            "energy_accuracy_shared_minus_single": e_shared_metric["accuracy"] - e_single_metric["accuracy"],
        })

    price_difference = np.mean([
        np.abs(price_truth - single) - np.abs(price_truth - shared)
        for single, shared in zip(price_single_predictions, price_shared_predictions)
    ], axis=0)
    add_ci(ci_rows, "Shared vs single-task Transformer", "price", "Log-MAE", "shared improves", bootstrap_mean_difference(price_difference, price_blocks, rng))
    for metric in ("macro_f1", "balanced_accuracy", "accuracy"):
        values = bootstrap_confusion_difference(
            energy_truth, energy_shared_predictions, energy_single_predictions, energy_blocks, metric, rng
        )
        add_ci(ci_rows, "Shared vs single-task Transformer", "energy", metric, "shared improves", values)

    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUT / "spatial_block_confidence_intervals.csv", index=False)
    pd.DataFrame(paired_seed_rows).to_csv(OUT / "paired_transformer_deltas_by_seed.csv", index=False)

    summary = pd.read_csv(HERE / "results/transformer/test_results_summary.csv")
    formal = formal_entity_rows(xgb, summary)
    n_summary = []
    for task in ("price", "energy"):
        frame = xgb[f"{task}_frame"]
        unseen = int(frame.unseen_entity_in_year.sum())
        n_summary.append({"task": task, "all_2025": len(frame), "unseen_entity_2025": unseen, "unseen_share": unseen / len(frame), "repeated_entity_2025": len(frame) - unseen})
    formal.to_csv(OUT / "unseen_entity_formal_results.csv", index=False)
    pd.DataFrame(n_summary).to_csv(OUT / "unseen_entity_sample_sizes.csv", index=False)

    manifest_out = {
        "status": "PASS",
        "bootstrap": "paired 3 km spatial-block resampling",
        "bootstrap_repetitions": REPETITIONS,
        "bootstrap_seed": RNG_SEED,
        "transformer_seeds": list(tt.SEEDS),
        "selection_boundary": "all configurations and lambda remained fixed from 2024 validation",
        "test": "2025 only",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest_out, indent=2) + "\n")
    print("SPATIAL-BLOCK CONFIDENCE INTERVALS")
    print(ci.to_string(index=False))
    print("\nTRANSFORMER DELTAS BY SEED")
    print(pd.DataFrame(paired_seed_rows).to_string(index=False))
    print("\nUNSEEN ENTITY SAMPLE SIZES")
    print(pd.DataFrame(n_summary).to_string(index=False))


if __name__ == "__main__":
    main()
