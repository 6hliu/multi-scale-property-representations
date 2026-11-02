#!/usr/bin/env python3
"""Consolidate the residual extension and compute paired uncertainty results."""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN = ROOT / "temporal_main_experiment_2026_09_14"
OUT = HERE / "results"
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN / "vendor"))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import run_temporal_transformer as tt
from run_full_experiment import bootstrap, spatial_blocks


SEEDS = tuple(tt.SEEDS)
LAMBDA_ENERGY = 0.20
REPETITIONS = 20_000
RNG_SEED = 20260923


def load_original(mode: str, seed: int, task: str) -> pd.DataFrame:
    lam = 0.0 if mode in {"price", "energy"} else LAMBDA_ENERGY
    path = MAIN / "results/transformer/refit" / f"{mode}_seed{seed}_lambda{lam:g}" / f"predictions_{task}.csv.gz"
    return pd.read_csv(path, dtype={"record_id": "string"})


def load_residual(mode: str, seed: int, task: str) -> pd.DataFrame:
    lam = 0.0 if mode == "price" else LAMBDA_ENERGY
    path = OUT / "refit" / f"{mode}_seed{seed}_lambda{lam:g}" / f"predictions_{task}.csv.gz"
    return pd.read_csv(path, dtype={"record_id": "string"})


def price_errors(frame: pd.DataFrame) -> np.ndarray:
    truth = frame["truth"].to_numpy(float) if "truth" in frame else frame.log_sale_price.to_numpy(float)
    return np.abs(truth - frame.prediction.to_numpy(float))


def price_comparison(candidate_loader, reference_loader, name: str) -> dict:
    candidate, reference = [], []
    base = None
    for seed in SEEDS:
        c = candidate_loader(seed)
        r = reference_loader(seed)
        if not np.array_equal(c.record_id.astype(str), r.record_id.astype(str)):
            raise AssertionError(f"unaligned price comparison: {name}, seed {seed}")
        candidate.append(price_errors(c))
        reference.append(price_errors(r))
        base = c
    candidate_mean = np.mean(candidate, axis=0)
    reference_mean = np.mean(reference, axis=0)
    result = bootstrap(reference_mean - candidate_mean, spatial_blocks(base))
    result.update({
        "comparison": name,
        "task": "price",
        "metric": "Log-MAE reduction",
        "positive_favours": "candidate",
        "candidate_mean": float(candidate_mean.mean()),
        "reference_mean": float(reference_mean.mean()),
        "relative_gain_percent": float(100 * result["point_gain"] / reference_mean.mean()),
    })
    return result


def block_confusions(y: np.ndarray, prediction: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    unique, codes = np.unique(blocks, return_inverse=True)
    matrices = np.zeros((len(unique), 7, 7), dtype=np.int64)
    for index in range(len(unique)):
        mask = codes == index
        matrices[index] = confusion_matrix(y[mask], prediction[mask], labels=list(range(7)))
    return matrices


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    value = matrix.astype(float)
    true_count = value.sum(1)
    predicted_count = value.sum(0)
    tp = np.diag(value)
    recall = np.divide(tp, true_count, out=np.zeros(7), where=true_count > 0)
    precision = np.divide(tp, predicted_count, out=np.zeros(7), where=predicted_count > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(7), where=(precision + recall) > 0)
    return {
        "Macro-F1": float(f1.mean()),
        "Balanced accuracy": float(recall.mean()),
        "Accuracy": float(tp.sum() / max(value.sum(), 1)),
    }


def energy_comparison(candidate_loader, reference_loader, name: str) -> list[dict]:
    candidate_matrices = []
    reference_matrices = []
    point_candidate = []
    point_reference = []
    blocks = None
    for seed in SEEDS:
        c = candidate_loader(seed)
        r = reference_loader(seed)
        if not np.array_equal(c.record_id.astype(str), r.record_id.astype(str)):
            raise AssertionError(f"unaligned energy comparison: {name}, seed {seed}")
        y = c.truth.to_numpy(int)
        if not np.array_equal(y, r.truth.to_numpy(int)):
            raise AssertionError("energy truth mismatch")
        if blocks is None:
            blocks = spatial_blocks(c)
        cm_c = block_confusions(y, c.prediction.to_numpy(int), blocks)
        cm_r = block_confusions(y, r.prediction.to_numpy(int), blocks)
        candidate_matrices.append(cm_c)
        reference_matrices.append(cm_r)
        point_candidate.append(metrics_from_confusion(cm_c.sum(0)))
        point_reference.append(metrics_from_confusion(cm_r.sum(0)))
    rng = np.random.default_rng(RNG_SEED)
    block_count = candidate_matrices[0].shape[0]
    draws = {metric: np.empty(REPETITIONS, float) for metric in point_candidate[0]}
    for repetition in range(REPETITIONS):
        sampled = rng.integers(0, block_count, block_count)
        differences = {metric: [] for metric in draws}
        for cm_c, cm_r in zip(candidate_matrices, reference_matrices):
            mc = metrics_from_confusion(cm_c[sampled].sum(0))
            mr = metrics_from_confusion(cm_r[sampled].sum(0))
            for metric in draws:
                differences[metric].append(mc[metric] - mr[metric])
        for metric in draws:
            draws[metric][repetition] = np.mean(differences[metric])
    rows = []
    for metric in draws:
        candidate_value = float(np.mean([value[metric] for value in point_candidate]))
        reference_value = float(np.mean([value[metric] for value in point_reference]))
        rows.append({
            "comparison": name,
            "task": "energy",
            "metric": metric,
            "point_gain": candidate_value - reference_value,
            "ci_lower": float(np.quantile(draws[metric], 0.025)),
            "ci_upper": float(np.quantile(draws[metric], 0.975)),
            "blocks": block_count,
            "repetitions": REPETITIONS,
            "positive_favours": "candidate",
            "candidate_mean": candidate_value,
            "reference_mean": reference_value,
            "relative_gain_percent": None,
        })
    return rows


def main() -> None:
    comparisons = []
    comparisons.append(price_comparison(
        lambda seed: load_residual("price", seed, "price"),
        lambda seed: load_original("price", seed, "price"),
        "Residual price-only vs direct price-only Transformer",
    ))
    comparisons.append(price_comparison(
        lambda seed: load_residual("multi", seed, "price"),
        lambda seed: load_original("multi", seed, "price"),
        "Residual shared vs direct shared Transformer",
    ))
    comparisons.append(price_comparison(
        lambda seed: load_residual("multi", seed, "price"),
        lambda seed: load_residual("price", seed, "price"),
        "Residual shared vs residual price-only Transformer",
    ))
    comparisons.extend(energy_comparison(
        lambda seed: load_residual("multi", seed, "energy"),
        lambda seed: load_original("multi", seed, "energy"),
        "Residual shared vs direct shared Transformer",
    ))
    comparisons.extend(energy_comparison(
        lambda seed: load_residual("multi", seed, "energy"),
        lambda seed: load_original("energy", seed, "energy"),
        "Residual shared vs energy-only Transformer",
    ))
    uncertainty = pd.DataFrame(comparisons)
    uncertainty["ci_excludes_zero"] = (uncertainty.ci_lower > 0) | (uncertainty.ci_upper < 0)
    uncertainty.to_csv(OUT / "extended_paired_uncertainty.csv", index=False)

    original = pd.read_csv(MAIN / "results/final/RQ2_SINGLE_VS_SHARED.csv")
    residual = pd.read_csv(OUT / "summary_results.csv")
    rolling = pd.read_csv(OUT / "rolling_backtests/summary_by_fold.csv")
    retrieval = pd.read_csv(OUT / "residual_aware_retrieval_results.csv")
    retrieval_ci = json.loads((OUT / "residual_aware_retrieval_uncertainty.json").read_text())

    rows = []
    for _, row in original.iterrows():
        if row.task == "price":
            rows.append({"section": "2025 direct prediction", "method": f"Original {row.model} ({row.training})", "task": "price", "metric": "Log-MAE", "mean": row.log_mae_mean, "std": row.log_mae_std})
        else:
            rows.append({"section": "2025 direct prediction", "method": f"Original {row.model} ({row.training})", "task": "energy", "metric": "Macro-F1", "mean": row.macro_f1_mean, "std": row.macro_f1_std})
    for _, row in residual.iterrows():
        rows.append({"section": "2025 residual extension", "method": f"{row.family} ({row['mode']})", "task": row.task, "metric": row.metric, "mean": row["mean"], "std": row["std"]})
    for _, row in rolling.iterrows():
        rows.append({"section": f"Rolling backtest {int(row.test_year)}", "method": f"{row.variant.title()} price-only Transformer", "task": "price", "metric": "Log-MAE", "mean": row.log_mae_mean, "std": row.log_mae_std})
    for _, row in retrieval.loc[retrieval.phase.eq("test_2025")].iterrows():
        rows.append({"section": "2025 retrieval diagnostic", "method": row.method, "task": "retrieval", "metric": "Log-MAE", "mean": row.log_mae, "std": None})
    pd.DataFrame(rows).to_csv(OUT / "consolidated_results.csv", index=False)

    manifest = {
        "status": "PASS",
        "analysis_date": "2026-09-23",
        "five_seed_main_experiment": True,
        "rolling_folds": [2023, 2024],
        "spatial_block_bootstrap_repetitions": REPETITIONS,
        "retrieval_primary": "Frozen raw-price aggregation; incompatible with residual embeddings and materially worse",
        "retrieval_post_hoc": {
            "status": retrieval_ci["status"],
            "selected_on": retrieval_ci["selection_population"],
            "selected_method": retrieval_ci["selected_method"],
        },
        "reporting_boundary": "Follow-up extension; do not silently replace the already frozen final registry",
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\nPAIRED UNCERTAINTY\n", uncertainty.to_string(index=False), flush=True)
    print("\nCONSOLIDATED RESULTS WRITTEN\n", flush=True)


if __name__ == "__main__":
    main()
