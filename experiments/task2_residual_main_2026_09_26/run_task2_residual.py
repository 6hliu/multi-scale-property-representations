#!/usr/bin/env python3
"""Fresh five-seed Task 2 rerun with a cutoff-frozen residual price target."""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN = ROOT / "temporal_main_experiment_2026_09_14"
RESIDUAL = ROOT / "temporal_residual_full_2026_09_23"
SCREEN = ROOT / "temporal_residual_screen_2026_09_23"
OUT = HERE / "results"
sys.path.insert(0, str(RESIDUAL))
sys.path.insert(0, str(SCREEN))
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN / "vendor"))

import numpy as np
import pandas as pd
import torch

# The reusable residual module also defines an XGBoost control that this Task 2
# run never calls.  The established PyTorch environment does not include
# XGBoost, so provide an explicit unused placeholder rather than changing the
# environment after the protocol has been frozen.
try:
    import xgboost  # noqa: F401
except ModuleNotFoundError:
    placeholder = types.ModuleType("xgboost")
    placeholder.XGBRegressor = object
    sys.modules["xgboost"] = placeholder

import analyse_full_experiment as analyse
import run_full_experiment as residual
import run_temporal_transformer as transformer
from run_residual_screen import market_index


SEEDS = tuple(transformer.SEEDS)
LAMBDA_ENERGY = 0.20


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def original_prediction(mode: str, seed: int, task: str) -> pd.DataFrame:
    lam = 0.0 if mode in {"price", "energy"} else LAMBDA_ENERGY
    path = MAIN / "results/transformer/refit" / f"{mode}_seed{seed}_lambda{lam:g}" / f"predictions_{task}.csv.gz"
    return pd.read_csv(path, dtype={"record_id": "string"})


def residual_prediction(mode: str, seed: int, task: str) -> pd.DataFrame:
    lam = 0.0 if mode == "price" else LAMBDA_ENERGY
    path = OUT / "refit" / f"{mode}_seed{seed}_lambda{lam:g}" / f"predictions_{task}.csv.gz"
    return pd.read_csv(path, dtype={"record_id": "string"})


def energy_metrics_from_file(frame: pd.DataFrame) -> dict:
    return residual.energy_metric(frame.truth.to_numpy(int), frame.prediction.to_numpy(int))


def result_markdown(summary: pd.DataFrame, uncertainty: pd.DataFrame) -> str:
    def row(model: str, task: str) -> pd.Series:
        return summary.loc[(summary.model == model) & (summary.task == task)].iloc[0]

    single_price = row("Single-task Residual Transformer", "price")
    shared_price = row("Shared Residual Transformer", "price")
    single_energy = row("Energy-only Transformer", "energy")
    shared_energy = row("Shared Residual Transformer", "energy")
    price_ci = uncertainty.loc[uncertainty.comparison.eq("Shared residual vs single residual")].iloc[0]
    energy_ci = uncertainty.loc[
        uncertainty.comparison.eq("Shared residual vs energy-only")
        & uncertainty.metric.eq("Macro-F1")
    ].iloc[0]
    return f"""# Task 2 residual-target rerun

## Frozen design

- Train 2021--2023; validate 2024; refit 2021--2024; evaluate 2025.
- Five paired seeds: {', '.join(map(str, SEEDS))}.
- Shared loss: residual-price Huber loss + 0.20 weighted energy cross-entropy.
- No retrieval loss, neighbour correction or prediction fusion.

## Main results

| Task | Model | Primary metric | Secondary metrics |
|---|---|---:|---|
| Price | Single-task Residual Transformer | Log-MAE {single_price.log_mae_mean:.4f} +/- {single_price.log_mae_std:.4f} | Euro MAE EUR {single_price.euro_mae_mean:,.0f} |
| Price | Shared Residual Transformer | Log-MAE {shared_price.log_mae_mean:.4f} +/- {shared_price.log_mae_std:.4f} | Euro MAE EUR {shared_price.euro_mae_mean:,.0f} |
| Energy | Energy-only Transformer | Macro-F1 {single_energy.macro_f1_mean:.4f} +/- {single_energy.macro_f1_std:.4f} | Balanced accuracy {single_energy.balanced_accuracy_mean:.4f} |
| Energy | Shared Residual Transformer | Macro-F1 {shared_energy.macro_f1_mean:.4f} +/- {shared_energy.macro_f1_std:.4f} | Balanced accuracy {shared_energy.balanced_accuracy_mean:.4f} |

Positive paired gains favour the shared model.

- Price Log-MAE gain: {price_ci.point_gain:.6f}, 95% CI [{price_ci.ci_lower:.6f}, {price_ci.ci_upper:.6f}].
- Energy Macro-F1 gain: {energy_ci.point_gain:.6f}, 95% CI [{energy_ci.ci_lower:.6f}, {energy_ci.ci_upper:.6f}].

## Interpretation

The result must be interpreted from the confidence intervals rather than from
the point estimates alone. This experiment is a protocol-frozen temporal rerun,
not a globally untouched test, because earlier project development had already
inspected 2025.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    residual.OUT = OUT
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    started = time.time()

    data = pd.read_csv(residual.DATA, low_memory=False, dtype={"record_id": "string"})
    data.observation_date = pd.to_datetime(data.observation_date)
    manifest = json.loads(residual.MANIFEST.read_text())
    groups = transformer.build_groups(manifest)
    raw = transformer.raw_feature_frame(data, manifest["feature_names"])
    dvf = data.loc[data.source.eq("DVF")].reset_index(drop=True)

    # Development: 2023-only market index, no 2024/2025 label in the baseline.
    dev = transformer.prepare(
        data,
        raw,
        data.temporal_split.eq("train").to_numpy(),
        data.temporal_split.eq("validation").to_numpy(),
    )
    dev_index, dev_index_audit = market_index(
        dvf,
        dvf.temporal_split.eq("train").to_numpy(),
        pd.Timestamp("2023-01-01"),
    )
    dev_residual, dev_baseline = residual.residual_lookup(dvf, dev_index)

    development_rows = []
    development = {}
    for seed in SEEDS:
        price_meta = residual.fit_development(
            dev, groups, dev_residual, dev_baseline,
            "price", seed, None, None,
        )
        energy_meta = json.loads(
            (MAIN / "results/transformer/development" / f"energy_seed{seed}" / "metadata.json").read_text()
        )
        multi_meta = residual.fit_development(
            dev, groups, dev_residual, dev_baseline,
            "multi", seed,
            float(price_meta["validation_price"]["log_mae"]),
            float(energy_meta["validation_energy"]["macro_f1"]),
        )
        development[("price", seed)] = price_meta
        development[("multi", seed)] = multi_meta
        for meta in (price_meta, multi_meta):
            development_rows.append({
                "mode": meta["mode"],
                "seed": seed,
                "selected_step": meta["selected_step"],
                "price_log_mae": meta["validation_price"]["log_mae"],
                "energy_macro_f1": (meta.get("validation_energy") or {}).get("macro_f1"),
                "energy_balanced_accuracy": (meta.get("validation_energy") or {}).get("balanced_accuracy"),
                "energy_accuracy": (meta.get("validation_energy") or {}).get("accuracy"),
            })
    pd.DataFrame(development_rows).to_csv(OUT / "development_2024_by_seed.csv", index=False)

    # Refit: 2024-only market index, with 2025 labels never used in training or selection.
    refit = transformer.prepare(
        data,
        raw,
        data.temporal_split.isin(["train", "validation"]).to_numpy(),
        data.temporal_split.eq("test").to_numpy(),
    )
    # The reused refit routine also exports price-task embeddings for audit.
    # They are not used in this Task 2 comparison, but the matching DVF rows
    # must be supplied so the routine can complete its standard output bundle.
    dvf_mask = data.source.eq("DVF").to_numpy()
    refit["all_dvf_x"] = refit["matrix"][dvf_mask]
    refit["all_dvf_f"] = data.loc[dvf_mask].reset_index(drop=True)
    refit_index, refit_index_audit = market_index(
        dvf,
        dvf.temporal_split.isin(["train", "validation"]).to_numpy(),
        pd.Timestamp("2024-01-01"),
    )
    refit_residual, refit_baseline = residual.residual_lookup(dvf, refit_index)

    seed_rows = []
    for seed in SEEDS:
        for mode in ("price", "multi"):
            meta = residual.fit_refit(
                refit, groups, refit_residual, refit_baseline,
                mode, seed, int(development[(mode, seed)]["selected_step"]),
            )
            seed_rows.append({
                "model": "Single-task Residual Transformer" if mode == "price" else "Shared Residual Transformer",
                "task": "price",
                "seed": seed,
                **meta["test_price"],
            })
            if mode == "multi":
                seed_rows.append({
                    "model": "Shared Residual Transformer",
                    "task": "energy",
                    "seed": seed,
                    **meta["test_energy"],
                })
        energy = original_prediction("energy", seed, "energy")
        seed_rows.append({
            "model": "Energy-only Transformer",
            "task": "energy",
            "seed": seed,
            **energy_metrics_from_file(energy),
        })
    by_seed = pd.DataFrame(seed_rows)
    by_seed.to_csv(OUT / "task2_by_seed.csv", index=False)

    summary_rows = []
    for (model, task), subset in by_seed.groupby(["model", "task"], sort=False):
        row = {"model": model, "task": task}
        metrics = ("log_mae", "log_rmse", "log_r2", "euro_mae") if task == "price" else ("macro_f1", "balanced_accuracy", "accuracy")
        for metric in metrics:
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_std"] = float(subset[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "TASK2_RESIDUAL_RESULTS.csv", index=False)

    uncertainty_rows = []
    uncertainty_rows.append(analyse.price_comparison(
        lambda seed: residual_prediction("multi", seed, "price"),
        lambda seed: residual_prediction("price", seed, "price"),
        "Shared residual vs single residual",
    ))
    uncertainty_rows.extend(analyse.energy_comparison(
        lambda seed: residual_prediction("multi", seed, "energy"),
        lambda seed: original_prediction("energy", seed, "energy"),
        "Shared residual vs energy-only",
    ))
    # Secondary target-formulation checks for interpretation.
    uncertainty_rows.append(analyse.price_comparison(
        lambda seed: residual_prediction("price", seed, "price"),
        lambda seed: original_prediction("price", seed, "price"),
        "Single residual vs single direct-price",
    ))
    uncertainty_rows.append(analyse.price_comparison(
        lambda seed: residual_prediction("multi", seed, "price"),
        lambda seed: original_prediction("multi", seed, "price"),
        "Shared residual vs shared direct-price",
    ))
    uncertainty = pd.DataFrame(uncertainty_rows)
    uncertainty["ci_excludes_zero"] = (uncertainty.ci_lower > 0) | (uncertainty.ci_upper < 0)
    uncertainty.to_csv(OUT / "PAIRED_SPATIAL_BLOCK_UNCERTAINTY.csv", index=False)

    dump(OUT / "market_index_audit.json", {
        "development_2023": dev_index_audit,
        "refit_2024": refit_index_audit,
    })
    dump(OUT / "run_manifest.json", {
        "status": "PASS",
        "protocol": str(HERE / "PROTOCOL.md"),
        "fresh_residual_models": True,
        "reused_model": "energy-only Transformer only; price-target change cannot affect it",
        "seeds": list(SEEDS),
        "lambda_energy": LAMBDA_ENERGY,
        "features": len(manifest["feature_names"]),
        "development": "train 2021-2023; validate 2024; market index source 2023",
        "refit_test": "refit 2021-2024; evaluate 2025; market index source 2024",
        "test_2025_used_for_selection": False,
        "retrieval_loss": False,
        "neighbour_fusion": False,
        "device": str(transformer.DEVICE),
        "elapsed_seconds": time.time() - started,
    })
    (OUT / "RESULTS_SUMMARY.md").write_text(result_markdown(summary, uncertainty))

    print("\nTASK 2 SUMMARY\n", summary.to_string(index=False), flush=True)
    print("\nPAIRED UNCERTAINTY\n", uncertainty.to_string(index=False), flush=True)
    print(f"\nCompleted in {time.time() - started:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
