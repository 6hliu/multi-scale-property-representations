#!/usr/bin/env python3
"""Exploratory 2024 screen for a lagged local-market-index residual target.

This script never reads 2025 labels.  It trains on 2021--2023 DVF sales and
evaluates on 2024 DVF sales.  The local market index is a smoothed log-price-
per-square-metre estimate frozen at the end of 2023.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN = ROOT / "temporal_main_experiment_2026_09_14"
DATA = MAIN / "data/temporal_model_ready_55.csv.gz"
MANIFEST = MAIN / "data/feature_manifest.json"
OUT = HERE / "results"

sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN / "vendor"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

import run_temporal_transformer as tt  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402


SEED = 42
BATCH = 1024
STEPS = 512
CHECKPOINT_INTERVAL = 128
INDEX_VARIANTS = {
    "all_2021_2023": None,
    "recent_2022_2023": pd.Timestamp("2022-01-01"),
    "recent_2023": pd.Timestamp("2023-01-01"),
}
XGB_CONFIGS = (
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 4, "learning_rate": 0.10, "min_child_weight": 10},
)


def metrics(y: np.ndarray, prediction: np.ndarray, euro: np.ndarray) -> dict:
    predicted_euro = np.exp(np.clip(prediction, np.log(10_000), np.log(20_000_000)))
    return {
        "log_mae": float(mean_absolute_error(y, prediction)),
        "log_rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "log_r2": float(r2_score(y, prediction)),
        "euro_mae": float(mean_absolute_error(euro, predicted_euro)),
    }


def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(keys, dropna=False, observed=True)["log_price_per_m2"]
        .agg(["sum", "count"])
        .reset_index()
    )


def attach_stats(
    rows: pd.DataFrame,
    stats: pd.DataFrame,
    keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    renamed = stats.rename(columns={"sum": f"{prefix}_sum", "count": f"{prefix}_count"})
    return rows.merge(renamed, on=keys, how="left", validate="many_to_one")


def market_index(
    price_frame: pd.DataFrame,
    train_mask: np.ndarray,
    start_date: pd.Timestamp | None,
) -> tuple[np.ndarray, dict]:
    """Return a cutoff-frozen hierarchical index for every DVF row.

    The hierarchy is property type -> commune/type -> postcode/type.  Local
    means are shrunk toward their parent means.  Training rows that contributed
    to the index use an exact leave-one-out numerator and denominator.
    """
    base = price_frame.loc[train_mask].copy()
    if start_date is not None:
        base = base.loc[base.observation_date.ge(start_date)].copy()
    base["log_price_per_m2"] = base.log_sale_price - np.log(base.floor_area_m2)
    if len(base) == 0:
        raise AssertionError("empty index base")

    type_stats = aggregate(base, ["property_type"])
    commune_stats = aggregate(base, ["property_type", "commune_code"])
    postcode_stats = aggregate(base, ["property_type", "postcode"])

    rows = price_frame[[
        "record_id", "property_type", "commune_code", "postcode",
        "log_sale_price", "floor_area_m2", "observation_date",
    ]].copy()
    rows["log_price_per_m2"] = rows.log_sale_price - np.log(rows.floor_area_m2)
    rows["in_index_base"] = train_mask
    if start_date is not None:
        rows["in_index_base"] &= rows.observation_date.ge(start_date)

    rows = attach_stats(rows, type_stats, ["property_type"], "type")
    rows = attach_stats(rows, commune_stats, ["property_type", "commune_code"], "commune")
    rows = attach_stats(rows, postcode_stats, ["property_type", "postcode"], "postcode")

    own = rows.log_price_per_m2.where(rows.in_index_base, 0.0)
    subtract = rows.in_index_base.astype(float)
    for level in ("type", "commune", "postcode"):
        rows[f"{level}_sum"] = rows[f"{level}_sum"].fillna(0.0) - own
        rows[f"{level}_count"] = rows[f"{level}_count"].fillna(0.0) - subtract

    global_mean = float(base.log_price_per_m2.mean())
    type_count = rows.type_count.clip(lower=0)
    type_mean = np.where(
        type_count > 0,
        rows.type_sum / np.maximum(type_count, 1),
        global_mean,
    )

    commune_count = rows.commune_count.clip(lower=0)
    commune_mean = (
        rows.commune_sum + 50.0 * type_mean
    ) / np.maximum(commune_count + 50.0, 1.0)

    postcode_count = rows.postcode_count.clip(lower=0)
    postcode_mean = (
        rows.postcode_sum + 30.0 * commune_mean
    ) / np.maximum(postcode_count + 30.0, 1.0)

    index = np.asarray(postcode_mean, dtype=np.float32)
    if not np.isfinite(index).all():
        raise AssertionError("non-finite market index")
    audit = {
        "base_rows": int(len(base)),
        "base_start": str(base.observation_date.min().date()),
        "base_end": str(base.observation_date.max().date()),
        "postcode_groups": int(len(postcode_stats)),
        "commune_groups": int(len(commune_stats)),
        "global_log_price_per_m2": global_mean,
        "shrinkage": {"postcode_to_commune": 30.0, "commune_to_type": 50.0},
        "training_leave_one_out": True,
    }
    return index, audit


def transformer_screen(
    arrays: dict,
    groups: dict[str, list[int]],
    residual_target: np.ndarray,
    baseline_log_price: np.ndarray,
) -> tuple[dict, np.ndarray, list[dict]]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = tt.TemporalGroupedTransformer(groups).to(tt.DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()

    train_residual = residual_target[arrays["price_train_positions"]].astype(np.float32)
    validation_residual = residual_target[arrays["price_validation_positions"]].astype(np.float32)
    center = float(np.median(train_residual))
    px = torch.from_numpy(arrays["price_train_x"]).to(tt.DEVICE)
    pr = torch.from_numpy(arrays["price_train_room"]).to(tt.DEVICE)
    py = torch.from_numpy(train_residual).to(tt.DEVICE)
    rng = np.random.default_rng(SEED + 10007)
    best = None
    best_state = None
    history = []
    for step in range(1, STEPS + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        index = torch.from_numpy(rng.integers(len(py), size=BATCH)).to(tt.DEVICE)
        loss = loss_fn(model.price(px[index], pr[index]), py[index] - center)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        if step % CHECKPOINT_INTERVAL == 0:
            residual_prediction = (
                tt.predict(
                    model,
                    arrays["price_evaluation_x"],
                    "price",
                    arrays["price_evaluation_room"],
                )
                + center
            )
            prediction = residual_prediction + baseline_log_price[arrays["price_validation_positions"]]
            value = metrics(
                arrays["price_evaluation_y"],
                prediction,
                arrays["price_evaluation_f"].sale_price_eur.to_numpy(float),
            )
            history.append({"step": step, "train_loss": float(loss.detach().cpu()), **value})
            print(f"Transformer step={step} validation Log-MAE={value['log_mae']:.6f}", flush=True)
            if best is None or value["log_mae"] < best["log_mae"]:
                best = {"selected_step": step, **value}
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best is None or best_state is None:
        raise AssertionError("no transformer checkpoint")
    model.load_state_dict(best_state)
    residual_prediction = (
        tt.predict(
            model,
            arrays["price_evaluation_x"],
            "price",
            arrays["price_evaluation_room"],
        )
        + center
    )
    prediction = residual_prediction + baseline_log_price[arrays["price_validation_positions"]]
    return best, prediction, history


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data = pd.read_csv(DATA, low_memory=False, dtype={"record_id": "string"})
    data.observation_date = pd.to_datetime(data.observation_date)
    price_mask = data.source.eq("DVF").to_numpy()
    price_frame = data.loc[price_mask].reset_index(drop=True)
    price_train = price_frame.temporal_split.eq("train").to_numpy()
    price_validation = price_frame.temporal_split.eq("validation").to_numpy()
    price_train_global = np.flatnonzero(price_mask)[price_train]
    price_validation_global = np.flatnonzero(price_mask)[price_validation]

    manifest = json.loads(MANIFEST.read_text())
    names = manifest["feature_names"]
    raw = tt.raw_feature_frame(data, names)
    train_global_mask = data.temporal_split.eq("train").to_numpy()
    validation_global_mask = data.temporal_split.eq("validation").to_numpy()
    arrays = tt.prepare(data, raw, train_global_mask, validation_global_mask)
    arrays["price_train_positions"] = np.arange(len(price_frame))[price_train]
    arrays["price_validation_positions"] = np.arange(len(price_frame))[price_validation]
    groups = tt.build_groups(manifest)

    full_matrix = tt.raw_feature_frame(data, names)
    full_matrix["log1p_room_count"] = np.log1p(pd.to_numeric(data.room_count, errors="coerce"))
    x_train = full_matrix.iloc[price_train_global]
    x_validation = full_matrix.iloc[price_validation_global]

    direct_transformer = json.loads(
        (MAIN / "results/transformer/development/price_seed42/metadata.json").read_text()
    )["reported_full_validation_price"]
    direct_xgb_rows = pd.read_csv(MAIN / "results/classical/validation_tuning.csv")
    direct_xgb = (
        direct_xgb_rows.loc[
            direct_xgb_rows.task.eq("price")
            & direct_xgb_rows.model.eq("XGBoost")
            & direct_xgb_rows.feature_set.eq("full_55")
        ]
        .sort_values("validation_log_mae")
        .iloc[0]
    )

    summary = [
        {
            "index_variant": "none",
            "model": "Direct XGBoost full-55",
            "seed": SEED,
            "selected_step_or_tree": int(direct_xgb.best_iteration) + 1,
            "log_mae": float(direct_xgb.validation_log_mae),
            "log_rmse": float(direct_xgb.validation_log_rmse),
            "log_r2": float(direct_xgb.validation_log_r2),
            "euro_mae": float(direct_xgb.validation_euro_mae),
        },
        {
            "index_variant": "none",
            "model": "Direct price-only Transformer",
            "seed": SEED,
            "selected_step_or_tree": int(
                json.loads((MAIN / "results/transformer/development/price_seed42/metadata.json").read_text())["selected_step"]
            ),
            **direct_transformer,
        },
    ]
    predictions = []
    audits = {}

    y_log_price = price_frame.log_sale_price.to_numpy(float)
    y_euro = price_frame.sale_price_eur.to_numpy(float)
    log_area = np.log(price_frame.floor_area_m2.to_numpy(float))

    for variant, start_date in INDEX_VARIANTS.items():
        print(f"\n=== {variant} ===", flush=True)
        index, audit = market_index(price_frame, price_train, start_date)
        audits[variant] = audit
        baseline_log_price = index + log_area
        residual = y_log_price - baseline_log_price

        baseline_prediction = baseline_log_price[price_validation]
        baseline_result = metrics(
            y_log_price[price_validation], baseline_prediction, y_euro[price_validation]
        )
        summary.append({
            "index_variant": variant,
            "model": "Lagged local index only",
            "seed": None,
            "selected_step_or_tree": None,
            **baseline_result,
        })

        xgb_candidates = []
        for config in XGB_CONFIGS:
            model = XGBRegressor(
                objective="reg:squarederror",
                eval_metric="mae",
                n_estimators=1200,
                early_stopping_rounds=60,
                random_state=SEED,
                n_jobs=4,
                tree_method="hist",
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                **config,
            )
            model.fit(
                x_train,
                residual[price_train],
                eval_set=[(x_validation, residual[price_validation])],
                verbose=False,
            )
            residual_prediction = model.predict(x_validation)
            prediction = residual_prediction + baseline_log_price[price_validation]
            value = metrics(y_log_price[price_validation], prediction, y_euro[price_validation])
            xgb_candidates.append((value["log_mae"], model, prediction, value, config))
        _, selected_xgb, xgb_prediction, xgb_value, xgb_config = min(xgb_candidates, key=lambda x: x[0])
        summary.append({
            "index_variant": variant,
            "model": "Residual XGBoost full-55",
            "seed": SEED,
            "selected_step_or_tree": int(selected_xgb.best_iteration) + 1,
            **xgb_config,
            **xgb_value,
        })

        transformer_value, transformer_prediction, history = transformer_screen(
            arrays, groups, residual, baseline_log_price
        )
        summary.append({
            "index_variant": variant,
            "model": "Residual price-only Transformer",
            "seed": SEED,
            "selected_step_or_tree": transformer_value.pop("selected_step"),
            **transformer_value,
        })
        pd.DataFrame(history).to_csv(OUT / f"transformer_curve_{variant}.csv", index=False)

        frame = price_frame.loc[price_validation, [
            "record_id", "observation_date", "property_type", "postcode",
            "commune_code", "floor_area_m2", "sale_price_eur", "log_sale_price",
        ]].copy()
        frame["index_variant"] = variant
        frame["market_index_log_price_per_m2"] = index[price_validation]
        frame["index_only_prediction"] = baseline_prediction
        frame["xgboost_residual_prediction"] = xgb_prediction
        frame["transformer_residual_prediction"] = transformer_prediction
        predictions.append(frame)

    result = pd.DataFrame(summary)
    result["delta_vs_direct_same_model"] = np.nan
    direct_lookup = {
        "Residual XGBoost full-55": float(direct_xgb.validation_log_mae),
        "Residual price-only Transformer": float(direct_transformer["log_mae"]),
    }
    for model, direct in direct_lookup.items():
        mask = result.model.eq(model)
        result.loc[mask, "delta_vs_direct_same_model"] = result.loc[mask, "log_mae"] - direct
    result.to_csv(OUT / "validation_2024_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "validation_2024_predictions.csv.gz", index=False)
    (OUT / "market_index_audit.json").write_text(json.dumps(audits, indent=2) + "\n")

    best_transformer = result.loc[result.model.eq("Residual price-only Transformer")].sort_values("log_mae").iloc[0]
    best_xgb = result.loc[result.model.eq("Residual XGBoost full-55")].sort_values("log_mae").iloc[0]
    readme = f"""# Lagged local market index + residual prediction: quick screen

## Boundary

- Development only: train 2021--2023, evaluate 2024.
- The 2025 test labels are not used or opened by this script.
- One fixed neural seed ({SEED}); this is a screening result, not a final claim.
- The index is a cutoff-frozen, smoothed log-price-per-square-metre hierarchy:
  property type -> commune/type -> postcode/type.
- Training rows use leave-one-out index statistics.

## Key comparison

- Direct XGBoost validation Log-MAE: {float(direct_xgb.validation_log_mae):.6f}
- Best residual XGBoost: {best_xgb.log_mae:.6f} ({best_xgb.index_variant}; delta {best_xgb.delta_vs_direct_same_model:+.6f})
- Direct price-only Transformer validation Log-MAE: {float(direct_transformer['log_mae']):.6f}
- Best residual Transformer: {best_transformer.log_mae:.6f} ({best_transformer.index_variant}; delta {best_transformer.delta_vs_direct_same_model:+.6f})

Lower Log-MAE is better.  Promotion would require a predefined multi-seed and
rolling-time validation experiment; this screen alone must not be mixed into the
frozen final registry.
"""
    (HERE / "README.md").write_text(readme)
    manifest_out = {
        "status": "PASS",
        "purpose": "exploratory validation-only residual-target screen",
        "train_period": "2021-2023",
        "evaluation_period": "2024",
        "test_2025_used": False,
        "seed": SEED,
        "index_variants": list(INDEX_VARIANTS),
        "elapsed_seconds": time.time() - started,
        "device": str(tt.DEVICE),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest_out, indent=2) + "\n")
    print("\n" + result.to_string(index=False), flush=True)
    print(f"\nElapsed: {manifest_out['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
