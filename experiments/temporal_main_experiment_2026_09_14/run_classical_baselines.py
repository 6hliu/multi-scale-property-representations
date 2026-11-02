#!/usr/bin/env python3
"""Validation-selected temporal linear and XGBoost baselines."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# XGBoost is kept in this experiment's isolated vendor directory so the
# student's global Python installation is not modified.
sys.path.append(str(Path(__file__).resolve().parent / "vendor"))
from xgboost import XGBClassifier, XGBRegressor


HERE = Path(__file__).resolve().parent
DATA = HERE / "data/temporal_model_ready_55.csv.gz"
MANIFEST = HERE / "data/feature_manifest.json"
OUT = HERE / "results/classical"
LABELS = dict(zip("ABCDEFG", range(7)))
SEED = 42
XGB_CONFIGS = (
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 4, "learning_rate": 0.10, "min_child_weight": 10},
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def feature_frame(data: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(index=data.index)
    for name in names:
        if name == "property_type_is_house":
            frame[name] = data.property_type.eq("House").astype(float)
        elif name == "log_floor_area":
            frame[name] = np.log(data.floor_area_m2.astype(float))
        elif "_count_" in name or name.endswith("_nearest_distance_m"):
            frame[name] = np.log1p(pd.to_numeric(data[name], errors="coerce"))
        elif name.startswith("bdnb_") and not name.endswith("_missing"):
            frame[name] = np.log1p(pd.to_numeric(data[name], errors="coerce").clip(lower=0))
        else:
            frame[name] = pd.to_numeric(data[name], errors="coerce")
    return frame


def feature_sets(manifest: dict) -> dict[str, list[str]]:
    groups = manifest["feature_groups"]
    intrinsic = groups["intrinsic"]
    coordinates = groups["coordinates"]
    temporal = groups["temporal"]
    sets = {
        "intrinsic_only": intrinsic,
        "intrinsic_coordinates": intrinsic + coordinates,
        "intrinsic_coordinates_time": intrinsic + coordinates + temporal,
        "intrinsic_250m": intrinsic + groups["osm_250m"],
        "intrinsic_500m": intrinsic + groups["osm_500m"],
        "intrinsic_1000m": intrinsic + groups["osm_1000m"],
        "intrinsic_all_osm": intrinsic + groups["osm_250m"] + groups["osm_500m"] + groups["osm_1000m"],
        "intrinsic_coordinates_all_osm": intrinsic + coordinates + groups["osm_250m"] + groups["osm_500m"] + groups["osm_1000m"],
        "full_55": manifest["feature_names"],
    }
    for key, value in sets.items():
        if len(value) != len(set(value)):
            raise AssertionError(key)
    return sets


def price_metrics(y: np.ndarray, prediction: np.ndarray, euro: np.ndarray) -> dict:
    predicted_euro = np.exp(np.clip(prediction, np.log(10_000), np.log(20_000_000)))
    return {
        "log_mae": float(mean_absolute_error(y, prediction)),
        "log_rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "log_r2": float(r2_score(y, prediction)),
        "euro_mae": float(mean_absolute_error(euro, predicted_euro)),
    }


def energy_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "macro_f1": float(f1_score(y, prediction, labels=list(range(7)), average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(y, prediction, labels=list(range(7)), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
    }


def sqrt_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=7).astype(float)
    weights = np.sqrt(len(y) / (7 * np.maximum(counts, 1)))
    weights /= np.average(weights, weights=counts)
    return weights[y]


def add_room(frame: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["log1p_room_count"] = np.log1p(pd.to_numeric(data.room_count, errors="coerce"))
    return result


def refit_regressor(config: dict, n_estimators: int, x, y):
    model = XGBRegressor(
        objective="reg:squarederror", eval_metric="mae", n_estimators=n_estimators,
        random_state=SEED, n_jobs=4, tree_method="hist", subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0, **config,
    )
    model.fit(x, y, verbose=False)
    return model


def refit_classifier(config: dict, n_estimators: int, x, y, sample_weight):
    model = XGBClassifier(
        objective="multi:softprob", num_class=7, eval_metric="mlogloss",
        n_estimators=n_estimators, random_state=SEED, n_jobs=4,
        tree_method="hist", subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, **config,
    )
    model.fit(x, y, sample_weight=sample_weight, verbose=False)
    return model


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    data = pd.read_csv(DATA, low_memory=False, dtype={"record_id": "string"})
    sets = feature_sets(manifest)
    matrices = {name: feature_frame(data, columns) for name, columns in sets.items()}

    dev_train = data.temporal_split.eq("train")
    validation = data.temporal_split.eq("validation")
    refit = data.temporal_split.isin(["train", "validation"])
    test = data.temporal_split.eq("test")
    price = data.source.eq("DVF")
    energy = data.source.eq("DPE")
    masks = {
        "price_train": price & dev_train, "price_validation": price & validation,
        "price_refit": price & refit, "price_test": price & test,
        "energy_train": energy & dev_train, "energy_validation": energy & validation,
        "energy_refit": energy & refit, "energy_test": energy & test,
    }
    py = data.log_sale_price.to_numpy(float)
    euro = data.sale_price_eur.to_numpy(float)
    ey = data.energy_class.map(LABELS).fillna(-1).to_numpy(int)

    results, tuning = [], []
    price_test = masks["price_test"]
    train_center = float(np.median(py[masks["price_refit"]]))
    prediction = np.full(price_test.sum(), train_center)
    results.append({"task": "price", "model": "Naive refit median", "feature_set": "none", "n_test": int(price_test.sum()), **price_metrics(py[price_test], prediction, euro[price_test])})
    energy_test = masks["energy_test"]
    majority = int(np.bincount(ey[masks["energy_refit"]], minlength=7).argmax())
    results.append({"task": "energy", "model": "Naive refit majority", "feature_set": "none", "n_test": int(energy_test.sum()), **energy_metrics(ey[energy_test], np.full(energy_test.sum(), majority))})

    # Interpretable linear baselines use the full representation inputs. Price
    # receives room count as the same task-specific auxiliary used by neural models.
    full_price = add_room(matrices["full_55"], data)
    ridge_candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
        model.fit(full_price.loc[masks["price_train"]], py[masks["price_train"]])
        metric = price_metrics(py[masks["price_validation"]], model.predict(full_price.loc[masks["price_validation"]]), euro[masks["price_validation"]])
        tuning.append({"task": "price", "model": "Ridge", "feature_set": "full_55_plus_room", "alpha": alpha, **{f"validation_{k}": v for k, v in metric.items()}})
        ridge_candidates.append((metric["log_mae"], alpha))
    _, selected_alpha = min(ridge_candidates)
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=selected_alpha))
    ridge.fit(full_price.loc[masks["price_refit"]], py[masks["price_refit"]])
    results.append({"task": "price", "model": "Ridge", "feature_set": "full_55_plus_room", "selected_alpha": selected_alpha, "n_test": int(price_test.sum()), **price_metrics(py[price_test], ridge.predict(full_price.loc[price_test]), euro[price_test])})

    class_count = np.bincount(ey[masks["energy_train"]], minlength=7).astype(float)
    class_value = np.sqrt(masks["energy_train"].sum() / (7 * np.maximum(class_count, 1)))
    class_value /= np.average(class_value, weights=class_count)
    logistic = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=1.0, max_iter=350, solver="lbfgs", class_weight={i: float(class_value[i]) for i in range(7)}),
    )
    logistic.fit(matrices["full_55"].loc[masks["energy_train"]], ey[masks["energy_train"]])
    validation_metric = energy_metrics(ey[masks["energy_validation"]], logistic.predict(matrices["full_55"].loc[masks["energy_validation"]]))
    tuning.append({"task": "energy", "model": "Logistic regression", "feature_set": "full_55", **{f"validation_{k}": v for k, v in validation_metric.items()}})
    class_count = np.bincount(ey[masks["energy_refit"]], minlength=7).astype(float)
    class_value = np.sqrt(masks["energy_refit"].sum() / (7 * np.maximum(class_count, 1)))
    class_value /= np.average(class_value, weights=class_count)
    logistic = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=1.0, max_iter=350, solver="lbfgs", class_weight={i: float(class_value[i]) for i in range(7)}),
    )
    logistic.fit(matrices["full_55"].loc[masks["energy_refit"]], ey[masks["energy_refit"]])
    results.append({"task": "energy", "model": "Logistic regression", "feature_set": "full_55", "n_test": int(energy_test.sum()), **energy_metrics(ey[energy_test], logistic.predict(matrices["full_55"].loc[energy_test]))})

    for set_name, matrix in matrices.items():
        # Room count is deliberately task-specific rather than an embedding input.
        price_matrix = add_room(matrices[set_name], data)
        price_candidates = []
        for config in XGB_CONFIGS:
            model = XGBRegressor(
                objective="reg:squarederror", eval_metric="mae", n_estimators=1200,
                early_stopping_rounds=60, random_state=SEED, n_jobs=4,
                tree_method="hist", subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, **config,
            )
            model.fit(
                price_matrix.loc[masks["price_train"]], py[masks["price_train"]],
                eval_set=[(price_matrix.loc[masks["price_validation"]], py[masks["price_validation"]])],
                verbose=False,
            )
            prediction = model.predict(price_matrix.loc[masks["price_validation"]])
            metric = price_metrics(py[masks["price_validation"]], prediction, euro[masks["price_validation"]])
            row = {"task": "price", "model": "XGBoost", "feature_set": set_name, **config, "best_iteration": int(model.best_iteration), **{f"validation_{k}": v for k, v in metric.items()}}
            tuning.append(row); price_candidates.append((metric["log_mae"], config, int(model.best_iteration) + 1))
        _, config, trees = min(price_candidates, key=lambda value: value[0])
        selected = refit_regressor(config, trees, price_matrix.loc[masks["price_refit"]], py[masks["price_refit"]])
        selected.get_booster().save_model(OUT / f"xgboost_price_{set_name}.ubj")
        results.append({"task": "price", "model": "XGBoost", "feature_set": set_name, "selected_trees": trees, **config, "n_test": int(price_test.sum()), **price_metrics(py[price_test], selected.predict(price_matrix.loc[price_test]), euro[price_test])})

        energy_candidates = []
        train_weight = sqrt_weights(ey[masks["energy_train"]])
        for config in XGB_CONFIGS:
            model = XGBClassifier(
                objective="multi:softprob", num_class=7, eval_metric="mlogloss",
                n_estimators=800, random_state=SEED,
                n_jobs=4, tree_method="hist", subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, **config,
            )
            model.fit(
                matrix.loc[masks["energy_train"]], ey[masks["energy_train"]],
                sample_weight=train_weight,
                verbose=False,
            )
            for trees in (100, 200, 300, 400, 600, 800):
                prediction = model.predict(
                    matrix.loc[masks["energy_validation"]],
                    iteration_range=(0, trees),
                )
                metric = energy_metrics(ey[masks["energy_validation"]], prediction)
                row = {"task": "energy", "model": "XGBoost", "feature_set": set_name, **config, "trees": trees, **{f"validation_{k}": v for k, v in metric.items()}}
                tuning.append(row); energy_candidates.append((-metric["macro_f1"], config, trees))
        _, config, trees = min(energy_candidates, key=lambda value: value[0])
        selected = refit_classifier(config, trees, matrix.loc[masks["energy_refit"]], ey[masks["energy_refit"]], sqrt_weights(ey[masks["energy_refit"]]))
        selected.get_booster().save_model(OUT / f"xgboost_energy_{set_name}.ubj")
        results.append({"task": "energy", "model": "XGBoost", "feature_set": set_name, "selected_trees": trees, **config, "n_test": int(energy_test.sum()), **energy_metrics(ey[energy_test], selected.predict(matrix.loc[energy_test]))})
        print("COMPLETE", set_name, flush=True)

    pd.DataFrame(tuning).to_csv(OUT / "validation_tuning.csv", index=False)
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(OUT / "test_results.csv", index=False)
    protocol = {
        "data": str(DATA), "data_sha256": digest(DATA), "feature_manifest_sha256": digest(MANIFEST),
        "development": "2021-2023 train; 2024 validation", "refit": "2021-2024", "test": "2025",
        "xgboost_candidates": list(XGB_CONFIGS), "selection_metrics": {"price": "minimum validation Log-MAE", "energy": "maximum validation Macro-F1"},
        "energy_training": "sqrt inverse-frequency sample weights recomputed on each training period",
        "price_auxiliary": "log1p room_count is included outside the shared 55-feature representation",
        "test_used_for_selection": False, "elapsed_seconds": time.time() - started,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(result_frame.to_string(index=False))


if __name__ == "__main__":
    main()
