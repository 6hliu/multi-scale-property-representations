#!/usr/bin/env python3
"""Frozen-model entity and within-2025 temporal sensitivity analyses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE / "vendor"))
import xgboost as xgb  # noqa: E402
import run_classical_baselines as classical  # noqa: E402


def main() -> None:
    out = HERE / "results/sensitivity"; out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(HERE / "data/temporal_model_ready_55.csv.gz", low_memory=False, dtype={"record_id": "string"})
    data["observation_date"] = pd.to_datetime(data.observation_date)
    manifest = json.loads((HERE / "data/feature_manifest.json").read_text())
    features = classical.feature_frame(data, manifest["feature_names"])
    price_features = classical.add_room(features, data)
    test = data.temporal_split.eq("test")

    price_mask = test & data.source.eq("DVF")
    price_booster = xgb.Booster(); price_booster.load_model(HERE / "results/classical/xgboost_price_full_55.ubj")
    price_frame = data.loc[price_mask].copy().reset_index(drop=True)
    price_prediction = price_booster.predict(xgb.DMatrix(price_features.loc[price_mask], feature_names=price_features.columns.tolist()))
    price_output = price_frame[["record_id", "entity_id", "observation_date", "sale_price_eur", "log_sale_price", "unseen_entity_in_year"]].copy()
    price_output["prediction"] = price_prediction
    price_output.to_csv(out / "xgboost_price_test_predictions.csv.gz", index=False)

    energy_mask = test & data.source.eq("DPE")
    energy_booster = xgb.Booster(); energy_booster.load_model(HERE / "results/classical/xgboost_energy_full_55.ubj")
    energy_frame = data.loc[energy_mask].copy().reset_index(drop=True)
    energy_probability = energy_booster.predict(xgb.DMatrix(features.loc[energy_mask], feature_names=features.columns.tolist()))
    energy_prediction = energy_probability.argmax(1)
    energy_truth = energy_frame.energy_class.map(classical.LABELS).to_numpy(int)
    energy_output = energy_frame[["record_id", "entity_id", "observation_date", "energy_class"]].copy()
    energy_output["prediction"] = energy_prediction
    energy_output.to_csv(out / "xgboost_energy_test_predictions.csv.gz", index=False)

    rows = []
    all_price = classical.price_metrics(price_frame.log_sale_price.to_numpy(), price_prediction, price_frame.sale_price_eur.to_numpy())
    rows.append({"model": "XGBoost full_55", "task": "price", "population": "all_2025", "n": len(price_frame), **all_price})
    unseen = price_frame.unseen_entity_in_year.to_numpy(bool)
    rows.append({"model": "XGBoost full_55", "task": "price", "population": "unseen_entity_2025", "n": int(unseen.sum()), **classical.price_metrics(price_frame.log_sale_price.to_numpy()[unseen], price_prediction[unseen], price_frame.sale_price_eur.to_numpy()[unseen])})
    rows.append({"model": "XGBoost full_55", "task": "energy", "population": "all_2025", "n": len(energy_frame), **classical.energy_metrics(energy_truth, energy_prediction)})
    entity_weight = 1 / energy_frame.entity_id.map(energy_frame.entity_id.value_counts()).to_numpy(float)
    rows.append({"model": "XGBoost full_55", "task": "energy", "population": "entity_equal_2025", "n": len(energy_frame), **{
        "macro_f1": float(classical.f1_score(energy_truth, energy_prediction, labels=list(range(7)), average="macro", zero_division=0, sample_weight=entity_weight)),
        "balanced_accuracy": float(classical.recall_score(energy_truth, energy_prediction, labels=list(range(7)), average="macro", zero_division=0, sample_weight=entity_weight)),
        "accuracy": float(classical.accuracy_score(energy_truth, energy_prediction, sample_weight=entity_weight)),
    }})
    pd.DataFrame(rows).to_csv(out / "entity_sensitivity.csv", index=False)

    price_month = []
    for month, group in price_output.groupby(price_output.observation_date.dt.month):
        price_month.append({"month": int(month), "n": len(group), **classical.price_metrics(group.log_sale_price.to_numpy(), group.prediction.to_numpy(), group.sale_price_eur.to_numpy())})
    pd.DataFrame(price_month).to_csv(out / "xgboost_price_metrics_by_2025_month.csv", index=False)
    energy_output["truth"] = energy_truth
    energy_month = []
    for month, group in energy_output.groupby(energy_output.observation_date.dt.month):
        energy_month.append({"month": int(month), "n": len(group), **classical.energy_metrics(group.truth.to_numpy(), group.prediction.to_numpy())})
    pd.DataFrame(energy_month).to_csv(out / "xgboost_energy_metrics_by_2025_month.csv", index=False)

    dvf = data.loc[data.source.eq("DVF")].copy()
    drift = dvf.groupby("observation_year").agg(
        sales=("record_id", "size"), median_price_eur=("sale_price_eur", "median"),
        median_log_price=("log_sale_price", "median"), median_floor_area_m2=("floor_area_m2", "median"),
    ).reset_index()
    drift["median_price_change"] = drift.median_price_eur.pct_change()
    drift.to_csv(out / "dvf_yearly_drift.csv", index=False)
    energy_distribution = data.loc[data.source.eq("DPE")].groupby(["observation_year", "energy_class"]).size().rename("rows").reset_index()
    energy_distribution["share"] = energy_distribution.rows / energy_distribution.groupby("observation_year").rows.transform("sum")
    energy_distribution.to_csv(out / "dpe_yearly_class_distribution.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(drift.to_string(index=False))


if __name__ == "__main__":
    main()
