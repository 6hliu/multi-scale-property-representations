"""Small synthetic tests of the included implementations; no real data or training run."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import unittest
from unittest.mock import patch

os.environ["PROPERTY_DEVICE"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS / "temporal_main_experiment_2026_09_14"))
import numpy as np
import pandas as pd
import torch
import run_temporal_transformer as tt

torch.set_num_threads(1)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ra = load("release_final_ra", "temporal_retrieval_aware_residual_fusion_2026_09_24/run_experiment.py")
mt = load("release_final_mt", "task2_residual_main_2026_09_26/run_task2_residual.py")
schema = json.loads((ROOT / "configs/feature_manifest.json").read_text())


class ReleaseSmokeTests(unittest.TestCase):
    def test_all_included_modules_import_without_running(self):
        for i, path in enumerate(sorted(EXPERIMENTS.rglob("*.py"))):
            load(f"release_import_{i}", path.relative_to(EXPERIMENTS))

    def test_feature_groups_and_forward_shapes(self):
        groups = tt.build_groups(schema)
        self.assertEqual(len(groups), 9)
        self.assertEqual(sum(map(len, groups.values())), 55)
        model = tt.TemporalGroupedTransformer(groups).eval()
        x, room = torch.randn(4, 55), torch.randn(4, 1)
        with torch.no_grad():
            z = model.embed(x, "price")
            self.assertEqual(tuple(z.shape), (4, 64))
            self.assertEqual(tuple(model.feature_tokens(x).shape), (4, 9, 64))
            self.assertEqual(tuple(model.price(x, room).shape), (4,))
            logits = model.energy(x)
            self.assertEqual(tuple(logits.shape), (4, 7))
            torch.testing.assert_close(logits.softmax(1).sum(1), torch.ones(4))

    def test_shared_loss_backward(self):
        model = tt.TemporalGroupedTransformer(tt.build_groups(schema))
        x, room = torch.randn(4, 55), torch.randn(4, 1)
        price_loss = torch.nn.HuberLoss()(model.price(x, room), torch.randn(4))
        energy_loss = torch.nn.CrossEntropyLoss()(model.energy(x), torch.tensor([0, 2, 4, 6]))
        (price_loss + 0.2 * energy_loss).backward()
        grad = model.projections["intrinsic"].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())

    def test_retrieval_loss_backward(self):
        model = tt.TemporalGroupedTransformer(tt.build_groups(schema))
        x = torch.randn(8, 55)
        tuples = pd.DataFrame({
            "anchor_index": [0, 1], "positive_index": [2, 3],
            "negative_1_index": [4, 4], "negative_2_index": [5, 5],
            "negative_3_index": [6, 6], "negative_4_index": [7, 7],
        })
        with patch.object(ra.metric, "METRIC_BATCH", 2):
            loss, _, _, _ = ra.metric.metric_loss(model, x, tuples, np.random.default_rng(42))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.price_token.grad).all())

    def test_tiny_xgboost_fit(self):
        # Match the separate-process training stages. The local release-check
        # environment crashed when fitting XGBoost after PyTorch backpropagation
        # in the same process; do not silently assert that mixed mode is tested.
        code = """
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
x = np.random.default_rng(42).normal(size=(70, 5))
reg = XGBRegressor(n_estimators=2, max_depth=2, n_jobs=1, random_state=42)
reg.fit(x, x[:, 0])
assert np.isfinite(reg.predict(x)).all()
clf = XGBClassifier(n_estimators=2, max_depth=2, n_jobs=1, random_state=42,
                    objective="multi:softprob", num_class=7)
clf.fit(x, np.arange(70) % 7)
np.testing.assert_allclose(clf.predict_proba(x).sum(1), 1., rtol=1e-6)
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_market_index_cutoff_and_self_removal(self):
        frame = pd.DataFrame({
            "record_id": ["synthetic-a", "synthetic-b", "synthetic-c", "synthetic-test"],
            "property_type": ["Apartment"] * 4,
            "commune_code": ["synthetic"] * 4, "postcode": ["synthetic"] * 4,
            "floor_area_m2": [50.] * 4, "log_sale_price": [11., 12., 13., 14.],
            "observation_date": pd.to_datetime(["2024-02-01", "2024-04-01", "2024-09-01", "2025-02-01"]),
        })
        mask = np.array([True, True, True, False])
        baseline, audit = ra.market_index(frame, mask, pd.Timestamp("2024-01-01"))
        changed = frame.copy()
        changed.loc[3, "log_sale_price"] = 30.
        second, _ = ra.market_index(changed, mask, pd.Timestamp("2024-01-01"))
        np.testing.assert_allclose(baseline, second)
        changed = frame.copy()
        changed.loc[0, "log_sale_price"] = 20.
        third, _ = ra.market_index(changed, mask, pd.Timestamp("2024-01-01"))
        self.assertAlmostEqual(float(baseline[0]), float(third[0]), places=5)
        self.assertEqual(audit["base_rows"], 3)

    def test_neighbour_fusion_formula(self):
        context = {"direct_prediction": np.array([12., 13.]), "query_baseline": np.array([10., 10.])}
        neighbour = np.array([1., -1.])
        config = {"model_weight": .75, "geo_beta": .1, "intrinsic_beta": .25,
                  "k": 10, "aggregation": "median", "temperature": .25}
        with patch.object(ra.fusion, "neighbour_residual",
                          return_value=(neighbour, np.zeros((2, 10)), np.ones(2) * 10)):
            actual, _, _ = ra.fusion.apply(context, config)
        np.testing.assert_allclose(actual, [11.75, 12.])

    def test_paired_bootstrap_gain_sign(self):
        frame = pd.DataFrame({"latitude": [45., 45., 45.1, 45.1],
                              "longitude": [4., 4., 4.1, 4.1], "log_sale_price": [12.] * 4})
        with patch.object(ra.fusion, "REPETITIONS", 40):
            result = ra.fusion.bootstrap(frame, np.array([12.2] * 4), np.array([12.1] * 4), "synthetic")
        self.assertAlmostEqual(result["point_gain"], .1)
        self.assertAlmostEqual(result["ci95_lower"], .1)
        self.assertAlmostEqual(result["ci95_upper"], .1)
        self.assertEqual(result["bootstrap_repetitions"], 40)


if __name__ == "__main__":
    unittest.main()
