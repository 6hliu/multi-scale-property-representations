#!/usr/bin/env python3
"""Grouped Transformer development, frozen refit and 2025 evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent / "vendor"))
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn


# CPU matches the saved main-run manifest; other devices are explicit opt-ins.
DEVICE = torch.device(os.environ.get("PROPERTY_DEVICE", "cpu"))

HERE = Path(__file__).resolve().parent
DATA = HERE / "data/temporal_model_ready_55.csv.gz"
MANIFEST = HERE / "data/feature_manifest.json"
OUT = HERE / "results/transformer"
LABELS = dict(zip("ABCDEFG", range(7)))
# The original three seeds are retained and two fixed robustness seeds are
# added.  The additional seeds do not change model selection or lambda.
SEEDS = (42, 2026, 314, 7, 99)
LAMBDA_CANDIDATES = (0.03, 0.10, 0.20, 0.30)
BATCH = 1024
INTERVAL = 128
SCREEN_STEPS = 256
FINAL_STEPS = 512


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def raw_feature_frame(data: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(index=data.index)
    for name in feature_names:
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


class TaskBalancedPreprocessor:
    """Fit transforms on equal-size DVF and DPE training samples only."""

    def __init__(self, names: list[str]):
        self.names = names
        self.median: pd.Series | None = None
        self.mean: pd.Series | None = None
        self.std: pd.Series | None = None
        self.fit_counts: dict[str, int] = {}

    def fit(self, raw: pd.DataFrame, data: pd.DataFrame, mask: np.ndarray) -> "TaskBalancedPreprocessor":
        indices = {}
        minimum = min(int((mask & data.source.eq(source)).sum()) for source in ("DVF", "DPE"))
        for offset, source in enumerate(("DVF", "DPE")):
            candidates = np.flatnonzero((mask & data.source.eq(source)).to_numpy())
            rng = np.random.default_rng(91000 + offset)
            indices[source] = rng.choice(candidates, size=minimum, replace=False)
        balanced = raw.iloc[np.concatenate([indices["DVF"], indices["DPE"]])]
        self.fit_counts = {"DVF": minimum, "DPE": minimum}
        self.median = balanced.median().fillna(0)
        filled = balanced.fillna(self.median)
        self.mean = filled.mean()
        self.std = filled.std().replace(0, 1).fillna(1)
        return self

    def transform(self, raw: pd.DataFrame) -> np.ndarray:
        if self.median is None or self.mean is None or self.std is None:
            raise RuntimeError("preprocessor is not fitted")
        values = ((raw.fillna(self.median) - self.mean) / self.std).to_numpy(np.float32)
        if not np.isfinite(values).all():
            raise AssertionError("non-finite transformed feature")
        return values

    def metadata(self) -> dict:
        return {
            "features": self.names, "fit_counts": self.fit_counts,
            "median": self.median.to_dict(), "mean": self.mean.to_dict(), "std": self.std.to_dict(),
            "fit_policy": "equal-size deterministic DVF and DPE samples from the applicable training period only",
            "transforms": "log area; log1p OSM counts, distances and positive BDNB values; standardise all",
        }


def build_groups(manifest: dict) -> dict[str, list[int]]:
    names = manifest["feature_names"]
    groups = {key: [names.index(name) for name in values] for key, values in manifest["feature_groups"].items()}
    flat = [index for values in groups.values() for index in values]
    if sorted(flat) != list(range(len(names))) or len(flat) != len(set(flat)):
        raise AssertionError(groups)
    return groups


@dataclass(frozen=True)
class Config:
    d_model: int = 64
    layers: int = 2
    heads: int = 4
    feedforward: int = 128
    dropout: float = 0.10


class TemporalGroupedTransformer(nn.Module):
    def __init__(self, groups: dict[str, list[int]], price_aux_dim: int = 1, config: Config = Config()):
        super().__init__()
        self.groups = groups
        self.group_names = list(groups)
        self.config = config
        self.projections = nn.ModuleDict({name: nn.Linear(len(indices), config.d_model) for name, indices in groups.items()})
        self.token_type = nn.Parameter(torch.empty(len(groups), config.d_model))
        self.price_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.energy_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.heads,
            dim_feedforward=config.feedforward, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.price_head = nn.Sequential(nn.ReLU(), nn.Linear(config.d_model + price_aux_dim, 48), nn.ReLU(), nn.Linear(48, 1))
        self.energy_head = nn.Sequential(nn.ReLU(), nn.Linear(config.d_model, 48), nn.ReLU(), nn.Linear(48, 7))
        nn.init.normal_(self.token_type, std=0.02)
        nn.init.normal_(self.price_token, std=0.02)
        nn.init.normal_(self.energy_token, std=0.02)

    def feature_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.projections[name](x[:, self.groups[name]]) for name in self.group_names], dim=1) + self.token_type.unsqueeze(0)

    def embed(self, x: torch.Tensor, task: str) -> torch.Tensor:
        token = self.price_token if task == "price" else self.energy_token
        encoded = self.transformer(torch.cat([token.expand(len(x), -1, -1), self.feature_tokens(x)], dim=1))
        return self.output_norm(encoded[:, 0])

    def price(self, x: torch.Tensor, room: torch.Tensor) -> torch.Tensor:
        return self.price_head(torch.cat([self.embed(x, "price"), room], dim=1)).squeeze(1)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy_head(self.embed(x, "energy"))


def price_metrics(y, prediction, euro, weights=None) -> dict:
    residual = np.abs(y - prediction)
    squared = (y - prediction) ** 2
    predicted_euro = np.exp(np.clip(prediction, np.log(10_000), np.log(20_000_000)))
    if weights is None:
        weights = np.ones(len(y))
    weights = np.asarray(weights, float)
    return {
        "log_mae": float(np.average(residual, weights=weights)),
        "log_rmse": float(np.sqrt(np.average(squared, weights=weights))),
        "log_r2": float(1 - np.sum(weights * squared) / np.sum(weights * (y - np.average(y, weights=weights)) ** 2)),
        "euro_mae": float(np.average(np.abs(euro - predicted_euro), weights=weights)),
    }


def energy_metrics(y, prediction, weights=None) -> dict:
    return {
        "macro_f1": float(f1_score(y, prediction, labels=list(range(7)), average="macro", zero_division=0, sample_weight=weights)),
        "balanced_accuracy": float(recall_score(y, prediction, labels=list(range(7)), average="macro", zero_division=0, sample_weight=weights)),
        "accuracy": float(accuracy_score(y, prediction, sample_weight=weights)),
    }


def class_weights(y: np.ndarray) -> torch.Tensor:
    count = np.bincount(y, minlength=7).astype(float)
    value = np.sqrt(len(y) / (7 * np.maximum(count, 1)))
    value /= np.average(value, weights=count)
    return torch.tensor(value, dtype=torch.float32)


def prepare(data: pd.DataFrame, raw: pd.DataFrame, train_mask: np.ndarray, evaluation_mask: np.ndarray, preprocessor: TaskBalancedPreprocessor | None = None):
    if preprocessor is None:
        preprocessor = TaskBalancedPreprocessor(raw.columns.tolist()).fit(raw, data, train_mask)
    matrix = preprocessor.transform(raw)
    room_raw = np.log1p(pd.to_numeric(data.room_count, errors="coerce")).to_numpy(float)
    price_train = train_mask & data.source.eq("DVF").to_numpy()
    room_mean = float(np.nanmean(room_raw[price_train])); room_std = float(np.nanstd(room_raw[price_train]))
    if room_std < 1e-8: room_std = 1.0
    room = np.nan_to_num((room_raw - room_mean) / room_std, nan=0.0).astype(np.float32)[:, None]
    energy_all = data.energy_class.map(LABELS).fillna(-1).to_numpy(np.int64)
    arrays = {"preprocessor": preprocessor, "matrix": matrix, "room_mean": room_mean, "room_std": room_std}
    for task, source in (("price", "DVF"), ("energy", "DPE")):
        for name, base_mask in (("train", train_mask), ("evaluation", evaluation_mask)):
            mask = base_mask & data.source.eq(source).to_numpy()
            arrays[f"{task}_{name}_x"] = matrix[mask]
            arrays[f"{task}_{name}_f"] = data.loc[mask].reset_index(drop=True)
            arrays[f"{task}_{name}_y"] = data.loc[mask, "log_sale_price"].to_numpy(np.float32) if task == "price" else energy_all[mask]
            if task == "price": arrays[f"price_{name}_room"] = room[mask]
    arrays["energy_class_weight"] = class_weights(arrays["energy_train_y"])
    # The 2024 energy cohort contains many repeated diagnostics. A fixed,
    # class-stratified 30k subset is used only for frequent checkpoint scoring;
    # the selected checkpoint is also reported on the complete 2024 cohort.
    y = arrays["energy_evaluation_y"]
    rng = np.random.default_rng(20260914)
    selection = []
    for label in range(7):
        candidates = np.flatnonzero(y == label)
        target = max(1, int(round(30_000 * len(candidates) / len(y))))
        selection.extend(rng.choice(candidates, size=min(target, len(candidates)), replace=False).tolist())
    selection = np.asarray(sorted(selection), dtype=np.int64)
    arrays["energy_selection_x"] = arrays["energy_evaluation_x"][selection]
    arrays["energy_selection_y"] = arrays["energy_evaluation_y"][selection]
    return arrays


@torch.no_grad()
def predict(model, x: np.ndarray, task: str, room: np.ndarray | None = None, embeddings: bool = False):
    model.eval(); parts = []
    for start in range(0, len(x), 4096):
        xb = torch.from_numpy(x[start:start + 4096]).to(DEVICE)
        if embeddings: value = model.embed(xb, task)
        elif task == "price": value = model.price(xb, torch.from_numpy(room[start:start + 4096]).to(DEVICE))
        else: value = model.energy(xb).argmax(1)
        parts.append(value.cpu().numpy())
    return np.concatenate(parts)


def validation_price(model, arrays, center):
    pp = predict(model, arrays["price_evaluation_x"], "price", arrays["price_evaluation_room"]) + center
    return price_metrics(arrays["price_evaluation_y"], pp, arrays["price_evaluation_f"].sale_price_eur.to_numpy())


def validation_energy(model, arrays, complete=False):
    prefix = "energy_evaluation" if complete else "energy_selection"
    ep = predict(model, arrays[f"{prefix}_x"], "energy")
    return energy_metrics(arrays[f"{prefix}_y"], ep)


def minimax(price, energy, reference):
    price_gain = (reference["price_log_mae"] - price["log_mae"]) / reference["price_log_mae"]
    energy_gain = (energy["macro_f1"] - reference["energy_macro_f1"]) / max(reference["energy_macro_f1"], 1e-8)
    return float(min(price_gain, energy_gain))


def fit_development(arrays, groups, mode, seed, lam, steps, reference, tag):
    output = OUT / "development" / tag
    metadata_path = output / "metadata.json"
    if metadata_path.exists(): return json.loads(metadata_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    model = TemporalGroupedTransformer(groups).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    huber = nn.HuberLoss(); cross_entropy = nn.CrossEntropyLoss(weight=arrays["energy_class_weight"].to(DEVICE))
    center = float(np.median(arrays["price_train_y"]))
    px = torch.from_numpy(arrays["price_train_x"]).to(DEVICE); py = torch.from_numpy(arrays["price_train_y"]).to(DEVICE)
    pr = torch.from_numpy(arrays["price_train_room"]).to(DEVICE)
    ex = torch.from_numpy(arrays["energy_train_x"]).to(DEVICE); ey = torch.from_numpy(arrays["energy_train_y"]).to(DEVICE)
    rng_p = np.random.default_rng(seed + 10007); rng_e = np.random.default_rng(seed + 20011)
    best_score = -np.inf; best_state = None; history = []; started = time.time()
    for step in range(1, steps + 1):
        model.train(); optimiser.zero_grad(set_to_none=True); losses = {}
        if mode in {"price", "multi"}:
            index = torch.from_numpy(rng_p.integers(len(py), size=BATCH)).to(DEVICE)
            losses["price"] = huber(model.price(px[index], pr[index]), py[index] - center)
        if mode in {"energy", "multi"}:
            index = torch.from_numpy(rng_e.integers(len(ey), size=BATCH)).to(DEVICE)
            losses["energy"] = cross_entropy(model.energy(ex[index]), ey[index])
        total = losses[mode] if mode in {"price", "energy"} else losses["price"] + lam * losses["energy"]
        total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimiser.step()
        if step % INTERVAL == 0:
            pm = validation_price(model, arrays, center) if mode in {"price", "multi"} else None
            em = validation_energy(model, arrays, complete=False) if mode in {"energy", "multi"} else None
            score = -pm["log_mae"] if mode == "price" else em["macro_f1"] if mode == "energy" else minimax(pm, em, reference)
            history.append({"step": step, "selection_score": score, **{f"price_{k}": v for k, v in (pm or {}).items()}, **{f"energy_{k}": v for k, v in (em or {}).items()}})
            if score > best_score:
                best_score = score; selected_step = step; selected_price = pm; selected_energy = em
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            print(f"DEV {tag} step={step} price={pm['log_mae']:.4f}" if pm else f"DEV {tag} step={step}",
                  f"energy={em['macro_f1']:.4f}" if em else "", flush=True)
    if best_state is None: raise AssertionError("checkpoint not selected")
    model.load_state_dict(best_state)
    reported_price = validation_price(model, arrays, center) if mode in {"price", "multi"} else None
    reported_energy = validation_energy(model, arrays, complete=True) if mode in {"energy", "multi"} else None
    metadata = {
        "mode": mode, "seed": seed, "lambda_energy": lam, "selected_step": selected_step,
        "training_steps": steps, "selection_score": best_score,
        "validation_price": selected_price if mode != "energy" else None,
        "validation_energy": selected_energy if mode != "price" else None,
        "reported_full_validation_price": reported_price,
        "reported_full_validation_energy": reported_energy,
        "price_center": center, "elapsed_seconds": time.time() - started,
    }
    torch.save({"state_dict": best_state, "metadata": metadata}, output / "model.pt")
    pd.DataFrame(history).to_csv(output / "validation_curve.csv", index=False)
    dump(metadata_path, metadata)
    return metadata


def fit_refit(arrays, groups, mode, seed, lam, steps, tag):
    output = OUT / "refit" / tag
    metadata_path = output / "metadata.json"
    if metadata_path.exists(): return json.loads(metadata_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    model = TemporalGroupedTransformer(groups).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    huber = nn.HuberLoss(); cross_entropy = nn.CrossEntropyLoss(weight=arrays["energy_class_weight"].to(DEVICE))
    center = float(np.median(arrays["price_train_y"]))
    px = torch.from_numpy(arrays["price_train_x"]).to(DEVICE); py = torch.from_numpy(arrays["price_train_y"]).to(DEVICE); pr = torch.from_numpy(arrays["price_train_room"]).to(DEVICE)
    ex = torch.from_numpy(arrays["energy_train_x"]).to(DEVICE); ey = torch.from_numpy(arrays["energy_train_y"]).to(DEVICE)
    rng_p = np.random.default_rng(seed + 10007); rng_e = np.random.default_rng(seed + 20011)
    started = time.time()
    for step in range(1, steps + 1):
        model.train(); optimiser.zero_grad(set_to_none=True); losses = {}
        if mode in {"price", "multi"}:
            index = torch.from_numpy(rng_p.integers(len(py), size=BATCH)).to(DEVICE)
            losses["price"] = huber(model.price(px[index], pr[index]), py[index] - center)
        if mode in {"energy", "multi"}:
            index = torch.from_numpy(rng_e.integers(len(ey), size=BATCH)).to(DEVICE)
            losses["energy"] = cross_entropy(model.energy(ex[index]), ey[index])
        total = losses[mode] if mode in {"price", "energy"} else losses["price"] + lam * losses["energy"]
        total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimiser.step()
        if step % 512 == 0: print(f"REFIT {tag} step={step}/{steps}", flush=True)
    metadata = {"mode": mode, "seed": seed, "lambda_energy": lam, "training_steps": steps, "price_center": center, "elapsed_seconds": time.time() - started}
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "metadata": metadata}, output / "model.pt")
    dump(metadata_path, metadata)
    return metadata


def load_model(path: Path, groups):
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    model = TemporalGroupedTransformer(groups).to(DEVICE); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    return model, checkpoint["metadata"]


def evaluate_refit(arrays, groups, mode, seed, lam, tag):
    output = OUT / "refit" / tag
    model, metadata = load_model(output, groups); rows = []
    tasks = ("price", "energy") if mode == "multi" else (mode,)
    for task in tasks:
        frame = arrays[f"{task}_evaluation_f"]
        unseen = frame.unseen_entity_in_year.to_numpy(bool)
        if task == "price":
            prediction = predict(model, arrays["price_evaluation_x"], "price", arrays["price_evaluation_room"]) + metadata["price_center"]
            metric = price_metrics(arrays["price_evaluation_y"], prediction, frame.sale_price_eur.to_numpy())
            unseen_metric = price_metrics(arrays["price_evaluation_y"][unseen], prediction[unseen], frame.sale_price_eur.to_numpy()[unseen])
            repeated_metric = price_metrics(arrays["price_evaluation_y"][~unseen], prediction[~unseen], frame.sale_price_eur.to_numpy()[~unseen])
        else:
            prediction = predict(model, arrays["energy_evaluation_x"], "energy")
            metric = energy_metrics(arrays["energy_evaluation_y"], prediction)
            weight = 1 / frame.entity_id.map(frame.entity_id.value_counts()).to_numpy(float)
            entity_equal_metric = energy_metrics(arrays["energy_evaluation_y"], prediction, weight)
            unseen_metric = energy_metrics(arrays["energy_evaluation_y"][unseen], prediction[unseen])
            repeated_metric = energy_metrics(arrays["energy_evaluation_y"][~unseen], prediction[~unseen])
        rows.append({"architecture": "Grouped Transformer", "training": mode, "task": task, "seed": seed, "lambda_energy": lam, "population": "all_2025", "n_test": len(frame), **metric})
        rows.append({"architecture": "Grouped Transformer", "training": mode, "task": task, "seed": seed, "lambda_energy": lam, "population": "unseen_entity_2025", "n_test": int(unseen.sum()), **unseen_metric})
        rows.append({"architecture": "Grouped Transformer", "training": mode, "task": task, "seed": seed, "lambda_energy": lam, "population": "repeated_entity_2025", "n_test": int((~unseen).sum()), **repeated_metric})
        if task == "energy":
            rows.append({"architecture": "Grouped Transformer", "training": mode, "task": task, "seed": seed, "lambda_energy": lam, "population": "entity_equal_2025", "n_test": len(frame), **entity_equal_metric})
        prediction_frame = frame[["source", "record_id", "entity_id", "observation_date", "temporal_split", "unseen_entity_in_year"]].copy()
        prediction_frame["truth"] = arrays[f"{task}_evaluation_y"]; prediction_frame["prediction"] = prediction
        prediction_frame.to_csv(output / f"predictions_{task}.csv.gz", index=False)
    # Price embeddings for RQ3 are exported for every DVF record under the
    # refit transform, including the 2021--2024 candidate pool and 2025 queries.
    if mode in {"price", "multi"}:
        dvf = arrays["all_dvf_f"]
        embedding = predict(model, arrays["all_dvf_x"], "price", embeddings=True)
        np.save(output / "dvf_price_embeddings.npy", embedding.astype(np.float32))
        dvf[["record_id", "parcel_id", "observation_date", "temporal_split"]].to_csv(output / "dvf_embedding_index.csv.gz", index=False)
    dump(output / "test_results.json", rows)
    return rows


def main() -> None:
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True); started = time.time()
    manifest = json.loads(MANIFEST.read_text()); names = manifest["feature_names"]; groups = build_groups(manifest)
    data = pd.read_csv(DATA, low_memory=False, dtype={"record_id": "string"})
    raw = raw_feature_frame(data, names)
    train_mask = data.temporal_split.eq("train").to_numpy(); validation_mask = data.temporal_split.eq("validation").to_numpy()
    dev = prepare(data, raw, train_mask, validation_mask)

    dev_metadata = {}
    for mode in ("price", "energy"):
        tag = f"{mode}_seed42"
        dev_metadata[tag] = fit_development(dev, groups, mode, 42, 0.0, FINAL_STEPS, None, tag)
    reference42 = {"price_log_mae": dev_metadata["price_seed42"]["validation_price"]["log_mae"], "energy_macro_f1": dev_metadata["energy_seed42"]["validation_energy"]["macro_f1"]}
    screen = []
    for lam in LAMBDA_CANDIDATES:
        tag = f"multi_screen_seed42_lambda{lam:g}"
        value = fit_development(dev, groups, "multi", 42, lam, SCREEN_STEPS, reference42, tag)
        screen.append({"lambda_energy": lam, "selection_score": value["selection_score"], "validation_price_log_mae": value["validation_price"]["log_mae"], "validation_energy_macro_f1": value["validation_energy"]["macro_f1"], "selected_step": value["selected_step"]})
    screen_frame = pd.DataFrame(screen).sort_values(["selection_score", "lambda_energy"], ascending=[False, True])
    selected_lambda = float(screen_frame.iloc[0].lambda_energy)
    screen_frame.to_csv(OUT / "lambda_validation_screen.csv", index=False)
    dump(OUT / "SELECTION.json", {"configuration": Config().__dict__, "selected_lambda_energy": selected_lambda, "criterion": "maximum worst relative validation gain against matched single-task models", "selection_seed": 42, "test_used": False})

    for seed in SEEDS:
        for mode in ("price", "energy"):
            tag = f"{mode}_seed{seed}"
            if tag not in dev_metadata: dev_metadata[tag] = fit_development(dev, groups, mode, seed, 0.0, FINAL_STEPS, None, tag)
        reference = {"price_log_mae": dev_metadata[f"price_seed{seed}"]["validation_price"]["log_mae"], "energy_macro_f1": dev_metadata[f"energy_seed{seed}"]["validation_energy"]["macro_f1"]}
        tag = f"multi_seed{seed}_lambda{selected_lambda:g}"
        dev_metadata[tag] = fit_development(dev, groups, "multi", seed, selected_lambda, FINAL_STEPS, reference, tag)
    pd.DataFrame([
        {"run": key, **{k: v for k, v in value.items() if k not in {"validation_price", "validation_energy"}},
         "validation_price_log_mae": (value.get("validation_price") or {}).get("log_mae"),
         "validation_energy_macro_f1": (value.get("validation_energy") or {}).get("macro_f1")}
        for key, value in dev_metadata.items() if "screen" not in key
    ]).to_csv(OUT / "development_selected_runs.csv", index=False)

    # Frozen refit: fit every transform on 2021--2024 and run exactly the
    # validation-selected number of optimisation steps for the matched seed/mode.
    refit_mask = data.temporal_split.isin(["train", "validation"]).to_numpy(); test_mask = data.temporal_split.eq("test").to_numpy()
    refit = prepare(data, raw, refit_mask, test_mask)
    dvf_mask = data.source.eq("DVF").to_numpy()
    refit["all_dvf_x"] = refit["matrix"][dvf_mask]
    refit["all_dvf_f"] = data.loc[dvf_mask].reset_index(drop=True)
    del refit["matrix"]
    dump(OUT / "development_preprocessing.json", dev["preprocessor"].metadata())
    dump(OUT / "refit_preprocessing.json", refit["preprocessor"].metadata())
    results = []
    for seed in SEEDS:
        for mode, lam, key in (
            ("price", 0.0, f"price_seed{seed}"),
            ("energy", 0.0, f"energy_seed{seed}"),
            ("multi", selected_lambda, f"multi_seed{seed}_lambda{selected_lambda:g}"),
        ):
            steps = int(dev_metadata[key]["selected_step"])
            tag = f"{mode}_seed{seed}_lambda{lam:g}"
            fit_refit(refit, groups, mode, seed, lam, steps, tag)
            results.extend(evaluate_refit(refit, groups, mode, seed, lam, tag))
    frame = pd.DataFrame(results); frame.to_csv(OUT / "test_results_by_seed.csv", index=False)
    metrics = ["log_mae", "log_rmse", "log_r2", "euro_mae", "macro_f1", "balanced_accuracy", "accuracy"]
    summary = frame.groupby(["architecture", "training", "task", "population"])[metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(OUT / "test_results_summary.csv", index=False)
    dump(OUT / "run_manifest.json", {
        "status": "PASS", "data": str(DATA), "data_sha256": digest(DATA), "protocol_sha256": digest(HERE / "PROTOCOL.md"),
        "architecture": {**Config().__dict__, "feature_groups": list(groups), "shared_feature_count": len(names), "embedding_dim": 64},
        "development": "2021-2023 train and 2024 checkpoint selection", "refit": "2021-2024; selected steps fixed", "test": "2025",
        "task_balancing": "one independent DVF batch and one independent DPE batch per shared update",
        "energy_weighting": "square-root inverse training-class-frequency cross entropy",
        "selected_lambda_energy": selected_lambda, "seeds": list(SEEDS), "test_used_for_selection": False,
        "device": str(DEVICE), "torch_version": torch.__version__, "elapsed_seconds": time.time() - started,
    })
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
