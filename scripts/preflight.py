#!/usr/bin/env python3
"""Check the release environment and optional data, without training or network I/O."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "requests")
DATA_DIR = ROOT / "experiments/temporal_main_experiment_2026_09_14/data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--hash-data", action="store_true", help="Read the entire compressed data file.")
    args = parser.parse_args()
    errors = []
    for name in PACKAGES:
        try:
            print(f"OK dependency: {name} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing dependency: {name}")
    for path in sorted(ROOT.rglob("*.py")):
        if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(ROOT).parts):
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"Syntax check failed: {path.relative_to(ROOT)}: {exc}")
    expected = json.loads((ROOT / "configs/feature_manifest.json").read_text())
    names = expected["feature_names"]
    groups = expected["feature_groups"]
    flat = [name for group in groups.values() for name in group]
    if len(names) != 55 or len(groups) != 9 or sorted(flat) != sorted(names) or len(set(flat)) != 55:
        errors.append("Feature schema must partition 55 unique inputs into nine groups.")
    path = DATA_DIR / "temporal_model_ready_55.csv.gz"
    manifest = DATA_DIR / "feature_manifest.json"
    if not path.exists() or not manifest.exists():
        message = "Model-ready data/manifest absent (intentionally excluded from the public release)."
        if args.require_data or args.hash_data:
            errors.append(message + " See docs/DATA.md.")
        else:
            print("NOTE " + message)
    else:
        actual = json.loads(manifest.read_text())
        if actual.get("feature_names") != names or actual.get("feature_groups") != groups:
            errors.append("Local feature manifest differs from the archived feature order/groups.")
        with gzip.open(path, "rt", newline="") as handle:
            columns = next(csv.reader(handle))
        required = json.loads((ROOT / "configs/data_columns.json").read_text())
        missing = sorted(set(required) - set(columns))
        if missing:
            errors.append("Missing data columns: " + ", ".join(missing))
        else:
            print("OK data header (not a full row/content validation)")
        if args.hash_data:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            reference = json.loads((ROOT / "configs/data_reference.json").read_text())["output_sha256"]
            if digest.hexdigest() != reference:
                errors.append("Data hash differs from the archived compressed file; exact snapshot not verified.")
            else:
                print("OK archived data SHA-256")
    for message in errors:
        print("ERROR " + message)
    print("No models trained, data downloaded or files modified.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
