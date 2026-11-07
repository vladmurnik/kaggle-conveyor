"""Source of the standalone ``submission.py`` written into inference bundles.

Keep this module dependency-free at import time: it is copied verbatim into
the bundle, where only numpy/pandas/joblib and the generated
``feature_runtime.py`` are available.
"""

from __future__ import annotations

SUBMISSION_SCRIPT = r'''#!/usr/bin/env python3
"""Standalone kconv inference bundle.

Usage:
    python submission.py --test test.csv --output submission.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:  # bundled runtime
    from feature_runtime import FeatureRuntime
except ImportError:  # unit tests / in-repo execution
    from kconv.features.runtime import FeatureRuntime

ROOT = Path(__file__).resolve().parent


def _read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if name.endswith((".csv", ".csv.gz", ".tsv")):
        separator = "\t" if name.endswith(".tsv") else ","
        return pd.read_csv(path, sep=separator)
    raise SystemExit(f"Unsupported test format: {path}")


def _predict(model, x: pd.DataFrame, task: str) -> np.ndarray:
    if task == "regression":
        return np.asarray(model.predict(x), dtype=float)
    proba = np.asarray(model.predict_proba(x), dtype=float)
    if task == "binary":
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.reshape(-1)
    if proba.ndim != 2:
        raise SystemExit("A multiclass model returned a 1-D probability array.")
    return proba


def _load_model_predictions(root: Path, metadata: dict, test: pd.DataFrame) -> dict[str, np.ndarray]:
    task = metadata["task"]
    predictions: dict[str, np.ndarray] = {}
    for model_name in metadata["models"]:
        model_root = root / "models" / model_name / "folds"
        if not model_root.is_dir():
            raise SystemExit(f"Missing model folds for {model_name}: {model_root}")
        fold_dirs = sorted(
            (path for path in model_root.iterdir() if path.is_dir()),
            key=lambda path: int(path.name.rsplit("_", 1)[-1]),
        )
        if not fold_dirs:
            raise SystemExit(f"No fold artifacts for {model_name}")
        fold_predictions: list[np.ndarray] = []
        for fold_dir in fold_dirs:
            state = joblib.load(fold_dir / "features.joblib")
            if not isinstance(state, dict):
                raise SystemExit(
                    f"{fold_dir / 'features.joblib'} is an old kconv FeatureBuilder; "
                    "re-export the bundle with this kconv version."
                )
            builder = FeatureRuntime(state)
            x_test = builder.transform(test)
            seed_files = sorted(
                fold_dir.glob("seed_*.joblib"),
                key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
            )
            if not seed_files:
                raise SystemExit(f"No seed artifacts in {fold_dir}")
            seed_predictions: list[np.ndarray] = []
            for path in seed_files:
                artifact = joblib.load(path)
                model = artifact.get("model") if isinstance(artifact, dict) else artifact
                if model is None:
                    raise SystemExit(f"Seed artifact has no model: {path}")
                seed_predictions.append(_predict(model, x_test, task))
            fold_predictions.append(np.mean(np.stack(seed_predictions, axis=0), axis=0))
        predictions[model_name] = np.mean(np.stack(fold_predictions, axis=0), axis=0)
    return predictions


def _normalize_probability_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(row_sums, 1e-12)


def _combine(
    predictions: dict[str, np.ndarray],
    metadata: dict,
) -> np.ndarray:
    names = metadata["models"]
    arrays = [predictions[name] for name in names]
    task = metadata["task"]
    multiclass = task == "multiclass"
    method = metadata["method"]

    if method == "best_single":
        selected = metadata["best_single"]["name"]
        return predictions[selected]

    if method == "mean":
        combined = np.average(np.stack(arrays, axis=0), axis=0)
        return _normalize_probability_matrix(combined) if multiclass else combined

    if method == "weighted":
        weights = np.asarray([metadata["weights"].get(name, 0.0) for name in names], dtype=float)
        if weights.sum() <= 0:
            raise SystemExit("Blend weights are empty or non-positive.")
        if multiclass:
            combined = np.tensordot(weights, np.stack(arrays, axis=0), axes=(0, 0))
            return _normalize_probability_matrix(combined)
        return np.average(np.stack(arrays, axis=0), axis=0, weights=weights)

    if method == "rank_mean":
        if metadata.get("metric") != "roc_auc":
            raise SystemExit("rank_mean is only valid for roc_auc bundles.")
        if multiclass:
            ranks = [np.apply_along_axis(rankdata, 0, array) for array in arrays]
            return _normalize_probability_matrix(np.mean(ranks, axis=0))
        ranks = np.stack([rankdata(array) for array in arrays], axis=0)
        combined = ranks.mean(axis=0)
        span = combined.max() - combined.min()
        return (combined - combined.min()) / span if span > 1e-12 else combined

    if method == "stack":
        meta_path = ROOT / "models" / "meta_model.joblib"
        if not meta_path.is_file():
            raise SystemExit(f"Stacking metadata requires {meta_path}")
        if multiclass:
            x_meta = np.concatenate(arrays, axis=1)
        else:
            x_meta = np.column_stack(arrays)
        estimator = joblib.load(meta_path)
        if task == "regression":
            return np.asarray(estimator.predict(x_meta), dtype=float)
        proba = np.asarray(estimator.predict_proba(x_meta), dtype=float)
        if task == "binary":
            values = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.reshape(-1)
            return np.clip(values, 0.0, 1.0)
        return _normalize_probability_matrix(proba)

    raise SystemExit(f"Unsupported blend method: {method}")


def _to_submission(prediction: np.ndarray, metadata: dict) -> np.ndarray:
    task = metadata["task"]
    metric = metadata["metric"]
    if task == "regression":
        output = np.asarray(prediction, dtype=float)
    elif task == "multiclass":
        prediction = np.asarray(prediction)
        classes = metadata.get("classes")
        if metric in {"roc_auc", "log_loss"}:
            if prediction.ndim != 2:
                raise SystemExit("Multiclass probability metrics need a probability matrix.")
            output = prediction.astype(float)
        else:
            labels = prediction.argmax(axis=1) if prediction.ndim == 2 else prediction.astype(int)
            shape_matches = not (prediction.ndim == 2 and len(classes or []) != prediction.shape[1])
            in_range = bool(labels.size) and int(labels.min()) >= 0 and int(labels.max()) < len(classes or [])
            if isinstance(classes, list) and classes and shape_matches and in_range:
                output = np.asarray(classes)[labels]
            else:
                output = labels
    else:
        probability = np.asarray(prediction, dtype=float)
        if metric in {"roc_auc", "log_loss"}:
            output = probability
        else:
            labels = (probability >= 0.5).astype(int)
            classes = metadata.get("classes")
            output = np.asarray(classes)[labels] if isinstance(classes, list) and len(classes) == 2 else labels

    clip = metadata.get("clip")
    if clip is not None:
        if not np.issubdtype(np.asarray(output).dtype, np.number):
            raise SystemExit("submit.clip cannot be applied to string class labels.")
        output = np.clip(output, clip[0], clip[1])
    digits = metadata.get("round")
    if digits is not None:
        if not np.issubdtype(np.asarray(output).dtype, np.number):
            raise SystemExit("submit.round cannot be applied to string class labels.")
        output = np.round(output, digits)
    return np.asarray(output)


def _build_submission(test: pd.DataFrame, output: np.ndarray, metadata: dict, sample: pd.DataFrame | None) -> pd.DataFrame:
    id_col = metadata.get("id_col")
    target_col = metadata.get("target_col")
    if not id_col or id_col not in test.columns:
        raise SystemExit(f"Cannot find id column '{id_col}' in the test file")
    if not target_col:
        raise SystemExit("metadata.target_col is missing")
    if sample is None:
        frame = pd.DataFrame({id_col: test[id_col].to_numpy()})
        if np.asarray(output).ndim == 2:
            classes = metadata.get("classes") or list(range(np.asarray(output).shape[1]))
            for index, label in enumerate(classes):
                frame[f"{target_col}_{label}"] = np.asarray(output)[:, index]
        else:
            frame[target_col] = output
        return frame

    sample_id = str(sample.columns[0])
    if sample_id != id_col and sample_id not in test.columns:
        raise SystemExit(
            f"Sample id column '{sample_id}' is not present in the test file "
            f"(expected '{id_col}')"
        )
    sample_ids = sample[sample_id].astype(str)
    if sample_ids.duplicated().any():
        raise SystemExit("Sample submission contains duplicate ids")
    test_ids = test[id_col].astype(str)
    if test_ids.duplicated().any():
        raise SystemExit("Test file contains duplicate ids")
    if set(sample_ids) != set(test_ids):
        missing = len(set(test_ids) - set(sample_ids))
        extra = len(set(sample_ids) - set(test_ids))
        raise SystemExit(
            f"Sample/test id mismatch: {missing} test ids missing from sample, {extra} sample ids missing from test"
        )
    non_id = [str(column) for column in sample.columns if str(column) != sample_id]
    output_array = np.asarray(output)
    if output_array.ndim == 2:
        classes = metadata.get("classes") or []
        if len(non_id) != output_array.shape[1]:
            raise SystemExit(
                f"Multiclass probability submission needs {output_array.shape[1]} columns, sample has {len(non_id)}"
            )
        if classes and sorted(non_id) == sorted([str(value) for value in classes]):
            order = [non_id.index(str(value)) for value in classes]
            matrix = output_array[:, order]
        else:
            matrix = output_array
        prediction = pd.DataFrame({"__kconv_id_key": test_ids.to_numpy()})
        for index, column in enumerate(non_id):
            prediction[column] = matrix[:, index]
    else:
        if len(non_id) != 1:
            raise SystemExit(
                f"Sample submission must have exactly one target column, found {len(non_id)}"
            )
        target_col = non_id[0]
        prediction = pd.DataFrame(
            {"__kconv_id_key": test_ids.to_numpy(), target_col: output_array}
        )
    merged = (
        sample[[sample_id]]
        .assign(__kconv_id_key=sample_ids.to_numpy())
        .merge(prediction, on="__kconv_id_key", how="left", validate="one_to_one", sort=False)
        .drop(columns=["__kconv_id_key"])
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate submission.csv from a kconv bundle.")
    parser.add_argument("--test", required=True, type=Path, help="Test CSV/parquet file.")
    parser.add_argument("--output", default="submission.csv", type=Path, help="Output CSV path.")
    args = parser.parse_args()

    metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    test = _read_table(args.test)
    predictions = _load_model_predictions(ROOT, metadata, test)
    combined = _combine(predictions, metadata)
    output = _to_submission(combined, metadata)

    sample_name = metadata.get("sample_submission")
    sample = pd.read_csv(ROOT / sample_name) if sample_name else None
    frame = _build_submission(test, output, metadata, sample)

    if len(frame) != len(test):
        raise SystemExit(f"Submission has {len(frame)} rows, expected {len(test)}")
    if frame.isna().any().any():
        raise SystemExit("Submission contains missing values")
    if np.issubdtype(np.asarray(output).dtype, np.number) and not np.isfinite(np.asarray(output)).all():
        raise SystemExit("Submission contains NaN/inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
'''
