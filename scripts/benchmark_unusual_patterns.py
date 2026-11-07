"""Benchmark the three cheap unusual-pattern prototypes on real UCI data.

Run::

    uv run python scripts/benchmark_unusual_patterns.py

The data is ``sklearn.datasets.load_breast_cancer`` (UCI Breast Cancer
Wisconsin, 569 rows, 30 numeric features), not a synthetic generator.  The
report records only cross-validated deltas; it does not claim leaderboard
performance.  Gaussian rank is fitted inside each training fold (the source
notebook ranked train+test jointly, which kconv does not copy).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kconv.features.unusual import (
    GaussianRankTransformer,
    add_modulo_features,
    add_row_statistics,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports"


def _classifier(kind: str, seed: int):
    if kind == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=seed,
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=seed),
    )


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].to_numpy(dtype=float)


def _run_cv(frame: pd.DataFrame, y: np.ndarray, variant: str, kind: str, folds: int, seed: int) -> dict:
    numeric = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    started = time.perf_counter()
    for train_idx, valid_idx in splitter.split(frame, y):
        train = frame.iloc[train_idx].reset_index(drop=True)
        valid = frame.iloc[valid_idx].reset_index(drop=True)
        if variant == "baseline":
            x_train, x_valid = _matrix(train, numeric), _matrix(valid, numeric)
        elif variant == "gaussian_rank":
            transformer = GaussianRankTransformer(numeric).fit(train)
            transformed_train = transformer.transform(train)
            transformed_valid = transformer.transform(valid)
            names = [f"{column}__gauss_rank" for column in numeric]
            x_train, x_valid = _matrix(transformed_train, names), _matrix(transformed_valid, names)
        elif variant == "modulo":
            enhanced_train = add_modulo_features(train, numeric, moduli=(2, 10, 100), steps=(2, 10))
            enhanced_valid = add_modulo_features(valid, numeric, moduli=(2, 10, 100), steps=(2, 10))
            names = [str(column) for column in enhanced_train.columns]
            x_train, x_valid = _matrix(enhanced_train, names), _matrix(enhanced_valid, names)
        elif variant == "row_statistics":
            enhanced_train = add_row_statistics(train, numeric, stats=("mean", "std", "min", "max"))
            enhanced_valid = add_row_statistics(valid, numeric, stats=("mean", "std", "min", "max"))
            names = [str(column) for column in enhanced_train.columns]
            x_train, x_valid = _matrix(enhanced_train, names), _matrix(enhanced_valid, names)
        else:
            raise ValueError(variant)
        model = _classifier(kind, seed)
        model.fit(x_train, y[train_idx])
        proba = model.predict_proba(x_valid)[:, 1]
        scores.append(float(roc_auc_score(y[valid_idx], proba)))
    return {
        "variant": variant,
        "model": kind,
        "auc_mean": float(np.mean(scores)),
        "auc_std": float(np.std(scores, ddof=1)),
        "auc_per_fold": scores,
        "seconds": round(time.perf_counter() - started, 3),
    }


def run(output_dir: Path, *, folds: int = 5, seed: int = 42, with_details: bool = False) -> dict:
    data = load_breast_cancer(as_frame=True)
    frame = data.frame.drop(columns=["target"]).copy()
    y = data.target.to_numpy(dtype=int)
    variants = ("baseline", "gaussian_rank", "modulo", "row_statistics")
    results: dict[str, dict] = {}
    baseline: dict[str, dict] = {}
    for kind in ("histgb", "logreg"):
        for variant in variants:
            result = _run_cv(frame, y, variant, kind, folds, seed)
            results[f"{kind}:{variant}"] = result
            if variant == "baseline":
                baseline[kind] = result
            else:
                result["delta_auc_vs_baseline"] = round(
                    result["auc_mean"] - baseline[kind]["auc_mean"], 6
                )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": "UCI Breast Cancer Wisconsin (Diagnostic)",
            "source": "sklearn.datasets.load_breast_cancer",
            "rows": int(len(frame)),
            "numeric_features": int(frame.shape[1]),
            "task": "binary",
        },
        "protocol": {
            "folds": folds,
            "seed": seed,
            "metric": "roc_auc",
            "notes": [
                "Gaussian rank reference is fitted on each training fold only.",
                "Modulo and row statistics are label-free and fitted inside each fold.",
                "No hyperparameter search and no leaderboard claim.",
            ],
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "results": results,
    }
    if with_details:
        payload["per_fold_scores"] = {
            key: value["auc_per_fold"] for key, value in results.items()
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "unusual_patterns_benchmark.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = _markdown(payload)
    md_path = output_dir / "unusual_patterns_benchmark.md"
    md_path.write_text(markdown, encoding="utf-8", newline="\n")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return payload


def _markdown(payload: dict) -> str:
    lines = [
        "# Cheap unusual-pattern benchmark",
        "",
        f"- Dataset: {payload['dataset']['name']} "
        f"({payload['dataset']['rows']} rows, {payload['dataset']['numeric_features']} numeric features)",
        f"- Protocol: {payload['protocol']['folds']}-fold stratified CV, seed {payload['protocol']['seed']}, "
        f"metric {payload['protocol']['metric']}",
        f"- Generated: {payload['generated_at']}",
        "",
        "| Model | Variant | ROC AUC mean | ROC AUC std | Delta vs baseline | Seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _key, result in payload["results"].items():
        delta = result.get("delta_auc_vs_baseline")
        delta_text = "baseline" if delta is None else f"{delta:+.5f}"
        lines.append(
            f"| {result['model']} | {result['variant']} | {result['auc_mean']:.6f} | "
            f"{result['auc_std']:.6f} | {delta_text} | {result['seconds']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Deltas are cross-validated mean differences on one small public dataset; they are evidence for a "
        "prototype, not a general claim.",
        "- Gaussian rank is monotone and therefore expected to be neutral for tree models; any difference is "
        "fold noise. It is more relevant for linear/distance-based models.",
        "- Modulo and row statistics are label-free; their risk is overfitting and interpretation, not target leakage.",
        "- No result here replaces a nested baseline-vs-experiment comparison on the target competition.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    run(args.output_dir, folds=args.folds, seed=args.seed, with_details=args.details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
