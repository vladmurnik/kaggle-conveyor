"""Measured TabPFN v2 vs GBDT benchmark for the 3.5 GB VRAM laptop profile.

Run from the repository root with the optional stack installed::

    uv sync --extra tabpfn
    uv run --with psutil python scripts/benchmark_tabpfn.py

The benchmark uses two real UCI datasets bundled with scikit-learn:

* Breast Cancer Wisconsin (binary, 569 x 30, ROC AUC);
* Diabetes (regression, 442 x 10, RMSE).

It also runs one synthetic stress case (10,000 train rows x 100 features) to
measure the actual CUDA footprint.  All models use the same folds and raw
numeric features; no kconv feature engineering is applied, so this is a model
comparison, not a leaderboard claim.  The generated report is committed as
evidence and checked by ``tests/test_tabpfn_benchmark.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from kconv.metrics import compute_metric
from kconv.models import create_model, model_available

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "tabpfn_benchmark.json"
MODELS = ("tabpfn", "lgbm", "xgb", "catboost", "histgb")


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def _gpu_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "torch": torch.__version__}
        free, total = torch.cuda.mem_get_info()
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "name": props.name,
            "total_mib": round(total / 1024**2, 1),
            "free_mib": round(free / 1024**2, 1),
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _tabpfn_version() -> str | None:
    try:
        import tabpfn

        return str(getattr(tabpfn, "__version__", "unknown"))
    except Exception:
        return None


def _datasets() -> list[dict[str, Any]]:
    cancer = load_breast_cancer(as_frame=True)
    diabetes = load_diabetes(as_frame=True)
    return [
        {
            "id": "uci_breast_cancer",
            "name": "UCI Breast Cancer Wisconsin (Diagnostic)",
            "task": "binary",
            "metric": "roc_auc",
            "X": cancer.frame.drop(columns=["target"]).reset_index(drop=True),
            "y": cancer.target.to_numpy(),
            "splitter": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        },
        {
            "id": "uci_diabetes",
            "name": "UCI Diabetes",
            "task": "regression",
            "metric": "rmse",
            "X": diabetes.frame.drop(columns=["target"]).reset_index(drop=True),
            "y": diabetes.target.to_numpy(),
            "splitter": KFold(n_splits=5, shuffle=True, random_state=42),
        },
    ]


def _fit_and_predict(
    name: str,
    task: str,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    device: str,
) -> np.ndarray:
    adapter = create_model(name)
    adapter.task = task
    if name == "tabpfn":
        adapter.fit(
            X_tr,
            y_tr,
            params={"device": device, "n_estimators": 8, "max_train_rows": 8_000},
            seed=42,
        )
    else:
        adapter.fit(X_tr, y_tr, seed=42)
    if task == "regression":
        return np.asarray(adapter.predict(X_va), dtype=float)
    return np.asarray(adapter.predict_proba(X_va), dtype=float)


def _benchmark_dataset(dataset: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    task = dataset["task"]
    metric = dataset["metric"]
    X = dataset["X"]
    y = dataset["y"]
    rows: list[dict[str, Any]] = []
    for name in MODELS:
        if name == "tabpfn" and not model_available(name):
            continue
        fold_scores: list[float] = []
        fold_seconds: list[float] = []
        peak_reserved = 0.0
        peak_allocated = 0.0
        peak_rss = 0.0
        started_model = time.perf_counter()
        for train_idx, valid_idx in dataset["splitter"].split(X, y if task == "binary" else None):
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            started_fold = time.perf_counter()
            prediction = _fit_and_predict(
                name,
                task,
                X.iloc[train_idx].reset_index(drop=True),
                y[train_idx],
                X.iloc[valid_idx].reset_index(drop=True),
                device,
            )
            fold_seconds.append(time.perf_counter() - started_fold)
            fold_scores.append(float(compute_metric(metric, y[valid_idx], prediction, task)))
            if device.startswith("cuda"):
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / 1024**3)
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated() / 1024**3)
            rss = _rss_mb()
            if rss is not None:
                peak_rss = max(peak_rss, rss)
        direction = "maximize" if metric in {"roc_auc", "accuracy", "f1", "r2"} else "minimize"
        rows.append(
            {
                "model": name,
                "mean_score": float(np.mean(fold_scores)),
                "std_score": float(np.std(fold_scores)),
                "fold_scores": fold_scores,
                "direction": direction,
                "total_sec": round(time.perf_counter() - started_model, 3),
                "mean_fold_sec": round(float(np.mean(fold_seconds)), 3),
                "peak_vram_reserved_gb": round(peak_reserved, 3),
                "peak_vram_allocated_gb": round(peak_allocated, 3),
                "peak_rss_mb": round(peak_rss, 1) if peak_rss else None,
            }
        )
        gc.collect()

    best_row = rows[0] if rows else None
    for row in rows[1:]:
        if best_row is None:
            best_row = row
            continue
        better = (
            row["mean_score"] > best_row["mean_score"]
            if row["direction"] == "maximize"
            else row["mean_score"] < best_row["mean_score"]
        )
        if better:
            best_row = row
    best = best_row["model"] if best_row else None
    return {
        "id": dataset["id"],
        "name": dataset["name"],
        "task": task,
        "metric": metric,
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "models": rows,
        "best_by_mean": best,
    }


def _stress(device: str, rows: int = 10_000, features: int = 100) -> dict[str, Any]:
    import torch

    X, y = make_classification(
        n_samples=rows,
        n_features=features,
        n_informative=25,
        n_redundant=10,
        random_state=42,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    columns = [f"x{i}" for i in range(features)]
    X_tr_df = pd.DataFrame(X_tr, columns=columns)
    X_te_df = pd.DataFrame(X_te, columns=columns)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    baseline_rss = _rss_mb()
    peak_rss = baseline_rss or 0.0
    stop = threading.Event()

    def sample_rss() -> None:
        nonlocal peak_rss
        while not stop.is_set():
            value = _rss_mb()
            if value is not None:
                peak_rss = max(peak_rss, value)
            stop.wait(0.1)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    started = time.perf_counter()
    adapter = create_model("tabpfn")
    adapter.task = "binary"
    adapter.fit(
        X_tr_df,
        y_tr,
        params={"device": device, "n_estimators": 8, "max_train_rows": 8_000},
        seed=42,
    )
    prediction = adapter.predict_proba(X_te_df)
    elapsed = time.perf_counter() - started
    stop.set()
    sampler.join(timeout=1.0)
    result = {
        "id": f"synthetic_{len(X_tr_df)}x{features}",
        "train_rows": int(len(X_tr_df)),
        "test_rows": int(len(X_te_df)),
        "features": int(features),
        "auc": float(compute_metric("roc_auc", y_te, prediction, "binary")),
        "seconds": round(elapsed, 3),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        if device.startswith("cuda")
        else 0.0,
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        if device.startswith("cuda")
        else 0.0,
        "adapter_peak_vram_allocated_gb": round(adapter.peak_vram_bytes_ / 1024**3, 3),
        "baseline_rss_mb": round(baseline_rss, 1) if baseline_rss else None,
        "peak_rss_mb": round(peak_rss, 1) if peak_rss else None,
    }
    del adapter
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# TabPFN v2 benchmark",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "Protocol: raw numeric features, same 5-fold split for every model, "
        "TabPFN v2 with `n_estimators=8`, `fit_mode=fit_preprocessors`, "
        "checkpoint cache warm. This is a model comparison, not a leaderboard claim.",
        "",
        "## Environment",
        "",
        f"- Python: {payload['environment']['python']}",
        f"- Platform: {payload['environment']['platform']}",
        f"- TabPFN: {payload['environment']['tabpfn']}",
        f"- Torch: {payload['environment']['gpu'].get('torch')}",
        f"- GPU: {payload['environment']['gpu'].get('name', 'CPU')} "
        f"({payload['environment']['gpu'].get('total_mib', 'n/a')} MiB total, "
        f"{payload['environment']['gpu'].get('free_mib', 'n/a')} MiB free at start)",
        "",
    ]
    for item in payload["benchmarks"]:
        lines += [
            f"## {item['name']}",
            "",
            f"`{item['id']}` · {item['rows']} rows × {item['features']} features · "
            f"{item['task']} · `{item['metric']}`",
            "",
            "| model | mean ± std | total s | mean fold s | peak reserved VRAM | peak RSS |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in item["models"]:
            lines.append(
                f"| `{row['model']}` | {row['mean_score']:.6f} ± {row['std_score']:.6f} | "
                f"{row['total_sec']:.2f} | {row['mean_fold_sec']:.2f} | "
                f"{row['peak_vram_reserved_gb']:.3f} GB | "
                f"{row['peak_rss_mb'] if row['peak_rss_mb'] is not None else 'n/a'} MB |"
            )
        lines += ["", f"Best mean {item['metric']}: **{item['best_by_mean']}**.", ""]
    lines += [
        "## 3.5 GB VRAM stress test",
        "",
        "| case | train × features | AUC | seconds | peak reserved | peak allocated | peak RSS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["stress"]:
        lines.append(
            f"| `{row['id']}` | {row['train_rows']} × {row['features']} | {row['auc']:.4f} | "
            f"{row['seconds']:.2f} | {row['peak_vram_reserved_gb']:.3f} GB | "
            f"{row['peak_vram_allocated_gb']:.3f} GB | "
            f"{row['peak_rss_mb'] if row['peak_rss_mb'] is not None else 'n/a'} MB |"
        )
    lines += [
        "",
        "Limits: TabPFN v2 supports 10,000 train rows, 500 features and 10 classes. "
        "The adapter's 3.5 GB-safe default is 8,000 train rows; for larger tables "
        "use LightGBM/XGBoost/CatBoost, or explicitly raise `params.max_train_rows`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-stress", action="store_true")
    args = parser.parse_args()
    if not model_available("tabpfn"):
        print("TabPFN is not installed. Run: uv sync --extra tabpfn")
        return 2

    import torch

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tabpfn": _tabpfn_version(),
            "gpu": _gpu_info(),
        },
        "protocol": {
            "device": device,
            "folds": 5,
            "tabpfn_n_estimators": 8,
            "tabpfn_fit_mode": "fit_preprocessors",
            "feature_engineering": "none (raw numeric sklearn datasets)",
            "dataset_source": "sklearn.datasets UCI built-ins",
        },
        "benchmarks": [],
        "stress": [],
    }
    for dataset in _datasets():
        payload["benchmarks"].append(_benchmark_dataset(dataset, device))
    if not args.skip_stress:
        payload["stress"].append(_stress(device))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(f"Wrote {args.output} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
