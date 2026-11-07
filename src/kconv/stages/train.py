"""Train all configured models with fold-level OOF and test predictions."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from kconv.atomic import (
    dump_joblib_atomic,
    write_csv_atomic,
    write_parquet_atomic,
)
from kconv.config import ModelConfig
from kconv.context import RunContext, read_json, write_json
from kconv.contracts import DatasetSpec
from kconv.cv.splitters import load_run_folds
from kconv.errors import ModelError, StageError
from kconv.feature_cache import load_or_build_features
from kconv.labels import native_labels
from kconv.metrics import compute_metric
from kconv.models import create_model
from kconv.models.validation import validate_model_params
from kconv.oof import validation_mask
from kconv.timeouts import fit_with_timeout, reduced_params

STAGE = "02_train"


def _spec(ctx: RunContext) -> DatasetSpec:
    if ctx.dataset is not None:
        return ctx.dataset
    raw = read_json(ctx.run_dir / "00_ingest" / "schema.json", default=None)
    if not isinstance(raw, dict):
        raise StageError("Missing ingest schema.", "Run ingest first.")
    return DatasetSpec.from_dict(raw)


def _predict(
    adapter,
    X: pd.DataFrame,
    task: str,
    n_classes: int | None = None,
) -> np.ndarray:
    if task == "regression":
        values = np.asarray(adapter.predict(X), dtype=float).reshape(-1)
    else:
        proba = np.asarray(adapter.predict_proba(X), dtype=float)
        if not np.isfinite(proba).all():
            raise ModelError(
                f"Model '{adapter.name}' produced non-finite predictions.",
                "Check numerical stability, feature ranges and model parameters.",
            )
        if task == "binary":
            if proba.ndim == 2:
                if proba.shape[1] != 2:
                    raise ModelError(
                        f"Model '{adapter.name}' returned {proba.shape[1]} probability columns for binary task.",
                        "Return positive-class probabilities or a two-column matrix.",
                    )
                if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-3):
                    raise ModelError(
                        f"Model '{adapter.name}' returned binary probabilities that do not sum to 1.",
                        "Check the adapter's predict_proba output.",
                    )
                values = proba[:, 1]
            else:
                values = proba.reshape(-1)
            if not np.all((values >= 0.0) & (values <= 1.0)):
                raise ModelError(
                    f"Model '{adapter.name}' returned probabilities outside [0, 1].",
                    "Return calibrated probabilities, not logits or class labels.",
                )
        else:
            if proba.ndim != 2:
                raise ModelError(
                    f"Model '{adapter.name}' returned a {proba.ndim}-D multiclass probability array.",
                    "Return an (n_rows, n_classes) probability matrix.",
                )
            if n_classes is not None and proba.shape[1] != n_classes:
                raise ModelError(
                    f"Model '{adapter.name}' returned {proba.shape[1]} probability columns; "
                    f"the training target has {n_classes} classes.",
                    "Check class ordering and the adapter's predict_proba output.",
                )
            if not np.all((proba >= 0.0) & (proba <= 1.0)):
                raise ModelError(
                    f"Model '{adapter.name}' returned multiclass probabilities outside [0, 1].",
                    "Return probabilities, not logits or class labels.",
                )
            if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-3):
                raise ModelError(
                    f"Model '{adapter.name}' returned multiclass probabilities that do not sum to 1.",
                    "Check the adapter's predict_proba output.",
                )
            values = proba
    if values.shape[0] != len(X):
        raise ModelError(
            f"Model '{adapter.name}' returned {values.shape[0]} predictions for {len(X)} rows.",
            "Check the adapter predict/predict_proba contract.",
        )
    if not np.isfinite(values).all():
        raise ModelError(
            f"Model '{adapter.name}' produced non-finite predictions.",
            "Check numerical stability, feature ranges and model parameters.",
        )
    return values


def _constant_prediction(values: np.ndarray) -> bool:
    if values.ndim == 1:
        return float(np.ptp(values)) <= 1e-12
    return bool(np.all(np.ptp(values, axis=0) <= 1e-12))


def _write_prediction(path: Path, frame: pd.DataFrame, task: str) -> None:
    del task
    write_parquet_atomic(frame, path)


def _prediction_frame(pred: np.ndarray, task: str) -> pd.DataFrame:
    frame = pd.DataFrame({"row_id": np.arange(len(pred), dtype=np.int64)})
    if task == "multiclass":
        pred = np.asarray(pred)
        for index in range(pred.shape[1]):
            frame[f"pred_{index}"] = pred[:, index]
    else:
        frame["pred"] = np.asarray(pred, dtype=float)
    return frame


def _encode_target(y: np.ndarray, task: str) -> tuple[np.ndarray, list[Any] | None]:
    if task == "regression":
        return np.asarray(y, dtype=float), None
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(np.asarray(y))
    return encoded.astype(np.int64), native_labels(encoder.classes_)


def _params_differ(params: dict[str, Any], base: dict[str, Any]) -> bool:
    """Return whether HPO actually changed a recipe parameter."""
    for key, value in params.items():
        if key not in base or base[key] != value:
            return True
    return False


def run_train(ctx: RunContext) -> dict:
    spec = _spec(ctx)
    train = pd.read_parquet(ctx.run_dir / "00_ingest" / "train.parquet")
    test_path = ctx.run_dir / "00_ingest" / "test.parquet"
    test = pd.read_parquet(test_path) if test_path.exists() else None
    folds = load_run_folds(ctx.run_dir)
    valid_mask = validation_mask(folds, len(train))
    if spec.target not in train.columns:
        raise StageError(f"Target '{spec.target}' missing from train artifact.", "Rerun ingest.")
    y_raw = train[spec.target].to_numpy()
    y, classes = _encode_target(y_raw, spec.task)
    model_configs = [m for m in ctx.config.models if m.enabled]
    if not model_configs:
        raise ModelError("No enabled models.", "Add at least one model to the config.")
    runtime = ctx.config.runtime
    existing_names = {m.name for m in model_configs}
    for fallback_name in list(getattr(runtime, "fallback_models", []) or []):
        if fallback_name in existing_names:
            continue
        try:
            validate_model_params(fallback_name, {})
        except Exception as exc:
            ctx.warnings.append(f"Fallback model '{fallback_name}' is invalid and was ignored: {exc}")
            continue
        model_configs.append(ModelConfig(name=fallback_name))
        existing_names.add(fallback_name)
        ctx.warnings.append(f"Fallback model '{fallback_name}' enabled by runtime config.")
    tuning = read_json(ctx.run_dir / "01_hpo" / "tuning.json", default={}) or {}
    tuning_folds = tuning.get("folds", {}) if isinstance(tuning, dict) else {}
    artifacts: list[str] = []
    summary: dict[str, float] = {}
    failed_models: list[dict[str, Any]] = []
    # Reject typos/ranges before the model loop.  A bad config is not a
    # per-model runtime failure; failing fast is clearer than silently skipping.
    for model_config in model_configs:
        validate_model_params(model_config.name, model_config.params)

    stage_root = ctx.stage_dir(STAGE).resolve()
    for model_config in model_configs:
        started = time.perf_counter()
        name = model_config.name
        # Model names come from user config.  Reject path syntax before any
        # mkdir/rmtree: ``foo/../bar`` stays inside the stage directory but
        # would still let a typo overwrite another model's artifacts.
        unsafe_name = (
            not name
            or name in {".", ".."}
            or name != name.strip()
            or ":" in name
            or any(sep in name for sep in ("/", "\\", "\x00"))
            or Path(name).name != name
        )
        model_dir = stage_root / name
        if unsafe_name or model_dir.resolve().parent != stage_root:
            failed_models.append(
                {
                    "model": name,
                    "error": "Invalid model name: path separators are not allowed.",
                    "duration_sec": 0.0,
                }
            )
            ctx.warnings.append(
                f"Model '{name}' failed and was skipped: invalid model name."
            )
            continue
        # Remove stale artifacts from an earlier attempt/resume so blend can
        # never pick up an OOF file from a model that failed this time.
        shutil.rmtree(model_dir, ignore_errors=True)
        try:
            adapter = create_model(model_config.name)
            adapter.task = spec.task
            if spec.task not in adapter.task_types:
                raise ModelError(
                    f"Model '{model_config.name}' does not support task '{spec.task}'.",
                    f"Use one of: {', '.join(sorted(adapter.task_types))}.",
                )
            if test is None and spec.task != "regression":
                # Training can still produce OOF; submit will fail with a clear hint.
                test = None
            n_rows = len(train)
            model_dir.mkdir(parents=True, exist_ok=True)
            # Rows outside every validation fold have no out-of-fold prediction.
            # Keep them as NaN instead of zeroes so no downstream metric or blend
            # can silently treat "unvalidated" as "predicted zero".
            if spec.task == "multiclass":
                oof = np.full((n_rows, len(np.unique(y))), np.nan, dtype=float)
            else:
                oof = np.full((n_rows, 1), np.nan, dtype=float)
            test_sum: np.ndarray | None = None
            test_count = 0
            fold_scores: list[float] = []
            last_params = dict(model_config.params)
            tuned_fold_count = 0
            cache_hits = 0
            fit_timeout = float(getattr(runtime, "model_timeout_sec", 0) or 0)
            if getattr(runtime, "isolate_models", False) and fit_timeout <= 0:
                fit_timeout = 3600.0
            retries = int(getattr(runtime, "model_retries", 0) or 0)

            for fold in folds:
                tr_idx = fold.train_idx
                va_idx = fold.valid_idx
                X_tr = train.iloc[tr_idx].reset_index(drop=True)
                X_va = train.iloc[va_idx].reset_index(drop=True)
                builder, Xt, Xv, X_test, cache_hit = load_or_build_features(
                    ctx,
                    spec,
                    adapter.family,
                    fold.index,
                    X_tr,
                    y[tr_idx],
                    X_va,
                    test,
                    fold_train_idx=tr_idx,
                    fold_valid_idx=va_idx,
                )
                cache_hits += int(cache_hit)
                for warning in getattr(builder, "warnings", []):
                    message = f"{model_config.name}: {warning}"
                    if message not in ctx.warnings:
                        ctx.warnings.append(message)
                fold_dir = model_dir / "folds" / f"fold_{fold.index}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                try:
                    dump_joblib_atomic(builder, fold_dir / "features.joblib", compress=3)
                except Exception as exc:  # pragma: no cover - filesystem/plugin specific
                    raise ModelError(
                        f"Could not save feature builder for {model_config.name} fold {fold.index}: {exc}",
                        "Check disk space and write permissions.",
                    ) from exc
                fold_tuning = tuning_folds.get(str(fold.index), {}) if isinstance(tuning_folds, dict) else {}
                tuned = fold_tuning.get(model_config.name, {}) if isinstance(fold_tuning, dict) else {}
                params = dict(model_config.params)
                if isinstance(tuned, dict) and isinstance(tuned.get("params"), dict):
                    params.update(tuned["params"])
                    if _params_differ(params, model_config.params):
                        tuned_fold_count += 1
                last_params = params
                seed_preds: list[np.ndarray] = []
                for seed_offset in range(model_config.seeds):
                    seed = int(ctx.config.seed) + seed_offset
                    for attempt in range(retries + 1):
                        try:
                            adapter = fit_with_timeout(
                                adapter,
                                Xt,
                                y[tr_idx],
                                Xv,
                                y[va_idx],
                                params=params,
                                seed=seed,
                                early_stopping_rounds=model_config.early_stopping_rounds,
                                timeout_sec=fit_timeout,
                            )
                            break
                        except Exception as exc:
                            if attempt >= retries:
                                if isinstance(exc, TimeoutError):
                                    exc.model_status = "timed_out"  # type: ignore[attr-defined]
                                raise
                            ctx.warnings.append(
                                f"{model_config.name}: fit attempt {attempt + 1} failed ({exc}); "
                                "retrying with reduced parameters."
                            )
                            params = reduced_params(params)
                            last_params = dict(params)
                            adapter = create_model(model_config.name)
                            adapter.task = spec.task
                    try:
                        dump_joblib_atomic(adapter, fold_dir / f"seed_{seed}.joblib", compress=3)
                    except Exception as exc:  # pragma: no cover - filesystem/plugin specific
                        raise ModelError(
                            f"Could not save {model_config.name} fold {fold.index} seed {seed}: {exc}",
                            "Check disk space and write permissions.",
                        ) from exc
                    pred_va = _predict(
                        adapter,
                        Xv,
                        spec.task,
                        n_classes=len(classes) if classes is not None else None,
                    )
                    if _constant_prediction(pred_va):
                        message = (
                            f"{model_config.name}: validation predictions are constant; "
                            "the model may be underfit or the target/features may be degenerate."
                        )
                        if message not in ctx.warnings:
                            ctx.warnings.append(message)
                    seed_preds.append(pred_va)
                    if X_test is not None:
                        pred_test = _predict(
                            adapter,
                            X_test,
                            spec.task,
                            n_classes=len(classes) if classes is not None else None,
                        )
                        if test_sum is None:
                            test_sum = np.zeros_like(pred_test, dtype=float)
                        test_sum += pred_test
                        test_count += 1
                fold_pred = np.mean(np.stack(seed_preds, axis=0), axis=0)
                if spec.task == "multiclass":
                    if fold_pred.ndim != 2 or fold_pred.shape[1] != oof.shape[1]:
                        raise ModelError("Multiclass fold prediction has an unexpected shape.", "Check class count.")
                    oof[va_idx, :] = fold_pred
                else:
                    oof[va_idx, 0] = np.asarray(fold_pred, dtype=float)
                fold_scores.append(compute_metric(spec.metric, y[va_idx], fold_pred, spec.task))

            if spec.task == "multiclass":
                oof_final = oof
                validated_oof = oof_final[valid_mask]
            else:
                oof_final = oof[:, 0]
                validated_oof = oof_final[valid_mask]
            validated_rows = int(valid_mask.sum())
            if validated_rows == 0:
                raise StageError(
                    "No row received an out-of-fold prediction.",
                    "Rerun the validate stage; folds do not cover the training data.",
                )
            oof_score = compute_metric(spec.metric, y[valid_mask], validated_oof, spec.task)
            oof_path = model_dir / "oof.parquet"
            test_pred_path = model_dir / "test_pred.parquet"
            _write_prediction(oof_path, _prediction_frame(oof_final, spec.task), spec.task)
            if test is not None and test_sum is not None and test_count:
                test_pred = test_sum / test_count
                _write_prediction(test_pred_path, _prediction_frame(test_pred, spec.task), spec.task)
            else:
                test_pred = np.empty(0)
            try:
                importance = adapter.importance()
            except Exception as exc:
                importance = None
                message = f"{model_config.name}: feature importance unavailable: {exc}"
                if message not in ctx.warnings:
                    ctx.warnings.append(message)
            importance_path = model_dir / "importance.csv"
            if importance is not None and len(importance):
                importance_frame = importance.to_frame("importance")
                importance_frame.index.name = "feature"
                write_csv_atomic(importance_frame, importance_path, index=True)
            duration = time.perf_counter() - started
            metrics_payload = {
                "model": model_config.name,
                "task": spec.task,
                "metric": spec.metric,
                "fold_scores": [float(v) for v in fold_scores],
                "oof_score": float(oof_score),
                "oof_rows": int(n_rows),
                "validated_rows": validated_rows,
                "partial_oof": bool(validated_rows < n_rows),
                "oof_coverage": float(validated_rows / max(n_rows, 1)),
                "mean_fold_score": float(np.mean(fold_scores)) if fold_scores else None,
                "classes": classes,
                "seeds": model_config.seeds,
                "duration_sec": round(duration, 3),
                "params": last_params,
                "recipe_params": model_config.params,
                "hpo_applied": bool(tuned_fold_count),
                "hpo_tuned_folds": int(tuned_fold_count),
                "feature_cache_hits": int(cache_hits),
                "feature_cache_misses": int(len(folds) - cache_hits),
            }
            write_json(model_dir / "metrics.json", metrics_payload)
            for path in (oof_path, test_pred_path, importance_path, model_dir / "metrics.json"):
                if path.exists():
                    artifacts.append(str(path.relative_to(ctx.run_dir)).replace("\\", "/"))
            summary[model_config.name] = float(oof_score)
            for warning in getattr(adapter, "warnings", []):
                message = f"{model_config.name}: {warning}"
                if message not in ctx.warnings:
                    ctx.warnings.append(message)
        except Exception as exc:
            failure: dict[str, Any] = {
                "model": model_config.name,
                "error": f"{type(exc).__name__}: {exc}",
                "duration_sec": round(time.perf_counter() - started, 3),
            }
            status = getattr(exc, "model_status", None)
            if status:
                failure["status"] = status
            failed_models.append(failure)
            ctx.warnings.append(
                f"Model '{model_config.name}' failed and was skipped: {exc}"
            )
            shutil.rmtree(model_dir, ignore_errors=True)
            continue

    if not summary:
        ctx.failed_models = list(failed_models)
        details = "; ".join(f"{item['model']}: {item['error']}" for item in failed_models)
        raise ModelError(
            "All configured models failed: " + details,
            "Fix at least one model's params/runtime, then rerun with --resume.",
        )

    return {
        "status": "ok",
        "artifacts": artifacts,
        "metrics": summary,
        "models": list(summary),
        "failed_models": failed_models,
    }
