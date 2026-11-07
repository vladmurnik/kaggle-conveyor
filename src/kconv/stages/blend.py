"""Blend model predictions on OOF and save the selected ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from kconv.atomic import (
    dump_joblib_atomic,
    verify_artifacts,
    write_parquet_atomic,
)
from kconv.context import RunContext, read_json, write_json
from kconv.contracts import DatasetSpec
from kconv.cv.splitters import load_run_folds
from kconv.errors import StageError
from kconv.labels import native_labels
from kconv.metrics import compute_metric, get_metric
from kconv.oof import validation_mask

STAGE = "03_blend"


def _spec(ctx: RunContext) -> DatasetSpec:
    raw = read_json(ctx.run_dir / "00_ingest" / "schema.json", default=None)
    if not isinstance(raw, dict):
        raise StageError("Missing ingest schema.", "Run ingest first.")
    return DatasetSpec.from_dict(raw)


def _load_prediction(path) -> tuple[np.ndarray, bool]:
    frame = pd.read_parquet(path)
    if "pred" in frame.columns:
        return frame["pred"].to_numpy(dtype=float), False
    columns = sorted((c for c in frame.columns if c.startswith("pred_")), key=lambda c: int(c.split("_")[-1]))
    if not columns:
        raise StageError(f"{path} has no prediction column.", "Rerun train.")
    return frame[columns].to_numpy(dtype=float), True


def _write_prediction(path, pred: np.ndarray, multiclass: bool) -> None:
    frame = pd.DataFrame({"row_id": np.arange(len(pred), dtype=np.int64)})
    if multiclass:
        for index in range(pred.shape[1]):
            frame[f"pred_{index}"] = pred[:, index]
    else:
        frame["pred"] = pred
    write_parquet_atomic(frame, path)


def _score(metric: str, y: np.ndarray, pred: np.ndarray, task: str) -> float:
    return compute_metric(metric, y, pred, task)


def _improves(
    metric_name: str,
    candidate: float,
    baseline: float,
    min_improvement: float,
) -> bool:
    """Return whether ``candidate`` beats ``baseline`` by the requested margin."""
    direction = get_metric(metric_name).direction
    margin = max(0.0, float(min_improvement))
    if direction == "maximize":
        return candidate > baseline + margin
    return candidate < baseline - margin


def _normalize_probability_matrix(values: np.ndarray) -> np.ndarray:
    """Keep multiclass blends valid after floating-point accumulation."""
    matrix = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(row_sums, 1e-12)


def _combine(weights: np.ndarray, predictions: list[np.ndarray], multiclass: bool) -> np.ndarray:
    if multiclass:
        combined = np.tensordot(weights, np.stack(predictions, axis=0), axes=(0, 0))
        return _normalize_probability_matrix(combined)
    return np.average(np.stack(predictions, axis=0), axis=0, weights=weights)


def _fill_masked(values: np.ndarray, mask: np.ndarray, n_rows: int, multiclass: bool) -> np.ndarray:
    """Put validated predictions back into a full-length OOF array."""
    values = np.asarray(values)
    if multiclass:
        full = np.full((n_rows, values.shape[1]), np.nan, dtype=float)
    else:
        full = np.full(n_rows, np.nan, dtype=float)
    full[np.asarray(mask, dtype=bool)] = values
    return full


def _nested_weight_score(
    predictions: list[np.ndarray],
    y: np.ndarray,
    folds,
    mask: np.ndarray,
    task: str,
    metric: str,
    multiclass: bool,
) -> float | None:
    """Evaluate blend-weight fitting with the outer validation folds.

    Weights are fitted only on the rows that are training rows for a fold and
    scored on that fold's validation rows.  This removes the optimistic bias of
    fitting weights and reporting the score on the same OOF rows.  If the folds
    cannot cover every validated row (holdout, early time-series folds), return
    ``None`` and let the caller fall back to the in-sample score explicitly.
    """
    valid_positions = np.flatnonzero(mask)
    if len(valid_positions) == 0:
        return None
    position = {int(original): index for index, original in enumerate(valid_positions)}
    if multiclass:
        nested = np.full((len(valid_positions), np.asarray(predictions[0]).shape[1]), np.nan, dtype=float)
    else:
        nested = np.full(len(valid_positions), np.nan, dtype=float)
    covered = np.zeros(len(valid_positions), dtype=bool)
    for fold in folds:
        train_idx = np.asarray(fold.train_idx, dtype=np.int64)
        valid_idx = np.asarray(fold.valid_idx, dtype=np.int64)
        train_idx = train_idx[mask[train_idx]]
        valid_idx = valid_idx[mask[valid_idx]]
        if len(train_idx) < max(2, len(predictions)) or len(valid_idx) == 0:
            continue
        try:
            weights = _optimize_weights(
                [np.asarray(pred)[train_idx] for pred in predictions],
                y[train_idx],
                task,
                metric,
                multiclass,
            )
            prediction = _combine(
                weights,
                [np.asarray(pred)[valid_idx] for pred in predictions],
                multiclass,
            )
        except (ValueError, FloatingPointError):
            continue
        rows = np.asarray([position[int(index)] for index in valid_idx], dtype=np.int64)
        nested[rows] = prediction
        covered[rows] = True
    if not covered.all():
        return None
    return _score(metric, y[mask], nested, task)


def _optimize_weights(predictions: list[np.ndarray], y: np.ndarray, task: str, metric: str, multiclass: bool) -> np.ndarray:
    n_models = len(predictions)
    initial = np.full(n_models, 1.0 / n_models)
    direction = get_metric(metric).direction

    def objective(weights: np.ndarray) -> float:
        score = _score(metric, y, _combine(weights, predictions, multiclass), task)
        return -score if direction == "maximize" else score

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        options={"maxiter": 200, "ftol": 1e-9},
    )
    if not result.success or not np.isfinite(result.fun):
        return initial
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    total = weights.sum()
    return weights / total if total > 0 else initial


def _rank_mean(predictions: list[np.ndarray], multiclass: bool) -> np.ndarray:
    if multiclass:
        # Rank each class independently, then average. Normalize across
        # classes per row so the result remains a valid probability matrix for
        # log_loss and for submission stages that consume probabilities.
        ranks = [np.apply_along_axis(rankdata, 0, p) for p in predictions]
        return _normalize_probability_matrix(np.mean(ranks, axis=0))
    ranks = np.stack([rankdata(p) for p in predictions], axis=0)
    combined = ranks.mean(axis=0)
    span = combined.max() - combined.min()
    return (combined - combined.min()) / span if span > 1e-12 else combined


def _stack_matrix(arrays: list[np.ndarray], multiclass: bool) -> np.ndarray:
    if multiclass:
        return np.concatenate(arrays, axis=1)
    return np.column_stack(arrays)


def _meta_estimator(task: str, meta_model: str, seed: int):
    if meta_model == "lgbm":
        params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 20,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_jobs": 1,
            "verbosity": -1,
        }
        if task == "regression":
            return LGBMRegressor(**params), "lgbm"
        return LGBMClassifier(**params), "lgbm"
    if task == "regression":
        # Ridge is the safe default for continuous meta-targets. A request for
        # logreg is resolved to ridge with the resolved name reported back.
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "ridge"
    # RidgeClassifier has no calibrated probabilities; keep the stacking
    # contract probability-based and resolve to logistic regression.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), "logreg"


def _meta_predict(estimator, X: np.ndarray, task: str) -> np.ndarray:
    if task == "regression":
        return np.asarray(estimator.predict(X), dtype=float)
    proba = np.asarray(estimator.predict_proba(X), dtype=float)
    if task == "binary":
        values = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.reshape(-1)
        return np.clip(values, 0.0, 1.0)
    return _normalize_probability_matrix(proba)


def _stack_predictions(
    oofs: list[np.ndarray],
    tests: list[np.ndarray],
    y: np.ndarray,
    task: str,
    meta_model: str,
    meta_splits: int,
    seed: int,
    *,
    meta_model_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a meta-model on OOF predictions.

    The meta-model is validated with its own cross-validation on the OOF rows,
    so the reported stack score is not the resubstitution score. The final
    meta-model is then fitted on all OOF rows before predicting the test set.
    """
    multiclass = task == "multiclass"
    X_meta = _stack_matrix(oofs, multiclass)
    X_test = _stack_matrix(tests, multiclass)
    n_rows = len(y)
    if n_rows < 8:
        raise StageError(
            f"Stacking needs at least 8 OOF rows, got {n_rows}.",
            "Use method 'weighted' or collect more training rows.",
        )
    n_classes = int(np.max(y)) + 1 if task != "regression" else 0
    if task == "regression":
        n_splits = min(meta_splits, n_rows)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        meta_oof = np.zeros(n_rows, dtype=float)
    else:
        counts = np.bincount(y, minlength=n_classes)
        smallest = int(counts.min()) if len(counts) else 0
        n_splits = min(meta_splits, smallest)
        if n_splits < 2:
            raise StageError(
                "Stacking needs at least two rows per class in OOF predictions.",
                "Use method 'weighted' or a larger training set.",
            )
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        meta_oof = np.zeros((n_rows, n_classes), dtype=float) if multiclass else np.zeros(n_rows, dtype=float)

    for train_index, valid_index in splitter.split(X_meta, y if task != "regression" else None):
        if task != "regression" and len(np.unique(y[train_index])) < n_classes:
            raise StageError(
                "A stacking validation fold is missing a class.",
                "Use method 'weighted' or reduce ensemble.meta_splits.",
            )
        estimator, _ = _meta_estimator(task, meta_model, seed)
        estimator.fit(X_meta[train_index], y[train_index])
        prediction = _meta_predict(estimator, X_meta[valid_index], task)
        if multiclass:
            if prediction.shape[1] != n_classes:
                raise StageError(
                    "The stacking meta-model returned an unexpected number of classes.",
                    "Use method 'weighted' instead.",
                )
            meta_oof[valid_index] = prediction
        else:
            meta_oof[valid_index] = prediction
    if not np.isfinite(meta_oof).all():
        raise StageError("Stacking produced non-finite OOF predictions.", "Check model predictions.")

    estimator, resolved_model = _meta_estimator(task, meta_model, seed)
    estimator.fit(X_meta, y)
    if meta_model_path is not None:
        meta_model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            dump_joblib_atomic(estimator, meta_model_path, compress=3)
        except Exception as exc:  # pragma: no cover - filesystem/plugin specific
            raise StageError(
                f"Could not save the stacking meta-model: {exc}",
                "Check disk space and write permissions.",
            ) from exc
    test_pred = _meta_predict(estimator, X_test, task)
    if multiclass and test_pred.shape[1] != n_classes:
        raise StageError("The stacking meta-model returned an unexpected test shape.", "Use method 'weighted'.")
    if not np.isfinite(test_pred).all():
        raise StageError("Stacking produced non-finite test predictions.", "Check model predictions.")

    info = {
        "meta_model": resolved_model,
        "meta_splits": int(n_splits),
        "meta_train_rows": int(n_rows),
        "meta_features": int(X_meta.shape[1]),
    }
    return meta_oof, test_pred, info


def run_blend(ctx: RunContext) -> dict:
    spec = _spec(ctx)
    train = pd.read_parquet(ctx.run_dir / "00_ingest" / "train.parquet")
    if spec.target not in train.columns:
        raise StageError("Target missing from train artifact.", "Rerun ingest.")
    if spec.task == "regression":
        y = train[spec.target].to_numpy(dtype=float)
        classes = None
    else:
        encoder = LabelEncoder()
        y = encoder.fit_transform(train[spec.target].to_numpy())
        classes = native_labels(encoder.classes_)

    model_dirs = [p for p in sorted((ctx.run_dir / "02_train").glob("*/oof.parquet")) if p.is_file()]
    if not model_dirs:
        raise StageError("No model OOF artifacts found.", "Run train before blend.")
    names: list[str] = []
    oofs: list[np.ndarray] = []
    tests: list[np.ndarray] = []
    multiclass = spec.task == "multiclass"
    n_rows = len(train)
    for oof_path in model_dirs:
        name = oof_path.parent.name
        test_path = oof_path.parent / "test_pred.parquet"
        integrity_errors = verify_artifacts(ctx.run_dir, [oof_path, test_path])
        if integrity_errors:
            raise StageError(
                f"{name} artifact integrity check failed: {integrity_errors[0]}",
                "Rerun train to regenerate corrupted artifacts.",
            )
        oof, is_multi = _load_prediction(oof_path)
        if len(oof) != n_rows:
            raise StageError(f"{name} OOF has {len(oof)} rows, expected {n_rows}.", "Rerun train.")
        if is_multi != multiclass:
            raise StageError(f"{name} prediction shape does not match task '{spec.task}'.", "Rerun train.")
        if not test_path.exists():
            raise StageError(f"{name} is missing test predictions.", "Provide a test file and rerun train.")
        test_pred, test_is_multi = _load_prediction(test_path)
        if test_is_multi != multiclass:
            raise StageError(f"{name} test prediction shape is inconsistent.", "Rerun train.")
        names.append(name)
        oofs.append(oof)
        tests.append(test_pred)

    folds = load_run_folds(ctx.run_dir)
    valid_mask = validation_mask(folds, n_rows)
    if not valid_mask.any():
        raise StageError(
            "No row received an out-of-fold prediction.",
            "Rerun the validate stage; folds do not cover the training data.",
        )
    scores = [_score(spec.metric, y[valid_mask], np.asarray(pred)[valid_mask], spec.task) for pred in oofs]
    best_index = int(np.argmin(scores) if get_metric(spec.metric).direction == "minimize" else np.argmax(scores))
    best_name = names[best_index]
    best_score = scores[best_index]
    method = ctx.config.ensemble.method
    if len(names) == 1:
        method = "best_single"
    elif method == "auto":
        method = "weighted"
    if method == "rank_mean" and spec.metric != "roc_auc":
        raise StageError(
            "rank_mean produces ranks, not calibrated probabilities; it is only valid for roc_auc.",
            f"Use method 'weighted'/'mean' or change the metric (current: {spec.metric}).",
        )

    out_dir = ctx.stage_dir(STAGE)
    meta_path = out_dir / "meta_model.joblib"
    weights = np.zeros(len(names), dtype=float)
    stack_info: dict | None = None
    score_in_sample: float | None = None
    score_cv: float | None = None
    score_source: str | None = None
    fallback = False
    fallback_reason: str | None = None
    if method == "best_single":
        weights[best_index] = 1.0
        oof = _fill_masked(np.asarray(oofs[best_index])[valid_mask], valid_mask, n_rows, multiclass)
        test = tests[best_index]
        score_in_sample = _score(spec.metric, y[valid_mask], oof[valid_mask], spec.task)
        score_cv = score_in_sample
        score_source = "in_sample_no_fit"
    elif method == "mean":
        weights[:] = 1.0 / len(names)
        oof_valid = _combine(weights, [np.asarray(pred)[valid_mask] for pred in oofs], multiclass)
        oof = _fill_masked(oof_valid, valid_mask, n_rows, multiclass)
        test = _combine(weights, tests, multiclass)
        score_in_sample = _score(spec.metric, y[valid_mask], oof_valid, spec.task)
        score_cv = score_in_sample
        score_source = "in_sample_no_fit"
    elif method == "rank_mean":
        weights[:] = 1.0 / len(names)
        oof_valid = _rank_mean([np.asarray(pred)[valid_mask] for pred in oofs], multiclass)
        oof = _fill_masked(oof_valid, valid_mask, n_rows, multiclass)
        test = _rank_mean(tests, multiclass)
        score_in_sample = _score(spec.metric, y[valid_mask], oof_valid, spec.task)
        score_cv = score_in_sample
        score_source = "in_sample_no_fit"
    elif method == "weighted":
        weights = _optimize_weights(
            [np.asarray(pred)[valid_mask] for pred in oofs],
            y[valid_mask],
            spec.task,
            spec.metric,
            multiclass,
        )
        oof_valid = _combine(weights, [np.asarray(pred)[valid_mask] for pred in oofs], multiclass)
        oof = _fill_masked(oof_valid, valid_mask, n_rows, multiclass)
        test = _combine(weights, tests, multiclass)
        score_in_sample = _score(spec.metric, y[valid_mask], oof_valid, spec.task)
        score_cv = _nested_weight_score(
            oofs,
            y,
            folds,
            valid_mask,
            spec.task,
            spec.metric,
            multiclass,
        )
        score_source = "nested" if score_cv is not None else None
    elif method == "stack":
        oof_valid, test, stack_info = _stack_predictions(
            [np.asarray(pred)[valid_mask] for pred in oofs],
            tests,
            y[valid_mask],
            spec.task,
            ctx.config.ensemble.meta_model,
            ctx.config.ensemble.meta_splits,
            ctx.config.seed,
            meta_model_path=meta_path,
        )
        oof = _fill_masked(oof_valid, valid_mask, n_rows, multiclass)
        score_cv = _score(spec.metric, y[valid_mask], oof_valid, spec.task)
        score_in_sample = None
        score_source = "nested"
    else:
        raise StageError(
            f"Unknown ensemble method '{method}'.",
            "Use auto|best_single|mean|rank_mean|weighted|stack.",
        )

    # Weighted blending fits weights on the same rows it scores.  If the folds
    # cannot provide a nested score (holdout or early time-series), do not use
    # that in-sample score for model selection: fall back to the best single
    # model instead.
    if method == "weighted" and score_cv is None:
        fallback = True
        fallback_reason = "no_nested_score"
        method = "best_single"
        weights = np.zeros(len(names), dtype=float)
        weights[best_index] = 1.0
        oof = _fill_masked(np.asarray(oofs[best_index])[valid_mask], valid_mask, n_rows, multiclass)
        test = tests[best_index]
        score_in_sample = best_score
        score_cv = best_score
        score_source = "in_sample_no_fit"

    blend_score = score_cv if score_cv is not None else score_in_sample
    if blend_score is None:
        blend_score = _score(spec.metric, y[valid_mask], oof[valid_mask], spec.task)
        score_source = "in_sample_no_fit"
    if method != "best_single" and not _improves(
        spec.metric, blend_score, best_score, ctx.config.ensemble.min_improvement
    ):
        # Fallback when the blend does not beat the best single model by the
        # configured margin.  Maximized and minimized metrics need opposite
        # signs for that margin.
        fallback = True
        fallback_reason = "blend_not_better"
        method = ctx.config.ensemble.fallback
        if method == "rank_mean" and spec.metric != "roc_auc":
            raise StageError(
                "ensemble.fallback=rank_mean is only valid for roc_auc.",
                f"Change ensemble.fallback or use roc_auc (current: {spec.metric}).",
            )
        if method == "best_single":
            weights = np.zeros(len(names), dtype=float)
            weights[best_index] = 1.0
            oof = _fill_masked(np.asarray(oofs[best_index])[valid_mask], valid_mask, n_rows, multiclass)
            test = tests[best_index]
            score_in_sample = _score(spec.metric, y[valid_mask], oof[valid_mask], spec.task)
            score_cv = score_in_sample
            score_source = "in_sample_no_fit"
        elif method == "rank_mean":
            weights[:] = 1.0 / len(names)
            oof_valid = _rank_mean([np.asarray(pred)[valid_mask] for pred in oofs], multiclass)
            oof = _fill_masked(oof_valid, valid_mask, n_rows, multiclass)
            test = _rank_mean(tests, multiclass)
            score_in_sample = _score(spec.metric, y[valid_mask], oof_valid, spec.task)
            score_cv = score_in_sample
            score_source = "in_sample_no_fit"
        elif method == "weighted":
            weights = _optimize_weights(
                [np.asarray(pred)[valid_mask] for pred in oofs],
                y[valid_mask],
                spec.task,
                spec.metric,
                multiclass,
            )
            oof_valid = _combine(weights, [np.asarray(pred)[valid_mask] for pred in oofs], multiclass)
            oof = _fill_masked(oof_valid, valid_mask, n_rows, multiclass)
            test = _combine(weights, tests, multiclass)
            fallback_cv = _nested_weight_score(
                oofs,
                y,
                folds,
                valid_mask,
                spec.task,
                spec.metric,
                multiclass,
            )
            if fallback_cv is None:
                fallback_reason = "no_nested_score"
                method = "best_single"
                weights = np.zeros(len(names), dtype=float)
                weights[best_index] = 1.0
                oof = _fill_masked(np.asarray(oofs[best_index])[valid_mask], valid_mask, n_rows, multiclass)
                test = tests[best_index]
                score_in_sample = best_score
                score_cv = best_score
                score_source = "in_sample_no_fit"
            else:
                score_in_sample = _score(spec.metric, y[valid_mask], oof_valid, spec.task)
                score_cv = fallback_cv
                score_source = "nested"
        else:
            weights[:] = 1.0 / len(names)
            oof_valid = _combine(weights, [np.asarray(pred)[valid_mask] for pred in oofs], multiclass)
            oof = _fill_masked(oof_valid, valid_mask, n_rows, multiclass)
            test = _combine(weights, tests, multiclass)
            score_in_sample = _score(spec.metric, y[valid_mask], oof_valid, spec.task)
            score_cv = score_in_sample
            score_source = "in_sample_no_fit"
        blend_score = score_cv if score_cv is not None else score_in_sample
        if blend_score is None:
            blend_score = _score(spec.metric, y[valid_mask], oof[valid_mask], spec.task)
            score_source = "in_sample_no_fit"

    oof_path = out_dir / "oof.parquet"
    test_path = out_dir / "test_pred.parquet"
    _write_prediction(oof_path, oof, multiclass)
    _write_prediction(test_path, test, multiclass)
    if method == "stack" and stack_info is not None:
        weights_payload: dict[str, float] = {}
    else:
        weights_payload = {name: float(weight) for name, weight in zip(names, weights, strict=False)}
    payload = {
        "method": method,
        "models": names,
        "weights": weights_payload,
        "best_single": {"name": best_name, "score": float(best_score)},
        "oof_score": float(blend_score),
        "score_in_sample": None if score_in_sample is None else float(score_in_sample),
        "score_cv": None if score_cv is None else float(score_cv),
        "score_cv_available": score_cv is not None,
        "score_source": score_source,
        "fallback_reason": fallback_reason,
        "oof_rows": int(n_rows),
        "validated_rows": int(valid_mask.sum()),
        "partial_oof": bool(valid_mask.sum() < n_rows),
        "metric": spec.metric,
        "task": spec.task,
        "classes": classes,
        "fallback_used": fallback,
        "model_scores": {name: float(score) for name, score in zip(names, scores, strict=False)},
        "stack": stack_info,
    }
    write_json(out_dir / "blend.json", payload)
    artifacts = [
        str(oof_path.relative_to(ctx.run_dir)).replace("\\", "/"),
        str(test_path.relative_to(ctx.run_dir)).replace("\\", "/"),
        str((out_dir / "blend.json").relative_to(ctx.run_dir)).replace("\\", "/"),
    ]
    if meta_path.exists():
        artifacts.append(str(meta_path.relative_to(ctx.run_dir)).replace("\\", "/"))
    return {
        "status": "ok",
        "artifacts": artifacts,
        "method": method,
        "oof_score": float(blend_score),
        "fallback_used": fallback,
        "stack": stack_info,
    }
