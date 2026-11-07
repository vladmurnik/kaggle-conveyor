"""Nested, time-boxed hyperparameter search for enabled models.

The search is deliberately conservative:

* it only runs when ``hpo.enabled`` is true (the ``deep`` strategy turns it on);
* for every outer CV fold it creates a separate inner validation split and
  tunes only on that fold's training rows, so the outer validation rows are
  never used for model selection;
* the best parameters are stored per outer fold and consumed by the train
  stage.  There is no global "best params" leak into OOF evaluation;
* the wall-clock budget is checked between trials and the stage degrades to
  recipe defaults instead of failing the whole run.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

from kconv.config import ModelConfig
from kconv.context import RunContext, read_json, write_json
from kconv.contracts import DatasetSpec, Fold
from kconv.cv.splitters import load_run_folds
from kconv.errors import KconvError, ModelError, StageError
from kconv.features.pipeline import FeatureBuilder
from kconv.metrics import compute_metric, get_metric
from kconv.models import create_model
from kconv.timeouts import fit_with_timeout

STAGE = "01_hpo"

# Search spaces are intentionally small.  They cover the parameters that move
# tabular GBDT baselines most, without turning a normal run into AutoML.
SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "lgbm": {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [10, 20, 50],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
    },
    "xgb": {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 4, 6],
        "min_child_weight": [1, 3, 8],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
    },
    "catboost": {
        "iterations": [300, 500, 800],
        "learning_rate": [0.03, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1.0, 3.0, 10.0],
    },
    "histgb": {
        "max_iter": [200, 300, 500],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [5, 10, 20],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    "logreg": {
        "C": [0.01, 0.1, 1.0, 10.0],
    },
    "ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
    },
}


def _spec(ctx: RunContext) -> DatasetSpec:
    if ctx.dataset is not None:
        return ctx.dataset
    raw = read_json(ctx.run_dir / "00_ingest" / "schema.json", default=None)
    if not isinstance(raw, dict):
        raise StageError("Missing ingest schema.", "Run ingest first.")
    return DatasetSpec.from_dict(raw)


def _encode_target(y: np.ndarray, task: str) -> np.ndarray:
    if task == "regression":
        return np.asarray(y, dtype=float)
    encoder = LabelEncoder()
    return encoder.fit_transform(np.asarray(y)).astype(np.int64)


def _predict(adapter, X: pd.DataFrame, task: str) -> np.ndarray:
    if task == "regression":
        return np.asarray(adapter.predict(X), dtype=float)
    proba = np.asarray(adapter.predict_proba(X), dtype=float)
    if task == "binary":
        return proba[:, 1] if proba.ndim == 2 else proba.reshape(-1)
    if proba.ndim == 1:
        raise ModelError("Multiclass model returned a 1-D probability array.", "Check the adapter contract.")
    return proba


def _better(metric_name: str, first: float, second: float | None) -> bool:
    if second is None:
        return True
    direction = get_metric(metric_name).direction
    return first > second if direction == "maximize" else first < second


def _sample_params(rng: np.random.Generator, model_name: str, base: dict[str, Any]) -> dict[str, Any]:
    """Draw one candidate, always keeping recipe parameters that are not searched."""
    params = dict(base)
    space = SEARCH_SPACES.get(model_name, {})
    for key, values in space.items():
        params[key] = values[int(rng.integers(0, len(values)))]
    return params


def _inner_split(
    outer: pd.DataFrame,
    y_outer: np.ndarray,
    config,
    spec: DatasetSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Create a validation split inside one outer training fold."""
    n_rows = len(outer)
    inner_rows = max(1, int(round(n_rows * config.hpo.inner_size)))
    if n_rows < 20 or n_rows - inner_rows < 10:
        return None

    group_col = config.validation.group_col
    if group_col:
        if group_col not in outer.columns:
            return None
        groups = outer[group_col].astype(str).to_numpy()
        if len(np.unique(groups)) < 2:
            return None
        splitter = GroupShuffleSplit(n_splits=1, test_size=config.hpo.inner_size, random_state=int(rng.integers(0, 2**31 - 1)))
        try:
            train_pos, valid_pos = next(splitter.split(outer, y_outer, groups=groups))
        except ValueError:
            return None
        return np.asarray(train_pos, dtype=np.int64), np.asarray(valid_pos, dtype=np.int64)

    time_col = config.validation.time_col
    if time_col:
        if time_col not in outer.columns:
            return None
        try:
            values = pd.to_datetime(outer[time_col], errors="coerce", format="mixed")
        except (TypeError, ValueError):
            values = pd.to_datetime(["NaT"] * n_rows)
        if values.isna().all():
            numeric = pd.to_numeric(outer[time_col], errors="coerce")
            if numeric.isna().all():
                return None
            order = np.argsort(numeric.to_numpy(), kind="stable")
        else:
            order = np.argsort(values.to_numpy(), kind="stable")
        split_at = max(1, n_rows - max(1, inner_rows))
        if split_at >= n_rows:
            return None
        return np.asarray(order[:split_at], dtype=np.int64), np.asarray(order[split_at:], dtype=np.int64)

    stratify = None
    counts = pd.Series(y_outer).value_counts(dropna=False)
    if len(counts) >= 2 and int(counts.min()) >= 2 and len(counts) <= 20:
        stratify = y_outer
    try:
        train_pos, valid_pos = train_test_split(
            np.arange(n_rows),
            test_size=config.hpo.inner_size,
            random_state=int(rng.integers(0, 2**31 - 1)),
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        return None
    return np.asarray(train_pos, dtype=np.int64), np.asarray(valid_pos, dtype=np.int64)


def _early_stop_split(
    outer: pd.DataFrame,
    y_outer: np.ndarray,
    train_pos: np.ndarray,
    config,
    spec: DatasetSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Hold out a small inner slice used only for early stopping.

    The HPO score must be computed on rows that never influenced the number of
    boosting rounds.  Using the same inner-validation rows for both early
    stopping and scoring optimistically biases every trial.
    """
    train_pos = np.asarray(train_pos, dtype=np.int64)
    if len(train_pos) < 30:
        return train_pos, None
    stop_size = max(5, int(round(len(train_pos) * 0.2)))
    if len(train_pos) - stop_size < 10:
        return train_pos, None
    group_col = config.validation.group_col
    if group_col and group_col in outer.columns:
        groups = outer.iloc[train_pos][group_col].astype(str).to_numpy()
        if len(np.unique(groups)) >= 2:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=0.2,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            try:
                fit_local, stop_local = next(splitter.split(train_pos, groups=groups))
            except ValueError:
                return train_pos, None
            fit_pos = train_pos[np.asarray(fit_local, dtype=np.int64)]
            stop_pos = train_pos[np.asarray(stop_local, dtype=np.int64)]
            if len(fit_pos) and len(stop_pos):
                return fit_pos, stop_pos
        return train_pos, None
    time_col = config.validation.time_col
    if time_col and time_col in outer.columns:
        values = pd.to_numeric(outer.iloc[train_pos][time_col], errors="coerce")
        if values.isna().all():
            try:
                parsed = pd.to_datetime(outer.iloc[train_pos][time_col], errors="coerce", format="mixed")
            except (TypeError, ValueError):
                parsed = pd.Series(pd.NaT, index=values.index)
            values = pd.Series(parsed.to_numpy(), index=values.index)
            if values.isna().all():
                return train_pos, None
        order = np.argsort(values.to_numpy(), kind="stable")
        split_at = len(order) - stop_size
        if split_at < 10:
            return train_pos, None
        return train_pos[order[:split_at]], train_pos[order[split_at:]]
    stratify = None
    counts = pd.Series(y_outer[train_pos]).value_counts(dropna=False)
    if spec.task in {"binary", "multiclass"} and len(counts) >= 2 and int(counts.min()) >= 2:
        stratify = y_outer[train_pos]
    try:
        fit_local, stop_local = train_test_split(
            np.arange(len(train_pos)),
            test_size=stop_size,
            random_state=int(rng.integers(0, 2**31 - 1)),
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        return train_pos, None
    fit_pos = train_pos[np.asarray(fit_local, dtype=np.int64)]
    stop_pos = train_pos[np.asarray(stop_local, dtype=np.int64)]
    if not len(fit_pos) or not len(stop_pos):
        return train_pos, None
    return fit_pos, stop_pos


def _evaluate(
    adapter,
    builder: FeatureBuilder,
    outer: pd.DataFrame,
    y_outer: np.ndarray,
    fit_pos: np.ndarray,
    stop_pos: np.ndarray | None,
    valid_pos: np.ndarray,
    params: dict[str, Any],
    model_config: ModelConfig,
    ctx: RunContext,
    spec: DatasetSpec,
    timeout_sec: float,
) -> tuple[float, Any]:
    X_fit = builder.fit_transform(outer.iloc[fit_pos].reset_index(drop=True), y_outer[fit_pos])
    X_valid = builder.transform(outer.iloc[valid_pos].reset_index(drop=True))
    if stop_pos is not None and model_config.early_stopping_rounds and model_config.early_stopping_rounds > 0:
        X_stop = builder.transform(outer.iloc[stop_pos].reset_index(drop=True))
        adapter = fit_with_timeout(
            adapter,
            X_fit,
            y_outer[fit_pos],
            X_stop,
            y_outer[stop_pos],
            params=params,
            seed=int(ctx.config.seed),
            early_stopping_rounds=model_config.early_stopping_rounds,
            timeout_sec=timeout_sec,
        )
    else:
        adapter = fit_with_timeout(
            adapter,
            X_fit,
            y_outer[fit_pos],
            None,
            None,
            params=params,
            seed=int(ctx.config.seed),
            early_stopping_rounds=0,
            timeout_sec=timeout_sec,
        )
    prediction = _predict(adapter, X_valid, spec.task)
    return float(compute_metric(spec.metric, y_outer[valid_pos], prediction, spec.task)), adapter


def _tune_fold(
    ctx: RunContext,
    spec: DatasetSpec,
    train: pd.DataFrame,
    y: np.ndarray,
    fold: Fold,
    model_config: ModelConfig,
    deadline: float,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    outer = train.iloc[fold.train_idx].reset_index(drop=True)
    y_outer = y[fold.train_idx]
    base_params = dict(model_config.params)
    default_result: dict[str, Any] = {
        "params": base_params,
        "score": None,
        "trials_run": 0,
        "used_default": True,
    }
    if model_config.name not in SEARCH_SPACES:
        return default_result, {"reason": "no HPO search space"}
    split = _inner_split(outer, y_outer, ctx.config, spec, rng)
    if split is None:
        return default_result, {"reason": "not enough rows for an inner split"}

    train_pos, valid_pos = split
    fit_pos, stop_pos = _early_stop_split(outer, y_outer, train_pos, ctx.config, spec, rng)
    adapter = create_model(model_config.name)
    adapter.task = spec.task
    builder = FeatureBuilder(ctx.config, spec, family=adapter.family)
    best_params = dict(base_params)
    best_score: float | None = None
    trials_run = 0
    attempts = max(1, int(ctx.config.hpo.trials))
    last_error: str | None = None
    timed_out = False

    for _ in range(attempts):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        candidate = _sample_params(rng, model_config.name, base_params) if trials_run else dict(base_params)
        trials_run += 1
        try:
            score, adapter = _evaluate(
                adapter,
                builder,
                outer,
                y_outer,
                fit_pos,
                stop_pos,
                valid_pos,
                candidate,
                model_config,
                ctx,
                spec,
                timeout_sec=remaining,
            )
        except TimeoutError as exc:
            last_error = str(exc)
            timed_out = True
            break
        except (KconvError, ValueError, TypeError, RuntimeError) as exc:
            last_error = str(exc)
            continue
        if _better(spec.metric, score, best_score):
            best_score = score
            best_params = dict(candidate)

    if best_score is None:
        return {**default_result, "trials_run": trials_run}, {
            "reason": "all trials failed",
            "last_error": last_error,
            "timeout_hit": timed_out,
        }
    return (
        {
            "params": best_params,
            "score": best_score,
            "trials_run": trials_run,
            "used_default": best_params == base_params,
        },
        {
            "reason": None,
            "last_error": last_error,
            "timeout_hit": timed_out,
            "fit_rows": int(len(fit_pos)),
            "early_stop_rows": int(len(stop_pos)) if stop_pos is not None else 0,
            "score_rows": int(len(valid_pos)),
        },
    )


def run_hpo(ctx: RunContext) -> dict:
    if not ctx.config.hpo.enabled:
        return {"status": "skipped", "enabled": False, "artifacts": [], "tuned": 0}

    spec = _spec(ctx)
    train = pd.read_parquet(ctx.run_dir / "00_ingest" / "train.parquet")
    folds = load_run_folds(ctx.run_dir)
    y = _encode_target(train[spec.target].to_numpy(), spec.task)
    model_configs = [m for m in ctx.config.models if m.enabled]
    started = time.perf_counter()
    deadline = started + float(ctx.config.hpo.timeout_sec)
    rng = np.random.default_rng(ctx.config.seed)
    fold_payload: dict[str, dict[str, Any]] = {}
    tuned = 0
    timeout_hit = False
    tuned_models: set[str] = set()
    skipped_reasons: dict[str, set[str]] = {}

    for fold in folds:
        fold_payload[str(fold.index)] = {}
        for model_config in model_configs:
            try:
                params, detail = _tune_fold(
                    ctx,
                    spec,
                    train,
                    y,
                    fold,
                    model_config,
                    deadline,
                    rng,
                )
            except Exception as exc:
                # HPO is an optimization, not a correctness requirement.  A
                # broken optional model or a feature setup error must not stop
                # the whole run; train will either skip that model or surface
                # the underlying data error with the remaining models intact.
                params = {
                    "params": dict(model_config.params),
                    "score": None,
                    "trials_run": 0,
                    "used_default": True,
                }
                detail = {
                    "reason": "HPO setup failed",
                    "last_error": f"{type(exc).__name__}: {exc}",
                }
                message = f"HPO skipped for '{model_config.name}': {exc}"
                if message not in ctx.warnings:
                    ctx.warnings.append(message)
            fold_payload[str(fold.index)][model_config.name] = {**params, "detail": detail}
            if params.get("score") is not None and not params.get("used_default"):
                tuned += 1
                tuned_models.add(model_config.name)
            reason = detail.get("reason") if isinstance(detail, dict) else None
            if reason:
                skipped_reasons.setdefault(model_config.name, set()).add(str(reason))
            if isinstance(detail, dict) and detail.get("timeout_hit"):
                timeout_hit = True
            if timeout_hit or time.perf_counter() >= deadline:
                timeout_hit = True
                break
        if timeout_hit:
            break

    if timeout_hit:
        for model_config in model_configs:
            skipped_reasons.setdefault(model_config.name, set()).add("HPO timeout")

    if timeout_hit:
        message = (
            f"HPO budget of {ctx.config.hpo.timeout_sec}s was exhausted; "
            "remaining folds/models use recipe defaults."
        )
        if message not in ctx.warnings:
            ctx.warnings.append(message)

    skipped = [
        {"model": name, "reasons": sorted(reasons)}
        for name, reasons in sorted(skipped_reasons.items())
        if reasons and name not in tuned_models
    ]
    if skipped:
        names = ", ".join(item["model"] for item in skipped)
        message = f"HPO was not applied to: {names}. See 01_hpo/tuning.json for reasons."
        if message not in ctx.warnings:
            ctx.warnings.append(message)

    payload = {
        "enabled": True,
        "metric": spec.metric,
        "task": spec.task,
        "trials": int(ctx.config.hpo.trials),
        "inner_size": float(ctx.config.hpo.inner_size),
        "timeout_sec": int(ctx.config.hpo.timeout_sec),
        "timeout_hit": timeout_hit,
        "duration_sec": round(time.perf_counter() - started, 3),
        "models": [m.name for m in model_configs],
        "models_tuned": sorted(tuned_models),
        "models_skipped": skipped,
        "folds": fold_payload,
        "tuned_fold_models": tuned,
    }
    path = ctx.artifact(STAGE, "tuning.json")
    write_json(path, payload)
    return {
        "status": "ok",
        "enabled": True,
        "artifacts": [str(path.relative_to(ctx.run_dir)).replace("\\", "/")],
        "tuned": tuned,
        "timeout_hit": timeout_hit,
    }
