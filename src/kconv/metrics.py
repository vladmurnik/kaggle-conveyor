"""Metric registry with explicit task/probability requirements."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn import metrics as skm

from kconv.errors import ModelError

Direction = Literal["maximize", "minimize"]
Requires = Literal["proba", "labels", "raw"]


@dataclass(frozen=True)
class Metric:
    name: str
    direction: Direction
    requires: Requires
    fn: Callable[[np.ndarray, np.ndarray], float]


def _labels(y_pred: np.ndarray) -> np.ndarray:
    array = np.asarray(y_pred)
    if array.ndim == 2:
        return np.argmax(array, axis=1)
    if array.ndim == 1 and np.issubdtype(array.dtype, np.floating):
        return (array >= 0.5).astype(int)
    return array


def _roc_auc(y_true, y_pred):
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2:
        # A two-column probability matrix is still a binary classifier, even when
        # the task was configured as multiclass.  sklearn's multiclass path expects
        # one-hot y_true and rejects a normal 1-D label vector here.
        if y_pred.shape[1] == 2:
            return float(skm.roc_auc_score(y_true, y_pred[:, 1]))
        return float(skm.roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro"))
    return float(skm.roc_auc_score(y_true, y_pred))


def _log_loss(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    if y_pred.ndim == 2:
        # XGBoost/CatBoost can leave a row at 0.99999994 after float32
        # accumulation.  Normalize the tiny round-off before sklearn warns
        # (or rejects) the otherwise valid probability matrix.
        row_sums = y_pred.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("Every probability row must have a positive sum.")
        y_pred = y_pred / row_sums
        return float(skm.log_loss(y_true, y_pred))
    return float(skm.log_loss(y_true, y_pred, labels=[0, 1]))


def _accuracy(y_true, y_pred):
    return float(skm.accuracy_score(y_true, _labels(y_pred)))


def _f1(y_true, y_pred):
    labels = _labels(y_pred)
    unique = np.unique(y_true)
    average = "binary" if len(unique) <= 2 else "macro"
    return float(skm.f1_score(y_true, labels, average=average))


def _rmse(y_true, y_pred):
    return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))


def _mae(y_true, y_pred):
    return float(skm.mean_absolute_error(y_true, y_pred))


def _r2(y_true, y_pred):
    return float(skm.r2_score(y_true, y_pred))


_METRICS: dict[str, Metric] = {
    "roc_auc": Metric("roc_auc", "maximize", "proba", _roc_auc),
    "log_loss": Metric("log_loss", "minimize", "proba", _log_loss),
    "accuracy": Metric("accuracy", "maximize", "labels", _accuracy),
    "f1": Metric("f1", "maximize", "labels", _f1),
    "rmse": Metric("rmse", "minimize", "raw", _rmse),
    "mae": Metric("mae", "minimize", "raw", _mae),
    "r2": Metric("r2", "maximize", "raw", _r2),
}

_METRIC_TASKS: dict[str, frozenset[str]] = {
    "roc_auc": frozenset({"binary", "multiclass"}),
    "log_loss": frozenset({"binary", "multiclass"}),
    "accuracy": frozenset({"binary", "multiclass"}),
    "f1": frozenset({"binary", "multiclass"}),
    "rmse": frozenset({"regression"}),
    "mae": frozenset({"regression"}),
    "r2": frozenset({"regression"}),
}


def metric_names() -> list[str]:
    return sorted(_METRICS)


def metric_tasks(name: str) -> frozenset[str]:
    """Return the task types supported by a metric."""
    get_metric(name)
    return _METRIC_TASKS[name]


def metric_supports_task(name: str, task: str) -> bool:
    return task in metric_tasks(name)


def validate_metric_task(name: str, task: str) -> None:
    if task not in metric_tasks(name):
        supported = ", ".join(sorted(metric_tasks(name)))
        raise ModelError(
            f"Metric '{name}' does not support task '{task}'.",
            f"Use one of: {supported}.",
        )


def default_metric(task: str) -> str:
    if task == "binary":
        return "roc_auc"
    if task == "multiclass":
        return "log_loss"
    return "rmse"


def get_metric(name: str) -> Metric:
    try:
        return _METRICS[name]
    except KeyError as exc:
        raise ModelError(f"Unknown metric '{name}'.", f"Available metrics: {', '.join(metric_names())}") from exc


def compute_metric(name: str, y_true, y_pred, task: str) -> float:
    metric = get_metric(name)
    validate_metric_task(name, task)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        raise ModelError("Cannot compute a metric on zero rows.", "Check the fold/validation artifacts.")
    if task in {"binary", "multiclass"} and metric.name == "roc_auc" and len(np.unique(y_true)) < 2:
        raise ModelError("roc_auc needs both classes in y_true.", "Use a different metric or more folds.")
    try:
        return float(metric.fn(y_true, y_pred))
    except Exception as exc:
        raise ModelError(f"Metric '{name}' failed: {exc}", "Check prediction shape/type for this task.") from exc
