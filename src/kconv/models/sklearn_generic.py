"""Generic adapter for the scikit-learn estimator catalog."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from kconv.errors import ModelError
from kconv.models.base import BaseAdapter


@dataclass(frozen=True)
class SklearnSpec:
    """Description of one scikit-learn estimator entry."""

    path: str
    task_types: frozenset[str]
    family: str = "linear"
    defaults: dict[str, Any] = field(default_factory=dict)
    seed_param: str | None = "random_state"


def _load_class(path: str) -> type:
    module_name, class_name = path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ModelError(
            f"Cannot load scikit-learn estimator '{path}': {exc}",
            "Reinstall scikit-learn or check the model catalog.",
        ) from exc


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=float).reshape(-1), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2:
        raise ModelError("A multiclass decision function must return a 2-D array.")
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.maximum(exponent.sum(axis=1, keepdims=True), 1e-12)


class SklearnAdapter(BaseAdapter):
    """Fit any sklearn-compatible estimator with kconv's task contract."""

    def __init__(self, name: str, spec: SklearnSpec) -> None:
        super().__init__()
        self.name = name
        self.spec = spec
        self.task_types = set(spec.task_types)
        self.family = spec.family

    def fit(
        self,
        X_tr,
        y_tr,
        X_va=None,
        y_va=None,
        params=None,
        seed=42,
        early_stopping_rounds=100,
    ):
        estimator_class = _load_class(self.spec.path)
        self._remember_features(X_tr)
        supplied = dict(params or {})
        merged = dict(self.spec.defaults)
        merged.update(supplied)
        if self.spec.seed_param and self.spec.seed_param not in supplied:
            signature = inspect.signature(estimator_class)
            if self.spec.seed_param in signature.parameters:
                merged[self.spec.seed_param] = seed
        try:
            model = estimator_class(**merged)
            model.fit(X_tr, y_tr)
        except Exception as exc:  # pragma: no cover - depends on estimator/data
            raise ModelError(
                f"{self.name} training failed: {exc}",
                "Check the feature family and model parameters.",
            ) from exc
        self.model = model

    def predict(self, X):
        if self.model is None:
            raise ModelError(f"Model '{self.name}' is not fitted.", "Call fit() first.")
        values = np.asarray(self.model.predict(X))
        if self.task == "regression":
            return values.reshape(-1)
        return values

    def predict_proba(self, X):
        if self.task == "regression":
            raise AttributeError("Regression models do not expose predict_proba")
        if self.model is None:
            raise ModelError(f"Model '{self.name}' is not fitted.", "Call fit() first.")
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(X), dtype=float)
            if self.task == "binary":
                if proba.ndim == 2 and proba.shape[1] > 1:
                    return proba[:, 1]
                return proba.reshape(-1)
            return proba
        if hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(X), dtype=float)
            if self.task == "binary":
                return _sigmoid(scores)
            return _softmax(scores)
        raise ModelError(
            f"Model '{self.name}' does not expose predict_proba or decision_function.",
            "Choose another classifier.",
        )

    def importance(self) -> pd.Series | None:
        model = self.model
        if model is None:
            return None
        if hasattr(model, "feature_importances_"):
            return super().importance()
        if not hasattr(model, "coef_"):
            return None
        coefficients = np.asarray(model.coef_)
        values = np.abs(coefficients).max(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
        columns = self.feature_columns or [f"f{i}" for i in range(len(values))]
        if len(columns) != len(values):
            columns = [f"f{i}" for i in range(len(values))]
        return pd.Series(values, index=columns, name="importance").sort_values(ascending=False)
