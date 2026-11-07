"""Shared model adapter behavior."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from kconv.errors import ModelError


class BaseAdapter:
    """Small convenience base; adapters may override any method."""

    name = "base"
    task_types: set[str] = set()
    family = "tree"
    task = "binary"
    optional = False

    def __init__(self) -> None:
        self.model: Any = None
        self.feature_columns: list[str] = []

    def _require(self, module: str, package: str) -> Any:
        try:
            return __import__(module, fromlist=["*"])
        except ImportError as exc:  # pragma: no cover - exercised only without extras
            raise ModelError(
                f"Model '{self.name}' needs the optional package '{package}'.",
                f"Install it with: uv add {package}",
            ) from exc

    def _remember_features(self, X: pd.DataFrame) -> None:
        self.feature_columns = [str(c) for c in X.columns]

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, params=None, seed=42, early_stopping_rounds=100):
        raise NotImplementedError

    def predict(self, X):
        if self.model is None:
            raise ModelError(f"Model '{self.name}' is not fitted.", "Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ModelError(f"Model '{self.name}' is not fitted.", "Call fit() first.")
        return self.model.predict_proba(X)

    def importance(self) -> pd.Series | None:
        model = self.model
        if model is None or not hasattr(model, "feature_importances_"):
            return None
        values = np.asarray(model.feature_importances_, dtype=float)
        columns = self.feature_columns or [f"f{i}" for i in range(len(values))]
        if len(columns) != len(values):
            columns = [f"f{i}" for i in range(len(values))]
        return pd.Series(values, index=columns, name="importance").sort_values(ascending=False)
