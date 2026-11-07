"""Scikit-learn model adapters: HistGradientBoosting, LogisticRegression, Ridge."""

from __future__ import annotations

from typing import Any

import numpy as np

from kconv.errors import ModelError
from kconv.models.base import BaseAdapter


class HistGBAdapter(BaseAdapter):
    name = "histgb"
    task_types = {"binary", "multiclass", "regression"}
    family = "tree"

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, params=None, seed=42, early_stopping_rounds=100):
        sklearn_ensemble = self._require("sklearn.ensemble", "scikit-learn")
        self._remember_features(X_tr)
        supplied = dict(params or {})
        defaults: dict[str, Any] = {
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.1,
            "random_state": seed,
            "early_stopping": False,
        }
        defaults.update(supplied)
        defaults["random_state"] = supplied.get("random_state", seed)
        if self.task == "regression":
            model = sklearn_ensemble.HistGradientBoostingRegressor(**defaults)
        else:
            model = sklearn_ensemble.HistGradientBoostingClassifier(**defaults)
        try:
            model.fit(X_tr, y_tr)
        except Exception as exc:  # pragma: no cover
            raise ModelError(f"HistGradientBoosting training failed: {exc}", "Check feature dtypes.") from exc
        self.model = model

    def predict_proba(self, X):
        if self.task == "regression":
            raise AttributeError("Regression models do not expose predict_proba")
        proba = np.asarray(self.model.predict_proba(X))
        if self.task == "binary":
            return proba[:, 1]
        return proba

    def importance(self):
        return None


class LogRegAdapter(BaseAdapter):
    name = "logreg"
    task_types = {"binary", "multiclass"}
    family = "linear"

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, params=None, seed=42, early_stopping_rounds=100):
        sklearn_linear = self._require("sklearn.linear_model", "scikit-learn")
        self._remember_features(X_tr)
        supplied = dict(params or {})
        defaults: dict[str, Any] = {
            "max_iter": 1000,
            "random_state": seed,
            "solver": "lbfgs",
        }
        defaults.update(supplied)
        defaults["random_state"] = supplied.get("random_state", seed)
        model = sklearn_linear.LogisticRegression(**defaults)
        try:
            model.fit(X_tr, y_tr)
        except Exception as exc:  # pragma: no cover
            raise ModelError(f"LogisticRegression training failed: {exc}", "Check feature scaling.") from exc
        self.model = model

    def predict_proba(self, X):
        proba = np.asarray(self.model.predict_proba(X))
        if self.task == "binary":
            return proba[:, 1]
        return proba

    def importance(self):
        if self.model is None:
            return None
        coefficients = np.asarray(self.model.coef_)
        values = np.abs(coefficients).max(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
        import pandas as pd

        return pd.Series(values, index=self.feature_columns, name="importance").sort_values(ascending=False)


class RidgeAdapter(BaseAdapter):
    name = "ridge"
    task_types = {"regression"}
    family = "linear"

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, params=None, seed=42, early_stopping_rounds=100):
        sklearn_linear = self._require("sklearn.linear_model", "scikit-learn")
        self._remember_features(X_tr)
        supplied = dict(params or {})
        defaults: dict[str, Any] = {"alpha": 1.0}
        defaults.update(supplied)
        try:
            model = sklearn_linear.Ridge(**defaults)
            model.fit(X_tr, y_tr)
        except Exception as exc:  # pragma: no cover
            raise ModelError(f"Ridge training failed: {exc}", "Check feature scaling.") from exc
        self.model = model

    def predict_proba(self, X):
        raise AttributeError("Regression models do not expose predict_proba")

    def importance(self):
        if self.model is None:
            return None
        import pandas as pd

        return pd.Series(np.abs(self.model.coef_), index=self.feature_columns, name="importance").sort_values(
            ascending=False
        )
