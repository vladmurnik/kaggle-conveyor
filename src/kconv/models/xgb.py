"""XGBoost adapter (optional at runtime, installed in the default extra)."""

from __future__ import annotations

from typing import Any

import numpy as np

from kconv.errors import ModelError
from kconv.models.base import BaseAdapter


class XGBAdapter(BaseAdapter):
    name = "xgb"
    task_types = {"binary", "multiclass", "regression"}
    family = "tree"

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, params=None, seed=42, early_stopping_rounds=100):
        xgb = self._require("xgboost", "xgboost")
        self._remember_features(X_tr)
        supplied = dict(params or {})
        n_estimators = int(supplied.pop("n_estimators", 500))
        defaults: dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
        }
        defaults.update(supplied)
        defaults["random_state"] = supplied.get("random_state", seed)
        if X_va is not None and y_va is not None and early_stopping_rounds and early_stopping_rounds > 0:
            defaults["early_stopping_rounds"] = int(early_stopping_rounds)
        if self.task == "regression":
            model = xgb.XGBRegressor(**defaults)
        else:
            model = xgb.XGBClassifier(**defaults)
        eval_set = [(X_va, y_va)] if X_va is not None and y_va is not None else None
        try:
            model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        except Exception as exc:  # pragma: no cover - depends on data/version
            raise ModelError(f"XGBoost training failed: {exc}", "Check target/feature dtypes and params.") from exc
        self.model = model

    def predict_proba(self, X):
        if self.task == "regression":
            raise AttributeError("Regression models do not expose predict_proba")
        proba = np.asarray(self.model.predict_proba(X))
        if self.task == "binary":
            return proba[:, 1]
        return proba
