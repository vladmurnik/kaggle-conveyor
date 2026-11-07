"""CatBoost adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from kconv.errors import ModelError
from kconv.models.base import BaseAdapter


class CatBoostAdapter(BaseAdapter):
    name = "catboost"
    task_types = {"binary", "multiclass", "regression"}
    family = "tree"

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, params=None, seed=42, early_stopping_rounds=100):
        catboost = self._require("catboost", "catboost")
        self._remember_features(X_tr)
        supplied = dict(params or {})
        defaults: dict[str, Any] = {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "random_seed": seed,
            "verbose": 0,
            "allow_writing_files": False,
            "thread_count": -1,
        }
        defaults.update(supplied)
        defaults["random_seed"] = supplied.get("random_seed", seed)
        if self.task == "regression":
            model = catboost.CatBoostRegressor(loss_function="RMSE", **defaults)
        elif self.task == "multiclass":
            model = catboost.CatBoostClassifier(loss_function="MultiClass", **defaults)
        else:
            model = catboost.CatBoostClassifier(loss_function="Logloss", **defaults)
        try:
            if X_va is not None and y_va is not None and early_stopping_rounds and early_stopping_rounds > 0:
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_va, y_va),
                    early_stopping_rounds=int(early_stopping_rounds),
                    use_best_model=True,
                )
            else:
                model.fit(X_tr, y_tr)
        except Exception as exc:  # pragma: no cover - depends on data/version
            raise ModelError(f"CatBoost training failed: {exc}", "Check target/feature dtypes and params.") from exc
        self.model = model

    def predict_proba(self, X):
        if self.task == "regression":
            raise AttributeError("Regression models do not expose predict_proba")
        proba = np.asarray(self.model.predict_proba(X))
        if self.task == "binary":
            return proba[:, 1]
        return proba
