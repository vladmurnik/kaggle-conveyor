"""LightGBM adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from kconv.errors import ModelError
from kconv.models.base import BaseAdapter


class LGBMAdapter(BaseAdapter):
    name = "lgbm"
    task_types = {"binary", "multiclass", "regression"}
    family = "tree"

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, params=None, seed=42, early_stopping_rounds=100):
        lgb = self._require("lightgbm", "lightgbm")
        self._remember_features(X_tr)
        supplied = dict(params or {})
        defaults: dict[str, Any] = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "random_state": seed,
            "num_threads": 0,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }
        defaults.update(supplied)
        defaults["random_state"] = supplied.get("random_state", seed)
        if self.task == "regression":
            model = lgb.LGBMRegressor(**defaults)
        elif self.task == "multiclass":
            model = lgb.LGBMClassifier(**defaults)
        else:
            model = lgb.LGBMClassifier(**defaults)
        callbacks = []
        eval_set = None
        if X_va is not None and y_va is not None and early_stopping_rounds and early_stopping_rounds > 0:
            eval_set = [(X_va, y_va)]
            callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))
        try:
            if eval_set is not None:
                # LightGBM 4.7 deprecates eval_set in favor of eval_X/eval_y.
                try:
                    model.fit(
                        X_tr,
                        y_tr,
                        eval_X=(X_va,),
                        eval_y=(y_va,),
                        callbacks=callbacks or None,
                    )
                except TypeError:
                    model.fit(X_tr, y_tr, eval_set=eval_set, callbacks=callbacks or None)
            else:
                model.fit(X_tr, y_tr)
        except Exception as exc:  # pragma: no cover - depends on data/version
            raise ModelError(f"LightGBM training failed: {exc}", "Check target/feature dtypes and params.") from exc
        self.model = model

    def predict_proba(self, X):
        if self.task == "regression":
            raise AttributeError("Regression models do not expose predict_proba")
        proba = self.model.predict_proba(X)
        proba = np.asarray(proba)
        if self.task == "binary":
            return proba[:, 1]
        return proba

    def importance(self):
        result = super().importance()
        return result
