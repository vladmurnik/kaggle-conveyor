"""Catalog-level tests: all registered estimators must fit the task contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kconv.models import create_model, model_available
from kconv.registry import model_names


def test_catalog_has_fifty_one_models() -> None:
    names = model_names()
    assert len(names) == 51
    assert {"lgbm", "xgb", "catboost", "histgb", "logreg", "ridge"} <= set(names)
    assert {"random_forest_classifier", "random_forest_regressor", "svc", "mlp_classifier"} <= set(names)
    assert "tabpfn" in names


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_all_catalog_models_fit_and_predict() -> None:
    rng = np.random.default_rng(7)
    n = 100
    features = pd.DataFrame(
        rng.normal(size=(n, 4)),
        columns=[f"x{i}" for i in range(4)],
    )
    binary_target = (features["x0"] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    regression_target = 2 * features["x0"] - features["x1"] + rng.normal(scale=0.2, size=n)

    failures: list[str] = []
    skipped: list[str] = []
    for name in model_names():
        if not model_available(name):
            skipped.append(name)
            continue
        adapter = create_model(name)
        task = "regression" if "regression" in adapter.task_types else "binary"
        adapter.task = task
        target = regression_target if task == "regression" else binary_target
        try:
            adapter.fit(
                features.iloc[:80],
                target[:80],
                features.iloc[80:],
                target[80:],
                seed=1,
                early_stopping_rounds=10,
            )
            prediction = np.asarray(adapter.predict(features.iloc[80:]))
            assert prediction.shape[0] == 20
            if task != "regression":
                probability = np.asarray(adapter.predict_proba(features.iloc[80:]))
                assert probability.shape[0] == 20
                assert np.isfinite(probability).all()
        except Exception as exc:  # pragma: no cover - only hit on a broken catalog entry
            failures.append(f"{name} ({task}): {exc}")

    assert not failures, "\n".join(failures)
    assert not set(skipped) - {"tabpfn"}, f"unexpected skipped optional models: {skipped}"
