"""Model adapter and metric tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from kconv.errors import ModelError
from kconv.metrics import compute_metric, default_metric, metric_names
from kconv.models import create_model


def _data(task: str = "binary") -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(3)
    n = 80
    frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    if task == "regression":
        target = 1.5 * frame["a"].to_numpy() - 0.5 * frame["b"].to_numpy() + rng.normal(scale=0.1, size=n)
    else:
        target = (frame["a"].to_numpy() + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return frame, target


@pytest.mark.parametrize("name", ["lgbm", "histgb", "logreg"])
def test_binary_adapters(name: str) -> None:
    X, y = _data()
    model = create_model(name)
    model.task = "binary"
    model.fit(X.iloc[:60], y[:60], X.iloc[60:], y[60:], params={}, seed=1, early_stopping_rounds=20)
    proba = model.predict_proba(X.iloc[60:])
    assert proba.shape == (20,)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()
    assert model.predict(X.iloc[60:]).shape == (20,)


def test_ridge_regression() -> None:
    X, y = _data("regression")
    model = create_model("ridge")
    model.task = "regression"
    model.fit(X.iloc[:60], y[:60])
    pred = model.predict(X.iloc[60:])
    assert pred.shape == (20,)
    with pytest.raises(AttributeError):
        model.predict_proba(X.iloc[60:])


@pytest.mark.parametrize(
    ("name", "task", "params"),
    [
        ("lgbm", "binary", {"num_leaves": 0}),
        ("xgb", "binary", {"learning_rate": -1.0}),
        ("catboost", "binary", {"iterations": 0}),
        ("histgb", "binary", {"max_iter": 0}),
        ("logreg", "binary", {"C": -1.0}),
        ("ridge", "regression", {"alpha": -1.0}),
    ],
)
def test_invalid_model_params_raise_actionable_model_error(
    name: str, task: str, params: dict[str, object]
) -> None:
    X, y = _data(task)
    model = create_model(name)
    model.task = task
    with pytest.raises(ModelError):
        model.fit(X.iloc[:60], y[:60], X.iloc[60:], y[60:], params=params, seed=1)


def test_metrics() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.2, 0.9])
    assert 0.0 <= compute_metric("roc_auc", y, p, "binary") <= 1.0
    assert compute_metric("accuracy", y, (p >= 0.5).astype(int), "binary") == 1.0
    assert compute_metric("rmse", np.array([1.0, 2.0]), np.array([1.0, 2.0]), "regression") == 0.0
    assert default_metric("binary") == "roc_auc"
    assert "log_loss" in metric_names()


def test_roc_auc_accepts_two_column_binary_probabilities() -> None:
    y = np.array([0, 0, 1, 1, 1, 0])
    positive = np.array([0.1, 0.35, 0.62, 0.8, 0.9, 0.2])
    probabilities = np.column_stack([1.0 - positive, positive])
    expected = compute_metric("roc_auc", y, positive, "binary")
    assert compute_metric("roc_auc", y, probabilities, "binary") == pytest.approx(expected)
    assert compute_metric("roc_auc", y, probabilities, "multiclass") == pytest.approx(expected)


def test_roc_auc_multiclass_probabilities() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[y]
    assert compute_metric("roc_auc", y, probabilities, "multiclass") == pytest.approx(1.0)


def test_log_loss_accepts_float32_probability_roundoff_without_warning() -> None:
    y = np.array([0, 1, 2])
    probabilities = np.array(
        [
            [0.33333331, 0.33333334, 0.33333335],
            [0.33333334, 0.33333331, 0.33333335],
            [0.33333335, 0.33333334, 0.33333331],
        ],
        dtype=np.float32,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = compute_metric("log_loss", y, probabilities, "multiclass")
    assert np.isfinite(score)
    assert not any("sum to one" in str(item.message) for item in caught)


def test_metric_task_mismatch_is_rejected() -> None:
    with pytest.raises(ModelError, match="does not support task"):
        compute_metric("rmse", np.array([1.0, 2.0]), np.array([1.0, 2.0]), "binary")


def test_tree_adapters_use_all_cores_by_default() -> None:
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from kconv.models import create_model

    X = pd.DataFrame({"a": [0, 1, 2, 3], "b": [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])
    cases = [
        ("xgb", "XGBClassifier", "n_jobs", -1),
        ("lgbm", "LGBMClassifier", "num_threads", 0),
        ("catboost", "CatBoostClassifier", "thread_count", -1),
    ]
    for name, estimator_name, key, expected in cases:
        captured = {}

        class FakeEstimator:
            def __init__(self, _captured=captured, **kwargs):
                _captured.update(kwargs)

            def fit(self, X, y, **kwargs):
                return self

            def predict_proba(self, X):
                return np.zeros((len(X), 2))

        fake = SimpleNamespace(
            **{
                estimator_name: FakeEstimator,
                estimator_name.replace("Classifier", "Regressor"): FakeEstimator,
            }
        )
        model = create_model(name)
        model.task = "binary"
        model._require = lambda package, extra, _fake=fake: _fake
        model.fit(X, y, params={}, seed=1, early_stopping_rounds=0)
        assert captured[key] == expected
