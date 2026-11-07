"""Optional TabPFN adapter tests.

The whole module is skipped when the ``tabpfn`` extra is not installed, so the
base CI job stays lightweight.  Run it locally with ``uv sync --extra tabpfn``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tabpfn")
pytest.importorskip("torch")

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine  # noqa: E402

from kconv.errors import ModelError  # noqa: E402
from kconv.models import create_model  # noqa: E402


def _binary() -> tuple[pd.DataFrame, np.ndarray]:
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame.drop(columns=["target"])
    return frame.iloc[:450].reset_index(drop=True), bunch.target.to_numpy()[:450]


def test_binary_cpu_fit_predict_and_pickle(tmp_path) -> None:
    X, y = _binary()
    model = create_model("tabpfn")
    model.task = "binary"
    model.fit(X.iloc[:380], y[:380], params={"device": "cpu", "n_estimators": 1}, seed=7)

    proba = model.predict_proba(X.iloc[380:])
    labels = model.predict(X.iloc[380:])
    assert proba.shape == (70,)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()
    assert labels.shape == (70,)
    assert model.device_ == "cpu"
    assert model.peak_vram_bytes_ == 0
    assert model.importance() is None

    path = tmp_path / "tabpfn.joblib"
    joblib.dump(model, path, compress=3)
    loaded = joblib.load(path)
    assert loaded.task == "binary"
    assert np.allclose(loaded.predict_proba(X.iloc[380:]), proba, atol=1e-6)


def test_regression_and_multiclass_cpu() -> None:
    diabetes = load_diabetes(as_frame=True)
    reg = create_model("tabpfn")
    reg.task = "regression"
    reg.fit(
        diabetes.frame.drop(columns=["target"]).iloc[:350].reset_index(drop=True),
        diabetes.target.to_numpy()[:350],
        params={"device": "cpu", "n_estimators": 1},
        seed=7,
    )
    prediction = reg.predict(diabetes.frame.drop(columns=["target"]).iloc[350:].reset_index(drop=True))
    assert prediction.shape == (92,)
    assert np.isfinite(prediction).all()
    with pytest.raises(AttributeError):
        reg.predict_proba(diabetes.frame.drop(columns=["target"]).iloc[350:])

    wine = load_wine(as_frame=True)
    multi = create_model("tabpfn")
    multi.task = "multiclass"
    multi.fit(
        wine.frame.drop(columns=["target"]).iloc[:140].reset_index(drop=True),
        wine.target.to_numpy()[:140],
        params={"device": "cpu", "n_estimators": 1},
        seed=7,
    )
    probabilities = multi.predict_proba(wine.frame.drop(columns=["target"]).iloc[140:].reset_index(drop=True))
    assert probabilities.shape == (38, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert multi.predict(wine.frame.drop(columns=["target"]).iloc[140:]).shape == (38,)


def test_limits_fail_before_download(monkeypatch) -> None:
    X = pd.DataFrame(np.arange(60, dtype=float).reshape(12, 5), columns=list("abcde"))
    y = np.array([0, 1] * 6)
    model = create_model("tabpfn")
    model.task = "binary"

    with pytest.raises(ModelError, match="configured limit"):
        model.fit(X, y, params={"device": "cpu", "max_train_rows": 10})
    with pytest.raises(ModelError, match="configured limit"):
        model.fit(X, y, params={"device": "cpu", "max_features": 4})

    y_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    with pytest.raises(ModelError, match="supports at most"):
        model.fit(X, y_multi, params={"device": "cpu", "max_classes": 2})


def test_release_clears_cuda_cache_even_without_a_fitted_model(monkeypatch) -> None:
    calls: list[bool] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: calls.append(True),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = create_model("tabpfn")
    model.model = None

    model._release()

    assert calls == [True]


def test_unknown_version_is_actionable() -> None:
    X, y = _binary()
    model = create_model("tabpfn")
    model.task = "binary"
    with pytest.raises(ModelError, match="Unknown TabPFN model version"):
        model.fit(X.iloc[:20], y[:20], params={"device": "cpu", "model_version": "v9"})


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA device is not available",
)
def test_cuda_auto_uses_gpu() -> None:
    X, y = _binary()
    model = create_model("tabpfn")
    model.task = "binary"
    model.fit(X.iloc[:80], y[:80], params={"n_estimators": 1}, seed=7)
    assert model.device_ == "cuda"
    assert model.peak_vram_bytes_ > 0
