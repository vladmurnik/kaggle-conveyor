"""Train/test adversarial-validation diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kconv.diagnostics import run_adversarial


def _frames(n_train: int = 600, n_test: int = 300, shifted: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(31)
    train = pd.DataFrame(
        {
            "x0": rng.normal(size=n_train),
            "x1": rng.normal(size=n_train),
            "cat": rng.choice(["a", "b", "c"], n_train),
        }
    )
    test = pd.DataFrame(
        {
            "x0": rng.normal(size=n_test),
            "x1": rng.normal(size=n_test),
            "cat": rng.choice(["a", "b", "c"], n_test),
        }
    )
    if shifted:
        test["x0"] += 2.0
        test["x1"] += 1.0
    return train, test


def test_same_distribution_has_no_strong_signal() -> None:
    train, test = _frames()
    report = run_adversarial(train, test, ["x0", "x1", "cat"], folds=3, threshold=0.75, seed=42)

    assert report["status"] == "ok"
    assert report["auc"] is not None
    assert report["auc"] < 0.75
    assert report["warning"] is False
    assert report["features"] == 3
    assert report["folds"] == 3


def test_shifted_distribution_triggers_warning() -> None:
    train, test = _frames(shifted=True)
    report = run_adversarial(train, test, ["x0", "x1", "cat"], folds=3, threshold=0.75, seed=42)

    assert report["status"] == "ok"
    assert report["auc"] is not None
    assert report["auc"] >= 0.75
    assert report["warning"] is True
    assert "distributions differ" in report["message"]


def test_missing_test_data_is_skipped() -> None:
    train, _ = _frames()
    report = run_adversarial(train, pd.DataFrame(), ["x0", "x1", "cat"])

    assert report["status"] == "skipped"
    assert report["warning"] is False
    assert "empty" in report["message"]


def test_datetime_and_text_columns_are_encoded() -> None:
    train = pd.DataFrame(
        {
            "when": pd.date_range("2026-01-01", periods=120, freq="h"),
            "label": ["alpha"] * 60 + ["beta"] * 60,
            "value": np.linspace(0.0, 1.0, 120),
        }
    )
    test = train.copy()
    report = run_adversarial(train, test, ["when", "label", "value"], folds=2, threshold=0.75, seed=42)

    assert report["status"] == "ok"
    assert report["features"] == 3
