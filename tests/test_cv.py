"""Cross-validation contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kconv.config import Config, DataConfig, ValidationConfig
from kconv.cv.splitters import build_folds, choose_strategy, load_folds, save_folds
from kconv.errors import DataError


def _config(task: str = "binary", **validation) -> Config:
    return Config(
        project="cv",
        task=task,
        metric="roc_auc" if task != "regression" else "rmse",
        data=DataConfig(train="train.csv", target="y"),
        validation=ValidationConfig(n_splits=3, **validation),
    )


def test_stratified_folds_cover_every_row_once() -> None:
    y = np.array([0, 1] * 15)
    folds = build_folds(_config(), y)
    seen = np.concatenate([fold.valid_idx for fold in folds])
    assert len(folds) == 3
    assert sorted(seen.tolist()) == list(range(len(y)))
    assert len(seen) == len(set(seen.tolist()))


def test_auto_strategy_classification_and_regression() -> None:
    assert choose_strategy(_config(), np.array([0, 1] * 5)) == "stratified_kfold"
    assert choose_strategy(_config("regression"), np.linspace(0, 1, 20)) == "kfold"


def test_resolved_task_stratifies_auto_cv() -> None:
    y = np.array([0, 1] * 15)
    config = _config(task="auto")
    assert choose_strategy(config, y, task="binary") == "stratified_kfold"
    folds = build_folds(config, y, task="binary")
    for fold in folds:
        assert set(y[fold.train_idx]) == {0, 1}
        assert set(y[fold.valid_idx]) == {0, 1}


def test_group_split_has_no_group_overlap() -> None:
    groups = np.repeat(np.arange(10), 3)
    y = np.array([0, 1, 0] * 10)
    config = _config(group_col="group")
    folds = build_folds(config, y, groups=groups)
    for fold in folds:
        assert set(groups[fold.train_idx]).isdisjoint(set(groups[fold.valid_idx]))


def test_holdout_and_save_load(tmp_path) -> None:
    y = np.array([0, 1] * 10)
    config = _config(strategy="holdout", holdout_size=0.25)
    folds = build_folds(config, y)
    assert len(folds) == 1
    path = tmp_path / "folds.parquet"
    save_folds(folds, path, len(y))
    loaded = load_folds(path)
    assert len(loaded) == 1
    assert sorted(loaded[0].valid_idx.tolist()) == sorted(folds[0].valid_idx.tolist())
    assert len(loaded[0].train_idx) == 15
    frame = pd.read_parquet(path)
    assert set(frame.columns) == {"row_id", "fold", "n_rows"}


def test_load_folds_rejects_train_valid_overlap(tmp_path, monkeypatch) -> None:
    from kconv.cv import splitters

    y = np.array([0, 1] * 10)
    config = _config(strategy="holdout", holdout_size=0.25)
    folds = build_folds(config, y)
    path = tmp_path / "folds.parquet"
    save_folds(folds, path, len(y))
    overlap = folds[0].valid_idx[:1].tolist()
    monkeypatch.setattr(
        splitters,
        "_fold_membership",
        lambda path, table: {"0": overlap},
    )

    with pytest.raises(DataError, match="overlap"):
        load_folds(path)


def test_time_series_respects_order() -> None:
    n = 30
    y = np.arange(n, dtype=float)
    times = np.arange(n)
    config = _config("regression", strategy="time_series", gap=1)
    folds = build_folds(config, y, time=times)
    assert folds
    for fold in folds:
        assert fold.train_idx.max() < fold.valid_idx.min()


def test_group_folds_survive_save_load(tmp_path) -> None:
    groups = np.repeat(np.arange(8), 4)
    y = np.tile([0, 1], 16)
    config = _config(group_col="group")
    folds = build_folds(config, y, groups=groups)
    path = tmp_path / "folds.parquet"
    save_folds(folds, path, len(y))
    loaded = load_folds(path)
    assert len(loaded) == len(folds)
    for before, after in zip(folds, loaded, strict=True):
        np.testing.assert_array_equal(after.train_idx, before.train_idx)
        np.testing.assert_array_equal(after.valid_idx, before.valid_idx)
        assert set(groups[after.train_idx]).isdisjoint(set(groups[after.valid_idx]))


def test_time_series_survives_save_load_without_future_leakage(tmp_path) -> None:
    n = 40
    y = np.arange(n, dtype=float)
    # Deliberately unsorted so the splitter must order rows itself.
    times = np.array([(i * 7) % n for i in range(n)])
    config = _config("regression", strategy="time_series", gap=2)
    folds = build_folds(config, y, time=times)
    path = tmp_path / "folds.parquet"
    save_folds(folds, path, len(y))
    loaded = load_folds(path)
    assert len(loaded) == len(folds)
    for before, after in zip(folds, loaded, strict=True):
        np.testing.assert_array_equal(after.train_idx, before.train_idx)
        np.testing.assert_array_equal(after.valid_idx, before.valid_idx)
        assert times[after.train_idx].max() < times[after.valid_idx].min()
        assert times[after.valid_idx].min() - times[after.train_idx].max() >= 3


def test_auto_selects_group_or_time_strategy() -> None:
    y = np.array([0, 1] * 10)
    assert choose_strategy(_config(group_col="group"), y) == "group_kfold"
    assert choose_strategy(_config(time_col="time"), y) == "time_series"


def test_group_split_rejects_single_group() -> None:
    y = np.array([0, 1] * 10)
    with pytest.raises(DataError, match="two distinct groups"):
        build_folds(_config(group_col="group"), y, groups=np.zeros(len(y), dtype=int))


def test_stratified_rejects_single_class() -> None:
    y = np.zeros(20, dtype=int)
    with pytest.raises(DataError, match="two target classes"):
        build_folds(_config(), y)


def test_build_folds_rejects_tiny_dataset() -> None:
    y = np.array([0, 1, 0])
    with pytest.raises(DataError, match="at least 4"):
        build_folds(_config(), y)


def test_time_series_rejects_too_large_gap() -> None:
    y = np.arange(12, dtype=float)
    config = _config("regression", strategy="time_series", gap=20)
    with pytest.raises(DataError, match="time_series CV"):
        build_folds(config, y, time=np.arange(len(y)))
