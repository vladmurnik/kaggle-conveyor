"""Cheap unusual-pattern feature prototypes: rank, modulo and row statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kconv.errors import KconvError
from kconv.features.unusual import (
    GaussianRankTransformer,
    add_modulo_features,
    add_row_statistics,
)


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1.0, 2.0, 3.0, 4.0],
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "label": ["x", "y", "x", "y"],
        }
    )


def test_gaussian_rank_is_fitted_on_train_and_applied_to_test(frame: pd.DataFrame) -> None:
    train = frame.iloc[:3].copy()
    test = pd.DataFrame({"a": [-100.0, 2.5, 100.0], "b": [5.0, 25.0, 500.0]})
    transformer = GaussianRankTransformer(["a", "b"]).fit(train)
    transformed = transformer.transform(test)

    assert {"a__gauss_rank", "b__gauss_rank"} <= set(transformed.columns)
    assert np.isfinite(transformed[["a__gauss_rank", "b__gauss_rank"]].to_numpy()).all()
    # Out-of-range test values remain finite and ordered.
    assert transformed["a__gauss_rank"].is_monotonic_increasing
    assert transformed["a__gauss_rank"].iloc[0] < transformed["a__gauss_rank"].iloc[2]


def test_gaussian_rank_does_not_rank_train_and_test_jointly(frame: pd.DataFrame) -> None:
    train = frame.iloc[:3]
    test = pd.DataFrame({"a": [-100.0, 2.5, 100.0]})
    transformer = GaussianRankTransformer(["a"]).fit(train)
    transformed = transformer.transform(test)
    alone = transformer.transform(pd.DataFrame({"a": [2.5]}))
    assert transformed["a__gauss_rank"].iloc[1] == pytest.approx(alone["a__gauss_rank"].iloc[0])
    bigger = pd.DataFrame({"a": [1000.0]})
    assert np.isfinite(transformer.transform(bigger)["a__gauss_rank"]).all()


def test_gaussian_rank_handles_missing_values(frame: pd.DataFrame) -> None:
    tensor = frame.copy()
    tensor.loc[1, "a"] = np.nan
    out = GaussianRankTransformer(["a"]).fit_transform(tensor)
    assert np.isfinite(out["a__gauss_rank"]).all()
    assert out.loc[1, "a__gauss_rank"] == pytest.approx(0.0)


def test_gaussian_rank_rejects_unknown_column_and_unfitted_transform(frame: pd.DataFrame) -> None:
    with pytest.raises(KconvError, match="not present"):
        GaussianRankTransformer(["missing"]).fit(frame)
    with pytest.raises(KconvError, match="before fit"):
        GaussianRankTransformer(["a"]).transform(frame)
    with pytest.raises(KconvError, match="No numeric columns"):
        GaussianRankTransformer().fit(pd.DataFrame({"text": ["a", "b"]}))


def test_modulo_features_are_deterministic_and_non_destructive(frame: pd.DataFrame) -> None:
    before = frame.copy()
    out = add_modulo_features(frame, ["a"], moduli=(2,), steps=(2,))
    pd.testing.assert_frame_equal(frame, before)
    assert out["a__mod_2"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert out["a__div_2"].tolist() == [0.0, 1.0, 1.0, 2.0]


def test_modulo_features_validate_inputs(frame: pd.DataFrame) -> None:
    with pytest.raises(KconvError, match="positive"):
        add_modulo_features(frame, ["a"], moduli=(0,), steps=())
    with pytest.raises(KconvError, match="At least one"):
        add_modulo_features(frame, ["a"], moduli=(), steps=())
    with pytest.raises(KconvError, match="not present"):
        add_modulo_features(frame, ["missing"])


def test_row_statistics_are_added_without_modifying_input(frame: pd.DataFrame) -> None:
    before = frame.copy()
    out = add_row_statistics(frame, ["a", "b"])
    pd.testing.assert_frame_equal(frame, before)
    assert out["row_mean"].tolist() == pytest.approx([5.5, 11.0, 16.5, 22.0])
    assert out["row_min"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert out["row_max"].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert (out["row_std"] >= 0).all()


def test_row_statistics_support_median_and_single_column(frame: pd.DataFrame) -> None:
    out = add_row_statistics(frame, ["a"], stats=("median",))
    assert out["row_median"].tolist() == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(KconvError, match="Unknown row statistic"):
        add_row_statistics(frame, ["a"], stats=("mode",))
    with pytest.raises(KconvError, match="At least one"):
        add_row_statistics(frame, ["a"], stats=())


def test_row_statistics_auto_select_numeric_columns(frame: pd.DataFrame) -> None:
    out = add_row_statistics(frame, stats=("mean",))
    assert {"row_mean"} <= set(out.columns)
    assert out["row_mean"].notna().all()


def test_prototypes_support_integer_column_labels() -> None:
    frame = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [10.0, 20.0, 30.0]})
    rank = GaussianRankTransformer([0]).fit_transform(frame)
    assert {"0__gauss_rank", "1__gauss_rank"} <= set(
        GaussianRankTransformer().fit_transform(frame).columns
    )
    assert rank["0__gauss_rank"].is_monotonic_increasing

    modulo = add_modulo_features(frame, [0], moduli=(2,), steps=())
    assert modulo["0__mod_2"].tolist() == [1.0, 0.0, 1.0]

    row = add_row_statistics(frame, stats=("mean",))
    assert row["row_mean"].tolist() == pytest.approx([5.5, 11.0, 16.5])
