"""Unit tests for the advanced fold-safe feature transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kconv.errors import DataError
from kconv.features.anomaly import AnomalyFeatureTransformer
from kconv.features.clustering import ClusterFeatureTransformer
from kconv.features.numeric_interactions import NumericInteractionTransformer
from kconv.features.time_series import TimeSeriesFeatureTransformer

# ----------------------------------------------------------------------
# numeric interactions
# ----------------------------------------------------------------------


def test_numeric_interactions_all_operations_and_clipping() -> None:
    frame = pd.DataFrame({"a": [2.0, 0.0, 5.0], "b": [4.0, 0.0, 0.0]})
    transformer = NumericInteractionTransformer(
        columns=["a", "b"],
        max_pairs=1,
        operations=["product", "ratio", "difference", "sum"],
        clip_value=3.0,
    )
    result = transformer.fit_transform(frame)

    assert list(result.columns) == [
        "a__x__b__product",
        "a__x__b__ratio",
        "a__x__b__diff",
        "a__x__b__sum",
    ]
    assert list(transformer.feature_names) == list(result.columns)
    np.testing.assert_allclose(result["a__x__b__product"], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(result["a__x__b__ratio"], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(result["a__x__b__diff"], [-2.0, 0.0, 3.0])
    np.testing.assert_allclose(result["a__x__b__sum"], [3.0, 0.0, 3.0])
    assert np.isfinite(result.to_numpy()).all()


def test_numeric_interactions_auto_columns_pair_cap_and_empty_frame() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
            "c": [5.0, 6.0],
            "cat": ["x", "y"],
        }
    )
    transformer = NumericInteractionTransformer(max_pairs=2)
    result = transformer.fit_transform(frame)

    assert list(result.columns) == [
        "a__x__b__product",
        "a__x__b__ratio",
        "a__x__c__product",
        "a__x__c__ratio",
    ]
    empty = transformer.transform(frame.iloc[0:0])
    assert list(empty.columns) == list(transformer.feature_names)
    assert empty.empty


def test_numeric_interactions_reject_bad_input() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "cat": ["x", "y"]})
    with pytest.raises(DataError, match="before fit"):
        NumericInteractionTransformer(columns=["a", "b"]).transform(frame)
    with pytest.raises(DataError, match="not numeric"):
        NumericInteractionTransformer(columns=["a", "cat"]).fit(frame)
    with pytest.raises(DataError, match="at least two"):
        NumericInteractionTransformer(columns=["a"]).fit(frame)
    with pytest.raises(ValueError, match="max_pairs"):
        NumericInteractionTransformer(max_pairs=0)
    with pytest.raises(ValueError, match="operation"):
        NumericInteractionTransformer(operations=["magic"])
    transformer = NumericInteractionTransformer(columns=["a", "b"]).fit(frame)
    with pytest.raises(DataError, match="not present"):
        transformer.transform(frame.rename(columns={"b": "c"}))


# ----------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------


def _cluster_frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n) * 2.0,
            "cat": rng.choice(["a", "b", "c"], n),
        },
        index=pd.Index([f"r{index}" for index in range(n)]),
    )


def test_cluster_features_shape_names_finite_and_index() -> None:
    frame = _cluster_frame()
    transformer = ClusterFeatureTransformer(n_clusters=3)
    result = transformer.fit_transform(frame)

    assert list(result.columns) == [
        "cluster__dist_0",
        "cluster__dist_1",
        "cluster__dist_2",
        "cluster__label",
    ]
    assert list(result.index) == list(frame.index)
    assert np.isfinite(result.to_numpy()).all()
    assert set(result["cluster__label"]).issubset({0.0, 1.0, 2.0})


def test_cluster_features_are_deterministic_and_shift_invariant() -> None:
    frame = _cluster_frame()
    first = ClusterFeatureTransformer(n_clusters=4).fit_transform(frame)
    second = ClusterFeatureTransformer(n_clusters=4).fit_transform(frame)
    pd.testing.assert_frame_equal(first, second)

    shifted = frame.copy()
    shifted[["x", "y"]] += 100.0
    shifted_result = ClusterFeatureTransformer(columns=["x", "y"], n_clusters=4).fit_transform(
        shifted
    )
    pd.testing.assert_frame_equal(first, shifted_result)


def test_cluster_features_explicit_columns_empty_and_tiny_frame() -> None:
    frame = _cluster_frame(12)
    transformer = ClusterFeatureTransformer(columns=["x"], n_clusters=5)
    result = transformer.fit_transform(frame)
    assert list(result.columns) == [
        "cluster__dist_0",
        "cluster__dist_1",
        "cluster__dist_2",
        "cluster__dist_3",
        "cluster__dist_4",
        "cluster__label",
    ]
    empty = transformer.transform(frame.iloc[0:0])
    assert empty.empty and list(empty.columns) == list(transformer.feature_names)

    tiny = ClusterFeatureTransformer(n_clusters=8)
    tiny_result = tiny.fit_transform(frame.iloc[:3])
    assert list(tiny_result.columns) == [
        "cluster__dist_0",
        "cluster__dist_1",
        "cluster__dist_2",
        "cluster__label",
    ]


def test_cluster_features_reject_bad_input() -> None:
    frame = _cluster_frame(10)
    with pytest.raises(DataError, match="before fit"):
        ClusterFeatureTransformer().transform(frame)
    with pytest.raises(DataError, match="not present"):
        ClusterFeatureTransformer(columns=["missing"]).fit(frame)
    with pytest.raises(DataError, match="convert"):
        ClusterFeatureTransformer(columns=["cat"]).fit(frame)
    with pytest.raises(ValueError, match="n_clusters"):
        ClusterFeatureTransformer(n_clusters=1)
    with pytest.raises(ValueError, match="include"):
        ClusterFeatureTransformer(include_distances=False, include_labels=False)
    transformer = ClusterFeatureTransformer(columns=["x"]).fit(frame)
    with pytest.raises(DataError, match="not present"):
        transformer.transform(frame.rename(columns={"x": "z"}))


# ----------------------------------------------------------------------
# anomaly
# ----------------------------------------------------------------------


def test_anomaly_features_names_finite_flag_and_index() -> None:
    frame = pd.DataFrame(
        {"x": np.linspace(0.0, 1.0, 30), "y": np.linspace(1.0, 2.0, 30)},
        index=pd.Index([f"r{index}" for index in range(30)]),
    )
    transformer = AnomalyFeatureTransformer(n_estimators=20)
    result = transformer.fit_transform(frame)

    assert list(result.columns) == ["anomaly__score", "anomaly__flag"]
    assert list(result.index) == list(frame.index)
    assert np.isfinite(result.to_numpy()).all()
    assert set(result["anomaly__flag"]).issubset({0.0, 1.0})


def test_anomaly_clear_outlier_has_highest_score_and_deterministic() -> None:
    frame = pd.DataFrame({"x": np.concatenate([np.zeros(30), [50.0]])})
    transformer = AnomalyFeatureTransformer(
        n_estimators=30, contamination=0.1, random_state=7
    )
    result = transformer.fit_transform(frame)
    assert result["anomaly__score"].iloc[-1] == result["anomaly__score"].max()
    assert result["anomaly__flag"].iloc[-1] == 1.0

    again = AnomalyFeatureTransformer(
        n_estimators=30, contamination=0.1, random_state=7
    ).fit_transform(frame)
    pd.testing.assert_frame_equal(result, again)


def test_anomaly_features_explicit_columns_empty_and_bad_input() -> None:
    frame = pd.DataFrame({"x": np.arange(20, dtype=float), "y": np.arange(20, dtype=float)})
    transformer = AnomalyFeatureTransformer(columns=["x"], n_estimators=10)
    result = transformer.fit_transform(frame)
    assert list(result.columns) == ["anomaly__score", "anomaly__flag"]
    empty = transformer.transform(frame.iloc[0:0])
    assert empty.empty and list(empty.columns) == list(transformer.feature_names)

    with pytest.raises(DataError, match="before fit"):
        AnomalyFeatureTransformer().transform(frame)
    with pytest.raises(DataError, match="not present"):
        AnomalyFeatureTransformer(columns=["missing"]).fit(frame)
    with pytest.raises(ValueError, match="n_estimators"):
        AnomalyFeatureTransformer(n_estimators=1)
    with pytest.raises(ValueError, match="contamination"):
        AnomalyFeatureTransformer(contamination=0.8)
    transformer = AnomalyFeatureTransformer(columns=["x"]).fit(frame)
    with pytest.raises(DataError, match="not present"):
        transformer.transform(frame.rename(columns={"x": "z"}))


# ----------------------------------------------------------------------
# time series
# ----------------------------------------------------------------------


def _time_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "when": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
            "g": ["a", "a", "a", "b", "b", "b"],
            "v": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        },
        index=pd.Index([4, 5, 6, 1, 2, 3]),
    )


def test_time_series_lags_rolling_elapsed_and_diff() -> None:
    frame = _time_frame()
    transformer = TimeSeriesFeatureTransformer(
        time_col="when",
        group_col="g",
        columns=["v"],
        lags=[1],
        windows=[2],
        stats=["mean", "max"],
        include_elapsed=True,
        include_diff=True,
    )
    result = transformer.fit_transform(frame)

    assert list(result.columns) == [
        "ts__v__lag_1",
        "ts__v__mean_2",
        "ts__v__max_2",
        "ts__elapsed_seconds",
        "ts__diff_seconds",
    ]
    assert list(result.index) == list(frame.index)
    np.testing.assert_allclose(result["ts__v__lag_1"], [0.0, 1.0, 2.0, 0.0, 10.0, 20.0])
    np.testing.assert_allclose(result["ts__v__mean_2"], [0.0, 1.0, 1.5, 0.0, 10.0, 15.0])
    np.testing.assert_allclose(result["ts__v__max_2"], [0.0, 1.0, 2.0, 0.0, 10.0, 20.0])
    np.testing.assert_allclose(
        result["ts__elapsed_seconds"],
        [0.0, 86400.0, 172800.0, 0.0, 86400.0, 172800.0],
    )
    np.testing.assert_allclose(
        result["ts__diff_seconds"],
        [0.0, 86400.0, 86400.0, 0.0, 86400.0, 86400.0],
    )
    assert np.isfinite(result.to_numpy()).all()


def test_time_series_transform_uses_fitted_history_without_future() -> None:
    train = _time_frame().iloc[[0, 1, 3, 4]].copy()
    test = pd.DataFrame(
        {
            "when": ["2024-01-03", "2024-01-03"],
            "g": ["a", "b"],
            "v": [3.0, 30.0],
        },
        index=pd.Index([20, 21]),
    )
    transformer = TimeSeriesFeatureTransformer(
        time_col="when",
        group_col="g",
        columns=["v"],
        lags=[1],
        windows=[2],
        stats=["mean", "max"],
    )
    transformer.fit(train)
    result = transformer.transform(test)

    assert list(result.index) == [20, 21]
    np.testing.assert_allclose(result["ts__v__lag_1"], [2.0, 20.0])
    np.testing.assert_allclose(result["ts__v__mean_2"], [1.5, 15.0])
    np.testing.assert_allclose(result["ts__v__max_2"], [2.0, 20.0])
    np.testing.assert_allclose(result["ts__elapsed_seconds"], [172800.0, 172800.0])
    np.testing.assert_allclose(result["ts__diff_seconds"], [86400.0, 86400.0])


def test_time_series_empty_and_bad_input() -> None:
    frame = _time_frame()
    with pytest.raises(DataError, match="before fit"):
        TimeSeriesFeatureTransformer(time_col="when", columns=["v"]).transform(frame)
    with pytest.raises(DataError, match="invalid or missing"):
        TimeSeriesFeatureTransformer(time_col="when", columns=["v"]).fit(
            frame.assign(when=["bad"] + list(frame["when"].iloc[1:]))
        )
    with pytest.raises(DataError, match="not present"):
        TimeSeriesFeatureTransformer(time_col="missing", columns=["v"]).fit(frame)
    with pytest.raises(ValueError, match="lags"):
        TimeSeriesFeatureTransformer(time_col="when", lags=[0])
    with pytest.raises(ValueError, match="windows"):
        TimeSeriesFeatureTransformer(time_col="when", windows=[1])
    transformer = TimeSeriesFeatureTransformer(time_col="when", columns=["v"]).fit(frame)
    empty = transformer.transform(frame.iloc[0:0])
    assert empty.empty and list(empty.columns) == list(transformer.feature_names)
    with pytest.raises(DataError, match="not numeric"):
        TimeSeriesFeatureTransformer(time_col="when", columns=["g"]).fit(frame)
