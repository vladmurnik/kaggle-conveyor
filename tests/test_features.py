"""Feature engineering tests, including target-encoding leakage checks."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from kconv.config import (
    Config,
    DataConfig,
    DatetimeCyclicalConfig,
    FeatureBlockConfig,
    FeatureConfig,
    InteractionConfig,
    KNNFeatureConfig,
    MissingnessConfig,
)
from kconv.contracts import DatasetSpec
from kconv.errors import DataError
from kconv.features.knn import KNNClassDistanceTransformer
from kconv.features.pipeline import FeatureBuilder


def _spec(**kwargs) -> DatasetSpec:
    base = dict(
        train=Path("train.csv"),
        test=None,
        sample_submission=None,
        target="y",
        id_col="id",
        task="binary",
        metric="roc_auc",
        feature_columns=["num", "cat", "when", "text"],
        numeric_columns=["num"],
        categorical_columns=["cat"],
        datetime_columns=["when"],
        text_columns=["text"],
    )
    base.update(kwargs)
    return DatasetSpec(**base)


def _frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "num": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], n),
            "when": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "text": ["short text " * (i % 3 + 1) for i in range(n)],
            "y": (rng.random(n) > 0.5).astype(int),
        }
    )


def test_feature_builder_numeric_categorical_datetime_text() -> None:
    config = Config(project="f", task="binary", metric="roc_auc", data=DataConfig(train="x", target="y"))
    fb = FeatureBuilder(config, _spec(), "tree")
    df = _frame()
    train = fb.fit_transform(df.iloc[:30], df["y"].iloc[:30].to_numpy())
    valid = fb.transform(df.iloc[30:])
    assert train.shape[0] == 30
    assert valid.shape[0] == 10
    assert list(valid.columns) == fb.feature_names
    assert "y" not in train.columns and "id" not in train.columns
    assert any("cat" in c for c in train.columns)


def test_linear_family_one_hot_and_scale() -> None:
    config = Config(project="f", task="binary", metric="roc_auc", data=DataConfig(train="x", target="y"))
    fb = FeatureBuilder(config, _spec(text_columns=[], datetime_columns=[]), "linear")
    df = _frame()
    train = fb.fit_transform(df.iloc[:30], df["y"].iloc[:30].to_numpy())
    assert train.select_dtypes(include=[np.number]).shape[1] == train.shape[1]
    assert any(c.startswith("cat__") for c in train.columns)


def test_target_encoding_does_not_use_own_row_target() -> None:
    n = 30
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "cat": [f"c{i}" for i in range(n)],
            "num": np.arange(n, dtype=float),
            "y": np.array([0, 1] * (n // 2)),
        }
    )
    spec = _spec(
        feature_columns=["num", "cat"],
        numeric_columns=["num"],
        categorical_columns=["cat"],
        datetime_columns=[],
        text_columns=[],
    )
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(target_encode=["cat"], drop_constant=False),
    )
    first = FeatureBuilder(config, spec, "tree").fit_transform(df, df["y"].to_numpy())
    changed = df.copy()
    changed.loc[0, "y"] = 1 - changed.loc[0, "y"]
    second = FeatureBuilder(config, spec, "tree").fit_transform(changed, changed["y"].to_numpy())
    assert first["cat__target"].iloc[0] == second["cat__target"].iloc[0]
    assert first.shape[1] == second.shape[1]


def test_target_encoding_tiny_fold_excludes_own_row_target() -> None:
    n = 6
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "cat": [f"c{i}" for i in range(n)],
            "num": np.arange(n, dtype=float),
            "y": np.array([0, 1, 0, 1, 0, 1]),
        }
    )
    spec = _spec(
        feature_columns=["num", "cat"],
        numeric_columns=["num"],
        categorical_columns=["cat"],
        datetime_columns=[],
        text_columns=[],
    )
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(target_encode=["cat"], drop_constant=False),
    )
    first = FeatureBuilder(config, spec, "tree").fit_transform(df, df["y"].to_numpy())
    changed = df.copy()
    changed.loc[0, "y"] = 1 - changed.loc[0, "y"]
    second = FeatureBuilder(config, spec, "tree").fit_transform(changed, changed["y"].to_numpy())
    assert first["cat__target"].iloc[0] == second["cat__target"].iloc[0]


def test_missing_values_are_imputed_in_transform() -> None:
    config = Config(project="f", task="binary", metric="roc_auc", data=DataConfig(train="x", target="y"))
    df = _frame()
    df.loc[0, "num"] = np.nan
    df.loc[1, "cat"] = None
    fb = FeatureBuilder(config, _spec(), "tree")
    train = fb.fit_transform(df.iloc[:30], df["y"].iloc[:30].to_numpy())
    valid = fb.transform(df.iloc[30:])
    assert not train.isna().any().any()
    assert not valid.isna().any().any()


def test_infinite_numeric_values_are_treated_as_missing() -> None:
    config = Config(project="f", task="binary", metric="roc_auc", data=DataConfig(train="x", target="y"))
    df = _frame()
    df.loc[0, "num"] = np.inf
    df.loc[1, "num"] = -np.inf
    fb = FeatureBuilder(config, _spec(), "tree")
    train = fb.fit_transform(df.iloc[:30], df["y"].iloc[:30].to_numpy())
    valid = fb.transform(df.iloc[30:])
    assert np.isfinite(train.to_numpy()).all()
    assert np.isfinite(valid.to_numpy()).all()


def test_all_constant_features_raise_a_clear_data_error() -> None:
    spec = _spec(
        feature_columns=["constant"],
        numeric_columns=["constant"],
        categorical_columns=[],
        datetime_columns=[],
        text_columns=[],
    )
    frame = pd.DataFrame(
        {
            "id": np.arange(20),
            "constant": np.ones(20),
            "y": np.array([0, 1] * 10),
        }
    )
    config = Config(project="f", task="binary", metric="roc_auc", data=DataConfig(train="x", target="y"))
    with pytest.raises(DataError, match="constant"):
        FeatureBuilder(config, spec, "tree").fit_transform(frame, frame["y"].to_numpy())


def test_knn_class_distance_is_fit_in_fold_and_excludes_self() -> None:
    frame = pd.DataFrame({"a": np.arange(12, dtype=float), "b": np.zeros(12)})
    target = np.array([0] * 6 + [1] * 6)
    transformer = KNNClassDistanceTransformer(["a"], neighbors=1, folds=3)

    oof = transformer.fit_transform_oof(frame, target)
    full = transformer.transform(frame)

    assert list(oof.columns) == ["knn_c0_k1", "knn_c1_k1"]
    assert full["knn_c0_k1"].iloc[0] == 0.0
    own_class_oof = np.where(
        target == 0,
        oof["knn_c0_k1"].to_numpy(),
        oof["knn_c1_k1"].to_numpy(),
    )
    # A training row never contributes itself to its own OOF distance.
    assert (own_class_oof > 0).all()

    probe = pd.DataFrame({"a": [100.0, -100.0], "b": [0.0, 0.0]})
    first = transformer.transform(probe)
    second = transformer.transform(probe)
    pd.testing.assert_frame_equal(first, second)
    assert np.isfinite(first.to_numpy()).all()
    assert first["knn_c1_k1"].iloc[0] < first["knn_c0_k1"].iloc[0]


def test_knn_class_distance_requires_two_rows_per_class() -> None:
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    target = np.array([0, 1, 1])
    with pytest.raises(DataError, match="at least two rows per class"):
        KNNClassDistanceTransformer(["a"], neighbors=1, folds=2).fit_transform_oof(frame, target)


def test_feature_builder_integrates_knn_features() -> None:
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            knn_features=KNNFeatureConfig(enabled=True, neighbors=2, folds=3),
            drop_constant=False,
        ),
    )
    spec = _spec(
        feature_columns=["num", "cat"],
        numeric_columns=["num"],
        categorical_columns=["cat"],
        datetime_columns=[],
        text_columns=[],
    )
    df = _frame(40)
    target = np.tile([0, 1], 20)
    fb = FeatureBuilder(config, spec, "tree")
    train = fb.fit_transform(df.iloc[:30], target[:30])
    valid = fb.transform(df.iloc[30:])

    assert any(column.startswith("knn_c") for column in train.columns)
    assert list(valid.columns) == fb.feature_names
    assert not train.isna().any().any()
    assert not valid.isna().any().any()


def test_feature_builder_rejects_knn_for_regression() -> None:
    config = Config(
        project="f",
        task="regression",
        metric="rmse",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(knn_features=KNNFeatureConfig(enabled=True)),
    )
    spec = _spec(task="regression", metric="rmse")
    with pytest.raises(DataError, match="only available for classification"):
        FeatureBuilder(config, spec, "tree")


def test_explicit_blocks_match_legacy_feature_matrix() -> None:
    spec = _spec(
        feature_columns=["num", "cat"],
        numeric_columns=["num"],
        categorical_columns=["cat"],
        datetime_columns=[],
        text_columns=[],
    )
    legacy = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            target_encode=["cat"],
            frequency_encode=["cat"],
            row_statistics=True,
            row_statistics_stats=["mean", "std", "min", "max"],
        ),
    )
    explicit = legacy.model_copy(deep=True)
    explicit.features.feature_blocks = [
        FeatureBlockConfig(type="base"),
        FeatureBlockConfig(type="target_encoding", params={"columns": ["cat"], "smoothing": 10}),
        FeatureBlockConfig(type="frequency_encoding", params={"columns": ["cat"]}),
        FeatureBlockConfig(type="row_statistics", params={"stats": ["mean", "std", "min", "max"]}),
    ]
    df = _frame(40)
    legacy_matrix = FeatureBuilder(legacy, spec, "tree").fit_transform(df, df["y"].to_numpy())
    explicit_matrix = FeatureBuilder(explicit, spec, "tree").fit_transform(df, df["y"].to_numpy())
    assert list(legacy_matrix.columns) == list(explicit_matrix.columns)
    np.testing.assert_allclose(legacy_matrix.to_numpy(), explicit_matrix.to_numpy(), rtol=0, atol=0)


def test_target_smoothing_is_controlled_by_block_params() -> None:
    spec = _spec(
        feature_columns=["num", "cat"],
        numeric_columns=["num"],
        categorical_columns=["cat"],
        datetime_columns=[],
        text_columns=[],
    )
    df = _frame(40)
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            feature_blocks=[
                FeatureBlockConfig(type="base"),
                FeatureBlockConfig(type="target_encoding", params={"columns": ["cat"], "smoothing": 1.0}),
            ],
            drop_constant=False,
        ),
    )
    small_smoothing = FeatureBuilder(config, spec, "tree").fit_transform(df, df["y"].to_numpy())
    config.features.feature_blocks[1].params["smoothing"] = 100.0
    large_smoothing = FeatureBuilder(config, spec, "tree").fit_transform(df, df["y"].to_numpy())
    assert not np.allclose(
        small_smoothing["cat__target"].to_numpy(),
        large_smoothing["cat__target"].to_numpy(),
    )


def test_feature_selection_variance_limits_the_matrix() -> None:
    spec = _spec()
    df = _frame(40)
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            feature_blocks=[
                FeatureBlockConfig(type="base"),
                FeatureBlockConfig(
                    type="feature_selection",
                    params={"method": "variance", "max_features": 2},
                ),
            ]
        ),
    )
    builder = FeatureBuilder(config, spec, "tree")
    train = builder.fit_transform(df.iloc[:30], df["y"].iloc[:30].to_numpy())
    valid = builder.transform(df.iloc[30:])

    assert train.shape[1] == 2
    assert list(valid.columns) == builder.feature_names
    assert not train.isna().any().any()
    assert not valid.isna().any().any()


def test_feature_selection_mutual_info_uses_outer_training_target() -> None:
    spec = _spec()
    df = _frame(40)
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            feature_blocks=[
                FeatureBlockConfig(type="base"),
                FeatureBlockConfig(
                    type="feature_selection",
                    params={"method": "mutual_info", "max_features": 3},
                ),
            ]
        ),
    )
    builder = FeatureBuilder(config, spec, "tree")
    train = builder.fit_transform(df.iloc[:30], df["y"].iloc[:30].to_numpy())
    valid = builder.transform(df.iloc[30:])

    assert train.shape[1] == 3
    assert list(valid.columns) == builder.feature_names
    assert not train.isna().any().any()


@pytest.mark.parametrize(
    ("task", "metric", "target"),
    [
        ("multiclass", "log_loss", np.tile([0, 1, 2], 14)[:40]),
        ("regression", "rmse", np.linspace(0.0, 1.0, 40)),
    ],
)
def test_feature_selection_mutual_info_supports_all_task_types(
    task: str, metric: str, target: np.ndarray
) -> None:
    spec = _spec(task=task, metric=metric)
    df = _frame(40).assign(y=target)
    config = Config(
        project="f",
        task=task,  # type: ignore[arg-type]
        metric=metric,
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            feature_blocks=[
                FeatureBlockConfig(type="base"),
                FeatureBlockConfig(
                    type="feature_selection",
                    params={"method": "mutual_info", "max_features": 2},
                ),
            ]
        ),
    )
    train = FeatureBuilder(config, spec, "tree").fit_transform(df, target)
    assert train.shape[1] == 2
    assert not train.isna().any().any()


@pytest.mark.parametrize(
    ("task", "metric", "target"),
    [
        ("binary", "roc_auc", np.tile([0, 1], 20)),
        ("multiclass", "log_loss", np.tile([0, 1, 2], 14)[:40]),
        ("regression", "rmse", np.linspace(0.0, 1.0, 40)),
    ],
)
def test_feature_selection_model_importance_supports_all_task_types(
    task: str, metric: str, target: np.ndarray
) -> None:
    spec = _spec(task=task, metric=metric)
    df = _frame(40).assign(y=target)
    config = Config(
        project="f",
        task=task,  # type: ignore[arg-type]
        metric=metric,
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            feature_blocks=[
                FeatureBlockConfig(type="base"),
                FeatureBlockConfig(
                    type="feature_selection",
                    params={"method": "model_importance", "max_features": 2},
                ),
            ]
        ),
    )
    first = FeatureBuilder(config, spec, "tree").fit_transform(df, target)
    second = FeatureBuilder(config, spec, "tree").fit_transform(df, target)
    assert first.shape[1] == 2
    assert list(first.columns) == list(second.columns)
    assert not first.isna().any().any()


def test_hash_encoding_is_stable_label_free_and_transform_consistent() -> None:
    spec = _spec(
        feature_columns=["num", "cat"],
        numeric_columns=["num"],
        categorical_columns=["cat"],
        datetime_columns=[],
        text_columns=[],
    )
    df = _frame(40)
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            drop_constant=False,
            feature_blocks=[
                FeatureBlockConfig(type="base"),
                FeatureBlockConfig(
                    type="hash_encoding",
                    params={"columns": ["cat"], "n_features": 8},
                ),
            ],
        ),
    )
    builder = FeatureBuilder(config, spec, "tree")
    train = builder.fit_transform(df.iloc[:30], df["y"].iloc[:30].to_numpy())
    valid = builder.transform(df.iloc[30:])
    hash_columns = [column for column in train.columns if column.startswith("hash__")]

    assert len(hash_columns) == 8
    assert list(valid.columns) == builder.feature_names
    assert np.allclose(train[hash_columns].sum(axis=1), 1.0)
    assert np.allclose(valid[hash_columns].sum(axis=1), 1.0)

    changed = df.copy()
    changed.loc[0, "y"] = 1 - changed.loc[0, "y"]
    other = FeatureBuilder(config, spec, "tree").fit_transform(
        changed.iloc[:30], changed["y"].iloc[:30].to_numpy()
    )
    pd.testing.assert_frame_equal(train[hash_columns], other[hash_columns])


def test_feature_selection_builder_survives_joblib_roundtrip(tmp_path: Path) -> None:
    spec = _spec()
    df = _frame(40)
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            feature_blocks=[
                FeatureBlockConfig(type="base"),
                FeatureBlockConfig(
                    type="feature_selection",
                    params={"method": "variance", "max_features": 2},
                ),
            ]
        ),
    )
    builder = FeatureBuilder(config, spec, "tree")
    builder.fit_transform(df.iloc[:30], df["y"].iloc[:30].to_numpy())
    path = tmp_path / "features.joblib"
    joblib.dump(builder, path, compress=3)
    loaded = joblib.load(path)

    valid = loaded.transform(df.iloc[30:])
    assert list(valid.columns) == builder.feature_names
    assert valid.shape[1] == 2


def test_feature_selection_variance_drops_constants_even_without_drop_constant() -> None:
    spec = _spec(
        feature_columns=["num", "cat", "when", "text", "constant"],
        numeric_columns=["num", "constant"],
        categorical_columns=["cat"],
    )
    df = _frame(40)
    df["constant"] = 1.0
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            drop_constant=False,
            feature_blocks=[
                FeatureBlockConfig(type="base"),
                FeatureBlockConfig(
                    type="feature_selection",
                    params={"method": "variance", "max_features": 50},
                ),
            ],
        ),
    )
    train = FeatureBuilder(config, spec, "tree").fit_transform(df, df["y"].to_numpy())
    assert "constant" not in train.columns


def test_feature_selection_can_remove_everything() -> None:
    spec = _spec()
    df = _frame(40)
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            feature_blocks=[
                FeatureBlockConfig(type="base"),
                FeatureBlockConfig(
                    type="feature_selection",
                    params={"method": "variance", "min_variance": 1e9},
                ),
            ]
        ),
    )
    with pytest.raises(DataError, match="removed all features"):
        FeatureBuilder(config, spec, "tree").fit_transform(df, df["y"].to_numpy())


def test_feature_builder_integrates_row_statistics_and_modulo() -> None:
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            row_statistics=True,
            row_statistics_stats=["mean", "std"],
            modulo_features=True,
            modulo_moduli=[10],
            modulo_steps=[5],
            drop_constant=False,
        ),
    )
    spec = _spec(
        feature_columns=["num", "cat"],
        numeric_columns=["num"],
        categorical_columns=["cat"],
        datetime_columns=[],
        text_columns=[],
    )
    df = _frame(40)
    fb = FeatureBuilder(config, spec, "tree")
    train = fb.fit_transform(df.iloc[:30], df["y"].iloc[:30].to_numpy())
    valid = fb.transform(df.iloc[30:])

    assert {"row_mean", "row_std", "num__mod_10", "num__div_5"} <= set(train.columns)
    assert list(valid.columns) == fb.feature_names
    assert not train.isna().any().any()
    assert not valid.isna().any().any()


def test_group_statistics_block_is_fit_in_fold_and_handles_unseen_groups() -> None:
    df = _frame(40)
    spec = _spec(
        feature_columns=["num", "cat"],
        numeric_columns=["num"],
        categorical_columns=["cat"],
        datetime_columns=[],
        text_columns=[],
    )
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            feature_blocks=[
                FeatureBlockConfig(type="base"),
                FeatureBlockConfig(
                    type="group_statistics",
                    params={
                        "group_col": "cat",
                        "columns": ["num"],
                        "stats": ["mean", "std", "count"],
                    },
                ),
            ],
            drop_constant=False,
        ),
    )
    fb = FeatureBuilder(config, spec, "tree")
    train = fb.fit_transform(df.iloc[:30], df["y"].iloc[:30].to_numpy())
    assert {
        "group__cat__num__mean",
        "group__cat__num__std",
        "group__cat__count",
    } <= set(train.columns)

    changed = df.iloc[:30].copy()
    changed.loc[changed.index[0], "cat"] = "unseen"
    valid = fb.transform(changed)
    assert np.isfinite(valid.to_numpy(dtype=float)).all()
    assert valid["group__cat__count"].iloc[0] == 0.0


def test_categorical_interaction_frequency_is_label_free() -> None:
    df = _frame(40)
    df["cat2"] = np.where(np.arange(len(df)) % 2 == 0, "u", "v")
    spec = _spec(
        feature_columns=["num", "cat", "cat2"],
        numeric_columns=["num"],
        categorical_columns=["cat", "cat2"],
        datetime_columns=[],
        text_columns=[],
    )
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            interactions=[InteractionConfig(columns=["cat", "cat2"], encoding="frequency")],
            drop_constant=False,
        ),
    )
    fb = FeatureBuilder(config, spec, "tree")
    train = fb.fit_transform(df.iloc[:30], df["y"].iloc[:30].to_numpy())
    valid = fb.transform(df.iloc[30:])
    assert "cat__x__cat2__frequency" in train.columns
    assert not train.isna().any().any()
    assert not valid.isna().any().any()
    assert np.isfinite(valid["cat__x__cat2__frequency"].to_numpy()).all()


def test_categorical_interaction_target_encoding_excludes_own_row_target() -> None:
    n = 30
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "cat": [f"c{i}" for i in range(n)],
            "cat2": ["z"] * n,
            "num": np.arange(n, dtype=float),
            "y": np.array([0, 1] * (n // 2)),
        }
    )
    spec = _spec(
        feature_columns=["num", "cat", "cat2"],
        numeric_columns=["num"],
        categorical_columns=["cat", "cat2"],
        datetime_columns=[],
        text_columns=[],
    )
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            interactions=[InteractionConfig(columns=["cat", "cat2"], encoding="target")],
            drop_constant=False,
        ),
    )
    first = FeatureBuilder(config, spec, "tree").fit_transform(df, df["y"].to_numpy())
    changed = df.copy()
    changed.loc[0, "y"] = 1 - changed.loc[0, "y"]
    second = FeatureBuilder(config, spec, "tree").fit_transform(changed, changed["y"].to_numpy())
    assert "cat__x__cat2__target" in first.columns
    assert first["cat__x__cat2__target"].iloc[0] == second["cat__x__cat2__target"].iloc[0]


def test_missingness_features_are_label_free() -> None:
    df = _frame(40)
    df.loc[0, "num"] = np.nan
    df.loc[1, "num"] = 0.0
    df.loc[2, "cat"] = None
    spec = _spec(
        feature_columns=["num", "cat"],
        numeric_columns=["num"],
        categorical_columns=["cat"],
        datetime_columns=[],
        text_columns=[],
    )
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            missingness=MissingnessConfig(
                enabled=True,
                columns=["num", "cat"],
                indicators=True,
                row_count=True,
                zero_count=True,
            ),
            drop_constant=False,
        ),
    )
    fb = FeatureBuilder(config, spec, "tree")
    train = fb.fit_transform(df.iloc[:30], df["y"].iloc[:30].to_numpy())
    valid = fb.transform(df.iloc[30:])
    assert {"missing__count", "num__missing", "cat__missing", "zero__count"} <= set(train.columns)
    assert train["missing__count"].iloc[0] == 1.0
    assert train["zero__count"].iloc[1] == 1.0
    assert not train.isna().any().any()
    assert not valid.isna().any().any()


def test_datetime_cyclical_features_are_label_free() -> None:
    df = _frame(40)
    spec = _spec(
        feature_columns=["num", "when"],
        numeric_columns=["num"],
        categorical_columns=[],
        datetime_columns=["when"],
        text_columns=[],
    )
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            datetime_cyclical=DatetimeCyclicalConfig(
                enabled=True,
                columns=["when"],
                periods=["month", "dayofweek", "dayofyear", "hour"],
            ),
            drop_constant=False,
        ),
    )
    fb = FeatureBuilder(config, spec, "tree")
    train = fb.fit_transform(df.iloc[:30], df["y"].iloc[:30].to_numpy())
    valid = fb.transform(df.iloc[30:])
    expected = {
        "when__month_sin",
        "when__month_cos",
        "when__dayofweek_sin",
        "when__dayofweek_cos",
        "when__dayofyear_sin",
        "when__dayofyear_cos",
        "when__hour_sin",
        "when__hour_cos",
    }
    assert expected <= set(train.columns)
    assert not train.isna().any().any()
    assert not valid.isna().any().any()
