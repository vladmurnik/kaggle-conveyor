"""Feature block registry, catalog and plan validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kconv.config import Config, DataConfig, FeatureBlockConfig, FeatureConfig
from kconv.contracts import DatasetSpec
from kconv.errors import DataError, RecipeError
from kconv.features.blocks import (
    BLOCK_REGISTRY,
    block_types,
    get_block_spec,
    validate_block_plan,
)
from kconv.features.group_stats import build_group_statistics_plan
from kconv.recipes import _ALLOWED_BLOCK_TYPES

BLOCK_TYPES = frozenset(
    {
        "base",
        "target_encoding",
        "frequency_encoding",
        "hash_encoding",
        "group_statistics",
        "categorical_interaction",
        "missingness",
        "datetime_cyclical",
        "numeric_interactions",
        "clustering",
        "anomaly",
        "time_series",
        "row_statistics",
        "modulo",
        "fold_plan",
        "feature_selection",
        "knn_class_distance",
    }
)


def _spec(**kwargs) -> DatasetSpec:
    base = dict(
        train=Path("train.csv"),
        test=None,
        sample_submission=None,
        target="y",
        id_col="id",
        task="binary",
        metric="roc_auc",
        feature_columns=["num", "cat"],
        numeric_columns=["num"],
        categorical_columns=["cat"],
        datetime_columns=[],
        text_columns=[],
    )
    base.update(kwargs)
    return DatasetSpec(**base)


def _config(blocks: list[FeatureBlockConfig] | None = None, task: str = "binary") -> Config:
    return Config(
        project="f",
        task=task,  # type: ignore[arg-type]
        metric="rmse" if task == "regression" else "roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(feature_blocks=blocks),
    )


def _binary_plan(*blocks: FeatureBlockConfig) -> list[FeatureBlockConfig]:
    return [
        FeatureBlockConfig(type="base"),
        *blocks,
    ]


def test_legacy_config_omits_new_fields() -> None:
    config = Config(project="f", task="binary", metric="roc_auc", data=DataConfig(train="x", target="y"))

    assert config.features.feature_blocks is None
    assert config.features.target_smoothing is None
    assert config.features.hash_encode == []
    assert config.features.hash_features == 16
    assert config.recipe_kind is None

    dumped = config.model_dump(mode="json", exclude_none=True)
    assert "feature_blocks" not in dumped["features"]
    assert "target_smoothing" not in dumped["features"]
    assert "recipe_kind" not in dumped


def test_explicit_valid_block_plan_passes() -> None:
    blocks = [
        FeatureBlockConfig(type="base"),
        FeatureBlockConfig(type="frequency_encoding", params={"columns": ["cat"]}),
        FeatureBlockConfig(type="target_encoding", params={"columns": ["cat"], "smoothing": 10.0}),
        FeatureBlockConfig(type="row_statistics", params={"stats": ["mean", "std", "min", "max"]}),
        FeatureBlockConfig(type="modulo", params={"moduli": [10, 100], "steps": [5]}),
    ]
    config = _config(blocks)

    assert validate_block_plan(config, _spec()) == blocks


def test_explicit_encoding_columns_must_exist() -> None:
    blocks = _binary_plan(
        FeatureBlockConfig(type="target_encoding", params={"columns": ["typo"], "smoothing": 10.0})
    )
    with pytest.raises(RecipeError, match="unknown column"):
        validate_block_plan(_config(blocks), _spec())


def test_base_block_must_be_first_and_appear_once() -> None:
    missing_base = [FeatureBlockConfig(type="target_encoding", params={"columns": ["cat"], "smoothing": 10.0})]
    config = _config(missing_base)
    with pytest.raises(RecipeError, match="base"):
        validate_block_plan(config, _spec())

    not_first = [
        FeatureBlockConfig(type="target_encoding", params={"columns": ["cat"], "smoothing": 10.0}),
        FeatureBlockConfig(type="base"),
    ]
    config = _config(not_first)
    with pytest.raises(RecipeError, match="base"):
        validate_block_plan(config, _spec())

    duplicate = [FeatureBlockConfig(type="base"), FeatureBlockConfig(type="base")]
    config = _config(duplicate)
    with pytest.raises(DataError, match="exactly once"):
        validate_block_plan(config, _spec())


def test_unknown_block_lists_allowed_types() -> None:
    config = _config([FeatureBlockConfig(type="not_a_block")])
    with pytest.raises(RecipeError, match="not_a_block"):
        validate_block_plan(config, _spec())
    with pytest.raises(RecipeError, match="Allowed feature block types"):
        get_block_spec("not_a_block")


def test_block_registry_catalog_is_complete() -> None:
    assert frozenset(BLOCK_REGISTRY) == BLOCK_TYPES
    assert _ALLOWED_BLOCK_TYPES == frozenset(BLOCK_REGISTRY)
    assert block_types() == sorted(BLOCK_REGISTRY)
    assert get_block_spec("base").required_params == ()
    assert get_block_spec("base").resource_hints == ("cpu",)
    assert get_block_spec("knn_class_distance").resource_hints == ("cpu", "ram", "cpu-heavy", "classification-only")
    assert get_block_spec("target_encoding").required_params == ("columns",)
    assert get_block_spec("hash_encoding").required_params == ("columns",)
    assert get_block_spec("hash_encoding").allowed_params == ("columns", "n_features")
    assert get_block_spec("hash_encoding").resource_hints == ("cpu",)
    assert get_block_spec("group_statistics").required_params == ("group_col",)
    assert get_block_spec("group_statistics").task_constraints == ("fit_in_outer_fold",)
    assert get_block_spec("categorical_interaction").required_params == ("columns",)
    assert get_block_spec("categorical_interaction").allowed_params == (
        "columns",
        "encoding",
        "max_pairs",
    )
    assert get_block_spec("missingness").task_constraints == ("label_free",)
    assert get_block_spec("datetime_cyclical").allowed_params == ("columns", "periods")
    assert get_block_spec("numeric_interactions").allowed_params == (
        "clip_value",
        "columns",
        "max_pairs",
        "operations",
    )
    assert get_block_spec("numeric_interactions").task_constraints == (
        "label_free",
        "numeric_columns",
    )
    assert get_block_spec("clustering").task_constraints == (
        "fit_in_outer_fold",
        "numeric_columns",
    )
    assert get_block_spec("anomaly").resource_hints == ("cpu", "ram")
    assert get_block_spec("time_series").required_params == ("time_col",)
    assert get_block_spec("time_series").task_constraints == (
        "datetime_columns",
        "fit_in_outer_fold",
        "numeric_columns",
    )


def test_knn_block_requires_classification_and_numeric_columns() -> None:
    regression_config = _config(_binary_plan(FeatureBlockConfig(type="knn_class_distance")), task="regression")
    with pytest.raises(DataError, match="classification-only"):
        validate_block_plan(regression_config, _spec(task="regression"))

    without_numeric = _config(_binary_plan(FeatureBlockConfig(type="knn_class_distance")))
    with pytest.raises(DataError, match="numeric"):
        validate_block_plan(without_numeric, _spec(numeric_columns=[], categorical_columns=[]))

    binary_config = _config(_binary_plan(FeatureBlockConfig(type="knn_class_distance")))
    assert validate_block_plan(binary_config, _spec()) == binary_config.features.feature_blocks


def test_missing_required_block_params_are_rejected() -> None:
    config = _config(
        [
            FeatureBlockConfig(type="base"),
            FeatureBlockConfig(type="target_encoding"),
        ]
    )
    with pytest.raises(RecipeError, match="required"):
        validate_block_plan(config, _spec())


def test_unknown_block_params_are_rejected() -> None:
    config = _config(
        _binary_plan(
            FeatureBlockConfig(
                type="target_encoding",
                params={"columns": ["cat"], "smoothing": 10.0, "typo": True},
            )
        )
    )
    with pytest.raises(RecipeError, match="unknown param"):
        validate_block_plan(config, _spec())


def test_hash_encoding_params_and_columns_are_validated() -> None:
    valid = _config(
        _binary_plan(
            FeatureBlockConfig(
                type="hash_encoding",
                params={"columns": ["cat"], "n_features": 8},
            )
        )
    )
    assert validate_block_plan(valid, _spec()) == valid.features.feature_blocks

    missing = _config(_binary_plan(FeatureBlockConfig(type="hash_encoding")))
    with pytest.raises(RecipeError, match="required"):
        validate_block_plan(missing, _spec())

    unknown = _config(
        _binary_plan(
            FeatureBlockConfig(type="hash_encoding", params={"columns": ["typo"]})
        )
    )
    with pytest.raises(RecipeError, match="unknown column"):
        validate_block_plan(unknown, _spec())

    for value in (0, 1025, True, "16"):
        invalid = _config(
            _binary_plan(
                FeatureBlockConfig(
                    type="hash_encoding",
                    params={"columns": ["cat"], "n_features": value},
                )
            )
        )
        with pytest.raises(RecipeError, match="n_features"):
            validate_block_plan(invalid, _spec())


def test_block_specs_allow_all_required_params() -> None:
    for spec in BLOCK_REGISTRY.values():
        assert set(spec.required_params) <= set(spec.allowed_params)


def test_base_block_requires_a_feature_column() -> None:
    config = _config([FeatureBlockConfig(type="base")])
    with pytest.raises(DataError, match="feature column"):
        validate_block_plan(
            config,
            _spec(feature_columns=[], numeric_columns=[], categorical_columns=[]),
        )


def test_numeric_blocks_require_a_numeric_feature_column() -> None:
    for block_type in ("row_statistics", "modulo"):
        params = {"stats": ["mean"]} if block_type == "row_statistics" else {"moduli": [10], "steps": [5]}
        config = _config(_binary_plan(FeatureBlockConfig(type=block_type, params=params)))
        with pytest.raises(DataError, match="numeric feature column"):
            validate_block_plan(
                config,
                _spec(numeric_columns=[], categorical_columns=["cat"]),
            )


def test_numeric_interaction_block_validation() -> None:
    spec = _spec(
        feature_columns=["num", "num2", "cat"],
        numeric_columns=["num", "num2"],
    )
    valid = _config(
        _binary_plan(
            FeatureBlockConfig(
                type="numeric_interactions",
                params={"columns": "auto", "max_pairs": 1, "operations": ["product", "sum"]},
            )
        )
    )
    assert validate_block_plan(valid, spec) == valid.features.feature_blocks

    for params in (
        {"columns": ["cat"]},
        {"max_pairs": 0},
        {"operations": []},
        {"operations": ["magic"]},
        {"clip_value": 0},
    ):
        invalid = _config(
            _binary_plan(FeatureBlockConfig(type="numeric_interactions", params=params))
        )
        with pytest.raises((RecipeError, DataError)):
            validate_block_plan(invalid, spec)


def test_clustering_block_validation() -> None:
    spec = _spec(
        feature_columns=["num", "num2", "cat"],
        numeric_columns=["num", "num2"],
    )
    valid = _config(
        _binary_plan(
            FeatureBlockConfig(
                type="clustering",
                params={"columns": "auto", "n_clusters": 2, "include_labels": False},
            )
        )
    )
    assert validate_block_plan(valid, spec) == valid.features.feature_blocks

    for params in (
        {"columns": ["cat"]},
        {"n_clusters": 1},
        {"include_distances": False, "include_labels": False},
        {"max_fit_rows": 99},
    ):
        invalid = _config(_binary_plan(FeatureBlockConfig(type="clustering", params=params)))
        with pytest.raises((RecipeError, DataError)):
            validate_block_plan(invalid, spec)


def test_anomaly_block_validation() -> None:
    spec = _spec(
        feature_columns=["num", "num2", "cat"],
        numeric_columns=["num", "num2"],
    )
    valid = _config(
        _binary_plan(
            FeatureBlockConfig(
                type="anomaly",
                params={"columns": "auto", "n_estimators": 10, "contamination": 0.1},
            )
        )
    )
    assert validate_block_plan(valid, spec) == valid.features.feature_blocks

    for params in (
        {"columns": ["cat"]},
        {"n_estimators": 5},
        {"contamination": 0.9},
        {"include_flag": "yes"},
    ):
        invalid = _config(_binary_plan(FeatureBlockConfig(type="anomaly", params=params)))
        with pytest.raises((RecipeError, DataError)):
            validate_block_plan(invalid, spec)


def test_time_series_block_validation() -> None:
    spec = _spec(
        feature_columns=["num", "num2", "cat", "when"],
        numeric_columns=["num", "num2"],
        datetime_columns=["when"],
    )
    valid = _config(
        _binary_plan(
            FeatureBlockConfig(
                type="time_series",
                params={
                    "time_col": "auto",
                    "group_col": "cat",
                    "columns": "auto",
                    "lags": [1, 2],
                    "windows": [2, 4],
                    "stats": ["mean", "std"],
                },
            )
        )
    )
    assert validate_block_plan(valid, spec) == valid.features.feature_blocks

    auto_group = _config(
        _binary_plan(
            FeatureBlockConfig(
                type="time_series",
                params={"time_col": "when", "group_col": "num2", "columns": "auto"},
            )
        )
    )
    assert validate_block_plan(auto_group, spec) == auto_group.features.feature_blocks

    for params in (
        {"time_col": "cat"},
        {"time_col": "when", "lags": [0]},
        {"time_col": "when", "windows": [1]},
        {"time_col": "when", "stats": ["median"]},
        {"time_col": "when", "group_col": "missing"},
        {"time_col": "when", "group_col": "num2", "columns": ["num2"]},
    ):
        invalid = _config(_binary_plan(FeatureBlockConfig(type="time_series", params=params)))
        with pytest.raises(RecipeError):
            validate_block_plan(invalid, spec)

    no_datetime = _config(
        _binary_plan(FeatureBlockConfig(type="time_series", params={"time_col": "auto"}))
    )
    with pytest.raises(DataError, match="datetime"):
        validate_block_plan(no_datetime, _spec(feature_columns=["num", "cat"]))


def test_fold_plan_requires_a_non_empty_plan() -> None:
    for plan in (None, {}, {"features": []}):
        config = _config(_binary_plan(FeatureBlockConfig(type="fold_plan", params={"plan": plan})))
        with pytest.raises(RecipeError, match="non-empty plan"):
            validate_block_plan(config, _spec())


def test_feature_selection_must_be_the_last_enabled_block() -> None:
    blocks = [
        FeatureBlockConfig(type="base"),
        FeatureBlockConfig(type="feature_selection", params={"method": "variance"}),
        FeatureBlockConfig(type="target_encoding", params={"columns": ["cat"], "smoothing": 10.0}),
    ]
    with pytest.raises(RecipeError, match="last enabled block"):
        validate_block_plan(_config(blocks), _spec())

    valid = [
        FeatureBlockConfig(type="base"),
        FeatureBlockConfig(type="target_encoding", params={"columns": ["cat"], "smoothing": 10.0}),
        FeatureBlockConfig(type="feature_selection", params={"method": "mutual_info", "max_features": 5}),
    ]
    assert validate_block_plan(_config(valid), _spec()) == valid
    importance = [
        FeatureBlockConfig(type="base"),
        FeatureBlockConfig(
            type="feature_selection",
            params={"method": "model_importance", "max_features": 5},
        ),
    ]
    assert validate_block_plan(_config(importance), _spec()) == importance


def test_feature_selection_params_are_validated() -> None:
    invalid_method = _config(
        _binary_plan(FeatureBlockConfig(type="feature_selection", params={"method": "magic"}))
    )
    with pytest.raises(RecipeError, match="unknown method"):
        validate_block_plan(invalid_method, _spec())

    invalid_max = _config(
        _binary_plan(FeatureBlockConfig(type="feature_selection", params={"max_features": 0}))
    )
    with pytest.raises(RecipeError, match="max_features"):
        validate_block_plan(invalid_max, _spec())

    invalid_variance = _config(
        _binary_plan(FeatureBlockConfig(type="feature_selection", params={"min_variance": -1.0}))
    )
    with pytest.raises(RecipeError, match="min_variance"):
        validate_block_plan(invalid_variance, _spec())


def test_target_smoothing_rejects_non_positive_values() -> None:
    assert FeatureConfig().target_smoothing is None
    assert FeatureConfig(target_smoothing=2.5).target_smoothing == 2.5
    with pytest.raises(ValidationError):
        FeatureConfig(target_smoothing=0.0)
    with pytest.raises(ValidationError):
        FeatureConfig(target_smoothing=-1.0)


def test_recipe_kind_defaults_and_accepts_pipeline() -> None:
    config = Config(project="f", data=DataConfig(train="x", target="y"))
    assert config.recipe_kind is None
    pipeline = Config(project="f", data=DataConfig(train="x", target="y"), recipe_kind="pipeline")
    assert pipeline.recipe_kind == "pipeline"
    with pytest.raises(ValidationError):
        Config(project="f", data=DataConfig(train="x", target="y"), recipe_kind="nope")


def test_group_statistics_block_validation() -> None:
    valid = _config(
        _binary_plan(
            FeatureBlockConfig(
                type="group_statistics",
                params={"group_col": "cat", "columns": ["num"], "stats": ["mean", "count"]},
            )
        )
    )
    assert validate_block_plan(valid, _spec()) == valid.features.feature_blocks

    count_only = _config(
        _binary_plan(
            FeatureBlockConfig(
                type="group_statistics",
                params={"group_col": "cat", "columns": [], "stats": ["count"]},
            )
        )
    )
    assert validate_block_plan(
        count_only, _spec(numeric_columns=[], categorical_columns=["cat"])
    ) == count_only.features.feature_blocks

    with pytest.raises(RecipeError, match="required"):
        validate_block_plan(
            _config(_binary_plan(FeatureBlockConfig(type="group_statistics"))), _spec()
        )
    with pytest.raises(RecipeError, match="stats"):
        validate_block_plan(
            _config(
                _binary_plan(
                    FeatureBlockConfig(
                        type="group_statistics",
                        params={"group_col": "cat", "stats": ["median"]},
                    )
                )
            ),
            _spec(),
        )
    with pytest.raises(RecipeError, match="numeric"):
        validate_block_plan(
            _config(
                _binary_plan(
                    FeatureBlockConfig(
                        type="group_statistics",
                        params={"group_col": "cat", "columns": ["cat"], "stats": ["mean"]},
                    )
                )
            ),
            _spec(),
        )


def test_group_statistics_conflicts_with_fold_plan() -> None:
    blocks = _binary_plan(
        FeatureBlockConfig(type="group_statistics", params={"group_col": "cat"}),
        FeatureBlockConfig(type="fold_plan", params={"plan": {"features": [{"name": "x", "expression": "row_mean(num)"}]}}),
    )
    with pytest.raises(RecipeError, match="both define"):
        validate_block_plan(_config(blocks), _spec())


def test_categorical_interaction_block_validation() -> None:
    spec = _spec(
        feature_columns=["num", "cat", "cat2"],
        categorical_columns=["cat", "cat2"],
    )
    valid = _config(
        _binary_plan(
            FeatureBlockConfig(
                type="categorical_interaction",
                params={"columns": ["cat", "cat2"], "max_pairs": 1, "encoding": "target"},
            )
        )
    )
    assert validate_block_plan(valid, spec) == valid.features.feature_blocks

    auto = _config(
        _binary_plan(
            FeatureBlockConfig(
                type="categorical_interaction",
                params={"columns": "auto", "max_pairs": 2, "encoding": "frequency"},
            )
        )
    )
    assert validate_block_plan(auto, spec) == auto.features.feature_blocks

    with pytest.raises(RecipeError, match="at least two"):
        validate_block_plan(
            _config(
                _binary_plan(
                    FeatureBlockConfig(
                        type="categorical_interaction",
                        params={"columns": ["cat"]},
                    )
                )
            ),
            spec,
        )
    with pytest.raises(RecipeError, match="categorical"):
        validate_block_plan(
            _config(
                _binary_plan(
                    FeatureBlockConfig(
                        type="categorical_interaction",
                        params={"columns": ["num", "cat"]},
                    )
                )
            ),
            spec,
        )
    with pytest.raises(RecipeError, match="encoding"):
        validate_block_plan(
            _config(
                _binary_plan(
                    FeatureBlockConfig(
                        type="categorical_interaction",
                        params={"columns": ["cat", "cat2"], "encoding": "magic"},
                    )
                )
            ),
            spec,
        )
    with pytest.raises(RecipeError, match="max_pairs"):
        validate_block_plan(
            _config(
                _binary_plan(
                    FeatureBlockConfig(
                        type="categorical_interaction",
                        params={"columns": ["cat", "cat2"], "max_pairs": 0},
                    )
                )
            ),
            spec,
        )


def test_missingness_block_validation() -> None:
    valid = _config(
        _binary_plan(
            FeatureBlockConfig(
                type="missingness",
                params={"columns": ["num", "cat"], "indicators": True, "row_count": True, "zero_count": True},
            )
        )
    )
    assert validate_block_plan(valid, _spec()) == valid.features.feature_blocks

    with pytest.raises(RecipeError, match="boolean"):
        validate_block_plan(
            _config(
                _binary_plan(
                    FeatureBlockConfig(type="missingness", params={"indicators": "yes"})
                )
            ),
            _spec(),
        )
    with pytest.raises(DataError, match="numeric"):
        validate_block_plan(
            _config(
                _binary_plan(
                    FeatureBlockConfig(
                        type="missingness",
                        params={"columns": ["cat"], "zero_count": True},
                    )
                )
            ),
            _spec(),
        )


def test_datetime_cyclical_block_validation() -> None:
    spec = _spec(
        feature_columns=["num", "cat", "when"],
        datetime_columns=["when"],
    )
    valid = _config(
        _binary_plan(
            FeatureBlockConfig(
                type="datetime_cyclical",
                params={"columns": ["when"], "periods": ["month", "hour"]},
            )
        )
    )
    assert validate_block_plan(valid, spec) == valid.features.feature_blocks

    with pytest.raises(RecipeError, match="datetime"):
        validate_block_plan(
            _config(
                _binary_plan(
                    FeatureBlockConfig(
                        type="datetime_cyclical",
                        params={"columns": ["num"]},
                    )
                )
            ),
            spec,
        )
    with pytest.raises(RecipeError, match="periods"):
        validate_block_plan(
            _config(
                _binary_plan(
                    FeatureBlockConfig(
                        type="datetime_cyclical",
                        params={"columns": ["when"], "periods": ["century"]},
                    )
                )
            ),
            spec,
        )


def test_group_statistics_hashes_long_feature_names_and_rejects_bad_identifiers() -> None:
    group_col = "group_" + "x" * 50
    value_col = "value_" + "y" * 50
    spec = _spec(
        feature_columns=[group_col, value_col],
        numeric_columns=[value_col],
        categorical_columns=[group_col],
    )
    plan = build_group_statistics_plan(
        {"group_col": group_col, "columns": [value_col], "stats": ["mean", "count"]},
        spec,
    )
    names = [feature["name"] for feature in plan.plan["features"]]
    assert len(names) == 2
    assert all(len(name) <= 40 and name.isidentifier() for name in names)
    assert len(set(names)) == len(names)

    bad_spec = _spec(
        feature_columns=["bad-name", "num"],
        numeric_columns=["num"],
        categorical_columns=["bad-name"],
    )
    with pytest.raises(RecipeError, match="identifier"):
        validate_block_plan(
            _config(
                _binary_plan(
                    FeatureBlockConfig(
                        type="group_statistics",
                        params={"group_col": "bad-name", "columns": ["num"]},
                    )
                )
            ),
            bad_spec,
        )


def test_group_statistics_hashes_only_feature_names_over_40_chars() -> None:
    group_40 = "g" * 26
    group_41 = "g" * 27
    plan_40 = build_group_statistics_plan(
        {"group_col": group_40, "columns": [], "stats": ["count"]},
        _spec(
            feature_columns=[group_40],
            numeric_columns=[],
            categorical_columns=[group_40],
        ),
    )
    plan_41 = build_group_statistics_plan(
        {"group_col": group_41, "columns": [], "stats": ["count"]},
        _spec(
            feature_columns=[group_41],
            numeric_columns=[],
            categorical_columns=[group_41],
        ),
    )

    name_40 = plan_40.plan["features"][0]["name"]
    name_41 = plan_41.plan["features"][0]["name"]
    assert name_40 == f"group__{group_40}__count"
    assert len(name_40) == 40
    assert name_41.startswith("group__") and len(name_41) <= 40
    assert name_41 != f"group__{group_41}__count"
