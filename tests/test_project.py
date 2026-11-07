"""Project creation tests: portability, auto-encoding and strategy profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kconv.config import (
    Config,
    DataConfig,
    FeatureBlockConfig,
    FeatureConfig,
    ValidationConfig,
    load_config,
)
from kconv.contracts import DatasetSpec
from kconv.errors import ConfigError, RecipeError
from kconv.project import (
    _compile_pipeline_features,
    _validate_column_roles,
    apply_fold_plan,
    build_config,
)
from kconv.recipes import Recipe, RecipeBlock, RecipeDefaults, RecipePipeline


def _dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    n = 300
    high_card = [f"account_{i}" for i in range(60)]
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "num": rng.normal(size=n),
            "small_cat": rng.choice(["a", "b", "c"], n),
            "account": rng.choice(high_card, n),
            "target": (rng.random(n) > 0.5).astype(int),
        }
    )
    train = frame.iloc[:240].copy()
    test = frame.iloc[240:].drop(columns=["target"]).copy()
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "target": 0}).to_csv(root / "sample_submission.csv", index=False)
    return root


def _time_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(23)
    n = 300
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "num": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], n),
            "when": pd.date_range("2024-01-01", periods=n, freq="h").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "target": (rng.random(n) > 0.5).astype(int),
        }
    )
    train = frame.iloc[:240].copy()
    test = frame.iloc[240:].drop(columns=["target"]).copy()
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "target": 0}).to_csv(root / "sample_submission.csv", index=False)
    return root


def _create(tmp_path: Path, monkeypatch, strategy: str = "fast"):
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    project_dir, config_path, cfg = build_config(
        data,
        target="target",
        recipe_id="pesnik/baseline",
        strategy=strategy,
        models="lgbm",
        name="portable",
    )
    return project_dir, config_path, cfg


def test_config_uses_relative_paths(tmp_path, monkeypatch) -> None:
    project_dir, config_path, cfg = _create(tmp_path, monkeypatch)
    assert not Path(cfg.data.train).is_absolute()
    assert not Path(cfg.data.test).is_absolute()
    assert ".." in Path(cfg.data.train).parts
    assert config_path.exists()
    assert (project_dir / cfg.data.train).resolve() == (tmp_path / "data" / "train.csv").resolve()


def test_high_cardinality_columns_are_auto_encoded(tmp_path, monkeypatch) -> None:
    _, _, cfg = _create(tmp_path, monkeypatch)
    assert "account" in cfg.features.target_encode
    assert "account" in cfg.features.frequency_encode
    assert "small_cat" not in cfg.features.target_encode


def test_auto_encoding_can_be_disabled(tmp_path, monkeypatch) -> None:
    _, config_path, cfg = _create(tmp_path, monkeypatch)
    raw = cfg.model_dump(mode="python")
    raw["features"]["auto_encode"] = False
    raw["features"]["target_encode"] = []
    raw["features"]["frequency_encode"] = []
    from kconv.config import Config

    disabled = Config.model_validate(raw)
    assert disabled.features.target_encode == []


def test_explicit_id_and_ignore_are_persisted(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    _, _, cfg = build_config(
        data,
        target="target",
        recipe_id="pesnik/baseline",
        strategy="fast",
        models="lgbm",
        name="roles",
        id_col="id",
        ignore="account",
    )
    assert cfg.data.id == "id"
    assert cfg.data.ignore == ["account"]
    assert "account" not in cfg.features.target_encode
    assert "account" not in cfg.features.frequency_encode


def test_pipeline_recipe_compiles_blocks_and_resolves_auto_columns(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    _, config_path, cfg = build_config(
        data,
        target="target",
        recipe_id="pipeline/dptools-row-profile",
        strategy="fast",
        models="lgbm",
        name="pipeline",
    )
    assert cfg.recipe_kind == "pipeline"
    assert cfg.features.auto_encode is False
    assert cfg.features.row_statistics is True
    blocks = cfg.features.feature_blocks
    assert blocks is not None
    assert [block.type for block in blocks] == [
        "base",
        "target_encoding",
        "frequency_encoding",
        "row_statistics",
    ]
    target = next(block for block in blocks if block.type == "target_encoding")
    frequency = next(block for block in blocks if block.type == "frequency_encoding")
    assert target.params["columns"] == ["account"]
    assert frequency.params["columns"] == ["account"]
    assert cfg.features.target_encode == ["account"]
    assert cfg.features.frequency_encode == ["account"]
    loaded = load_config(config_path)
    assert loaded.recipe_kind == "pipeline"
    assert loaded.features.feature_blocks is not None
    assert loaded.features.feature_blocks[0].type == "base"


def test_feature_selection_pipeline_compiles_into_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    _, config_path, cfg = build_config(
        data,
        target="target",
        recipe_id="pipeline/arnabbiswas-staged",
        strategy="fast",
        models="lgbm",
        name="staged",
    )
    assert cfg.recipe_kind == "pipeline"
    assert cfg.features.feature_selection.enabled is True
    assert cfg.features.feature_selection.method == "mutual_info"
    assert cfg.features.feature_selection.max_features == 25
    blocks = cfg.features.feature_blocks
    assert blocks is not None
    assert blocks[-1].type == "feature_selection"
    loaded = load_config(config_path)
    assert loaded.features.feature_selection.enabled is True
    assert loaded.features.feature_selection.method == "mutual_info"


def test_hash_pipeline_compiles_into_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    _, config_path, cfg = build_config(
        data,
        target="target",
        recipe_id="pipeline/kozodoi-high-cardinality",
        strategy="fast",
        models="lgbm",
        name="hash",
    )
    assert cfg.recipe_kind == "pipeline"
    assert cfg.features.hash_encode == ["account"]
    assert cfg.features.hash_features == 32
    assert cfg.features.feature_selection.enabled is True
    assert cfg.features.feature_selection.method == "model_importance"
    blocks = cfg.features.feature_blocks
    assert blocks is not None
    hash_block = next(block for block in blocks if block.type == "hash_encoding")
    assert hash_block.params["columns"] == ["account"]
    assert hash_block.params["n_features"] == 32
    loaded = load_config(config_path)
    assert loaded.features.hash_encode == ["account"]
    assert loaded.features.hash_features == 32


def test_invalid_hash_block_raises_recipe_error() -> None:
    recipe = Recipe(
        id="pipeline/bad-hash",
        kind="pipeline",
        title="bad hash",
        author="test",
        source="local",
        license="MIT",
        best_for="test",
        defaults=RecipeDefaults(),
        pipeline=RecipePipeline(
            blocks=[
                RecipeBlock(type="base"),
                RecipeBlock(
                    type="hash_encoding",
                    params={"columns": ["num"], "n_features": 0},
                ),
            ]
        ),
        attribution="test",
    )
    spec = DatasetSpec(
        train=Path("train.csv"),
        test=None,
        sample_submission=None,
        target="target",
        id_col="id",
        task="binary",
        metric="roc_auc",
        feature_columns=["num"],
        numeric_columns=["num"],
        categorical_columns=[],
        datetime_columns=[],
        text_columns=[],
    )
    with pytest.raises(RecipeError, match="n_features"):
        _compile_pipeline_features(
            recipe,
            FeatureConfig(),
            spec,
            Path("train.csv"),
            [],
        )


def test_invalid_feature_selection_block_raises_recipe_error() -> None:
    recipe = Recipe(
        id="pipeline/bad-selection",
        kind="pipeline",
        title="bad selection",
        author="test",
        source="local",
        license="MIT",
        best_for="test",
        defaults=RecipeDefaults(),
        pipeline=RecipePipeline(
            blocks=[
                RecipeBlock(type="base"),
                RecipeBlock(type="feature_selection", params={"method": "magic"}),
            ]
        ),
        attribution="test",
    )
    spec = DatasetSpec(
        train=Path("train.csv"),
        test=None,
        sample_submission=None,
        target="target",
        id_col="id",
        task="binary",
        metric="roc_auc",
        feature_columns=["num"],
        numeric_columns=["num"],
        categorical_columns=[],
        datetime_columns=[],
        text_columns=[],
    )
    with pytest.raises(RecipeError, match="Invalid feature_selection"):
        _compile_pipeline_features(
            recipe,
            FeatureConfig(),
            spec,
            Path("train.csv"),
            [],
        )


def test_validation_group_and_time_columns_must_exist(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "data")
    spec = DatasetSpec(
        train=data / "train.csv",
        test=data / "test.csv",
        sample_submission=data / "sample_submission.csv",
        target="target",
        id_col="id",
        task="binary",
        metric="roc_auc",
        feature_columns=["num", "small_cat", "account"],
        numeric_columns=["num"],
        categorical_columns=["small_cat", "account"],
    )
    with pytest.raises(ConfigError, match="group_col"):
        _validate_column_roles(ValidationConfig(group_col="missing_group"), spec)
    with pytest.raises(ConfigError, match="time_col"):
        _validate_column_roles(ValidationConfig(time_col="missing_time"), spec)
    _validate_column_roles(ValidationConfig(group_col="account", time_col="num"), spec)


def test_new_advanced_blocks_compile_into_pipeline_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    expectations = {
        "pipeline/dptools-group-profile": ("group_statistics", "fold_plan"),
        "pipeline/kozodoi-combination-frequency": ("categorical_interaction", "interactions"),
        "pipeline/kaggle-ieee-missingness": ("missingness", "missingness"),
        "pipeline/cdeotte-datetime-cyclical": ("datetime_cyclical", "datetime_cyclical"),
        "pipeline/ensemble-amazon-interaction-target": ("categorical_interaction", "interactions"),
    }
    for recipe_id, (block_type, feature_attr) in expectations.items():
        _, config_path, cfg = build_config(
            data,
            target="target",
            recipe_id=recipe_id,
            strategy="fast",
            models="lgbm",
            name="advanced-" + recipe_id.split("/")[-1],
        )
        assert cfg.recipe_kind == "pipeline"
        assert cfg.features.feature_blocks is not None
        assert any(block.type == block_type for block in cfg.features.feature_blocks)
        if feature_attr == "fold_plan":
            assert cfg.features.fold_plan is not None
        elif feature_attr == "interactions":
            assert cfg.features.interactions
        elif feature_attr == "missingness":
            assert cfg.features.missingness.enabled is True
        else:
            assert cfg.features.datetime_cyclical.enabled is True
        loaded = load_config(config_path)
        assert loaded.features.feature_blocks is not None
        assert any(block.type == block_type for block in loaded.features.feature_blocks)


def test_pipeline_interaction_auto_is_noop_with_one_categorical(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    train = pd.read_csv(data / "train.csv").drop(columns=["account"])
    test = pd.read_csv(data / "test.csv").drop(columns=["account"])
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    _, _, cfg = build_config(
        data,
        target="target",
        recipe_id="pipeline/kozodoi-combination-frequency",
        strategy="fast",
        models="lgbm",
        name="one-cat-interaction",
    )
    assert cfg.features.interactions == []
    block = next(
        block for block in cfg.features.feature_blocks or [] if block.type == "categorical_interaction"
    )
    assert block.params["columns"] == "auto"


def test_group_statistics_conflicts_with_apply_fold_plan() -> None:
    config = Config(
        project="f",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            feature_blocks=[
                FeatureBlockConfig(type="base"),
                FeatureBlockConfig(type="group_statistics", params={"group_col": "cat"}),
            ]
        ),
    )
    with pytest.raises(RecipeError, match="group_statistics"):
        apply_fold_plan(
            config,
            {"features": [{"name": "x", "expression": "row_mean(num)"}]},
        )


def test_apply_fold_plan_appends_pipeline_block_and_keeps_preset_shape() -> None:
    payload = {"features": [{"name": "group_mean", "expression": "mean(x) over group"}]}
    preset = Config(
        project="f",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(),
    )
    updated_preset = apply_fold_plan(preset, payload)
    assert updated_preset.features.feature_blocks is None
    assert updated_preset.features.fold_plan == payload

    pipeline = preset.model_copy(
        update={
            "features": preset.features.model_copy(
                update={"feature_blocks": [FeatureBlockConfig(type="base")]}
            )
        }
    )
    updated = apply_fold_plan(pipeline, payload)
    assert updated.features.feature_blocks is not None
    assert [block.type for block in updated.features.feature_blocks] == ["base", "fold_plan"]
    assert updated.features.feature_blocks[1].params["plan"] == payload
    assert updated.features.fold_plan == payload


def test_apply_fold_plan_inserts_before_feature_selection() -> None:
    payload = {"features": [{"name": "grp", "expression": "group_mean(num, cat)"}]}
    config = Config(
        project="f",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            feature_blocks=[
                FeatureBlockConfig(type="base"),
                FeatureBlockConfig(type="feature_selection"),
            ]
        ),
    )
    updated = apply_fold_plan(config, payload)
    assert updated.features.feature_blocks is not None
    assert [block.type for block in updated.features.feature_blocks] == [
        "base",
        "fold_plan",
        "feature_selection",
    ]
    assert updated.features.feature_blocks[1].params["plan"] == payload


def test_apply_fold_plan_moves_existing_block_before_feature_selection() -> None:
    payload = {"features": [{"name": "grp", "expression": "group_mean(num, cat)"}]}
    config = Config(
        project="f",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(
            feature_blocks=[
                FeatureBlockConfig(type="base"),
                FeatureBlockConfig(type="feature_selection"),
                FeatureBlockConfig(type="fold_plan", params={"plan": {"features": []}}),
            ]
        ),
    )
    updated = apply_fold_plan(config, payload)
    assert updated.features.feature_blocks is not None
    assert [block.type for block in updated.features.feature_blocks] == [
        "base",
        "fold_plan",
        "feature_selection",
    ]
    assert updated.features.feature_blocks[1].params["plan"] == payload


def test_deep_strategy_enables_bounded_hpo(tmp_path, monkeypatch) -> None:
    _, config_path, cfg = _create(tmp_path, monkeypatch, strategy="deep")
    assert cfg.hpo.enabled is True
    assert cfg.hpo.trials == 8
    loaded = load_config(config_path)
    assert loaded.hpo.enabled is True
    assert loaded.hpo.inner_size == 0.2


@pytest.mark.parametrize(
    ("strategy", "n_splits", "seeds", "hpo"),
    [
        ("fast", 3, 1, False),
        ("standard", 5, 2, False),
        ("deep", 5, 3, True),
    ],
)
def test_strategy_profiles_are_applied(
    tmp_path, monkeypatch, strategy: str, n_splits: int, seeds: int, hpo: bool
) -> None:
    _, _, cfg = _create(tmp_path, monkeypatch, strategy=strategy)
    assert cfg.strategy == strategy
    assert cfg.validation.n_splits == n_splits
    assert {model.seeds for model in cfg.models} == {seeds}
    assert cfg.hpo.enabled is hpo
    assert cfg.hpo.trials == 8


def test_runtime_overrides_and_manual_model_params_are_applied(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    _, config_path, cfg = build_config(
        data,
        target="target",
        recipe_id="pesnik/baseline",
        strategy="fast",
        models="lgbm,xgb",
        name="tuned",
        profile=3,
        device="cpu",
        cpu_threads=2,
        ram_gb=12,
        vram_gb=0,
        model_params={
            "lgbm": {"n_estimators": 123, "num_threads": 3},
            "xgb": {"max_depth": 4},
        },
    )

    assert cfg.strategy == "standard"  # profile 3 wins over the explicit strategy
    assert cfg.runtime is not None
    assert cfg.runtime.profile == 3
    assert cfg.runtime.device == "cpu"
    assert cfg.runtime.cpu_threads == 2
    assert cfg.runtime.ram_gb == 12
    assert cfg.runtime.vram_gb == 0
    by_name = {model.name: model for model in cfg.models}
    assert by_name["lgbm"].params["n_estimators"] == 123
    assert by_name["lgbm"].params["num_threads"] == 3  # manual overrides runtime
    assert by_name["xgb"].params["max_depth"] == 4

    loaded = load_config(config_path)
    assert loaded.runtime is not None
    assert loaded.runtime.ram_gb == 12
    assert {model.name: model for model in loaded.models}["lgbm"].params["n_estimators"] == 123

    with pytest.raises(ConfigError):
        build_config(
            data,
            target="target",
            recipe_id="pesnik/baseline",
            models="lgbm",
            name="unselected-params",
            model_params={"xgb": {"max_depth": 4}},
        )
    with pytest.raises(ConfigError):
        build_config(
            data,
            target="target",
            recipe_id="pesnik/baseline",
            models="lgbm",
            name="resources-without-profile",
            cpu_threads=2,
        )
    with pytest.raises(ConfigError):
        build_config(
            data,
            target="target",
            recipe_id="pesnik/baseline",
            models="lgbm",
            name="non-string-param",
            model_params={"lgbm": {1: "x"}},
        )


def test_recipe_validation_and_hpo_defaults_are_preserved(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    (recipes / "custom.yaml").write_text(
        "id: test/custom-defaults\n"
        "title: Custom defaults\n"
        "author: Me\n"
        "source: local\n"
        "license: MIT\n"
        "tasks: [binary]\n"
        "best_for: test\n"
        "defaults:\n"
        "  validation:\n"
        "    adversarial: false\n"
        "    n_splits: 9\n"
        "  hpo:\n"
        "    inner_size: 0.3\n"
        "    trials: 30\n"
        "  models: [lgbm]\n"
        "attribution: test\n",
        encoding="utf-8",
    )

    _, _, cfg = build_config(
        data,
        target="target",
        recipe_id="test/custom-defaults",
        strategy="fast",
        name="custom-defaults",
    )

    assert cfg.validation.adversarial is False
    assert cfg.validation.n_splits == 3
    assert cfg.hpo.inner_size == 0.3
    assert cfg.hpo.trials == 8


def test_numeric_interactions_pipeline_resolves_auto_columns(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _time_dataset(tmp_path / "data")
    _, config_path, cfg = build_config(
        data,
        target="target",
        recipe_id="pipeline/kozodoi-numeric-ratios",
        strategy="fast",
        models="lgbm",
        name="numeric",
    )
    assert cfg.features.numeric_interactions.enabled is True
    assert cfg.features.numeric_interactions.columns == ["num", "num2"]
    block = next(
        item for item in cfg.features.feature_blocks or [] if item.type == "numeric_interactions"
    )
    assert block.params["columns"] == ["num", "num2"]
    loaded = load_config(config_path)
    assert loaded.features.numeric_interactions.columns == ["num", "num2"]


def test_cluster_and_anomaly_pipeline_resolve_numeric_columns(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _time_dataset(tmp_path / "data")
    _, config_path, cfg = build_config(
        data,
        target="target",
        recipe_id="pipeline/kaggle-ieee-cluster-anomaly",
        strategy="fast",
        models="lgbm",
        name="cluster-anomaly",
    )
    assert cfg.features.clustering.enabled is True
    assert cfg.features.clustering.columns == ["num", "num2"]
    assert cfg.features.anomaly.enabled is True
    assert cfg.features.anomaly.columns == ["num", "num2"]
    loaded = load_config(config_path)
    assert loaded.features.clustering.columns == ["num", "num2"]
    assert loaded.features.anomaly.columns == ["num", "num2"]


def test_time_series_pipeline_resolves_auto_time_and_group(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _time_dataset(tmp_path / "data")
    _, config_path, cfg = build_config(
        data,
        target="target",
        recipe_id="pipeline/kozodoi-time-profile",
        strategy="fast",
        models="lgbm",
        name="time-series",
    )
    time_series = cfg.features.time_series
    assert time_series.enabled is True
    assert time_series.time_col == "when"
    assert time_series.group_col == "cat"
    assert time_series.columns == ["num", "num2"]
    block = next(
        item for item in cfg.features.feature_blocks or [] if item.type == "time_series"
    )
    assert block.params["time_col"] == "when"
    assert block.params["group_col"] == "cat"
    assert block.params["columns"] == ["num", "num2"]
    loaded = load_config(config_path)
    assert loaded.features.time_series.time_col == "when"
    assert loaded.features.time_series.group_col == "cat"


def test_time_series_pipeline_without_datetime_raises_recipe_error(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    with pytest.raises(RecipeError, match="datetime"):
        build_config(
            data,
            target="target",
            recipe_id="pipeline/kozodoi-time-profile",
            strategy="fast",
            models="lgbm",
            name="time-series-missing",
        )
