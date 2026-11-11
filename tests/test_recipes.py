"""Recipe catalog tests."""

from __future__ import annotations

import pytest

from kconv.errors import ConfigError, RecipeError
from kconv.models import create_model
from kconv.project import _parse_models
from kconv.recipes import Recipe, RecipeBlock, RecipeDefaults, RecipePipeline, list_recipes, load_recipe


def test_all_builtin_recipes_load() -> None:
    recipes = list_recipes()
    assert len(recipes) == 65
    presets = [recipe for recipe in recipes if recipe.kind == "preset"]
    pipelines = [recipe for recipe in recipes if recipe.kind == "pipeline"]
    assert len(presets) == 16
    assert len(pipelines) == 49
    ids = {recipe.id for recipe in recipes}
    assert "pesnik/baseline" in ids
    assert "cdeotte/tabular" in ids
    assert "kconv/sklearn-diverse" in ids
    assert "kconv/linear-stack" in ids
    assert "pipeline/pesnik-clean" in ids
    assert "pipeline/knnfeat-class-distance" in ids
    assert "pipeline/arnabbiswas-staged" in ids
    assert "pipeline/kozodoi-high-cardinality" in ids
    assert "pipeline/sergeyshilin-contracts" in ids
    assert "pipeline/nyaggle-oof-ensemble" in ids
    assert "pipeline/kconv-missing-row-profile" in ids
    assert "pipeline/kaggle-ieee-time-group" in ids
    assert "pipeline/kozodoi-hash-missing-numeric" in ids
    assert "pipeline/kconv-modulo-anomaly" in ids
    assert "pipeline/kconv-datetime-cluster" in ids
    assert "pipeline/ensemble-amazon-time-interaction" in ids
    assert "pipeline/knnfeat-hash-distance" in ids
    assert "pipeline/kozodoi-numeric-cluster" in ids
    assert "pipeline/ensemble-amazon-stack" in ids
    assert "pipeline/kconv-sklearn-diverse" in ids
    assert "pipeline/statsmaster-honest-cv" in ids
    assert "pipeline/dptools-group-profile" in ids
    assert "pipeline/kozodoi-combination-frequency" in ids
    assert "pipeline/kaggle-ieee-missingness" in ids
    assert "pipeline/cdeotte-datetime-cyclical" in ids
    assert "pipeline/ensemble-amazon-interaction-target" in ids
    assert {
        "nyaggle/oof-ensemble",
        "kaggler/target-encoding",
        "dptools/row-features",
        "knnfeat/knn-features",
        "ensemble-amazon/stacking",
        "kaggle-ieee/fraud-pipeline",
    } <= ids
    for recipe in recipes:
        assert recipe.title
        assert recipe.author
        assert recipe.source
        assert recipe.license
        assert recipe.defaults.models
    assert all(recipe.pipeline is None for recipe in presets)
    assert all(recipe.pipeline is not None for recipe in pipelines)
    assert all(recipe.pipeline.blocks[0].type == "base" for recipe in pipelines)
    assert load_recipe("minerva/stacking").defaults.ensemble.method == "stack"
    assert load_recipe("dptools/row-features").defaults.features.row_statistics is True
    assert load_recipe("knnfeat/knn-features").defaults.features.knn_features.enabled is True
    assert load_recipe("knnfeat/knn-features").tasks == ["binary", "multiclass"]
    pipeline = load_recipe("pipeline/dptools-row-profile")
    assert pipeline.kind == "pipeline"
    assert pipeline.derived_from == "dptools/row-features"
    assert [block.type for block in pipeline.pipeline.blocks] == [
        "base",
        "target_encoding",
        "frequency_encoding",
        "row_statistics",
    ]
    staged = load_recipe("pipeline/arnabbiswas-staged")
    assert staged.derived_from == "arnabbiswas1/staged"
    assert [block.type for block in staged.pipeline.blocks] == [
        "base",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ]

    high_cardinality = load_recipe("pipeline/kozodoi-high-cardinality")
    assert [block.type for block in high_cardinality.pipeline.blocks] == [
        "base",
        "hash_encoding",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ]
    assert high_cardinality.pipeline.blocks[1].params["n_features"] == 32
    assert high_cardinality.pipeline.blocks[-1].params["method"] == "model_importance"

    group_profile = load_recipe("pipeline/dptools-group-profile")
    assert [block.type for block in group_profile.pipeline.blocks] == [
        "base",
        "target_encoding",
        "frequency_encoding",
        "group_statistics",
        "feature_selection",
    ]
    assert group_profile.pipeline.blocks[3].params["group_col"] == "auto"
    assert group_profile.pipeline.blocks[3].params["stats"] == ["mean", "std", "count"]

    combination = load_recipe("pipeline/kozodoi-combination-frequency")
    assert combination.pipeline.blocks[2].type == "categorical_interaction"
    assert combination.pipeline.blocks[2].params["encoding"] == "frequency"

    missingness = load_recipe("pipeline/kaggle-ieee-missingness")
    assert missingness.pipeline.blocks[1].type == "missingness"
    assert missingness.pipeline.blocks[1].params["zero_count"] is True

    cyclical = load_recipe("pipeline/cdeotte-datetime-cyclical")
    assert cyclical.pipeline.blocks[1].type == "datetime_cyclical"
    assert cyclical.pipeline.blocks[1].params["periods"] == [
        "month",
        "dayofweek",
        "dayofyear",
        "hour",
    ]

    interaction_target = load_recipe("pipeline/ensemble-amazon-interaction-target")
    assert interaction_target.pipeline.blocks[1].type == "categorical_interaction"
    assert interaction_target.pipeline.blocks[1].params["encoding"] == "target"
    assert interaction_target.defaults.ensemble.method == "stack"

    contracts = load_recipe("pipeline/sergeyshilin-contracts")
    assert contracts.defaults.validation.adversarial is True
    assert contracts.defaults.validation.adversarial_threshold == 0.8
    assert contracts.defaults.ensemble.method == "mean"
    assert contracts.pipeline.blocks[-1].params["method"] == "model_importance"


def test_recipe_kind_defaults_to_preset() -> None:
    recipe = Recipe(
        id="custom/default-kind",
        title="Custom",
        author="Me",
        source="local",
        license="MIT",
        best_for="test",
        defaults=RecipeDefaults(models=["lgbm"]),
        attribution="test",
    )
    assert recipe.kind == "preset"
    assert recipe.derived_from is None
    assert recipe.pipeline is None


def test_pipeline_requires_base_first_and_nonempty() -> None:
    base = RecipeBlock(type="base")
    pipeline = RecipePipeline(blocks=[base, RecipeBlock(type="target_encoding", params={"columns": ["cat"]})])
    recipe = Recipe(
        id="custom/pipeline",
        kind="pipeline",
        title="Custom pipeline",
        author="Me",
        source="local",
        license="MIT",
        best_for="test",
        defaults=RecipeDefaults(models=["lgbm"]),
        pipeline=pipeline,
        attribution="test",
    )
    assert recipe.pipeline is not None
    assert [block.type for block in recipe.pipeline.blocks] == ["base", "target_encoding"]

    with pytest.raises(Exception, match="base"):
        RecipePipeline(blocks=[RecipeBlock(type="target_encoding", params={"columns": ["cat"]})])
    with pytest.raises(Exception, match="exactly once"):
        RecipePipeline(blocks=[RecipeBlock(type="base"), RecipeBlock(type="base")])


def test_pipeline_recipe_rejects_unknown_block_and_preset_pipeline() -> None:
    with pytest.raises(Exception, match="allowed feature block types"):
        RecipePipeline(blocks=[RecipeBlock(type="base"), RecipeBlock(type="unknown")])

    with pytest.raises(Exception, match="must not declare pipeline"):
        Recipe(
            id="custom/bad",
            kind="preset",
            title="Bad",
            author="Me",
            source="local",
            license="MIT",
            best_for="test",
            defaults=RecipeDefaults(models=["lgbm"]),
            pipeline=RecipePipeline(blocks=[RecipeBlock(type="base")]),
            attribution="test",
        )

    with pytest.raises(Exception, match="requires a non-empty pipeline.blocks"):
        Recipe(
            id="custom/empty",
            kind="pipeline",
            title="Empty",
            author="Me",
            source="local",
            license="MIT",
            best_for="test",
            defaults=RecipeDefaults(models=["lgbm"]),
            attribution="test",
        )


def test_derived_from_must_not_be_empty() -> None:
    with pytest.raises(Exception, match="derived_from"):
        Recipe(
            id="custom/derived",
            derived_from="",
            title="Derived",
            author="Me",
            source="local",
            license="MIT",
            best_for="test",
            defaults=RecipeDefaults(models=["lgbm"]),
            attribution="test",
        )


def test_every_recipe_has_a_compatible_default_model_for_its_tasks() -> None:
    for recipe in list_recipes():
        for task in recipe.tasks:
            compatible = [
                name for name in recipe.defaults.models if task in create_model(name).task_types
            ]
            assert compatible, f"{recipe.id} declares {task} but has no compatible default model"


def test_unknown_recipe_raises() -> None:
    with pytest.raises(RecipeError):
        load_recipe("does/not-exist")


def test_user_recipe_override(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    (recipes_dir / "custom.yaml").write_text(
        "id: custom/recipe\n"
        "title: Custom\n"
        "author: Me\n"
        "source: local\n"
        "license: MIT\n"
        "tasks: [binary]\n"
        "best_for: test\n"
        "defaults:\n"
        "  models: [lgbm]\n"
        "attribution: test\n",
        encoding="utf-8",
    )
    recipe = load_recipe("custom/recipe")
    assert recipe.title == "Custom"
    assert "custom/recipe" in {r.id for r in list_recipes()}


def _write_recipe(path, *, recipe_id: str, title: str = "Custom", models: str = "lgbm", tasks: str = "binary") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"id: {recipe_id}\n"
        f"title: {title}\n"
        "author: Me\n"
        "source: local\n"
        "license: MIT\n"
        f"tasks: [{tasks}]\n"
        "best_for: test\n"
        "defaults:\n"
        f"  models: [{models}]\n"
        "attribution: test\n",
        encoding="utf-8",
    )


def test_catalog_user_recipe_overrides_builtin(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_recipe(tmp_path / "recipes" / "pesnik__baseline.yaml", recipe_id="pesnik/baseline", title="Team baseline")

    assert load_recipe("pesnik/baseline").title == "Team baseline"
    assert next(r.title for r in list_recipes() if r.id == "pesnik/baseline") == "Team baseline"


def test_nested_user_recipe_is_loaded_and_listed(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_recipe(
        tmp_path / "recipes" / "team" / "recipe" / "recipe.yaml",
        recipe_id="team/recipe",
        title="Nested recipe",
    )

    assert load_recipe("team/recipe").title == "Nested recipe"
    assert "team/recipe" in {recipe.id for recipe in list_recipes()}


def test_recipe_tasks_are_validated(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_recipe(tmp_path / "recipes" / "bad.yaml", recipe_id="bad/tasks", tasks="ranking")

    with pytest.raises(RecipeError, match="unknown task"):
        load_recipe("bad/tasks")


def test_incompatible_recipe_defaults_do_not_silently_fallback() -> None:
    recipe = Recipe(
        id="test/ridge-only",
        title="Ridge only",
        author="Me",
        source="local",
        license="MIT",
        tasks=["binary"],
        best_for="test",
        defaults=RecipeDefaults(models=["ridge"]),
        attribution="test",
    )
    with pytest.raises(RecipeError, match="no installed model compatible"):
        _parse_models(None, recipe, "binary")
    with pytest.raises(ConfigError, match="does not support task"):
        _parse_models("ridge", recipe, "binary")
