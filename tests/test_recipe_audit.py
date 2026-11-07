"""Recipe audit: duplicate presets, strict pipelines and JSON safety."""

from __future__ import annotations

import json

import pytest

from kconv.errors import RecipeError
from kconv.recipe_audit import audit_recipes, normalized_signature, signature_key
from kconv.recipes import Recipe, RecipeBlock, RecipeDefaults, RecipePipeline, list_recipes


def _pipeline(recipe_id: str) -> Recipe:
    return Recipe(
        id=recipe_id,
        kind="pipeline",
        title=recipe_id,
        author="Me",
        source="local",
        license="MIT",
        best_for="test",
        defaults=RecipeDefaults(models=["lgbm"]),
        pipeline=RecipePipeline(
            blocks=[
                RecipeBlock(type="base"),
                RecipeBlock(type="target_encoding", params={"columns": ["cat"]}),
            ]
        ),
        attribution="test",
    )


def test_audit_finds_known_preset_duplicates() -> None:
    result = audit_recipes()
    assert result.kind_counts == {"pipeline": 29, "preset": 16}
    groups = {group.recipe_ids for group in result.duplicate_groups}
    assert ("cdeotte/tabular", "statsmaster/tabular") in groups
    assert (
        "congmnguyen/leakage-safe",
        "kaggle-ieee/fraud-pipeline",
        "kaggler/target-encoding",
    ) in groups
    assert not result.has_pipeline_duplicates


def test_audit_strict_is_ok_for_preset_duplicates() -> None:
    result = audit_recipes(strict=True)
    assert result.duplicate_groups
    assert not result.has_pipeline_duplicates


def test_audit_strict_rejects_duplicate_pipelines() -> None:
    first = _pipeline("pipeline/one")
    second = _pipeline("pipeline/two")
    assert signature_key(first) == signature_key(second)
    with pytest.raises(RecipeError, match="Duplicate executable pipeline"):
        audit_recipes([first, second], strict=True)


def test_audit_payload_is_json_safe_and_signature_is_stable() -> None:
    recipes = list_recipes()
    recipe = recipes[0]
    signature = normalized_signature(recipe)
    assert signature["kind"] == recipe.kind
    assert signature == normalized_signature(recipe)
    assert json.loads(signature_key(recipe)) == signature
    payload = audit_recipes().to_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=True))["kind_counts"] == {
        "pipeline": 29,
        "preset": 16,
    }
