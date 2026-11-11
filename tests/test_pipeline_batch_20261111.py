"""Batch of executable pipelines added on 2026-11-11."""

from __future__ import annotations

import pytest

from kconv.pipeline_presets import ALL_PIPELINES, PIPELINE_PRESETS
from kconv.recipe_audit import signature_key
from kconv.recipes import list_recipes, load_recipe

NEW_PIPELINES: tuple[str, ...] = (
    "pipeline/kconv-missing-row-profile",
    "pipeline/kozodoi-hash-group-profile",
    "pipeline/cdeotte-datetime-numeric",
    "pipeline/kaggle-ieee-time-group",
    "pipeline/ensemble-amazon-interaction-group",
    "pipeline/knnfeat-cluster-anomaly",
    "pipeline/kaggle-ieee-missing-cluster",
    "pipeline/kozodoi-hash-missing-numeric",
    "pipeline/arnabbiswas-mutual-profile",
    "pipeline/kconv-modulo-anomaly",
    "pipeline/kconv-datetime-row-profile",
    "pipeline/kozodoi-hash-datetime",
    "pipeline/kaggle-ieee-time-row-profile",
    "pipeline/ensemble-amazon-interaction-missing",
    "pipeline/knnfeat-hash-distance",
    "pipeline/kaggle-ieee-missing-group",
    "pipeline/kozodoi-numeric-cluster",
    "pipeline/arnabbiswas-modulo-profile",
    "pipeline/kconv-datetime-cluster",
    "pipeline/ensemble-amazon-time-interaction",
)

EXPECTED_BLOCKS: dict[str, tuple[str, ...]] = {
    "pipeline/kconv-missing-row-profile": (
        "base",
        "missingness",
        "row_statistics",
        "numeric_interactions",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/kozodoi-hash-group-profile": (
        "base",
        "hash_encoding",
        "group_statistics",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/cdeotte-datetime-numeric": (
        "base",
        "datetime_cyclical",
        "numeric_interactions",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/kaggle-ieee-time-group": (
        "base",
        "time_series",
        "group_statistics",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/ensemble-amazon-interaction-group": (
        "base",
        "categorical_interaction",
        "group_statistics",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/knnfeat-cluster-anomaly": (
        "base",
        "clustering",
        "anomaly",
        "knn_class_distance",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/kaggle-ieee-missing-cluster": (
        "base",
        "missingness",
        "datetime_cyclical",
        "clustering",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/kozodoi-hash-missing-numeric": (
        "base",
        "hash_encoding",
        "missingness",
        "numeric_interactions",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/arnabbiswas-mutual-profile": (
        "base",
        "numeric_interactions",
        "row_statistics",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/kconv-modulo-anomaly": (
        "base",
        "modulo",
        "anomaly",
        "numeric_interactions",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/kconv-datetime-row-profile": (
        "base",
        "datetime_cyclical",
        "row_statistics",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/kozodoi-hash-datetime": (
        "base",
        "hash_encoding",
        "datetime_cyclical",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/kaggle-ieee-time-row-profile": (
        "base",
        "time_series",
        "row_statistics",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/ensemble-amazon-interaction-missing": (
        "base",
        "categorical_interaction",
        "missingness",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/knnfeat-hash-distance": (
        "base",
        "hash_encoding",
        "knn_class_distance",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/kaggle-ieee-missing-group": (
        "base",
        "missingness",
        "group_statistics",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/kozodoi-numeric-cluster": (
        "base",
        "numeric_interactions",
        "clustering",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/arnabbiswas-modulo-profile": (
        "base",
        "modulo",
        "row_statistics",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/kconv-datetime-cluster": (
        "base",
        "datetime_cyclical",
        "clustering",
        "anomaly",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
    "pipeline/ensemble-amazon-time-interaction": (
        "base",
        "time_series",
        "categorical_interaction",
        "target_encoding",
        "frequency_encoding",
        "feature_selection",
    ),
}


@pytest.mark.parametrize("pipeline_id", NEW_PIPELINES)
def test_new_pipeline_declares_expected_blocks(pipeline_id: str) -> None:
    recipe = load_recipe(pipeline_id)
    assert recipe.kind == "pipeline"
    assert recipe.pipeline is not None
    blocks = recipe.pipeline.blocks
    assert tuple(block.type for block in blocks) == EXPECTED_BLOCKS[pipeline_id]
    assert blocks[0].type == "base"
    assert blocks[-1].type == "feature_selection"
    assert recipe.attribution.strip()
    assert recipe.tasks


def test_new_pipelines_keep_lineage_metadata() -> None:
    recipes = {recipe.id: recipe for recipe in list_recipes()}
    for pipeline_id in NEW_PIPELINES:
        pipeline = recipes[pipeline_id]
        preset = recipes[pipeline.derived_from]
        assert pipeline.author == preset.author
        assert pipeline.source == preset.source
        assert pipeline.license == preset.license


def test_new_pipelines_are_unique_and_curated() -> None:
    signatures = [signature_key(load_recipe(pipeline_id)) for pipeline_id in NEW_PIPELINES]
    assert len(signatures) == len(set(signatures))

    assert set(NEW_PIPELINES) <= set(ALL_PIPELINES)
    curated = {
        pipeline_id
        for preset in PIPELINE_PRESETS
        if preset.kind != "all"
        for pipeline_id in preset.pipelines
    }
    assert set(NEW_PIPELINES) <= curated
