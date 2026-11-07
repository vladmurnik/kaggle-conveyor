"""Advanced executable pipelines: signatures, lineage and catalog uniqueness."""

from __future__ import annotations

import pytest

from kconv.recipes import list_recipes, load_recipe

DERIVED_FROM: dict[str, str] = {
    "pipeline/kozodoi-numeric-ratios": "kozodoi/tabular",
    "pipeline/kozodoi-time-profile": "kozodoi/tabular",
    "pipeline/kaggle-ieee-cluster-anomaly": "kaggle-ieee/fraud-pipeline",
    "pipeline/kaggle-ieee-time-profile": "kaggle-ieee/fraud-pipeline",
    "pipeline/nyaggle-cluster-stack": "nyaggle/oof-ensemble",
    "pipeline/ensemble-amazon-numeric-cross": "ensemble-amazon/stacking",
    "pipeline/sergeyshilin-anomaly-contracts": "sergeyshilin/contracts",
    "pipeline/dptools-cluster-profile": "dptools/row-features",
    "pipeline/kconv-unsupervised-profile": "kconv/sklearn-diverse",
}

EXPECTED_BLOCKS: dict[str, list[tuple[str, dict[str, object]]]] = {
    "pipeline/kozodoi-numeric-ratios": [
        ("base", {"max_categories": 100}),
        (
            "numeric_interactions",
            {"columns": "auto", "max_pairs": 6, "operations": ["product", "ratio", "difference"]},
        ),
        ("target_encoding", {"columns": "auto", "smoothing": 10}),
        ("frequency_encoding", {"columns": "auto"}),
        ("feature_selection", {"method": "model_importance", "max_features": 40}),
    ],
    "pipeline/kozodoi-time-profile": [
        ("base", {"max_categories": 100}),
        (
            "time_series",
            {
                "time_col": "auto",
                "group_col": "auto",
                "columns": "auto",
                "lags": [1, 3],
                "windows": [3, 7],
                "stats": ["mean", "std", "min", "max"],
                "include_elapsed": True,
                "include_diff": True,
            },
        ),
        ("numeric_interactions", {"columns": "auto", "max_pairs": 4, "operations": ["ratio", "difference"]}),
        ("target_encoding", {"columns": "auto", "smoothing": 10}),
        ("frequency_encoding", {"columns": "auto"}),
        ("feature_selection", {"method": "model_importance", "max_features": 40}),
    ],
    "pipeline/kaggle-ieee-cluster-anomaly": [
        ("base", {"max_categories": 100}),
        ("missingness", {"columns": "auto", "indicators": True, "row_count": True, "zero_count": True}),
        (
            "clustering",
            {
                "columns": "auto",
                "n_clusters": 8,
                "max_iter": 100,
                "include_distances": True,
                "include_labels": True,
                "max_fit_rows": 50000,
            },
        ),
        (
            "anomaly",
            {"columns": "auto", "n_estimators": 100, "contamination": "auto", "include_flag": True},
        ),
        ("target_encoding", {"columns": "auto", "smoothing": 20}),
        ("frequency_encoding", {"columns": "auto"}),
        ("feature_selection", {"method": "model_importance", "max_features": 40}),
    ],
    "pipeline/kaggle-ieee-time-profile": [
        ("base", {"max_categories": 100}),
        (
            "time_series",
            {
                "time_col": "auto",
                "group_col": "auto",
                "columns": "auto",
                "lags": [1, 2],
                "windows": [3, 7],
                "stats": ["mean", "std"],
                "include_elapsed": True,
                "include_diff": True,
            },
        ),
        ("datetime_cyclical", {"columns": "auto", "periods": ["month", "dayofweek", "hour"]}),
        ("frequency_encoding", {"columns": "auto"}),
        ("target_encoding", {"columns": "auto", "smoothing": 20}),
        ("feature_selection", {"method": "model_importance", "max_features": 40}),
    ],
    "pipeline/nyaggle-cluster-stack": [
        ("base", {"max_categories": 100}),
        (
            "clustering",
            {
                "columns": "auto",
                "n_clusters": 10,
                "max_iter": 100,
                "include_distances": True,
                "include_labels": False,
                "max_fit_rows": 50000,
            },
        ),
        ("target_encoding", {"columns": "auto", "smoothing": 10}),
        ("frequency_encoding", {"columns": "auto"}),
        ("feature_selection", {"method": "model_importance", "max_features": 40}),
    ],
    "pipeline/ensemble-amazon-numeric-cross": [
        ("base", {"max_categories": 100}),
        ("numeric_interactions", {"columns": "auto", "max_pairs": 8, "operations": ["product", "ratio"]}),
        ("categorical_interaction", {"columns": "auto", "max_pairs": 4, "encoding": "frequency"}),
        ("target_encoding", {"columns": "auto", "smoothing": 10}),
        ("frequency_encoding", {"columns": "auto"}),
        ("feature_selection", {"method": "model_importance", "max_features": 40}),
    ],
    "pipeline/sergeyshilin-anomaly-contracts": [
        ("base", {}),
        (
            "anomaly",
            {"columns": "auto", "n_estimators": 100, "contamination": 0.05, "include_flag": True},
        ),
        ("numeric_interactions", {"columns": "auto", "max_pairs": 4, "operations": ["ratio", "difference"]}),
        ("target_encoding", {"columns": "auto", "smoothing": 10}),
        ("frequency_encoding", {"columns": "auto"}),
        ("feature_selection", {"method": "model_importance", "max_features": 30}),
    ],
    "pipeline/dptools-cluster-profile": [
        ("base", {}),
        (
            "clustering",
            {
                "columns": "auto",
                "n_clusters": 6,
                "max_iter": 100,
                "include_distances": True,
                "include_labels": True,
                "max_fit_rows": 50000,
            },
        ),
        ("row_statistics", {"stats": ["mean", "std", "min", "max", "median"]}),
        ("target_encoding", {"columns": "auto", "smoothing": 10}),
        ("frequency_encoding", {"columns": "auto"}),
        ("feature_selection", {"method": "model_importance", "max_features": 40}),
    ],
    "pipeline/kconv-unsupervised-profile": [
        ("base", {"scale": "standard"}),
        (
            "numeric_interactions",
            {"columns": "auto", "max_pairs": 6, "operations": ["product", "ratio", "difference", "sum"]},
        ),
        (
            "clustering",
            {
                "columns": "auto",
                "n_clusters": 8,
                "max_iter": 100,
                "include_distances": True,
                "include_labels": False,
                "max_fit_rows": 50000,
            },
        ),
        ("anomaly", {"columns": "auto", "n_estimators": 80, "contamination": "auto", "include_flag": True}),
        ("feature_selection", {"method": "variance", "max_features": 50}),
    ],
}

EXPECTED_MODELS: dict[str, list[str]] = {
    "pipeline/kozodoi-numeric-ratios": ["lgbm", "catboost"],
    "pipeline/kozodoi-time-profile": ["lgbm", "catboost"],
    "pipeline/kaggle-ieee-cluster-anomaly": ["lgbm", "xgb"],
    "pipeline/kaggle-ieee-time-profile": ["lgbm", "xgb"],
    "pipeline/nyaggle-cluster-stack": ["lgbm", "xgb", "catboost"],
    "pipeline/ensemble-amazon-numeric-cross": ["lgbm", "xgb", "catboost", "logreg"],
    "pipeline/sergeyshilin-anomaly-contracts": ["lgbm", "catboost", "logreg"],
    "pipeline/dptools-cluster-profile": ["lgbm", "catboost"],
    "pipeline/kconv-unsupervised-profile": [
        "lgbm",
        "histgb",
        "random_forest_classifier",
        "random_forest_regressor",
    ],
}

EXPECTED_ENSEMBLE: dict[str, str] = {
    "pipeline/kozodoi-numeric-ratios": "weighted",
    "pipeline/kozodoi-time-profile": "weighted",
    "pipeline/kaggle-ieee-cluster-anomaly": "weighted",
    "pipeline/kaggle-ieee-time-profile": "weighted",
    "pipeline/nyaggle-cluster-stack": "stack",
    "pipeline/ensemble-amazon-numeric-cross": "stack",
    "pipeline/sergeyshilin-anomaly-contracts": "mean",
    "pipeline/dptools-cluster-profile": "weighted",
    "pipeline/kconv-unsupervised-profile": "weighted",
}


@pytest.mark.parametrize("recipe_id", sorted(EXPECTED_BLOCKS))
def test_advanced_pipeline_loads_with_declared_signature(recipe_id: str) -> None:
    recipe = load_recipe(recipe_id)

    assert recipe.kind == "pipeline"
    assert recipe.derived_from == DERIVED_FROM[recipe_id]
    assert recipe.pipeline is not None

    blocks = recipe.pipeline.blocks
    types = [block.type for block in blocks]
    assert types[0] == "base"
    assert types.count("base") == 1
    assert [(block.type, block.params) for block in blocks] == EXPECTED_BLOCKS[recipe_id]
    assert blocks[-1].type == "feature_selection"

    assert recipe.defaults.models == EXPECTED_MODELS[recipe_id]
    assert recipe.defaults.ensemble.method == EXPECTED_ENSEMBLE[recipe_id]
    assert recipe.attribution.strip()


def test_advanced_pipelines_keep_preset_attribution() -> None:
    recipes = {recipe.id: recipe for recipe in list_recipes()}
    for pipeline_id, preset_id in DERIVED_FROM.items():
        pipeline = load_recipe(pipeline_id)
        preset = recipes[preset_id]
        assert pipeline.derived_from == preset_id
        assert pipeline.author == preset.author
        assert pipeline.source == preset.source
        assert pipeline.license == preset.license


def test_pipeline_ids_are_unique_across_all_recipes() -> None:
    recipes = list_recipes()
    ids = [recipe.id for recipe in recipes]
    assert len(ids) == len(set(ids))

    pipeline_ids = [recipe.id for recipe in recipes if recipe.kind == "pipeline"]
    assert len(pipeline_ids) == len(set(pipeline_ids))
    assert set(EXPECTED_BLOCKS) <= set(pipeline_ids)
