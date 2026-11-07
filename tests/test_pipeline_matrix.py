"""Every built-in recipe must build and run for every task it declares."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kconv.api import list_pipelines
from kconv.config import load_config, save_config
from kconv.models import create_model
from kconv.project import build_config
from kconv.recipes import load_recipe
from kconv.stages.run import run_pipeline
from kconv.state import PipelineState

TASKS = ("binary", "multiclass", "regression")
RECIPES = tuple(pipeline.id for pipeline in list_pipelines())
PIPELINE_STAGES = ("ingest", "validate", "hpo", "train", "blend", "submit", "report")
DEFAULT_METRICS = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
CHEAP_PARAMS: dict[str, dict[str, object]] = {
    "lgbm": {"n_estimators": 40, "verbosity": -1, "num_threads": 1, "deterministic": True},
    "xgb": {"n_estimators": 40, "n_jobs": 1, "verbosity": 0},
    "catboost": {"iterations": 40, "verbose": 0, "thread_count": 1},
    "histgb": {"max_iter": 40},
    "logreg": {"max_iter": 1000},
    "ridge": {"alpha": 1.0},
    "random_forest_classifier": {"n_estimators": 20, "n_jobs": 1},
    "random_forest_regressor": {"n_estimators": 20, "n_jobs": 1},
}


def _write_dataset(root: Path, task: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    n = 150
    if task == "binary":
        target = (rng.random(n) > 0.5).astype(int)
    elif task == "multiclass":
        target = np.tile(np.arange(3), n // 3)
    else:
        target = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], n),
            "cat2": rng.choice(["u", "v", "w"], n),
            "when": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "target": target,
        }
    )
    train = frame.iloc[:110].copy()
    test = frame.iloc[110:].drop(columns=["target"]).copy()
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    if task == "multiclass":
        sample = pd.DataFrame(
            {
                "id": test["id"].to_numpy(),
                "class_0": 1.0 / 3.0,
                "class_1": 1.0 / 3.0,
                "class_2": 1.0 / 3.0,
            }
        )
    else:
        sample = pd.DataFrame({"id": test["id"], "target": 0})
    sample.to_csv(root / "sample_submission.csv", index=False)
    return root


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("recipe_id", RECIPES)
def test_recipe_builds_for_every_declared_task(tmp_path, monkeypatch, recipe_id: str, task: str) -> None:
    recipe = load_recipe(recipe_id)
    if task not in recipe.tasks:
        pytest.skip(f"{recipe_id} does not declare task {task}")
    monkeypatch.chdir(tmp_path)
    data = _write_dataset(tmp_path / "data", task)
    project_dir, config_path, cfg = build_config(
        data,
        target="target",
        recipe_id=recipe_id,
        strategy="fast",
        task=task,
        name=f"build-{task}-{recipe_id.replace('/', '-')}",
    )

    assert project_dir.is_dir()
    assert config_path.is_file()
    assert cfg.recipe == recipe_id
    assert cfg.task == task
    assert cfg.metric == DEFAULT_METRICS[task]
    assert cfg.validation.n_splits == 3
    assert cfg.hpo.enabled is False
    assert {model.seeds for model in cfg.models} == {1}
    assert cfg.ensemble == recipe.defaults.ensemble
    assert cfg.validation.adversarial == recipe.defaults.validation.adversarial
    assert (
        cfg.validation.adversarial_threshold
        == recipe.defaults.validation.adversarial_threshold
    )
    assert cfg.models
    for model in cfg.models:
        assert model.name in recipe.defaults.models
        assert task in create_model(model.name).task_types

    if recipe_id == "kconv/sklearn-diverse":
        names = {model.name for model in cfg.models}
        if task == "regression":
            assert "random_forest_classifier" not in names
        else:
            assert "random_forest_regressor" not in names

    loaded = load_config(config_path)
    assert loaded.models == cfg.models
    assert loaded.ensemble == cfg.ensemble


@pytest.mark.slow
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("recipe_id", RECIPES)
def test_recipe_pipeline_runs_for_every_declared_task(
    tmp_path, monkeypatch, recipe_id: str, task: str
) -> None:
    recipe = load_recipe(recipe_id)
    if task not in recipe.tasks:
        pytest.skip(f"{recipe_id} does not declare task {task}")
    monkeypatch.chdir(tmp_path)
    data = _write_dataset(tmp_path / "data", task)
    project_dir, config_path, cfg = build_config(
        data,
        target="target",
        recipe_id=recipe_id,
        strategy="fast",
        task=task,
        name=f"run-{task}-{recipe_id.replace('/', '-')}",
    )
    cfg.validation.n_splits = 2
    cfg.validation.adversarial = False
    for model in cfg.models:
        model.params.update(CHEAP_PARAMS.get(model.name, {}))
    save_config(cfg, config_path)

    ctx = run_pipeline(project_dir, config_path, cfg, until="report")
    state = PipelineState.load(ctx.run_dir / "state.json")
    assert all(state.is_done(stage) for stage in PIPELINE_STAGES), state.summary()

    blend = json.loads((ctx.run_dir / "03_blend" / "blend.json").read_text(encoding="utf-8"))
    assert blend["task"] == task
    assert np.isfinite(float(blend["oof_score"]))
    assert blend["method"] in {"best_single", "mean", "rank_mean", "weighted", "stack"}

    oof = pd.read_parquet(ctx.run_dir / "03_blend" / "oof.parquet")
    test = pd.read_parquet(ctx.run_dir / "03_blend" / "test_pred.parquet")
    assert len(oof) == 110
    assert len(test) == 40
    assert not oof.isna().any().any()
    assert not test.isna().any().any()
    if task == "multiclass":
        prediction_columns = [column for column in oof.columns if column.startswith("pred_")]
        assert len(prediction_columns) == 3
        np.testing.assert_allclose(oof[prediction_columns].sum(axis=1), 1.0, atol=1e-5)

    submission = pd.read_csv(ctx.run_dir / "04_submit" / "submission.csv")
    assert len(submission) == 40
    assert not submission.isna().any().any()
    if task == "multiclass":
        probability_columns = [column for column in submission.columns if column != "id"]
        assert len(probability_columns) == 3
        np.testing.assert_allclose(submission[probability_columns].sum(axis=1), 1.0, atol=1e-5)
    else:
        assert set(submission.columns) == {"id", "target"}
    report = (ctx.run_dir / "05_report" / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# kconv report")
