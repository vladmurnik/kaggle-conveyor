"""Fault injection: one broken model must not discard healthy model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kconv.config import Config, DataConfig, ModelConfig, ValidationConfig, save_config
from kconv.errors import ModelError, StageError
from kconv.stages import train as train_stage
from kconv.stages.run import run_pipeline
from kconv.state import PipelineState


def _make_project(
    root: Path,
    models: list[ModelConfig],
) -> tuple[Path, Path, Config]:
    rng = np.random.default_rng(19)
    n_train, n_test = 80, 20
    n = n_train + n_test
    target = np.tile(np.array([0, 1]), n // 2)
    rng.shuffle(target)
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
            "target": target,
        }
    )
    data = root / "data"
    data.mkdir(parents=True)
    frame.iloc[:n_train].to_csv(data / "train.csv", index=False)
    frame.iloc[n_train:].drop(columns=["target"]).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": frame.iloc[n_train:]["id"], "target": 0}).to_csv(
        data / "sample_submission.csv", index=False
    )

    project = root / "project"
    project.mkdir()
    config = Config(
        project="robustness",
        seed=42,
        task="binary",
        metric="roc_auc",
        data=DataConfig(
            train=str(data / "train.csv"),
            test=str(data / "test.csv"),
            sample_submission=str(data / "sample_submission.csv"),
            target="target",
            id="id",
        ),
        validation=ValidationConfig(strategy="stratified_kfold", n_splits=2, random_state=42),
        models=models,
    )
    config_path = project / "config.yaml"
    save_config(config, config_path)
    return project, config_path, config


def _lgbm_model() -> ModelConfig:
    return ModelConfig(
        name="lgbm",
        seeds=1,
        params={"n_estimators": 30, "verbosity": -1, "num_threads": 1, "deterministic": True},
    )


class _FailingFitAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.task_types = {"binary", "multiclass", "regression"}
        self.family = "tree"
        self.task = "binary"
        self.warnings: list[str] = []

    def fit(self, *args, **kwargs):
        raise ModelError(
            "simulated optional package failure",
            "Install the optional package and rerun.",
        )

    def predict(self, X):
        raise AssertionError("predict must not run after a failed fit")

    def predict_proba(self, X):
        raise AssertionError("predict_proba must not run after a failed fit")

    def importance(self):
        return None


class _NonFiniteAdapter(_FailingFitAdapter):
    def fit(self, *args, **kwargs):
        self.model = object()
        return self

    def predict(self, X):
        return np.full(len(X), np.nan)

    def predict_proba(self, X):
        return np.full(len(X), np.nan)


class _ImportanceFailureAdapter:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = inner.name
        self.task_types = inner.task_types
        self.family = inner.family
        self.task = "binary"
        self.warnings: list[str] = []

    def fit(self, *args, **kwargs):
        self.inner.fit(*args, **kwargs)
        return self

    def predict(self, X):
        return self.inner.predict(X)

    def predict_proba(self, X):
        return self.inner.predict_proba(X)

    def importance(self):
        raise RuntimeError("importance exploded")


def test_one_model_failure_is_isolated_and_blend_continues(tmp_path, monkeypatch) -> None:
    project, config_path, config = _make_project(
        tmp_path,
        [_lgbm_model(), ModelConfig(name="tabpfn", seeds=1)],
    )
    real_create = train_stage.create_model

    def factory(name: str):
        return _FailingFitAdapter(name) if name == "tabpfn" else real_create(name)

    monkeypatch.setattr(train_stage, "create_model", factory)
    ctx = run_pipeline(project, config_path, config, until="blend")
    state = PipelineState.load(ctx.run_dir / "state.json")

    assert state.is_done("train")
    assert state.is_done("blend")
    train_result = state.stage("train").artifacts
    assert train_result["models"] == ["lgbm"]
    assert [item["model"] for item in train_result["failed_models"]] == ["tabpfn"]
    assert "simulated optional package failure" in train_result["failed_models"][0]["error"]
    assert not (ctx.run_dir / "02_train" / "tabpfn").exists()
    assert (ctx.run_dir / "03_blend" / "blend.json").is_file()
    assert any("tabpfn" in warning and "skipped" in warning for warning in ctx.warnings)


def test_all_models_failing_marks_train_failed_and_resume_recovers(tmp_path, monkeypatch) -> None:
    project, config_path, config = _make_project(tmp_path, [_lgbm_model()])

    with monkeypatch.context() as patch:
        patch.setattr(
            train_stage,
            "create_model",
            lambda name: _FailingFitAdapter(name),
        )
        with pytest.raises(StageError, match="All configured models failed"):
            run_pipeline(project, config_path, config, until="train")

    state = PipelineState.load(project / "runs" / _latest_run_id(project) / "state.json")
    assert state.stage("train").status == "failed"
    assert not state.is_done("train")
    latest = json.loads((project / "latest.json").read_text(encoding="utf-8"))
    assert latest["status"] == "failed"
    assert not (project / "runs" / state.run_id / "02_train" / "lgbm").exists()

    resumed = run_pipeline(project, config_path, config, until="blend", resume=True)
    resumed_state = PipelineState.load(resumed.run_dir / "state.json")
    assert resumed_state.is_done("train")
    assert resumed_state.is_done("blend")
    assert (resumed.run_dir / "02_train" / "lgbm" / "metrics.json").is_file()


def test_non_finite_predictions_are_rejected_and_model_is_skipped(
    tmp_path, monkeypatch
) -> None:
    project, config_path, config = _make_project(
        tmp_path,
        [_lgbm_model(), ModelConfig(name="tabpfn", seeds=1)],
    )
    real_create = train_stage.create_model

    def factory(name: str):
        return _NonFiniteAdapter(name) if name == "tabpfn" else real_create(name)

    monkeypatch.setattr(train_stage, "create_model", factory)
    ctx = run_pipeline(project, config_path, config, until="train")
    state = PipelineState.load(ctx.run_dir / "state.json")

    assert state.is_done("train")
    failed = state.stage("train").artifacts["failed_models"]
    assert [item["model"] for item in failed] == ["tabpfn"]
    assert "non-finite predictions" in failed[0]["error"]
    assert not (ctx.run_dir / "02_train" / "tabpfn").exists()


def test_importance_failure_does_not_discard_a_trained_model(tmp_path, monkeypatch) -> None:
    project, config_path, config = _make_project(tmp_path, [_lgbm_model()])
    real_create = train_stage.create_model

    def factory(name: str):
        return _ImportanceFailureAdapter(real_create(name))

    monkeypatch.setattr(train_stage, "create_model", factory)
    ctx = run_pipeline(project, config_path, config, until="train")
    state = PipelineState.load(ctx.run_dir / "state.json")

    assert state.is_done("train")
    assert state.stage("train").artifacts["models"] == ["lgbm"]
    assert (ctx.run_dir / "02_train" / "lgbm" / "metrics.json").is_file()
    assert any("feature importance unavailable" in warning for warning in ctx.warnings)


@pytest.mark.parametrize("name", ["../escape", "lgbm/../escape", "..\\escape"])
def test_model_name_cannot_escape_the_run_directory(tmp_path, name: str) -> None:
    project, config_path, config = _make_project(
        tmp_path,
        [ModelConfig(name=name, seeds=1)],
    )

    with pytest.raises(StageError, match="All configured models failed"):
        run_pipeline(project, config_path, config, until="train")

    run_dir = project / "runs" / _latest_run_id(project)
    state = PipelineState.load(run_dir / "state.json")
    assert "Invalid model name" in (state.stage("train").error or "")
    assert not (run_dir / "escape").exists()
    assert not (run_dir / "02_train" / "escape").exists()


def _latest_run_id(project: Path) -> str:
    payload = json.loads((project / "latest.json").read_text(encoding="utf-8"))
    return str(payload["run_id"])
