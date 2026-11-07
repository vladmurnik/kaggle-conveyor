"""Quick-win regressions: postmortems, diagnostics, preflight and atomic IO."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kconv.atomic import (
    dump_joblib_atomic,
    verify_artifacts,
    write_artifact_manifest,
    write_parquet_atomic,
)
from kconv.cli import app
from kconv.config import Config, DataConfig, ModelConfig, RuntimeConfig, ValidationConfig, save_config
from kconv.context import read_json, write_json
from kconv.errors import ConfigError, ModelError, StageError
from kconv.models.validation import validate_model_params
from kconv.stages import run as run_stage
from kconv.stages import train as train_stage
from kconv.stages.run import run_pipeline


def _make_project(
    project: Path,
    models: list[ModelConfig] | None = None,
) -> tuple[Path, Config]:
    rng = np.random.default_rng(31)
    n_train, n_test = 80, 20
    n = n_train + n_test
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
            "target": (rng.random(n) < 0.5).astype(int),
        }
    )
    data = project / "data"
    data.mkdir(parents=True)
    frame.iloc[:n_train].to_csv(data / "train.csv", index=False)
    frame.iloc[n_train:].drop(columns=["target"]).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": frame.iloc[n_train:]["id"], "target": 0}).to_csv(
        data / "sample_submission.csv", index=False
    )
    config = Config(
        project="quick-wins",
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
        models=models
        or [
            ModelConfig(
                name="lgbm",
                seeds=1,
                params={"n_estimators": 30, "verbosity": -1, "num_threads": 1, "deterministic": True},
            )
        ],
    )
    config_path = project / "config.yaml"
    save_config(config, config_path)
    return config_path, config


def test_failed_stage_writes_postmortem_and_diagnostics(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path, config = _make_project(project)
    real_loader = run_stage._load_stage_runner

    def boom(ctx):
        raise RuntimeError("simulated validator crash")

    def loader(name: str):
        return boom if name == "validate" else real_loader(name)

    monkeypatch.setattr(run_stage, "_load_stage_runner", loader)
    with pytest.raises(StageError, match="validate"):
        run_pipeline(project, config_path, config, until="validate")

    run_id = json.loads((project / "latest.json").read_text(encoding="utf-8"))["run_id"]
    run_dir = project / "runs" / run_id
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["stage"] == "validate"
    assert failure["exception"]["type"] == "RuntimeError"
    assert "simulated validator crash" in failure["exception"]["message"]
    assert "RuntimeError" in failure["exception"]["traceback"]
    assert "resume_hint" in failure
    assert (run_dir / "failure_report.md").read_text(encoding="utf-8").startswith("# kconv failure report")
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["stage_status"]["validate"] == "failed"
    run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_meta["status"] == "failed"
    assert run_meta["failed_stage"] == "validate"
    assert run_meta["failure_report"] == "failure_report.md"


def test_resume_merges_persisted_warnings_and_failed_models(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path, config = _make_project(project)
    first = run_pipeline(project, config_path, config, until="validate")
    run_meta = read_json(first.run_dir / "run.json")
    run_meta["warnings"] = ["previous warning"]
    run_meta["failed_models"] = [{"model": "lgbm", "error": "old failure"}]
    write_json(first.run_dir / "run.json", run_meta)

    resumed = run_pipeline(project, config_path, config, until="validate", resume=True)
    merged = read_json(resumed.run_dir / "run.json")
    assert "previous warning" in merged["warnings"]
    assert merged["failed_models"] == [{"model": "lgbm", "error": "old failure"}]
    diagnostics = read_json(resumed.run_dir / "diagnostics.json")
    assert "previous warning" in diagnostics["warnings"]
    assert diagnostics["failed_models"][0]["model"] == "lgbm"


class _BrokenAdapter:
    name = "tabpfn"
    task_types = {"binary", "multiclass", "regression"}
    family = "tree"
    task = "binary"

    def fit(self, *args, **kwargs):
        raise ModelError("simulated model crash", "This is a test adapter.")


def test_completed_with_warnings_when_one_model_is_skipped(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path, config = _make_project(
        project,
        [
            ModelConfig(
                name="lgbm",
                seeds=1,
                params={"n_estimators": 30, "verbosity": -1, "num_threads": 1, "deterministic": True},
            ),
            ModelConfig(name="tabpfn", seeds=1),
        ],
    )
    real_create = train_stage.create_model

    def factory(name: str):
        return _BrokenAdapter() if name == "tabpfn" else real_create(name)

    monkeypatch.setattr(train_stage, "create_model", factory)
    ctx = run_pipeline(project, config_path, config, until="report")
    run_meta = read_json(ctx.run_dir / "run.json")
    assert run_meta["status"] == "completed_with_warnings"
    assert [item["model"] for item in run_meta["failed_models"]] == ["tabpfn"]
    assert "simulated model crash" in run_meta["failed_models"][0]["error"]
    diagnostics = read_json(ctx.run_dir / "diagnostics.json")
    assert diagnostics["failed_models"][0]["model"] == "tabpfn"
    report = (ctx.run_dir / "05_report" / "report.md").read_text(encoding="utf-8")
    assert "Skipped models" in report
    assert "tabpfn" in report


def test_doctor_json_reports_config_param_typo(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path, config = _make_project(project)
    config.models[0].params = {"n_estimator": 10}
    save_config(config, config_path)
    result = CliRunner().invoke(app, ["doctor", "--config", str(config_path), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any("n_estimator" in check["message"] for check in payload["checks"])


def test_model_param_validation_rejects_typos_and_ranges() -> None:
    with pytest.raises(ConfigError, match="n_estimator"):
        validate_model_params("lgbm", {"n_estimator": 10})
    with pytest.raises(ConfigError, match=">= 1"):
        validate_model_params("lgbm", {"n_estimators": 0})
    with pytest.raises(ConfigError, match="<= 1.0"):
        validate_model_params("lgbm", {"learning_rate": 2.0})
    with pytest.raises(ConfigError, match="Unknown parameter"):
        validate_model_params("random_forest_classifier", {"n_estimator": 10})
    assert validate_model_params("lgbm", {"n_estimators": 10, "verbosity": -1}) == {
        "n_estimators": 10,
        "verbosity": -1,
    }


class _BadProbabilityAdapter:
    name = "bad"

    def __init__(self, values):
        self.values = values

    def predict_proba(self, X):
        return self.values


def test_prediction_sanity_rejects_invalid_probabilities() -> None:
    X = pd.DataFrame({"x": [0.0, 1.0]})
    with pytest.raises(ModelError, match="outside \\[0, 1\\]"):
        train_stage._predict(_BadProbabilityAdapter(np.array([0.1, 1.4])), X, "binary")
    with pytest.raises(ModelError, match="do not sum to 1"):
        train_stage._predict(
            _BadProbabilityAdapter(np.array([[0.2, 0.2], [0.3, 0.3]])),
            X,
            "binary",
        )
    with pytest.raises(ModelError, match="training target has 3 classes"):
        train_stage._predict(
            _BadProbabilityAdapter(np.array([[0.2, 0.8], [0.3, 0.7]])),
            X,
            "multiclass",
            n_classes=3,
        )


def test_atomic_artifact_checksum_detects_corruption(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "02_train" / "lgbm" / "oof.parquet"
    frame = pd.DataFrame({"row_id": [0, 1], "pred": [0.1, 0.9]})
    write_parquet_atomic(frame, path)
    model_path = run_dir / "model.joblib"
    dump_joblib_atomic({"value": 1}, model_path)
    write_artifact_manifest(run_dir, [path, model_path])
    assert verify_artifacts(run_dir) == []
    path.write_bytes(b"corrupt")
    errors = verify_artifacts(run_dir)
    assert errors and "checksum mismatch" in errors[0]



def test_feature_cache_reuses_fold_features_and_recovers_from_corruption(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path, config = _make_project(project)
    first = run_pipeline(project, config_path, config, until="train")
    second = run_pipeline(project, config_path, config, until="train")
    second_metrics = read_json(second.run_dir / "02_train" / "lgbm" / "metrics.json")
    assert second_metrics["feature_cache_hits"] == 2
    assert second_metrics["feature_cache_misses"] == 0
    cache_dirs = [path for path in (project / ".cache" / "features").iterdir() if path.is_dir()]
    assert len(cache_dirs) == 2
    (cache_dirs[0] / "train.parquet").write_bytes(b"corrupt")
    third = run_pipeline(project, config_path, config, until="train")
    third_metrics = read_json(third.run_dir / "02_train" / "lgbm" / "metrics.json")
    assert third_metrics["feature_cache_misses"] >= 1
    assert first.run_id != second.run_id != third.run_id


def test_model_timeout_is_reported_as_timed_out(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path, config = _make_project(project)
    config.runtime = RuntimeConfig(profile=1, model_timeout_sec=1)
    real_fit = train_stage.fit_with_timeout

    def fake_fit(adapter, *args, **kwargs):
        if getattr(adapter, "name", None) == "lgbm":
            raise TimeoutError("simulated model timeout")
        return real_fit(adapter, *args, **kwargs)

    monkeypatch.setattr(train_stage, "fit_with_timeout", fake_fit)
    with pytest.raises(StageError, match="train"):
        run_pipeline(project, config_path, config, until="train")
    run_id = json.loads((project / "latest.json").read_text(encoding="utf-8"))["run_id"]
    run_meta = read_json(project / "runs" / run_id / "run.json")
    assert run_meta["failed_models"][0]["model"] == "lgbm"
    assert run_meta["failed_models"][0]["status"] == "timed_out"


def test_fallback_model_runs_when_primary_model_fails(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path, config = _make_project(project)
    config.runtime = RuntimeConfig(profile=1, fallback_models=["histgb"])
    real_create = train_stage.create_model

    def factory(name: str):
        return _BrokenAdapter() if name == "lgbm" else real_create(name)

    monkeypatch.setattr(train_stage, "create_model", factory)
    ctx = run_pipeline(project, config_path, config, until="train")
    assert (ctx.run_dir / "02_train" / "histgb" / "metrics.json").is_file()
    run_meta = read_json(ctx.run_dir / "run.json")
    assert any(item["model"] == "lgbm" for item in run_meta["failed_models"])


def test_stage_timeout_runs_ingest_in_child_and_merges_context(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path, config = _make_project(project)
    config.runtime = RuntimeConfig(profile=1, stage_timeout_sec=60)
    ctx = run_pipeline(project, config_path, config, until="ingest")
    assert ctx.data_hash
    assert ctx.dataset is not None
    assert (ctx.run_dir / "00_ingest" / "train.parquet").is_file()
