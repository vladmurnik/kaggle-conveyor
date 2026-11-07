"""Nested HPO stage tests on a tiny synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kconv.config import Config, DataConfig, HPOConfig, ModelConfig, ValidationConfig
from kconv.errors import ModelError
from kconv.stages import hpo as hpo_stage
from kconv.stages.run import run_pipeline
from kconv.state import PipelineState


def _make_competition(root: Path) -> None:
    rng = np.random.default_rng(3)
    n = 240
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c", "d"], n),
            "target": (rng.random(n) > 0.5).astype(int),
        }
    )
    data = root / "data"
    data.mkdir(parents=True)
    frame.iloc[:180].to_csv(data / "train.csv", index=False)
    frame.iloc[180:].drop(columns=["target"]).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": frame.iloc[180:]["id"], "target": 0}).to_csv(data / "sample_submission.csv", index=False)


def _config(tmp_path: Path, *, trials: int, timeout_sec: int = 60) -> Config:
    return Config(
        project="hpo",
        seed=42,
        task="binary",
        metric="roc_auc",
        strategy="deep",
        data=DataConfig(
            train=str(tmp_path / "data" / "train.csv"),
            test=str(tmp_path / "data" / "test.csv"),
            sample_submission=str(tmp_path / "data" / "sample_submission.csv"),
            target="target",
            id="id",
        ),
        validation=ValidationConfig(strategy="stratified_kfold", n_splits=2, random_state=42),
        models=[ModelConfig(name="lgbm", seeds=1, params={"verbosity": -1, "num_threads": 1})],
        hpo=HPOConfig(enabled=True, trials=trials, timeout_sec=timeout_sec, inner_size=0.25),
    )


def test_hpo_writes_per_fold_params_and_train_uses_them(tmp_path: Path) -> None:
    _make_competition(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    config = _config(tmp_path, trials=2)
    ctx = run_pipeline(project, project / "config.yaml", config, until="train")
    state = PipelineState.load(ctx.run_dir / "state.json")
    assert state.is_done("hpo")
    assert state.is_done("train")

    tuning_path = ctx.run_dir / "01_hpo" / "tuning.json"
    payload = json.loads(tuning_path.read_text(encoding="utf-8"))
    assert payload["enabled"] is True
    assert set(payload["folds"]) == {"0", "1"}
    for fold in payload["folds"].values():
        assert "lgbm" in fold
        assert isinstance(fold["lgbm"]["params"], dict)
        assert fold["lgbm"]["trials_run"] >= 1

    metrics = json.loads((ctx.run_dir / "02_train" / "lgbm" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["hpo_applied"] is True
    assert isinstance(metrics["params"], dict)


def test_hpo_timeout_falls_back_to_defaults(tmp_path: Path, monkeypatch) -> None:
    _make_competition(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    config = _config(tmp_path, trials=3, timeout_sec=10)
    ticks = {"count": 0}

    def fake_clock() -> float:
        ticks["count"] += 1
        return 0.0 if ticks["count"] <= 1 else 100.0

    monkeypatch.setattr(hpo_stage, "time", SimpleNamespace(perf_counter=fake_clock))
    ctx = run_pipeline(project, project / "config.yaml", config, until="hpo")

    payload = json.loads((ctx.run_dir / "01_hpo" / "tuning.json").read_text(encoding="utf-8"))
    assert payload["enabled"] is True
    assert payload["timeout_hit"] is True
    assert payload["folds"]["0"]["lgbm"]["params"] == {"verbosity": -1, "num_threads": 1}
    assert payload["folds"]["0"]["lgbm"]["used_default"] is True


def test_hpo_fit_timeout_stops_trials(tmp_path: Path, monkeypatch) -> None:
    _make_competition(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    config = _config(tmp_path, trials=3)
    calls = {"count": 0}

    def fake_fit(*args, **kwargs):
        calls["count"] += 1
        raise TimeoutError("simulated fit timeout")

    monkeypatch.setattr(hpo_stage, "fit_with_timeout", fake_fit)
    ctx = run_pipeline(project, project / "config.yaml", config, until="hpo")

    payload = json.loads((ctx.run_dir / "01_hpo" / "tuning.json").read_text(encoding="utf-8"))
    assert payload["timeout_hit"] is True
    fold = payload["folds"]["0"]["lgbm"]
    assert fold["used_default"] is True
    assert fold["trials_run"] == 1
    assert "simulated fit timeout" in fold["detail"]["last_error"]
    assert calls["count"] == 1


def test_hpo_setup_failure_degrades_to_defaults(tmp_path, monkeypatch) -> None:
    _make_competition(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    config = _config(tmp_path, trials=1)

    def broken_factory(name: str):
        raise ModelError("simulated unavailable optional model", "Install the extra.")

    monkeypatch.setattr(hpo_stage, "create_model", broken_factory)
    ctx = run_pipeline(project, project / "config.yaml", config, until="hpo")

    state = PipelineState.load(ctx.run_dir / "state.json")
    assert state.is_done("hpo")
    payload = json.loads((ctx.run_dir / "01_hpo" / "tuning.json").read_text(encoding="utf-8"))
    fold = payload["folds"]["0"]["lgbm"]
    assert fold["used_default"] is True
    assert fold["detail"]["reason"] == "HPO setup failed"
    assert "simulated unavailable optional model" in fold["detail"]["last_error"]
    assert any("HPO skipped for 'lgbm'" in warning for warning in ctx.warnings)
