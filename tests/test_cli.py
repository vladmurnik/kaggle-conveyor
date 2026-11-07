"""CLI smoke tests with a tiny synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kconv.cli import app
from kconv.config import load_config


def _dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    n = 60
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "x": rng.normal(size=n),
            "cat": rng.choice(["a", "b"], n),
            "target": (rng.random(n) > 0.5).astype(int),
        }
    )
    train = frame.iloc[:45].copy()
    test = frame.iloc[45:].drop(columns=["target"]).copy()
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "target": 0}).to_csv(root / "sample_submission.csv", index=False)
    return root


def test_help_and_lists() -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["recipes"]).exit_code == 0
    assert runner.invoke(app, ["recipes", "--group", "baseline"]).exit_code == 0
    assert runner.invoke(app, ["recipes", "--group", "ensemble", "--json"]).exit_code == 0
    assert runner.invoke(app, ["recipes", "show", "pesnik/baseline"]).exit_code == 0
    assert runner.invoke(app, ["models"]).exit_code == 0
    assert runner.invoke(app, ["models", "--details"]).exit_code == 0
    assert runner.invoke(app, ["sources", "--limit", "2"]).exit_code == 0
    assert runner.invoke(app, ["sources", "--all", "--limit", "2"]).exit_code == 0
    assert runner.invoke(app, ["sources", "show", "arnabbiswas1__kaggle_pipeline_tps_aug_22"]).exit_code == 0
    assert runner.invoke(app, ["plugins"]).exit_code == 0
    assert runner.invoke(app, ["metrics"]).exit_code == 0
    assert runner.invoke(app, ["doctor"]).exit_code == 0


def test_web_requires_allow_remote_for_non_loopback_host(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Rich truncates the options table on very narrow terminals (CI defaults
    # to 40 columns), so pin the width before rendering help.
    monkeypatch.setenv("COLUMNS", "120")
    runner = CliRunner()
    help_result = runner.invoke(app, ["web", "--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "--allow-remote" in help_result.output

    result = runner.invoke(app, ["web", "--host", "0.0.0.0", "--port", "0", "--no-open"])
    assert result.exit_code != 0
    assert "--allow-remote" in result.output


def test_json_commands_are_ascii_safe() -> None:
    result = CliRunner().invoke(app, ["sources", "--all", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, list) and payload
    assert result.output.isascii()


def test_one_liner_no_run_creates_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "target",
            "--recipe",
            "pesnik/baseline",
            "--models",
            "lgbm",
            "--name",
            "smoke",
            "--no-run",
        ],
    )
    assert result.exit_code == 0, result.output
    config = tmp_path / "competitions" / "smoke" / "config.yaml"
    assert config.exists()
    validate = runner.invoke(app, ["config", "validate", "--config", str(config)])
    assert validate.exit_code == 0, validate.output
    validate_comp = runner.invoke(app, ["config", "validate", "--comp", "smoke"])
    assert validate_comp.exit_code == 0, validate_comp.output


def test_run_accepts_resource_and_model_param_overrides(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "target",
            "--recipe",
            "pesnik/baseline",
            "--models",
            "lgbm",
            "--name",
            "tuned-cli",
            "--profile",
            "3",
            "--device",
            "cpu",
            "--cpu-threads",
            "2",
            "--ram-gb",
            "12",
            "--vram-gb",
            "0",
            "--model-params",
            '{"lgbm": {"n_estimators": 123, "learning_rate": 0.03}}',
            "--no-run",
        ],
    )
    assert result.exit_code == 0, result.output
    config = load_config(tmp_path / "competitions" / "tuned-cli" / "config.yaml")
    assert config.runtime is not None
    assert config.runtime.cpu_threads == 2
    assert config.runtime.ram_gb == 12
    assert config.runtime.vram_gb == 0
    lgbm = next(model for model in config.models if model.name == "lgbm")
    assert lgbm.params["n_estimators"] == 123
    assert lgbm.params["learning_rate"] == 0.03
    assert lgbm.params["num_threads"] == 2

    rejected = runner.invoke(
        app,
        ["run", "--config", str(tmp_path / "competitions" / "tuned-cli" / "config.yaml"), "--cpu-threads", "2"],
    )
    assert rejected.exit_code != 0
    assert "require --data" in rejected.output

    bad_json = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "target",
            "--recipe",
            "pesnik/baseline",
            "--models",
            "lgbm",
            "--model-params",
            "{bad",
            "--no-run",
        ],
    )
    assert bad_json.exit_code != 0
    assert "not valid JSON" in bad_json.output


def test_interactive_new_creates_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["new"],
        input=f"{data}\ntarget\n1\nfast\nlgbm\nwizard\nn\n",
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "competitions" / "wizard" / "config.yaml").exists()



def test_runs_compare_best_and_tag_commands(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    from kconv.context import write_json

    for run_id, score in (("run-1", 0.7), ("run-2", 0.9)):
        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True)
        write_json(
            run_dir / "run.json",
            {
                "run_id": run_id,
                "project": tmp_path.name,
                "status": "completed",
                "started_at": f"2026-01-0{1 if run_id == 'run-1' else 2}T00:00:00+00:00",
                "config": {
                    "project": tmp_path.name,
                    "recipe": "pesnik/baseline",
                    "strategy": "fast",
                    "task": "binary",
                    "metric": "roc_auc",
                    "models": [{"name": "lgbm", "enabled": True}],
                },
            },
        )
        write_json(run_dir / "03_blend" / "blend.json", {"oof_score": score, "metric": "roc_auc"})

    runner = CliRunner()
    listed = runner.invoke(app, ["runs", "--json"])
    assert listed.exit_code == 0, listed.output
    assert len(json.loads(listed.output)["runs"]) == 2

    compared = runner.invoke(app, ["compare", "run-1", "run-2", "--json"])
    assert compared.exit_code == 0, compared.output
    assert json.loads(compared.output)["score_delta"] == pytest.approx(0.2)

    best = runner.invoke(app, ["best", "--metric", "roc_auc", "--json"])
    assert best.exit_code == 0, best.output
    assert json.loads(best.output)["run_id"] == "run-2"

    tagged = runner.invoke(app, ["tag", "run-1", "baseline", "fast", "--json"])
    assert tagged.exit_code == 0, tagged.output
    assert json.loads(tagged.output)["tags"] == ["baseline", "fast"]
