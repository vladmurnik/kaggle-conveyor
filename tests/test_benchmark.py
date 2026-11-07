"""Benchmark registry, materialization and scorecard tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from kconv.benchmark import load_benchmark_registry, run_benchmark, write_scorecard
from kconv.cli import app
from kconv.errors import ConfigError


def test_default_registry_has_four_datasets() -> None:
    registry = load_benchmark_registry()
    assert [dataset.id for dataset in registry.datasets] == [
        "local-binary",
        "kgmon-regression",
        "wine-multiclass",
        "synthetic-contract",
    ]
    local = registry.get("local-binary")
    assert local.kind == "local"
    assert local.task == "binary"
    assert local.metric == "roc_auc"
    assert local.sample_fraction == 0.10


def test_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    payload = {
        "datasets": [
            {
                "id": "same",
                "title": "One",
                "kind": "synthetic",
                "task": "binary",
                "metric": "roc_auc",
                "target": "target",
                "source": "test",
                "license": "test",
            },
            {
                "id": "same",
                "title": "Two",
                "kind": "synthetic",
                "task": "binary",
                "metric": "roc_auc",
                "target": "target",
                "source": "test",
                "license": "test",
            },
        ]
    }
    path = tmp_path / "datasets.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="Duplicate benchmark dataset id"):
        load_benchmark_registry(path, root=tmp_path)


def test_synthetic_materialization(tmp_path: Path) -> None:
    registry = load_benchmark_registry()
    prepared = registry.get("synthetic-contract").materialize(tmp_path / "work", registry.root)
    assert prepared.rows_train == 110
    assert prepared.rows_test == 40
    assert prepared.target == "target"
    assert prepared.train_path.exists()
    assert prepared.test_path.exists()
    assert prepared.sample_submission_path.exists()
    train = pd.read_csv(prepared.train_path)
    test = pd.read_csv(prepared.test_path)
    sample = pd.read_csv(prepared.sample_submission_path)
    assert "target" in train.columns
    assert "target" not in test.columns
    assert list(sample.columns) == ["id", "target"]


def test_benchmark_smoke_and_scorecard(tmp_path: Path) -> None:
    output = tmp_path / "scorecards"
    run = run_benchmark(
        "synthetic-contract",
        recipes=["pesnik/baseline"],
        strategy="fast",
        n_splits=2,
        output_dir=output,
        keep_artifacts=True,
    )
    assert run.ok
    assert len(run.results) == 1
    result = run.results[0]
    assert result.status == "ok"
    assert result.mean is not None
    assert len(result.fold_scores) == 2
    assert result.submission_valid
    assert result.n_train == 110
    assert result.n_test == 40
    json_path = output / "scorecard.json"
    markdown_path = output / "scorecard.md"
    assert json_path.exists()
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["results"][0]["recipe_id"] == "pesnik/baseline"
    assert json_path.read_text(encoding="utf-8").isascii()
    assert markdown_path.read_text(encoding="utf-8").isascii()


def test_write_scorecard_accepts_explicit_directory(tmp_path: Path) -> None:
    run = run_benchmark(
        "synthetic-contract",
        recipes=["pesnik/baseline"],
        strategy="fast",
        n_splits=2,
        output_dir=tmp_path / "first",
        keep_artifacts=True,
    )
    target = write_scorecard(run, tmp_path / "second")
    assert (target / "scorecard.json").exists()
    assert (target / "scorecard.md").exists()


def test_benchmark_cli_list_json() -> None:
    result = CliRunner().invoke(app, ["benchmark", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [item["id"] for item in payload] == [
        "local-binary",
        "kgmon-regression",
        "wine-multiclass",
        "synthetic-contract",
    ]
    assert result.stdout.isascii()


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown benchmark dataset"):
        run_benchmark("does-not-exist", recipes=["pesnik/baseline"])


def test_unknown_recipe_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown benchmark recipe"):
        run_benchmark("synthetic-contract", recipes=["does-not-exist"])


def test_registry_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "datasets": [
            {
                "id": "env-contract",
                "title": "Env contract",
                "kind": "synthetic",
                "task": "binary",
                "metric": "roc_auc",
                "target": "target",
                "source": "test",
                "license": "test",
            }
        ]
    }
    path = tmp_path / "env-datasets.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setenv("KCONV_BENCHMARK_REGISTRY", str(path))
    registry = load_benchmark_registry()
    assert registry.path == path.resolve()
    assert registry.get("env-contract").title == "Env contract"
