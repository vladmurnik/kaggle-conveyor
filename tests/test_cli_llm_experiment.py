"""CLI wiring for the LLM feature experiment: propose -> approve -> run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from kconv.cli import app
from kconv.config import load_config
from kconv.io import read_table


def _dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n = 80
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "target": (rng.random(n) > 0.5).astype(int),
        }
    )
    frame.iloc[:60].to_csv(root / "train.csv", index=False)
    frame.iloc[60:].drop(columns=["target"]).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": frame.iloc[60:]["id"], "target": 0}).to_csv(root / "sample_submission.csv", index=False)
    return root


def test_propose_and_run_with_llm_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    plan = tmp_path / "plan.json"
    augmented = tmp_path / "augmented.csv"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "experiments",
            "propose",
            "--data",
            str(data),
            "--target",
            "target",
            "--provider",
            "mock",
            "--output",
            str(plan),
            "--apply",
            str(augmented),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(plan.read_text(encoding="utf-8"))
    assert payload["features"]
    frame = pd.read_csv(augmented)
    assert {"llm_sum_1", "llm_ratio_1"} <= set(frame.columns)

    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "target",
            "--experiment",
            "llm-feature-proposals",
            "--approve-llm-features",
            "--llm-plan-file",
            str(plan),
            "--models",
            "lgbm",
            "--name",
            "llm-smoke",
            "--no-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "competitions" / "llm-smoke" / "config.yaml").exists()

    auto_named = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "target",
            "--experiment",
            "llm-feature-proposals",
            "--approve-llm-features",
            "--llm-plan-file",
            str(plan),
            "--models",
            "lgbm",
            "--no-run",
        ],
    )
    assert auto_named.exit_code == 0, auto_named.output
    config_file = tmp_path / "competitions" / "data" / "config.yaml"
    assert config_file.exists()
    config = load_config(config_file)
    assert config.features.fold_plan is not None
    planned = {item["name"] for item in config.features.fold_plan["features"]}
    assert {"llm_sum_1", "llm_ratio_1"} <= planned
    assert (config_file.parent / config.data.train).resolve() == (data / "train.csv").resolve()
    assert config.data.test is not None
    assert (config_file.parent / config.data.test).resolve() == (data / "test.csv").resolve()
    train = read_table((config_file.parent / config.data.train).resolve())
    assert {"llm_sum_1", "llm_ratio_1"}.isdisjoint(train.columns)
    test = read_table((config_file.parent / config.data.test).resolve())
    assert {"llm_sum_1", "llm_ratio_1"}.isdisjoint(test.columns)


def test_run_requires_approval_and_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    runner = CliRunner()
    base = [
        "run",
        "--data",
        str(data),
        "--target",
        "target",
        "--experiment",
        "llm-feature-proposals",
        "--name",
        "blocked",
        "--no-run",
    ]
    no_approval = runner.invoke(app, base)
    assert no_approval.exit_code != 0
    with_approval = runner.invoke(app, [*base, "--approve-llm-features"])
    assert with_approval.exit_code != 0
    unknown = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "target",
            "--experiment",
            "rank-blend",
            "--approve-llm-features",
            "--name",
            "unsupported",
            "--no-run",
        ],
    )
    assert unknown.exit_code != 0
    orphan_flag = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "target",
            "--llm-plan-file",
            str(tmp_path / "missing.json"),
            "--name",
            "orphan",
            "--no-run",
        ],
    )
    assert orphan_flag.exit_code != 0


def test_plan_provider_filters_unsafe_expressions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "features": [
                    {"name": "safe_sum", "kind": "numeric", "expression": "a + b", "rationale": "safe"},
                    {"name": "uses_target", "kind": "numeric", "expression": "target + a", "rationale": "leak"},
                    {"name": "calls_code", "kind": "numeric", "expression": "__import__('os')", "rationale": "code"},
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "propose",
            "--data",
            str(data),
            "--target",
            "target",
            "--provider",
            "plan",
            "--plan-file",
            str(plan),
            "--output",
            str(tmp_path / "filtered.json"),
            "--apply",
            str(tmp_path / "filtered.csv"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    filtered = json.loads((tmp_path / "filtered.json").read_text(encoding="utf-8"))
    assert [feature["name"] for feature in filtered["features"]] == ["safe_sum"]
    columns = (tmp_path / "filtered.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "safe_sum" in columns
    assert "uses_target" not in columns and "calls_code" not in columns

    evil = tmp_path / "evil.json"
    evil.write_text(
        json.dumps(
            {
                "features": [
                    {"name": "calls_code", "kind": "numeric", "expression": "__import__('os')", "rationale": "code"}
                ]
            }
        ),
        encoding="utf-8",
    )
    blocked = runner.invoke(
        app,
        [
            "experiments",
            "propose",
            "--data",
            str(data),
            "--target",
            "target",
            "--provider",
            "plan",
            "--plan-file",
            str(evil),
        ],
    )
    assert blocked.exit_code != 0


def test_apply_refuses_to_overwrite_input(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _dataset(tmp_path / "data")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "propose",
            "--data",
            str(data),
            "--target",
            "target",
            "--provider",
            "mock",
            "--apply",
            str(data / "train.csv"),
        ],
    )
    assert result.exit_code != 0
    assert "overwrite" in result.output.lower()
