"""Offline package tests: contents, checksums, CLI and safety errors."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kconv.cli import app
from kconv.config import Config, DataConfig, ModelConfig, ValidationConfig, save_config
from kconv.errors import KconvError
from kconv.package import create_package
from kconv.stages.run import run_pipeline


@pytest.fixture(scope="module")
def packaged_project(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("package")
    project = root / "competitions" / "offline"
    data = project / "data"
    data.mkdir(parents=True)
    rng = np.random.default_rng(11)
    n_train, n_test = 120, 40
    n = n_train + n_test
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], n),
            "target": np.tile([0, 1], n // 2),
        }
    )
    test = frame.iloc[n_train:].drop(columns=["target"]).copy()
    frame.iloc[:n_train].to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "target": 0}).to_csv(data / "sample_submission.csv", index=False)

    config = Config(
        project="offline",
        seed=42,
        recipe="pesnik/baseline",
        strategy="fast",
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
        models=[
            ModelConfig(
                name="lgbm",
                params={"verbosity": -1, "num_threads": 1, "deterministic": True},
            )
        ],
    )
    config_path = project / "config.yaml"
    save_config(config, config_path)
    ctx = run_pipeline(project, config_path, config, until="report")
    return root, project, config_path, ctx


def test_package_contains_metadata_and_checksums(
    packaged_project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, project, config_path, ctx = packaged_project
    output = tmp_path / "bundle.zip"
    result = create_package(project, config_path, output=output)

    assert Path(result["path"]) == output
    assert result["run_id"] == ctx.run_id
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        required = {
            "submission.csv",
            "report.md",
            "blend.json",
            "schema.json",
            "data_card.json",
            "adversarial.json",
            "run.json",
            "state.json",
            "config.yaml",
            "SHA256SUMS",
        }
        assert required <= names
        assert "train.parquet" not in names
        assert "test.parquet" not in names
        assert not any(name.startswith("runs/") for name in names)
        assert not any(name.endswith("oof.parquet") for name in names)

        sums = archive.read("SHA256SUMS").decode("utf-8")
        for line in sums.splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest

    # The CLI resolves the latest run and writes to packages/ by default.
    runner = CliRunner()
    monkeypatch.chdir(root)
    invocation = runner.invoke(app, ["package", "--comp", "offline"])
    assert invocation.exit_code == 0, invocation.output
    default_zip = project / "packages" / f"offline-{ctx.run_id}.zip"
    assert default_zip.exists()


def test_package_force_and_missing_report(packaged_project, tmp_path: Path) -> None:
    _root, project, config_path, ctx = packaged_project
    output = tmp_path / "bundle.zip"
    create_package(project, config_path, output=output)

    with pytest.raises(KconvError, match="already exists"):
        create_package(project, config_path, output=output)
    create_package(project, config_path, output=output, force=True)

    report = ctx.run_dir / "05_report" / "report.md"
    backup = report.with_name("report.md.bak")
    report.rename(backup)
    try:
        with pytest.raises(KconvError, match="report.md"):
            create_package(project, config_path, output=tmp_path / "missing.zip")
    finally:
        backup.rename(report)
