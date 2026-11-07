"""Portable inference bundle: structure and end-to-end predictions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kconv.cli import app
from kconv.config import (
    Config,
    DataConfig,
    EnsembleConfig,
    ModelConfig,
    ValidationConfig,
    save_config,
)
from kconv.errors import KconvError
from kconv.inference import _SUBMISSION_SCRIPT, export_inference
from kconv.stages.run import run_pipeline


def test_multiclass_submission_maps_classes_even_when_high_class_is_absent(tmp_path: Path) -> None:
    namespace = {"__name__": "kconv_submission_test", "__file__": str(tmp_path / "submission.py")}
    exec(_SUBMISSION_SCRIPT, namespace)
    prediction = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    probabilities = namespace["_to_submission"](
        prediction,
        {"task": "multiclass", "metric": "log_loss", "classes": ["a", "b", "c"]},
    )
    assert probabilities.tolist() == prediction.tolist()
    labels = namespace["_to_submission"](
        prediction,
        {"task": "multiclass", "metric": "accuracy", "classes": ["a", "b", "c"]},
    )
    assert labels.tolist() == ["a", "b"]


@pytest.fixture()
def inference_project(tmp_path: Path):
    project = tmp_path / "competitions" / "portable"
    data = project / "data"
    data.mkdir(parents=True)
    rng = np.random.default_rng(52)
    n_train, n_test = 180, 60
    n = n_train + n_test
    x = rng.normal(size=(n, 3))
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "x0": x[:, 0],
            "x1": x[:, 1],
            "x2": x[:, 2],
            "cat": rng.choice(["a", "b", "c"], n),
            "target": (rng.random(n) < 1.0 / (1.0 + np.exp(-x[:, 0]))).astype(int),
        }
    )
    test = frame.iloc[n_train:].drop(columns=["target"]).copy()
    frame.iloc[:n_train].to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "target": 0}).to_csv(data / "sample_submission.csv", index=False)

    config = Config(
        project="portable",
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
        models=[
            ModelConfig(
                name="lgbm",
                params={"n_estimators": 40, "verbosity": -1, "num_threads": 1, "deterministic": True},
            )
        ],
    )
    config_path = project / "config.yaml"
    save_config(config, config_path)
    ctx = run_pipeline(project, config_path, config, until="report")
    return project, config, ctx, test


def test_inference_bundle_runs_standalone(
    inference_project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, config, ctx, test = inference_project
    bundle = tmp_path / "bundle"
    result = export_inference(project, project / "config.yaml", output=bundle)

    assert result["run_id"] == ctx.run_id
    assert result["models"] == ["lgbm"]
    assert (bundle / "submission.py").exists()
    assert (bundle / "metadata.json").exists()
    assert (bundle / "requirements.txt").exists()
    assert (bundle / "inference.ipynb").exists()
    notebook = json.loads((bundle / "inference.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(ids)
    assert len(set(ids)) == len(ids)
    assert any(cell["cell_type"] == "code" for cell in notebook["cells"])
    assert (bundle / "models" / "lgbm" / "folds" / "fold_0" / "features.joblib").exists()
    assert list((bundle / "models" / "lgbm" / "folds" / "fold_0").glob("seed_*.joblib"))
    assert not (bundle / "train.csv").exists()
    assert not (bundle / "test.csv").exists()
    assert not any(path.name == "train.parquet" for path in bundle.rglob("*"))

    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["method"] == "best_single"
    assert metadata["task"] == "binary"
    assert metadata["classes"] == [0, 1]
    assert metadata["id_col"] == "id"
    assert metadata["target_col"] == "target"

    output = tmp_path / "from_bundle.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(bundle / "submission.py"),
            "--test",
            str(project / "data" / "test.csv"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    expected = pd.read_csv(ctx.run_dir / "04_submit" / "submission.csv")
    actual = pd.read_csv(output)
    assert list(actual.columns) == list(expected.columns)
    assert actual["id"].tolist() == expected["id"].tolist()
    np.testing.assert_allclose(actual["target"], expected["target"], rtol=0.0, atol=1e-10)
    assert actual.isna().sum().sum() == 0

    monkeypatch.chdir(project.parent.parent)
    cli_bundle = tmp_path / "cli_bundle"
    invocation = CliRunner().invoke(
        app,
        ["export", "--comp", "portable", "-o", str(cli_bundle)],
    )
    assert invocation.exit_code == 0, invocation.output
    assert (cli_bundle / "submission.py").exists()


def test_inference_export_refuses_existing_directory(inference_project, tmp_path: Path) -> None:
    project, _config, _ctx, _test = inference_project
    bundle = tmp_path / "bundle"
    export_inference(project, project / "config.yaml", output=bundle)
    with pytest.raises(KconvError, match="already exists"):
        export_inference(project, project / "config.yaml", output=bundle)


def test_stack_bundle_exports_meta_model(inference_project, tmp_path: Path) -> None:
    project, base_config, _base_ctx, _test = inference_project
    config = base_config.model_copy(deep=True)
    config.project = "portable_stack"
    config.models = [
        ModelConfig(
            name="lgbm",
            params={"n_estimators": 40, "verbosity": -1, "num_threads": 1, "deterministic": True},
        ),
        ModelConfig(name="histgb", params={"max_iter": 40}),
    ]
    config.ensemble = EnsembleConfig(method="stack", meta_splits=3)
    config_path = project / "stack_config.yaml"
    save_config(config, config_path)
    ctx = run_pipeline(project, config_path, config, until="report")

    blend = json.loads((ctx.run_dir / "03_blend" / "blend.json").read_text(encoding="utf-8"))
    assert blend["stack"] is not None
    assert blend["method"] == "stack"
    assert (ctx.run_dir / "03_blend" / "meta_model.joblib").exists()

    bundle = tmp_path / "stack_bundle"
    export_inference(project, config_path, output=bundle)
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    assert (bundle / "models" / "meta_model.joblib").exists()
    assert metadata["method"] == "stack"

    output = tmp_path / "stack_from_bundle.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(bundle / "submission.py"),
            "--test",
            str(project / "data" / "test.csv"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    expected = pd.read_csv(ctx.run_dir / "04_submit" / "submission.csv")
    actual = pd.read_csv(output)
    np.testing.assert_allclose(actual["target"], expected["target"], rtol=0.0, atol=1e-10)
