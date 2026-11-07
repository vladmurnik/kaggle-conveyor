"""Submission-stage edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kconv.config import Config, DataConfig, ModelConfig, ValidationConfig
from kconv.errors import StageError
from kconv.stages.run import run_pipeline


def _multiclass_data(tmp_path: Path, labels: np.ndarray, metric: str) -> tuple[Path, pd.DataFrame, Path]:
    project = tmp_path / "multiclass"
    data = project / "data"
    data.mkdir(parents=True)
    rng = np.random.default_rng(37)
    n = 240
    class_id = np.tile(np.arange(3), n // 3)
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "x0": class_id + rng.normal(scale=0.4, size=n),
            "x1": rng.normal(size=n),
            "cat": rng.choice(["a", "b"], n),
            "target": labels[class_id],
        }
    )
    frame.iloc[:180].to_csv(data / "train.csv", index=False)
    test = frame.iloc[180:].drop(columns=["target"])
    test.to_csv(data / "test.csv", index=False)
    if metric in {"log_loss", "roc_auc"}:
        probability_columns = [str(value) for value in labels.tolist()]
        sample = {"id": test["id"].to_numpy()}
        for column in probability_columns:
            sample[column] = 1.0 / len(probability_columns)
        pd.DataFrame(sample).to_csv(data / "sample_submission.csv", index=False)
    else:
        pd.DataFrame({"id": test["id"], "target": 0}).to_csv(data / "sample_submission.csv", index=False)
    return project, test, data / "sample_submission.csv"


@pytest.mark.parametrize(
    "labels",
    [
        np.array(["red", "green", "blue"]),
        np.array([10, 20, 30]),
    ],
    ids=["string", "integer"],
)
@pytest.mark.parametrize("metric", ["log_loss", "accuracy"])
def test_multiclass_submission_matches_metric_contract(
    tmp_path: Path, labels: np.ndarray, metric: str
) -> None:
    project, test, _ = _multiclass_data(tmp_path, labels, metric)
    config = Config(
        project="multiclass",
        seed=42,
        task="multiclass",
        metric=metric,
        data=DataConfig(
            train=str(project / "data" / "train.csv"),
            test=str(project / "data" / "test.csv"),
            sample_submission=str(project / "data" / "sample_submission.csv"),
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

    ctx = run_pipeline(project, project / "config.yaml", config, until="report")
    submission = pd.read_csv(ctx.run_dir / "04_submit" / "submission.csv")
    check = json.loads((ctx.run_dir / "04_submit" / "submission_check.json").read_text(encoding="utf-8"))
    assert list(submission["id"]) == test["id"].tolist()
    assert len(submission) == len(test)
    assert check["task"] == "multiclass"
    assert check["nan_count"] == 0
    assert (ctx.run_dir / "05_report" / "report.md").exists()

    if metric in {"log_loss", "roc_auc"}:
        # Kaggle multiclass probability metrics need one column per class, not
        # a single argmax label column.
        probability_columns = [str(value) for value in labels.tolist()]
        assert set(submission.columns) == {"id", *probability_columns}
        matrix = submission[probability_columns].to_numpy(dtype=float)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(matrix >= 0.0) and np.all(matrix <= 1.0)
        assert 0.0 <= float(check["min"]) <= float(check["max"]) <= 1.0
    else:
        expected = set(labels.tolist())
        assert set(submission["target"]) <= expected
        if labels.dtype.kind in "iu":
            assert pd.api.types.is_integer_dtype(submission["target"])
        assert check["min"] in expected
        assert check["max"] in expected


def test_multiclass_label_metric_rejects_probability_sample(tmp_path: Path) -> None:
    labels = np.array(["red", "green", "blue"])
    project, test, sample_path = _multiclass_data(tmp_path, labels, "accuracy")
    sample = {"id": test["id"].to_numpy()}
    for label in labels:
        sample[str(label)] = 1.0 / len(labels)
    pd.DataFrame(sample).to_csv(sample_path, index=False)
    config = Config(
        project="multiclass",
        seed=42,
        task="multiclass",
        metric="accuracy",
        data=DataConfig(
            train=str(project / "data" / "train.csv"),
            test=str(project / "data" / "test.csv"),
            sample_submission=str(project / "data" / "sample_submission.csv"),
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

    with pytest.raises(StageError, match="exactly one target column"):
        run_pipeline(project, project / "config.yaml", config, until="report")
