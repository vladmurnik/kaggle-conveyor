"""Lifecycle regressions: data hashes, locks, partial runs, schema checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kconv.autodetect import detect_dataset
from kconv.config import Config, DataConfig, ModelConfig, ValidationConfig, save_config
from kconv.context import ProjectLock, RunContext, write_json
from kconv.errors import DataError, KconvError
from kconv.io import safe_read_edges
from kconv.stages.ingest import run_ingest
from kconv.stages.run import run_pipeline


def _make_project(project: Path) -> tuple[Path, Config]:
    rng = np.random.default_rng(11)
    n_train, n_test = 60, 20
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
        project="lifecycle",
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
                seeds=1,
                params={"verbosity": -1, "num_threads": 1, "deterministic": True},
            )
        ],
    )
    config_path = project / "config.yaml"
    save_config(config, config_path)
    return config_path, config


def test_resume_rejects_changed_raw_data(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path, config = _make_project(project)
    first = run_pipeline(project, config_path, config, until="validate")
    assert first.data_hash
    state = json.loads((first.run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["data_hash"] == first.data_hash
    assert state["data_files"]

    train_path = project / "data" / "train.csv"
    frame = pd.read_csv(train_path)
    frame.loc[0, "x0"] = frame.loc[0, "x0"] + 100.0
    frame.to_csv(train_path, index=False)

    resumed = run_pipeline(project, config_path, config, until="validate", resume=True)
    assert resumed.run_id != first.run_id
    run_meta = json.loads((resumed.run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_meta["data_hash"] != first.data_hash


def test_until_validate_marks_run_partial(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path, config = _make_project(project)
    ctx = run_pipeline(project, config_path, config, until="validate")

    run_meta = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_meta["status"] == "partial"
    assert "train" in run_meta["remaining_stages"]
    latest = json.loads((project / "latest.json").read_text(encoding="utf-8"))
    assert latest["status"] == "partial"
    assert run_meta["python"]
    assert run_meta["kconv_version"]
    assert run_meta["packages"]["pandas"]
    assert run_meta["environment"]["python"] == run_meta["python"]


def test_project_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    with ProjectLock(tmp_path):
        with pytest.raises(KconvError, match="already in use"):
            ProjectLock(tmp_path).acquire()


def test_write_json_is_safe_for_concurrent_writers(tmp_path: Path) -> None:
    import threading

    path = tmp_path / "state.json"
    errors: list[BaseException] = []

    def write(index: int) -> None:
        try:
            write_json(path, {"writer": index, "values": list(range(20))})
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [
        threading.Thread(target=write, args=(index,), daemon=True)
        for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["values"] == list(range(20))


def test_ingest_reports_missing_test_feature(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_path, config = _make_project(project)
    test_path = project / "data" / "test.csv"
    test = pd.read_csv(test_path).drop(columns=["x1"])
    test.to_csv(test_path, index=False)
    ctx = RunContext.create(project, config_path, config)
    with pytest.raises(DataError, match="missing feature columns"):
        run_ingest(ctx)
    assert not (ctx.run_dir / "00_ingest" / "schema.json").exists()


def test_safe_read_edges_includes_tail_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    pd.DataFrame({"value": np.arange(10)}).to_csv(path, index=False)
    sample = safe_read_edges(path, nrows=6)
    assert len(sample) == 6
    assert sample["value"].tolist() == [0, 1, 2, 7, 8, 9]


def test_role_detection_sees_signal_after_head(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    rows = 120
    frame = pd.DataFrame(
        {
            "id": np.arange(rows),
            "x": np.arange(rows),
            "late_signal": [0] * 100 + [1] * 20,
            "target": [0, 1] * (rows // 2),
        }
    )
    frame.to_csv(path, index=False)
    spec = detect_dataset(path, target="target", id="id", task="binary", sample_rows=100)
    assert "late_signal" in spec.feature_columns
