"""Tests for dataset auto-detection, table IO and the ingest stage."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from kconv.autodetect import detect_dataset
from kconv.config import Config, DataConfig, save_config
from kconv.context import RunContext
from kconv.errors import DataError
from kconv.io import ensure_dirs, read_table, safe_read_sample, write_table
from kconv.stages.ingest import run_ingest


def _titanic_like(directory: Path, *, with_sample_submission: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Age": [22.0, 38.0, float("nan"), 35.0],
            "Fare": [7.25, 71.28, 8.05, 53.1],
            "Survived": [0, 1, 0, 1],
        }
    )
    # Keep all train feature columns present so a target override (e.g. Fare)
    # can be ingested without a false schema mismatch on Survived.
    test = pd.DataFrame(
        {
            "PassengerId": [5, 6],
            "Age": [30.0, 22.0],
            "Fare": [10.0, 20.0],
            "Survived": [0, 1],
        }
    )
    gender = pd.DataFrame({"PassengerId": [5, 6], "Survived": [0, 1]})
    train.to_csv(directory / "train.csv", index=False)
    test.to_csv(directory / "test.csv", index=False)
    gender.to_csv(directory / "gender_submission.csv", index=False)
    if with_sample_submission:
        gender.to_csv(directory / "sample_submission.csv", index=False)
    return directory


def test_detect_rejects_unknown_or_incompatible_metric(tmp_path: Path) -> None:
    data_dir = _titanic_like(tmp_path / "raw")
    with pytest.raises(DataError, match="not compatible"):
        detect_dataset(data_dir, task="binary", metric="rmse")
    with pytest.raises(DataError, match="Unknown metric"):
        detect_dataset(data_dir, task="binary", metric="not-a-metric")


def test_detect_directory_titanic_like(tmp_path: Path) -> None:
    data_dir = _titanic_like(tmp_path / "raw")

    spec = detect_dataset(data_dir)

    assert spec.train == data_dir / "train.csv"
    assert spec.test == data_dir / "test.csv"
    # sample_submission wins over gender_submission
    assert spec.sample_submission == data_dir / "sample_submission.csv"
    assert spec.target == "Survived"
    assert spec.id_col == "PassengerId"
    assert spec.task == "binary"
    assert spec.metric == "roc_auc"
    assert spec.feature_columns == ["Age", "Fare"]
    assert spec.numeric_columns == ["Age", "Fare"]
    assert spec.ignore_columns == []
    assert spec.test is not None


def test_detect_directory_without_sample_submission(tmp_path: Path) -> None:
    data_dir = _titanic_like(tmp_path / "raw", with_sample_submission=False)

    spec = detect_dataset(data_dir)

    assert spec.sample_submission == data_dir / "gender_submission.csv"
    assert spec.target == "Survived"


def test_detect_single_train_only_csv(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": ["x", "y", "x", "y"],
            "target": [0, 1, 0, 1],
        }
    ).to_csv(train, index=False)

    spec = detect_dataset(train)

    assert spec.train == train
    assert spec.test is None
    assert spec.sample_submission is None
    assert spec.target == "target"  # name heuristic beats last-column fallback
    assert spec.id_col is None
    assert spec.task == "binary"
    assert spec.metric == "roc_auc"
    assert spec.categorical_columns == ["feature_b"]


def test_detect_parquet_directory_last_column_regression(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "f1": [1.0, 1.0, 2.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 4.0, 6.0, 5.0],
            "demand": [0.15, 0.93, 1.71, 2.34, 3.55],
        }
    ).to_parquet(tmp_path / "train.parquet", index=False)
    pd.DataFrame({"f1": [1.0, 2.0], "f2": [5.0, 6.0]}).to_parquet(tmp_path / "test.parquet", index=False)

    spec = detect_dataset(tmp_path)

    assert spec.train == tmp_path / "train.parquet"
    assert spec.test == tmp_path / "test.parquet"
    assert spec.target == "demand"  # no name hint: last column wins
    assert spec.id_col is None  # no unique column shared with test
    assert spec.task == "regression"
    assert spec.metric == "rmse"


def test_explicit_target_override_wins(tmp_path: Path) -> None:
    data_dir = _titanic_like(tmp_path / "raw")

    spec = detect_dataset(data_dir, target="Fare")

    assert spec.target == "Fare"
    assert spec.task == "regression"  # inferred from the overridden column
    assert spec.metric == "rmse"
    assert spec.id_col == "PassengerId"


def test_explicit_task_and_metric_overrides(tmp_path: Path) -> None:
    data_dir = _titanic_like(tmp_path / "raw")

    spec = detect_dataset(data_dir, task="regression", metric="mae")

    assert spec.task == "regression"
    assert spec.metric == "mae"


def test_unknown_task_raises(tmp_path: Path) -> None:
    data_dir = _titanic_like(tmp_path / "raw")

    with pytest.raises(DataError) as excinfo:
        detect_dataset(data_dir, task="multilabel")

    assert "--task" in str(excinfo.value)


def test_missing_target_override_raises(tmp_path: Path) -> None:
    data_dir = _titanic_like(tmp_path / "raw")

    with pytest.raises(DataError) as excinfo:
        detect_dataset(data_dir, target="NoSuchColumn")

    assert "--target" in str(excinfo.value)


def test_no_target_raises(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    pd.DataFrame({"only": [1, 2, 3]}).to_csv(train, index=False)

    with pytest.raises(DataError) as excinfo:
        detect_dataset(train)

    message = str(excinfo.value)
    assert "--target" in message
    assert "--task" in message


def test_ambiguous_target_raises(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "label_a": [0, 1, 0, 1],
            "label_b": [1, 0, 1, 0],
        }
    ).to_csv(train, index=False)

    with pytest.raises(DataError) as excinfo:
        detect_dataset(train)

    message = str(excinfo.value)
    assert "label_a" in message and "label_b" in message
    assert "--target" in message


def test_missing_path_and_empty_dir_raise(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        detect_dataset(tmp_path / "nope.csv")

    with pytest.raises(DataError) as excinfo:
        detect_dataset(tmp_path)

    assert "--target" in str(excinfo.value)


def test_id_inference_from_unique_column_shared_with_test(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "key": ["a", "b", "c", "d"],
            "feature": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 1, 0, 1],
        }
    ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"key": ["e", "f"], "feature": [5.0, 6.0]}).to_csv(tmp_path / "test.csv", index=False)

    spec = detect_dataset(tmp_path)

    assert spec.target == "target"
    assert spec.id_col == "key"


def test_multiclass_inference(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
            "cls": [0, 1, 2, 0, 1, 2],
        }
    ).to_csv(tmp_path / "train.csv", index=False)

    spec = detect_dataset(tmp_path)

    assert spec.target == "cls"
    assert spec.task == "multiclass"
    assert spec.metric == "log_loss"


def test_detect_zip_archive(tmp_path: Path) -> None:
    data_dir = tmp_path / "packed"
    _titanic_like(data_dir, with_sample_submission=False)
    archive = tmp_path / "competition.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for file in data_dir.iterdir():
            handle.write(file, arcname=f"competition/{file.name}")

    spec = detect_dataset(archive)

    assert spec.train.is_file()
    assert spec.train.suffix == ".csv"
    assert archive not in spec.train.parents  # real extracted paths, not the zip
    assert spec.target == "Survived"
    # content-addressed cache: same archive resolves to the same extraction
    assert detect_dataset(archive).train == spec.train


def test_detect_rejects_unsupported_path(tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("hello", encoding="utf-8")

    with pytest.raises(DataError):
        detect_dataset(bad)


def test_io_roundtrip_and_sample(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": range(50), "b": ["x"] * 50})

    parquet = write_table(frame, tmp_path / "sub" / "dir" / "t.parquet")
    assert parquet.is_file()
    assert len(read_table(parquet)) == 50

    csv = write_table(frame, tmp_path / "t.csv")
    assert read_table(csv).shape == (50, 2)
    assert len(safe_read_sample(csv, nrows=10)) == 10
    assert list(safe_read_sample(parquet, nrows=7).columns) == ["a", "b"]

    assert ensure_dirs(tmp_path / "nested" / "dirs").is_dir()  # no suffix: the dir itself
    with pytest.raises(DataError):
        read_table(tmp_path / "missing.csv")


def _make_project(tmp_path: Path, **data_kwargs: object) -> tuple[Path, RunContext]:
    project = tmp_path / "proj"
    raw = project / "data" / "raw"
    _titanic_like(raw, with_sample_submission=bool(data_kwargs.pop("with_sample_submission", True)))
    data = DataConfig(train="data/raw/train.csv", **data_kwargs)  # type: ignore[arg-type]
    config = Config(project="titanic", data=data)
    config_path = project / "config.yaml"
    save_config(config, config_path)
    # Construct the context directly: RunContext.create() hashes the config,
    # which currently trips a pre-existing pydantic bug in context.config_hash.
    run_dir = project / "runs" / "manual-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(
        project_dir=project,
        config_path=config_path,
        config=config,
        run_id="manual-run",
        run_dir=run_dir,
        seed=config.seed,
    )
    return project, ctx


@pytest.mark.parametrize(
    "target_values",
    [
        [0, 1, 0, 1],
        [0, 1, 2, 0],
        [1.5, 2.5, 3.5, 4.5],
    ],
    ids=["binary", "multiclass", "regression"],
)
def test_run_ingest_rejects_partial_nan_target(tmp_path: Path, target_values: list[float]) -> None:
    project, ctx = _make_project(tmp_path)
    train_path = project / "data" / "raw" / "train.csv"
    frame = pd.read_csv(train_path)
    frame["Survived"] = target_values
    frame.loc[1, "Survived"] = float("nan")
    frame.to_csv(train_path, index=False)

    with pytest.raises(DataError, match="missing value"):
        run_ingest(ctx)


def test_run_ingest_writes_artifacts(tmp_path: Path) -> None:
    project, ctx = _make_project(
        tmp_path,
        test="data/raw/test.csv",
        sample_submission="data/raw/gender_submission.csv",
        with_sample_submission=False,
    )

    result = run_ingest(ctx)

    assert result["status"] == "ok"
    assert result["artifacts"]
    assert all(not Path(a).is_absolute() for a in result["artifacts"])

    stage_dir = ctx.run_dir / "00_ingest"
    train_parquet = stage_dir / "train.parquet"
    assert train_parquet.is_file()
    assert (stage_dir / "test.parquet").is_file()

    schema_path = stage_dir / "schema.json"
    assert schema_path.is_file()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["target"] == "Survived"
    assert schema["id_col"] == "PassengerId"
    assert schema["task"] == "binary"
    assert schema["metric"] == "roc_auc"
    assert not Path(schema["train"]).is_absolute()
    assert schema["train"].endswith("00_ingest/train.parquet")
    # schema.json must describe data that actually exists
    assert (project / schema["train"]).is_file()
    assert len(pd.read_parquet(project / schema["train"])) == 4

    card_path = stage_dir / "data_card.json"
    assert card_path.is_file()
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["rows"] == {"train": 4, "test": 2}
    assert card["target"] == "Survived"
    assert card["source"]["train"] == "data/raw/train.csv"
    assert card["class_balance"] == {"0": 2, "1": 2}

    assert ctx.dataset is not None
    assert ctx.dataset.target == "Survived"
    assert ctx.dataset.task == "binary"


def test_run_ingest_samples_train_percentage(tmp_path: Path) -> None:
    _, ctx = _make_project(
        tmp_path,
        sample_pct=50,
        test="data/raw/test.csv",
        with_sample_submission=False,
    )

    result = run_ingest(ctx)

    assert result["status"] == "ok"
    stage_dir = ctx.run_dir / "00_ingest"
    card = json.loads((stage_dir / "data_card.json").read_text(encoding="utf-8"))
    assert card["rows"] == {"train": 2, "test": 2}
    assert card["sample_pct"] == 50
    assert card["rows_before_sample"] == 4
    assert card["class_balance"] == {"0": 1, "1": 1}
    assert any("sampled to 50%" in warning for warning in result["warnings"])
    schema = json.loads((stage_dir / "schema.json").read_text(encoding="utf-8"))
    assert len(pd.read_parquet(ctx.project_dir / schema["train"])) == 2


def test_run_ingest_honours_config_target(tmp_path: Path) -> None:
    _, ctx = _make_project(
        tmp_path,
        target="Fare",
        test="data/raw/test.csv",
        sample_submission="data/raw/gender_submission.csv",
        with_sample_submission=False,
    )

    result = run_ingest(ctx)

    assert result["status"] == "ok"
    assert ctx.dataset is not None
    assert ctx.dataset.target == "Fare"
    assert ctx.dataset.task == "regression"
    schema = json.loads((ctx.run_dir / "00_ingest" / "schema.json").read_text(encoding="utf-8"))
    assert schema["target"] == "Fare"
    assert schema["task"] == "regression"
