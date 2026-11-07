"""End-to-end synthetic pipeline test (T1.6 acceptance).

Builds a small tabular competition on disk (600 train rows, numeric +
categorical features, binary target), runs the full pipeline up to
``submit`` and checks:

- ``submission.csv`` exists with the sample-submission schema and no NaN;
- ``state.json`` marks ingest/validate/train/blend/submit as done;
- the OOF file has exactly one finite prediction per train row;
- a second run with the same seed reproduces the OOF hash or metric (1e-9);
- ``--resume`` reuses the latest run instead of starting a new one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kconv.config import Config, DataConfig, ModelConfig, ValidationConfig, save_config
from kconv.errors import StageError
from kconv.stages.run import run_pipeline
from kconv.state import PipelineState, read_latest_pointer

N_TRAIN = 600
N_TEST = 150
N_NUMERIC = 5


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_competition(project: Path, seed: int = 7) -> None:
    """Create train/test/sample_submission CSVs under ``project/data``."""
    rng = np.random.default_rng(seed)
    n_total = N_TRAIN + N_TEST

    numeric = rng.normal(size=(n_total, N_NUMERIC))
    cat_a = rng.choice(["alpha", "beta", "gamma", "delta"], size=n_total)
    cat_b = rng.choice(["red", "blue"], size=n_total)

    cat_a_effect = pd.Series(cat_a).map({"alpha": 1.0, "beta": -0.7, "gamma": 0.3, "delta": -0.2}).to_numpy()
    cat_b_effect = np.where(cat_b == "red", 0.5, -0.5)
    logit = (
        1.3 * numeric[:, 0]
        - 0.9 * numeric[:, 1]
        + 0.7 * numeric[:, 2] * numeric[:, 3]
        + cat_a_effect
        + cat_b_effect
        + rng.normal(scale=0.5, size=n_total)
    )
    target = (_sigmoid(logit) > rng.uniform(size=n_total)).astype(int)

    frame = pd.DataFrame(
        {
            "id": np.arange(n_total),
            "num_0": numeric[:, 0],
            "num_1": numeric[:, 1],
            "num_2": numeric[:, 2],
            "num_3": numeric[:, 3],
            "num_4": numeric[:, 4],
            "cat_a": cat_a,
            "cat_b": cat_b,
            "target": target,
        }
    )
    train = frame.iloc[:N_TRAIN].copy()
    test = frame.iloc[N_TRAIN:].drop(columns=["target"]).copy()
    sample_submission = pd.DataFrame({"id": test["id"].to_numpy(), "target": 0})

    data_dir = project / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)


def _make_config(project: Path) -> Config:
    data_dir = project / "data"
    return Config(
        project="synthetic",
        seed=42,
        recipe="pesnik/baseline",
        strategy="fast",
        task="binary",
        metric="auto",
        data=DataConfig(
            train=str(data_dir / "train.csv"),
            test=str(data_dir / "test.csv"),
            sample_submission=str(data_dir / "sample_submission.csv"),
            target="target",
            id="id",
        ),
        validation=ValidationConfig(strategy="stratified_kfold", n_splits=3, random_state=42),
        models=[
            ModelConfig(
                name="lgbm",
                seeds=1,
                params={"verbosity": -1, "num_threads": 1, "deterministic": True},
            )
        ],
    )


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "synthetic"
    project_dir.mkdir()
    _make_competition(project_dir)
    return project_dir


def test_auto_task_config_uses_detected_task_for_folds(project: Path) -> None:
    config = _make_config(project)
    config.task = "auto"
    config.metric = "auto"
    config.validation = ValidationConfig(strategy="auto", n_splits=3, random_state=42)
    config_path = project / "auto.yaml"
    save_config(config, config_path)

    ctx = run_pipeline(project, config_path, config, until="validate")

    summary = json.loads((ctx.run_dir / "01_validate" / "cv_summary.json").read_text(encoding="utf-8"))
    assert summary["strategy"] == "stratified_kfold"


def _find_submission(run_dir: Path) -> Path:
    exact = run_dir / "submit" / "submission.csv"
    if exact.exists():
        return exact
    candidates = sorted(p for p in run_dir.glob("**/submission.csv") if p.is_file())
    assert candidates, f"submission.csv not found under {run_dir}"
    return candidates[0]


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _find_oof(run_dir: Path) -> np.ndarray:
    """Locate the OOF predictions (blend's if present, else the model's)."""
    candidates = sorted(run_dir.glob("**/oof*.parquet")) + sorted(run_dir.glob("**/oof*.csv"))
    assert candidates, f"no OOF artifact found under {run_dir}"
    blend = [p for p in candidates if "blend" in p.parts]
    path = (blend or candidates)[0]
    frame = _load_table(path)
    for column in ("pred", "prediction", "oof", "target_pred", "y_pred"):
        if column in frame.columns:
            return frame[column].to_numpy(dtype=float)
    numeric = frame.select_dtypes(include=[np.number])
    assert numeric.shape[1] >= 1, f"OOF artifact {path} has no numeric prediction column"
    return numeric.iloc[:, -1].to_numpy(dtype=float)


def _collect_oof_scores(run_dir: Path) -> list[float]:
    """Gather every ``oof_score`` value serialized in the run's JSON files."""
    scores: list[float] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "oof_score" and isinstance(value, (int, float)) and not isinstance(value, bool):
                    scores.append(float(value))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for path in sorted(run_dir.glob("**/*.json")):
        try:
            walk(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return scores


def test_pipeline_e2e_synthetic(project: Path) -> None:
    config = _make_config(project)
    config_path = project / "config.yaml"

    # --- first run: full pipeline up to submit ----------------------------
    ctx = run_pipeline(project, config_path, config, until="submit")
    run_dir = ctx.run_dir

    # submission: schema matches sample_submission, rows match test, no NaN
    submission = pd.read_csv(_find_submission(run_dir))
    sample = pd.read_csv(project / "data" / "sample_submission.csv")
    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == N_TEST
    assert not submission.isna().any().any(), "submission contains NaN"
    assert set(submission["id"]) == set(sample["id"])

    # adversarial validation runs on the same-distribution synthetic split
    adversarial = json.loads((run_dir / "01_validate" / "adversarial.json").read_text(encoding="utf-8"))
    assert adversarial["status"] == "ok"
    assert adversarial["auc"] < 0.75
    assert adversarial["warning"] is False

    # state.json: every executed stage is done
    state = PipelineState.load(run_dir / "state.json")
    for stage_name in ("ingest", "validate", "train", "blend", "submit"):
        assert state.is_done(stage_name), f"stage '{stage_name}' is {state.stages[stage_name].status}"
    assert state.config_hash == config_hash_of(config)

    # OOF: one finite prediction per train row
    oof_1 = _find_oof(run_dir)
    assert len(oof_1) == N_TRAIN, f"expected {N_TRAIN} OOF predictions, got {len(oof_1)}"
    assert not np.isnan(oof_1).any(), "OOF contains NaN"
    assert np.isfinite(oof_1).all(), "OOF contains non-finite values"

    # latest pointer points at this run
    assert read_latest_pointer(project) == ctx.run_id

    # --- second run with the same seed: reproducibility --------------------
    ctx2 = run_pipeline(project, config_path, config, until="submit")
    assert ctx2.run_id != ctx.run_id, "second run should be a fresh run"
    oof_2 = _find_oof(ctx2.run_dir)
    assert len(oof_2) == N_TRAIN

    hash_1 = hashlib.sha256(oof_1.tobytes()).hexdigest()
    hash_2 = hashlib.sha256(oof_2.tobytes()).hexdigest()
    metrics_1 = _collect_oof_scores(run_dir)
    metrics_2 = _collect_oof_scores(ctx2.run_dir)
    metric_ok = (
        bool(metrics_1)
        and bool(metrics_2)
        and all(abs(a - b) <= 1e-9 for a, b in zip(metrics_1, metrics_2, strict=False))
    )
    assert hash_1 == hash_2 or metric_ok, (
        f"reproducibility broken: oof hash {hash_1[:12]} != {hash_2[:12]}, "
        f"metrics {metrics_1} vs {metrics_2}"
    )

    # --- resume: latest run is reused, done stages are not rerun -----------
    ctx3 = run_pipeline(project, config_path, config, until="submit", resume=True)
    assert ctx3.run_id == ctx2.run_id, "--resume should reuse the latest run"
    resumed = PipelineState.load(ctx3.run_dir / "state.json")
    for stage_name in ("ingest", "validate", "train", "blend", "submit"):
        assert resumed.is_done(stage_name)
        assert resumed.stage(stage_name).attempts == 1, f"stage '{stage_name}' was rerun on resume"


def test_run_pipeline_rejects_unknown_until(project: Path) -> None:
    config = _make_config(project)
    with pytest.raises(StageError, match="until"):
        run_pipeline(project, project / "config.yaml", config, until="no_such_stage")


def test_resume_without_runs_starts_fresh(project: Path) -> None:
    config = _make_config(project)
    ctx = run_pipeline(project, project / "config.yaml", config, until="ingest", resume=True)
    assert ctx.run_id
    assert PipelineState.load(ctx.run_dir / "state.json").is_done("ingest")
    assert read_latest_pointer(project) == ctx.run_id


def config_hash_of(config: Config) -> str:
    from kconv.context import config_hash

    return config_hash(config)
