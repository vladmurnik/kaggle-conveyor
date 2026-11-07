"""Opt-in real-data smoke tests on a 10% sample of the local Kaggle dataset.

The tests are intentionally skipped unless ``KCONV_REAL_DATA_TESTS=1``.  A
full run fits every binary-compatible model and executes every built-in
pipeline on the same 10% sample, which takes several minutes on CPU and is not
appropriate for the normal unit-test suite.

Run everything::

    $env:KCONV_REAL_DATA_TESTS = "1"
    uv run pytest -q tests/test_real_data_10pct.py

Run selected models/pipelines::

    $env:KCONV_REAL_DATA_TESTS = "1"
    $env:KCONV_REAL_DATA_MODELS = "lgbm,histgb,svc"
    $env:KCONV_REAL_DATA_RECIPES = "pesnik/baseline,kconv/linear-stack"
    uv run pytest -q tests/test_real_data_10pct.py

The sample is stratified for the target.  Model tests use an 80/20 split of the
sample; pipeline tests use the 10% train/test/sample-submission files.  The
expensive ``svc``/``nu_svc`` adapters get a finite iteration budget so the
smoke test remains bounded; all other adapters use their production defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from kconv.api import list_models, list_pipelines
from kconv.autodetect import detect_dataset
from kconv.config import Config, DataConfig, save_config
from kconv.contracts import DatasetSpec
from kconv.features.pipeline import FeatureBuilder
from kconv.models import create_model, model_available
from kconv.project import build_config
from kconv.stages.run import run_pipeline
from kconv.state import PipelineState

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TARGET = os.environ.get("KCONV_REAL_DATA_TARGET", "satisfaction")
ID_COL = os.environ.get("KCONV_REAL_DATA_ID", "id")
FRACTION = 0.10
SEED = 42
RUN_REAL_DATA = os.environ.get("KCONV_REAL_DATA_TESTS") == "1"
PIPELINE_STAGES = ("ingest", "validate", "hpo", "train", "blend", "submit", "report")
SLOW_MODELS = {"svc", "nu_svc"}


def _filtered(values: list[str], env_name: str) -> list[str]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return values
    wanted = [item.strip() for item in raw.split(",") if item.strip()]
    return [item for item in values if item in wanted]


BINARY_MODELS = _filtered(sorted(model.id for model in list_models(task="binary")), "KCONV_REAL_DATA_MODELS")
BINARY_PIPELINES = _filtered(
    sorted(pipeline.id for pipeline in list_pipelines(task="binary")),
    "KCONV_REAL_DATA_RECIPES",
)

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.slow,
    pytest.mark.skipif(
        not RUN_REAL_DATA,
        reason="set KCONV_REAL_DATA_TESTS=1 to run the 10% real-data smoke suite",
    ),
]


@dataclass
class RealDataSample:
    """The shared 10% sample and lazily built feature matrices."""

    data_dir: Path
    train_path: Path
    test_path: Path
    sample_submission_path: Path
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    spec: DatasetSpec
    builder_config: Config
    y_train: np.ndarray
    y_valid: np.ndarray
    total_train_rows: int
    total_test_rows: int
    full_target_share: dict[str, float]
    sample_target_share: dict[str, float]
    _matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = field(default_factory=dict)

    def matrices(self, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if family not in self._matrices:
            builder = FeatureBuilder(self.builder_config, self.spec, family=family)
            xt = builder.fit_transform(self.train, self.y_train)
            xv = builder.transform(self.valid)
            self._matrices[family] = (xt, xv)
        return self._matrices[family]


def _make_sample(tmp_path: Path) -> RealDataSample:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    sample_submission_path = DATA_DIR / "sample_submission.csv"
    if not train_path.is_file() or not test_path.is_file() or not sample_submission_path.is_file():
        pytest.skip("data/train.csv, data/test.csv and data/sample_submission.csv are required")

    full_train = pd.read_csv(train_path)
    if TARGET not in full_train.columns:
        pytest.skip(f"target column '{TARGET}' is not present in {train_path}")
    if ID_COL not in full_train.columns:
        pytest.skip(f"id column '{ID_COL}' is not present in {train_path}")

    # Stratified 10% sample.  GroupBy.sample keeps every class represented and
    # gives exactly the requested fraction up to integer rounding.
    train_sample = full_train.groupby(TARGET, group_keys=False).sample(frac=FRACTION, random_state=SEED)
    train_sample = train_sample.reset_index(drop=True)

    full_test = pd.read_csv(test_path)
    test_sample = full_test.sample(frac=FRACTION, random_state=SEED).reset_index(drop=True)
    full_sample_submission = pd.read_csv(sample_submission_path)
    if ID_COL not in full_sample_submission.columns:
        pytest.skip(f"id column '{ID_COL}' is not present in {sample_submission_path}")
    test_ids = set(test_sample[ID_COL].tolist())
    sample_submission = full_sample_submission[full_sample_submission[ID_COL].isin(test_ids)].copy()
    if len(sample_submission) != len(test_sample):
        pytest.skip("sample_submission does not cover the sampled test ids")

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_sample.to_csv(data_dir / "train.csv", index=False)
    test_sample.to_csv(data_dir / "test.csv", index=False)
    sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)

    # The real dataset is a binary target; use it explicitly so a noisy
    # auto-detection heuristic cannot change the test contract.
    spec = detect_dataset(
        data_dir,
        target=TARGET,
        id=ID_COL,
        task="binary",
        metric="roc_auc",
    )
    train_frame, valid_frame = train_test_split(
        train_sample,
        test_size=0.20,
        random_state=SEED,
        stratify=train_sample[TARGET],
    )
    train_frame = train_frame.reset_index(drop=True)
    valid_frame = valid_frame.reset_index(drop=True)
    builder_config = Config(
        project="realdata-10pct",
        seed=SEED,
        task="binary",
        metric="roc_auc",
        data=DataConfig(
            train=str(data_dir / "train.csv"),
            test=str(data_dir / "test.csv"),
            sample_submission=str(data_dir / "sample_submission.csv"),
            target=TARGET,
            id=ID_COL,
        ),
    )
    assert set(train_sample[TARGET].unique()) == set(valid_frame[TARGET].unique())
    return RealDataSample(
        data_dir=data_dir,
        train_path=data_dir / "train.csv",
        test_path=data_dir / "test.csv",
        sample_submission_path=data_dir / "sample_submission.csv",
        train=train_frame,
        valid=valid_frame,
        test=test_sample,
        sample_submission=sample_submission,
        spec=spec,
        builder_config=builder_config,
        y_train=train_frame[TARGET].astype(int).to_numpy(),
        y_valid=valid_frame[TARGET].astype(int).to_numpy(),
        total_train_rows=len(full_train),
        total_test_rows=len(full_test),
        full_target_share={
            str(label): float(share)
            for label, share in full_train[TARGET].value_counts(normalize=True).items()
        },
        sample_target_share={
            str(label): float(share)
            for label, share in train_sample[TARGET].value_counts(normalize=True).items()
        },
    )


@pytest.fixture(scope="session")
def real_data_sample(tmp_path_factory: pytest.TempPathFactory) -> RealDataSample:
    return _make_sample(tmp_path_factory.mktemp("realdata-10pct"))


def _assert_probabilities(proba: np.ndarray, expected_rows: int) -> None:
    values = np.asarray(proba, dtype=float)
    assert values.shape == (expected_rows,), f"expected one probability per row, got {values.shape}"
    assert np.isfinite(values).all()
    assert ((values >= 0.0) & (values <= 1.0)).all()


def test_sample_is_ten_percent(real_data_sample: RealDataSample) -> None:
    sample_rows = len(real_data_sample.train) + len(real_data_sample.valid)
    assert 0.095 <= sample_rows / real_data_sample.total_train_rows <= 0.105
    assert 0.095 <= len(real_data_sample.test) / real_data_sample.total_test_rows <= 0.105
    assert len(real_data_sample.test) == len(real_data_sample.sample_submission)


def test_train_sample_is_stratified_by_target(real_data_sample: RealDataSample) -> None:
    for label, full_share in real_data_sample.full_target_share.items():
        assert real_data_sample.sample_target_share[label] == pytest.approx(full_share, abs=0.01)


@pytest.mark.parametrize("model_name", BINARY_MODELS)
def test_binary_model_on_10pct(real_data_sample: RealDataSample, model_name: str) -> None:
    if model_name == "tabpfn":
        pytest.skip("TabPFN v2 has an 8k safe row limit; the 10% sample is larger")
    if not model_available(model_name):
        pytest.skip(f"optional runtime for model '{model_name}' is not installed")

    adapter = create_model(model_name)
    adapter.task = "binary"
    params: dict[str, object] = {}
    if model_name in SLOW_MODELS:
        # NuSVC on 56k rows is quadratic.  This still exercises fit/predict and
        # the adapter contract while keeping the opt-in suite bounded.
        params["max_iter"] = 1000
    xt, xv = real_data_sample.matrices(adapter.family)
    adapter.fit(
        xt,
        real_data_sample.y_train,
        xv,
        real_data_sample.y_valid,
        params=params,
        seed=SEED,
        early_stopping_rounds=20,
    )
    _assert_probabilities(adapter.predict_proba(xv), len(xv))
    predictions = np.asarray(adapter.predict(xv))
    assert predictions.shape == (len(xv),)


@pytest.mark.parametrize("recipe_id", BINARY_PIPELINES)
def test_pipeline_on_10pct(
    real_data_sample: RealDataSample,
    recipe_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # build_config writes competitions/<name>; keep every parameterized test
    # isolated in its own pytest temp directory.
    monkeypatch.chdir(tmp_path)
    project_dir, config_path, config = build_config(
        real_data_sample.data_dir,
        target=TARGET,
        recipe_id=recipe_id,
        strategy="fast",
        models=None,
        name="realdata",
        task="binary",
        metric="roc_auc",
        id_col=ID_COL,
    )
    assert config.recipe == recipe_id
    assert config.models
    # Two folds are enough for a structural smoke test; adversarial validation
    # is covered by the fast synthetic suite and would add no model coverage.
    config.validation.n_splits = 2
    config.validation.adversarial = False
    save_config(config, config_path)

    ctx = run_pipeline(project_dir, config_path, config, until="report", force=True)
    state = PipelineState.load(ctx.run_dir / "state.json")
    for stage in PIPELINE_STAGES:
        assert state.is_done(stage), f"{recipe_id}: stage {stage} is {state.stage(stage).status}"

    submission_path = ctx.run_dir / "04_submit" / "submission.csv"
    report_path = ctx.run_dir / "05_report" / "report.md"
    assert submission_path.is_file(), recipe_id
    assert report_path.is_file(), recipe_id
    submission = pd.read_csv(submission_path)
    assert list(submission.columns) == list(real_data_sample.sample_submission.columns)
    assert len(submission) == len(real_data_sample.test)
    assert not submission.isna().any().any(), f"{recipe_id}: submission contains NaN"
    assert set(submission[ID_COL]) == set(real_data_sample.sample_submission[ID_COL])

    blend = json.loads((ctx.run_dir / "03_blend" / "blend.json").read_text(encoding="utf-8"))
    assert np.isfinite(float(blend["oof_score"])), recipe_id
    assert blend["models"], recipe_id
