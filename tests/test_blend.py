"""OOF stacking tests: meta-model CV and the blend stage contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kconv.config import Config, DataConfig, EnsembleConfig, ModelConfig, ValidationConfig
from kconv.context import RunContext, write_json
from kconv.errors import StageError
from kconv.stages.blend import _combine, _improves, _rank_mean, _stack_predictions
from kconv.stages.report import run_report
from kconv.stages.run import run_pipeline


def _regression_arrays(rows: int = 90, test_rows: int = 17):
    rng = np.random.default_rng(13)
    y = rng.normal(size=rows)
    oofs = [y + rng.normal(scale=0.8, size=rows), y + rng.normal(scale=1.1, size=rows)]
    tests = [rng.normal(size=test_rows), rng.normal(size=test_rows)]
    return oofs, tests, y


def test_stack_regression_has_finite_oof_and_test() -> None:
    oofs, tests, y = _regression_arrays()
    meta_oof, meta_test, info = _stack_predictions(oofs, tests, y, "regression", "auto", 4, 42)

    assert meta_oof.shape == y.shape
    assert meta_test.shape == (len(tests[0]),)
    assert np.isfinite(meta_oof).all()
    assert np.isfinite(meta_test).all()
    assert info["meta_model"] == "ridge"
    assert info["meta_features"] == 2


def test_stack_binary_probabilities_are_bounded() -> None:
    rng = np.random.default_rng(17)
    n = 120
    y = np.array([0, 1] * (n // 2))
    logits = y * 0.8 + rng.normal(scale=0.5, size=n)
    oofs = [1.0 / (1.0 + np.exp(-logits)), 1.0 / (1.0 + np.exp(-(logits + rng.normal(scale=0.4, size=n))))]
    tests = [rng.uniform(0.1, 0.9, size=25), rng.uniform(0.1, 0.9, size=25)]

    meta_oof, meta_test, info = _stack_predictions(oofs, tests, y, "binary", "logreg", 4, 42)

    assert meta_oof.shape == (n,)
    assert meta_test.shape == (25,)
    assert np.all((meta_oof >= 0.0) & (meta_oof <= 1.0))
    assert np.all((meta_test >= 0.0) & (meta_test <= 1.0))
    assert info["meta_model"] == "logreg"


def test_stack_multiclass_probabilities_sum_to_one() -> None:
    rng = np.random.default_rng(19)
    n = 150
    y = np.repeat(np.arange(3), n // 3)
    first = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    second = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    tests = [rng.dirichlet([1.0, 1.0, 1.0], size=20), rng.dirichlet([1.0, 1.0, 1.0], size=20)]

    meta_oof, meta_test, info = _stack_predictions([first, second], tests, y, "multiclass", "auto", 3, 42)

    assert meta_oof.shape == (n, 3)
    assert meta_test.shape == (20, 3)
    np.testing.assert_allclose(meta_oof.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(meta_test.sum(axis=1), 1.0, atol=1e-6)
    assert info["meta_features"] == 6


def test_stack_lgbm_regression_is_finite() -> None:
    oofs, tests, y = _regression_arrays(rows=60, test_rows=12)
    meta_oof, meta_test, info = _stack_predictions(oofs, tests, y, "regression", "lgbm", 3, 42)

    assert meta_oof.shape == y.shape
    assert meta_test.shape == (12,)
    assert np.isfinite(meta_oof).all()
    assert np.isfinite(meta_test).all()
    assert info["meta_model"] == "lgbm"


def test_stack_lgbm_binary_probabilities_are_bounded() -> None:
    rng = np.random.default_rng(41)
    n = 90
    y = np.array([0, 1] * (n // 2))
    signal = y * 0.9 + rng.normal(scale=0.5, size=n)
    oofs = [1.0 / (1.0 + np.exp(-signal)), 1.0 / (1.0 + np.exp(-(signal + 0.3)))]
    tests = [rng.uniform(0.1, 0.9, size=15), rng.uniform(0.1, 0.9, size=15)]

    meta_oof, meta_test, info = _stack_predictions(oofs, tests, y, "binary", "lgbm", 3, 42)

    assert meta_oof.shape == (n,)
    assert meta_test.shape == (15,)
    assert np.all((meta_oof >= 0.0) & (meta_oof <= 1.0))
    assert np.all((meta_test >= 0.0) & (meta_test <= 1.0))
    assert info["meta_model"] == "lgbm"


def test_min_improvement_uses_the_metric_direction() -> None:
    # A maximized metric must clear baseline + margin, not baseline - margin.
    assert _improves("roc_auc", 0.81, 0.80, 0.005)
    assert not _improves("roc_auc", 0.805, 0.80, 0.01)
    # A minimized metric must be below baseline - margin.
    assert _improves("rmse", 0.79, 0.80, 0.005)
    assert not _improves("rmse", 0.795, 0.80, 0.01)


def test_multiclass_weighted_blend_is_a_probability_matrix() -> None:
    predictions = [
        np.array([[0.50000001, 0.3, 0.19999999], [0.2, 0.7, 0.1]]),
        np.array([[0.50000001, 0.3, 0.19999999], [0.2, 0.7, 0.1]]),
    ]
    combined = _combine(np.array([0.5, 0.5]), predictions, multiclass=True)

    assert combined.shape == (2, 3)
    assert (combined >= 0.0).all()
    assert (combined <= 1.0).all()
    np.testing.assert_allclose(combined.sum(axis=1), 1.0, atol=1e-12)


def test_multiclass_rank_mean_is_a_probability_matrix() -> None:
    rng = np.random.default_rng(29)
    predictions = [
        rng.dirichlet([1.0, 1.0, 1.0], size=40),
        rng.dirichlet([2.0, 1.0, 1.0], size=40),
    ]

    combined = _rank_mean(predictions, multiclass=True)

    assert combined.shape == (40, 3)
    assert np.isfinite(combined).all()
    np.testing.assert_allclose(combined.sum(axis=1), 1.0, atol=1e-12)


def test_stack_rejects_tiny_oof() -> None:
    rng = np.random.default_rng(23)
    y = rng.normal(size=5)
    with pytest.raises(StageError, match="at least 8"):
        _stack_predictions([y, y], [y[:2], y[:2]], y, "regression", "auto", 3, 42)


def _make_dataset(project: Path) -> None:
    rng = np.random.default_rng(29)
    n = 300
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        }
    )
    logit = 1.2 * frame["x0"] - 0.8 * frame["x1"] + np.where(frame["cat"] == "a", 0.7, -0.2)
    frame["target"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    data = project / "data"
    data.mkdir(parents=True)
    frame.iloc[:240].to_csv(data / "train.csv", index=False)
    frame.iloc[240:].drop(columns=["target"]).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": frame.iloc[240:]["id"], "target": 0}).to_csv(data / "sample_submission.csv", index=False)


def test_blend_stage_runs_stack_and_records_meta_model(tmp_path: Path) -> None:
    project = tmp_path / "stack"
    project.mkdir()
    _make_dataset(project)
    config = Config(
        project="stack",
        seed=42,
        task="binary",
        metric="roc_auc",
        data=DataConfig(
            train=str(project / "data" / "train.csv"),
            test=str(project / "data" / "test.csv"),
            sample_submission=str(project / "data" / "sample_submission.csv"),
            target="target",
            id="id",
        ),
        validation=ValidationConfig(strategy="stratified_kfold", n_splits=3, random_state=42),
        models=[
            ModelConfig(
                name="lgbm",
                params={"n_estimators": 80, "verbosity": -1, "num_threads": 1},
            ),
            ModelConfig(name="histgb", params={"max_iter": 80}),
        ],
        ensemble=EnsembleConfig(method="stack", meta_splits=3),
    )

    ctx = run_pipeline(project, project / "config.yaml", config, until="blend")

    payload = pd.read_json(ctx.run_dir / "03_blend" / "blend.json", typ="series")
    assert payload["stack"] is not None
    assert payload["stack"]["meta_model"] == "logreg"
    assert (ctx.run_dir / "03_blend" / "meta_model.joblib").exists()
    assert payload["stack"]["meta_features"] == 2
    assert payload["method"] in {"stack", "best_single", "mean"}
    oof = pd.read_parquet(ctx.run_dir / "03_blend" / "oof.parquet")
    assert len(oof) == 240
    assert not oof.isna().any().any()


def test_report_displays_meta_model_instead_of_zero_weights(tmp_path: Path) -> None:
    project = tmp_path / "report"
    project.mkdir()
    config = Config(project="stack-report", data=DataConfig(train="train.csv"))
    ctx = RunContext.create(project, project / "config.yaml", config)
    write_json(
        ctx.run_dir / "03_blend" / "blend.json",
        {
            "method": "stack",
            "models": ["lgbm", "xgb"],
            "weights": {},
            "model_scores": {"lgbm": 1.1, "xgb": 1.2},
            "oof_score": 1.05,
            "fallback_used": False,
            "stack": {"meta_model": "ridge", "meta_splits": 5, "meta_features": 2},
        },
    )

    run_report(ctx)

    text = (ctx.run_dir / "05_report" / "report.md").read_text(encoding="utf-8")
    assert "Meta model: `ridge`" in text
    assert "Meta CV folds: 5" in text
    assert "| Model | OOF |" in text
    assert "| Model | Weight |" not in text
