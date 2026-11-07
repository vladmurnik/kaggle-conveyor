"""Fold-aware feature plans: group mappings never learn from validation/test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kconv.cli import app
from kconv.config import Config, DataConfig, FeatureConfig, load_config
from kconv.contracts import DatasetSpec
from kconv.errors import KconvError
from kconv.features.fold_plan import (
    FoldPlanTransformer,
    expression_columns,
    parse_fold_expression,
    plan_requires_fold_transformer,
)
from kconv.features.llm_expr import materialize_plan
from kconv.features.pipeline import FeatureBuilder
from kconv.io import read_table
from kconv.llm_features import parse_plan


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            "other": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "grp": ["a", "a", "b", "b", "a", "c"],
        }
    )


def _spec() -> DatasetSpec:
    return DatasetSpec(
        train=Path("train.csv"),
        test=None,
        sample_submission=None,
        target="y",
        id_col="id",
        task="binary",
        metric="roc_auc",
        feature_columns=["num", "other", "grp"],
        numeric_columns=["num", "other"],
        categorical_columns=["grp"],
        datetime_columns=[],
        text_columns=[],
    )


def test_group_aggregates_are_fitted_on_train_and_reused_for_test() -> None:
    train = _frame().iloc[:4]
    test = _frame().iloc[4:]
    transformer = FoldPlanTransformer(
        {
            "features": [
                {"name": "grp_mean", "expression": "group_mean(num, grp)"},
                {"name": "grp_std", "expression": "group_std(num, grp)"},
                {"name": "grp_count", "expression": "group_count(grp)"},
            ]
        },
        columns=["num", "other", "grp"],
    ).fit(train)

    train_out = transformer.transform(train)
    test_out = transformer.transform(test)

    assert train_out["grp_mean"].tolist() == [1.5, 1.5, 3.5, 3.5]
    assert train_out["grp_count"].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert test_out["grp_mean"].tolist()[0] == pytest.approx(1.5)  # seen group
    assert test_out["grp_mean"].tolist()[1] == pytest.approx(2.5)  # unseen -> train prior
    assert test_out["grp_count"].tolist()[1] == 0.0
    # Transforming the test set never changes the fitted mapping.
    assert transformer.transform(train)["grp_mean"].tolist() == [1.5, 1.5, 3.5, 3.5]


def test_group_keys_canonicalize_integral_float_dtype() -> None:
    train = pd.DataFrame({"num": [1.0, 2.0], "grp": [1, 2]})
    test = pd.DataFrame({"num": [10.0, 20.0], "grp": [1.0, 2.0]})
    transformer = FoldPlanTransformer(
        {
            "features": [
                {"name": "grp_mean", "expression": "group_mean(num, grp)"},
                {"name": "grp_count", "expression": "group_count(grp)"},
            ]
        },
        columns=["num", "grp"],
    ).fit(train)
    output = transformer.transform(test)
    assert output["grp_mean"].tolist() == [1.0, 2.0]
    assert output["grp_count"].tolist() == [1.0, 1.0]


def test_fold_plan_supports_col_references_for_spaced_columns() -> None:
    frame = pd.DataFrame(
        {
            "Cleanliness": [1.0, 2.0, 3.0, 4.0],
            "Customer Type": ["loyal", "loyal", "new", "new"],
            "Gender": ["m", "f", "m", "f"],
        }
    )
    transformer = FoldPlanTransformer(
        {
            "features": [
                {
                    "name": "cleanliness_by_customer",
                    "expression": 'group_mean(Cleanliness, col("Customer Type"))',
                },
                {
                    "name": "cleanliness_std_by_customer",
                    "expression": 'group_std(Cleanliness, col("Customer Type"))',
                },
                {"name": "gender_count", "expression": 'group_count(col("Gender"))'},
            ]
        },
        columns=list(frame.columns),
    ).fit(frame)
    output = transformer.transform(frame)
    assert output["cleanliness_by_customer"].tolist() == pytest.approx([1.5, 1.5, 3.5, 3.5])
    assert output["cleanliness_std_by_customer"].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert output["gender_count"].tolist() == [2.0, 2.0, 2.0, 2.0]


def test_group_mapping_is_fold_specific() -> None:
    frame = _frame()
    first = FoldPlanTransformer(
        {"features": [{"name": "grp_mean", "expression": "group_mean(num, grp)"}]},
        columns=["num", "grp"],
    ).fit(frame.iloc[:3])
    second = FoldPlanTransformer(
        {"features": [{"name": "grp_mean", "expression": "group_mean(num, grp)"}]},
        columns=["num", "grp"],
    ).fit(frame.iloc[3:])
    assert first.transform(frame.iloc[3:])["grp_mean"].tolist()[0] == pytest.approx(3.0)
    assert second.transform(frame.iloc[3:])["grp_mean"].tolist()[0] == pytest.approx(4.0)


def test_feature_builder_applies_fold_plan() -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "other": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "grp": ["a", "a", "b", "b", "a", "b"],
            "y": [0, 1, 0, 1, 0, 1],
        }
    )
    plan = {
        "features": [
            {"name": "grp_mean", "expression": "group_mean(num, grp)"},
            {"name": "row_total", "expression": "row_sum(num, num)"},
        ]
    }
    config = Config(
        project="f",
        task="binary",
        metric="roc_auc",
        data=DataConfig(train="x", target="y"),
        features=FeatureConfig(fold_plan=plan),
    )
    builder = FeatureBuilder(config, _spec(), "tree")
    train = builder.fit_transform(frame.iloc[:4], frame["y"].iloc[:4].to_numpy())
    valid = builder.transform(frame.iloc[4:])
    assert "grp_mean" in train.columns
    assert "row_total" in train.columns
    assert valid.loc[0, "grp_mean"] == pytest.approx(1.5)
    assert valid.loc[1, "grp_mean"] == pytest.approx(3.5)


@pytest.mark.parametrize(
    "expression",
    [
        "num.attr",
        "group_mean(num, grp) + unknown_col",
        "group_mean(num, 1)",
        "group_mean(num, grp, other)",
        "group_mean(group_mean(num, grp), grp)",
        "row_mean(num)",
        "row_mean(group_mean(num, grp), other)",
        "group_count(num, grp)",
        "group_mean(num, grp, extra=1)",
        "evil(num, grp)",
    ],
)
def test_fold_expression_rejects_unsafe_grammar(expression: str) -> None:
    with pytest.raises(KconvError):
        parse_fold_expression(expression, ["num", "other", "grp"])


def test_expression_columns_excludes_functions_and_constants() -> None:
    assert expression_columns("row_mean(num, 1e3) + group_count(grp)") == {"num", "grp"}
    assert expression_columns("num / other") == {"num", "other"}


def test_fold_expression_rejects_target_or_id_outside_allowlist() -> None:
    with pytest.raises(KconvError, match="Unknown column"):
        parse_fold_expression("group_mean(target, grp)", ["num", "grp"])
    with pytest.raises(KconvError, match="Unknown column"):
        parse_fold_expression("num + id", ["num", "grp"])


def test_parse_and_materialize_contract_for_functions() -> None:
    plan = parse_plan(
        json.dumps(
            {
                "features": [
                    {
                        "name": "grp_mean",
                        "kind": "numeric",
                        "expression": "group_mean(num, grp)",
                        "rationale": "group signal",
                    }
                ]
            }
        ),
        columns=["num", "grp"],
    )
    assert plan.proposals[0].name == "grp_mean"
    assert plan_requires_fold_transformer(plan) is True
    with pytest.raises(KconvError, match="fold-aware"):
        materialize_plan(plan, _frame(), columns=["num", "grp"])
    assert plan_requires_fold_transformer(
        parse_plan(
            json.dumps(
                {
                    "features": [
                        {
                            "name": "ratio",
                            "kind": "numeric",
                            "expression": "num / other",
                            "rationale": "ratio",
                        }
                    ]
                }
            ),
            columns=["num", "other"],
        )
    ) is False


def _cli_dataset(root: Path) -> Path:
    root.mkdir()
    n = 48
    frame = pd.DataFrame(
        {
            "id": list(range(n)),
            "num": [float(i) for i in range(n)],
            "grp": ["a", "b", "c", "d"] * (n // 4),
            "target": [i % 2 for i in range(n)],
        }
    )
    frame.to_csv(root / "train.csv", index=False)
    frame.iloc[:12].drop(columns=["target"]).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": frame.iloc[:12]["id"], "target": 0}).to_csv(
        root / "sample_submission.csv", index=False
    )
    return root


def test_cli_stages_group_plan_as_fold_features(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _cli_dataset(tmp_path / "data")
    source_plan = tmp_path / "group-plan.json"
    source_plan.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "name": "grp_mean_num",
                        "kind": "numeric",
                        "expression": "group_mean(num, grp)",
                        "rationale": "group signal",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
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
            str(source_plan),
            "--models",
            "lgbm",
            "--name",
            "fold-smoke",
            "--no-run",
        ],
    )
    assert result.exit_code == 0, result.output
    config_path = tmp_path / "competitions" / "fold-smoke" / "config.yaml"
    config = load_config(config_path)
    assert config.features.fold_plan is not None
    assert config.features.fold_plan["features"][0]["expression"] == "group_mean(num, grp)"
    assert (config_path.parent / config.data.train).resolve() == (data / "train.csv").resolve()
    staged = read_table((config_path.parent / config.data.train).resolve())
    assert "grp_mean_num" not in staged.columns


def test_cli_stages_arithmetic_plan_as_fold_features(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _cli_dataset(tmp_path / "data")
    source_plan = tmp_path / "arithmetic-plan.json"
    source_plan.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "name": "num_times_two",
                        "kind": "numeric",
                        "expression": "num * 2",
                        "rationale": "arithmetic signal",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
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
            str(source_plan),
            "--models",
            "lgbm",
            "--name",
            "arithmetic-smoke",
            "--no-run",
        ],
    )
    assert result.exit_code == 0, result.output
    config_path = tmp_path / "competitions" / "arithmetic-smoke" / "config.yaml"
    config = load_config(config_path)
    assert config.features.fold_plan is not None
    assert config.features.fold_plan["features"][0]["expression"] == "num * 2"
    assert (config_path.parent / config.data.train).resolve() == (data / "train.csv").resolve()
    staged = read_table((config_path.parent / config.data.train).resolve())
    assert "num_times_two" not in staged.columns


def test_cli_runs_fold_plan_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _cli_dataset(tmp_path / "data")
    source = tmp_path / "run-plan.json"
    source.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "name": "grp_mean_num",
                        "kind": "numeric",
                        "expression": "group_mean(num, grp)",
                        "rationale": "group signal",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
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
            str(source),
            "--models",
            "lgbm",
            "--name",
            "fold-run",
            "--until",
            "report",
        ],
    )
    assert result.exit_code == 0, result.output
    project = tmp_path / "competitions" / "fold-run"
    config = load_config(project / "config.yaml")
    assert (project / config.data.train).resolve() == (data / "train.csv").resolve()
    submissions = list((project / "runs").glob("*/04_submit/submission.csv"))
    assert submissions, result.output
    assert submissions[0].is_file()
    builder_files = list((project / "runs").glob("*/02_train/**/features.joblib"))
    assert builder_files, result.output


def test_cli_apply_writes_plan_before_rejecting_fold_functions(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data = _cli_dataset(tmp_path / "data")
    source = tmp_path / "source-plan.json"
    source.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "name": "grp_mean_num",
                        "kind": "numeric",
                        "expression": "group_mean(num, grp)",
                        "rationale": "group signal",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    reviewed = tmp_path / "reviewed.json"
    sidecar = tmp_path / "reviewed.llm.json"
    applied = tmp_path / "applied.csv"
    result = CliRunner().invoke(
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
            str(source),
            "--output",
            str(reviewed),
            "--apply",
            str(applied),
        ],
    )
    assert result.exit_code != 0
    assert reviewed.is_file()
    assert sidecar.is_file()
    assert not applied.exists()
    assert "plan was written" in result.output.lower()


def test_cli_rejects_plan_using_ignored_column_and_malformed_expression(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data = _cli_dataset(tmp_path / "data")
    runner = CliRunner()
    ignored_plan = tmp_path / "ignored-plan.json"
    ignored_plan.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "name": "grp_mean_num",
                        "kind": "numeric",
                        "expression": "group_mean(num, grp)",
                        "rationale": "group signal",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            str(data),
            "--target",
            "target",
            "--ignore",
            "grp",
            "--experiment",
            "llm-feature-proposals",
            "--approve-llm-features",
            "--llm-plan-file",
            str(ignored_plan),
            "--models",
            "lgbm",
            "--name",
            "ignored-plan",
            "--no-run",
        ],
    )
    assert result.exit_code != 0
    assert "no valid feature proposals" in result.output.lower()

    malformed_plan = tmp_path / "malformed-plan.json"
    malformed_plan.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "name": "bad_row",
                        "kind": "numeric",
                        "expression": "row_mean(num)",
                        "rationale": "malformed",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
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
            str(malformed_plan),
            "--models",
            "lgbm",
            "--name",
            "malformed-plan",
            "--no-run",
        ],
    )
    assert result.exit_code != 0
    assert "at least two arguments" in result.output.lower()


def test_fold_plan_ratio_with_zero_denominator_stays_finite() -> None:
    frame = pd.DataFrame({"num": [1.0, 2.0, 3.0], "den": [0.0, 0.0, 1.0]})
    transformer = FoldPlanTransformer(
        {"features": [{"name": "ratio", "expression": "num / den"}]},
        columns=["num", "den"],
    ).fit(frame)

    output = transformer.transform(frame)

    assert np.isfinite(output["ratio"].to_numpy()).all()
    assert output["ratio"].tolist() == [0.0, 0.0, 3.0]
