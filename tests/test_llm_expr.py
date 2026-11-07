"""Safe materialization of LLM-proposed features: strict whitelist, no eval()."""

from __future__ import annotations

import ast
import inspect
import sys

import numpy as np
import pandas as pd
import pytest

from kconv.errors import KconvError
from kconv.features import llm_expr
from kconv.features.llm_expr import evaluate_expression, materialize_plan, parse_expression
from kconv.llm_features import FeaturePlan, FeatureProposal


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [4.0, 0.0, 8.0]},
        index=[10, 11, 12],
    )


def _plan(*proposals: FeatureProposal) -> FeaturePlan:
    return FeaturePlan(proposals=proposals, warnings=())


def _proposal(name: str, expression: str, kind: str = "numeric") -> FeatureProposal:
    return FeatureProposal(name=name, kind=kind, expression=expression, rationale="test")


# ----------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------


def test_parse_expression_returns_expression_tree() -> None:
    tree = parse_expression("a + b", ["a", "b"])
    assert isinstance(tree, ast.Expression)
    assert isinstance(tree.body, ast.BinOp)


def test_parse_expression_accepts_parentheses_and_unary_minus() -> None:
    tree = parse_expression("-(a + b) * 2", ["a", "b"])
    assert isinstance(tree.body, ast.BinOp)


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------


def test_evaluate_addition(frame: pd.DataFrame) -> None:
    values = evaluate_expression("a + b", frame)
    pd.testing.assert_series_equal(values, pd.Series([5.0, 2.0, 11.0], index=[10, 11, 12], name="expression"))


def test_evaluate_division(frame: pd.DataFrame) -> None:
    values = evaluate_expression("a / b", frame)
    assert np.isnan(values.loc[11])
    assert values.loc[10] == pytest.approx(0.25)
    assert values.loc[12] == pytest.approx(0.375)


def test_evaluate_subtraction_and_multiplication(frame: pd.DataFrame) -> None:
    assert evaluate_expression("a - b", frame).tolist() == pytest.approx([-3.0, 2.0, -5.0])
    assert evaluate_expression("a * b", frame).tolist() == pytest.approx([4.0, 0.0, 24.0])


def test_evaluate_unary_minus(frame: pd.DataFrame) -> None:
    values = evaluate_expression("-a", frame)
    assert values.tolist() == [-1.0, -2.0, -3.0]


def test_evaluate_unary_plus(frame: pd.DataFrame) -> None:
    assert evaluate_expression("+a", frame).tolist() == [1.0, 2.0, 3.0]


def test_evaluate_integer_and_float_literals(frame: pd.DataFrame) -> None:
    assert evaluate_expression("2", frame).tolist() == [2.0, 2.0, 2.0]
    assert evaluate_expression("1.5", frame).tolist() == [1.5, 1.5, 1.5]
    assert evaluate_expression("a * 2 + 0.5", frame).tolist() == pytest.approx([2.5, 4.5, 6.5])


def test_evaluate_returns_float_series_named_expression(frame: pd.DataFrame) -> None:
    values = evaluate_expression("a + b", frame)
    assert values.name == "expression"
    assert values.dtype == np.float64
    assert list(values.index) == list(frame.index)


def test_evaluate_division_by_zero_is_nan(frame: pd.DataFrame) -> None:
    values = evaluate_expression("a / (b - b)", frame)
    assert values.isna().all()
    assert not np.isinf(values.to_numpy()).any()


def test_evaluate_division_by_zero_in_series_is_nan(frame: pd.DataFrame) -> None:
    values = evaluate_expression("a / b", frame)
    assert np.isnan(values.loc[11])
    assert not np.isinf(values.to_numpy()).any()


def test_evaluate_col_references_with_spaces_and_punctuation() -> None:
    frame = pd.DataFrame(
        {
            "Flight Distance": [100.0, 200.0],
            "Departure/Arrival time convenient": [1.0, 2.0],
        }
    )
    values = evaluate_expression(
        'col("Flight Distance") + col("Departure/Arrival time convenient")',
        frame,
    )
    assert values.tolist() == pytest.approx([101.0, 202.0])


def test_evaluate_rejects_unknown_col_reference(frame: pd.DataFrame) -> None:
    with pytest.raises(KconvError):
        evaluate_expression('col("missing column") + a', frame)


def test_evaluate_restricts_columns_when_given(frame: pd.DataFrame) -> None:
    with pytest.raises(KconvError):
        evaluate_expression("a + b", frame, columns=["a"])


def test_evaluate_does_not_modify_the_frame(frame: pd.DataFrame) -> None:
    before = frame.copy()
    evaluate_expression("a + b", frame)
    pd.testing.assert_frame_equal(frame, before)


# ----------------------------------------------------------------------
# rejections
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "a.__class__",
        "a[0]",
        "__import__('os')",
        "lambda x: x",
        "a if b else 0",
        "a > b",
        "a ** 2",
        "a % b",
        "'text'",
        "True",
        "None",
        "f'{a}'",
        "*a",
        "a; b",
    ],
)
def test_rejects_unsafe_expressions(expression: str, frame: pd.DataFrame) -> None:
    with pytest.raises(KconvError):
        parse_expression(expression, frame.columns)
    with pytest.raises(KconvError):
        evaluate_expression(expression, frame)


def test_rejects_attribute_access(frame: pd.DataFrame) -> None:
    with pytest.raises(KconvError):
        evaluate_expression("a.__class__", frame)


def test_rejects_subscript(frame: pd.DataFrame) -> None:
    with pytest.raises(KconvError):
        evaluate_expression("a[0]", frame)


def test_rejects_import_call(frame: pd.DataFrame) -> None:
    with pytest.raises(KconvError):
        evaluate_expression("__import__('os')", frame)


def test_rejects_lambda(frame: pd.DataFrame) -> None:
    with pytest.raises(KconvError):
        evaluate_expression("lambda x: x", frame)


def test_rejects_unknown_column(frame: pd.DataFrame) -> None:
    with pytest.raises(KconvError):
        evaluate_expression("a + missing", frame)
    with pytest.raises(KconvError):
        parse_expression("missing", frame.columns)


def test_rejects_empty_expression(frame: pd.DataFrame) -> None:
    with pytest.raises(KconvError):
        parse_expression("", frame.columns)
    with pytest.raises(KconvError):
        evaluate_expression("   ", frame)


def test_rejection_error_carries_a_hint(frame: pd.DataFrame) -> None:
    with pytest.raises(KconvError) as excinfo:
        evaluate_expression("a[0]", frame)
    assert excinfo.value.hint


# ----------------------------------------------------------------------
# materialize_plan
# ----------------------------------------------------------------------


def test_materialize_plan_adds_columns(frame: pd.DataFrame) -> None:
    plan = _plan(
        _proposal("sum_ab", "a + b"),
        _proposal("ratio_ab", "a / b"),
    )
    result = materialize_plan(plan, frame)
    assert "sum_ab" in result.columns
    assert "ratio_ab" in result.columns
    assert len(result.columns) == len(frame.columns) + 2
    assert result["sum_ab"].tolist() == pytest.approx([5.0, 2.0, 11.0])
    assert result["sum_ab"].dtype == np.float64
    assert list(result.index) == list(frame.index)


def test_materialize_plan_accepts_col_references() -> None:
    frame = pd.DataFrame({"Flight Distance": [100.0, 200.0], "Delay": [5.0, 10.0]})
    plan = _plan(
        _proposal(
            "distance_per_delay",
            'col("Flight Distance") / Delay',
        )
    )
    result = materialize_plan(plan, frame)
    assert result["distance_per_delay"].tolist() == pytest.approx([20.0, 20.0])


def test_materialize_plan_does_not_modify_the_input_frame(frame: pd.DataFrame) -> None:
    plan = _plan(_proposal("sum_ab", "a + b"))
    materialize_plan(plan, frame)
    assert list(frame.columns) == ["a", "b"]


def test_materialize_plan_treats_categorical_kind_as_numeric(frame: pd.DataFrame) -> None:
    plan = _plan(_proposal("mixed_kind", "a + b", kind="categorical"))
    result = materialize_plan(plan, frame)
    assert result["mixed_kind"].dtype == np.float64
    assert result["mixed_kind"].tolist() == pytest.approx([5.0, 2.0, 11.0])


def test_materialize_plan_rejects_name_colliding_with_existing_column(frame: pd.DataFrame) -> None:
    plan = _plan(_proposal("a", "a + b"))
    with pytest.raises(KconvError):
        materialize_plan(plan, frame)


def test_materialize_plan_rejects_duplicate_names(frame: pd.DataFrame) -> None:
    plan = _plan(_proposal("dup", "a + b"), _proposal("dup", "a - b"))
    with pytest.raises(KconvError):
        materialize_plan(plan, frame)


@pytest.mark.parametrize("name", ["1bad", "with space", "", "a" * 42])
def test_materialize_plan_rejects_invalid_names(name: str, frame: pd.DataFrame) -> None:
    plan = _plan(_proposal(name, "a + b"))
    with pytest.raises(KconvError):
        materialize_plan(plan, frame)


def test_materialize_plan_rejects_invalid_expression(frame: pd.DataFrame) -> None:
    plan = _plan(_proposal("unsafe", "a.__class__"))
    with pytest.raises(KconvError):
        materialize_plan(plan, frame)


def test_materialize_plan_proposals_cannot_reference_each_other(frame: pd.DataFrame) -> None:
    plan = _plan(
        _proposal("first", "a + b"),
        _proposal("second", "first + a"),
    )
    with pytest.raises(KconvError):
        materialize_plan(plan, frame)


def test_materialize_plan_respects_column_allow_list(frame: pd.DataFrame) -> None:
    allowed = materialize_plan(_plan(_proposal("double_a", "a + a")), frame, columns=["a"])
    assert allowed["double_a"].tolist() == pytest.approx([2.0, 4.0, 6.0])
    with pytest.raises(KconvError):
        materialize_plan(_plan(_proposal("uses_b", "a + b")), frame, columns=["a"])


# ----------------------------------------------------------------------
# safety guarantees
# ----------------------------------------------------------------------


def test_module_never_calls_eval_or_exec() -> None:
    module = sys.modules[llm_expr.__name__]
    tree = ast.parse(inspect.getsource(module))
    forbidden_names = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "open",
        "input",
    }
    # ``re.compile`` and friends are fine; only dangerous callables matter.
    forbidden_attributes = {
        "eval",
        "exec",
        "system",
        "popen",
        "Popen",
        "check_output",
        "import_module",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_names
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_attributes
