"""Fit-in-fold feature materializer for reviewed feature plans.

The arithmetic-only materializer in :mod:`kconv.features.llm_expr` is safe for
row-wise features because every row is independent.  Group aggregates are not:
their mapping must be learned from the outer training fold and then applied
unchanged to validation/test rows.  This module provides that contract.

Allowed function grammar
------------------------

``row_mean``, ``row_std``, ``row_min``, ``row_max``, ``row_sum``
    Row-wise reductions over two or more numeric expressions.

``group_mean(value, group)``, ``group_std(value, group)``
    Label-free group aggregates.  ``value`` is an arithmetic expression;
    ``group`` is a column reference.  Training-fold mappings are stored and
    reused for validation/test; unseen groups use the training-fold prior.

``group_count(group)``
    Number of training-fold rows per group.  Unseen groups map to zero.

Column references may be bare identifiers or ``col("exact name")`` for names
with spaces or punctuation.  Only arithmetic operators and the functions above
are accepted.  Arbitrary calls, attribute access, subscripts, keywords, lambdas
and imports remain rejected.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from kconv.errors import KconvError
from kconv.features.llm_expr import _column_series, _divide

ROW_FUNCTIONS = frozenset({"row_mean", "row_std", "row_min", "row_max", "row_sum"})
GROUP_FUNCTIONS = frozenset({"group_mean", "group_std", "group_count"})
ALLOWED_FUNCTIONS = ROW_FUNCTIONS | GROUP_FUNCTIONS
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
_GRAMMAR_HINT = (
    "Use arithmetic, row_mean/row_std/row_min/row_max/row_sum, or "
    "group_mean/group_std/group_count(value, group)."
)


def _col_name(node: ast.AST) -> str | None:
    """Return the exact column name from ``col("...")``, or None for another node."""
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name) or node.func.id != "col":
        return None
    if node.keywords or len(node.args) != 1:
        raise KconvError(
            'col() requires exactly one string argument, e.g. col("Flight Distance").',
            "Use the exact column name from the dataset schema.",
        )
    argument = node.args[0]
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        raise KconvError(
            'col() requires a string literal, e.g. col("Flight Distance").',
            "Use the exact column name from the dataset schema.",
        )
    return argument.value


def _check_name(node: ast.AST, allowed: frozenset[str], *, role: str) -> str:
    name = node.id if isinstance(node, ast.Name) else _col_name(node)
    if name is None:
        raise KconvError(
            f"{role} must be a column name.",
            'Use a bare column name or col("exact column name") for names with spaces.',
        )
    if name not in allowed:
        raise KconvError(
            f"Unknown column '{name}' in expression.",
            f"Use only existing columns: {', '.join(sorted(allowed)) or '<none>'}.",
        )
    return name


def _contains_group_call(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id in GROUP_FUNCTIONS
        for child in ast.walk(node)
    )


def _check_node(node: ast.AST, allowed: frozenset[str]) -> None:
    if isinstance(node, ast.Expression):
        _check_node(node.body, allowed)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise KconvError(
                f"Unsupported operator {type(node.op).__name__} in expression.",
                "Only +, -, * and / are allowed.",
            )
        _check_node(node.left, allowed)
        _check_node(node.right, allowed)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise KconvError(
                f"Unsupported unary operator {type(node.op).__name__} in expression.",
                "Only unary + and - are allowed.",
            )
        _check_node(node.operand, allowed)
        return
    if isinstance(node, ast.Name):
        if node.id not in allowed:
            raise KconvError(
                f"Unknown column '{node.id}' in expression.",
                f"Use only existing columns: {', '.join(sorted(allowed)) or '<none>'}.",
            )
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise KconvError(
                "Unsupported constant in expression.",
                "Only integer and float literals are allowed.",
            )
        return
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "col":
            _check_name(node, allowed, role="col()")
            return
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise KconvError(
                "Unsupported function in expression.",
                f"Allowed functions: {', '.join(sorted(ALLOWED_FUNCTIONS))}.",
            )
        if node.keywords:
            raise KconvError("Keyword arguments are not allowed in feature functions.")
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            raise KconvError("Starred arguments are not allowed in feature functions.")
        name = node.func.id
        if name in ROW_FUNCTIONS:
            if len(node.args) < 2:
                raise KconvError(f"{name} requires at least two arguments.")
            for argument in node.args:
                if _contains_group_call(argument):
                    raise KconvError(
                        "Group aggregates cannot be nested inside row functions.",
                        "Compute the group feature separately, then reference its name in the plan.",
                    )
                _check_node(argument, allowed)
            return
        if name == "group_count":
            if len(node.args) != 1:
                raise KconvError("group_count requires exactly one group column.")
            _check_name(node.args[0], allowed, role="group_count argument")
            return
        if len(node.args) != 2:
            raise KconvError(f"{name} requires (value, group).")
        _check_node(node.args[0], allowed)
        group = _check_name(node.args[1], allowed, role=f"{name} group")
        if _contains_group_call(node.args[0]):
            raise KconvError(
                f"Nested group aggregate inside {name}(value, group) is not supported.",
                "Compute the inner aggregate as a separate feature.",
            )
        if not isinstance(group, str):  # pragma: no cover - defensive typing guard
            raise KconvError("Group column must be a string.")
        return
    raise KconvError(
        f"Unsupported expression element: {type(node).__name__}.",
        _GRAMMAR_HINT,
    )


def parse_fold_expression(expression: str, columns: Iterable[str]) -> ast.Expression:
    """Parse and validate a fold-aware feature expression."""
    if not isinstance(expression, str):
        raise KconvError("Expression must be a string.", _GRAMMAR_HINT)
    text = expression.strip()
    if not text:
        raise KconvError("Expression is empty.", "Provide a feature expression.")
    allowed = frozenset(str(column) for column in columns)
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError) as exc:
        detail = exc.msg if isinstance(exc, SyntaxError) else str(exc)
        raise KconvError(
            f"Could not parse expression: {detail}.",
            "Use arithmetic or an allowed row/group function.",
        ) from exc
    _check_node(tree, allowed)
    return tree


def _call_key(node: ast.Call) -> tuple[str, str, str]:
    if not isinstance(node.func, ast.Name):
        raise KconvError("Unsupported callable in expression.")
    name = node.func.id
    if name == "group_count":
        group = _name_from_node(node.args[0])
        return name, "", group
    return name, ast.dump(node.args[0], annotate_fields=True, include_attributes=False), _name_from_node(node.args[1])


def _name_from_node(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    name = _col_name(node)
    if name is None:  # pragma: no cover - parser already checks
        raise KconvError("Group argument must be a column name.")
    return name


def _group_keys(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KconvError(f"Group column '{column}' is not present in the frame.")
    series = frame[column]
    if isinstance(series, pd.DataFrame):
        raise KconvError(f"Group column '{column}' is duplicated.")
    # CSV inference often reads the same integer group as int64 in train and
    # float64 in test when one test cell is blank.  Canonicalize integral
    # floats so an unseen-group fallback is not triggered by dtype drift.
    keys = series.astype("string").fillna("__missing__").astype(str)
    return keys.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def _row_reduce(name: str, values: list[pd.Series]) -> pd.Series:
    frame = pd.concat(values, axis=1)
    if name == "row_mean":
        result = frame.mean(axis=1)
    elif name == "row_std":
        result = frame.std(axis=1, ddof=0)
    elif name == "row_min":
        result = frame.min(axis=1)
    elif name == "row_max":
        result = frame.max(axis=1)
    else:
        result = frame.sum(axis=1)
    return result.astype("float64").fillna(0.0)


class FoldPlanTransformer:
    """Apply a reviewed feature plan with fit-in-fold group aggregates.

    ``fit`` parses the plan and learns group mappings from the supplied outer
    training frame.  ``transform`` never changes those mappings, so validation
    and test rows cannot influence training statistics.
    """

    def __init__(self, plan: object, *, columns: Iterable[str] | None = None) -> None:
        self._items = _normalize_plan(plan)
        self._columns = None if columns is None else [str(column) for column in columns]
        self._parsed: list[tuple[str, ast.Expression]] = []
        self._maps: dict[tuple[str, str, str], dict[str, float]] = {}
        self._fallbacks: dict[tuple[str, str, str], float] = {}
        self._fitted = False
        self._frame: pd.DataFrame | None = None

    @property
    def feature_names(self) -> list[str]:
        return [name for name, _ in self._items]

    def fit(self, frame: pd.DataFrame) -> FoldPlanTransformer:
        if not isinstance(frame, pd.DataFrame):
            raise KconvError("FoldPlanTransformer.fit expects a pandas DataFrame.")
        if len(frame) == 0:
            raise KconvError("Cannot fit fold features on an empty frame.")
        allowed = (
            frozenset(str(column) for column in frame.columns)
            if self._columns is None
            else frozenset(self._columns)
        )
        parsed: list[tuple[str, ast.Expression]] = []
        seen: set[str] = set()
        for name, expression in self._items:
            if name in seen:
                raise KconvError(f"Duplicate feature name '{name}' in fold plan.")
            if name in frame.columns:
                raise KconvError(
                    f"Feature '{name}' already exists in the dataset.",
                    "Rename the proposal or drop the existing column.",
                )
            parsed.append((name, parse_fold_expression(expression, allowed)))
            seen.add(name)
        self._parsed = parsed
        self._maps = {}
        self._fallbacks = {}
        self._frame = frame
        try:
            for _, tree in parsed:
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in GROUP_FUNCTIONS
                    ):
                        self._fit_call(node, frame)
        finally:
            self._frame = None
        self._fitted = True
        return self

    def _fit_call(self, node: ast.Call, frame: pd.DataFrame) -> None:
        key = _call_key(node)
        if key in self._maps:
            return
        if key[0] == "group_count":
            keys = _group_keys(frame, key[2])
            counts = keys.value_counts(dropna=False).astype(float)
            self._maps[key] = {str(group): float(count) for group, count in counts.items()}
            self._fallbacks[key] = 0.0
            return
        keys = _group_keys(frame, key[2])
        values = self._eval_node(node.args[0])
        values = pd.to_numeric(values, errors="coerce").astype(float)
        if not values.notna().any():
            raise KconvError(
                f"Group feature value for '{key[0]}' has no numeric values.",
                "Use a numeric expression or a different column.",
            )
        grouped = pd.DataFrame({"group": keys, "value": values}).groupby("group", observed=True)["value"]
        if key[0] == "group_mean":
            mapping = grouped.mean()
            fallback = float(values.mean()) if values.notna().any() else 0.0
        elif key[0] == "group_std":
            mapping = grouped.std(ddof=0)
            fallback = float(values.std(ddof=0)) if values.notna().any() else 0.0
        else:  # pragma: no cover - parser guards this
            raise KconvError(f"Unsupported group function '{key[0]}'.")
        self._maps[key] = {
            str(group): float(value) if pd.notna(value) else 0.0 for group, value in mapping.items()
        }
        self._fallbacks[key] = fallback if np.isfinite(fallback) else 0.0

    def _eval_call(self, node: ast.Call) -> pd.Series:
        if not isinstance(node.func, ast.Name):  # pragma: no cover - parser already checks
            raise KconvError("Unsupported function in expression.")
        name = node.func.id
        if name == "col":
            column = _col_name(node)
            if column is None:  # pragma: no cover - parser already checks
                raise KconvError('col() requires a string literal, e.g. col("Flight Distance").')
            if self._frame is None:
                raise KconvError("FoldPlanTransformer internal frame is missing.")
            return _column_series(column, self._frame)
        if name in ROW_FUNCTIONS:
            values = [self._eval_node(argument) for argument in node.args]
            return _row_reduce(name, values)
        key = _call_key(node)
        if key not in self._maps:
            raise KconvError(
                f"Group mapping for '{name}' is missing.",
                "Call fit() on the outer training fold before transform().",
            )
        if self._frame is None:
            raise KconvError("FoldPlanTransformer internal frame is missing.")
        group = _group_keys(self._frame, key[2])
        if name == "group_count":
            return group.map(self._maps[key]).fillna(0.0).astype("float64")
        return group.map(self._maps[key]).fillna(self._fallbacks[key]).astype("float64")

    def _eval_node(self, node: ast.AST) -> pd.Series:
        frame = self._frame
        if frame is None:
            raise KconvError("FoldPlanTransformer internal frame is missing.")
        if isinstance(node, ast.Constant):
            return pd.Series(float(node.value), index=frame.index, dtype="float64")
        if isinstance(node, ast.Name):
            return _column_series(node.id, frame)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return _divide(left, right)
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        raise KconvError(
            f"Unsupported expression element: {type(node).__name__}.",
            _GRAMMAR_HINT,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise KconvError(
                "FoldPlanTransformer.transform() called before fit().",
                "Fit on the outer training fold first.",
            )
        if not isinstance(frame, pd.DataFrame):
            raise KconvError("FoldPlanTransformer.transform expects a pandas DataFrame.")
        self._frame = frame
        try:
            output: dict[str, pd.Series] = {}
            for name, tree in self._parsed:
                values = self._eval_node(tree.body).astype("float64")
                # Ratios with a zero denominator are intentionally NaN in the
                # arithmetic evaluator.  Feature selection and model adapters
                # require finite matrices, so use the same neutral fallback as
                # row aggregates instead of failing the whole train stage.
                output[name] = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        finally:
            self._frame = None
        return pd.DataFrame(output, index=frame.index)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


def _normalize_plan(plan: object) -> list[tuple[str, str]]:
    if hasattr(plan, "proposals"):
        items: Iterable[Any] = cast(Any, plan).proposals
    elif isinstance(plan, Mapping):
        items = plan.get("features") or plan.get("proposals") or []
    else:
        items = plan  # type: ignore[assignment]
    if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
        raise KconvError("Fold plan must be a list of feature proposals.")
    output: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, Mapping):
            name = str(item.get("name", "")).strip()
            expression = str(item.get("expression", "")).strip()
        else:
            name = str(getattr(item, "name", "")).strip()
            expression = str(getattr(item, "expression", "")).strip()
        if not _NAME_RE.fullmatch(name):
            raise KconvError(f"Invalid fold feature name '{name}'.")
        if not expression:
            raise KconvError(f"Fold feature '{name}' has an empty expression.")
        if len(expression) > 1000:
            raise KconvError(f"Fold feature '{name}' expression is too long.")
        output.append((name, expression))
    if not output:
        raise KconvError("Fold plan contains no feature proposals.")
    if len(output) > 50:
        raise KconvError("Fold plan contains more than 50 feature proposals.")
    return output


def expression_columns(expression: str) -> set[str]:
    """Return raw column names referenced by an expression.

    Function names are excluded, so ``row_mean(a, b)`` yields ``{"a", "b"}``.
    ``col("Flight Distance")`` contributes ``"Flight Distance"``.  The function
    is used by CLI staging to verify that every referenced column also exists
    in the test frame.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        return set()
    function_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in function_names
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "col"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    return names


def plan_requires_fold_transformer(plan: object) -> bool:
    """Return True when a plan contains a row/group function.

    Arithmetic-only plans (including ``col("...")`` references) can be
    materialized before CV; row/group plans need the fold-aware transformer
    even for row functions because the arithmetic evaluator intentionally
    rejects those calls.
    """
    try:
        items = _normalize_plan(plan)
    except KconvError:
        return False
    for _, expression in items:
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ALLOWED_FUNCTIONS:
                    return True
    return False


__all__ = [
    "ALLOWED_FUNCTIONS",
    "GROUP_FUNCTIONS",
    "ROW_FUNCTIONS",
    "FoldPlanTransformer",
    "expression_columns",
    "parse_fold_expression",
    "plan_requires_fold_transformer",
]
