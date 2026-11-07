"""Safe materialization of LLM-proposed tabular features.

The LLM may only suggest simple arithmetic over existing columns.  This module
parses such expressions with :mod:`ast` and evaluates them by walking the
parsed tree.  ``eval()``/``exec()`` are never used, and the accepted grammar is
a strict whitelist.

Allowed grammar (first version, numeric only)::

    expression := term (("+" | "-") term)*
    term       := factor (("*" | "/") factor)*
    factor     := ("+" | "-") factor | number | column | "(" expression ")"
    number     := int | float
    column     := identifier that exists in the frame
                | "col(" string ")"

The ``col("exact name")`` form exists for datasets whose column names are not
valid Python identifiers (for example ``"Flight Distance"``).  The string must
match an allowed column exactly.  Everything else -- arbitrary calls, attribute
access, subscripts, lambdas, comparisons, conditional expressions, booleans,
``None``, f-strings, starred expressions -- is rejected with
:class:`kconv.errors.KconvError`.

Notes
-----
* Only numeric arithmetic is supported.  A proposal whose ``kind`` is
  ``"categorical"`` is still materialized as a numeric column; the kind is
  currently ignored on purpose (documented first-version limitation).
* Operands are coerced with :func:`pandas.to_numeric`, so a non-numeric column
  contributes missing values instead of raising.
* Division by zero yields ``numpy.nan`` (infinities are replaced by NaN).
* The caller owns the leakage question: :func:`materialize_plan` evaluates
  expressions against the columns of the frame it receives, so the frame must
  not contain the target when the proposal is meant to be leakage-free.
* Proposals of one plan cannot reference each other; every expression is
  evaluated against the original frame only.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from kconv.errors import KconvError

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
_GRAMMAR_HINT = (
    'Use column names, col("exact column name") for names with spaces or punctuation, '
    "integer or float literals, the operators + - * / and parentheses."
)


def _check_col_call(node: ast.Call, allowed: frozenset[str]) -> None:
    """Validate the only call this arithmetic grammar accepts: ``col("name")``."""
    if not isinstance(node.func, ast.Name) or node.func.id != "col":
        raise KconvError(
            "Unsupported function in expression.",
            'Only col("exact column name") is allowed here; fold functions are not arithmetic.',
        )
    if node.keywords:
        raise KconvError("Keyword arguments are not allowed in col().")
    if len(node.args) != 1:
        raise KconvError('col() requires exactly one string argument, e.g. col("Flight Distance").')
    argument = node.args[0]
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        raise KconvError('col() requires a string literal, e.g. col("Flight Distance").')
    if argument.value not in allowed:
        raise KconvError(
            f"Unknown column '{argument.value}' in expression.",
            f"Use only existing columns: {', '.join(sorted(allowed)) or '<none>'}.",
        )


def _check_node(node: ast.AST, allowed: frozenset[str]) -> None:
    """Validate an AST recursively against the whitelist."""
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
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise KconvError(
                "Unsupported constant in expression.",
                "Only integer and float literals are allowed.",
            )
        return
    if isinstance(node, ast.Call):
        _check_col_call(node, allowed)
        return
    raise KconvError(
        f"Unsupported expression element: {type(node).__name__}.",
        _GRAMMAR_HINT,
    )


def parse_expression(expression: str, columns: Iterable[str]) -> ast.Expression:
    """Parse and whitelist-check an expression.

    Parameters
    ----------
    expression:
        A single arithmetic expression, e.g. ``"a + b"`` or ``"rate * term"``.
    columns:
        Identifiers the expression may reference.

    Returns
    -------
    ast.Expression
        The parsed tree, ready for the recursive evaluator.

    Raises
    ------
    KconvError
        If the expression cannot be parsed or contains anything outside the
        whitelist.
    """
    if not isinstance(expression, str):
        raise KconvError("Expression must be a string.", _GRAMMAR_HINT)
    text = expression.strip()
    if not text:
        raise KconvError(
            "Expression is empty.",
            "Provide a simple arithmetic expression over existing columns.",
        )
    allowed = frozenset(str(column) for column in columns)
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError) as exc:
        detail = exc.msg if isinstance(exc, SyntaxError) else str(exc)
        raise KconvError(
            f"Could not parse expression: {detail}.",
            "Use a single arithmetic expression over existing columns.",
        ) from exc
    _check_node(tree, allowed)
    return tree


def _column_series(name: str, frame: pd.DataFrame) -> pd.Series:
    """Return one frame column as a float Series."""
    if name not in frame.columns:
        raise KconvError(
            f"Column '{name}' is not present in the frame.",
            "Check the expression against the dataset schema.",
        )
    if (frame.columns == name).sum() > 1:
        raise KconvError(
            f"Column '{name}' is duplicated in the frame.",
            "Rename duplicated columns before materializing features.",
        )
    series = frame[name]
    if not isinstance(series, pd.Series):
        raise KconvError(
            f"Column '{name}' is not a single column.",
            "Rename duplicated columns before materializing features.",
        )
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        raise KconvError(
            f"Column '{name}' is a date/time column and cannot be used in arithmetic.",
            "Pre-compute numeric features from dates before materializing.",
        )
    try:
        numeric = pd.to_numeric(series, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise KconvError(
            f"Column '{name}' cannot be used as a numeric operand.",
            "Only numeric columns can be combined with arithmetic.",
        ) from exc
    return numeric.astype("float64")


def _divide(left: pd.Series, right: pd.Series) -> pd.Series:
    """Divide two Series and map division by zero to NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = left / right
    return result.replace([np.inf, -np.inf], np.nan)


def _eval_node(node: ast.AST, frame: pd.DataFrame) -> pd.Series:
    """Evaluate one whitelisted AST node into a Series aligned to frame.index."""
    if isinstance(node, ast.Constant):
        return pd.Series(float(node.value), index=frame.index, dtype="float64")
    if isinstance(node, ast.Name):
        return _column_series(node.id, frame)
    if isinstance(node, ast.Call):
        # The parser accepts exactly one call form: col("exact column name").
        argument = node.args[0]
        if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
            raise KconvError('col() requires a string literal, e.g. col("Flight Distance").')
        return _column_series(argument.value, frame)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, frame)
        if isinstance(node.op, ast.UAdd):
            return operand
        return -operand
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, frame)
        right = _eval_node(node.right, frame)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return _divide(left, right)
    # The parser already rejected everything else; this is a defensive guard.
    raise KconvError(
        f"Unsupported expression element: {type(node).__name__}.",
        _GRAMMAR_HINT,
    )


def evaluate_expression(
    expression: str,
    frame: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> pd.Series:
    """Evaluate an expression against a frame without using ``eval``.

    Parameters
    ----------
    expression:
        A single arithmetic expression over existing columns.
    frame:
        The source frame; its index defines the alignment of the result.
    columns:
        Identifiers the expression may reference.  ``None`` means every column
        of the frame.

    Returns
    -------
    pandas.Series
        A float Series named ``"expression"`` aligned to ``frame.index``.
        Division by zero produces ``numpy.nan``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise KconvError(
            "Frame must be a pandas DataFrame.",
            "Pass the dataset frame that holds the feature columns.",
        )
    allowed = [str(column) for column in frame.columns] if columns is None else [
        str(column) for column in columns
    ]
    tree = parse_expression(expression, allowed)
    values = _eval_node(tree.body, frame).astype("float64")
    values.name = "expression"
    return values


def materialize_plan(plan, frame: pd.DataFrame, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Add every proposal of a plan to a copy of the frame.

    Parameters
    ----------
    plan:
        A :class:`kconv.llm_features.FeaturePlan` (anything exposing a
        ``proposals`` sequence of proposals with ``name`` and ``expression``).
    frame:
        The frame to extend; it is never modified in place.
    columns:
        Optional allow-list for expression identifiers.  Pass the feature
        columns (target and id excluded) when materializing a plan that came
        from an LLM.  ``None`` means every frame column is allowed and is only
        appropriate when the caller has already validated the expressions.

    Returns
    -------
    pandas.DataFrame
        A copy of ``frame`` with one extra numeric column per proposal.

    Notes
    -----
    * Proposal names must match ``^[A-Za-z_][A-Za-z0-9_]{0,40}$``, must be
      unique inside the plan and must not collide with an existing column.
    * The ``kind`` of a proposal is ignored in this first version: a
      ``"categorical"`` proposal is materialized as a numeric column.
    * Every expression is evaluated against the original frame, so proposals
      cannot reference each other.
    """
    if not isinstance(frame, pd.DataFrame):
        raise KconvError(
            "Frame must be a pandas DataFrame.",
            "Pass the dataset frame that holds the feature columns.",
        )
    allowed = [str(column) for column in frame.columns] if columns is None else [
        str(column) for column in columns
    ]
    try:
        proposals = tuple(plan.proposals)
    except AttributeError as exc:
        raise KconvError(
            "Plan must be a FeaturePlan with proposals.",
            "Parse the LLM answer with kconv.llm_features.parse_plan() first.",
        ) from exc

    existing = {str(column) for column in frame.columns}
    seen: set[str] = set()
    for proposal in proposals:
        name = proposal.name
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise KconvError(
                f"Invalid feature name {name!r}.",
                "Names must match ^[A-Za-z_][A-Za-z0-9_]{0,40}$.",
            )
        if name in existing:
            raise KconvError(
                f"Feature name '{name}' already exists in the frame.",
                "Pick a name that does not collide with an existing column.",
            )
        if name in seen:
            raise KconvError(
                f"Duplicate feature name '{name}' in the plan.",
                "Every proposal must use a unique name.",
            )
        seen.add(name)

    result = frame.copy()
    for proposal in proposals:
        try:
            tree = ast.parse(proposal.expression, mode="eval")
        except (SyntaxError, ValueError):
            tree = None
        fold_call = tree is not None and any(
            isinstance(node, ast.Call)
            and not (isinstance(node.func, ast.Name) and node.func.id == "col")
            for node in ast.walk(tree)
        )
        if fold_call:
            raise KconvError(
                f"Feature '{proposal.name}' uses a fold-aware function.",
                "Pass the plan to 'kconv run --llm-plan-file' (FoldPlanTransformer); "
                "--apply supports arithmetic only.",
            )
        values = evaluate_expression(proposal.expression, frame, columns=allowed)
        result[proposal.name] = values
    return result


__all__ = ["evaluate_expression", "materialize_plan", "parse_expression"]
