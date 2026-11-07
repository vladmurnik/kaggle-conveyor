"""Provider-agnostic LLM feature proposals with strict, non-executing validation.

The LLM never gets to run code.  It receives a compact schema/sample prompt and
must return JSON.  The parser accepts arithmetic, the whitelisted row/group
functions and ``col("exact name")`` references over known columns; the caller
decides whether to materialize a proposal and must evaluate it with a clean,
fit-in-fold CV split.
"""

from __future__ import annotations

import ast
import json
import keyword
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from kconv.errors import KconvError
from kconv.features.fold_plan import (
    ALLOWED_FUNCTIONS,
    GROUP_FUNCTIONS,
    ROW_FUNCTIONS,
    parse_fold_expression,
)

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
_SIMPLE_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_KINDS = {"numeric", "categorical"}
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_FUNCTION_NAMES = ALLOWED_FUNCTIONS | {"col"}


@dataclass(frozen=True)
class FeatureProposal:
    """One validated feature idea; ``expression`` is never executed here."""

    name: str
    kind: str
    expression: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeaturePlan:
    """A parsed LLM answer plus non-fatal validation warnings."""

    proposals: tuple[FeatureProposal, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "warnings": list(self.warnings),
        }


def _columns_from_schema(schema: Mapping[str, Any] | Iterable[str]) -> list[str]:
    if isinstance(schema, Mapping):
        columns = schema.get("columns")
        if isinstance(columns, Mapping):
            return [str(column) for column in columns]
        if isinstance(columns, (list, tuple, set)):
            return [str(column) for column in columns]
        return [str(column) for column in schema if column != "columns"]
    return [str(column) for column in schema]


def _is_word_char(value: str) -> bool:
    return bool(value) and (value.isalnum() or value == "_")


def _col_reference(column: str) -> str:
    return f"col({json.dumps(column, ensure_ascii=False)})"


def _read_quoted(text: str, start: int, quote: str) -> int:
    """Return the index just after a quoted string starting at ``start``."""
    index = start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == quote:
            return index + 1
        index += 1
    return len(text)


def normalize_expression(expression: str, columns: Iterable[str]) -> str:
    """Rewrite non-identifier column references to ``col("exact name")``.

    Small local models often emit ``row_mean(Flight Distance, ...)`` even when
    asked for JSON.  Python cannot parse such an expression because the column
    name is not an identifier.  This scanner only rewrites exact, allow-listed
    column names outside strings; it never evaluates anything.  Simple
    identifiers are left untouched so existing plans stay readable.
    """
    column_list = [str(column) for column in columns]
    allowed = set(column_list)
    non_simple = sorted(
        (
            column
            for column in allowed
            if not _SIMPLE_COLUMN_RE.fullmatch(column) or keyword.iskeyword(column)
        ),
        key=len,
        reverse=True,
    )
    output: list[str] = []
    index = 0
    while index < len(expression):
        char = expression[index]
        if char in {"'", '"'}:
            end = _read_quoted(expression, index, char)
            output.append(expression[index:end])
            index = end
            continue
        if char == "`":
            end = expression.find("`", index + 1)
            if end < 0:
                raise KconvError(
                    "Unterminated backtick column reference.",
                    'Use col("exact column name") for columns with spaces or punctuation.',
                )
            name = expression[index + 1 : end]
            if name not in allowed:
                raise KconvError(
                    f"Unknown column '{name}' in expression.",
                    'Use only existing columns; wrap names with spaces in col("...").',
                )
            output.append(_col_reference(name))
            index = end + 1
            continue
        if char.isalpha() or char == "_":
            end = index + 1
            while end < len(expression) and (
                expression[end].isalnum() or expression[end] == "_"
            ):
                end += 1
            word = expression[index:end]
            lookahead = end
            while lookahead < len(expression) and expression[lookahead].isspace():
                lookahead += 1
            if word in _FUNCTION_NAMES and lookahead < len(expression) and expression[lookahead] == "(":
                output.append(word)
                index = end
                continue
            matched: str | None = None
            for column in non_simple:
                if not expression.startswith(column, index):
                    continue
                finish = index + len(column)
                previous = expression[index - 1] if index else ""
                following = expression[finish] if finish < len(expression) else ""
                if (not previous or not _is_word_char(previous)) and (
                    not following or not _is_word_char(following)
                ):
                    matched = column
                    break
            if matched is not None:
                output.append(_col_reference(matched))
                index += len(matched)
                continue
            output.append(word)
            index = end
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _slug_name(name: str) -> str:
    """Turn a model-generated feature name into a safe DataFrame column name."""
    value = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
    if not value:
        value = "feature"
    if value[0].isdigit():
        value = f"f_{value}"
    if keyword.iskeyword(value) or value in _FUNCTION_NAMES:
        value = f"f_{value}"
    return value[:40].rstrip("_") or "feature"


def _unique_feature_name(name: str, taken: set[str]) -> tuple[str, bool]:
    """Return a unique safe feature name and whether it was changed."""
    raw = name.strip()
    if _NAME_RE.fullmatch(raw) and raw not in taken:
        return raw, False
    base = _slug_name(raw)
    if base not in taken:
        return base, True
    for suffix_index in range(2, 1000):
        suffix = f"_{suffix_index}"
        candidate = f"{base[: 40 - len(suffix)].rstrip('_')}{suffix}"
        if candidate not in taken:
            return candidate, True
    raise KconvError(
        "Could not derive a unique feature name.",
        "Use shorter snake_case feature names in the LLM plan.",
    )


def _warning_for_expression_error(index: int, exc: KconvError) -> str:
    if "Unknown column" in exc.message:
        names = sorted(set(re.findall(r"Unknown column '([^']+)'", exc.message)))
        return f"item {index}: unknown columns {names}; skipped"
    return f"item {index}: {exc.message}; skipped"


def _node_column_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "col"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def _check_numeric_operands(node: ast.AST, numeric: set[str], *, group: bool = False) -> None:
    """Reject categorical columns in arithmetic/value positions.

    Group keys may be categorical, but arithmetic, row functions and the value
    argument of group_mean/group_std need numeric columns.  Without this check
    a local model can produce a plan that only fails later inside the training
    fold, after ingest and validation have already run.
    """
    if isinstance(node, ast.Expression):
        _check_numeric_operands(node.body, numeric, group=group)
        return
    if isinstance(node, (ast.Name, ast.Call)):
        name = _node_column_name(node)
        if name is not None:
            if not group and name not in numeric:
                raise KconvError(
                    f"Column '{name}' is not numeric.",
                    "Use it only as a group argument or encode it before generating arithmetic features.",
                )
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = node.func.id
            if function in ROW_FUNCTIONS:
                for argument in node.args:
                    _check_numeric_operands(argument, numeric)
                return
            if function == "group_count":
                for argument in node.args:
                    _check_numeric_operands(argument, numeric, group=True)
                return
            if function in GROUP_FUNCTIONS:
                if node.args:
                    _check_numeric_operands(node.args[0], numeric)
                for argument in node.args[1:]:
                    _check_numeric_operands(argument, numeric, group=True)
                return
        return
    if isinstance(node, ast.BinOp):
        _check_numeric_operands(node.left, numeric, group=group)
        _check_numeric_operands(node.right, numeric, group=group)
        return
    if isinstance(node, ast.UnaryOp):
        _check_numeric_operands(node.operand, numeric, group=group)


def build_prompt(
    schema: Mapping[str, Any] | Iterable[str],
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    max_features: int = 5,
) -> str:
    """Build a compact prompt; samples are filtered to the schema columns.

    Any target/ID/extra keys are dropped even if the caller accidentally passes
    a raw frame row, so a prompt builder cannot leak them by omission.
    """
    if max_features < 1:
        raise KconvError("max_features must be at least 1.")
    columns = _columns_from_schema(schema)
    schema_view: Any = schema if isinstance(schema, Mapping) else {"columns": columns}
    allowed = {column for column in columns}
    samples: list[dict[str, Any]] = []
    for row in list(sample_rows)[:5]:
        if not isinstance(row, Mapping):
            continue
        filtered = {str(key): value for key, value in row.items() if str(key) in allowed}
        samples.append(filtered)
    non_simple_examples = [
        column
        for column in columns
        if not _SIMPLE_COLUMN_RE.fullmatch(column) or keyword.iskeyword(column)
    ][:2]
    example = (
        " + ".join(_col_reference(column) for column in non_simple_examples)
        if non_simple_examples
        else 'col("exact column name")'
    )
    return (
        "You are proposing tabular feature engineering ideas.\n"
        "Return ONLY valid JSON, with no Markdown and no code execution.\n"
        'Schema: {"features":[{"name":"safe_identifier","kind":"numeric|categorical",'
        '"expression":"arithmetic or an allowed row/group function",'
        '"rationale":"one sentence"}]}\n'
        "Feature names must match ^[A-Za-z_][A-Za-z0-9_]{0,40}$ and use snake_case.\n"
        'Reference a column by its exact bare name when it is a simple identifier. '
        'For names with spaces, slashes or other punctuation use col("exact column name").\n'
        f"Example: {example}.\n"
        f"Only use these columns: {json.dumps(columns, ensure_ascii=False)}\n"
        "Allowed functions: row_mean(...), row_std(...), row_min(...), row_max(...), "
        "row_sum(...), group_mean(value, group), group_std(value, group), group_count(group).\n"
        "Do not use the target, IDs, row indexes, external data, or any other Python functions.\n"
        f"Return at most {max_features} features. Prefer simple, interpretable expressions.\n"
        f"Schema metadata: {json.dumps(schema_view, ensure_ascii=False, default=str)[:4000]}\n"
        f"Sample rows (may be truncated): {json.dumps(samples, ensure_ascii=False, default=str)[:4000]}\n"
    )


def _strip_fence(raw: str) -> str:
    text = raw.strip().lstrip("\ufeff")
    if text.startswith("```"):
        text = _JSON_FENCE_RE.sub("", text).strip()
    return text


def _parse_json_text(raw: str) -> Any:
    """Parse JSON even when a local model wraps it in prose or a fence."""
    text = _strip_fence(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                value, _end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, (Mapping, list)):
                return value
        raise first_error from None


def parse_plan(
    raw: str,
    *,
    columns: Iterable[str] | None = None,
    max_features: int = 5,
    reserved: Iterable[str] = (),
    numeric_columns: Iterable[str] | None = None,
) -> FeaturePlan:
    """Parse and validate JSON from an LLM without evaluating expressions.

    ``reserved`` names (target and ID columns) are never used as feature names.
    Non-identifier column references are rewritten to ``col("exact name")`` and
    the expression is checked with the same fold-aware whitelist used by runs.
    When ``numeric_columns`` is provided, categorical columns are rejected in
    arithmetic/value positions but remain valid as group keys.
    """
    if max_features < 1:
        raise KconvError("max_features must be at least 1.")
    try:
        payload = _parse_json_text(raw)
    except json.JSONDecodeError as exc:
        raise KconvError("LLM feature plan is not valid JSON.", "Ask the model for JSON only.") from exc
    if isinstance(payload, Mapping):
        payload = payload.get("features")
    if not isinstance(payload, list):
        raise KconvError("LLM feature plan must contain a 'features' list.")

    column_list = [str(column) for column in columns] if columns is not None else None
    allowed = set(column_list) if column_list is not None else None
    numeric = (
        {str(column) for column in numeric_columns} if numeric_columns is not None else None
    )
    taken = set(column_list or ()) | {str(column) for column in reserved}
    warnings: list[str] = []
    proposals: list[FeatureProposal] = []
    seen_raw: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            warnings.append(f"item {index}: not an object; skipped")
            continue
        raw_name = str(item.get("name", "")).strip()
        kind = str(item.get("kind", "")).strip().lower()
        expression = str(item.get("expression", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        if not raw_name:
            warnings.append(f"item {index}: invalid name; skipped")
            continue
        if raw_name in seen_raw:
            warnings.append(f"item {index}: duplicate name '{raw_name}'; skipped")
            continue
        seen_raw.add(raw_name)
        if kind not in _ALLOWED_KINDS:
            warnings.append(f"item {index}: invalid kind '{kind}'; skipped")
            continue
        if not expression or len(expression) > 1000:
            warnings.append(f"item {index}: empty or overlong expression; skipped")
            continue
        canonical = expression
        if column_list is not None and allowed is not None:
            try:
                canonical = normalize_expression(expression, column_list)
                tree = parse_fold_expression(canonical, allowed)
                if numeric is not None:
                    _check_numeric_operands(tree, numeric)
            except KconvError as exc:
                warnings.append(_warning_for_expression_error(index, exc))
                continue
        name, renamed = _unique_feature_name(raw_name, taken)
        if renamed:
            warnings.append(f"item {index}: renamed feature '{raw_name}' to '{name}'")
        if len(proposals) >= max_features:
            warnings.append(f"truncated to max_features={max_features}")
            break
        taken.add(name)
        proposals.append(
            FeatureProposal(
                name=name,
                kind=kind,
                expression=canonical,
                rationale=rationale[:300],
            )
        )
    if not proposals:
        hint = (
            "Check the JSON shape and that expressions use only known columns; "
            'use col("exact column name") for names with spaces or punctuation.'
        )
        if warnings:
            hint += " Rejected: " + "; ".join(warnings[:5])
        raise KconvError("LLM returned no valid feature proposals.", hint)
    return FeaturePlan(proposals=tuple(proposals), warnings=tuple(warnings))


def propose_features(
    schema: Mapping[str, Any] | Iterable[str],
    sample_rows: Sequence[Mapping[str, Any]],
    llm: Callable[[str], str],
    *,
    max_features: int = 5,
    reserved: Iterable[str] = (),
    numeric_columns: Iterable[str] | None = None,
) -> FeaturePlan:
    """Call an injected LLM function and validate its JSON answer.

    This function does not make network calls.  Pass a callable that takes the
    prompt and returns text, which makes the experiment testable and lets the
    caller choose its provider, model, budget and privacy controls.
    """
    columns = _columns_from_schema(schema)
    prompt = build_prompt(schema, sample_rows, max_features=max_features)
    answer = llm(prompt)
    if not isinstance(answer, str):
        raise KconvError("LLM callable must return a string.")
    return parse_plan(
        answer,
        columns=columns,
        max_features=max_features,
        reserved=reserved,
        numeric_columns=numeric_columns,
    )


__all__ = [
    "FeaturePlan",
    "FeatureProposal",
    "build_prompt",
    "normalize_expression",
    "parse_plan",
    "propose_features",
]
