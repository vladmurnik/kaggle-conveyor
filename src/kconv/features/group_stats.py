"""Declarative group-statistics block.

The block deliberately compiles to the existing reviewed ``fold_plan``
representation.  That keeps one implementation of fit-in-fold group
aggregates and means a pipeline cannot accidentally introduce arbitrary
Python through a group feature.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from kconv.contracts import DatasetSpec

GROUP_STATISTICS = frozenset({"mean", "std", "count"})
DEFAULT_GROUP_STATS = ("mean", "std", "count")
DEFAULT_MAX_COLUMNS = 8
_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FEATURE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,39}$")


def _check_identifier(value: str, role: str) -> str:
    """Keep generated fold expressions parseable as bare identifiers."""

    if not _COLUMN_RE.fullmatch(value):
        raise ValueError(
            f"group_statistics {role} {value!r} is not a supported column identifier; "
            "use ASCII letters, digits and underscores, starting with a letter or underscore"
        )
    return value


def _feature_name(raw: str) -> str:
    """Respect the fold-plan 40-character feature-name limit."""

    if _FEATURE_NAME_RE.fullmatch(raw):
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:32]
    return f"group__{digest}"



@dataclass(frozen=True)
class GroupStatisticsPlan:
    """Resolved group block plus the fold-plan payload it compiles to."""

    group_col: str
    columns: tuple[str, ...]
    stats: tuple[str, ...]
    plan: dict[str, Any]


def _string_list(value: object, field: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        if not all(isinstance(item, str) and item for item in value):
            raise ValueError(f"{field} must contain non-empty strings")
        return list(dict.fromkeys(value))
    raise ValueError(f"{field} must be a string or a list of strings")


def build_group_statistics_plan(
    params: dict[str, Any],
    spec: DatasetSpec,
) -> GroupStatisticsPlan:
    """Validate block parameters and compile them to a fold-plan dict.

    ``group_col='auto'`` picks the first categorical feature, falling back to
    the first feature column.  Numeric columns default to all numeric feature
    columns and are capped by ``max_columns`` to keep wide tables manageable.
    ``count`` is emitted once per group column and needs no numeric column.
    """

    feature_columns = [column for column in spec.feature_columns if column not in {spec.target, spec.id_col}]
    numeric_columns = [column for column in spec.numeric_columns if column in feature_columns]
    categorical_columns = [column for column in spec.categorical_columns if column in feature_columns]

    raw_group = params.get("group_col", "auto")
    if raw_group in (None, "auto"):
        group_col = categorical_columns[0] if categorical_columns else (feature_columns[0] if feature_columns else "")
    elif isinstance(raw_group, str) and raw_group:
        group_col = raw_group
    else:
        raise ValueError("group_statistics.group_col must be 'auto' or a non-empty column name")
    if not group_col or group_col not in feature_columns:
        raise ValueError(
            "group_statistics.group_col must name a feature column; "
            f"got {group_col!r}"
        )
    _check_identifier(group_col, "group_col")

    raw_stats = params.get("stats", list(DEFAULT_GROUP_STATS))
    stats = _string_list(raw_stats, "group_statistics.stats")
    unknown_stats = [stat for stat in stats if stat not in GROUP_STATISTICS]
    if unknown_stats:
        raise ValueError(
            "group_statistics.stats supports mean, std and count; "
            f"got {unknown_stats!r}"
        )
    if not stats:
        raise ValueError("group_statistics.stats must not be empty")

    raw_columns = params.get("columns", "auto")
    if raw_columns in (None, "auto"):
        columns = list(numeric_columns)
    else:
        columns = _string_list(raw_columns, "group_statistics.columns")
        unknown_columns = [column for column in columns if column not in feature_columns]
        if unknown_columns:
            raise ValueError(
                "group_statistics.columns contains unknown columns: "
                f"{unknown_columns!r}"
            )
        non_numeric = [column for column in columns if column not in numeric_columns]
        if non_numeric:
            raise ValueError(
                "group_statistics.columns must be numeric; "
                f"got {non_numeric!r}"
            )

    raw_max = params.get("max_columns", DEFAULT_MAX_COLUMNS)
    if isinstance(raw_max, bool) or not isinstance(raw_max, int) or not 1 <= raw_max <= 100:
        raise ValueError("group_statistics.max_columns must be an integer between 1 and 100")
    columns = columns[:raw_max]
    for column in columns:
        _check_identifier(column, "column")

    if any(stat in {"mean", "std"} for stat in stats) and not columns:
        raise ValueError(
            "group_statistics mean/std needs at least one numeric feature column; "
            "use stats: [count] for count-only group features"
        )

    features: list[dict[str, str]] = []
    for column in columns:
        for stat in stats:
            if stat == "count":
                continue
            features.append(
                {
                    "name": _feature_name(f"group__{group_col}__{column}__{stat}"),
                    "expression": f"group_{stat}({column}, {group_col})",
                }
            )
    if "count" in stats:
        features.append(
            {
                "name": _feature_name(f"group__{group_col}__count"),
                "expression": f"group_count({group_col})",
            }
        )
    if not features:
        raise ValueError("group_statistics produced no features")

    return GroupStatisticsPlan(
        group_col=group_col,
        columns=tuple(columns),
        stats=tuple(stats),
        plan={"features": features},
    )
