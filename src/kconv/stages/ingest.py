"""Ingest stage: detect the dataset, normalize it, write 00_ingest artifacts."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from kconv.autodetect import detect_dataset
from kconv.context import RunContext, data_hash, file_sha256, write_json
from kconv.contracts import DatasetSpec
from kconv.errors import DataError
from kconv.io import read_table, write_table

STAGE = "00_ingest"
_ROLE_FIELDS = (
    "feature_columns",
    "numeric_columns",
    "categorical_columns",
    "datetime_columns",
    "text_columns",
    "ignore_columns",
)


def _resolve(ctx: RunContext, value: str | Path) -> Path:
    """Config paths resolve against the project dir unless already absolute."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ctx.project_dir / candidate
    return candidate.resolve()


def _relative(ctx: RunContext, path: Path) -> str:
    """Path relative to the project dir when possible, absolute otherwise."""
    try:
        rel = os.path.relpath(path, ctx.project_dir)
    except ValueError:  # different drives on Windows
        return str(path)
    if rel.startswith(".."):
        return str(path)
    return rel.replace(os.sep, "/")


def _apply_config_ignore(spec: DatasetSpec, ctx: RunContext, warnings: list[str]) -> None:
    """Move columns from data.ignore out of the feature/role lists."""
    known: set[str] = set(spec.feature_columns)
    for field in _ROLE_FIELDS:
        known.update(getattr(spec, field))
    known.add(spec.target)
    if spec.id_col:
        known.add(spec.id_col)
    for column in dict.fromkeys(ctx.config.data.ignore):
        if column == spec.target:
            warnings.append(f"data.ignore column '{column}' is the target; skipped.")
            continue
        if column not in known:
            warnings.append(f"data.ignore column '{column}' not found in train; skipped.")
            continue
        if column not in spec.ignore_columns:
            spec.ignore_columns.append(column)
        for field in _ROLE_FIELDS:
            setattr(spec, field, [c for c in getattr(spec, field) if c != column])


def _sample_train(
    train_df: pd.DataFrame,
    pct: int,
    seed: int,
    target: str | None,
    task: str,
) -> pd.DataFrame:
    """Return a deterministic, order-preserving sample of the training frame."""
    if pct >= 100 or len(train_df) <= 1:
        return train_df
    n_total = max(1, int(round(len(train_df) * pct / 100)))
    if task in {"binary", "multiclass"} and target and target in train_df.columns:
        parts: list[pd.DataFrame] = []
        for _, group in train_df.groupby(target, dropna=False, sort=False):
            n_group = max(1, min(len(group), int(round(len(group) * pct / 100))))
            parts.append(group.sample(n=n_group, random_state=seed))
        return pd.concat(parts).sort_index()
    return train_df.sample(n=min(n_total, len(train_df)), random_state=seed).sort_index()


def _data_card(
    ctx: RunContext,
    spec: DatasetSpec,
    source: dict[str, str | None],
    artifacts: dict[str, str],
    train_df: pd.DataFrame,
    test_rows: int | None,
    schema_diff: dict | None = None,
    rows_before_sample: int | None = None,
) -> dict:
    card: dict = {
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": ctx.run_id,
        "project": ctx.config.project,
        "source": source,
        "artifacts": artifacts,
        "rows": {"train": int(len(train_df)), "test": test_rows},
        "sample_pct": int(ctx.config.data.sample_pct),
        "rows_before_sample": int(rows_before_sample) if rows_before_sample is not None else None,
        "data_hash": ctx.data_hash,
        "data_files": list(ctx.data_files),
        "schema_diff": schema_diff,
        "target": spec.target,
        "id_col": spec.id_col,
        "task": spec.task,
        "metric": spec.metric,
        "roles": {field: list(getattr(spec, field)) for field in _ROLE_FIELDS},
        "missing_cells_train": int(train_df.isna().sum().sum()),
        "duplicate_rows_train": int(train_df.duplicated().sum()),
        "warnings": list(spec.warnings),
    }
    if spec.task in {"binary", "multiclass"}:
        counts = train_df[spec.target].value_counts(dropna=True)
        if len(counts) <= 50:
            card["class_balance"] = {str(k): int(v) for k, v in counts.items()}
    return card


def run_ingest(ctx: RunContext) -> dict:
    """Load config.data, detect the dataset and write normalized artifacts.

    Writes ``00_ingest/train.parquet``, ``test.parquet`` (when a test file
    exists), ``schema.json`` and ``data_card.json``; sets ``ctx.dataset``.
    """
    data = ctx.config.data
    warnings: list[str] = []

    train_path = _resolve(ctx, data.train)
    test_override = _resolve(ctx, data.test) if data.test else None
    ss_override = _resolve(ctx, data.sample_submission) if data.sample_submission else None
    task_override = None if ctx.config.task == "auto" else ctx.config.task
    metric_override = None if ctx.config.metric in {None, "auto"} else ctx.config.metric

    spec = detect_dataset(
        train_path,
        target=data.target,
        test=test_override,
        sample_submission=ss_override,
        id=data.id,
        task=task_override,
        metric=metric_override,
    )
    _apply_config_ignore(spec, ctx, warnings)

    raw_paths = [
        path
        for path in (spec.train, spec.test, spec.sample_submission)
        if path is not None
    ]
    ctx.data_files = [
        {
            "path": str(path.resolve()),
            "relative_path": _relative(ctx, path),
            "sha256": file_sha256(path),
        }
        for path in raw_paths
        if path.exists()
    ]
    ctx.data_hash = data_hash([Path(item["path"]) for item in ctx.data_files]) if ctx.data_files else None

    source = {
        "train": _relative(ctx, spec.train),
        "test": _relative(ctx, spec.test) if spec.test else None,
        "sample_submission": _relative(ctx, spec.sample_submission) if spec.sample_submission else None,
    }

    train_df = read_table(spec.train)
    missing_target = int(train_df[spec.target].isna().sum())
    if missing_target:
        raise DataError(
            f"Target column '{spec.target}' contains {missing_target:,} missing value(s).",
            "Fill or drop rows with a missing target before training; kconv does not impute the target.",
        )
    rows_before_sample = int(len(train_df))
    sample_pct = int(ctx.config.data.sample_pct)
    if sample_pct < 100 and len(train_df) > 1:
        train_df = _sample_train(train_df, sample_pct, int(ctx.config.seed), spec.target, spec.task)
        warnings.append(
            f"Training data sampled to {sample_pct}%: {len(train_df):,} of {rows_before_sample:,} rows."
        )
    artifacts_paths: dict[str, Path] = {}
    train_out = ctx.artifact(STAGE, "train.parquet")
    write_table(train_df, train_out)
    artifacts_paths["train"] = train_out

    test_rows: int | None = None
    test_out: Path | None = None
    schema_diff: dict | None = None
    if spec.test is not None:
        test_df = read_table(spec.test)
        test_rows = int(len(test_df))
        train_columns = [str(column) for column in train_df.columns]
        test_columns = [str(column) for column in test_df.columns]
        missing_in_test = [column for column in train_columns if column not in test_columns]
        extra_in_test = [column for column in test_columns if column not in train_columns]
        type_changes = [
            {
                "column": column,
                "train": str(train_df[column].dtype),
                "test": str(test_df[column].dtype),
            }
            for column in train_columns
            if column in test_columns and str(train_df[column].dtype) != str(test_df[column].dtype)
        ]
        schema_diff = {
            "missing_in_test": missing_in_test,
            "extra_in_test": extra_in_test,
            "type_changes": type_changes,
        }
        missing_features = [column for column in spec.feature_columns if column not in test_columns]
        if missing_features:
            raise DataError(
                "Test data is missing feature columns: " + ", ".join(missing_features) + ".",
                "Fix the train/test schema mismatch; these columns are required by the fitted pipeline.",
            )
        if extra_in_test:
            warnings.append(
                "Test data has extra columns that will be ignored: " + ", ".join(extra_in_test) + "."
            )
        test_out = ctx.artifact(STAGE, "test.parquet")
        write_table(test_df, test_out)
        artifacts_paths["test"] = test_out

    # The normalized spec points at the parquet artifacts, not the raw files;
    # paths are recorded relative to the project dir when possible.
    spec.train = Path(_relative(ctx, train_out))
    if test_out is not None:
        spec.test = Path(_relative(ctx, test_out))
    if spec.sample_submission is not None:
        spec.sample_submission = Path(_relative(ctx, spec.sample_submission))

    schema_path = ctx.artifact(STAGE, "schema.json")
    write_json(schema_path, spec.to_dict())
    artifacts_paths["schema"] = schema_path
    if schema_diff is not None:
        diff_path = ctx.artifact(STAGE, "schema_diff.json")
        write_json(diff_path, schema_diff)
        artifacts_paths["schema_diff"] = diff_path

    card = _data_card(
        ctx,
        spec,
        source,
        {key: _relative(ctx, value) for key, value in artifacts_paths.items()},
        train_df,
        test_rows,
        schema_diff,
        rows_before_sample=rows_before_sample,
    )
    card_path = ctx.artifact(STAGE, "data_card.json")
    write_json(card_path, card)
    artifacts_paths["data_card"] = card_path

    all_warnings = list(spec.warnings) + warnings
    for message in all_warnings:
        if message not in ctx.warnings:
            ctx.warnings.append(message)

    ctx.dataset = spec

    return {
        "status": "ok",
        "artifacts": [_relative(ctx, p) for p in artifacts_paths.values()],
        "warnings": all_warnings,
        "data_hash": ctx.data_hash,
        "data_files": list(ctx.data_files),
    }
