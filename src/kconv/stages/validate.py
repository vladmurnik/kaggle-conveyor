"""Validate stage: create and persist cross-validation folds."""

from __future__ import annotations

import pandas as pd

from kconv.context import RunContext, read_json, write_json
from kconv.contracts import DatasetSpec
from kconv.cv.splitters import build_folds, choose_strategy, save_folds
from kconv.diagnostics import run_adversarial
from kconv.errors import DataError, StageError

STAGE = "01_validate"


def _dataset(ctx: RunContext) -> DatasetSpec:
    if ctx.dataset is not None:
        return ctx.dataset
    raw = read_json(ctx.run_dir / "00_ingest" / "schema.json", default=None)
    if not isinstance(raw, dict):
        raise StageError("Missing ingest schema.", "Run the ingest stage first.")
    return DatasetSpec.from_dict(raw)


def run_validate(ctx: RunContext) -> dict:
    spec = _dataset(ctx)
    train = pd.read_parquet(ctx.run_dir / "00_ingest" / "train.parquet")
    if spec.target not in train.columns:
        raise StageError(
            f"Target '{spec.target}' is missing from the ingested train table.",
            "Rerun ingest; the schema and data are inconsistent.",
        )
    missing_target = int(train[spec.target].isna().sum())
    if missing_target:
        raise DataError(
            f"Target column '{spec.target}' contains {missing_target:,} missing value(s).",
            "Rerun ingest after filling or dropping rows with a missing target.",
        )
    y = train[spec.target].to_numpy()
    groups = None
    time_values = None
    if ctx.config.validation.group_col:
        column = ctx.config.validation.group_col
        if column not in train.columns:
            raise DataError(f"Group column '{column}' not found in train.", "Set validation.group_col correctly.")
        groups = train[column].to_numpy()
    if ctx.config.validation.time_col:
        column = ctx.config.validation.time_col
        if column not in train.columns:
            raise DataError(f"Time column '{column}' not found in train.", "Set validation.time_col correctly.")
        time_values = train[column].to_numpy()

    folds = build_folds(ctx.config, y, groups=groups, time=time_values, task=spec.task)
    path = ctx.artifact(STAGE, "folds.parquet")
    save_folds(folds, path, len(train))
    summary = {
        "strategy": choose_strategy(ctx.config, y, task=spec.task),
        "n_splits": len(folds),
        "rows": int(len(train)),
        "validation_rows": int(sum(len(f.valid_idx) for f in folds)),
        "fold_sizes": [int(len(f.valid_idx)) for f in folds],
        "group_col": ctx.config.validation.group_col,
        "time_col": ctx.config.validation.time_col,
    }
    summary_path = ctx.artifact(STAGE, "cv_summary.json")
    write_json(summary_path, summary)

    if ctx.config.validation.adversarial:
        test_path = ctx.run_dir / "00_ingest" / "test.parquet"
        test = pd.read_parquet(test_path) if test_path.exists() else pd.DataFrame()
        try:
            adversarial = run_adversarial(
                train,
                test,
                list(spec.feature_columns),
                folds=ctx.config.validation.adversarial_folds,
                threshold=ctx.config.validation.adversarial_threshold,
                seed=ctx.config.seed,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must not stop a submission
            adversarial = {
                "enabled": True,
                "status": "skipped",
                "warning": False,
                "message": f"Adversarial validation failed: {exc}",
            }
    else:
        adversarial = {"enabled": False, "status": "disabled", "warning": False}
    adversarial_path = ctx.artifact(STAGE, "adversarial.json")
    write_json(adversarial_path, adversarial)
    if adversarial.get("warning") and adversarial.get("message") not in ctx.warnings:
        ctx.warnings.append(str(adversarial["message"]))

    artifacts = [
        str(path.relative_to(ctx.run_dir)).replace("\\", "/"),
        str(summary_path.relative_to(ctx.run_dir)).replace("\\", "/"),
        str(adversarial_path.relative_to(ctx.run_dir)).replace("\\", "/"),
    ]
    return {
        "status": "ok",
        "artifacts": artifacts,
        "n_splits": len(folds),
        "strategy": summary["strategy"],
        "adversarial": adversarial,
    }
