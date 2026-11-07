"""Submission stage: turn blended test predictions into submission.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kconv.atomic import write_csv_atomic
from kconv.context import RunContext, read_json, write_json
from kconv.contracts import DatasetSpec
from kconv.errors import SubmissionError

STAGE = "04_submit"


def _spec(ctx: RunContext) -> DatasetSpec:
    raw = read_json(ctx.run_dir / "00_ingest" / "schema.json", default=None)
    if not isinstance(raw, dict):
        raise SubmissionError("Missing ingest schema.", "Run ingest first.")
    return DatasetSpec.from_dict(raw)


def _resolve(ctx: RunContext, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ctx.project_dir / path)


def _proba_or_labels(pred: np.ndarray, task: str, metric: str) -> np.ndarray:
    pred = np.asarray(pred)
    if task == "regression":
        return pred.astype(float)
    if task == "multiclass":
        # Probability metrics need the full class matrix. Accuracy/F1 keep the
        # compact label form, which is what Kaggle's label submission expects.
        if metric in {"roc_auc", "log_loss"} and pred.ndim == 2:
            return pred.astype(float)
        return pred.argmax(axis=1) if pred.ndim == 2 else pred.astype(int)
    if metric in {"roc_auc", "log_loss"}:
        return pred.astype(float)
    return (pred >= 0.5).astype(int)


def run_submit(ctx: RunContext) -> dict:
    spec = _spec(ctx)
    test_path = ctx.run_dir / "00_ingest" / "test.parquet"
    if not test_path.exists():
        raise SubmissionError(
            "No test file was ingested, so a submission cannot be built.",
            "Pass --test pointing at the test CSV/parquet (or use --until train).",
        )
    test = pd.read_parquet(test_path)
    blend_path = ctx.run_dir / "03_blend" / "test_pred.parquet"
    if not blend_path.exists():
        raise SubmissionError("Missing blended test predictions.", "Run the blend stage first.")
    pred_frame = pd.read_parquet(blend_path)
    if "pred" in pred_frame.columns:
        raw_pred = pred_frame["pred"].to_numpy(dtype=float)
        multiclass = False
    else:
        columns = sorted((c for c in pred_frame.columns if c.startswith("pred_")), key=lambda c: int(c.split("_")[-1]))
        raw_pred = pred_frame[columns].to_numpy(dtype=float)
        multiclass = True
    if len(raw_pred) != len(test):
        raise SubmissionError(
            f"Prediction rows ({len(raw_pred)}) do not match test rows ({len(test)}).",
            "Rerun train/blend; the artifacts are inconsistent.",
        )
    if spec.task == "multiclass" and not multiclass:
        raise SubmissionError("Multiclass task but blended predictions are 1-D.", "Rerun blend.")
    if spec.task != "multiclass" and multiclass:
        raise SubmissionError("Binary/regression task but blended predictions are 2-D.", "Rerun blend.")

    sample_path = _resolve(ctx, spec.sample_submission) if spec.sample_submission else None
    sample = None
    sample_id: str | None = None
    if sample_path is not None and sample_path.exists():
        sample = pd.read_csv(sample_path)
    id_column = ctx.config.submit.id_col or spec.id_col
    if sample is not None:
        sample_id = str(sample.columns[0])
        if sample_id in test.columns:
            id_column = sample_id
        elif id_column is not None and id_column in test.columns:
            pass
        else:
            raise SubmissionError(
                f"Cannot map sample-submission id column '{sample_id}' to the test data.",
                "Set submit.id_col or data.id to the shared id column.",
            )
        non_id = [str(column) for column in sample.columns if str(column) != sample_id]
        target_column = ctx.config.submit.target_col
        if target_column is None:
            target_column = non_id[0] if non_id else str(spec.target)
        if spec.task == "multiclass" and spec.metric not in {"roc_auc", "log_loss"} and len(non_id) != 1:
            raise SubmissionError(
                f"Multiclass label metric '{spec.metric}' needs exactly one target column "
                f"in the sample submission, but found {len(non_id)}.",
                "Use a label-format sample submission (one target column) or choose "
                "a probability metric such as log_loss/roc_auc.",
            )
    else:
        target_column = ctx.config.submit.target_col or str(spec.target)
    if id_column is None or id_column not in test.columns:
        raise SubmissionError(
            "Cannot determine the test id column for submission.",
            "Set submit.id_col or data.id in config.",
        )

    blend = read_json(ctx.run_dir / "03_blend" / "blend.json", default={}) or {}
    classes = blend.get("classes") if isinstance(blend, dict) else None
    probabilistic_multiclass = spec.task == "multiclass" and spec.metric in {"roc_auc", "log_loss"}
    output = _proba_or_labels(raw_pred, spec.task, spec.metric)
    if probabilistic_multiclass:
        if np.asarray(output).ndim != 2:
            raise SubmissionError(
                "Multiclass probability metric but blended predictions are 1-D.",
                "Rerun blend; log_loss/roc_auc need one probability column per class.",
            )
        if isinstance(classes, list) and len(classes) != output.shape[1]:
            raise SubmissionError(
                f"Blend classes ({len(classes)}) do not match prediction columns ({output.shape[1]}).",
                "Rerun blend; the class order is inconsistent.",
            )
    elif spec.task == "multiclass" and isinstance(classes, list) and len(classes) == raw_pred.shape[1]:
        output = np.asarray(classes)[np.asarray(output, dtype=int)]
    elif spec.task == "binary" and spec.metric in {"accuracy", "f1"} and isinstance(classes, list) and len(classes) == 2:
        output = np.asarray(classes)[np.asarray(output, dtype=int)]
    if ctx.config.submit.clip is not None:
        output = np.clip(output, *ctx.config.submit.clip)
    if ctx.config.submit.round is not None:
        output = np.round(output, ctx.config.submit.round)
    if np.issubdtype(np.asarray(output).dtype, np.number) and not np.isfinite(np.asarray(output)).all():
        raise SubmissionError("Submission contains NaN/inf.", "Check the training predictions.")

    if sample is not None:
        assert sample_id is not None
        sample_keys = sample[sample_id].astype(str)
        test_keys = test[id_column].astype(str)
        if sample_keys.duplicated().any():
            raise SubmissionError(
                "Sample submission contains duplicate ids.",
                "Fix the sample submission so each test id appears exactly once.",
            )
        if test_keys.duplicated().any():
            raise SubmissionError(
                "Test data contains duplicate ids.",
                "The id column must uniquely identify each test row.",
            )
        if set(sample_keys) != set(test_keys):
            missing = len(set(test_keys) - set(sample_keys))
            extra = len(set(sample_keys) - set(test_keys))
            raise SubmissionError(
                f"Sample/test id mismatch: {missing} test ids are absent from the sample, "
                f"{extra} sample ids are absent from the test data.",
                "Use the official sample submission for this competition/test file.",
            )

    if probabilistic_multiclass:
        matrix = np.asarray(output, dtype=float)
        if sample is not None:
            assert sample_id is not None
            non_id = [str(column) for column in sample.columns if str(column) != sample_id]
            if len(non_id) != matrix.shape[1]:
                raise SubmissionError(
                    f"Multiclass probability submission needs {matrix.shape[1]} prediction columns "
                    f"but the sample has {len(non_id)}.",
                    "Use a sample submission with one column per class "
                    "(or set submit.target_col only when the sample has one column per class).",
                )
            labels = [str(value) for value in classes] if isinstance(classes, list) else []
            if labels and sorted(non_id) == sorted(labels):
                order = [labels.index(column) for column in non_id]
                matrix = matrix[:, order]
            prediction = pd.DataFrame({"__kconv_key": test_keys.to_numpy()})
            for index, column in enumerate(non_id):
                prediction[column] = matrix[:, index]
            frame = (
                sample[[sample_id]]
                .assign(__kconv_key=sample_keys.to_numpy())
                .merge(prediction, on="__kconv_key", how="left", validate="one_to_one", sort=False)
                .drop(columns=["__kconv_key"])
            )
        else:
            labels = (
                [str(value) for value in classes]
                if isinstance(classes, list)
                else [str(index) for index in range(matrix.shape[1])]
            )
            frame = pd.DataFrame({id_column: test[id_column].to_numpy()})
            for index, label in enumerate(labels):
                frame[f"{target_column}_{label}"] = matrix[:, index]
    elif sample is not None:
        assert sample_id is not None
        non_id = [str(column) for column in sample.columns if str(column) != sample_id]
        if len(non_id) != 1:
            raise SubmissionError(
                f"Sample submission must contain exactly one target column, found {len(non_id)}.",
                "Use the official sample submission (id column plus one prediction column).",
            )
        target_column = non_id[0]
        prediction = pd.DataFrame(
            {"__kconv_key": test_keys.to_numpy(), target_column: np.asarray(output)}
        )
        frame = (
            sample[[sample_id]]
            .assign(__kconv_key=sample_keys.to_numpy())
            .merge(prediction, on="__kconv_key", how="left", validate="one_to_one", sort=False)
            .drop(columns=["__kconv_key"])
        )
        if frame[target_column].isna().any():
            raise SubmissionError(
                "Some sample-submission rows did not receive a prediction.",
                "Check that sample ids exactly match the test ids.",
            )
    else:
        frame = pd.DataFrame({id_column: test[id_column].to_numpy(), target_column: output})

    expected_rows = len(test)
    if len(frame) != expected_rows:
        raise SubmissionError(
            f"Submission has {len(frame)} rows, expected {expected_rows}.",
            "Check the sample submission id ordering.",
        )
    if frame.isna().any().any():
        raise SubmissionError("Submission contains missing values.", "Check id and prediction columns.")
    out_path = ctx.artifact(STAGE, "submission.csv")
    write_csv_atomic(frame, out_path, index=False)
    output_array = np.asarray(output)
    numeric_output = np.issubdtype(output_array.dtype, np.number)
    if numeric_output:
        min_value: float | str = float(np.min(output_array))
        max_value: float | str = float(np.max(output_array))
    else:
        values = output_array.tolist()
        min_value = str(min(values))
        max_value = str(max(values))
    check = {
        "rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "id_col": str(id_column),
        "target_col": str(target_column),
        "task": spec.task,
        "metric": spec.metric,
        "nan_count": int(frame.isna().sum().sum()),
        "min": min_value,
        "max": max_value,
    }
    check_path = ctx.artifact(STAGE, "submission_check.json")
    write_json(check_path, check)
    return {
        "status": "ok",
        "artifacts": [
            str(out_path.relative_to(ctx.run_dir)).replace("\\", "/"),
            str(check_path.relative_to(ctx.run_dir)).replace("\\", "/"),
        ],
        "rows": int(len(frame)),
        "columns": check["columns"],
    }
