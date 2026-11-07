"""Dataset auto-detection: point at a folder/file/zip, get a DatasetSpec.

Supported inputs:

1. a directory containing ``train``/``test``/``sample_submission`` files
   (``train.csv|parquet``, ``test.*``, ``sample_submission.*``,
   ``gender_submission.*``), searched recursively;
2. a single CSV/parquet used as train (test/sample discovered as siblings);
3. a zip archive with the same files (extracted to a temp cache).

Explicit ``target``/``test``/``sample_submission``/``id``/``task``/``metric``
overrides always win over inference.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from kconv.contracts import DatasetSpec, TaskType
from kconv.errors import DataError
from kconv.io import is_table, safe_read_edges, safe_read_sample
from kconv.metrics import metric_names, metric_supports_task, metric_tasks

_TRAIN_NAMES = ("train.csv", "train.parquet", "train.pq")
_TEST_NAMES = ("test.csv", "test.parquet", "test.pq")
_SAMPLE_SUBMISSION_NAMES = ("sample_submission.csv", "sample_submission.parquet", "sample_submission.pq")
_GENDER_SUBMISSION_NAMES = ("gender_submission.csv", "gender_submission.parquet", "gender_submission.pq")

# Never harvest artifacts or VCS/venv internals as "the dataset".
_SKIP_DIR_NAMES = {".git", ".venv", "venv", "__pycache__", "node_modules", "runs"}
_ARCHIVE_MEMBER_SUFFIXES = {".csv", ".parquet", ".pq"}

_DEFAULT_METRICS: dict[str, str] = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
_TASKS: tuple[str, ...] = ("binary", "multiclass", "regression")
_MAX_CLASS_CARDINALITY = 20
_MAX_TEXT_TARGET_CLASSES = 100

_TARGET_EXACT_NAMES = frozenset(
    {
        "y", "class", "classes", "outcome", "response", "survived", "saleprice", "sale_price",
        "price", "amount", "churn", "readmitted", "default", "defaulted", "clicked",
        "is_fraud", "isfraud", "is_churn", "target_value",
    }
)
_ID_EXACT_NAMES = frozenset({"id", "uuid", "uid", "rowid", "index"})


def _hint(columns: list[str] | None = None) -> str:
    concrete = f" (e.g. --target {columns[-1]})" if columns else ""
    return (
        f"Pass --target <column>{concrete} and --task binary|multiclass|regression; "
        "add --test <path> if the test file is named differently."
    )


# ---------------------------------------------------------------------------
# file discovery
# ---------------------------------------------------------------------------


def _table_files(root: Path) -> list[Path]:
    files: list[Path] = []
    try:
        for candidate in root.rglob("*"):
            if not candidate.is_file() or not is_table(candidate):
                continue
            try:
                rel_parts = candidate.relative_to(root).parts[:-1]
            except ValueError:  # pragma: no cover - defensive
                rel_parts = ()
            if any(part.lower() in _SKIP_DIR_NAMES for part in rel_parts):
                continue
            files.append(candidate)
    except OSError:  # unreadable subtree: best effort
        pass
    return files


def _pick_named(files: list[Path], root: Path, preferred: tuple[str, ...]) -> Path | None:
    """Exact filename matches first, then shallowest path wins."""
    matches = [f for f in files if f.name.lower() in preferred]
    if not matches:
        return None

    def key(candidate: Path) -> tuple[int, int, str]:
        rel = candidate.relative_to(root)
        return (preferred.index(candidate.name.lower()), len(rel.parts), str(rel))

    return min(matches, key=key)


def _shallowest(files: list[Path], root: Path) -> Path:
    return min(files, key=lambda f: (len(f.relative_to(root).parts), str(f.relative_to(root))))


def _locate_train(root: Path, warnings: list[str]) -> Path:
    files = _table_files(root)
    picked = _pick_named(files, root, _TRAIN_NAMES)
    if picked is not None:
        return picked
    if len(files) == 1:
        warnings.append(f"No train.* file found; using the only table file: {files[0].name}.")
        return files[0]
    if not files:
        raise DataError(
            f"No CSV/parquet files found under {root}.",
            "Expected train.csv/train.parquet there; point data.train at the file, "
            + _hint(),
        )
    found = ", ".join(sorted(f.name for f in files)[:5])
    raise DataError(
        f"Cannot identify the train file under {root} (found: {found}).",
        "Name it train.csv/train.parquet or set data.train to the exact file; " + _hint(),
    )


def _discover_test(root: Path, exclude: set[Path]) -> Path | None:
    files = [f for f in _table_files(root) if f not in exclude]
    return _pick_named(files, root, _TEST_NAMES)


def _discover_sample_submission(root: Path, exclude: set[Path], warnings: list[str]) -> Path | None:
    files = [f for f in _table_files(root) if f not in exclude]
    picked = _pick_named(files, root, _SAMPLE_SUBMISSION_NAMES)
    if picked is not None:
        return picked
    by_name = [f for f in files if "sample_submission" in f.name.lower()]
    if by_name:
        return _shallowest(by_name, root)
    picked = _pick_named(files, root, _GENDER_SUBMISSION_NAMES)
    if picked is not None:
        return picked
    by_name = [f for f in files if "submission" in f.name.lower()]
    if by_name:
        warnings.append(f"Using {by_name[0].name} as the sample submission.")
        return _shallowest(by_name, root)
    return None


def _extract_zip(zip_path: Path, warnings: list[str]) -> Path:
    """Safely extract CSV/parquet members to a temp cache keyed by file identity."""
    try:
        stat = zip_path.stat()
        identity = f"{zip_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError as exc:
        raise DataError(
            f"Cannot read zip archive {zip_path}: {exc}",
            "Re-download the archive or pass a folder with the extracted files.",
        ) from exc
    cache_root = Path(tempfile.gettempdir()) / "kconv_autodetect"
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    dest = cache_root / key
    if dest.is_dir() and any(dest.iterdir()):
        warnings.append(f"Reusing extracted archive cache: {dest}")
        return dest

    cache_root.mkdir(parents=True, exist_ok=True)
    part = cache_root / f"{key}.part-{os.getpid()}"
    if part.exists():
        shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    extracted = 0
    try:
        with zipfile.ZipFile(zip_path) as archive:
            base = part.resolve()
            for info in archive.infolist():
                name = info.filename
                if info.is_dir() or not name:
                    continue
                parts = Path(name).parts
                if any(p.startswith(".") or p == "__MACOSX" for p in parts):
                    continue
                if Path(name).suffix.lower() not in _ARCHIVE_MEMBER_SUFFIXES:
                    continue
                target = (part / name).resolve()
                if not str(target).startswith(str(base) + os.sep):  # zip-slip guard
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                extracted += 1
        if extracted == 0:
            raise DataError(
                f"No CSV/parquet files inside {zip_path.name}.",
                "Archive must contain train/test CSV or parquet files; " + _hint(),
            )
        if not dest.exists():
            try:
                os.replace(part, dest)
            except OSError:  # another process won the race
                if not dest.exists():
                    raise
        else:  # pragma: no cover - race fallback
            shutil.rmtree(part, ignore_errors=True)
    finally:
        if part.exists():
            shutil.rmtree(part, ignore_errors=True)
    warnings.append(f"Extracted {zip_path.name} to {dest}")
    return dest


# ---------------------------------------------------------------------------
# name heuristics
# ---------------------------------------------------------------------------


def _is_target_name(name: str) -> bool:
    lowered = name.strip().lower()
    if lowered in _TARGET_EXACT_NAMES:
        return True
    return "target" in lowered or "label" in lowered


def _is_id_name(name: str) -> bool:
    lowered = name.strip().lower()
    if lowered in _ID_EXACT_NAMES:
        return True
    if lowered.endswith("_id"):
        return True
    # "PassengerId"/"CustomerId" but not short words like "paid".
    return lowered.endswith("id") and len(lowered) >= 7 and lowered[:-2].isalpha()


def _sample_submission_id_column(ss_df: pd.DataFrame, id_override: str | None) -> str:
    if id_override and id_override in ss_df.columns:
        return str(id_override)
    for column in ss_df.columns:
        if _is_id_name(str(column)):
            return str(column)
    return str(ss_df.columns[0])


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def _infer_target(
    train_df: pd.DataFrame,
    ss_df: pd.DataFrame | None,
    id_override: str | None,
    warnings: list[str],
) -> str:
    columns = [str(c) for c in train_df.columns]
    if ss_df is not None and ss_df.shape[1] >= 1:
        ss_id = _sample_submission_id_column(ss_df, id_override)
        candidates = [str(c) for c in ss_df.columns if str(c) != ss_id and str(c) in columns]
        if candidates:
            if len(candidates) > 1:
                warnings.append(
                    f"Sample submission has several non-id columns {candidates}; using '{candidates[0]}'."
                )
            return candidates[0]
        if ss_df.shape[1] > 1:
            warnings.append("Sample submission target not found in train; trying column-name heuristics.")
    matches = [c for c in columns if _is_target_name(c)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise DataError(
            f"Ambiguous target column in train: {matches}.",
            f"Pass --target <column> to pick one (e.g. --target {matches[0]}); "
            "add --task binary|multiclass|regression if needed.",
        )
    if len(columns) >= 2:
        return columns[-1]
    raise DataError(
        f"Cannot infer target: train has a single column {columns}.",
        _hint(columns),
    )


def _infer_id(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    ss_df: pd.DataFrame | None,
    target_col: str,
    warnings: list[str],
) -> str | None:
    columns = [str(c) for c in train_df.columns]
    if ss_df is not None and ss_df.shape[1] >= 1:
        first = str(ss_df.columns[0])
        if first in columns and first != target_col:
            return first
    name_rejected = False
    for col in train_df.columns:
        name = str(col)
        if name == target_col or not _is_id_name(name):
            continue
        series = train_df[col]
        if bool(series.is_unique) and bool(series.notna().all()):
            return name
        name_rejected = True
    if name_rejected:
        warnings.append("An id-like column has duplicate/missing values; id column left unset.")
    if test_df is not None:
        test_cols = {str(c) for c in test_df.columns}
        for col in train_df.columns:
            name = str(col)
            if name == target_col or name not in test_cols:
                continue
            series = train_df[col]
            if bool(series.is_unique) and bool(series.notna().all()):
                return name
    return None


def _all_integral(values: pd.Series) -> bool:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(array).all():
        return False
    return bool((array == np.floor(array)).all())


def _infer_task(train_df: pd.DataFrame, target_col: str) -> TaskType:
    values = train_df[target_col].dropna()
    if values.empty:
        raise DataError(
            f"Target column '{target_col}' contains only missing values.",
            _hint([str(c) for c in train_df.columns]),
        )
    nunique = int(values.nunique())
    if nunique < 2:
        raise DataError(
            f"Target column '{target_col}' has a single unique value ({values.iloc[0]!r}); cannot infer task.",
            "Pass --target <column> and --task binary|multiclass|regression to force it.",
        )
    if nunique == 2:
        return "binary"
    kind = values.dtype.kind
    if kind in {"M", "m"}:
        raise DataError(
            f"Target column '{target_col}' is a datetime; expected numeric or categorical values.",
            _hint([str(c) for c in train_df.columns]),
        )
    if kind in {"i", "u"} or (kind == "f" and _all_integral(values)):
        return "multiclass" if nunique <= _MAX_CLASS_CARDINALITY else "regression"
    if kind == "f":
        return "regression"
    if nunique > _MAX_TEXT_TARGET_CLASSES:
        raise DataError(
            f"Target column '{target_col}' has {nunique} distinct text values; cannot infer task.",
            "Pass --task regression (or binary|multiclass) and --target <column> if the target is wrong.",
        )
    return "multiclass"


def _looks_datetime(name: str, values: pd.Series) -> bool:
    lowered = name.lower()
    name_hit = any(token in lowered for token in ("date", "time", "timestamp"))
    sample = values.astype(str).head(500)
    if sample.empty:
        return False
    has_separator = bool(sample.str.contains(r"[-/:]", regex=True).mean() >= 0.8)
    if not name_hit and not has_separator:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (ValueError, TypeError):  # pragma: no cover - pandas version quirks
        return False
    fraction = float(parsed.notna().mean())
    return fraction >= (0.8 if name_hit else 0.95)


def _column_role(name: str, series: pd.Series) -> str:
    values = series.dropna()
    if values.empty or int(values.nunique()) <= 1:
        return "ignore"
    kind = series.dtype.kind
    if kind == "M":
        return "datetime"
    if kind == "b":
        return "categorical"
    if kind in {"i", "u", "f"}:
        return "numeric"
    if _looks_datetime(name, values):
        return "datetime"
    text = values.astype(str)
    mean_len = float(text.str.len().mean())
    unique_ratio = float(values.nunique() / max(len(values), 1))
    if mean_len >= 20.0 or (mean_len >= 10.0 and unique_ratio >= 0.8):
        return "text"
    return "categorical"


def _assign_roles(
    train_df: pd.DataFrame, target_col: str, id_col: str | None, warnings: list[str]
) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {
        "feature_columns": [],
        "numeric_columns": [],
        "categorical_columns": [],
        "datetime_columns": [],
        "text_columns": [],
        "ignore_columns": [],
    }
    buckets = {
        "numeric": "numeric_columns",
        "categorical": "categorical_columns",
        "datetime": "datetime_columns",
        "text": "text_columns",
        "ignore": "ignore_columns",
    }
    for column in train_df.columns:
        name = str(column)
        if name == target_col or name == id_col:
            continue
        role = _column_role(name, train_df[column])
        if role == "ignore":
            roles["ignore_columns"].append(name)
            warnings.append(f"Column '{name}' is constant/empty and will be ignored.")
            continue
        roles["feature_columns"].append(name)
        roles[buckets[role]].append(name)
    return roles


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def detect_dataset(
    path: str | Path,
    *,
    target: str | None = None,
    test: str | Path | None = None,
    sample_submission: str | Path | None = None,
    id: str | None = None,
    task: str | None = None,
    metric: str | None = None,
    sample_rows: int = 100_000,
) -> DatasetSpec:
    """Detect a dataset under ``path`` and return a normalized ``DatasetSpec``.

    ``path`` may be a directory, a single CSV/parquet (used as train) or a
    zip archive.  Every explicit keyword override wins over inference.
    """
    warnings: list[str] = []
    path = Path(path)
    if not path.exists():
        raise DataError(f"Data path not found: {path}.", "Check data.train in config; " + _hint())

    if path.is_dir():
        root = path
        train_path = _locate_train(root, warnings)
    elif path.suffix.lower() == ".zip":
        root = _extract_zip(path, warnings)
        train_path = _locate_train(root, warnings)
    elif is_table(path):
        root = path.parent
        train_path = path
    else:
        raise DataError(
            f"Unsupported data path: {path}.",
            "Point at a folder with train/test files, a .csv/.parquet train file, or a .zip archive.",
        )

    exclude = {train_path}
    if test is not None:
        test_path = Path(test)
        if not test_path.exists():
            raise DataError(f"Test file not found: {test_path}", "Pass --test <path> pointing at an existing file.")
    else:
        test_path = _discover_test(root, exclude)
        if test_path is None:
            warnings.append("No test file detected; pass --test <path> if one exists.")

    if sample_submission is not None:
        ss_path = Path(sample_submission)
        if not ss_path.exists():
            raise DataError(
                f"Sample submission not found: {ss_path}",
                "Set data.sample_submission to an existing file.",
            )
    else:
        ss_path = _discover_sample_submission(root, exclude, warnings)
        if ss_path is None:
            warnings.append("No sample submission detected; set data.sample_submission if one exists.")

    train_df = safe_read_edges(train_path, nrows=sample_rows)
    if train_df.empty:
        raise DataError(f"Train file {train_path} has no rows.", _hint([str(c) for c in train_df.columns]))
    train_df.columns = [str(c) for c in train_df.columns]
    if train_df.shape[1] < 2:
        raise DataError(
            f"Train file {train_path.name} has a single column {list(train_df.columns)}; no features to model.",
            _hint(list(train_df.columns)),
        )

    ss_df = None
    if ss_path is not None:
        ss_df = safe_read_sample(ss_path, nrows=min(sample_rows, 10_000))
        ss_df.columns = [str(c) for c in ss_df.columns]
        if ss_df.shape[1] == 0:
            ss_df = None

    test_df = None
    if test_path is not None:
        test_df = safe_read_sample(test_path, nrows=sample_rows)
        test_df.columns = [str(c) for c in test_df.columns]

    if target is not None:
        if target not in train_df.columns:
            raise DataError(
                f"Target column '{target}' not found in {train_path.name} "
                f"(columns: {list(train_df.columns)}).",
                "Pass --target <column> with an exact train column name.",
            )
        target_col = str(target)
    else:
        target_col = _infer_target(train_df, ss_df, target, warnings)

    if id is not None:
        if id not in train_df.columns:
            raise DataError(
                f"Id column '{id}' not found in {train_path.name} "
                f"(columns: {list(train_df.columns)}).",
                "Pass --id <column> with an exact train column name.",
            )
        id_col = str(id)
    else:
        id_col = _infer_id(train_df, test_df, ss_df, target_col, warnings)

    if task is None or task == "auto":
        task_val = _infer_task(train_df, target_col)
    elif task not in _TASKS:
        raise DataError(f"Unknown task '{task}'.", "Pass --task binary|multiclass|regression.")
    else:
        task_val = task  # type: ignore[assignment]

    if metric is None or metric == "auto":
        metric_val = _DEFAULT_METRICS[task_val]
    else:
        metric_val = str(metric)
    if metric_val not in metric_names():
        raise DataError(
            f"Unknown metric '{metric_val}'.",
            f"Use one of: {', '.join(metric_names())}.",
        )
    if not metric_supports_task(metric_val, task_val):
        raise DataError(
            f"Metric '{metric_val}' is not compatible with task '{task_val}'.",
            f"Use one of: {', '.join(sorted(metric_tasks(metric_val)))}.",
        )

    roles = _assign_roles(train_df, target_col, id_col, warnings)
    if not roles["feature_columns"]:
        raise DataError(
            f"No usable feature columns in {train_path.name} after excluding target/id/constant columns.",
            "Check data.ignore and pass --target <column> pointing at the answer column.",
        )

    return DatasetSpec(
        train=train_path,
        test=test_path,
        sample_submission=ss_path,
        target=target_col,
        id_col=id_col,
        task=task_val,
        metric=metric_val,
        warnings=warnings,
        **roles,
    )
