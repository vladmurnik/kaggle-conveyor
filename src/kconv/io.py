"""Table IO helpers: read/write CSV and parquet with typed errors.

Format selection is by file extension only, so a typo never silently
produces an empty frame.  Every failure surfaces as a ``DataError`` with
an actionable hint.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from kconv.atomic import write_csv_atomic, write_parquet_atomic
from kconv.errors import DataError

_FORMAT_HINT = "Supported formats: .csv (also .csv.gz) and .parquet/.pq."
_NOT_FOUND_HINT = "Check data.train/data.test in config; relative paths resolve from the project folder."


def is_table(path: str | Path) -> bool:
    """True when the path looks like a supported CSV/parquet file."""
    path = Path(path)
    name = path.name.lower()
    return name.endswith(".csv.gz") or path.suffix.lower() in {".csv", ".parquet", ".pq"}


def _format_of(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv.gz") or path.suffix.lower() == ".csv":
        return "csv"
    if path.suffix.lower() in {".parquet", ".pq"}:
        return "parquet"
    raise DataError(f"Unsupported table format: {path}", _FORMAT_HINT)


def ensure_dirs(path: str | Path) -> Path:
    """Create the directory ``path`` refers to (parent for file paths).

    Returns the original path so it can be chained with writes.
    """
    path = Path(path)
    directory = path.parent if path.suffix else path
    directory.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a full CSV/parquet table into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"Data file not found: {path}", _NOT_FOUND_HINT)
    fmt = _format_of(path)
    try:
        if fmt == "csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed error with context
        raise DataError(
            f"Failed to read {path}: {exc}",
            f"{_FORMAT_HINT} Check that the file is not corrupted.",
        ) from exc


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame as CSV or parquet (by extension), creating parent dirs."""
    path = Path(path)
    fmt = _format_of(path)
    ensure_dirs(path)
    try:
        if fmt == "csv":
            write_csv_atomic(df, path, index=False)
        else:
            write_parquet_atomic(df, path)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed error with context
        raise DataError(
            f"Failed to write {path}: {exc}",
            "Check folder permissions and available disk space.",
        ) from exc
    return path


def safe_read_sample(path: str | Path, nrows: int = 100_000) -> pd.DataFrame:
    """Read at most ``nrows`` rows without loading a potentially huge file.

    CSVs use ``nrows`` directly; parquet reads row groups until the budget
    is reached.  Column names always reflect the full file.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"Data file not found: {path}", _NOT_FOUND_HINT)
    fmt = _format_of(path)
    if fmt == "csv":
        try:
            return pd.read_csv(path, nrows=nrows)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error with context
            raise DataError(
                f"Failed to read {path}: {exc}",
                f"{_FORMAT_HINT} Check that the file is not corrupted.",
            ) from exc
    try:
        import pyarrow.parquet as pq  # local import keeps CSV-only flows light
    except ImportError:  # pragma: no cover - pyarrow is a hard dependency
        frame = pd.read_parquet(path)
        return frame.head(nrows)
    try:
        parquet = pq.ParquetFile(path)
        frames: list[pd.DataFrame] = []
        rows = 0
        for batch in parquet.iter_batches(batch_size=min(nrows, 65_536)):
            frames.append(batch.to_pandas())
            rows += len(frames[-1])
            if rows >= nrows:
                break
        if not frames:
            return pd.DataFrame(columns=parquet.schema_arrow.names)
        return pd.concat(frames, ignore_index=True).head(nrows)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed error with context
        raise DataError(
            f"Failed to read {path}: {exc}",
            f"{_FORMAT_HINT} Check that the file is not corrupted.",
        ) from exc


def _csv_row_count(path: Path) -> int:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def safe_read_edges(path: str | Path, nrows: int = 100_000) -> pd.DataFrame:
    """Read a head+tail sample so late-row signal is not silently lost.

    Role detection used to inspect only the first 100k rows.  A column that is
    constant in the head but varies after row 100k was then dropped as
    "constant".  This helper reads roughly half the budget from each edge;
    parquet uses row groups and CSV uses a skip to the final rows.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"Data file not found: {path}", _NOT_FOUND_HINT)
    if nrows <= 0:
        return safe_read_sample(path, nrows=0)
    fmt = _format_of(path)
    head_budget = max(1, nrows // 2)
    head = safe_read_sample(path, nrows=head_budget)
    if fmt == "csv":
        try:
            total = _csv_row_count(path)
            if total <= len(head):
                return head
            tail_budget = max(1, nrows - len(head))
            tail = pd.read_csv(
                path,
                skiprows=range(1, max(1, total - tail_budget + 1)),
                nrows=tail_budget,
            )
            return pd.concat([head, tail], ignore_index=True)
        except Exception as exc:  # noqa: BLE001 - typed error with context
            raise DataError(
                f"Failed to read {path}: {exc}",
                f"{_FORMAT_HINT} Check that the file is not corrupted.",
            ) from exc
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        total = int(parquet.metadata.num_rows)
        if total <= len(head):
            return head
        tail_budget = max(1, nrows - len(head))
        frames: list[pd.DataFrame] = []
        rows = 0
        for index in range(parquet.num_row_groups - 1, -1, -1):
            frame = parquet.read_row_group(index).to_pandas()
            frames.append(frame)
            rows += len(frame)
            if rows >= tail_budget:
                break
        if not frames:
            return head
        tail = pd.concat(list(reversed(frames)), ignore_index=True).tail(tail_budget)
        return pd.concat([head, tail], ignore_index=True)
    except Exception as exc:  # noqa: BLE001 - typed error with context
        raise DataError(
            f"Failed to read {path}: {exc}",
            f"{_FORMAT_HINT} Check that the file is not corrupted.",
        ) from exc
