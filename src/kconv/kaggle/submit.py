"""Kaggle submission planning and execution with a safe dry-run default.

The command never uploads unless the caller explicitly passes ``execute=True``
(CLI: ``--execute``).  Planning validates the local CSV and prints the exact
Kaggle CLI argv; credentials and the network are touched only in execute mode.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from kconv.errors import KconvError
from kconv.kaggle.auth import AuthStatus, check_auth

_COMPETITION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MESSAGE_LIMIT = 500
Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class SubmissionPlan:
    """Validated local file plus the command that would be executed."""

    competition: str
    file: str
    message: str
    rows: int
    columns: tuple[str, ...]
    sha256: str
    command: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "competition": self.competition,
            "file": self.file,
            "message": self.message,
            "rows": self.rows,
            "columns": list(self.columns),
            "sha256": self.sha256,
            "command": list(self.command),
            "dry_run": True,
        }


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of a dry-run or a real Kaggle CLI invocation."""

    plan: SubmissionPlan
    executed: bool
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = self.plan.to_dict()
        payload.update(
            {
                "dry_run": not self.executed,
                "executed": self.executed,
                "returncode": self.returncode,
                "stdout": self.stdout,
                "stderr": self.stderr,
            }
        )
        return payload


def _validate_competition(competition: str) -> str:
    value = competition.strip()
    if not value or not _COMPETITION_RE.fullmatch(value):
        raise KconvError(
            f"Invalid Kaggle competition slug: {competition!r}.",
            "Use the slug from the competition URL, for example 'titanic'.",
        )
    return value


def _raw_header(path: Path) -> list[str]:
    """Read the literal header before pandas can rename duplicate/empty names."""
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), None)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise KconvError(
            f"Could not read submission CSV header: {path}",
            "Check the delimiter, quoting and gzip integrity.",
        ) from exc
    if header is None or not header:
        raise KconvError("Submission CSV is empty (no header row).")
    return [str(column) for column in header]


def validate_submission(path: str | Path, *, expected_columns: list[str] | tuple[str, ...] | None = None) -> tuple[int, tuple[str, ...]]:
    """Validate a CSV submission and return ``(rows, columns)``.

    Checks: readable CSV/CSV.GZ, non-empty, at least two columns, unique
    headers, no empty header, no all-NaN values, and optional exact column
    order when ``expected_columns`` is supplied.  The literal header is
    inspected before pandas so duplicate/empty names cannot be silently
    renamed by the parser.
    """
    candidate = Path(path)
    if not candidate.is_file():
        raise KconvError(
            f"Submission file not found: {candidate}",
            "Run the pipeline first or pass --file explicitly.",
        )
    suffix = candidate.name.lower()
    if not suffix.endswith((".csv", ".csv.gz")):
        raise KconvError(
            f"Unsupported submission format: {candidate.suffix or '<none>'}.",
            "Kaggle competitions accept .csv (and occasionally .csv.gz).",
        )
    raw_headers = _raw_header(candidate)
    if len(raw_headers) < 2:
        raise KconvError("Submission must have an id column and at least one prediction column.")
    if any(not header.strip() for header in raw_headers):
        raise KconvError("Submission contains an empty column name.")
    normalized = [header.strip() for header in raw_headers]
    if len(set(normalized)) != len(normalized):
        raise KconvError("Submission column names must be unique.")
    try:
        frame = pd.read_csv(candidate)
    except Exception as exc:  # noqa: BLE001 - pandas raises many parser exceptions
        raise KconvError(
            f"Could not read submission CSV: {candidate}",
            "Check the delimiter, quoting and gzip integrity.",
        ) from exc
    if frame.shape[1] < 2:
        raise KconvError("Submission must have an id column and at least one prediction column.")
    headers = [str(column).strip() for column in frame.columns]
    if headers != normalized:
        raise KconvError(
            "Submission header changed while parsing.",
            "Check the CSV for duplicate/empty names or unsupported quoting.",
        )
    if frame.empty:
        raise KconvError("Submission has zero data rows.")
    if frame.isna().all(axis=0).any():
        bad = [headers[index] for index, value in enumerate(frame.isna().all(axis=0)) if value]
        raise KconvError(f"Submission column(s) are entirely empty: {', '.join(bad)}.")
    if expected_columns is not None and tuple(headers) != tuple(str(column) for column in expected_columns):
        raise KconvError(
            "Submission columns do not match the expected order.",
            f"Expected {list(expected_columns)}, got {headers}.",
        )
    return int(len(frame)), tuple(headers)


def plan_submission(
    competition: str,
    submission: str | Path,
    *,
    message: str = "kconv submission",
    expected_columns: list[str] | tuple[str, ...] | None = None,
) -> SubmissionPlan:
    """Build and validate a submission plan without any network or secrets."""
    slug = _validate_competition(competition)
    text = message.strip()
    if not text:
        raise KconvError("Kaggle submission message must not be empty.")
    if len(text) > _MESSAGE_LIMIT:
        raise KconvError(f"Kaggle submission message exceeds {_MESSAGE_LIMIT} characters.")
    path = Path(submission).resolve()
    rows, columns = validate_submission(path, expected_columns=expected_columns)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    command = (
        "kaggle",
        "competitions",
        "submit",
        "-c",
        slug,
        "-f",
        str(path),
        "-m",
        text,
    )
    return SubmissionPlan(
        competition=slug,
        file=str(path),
        message=text,
        rows=rows,
        columns=columns,
        sha256=digest,
        command=command,
    )


def _auth_is_ready(auth: AuthStatus) -> bool:
    return auth.cli_available and auth.credentials_present


def submit_submission(
    competition: str,
    submission: str | Path,
    *,
    message: str = "kconv submission",
    execute: bool = False,
    expected_columns: list[str] | tuple[str, ...] | None = None,
    runner: Runner | None = None,
    auth: AuthStatus | None = None,
    timeout: float = 600.0,
) -> SubmitResult:
    """Plan a submission and, only when ``execute=True``, call the Kaggle CLI.

    A dry run never imports/executes the Kaggle CLI and never reads credentials.
    Execute mode requires a ready :class:`AuthStatus`; the default checker is
    used only in that branch.
    """
    plan = plan_submission(
        competition,
        submission,
        message=message,
        expected_columns=expected_columns,
    )
    if not execute:
        return SubmitResult(plan=plan, executed=False)
    status = auth if auth is not None else check_auth()
    if not _auth_is_ready(status):
        raise KconvError(
            "Kaggle credentials/CLI are not ready for an upload.",
            "Run 'kconv kaggle auth' and install/configure the Kaggle CLI; dry-run still works.",
        )
    command = list(plan.command)
    invoke: Runner = runner or subprocess.run
    try:
        completed = invoke(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise KconvError(
            "Kaggle CLI executable was not found.",
            "Run 'uv sync' or install it with 'pip install kaggle'.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise KconvError(
            f"Kaggle submission timed out after {timeout:g}s.",
            "Check the network and rerun; the local file was not modified.",
        ) from exc
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        raise KconvError(
            f"Kaggle CLI exited with code {completed.returncode}.",
            (stderr or stdout or "No output from the Kaggle CLI.")[:1000],
        )
    return SubmitResult(
        plan=plan,
        executed=True,
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
    )


__all__ = [
    "SubmissionPlan",
    "SubmitResult",
    "plan_submission",
    "submit_submission",
    "validate_submission",
]
