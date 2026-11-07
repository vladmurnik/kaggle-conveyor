"""Safe Kaggle submission planning, dry-run default and execute gate."""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from kconv.cli import app
from kconv.errors import KconvError
from kconv.kaggle.auth import AuthStatus
from kconv.kaggle.submit import plan_submission, submit_submission, validate_submission
from kconv.state import write_latest_pointer


def _csv(path: Path, *, rows: int = 3) -> Path:
    pd.DataFrame({"id": range(rows), "target": [0.1] * rows}).to_csv(path, index=False)
    return path


def _ready_auth() -> AuthStatus:
    return AuthStatus(
        cli_available=True,
        cli_path="kaggle",
        cli_version="Kaggle API 1.6.0",
        python_package=True,
        credentials_present=True,
        credential_source="environment: KAGGLE_USERNAME/KAGGLE_KEY",
        config_path=None,
    )


def _not_ready_auth() -> AuthStatus:
    return AuthStatus(
        cli_available=True,
        cli_path="kaggle",
        cli_version="Kaggle API 1.6.0",
        python_package=True,
        credentials_present=False,
        credential_source=None,
        config_path=None,
    )


def test_plan_validates_csv_and_builds_exact_command(tmp_path: Path) -> None:
    path = _csv(tmp_path / "submission.csv", rows=4)
    plan = plan_submission("titanic", path, message="run 42")
    assert plan.rows == 4
    assert plan.columns == ("id", "target")
    assert plan.command == (
        "kaggle",
        "competitions",
        "submit",
        "-c",
        "titanic",
        "-f",
        str(path.resolve()),
        "-m",
        "run 42",
    )
    assert len(plan.sha256) == 64


def test_plan_rejects_bad_names_messages_and_columns(tmp_path: Path) -> None:
    path = _csv(tmp_path / "submission.csv")
    with pytest.raises(KconvError, match="Invalid Kaggle competition slug"):
        plan_submission("bad slug", path)
    with pytest.raises(KconvError, match="message"):
        plan_submission("titanic", path, message="   ")
    with pytest.raises(KconvError, match="columns do not match"):
        plan_submission("titanic", path, expected_columns=["id", "prediction"])
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"id": [1, 2], "target": [float("nan"), float("nan")]}).to_csv(bad, index=False)
    with pytest.raises(KconvError, match="entirely empty"):
        validate_submission(bad)
    with pytest.raises(KconvError, match="not found"):
        validate_submission(tmp_path / "missing.csv")
    text = tmp_path / "submission.txt"
    text.write_text("id,target\n1,0\n", encoding="utf-8")
    with pytest.raises(KconvError, match="Unsupported submission format"):
        validate_submission(text)


def test_plan_rejects_literal_duplicate_empty_and_gzipped_headers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("id,id,target\n1,2,0.5\n", encoding="utf-8")
    with pytest.raises(KconvError, match="unique"):
        validate_submission(duplicate)

    empty = tmp_path / "empty.csv"
    empty.write_text(",target\n1,0.5\n", encoding="utf-8")
    with pytest.raises(KconvError, match="empty column name"):
        validate_submission(empty)

    whitespace = tmp_path / "whitespace.csv"
    whitespace.write_text("id,   \n1,0.5\n", encoding="utf-8")
    with pytest.raises(KconvError, match="empty column name"):
        validate_submission(whitespace)

    gz = tmp_path / "duplicate.csv.gz"
    with gzip.open(gz, "wt", encoding="utf-8", newline="") as handle:
        handle.write("id,id,target\n1,2,0.5\n")
    with pytest.raises(KconvError, match="unique"):
        validate_submission(gz)


def test_valid_gzipped_submission_is_accepted(tmp_path: Path) -> None:
    gz = tmp_path / "submission.csv.gz"
    with gzip.open(gz, "wt", encoding="utf-8", newline="") as handle:
        handle.write("id,target\n1,0.25\n2,0.75\n")
    rows, columns = validate_submission(gz)
    assert rows == 2
    assert columns == ("id", "target")


def test_dry_run_never_calls_runner_or_auth(tmp_path: Path) -> None:
    path = _csv(tmp_path / "submission.csv")
    calls: list[list[str]] = []

    def runner(command, **kwargs):  # noqa: ANN001, ANN003
        calls.append(command)
        raise AssertionError("dry-run must not invoke subprocess")

    result = submit_submission("titanic", path, runner=runner)
    assert not result.executed
    assert result.returncode is None
    assert calls == []
    assert result.to_dict()["dry_run"] is True


def test_execute_requires_ready_auth(tmp_path: Path) -> None:
    path = _csv(tmp_path / "submission.csv")
    with pytest.raises(KconvError, match="credentials"):
        submit_submission("titanic", path, execute=True, auth=_not_ready_auth())


def test_execute_calls_kaggle_and_returns_output(tmp_path: Path) -> None:
    path = _csv(tmp_path / "submission.csv")
    captured: list[list[str]] = []

    def runner(command, **kwargs):  # noqa: ANN001, ANN003
        captured.append(command)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(command, 0, stdout="Success", stderr="")

    result = submit_submission(
        "titanic",
        path,
        execute=True,
        auth=_ready_auth(),
        runner=runner,
        message="hello",
    )
    assert result.executed
    assert result.returncode == 0
    assert result.stdout == "Success"
    assert captured[0][:5] == ["kaggle", "competitions", "submit", "-c", "titanic"]


def test_execute_failure_is_reported_without_leaking_secrets(tmp_path: Path) -> None:
    path = _csv(tmp_path / "submission.csv")

    def runner(command, **kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="not authorized")

    with pytest.raises(KconvError) as exc:
        submit_submission("titanic", path, execute=True, auth=_ready_auth(), runner=runner)
    assert "exited with code 1" in exc.value.message
    assert "not authorized" in (exc.value.hint or "")


def test_cli_dry_run_uses_latest_submission(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "competitions" / "demo"
    run_dir = project / "runs" / "run-1" / "04_submit"
    run_dir.mkdir(parents=True)
    _csv(run_dir / "submission.csv")
    write_latest_pointer(project, "run-1")
    result = CliRunner().invoke(
        app,
        ["kaggle", "submit", "--comp", "demo", "--competition", "titanic", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["competition"] == "titanic"
    assert payload["rows"] == 3
    assert payload["executed"] is False


def test_cli_execute_fails_cleanly_without_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kconv.kaggle.submit.check_auth", _not_ready_auth)
    path = _csv(tmp_path / "submission.csv")
    result = CliRunner().invoke(
        app,
        ["kaggle", "submit", "--competition", "titanic", "--file", str(path), "--execute"],
    )
    assert result.exit_code != 0
    assert "not ready" in result.output.lower() or "credentials" in result.output.lower()
