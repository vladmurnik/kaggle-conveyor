"""Kaggle credential diagnostics: no network and no secret leakage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kconv.cli import app
from kconv.errors import KconvError
from kconv.kaggle import auth as kaggle_auth
from kconv.kaggle.auth import AuthStatus, check_auth, save_credentials


def test_environment_credentials_are_detected_without_exposing_secrets(tmp_path: Path) -> None:
    status = check_auth(
        environ={"KAGGLE_USERNAME": "alice", "KAGGLE_KEY": "super-secret"},
        config_dir=tmp_path,
        which=lambda _: "/fake/kaggle",
        version_runner=lambda _: "Kaggle API 1.6.0",
    )

    assert status.ready
    assert status.cli_available
    assert status.cli_version == "Kaggle API 1.6.0"
    assert status.credentials_present
    assert status.credential_source == "environment: KAGGLE_USERNAME/KAGGLE_KEY"
    assert "super-secret" not in json.dumps(status.to_dict())


def test_kaggle_json_credentials_are_detected(tmp_path: Path) -> None:
    credential_file = tmp_path / "kaggle.json"
    credential_file.write_text('{"username": "alice", "key": "file-secret"}', encoding="utf-8")
    if os.name != "nt":
        credential_file.chmod(0o600)

    status = check_auth(
        environ={},
        config_dir=tmp_path,
        which=lambda _: "/fake/kaggle",
        version_runner=lambda _: "Kaggle API 1.6.0",
    )

    assert status.credentials_present
    assert status.credential_source == f"file: {credential_file}"
    assert "file-secret" not in json.dumps(status.to_dict())
    assert not status.warnings


def test_missing_credentials_and_cli_are_reported(tmp_path: Path) -> None:
    status = check_auth(
        environ={},
        config_dir=tmp_path,
        token_path=tmp_path / "access_token",
        which=lambda _: None,
        version_runner=lambda _: None,
    )

    assert not status.ready
    assert not status.credentials_present
    assert not status.cli_available
    assert any("credentials not found" in warning for warning in status.warnings)
    assert any("CLI not found" in warning for warning in status.warnings)


def test_invalid_credential_file_is_reported_as_warning(tmp_path: Path) -> None:
    (tmp_path / "kaggle.json").write_text("not json", encoding="utf-8")
    status = check_auth(
        environ={},
        config_dir=tmp_path,
        token_path=tmp_path / "access_token",
        which=lambda _: "/fake/kaggle",
        version_runner=lambda _: "Kaggle API 1.6.0",
    )

    assert not status.credentials_present
    assert any("not valid JSON" in warning for warning in status.warnings)


def test_save_credentials_writes_private_kaggle_json(tmp_path: Path) -> None:
    path = save_credentials("alice", "super-secret", config_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / "kaggle.json"
    assert payload == {"username": "alice", "key": "super-secret"}
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600

    status = check_auth(
        environ={},
        config_dir=tmp_path,
        which=lambda _: "/fake/kaggle",
        version_runner=lambda _: "Kaggle API 2.2.5",
    )
    assert status.ready
    assert status.credential_source == f"file: {path}"
    assert "super-secret" not in json.dumps(status.to_dict())


def test_access_token_file_is_detected_and_saved(tmp_path: Path) -> None:
    token_path = tmp_path / "access_token"
    path = save_credentials(token="token-secret", token_path=token_path)

    assert path == token_path
    assert token_path.read_text(encoding="utf-8") == "token-secret\n"
    if os.name != "nt":
        assert token_path.stat().st_mode & 0o777 == 0o600

    status = check_auth(
        environ={},
        config_dir=tmp_path / "config",
        token_path=token_path,
        which=lambda _: "/fake/kaggle",
        version_runner=lambda _: "Kaggle CLI 2.2.5",
    )
    assert status.ready
    assert status.credential_source == f"file: {token_path} (access token)"
    assert "token-secret" not in json.dumps(status.to_dict())


def test_api_token_environment_is_detected(tmp_path: Path) -> None:
    status = check_auth(
        environ={"KAGGLE_API_TOKEN": "token-secret"},
        config_dir=tmp_path,
        token_path=tmp_path / "access_token",
        which=lambda _: "/fake/kaggle",
        version_runner=lambda _: "Kaggle CLI 2.2.5",
    )

    assert status.ready
    assert status.credential_source == "environment: KAGGLE_API_TOKEN"
    assert "token-secret" not in json.dumps(status.to_dict())


def test_save_credentials_rejects_empty_or_unsafe_values(tmp_path: Path) -> None:
    with pytest.raises(KconvError, match="username"):
        save_credentials("", "secret", config_dir=tmp_path)
    with pytest.raises(KconvError, match="API key"):
        save_credentials("alice", "has a space", config_dir=tmp_path)
    with pytest.raises(KconvError, match="username"):
        save_credentials("bad/name", "secret", config_dir=tmp_path)
    with pytest.raises(KconvError, match="token"):
        save_credentials(token="has a space", token_path=tmp_path / "access_token")
    assert not (tmp_path / "kaggle.json").exists()
    assert not (tmp_path / "access_token").exists()


def test_local_venv_kaggle_script_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = "kaggle.exe" if os.name == "nt" else "kaggle"
    script = tmp_path / executable
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr(kaggle_auth.sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setattr(kaggle_auth.shutil, "which", lambda _: None)

    status = check_auth(
        environ={},
        config_dir=tmp_path,
        version_runner=lambda _: "Kaggle CLI test",
    )

    assert status.cli_available
    assert status.cli_path == str(script)
    assert status.cli_version == "Kaggle CLI test"


def test_kaggle_auth_cli_json(monkeypatch) -> None:
    fake = AuthStatus(
        cli_available=True,
        cli_path="/fake/kaggle",
        cli_version="Kaggle API 1.6.0",
        python_package=True,
        credentials_present=True,
        credential_source="environment: KAGGLE_USERNAME/KAGGLE_KEY",
        config_path="/fake/.kaggle",
        warnings=(),
    )
    monkeypatch.setattr("kconv.cli.check_auth", lambda: fake)
    result = CliRunner().invoke(app, ["kaggle", "auth", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert payload["cli_available"] is True
    assert payload["credentials_present"] is True
    assert "super-secret" not in result.output
