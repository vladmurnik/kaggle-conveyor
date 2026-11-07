"""Kaggle credential diagnostics and safe local credential writes.

The checker never prints or returns secret values. It recognises the credential
locations used by the Kaggle CLI:

* environment variables ``KAGGLE_USERNAME``/``KAGGLE_KEY`` and
  ``KAGGLE_API_TOKEN``;
* legacy ``$KAGGLE_CONFIG_DIR/kaggle.json`` (normally
  ``~/.kaggle/kaggle.json``);
* the Kaggle CLI 2.x access-token file ``~/.kaggle/access_token`` (and
  ``access_token.txt`` on Windows);
* the OAuth credential cache ``~/.kaggle/credentials.json``.

``save_credentials`` writes a legacy JSON file or an access-token file
atomically for the local web UI; it never echoes the secret back to a caller.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from kconv.errors import KconvError


@dataclass(frozen=True)
class AuthStatus:
    """Safe-to-display summary of the local Kaggle setup."""

    cli_available: bool
    cli_path: str | None
    cli_version: str | None
    python_package: bool
    credentials_present: bool
    credential_source: str | None
    config_path: str | None
    warnings: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.cli_available and self.credentials_present

    def to_dict(self) -> dict:
        return {
            "cli_available": self.cli_available,
            "cli_path": self.cli_path,
            "cli_version": self.cli_version,
            "python_package": self.python_package,
            "credentials_present": self.credentials_present,
            "credential_source": self.credential_source,
            "config_path": self.config_path,
            "ready": self.ready,
            "warnings": list(self.warnings),
        }


def _run_version(path: str) -> str | None:
    """Ask the CLI for its version without failing the whole diagnostic."""
    try:
        completed = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output[:200] if completed.returncode == 0 and output else None


def _python_package_available() -> bool:
    try:
        return importlib.util.find_spec("kaggle") is not None
    except (ImportError, ValueError):
        return False


def _local_kaggle_cli() -> str | None:
    """Find the console script installed next to the running interpreter.

    ``uv run`` puts the virtualenv's scripts directory on ``PATH``, but a user
    can also start the web UI with ``.venv\\Scripts\\python -m kconv.cli web``.
    In that case ``shutil.which`` may miss the sibling ``kaggle`` executable.
    """
    executable = "kaggle.exe" if os.name == "nt" else "kaggle"
    candidate = Path(sys.executable).with_name(executable)
    if candidate.is_file():
        return str(candidate)
    return None


def _read_env_credentials(environ: Mapping[str, str]) -> str | None:
    username = environ.get("KAGGLE_USERNAME", "").strip()
    key = environ.get("KAGGLE_KEY", "").strip()
    if username and key:
        return "environment: KAGGLE_USERNAME/KAGGLE_KEY"
    if environ.get("KAGGLE_API_TOKEN", "").strip():
        return "environment: KAGGLE_API_TOKEN"
    return None


def _read_file_credentials(path: Path, warnings: list[str]) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        warnings.append(f"Kaggle credential file is not valid JSON: {path}")
        return None
    if not isinstance(payload, dict):
        warnings.append(f"Kaggle credential file must contain an object: {path}")
        return None
    username = str(payload.get("username", "")).strip()
    key = str(payload.get("key", "")).strip()
    if not username or not key:
        warnings.append(f"Kaggle credential file is missing username/key: {path}")
        return None
    if os.name != "nt":
        try:
            if path.stat().st_mode & 0o077:
                warnings.append(
                    f"Kaggle credential file is readable by group/others; run chmod 600 {path}"
                )
        except OSError:
            pass
    return f"file: {path}"


def _read_access_token_file(path: Path, warnings: list[str]) -> str | None:
    candidates = [path]
    if path.name == "access_token":
        candidates.append(path.with_name("access_token.txt"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            token = candidate.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            warnings.append(f"Kaggle access token file is not readable: {candidate}")
            continue
        if token:
            return f"file: {candidate} (access token)"
        warnings.append(f"Kaggle access token file is empty: {candidate}")
    return None


def _read_oauth_credentials(path: Path, warnings: list[str]) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        warnings.append(f"Kaggle OAuth credential file is not valid JSON: {path}")
        return None
    if isinstance(payload, dict) and str(payload.get("refresh_token", "")).strip():
        return f"file: {path} (oauth)"
    warnings.append(f"Kaggle OAuth credential file is missing refresh_token: {path}")
    return None


def _config_dir(explicit: str | Path | None, environ: Mapping[str, str]) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    env_dir = environ.get("KAGGLE_CONFIG_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / ".kaggle"


def _token_path(explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    return Path.home() / ".kaggle" / "access_token"


def _oauth_path() -> Path:
    return Path.home() / ".kaggle" / "credentials.json"


def _source_path(source: str | None) -> str | None:
    if not source or not source.startswith("file: "):
        return None
    value = source[len("file: ") :]
    for suffix in (" (access token)", " (oauth)"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _validated_credentials(username: str, key: str) -> tuple[str, str]:
    user = str(username or "").strip()
    secret = str(key or "").strip()
    if not user or not secret:
        raise KconvError(
            "Укажите Kaggle username и API key.",
            "Оба поля обязательны; key не сохраняется в браузере и не выводится в ответах.",
        )
    if not _USERNAME_RE.fullmatch(user):
        raise KconvError(
            "Некорректный Kaggle username.",
            "Используйте имя из kaggle.json: латиница, цифры, точка, дефис или подчёркивание.",
        )
    if len(secret) > 512 or any(character.isspace() or ord(character) < 32 for character in secret):
        raise KconvError(
            "Некорректный Kaggle API key.",
            "Ключ не должен содержать пробелы и управляющие символы.",
        )
    return user, secret


def _validated_token(token: str) -> str:
    secret = str(token or "").strip()
    if not secret:
        raise KconvError(
            "Укажите Kaggle API token.",
            "Токен создаётся в Kaggle Settings → API → Generate New Token.",
        )
    if len(secret) > 4096 or any(character.isspace() or ord(character) < 32 for character in secret):
        raise KconvError(
            "Некорректный Kaggle API token.",
            "Токен не должен содержать пробелы и управляющие символы.",
        )
    return secret


def _write_private_text(path: Path, text: str) -> None:
    """Atomically write a secret file with owner-only permissions."""
    directory = path.parent
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            directory.chmod(0o700)
        except OSError:
            pass
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".kaggle-",
            suffix=".tmp",
            dir=directory,
        )
        temporary = Path(temporary_name)
        handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        descriptor = None
        with handle:
            handle.write(text)
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    except OSError as exc:
        raise KconvError(
            f"Не удалось сохранить Kaggle credentials: {exc}",
            f"Проверьте права на папку {directory}.",
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def save_credentials(
    username: str = "",
    key: str = "",
    *,
    token: str = "",
    config_dir: str | Path | None = None,
    token_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Write Kaggle credentials atomically and return the written path.

    ``token`` writes the Kaggle CLI 2.x access-token file; otherwise
    ``username``/``key`` write the legacy ``kaggle.json``.  Files use
    owner-only permissions where the platform supports them.  Secrets are
    never returned to callers or included in diagnostics.
    """
    if str(token or "").strip():
        secret = _validated_token(token)
        path = _token_path(token_path)
        _write_private_text(path, secret + "\n")
        return path

    user, secret = _validated_credentials(username, key)
    env = os.environ if environ is None else environ
    directory = _config_dir(config_dir, env)
    path = directory / "kaggle.json"
    payload = json.dumps({"username": user, "key": secret}, ensure_ascii=True) + "\n"
    _write_private_text(path, payload)
    return path


def check_auth(
    *,
    environ: Mapping[str, str] | None = None,
    config_dir: str | Path | None = None,
    token_path: str | Path | None = None,
    which: Callable[[str], str | None] | None = None,
    version_runner: Callable[[str], str | None] | None = None,
) -> AuthStatus:
    """Inspect the local Kaggle CLI and credential setup.

    ``which`` and ``version_runner`` are injectable for deterministic tests.
    The function only reads metadata; it never validates credentials online.
    """
    env = os.environ if environ is None else environ
    finder = which or shutil.which
    cli_path = finder("kaggle")
    if cli_path is None and which is None:
        cli_path = _local_kaggle_cli()
    cli_version = None
    if cli_path:
        cli_version = (version_runner or _run_version)(cli_path)

    directory = _config_dir(config_dir, env)
    file_path = directory / "kaggle.json"
    warnings: list[str] = []
    source = _read_env_credentials(env)
    if source is None:
        source = _read_file_credentials(file_path, warnings)
    if source is None:
        source = _read_access_token_file(_token_path(token_path), warnings)
    if source is None:
        source = _read_oauth_credentials(_oauth_path(), warnings)
    if source is None and not warnings:
        warnings.append(
            "Kaggle credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY, "
            "KAGGLE_API_TOKEN, or create ~/.kaggle/kaggle.json or "
            "~/.kaggle/access_token."
        )
    if cli_path is None:
        warnings.append("Kaggle CLI not found; run 'uv sync' or install it with pip.")

    return AuthStatus(
        cli_available=cli_path is not None,
        cli_path=cli_path,
        cli_version=cli_version,
        python_package=_python_package_available(),
        credentials_present=source is not None,
        credential_source=source,
        config_path=_source_path(source) or str(file_path),
        warnings=tuple(warnings),
    )
