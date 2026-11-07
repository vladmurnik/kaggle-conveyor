"""Atomic artifact writers and checksum manifests.

Every helper writes to a private temporary file in the destination directory,
flushes it, and then replaces the destination with ``os.replace``.  Readers can
therefore never observe a half-written parquet/joblib/CSV file, even when a
training process is killed or the disk fills up.

The checksum manifest is intentionally separate from ``state.json``: it can be
rebuilt or verified independently, and old runs without a manifest remain
readable.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib


def _replace(tmp: Path, path: Path) -> None:
    """Replace ``path`` with ``tmp``, tolerating short Windows read locks."""
    for attempt in range(25):
        try:
            os.replace(tmp, path)
            return
        except (PermissionError, FileNotFoundError):
            if attempt == 24:
                raise
            time.sleep(0.02 * (attempt + 1))
    raise AssertionError("unreachable")


def _temp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=path.suffix or ".tmp",
        dir=path.parent,
    )
    os.close(fd)
    return Path(name)


def _cleanup(tmp: Path) -> None:
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass


def write_bytes_atomic(path: str | Path, data: bytes) -> Path:
    """Write bytes atomically and fsync before replacing the destination."""
    path = Path(path)
    tmp = _temp_path(path)
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _replace(tmp, path)
        return path
    finally:
        _cleanup(tmp)


def write_text_atomic(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> Path:
    """Write text atomically (used for reports, configs and manifests)."""
    path = Path(path)
    tmp = _temp_path(path)
    try:
        with tmp.open("w", encoding=encoding, newline=newline) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _replace(tmp, path)
        return path
    finally:
        _cleanup(tmp)


def write_parquet_atomic(frame: Any, path: str | Path) -> Path:
    """Write a DataFrame to parquet through a same-directory temp file."""
    path = Path(path)
    tmp = _temp_path(path)
    try:
        frame.to_parquet(tmp, index=False)
        _replace(tmp, path)
        return path
    finally:
        _cleanup(tmp)


def write_csv_atomic(frame: Any, path: str | Path, *, index: bool = False) -> Path:
    """Write a DataFrame to CSV through a same-directory temp file."""
    path = Path(path)
    tmp = _temp_path(path)
    try:
        frame.to_csv(tmp, index=index)
        _replace(tmp, path)
        return path
    finally:
        _cleanup(tmp)


def dump_joblib_atomic(obj: Any, path: str | Path, *, compress: int = 3) -> Path:
    """Serialize a Python object and validate the temp file before replacing."""
    path = Path(path)
    tmp = _temp_path(path)
    try:
        joblib.dump(obj, tmp, compress=compress)
        # A truncated/corrupt pickle is caught here rather than during blend or
        # inference, when the original object is no longer available.
        joblib.load(tmp)
        _replace(tmp, path)
        return path
    finally:
        _cleanup(tmp)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "artifact_checksums.json"


def _read_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = _manifest_path(run_dir)
    if not path.exists():
        return {"schema_version": 1, "artifacts": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "artifacts": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("artifacts"), dict):
        return {"schema_version": 1, "artifacts": {}}
    return payload


def write_artifact_manifest(run_dir: str | Path, paths: list[str | Path]) -> Path:
    """Merge ``paths`` into ``artifact_checksums.json`` and return its path."""
    run_dir = Path(run_dir)
    payload = _read_manifest(run_dir)
    artifacts = payload["artifacts"]
    root = run_dir.resolve()
    for item in paths:
        path = Path(item)
        if not path.is_absolute():
            path = run_dir / path
        if not path.is_file():
            continue
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if relative == "artifact_checksums.json":
            continue
        artifacts[relative] = {
            "sha256": sha256_file(path),
            "size": int(path.stat().st_size),
        }
    payload["schema_version"] = 1
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = _manifest_path(run_dir)
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def verify_artifacts(run_dir: str | Path, paths: list[str | Path] | None = None) -> list[str]:
    """Return human-readable checksum errors for known artifacts.

    Missing manifest entries are ignored so runs produced before this feature
    (and test fixtures that create only an OOF file) keep working.  A recorded
    artifact that is missing or has a different hash is an error.
    """
    run_dir = Path(run_dir)
    payload = _read_manifest(run_dir)
    records = payload.get("artifacts") or {}
    if not isinstance(records, dict):
        return []
    if paths is None:
        targets = [run_dir / relative for relative in records]
    else:
        targets = []
        for item in paths:
            path = Path(item)
            if not path.is_absolute():
                path = run_dir / path
            targets.append(path)
    errors: list[str] = []
    for path in targets:
        try:
            relative = path.resolve().relative_to(run_dir.resolve()).as_posix()
        except ValueError:
            continue
        record = records.get(relative)
        if not isinstance(record, dict) or not record.get("sha256"):
            continue
        if not path.is_file():
            errors.append(f"Artifact is missing: {relative}")
            continue
        actual = sha256_file(path)
        if actual != record["sha256"]:
            errors.append(
                f"Artifact checksum mismatch: {relative} "
                f"(expected {record['sha256'][:12]}, got {actual[:12]})"
            )
    return errors
