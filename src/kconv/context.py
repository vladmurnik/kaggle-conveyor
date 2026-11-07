"""Run context, hashes, artifact paths and small atomic IO helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from kconv.config import Config, FeatureConfig
from kconv.contracts import DatasetSpec
from kconv.errors import KconvError
from kconv.io import is_table


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    return value or "project"


def _hash_payload(config: Config) -> dict[str, Any]:
    """Return the behavior-stable config payload used for ``config_hash``.

    Feature additions after the original hash format use safe defaults
    (``False``, empty/default lists, a disabled kNN object).  They must not
    change the hash of configs that predate the fields, otherwise ``--resume``
    rejects old runs.  Remove a field only when its value is the default; any
    non-default value remains part of the hash.
    """
    payload = config.model_dump(mode="json", exclude_none=True)
    # Provenance only; it does not affect the resolved feature matrix.
    payload.pop("recipe_kind", None)
    # New runtime reliability fields must not invalidate old resumable runs
    # when they keep their safe defaults.  Non-default values still hash.
    runtime = payload.get("runtime")
    if isinstance(runtime, dict):
        for key, default in (
            ("stage_timeout_sec", 0),
            ("model_timeout_sec", 0),
            ("model_retries", 0),
            ("fallback_models", []),
            ("isolate_models", False),
            ("feature_cache", True),
        ):
            if runtime.get(key) == default:
                runtime.pop(key, None)
    features = payload.get("features")
    if not isinstance(features, dict):
        return payload

    defaults = FeatureConfig().model_dump(mode="json")
    for key in ("row_statistics_stats", "modulo_moduli", "modulo_steps"):
        if features.get(key) == defaults[key]:
            features.pop(key, None)
    if features.get("row_statistics") is False:
        features.pop("row_statistics", None)
        features.pop("row_statistics_stats", None)
    if features.get("modulo_features") is False:
        features.pop("modulo_features", None)
        features.pop("modulo_moduli", None)
        features.pop("modulo_steps", None)
    knn = features.get("knn_features")
    if isinstance(knn, dict) and knn.get("enabled") is False:
        features.pop("knn_features", None)
    selection = features.get("feature_selection")
    if isinstance(selection, dict) and selection.get("enabled") is False:
        features.pop("feature_selection", None)
    if not features.get("hash_encode"):
        features.pop("hash_encode", None)
        features.pop("hash_features", None)
    if not features.get("interactions"):
        features.pop("interactions", None)
    missingness = features.get("missingness")
    if isinstance(missingness, dict) and missingness.get("enabled") is False:
        features.pop("missingness", None)
    cyclical = features.get("datetime_cyclical")
    if isinstance(cyclical, dict) and cyclical.get("enabled") is False:
        features.pop("datetime_cyclical", None)
    for key in ("numeric_interactions", "clustering", "anomaly", "time_series"):
        value = features.get(key)
        if isinstance(value, dict) and value.get("enabled") is False:
            features.pop(key, None)
    return payload


def config_hash(config: Config) -> str:
    payload = json.dumps(
        _hash_payload(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_hash(paths: list[Path | None]) -> str:
    digest = hashlib.sha256()
    for path in sorted((p for p in paths if p is not None), key=lambda p: str(p)):
        digest.update(str(path).encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


_PACKAGE_DISTS: tuple[tuple[str, str], ...] = (
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("xgboost", "xgboost"),
    ("catboost", "catboost"),
    ("pyarrow", "pyarrow"),
)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def environment_versions() -> dict[str, Any]:
    """Return reproducible runtime provenance for ``run.json`` and reports."""
    from kconv import __version__

    packages = {name: _distribution_version(dist) for name, dist in _PACKAGE_DISTS}
    return {
        "kconv_version": _distribution_version("kconv") or __version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
    }


def resolve_data_paths(project_dir: str | Path, config: Config) -> list[Path]:
    """Resolve configured raw data files that exist on disk."""
    base = Path(project_dir).resolve()
    paths: list[Path] = []
    for value in (config.data.train, config.data.test, config.data.sample_submission):
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = base / candidate
        if not candidate.exists():
            continue
        if candidate.is_dir():
            paths.extend(
                child.resolve()
                for child in sorted(candidate.iterdir())
                if child.is_file() and is_table(child)
            )
        elif candidate.is_file():
            paths.append(candidate.resolve())
    return paths


def config_data_hash(project_dir: str | Path, config: Config) -> tuple[str | None, list[dict[str, str]]]:
    """Hash raw data files and return provenance records for the run metadata."""
    paths = resolve_data_paths(project_dir, config)
    if not paths:
        return None, []
    files = [{"path": str(path), "sha256": file_sha256(path)} for path in paths]
    return data_hash(paths), files


def new_run_id(config: Config) -> str:
    # Microseconds keep two quick successive runs distinct even when they
    # share the same config hash and start within the same second.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{config_hash(config)}"


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_run_id(run_id: str) -> str:
    """Return a safe run-directory name or raise ``KconvError``."""
    value = str(run_id or "").strip()
    if not _RUN_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise KconvError(f"Invalid run id: {run_id!r}", "Use a run directory name from runs/.")
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    """Atomically write JSON using a unique temp file per writer.

    A fixed ``<name>.tmp`` is unsafe when two kconv processes (or the web UI)
    write the same state file concurrently.  ``mkstemp`` gives every writer a
    private file, and ``os.replace`` is atomic on both POSIX and Windows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
        # On Windows os.replace can fail if another process (the web UI, kconv
        # status, an editor) has the destination open for reading.  The read is
        # short-lived, so retry briefly instead of failing a training run.
        for attempt in range(25):
            try:
                os.replace(tmp, path)
                return path
            except (PermissionError, FileNotFoundError):
                if attempt == 24:
                    raise
                time.sleep(0.02 * (attempt + 1))
        raise AssertionError("unreachable")
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is not a harmless existence check on Windows: the
        # Windows implementation may terminate the process for non-console
        # signals.  OpenProcess is the safe primitive.
        import ctypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


class ProjectLock:
    """Cross-process lock for one project directory.

    The lock file is created with ``O_EXCL`` so acquisition is atomic on both
    Windows and POSIX.  A lock is stale when its PID is gone or it is older
    than ``stale_after_sec``; this prevents a crashed run from blocking the
    project forever without silently stealing a live run's lock.
    """

    def __init__(self, project_dir: str | Path, *, stale_after_sec: float = 12 * 3600) -> None:
        self.path = Path(project_dir).resolve() / ".kconv.lock"
        self.stale_after_sec = float(stale_after_sec)
        self.token = uuid.uuid4().hex
        self.acquired = False

    def _stale(self, info: Any) -> bool:
        if not isinstance(info, dict):
            return True
        created = info.get("created_at")
        if isinstance(created, str):
            try:
                age = (datetime.now(UTC) - datetime.fromisoformat(created)).total_seconds()
                if age > self.stale_after_sec:
                    return True
            except ValueError:
                pass
        pid = info.get("pid")
        if isinstance(pid, int) and not _pid_alive(pid):
            return True
        return False

    def acquire(self) -> None:
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at": datetime.now(UTC).isoformat(),
            "token": self.token,
        }
        for attempt in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
                self.acquired = True
                return
            except FileExistsError:
                try:
                    info = read_json(self.path, default=None)
                except ValueError:
                    info = None
                if attempt == 0 and self._stale(info):
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                owner = info if isinstance(info, dict) else {}
                owner_text = (
                    f"pid {owner.get('pid')} on {owner.get('hostname')}"
                    if owner
                    else "another process"
                )
                raise KconvError(
                    f"Project is already in use by {owner_text}: {self.path}",
                    "Stop the other kconv run, or remove the lock file if that process is gone.",
                ) from None
        raise KconvError(f"Could not acquire project lock: {self.path}")

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            info = read_json(self.path, default=None)
            if isinstance(info, dict) and info.get("token") == self.token:
                self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self.acquired = False

    def __enter__(self) -> ProjectLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


@dataclass
class RunContext:
    project_dir: Path
    config_path: Path
    config: Config
    run_id: str
    run_dir: Path
    dataset: DatasetSpec | None = None
    seed: int = 42
    data_hash: str | None = None
    data_files: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failed_models: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        project_dir: str | Path,
        config_path: str | Path,
        config: Config,
        run_id: str | None = None,
    ) -> RunContext:
        project_dir = Path(project_dir).resolve()
        config_path = Path(config_path).resolve()
        run_id = validate_run_id(run_id) if run_id else new_run_id(config)
        run_dir = project_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            project_dir=project_dir,
            config_path=config_path,
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            seed=config.seed,
        )

    def stage_dir(self, name: str) -> Path:
        path = self.run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact(self, stage: str, *parts: str) -> Path:
        path = self.stage_dir(stage).joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_run_metadata(self, extra: dict[str, Any] | None = None) -> Path:
        path = self.run_dir / "run.json"
        try:
            existing = read_json(path, default={})
        except ValueError:
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        environment = environment_versions()
        payload = {
            "run_id": self.run_id,
            "project": self.config.project,
            "recipe": self.config.recipe,
            "strategy": self.config.strategy,
            "config_hash": config_hash(self.config),
            "seed": self.seed,
            "created_at": existing.get("created_at") or datetime.now(UTC).isoformat(),
            # Keep both the flat, documented keys and a grouped copy for the
            # report/UI; flat keys are what the audit and bug reports expect.
            "environment": environment,
            "kconv_version": environment["kconv_version"],
            "python": environment["python"],
            "platform": environment["platform"],
            "packages": environment["packages"],
            "data_hash": self.data_hash,
            "data_files": list(self.data_files),
            "warnings": list(dict.fromkeys(self.warnings)),
            "failed_models": list(self.failed_models),
            "config": self.config.model_dump(mode="json", exclude_none=True),
        }
        if extra:
            payload.update(extra)
        return write_json(path, payload)
