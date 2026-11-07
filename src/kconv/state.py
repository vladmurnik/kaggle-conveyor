"""Pipeline run state: atomic state.json IO, stage statuses and latest-run pointer.

``state.json`` lives inside a run directory (``runs/<run_id>/state.json``) and
tracks one :class:`StageState` per pipeline stage: status, timestamps, duration,
error text and the artifacts the stage reported.  Writes are atomic
(temp file + ``os.replace``) via :func:`kconv.context.write_json`.

``latest.json`` lives in the project root and points at the most recent run so
that ``--resume`` can find it without symlinks (Windows-friendly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from kconv.context import read_json, write_json

StageStatus = Literal["pending", "running", "done", "failed"]
STAGE_STATUSES: tuple[str, ...] = ("pending", "running", "done", "failed")

STATE_FILENAME = "state.json"
LATEST_FILENAME = "latest.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class StageState:
    """Status of a single pipeline stage."""

    name: str
    status: StageStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    attempts: int = 0
    error: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.duration_sec,
            "attempts": self.attempts,
            "error": self.error,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StageState:
        status = raw.get("status", "pending")
        if status not in STAGE_STATUSES:
            status = "pending"
        artifacts = raw.get("artifacts")
        return cls(
            name=str(raw.get("name", "")),
            status=status,  # type: ignore[arg-type]
            started_at=raw.get("started_at"),
            finished_at=raw.get("finished_at"),
            duration_sec=raw.get("duration_sec"),
            attempts=int(raw.get("attempts", 0) or 0),
            error=raw.get("error"),
            artifacts=artifacts if isinstance(artifacts, dict) else {},
        )


@dataclass
class PipelineState:
    """Full run state persisted as ``state.json`` inside a run directory."""

    run_id: str
    config_hash: str
    data_hash: str | None = None
    data_files: list[dict[str, str]] = field(default_factory=list)
    kconv_version: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    stages: dict[str, StageState] = field(default_factory=dict)

    # -- stage helpers -----------------------------------------------------

    def stage(self, name: str) -> StageState:
        """Get (or lazily create) the state entry for ``name``."""
        entry = self.stages.get(name)
        if entry is None:
            entry = StageState(name=name)
            self.stages[name] = entry
        return entry

    def is_done(self, name: str) -> bool:
        entry = self.stages.get(name)
        return entry is not None and entry.status == "done"

    def mark_running(self, name: str) -> StageState:
        entry = self.stage(name)
        entry.status = "running"
        entry.started_at = utc_now()
        entry.finished_at = None
        entry.duration_sec = None
        entry.attempts += 1
        entry.error = None
        self.updated_at = utc_now()
        return entry

    def mark_done(self, name: str, artifacts: dict[str, Any] | None = None, duration_sec: float | None = None) -> StageState:
        entry = self.stage(name)
        entry.status = "done"
        entry.finished_at = utc_now()
        if duration_sec is not None:
            entry.duration_sec = round(duration_sec, 3)
        if artifacts is not None:
            entry.artifacts = artifacts
        entry.error = None
        self.updated_at = utc_now()
        return entry

    def mark_failed(self, name: str, error: str, duration_sec: float | None = None) -> StageState:
        entry = self.stage(name)
        entry.status = "failed"
        entry.finished_at = utc_now()
        if duration_sec is not None:
            entry.duration_sec = round(duration_sec, 3)
        entry.error = error
        self.updated_at = utc_now()
        return entry

    def durations(self) -> dict[str, float]:
        return {
            name: entry.duration_sec
            for name, entry in self.stages.items()
            if entry.duration_sec is not None
        }

    def summary(self) -> dict[str, str]:
        return {name: entry.status for name, entry in self.stages.items()}

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "data_files": self.data_files,
            "kconv_version": self.kconv_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stages": {name: entry.to_dict() for name, entry in self.stages.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PipelineState:
        stages_raw = raw.get("stages")
        stages: dict[str, StageState] = {}
        if isinstance(stages_raw, dict):
            for name, entry in stages_raw.items():
                if isinstance(entry, dict):
                    state = StageState.from_dict({**entry, "name": entry.get("name", name)})
                    stages[str(name)] = state
        return cls(
            run_id=str(raw.get("run_id", "")),
            config_hash=str(raw.get("config_hash", "")),
            data_hash=raw.get("data_hash"),
            data_files=raw.get("data_files") if isinstance(raw.get("data_files"), list) else [],
            kconv_version=raw.get("kconv_version"),
            created_at=raw.get("created_at") or utc_now(),
            updated_at=raw.get("updated_at") or utc_now(),
            stages=stages,
        )

    def save(self, path: str | Path) -> Path:
        self.updated_at = utc_now()
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> PipelineState:
        raw = read_json(path, default=None)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid state file (expected JSON object): {path}")
        return cls.from_dict(raw)


# -- latest-run pointer ------------------------------------------------------


def latest_pointer_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / LATEST_FILENAME


def write_latest_pointer(
    project_dir: str | Path,
    run_id: str,
    config_hash: str | None = None,
    status: str | None = None,
) -> Path:
    payload: dict[str, Any] = {"run_id": run_id, "updated_at": utc_now()}
    if config_hash is not None:
        payload["config_hash"] = config_hash
    if status is not None:
        payload["status"] = status
    return write_json(latest_pointer_path(project_dir), payload)


def read_latest_pointer(project_dir: str | Path) -> str | None:
    """Return the latest run id, or ``None`` when the project has no runs."""
    raw = read_json(latest_pointer_path(project_dir), default=None)
    if isinstance(raw, dict):
        run_id = raw.get("run_id")
        return str(run_id) if run_id else None
    if isinstance(raw, str) and raw:
        return raw
    return None
