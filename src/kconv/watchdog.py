"""Run heartbeat, stale-run detection and crash recovery.

A Python process cannot catch ``SIGKILL``, a native crash or an OS OOM kill.
The next best thing is a small heartbeat file: when a later command sees a
``running`` run whose heartbeat is old (or whose process is gone), it can mark
the run as ``interrupted`` and produce the same postmortem a caught exception
would produce.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kconv.context import _pid_alive, read_json, write_json
from kconv.failure_report import write_failure_report
from kconv.state import PipelineState, read_latest_pointer, write_latest_pointer

HEARTBEAT_FILE = "heartbeat.json"
DEFAULT_STALE_AFTER_SEC = 120.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_time(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _age_sec(value: Any) -> float | None:
    timestamp = _parse_time(value)
    if timestamp is None:
        return None
    return max(0.0, time.time() - timestamp)


def heartbeat_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / HEARTBEAT_FILE


def write_heartbeat(
    run_dir: str | Path,
    *,
    run_id: str,
    stage: str | None = None,
    done: bool = False,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "updated_at": _now(),
        "stage": stage,
        "done": bool(done),
    }
    if extra:
        payload.update(extra)
    return write_json(heartbeat_path(run_dir), payload)


class Heartbeat:
    """Background heartbeat writer used by the pipeline orchestrator."""

    def __init__(self, ctx: Any, interval_sec: float = 10.0) -> None:
        self.ctx = ctx
        self.interval_sec = max(1.0, float(interval_sec))
        self._stage: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _write(self, done: bool = False) -> None:
        try:
            write_heartbeat(
                self.ctx.run_dir,
                run_id=self.ctx.run_id,
                stage=self._stage,
                done=done,
            )
        except OSError:
            # A heartbeat must never be the reason a training run fails.
            pass

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._write()

    def start(self) -> Heartbeat:
        self._write()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"kconv-heartbeat-{self.ctx.run_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def set_stage(self, stage: str | None) -> None:
        self._stage = stage
        self._write()

    def stop(self, done: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._write(done=done)


def inspect_run(
    project_dir: str | Path,
    run_id: str | None = None,
    *,
    stale_after_sec: float = DEFAULT_STALE_AFTER_SEC,
) -> dict[str, Any]:
    """Return status plus a conservative stale/interrupted diagnosis."""
    project = Path(project_dir).resolve()
    run_id = run_id or read_latest_pointer(project)
    if not run_id:
        return {"run_id": None, "status": "no_run", "stale": False, "reason": None}
    run_dir = project / "runs" / run_id
    run_meta = read_json(run_dir / "run.json", default={}) or {}
    if not isinstance(run_meta, dict):
        run_meta = {}
    status = str(run_meta.get("status") or "unknown")
    heartbeat = read_json(heartbeat_path(run_dir), default={}) or {}
    if not isinstance(heartbeat, dict):
        heartbeat = {}
    result: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "stale": False,
        "reason": None,
        "heartbeat_age_sec": _age_sec(heartbeat.get("updated_at")),
        "pid": heartbeat.get("pid") or run_meta.get("pid"),
        "hostname": heartbeat.get("hostname") or run_meta.get("hostname"),
        "failure_report": run_meta.get("failure_report"),
        "failure_json": run_meta.get("failure_json"),
        "heartbeat": heartbeat,
    }
    if status != "running":
        return result

    pid = result["pid"]
    hostname = result["hostname"]
    same_host = not hostname or str(hostname) == socket.gethostname()
    alive: bool | None = None
    if isinstance(pid, int) and pid > 0 and same_host:
        alive = _pid_alive(pid)
        result["process_alive"] = alive
    if alive is False:
        result.update(stale=True, reason="process is no longer running")
        return result

    age = result["heartbeat_age_sec"]
    if age is not None and age > stale_after_sec:
        result.update(stale=True, reason=f"heartbeat is {age:.0f}s old")
        return result

    if age is None:
        started_age = _age_sec(run_meta.get("started_at"))
        # Old runs created before heartbeats existed have no pid either.  Be
        # generous: only call them stale after a long grace period.
        grace = max(stale_after_sec, 300.0)
        if started_age is not None and started_age > grace:
            result.update(stale=True, reason="no heartbeat and run is old")
    return result


def _load_state(run_dir: Path, run_meta: dict[str, Any]) -> PipelineState:
    path = run_dir / "state.json"
    if path.exists():
        try:
            return PipelineState.load(path)
        except Exception:
            pass
    return PipelineState(
        run_id=str(run_meta.get("run_id") or run_dir.name),
        config_hash=run_meta.get("config_hash"),
        data_hash=run_meta.get("data_hash"),
        data_files=list(run_meta.get("data_files") or []),
        kconv_version=run_meta.get("kconv_version"),
    )


def recover_stale_run(
    project_dir: str | Path,
    run_id: str | None = None,
    *,
    reason: str | None = None,
    stale_after_sec: float = DEFAULT_STALE_AFTER_SEC,
) -> dict[str, Any]:
    """Mark a stale run interrupted and write a postmortem.

    Returns ``{"recovered": False, ...}`` for a live or already finished run.
    """
    project = Path(project_dir).resolve()
    diagnosis = inspect_run(project, run_id, stale_after_sec=stale_after_sec)
    if not diagnosis.get("stale"):
        return {"recovered": False, **diagnosis}
    run_id = str(diagnosis["run_id"])
    run_dir = project / "runs" / run_id
    run_meta = read_json(run_dir / "run.json", default={}) or {}
    if not isinstance(run_meta, dict):
        run_meta = {}
    state = _load_state(run_dir, run_meta)
    interrupted_stage = "unknown"
    for name, entry in state.stages.items():
        if entry.status in {"running", "pending"}:
            entry.error = f"Interrupted: {reason or diagnosis.get('reason') or 'stale heartbeat'}"
            if entry.status == "running":
                interrupted_stage = name
            state.mark_failed(name, entry.error)
    state.save(run_dir / "state.json")

    config_payload = run_meta.get("config") if isinstance(run_meta.get("config"), dict) else {}
    fake_ctx = SimpleNamespace(
        run_id=run_id,
        run_dir=run_dir,
        config=SimpleNamespace(project=str(config_payload.get("project") or run_meta.get("project") or project.name)),
        data_hash=run_meta.get("data_hash"),
        warnings=list(run_meta.get("warnings") or []),
        failed_models=list(run_meta.get("failed_models") or []),
    )
    exc = InterruptedError(
        reason or diagnosis.get("reason") or "The process stopped without a final status update."
    )
    report = write_failure_report(
        fake_ctx,
        stage=interrupted_stage,
        exc=exc,
        state=state,
        run_meta=run_meta,
        log_tail=(diagnosis.get("heartbeat") or {}).get("last_log_lines") or [],
        resume_hint=f"kconv recover --comp {project.name} --run-id {run_id} --resume",
    )
    recovered_meta = {
        **run_meta,
        "status": "interrupted",
        "finished_at": _now(),
        "interrupted_reason": str(exc),
        "failure_report": report.get("markdown"),
        "failure_json": report.get("json"),
        "recovered_at": _now(),
    }
    write_json(run_dir / "run.json", recovered_meta)
    diagnostics = read_json(run_dir / "diagnostics.json", default={}) or {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    diagnostics.update(
        {
            "status": "interrupted",
            "interrupted_reason": str(exc),
            "failure_report": report.get("markdown"),
            "recovered_at": _now(),
        }
    )
    write_json(run_dir / "diagnostics.json", diagnostics)
    latest_before = read_latest_pointer(project)
    latest_pointer_updated = latest_before is None or latest_before == run_id
    if latest_pointer_updated:
        write_latest_pointer(project, run_id, status="interrupted")
    write_heartbeat(run_dir, run_id=run_id, done=True, extra={"recovered": True})
    return {
        "recovered": True,
        "run_id": run_id,
        "status": "interrupted",
        "reason": str(exc),
        "latest_pointer_updated": latest_pointer_updated,
        "failure_report": report.get("markdown"),
        "failure_json": report.get("json"),
    }


def recover_latest(project_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    return recover_stale_run(project_dir, **kwargs)
