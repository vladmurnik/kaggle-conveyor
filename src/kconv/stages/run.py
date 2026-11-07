"""Pipeline orchestrator: stage ordering, resume/skip logic, state and pointers.

``run_pipeline`` is the single entry point behind ``kconv run``.  It runs the
fixed stage order ``ingest -> validate -> hpo -> train -> blend -> submit -> report``,
stops at ``until``, skips stages already marked ``done`` in a resumable run
with a matching config hash, and persists everything needed for ``--resume``:

- ``runs/<run_id>/run.json``   — config snapshot, hash, timings, status;
- ``runs/<run_id>/state.json`` — per-stage status/timestamps/error/artifacts;
- ``latest.json``              — project-level pointer to the newest run.
"""

from __future__ import annotations

import importlib
import logging
import os
import socket
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kconv.atomic import write_artifact_manifest
from kconv.config import Config
from kconv.context import (
    ProjectLock,
    RunContext,
    config_data_hash,
    config_hash,
    data_hash,
    environment_versions,
    read_json,
    validate_run_id,
    write_json,
)
from kconv.contracts import DatasetSpec
from kconv.errors import KconvError, StageError
from kconv.failure_report import write_failure_report
from kconv.runs_index import refresh_index
from kconv.state import PipelineState, read_latest_pointer, utc_now, write_latest_pointer
from kconv.timeouts import run_stage_isolated
from kconv.watchdog import Heartbeat

logger = logging.getLogger("kconv.run")


class _RecentLogHandler(logging.Handler):
    """Keep the last log lines in memory for failure postmortems."""

    def __init__(self, capacity: int = 200) -> None:
        super().__init__()
        self.records: deque[str] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
        except Exception:  # pragma: no cover - logging must never break a run
            self.handleError(record)


_recent_logs = _RecentLogHandler()
_recent_logs.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.getLogger("kconv").addHandler(_recent_logs)


def _recent_log_lines() -> list[str]:
    return list(_recent_logs.records)


def _write_diagnostics(
    ctx: RunContext,
    state: PipelineState,
    run_meta: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist a machine-readable snapshot that survives resume/reporting."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": ctx.run_id,
        "updated_at": utc_now(),
        "status": run_meta.get("status"),
        "stage_status": state.summary(),
        "durations_sec": state.durations(),
        "warnings": list(dict.fromkeys(ctx.warnings)),
        "failed_models": list(ctx.failed_models),
        "environment": environment_versions(),
        "config_hash": run_meta.get("config_hash"),
        "data_hash": ctx.data_hash,
    }
    if extra:
        payload.update(extra)
    write_json(ctx.run_dir / "diagnostics.json", payload)


def _restore_persisted_context(ctx: RunContext, state: PipelineState) -> None:
    """Merge warnings and skipped-model records from the previous attempt."""
    previous = read_json(ctx.run_dir / "run.json", default={})
    if isinstance(previous, dict):
        for warning in previous.get("warnings", []) or []:
            if isinstance(warning, str) and warning not in ctx.warnings:
                ctx.warnings.append(warning)
        for item in previous.get("failed_models", []) or []:
            if isinstance(item, dict) and item not in ctx.failed_models:
                ctx.failed_models.append(item)
    train_artifacts = state.stage("train").artifacts
    if isinstance(train_artifacts, dict):
        for item in train_artifacts.get("failed_models", []) or []:
            if isinstance(item, dict) and item not in ctx.failed_models:
                ctx.failed_models.append(item)


def _fail_stage(
    ctx: RunContext,
    state: PipelineState,
    name: str,
    exc: BaseException,
    started: float,
    run_meta: dict[str, Any],
    project_dir: Path,
    current_hash: str,
    error_text: str,
) -> float:
    """Persist a failed stage, write postmortem artifacts and return elapsed."""
    elapsed = time.perf_counter() - started
    state.mark_failed(name, error_text, duration_sec=elapsed)
    state.save(ctx.run_dir / "state.json")
    write_latest_pointer(project_dir, ctx.run_id, config_hash=current_hash, status="failed")
    failure_paths: dict[str, str] = {}
    try:
        failure_paths = write_failure_report(
            ctx,
            stage=name,
            exc=exc,
            state=state,
            run_meta=run_meta,
            log_tail=_recent_log_lines(),
        )
    except Exception:  # noqa: BLE001 - never mask the original stage failure
        logger.exception("could not write failure report for stage '%s'", name)
    payload = {
        **run_meta,
        "status": "failed",
        "failed_stage": name,
        "durations_sec": state.durations(),
        "warnings": list(dict.fromkeys(ctx.warnings)),
        "failed_models": list(ctx.failed_models),
    }
    if failure_paths:
        payload["failure_report"] = failure_paths["markdown"]
        payload["failure_json"] = failure_paths["json"]
    ctx.write_run_metadata(payload)
    _write_diagnostics(
        ctx,
        state,
        payload,
        extra={"failed_stage": name, "error": error_text},
    )
    try:
        refresh_index(project_dir)
    except Exception as exc:  # noqa: BLE001 - index is a convenience cache
        logger.warning("could not refresh run index after failure: %s", exc)
    return elapsed

STAGE_ORDER: tuple[str, ...] = ("ingest", "validate", "hpo", "train", "blend", "submit", "report")

# stage name -> (module, factory function); imported lazily so that a missing
# optional stage only fails when it is actually reached.
_STAGE_SPECS: dict[str, tuple[str, str]] = {
    "ingest": ("kconv.stages.ingest", "run_ingest"),
    "validate": ("kconv.stages.validate", "run_validate"),
    "hpo": ("kconv.stages.hpo", "run_hpo"),
    "train": ("kconv.stages.train", "run_train"),
    "blend": ("kconv.stages.blend", "run_blend"),
    "submit": ("kconv.stages.submit", "run_submit"),
    "report": ("kconv.stages.report", "run_report"),
}


def _load_stage_runner(name: str) -> Callable[[RunContext], Any]:
    module_name, func_name = _STAGE_SPECS[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise StageError(
            f"Stage '{name}' is unavailable: cannot import {module_name} ({exc}).",
            "Reinstall kconv or stop the run earlier with --until.",
        ) from exc
    runner = getattr(module, func_name, None)
    if runner is None or not callable(runner):
        raise StageError(
            f"Stage module {module_name} does not define {func_name}().",
            "This is a packaging bug; report it with the traceback.",
        )
    return runner


def _resolve_stage_plan(until: str) -> tuple[str, ...]:
    if until not in _STAGE_SPECS:
        raise StageError(
            f"Unknown --until stage '{until}'.",
            f"Valid values: {', '.join(STAGE_ORDER)}",
        )
    return STAGE_ORDER[: STAGE_ORDER.index(until) + 1]


def _restore_dataset(ctx: RunContext, state: PipelineState) -> None:
    """Best-effort restore of ``ctx.dataset`` when ingest was already done."""
    artifacts = state.stage("ingest").artifacts
    raw = artifacts.get("dataset") if isinstance(artifacts, dict) else None
    candidates: list[Any] = [raw]
    for filename in ("dataset.json", "schema.json"):
        data = read_json(ctx.run_dir / "00_ingest" / filename, default=None)
        candidates.append(data)
    for candidate in candidates:
        if not isinstance(candidate, dict) or "train" not in candidate or "target" not in candidate:
            continue
        try:
            ctx.dataset = DatasetSpec.from_dict(candidate)
            return
        except Exception:  # noqa: BLE001 - restore is best-effort only
            continue


def run_pipeline(
    project_dir: str | Path,
    config_path: str | Path,
    config: Config,
    *,
    until: str = "report",
    resume: bool = False,
    force: bool = False,
) -> RunContext:
    """Acquire the project lock and execute one pipeline run."""
    project_dir = Path(project_dir).resolve()
    with ProjectLock(project_dir):
        return _run_pipeline(
            project_dir,
            config_path,
            config,
            until=until,
            resume=resume,
            force=force,
        )


def _run_pipeline(
    project_dir: str | Path,
    config_path: str | Path,
    config: Config,
    *,
    until: str = "report",
    resume: bool = False,
    force: bool = False,
) -> RunContext:
    """Run pipeline stages in order up to ``until`` (inclusive).

    Returns the :class:`RunContext` of the executed (or resumed) run.  On stage
    failure the stage is marked ``failed`` in ``state.json`` and a
    :class:`~kconv.errors.StageError` carrying the stage name plus the original
    message is raised.
    """
    stage_plan = _resolve_stage_plan(until)
    project_dir = Path(project_dir).resolve()
    current_hash = config_hash(config)
    current_version = environment_versions()["kconv_version"]
    current_data_hash, current_data_files = config_data_hash(project_dir, config)

    ctx: RunContext | None = None
    state: PipelineState | None = None

    if resume:
        pointer_id = read_latest_pointer(project_dir)
        run_id = None
        if pointer_id:
            try:
                run_id = validate_run_id(pointer_id)
            except KconvError:
                logger.warning(
                    "Ignoring invalid latest run id %r; starting a fresh run.",
                    pointer_id,
                )
        state_path = project_dir / "runs" / run_id / "state.json" if run_id else None
        if run_id and state_path is not None and state_path.exists():
            loaded = PipelineState.load(state_path)
            config_matches = loaded.config_hash == current_hash
            version_matches = loaded.kconv_version == current_version
            stored_files = [
                Path(str(item.get("path")))
                for item in loaded.data_files
                if isinstance(item, dict) and item.get("path")
            ]
            if loaded.data_hash is not None and stored_files:
                if all(path.exists() for path in stored_files):
                    current_data_hash = data_hash(stored_files)
                    data_matches = loaded.data_hash == current_data_hash
                else:
                    data_matches = False
            else:
                data_matches = (
                    loaded.data_hash is None
                    or current_data_hash is None
                    or loaded.data_hash == current_data_hash
                )
            if config_matches and data_matches and version_matches:
                ctx = RunContext.create(project_dir, config_path, config, run_id=run_id)
                ctx.data_hash = current_data_hash
                ctx.data_files = list(loaded.data_files) or current_data_files
                state = loaded
                _restore_persisted_context(ctx, state)
                ctx.warnings.append(f"Resuming run {run_id}.")
                logger.info("resuming run %s (config/data/version hashes match)", run_id)
            elif not config_matches:
                warning = (
                    f"Cannot resume run {run_id}: config hash changed "
                    f"({loaded.config_hash} -> {current_hash}); starting a fresh run."
                )
                logger.warning(warning)
            elif not version_matches:
                warning = (
                    f"Cannot resume run {run_id}: kconv version changed "
                    f"({loaded.kconv_version} -> {current_version}); starting a fresh run."
                )
                logger.warning(warning)
            else:
                warning = (
                    f"Cannot resume run {run_id}: raw data changed "
                    f"({loaded.data_hash} -> {current_data_hash}); starting a fresh run."
                )
                logger.warning(warning)
        else:
            logger.warning("No previous run found for this project; starting a fresh run.")

    if ctx is None or state is None:
        ctx = RunContext.create(project_dir, config_path, config)
        ctx.data_hash = current_data_hash
        ctx.data_files = current_data_files
        state = PipelineState(
            run_id=ctx.run_id,
            config_hash=current_hash,
            data_hash=current_data_hash,
            data_files=current_data_files,
            kconv_version=current_version,
        )
        for name in STAGE_ORDER:
            state.stage(name)

    _restore_dataset(ctx, state)

    state.data_hash = ctx.data_hash
    state.data_files = list(ctx.data_files)
    state.kconv_version = current_version
    run_meta: dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "until": until,
        "resume": resume,
        "force": force,
        "stage_plan": list(stage_plan),
        "config_hash": current_hash,
        "environment": environment_versions(),
        "data_hash": ctx.data_hash,
        "data_files": list(ctx.data_files),
        "config_path": str(Path(config_path).resolve()),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
    }
    ctx.write_run_metadata(run_meta)
    state.save(ctx.run_dir / "state.json")
    write_latest_pointer(project_dir, ctx.run_id, config_hash=current_hash, status="running")
    _write_diagnostics(ctx, state, run_meta)
    heartbeat = Heartbeat(ctx).start()
    runtime = ctx.config.runtime
    stage_timeout = float(getattr(runtime, "stage_timeout_sec", 0) or 0)

    for name in stage_plan:
        if resume and not force and state.is_done(name):
            logger.info("stage '%s' skipped: already done in run %s", name, ctx.run_id)
            continue

        state.mark_running(name)
        state.save(ctx.run_dir / "state.json")
        heartbeat.set_stage(name)
        logger.info("stage '%s' started (run %s)", name, ctx.run_id)
        started = time.perf_counter()
        try:
            spec = _STAGE_SPECS.get(name)
            if stage_timeout > 0 and spec is not None:
                result = run_stage_isolated(spec[0], spec[1], ctx, stage_timeout)
            else:
                runner = _load_stage_runner(name)
                result = runner(ctx)
        except StageError as exc:
            heartbeat.stop(done=False)
            elapsed = _fail_stage(
                ctx,
                state,
                name,
                exc,
                started,
                run_meta,
                project_dir,
                current_hash,
                str(exc) or "StageError",
            )
            raise
        except Exception as exc:
            heartbeat.stop(done=False)
            elapsed = _fail_stage(
                ctx,
                state,
                name,
                exc,
                started,
                run_meta,
                project_dir,
                current_hash,
                f"{type(exc).__name__}: {exc}",
            )
            raise StageError(
                f"Stage '{name}' failed after {elapsed:.1f}s: {exc}",
                f"Fix the issue and rerun with --resume to continue from '{name}'.",
            ) from exc

        elapsed = time.perf_counter() - started
        artifacts = result if isinstance(result, dict) else {}
        if isinstance(result, dict):
            # A stage may have run in a child process: mutations to ``ctx``
            # there are not visible here, so merge the serializable fields
            # that later stages and the final run metadata depend on.
            if result.get("data_hash") is not None:
                ctx.data_hash = result["data_hash"]
            if isinstance(result.get("data_files"), list):
                ctx.data_files = [
                    item for item in result["data_files"] if isinstance(item, dict)
                ]
            for warning in result.get("warnings") or []:
                text = str(warning)
                if text and text not in ctx.warnings:
                    ctx.warnings.append(text)
            current_failed = [
                item
                for item in (result.get("failed_models", []) or [])
                if isinstance(item, dict)
            ]
            if name == "train":
                # A resumed train replaces the previous failure list; a model
                # that now succeeds must not keep an old "skipped" status.
                ctx.failed_models = current_failed
            else:
                for item in current_failed:
                    if item not in ctx.failed_models:
                        ctx.failed_models.append(item)
        state.data_hash = ctx.data_hash
        state.data_files = list(ctx.data_files)
        state.kconv_version = current_version
        state.mark_done(name, artifacts=artifacts, duration_sec=elapsed)
        state.save(ctx.run_dir / "state.json")
        _restore_dataset(ctx, state)
        artifact_paths = artifacts.get("artifacts", []) if isinstance(artifacts, dict) else []
        if artifact_paths:
            try:
                write_artifact_manifest(ctx.run_dir, list(artifact_paths))
            except Exception as exc:  # noqa: BLE001 - manifest is best-effort
                logger.warning("could not write artifact checksums: %s", exc)
        _write_diagnostics(ctx, state, run_meta, extra={"last_stage": name})
        logger.info("stage '%s' done in %.2fs", name, elapsed)

    finished_stages = [name for name in STAGE_ORDER if state.is_done(name)]
    remaining_stages = [name for name in STAGE_ORDER if not state.is_done(name)]
    complete = until == "report" and not remaining_stages
    if complete and ctx.failed_models:
        final_status = "completed_with_warnings"
        partial_reason = "One or more models were skipped; see warnings and failed_models."
    elif complete:
        final_status = "completed"
        partial_reason = None
    else:
        final_status = "partial"
        partial_reason = (
            f"Stopped after --until {until}." if until != "report" else "Some stages are not done."
        )
    state.save(ctx.run_dir / "state.json")
    final_meta = {
        **run_meta,
        "status": final_status,
        "finished_at": utc_now(),
        "durations_sec": state.durations(),
        "finished_stages": finished_stages,
        "remaining_stages": remaining_stages,
        "partial_reason": partial_reason,
        "data_hash": ctx.data_hash,
        "data_files": list(ctx.data_files),
    }
    heartbeat.stop(done=True)
    ctx.write_run_metadata(final_meta)
    _write_diagnostics(ctx, state, final_meta, extra={"finished_stages": finished_stages})
    write_latest_pointer(
        project_dir,
        ctx.run_id,
        config_hash=current_hash,
        status="done" if final_status == "completed" else final_status,
    )
    try:
        refresh_index(project_dir)
    except Exception as exc:  # noqa: BLE001 - index is a convenience cache
        logger.warning("could not refresh run index: %s", exc)
    logger.info(
        "run %s %s: %s",
        ctx.run_id,
        final_status,
        ", ".join(f"{k}={v:.2f}s" for k, v in state.durations().items()),
    )
    return ctx
