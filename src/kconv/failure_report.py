"""Failure postmortem writer: ``failure.json`` plus a human-readable report."""

from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import Any

from kconv.atomic import write_text_atomic
from kconv.context import RunContext, write_json
from kconv.state import utc_now

_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization|cookie)\b"
    r"(\s*[:=]\s*)([^\s,;]+)"
)


def redact(text: str) -> str:
    """Remove obvious credentials from tracebacks and log tails."""
    return _SECRET_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", text)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _stage_table(stages: dict[str, Any]) -> list[str]:
    lines = ["| Stage | Status | Seconds | Error |", "| --- | --- | ---: | --- |"]
    for name, entry in stages.items():
        if not isinstance(entry, dict):
            continue
        error = str(entry.get("error") or "").replace("|", "\\|").replace("\n", " ")
        if len(error) > 180:
            error = error[:177] + "..."
        seconds = entry.get("duration_sec")
        lines.append(
            f"| {name} | {entry.get('status', '?')} | "
            f"{seconds if seconds is not None else ''} | {error} |"
        )
    return lines


def write_failure_report(
    ctx: RunContext,
    *,
    stage: str,
    exc: BaseException,
    state: Any,
    run_meta: dict[str, Any] | None = None,
    log_tail: list[str] | None = None,
    resume_hint: str | None = None,
) -> dict[str, str]:
    """Write failure metadata and Markdown; return run-relative artifact paths."""
    run_meta = dict(run_meta or {})
    state_dict = state.to_dict() if hasattr(state, "to_dict") else dict(state or {})
    stages = state_dict.get("stages", {}) if isinstance(state_dict, dict) else {}
    if not isinstance(stages, dict):
        stages = {}
    message = redact(str(exc) or type(exc).__name__)
    trace = redact("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    if len(trace) > 20_000:
        trace = trace[:19_800] + "\n... (traceback truncated)"
    warnings = list(dict.fromkeys(redact(str(item)) for item in ctx.warnings))
    failed_models = [
        {
            "model": redact(str(item.get("model", "?"))),
            "error": redact(str(item.get("error", ""))),
            "duration_sec": item.get("duration_sec"),
        }
        for item in ctx.failed_models
        if isinstance(item, dict)
    ]
    logs = [redact(str(line)) for line in (log_tail or [])][-100:]
    hint = resume_hint or (
        f"Fix the issue and rerun with --resume to continue from '{stage}'."
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": ctx.run_id,
        "stage": stage,
        "status": "failed",
        "created_at": utc_now(),
        "exception": {
            "type": type(exc).__name__,
            "message": message,
            "traceback": trace,
        },
        "config_hash": run_meta.get("config_hash"),
        "data_hash": run_meta.get("data_hash", ctx.data_hash),
        "environment": run_meta.get("environment") or {},
        "warnings": warnings,
        "failed_models": failed_models,
        "stages": stages,
        "durations_sec": state_dict.get("durations_sec") if isinstance(state_dict, dict) else None,
        "last_log_lines": logs,
        "resume_hint": hint,
    }
    json_path = ctx.run_dir / "failure.json"
    md_path = ctx.run_dir / "failure_report.md"
    write_json(json_path, payload)

    lines = [
        f"# kconv failure report — {ctx.config.project}",
        "",
        f"- Run: `{ctx.run_id}`",
        f"- Failed stage: `{stage}`",
        f"- Exception: `{type(exc).__name__}`",
        f"- Time: `{payload['created_at']}`",
        f"- Config hash: `{payload.get('config_hash') or 'n/a'}`",
        f"- Data hash: `{payload.get('data_hash') or 'n/a'}`",
        "",
        "## Error",
        "",
        "```",
        message,
        "```",
        "",
        "## Resume",
        "",
        hint,
        "",
        "## Stages",
        "",
        *_stage_table(stages),
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {item}" for item in warnings) if warnings else lines.append("_None._")
    lines += ["", "## Skipped models", ""]
    if failed_models:
        lines += ["| Model | Error | Seconds |", "| --- | --- | ---: |"]
        for item in failed_models:
            error = item["error"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {item['model']} | {error} | {item.get('duration_sec', '')} |")
    else:
        lines.append("_None._")
    lines += ["", "## Last log lines", "", "```"]
    lines.extend(logs or ["(no captured log lines)"])
    lines += ["```", "", "## Traceback", "", "```", trace, "```", ""]
    write_text_atomic(md_path, "\n".join(lines))
    return {
        "json": _relative(json_path, ctx.run_dir),
        "markdown": _relative(md_path, ctx.run_dir),
    }
