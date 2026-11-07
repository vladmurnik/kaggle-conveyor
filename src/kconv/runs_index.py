"""Project run index, comparison and leaderboard helpers.

The index is intentionally a plain JSON file: it is easy to inspect, works
without a database and stays compatible with the stdlib-only web UI.  The
authoritative run data remains in ``runs/<id>/run.json`` and stage artifacts;
the index can always be rebuilt from disk.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kconv.context import read_json, write_json

INDEX_FILE = "index.json"
INDEX_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _read(path: Path, default: Any = None) -> Any:
    try:
        return read_json(path, default=default)
    except (OSError, ValueError):
        return default


def _state_duration(state: Any) -> float | None:
    if not isinstance(state, dict):
        return None
    stages = state.get("stages")
    if not isinstance(stages, dict):
        return None
    total = 0.0
    seen = False
    for entry in stages.values():
        if not isinstance(entry, dict):
            continue
        value = _float(entry.get("duration_sec"))
        if value is not None:
            total += value
            seen = True
    return round(total, 3) if seen else None


def _models(run_meta: dict[str, Any], run_dir: Path) -> list[str]:
    config = run_meta.get("config")
    if isinstance(config, dict):
        models = config.get("models")
        if isinstance(models, list):
            names = [
                str(item.get("name"))
                for item in models
                if isinstance(item, dict) and item.get("enabled", True) and item.get("name")
            ]
            if names:
                return names
    train_dir = run_dir / "02_train"
    if train_dir.is_dir():
        return sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    return []


def _score(run_dir: Path) -> tuple[float | None, str | None]:
    blend = _read(run_dir / "03_blend" / "blend.json")
    if isinstance(blend, dict):
        score = _float(blend.get("oof_score"))
        metric = blend.get("metric") if isinstance(blend.get("metric"), str) else None
        if score is not None:
            return score, metric
    report = _read(run_dir / "03_blend" / "metrics.json")
    if isinstance(report, dict):
        score = _float(report.get("oof_score") or report.get("score"))
        metric = report.get("metric") if isinstance(report.get("metric"), str) else None
        if score is not None:
            return score, metric
    return None, None


def _summary(run_dir: Path, tags: list[str] | None = None) -> dict[str, Any] | None:
    run_meta = _read(run_dir / "run.json")
    if not isinstance(run_meta, dict):
        return None
    state = _read(run_dir / "state.json", default={})
    score, metric = _score(run_dir)
    config = run_meta.get("config") if isinstance(run_meta.get("config"), dict) else {}
    failed = run_meta.get("failed_models") if isinstance(run_meta.get("failed_models"), list) else []
    warnings = run_meta.get("warnings") if isinstance(run_meta.get("warnings"), list) else []
    duration = _state_duration(state)
    if duration is None:
        duration = _float(run_meta.get("duration_sec"))
    return {
        "run_id": run_dir.name,
        "project": run_meta.get("project") or config.get("project"),
        "status": run_meta.get("status") or "unknown",
        "recipe": run_meta.get("recipe") or config.get("recipe"),
        "strategy": run_meta.get("strategy") or config.get("strategy"),
        "task": config.get("task"),
        "metric": metric or config.get("metric"),
        "score": score,
        "oof_score": score,
        "models": _models(run_meta, run_dir),
        "failed_models": failed,
        "warning_count": len(warnings),
        "warnings": warnings[:20],
        "started_at": run_meta.get("started_at"),
        "finished_at": run_meta.get("finished_at"),
        "duration_sec": duration,
        "until": run_meta.get("until"),
        "config_hash": run_meta.get("config_hash"),
        "data_hash": run_meta.get("data_hash"),
        "failure_report": run_meta.get("failure_report"),
        "failure_json": run_meta.get("failure_json"),
        "tags": list(tags or run_meta.get("tags") or []),
    }


def _existing_tags(project_dir: Path) -> dict[str, list[str]]:
    index = _read(project_dir / "runs" / INDEX_FILE, default={})
    result: dict[str, list[str]] = {}
    if isinstance(index, dict) and isinstance(index.get("runs"), list):
        for run in index["runs"]:
            if isinstance(run, dict) and run.get("run_id"):
                result[str(run["run_id"])] = list(run.get("tags") or [])
    return result


def refresh_index(project_dir: str | Path) -> dict[str, Any]:
    project = Path(project_dir).resolve()
    runs_dir = project / "runs"
    tags = _existing_tags(project)
    runs: list[dict[str, Any]] = []
    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir(), key=lambda path: path.name, reverse=True):
            if not run_dir.is_dir():
                continue
            summary = _summary(run_dir, tags.get(run_dir.name))
            if summary is not None:
                runs.append(summary)
    runs.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "project": project.name,
        "generated_at": _now(),
        "runs": runs,
    }
    write_json(runs_dir / INDEX_FILE, payload)
    return payload


def list_runs(
    project_dir: str | Path,
    *,
    status: str | None = None,
    limit: int | None = None,
    refresh: bool = True,
) -> list[dict[str, Any]]:
    payload = refresh_index(project_dir) if refresh else (_read(Path(project_dir) / "runs" / INDEX_FILE, {}) or {})
    runs = payload.get("runs") if isinstance(payload, dict) else []
    result = [item for item in runs if isinstance(item, dict)]
    if status:
        result = [item for item in result if item.get("status") == status]
    if limit is not None:
        result = result[: max(0, int(limit))]
    return result


def _config(run_dir: Path) -> dict[str, Any]:
    run_meta = _read(run_dir / "run.json", default={}) or {}
    config = run_meta.get("config") if isinstance(run_meta, dict) else {}
    return config if isinstance(config, dict) else {}


def _diff(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left:
                result.append({"path": child, "left": None, "right": right[key]})
            elif key not in right:
                result.append({"path": child, "left": left[key], "right": None})
            else:
                result.extend(_diff(left[key], right[key], child))
        return result
    if left != right:
        return [{"path": path or "$", "left": left, "right": right}]
    return []


def _run_dir(project_dir: str | Path, run_id: str) -> Path:
    safe = str(run_id).strip()
    if not safe or "/" in safe or "\\" in safe or safe in {".", ".."}:
        raise ValueError(f"Invalid run id: {run_id!r}")
    path = Path(project_dir).resolve() / "runs" / safe
    if not (path / "run.json").exists():
        raise ValueError(f"Run not found: {run_id}")
    return path


def compare_runs(project_dir: str | Path, left_id: str, right_id: str) -> dict[str, Any]:
    project = Path(project_dir).resolve()
    left_dir = _run_dir(project, left_id)
    right_dir = _run_dir(project, right_id)
    left = _summary(left_dir)
    right = _summary(right_dir)
    if left is None or right is None:
        raise ValueError("One of the runs has no run.json.")
    return {
        "left": left,
        "right": right,
        "config_diff": _diff(_config(left_dir), _config(right_dir)),
        "score_delta": (
            (right.get("score") or 0.0) - (left.get("score") or 0.0)
            if right.get("score") is not None and left.get("score") is not None
            else None
        ),
    }


_MINIMIZE = {"logloss", "rmse", "mae", "mse", "rmsle", "mape"}


def best_run(
    project_dir: str | Path,
    *,
    metric: str | None = None,
    status: str | None = "completed",
) -> dict[str, Any] | None:
    runs = list_runs(project_dir, status=status)
    candidates = [run for run in runs if run.get("score") is not None]
    if metric:
        candidates = [run for run in candidates if run.get("metric") == metric]
    if not candidates:
        return None
    minimize = bool(metric and metric in _MINIMIZE)
    return sorted(
        candidates,
        key=lambda run: float(run.get("score") or 0.0),
        reverse=not minimize,
    )[0]


def tag_run(project_dir: str | Path, run_id: str, tags: list[str]) -> dict[str, Any]:
    project = Path(project_dir).resolve()
    run_dir = _run_dir(project, run_id)
    run_meta = _read(run_dir / "run.json", default={}) or {}
    if not isinstance(run_meta, dict):
        run_meta = {}
    clean = list(dict.fromkeys(str(tag).strip() for tag in tags if str(tag).strip()))
    run_meta["tags"] = clean
    write_json(run_dir / "run.json", run_meta)
    refresh_index(project)
    summary = _summary(run_dir, clean)
    if summary is None:
        raise ValueError(f"Run not found: {run_id}")
    return summary
