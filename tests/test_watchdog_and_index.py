"""Watchdog recovery, run index and hard-timeout regressions."""

from __future__ import annotations

import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kconv.context import read_json, write_json
from kconv.runs_index import best_run, compare_runs, list_runs, refresh_index, tag_run
from kconv.state import write_latest_pointer
from kconv.timeouts import run_stage_isolated
from kconv.watchdog import inspect_run, recover_stale_run


def _old_iso(seconds: int = 600) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()


def test_stale_run_is_detected_and_recovered(tmp_path: Path) -> None:
    project = tmp_path / "project"
    run_dir = project / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "run.json",
        {
            "run_id": "run-1",
            "project": "demo",
            "status": "running",
            "started_at": _old_iso(),
            "config": {"project": "demo"},
            "warnings": [],
            "failed_models": [],
        },
    )
    write_json(
        run_dir / "heartbeat.json",
        {
            "run_id": "run-1",
            "pid": 999999,
            "hostname": socket.gethostname(),
            "updated_at": _old_iso(),
            "stage": "train",
        },
    )
    write_latest_pointer(project, "run-1", status="running")

    diagnosis = inspect_run(project, "run-1")
    assert diagnosis["stale"] is True
    assert diagnosis["reason"]

    result = recover_stale_run(project, "run-1")
    assert result["recovered"] is True
    meta = read_json(run_dir / "run.json")
    assert meta["status"] == "interrupted"
    assert (run_dir / "failure.json").is_file()
    assert (run_dir / "failure_report.md").is_file()
    heartbeat = read_json(run_dir / "heartbeat.json")
    assert heartbeat["done"] is True


def test_recovery_keeps_a_newer_latest_pointer(tmp_path: Path) -> None:
    project = tmp_path / "project"
    run_dir = project / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "run.json",
        {
            "run_id": "run-1",
            "project": "demo",
            "status": "running",
            "started_at": _old_iso(),
            "config": {"project": "demo"},
            "warnings": [],
            "failed_models": [],
        },
    )
    write_json(
        run_dir / "heartbeat.json",
        {
            "run_id": "run-1",
            "pid": 999999,
            "hostname": socket.gethostname(),
            "updated_at": _old_iso(),
            "stage": "train",
        },
    )
    write_latest_pointer(project, "run-1", status="running")
    _write_run(project, "run-2", 0.9, "b")
    write_latest_pointer(project, "run-2", status="completed")

    result = recover_stale_run(project, "run-1")

    assert result["recovered"] is True
    assert result["latest_pointer_updated"] is False
    assert read_json(project / "latest.json")["run_id"] == "run-2"


def _write_run(project: Path, run_id: str, score: float, recipe: str) -> None:
    run_dir = project / "runs" / run_id
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "project": project.name,
            "status": "completed",
            "started_at": f"2026-01-0{1 if run_id.endswith('1') else 2}T00:00:00+00:00",
            "finished_at": "2026-01-02T00:10:00+00:00",
            "config": {
                "project": project.name,
                "recipe": recipe,
                "strategy": "fast",
                "task": "binary",
                "metric": "roc_auc",
                "models": [{"name": "lgbm", "enabled": True}],
            },
            "warnings": [],
            "failed_models": [],
        },
    )
    write_json(run_dir / "03_blend" / "blend.json", {"oof_score": score, "metric": "roc_auc"})


def test_run_index_compare_best_and_tags(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_run(project, "run-1", 0.8, "a")
    _write_run(project, "run-2", 0.9, "b")

    payload = refresh_index(project)
    assert len(payload["runs"]) == 2
    assert len(list_runs(project)) == 2
    comparison = compare_runs(project, "run-1", "run-2")
    assert comparison["score_delta"] == pytest.approx(0.1)
    assert comparison["config_diff"]
    assert best_run(project, metric="roc_auc")["run_id"] == "run-2"

    tagged = tag_run(project, "run-1", ["baseline", "baseline", "fast"])
    assert tagged["tags"] == ["baseline", "fast"]
    assert next(item for item in list_runs(project) if item["run_id"] == "run-1")["tags"] == ["baseline", "fast"]


def test_stage_timeout_kills_a_slow_worker() -> None:
    with pytest.raises(TimeoutError, match="exceeded"):
        run_stage_isolated("time", "sleep", 5.0, 0.2)
