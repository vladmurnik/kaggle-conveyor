"""Tests for the shared JSON/state helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kconv import context
from kconv.context import validate_run_id
from kconv.errors import KconvError


def test_write_json_retries_a_short_windows_file_lock(monkeypatch, tmp_path: Path) -> None:
    real_replace = os.replace
    calls = 0

    def flaky_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("simulated reader holding state.json")
        real_replace(source, destination)

    monkeypatch.setattr(context.os, "replace", flaky_replace)
    path = context.write_json(tmp_path / "state.json", {"status": "done", "value": 1})

    assert calls == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "done", "value": 1}
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("bad", ["..", "../x", "a/b", "a\\b", "", ".", "a b", "-bad"])
def test_validate_run_id_rejects_path_traversal(bad: str) -> None:
    with pytest.raises(KconvError, match="Invalid run id"):
        validate_run_id(bad)


def test_validate_run_id_accepts_generated_ids() -> None:
    run_id = "20261101T202548667548Z-8d15db8dd8"
    assert validate_run_id(run_id) == run_id
