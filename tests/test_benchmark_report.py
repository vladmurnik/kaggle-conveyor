"""The checked-in benchmark report stays internally consistent.

The report is a local artifact and is not part of the public repository, so
this contract test only runs when the file is present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPORT = Path(__file__).parent.parent / "reports" / "runtime_benchmark_followup.json"

pytestmark = pytest.mark.skipif(
    not REPORT.is_file(),
    reason="local benchmark report is not part of the public repository",
)


def test_followup_protocol_summary_and_runs_agree() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    profiles = {int(profile) for profile in report["protocol"]["profiles"]}
    summary = {int(profile) for profile in report["summary"]}
    run_profiles = {int(run["profile"]) for run in report["runs"] if run["rep"] > 0}
    assert profiles == summary == run_profiles == {1, 2, 3, 4, 5}
    assert {int(profile) for profile in report["protocol"]["profile_reps"]} == profiles
    assert {int(profile) for profile in report["protocol"]["profile_timeouts"]} == profiles
    for profile in profiles:
        item = report["summary"][str(profile)]
        assert item["successes"] >= 1
        assert item["wall_sec_median"] > 0
        assert item["ram_peak_mib_median"] > 0
