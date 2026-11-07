"""Contract for the optional local TabPFN v2 benchmark evidence.

The JSON/Markdown reports are local artifacts and are not part of the public
repository. The test runs only when both files are present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JSON_REPORT = ROOT / "reports" / "tabpfn_benchmark.json"
MARKDOWN_REPORT = ROOT / "reports" / "tabpfn_benchmark.md"

pytestmark = pytest.mark.skipif(
    not (JSON_REPORT.is_file() and MARKDOWN_REPORT.is_file()),
    reason="local TabPFN benchmark reports are not part of the public repository",
)


def test_tabpfn_benchmark_report_contract() -> None:
    payload = json.loads(JSON_REPORT.read_text(encoding="utf-8"))

    assert payload["environment"]["tabpfn"]
    assert payload["environment"]["gpu"]["available"] is True
    assert payload["protocol"]["tabpfn_n_estimators"] == 8

    ids = {item["id"] for item in payload["benchmarks"]}
    assert {"uci_breast_cancer", "uci_diabetes"} <= ids
    for item in payload["benchmarks"]:
        models = {row["model"]: row for row in item["models"]}
        assert "tabpfn" in models
        assert {"lgbm", "xgb", "catboost", "histgb"} <= set(models)
        assert models["tabpfn"]["peak_vram_reserved_gb"] < 3.5
        assert item["best_by_mean"] == "tabpfn"

    stress = payload["stress"][0]
    assert stress["train_rows"] <= 8_000
    assert stress["features"] == 100
    assert stress["peak_vram_reserved_gb"] < 3.5
    assert stress["peak_rss_mb"] is not None
    assert stress["peak_rss_mb"] < 7 * 1024

    markdown = MARKDOWN_REPORT.read_text(encoding="utf-8")
    assert "TabPFN v2" in markdown
    assert "synthetic_8000x100" in markdown
