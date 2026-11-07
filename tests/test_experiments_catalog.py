"""Generated experiment catalog: loader, sync, validation and packaging."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import pytest

from kconv.errors import KconvError
from kconv.experiments import (
    catalog_payload,
    get_experiment,
    list_experiments,
    load_catalog,
    validate_catalog,
)


def test_packaged_catalog_is_present_and_loadable() -> None:
    asset = files("kconv.assets").joinpath("experiments.json")
    assert asset.is_file()
    payload = json.loads(asset.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["experiments"]) == 25
    loaded = load_catalog()
    assert len(loaded) == 25
    assert loaded[0].id == "target-encoding-oof"
    assert loaded[-1].id == "label-free-structure-audit"


def test_catalog_json_matches_the_python_source_of_truth() -> None:
    expected = catalog_payload()
    actual = json.loads(files("kconv.assets").joinpath("experiments.json").read_text(encoding="utf-8"))
    assert actual == expected
    assert validate_catalog(actual) == tuple(list_experiments())


def test_new_triage_entries_are_present_and_statuses_are_tracked() -> None:
    ids = {item.id for item in list_experiments()}
    expected = {
        "knn-class-distance-features",
        "gaussian-rank-normalization",
        "modulo-decomposition",
        "dense-crossfit-target-encoding",
        "risk-adjusted-target-encoding",
        "transductive-frequency-features",
        "offset-boosting",
        "tabular-foundation-baseline",
        "conditional-permutation-importance",
        "label-free-structure-audit",
    }
    assert expected <= ids
    assert get_experiment("gaussian-rank-normalization").status == "prototype"
    assert get_experiment("modulo-decomposition").status == "implemented"
    assert get_experiment("row-statistics").status == "implemented"
    assert get_experiment("knn-class-distance-features").status == "implemented"


def test_catalog_loader_rejects_stale_schema_and_duplicate_ids(tmp_path: Path) -> None:
    payload = catalog_payload()
    payload["schema_version"] = 99
    path = tmp_path / "experiments.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KconvError, match="schema_version"):
        load_catalog(path)

    payload = catalog_payload()
    payload["experiments"].append(payload["experiments"][0])
    with pytest.raises(KconvError, match="Duplicate experiment id"):
        validate_catalog(payload)


def test_catalog_loader_rejects_unknown_or_missing_fields() -> None:
    payload = catalog_payload()
    payload["experiments"][0]["surprise"] = True
    with pytest.raises(KconvError, match="Unknown experiment field"):
        validate_catalog(payload)

    payload = catalog_payload()
    del payload["experiments"][0]["risk"]
    with pytest.raises(KconvError, match="Missing experiment field"):
        validate_catalog(payload)

    payload = catalog_payload()
    payload["experiments"][0]["status"] = "maybe"
    with pytest.raises(KconvError, match="Unknown experiment status"):
        validate_catalog(payload)


def test_catalog_loader_rejects_wrong_semantic_types() -> None:
    payload = catalog_payload()
    payload["experiments"][0]["risk"] = None
    with pytest.raises(KconvError, match="risk"):
        validate_catalog(payload)

    payload = catalog_payload()
    payload["experiments"][0]["cost"] = 123
    with pytest.raises(KconvError, match="cost"):
        validate_catalog(payload)

    payload = catalog_payload()
    payload["experiments"][0]["task_types"] = []
    with pytest.raises(KconvError, match="task_types"):
        validate_catalog(payload)

    payload = catalog_payload()
    payload["experiments"][0]["stages"] = []
    with pytest.raises(KconvError, match="stages"):
        validate_catalog(payload)

    payload = catalog_payload()
    payload["experiments"][0]["recipe_id"] = ""
    with pytest.raises(KconvError, match="recipe_id"):
        validate_catalog(payload)


def test_catalog_loader_reports_invalid_json_and_missing_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(KconvError, match="not valid JSON"):
        load_catalog(bad)

    with pytest.raises(KconvError, match="not found"):
        load_catalog(tmp_path / "missing.json")
