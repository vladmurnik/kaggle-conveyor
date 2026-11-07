"""Integrity and behavior tests for the bundled source catalog."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kconv.errors import KconvError
from kconv.sources import SourceInfo, catalog_stats, get_source, list_sources

VALID_LICENSE_CLASSES = {"permissive", "copyleft", "unlicensed", "restricted", "unknown"}


def test_catalog_integrity() -> None:
    sources = list_sources(downloaded=None)
    assert len(sources) == 1173
    assert len({source.id for source in sources}) == len(sources)
    assert len({source.repo for source in sources}) == len(sources)

    downloaded = [source for source in sources if source.downloaded]
    executable = [source for source in sources if source.executable]
    pipeline_sources = [source for source in sources if source.pipeline_stages]
    assert len(downloaded) == 211
    assert len(executable) == 14
    assert len(pipeline_sources) == 20

    assert all(source.license_class in VALID_LICENSE_CLASSES for source in sources)
    assert all(source.reason for source in sources)
    assert all(source.local_dir for source in downloaded)
    assert all(source.recipe_id for source in executable)
    assert all(source.downloaded for source in executable)

    stats = catalog_stats()
    assert stats["total"] == 1173
    assert stats["downloaded"] == 211
    assert stats["not_downloaded"] == 962
    assert stats["executable"] == 14
    assert stats["recipe_backed_sources"] == 14
    assert stats["pipeline_sources"] == 20
    assert stats["by_license_class"] == {
        "unlicensed": 88,
        "permissive": 109,
        "copyleft": 6,
        "unknown": 969,
        "restricted": 1,
    }


def test_source_filters() -> None:
    assert len(list_sources()) == 211
    assert len(list_sources(downloaded=None)) == 1173
    assert len(list_sources(executable=True)) == 14
    assert len(list_sources(has_pipeline=True)) == 20
    assert len(list_sources(category="pipeline")) == 18
    assert len(list_sources(license_class="permissive")) == 109

    matches = list_sources(query="arnabbiswas1")
    assert len(matches) == 1
    assert matches[0].repo == "arnabbiswas1/kaggle_pipeline_tps_aug_22"

    limited = list_sources(limit=5)
    assert len(limited) == 5
    assert list_sources(limit=0) == []

    with pytest.raises(KconvError):
        list_sources(limit=-1)


def test_get_source_by_id_repo_and_directory() -> None:
    by_id = get_source("arnabbiswas1__kaggle_pipeline_tps_aug_22")
    by_repo = get_source("arnabbiswas1/kaggle_pipeline_tps_aug_22")
    assert by_id == by_repo
    assert by_id.recipe_id == "arnabbiswas1/staged"
    assert by_id.executable is True
    assert by_id.license_class == "permissive"

    assert get_source((by_id.local_dir or "").upper()) == by_id

    with pytest.raises(KconvError):
        get_source("does/not/exist")
    with pytest.raises(KconvError):
        get_source("   ")


def test_local_path_resolution(tmp_path: Path, monkeypatch) -> None:
    source = get_source("arnabbiswas1__kaggle_pipeline_tps_aug_22")
    toolkit = tmp_path / "kaggle-toolkit"
    local = toolkit / "templates" / (source.local_dir or "")
    local.mkdir(parents=True)

    monkeypatch.setenv("KCONV_TOOLKIT_DIR", str(toolkit))
    assert source.local_path() == local
    assert source.local_path(toolkit) == local

    monkeypatch.delenv("KCONV_TOOLKIT_DIR")
    shutil.rmtree(toolkit)
    monkeypatch.chdir(tmp_path)
    assert source.local_path() is None

    not_downloaded = next(item for item in list_sources(downloaded=None) if not item.downloaded)
    assert not_downloaded.local_path(toolkit) is None


def test_source_info_is_immutable_and_serializable() -> None:
    source = get_source("arnabbiswas1__kaggle_pipeline_tps_aug_22")
    with pytest.raises(AttributeError):
        source.repo = "changed"  # type: ignore[misc]

    payload = source.to_dict()
    assert payload["repo"] == source.repo
    assert payload["pipeline_stages"] == list(source.pipeline_stages)


def _source(**overrides) -> SourceInfo:
    values = {
        "id": "owner__repo",
        "repo": "owner/repo",
        "url": "https://github.com/owner/repo",
        "downloaded": True,
        "category": "pipeline",
        "license": "MIT License",
        "license_class": "permissive",
        "files": 1,
        "bytes": 10,
        "local_dir": "owner__repo",
        "description": "",
        "stars": 1,
        "language": "Python",
        "updated": "2026-01-01T00:00:00Z",
        "recipe_id": None,
        "executable": False,
        "pipeline_stages": (),
        "partial_stages": (),
        "reason": "test",
    }
    values.update(overrides)
    return SourceInfo(**values)


def test_ambiguous_source_is_rejected(monkeypatch) -> None:
    first = _source(id="a", repo="owner/a", local_dir="shared")
    second = _source(id="b", repo="owner/b", local_dir="shared")
    monkeypatch.setattr("kconv.sources._sources", lambda: (first, second))

    with pytest.raises(KconvError, match="ambiguous"):
        get_source("shared")
