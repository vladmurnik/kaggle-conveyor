"""Read-only access to the bundled kaggle-toolkit source catalog.

The catalog contains metadata only.  It deliberately does not import, execute
or vendor code from the indexed repositories.  A source is ``executable`` only
when kconv ships an original recipe that reimplements its methodology.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from kconv.errors import KconvError

CATALOG_SCHEMA_VERSION = 1
_CATALOG_PACKAGE = "kconv.assets"
_CATALOG_FILENAME = "catalog.json"


@dataclass(frozen=True)
class SourceInfo:
    """One repository from the kaggle-toolkit index."""

    id: str
    repo: str
    url: str
    downloaded: bool
    category: str | None
    license: str | None
    license_class: str
    files: int | None
    bytes: int | None
    local_dir: str | None
    description: str
    stars: int | None
    language: str | None
    updated: str | None
    recipe_id: str | None
    executable: bool
    pipeline_stages: tuple[str, ...]
    partial_stages: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pipeline_stages"] = list(self.pipeline_stages)
        payload["partial_stages"] = list(self.partial_stages)
        return payload

    def local_path(self, toolkit_dir: str | Path | None = None) -> Path | None:
        """Return the local template directory when the toolkit is available."""
        if not self.downloaded or not self.local_dir:
            return None
        root = _toolkit_root(toolkit_dir)
        if root is None:
            return None
        candidate = root / "templates" / self.local_dir
        return candidate if candidate.is_dir() else None


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    try:
        from importlib import resources

        text = resources.files(_CATALOG_PACKAGE).joinpath(_CATALOG_FILENAME).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:  # pragma: no cover - packaging failure
        raise KconvError(
            f"Bundled source catalog is unavailable: {exc}",
            "Reinstall kconv or run scripts/build_catalog.py.",
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - generated file corruption
        raise KconvError(f"Bundled source catalog is invalid JSON: {exc}", "Rebuild it with scripts/build_catalog.py.") from exc
    if payload.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise KconvError(
            f"Unsupported source catalog schema: {payload.get('schema_version')!r}",
            f"Expected schema_version={CATALOG_SCHEMA_VERSION}.",
        )
    if not isinstance(payload.get("sources"), list):
        raise KconvError("Bundled source catalog has no sources list.", "Rebuild the catalog.")
    return payload


def _source_from_dict(raw: dict[str, Any]) -> SourceInfo:
    return SourceInfo(
        id=str(raw["id"]),
        repo=str(raw["repo"]),
        url=str(raw.get("url", "")),
        downloaded=bool(raw.get("downloaded", False)),
        category=raw.get("category"),
        license=raw.get("license"),
        license_class=str(raw.get("license_class", "unknown")),
        files=raw.get("files"),
        bytes=raw.get("bytes"),
        local_dir=raw.get("local_dir"),
        description=str(raw.get("description", "")),
        stars=raw.get("stars"),
        language=raw.get("language"),
        updated=raw.get("updated"),
        recipe_id=raw.get("recipe_id"),
        executable=bool(raw.get("executable", False)),
        pipeline_stages=tuple(raw.get("pipeline_stages", [])),
        partial_stages=tuple(raw.get("partial_stages", [])),
        reason=str(raw.get("reason", "")),
    )


@lru_cache(maxsize=1)
def _sources() -> tuple[SourceInfo, ...]:
    """Parse the generated JSON once; SourceInfo is immutable."""
    return tuple(_source_from_dict(raw) for raw in _catalog()["sources"])


def list_sources(
    *,
    downloaded: bool | None = True,
    category: str | None = None,
    license_class: str | None = None,
    executable: bool | None = None,
    has_pipeline: bool | None = None,
    query: str | None = None,
    limit: int | None = None,
) -> list[SourceInfo]:
    """List indexed repositories.

    ``downloaded=True`` (the default) returns the 211 locally available
    templates.  Pass ``downloaded=None`` to include all discovered sources.
    """
    if limit is not None and limit < 0:
        raise KconvError("limit must be non-negative.", "Pass a positive limit or omit it.")
    needle = query.casefold().strip() if query else None
    result: list[SourceInfo] = []
    for item in _sources():
        if downloaded is not None and item.downloaded != downloaded:
            continue
        if category is not None and (item.category or "").casefold() != category.casefold():
            continue
        if license_class is not None and item.license_class.casefold() != license_class.casefold():
            continue
        if executable is not None and item.executable is not executable:
            continue
        if has_pipeline is not None and bool(item.pipeline_stages) != has_pipeline:
            continue
        if needle:
            haystack = " ".join(
                filter(
                    None,
                    (
                        item.id,
                        item.repo,
                        item.category,
                        item.license,
                        item.description,
                        item.recipe_id,
                    ),
                )
            ).casefold()
            if needle not in haystack:
                continue
        result.append(item)
    result.sort(key=lambda item: (not item.downloaded, item.repo.casefold()))
    return result[:limit] if limit is not None else result


def get_source(identifier: str) -> SourceInfo:
    """Find a source by catalog id, repo name, or local directory name."""
    needle = identifier.casefold().strip()
    if not needle:
        raise KconvError("Source id must not be empty.", "Run 'kconv sources' to list ids.")
    matches: list[SourceInfo] = []
    for item in _sources():
        keys = {item.id.casefold(), item.repo.casefold()}
        if item.local_dir:
            keys.add(item.local_dir.casefold())
        if needle in keys:
            matches.append(item)
    if not matches:
        raise KconvError(
            f"Unknown source '{identifier}'.",
            "Run 'kconv sources --all' to search the complete index.",
        )
    if len(matches) > 1:
        raise KconvError(
            f"Source id '{identifier}' is ambiguous.",
            "Use the repository name (owner/name) or the catalog id.",
        )
    return matches[0]


def catalog_stats() -> dict[str, Any]:
    """Return aggregate counts used by CLI/API/doctor."""
    sources = list_sources(downloaded=None)
    downloaded = [item for item in sources if item.downloaded]
    return {
        "total": len(sources),
        "downloaded": len(downloaded),
        "not_downloaded": len(sources) - len(downloaded),
        "executable": sum(item.executable for item in sources),
        "recipe_backed_sources": len({item.recipe_id for item in sources if item.recipe_id}),
        "pipeline_sources": sum(bool(item.pipeline_stages) for item in sources),
        "by_license_class": dict(Counter(item.license_class for item in sources)),
        "by_downloaded_category": dict(Counter(item.category or "unknown" for item in downloaded)),
    }


def _toolkit_root(toolkit_dir: str | Path | None = None) -> Path | None:
    if toolkit_dir is not None:
        root = Path(toolkit_dir).expanduser().resolve()
        return root if root.is_dir() else None
    env = os.environ.get("KCONV_TOOLKIT_DIR")
    if env:
        root = Path(env).expanduser().resolve()
        if root.is_dir():
            return root
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / "kaggle-toolkit"
        if candidate.is_dir():
            return candidate
    return None


__all__ = [
    "SourceInfo",
    "catalog_stats",
    "get_source",
    "list_sources",
]
