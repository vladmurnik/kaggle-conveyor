"""Recipe catalog audit: normalized signatures, variants and duplicates.

Recipes are configuration presets unless they declare ``kind: pipeline``.  A
preset may intentionally duplicate another preset (a historical variant), but
two executable pipelines must not silently resolve to the same runnable
configuration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from kconv.errors import RecipeError
from kconv.recipes import Recipe, list_recipes


def normalized_signature(recipe: Recipe) -> dict[str, Any]:
    """Return the behavior-relevant, JSON-safe recipe signature."""
    payload: dict[str, Any] = recipe.defaults.model_dump(mode="json")
    payload["kind"] = recipe.kind
    if recipe.pipeline is not None:
        payload["pipeline"] = {
            "blocks": [
                {
                    "type": block.type,
                    "params": block.params,
                    "enabled": block.enabled,
                }
                for block in recipe.pipeline.blocks
            ]
        }
    return payload


def signature_key(recipe: Recipe) -> str:
    """Canonical key used to group semantically identical recipes."""
    return json.dumps(normalized_signature(recipe), sort_keys=True, separators=(",", ":"), default=str)


def signature_digest(recipe: Recipe) -> str:
    """Short stable digest for display in CLI/UI."""
    return hashlib.sha256(signature_key(recipe).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class AuditEntry:
    """One catalog entry and its behavior signature."""

    id: str
    kind: str
    group: str
    derived_from: str | None
    signature: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "group": self.group,
            "derived_from": self.derived_from,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class DuplicateGroup:
    """Recipes sharing one normalized signature."""

    signature: dict[str, Any]
    recipe_ids: tuple[str, ...]
    kinds: tuple[str, ...]

    @property
    def has_pipeline(self) -> bool:
        return "pipeline" in self.kinds

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_ids": list(self.recipe_ids),
            "kinds": list(self.kinds),
            "has_pipeline": self.has_pipeline,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class RecipeAudit:
    """Complete audit result."""

    entries: tuple[AuditEntry, ...] = field(default_factory=tuple)
    duplicate_groups: tuple[DuplicateGroup, ...] = field(default_factory=tuple)
    kind_counts: dict[str, int] = field(default_factory=dict)

    @property
    def duplicate_ids(self) -> set[str]:
        return {recipe_id for group in self.duplicate_groups for recipe_id in group.recipe_ids}

    @property
    def has_pipeline_duplicates(self) -> bool:
        return any(group.has_pipeline for group in self.duplicate_groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind_counts": dict(self.kind_counts),
            "recipes": [entry.to_dict() for entry in self.entries],
            "duplicate_groups": [group.to_dict() for group in self.duplicate_groups],
            "has_pipeline_duplicates": self.has_pipeline_duplicates,
        }


def audit_recipes(recipes: list[Recipe] | None = None, *, strict: bool = False) -> RecipeAudit:
    """Audit the catalog; optionally reject duplicate executable pipelines."""
    items = list(recipes) if recipes is not None else list_recipes()
    entries: list[AuditEntry] = []
    grouped: dict[str, list[Recipe]] = {}
    kind_counts: dict[str, int] = {}
    for recipe in sorted(items, key=lambda item: item.id):
        entries.append(
            AuditEntry(
                id=recipe.id,
                kind=recipe.kind,
                group=recipe.group,
                derived_from=recipe.derived_from,
                signature=normalized_signature(recipe),
            )
        )
        grouped.setdefault(signature_key(recipe), []).append(recipe)
        kind_counts[recipe.kind] = kind_counts.get(recipe.kind, 0) + 1

    duplicates: list[DuplicateGroup] = []
    for _signature, members in sorted(grouped.items(), key=lambda item: item[0]):
        if len(members) < 2:
            continue
        duplicates.append(
            DuplicateGroup(
                signature=normalized_signature(members[0]),
                recipe_ids=tuple(sorted(member.id for member in members)),
                kinds=tuple(sorted({member.kind for member in members})),
            )
        )
    result = RecipeAudit(
        entries=tuple(entries),
        duplicate_groups=tuple(duplicates),
        kind_counts=dict(sorted(kind_counts.items())),
    )
    if strict and result.has_pipeline_duplicates:
        groups = [
            ", ".join(group.recipe_ids)
            for group in result.duplicate_groups
            if group.has_pipeline
        ]
        raise RecipeError(
            "Duplicate executable pipeline signatures: " + "; ".join(groups) + ".",
            "Give each pipeline a distinct block/feature/model configuration, or mark it as a preset.",
        )
    return result
