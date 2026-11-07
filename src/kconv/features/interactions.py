"""Helpers for pairwise categorical interaction features."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations


def interaction_pairs(columns: Iterable[str], max_pairs: int = 4) -> list[tuple[str, str]]:
    """Return deterministic pairwise combinations, capped at ``max_pairs``."""

    unique = list(dict.fromkeys(columns))
    return list(combinations(unique, 2))[:max_pairs]


def interaction_name(columns: tuple[str, str] | list[str], encoding: str) -> str:
    """Stable output name for one interaction encoding."""

    return f"{'__x__'.join(columns)}__{encoding}"
