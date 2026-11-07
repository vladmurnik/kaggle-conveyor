"""Tiny explicit registries.

No plugin magic in v1: modules register themselves at import time, and the
core imports the built-in modules once.  Custom recipes can still choose
from the registered names.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from kconv.errors import ModelError

T = TypeVar("T")

_MODELS: dict[str, Callable[..., Any]] = {}
_METRICS: dict[str, Any] = {}
_SPLITTERS: dict[str, Any] = {}
_ENSEMBLES: dict[str, Any] = {}


def register_model(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def deco(factory: Callable[..., T]) -> Callable[..., T]:
        _MODELS[name] = factory
        return factory

    return deco


def register_metric(name: str) -> Callable[[T], T]:
    def deco(obj: T) -> T:
        _METRICS[name] = obj
        return obj

    return deco


def register_splitter(name: str) -> Callable[[T], T]:
    def deco(obj: T) -> T:
        _SPLITTERS[name] = obj
        return obj

    return deco


def register_ensemble(name: str) -> Callable[[T], T]:
    def deco(obj: T) -> T:
        _ENSEMBLES[name] = obj
        return obj

    return deco


def get_model(name: str) -> Callable[..., Any]:
    try:
        return _MODELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_MODELS)) or "none"
        raise ModelError(f"Unknown model '{name}'.", f"Available models: {available}") from exc


def model_names() -> list[str]:
    return sorted(_MODELS)


def metric_names() -> list[str]:
    return sorted(_METRICS)


def get_metric(name: str) -> Any:
    try:
        return _METRICS[name]
    except KeyError as exc:
        raise ModelError(f"Unknown metric '{name}'.", f"Available metrics: {metric_names()}") from exc


def get_splitter(name: str) -> Any:
    return _SPLITTERS[name]


def splitter_names() -> list[str]:
    return sorted(_SPLITTERS)


def get_ensemble(name: str) -> Any:
    return _ENSEMBLES[name]


def ensemble_names() -> list[str]:
    return sorted(_ENSEMBLES)
