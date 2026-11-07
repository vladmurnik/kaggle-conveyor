"""Explicit plugin discovery for external model adapters.

A plugin is an installed Python distribution exposing an entry point in the
``kconv.plugins`` group.  The entry point may point to a module (import side
effects register models) or to a zero-argument callable that performs the
registration.  Nothing is imported until :func:`load_plugins` is called.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any

PLUGIN_GROUP = "kconv.plugins"


@dataclass(frozen=True)
class PluginInfo:
    name: str
    value: str
    group: str
    loaded: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entry_points() -> list[metadata.EntryPoint]:
    try:
        entries = metadata.entry_points()
    except Exception:  # pragma: no cover - metadata backend failure
        return []
    if hasattr(entries, "select"):
        return list(entries.select(group=PLUGIN_GROUP))
    return list(entries.get(PLUGIN_GROUP, []))  # type: ignore[union-attr]


def list_plugins() -> list[PluginInfo]:
    """List installed plugin entry points without importing them."""
    return [
        PluginInfo(name=entry.name, value=entry.value, group=PLUGIN_GROUP)
        for entry in _entry_points()
    ]


def load_plugins() -> list[PluginInfo]:
    """Import/call every installed plugin and report per-plugin failures.

    This is intentionally explicit: importing kconv, listing models, or running
    a pipeline never executes third-party plugin code.
    """
    result: list[PluginInfo] = []
    for entry in _entry_points():
        info = PluginInfo(name=entry.name, value=entry.value, group=PLUGIN_GROUP)
        try:
            target = entry.load()
            if callable(target):
                target()
        except Exception as exc:  # noqa: BLE001 - isolate a broken plugin
            info = PluginInfo(
                name=entry.name,
                value=entry.value,
                group=PLUGIN_GROUP,
                loaded=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            info = PluginInfo(name=entry.name, value=entry.value, group=PLUGIN_GROUP, loaded=True)
        result.append(info)
    return result


__all__ = ["PLUGIN_GROUP", "PluginInfo", "list_plugins", "load_plugins"]
