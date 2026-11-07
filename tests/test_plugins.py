"""Plugin discovery is explicit and failure-isolated."""

from __future__ import annotations

from typing import Any

from kconv.plugins import list_plugins, load_plugins


class _FakeEntryPoint:
    def __init__(self, name: str, value: str, loader, error: Exception | None = None) -> None:
        self.name = name
        self.value = value
        self._loader = loader
        self._error = error
        self.load_calls = 0

    def load(self) -> Any:
        self.load_calls += 1
        if self._error is not None:
            raise self._error
        return self._loader


class _FakeEntryPoints:
    def __init__(self, entries: list[_FakeEntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_FakeEntryPoint]:
        assert group == "kconv.plugins"
        return self._entries


def test_plugin_discovery_does_not_load(monkeypatch) -> None:
    calls: list[str] = []

    def loader() -> None:
        calls.append("loaded")

    entry = _FakeEntryPoint("demo", "demo.plugin", loader)
    monkeypatch.setattr("kconv.plugins.metadata.entry_points", lambda: _FakeEntryPoints([entry]))

    listed = list_plugins()
    assert [item.name for item in listed] == ["demo"]
    assert listed[0].loaded is False
    assert entry.load_calls == 0
    assert calls == []


def test_load_plugins_isolates_failures(monkeypatch) -> None:
    calls: list[str] = []

    def good_loader() -> None:
        calls.append("good")

    good = _FakeEntryPoint("good", "good.plugin", good_loader)
    bad = _FakeEntryPoint("bad", "bad.plugin", lambda: None, error=RuntimeError("boom"))
    monkeypatch.setattr("kconv.plugins.metadata.entry_points", lambda: _FakeEntryPoints([good, bad]))

    loaded = load_plugins()
    assert [item.loaded for item in loaded] == [True, False]
    assert loaded[0].error is None
    assert loaded[1].error == "RuntimeError: boom"
    assert calls == ["good"]
