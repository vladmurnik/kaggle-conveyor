"""Curated pipeline groups must stay in sync with the executable catalog."""

from __future__ import annotations

import json

from kconv.api import list_pipelines
from kconv.pipeline_presets import ALL_PIPELINES, PIPELINE_PRESETS, pipeline_presets


def test_pipeline_presets_match_catalog() -> None:
    available = {pipeline.id for pipeline in list_pipelines(kind="pipeline")}
    assert set(ALL_PIPELINES) == available
    assert len(ALL_PIPELINES) == len(available) == 29

    preset_ids = [preset.id for preset in PIPELINE_PRESETS]
    assert len(preset_ids) == len(set(preset_ids))
    assert {preset.kind for preset in PIPELINE_PRESETS} == {"tier", "specialty", "all"}

    for preset in PIPELINE_PRESETS:
        assert preset.pipelines, preset.id
        assert len(preset.pipelines) == len(set(preset.pipelines)), preset.id
        assert set(preset.pipelines) <= available, preset.id

    by_id = {preset.id: preset for preset in PIPELINE_PRESETS}
    assert set(by_id["starter"].pipelines) <= set(by_id["fast"].pipelines)
    assert set(by_id["fast"].pipelines) <= set(by_id["balanced"].pipelines)
    assert set(by_id["all"].pipelines) == available


def test_pipeline_presets_payload_is_json_safe() -> None:
    payload = pipeline_presets()
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert "Быстрый старт" in encoded
    assert all(set(item) == {"id", "title", "description", "kind", "pipelines"} for item in payload)
