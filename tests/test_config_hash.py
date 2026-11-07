"""Hash compatibility for optional feature fields."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from kconv.config import Config, DataConfig, FeatureBlockConfig, FeatureConfig
from kconv.context import config_hash


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:10]


def _legacy_payload(config: Config) -> dict[str, Any]:
    """Payload shape used before the optional feature fields were added."""

    payload = config.model_dump(mode="json", exclude_none=True)
    payload.pop("recipe_kind", None)
    features = payload["features"]
    for key in (
        "row_statistics",
        "row_statistics_stats",
        "modulo_features",
        "modulo_moduli",
        "modulo_steps",
        "knn_features",
        "feature_selection",
        "hash_encode",
        "hash_features",
        "interactions",
        "missingness",
        "datetime_cyclical",
        "numeric_interactions",
        "clustering",
        "anomaly",
        "time_series",
        "feature_blocks",
        "target_smoothing",
    ):
        features.pop(key, None)
    return payload


def _config(features: FeatureConfig | None = None) -> Config:
    return Config(
        project="hash",
        data=DataConfig(train="train.csv"),
        features=features or FeatureConfig(),
    )


def test_default_optional_feature_fields_keep_legacy_hash() -> None:
    config = _config()
    assert config_hash(config) == _digest(_legacy_payload(config))


def test_non_default_feature_fields_change_hash() -> None:
    base = _config()
    base_hash = config_hash(base)
    variants = [
        FeatureConfig(row_statistics=True),
        FeatureConfig(modulo_features=True),
        FeatureConfig(knn_features={"enabled": True, "neighbors": 3}),
        FeatureConfig(feature_selection={"enabled": True, "max_features": 5}),
        FeatureConfig(feature_selection={"enabled": True, "method": "model_importance"}),
        FeatureConfig(hash_encode=["cat"], hash_features=32),
        FeatureConfig(interactions=[{"columns": ["cat", "cat2"], "encoding": "frequency"}]),
        FeatureConfig(missingness={"enabled": True}),
        FeatureConfig(datetime_cyclical={"enabled": True}),
        FeatureConfig(numeric_interactions={"enabled": True}),
        FeatureConfig(clustering={"enabled": True}),
        FeatureConfig(anomaly={"enabled": True}),
        FeatureConfig(time_series={"enabled": True, "time_col": "when"}),
        FeatureConfig(feature_blocks=[FeatureBlockConfig(type="base")]),
        FeatureConfig(target_smoothing=20.0),
    ]
    assert all(config_hash(_config(variant)) != base_hash for variant in variants)
