"""The public feature package exports the fold-safe transformers."""

from __future__ import annotations

from kconv.features import (
    AnomalyFeatureTransformer,
    ClusterFeatureTransformer,
    NumericInteractionTransformer,
    TimeSeriesFeatureTransformer,
)


def test_advanced_transformers_are_exported() -> None:
    assert AnomalyFeatureTransformer.__name__ == "AnomalyFeatureTransformer"
    assert ClusterFeatureTransformer.__name__ == "ClusterFeatureTransformer"
    assert NumericInteractionTransformer.__name__ == "NumericInteractionTransformer"
    assert TimeSeriesFeatureTransformer.__name__ == "TimeSeriesFeatureTransformer"
