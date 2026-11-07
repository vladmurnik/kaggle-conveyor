"""Distribution-level ensemble helpers.

The stage implementation lives in :mod:`kconv.stages.blend`; these helpers are
kept small and pure so they can be reused by notebooks or future stackers.
"""

from __future__ import annotations

import numpy as np


def weighted_average(predictions: list[np.ndarray], weights: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.sum() <= 0:
        values = np.full(len(predictions), 1.0 / len(predictions))
    values = values / values.sum()
    return np.tensordot(values, np.stack(predictions, axis=0), axes=(0, 0))


def mean_average(predictions: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(predictions, axis=0), axis=0)
