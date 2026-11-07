"""Helpers for keeping class labels usable in JSON and submissions."""

from __future__ import annotations

from typing import Any

import numpy as np


def native_labels(values: np.ndarray) -> list[Any]:
    """Convert numpy scalar labels to plain Python values.

    ``LabelEncoder.classes_`` can contain numpy integers, floats, booleans or
    strings. JSON serialization handles Python values, while numpy integer
    scalars would otherwise be coerced to strings by the generic JSON encoder.
    """
    return [value.item() if isinstance(value, np.generic) else value for value in values]
