"""Helpers for working with partially validated out-of-fold predictions.

Holdout and time-series validation intentionally leave some training rows
without an out-of-fold prediction.  Treating those rows as zeroes silently
corrupts every downstream score and blend, so all OOF consumers must use the
same validation mask.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from kconv.contracts import Fold
from kconv.errors import DataError


def validation_mask(folds: Iterable[Fold], n_rows: int) -> np.ndarray:
    """Return a boolean mask selecting rows that appear in a validation fold."""
    mask = np.zeros(int(n_rows), dtype=bool)
    for fold in folds:
        valid = np.asarray(fold.valid_idx, dtype=np.int64)
        if valid.size == 0:
            continue
        if valid.min() < 0 or valid.max() >= int(n_rows):
            raise DataError(
                "Fold validation indices are outside the training data.",
                "Rerun the validate stage; this artifact is corrupted.",
            )
        mask[valid] = True
    return mask


def validated_indices(folds: Iterable[Fold], n_rows: int) -> np.ndarray:
    """Return positional indices that have an out-of-fold prediction."""
    return np.flatnonzero(validation_mask(folds, n_rows))


def masked_slice(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Slice the first axis of ``values`` with ``mask`` and validate its length."""
    array = np.asarray(values)
    mask = np.asarray(mask, dtype=bool)
    if array.shape[0] != mask.shape[0]:
        raise DataError(
            "OOF prediction length does not match the training data.",
            "Rerun the train stage; this artifact is corrupted.",
        )
    return array[mask]
