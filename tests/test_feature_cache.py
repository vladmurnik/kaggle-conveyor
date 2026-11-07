"""Feature-cache key tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from kconv.config import Config, DataConfig
from kconv.feature_cache import cache_key


def test_cache_key_includes_fold_membership() -> None:
    ctx = SimpleNamespace(
        config=Config(project="cache", data=DataConfig(train="train.csv")),
        data_hash="abc",
    )
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    target = np.array([0, 1, 0])

    first = cache_key(
        ctx,
        "linear",
        0,
        frame,
        target,
        frame,
        None,
        fold_train_idx=np.array([0, 1]),
        fold_valid_idx=np.array([2]),
    )
    second = cache_key(
        ctx,
        "linear",
        0,
        frame,
        target,
        frame,
        None,
        fold_train_idx=np.array([0, 2]),
        fold_valid_idx=np.array([1]),
    )

    assert first != second
