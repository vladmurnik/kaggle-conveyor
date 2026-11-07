"""Dataset sampling percentage wiring in project configs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kconv.config import load_config
from kconv.errors import ConfigError
from kconv.project import build_config


def _dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    n = 120
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "num": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], n),
            "target": (rng.random(n) > 0.5).astype(int),
        }
    )
    frame.iloc[:96].to_csv(root / "train.csv", index=False)
    frame.iloc[96:].drop(columns=["target"]).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": frame.iloc[96:]["id"], "target": 0}).to_csv(
        root / "sample_submission.csv", index=False
    )
    return root


def test_build_config_persists_sample_pct(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "data")

    _, config_path, cfg = build_config(
        data,
        target="target",
        recipe_id="pesnik/baseline",
        models="lgbm",
        name="sample-pct",
        sample_pct=37,
    )

    assert cfg.data.sample_pct == 37
    assert load_config(config_path).data.sample_pct == 37


@pytest.mark.parametrize("value", [0, 101])
def test_build_config_rejects_out_of_range_sample_pct(tmp_path: Path, value: int) -> None:
    data = _dataset(tmp_path / "data")

    with pytest.raises(ConfigError, match="--sample-pct"):
        build_config(
            data,
            target="target",
            recipe_id="pesnik/baseline",
            models="lgbm",
            name=f"sample-pct-{value}",
            sample_pct=value,
        )
