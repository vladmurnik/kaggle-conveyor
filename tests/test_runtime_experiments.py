"""Runtime profile and experiment catalog contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kconv.api import (
    KConv,
    build_project,
    get_experiment,
    get_runtime_profile,
    list_experiments,
    list_runtime_profiles,
    runtime_models,
)
from kconv.errors import KconvError
from kconv.models import model_available
from kconv.runtime import GpuCapabilities


def _fake_gpu_caps(**overrides: bool) -> GpuCapabilities:
    values = {
        "gpu_available": True,
        "lgbm_gpu": True,
        "xgb_cuda": True,
        "catboost_cuda": True,
    }
    values.update(overrides)
    return GpuCapabilities(**values, details="fake test capabilities")


def _binary_data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "id": np.arange(80),
            "x1": rng.normal(size=80),
            "x2": rng.normal(size=80),
            "target": (rng.random(80) > 0.5).astype(int),
        }
    )
    frame.iloc[:60].to_csv(data / "train.csv", index=False)
    frame.iloc[60:].drop(columns=["target"]).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": frame.iloc[60:]["id"], "target": 0}).to_csv(
        data / "sample_submission.csv",
        index=False,
    )
    return data


def test_runtime_profiles_are_ordered_and_task_aware() -> None:
    profiles = list_runtime_profiles()
    assert [profile.level for profile in profiles] == [1, 2, 3, 4, 5]
    assert profiles[0].ram_gb < profiles[-1].ram_gb
    assert profiles[0].device == "cpu"
    assert profiles[3].vram_gb > 0
    assert profiles[3].device == "auto"
    assert profiles[4].device == "auto"
    assert len(runtime_models(1, "binary")) == 2
    assert len(runtime_models(5, "regression")) == 29 + int(model_available("tabpfn"))
    assert get_runtime_profile(3).strategy == "standard"
    with pytest.raises(KconvError):
        get_runtime_profile(1, device="cuda")
    with pytest.raises(KconvError):
        get_runtime_profile(9)


def test_profile_five_skips_unavailable_optional(monkeypatch) -> None:
    monkeypatch.setattr("kconv.runtime.model_available", lambda name: name != "tabpfn")
    regression = runtime_models(5, "regression")
    assert "tabpfn" not in regression
    assert len(regression) == 29


def test_build_project_with_runtime_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _binary_data(tmp_path)

    project_dir, config_path, config = build_project(
        data,
        target="target",
        task="binary",
        metric="roc_auc",
        profile=1,
        name="runtime-micro",
    )
    assert project_dir.is_dir()
    assert config_path.is_file()
    assert config.strategy == "fast"
    assert len(config.models) == 2
    assert config.models[0].seeds == 1
    assert config.runtime is not None
    assert config.runtime.profile == 1
    assert config.runtime.device == "cpu"
    assert config.models[0].params.get("num_threads") == 1

    monkeypatch.setattr("kconv.runtime.gpu_capabilities", lambda: _fake_gpu_caps())
    _, _, gpu = build_project(
        data,
        target="target",
        task="binary",
        metric="roc_auc",
        profile=4,
        name="runtime-gpu",
    )
    assert gpu.strategy == "standard"
    assert gpu.runtime is not None
    assert gpu.runtime.device == "auto"
    assert gpu.models[0].params.get("device") == "cuda"
    assert any(model.params.get("device") == "cuda" for model in gpu.models)
    assert any(model.params.get("task_type") == "GPU" for model in gpu.models)

    with pytest.raises(KconvError):
        build_project(
            data,
            target="target",
            task="binary",
            metric="roc_auc",
            device="cpu",
            name="runtime-device-without-profile",
        )


def test_runtime_auto_device_falls_back_to_cpu(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = _binary_data(tmp_path)
    monkeypatch.setattr(
        "kconv.runtime.gpu_capabilities",
        lambda: _fake_gpu_caps(
            gpu_available=False,
            lgbm_gpu=False,
            xgb_cuda=False,
            catboost_cuda=False,
        ),
    )
    _, _, config = build_project(
        data,
        target="target",
        task="binary",
        metric="roc_auc",
        profile=4,
        name="runtime-auto-cpu",
    )
    assert config.runtime is not None
    assert config.runtime.device == "auto"
    assert all("device" not in model.params for model in config.models)

    with pytest.raises(KconvError):
        build_project(
            data,
            target="target",
            task="binary",
            metric="roc_auc",
            profile=4,
            device="cuda",
            name="runtime-forced-cuda",
        )


def test_experiment_catalog_and_facade() -> None:
    experiments = list_experiments()
    assert len(experiments) == 25
    assert all(item.category == "experiment" for item in experiments)
    assert len(list_experiments(status="implemented")) >= 8
    assert any(item.id == "llm-feature-proposals" for item in list_experiments(status="idea"))
    assert get_experiment("rank-blend").status == "implemented"
    with pytest.raises(KconvError):
        get_experiment("missing")

    api = KConv()
    assert api.experiments() == experiments
    assert api.experiment("rank-blend").id == "rank-blend"
    assert len(api.profiles()) == 5
    assert api.profile(4).device == "auto"
    assert api.profile_models(1, "binary") == runtime_models(1, "binary")
