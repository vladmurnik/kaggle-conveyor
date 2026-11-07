"""Public API contracts: discovery, filters, source catalog and config build."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kconv.api import (
    KConv,
    build_project,
    catalog_stats,
    create_model,
    get_model_info,
    get_pipeline,
    get_source,
    list_models,
    list_pipelines,
    list_sources,
)
from kconv.errors import KconvError, ModelError
from kconv.models import model_available


def test_model_catalog_api_filters() -> None:
    assert len(list_models()) == 51
    assert len(list_models(source="native")) == 6
    assert len(list_models(source="sklearn")) == 44
    assert len(list_models(source="tabpfn")) == 1
    assert len(list_models(task="binary")) == 26
    assert len(list_models(task="multiclass")) == 26
    assert len(list_models(task="regression")) == 30
    assert all(info.family == "linear" for info in list_models(family="linear"))
    tabpfn = get_model_info("tabpfn")
    assert tabpfn.family == "foundation"
    assert tabpfn.source == "tabpfn"
    assert set(tabpfn.tasks) == {"binary", "multiclass", "regression"}

    info = get_model_info("random_forest_classifier")
    assert info.source == "sklearn"
    assert info.estimator_path == "sklearn.ensemble.RandomForestClassifier"
    assert "binary" in info.tasks

    adapter = create_model("ridge", task="regression")
    assert adapter.task == "regression"
    with pytest.raises(ModelError):
        create_model("ridge", task="binary")
    with pytest.raises(KconvError):
        list_models(task="ranking")


def test_pipeline_api() -> None:
    pipelines = list_pipelines()
    assert len(pipelines) == 45
    assert len(list_pipelines(kind="preset")) == 16
    assert len(list_pipelines(kind="pipeline")) == 29
    assert len(list_pipelines(group="ensemble")) == 19
    assert {pipeline.id for pipeline in pipelines} >= {
        "kconv/linear-stack",
        "kconv/sklearn-diverse",
        "nyaggle/oof-ensemble",
        "knnfeat/knn-features",
        "pipeline/pesnik-clean",
        "pipeline/knnfeat-class-distance",
        "pipeline/kozodoi-time-profile",
        "pipeline/kaggle-ieee-cluster-anomaly",
        "pipeline/kconv-unsupervised-profile",
    }

    pipeline = get_pipeline("kconv/linear-stack")
    assert pipeline.kind == "preset"
    assert pipeline.pipeline is None
    assert pipeline.resource_hints == ()
    assert pipeline.defaults["ensemble"]["method"] == "stack"
    assert pipeline.tasks == ("binary", "multiclass", "regression")

    executable = get_pipeline("pipeline/knnfeat-class-distance")
    assert executable.kind == "pipeline"
    assert executable.derived_from == "knnfeat/knn-features"
    assert executable.pipeline is not None
    assert "knn_class_distance" in {
        block["type"] for block in executable.pipeline["blocks"]
    }
    assert "classification-only" in executable.resource_hints

    high_cardinality = get_pipeline("pipeline/kozodoi-high-cardinality")
    assert high_cardinality.kind == "pipeline"
    assert "cpu-heavy" in high_cardinality.resource_hints
    assert "hash_encoding" in {
        block["type"] for block in high_cardinality.pipeline["blocks"]
    }

    with pytest.raises(KconvError):
        list_pipelines(kind="not-a-kind")


def test_source_api_and_local_path(tmp_path: Path, monkeypatch) -> None:
    downloaded = list_sources()
    assert len(downloaded) == 211
    assert len(list_sources(downloaded=None)) == 1173
    assert all(source.downloaded for source in downloaded)

    source = get_source("arnabbiswas1__kaggle_pipeline_tps_aug_22")
    assert source.repo == "arnabbiswas1/kaggle_pipeline_tps_aug_22"
    assert source.license_class == "permissive"
    assert source.executable is True
    assert source.recipe_id == "arnabbiswas1/staged"
    assert source.pipeline_stages

    assert len(list_sources(executable=True)) == 14
    stats = catalog_stats()
    assert stats["total"] == 1173
    assert stats["downloaded"] == 211
    assert stats["executable"] == 14

    # Local path resolution is opt-in and does not leak an absolute path into
    # the bundled catalog.
    toolkit = tmp_path / "kaggle-toolkit"
    local = toolkit / "templates" / (source.local_dir or "")
    local.mkdir(parents=True)
    monkeypatch.setenv("KCONV_TOOLKIT_DIR", str(toolkit))
    assert source.local_path() == local

    with pytest.raises(KconvError):
        get_source("does/not/exist")


def test_build_project_with_all_models(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "id": np.arange(60),
            "x": rng.normal(size=60),
            "target": (rng.random(60) > 0.5).astype(int),
        }
    )
    frame.iloc[:45].to_csv(data / "train.csv", index=False)
    frame.iloc[45:].drop(columns=["target"]).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": frame.iloc[45:]["id"], "target": 0}).to_csv(
        data / "sample_submission.csv",
        index=False,
    )

    project_dir, config_path, config = build_project(
        data,
        target="target",
        task="binary",
        metric="roc_auc",
        models="all",
        name="api-all",
    )

    assert project_dir.is_dir()
    assert config_path.is_file()
    expected_models = 25 + int(model_available("tabpfn"))
    assert len(config.models) == expected_models
    assert all(model.name != "ridge" for model in config.models)

    _, _, mixed = build_project(
        data,
        target="target",
        task="binary",
        metric="roc_auc",
        models="all,lgbm",
        name="api-all-mixed",
    )
    assert len(mixed.models) == expected_models

    _, _, tuned = build_project(
        data,
        target="target",
        task="binary",
        metric="roc_auc",
        models="lgbm,xgb",
        name="api-tuned",
        profile=3,
        device="cpu",
        cpu_threads=2,
        ram_gb=12,
        vram_gb=0,
        model_params={"lgbm": {"n_estimators": 321}},
    )
    assert {model.name for model in tuned.models} == {"lgbm", "xgb"}
    assert next(model for model in tuned.models if model.name == "lgbm").params[
        "n_estimators"
    ] == 321
    assert tuned.runtime is not None
    assert tuned.runtime.device == "cpu"
    assert tuned.runtime.cpu_threads == 2
    assert tuned.runtime.ram_gb == 12
    assert tuned.runtime.vram_gb == 0


def test_kconv_facade_delegates_to_functions() -> None:
    api = KConv()

    assert api.models() == list_models()
    assert api.models(task="regression") == list_models(task="regression")
    assert api.model("lgbm") == get_model_info("lgbm")
    assert api.pipelines() == list_pipelines()
    assert api.pipeline("kconv/linear-stack") == get_pipeline("kconv/linear-stack")
    assert api.sources(limit=3) == list_sources(limit=3)
    assert api.sources(downloaded=None, limit=3) == list_sources(downloaded=None, limit=3)
    assert api.source("arnabbiswas1__kaggle_pipeline_tps_aug_22") == get_source(
        "arnabbiswas1__kaggle_pipeline_tps_aug_22"
    )
    assert api.stats() == catalog_stats()
    assert isinstance(api.plugins(), list)
