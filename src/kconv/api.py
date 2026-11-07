"""Public, stable-ish Python API for kconv.

The CLI is a thin layer over the same functions.  Use this module when you
want to discover models/pipelines/sources, build a config, or run the
dataset-first pipeline from Python:

    from kconv.api import list_models, list_pipelines, run_dataset

The API never imports external repositories.  Source catalog entries are
metadata; only kconv's own adapters and recipes are executable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kconv.config import Config, load_config
from kconv.contracts import ModelAdapter
from kconv.errors import KconvError, ModelError
from kconv.experiments import ExperimentInfo, get_experiment
from kconv.experiments import list_experiments as _list_experiments
from kconv.models import create_model as _create_model
from kconv.models.catalog import SKLEARN_SPECS
from kconv.plugins import PluginInfo, list_plugins, load_plugins
from kconv.project import build_config
from kconv.recipes import Recipe, list_recipes, load_recipe
from kconv.registry import model_names, register_model
from kconv.runtime import RuntimeProfile, get_profile, list_profiles, select_models
from kconv.sources import SourceInfo, catalog_stats, get_source, list_sources

_TASKS = {"binary", "multiclass", "regression"}
_SOURCES = {"native", "sklearn", "tabpfn"}


@dataclass(frozen=True)
class ModelInfo:
    """One registered model adapter."""

    id: str
    family: str
    tasks: tuple[str, ...]
    source: str
    estimator_path: str | None
    description: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = list(self.tasks)
        return payload


@dataclass(frozen=True)
class PipelineInfo:
    """One recipe entry: an executable pipeline or a config preset."""

    id: str
    title: str
    group: str
    author: str
    source: str
    license: str
    tasks: tuple[str, ...]
    models: tuple[str, ...]
    best_for: str
    attribution: str
    defaults: dict[str, Any]
    kind: str = "preset"
    derived_from: str | None = None
    pipeline: dict[str, Any] | None = None
    resource_hints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = list(self.tasks)
        payload["models"] = list(self.models)
        payload["resource_hints"] = list(self.resource_hints)
        return payload


def list_models(
    *,
    task: str | None = None,
    family: str | None = None,
    source: str | None = None,
) -> list[ModelInfo]:
    """List registered models with optional task/family/source filters."""
    if task is not None and task not in _TASKS:
        raise KconvError(f"Unknown task '{task}'.", f"Use one of: {', '.join(sorted(_TASKS))}.")
    if source is not None and source not in _SOURCES:
        raise KconvError(f"Unknown model source '{source}'.", f"Use one of: {', '.join(sorted(_SOURCES))}.")
    result: list[ModelInfo] = []
    for name in model_names():
        adapter = _create_model(name)
        if task is not None and task not in adapter.task_types:
            continue
        if family is not None and adapter.family != family:
            continue
        model_source = "tabpfn" if name == "tabpfn" else ("sklearn" if name in SKLEARN_SPECS else "native")
        if source is not None and model_source != source:
            continue
        estimator_path = SKLEARN_SPECS[name].path if name in SKLEARN_SPECS else None
        description = (
            f"scikit-learn {estimator_path}"
            if estimator_path
            else f"native kconv adapter ({adapter.__class__.__name__})"
        )
        result.append(
            ModelInfo(
                id=name,
                family=adapter.family,
                tasks=tuple(sorted(adapter.task_types)),
                source=model_source,
                estimator_path=estimator_path,
                description=description,
            )
        )
    return result


def get_model_info(name: str) -> ModelInfo:
    """Return metadata for one model; raises ModelError for unknown ids."""
    adapter = _create_model(name)
    source = "tabpfn" if name == "tabpfn" else ("sklearn" if name in SKLEARN_SPECS else "native")
    path = SKLEARN_SPECS[name].path if name in SKLEARN_SPECS else None
    return ModelInfo(
        id=name,
        family=adapter.family,
        tasks=tuple(sorted(adapter.task_types)),
        source=source,
        estimator_path=path,
        description=f"scikit-learn {path}" if path else f"native kconv adapter ({adapter.__class__.__name__})",
    )


def create_model(name: str, *, task: str | None = None) -> ModelAdapter:
    """Create an adapter, optionally binding and validating its task."""
    if task is not None and task not in _TASKS:
        raise KconvError(f"Unknown task '{task}'.", f"Use one of: {', '.join(sorted(_TASKS))}.")
    adapter = _create_model(name)
    if task is not None:
        if task not in adapter.task_types:
            raise ModelError(
                f"Model '{name}' does not support task '{task}'.",
                f"Use one of: {', '.join(sorted(adapter.task_types))}.",
            )
        adapter.task = task
    return adapter


def _resource_hints(recipe: Recipe) -> tuple[str, ...]:
    """Derive coarse UI/runtime hints from a recipe's block graph."""
    if recipe.kind != "pipeline" or recipe.pipeline is None:
        return ()
    hints: list[str] = ["cpu"]
    for block in recipe.pipeline.blocks:
        if block.type == "knn_class_distance":
            hints.extend(["ram", "cpu-heavy", "classification-only"])
        elif block.type == "feature_selection" and block.params.get("method") == "model_importance":
            hints.append("cpu-heavy")
    return tuple(dict.fromkeys(hints))


def _pipeline_from_recipe(recipe: Recipe) -> PipelineInfo:
    return PipelineInfo(
        id=recipe.id,
        title=recipe.title,
        group=recipe.group,
        author=recipe.author,
        source=recipe.source,
        license=recipe.license,
        tasks=tuple(recipe.tasks),
        models=tuple(recipe.defaults.models),
        best_for=recipe.best_for,
        attribution=recipe.attribution,
        defaults=recipe.defaults.model_dump(mode="json"),
        kind=recipe.kind,
        derived_from=recipe.derived_from,
        pipeline=recipe.pipeline.model_dump(mode="json") if recipe.pipeline is not None else None,
        resource_hints=_resource_hints(recipe),
    )


def list_pipelines(
    *,
    task: str | None = None,
    group: str | None = None,
    kind: str | None = None,
) -> list[PipelineInfo]:
    """List recipe entries, optionally filtered by task/group/kind."""
    if task is not None and task not in _TASKS:
        raise KconvError(f"Unknown task '{task}'.", f"Use one of: {', '.join(sorted(_TASKS))}.")
    if kind is not None and kind not in {"preset", "pipeline"}:
        raise KconvError(
            f"Unknown recipe kind '{kind}'.",
            "Use one of: preset, pipeline.",
        )
    result = []
    for recipe in list_recipes():
        if task is not None and task not in recipe.tasks:
            continue
        if group is not None and recipe.group != group:
            continue
        if kind is not None and recipe.kind != kind:
            continue
        result.append(_pipeline_from_recipe(recipe))
    return result


def get_pipeline(recipe_id: str) -> PipelineInfo:
    """Return one pipeline preset by recipe id."""
    return _pipeline_from_recipe(load_recipe(recipe_id))


def list_runtime_profiles() -> list[RuntimeProfile]:
    """Return the five coarse runtime presets (CPU/GPU/memory)."""
    return list_profiles()


def get_runtime_profile(level: int, *, device: str | None = None) -> RuntimeProfile:
    """Return one runtime preset; profile 1-3 reject GPU/CUDA requests."""
    return get_profile(level, device=device)


def runtime_models(level: int, task: str, *, device: str | None = None) -> list[str]:
    """Return the task-compatible models auto-selected by a runtime profile."""
    profile = get_profile(level, device=device)
    return select_models(profile, task)


def list_experiments(*, status: str | None = None, task: str | None = None) -> list[ExperimentInfo]:
    """List curated experiments, optionally filtered by status or task."""
    return _list_experiments(status=status, task=task)


def build_project(
    data: str | Path,
    *,
    target: str | None = None,
    recipe: str = "pesnik/baseline",
    strategy: str = "fast",
    models: str | None = None,
    name: str | None = None,
    test: str | Path | None = None,
    sample_submission: str | Path | None = None,
    task: str = "auto",
    metric: str = "auto",
    id_col: str | None = None,
    ignore: str | None = None,
    profile: int | None = None,
    device: str | None = None,
    cpu_threads: int | None = None,
    ram_gb: int | None = None,
    vram_gb: int | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, Path, Config]:
    """Create a project config through the same path as ``kconv run``."""
    return build_config(
        data=data,
        target=target,
        recipe_id=recipe,
        strategy=strategy,
        models=models,
        name=name,
        test=test,
        sample_submission=sample_submission,
        task=task,
        metric=metric,
        id_col=id_col,
        ignore=ignore,
        profile=profile,
        device=device,
        cpu_threads=cpu_threads,
        ram_gb=ram_gb,
        vram_gb=vram_gb,
        model_params=model_params,
    )


def run_project(
    project_dir: str | Path,
    config_path: str | Path | None = None,
    *,
    until: str = "report",
    resume: bool = False,
    force: bool = False,
):
    """Run an existing project.  Imported lazily to keep API discovery cheap."""
    from kconv.stages.run import run_pipeline

    project = Path(project_dir).resolve()
    path = Path(config_path).resolve() if config_path is not None else project / "config.yaml"
    config = load_config(path)
    return run_pipeline(project, path, config, until=until, resume=resume, force=force)


def run_dataset(
    data: str | Path,
    *,
    target: str | None = None,
    recipe: str = "pesnik/baseline",
    strategy: str = "fast",
    models: str | None = None,
    name: str | None = None,
    test: str | Path | None = None,
    sample_submission: str | Path | None = None,
    task: str = "auto",
    metric: str = "auto",
    id_col: str | None = None,
    ignore: str | None = None,
    profile: int | None = None,
    device: str | None = None,
    cpu_threads: int | None = None,
    ram_gb: int | None = None,
    vram_gb: int | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
    until: str = "report",
    resume: bool = False,
    force: bool = False,
):
    """Build a project and execute the pipeline; returns a RunContext."""
    project_dir, config_path, _ = build_project(
        data,
        target=target,
        recipe=recipe,
        strategy=strategy,
        models=models,
        name=name,
        test=test,
        sample_submission=sample_submission,
        task=task,
        metric=metric,
        id_col=id_col,
        ignore=ignore,
        profile=profile,
        device=device,
        cpu_threads=cpu_threads,
        ram_gb=ram_gb,
        vram_gb=vram_gb,
        model_params=model_params,
    )
    return run_project(project_dir, config_path, until=until, resume=resume, force=force)


class KConv:
    """Small facade over the module-level API.

    It keeps the functional API primary, while giving applications a single
    object to hold and mock:

        kconv = KConv()
        kconv.models(task="regression")
        kconv.pipelines()
        kconv.sources(downloaded=None)
    """

    def models(
        self,
        *,
        task: str | None = None,
        family: str | None = None,
        source: str | None = None,
    ) -> list[ModelInfo]:
        return list_models(task=task, family=family, source=source)

    def model(self, name: str) -> ModelInfo:
        return get_model_info(name)

    def create(self, name: str, *, task: str | None = None) -> ModelAdapter:
        return create_model(name, task=task)

    def pipelines(
        self,
        *,
        task: str | None = None,
        group: str | None = None,
        kind: str | None = None,
    ) -> list[PipelineInfo]:
        return list_pipelines(task=task, group=group, kind=kind)

    def pipeline(self, recipe_id: str) -> PipelineInfo:
        return get_pipeline(recipe_id)

    def experiments(self, *, status: str | None = None, task: str | None = None) -> list[ExperimentInfo]:
        return list_experiments(status=status, task=task)

    def experiment(self, experiment_id: str) -> ExperimentInfo:
        return get_experiment(experiment_id)

    def profiles(self) -> list[RuntimeProfile]:
        return list_runtime_profiles()

    def profile(self, level: int, *, device: str | None = None) -> RuntimeProfile:
        return get_runtime_profile(level, device=device)

    def profile_models(self, level: int, task: str, *, device: str | None = None) -> list[str]:
        return runtime_models(level, task, device=device)

    def sources(
        self,
        *,
        downloaded: bool | None = True,
        category: str | None = None,
        license_class: str | None = None,
        executable: bool | None = None,
        has_pipeline: bool | None = None,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[SourceInfo]:
        return list_sources(
            downloaded=downloaded,
            category=category,
            license_class=license_class,
            executable=executable,
            has_pipeline=has_pipeline,
            query=query,
            limit=limit,
        )

    def source(self, identifier: str) -> SourceInfo:
        return get_source(identifier)

    def stats(self) -> dict[str, Any]:
        return catalog_stats()

    def build(
        self,
        data: str | Path,
        *,
        target: str | None = None,
        recipe: str = "pesnik/baseline",
        strategy: str = "fast",
        models: str | None = None,
        name: str | None = None,
        test: str | Path | None = None,
        sample_submission: str | Path | None = None,
        task: str = "auto",
        metric: str = "auto",
        id_col: str | None = None,
        ignore: str | None = None,
        profile: int | None = None,
        device: str | None = None,
        cpu_threads: int | None = None,
        ram_gb: int | None = None,
        vram_gb: int | None = None,
        model_params: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[Path, Path, Config]:
        return build_project(
            data,
            target=target,
            recipe=recipe,
            strategy=strategy,
            models=models,
            name=name,
            test=test,
            sample_submission=sample_submission,
            task=task,
            metric=metric,
            id_col=id_col,
            ignore=ignore,
            profile=profile,
            device=device,
            cpu_threads=cpu_threads,
            ram_gb=ram_gb,
            vram_gb=vram_gb,
            model_params=model_params,
        )

    def run(
        self,
        data: str | Path,
        *,
        target: str | None = None,
        recipe: str = "pesnik/baseline",
        strategy: str = "fast",
        models: str | None = None,
        name: str | None = None,
        test: str | Path | None = None,
        sample_submission: str | Path | None = None,
        task: str = "auto",
        metric: str = "auto",
        id_col: str | None = None,
        ignore: str | None = None,
        profile: int | None = None,
        device: str | None = None,
        cpu_threads: int | None = None,
        ram_gb: int | None = None,
        vram_gb: int | None = None,
        model_params: dict[str, dict[str, Any]] | None = None,
        until: str = "report",
        resume: bool = False,
        force: bool = False,
    ):
        return run_dataset(
            data,
            target=target,
            recipe=recipe,
            strategy=strategy,
            models=models,
            name=name,
            test=test,
            sample_submission=sample_submission,
            task=task,
            metric=metric,
            id_col=id_col,
            ignore=ignore,
            profile=profile,
            device=device,
            cpu_threads=cpu_threads,
            ram_gb=ram_gb,
            vram_gb=vram_gb,
            model_params=model_params,
            until=until,
            resume=resume,
            force=force,
        )

    def run_existing(
        self,
        project_dir: str | Path,
        config_path: str | Path | None = None,
        *,
        until: str = "report",
        resume: bool = False,
        force: bool = False,
    ):
        return run_project(project_dir, config_path, until=until, resume=resume, force=force)

    def plugins(self) -> list[PluginInfo]:
        return list_plugins()

    def load_plugins(self) -> list[PluginInfo]:
        return load_plugins()

    def register_model(self, name: str):
        return register_model(name)


__all__ = [
    "ExperimentInfo",
    "KConv",
    "ModelInfo",
    "PipelineInfo",
    "PluginInfo",
    "RuntimeProfile",
    "SourceInfo",
    "build_project",
    "catalog_stats",
    "create_model",
    "get_experiment",
    "get_model_info",
    "get_pipeline",
    "get_runtime_profile",
    "get_source",
    "list_experiments",
    "list_models",
    "list_pipelines",
    "list_plugins",
    "list_runtime_profiles",
    "list_sources",
    "load_plugins",
    "register_model",
    "run_dataset",
    "run_project",
    "runtime_models",
]
