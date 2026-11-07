"""Benchmark registry, dataset materialization and scorecards.

The benchmark layer deliberately reuses the normal project and pipeline
entrypoints.  It does not implement a second trainer: ``run_benchmark`` calls
``build_config`` and ``run_pipeline`` exactly like the CLI, then extracts a
small, reproducible scorecard from the run artifacts.

Datasets are materialized into a temporary workspace.  The repository's
``data/`` and ``kaggle-toolkit/`` folders are never modified, and generated
scorecards go to ``runs/benchmarks/`` (gitignored) by default.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_diabetes, load_wine
from sklearn.model_selection import train_test_split

from kconv.config import save_config
from kconv.context import RunContext, config_hash
from kconv.contracts import TaskType
from kconv.errors import ConfigError, DataError, KconvError
from kconv.io import read_table, write_table
from kconv.metrics import get_metric
from kconv.project import build_config
from kconv.recipes import Recipe, list_recipes, load_recipe
from kconv.stages.run import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "benchmarks" / "datasets.yaml"
_DATASET_KINDS = frozenset({"local", "sklearn", "synthetic"})
_TASKS = frozenset({"binary", "multiclass", "regression"})


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"Benchmark dataset field '{key}' must be a non-empty string.",
            "Fix benchmarks/datasets.yaml or pass a valid registry path.",
        )
    return value.strip()


def _optional_text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"Benchmark dataset field '{key}' must be a non-empty string or null.",
            "Fix benchmarks/datasets.yaml.",
        )
    return value.strip()


@dataclass(frozen=True)
class BenchmarkDataset:
    """One fixed dataset definition from ``benchmarks/datasets.yaml``."""

    id: str
    title: str
    kind: Literal["local", "sklearn", "synthetic"]
    task: TaskType
    metric: str
    target: str
    source: str
    license: str
    id_col: str | None = "id"
    train: str | None = None
    test: str | None = None
    sample_submission: str | None = None
    sklearn_dataset: str | None = None
    fallback_sklearn: str | None = None
    sample_fraction: float | None = None
    seed: int = 42
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: Any) -> BenchmarkDataset:
        if not isinstance(payload, dict):
            raise ConfigError(
                "Every benchmark dataset entry must be a mapping.",
                "Fix benchmarks/datasets.yaml.",
            )
        dataset_id = _text(payload, "id")
        title = _text(payload, "title")
        kind = _text(payload, "kind")
        task = _text(payload, "task")
        metric = _text(payload, "metric")
        target = _text(payload, "target")
        source = _text(payload, "source")
        license_name = _text(payload, "license")
        if kind not in _DATASET_KINDS:
            raise ConfigError(
                f"Benchmark dataset '{dataset_id}' has unknown kind '{kind}'.",
                "Use local, sklearn or synthetic.",
            )
        if task not in _TASKS:
            raise ConfigError(
                f"Benchmark dataset '{dataset_id}' has unknown task '{task}'.",
                "Use binary, multiclass or regression.",
            )
        try:
            get_metric(metric)
        except KconvError as exc:
            raise ConfigError(
                f"Benchmark dataset '{dataset_id}' has unknown metric '{metric}'.",
                "Use a metric from 'kconv metrics'.",
            ) from exc
        id_col = _optional_text(payload, "id_col")
        train = _optional_text(payload, "train")
        test = _optional_text(payload, "test")
        sample_submission = _optional_text(payload, "sample_submission")
        sklearn_dataset = _optional_text(payload, "sklearn_dataset")
        fallback_sklearn = _optional_text(payload, "fallback_sklearn")
        if kind == "local" and not train:
            raise ConfigError(
                f"Benchmark dataset '{dataset_id}' is local but has no train path.",
                "Add train and test paths.",
            )
        if kind == "local" and not test:
            raise ConfigError(
                f"Benchmark dataset '{dataset_id}' is local but has no test path.",
                "Add train and test paths.",
            )
        if kind == "sklearn" and not sklearn_dataset:
            raise ConfigError(
                f"Benchmark dataset '{dataset_id}' is sklearn but has no sklearn_dataset.",
                "Use wine or diabetes.",
            )
        if fallback_sklearn is not None and fallback_sklearn not in {"wine", "diabetes"}:
            raise ConfigError(
                f"Benchmark dataset '{dataset_id}' has unknown fallback_sklearn.",
                "Use wine or diabetes.",
            )
        raw_fraction = payload.get("sample_fraction")
        if raw_fraction is not None:
            if isinstance(raw_fraction, bool):
                raise ConfigError("sample_fraction must be a number in (0, 1].")
            try:
                fraction = float(raw_fraction)
            except (TypeError, ValueError) as exc:
                raise ConfigError("sample_fraction must be a number in (0, 1].") from exc
            if not 0.0 < fraction <= 1.0:
                raise ConfigError("sample_fraction must be a number in (0, 1].")
        else:
            fraction = None
        raw_seed = payload.get("seed", 42)
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise ConfigError("Benchmark seed must be an integer.")
        notes = payload.get("notes", "")
        if notes is None:
            notes = ""
        if not isinstance(notes, str):
            raise ConfigError("Benchmark notes must be a string.")
        return cls(
            id=dataset_id,
            title=title,
            kind=kind,  # type: ignore[arg-type]
            task=task,  # type: ignore[arg-type]
            metric=metric,
            target=target,
            source=source,
            license=license_name,
            id_col=id_col,
            train=train,
            test=test,
            sample_submission=sample_submission,
            sklearn_dataset=sklearn_dataset,
            fallback_sklearn=fallback_sklearn,
            sample_fraction=fraction,
            seed=raw_seed,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def materialize(self, workspace: str | Path, root: str | Path) -> PreparedDataset:
        """Write train/test/sample files into ``workspace/data``.

        Local files are sampled without modifying the originals.  A missing
        local path uses ``fallback_sklearn`` when configured; otherwise it
        raises a typed error with the expected path.
        """
        workspace = Path(workspace).resolve()
        root = Path(root).resolve()
        data_dir = workspace / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        if self.kind == "local":
            train_path = _resolve_path(root, self.train)
            test_path = _resolve_path(root, self.test)
            if not train_path.exists() or not test_path.exists():
                if self.fallback_sklearn:
                    return self._materialize_sklearn(
                        self.fallback_sklearn,
                        data_dir,
                        fallback_note=(
                            "Local dataset path is missing; using the configured "
                            f"scikit-learn fallback '{self.fallback_sklearn}'."
                        ),
                    )
                missing = train_path if not train_path.exists() else test_path
                raise DataError(
                    f"Benchmark dataset '{self.id}' is missing {missing}.",
                    "Download the local dataset or set KCONV_BENCHMARK_REGISTRY to a valid registry.",
                )
            train = read_table(train_path)
            test = read_table(test_path)
            sample_path = _resolve_path(root, self.sample_submission) if self.sample_submission else None
            sample = read_table(sample_path) if sample_path is not None and sample_path.exists() else None
            if self.sample_fraction is not None:
                train = _sample_frame(
                    train,
                    self.sample_fraction,
                    self.seed,
                    target=self.target,
                    stratified=self.task in {"binary", "multiclass"},
                )
                test = _sample_frame(test, self.sample_fraction, self.seed)
            sample = _filter_sample(sample, test, self.id_col)
            return _write_prepared(
                data_dir,
                self,
                train=train,
                test=test,
                sample=sample,
                target=self.target,
                id_col=self.id_col,
                task=self.task,
                metric=self.metric,
                source=self.source,
                license_name=self.license,
                notes=self.notes,
            )

        if self.kind == "sklearn":
            assert self.sklearn_dataset is not None
            return self._materialize_sklearn(self.sklearn_dataset, data_dir)

        return self._materialize_synthetic(data_dir)

    def _materialize_sklearn(
        self,
        dataset_name: str,
        data_dir: Path,
        *,
        fallback_note: str = "",
    ) -> PreparedDataset:
        name = dataset_name.lower()
        if name == "wine":
            bunch = load_wine(as_frame=True)
            frame = bunch.frame.copy()
            target = "target"
            task: TaskType = "multiclass"
            metric = "log_loss"
            test_size = 0.30
            stratify = frame[target]
        elif name == "diabetes":
            bunch = load_diabetes(as_frame=True)
            frame = bunch.frame.copy()
            target = "target"
            task = "regression"
            metric = "rmse"
            test_size = 0.20
            stratify = None
        else:
            raise ConfigError(
                f"Unknown scikit-learn benchmark dataset '{dataset_name}'.",
                "Use wine or diabetes.",
            )
        frame.insert(0, "id", np.arange(len(frame), dtype=int))
        train, test = train_test_split(
            frame,
            test_size=test_size,
            random_state=self.seed,
            stratify=stratify,
        )
        test = test.drop(columns=[target])
        sample = _make_sample(train, test, target, "id", task)
        notes = " ".join(part for part in (self.notes, fallback_note) if part).strip()
        return _write_prepared(
            data_dir,
            self,
            train=train,
            test=test,
            sample=sample,
            target=target,
            id_col="id",
            task=task,
            metric=metric,
            source=f"scikit-learn load_{name} (UCI public dataset)",
            license_name="public UCI dataset; scikit-learn BSD-3-Clause loader",
            notes=notes,
        )

    def _materialize_synthetic(self, data_dir: Path) -> PreparedDataset:
        rng = np.random.default_rng(self.seed)
        n = 150
        frame = pd.DataFrame(
            {
                "id": np.arange(n),
                "x0": rng.normal(size=n),
                "x1": rng.normal(size=n),
                "cat": rng.choice(["a", "b", "c"], n),
                "when": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
                "target": (rng.random(n) > 0.5).astype(int),
            }
        )
        train = frame.iloc[:110].reset_index(drop=True)
        test = frame.iloc[110:].drop(columns=["target"]).reset_index(drop=True)
        sample = _make_sample(train, test, "target", "id", "binary")
        return _write_prepared(
            data_dir,
            self,
            train=train,
            test=test,
            sample=sample,
            target="target",
            id_col="id",
            task="binary",
            metric="roc_auc",
            source="kconv deterministic generator",
            license_name="generated data; no third-party dataset",
            notes="Contract-only dataset; not evidence of real-world quality.",
        )


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    data_dir: Path
    train_path: Path
    test_path: Path
    sample_submission_path: Path
    target: str
    id_col: str | None
    task: TaskType
    metric: str
    source: str
    license: str
    notes: str
    rows_train: int
    rows_test: int


@dataclass(frozen=True)
class BenchmarkRegistry:
    path: Path
    root: Path
    datasets: tuple[BenchmarkDataset, ...]

    def get(self, dataset_id: str) -> BenchmarkDataset:
        for dataset in self.datasets:
            if dataset.id == dataset_id:
                return dataset
        available = ", ".join(dataset.id for dataset in self.datasets)
        raise ConfigError(
            f"Unknown benchmark dataset '{dataset_id}'.",
            f"Available datasets: {available}",
        )


@dataclass
class BenchmarkResult:
    dataset_id: str
    dataset_title: str
    recipe_id: str
    recipe_kind: str | None
    task: str
    metric: str
    status: str
    mean: float | None = None
    std: float | None = None
    fold_scores: list[float] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    duration_sec: float = 0.0
    peak_rss_mb: float | None = None
    submission_valid: bool = False
    warnings: list[str] = field(default_factory=list)
    config_hash: str | None = None
    seed: int = 42
    git_sha: str | None = None
    error: str | None = None
    source: str | None = None
    license: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Public compatibility alias: one scorecard row is a BenchmarkResult.
Scorecard = BenchmarkResult


@dataclass
class BenchmarkRun:
    run_id: str
    dataset_id: str
    dataset_title: str
    strategy: str
    started_at: str
    output_dir: Path
    results: list[BenchmarkResult]

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(
            result.status in {"ok", "skipped"} for result in self.results
        ) and any(result.status == "ok" for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "dataset_title": self.dataset_title,
            "strategy": self.strategy,
            "started_at": self.started_at,
            "output_dir": str(self.output_dir),
            "ok": self.ok,
            "results": [result.to_dict() for result in self.results],
        }


def load_benchmark_registry(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
) -> BenchmarkRegistry:
    """Load and validate the fixed benchmark registry.

    ``root`` is the directory used to resolve relative dataset paths.  It
    defaults to the repository root for the bundled registry and to the YAML
    file's parent for a custom registry.
    """
    env_path = os.environ.get("KCONV_BENCHMARK_REGISTRY")
    if path is not None:
        registry_path = Path(path).resolve()
    elif env_path:
        registry_path = Path(env_path).expanduser().resolve()
    else:
        registry_path = DEFAULT_REGISTRY
    if root is not None:
        registry_root = Path(root).resolve()
    elif path is None and not env_path:
        registry_root = REPO_ROOT
    else:
        registry_root = registry_path.parent
    if not registry_path.exists():
        raise ConfigError(
            f"Benchmark registry not found: {registry_path}",
            "Set KCONV_BENCHMARK_REGISTRY or create benchmarks/datasets.yaml.",
        )
    try:
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"Could not read benchmark registry {registry_path}: {exc}",
            "Fix the YAML syntax.",
        ) from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("datasets"), list):
        raise ConfigError(
            "Benchmark registry must contain a top-level 'datasets' list.",
            "See docs/benchmark.md for the format.",
        )
    datasets: list[BenchmarkDataset] = []
    seen: set[str] = set()
    for item in raw["datasets"]:
        dataset = BenchmarkDataset.from_dict(item)
        if dataset.id in seen:
            raise ConfigError(
                f"Duplicate benchmark dataset id '{dataset.id}'.",
                "Every dataset id must be unique.",
            )
        seen.add(dataset.id)
        datasets.append(dataset)
    if not datasets:
        raise ConfigError("Benchmark registry has no datasets.")
    return BenchmarkRegistry(path=registry_path, root=registry_root, datasets=tuple(datasets))


def run_benchmark(
    dataset_id: str,
    *,
    registry: BenchmarkRegistry | None = None,
    recipes: Sequence[str] | str | None = None,
    strategy: str = "fast",
    models: str | None = None,
    n_splits: int | None = None,
    output_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
    keep_artifacts: bool = False,
    max_minutes: float | None = None,
) -> BenchmarkRun:
    """Run selected recipes on one benchmark dataset and write a scorecard.

    The dataset is materialized once in a temporary workspace.  Each recipe is
    built and executed through the normal ``build_config``/``run_pipeline``
    path.  A failed recipe is recorded as ``status=failed`` and does not stop
    the remaining recipes.
    """
    if n_splits is not None and n_splits < 2:
        raise ConfigError("n_splits must be at least 2.")
    if max_minutes is not None and max_minutes <= 0:
        raise ConfigError("max_minutes must be positive.")
    registry = registry or load_benchmark_registry()
    dataset = registry.get(dataset_id)
    recipe_ids = _select_recipe_ids(recipes)
    started_at = datetime.now(UTC).isoformat()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (Path.cwd() / "runs" / "benchmarks" / run_id).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    cleanup = False
    if work_dir is not None:
        work_root = Path(work_dir).resolve()
        work_root.mkdir(parents=True, exist_ok=True)
    elif keep_artifacts:
        work_root = output / "work"
        work_root.mkdir(parents=True, exist_ok=True)
    else:
        work_root = Path(tempfile.mkdtemp(prefix="kconv-bench-"))
        cleanup = True
    results: list[BenchmarkResult] = []
    started_monotonic = time.monotonic()
    try:
        prepared = dataset.materialize(work_root, registry.root)
        for recipe_id in recipe_ids:
            recipe = load_recipe(recipe_id)
            if prepared.task not in recipe.tasks:
                results.append(
                    _skipped_result(
                        dataset,
                        prepared,
                        recipe,
                        f"Recipe '{recipe_id}' does not declare task '{prepared.task}'.",
                    )
                )
                continue
            if max_minutes is not None and time.monotonic() - started_monotonic > max_minutes * 60:
                results.append(
                    _skipped_result(
                        dataset,
                        prepared,
                        recipe,
                        f"Benchmark budget of {max_minutes:g} minutes exhausted.",
                    )
                )
                continue
            results.append(
                _run_one(
                    dataset,
                    prepared,
                    recipe,
                    registry.root,
                    work_root,
                    strategy=strategy,
                    models=models,
                    n_splits=n_splits,
                )
            )
        run = BenchmarkRun(
            run_id=run_id,
            dataset_id=dataset.id,
            dataset_title=dataset.title,
            strategy=strategy,
            started_at=started_at,
            output_dir=output,
            results=results,
        )
        write_scorecard(run)
        return run
    finally:
        if cleanup and not _is_relative_to(output, work_root):
            shutil.rmtree(work_root, ignore_errors=True)


def write_scorecard(run: BenchmarkRun, output_dir: str | Path | None = None) -> Path:
    """Write ``scorecard.json`` and ``scorecard.md`` and return the folder."""
    target = Path(output_dir).resolve() if output_dir is not None else run.output_dir
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "scorecard.json"
    json_path.write_text(
        json.dumps(run.to_dict(), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (target / "scorecard.md").write_text(_scorecard_markdown(run), encoding="utf-8")
    return target


def _run_one(
    dataset: BenchmarkDataset,
    prepared: PreparedDataset,
    recipe: Recipe,
    registry_root: Path,
    work_root: Path,
    *,
    strategy: str,
    models: str | None,
    n_splits: int | None,
) -> BenchmarkResult:
    started = time.perf_counter()
    try:
        with contextlib.chdir(work_root):
            project_dir, config_path, config = build_config(
                data=prepared.data_dir,
                target=prepared.target,
                recipe_id=recipe.id,
                strategy=strategy,
                models=models,
                name=f"bench-{dataset.id}-{recipe.id}",
                task=prepared.task,
                metric=prepared.metric,
                id_col=prepared.id_col,
            )
            if n_splits is not None:
                config.validation.n_splits = int(n_splits)
                save_config(config, config_path)
            ctx = run_pipeline(project_dir, config_path, config, until="report")
        return _result_from_ctx(dataset, prepared, recipe, ctx, registry_root, started)
    except Exception as exc:  # noqa: BLE001 - one failed recipe must not abort the run
        return BenchmarkResult(
            dataset_id=dataset.id,
            dataset_title=dataset.title,
            recipe_id=recipe.id,
            recipe_kind=recipe.kind,
            task=prepared.task,
            metric=prepared.metric,
            status="failed",
            duration_sec=round(time.perf_counter() - started, 3),
            peak_rss_mb=_peak_rss_mb(),
            error=f"{type(exc).__name__}: {exc}",
            source=prepared.source,
            license=prepared.license,
            notes=prepared.notes,
        )


def _result_from_ctx(
    dataset: BenchmarkDataset,
    prepared: PreparedDataset,
    recipe: Recipe,
    ctx: RunContext,
    registry_root: Path,
    started: float,
) -> BenchmarkResult:
    blend = _read_json(ctx.run_dir / "03_blend" / "blend.json")
    card = _read_json(ctx.run_dir / "00_ingest" / "data_card.json")
    adversarial = _read_json(ctx.run_dir / "01_validate" / "adversarial.json")
    metric = str(blend.get("metric") or prepared.metric)
    fold_scores = _best_fold_scores(ctx.run_dir / "02_train", metric)
    mean = _as_float(blend.get("oof_score"))
    std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    rows = card.get("rows") if isinstance(card.get("rows"), dict) else {}
    warnings = list(ctx.warnings)
    raw_warnings = card.get("warnings")
    if isinstance(raw_warnings, list):
        warnings.extend(str(item) for item in raw_warnings)
    if adversarial.get("warning"):
        warnings.append(str(adversarial.get("message", "Train/test shift detected.")))
    if blend.get("fallback_used"):
        warnings.append("Ensemble fell back to its configured fallback.")
    warnings = list(dict.fromkeys(warnings))
    submission_path = ctx.run_dir / "04_submit" / "submission.csv"
    return BenchmarkResult(
        dataset_id=dataset.id,
        dataset_title=dataset.title,
        recipe_id=recipe.id,
        recipe_kind=recipe.kind,
        task=str(blend.get("task") or prepared.task),
        metric=metric,
        status="ok" if mean is not None else "failed",
        mean=mean,
        std=std,
        fold_scores=fold_scores,
        n_train=int(rows.get("train") or 0),
        n_test=int(rows.get("test") or 0),
        duration_sec=round(time.perf_counter() - started, 3),
        peak_rss_mb=_peak_rss_mb(),
        submission_valid=_submission_valid(submission_path),
        warnings=warnings,
        config_hash=config_hash(ctx.config),
        seed=ctx.config.seed,
        git_sha=_git_sha(registry_root),
        error=None if mean is not None else "Blend did not produce a finite OOF score.",
        source=prepared.source,
        license=prepared.license,
        notes=prepared.notes,
    )


def _skipped_result(
    dataset: BenchmarkDataset,
    prepared: PreparedDataset,
    recipe: Recipe,
    reason: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        dataset_id=dataset.id,
        dataset_title=dataset.title,
        recipe_id=recipe.id,
        recipe_kind=recipe.kind,
        task=prepared.task,
        metric=prepared.metric,
        status="skipped",
        warnings=[reason],
        source=prepared.source,
        license=prepared.license,
        notes=prepared.notes,
    )


def _select_recipe_ids(recipes: Sequence[str] | str | None) -> list[str]:
    all_ids = [recipe.id for recipe in list_recipes()]
    if recipes is None:
        return all_ids
    requested = (
        [part.strip() for part in recipes.split(",") if part.strip()]
        if isinstance(recipes, str)
        else [str(part).strip() for part in recipes if str(part).strip()]
    )
    if not requested:
        raise ConfigError("No recipes selected.", "Pass --recipes all or a comma-separated list.")
    if any(item == "all" for item in requested):
        return all_ids
    known = set(all_ids)
    unknown = [item for item in requested if item not in known]
    if unknown:
        raise ConfigError(
            f"Unknown benchmark recipe(s): {', '.join(unknown)}.",
            "Use 'kconv recipes' to list ids.",
        )
    return list(dict.fromkeys(requested))


def _resolve_path(root: Path, value: str | None) -> Path:
    if not value:
        raise DataError("Benchmark dataset path is missing.", "Fix benchmarks/datasets.yaml.")
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def _sample_frame(
    frame: pd.DataFrame,
    fraction: float,
    seed: int,
    *,
    target: str | None = None,
    stratified: bool = False,
) -> pd.DataFrame:
    if fraction >= 1.0 or frame.empty:
        return frame.reset_index(drop=True)
    if stratified and target and target in frame.columns:
        parts: list[pd.DataFrame] = []
        for _, group in frame.groupby(target, dropna=False):
            count = max(1, int(round(len(group) * fraction)))
            parts.append(group.sample(n=min(count, len(group)), random_state=seed))
        sampled = pd.concat(parts, ignore_index=True)
        return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return frame.sample(frac=fraction, random_state=seed).reset_index(drop=True)


def _filter_sample(
    sample: pd.DataFrame | None,
    test: pd.DataFrame,
    id_col: str | None,
) -> pd.DataFrame | None:
    if sample is None:
        return None
    if id_col and id_col in sample.columns and id_col in test.columns:
        filtered = sample[sample[id_col].isin(test[id_col])].reset_index(drop=True)
        return filtered if not filtered.empty else None
    return sample.reset_index(drop=True)


def _make_sample(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    id_col: str | None,
    task: TaskType,
) -> pd.DataFrame:
    id_name = id_col or "id"
    ids = test[id_name].to_numpy() if id_name in test.columns else np.arange(len(test))
    if task == "multiclass":
        classes = sorted(pd.unique(train[target])) if target in train.columns else [0, 1, 2]
        frame = pd.DataFrame({id_name: ids})
        for class_value in classes:
            frame[f"class_{class_value}"] = 1.0 / max(len(classes), 1)
        return frame
    return pd.DataFrame({id_name: ids, target: 0.0 if task == "regression" else 0})


def _write_prepared(
    data_dir: Path,
    dataset: BenchmarkDataset,
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame | None,
    target: str,
    id_col: str | None,
    task: TaskType,
    metric: str,
    source: str,
    license_name: str,
    notes: str,
) -> PreparedDataset:
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    if sample is None or sample.empty:
        sample = _make_sample(train, test, target, id_col, task)
    train_path = write_table(train, data_dir / "train.csv")
    test_path = write_table(test, data_dir / "test.csv")
    sample_path = write_table(sample, data_dir / "sample_submission.csv")
    return PreparedDataset(
        dataset_id=dataset.id,
        data_dir=data_dir,
        train_path=train_path,
        test_path=test_path,
        sample_submission_path=sample_path,
        target=target,
        id_col=id_col,
        task=task,
        metric=metric,
        source=source,
        license=license_name,
        notes=notes,
        rows_train=int(len(train)),
        rows_test=int(len(test)),
    )


def _best_fold_scores(train_dir: Path, metric: str) -> list[float]:
    best: float | None = None
    best_scores: list[float] = []
    direction = "maximize"
    try:
        direction = get_metric(metric).direction
    except KconvError:
        pass
    for path in sorted(train_dir.glob("*/metrics.json")):
        payload = _read_json(path)
        score = _as_float(payload.get("oof_score"))
        if score is None:
            continue
        if best is None or (score > best if direction == "maximize" else score < best):
            best = score
            raw = payload.get("fold_scores")
            if isinstance(raw, list):
                best_scores = [float(value) for value in raw if _as_float(value) is not None]
    return best_scores


def _submission_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:  # noqa: BLE001 - invalid artifact is reported as false
        return False
    return not frame.empty and not frame.isna().any().any()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(float(usage) / divisor, 1)


def _git_sha(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _scorecard_markdown(run: BenchmarkRun) -> str:
    lines = [
        f"# kconv benchmark - {run.dataset_id}",
        "",
        f"- Run: `{run.run_id}`",
        f"- Dataset: `{run.dataset_title}`",
        f"- Strategy: `{run.strategy}`",
        f"- Started: `{run.started_at}`",
        f"- Status: `{'ok' if run.ok else 'failed'}`",
        "",
        "| Recipe | Kind | Status | Metric | Mean | Std | Folds | Train | Test | Seconds | Submission |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in run.results:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(result.recipe_id),
                    _md(result.recipe_kind or "-"),
                    _md(result.status),
                    _md(result.metric),
                    _fmt(result.mean),
                    _fmt(result.std),
                    str(len(result.fold_scores)),
                    str(result.n_train),
                    str(result.n_test),
                    f"{result.duration_sec:.3f}",
                    "yes" if result.submission_valid else "no",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "> 10% samples and synthetic data are smoke/ranking evidence, not a",
        "> leaderboard result.  Selection must use train/CV only.",
        "",
    ]
    return "\n".join(lines)
