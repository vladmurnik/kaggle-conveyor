"""Project creation: turn a dataset path + a recipe into a runnable config."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from kconv.autodetect import detect_dataset
from kconv.config import (
    AnomalyFeatureConfig,
    ClusterFeatureConfig,
    Config,
    DataConfig,
    DatetimeCyclicalConfig,
    FeatureBlockConfig,
    FeatureConfig,
    FeatureSelectionConfig,
    InteractionConfig,
    KNNFeatureConfig,
    MissingnessConfig,
    ModelConfig,
    NumericInteractionConfig,
    RuntimeConfig,
    SubmitConfig,
    TimeSeriesFeatureConfig,
    save_config,
)
from kconv.context import slugify
from kconv.errors import ConfigError, KconvError, RecipeError
from kconv.features.blocks import FEATURE_SELECTION_BLOCK, validate_block_plan
from kconv.features.group_stats import build_group_statistics_plan
from kconv.features.interactions import interaction_pairs
from kconv.io import safe_read_edges, safe_read_sample
from kconv.models import create_model, model_available
from kconv.models.validation import validate_model_params
from kconv.recipes import Recipe, load_recipe
from kconv.registry import model_names
from kconv.runtime import device_params, get_profile, select_models

Strategy = Literal["fast", "standard", "deep"]
STRATEGY_PROFILES: dict[str, dict[str, object]] = {
    "fast": {"n_splits": 3, "seeds": 1, "hpo": False, "hpo_trials": 8},
    "standard": {"n_splits": 5, "seeds": 2, "hpo": False, "hpo_trials": 8},
    "deep": {"n_splits": 5, "seeds": 3, "hpo": True, "hpo_trials": 8},
}


def _portable_path(path: str | Path, base: Path) -> str:
    """Prefer a path relative to the project folder; keep absolute across drives."""
    resolved = Path(path).resolve()
    try:
        relative = os.path.relpath(resolved, base)
    except ValueError:  # different Windows drives
        return str(resolved)
    return relative.replace(os.sep, "/")


def _auto_encode_columns(features: FeatureConfig, spec, source: Path, ignored: list[str] | None = None) -> None:
    """Fill empty target/frequency lists from empirical cardinality.

    This runs only at project-creation time and the resolved lists are written
    to ``config.yaml``.  Training itself stays deterministic and fit-in-fold.
    """
    if not features.auto_encode:
        return
    if features.target_encode and features.frequency_encode:
        return
    try:
        sample = safe_read_edges(source, nrows=100_000)
    except KconvError:
        return
    if sample.empty:
        return
    target_columns = list(features.target_encode)
    frequency_columns = list(features.frequency_encode)
    excluded = set(ignored or [])
    for column in spec.categorical_columns:
        if column in excluded or column not in sample.columns:
            continue
        values = sample[column].dropna()
        if len(values) < 20:
            continue
        cardinality = int(values.nunique())
        uniqueness = cardinality / max(len(values), 1)
        # Unique-ish strings are identifiers in disguise; encoding them would
        # memorize rows instead of generalizing.  Very large vocabularies are
        # also unsafe for a tabular target encoder.
        if cardinality > 5000 or uniqueness > 0.8:
            continue
        high_cardinality = cardinality > features.max_categories
        if high_cardinality and column not in target_columns:
            target_columns.append(column)
        if (high_cardinality or cardinality > 10) and column not in frequency_columns:
            frequency_columns.append(column)
    features.target_encode = target_columns
    features.frequency_encode = frequency_columns


def _auto_block_columns(
    features: FeatureConfig,
    spec,
    source: Path,
    ignored: list[str] | None,
    *,
    target: bool,
) -> list[str]:
    """Choose auto target/frequency columns for one pipeline block.

    Pipeline blocks opt in explicitly.  The heuristic mirrors the preset
    auto-encoder: skip identifiers, very large vocabularies and tiny columns.
    """
    try:
        sample = safe_read_edges(source, nrows=100_000)
    except KconvError:
        return []
    if sample.empty:
        return []
    excluded = set(ignored or [])
    columns: list[str] = []
    for column in spec.categorical_columns:
        if column in excluded or column not in sample.columns:
            continue
        values = sample[column].dropna()
        if len(values) < 20:
            continue
        cardinality = int(values.nunique())
        uniqueness = cardinality / max(len(values), 1)
        if cardinality > 5000 or uniqueness > 0.8:
            continue
        high_cardinality = cardinality > features.max_categories
        if target:
            if high_cardinality:
                columns.append(column)
        elif high_cardinality or cardinality > 10:
            columns.append(column)
    return columns


def _compile_pipeline_features(
    recipe: Recipe,
    defaults: FeatureConfig,
    spec,
    source: Path,
    ignored: list[str] | None,
) -> FeatureConfig:
    """Compile a pipeline recipe into a resolved FeatureConfig."""
    if recipe.pipeline is None:
        raise RecipeError(
            f"Pipeline recipe '{recipe.id}' has no pipeline blocks.",
            "Add a non-empty pipeline.blocks list.",
        )
    features = FeatureConfig(**defaults.model_dump())
    features.feature_blocks = [
        FeatureBlockConfig(
            type=block.type,
            params=dict(block.params),
            enabled=block.enabled,
        )
        for block in recipe.pipeline.blocks
    ]
    # The block list is authoritative for pipelines; reset legacy flags so a
    # preset-style default cannot silently add undeclared target encoding.
    features.auto_encode = False
    features.target_encode = []
    features.frequency_encode = []
    features.hash_encode = []
    features.hash_features = 16
    features.row_statistics = False
    features.modulo_features = False
    features.fold_plan = None
    features.interactions = []
    features.missingness = MissingnessConfig()
    features.datetime_cyclical = DatetimeCyclicalConfig()
    features.numeric_interactions = NumericInteractionConfig()
    features.clustering = ClusterFeatureConfig()
    features.anomaly = AnomalyFeatureConfig()
    features.time_series = TimeSeriesFeatureConfig()
    features.knn_features = KNNFeatureConfig()
    features.feature_selection = FeatureSelectionConfig()

    for block, compiled in zip(recipe.pipeline.blocks, features.feature_blocks, strict=True):
        if not block.enabled:
            continue
        params = dict(block.params)
        if block.type == "base":
            for key in ("scale", "impute", "max_categories", "auto_encode", "drop_constant"):
                if key in params:
                    setattr(features, key, params[key])
        elif block.type == "row_statistics":
            features.row_statistics = True
            features.row_statistics_stats = list(params.get("stats", ["mean", "std", "min", "max"]))
        elif block.type == "modulo":
            features.modulo_features = True
            features.modulo_moduli = list(params.get("moduli", [10, 100]))
            features.modulo_steps = list(params.get("steps", [5]))
        elif block.type == "fold_plan":
            features.fold_plan = params.get("plan")
        elif block.type == "group_statistics":
            try:
                group_plan = build_group_statistics_plan(params, spec)
            except ValueError as exc:
                raise RecipeError(
                    f"Invalid group_statistics block: {exc}",
                    "Use group_col, numeric columns, and stats from mean/std/count.",
                ) from exc
            features.fold_plan = group_plan.plan
            compiled.params.update(
                {
                    "group_col": group_plan.group_col,
                    "columns": list(group_plan.columns),
                    "stats": list(group_plan.stats),
                }
            )
        elif block.type == "categorical_interaction":
            raw = params.get("columns", "auto")
            if raw in (None, "auto"):
                columns = [
                    column
                    for column in spec.categorical_columns
                    if column not in set(ignored or [])
                ]
            elif isinstance(raw, str):
                columns = [raw]
            elif isinstance(raw, (list, tuple)):
                columns = [str(column) for column in raw]
            else:
                raise RecipeError(
                    "Pipeline block 'categorical_interaction' columns must be 'auto' or a list.",
                    "Use columns: auto or columns: [name, ...].",
                )
            non_categorical = [
                column for column in columns if column not in spec.categorical_columns
            ]
            if non_categorical:
                raise RecipeError(
                    "Pipeline block 'categorical_interaction' requires categorical columns: "
                    f"{', '.join(non_categorical)}.",
                    "Use categorical columns or remove the block.",
                )
            max_pairs = params.get("max_pairs", 4)
            if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or not 1 <= max_pairs <= 50:
                raise RecipeError(
                    "Pipeline block 'categorical_interaction' max_pairs must be 1..50.",
                    "Use max_pairs: 4 or a smaller value.",
                )
            if raw not in (None, "auto") and len(columns) < 2:
                raise RecipeError(
                    "Pipeline block 'categorical_interaction' needs at least two columns.",
                    "Provide two or more categorical columns or use columns: auto.",
                )
            encoding = params.get("encoding", "frequency")
            if encoding not in {"frequency", "target"}:
                raise RecipeError(
                    f"Pipeline block 'categorical_interaction' has unknown encoding '{encoding}'.",
                    "Use encoding: frequency or encoding: target.",
                )
            pairs = interaction_pairs(columns, max_pairs)
            features.interactions = [
                InteractionConfig(columns=list(pair), encoding=encoding) for pair in pairs
            ]
            compiled.params.update(
                {
                    "columns": columns if raw not in (None, "auto") or len(columns) >= 2 else "auto",
                    "max_pairs": max_pairs,
                    "encoding": encoding,
                }
            )
        elif block.type == "missingness":
            raw = params.get("columns", "auto")
            if raw in (None, "auto"):
                columns: list[str] = []
            elif isinstance(raw, str):
                columns = [raw]
            elif isinstance(raw, (list, tuple)):
                columns = [str(column) for column in raw]
            else:
                raise RecipeError(
                    "Pipeline block 'missingness' columns must be 'auto' or a list.",
                    "Use columns: auto or columns: [name, ...].",
                )
            missing_params = {
                key: params[key]
                for key in ("indicators", "row_count", "zero_count")
                if key in params
            }
            features.missingness = MissingnessConfig(
                enabled=True, columns=columns, **missing_params
            )
            compiled.params["columns"] = columns if columns else "auto"
        elif block.type == "datetime_cyclical":
            raw = params.get("columns", "auto")
            if raw in (None, "auto"):
                columns = []
            elif isinstance(raw, str):
                columns = [raw]
            elif isinstance(raw, (list, tuple)):
                columns = [str(column) for column in raw]
            else:
                raise RecipeError(
                    "Pipeline block 'datetime_cyclical' columns must be 'auto' or a list.",
                    "Use columns: auto or columns: [name, ...].",
                )
            non_datetime = [
                column for column in columns if column not in spec.datetime_columns
            ]
            if non_datetime:
                raise RecipeError(
                    "Pipeline block 'datetime_cyclical' requires datetime columns: "
                    f"{', '.join(non_datetime)}.",
                    "Use datetime columns or remove the block.",
                )
            periods = list(
                params.get("periods", ["month", "dayofweek", "dayofyear", "hour"])
            )
            features.datetime_cyclical = DatetimeCyclicalConfig(
                enabled=True, columns=columns, periods=periods
            )
            compiled.params["columns"] = columns if columns else "auto"
            compiled.params["periods"] = periods
        elif block.type == "numeric_interactions":
            raw = params.get("columns", "auto")
            if raw in (None, "auto"):
                columns = [
                    column
                    for column in spec.numeric_columns
                    if column not in set(ignored or [])
                ]
            elif isinstance(raw, str):
                columns = [raw]
            elif isinstance(raw, (list, tuple)):
                columns = [str(column) for column in raw]
            else:
                raise RecipeError(
                    "Pipeline block 'numeric_interactions' columns must be 'auto' or a list.",
                    "Use columns: auto or columns: [name, ...].",
                )
            non_numeric = [
                column for column in columns if column not in spec.numeric_columns
            ]
            if non_numeric:
                raise RecipeError(
                    "Pipeline block 'numeric_interactions' requires numeric columns: "
                    f"{', '.join(non_numeric)}.",
                    "Use numeric columns or remove the block.",
                )
            if len(columns) < 2:
                raise RecipeError(
                    "Pipeline block 'numeric_interactions' needs at least two numeric columns.",
                    "Add a numeric column or remove the block.",
                )
            block_params = {
                key: params[key]
                for key in ("max_pairs", "operations", "clip_value")
                if key in params
            }
            try:
                features.numeric_interactions = NumericInteractionConfig(
                    enabled=True, columns=columns, **block_params
                )
            except ValidationError as exc:
                first = exc.errors()[0]
                raise RecipeError(
                    "Invalid numeric_interactions block at "
                    f"'{first.get('loc', ())}': {first.get('msg', 'invalid value')}",
                    "Check max_pairs, operations and clip_value.",
                ) from exc
            compiled.params.update({"columns": columns, **block_params})
        elif block.type == "clustering":
            raw = params.get("columns", "auto")
            if raw in (None, "auto"):
                columns = [
                    column
                    for column in spec.numeric_columns
                    if column not in set(ignored or [])
                ]
            elif isinstance(raw, str):
                columns = [raw]
            elif isinstance(raw, (list, tuple)):
                columns = [str(column) for column in raw]
            else:
                raise RecipeError(
                    "Pipeline block 'clustering' columns must be 'auto' or a list.",
                    "Use columns: auto or columns: [name, ...].",
                )
            non_numeric = [
                column for column in columns if column not in spec.numeric_columns
            ]
            if non_numeric:
                raise RecipeError(
                    "Pipeline block 'clustering' requires numeric columns: "
                    f"{', '.join(non_numeric)}.",
                    "Use numeric columns or remove the block.",
                )
            if not columns:
                raise RecipeError(
                    "Pipeline block 'clustering' needs at least one numeric column.",
                    "Add a numeric column or remove the block.",
                )
            block_params = {
                key: params[key]
                for key in (
                    "n_clusters",
                    "max_iter",
                    "include_distances",
                    "include_labels",
                    "max_fit_rows",
                )
                if key in params
            }
            try:
                features.clustering = ClusterFeatureConfig(
                    enabled=True, columns=columns, **block_params
                )
            except ValidationError as exc:
                first = exc.errors()[0]
                raise RecipeError(
                    "Invalid clustering block at "
                    f"'{first.get('loc', ())}': {first.get('msg', 'invalid value')}",
                    "Check n_clusters, max_iter and output flags.",
                ) from exc
            compiled.params.update({"columns": columns, **block_params})
        elif block.type == "anomaly":
            raw = params.get("columns", "auto")
            if raw in (None, "auto"):
                columns = [
                    column
                    for column in spec.numeric_columns
                    if column not in set(ignored or [])
                ]
            elif isinstance(raw, str):
                columns = [raw]
            elif isinstance(raw, (list, tuple)):
                columns = [str(column) for column in raw]
            else:
                raise RecipeError(
                    "Pipeline block 'anomaly' columns must be 'auto' or a list.",
                    "Use columns: auto or columns: [name, ...].",
                )
            non_numeric = [
                column for column in columns if column not in spec.numeric_columns
            ]
            if non_numeric:
                raise RecipeError(
                    "Pipeline block 'anomaly' requires numeric columns: "
                    f"{', '.join(non_numeric)}.",
                    "Use numeric columns or remove the block.",
                )
            if not columns:
                raise RecipeError(
                    "Pipeline block 'anomaly' needs at least one numeric column.",
                    "Add a numeric column or remove the block.",
                )
            block_params = {
                key: params[key]
                for key in ("n_estimators", "contamination", "include_flag")
                if key in params
            }
            try:
                features.anomaly = AnomalyFeatureConfig(
                    enabled=True, columns=columns, **block_params
                )
            except ValidationError as exc:
                first = exc.errors()[0]
                raise RecipeError(
                    "Invalid anomaly block at "
                    f"'{first.get('loc', ())}': {first.get('msg', 'invalid value')}",
                    "Check n_estimators and contamination.",
                ) from exc
            compiled.params.update({"columns": columns, **block_params})
        elif block.type == "time_series":
            raw_time = params.get("time_col")
            if raw_time in (None, "auto"):
                candidates = [
                    column
                    for column in spec.datetime_columns
                    if column not in set(ignored or [])
                ]
                if not candidates:
                    raise RecipeError(
                        "Pipeline block 'time_series' needs a datetime feature column.",
                        "Add a datetime column or remove the time_series block.",
                    )
                time_col = candidates[0]
            else:
                time_col = str(raw_time)
                if time_col not in spec.datetime_columns:
                    raise RecipeError(
                        "Pipeline block 'time_series' time_col must be a datetime column.",
                        "Use time_col: auto or a column listed in spec.datetime_columns.",
                    )
                if time_col in set(ignored or []):
                    raise RecipeError(
                        "Pipeline block 'time_series' time_col is ignored by data.ignore.",
                        "Use another datetime column or remove it from data.ignore.",
                    )
            raw_group = params.get("group_col")
            if raw_group in (None, "auto"):
                candidates = [
                    column
                    for column in spec.categorical_columns
                    if column not in set(ignored or [])
                ]
                group_col = candidates[0] if candidates else None
            else:
                group_col = str(raw_group)
                if group_col not in spec.feature_columns:
                    raise RecipeError(
                        "Pipeline block 'time_series' group_col must be a feature column.",
                        "Use group_col: auto or a feature column name.",
                    )
                if group_col in set(ignored or []):
                    raise RecipeError(
                        "Pipeline block 'time_series' group_col is ignored by data.ignore.",
                        "Use another group column or remove it from data.ignore.",
                    )
                if group_col == time_col:
                    raise RecipeError(
                        "Pipeline block 'time_series' group_col and time_col must differ.",
                        "Use a categorical group column or remove group_col.",
                    )
            raw = params.get("columns", "auto")
            if raw in (None, "auto"):
                columns = [
                    column
                    for column in spec.numeric_columns
                    if column not in set(ignored or [])
                    and column not in {time_col, group_col}
                ]
            elif isinstance(raw, str):
                columns = [raw]
            elif isinstance(raw, (list, tuple)):
                columns = [str(column) for column in raw]
            else:
                raise RecipeError(
                    "Pipeline block 'time_series' columns must be 'auto' or a list.",
                    "Use columns: auto or columns: [name, ...].",
                )
            non_numeric = [
                column for column in columns if column not in spec.numeric_columns
            ]
            if non_numeric:
                raise RecipeError(
                    "Pipeline block 'time_series' requires numeric columns: "
                    f"{', '.join(non_numeric)}.",
                    "Use numeric columns or remove the block.",
                )
            forbidden = [column for column in (time_col, group_col) if column in columns]
            if forbidden:
                raise RecipeError(
                    "Pipeline block 'time_series' columns must not include time_col or "
                    f"group_col: {', '.join(forbidden)}.",
                    "Use value columns only; time_col and group_col are separate.",
                )
            if not columns:
                raise RecipeError(
                    "Pipeline block 'time_series' needs at least one numeric value column.",
                    "Add a numeric column or remove the block.",
                )
            block_params = {
                key: params[key]
                for key in (
                    "lags",
                    "windows",
                    "stats",
                    "include_elapsed",
                    "include_diff",
                    "max_history_rows",
                )
                if key in params
            }
            try:
                features.time_series = TimeSeriesFeatureConfig(
                    enabled=True,
                    time_col=time_col,
                    group_col=group_col,
                    columns=columns,
                    **block_params,
                )
            except ValidationError as exc:
                first = exc.errors()[0]
                raise RecipeError(
                    "Invalid time_series block at "
                    f"'{first.get('loc', ())}': {first.get('msg', 'invalid value')}",
                    "Check lags, windows, stats and max_history_rows.",
                ) from exc
            compiled.params.update(
                {
                    "time_col": time_col,
                    "group_col": group_col if group_col is not None else "auto",
                    "columns": columns,
                    **block_params,
                }
            )
        elif block.type == "feature_selection":
            selection_params = {
                key: params[key]
                for key in ("method", "max_features", "min_variance")
                if key in params
            }
            try:
                features.feature_selection = FeatureSelectionConfig(
                    enabled=True, **selection_params
                )
            except ValidationError as exc:
                first = exc.errors()[0]
                raise RecipeError(
                    "Invalid feature_selection block at "
                    f"'{first.get('loc', ())}': {first.get('msg', 'invalid value')}",
                    "Check method, max_features and min_variance.",
                ) from exc
        elif block.type == "knn_class_distance":
            knn_params = {
                key: params[key]
                for key in ("columns", "neighbors", "folds", "metric", "max_reference_rows")
                if key in params
            }
            features.knn_features = KNNFeatureConfig(enabled=True, **knn_params)
        elif block.type in {"target_encoding", "frequency_encoding", "hash_encoding"}:
            raw = params.get("columns", "auto")
            if raw in (None, "auto"):
                columns = _auto_block_columns(
                    features,
                    spec,
                    source,
                    ignored,
                    target=block.type == "target_encoding",
                )
            elif isinstance(raw, str):
                columns = [raw]
            elif isinstance(raw, (list, tuple)):
                columns = [str(column) for column in raw]
            else:
                raise RecipeError(
                    f"Pipeline block '{block.type}' columns must be 'auto' or a list of names.",
                    "Use columns: auto or columns: [name, ...].",
                )
            compiled.params["columns"] = columns
            if block.type == "target_encoding":
                features.target_encode = columns
                if "smoothing" in params:
                    features.target_smoothing = params["smoothing"]
            elif block.type == "frequency_encoding":
                features.frequency_encode = columns
            else:
                raw_features = params.get("n_features", 16)
                if (
                    isinstance(raw_features, bool)
                    or not isinstance(raw_features, int)
                    or not 1 <= raw_features <= 1024
                ):
                    raise RecipeError(
                        f"Pipeline block '{block.type}' n_features must be an integer in 1..1024.",
                        "Use a small number of hash buckets, for example n_features: 16.",
                    )
                features.hash_encode = columns
                features.hash_features = raw_features
                compiled.params["n_features"] = features.hash_features

    # Re-validate after applying free-form YAML params.  This keeps invalid
    # block params from reaching FeatureBuilder or a run.
    try:
        return FeatureConfig.model_validate(features.model_dump())
    except ValidationError as exc:
        first = exc.errors()[0]
        raise RecipeError(
            "Invalid pipeline feature config at "
            f"'{first.get('loc', ())}': {first.get('msg', 'invalid value')}",
            "Check block params against docs/pipelines.md.",
        ) from exc


def apply_fold_plan(config: Config, payload: dict) -> Config:
    """Attach a reviewed fold-aware LLM plan to an existing config.

    Preset configs keep using the legacy ``features.fold_plan`` field.  A
    pipeline config gets an explicit ``fold_plan`` block before feature
    selection, so the block list remains the authoritative execution plan.
    """
    features = config.features
    if features.feature_blocks is None:
        return config.model_copy(update={"features": features.model_copy(update={"fold_plan": payload})})
    blocks = list(features.feature_blocks)
    if any(item.type == "group_statistics" and item.enabled for item in blocks):
        raise RecipeError(
            "This pipeline already defines group_statistics fold aggregates.",
            "Use one fold-aggregate mechanism: group_statistics or a reviewed fold_plan.",
        )
    # Feature selection is a post-feature block and must stay last.  Insert the
    # fold plan before it instead of appending after the selector.  An older
    # config may already contain a fold_plan after the selector, so move that
    # block as well.
    insert_at = next(
        (
            index
            for index, item in enumerate(blocks)
            if item.type == FEATURE_SELECTION_BLOCK and item.enabled
        ),
        len(blocks),
    )
    block = next((item for item in blocks if item.type == "fold_plan"), None)
    if block is None:
        blocks.insert(insert_at, FeatureBlockConfig(type="fold_plan", params={"plan": payload}))
    else:
        block.params["plan"] = payload
        block.enabled = True
        current = blocks.index(block)
        if current > insert_at:
            blocks.pop(current)
            blocks.insert(insert_at, block)
    return config.model_copy(
        update={
            "features": features.model_copy(
                update={"fold_plan": payload, "feature_blocks": blocks}
            )
        }
    )


def _validate_column_roles(validation, spec) -> None:
    """Fail early when a configured group/time column is absent."""
    if not (validation.group_col or validation.time_col):
        return
    try:
        sample = safe_read_sample(Path(spec.train), nrows=5)
    except KconvError:
        return
    available = set(sample.columns)
    for label, column in (("group_col", validation.group_col), ("time_col", validation.time_col)):
        if column and column not in available:
            raise ConfigError(
                f"Validation {label} '{column}' is not present in the training data.",
                "Use a column name from the dataset or remove the override.",
            )


def _parse_models(models: str | None, recipe: Recipe, task: str) -> list[str]:
    requested = [part.strip() for part in models.split(",") if part.strip()] if models else []
    if any(item in {"all", "*"} for item in requested):
        expanded = [
            name
            for name in model_names()
            if task in create_model(name).task_types and model_available(name)
        ]
        requested = expanded + [item for item in requested if item not in {"all", "*"}]
    explicit = bool(requested)
    selected = requested or list(recipe.defaults.models)
    available = set(model_names())
    if not available:
        # Importing the package normally registers all adapters; this is a
        # safety net for unusual partial imports.
        raise ConfigError("No model adapters are registered.", "Reinstall kconv.")
    result: list[str] = []
    for name in selected:
        if name not in available:
            raise ConfigError(f"Unknown model '{name}'.", f"Available models: {', '.join(sorted(available))}")
        adapter = create_model(name)
        if task not in adapter.task_types:
            if explicit:
                raise ConfigError(
                    f"Model '{name}' does not support task '{task}'.",
                    f"Use one of: {', '.join(sorted(adapter.task_types))}.",
                )
            continue
        if not model_available(name):
            if explicit:
                raise ConfigError(
                    f"Model '{name}' needs an optional runtime that is not installed.",
                    "Install it with: uv sync --extra tabpfn (or select another model).",
                )
            continue
        if name not in result:
            result.append(name)
    if not result:
        if explicit:
            raise ConfigError(
                f"No installed model from the explicit list supports task '{task}'.",
                "Choose a compatible model from 'kconv models'.",
            )
        raise RecipeError(
            f"Recipe '{recipe.id}' has no installed model compatible with task '{task}'.",
            "Pass --models with a compatible model or install the optional runtime.",
        )
    return result


def _validate_model_params(
    raw: dict[str, dict[str, Any]] | None,
    selected_models: list[str],
) -> dict[str, dict[str, Any]]:
    """Validate optional per-model overrides before they reach a config.

    The web UI and CLI both accept a JSON object keyed by model id.  Unknown or
    unselected model ids are rejected instead of being silently ignored, so a
    typo cannot produce a run that looks tuned but is not.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            "--model-params must be a JSON object keyed by model id.",
            'Example: {"lgbm": {"n_estimators": 300}, "xgb": {"max_depth": 4}}.',
        )
    selected = set(selected_models)
    for name in raw:
        if not isinstance(name, str):
            raise ConfigError(
                "--model-params keys must be model ids as strings.",
                'Example: {"lgbm": {"n_estimators": 300}}.',
            )
    unknown = sorted(name for name in raw if name not in selected)
    if unknown:
        raise ConfigError(
            f"--model-params contains unselected model(s): {', '.join(unknown)}.",
            "Add them to --models or remove their parameter objects.",
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise ConfigError(
                f"--model-params for '{name}' must be a JSON object.",
                'Example: {"n_estimators": 300, "learning_rate": 0.05}.',
            )
        for param_name in value:
            if not isinstance(param_name, str):
                raise ConfigError(
                    f"--model-params for '{name}' has a non-string parameter name.",
                    "Use adapter parameter names as JSON object keys.",
                )
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"--model-params for '{name}' contains a non-finite or non-JSON value.",
                "Use finite JSON numbers, strings, booleans, lists or objects.",
            ) from exc
        normalized[name] = dict(value)
    return normalized


def _validate_resource_override(name: str, value: int | None, minimum: int, maximum: int) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ConfigError(
            f"{name} must be an integer from {minimum} to {maximum}.",
            "Leave it empty to use the runtime profile default.",
        )


def build_config(
    data: str | Path,
    target: str | None,
    recipe_id: str,
    strategy: str = "fast",
    models: str | None = None,
    name: str | None = None,
    test: str | Path | None = None,
    sample_submission: str | Path | None = None,
    task: str = "auto",
    metric: str = "auto",
    id_col: str | None = None,
    ignore: str | None = None,
    sample_pct: int = 100,
    profile: int | None = None,
    device: str | None = None,
    cpu_threads: int | None = None,
    ram_gb: int | None = None,
    vram_gb: int | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, Path, Config]:
    """Detect a dataset, apply a recipe and write ``competitions/<slug>/config.yaml``.

    ``profile`` resolves one of the five coarse CPU/GPU presets and overrides
    ``strategy``.  It only auto-selects models when ``models`` is not supplied;
    an explicit model list always wins.

    ``cpu_threads``/``ram_gb``/``vram_gb`` refine a selected profile.  CPU
    threads and the device are applied to supported adapters; RAM/VRAM are
    recorded budgets rather than OS-enforced limits.  ``model_params`` maps
    model ids to manual adapter parameters and is layered after recipe and
    runtime defaults.
    """
    if profile is None and device is not None:
        raise ConfigError(
            "--device requires --profile.",
            "Choose a runtime profile 1..5 (for example --profile 3 --device cpu).",
        )
    if profile is None and any(value is not None for value in (cpu_threads, ram_gb, vram_gb)):
        raise ConfigError(
            "--cpu-threads/--ram-gb/--vram-gb require --profile.",
            "Choose a runtime profile 1..5 before overriding its resource budget.",
        )
    _validate_resource_override("--cpu-threads", cpu_threads, 0, 256)
    _validate_resource_override("--ram-gb", ram_gb, 1, 1024)
    _validate_resource_override("--vram-gb", vram_gb, 0, 256)
    _validate_resource_override("--sample-pct", sample_pct, 1, 100)
    runtime = get_profile(profile, device=device) if profile is not None else None
    if runtime is not None:
        runtime = replace(
            runtime,
            cpu_threads=runtime.cpu_threads if cpu_threads is None else cpu_threads,
            ram_gb=runtime.ram_gb if ram_gb is None else ram_gb,
            vram_gb=runtime.vram_gb if vram_gb is None else vram_gb,
        )
        strategy = runtime.strategy
    if strategy not in STRATEGY_PROFILES:
        raise ConfigError(
            f"Unknown strategy '{strategy}'.",
            f"Use one of: {', '.join(STRATEGY_PROFILES)}.",
        )
    strategy_profile = STRATEGY_PROFILES[strategy]
    recipe = load_recipe(recipe_id)
    ignored = [part.strip() for part in ignore.split(",") if part.strip()] if ignore else []
    spec = detect_dataset(
        data,
        target=target,
        test=test,
        sample_submission=sample_submission,
        id=id_col,
        task=None if task == "auto" else task,
        metric=None if metric == "auto" else metric,
    )
    if spec.task not in recipe.tasks:
        raise RecipeError(
            f"Recipe '{recipe.id}' is not intended for task '{spec.task}'.",
            f"Choose one of: {', '.join(recipe.tasks)}.",
        )
    if runtime is not None and models is None:
        models = ",".join(select_models(runtime, spec.task))
    selected_models = _parse_models(models, recipe, spec.task)
    manual_params = _validate_model_params(model_params, selected_models)
    runtime_params = device_params(runtime) if runtime is not None else {}
    defaults = recipe.defaults
    project = slugify(name or Path(data).stem or "project")
    project_dir = Path.cwd() / "competitions" / project
    project_dir.mkdir(parents=True, exist_ok=True)
    validation = defaults.validation.model_copy(
        update={"n_splits": int(strategy_profile["n_splits"])}
    )
    features = FeatureConfig(**defaults.features.model_dump())
    if recipe.kind == "pipeline":
        features = _compile_pipeline_features(recipe, defaults.features, spec, Path(spec.train), ignored)
    else:
        _auto_encode_columns(features, spec, Path(spec.train), ignored)
    _validate_column_roles(validation, spec)
    model_configs = [
        ModelConfig(
            name=model,
            seeds=int(runtime.seeds if runtime is not None else strategy_profile["seeds"]),
            params=validate_model_params(
                model,
                {
                    **defaults.model_params.get(model, {}),
                    **runtime_params.get(model, {}),
                    **manual_params.get(model, {}),
                },
            ),
        )
        for model in selected_models
    ]
    hpo = defaults.hpo.model_copy(
        update={
            "enabled": bool(runtime.hpo if runtime is not None else strategy_profile["hpo"]),
            "trials": int(strategy_profile["hpo_trials"]),
        }
    )
    ensemble = defaults.ensemble.model_copy()
    # A zip must stay a zip in the config: its extracted cache is temporary and
    # must not leak into a portable config.  For folders/files, persisting the
    # discovered paths makes the run self-describing.
    archive = Path(data).suffix.lower() == ".zip"
    train_source = data if archive else spec.train
    test_source = None if archive else spec.test
    sample_source = None if archive else spec.sample_submission
    config = Config(
        project=project,
        recipe=recipe.id,
        recipe_kind=recipe.kind,
        strategy=strategy,  # type: ignore[arg-type]
        task=spec.task,
        metric=spec.metric,
        data=DataConfig(
            train=_portable_path(train_source, project_dir),
            test=_portable_path(test_source, project_dir) if test_source else None,
            sample_submission=_portable_path(sample_source, project_dir) if sample_source else None,
            target=spec.target,
            id=spec.id_col,
            ignore=ignored,
            sample_pct=sample_pct,
        ),
        validation=validation,
        features=features,
        models=model_configs,
        hpo=hpo,
        ensemble=ensemble,
        submit=SubmitConfig(),
        runtime=(
            RuntimeConfig(
                profile=runtime.level,
                device=runtime.device,
                ram_gb=runtime.ram_gb,
                vram_gb=runtime.vram_gb,
                cpu_threads=runtime.cpu_threads,
            )
            if runtime is not None
            else None
        ),
    )
    if config.features.feature_blocks is not None:
        validate_block_plan(config, spec)
    config_path = project_dir / "config.yaml"
    save_config(config, config_path)
    return project_dir, config_path, config
