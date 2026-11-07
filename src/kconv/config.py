"""Strict configuration schema and YAML helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from kconv.errors import ConfigError

TaskChoice = Literal["auto", "binary", "multiclass", "regression"]
StrategyChoice = Literal["fast", "standard", "deep"]
MetricChoice = str


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    train: str
    test: str | None = None
    sample_submission: str | None = None
    target: str | None = None
    id: str | None = None
    ignore: list[str] = Field(default_factory=list)
    sample_pct: int = Field(default=100, ge=1, le=100)

    @field_validator("train")
    @classmethod
    def _train_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("data.train must not be empty")
        return value


class ValidationConfig(StrictModel):
    strategy: Literal[
        "auto",
        "kfold",
        "stratified_kfold",
        "group_kfold",
        "time_series",
        "holdout",
    ] = "auto"
    n_splits: int = Field(default=5, ge=2, le=20)
    shuffle: bool = True
    random_state: int = 42
    group_col: str | None = None
    time_col: str | None = None
    gap: int = Field(default=0, ge=0)
    holdout_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    adversarial: bool = True
    adversarial_folds: int = Field(default=3, ge=2, le=10)
    adversarial_threshold: float = Field(default=0.75, ge=0.5, le=1.0)


class KNNFeatureConfig(StrictModel):
    """Class-conditional kNN distance features.

    The transformer is fitted on the outer training fold and its training-row
    values are produced with an inner OOF split, so a row never sees its own
    label or itself as a neighbour.  It is classification-only and uses the
    numeric feature columns by default.
    """

    enabled: bool = False
    columns: list[str] = Field(default_factory=list)
    neighbors: int = Field(default=3, ge=1, le=20)
    folds: int = Field(default=5, ge=2, le=10)
    metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean"
    max_reference_rows: int = Field(default=50_000, ge=100, le=1_000_000)


class InteractionConfig(StrictModel):
    """Pairwise categorical interaction encoding.

    ``frequency`` is label-free and fitted on the outer training fold.
    ``target`` uses the same inner-OOF target encoder as normal target
    encoding, so a row never contributes its own target to its own feature.
    """

    columns: list[str] = Field(min_length=2, max_length=2)
    encoding: Literal["frequency", "target"] = "frequency"

    @field_validator("columns")
    @classmethod
    def _distinct_columns(cls, value: list[str]) -> list[str]:
        if value[0] == value[1]:
            raise ValueError("interaction columns must be two distinct columns")
        return value


class MissingnessConfig(StrictModel):
    """Label-free missingness and zero-count features.

    Empty ``columns`` means all feature columns.  ``indicators`` adds one
    missing flag per selected column, ``row_count`` adds the row-wise missing
    count, and ``zero_count`` adds the row-wise zero count over selected
    numeric columns.
    """

    enabled: bool = False
    columns: list[str] = Field(default_factory=list)
    indicators: bool = True
    row_count: bool = True
    zero_count: bool = False

    @field_validator("columns")
    @classmethod
    def _dedupe_columns(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class DatetimeCyclicalConfig(StrictModel):
    """Label-free cyclical datetime features.

    Empty ``columns`` means every datetime feature column.  Each period emits
    ``sin``/``cos`` pairs; the raw date parts remain available from ``base``.
    """

    enabled: bool = False
    columns: list[str] = Field(default_factory=list)
    periods: list[Literal["month", "dayofweek", "dayofyear", "hour"]] = Field(
        default_factory=lambda: ["month", "dayofweek", "dayofyear", "hour"]
    )

    @field_validator("columns", "periods")
    @classmethod
    def _dedupe_values(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class NumericInteractionConfig(StrictModel):
    """Label-free pairwise numeric interaction features.

    Empty ``columns`` means every numeric feature column.  The block emits
    deterministic products, ratios, differences and/or sums for at most
    ``max_pairs`` pairs.  Ratios with a zero denominator become zero and all
    outputs are clipped to a finite range.
    """

    enabled: bool = False
    columns: list[str] = Field(default_factory=list)
    max_pairs: int = Field(default=6, ge=1, le=50)
    operations: list[Literal["product", "ratio", "difference", "sum"]] = Field(
        default_factory=lambda: ["product", "ratio"]
    )
    clip_value: float = Field(default=1e12, gt=0.0)

    @field_validator("columns", "operations")
    @classmethod
    def _dedupe_values(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @field_validator("operations")
    @classmethod
    def _nonempty_operations(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("numeric_interactions operations must not be empty")
        return value


class ClusterFeatureConfig(StrictModel):
    """Fit-in-fold unsupervised clustering features.

    A MiniBatchKMeans model is fitted on the outer training fold only.  The
    block can emit distances to every centroid and/or the cluster label; it
    never reads the target.
    """

    enabled: bool = False
    columns: list[str] = Field(default_factory=list)
    n_clusters: int = Field(default=8, ge=2, le=50)
    max_iter: int = Field(default=100, ge=10, le=1000)
    include_distances: bool = True
    include_labels: bool = True
    max_fit_rows: int = Field(default=50_000, ge=100, le=1_000_000)

    @field_validator("columns")
    @classmethod
    def _dedupe_columns(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def _at_least_one_output(self) -> ClusterFeatureConfig:
        if not self.include_distances and not self.include_labels:
            raise ValueError("clustering must include distances, labels, or both")
        return self


class AnomalyFeatureConfig(StrictModel):
    """Fit-in-fold IsolationForest anomaly features."""

    enabled: bool = False
    columns: list[str] = Field(default_factory=list)
    n_estimators: int = Field(default=100, ge=10, le=500)
    contamination: float | Literal["auto"] = "auto"
    include_flag: bool = True

    @field_validator("columns")
    @classmethod
    def _dedupe_columns(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @field_validator("contamination")
    @classmethod
    def _valid_contamination(cls, value: float | str) -> float | str:
        if value == "auto":
            return value
        if not 0.0 < float(value) <= 0.5:
            raise ValueError("anomaly contamination must be 'auto' or in (0, 0.5]")
        return float(value)


class TimeSeriesFeatureConfig(StrictModel):
    """Past-only grouped time-series features.

    ``time_col`` is required by the pipeline block but may be ``auto`` before
    compilation.  Empty ``columns`` means every numeric feature column.
    Lags and rolling windows are computed per group from past rows only; fit
    stores a bounded history tail for validation/test transformation.
    """

    enabled: bool = False
    time_col: str | None = None
    group_col: str | None = None
    columns: list[str] = Field(default_factory=list)
    lags: list[int] = Field(default_factory=lambda: [1])
    windows: list[int] = Field(default_factory=lambda: [3])
    stats: list[Literal["mean", "std", "min", "max"]] = Field(
        default_factory=lambda: ["mean"]
    )
    include_elapsed: bool = True
    include_diff: bool = True
    max_history_rows: int = Field(default=10_000, ge=100, le=1_000_000)

    @field_validator("columns", "lags", "windows", "stats")
    @classmethod
    def _dedupe_values(cls, value: list) -> list:
        return list(dict.fromkeys(value))

    @field_validator("lags")
    @classmethod
    def _valid_lags(cls, value: list[int]) -> list[int]:
        if not value or any(item < 1 or item > 100 for item in value):
            raise ValueError("time_series lags must be non-empty integers in 1..100")
        return value

    @field_validator("windows")
    @classmethod
    def _valid_windows(cls, value: list[int]) -> list[int]:
        if not value or any(item < 2 or item > 100 for item in value):
            raise ValueError("time_series windows must be non-empty integers in 2..100")
        return value

    @model_validator(mode="after")
    def _has_output(self) -> TimeSeriesFeatureConfig:
        if not self.include_elapsed and not self.include_diff and not self.lags and not self.windows:
            raise ValueError("time_series must produce at least one feature")
        return self


class FeatureSelectionConfig(StrictModel):
    """Optional final feature-selection step.

    ``variance`` is label-free and uses ``min_variance``/``max_features``.
    ``mutual_info`` is fitted on the outer training fold only and ranks
    features with mutual information; ``min_variance`` is ignored there.
    ``model_importance`` fits a small ExtraTrees model on the outer training
    fold and keeps the highest-importance features; ``min_variance`` is ignored
    there too.
    """

    enabled: bool = False
    method: Literal["variance", "mutual_info", "model_importance"] = "variance"
    max_features: int = Field(default=50, ge=1, le=5000)
    min_variance: float = Field(default=0.0, ge=0.0)


class FeatureBlockConfig(StrictModel):
    """One declarative feature-engineering block of a pipeline plan.

    ``type`` names a block from the feature block registry; ``params`` carries
    block-specific options; ``enabled`` lets a plan keep disabled blocks for
    later activation.  Unknown block types are rejected during plan validation,
    not here, so error messages can list the allowed catalogue.
    """

    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class FeatureConfig(StrictModel):
    target_encode: list[str] = Field(default_factory=list)
    frequency_encode: list[str] = Field(default_factory=list)
    auto_encode: bool = True
    scale: Literal["auto", "none", "standard"] = "auto"
    impute: Literal["median", "most_frequent", "constant"] = "median"
    drop_constant: bool = True
    max_categories: int = Field(default=50, ge=2, le=5000)
    # Label-free aggregates over numeric features.  They are computed row-wise
    # and therefore need no fitted state.
    row_statistics: bool = False
    row_statistics_stats: list[Literal["mean", "std", "min", "max", "median"]] = Field(
        default_factory=lambda: ["mean", "std", "min", "max"]
    )
    # Opt-in modulo/quotient views for integer-like identifiers.  The caller
    # must decide that the selected numeric columns are identifier-like.
    modulo_features: bool = False
    modulo_moduli: list[int] = Field(default_factory=lambda: [10, 100])
    modulo_steps: list[int] = Field(default_factory=lambda: [5])
    knn_features: KNNFeatureConfig = Field(default_factory=KNNFeatureConfig)
    # Optional final selector applied to the assembled matrix.  It is fitted
    # inside the outer training fold; validation/test rows only use the
    # selected column names.
    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)
    # Pairwise categorical interactions.  Frequency encoding is label-free;
    # target encoding is fitted with the same inner-OOF guard as the regular
    # target encoder.
    interactions: list[InteractionConfig] = Field(default_factory=list)
    # Label-free missingness / row-profile features.
    missingness: MissingnessConfig = Field(default_factory=MissingnessConfig)
    # Label-free sin/cos views of parsed datetime columns.
    datetime_cyclical: DatetimeCyclicalConfig = Field(default_factory=DatetimeCyclicalConfig)
    # Label-free pairwise numeric interactions.
    numeric_interactions: NumericInteractionConfig = Field(
        default_factory=NumericInteractionConfig
    )
    # Fit-in-fold unsupervised cluster labels/distances.
    clustering: ClusterFeatureConfig = Field(default_factory=ClusterFeatureConfig)
    # Fit-in-fold IsolationForest anomaly score/flag.
    anomaly: AnomalyFeatureConfig = Field(default_factory=AnomalyFeatureConfig)
    # Past-only grouped lags, rolling statistics and elapsed-time features.
    time_series: TimeSeriesFeatureConfig = Field(default_factory=TimeSeriesFeatureConfig)
    # Stable, label-free hash encoding for high-cardinality categorical
    # columns.  The mapping is deterministic and needs no fitted vocabulary;
    # ``hash_features`` is the number of shared hash buckets.
    hash_encode: list[str] = Field(default_factory=list)
    hash_features: int = Field(default=16, ge=1, le=1024)
    # A reviewed, fold-aware feature plan (for example group aggregates).  It
    # is applied inside FeatureBuilder so validation/test rows never influence
    # the fitted mappings.
    fold_plan: dict[str, Any] | None = None
    # Optional ordered feature-block plan for `kind: pipeline` recipes.  When
    # absent, FeatureBuilder builds the canonical legacy plan from the flags
    # above and old configs keep hashing exactly as before.
    feature_blocks: list[FeatureBlockConfig] | None = None
    # Target smoothing for the target encoder; ``None`` keeps the legacy
    # smoothing of 10.0.
    target_smoothing: float | None = Field(default=None, gt=0)

    @field_validator("row_statistics_stats")
    @classmethod
    def _dedupe_row_statistics(
        cls, value: list[Literal["mean", "std", "min", "max", "median"]]
    ) -> list[Literal["mean", "std", "min", "max", "median"]]:
        return list(dict.fromkeys(value))

    @field_validator("hash_encode")
    @classmethod
    def _dedupe_hash_columns(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @field_validator("interactions")
    @classmethod
    def _dedupe_interactions(cls, value: list[InteractionConfig]) -> list[InteractionConfig]:
        seen: set[tuple[tuple[str, ...], str]] = set()
        for item in value:
            key = (tuple(item.columns), item.encoding)
            if key in seen:
                raise ValueError("duplicate categorical interaction")
            seen.add(key)
        return value

    @field_validator("modulo_moduli", "modulo_steps")
    @classmethod
    def _positive_modulo_values(cls, value: list[int]) -> list[int]:
        if any(item <= 0 for item in value):
            raise ValueError("modulo values must be positive integers")
        return list(dict.fromkeys(value))


class ModelConfig(StrictModel):
    name: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    seeds: int = Field(default=1, ge=1, le=10)
    early_stopping_rounds: int = Field(default=100, ge=0, le=5000)


class HPOConfig(StrictModel):
    enabled: bool = False
    trials: int = Field(default=20, ge=1, le=500)
    timeout_sec: int = Field(default=600, ge=10, le=86400)
    inner_size: float = Field(default=0.2, ge=0.1, le=0.5)


class EnsembleConfig(StrictModel):
    method: Literal["auto", "best_single", "mean", "rank_mean", "weighted", "stack"] = "auto"
    fallback: Literal["best_single", "mean"] = "best_single"
    min_improvement: float = Field(default=0.0, ge=0.0)
    meta_model: Literal["auto", "logreg", "ridge", "lgbm"] = "auto"
    meta_splits: int = Field(default=5, ge=2, le=20)


class SubmitConfig(StrictModel):
    id_col: str | None = None
    target_col: str | None = None
    clip: tuple[float, float] | None = None
    round: int | None = Field(default=None, ge=0, le=10)

    @field_validator("clip")
    @classmethod
    def _clip_order(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and value[0] > value[1]:
            raise ValueError("submit.clip must be [low, high] with low <= high")
        return value


class RuntimeConfig(StrictModel):
    """Resolved 1-5 runtime preset stored with a generated project."""

    profile: int = Field(ge=1, le=5)
    device: Literal["auto", "cpu", "gpu", "cuda"] = "auto"
    ram_gb: int | None = Field(default=None, ge=1)
    vram_gb: int | None = Field(default=None, ge=0)
    cpu_threads: int | None = Field(default=None, ge=0)
    # Reliability controls.  Zero disables a timeout; isolation is opt-in so
    # existing tests, plugins and monkeypatched adapters keep working in-process.
    stage_timeout_sec: int = Field(default=0, ge=0, le=86_400)
    model_timeout_sec: int = Field(default=0, ge=0, le=86_400)
    model_retries: int = Field(default=0, ge=0, le=3)
    fallback_models: list[str] = Field(default_factory=list)
    isolate_models: bool = False
    feature_cache: bool = True


class Config(StrictModel):
    schema_version: int = 1
    project: str
    seed: int = 42
    recipe: str = "pesnik/baseline"
    recipe_kind: Literal["preset", "pipeline"] | None = None
    strategy: StrategyChoice = "fast"
    task: TaskChoice = "auto"
    metric: MetricChoice = "auto"
    data: DataConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: list[ModelConfig] = Field(default_factory=lambda: [ModelConfig(name="lgbm")])
    hpo: HPOConfig = Field(default_factory=HPOConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    submit: SubmitConfig = Field(default_factory=SubmitConfig)
    runtime: RuntimeConfig | None = None

    @field_validator("models")
    @classmethod
    def _unique_enabled_models(cls, value: list[ModelConfig]) -> list[ModelConfig]:
        names = [m.name for m in value if m.enabled]
        if not names:
            raise ValueError("At least one model must be enabled")
        if len(names) != len(set(names)):
            raise ValueError("Model names must be unique")
        return value


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config not found: {path}", "Run 'kconv new' or pass --data to create one.")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    try:
        config = Config.model_validate(raw)
        # Lazy import avoids a config <-> models import cycle at module load.
        from kconv.models.validation import validate_config_models

        validate_config_models(config)
        return config
    except ValidationError as exc:
        # Keep the message short and actionable; pydantic's full dump is too noisy.
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        msg = first.get("msg", "invalid config")
        raise ConfigError(
            f"Invalid config at '{loc}': {msg}",
            "Run 'kconv config explain' or see docs/config.md.",
        ) from exc


def save_config(config: Config, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json", exclude_none=True)
    from kconv.atomic import write_text_atomic

    return write_text_atomic(
        path,
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    )


def apply_overrides(config: Config, **overrides: Any) -> Config:
    """Return a copy with only explicitly supplied values changed."""
    raw = config.model_dump(mode="python")
    for key, value in overrides.items():
        if value is not None:
            raw[key] = value
    return Config.model_validate(raw)
