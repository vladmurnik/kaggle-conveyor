"""Feature block registry and declarative plan validation.

Phase 2 of the recipe/pipeline plan: a resolved :class:`~kconv.config.Config`
may carry an optional, ordered list of feature blocks.  This module owns the
block catalogue, lookup and the static validation rules that run before a run
starts.  Execution (``FeatureBuilder``), recipe compilation and CLI/API wiring
are owned by other phases and stay untouched here.
"""

from __future__ import annotations

from dataclasses import dataclass

from kconv.config import Config, FeatureBlockConfig
from kconv.contracts import DatasetSpec
from kconv.errors import DataError, RecipeError
from kconv.features.group_stats import build_group_statistics_plan
from kconv.features.interactions import interaction_pairs

BASE_BLOCK = "base"
FEATURE_SELECTION_BLOCK = "feature_selection"
HASH_BLOCK = "hash_encoding"
KNN_BLOCK = "knn_class_distance"
GROUP_STATISTICS_BLOCK = "group_statistics"
INTERACTION_BLOCK = "categorical_interaction"
MISSINGNESS_BLOCK = "missingness"
DATETIME_CYCLICAL_BLOCK = "datetime_cyclical"
NUMERIC_INTERACTION_BLOCK = "numeric_interactions"
CLUSTER_BLOCK = "clustering"
ANOMALY_BLOCK = "anomaly"
TIME_SERIES_BLOCK = "time_series"
CLASSIFICATION_TASKS = frozenset({"binary", "multiclass"})


@dataclass(frozen=True)
class BlockSpec:
    """Static metadata for one feature block type."""

    type: str
    required_params: tuple[str, ...] = ()
    allowed_params: tuple[str, ...] = ()
    task_constraints: tuple[str, ...] = ()
    resource_hints: tuple[str, ...] = ("cpu",)
    description: str = ""


BLOCK_REGISTRY: dict[str, BlockSpec] = {
    "base": BlockSpec(
        type="base",
        allowed_params=("auto_encode", "drop_constant", "impute", "max_categories", "scale"),
        task_constraints=("at_least_one_feature_column",),
        resource_hints=("cpu",),
        description="Baseline numeric/categorical matrix with imputation, scaling and drop-constant.",
    ),
    "target_encoding": BlockSpec(
        type="target_encoding",
        required_params=("columns",),
        allowed_params=("columns", "smoothing"),
        task_constraints=("fit_in_outer_fold",),
        resource_hints=("cpu",),
        description="Target encoding with inner out-of-fold splits and smoothing.",
    ),
    "frequency_encoding": BlockSpec(
        type="frequency_encoding",
        required_params=("columns",),
        allowed_params=("columns",),
        task_constraints=("fit_in_outer_fold",),
        resource_hints=("cpu",),
        description="Frequency encoding of categorical columns from the outer training fold.",
    ),
    "hash_encoding": BlockSpec(
        type="hash_encoding",
        required_params=("columns",),
        allowed_params=("columns", "n_features"),
        task_constraints=("label_free",),
        resource_hints=("cpu",),
        description="Stable label-free hash buckets for high-cardinality categorical columns.",
    ),
    "group_statistics": BlockSpec(
        type="group_statistics",
        required_params=("group_col",),
        allowed_params=("columns", "group_col", "max_columns", "stats"),
        task_constraints=("fit_in_outer_fold",),
        resource_hints=("cpu",),
        description="Fit-in-fold group mean/std/count aggregates compiled to a reviewed fold plan.",
    ),
    "categorical_interaction": BlockSpec(
        type="categorical_interaction",
        required_params=("columns",),
        allowed_params=("columns", "encoding", "max_pairs"),
        task_constraints=("fit_in_outer_fold", "categorical_columns"),
        resource_hints=("cpu",),
        description="Pairwise categorical interactions encoded by fold-fitted frequency or target statistics.",
    ),
    "missingness": BlockSpec(
        type="missingness",
        allowed_params=("columns", "indicators", "row_count", "zero_count"),
        task_constraints=("label_free",),
        resource_hints=("cpu",),
        description="Label-free missing indicators, row missing count and numeric zero count.",
    ),
    "datetime_cyclical": BlockSpec(
        type="datetime_cyclical",
        allowed_params=("columns", "periods"),
        task_constraints=("label_free",),
        resource_hints=("cpu",),
        description="Label-free sin/cos cyclical views of parsed datetime columns.",
    ),
    "numeric_interactions": BlockSpec(
        type="numeric_interactions",
        allowed_params=("clip_value", "columns", "max_pairs", "operations"),
        task_constraints=("label_free", "numeric_columns"),
        resource_hints=("cpu",),
        description="Label-free pairwise numeric products, ratios, differences and sums.",
    ),
    "clustering": BlockSpec(
        type="clustering",
        allowed_params=(
            "columns",
            "include_distances",
            "include_labels",
            "max_fit_rows",
            "max_iter",
            "n_clusters",
        ),
        task_constraints=("fit_in_outer_fold", "numeric_columns"),
        resource_hints=("cpu", "ram"),
        description="Fit-in-fold MiniBatchKMeans cluster labels and centroid distances.",
    ),
    "anomaly": BlockSpec(
        type="anomaly",
        allowed_params=("columns", "contamination", "include_flag", "n_estimators"),
        task_constraints=("fit_in_outer_fold", "numeric_columns"),
        resource_hints=("cpu", "ram"),
        description="Fit-in-fold IsolationForest anomaly score and optional flag.",
    ),
    "time_series": BlockSpec(
        type="time_series",
        required_params=("time_col",),
        allowed_params=(
            "columns",
            "group_col",
            "include_diff",
            "include_elapsed",
            "lags",
            "max_history_rows",
            "stats",
            "time_col",
            "windows",
        ),
        task_constraints=("datetime_columns", "fit_in_outer_fold", "numeric_columns"),
        resource_hints=("cpu", "ram"),
        description="Past-only grouped lags, rolling statistics and elapsed-time features.",
    ),
    "row_statistics": BlockSpec(
        type="row_statistics",
        required_params=("stats",),
        allowed_params=("stats",),
        task_constraints=("numeric_columns",),
        resource_hints=("cpu",),
        description="Label-free row-wise statistics over numeric columns.",
    ),
    "modulo": BlockSpec(
        type="modulo",
        required_params=("moduli", "steps"),
        allowed_params=("moduli", "steps"),
        task_constraints=("numeric_columns",),
        resource_hints=("cpu",),
        description="Modulo/quotient views for identifier-like numeric columns.",
    ),
    "fold_plan": BlockSpec(
        type="fold_plan",
        required_params=("plan",),
        allowed_params=("plan",),
        task_constraints=("fold_aware",),
        resource_hints=("cpu",),
        description="Reviewed fold-aware feature plan.",
    ),
    "feature_selection": BlockSpec(
        type="feature_selection",
        allowed_params=("max_features", "method", "min_variance"),
        task_constraints=("fit_in_outer_fold", "post_features"),
        resource_hints=("cpu",),
        description="Final variance, mutual-information or model-importance feature selection.",
    ),
    "knn_class_distance": BlockSpec(
        type="knn_class_distance",
        allowed_params=("columns", "folds", "max_reference_rows", "metric", "neighbors"),
        task_constraints=("classification_only", "numeric_columns", "fit_in_outer_fold"),
        resource_hints=("cpu", "ram", "cpu-heavy", "classification-only"),
        description="Class-conditional kNN distance features.",
    ),
}

SINGLETON_BLOCKS: frozenset[str] = frozenset(BLOCK_REGISTRY)


def block_types() -> list[str]:
    """Return the sorted catalogue of known feature block types."""
    return sorted(BLOCK_REGISTRY)


def get_block_spec(block_type: str | FeatureBlockConfig) -> BlockSpec:
    """Return the catalog entry for a block type or fail with the allowed list."""
    key = block_type.type if isinstance(block_type, FeatureBlockConfig) else str(block_type)
    spec = BLOCK_REGISTRY.get(key)
    if spec is None:
        raise RecipeError(
            f"Unknown feature block type '{key}'.",
            f"Allowed feature block types: {', '.join(block_types())}.",
        )
    return spec


def _explicit_encoding_columns(block: FeatureBlockConfig, spec: DatasetSpec) -> list[str]:
    """Validate explicit encoding columns before a run starts."""
    raw = block.params.get("columns")
    if raw is None or raw == "auto":
        return []
    if isinstance(raw, str):
        columns = [raw]
    elif isinstance(raw, (list, tuple)):
        columns = [str(column) for column in raw]
    else:
        raise RecipeError(
            f"Feature block '{block.type}' columns must be 'auto' or a list of names.",
            "Use columns: auto or columns: [name, ...].",
        )
    allowed = set(spec.feature_columns) - {spec.target, spec.id_col}
    unknown = [column for column in columns if column not in allowed]
    if unknown:
        raise RecipeError(
            f"Feature block '{block.type}' references unknown column(s): {', '.join(unknown)}.",
            "Use a column from the dataset or columns: auto.",
        )
    return columns


def _fold_plan_is_empty(plan: object) -> bool:
    """Return whether a reviewed fold plan contains no proposals."""
    if not plan:
        return True
    if isinstance(plan, dict):
        proposals = plan.get("features") or plan.get("proposals")
        return not proposals
    return False


def validate_block_plan(config: Config, spec: DatasetSpec) -> list[FeatureBlockConfig]:
    """Validate an optional feature-block plan and return it.

    ``None`` means "no explicit plan": the legacy flag-driven feature config is
    used and no plan validation applies.  An explicit plan must be a non-empty
    sequence whose first block is an enabled ``base`` block, may not repeat
    singleton blocks and must satisfy task-level constraints of its enabled
    blocks.
    """
    blocks = config.features.feature_blocks
    if blocks is None:
        return []
    if not blocks:
        raise RecipeError(
            "Feature block plan must not be empty.",
            "Add a 'base' block as the first block of the plan.",
        )

    counts: dict[str, int] = {}
    for block in blocks:
        get_block_spec(block.type)
        counts[block.type] = counts.get(block.type, 0) + 1
    for block_type, count in sorted(counts.items()):
        if count > 1 and block_type in SINGLETON_BLOCKS:
            raise DataError(
                f"Feature block '{block_type}' must appear exactly once in a block plan.",
                f"Remove the duplicate entry; allowed feature block types: {', '.join(block_types())}.",
            )

    if BASE_BLOCK not in counts or blocks[0].type != BASE_BLOCK:
        raise RecipeError(
            f"Feature block plan must start with the '{BASE_BLOCK}' block.",
            f"First block must be '{BASE_BLOCK}'; allowed feature block types: {', '.join(block_types())}.",
        )
    if not blocks[0].enabled:
        raise RecipeError(
            f"Feature block '{BASE_BLOCK}' must be enabled.",
            "Set enabled: true for the first block of the plan.",
        )

    enabled_blocks = [block for block in blocks if block.enabled]
    enabled_types = {block.type for block in enabled_blocks}
    if {GROUP_STATISTICS_BLOCK, "fold_plan"} <= enabled_types:
        raise RecipeError(
            "Feature blocks 'group_statistics' and 'fold_plan' both define fold aggregates.",
            "Use one of them: group_statistics compiles to a reviewed fold plan automatically.",
        )
    if (
        any(block.type == FEATURE_SELECTION_BLOCK for block in enabled_blocks)
        and enabled_blocks[-1].type != FEATURE_SELECTION_BLOCK
    ):
        raise RecipeError(
            f"Feature block '{FEATURE_SELECTION_BLOCK}' must be the last enabled block.",
            "Move feature selection to the end of the block plan.",
        )

    feature_columns = [
        column for column in spec.feature_columns if column not in {spec.target, spec.id_col}
    ]
    numeric_features = [column for column in spec.numeric_columns if column in feature_columns]

    for block in blocks:
        if not block.enabled:
            continue
        block_spec = get_block_spec(block.type)
        unknown = sorted(set(block.params) - set(block_spec.allowed_params))
        if unknown:
            raise RecipeError(
                f"Feature block '{block.type}' has unknown param(s): {', '.join(unknown)}.",
                f"Allowed params for feature block '{block.type}': "
                f"{', '.join(block_spec.allowed_params) or '<none>'}.",
            )
        missing = [name for name in block_spec.required_params if name not in block.params]
        if missing:
            raise RecipeError(
                f"Feature block '{block.type}' is missing required param(s): {', '.join(missing)}.",
                f"Provide the required params for feature block '{block.type}': {', '.join(block_spec.required_params)}.",
            )
        if block.type == BASE_BLOCK and not feature_columns:
            raise DataError(
                f"Feature block '{BASE_BLOCK}' requires at least one feature column.",
                "Add a feature column to the dataset or remove the explicit block plan.",
            )
        if block.type in {"target_encoding", "frequency_encoding", HASH_BLOCK}:
            _explicit_encoding_columns(block, spec)
        if block.type in {"row_statistics", "modulo"} and not numeric_features:
            raise DataError(
                f"Feature block '{block.type}' requires at least one numeric feature column.",
                f"Add a numeric column or disable the {block.type} block.",
            )
        if block.type == "fold_plan" and _fold_plan_is_empty(block.params.get("plan")):
            raise RecipeError(
                "Feature block 'fold_plan' requires a non-empty plan.",
                "Review and attach a plan with at least one feature proposal.",
            )
        if block.type == GROUP_STATISTICS_BLOCK:
            try:
                build_group_statistics_plan(block.params, spec)
            except ValueError as exc:
                raise RecipeError(
                    f"Feature block '{GROUP_STATISTICS_BLOCK}' is invalid: {exc}",
                    "Use group_col, numeric columns, and stats from mean/std/count.",
                ) from exc
        if block.type == INTERACTION_BLOCK:
            raw_columns = block.params.get("columns")
            columns = _explicit_encoding_columns(block, spec)
            if raw_columns not in (None, "auto"):
                if len(columns) < 2:
                    raise RecipeError(
                        "Feature block 'categorical_interaction' needs at least two columns.",
                        "Provide two or more categorical columns or use columns: auto.",
                    )
                non_categorical = [
                    column for column in columns if column not in spec.categorical_columns
                ]
                if non_categorical:
                    raise RecipeError(
                        "Feature block 'categorical_interaction' requires categorical columns; "
                        f"got {', '.join(non_categorical)}.",
                        "Use categorical columns or remove the block.",
                    )
            else:
                columns = [
                    column for column in spec.categorical_columns if column in feature_columns
                ]
            max_pairs = block.params.get("max_pairs", 4)
            if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or not 1 <= max_pairs <= 50:
                raise RecipeError(
                    "Feature block 'categorical_interaction' max_pairs must be an integer in 1..50.",
                    "Use a small number of pairs, for example max_pairs: 4.",
                )
            encoding = block.params.get("encoding", "frequency")
            if encoding not in {"frequency", "target"}:
                raise RecipeError(
                    f"Feature block 'categorical_interaction' has unknown encoding '{encoding}'.",
                    "Use encoding: frequency or encoding: target.",
                )
            interaction_pairs(columns, max_pairs)
        if block.type == MISSINGNESS_BLOCK:
            columns = _explicit_encoding_columns(block, spec)
            for name in ("indicators", "row_count", "zero_count"):
                value = block.params.get(name)
                if value is not None and not isinstance(value, bool):
                    raise RecipeError(
                        f"Feature block 'missingness' {name} must be a boolean.",
                        "Use true or false.",
                    )
            zero_count = block.params.get("zero_count", False)
            if zero_count and not numeric_features:
                raise DataError(
                    "Feature block 'missingness' zero_count requires a numeric feature column.",
                    "Disable zero_count or add a numeric column.",
                )
            if zero_count and columns and not any(column in numeric_features for column in columns):
                raise DataError(
                    "Feature block 'missingness' zero_count has no selected numeric column.",
                    "Include a numeric column in columns or disable zero_count.",
                )
        if block.type == DATETIME_CYCLICAL_BLOCK:
            raw_columns = block.params.get("columns")
            columns = _explicit_encoding_columns(block, spec)
            if raw_columns not in (None, "auto"):
                if not columns:
                    raise RecipeError(
                        "Feature block 'datetime_cyclical' needs at least one datetime column.",
                        "Provide a datetime column or use columns: auto.",
                    )
                non_datetime = [
                    column for column in columns if column not in spec.datetime_columns
                ]
                if non_datetime:
                    raise RecipeError(
                        "Feature block 'datetime_cyclical' requires datetime columns; "
                        f"got {', '.join(non_datetime)}.",
                        "Use datetime columns or remove the block.",
                    )
            periods = block.params.get(
                "periods", ["month", "dayofweek", "dayofyear", "hour"]
            )
            if not isinstance(periods, (list, tuple)) or not periods:
                raise RecipeError(
                    "Feature block 'datetime_cyclical' periods must be a non-empty list.",
                    "Use month, dayofweek, dayofyear and/or hour.",
                )
            allowed_periods = {"month", "dayofweek", "dayofyear", "hour"}
            unknown_periods = [period for period in periods if period not in allowed_periods]
            if unknown_periods:
                raise RecipeError(
                    "Feature block 'datetime_cyclical' has unknown periods: "
                    f"{', '.join(map(str, unknown_periods))}.",
                    "Use month, dayofweek, dayofyear or hour.",
                )
        if block.type == HASH_BLOCK:
            n_features = block.params.get("n_features", 16)
            if (
                isinstance(n_features, bool)
                or not isinstance(n_features, int)
                or not 1 <= n_features <= 1024
            ):
                raise RecipeError(
                    f"Feature block '{HASH_BLOCK}' n_features must be an integer in 1..1024.",
                    "Use a small number of hash buckets, for example n_features: 16.",
                )
        if block.type == FEATURE_SELECTION_BLOCK:
            method = block.params.get("method", "variance")
            if method not in {"variance", "mutual_info", "model_importance"}:
                raise RecipeError(
                    f"Feature block '{FEATURE_SELECTION_BLOCK}' has unknown method '{method}'.",
                    "Use method: variance, mutual_info or model_importance.",
                )
            max_features = block.params.get("max_features", 50)
            if isinstance(max_features, bool) or not isinstance(max_features, int) or not 1 <= max_features <= 5000:
                raise RecipeError(
                    f"Feature block '{FEATURE_SELECTION_BLOCK}' max_features must be an integer in 1..5000.",
                    "Set a positive max_features value.",
                )
            min_variance = block.params.get("min_variance", 0.0)
            if (
                isinstance(min_variance, bool)
                or not isinstance(min_variance, (int, float))
                or min_variance < 0
            ):
                raise RecipeError(
                    f"Feature block '{FEATURE_SELECTION_BLOCK}' min_variance must be a non-negative number.",
                    "Use min_variance: 0 or a positive threshold.",
                )
        if block.type == KNN_BLOCK:
            if spec.task not in CLASSIFICATION_TASKS:
                raise DataError(
                    f"Feature block '{KNN_BLOCK}' is classification-only; task '{spec.task}' does not support it.",
                    "Disable the knn_class_distance block or use a binary/multiclass task.",
                )
            if not numeric_features:
                raise DataError(
                    f"Feature block '{KNN_BLOCK}' requires at least one numeric feature column.",
                    "Add a numeric column or disable the knn_class_distance block.",
                )
        if block.type == NUMERIC_INTERACTION_BLOCK:
            raw = block.params.get("columns")
            columns = (
                numeric_features
                if raw in (None, "auto", [])
                else _explicit_encoding_columns(block, spec)
            )
            non_numeric = [column for column in columns if column not in numeric_features]
            if non_numeric:
                raise RecipeError(
                    "Feature block 'numeric_interactions' requires numeric columns; "
                    f"got {', '.join(non_numeric)}.",
                    "Use numeric columns or remove the block.",
                )
            if not columns:
                raise DataError(
                    "Feature block 'numeric_interactions' requires at least one numeric feature column.",
                    "Add a numeric column or disable the numeric_interactions block.",
                )
            max_pairs = block.params.get("max_pairs", 6)
            if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or not 1 <= max_pairs <= 50:
                raise RecipeError(
                    "Feature block 'numeric_interactions' max_pairs must be an integer in 1..50.",
                    "Use max_pairs: 6 or a smaller value.",
                )
            operations = block.params.get("operations", ["product", "ratio"])
            if not isinstance(operations, (list, tuple)) or not operations:
                raise RecipeError(
                    "Feature block 'numeric_interactions' operations must be a non-empty list.",
                    "Use product, ratio, difference and/or sum.",
                )
            unknown_operations = [
                operation
                for operation in operations
                if operation not in {"product", "ratio", "difference", "sum"}
            ]
            if unknown_operations:
                raise RecipeError(
                    "Feature block 'numeric_interactions' has unknown operations: "
                    f"{', '.join(map(str, unknown_operations))}.",
                    "Use product, ratio, difference or sum.",
                )
            clip_value = block.params.get("clip_value", 1e12)
            if (
                isinstance(clip_value, bool)
                or not isinstance(clip_value, (int, float))
                or float(clip_value) <= 0
            ):
                raise RecipeError(
                    "Feature block 'numeric_interactions' clip_value must be positive.",
                    "Use a finite positive clip_value, for example 1e12.",
                )
        if block.type == CLUSTER_BLOCK:
            raw = block.params.get("columns")
            columns = (
                numeric_features
                if raw in (None, "auto", [])
                else _explicit_encoding_columns(block, spec)
            )
            non_numeric = [column for column in columns if column not in numeric_features]
            if non_numeric:
                raise RecipeError(
                    "Feature block 'clustering' requires numeric columns; "
                    f"got {', '.join(non_numeric)}.",
                    "Use numeric columns or remove the block.",
                )
            if not columns:
                raise DataError(
                    "Feature block 'clustering' requires at least one numeric feature column.",
                    "Add a numeric column or disable the clustering block.",
                )
            n_clusters = block.params.get("n_clusters", 8)
            if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or not 2 <= n_clusters <= 50:
                raise RecipeError(
                    "Feature block 'clustering' n_clusters must be an integer in 2..50.",
                    "Use n_clusters: 8 or a smaller value.",
                )
            max_iter = block.params.get("max_iter", 100)
            if isinstance(max_iter, bool) or not isinstance(max_iter, int) or not 10 <= max_iter <= 1000:
                raise RecipeError(
                    "Feature block 'clustering' max_iter must be an integer in 10..1000.",
                    "Use max_iter: 100.",
                )
            max_fit_rows = block.params.get("max_fit_rows", 50_000)
            if (
                isinstance(max_fit_rows, bool)
                or not isinstance(max_fit_rows, int)
                or not 100 <= max_fit_rows <= 1_000_000
            ):
                raise RecipeError(
                    "Feature block 'clustering' max_fit_rows must be an integer in 100..1000000.",
                    "Use max_fit_rows: 50000.",
                )
            include_distances = block.params.get("include_distances", True)
            include_labels = block.params.get("include_labels", True)
            if not isinstance(include_distances, bool) or not isinstance(include_labels, bool):
                raise RecipeError(
                    "Feature block 'clustering' include_distances/include_labels must be booleans.",
                    "Use true or false.",
                )
            if not include_distances and not include_labels:
                raise RecipeError(
                    "Feature block 'clustering' must include distances, labels, or both.",
                    "Enable include_distances or include_labels.",
                )
        if block.type == ANOMALY_BLOCK:
            raw = block.params.get("columns")
            columns = (
                numeric_features
                if raw in (None, "auto", [])
                else _explicit_encoding_columns(block, spec)
            )
            non_numeric = [column for column in columns if column not in numeric_features]
            if non_numeric:
                raise RecipeError(
                    "Feature block 'anomaly' requires numeric columns; "
                    f"got {', '.join(non_numeric)}.",
                    "Use numeric columns or remove the block.",
                )
            if not columns:
                raise DataError(
                    "Feature block 'anomaly' requires at least one numeric feature column.",
                    "Add a numeric column or disable the anomaly block.",
                )
            n_estimators = block.params.get("n_estimators", 100)
            if (
                isinstance(n_estimators, bool)
                or not isinstance(n_estimators, int)
                or not 10 <= n_estimators <= 500
            ):
                raise RecipeError(
                    "Feature block 'anomaly' n_estimators must be an integer in 10..500.",
                    "Use n_estimators: 100.",
                )
            contamination = block.params.get("contamination", "auto")
            if contamination != "auto" and (
                isinstance(contamination, bool)
                or not isinstance(contamination, (int, float))
                or not 0.0 < float(contamination) <= 0.5
            ):
                raise RecipeError(
                    "Feature block 'anomaly' contamination must be 'auto' or in (0, 0.5].",
                    "Use contamination: auto or a small fraction such as 0.05.",
                )
            include_flag = block.params.get("include_flag", True)
            if not isinstance(include_flag, bool):
                raise RecipeError(
                    "Feature block 'anomaly' include_flag must be a boolean.",
                    "Use include_flag: true or false.",
                )
        if block.type == TIME_SERIES_BLOCK:
            time_col = block.params.get("time_col")
            if time_col in (None, "auto"):
                datetime_features = [
                    column for column in spec.datetime_columns if column in feature_columns
                ]
                if not datetime_features:
                    raise DataError(
                        "Feature block 'time_series' needs a datetime feature column.",
                        "Add a datetime column or remove the time_series block.",
                    )
            elif not isinstance(time_col, str) or time_col not in spec.datetime_columns:
                raise RecipeError(
                    "Feature block 'time_series' time_col must be 'auto' or a datetime column.",
                    "Use time_col: auto or a column listed in spec.datetime_columns.",
                )
            group_col = block.params.get("group_col")
            if group_col not in (None, "auto") and group_col not in feature_columns:
                raise RecipeError(
                    "Feature block 'time_series' group_col must be a feature column or auto.",
                    "Use group_col: auto or a feature column name.",
                )
            if (
                group_col not in (None, "auto")
                and time_col not in (None, "auto")
                and group_col == time_col
            ):
                raise RecipeError(
                    "Feature block 'time_series' group_col and time_col must differ.",
                    "Use a categorical group column or remove group_col.",
                )
            raw = block.params.get("columns")
            if raw in (None, "auto", []):
                columns = [
                    column
                    for column in numeric_features
                    if group_col in (None, "auto") or column != group_col
                ]
            else:
                columns = _explicit_encoding_columns(block, spec)
            non_numeric = [column for column in columns if column not in numeric_features]
            if non_numeric:
                raise RecipeError(
                    "Feature block 'time_series' requires numeric columns; "
                    f"got {', '.join(non_numeric)}.",
                    "Use numeric columns or remove the block.",
                )
            forbidden = [
                column
                for column in (time_col, group_col)
                if column not in (None, "auto") and column in columns
            ]
            if forbidden:
                raise RecipeError(
                    "Feature block 'time_series' columns must not include time_col or "
                    f"group_col: {', '.join(forbidden)}.",
                    "Use value columns only; time_col and group_col are separate.",
                )
            if not columns:
                raise DataError(
                    "Feature block 'time_series' requires at least one numeric feature column.",
                    "Add a numeric column or disable the time_series block.",
                )
            lags = block.params.get("lags", [1])
            if (
                not isinstance(lags, (list, tuple))
                or not lags
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or not 1 <= item <= 100
                    for item in lags
                )
            ):
                raise RecipeError(
                    "Feature block 'time_series' lags must be a non-empty list of integers in 1..100.",
                    "Use lags: [1, 2].",
                )
            windows = block.params.get("windows", [3])
            if (
                not isinstance(windows, (list, tuple))
                or not windows
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or not 2 <= item <= 100
                    for item in windows
                )
            ):
                raise RecipeError(
                    "Feature block 'time_series' windows must be a non-empty list of integers in 2..100.",
                    "Use windows: [3, 7].",
                )
            stats = block.params.get("stats", ["mean"])
            if not isinstance(stats, (list, tuple)) or not stats:
                raise RecipeError(
                    "Feature block 'time_series' stats must be a non-empty list.",
                    "Use mean, std, min and/or max.",
                )
            unknown_stats = [stat for stat in stats if stat not in {"mean", "std", "min", "max"}]
            if unknown_stats:
                raise RecipeError(
                    "Feature block 'time_series' has unknown stats: "
                    f"{', '.join(map(str, unknown_stats))}.",
                    "Use mean, std, min or max.",
                )
            max_history_rows = block.params.get("max_history_rows", 10_000)
            if (
                isinstance(max_history_rows, bool)
                or not isinstance(max_history_rows, int)
                or not 100 <= max_history_rows <= 1_000_000
            ):
                raise RecipeError(
                    "Feature block 'time_series' max_history_rows must be an integer in 100..1000000.",
                    "Use max_history_rows: 10000.",
                )
            for name in ("include_elapsed", "include_diff"):
                value = block.params.get(name, True)
                if not isinstance(value, bool):
                    raise RecipeError(
                        f"Feature block 'time_series' {name} must be a boolean.",
                        "Use true or false.",
                    )

    return list(blocks)
