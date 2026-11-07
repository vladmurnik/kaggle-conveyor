"""Fit-in-fold feature engineering for tabular kconv workflows.

The builder is deliberately conservative.  It only learns mappings from the
outer training fold, and target encoding uses an inner out-of-fold scheme so a
row never contributes its own target to its own encoded feature.
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.model_selection import KFold, StratifiedKFold

from kconv.config import (
    AnomalyFeatureConfig,
    ClusterFeatureConfig,
    Config,
    DatetimeCyclicalConfig,
    FeatureConfig,
    FeatureSelectionConfig,
    InteractionConfig,
    KNNFeatureConfig,
    MissingnessConfig,
    NumericInteractionConfig,
    TimeSeriesFeatureConfig,
)
from kconv.contracts import DatasetSpec
from kconv.errors import DataError
from kconv.features.anomaly import AnomalyFeatureTransformer
from kconv.features.blocks import validate_block_plan
from kconv.features.clustering import ClusterFeatureTransformer
from kconv.features.fold_plan import FoldPlanTransformer
from kconv.features.group_stats import build_group_statistics_plan
from kconv.features.interactions import interaction_name, interaction_pairs
from kconv.features.knn import KNNClassDistanceTransformer
from kconv.features.numeric_interactions import NumericInteractionTransformer
from kconv.features.time_series import TimeSeriesFeatureTransformer


def _as_str_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__missing__").astype(str)


def _safe_numeric(series: pd.Series) -> pd.Series:
    # Infinities cannot be imputed by median/mean and most sklearn estimators
    # reject them.  Treat them like missing values so the fold-local imputer
    # produces a finite matrix for every model family.
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )


class FeatureBuilder:
    """Build a numeric matrix from a raw tabular fold.

    Parameters
    ----------
    config:
        Resolved pipeline config.
    spec:
        Dataset schema from ingest.
    family:
        ``tree`` and ``foundation`` use ordinal categoricals and no scaling;
        ``linear`` uses one-hot categoricals and standard scaling.
    """

    def __init__(self, config: Config, spec: DatasetSpec, family: str = "tree") -> None:
        self.config = config
        self.spec = spec
        self.family = family
        self.warnings: list[str] = []
        self.feature_names: list[str] = []
        self._fitted = False
        self._keep: list[str] = []
        self._numeric_fill: dict[str, float] = {}
        self._numeric_mean: dict[str, float] = {}
        self._numeric_std: dict[str, float] = {}
        self._cat_categories: dict[str, list[str]] = {}
        self._cat_maps: dict[str, dict[str, int]] = {}
        self._cat_fill: dict[str, str] = {}
        self._datetime_fill: dict[str, float] = {}
        self._text_fill: dict[str, float] = {}
        self._freq_maps: dict[str, dict[str, float]] = {}
        self._hash_encode: list[str] = []
        self._hash_features = 16
        self._te_maps: dict[str, dict[str, float]] = {}
        self._te_prior: dict[str, float] = {}
        self._te_oof: dict[str, np.ndarray] = {}
        self._fold_plan: FoldPlanTransformer | None = None
        self._knn: KNNClassDistanceTransformer | None = None
        self._knn_oof: pd.DataFrame | None = None
        self._interactions: list[InteractionConfig] = []
        self._interaction_maps: dict[str, dict[str, float]] = {}
        self._interaction_prior: dict[str, float] = {}
        self._interaction_oof: dict[str, np.ndarray] = {}
        self._missingness_columns: list[str] = []
        self._datetime_cyclical_columns: list[str] = []

        self._feature_columns = [c for c in spec.feature_columns if c not in {spec.target, spec.id_col}]
        self._numeric = [c for c in spec.numeric_columns if c in self._feature_columns]
        self._categorical = [c for c in spec.categorical_columns if c in self._feature_columns]
        self._datetime = [c for c in spec.datetime_columns if c in self._feature_columns]
        self._text = [c for c in spec.text_columns if c in self._feature_columns]
        # A role can be absent for manually written configs; unknown columns are
        # treated as categorical, which is the safest fallback for trees.
        known = set(self._numeric) | set(self._categorical) | set(self._datetime) | set(self._text)
        for column in self._feature_columns:
            if column not in known:
                self._categorical.append(column)

        # ``feature_blocks`` is optional and authoritative when present.  The
        # legacy flag-driven path stays untouched, so old configs keep their
        # exact feature matrix and config hash.
        self._features = self._resolve_features(config)
        self._feature_selection = self._features.feature_selection
        self._row_statistics = bool(self._features.row_statistics)
        self._row_statistics_stats = tuple(self._features.row_statistics_stats)
        self._modulo_features = bool(self._features.modulo_features)
        self._modulo_moduli = tuple(self._features.modulo_moduli)
        self._modulo_steps = tuple(self._features.modulo_steps)
        self._target_encode = [c for c in self._features.target_encode if c in self._feature_columns]
        if spec.task == "multiclass" and self._target_encode:
            self.warnings.append(
                "Target encoding is disabled for multiclass tasks: averaging class "
                "ordinals would encode an artificial class order."
            )
            self._target_encode = []
        self._frequency = [c for c in self._features.frequency_encode if c in self._feature_columns]
        self._hash_encode = [c for c in self._features.hash_encode if c in self._feature_columns]
        self._hash_features = int(self._features.hash_features)
        self._smoothing = 10.0 if self._features.target_smoothing is None else float(
            self._features.target_smoothing
        )
        if self._features.fold_plan:
            self._fold_plan = FoldPlanTransformer(
                self._features.fold_plan,
                columns=self._feature_columns,
            )

        self._interactions = list(self._features.interactions)
        if spec.task == "multiclass":
            skipped = [item for item in self._interactions if item.encoding == "target"]
            if skipped:
                self.warnings.append(
                    "Target-encoded categorical interactions are disabled for "
                    "multiclass tasks; use frequency encoding instead."
                )
                self._interactions = [item for item in self._interactions if item.encoding != "target"]
        for interaction in self._interactions:
            unknown = [column for column in interaction.columns if column not in self._feature_columns]
            if unknown:
                raise DataError(
                    "features.interactions references unknown columns: "
                    f"{', '.join(unknown)}.",
                    "Use columns from spec.feature_columns.",
                )
            non_categorical = [
                column for column in interaction.columns if column not in self._categorical
            ]
            if non_categorical:
                raise DataError(
                    "features.interactions columns must be categorical: "
                    f"{', '.join(non_categorical)}.",
                    "Use categorical columns or remove the interaction.",
                )

        self._missingness = self._features.missingness
        if self._missingness.enabled:
            selected = (
                list(self._missingness.columns)
                if self._missingness.columns
                else list(self._feature_columns)
            )
            unknown = [column for column in selected if column not in self._feature_columns]
            if unknown:
                raise DataError(
                    "features.missingness references unknown columns: "
                    f"{', '.join(unknown)}.",
                    "Use columns from spec.feature_columns.",
                )
            if self._missingness.zero_count and not [
                column for column in selected if column in self._numeric
            ]:
                raise DataError(
                    "features.missingness.zero_count requires a selected numeric column.",
                    "Include a numeric column or disable zero_count.",
                )
            self._missingness_columns = selected

        self._datetime_cyclical = self._features.datetime_cyclical
        if self._datetime_cyclical.enabled:
            selected = (
                list(self._datetime_cyclical.columns)
                if self._datetime_cyclical.columns
                else list(self._datetime)
            )
            unknown = [column for column in selected if column not in self._feature_columns]
            if unknown:
                raise DataError(
                    "features.datetime_cyclical references unknown columns: "
                    f"{', '.join(unknown)}.",
                    "Use columns from spec.feature_columns.",
                )
            non_datetime = [column for column in selected if column not in self._datetime]
            if non_datetime:
                raise DataError(
                    "features.datetime_cyclical columns must be datetime: "
                    f"{', '.join(non_datetime)}.",
                    "Use datetime columns or remove the cyclical block.",
                )
            self._datetime_cyclical_columns = selected

        self._numeric_interactions = self._features.numeric_interactions
        self._numeric_interaction_transformer: NumericInteractionTransformer | None = None
        if self._numeric_interactions.enabled:
            selected = (
                list(self._numeric_interactions.columns)
                if self._numeric_interactions.columns
                else list(self._numeric)
            )
            non_numeric = [column for column in selected if column not in self._numeric]
            if non_numeric:
                raise DataError(
                    "features.numeric_interactions columns must be numeric: "
                    f"{', '.join(map(str, non_numeric))}.",
                    "Select columns listed in spec.numeric_columns.",
                )
            if len(selected) < 2:
                raise DataError(
                    "features.numeric_interactions requires at least two numeric columns.",
                    "Add a numeric column or disable numeric interactions.",
                )
            self._numeric_interaction_transformer = NumericInteractionTransformer(
                columns=selected,
                max_pairs=self._numeric_interactions.max_pairs,
                operations=list(self._numeric_interactions.operations),
                clip_value=self._numeric_interactions.clip_value,
            )

        self._clustering = self._features.clustering
        self._cluster_transformer: ClusterFeatureTransformer | None = None
        if self._clustering.enabled:
            selected = (
                list(self._clustering.columns)
                if self._clustering.columns
                else list(self._numeric)
            )
            non_numeric = [column for column in selected if column not in self._numeric]
            if non_numeric:
                raise DataError(
                    "features.clustering columns must be numeric: "
                    f"{', '.join(map(str, non_numeric))}.",
                    "Select columns listed in spec.numeric_columns.",
                )
            if not selected:
                raise DataError(
                    "features.clustering requires at least one numeric column.",
                    "Add a numeric column or disable clustering.",
                )
            self._cluster_transformer = ClusterFeatureTransformer(
                columns=selected,
                n_clusters=self._clustering.n_clusters,
                max_iter=self._clustering.max_iter,
                random_state=config.seed,
                include_distances=self._clustering.include_distances,
                include_labels=self._clustering.include_labels,
                max_fit_rows=self._clustering.max_fit_rows,
            )

        self._anomaly = self._features.anomaly
        self._anomaly_transformer: AnomalyFeatureTransformer | None = None
        if self._anomaly.enabled:
            selected = (
                list(self._anomaly.columns)
                if self._anomaly.columns
                else list(self._numeric)
            )
            non_numeric = [column for column in selected if column not in self._numeric]
            if non_numeric:
                raise DataError(
                    "features.anomaly columns must be numeric: "
                    f"{', '.join(map(str, non_numeric))}.",
                    "Select columns listed in spec.numeric_columns.",
                )
            if not selected:
                raise DataError(
                    "features.anomaly requires at least one numeric column.",
                    "Add a numeric column or disable anomaly features.",
                )
            self._anomaly_transformer = AnomalyFeatureTransformer(
                columns=selected,
                n_estimators=self._anomaly.n_estimators,
                contamination=self._anomaly.contamination,
                random_state=config.seed,
                include_flag=self._anomaly.include_flag,
            )

        self._time_series = self._features.time_series
        self._time_series_transformer: TimeSeriesFeatureTransformer | None = None
        self._time_series_train_frame: pd.DataFrame | None = None
        if self._time_series.enabled:
            time_col = self._time_series.time_col
            if time_col in (None, "auto"):
                time_col = self._datetime[0] if self._datetime else None
            if time_col is None or time_col not in self._datetime:
                raise DataError(
                    "features.time_series requires a datetime feature column.",
                    "Add a datetime column or disable time-series features.",
                )
            group_col = self._time_series.group_col
            if group_col in (None, "auto"):
                group_col = self._categorical[0] if self._categorical else None
            if group_col is not None:
                if group_col not in self._feature_columns:
                    raise DataError(
                        f"features.time_series group_col '{group_col}' is not a feature column.",
                        "Use a feature column or group_col: auto.",
                    )
                if group_col == time_col:
                    raise DataError(
                        "features.time_series group_col and time_col must differ.",
                        "Use a categorical group column or no group column.",
                    )
            selected = (
                list(self._time_series.columns)
                if self._time_series.columns
                else [
                    column
                    for column in self._numeric
                    if column not in {time_col, group_col}
                ]
            )
            non_numeric = [column for column in selected if column not in self._numeric]
            if non_numeric:
                raise DataError(
                    "features.time_series columns must be numeric: "
                    f"{', '.join(map(str, non_numeric))}.",
                    "Select columns listed in spec.numeric_columns.",
                )
            forbidden = [column for column in (time_col, group_col) if column in selected]
            if forbidden:
                raise DataError(
                    "features.time_series columns must not include time_col or group_col: "
                    f"{', '.join(map(str, forbidden))}.",
                    "Use value columns only; time_col and group_col are separate.",
                )
            if not selected:
                raise DataError(
                    "features.time_series requires at least one numeric value column.",
                    "Add a numeric column or disable time-series features.",
                )
            self._time_series_transformer = TimeSeriesFeatureTransformer(
                time_col=time_col,
                group_col=group_col,
                columns=selected,
                lags=list(self._time_series.lags),
                windows=list(self._time_series.windows),
                stats=list(self._time_series.stats),
                include_elapsed=self._time_series.include_elapsed,
                include_diff=self._time_series.include_diff,
                max_history_rows=self._time_series.max_history_rows,
            )

        if self._row_statistics and not self._numeric:
            raise DataError(
                "features.row_statistics requires at least one numeric feature.",
                "Add a numeric column or disable row statistics.",
            )
        if self._modulo_features:
            if not self._numeric:
                raise DataError(
                    "features.modulo_features requires at least one numeric feature.",
                    "Add a numeric identifier-like column or disable modulo features.",
                )
            if not self._modulo_moduli and not self._modulo_steps:
                raise DataError(
                    "features.modulo_features needs at least one modulus or step.",
                    "Set features.modulo_moduli or features.modulo_steps.",
                )

        knn_config = self._features.knn_features
        if knn_config.enabled:
            if spec.task == "regression":
                raise DataError(
                    "features.knn_features is only available for classification tasks.",
                    "Disable kNN features for regression or use a classification recipe.",
                )
            if not self._numeric:
                raise DataError(
                    "features.knn_features requires at least one numeric feature.",
                    "Add a numeric column or disable kNN features.",
                )
            requested = list(knn_config.columns) if knn_config.columns else list(self._numeric)
            non_numeric = [column for column in requested if column not in self._numeric]
            if non_numeric:
                raise DataError(
                    f"features.knn_features columns must be numeric: {', '.join(map(str, non_numeric))}.",
                    "Select columns listed in spec.numeric_columns.",
                )
            self._knn = KNNClassDistanceTransformer(
                columns=requested,
                neighbors=knn_config.neighbors,
                folds=knn_config.folds,
                metric=knn_config.metric,
                max_reference_rows=knn_config.max_reference_rows,
                random_state=config.seed,
            )

    def runtime_unsupported_blocks(self) -> list[str]:
        """Return fitted blocks the portable runtime cannot reproduce yet."""
        blocks: list[str] = []
        if self._fold_plan is not None:
            blocks.append("fold_plan")
        if self._knn is not None:
            blocks.append("knn_features")
        if self._numeric_interaction_transformer is not None:
            blocks.append("numeric_interactions")
        if self._cluster_transformer is not None:
            blocks.append("clustering")
        if self._anomaly_transformer is not None:
            blocks.append("anomaly")
        if self._time_series_transformer is not None:
            blocks.append("time_series")
        return blocks

    def to_runtime_state(self) -> dict:
        """Serialize the fitted, label-free transform state for a bundle.

        The state contains only plain Python values and lists, so the portable
        runtime can apply it with numpy/pandas and no kconv import.  Advanced
        blocks are reported separately by :meth:`runtime_unsupported_blocks`.
        """
        if not self._fitted:
            raise DataError(
                "Cannot export a feature builder before fit_transform().",
                "This is an internal error.",
            )
        return {
            "schema_version": 1,
            "family": self.family,
            "feature_columns": list(self._feature_columns),
            "numeric": list(self._numeric),
            "categorical": list(self._categorical),
            "datetime": list(self._datetime),
            "text": list(self._text),
            "frequency": list(self._frequency),
            "hash_encode": list(self._hash_encode),
            "hash_features": int(self._hash_features),
            "target_encode": list(self._target_encode),
            "interactions": [
                {"columns": list(item.columns), "encoding": item.encoding}
                for item in self._interactions
            ],
            "keep": list(self._keep),
            "numeric_fill": dict(self._numeric_fill),
            "numeric_mean": dict(self._numeric_mean),
            "numeric_std": dict(self._numeric_std),
            "cat_categories": {key: list(value) for key, value in self._cat_categories.items()},
            "cat_maps": {key: dict(value) for key, value in self._cat_maps.items()},
            "cat_fill": dict(self._cat_fill),
            "datetime_fill": dict(self._datetime_fill),
            "text_fill": dict(self._text_fill),
            "freq_maps": {key: dict(value) for key, value in self._freq_maps.items()},
            "te_maps": {key: dict(value) for key, value in self._te_maps.items()},
            "te_prior": dict(self._te_prior),
            "interaction_maps": {key: dict(value) for key, value in self._interaction_maps.items()},
            "interaction_prior": dict(self._interaction_prior),
            "missingness": {
                "enabled": bool(self._missingness.enabled),
                "columns": list(self._missingness_columns),
                "indicators": bool(self._missingness.indicators),
                "row_count": bool(self._missingness.row_count),
                "zero_count": bool(self._missingness.zero_count),
            },
            "datetime_cyclical": {
                "enabled": bool(self._datetime_cyclical.enabled),
                "columns": list(self._datetime_cyclical_columns),
                "periods": list(self._datetime_cyclical.periods),
            },
            "row_statistics": bool(self._row_statistics),
            "row_statistics_stats": list(self._row_statistics_stats),
            "modulo_features": bool(self._modulo_features),
            "modulo_moduli": list(self._modulo_moduli),
            "modulo_steps": list(self._modulo_steps),
            "scale": self._features.scale,
            "max_categories": int(self._features.max_categories),
            "unsupported_blocks": self.runtime_unsupported_blocks(),
        }

    def _block_columns(self, value: object) -> list[str]:
        """Resolve an encoding block's ``columns`` parameter."""
        if value is None or value == "auto":
            return [column for column in self._categorical if column in self._feature_columns]
        if isinstance(value, str):
            values: list[object] = [value]
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            raise DataError(
                "Target/frequency block columns must be a list or 'auto'.",
                "Use columns: [name] or columns: auto.",
            )
        return [str(column) for column in values if str(column) in self._feature_columns]

    def _block_feature_columns(self, value: object) -> list[str]:
        """Resolve a label-free block's columns against all feature columns."""
        if value is None or value == "auto":
            return list(self._feature_columns)
        if isinstance(value, str):
            values: list[object] = [value]
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            raise DataError(
                "Feature block columns must be a list or 'auto'.",
                "Use columns: [name] or columns: auto.",
            )
        return [str(column) for column in values if str(column) in self._feature_columns]

    def _resolve_features(self, config: Config) -> FeatureConfig:
        """Apply an explicit block plan to a validated FeatureConfig view."""
        features = config.features
        if features.feature_blocks is None:
            return features
        blocks = validate_block_plan(config, self.spec)
        payload = features.model_dump(mode="python")
        payload.update(
            {
                "target_encode": [],
                "frequency_encode": [],
                "hash_encode": [],
                "hash_features": 16,
                "row_statistics": False,
                "modulo_features": False,
                "fold_plan": None,
                "interactions": [],
                "missingness": MissingnessConfig().model_dump(mode="python"),
                "datetime_cyclical": DatetimeCyclicalConfig().model_dump(mode="python"),
                "numeric_interactions": NumericInteractionConfig().model_dump(mode="python"),
                "clustering": ClusterFeatureConfig().model_dump(mode="python"),
                "anomaly": AnomalyFeatureConfig().model_dump(mode="python"),
                "time_series": TimeSeriesFeatureConfig().model_dump(mode="python"),
                "knn_features": KNNFeatureConfig().model_dump(mode="python"),
                "feature_selection": FeatureSelectionConfig().model_dump(mode="python"),
                "feature_blocks": [block.model_dump(mode="python") for block in blocks],
            }
        )
        for block in blocks:
            if not block.enabled:
                continue
            params = dict(block.params)
            if block.type == "base":
                for key in ("scale", "impute", "max_categories", "auto_encode", "drop_constant"):
                    if key in params:
                        payload[key] = params[key]
            elif block.type == "target_encoding":
                payload["target_encode"] = self._block_columns(params.get("columns"))
                if "smoothing" in params:
                    payload["target_smoothing"] = params["smoothing"]
            elif block.type == "frequency_encoding":
                payload["frequency_encode"] = self._block_columns(params.get("columns"))
            elif block.type == "hash_encoding":
                payload["hash_encode"] = self._block_columns(params.get("columns"))
                payload["hash_features"] = params.get("n_features", 16)
            elif block.type == "group_statistics":
                try:
                    group_plan = build_group_statistics_plan(params, self.spec)
                except ValueError as exc:
                    raise DataError(
                        f"Invalid group_statistics block: {exc}",
                        "Use group_col, numeric columns, and stats from mean/std/count.",
                    ) from exc
                payload["fold_plan"] = group_plan.plan
            elif block.type == "categorical_interaction":
                columns = self._block_columns(params.get("columns"))
                non_categorical = [column for column in columns if column not in self._categorical]
                if non_categorical:
                    raise DataError(
                        "categorical_interaction columns must be categorical: "
                        f"{', '.join(non_categorical)}.",
                        "Use categorical columns or remove the block.",
                    )
                max_pairs = params.get("max_pairs", 4)
                if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or not 1 <= max_pairs <= 50:
                    raise DataError(
                        "categorical_interaction.max_pairs must be an integer in 1..50.",
                        "Use a small number of pairs, for example max_pairs: 4.",
                    )
                encoding = params.get("encoding", "frequency")
                if encoding not in {"frequency", "target"}:
                    raise DataError(
                        f"Unknown categorical_interaction encoding '{encoding}'.",
                        "Use encoding: frequency or encoding: target.",
                    )
                payload["interactions"] = [
                    {"columns": list(pair), "encoding": encoding}
                    for pair in interaction_pairs(columns, max_pairs)
                ]
            elif block.type == "missingness":
                missingness = dict(payload["missingness"])
                missingness["enabled"] = True
                raw_columns = params.get("columns", "auto")
                if raw_columns not in (None, "auto"):
                    missingness["columns"] = self._block_feature_columns(raw_columns)
                for key in ("indicators", "row_count", "zero_count"):
                    if key in params:
                        missingness[key] = params[key]
                payload["missingness"] = missingness
            elif block.type == "datetime_cyclical":
                cyclical = dict(payload["datetime_cyclical"])
                cyclical["enabled"] = True
                raw_columns = params.get("columns", "auto")
                if raw_columns not in (None, "auto"):
                    cyclical["columns"] = self._block_feature_columns(raw_columns)
                if "periods" in params:
                    cyclical["periods"] = list(params["periods"])
                payload["datetime_cyclical"] = cyclical
            elif block.type == "numeric_interactions":
                interactions = dict(payload["numeric_interactions"])
                interactions["enabled"] = True
                raw_columns = params.get("columns", "auto")
                if raw_columns not in (None, "auto"):
                    interactions["columns"] = self._block_feature_columns(raw_columns)
                for key in ("max_pairs", "operations", "clip_value"):
                    if key in params:
                        interactions[key] = params[key]
                payload["numeric_interactions"] = interactions
            elif block.type == "clustering":
                clustering = dict(payload["clustering"])
                clustering["enabled"] = True
                raw_columns = params.get("columns", "auto")
                if raw_columns not in (None, "auto"):
                    clustering["columns"] = self._block_feature_columns(raw_columns)
                for key in (
                    "n_clusters",
                    "max_iter",
                    "include_distances",
                    "include_labels",
                    "max_fit_rows",
                ):
                    if key in params:
                        clustering[key] = params[key]
                payload["clustering"] = clustering
            elif block.type == "anomaly":
                anomaly = dict(payload["anomaly"])
                anomaly["enabled"] = True
                raw_columns = params.get("columns", "auto")
                if raw_columns not in (None, "auto"):
                    anomaly["columns"] = self._block_feature_columns(raw_columns)
                for key in ("n_estimators", "contamination", "include_flag"):
                    if key in params:
                        anomaly[key] = params[key]
                payload["anomaly"] = anomaly
            elif block.type == "time_series":
                time_series = dict(payload["time_series"])
                time_series["enabled"] = True
                raw_time = params.get("time_col")
                if raw_time in (None, "auto"):
                    time_series["time_col"] = self._datetime[0] if self._datetime else None
                else:
                    time_series["time_col"] = str(raw_time)
                raw_group = params.get("group_col")
                if raw_group in (None, "auto"):
                    time_series["group_col"] = self._categorical[0] if self._categorical else None
                else:
                    time_series["group_col"] = str(raw_group)
                raw_columns = params.get("columns", "auto")
                if raw_columns not in (None, "auto"):
                    time_series["columns"] = self._block_feature_columns(raw_columns)
                for key in (
                    "lags",
                    "windows",
                    "stats",
                    "include_elapsed",
                    "include_diff",
                    "max_history_rows",
                ):
                    if key in params:
                        time_series[key] = params[key]
                payload["time_series"] = time_series
            elif block.type == "row_statistics":
                payload["row_statistics"] = True
                payload["row_statistics_stats"] = list(params.get("stats", ["mean", "std", "min", "max"]))
            elif block.type == "modulo":
                payload["modulo_features"] = True
                payload["modulo_moduli"] = list(params.get("moduli", [10, 100]))
                payload["modulo_steps"] = list(params.get("steps", [5]))
            elif block.type == "fold_plan":
                payload["fold_plan"] = params.get("plan")
            elif block.type == "feature_selection":
                selection = dict(payload["feature_selection"])
                selection["enabled"] = True
                for key in ("method", "max_features", "min_variance"):
                    if key in params:
                        selection[key] = params[key]
                payload["feature_selection"] = selection
            elif block.type == "knn_class_distance":
                knn = dict(payload["knn_features"])
                knn["enabled"] = True
                for key in ("columns", "neighbors", "folds", "metric", "max_reference_rows"):
                    if key in params:
                        knn[key] = params[key]
                payload["knn_features"] = knn
        try:
            return FeatureConfig.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            raise DataError(
                f"Invalid feature block plan at '{first.get('loc', ())}': {first.get('msg', 'invalid value')}",
                "Check block params against docs/pipelines.md.",
            ) from exc

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def _fit_numeric(self, df: pd.DataFrame) -> None:
        for column in self._numeric:
            values = _safe_numeric(df[column])
            median = float(values.median()) if values.notna().any() else 0.0
            self._numeric_fill[column] = median
            filled = values.fillna(median)
            mean = float(filled.mean()) if len(filled) else 0.0
            std = float(filled.std(ddof=0)) if len(filled) > 1 else 0.0
            self._numeric_mean[column] = mean
            self._numeric_std[column] = std if std > 1e-12 else 1.0

    def _fit_categorical(self, df: pd.DataFrame) -> None:
        for column in self._categorical:
            values = _as_str_series(df[column])
            counts = values.value_counts()
            categories = [str(v) for v in counts.index[: max(2, self._features.max_categories)]]
            self._cat_categories[column] = categories
            self._cat_maps[column] = {value: idx for idx, value in enumerate(categories)}
            self._cat_fill[column] = "__missing__"

    def _fit_datetime(self, df: pd.DataFrame) -> None:
        for column in self._datetime:
            values = self._parse_datetime(df[column])
            ordinal = values.map(lambda v: v.toordinal() if pd.notna(v) else np.nan)
            fill = float(ordinal.median()) if ordinal.notna().any() else 0.0
            self._datetime_fill[column] = fill

    def _fit_text(self, df: pd.DataFrame) -> None:
        for column in self._text:
            values = _as_str_series(df[column])
            length = values.str.len().astype(float)
            words = values.str.split().str.len().astype(float)
            self._text_fill[f"{column}__len"] = float(length.median()) if length.notna().any() else 0.0
            self._text_fill[f"{column}__words"] = float(words.median()) if words.notna().any() else 0.0

    def _fit_frequency(self, df: pd.DataFrame) -> None:
        for column in self._frequency:
            values = _as_str_series(df[column])
            counts = values.value_counts(normalize=True, dropna=False)
            self._freq_maps[column] = {str(k): float(v) for k, v in counts.items()}

    def _target_map(self, values: pd.Series, target: np.ndarray) -> tuple[dict[str, float], float]:
        strings = _as_str_series(values)
        frame = pd.DataFrame({"value": strings, "target": np.asarray(target, dtype=float)})
        prior = float(frame["target"].mean()) if len(frame) else 0.0
        grouped = frame.groupby("value", observed=True)["target"].agg(["sum", "count"])
        mapping: dict[str, float] = {}
        for key, row in grouped.iterrows():
            count = float(row["count"])
            mapping[str(key)] = float((row["sum"] + prior * self._smoothing) / (count + self._smoothing))
        return mapping, prior

    def _fit_target_encoding(self, df: pd.DataFrame, y: np.ndarray) -> None:
        for column in self._target_encode:
            mapping, prior = self._target_map(df[column], y)
            self._te_maps[column] = mapping
            self._te_prior[column] = prior
            self._te_oof[column] = self._oof_target(df[column], y)

    @staticmethod
    def _interaction_series(df: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.Series:
        """Concatenate two categorical columns with a collision-resistant separator."""
        parts = [_as_str_series(df[column]) for column in columns]
        combined = parts[0]
        for part in parts[1:]:
            combined = combined + "\x1f" + part
        return combined.astype(str)

    def _fit_interactions(self, df: pd.DataFrame, y: np.ndarray) -> None:
        for interaction in self._interactions:
            name = interaction_name(interaction.columns, interaction.encoding)
            values = self._interaction_series(df, interaction.columns)
            if interaction.encoding == "frequency":
                counts = values.value_counts(normalize=True, dropna=False)
                self._interaction_maps[name] = {
                    str(key): float(value) for key, value in counts.items()
                }
            else:
                mapping, prior = self._target_map(values, y)
                self._interaction_maps[name] = mapping
                self._interaction_prior[name] = prior
                self._interaction_oof[name] = self._oof_target(values, y)

    def _fit_knn(self, df: pd.DataFrame, y: np.ndarray) -> None:
        if self._knn is None:
            return
        base = df.loc[:, self._feature_columns]
        self._knn_oof = self._knn.fit_transform_oof(base, y)

    def _fit_advanced(self, df: pd.DataFrame) -> None:
        """Fit the label-free/fold-safe advanced blocks on the outer train fold."""
        if self._numeric_interaction_transformer is not None:
            self._numeric_interaction_transformer.fit(df)
        if self._cluster_transformer is not None:
            self._cluster_transformer.fit(df)
        if self._anomaly_transformer is not None:
            self._anomaly_transformer.fit(df)
        if self._time_series_transformer is not None:
            self._time_series_train_frame = self._time_series_transformer.fit_transform(df)

    def _oof_target(self, values: pd.Series, y: np.ndarray) -> np.ndarray:
        n = len(values)
        strings = _as_str_series(values)
        target = np.asarray(y, dtype=float)
        if n < 8:
            # Too few rows for an inner split.  Leave-one-out keeps the
            # training-row encoding free of the row's own target while still
            # using the other rows in the outer training fold.
            total = float(target.sum())
            grouped = pd.DataFrame({"value": strings, "target": target}).groupby(
                "value", observed=True
            )["target"].agg(["sum", "count"])
            output = np.empty(n, dtype=float)
            for position, (key, target_value) in enumerate(zip(strings, target, strict=True)):
                count = float(grouped.loc[key, "count"])
                group_sum = float(grouped.loc[key, "sum"])
                prior = (total - target_value) / max(n - 1, 1)
                denominator = count - 1 + self._smoothing
                output[position] = (
                    (group_sum - target_value + prior * self._smoothing) / denominator
                    if denominator > 0
                    else prior
                )
            return output
        n_splits = min(5, max(2, n // 4))
        # Regression and multiclass both work with KFold; classification labels
        # are already integer-encoded by the training stage.  Materialise the
        # split list so an invalid stratified target can fall back to KFold
        # before iteration starts.
        try:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.seed)
            iterator = list(splitter.split(np.zeros(n), y))
        except ValueError:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.config.seed)
            iterator = list(splitter.split(np.zeros(n)))
        output = np.full(n, np.nan, dtype=float)
        for train_idx, valid_idx in iterator:
            mapping, prior = self._target_map(strings.iloc[train_idx], y[train_idx])
            output[valid_idx] = strings.iloc[valid_idx].map(mapping).fillna(prior).to_numpy(dtype=float)
        if np.isnan(output).any():
            raise DataError(
                "Target encoding produced missing out-of-fold values.",
                "Check for missing target values or rerun ingest.",
            )
        return output

    # ------------------------------------------------------------------
    # transformations
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_datetime(values: pd.Series) -> pd.Series:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return pd.to_datetime(values, errors="coerce", format="mixed")
            except (TypeError, ValueError):
                return pd.to_datetime(values, errors="coerce")

    def _numeric_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        scale = self._features.scale == "standard" or (
            self._features.scale == "auto" and self.family == "linear"
        )
        for column in self._numeric:
            values = _safe_numeric(df[column]).fillna(self._numeric_fill[column])
            if scale:
                values = (values - self._numeric_mean[column]) / self._numeric_std[column]
            out[column] = values
        return out

    def _categorical_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for column in self._categorical:
            values = _as_str_series(df[column])
            if self.family == "linear":
                categories = self._cat_categories[column]
                for category in categories:
                    out[f"{column}__{category}"] = (values == category).astype(float)
            else:
                mapping = self._cat_maps[column]
                out[column] = values.map(mapping).fillna(-1).astype(float)
        return out

    def _datetime_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for column in self._datetime:
            parsed = self._parse_datetime(df[column])
            ordinal = parsed.map(lambda v: v.toordinal() if pd.notna(v) else np.nan).astype(float)
            ordinal = ordinal.fillna(self._datetime_fill[column])
            out[f"{column}__ordinal"] = ordinal
            out[f"{column}__year"] = parsed.dt.year.astype(float).fillna(0.0)
            out[f"{column}__month"] = parsed.dt.month.astype(float).fillna(0.0)
            out[f"{column}__day"] = parsed.dt.day.astype(float).fillna(0.0)
            out[f"{column}__dayofweek"] = parsed.dt.dayofweek.astype(float).fillna(0.0)
        return out

    def _text_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for column in self._text:
            values = _as_str_series(df[column])
            length = values.str.len().astype(float)
            words = values.str.split().str.len().astype(float)
            out[f"{column}__len"] = length.fillna(self._text_fill.get(f"{column}__len", 0.0))
            out[f"{column}__words"] = words.fillna(self._text_fill.get(f"{column}__words", 0.0))
        return out

    def _frequency_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for column in self._frequency:
            values = _as_str_series(df[column])
            out[f"{column}__freq"] = values.map(self._freq_maps[column]).fillna(0.0).astype(float)
        return out

    def _hash_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Return stable, label-free hash buckets for categorical columns.

        The hash is deterministic across processes and pandas does not need a
        fitted vocabulary.  All selected columns share one bucket space, which
        keeps the matrix small; collisions are intentional and tree-friendly.
        """
        if not self._hash_encode:
            return {}
        matrix = np.zeros((len(df), self._hash_features), dtype=float)
        positions = np.arange(len(df))
        for column in self._hash_encode:
            values = _as_str_series(df[column])
            hashed = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64)
            salt = int.from_bytes(
                hashlib.blake2b(column.encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            buckets = ((hashed ^ np.uint64(salt)) % np.uint64(self._hash_features)).astype(int)
            matrix[positions, buckets] += 1.0
        return {
            f"hash__{index}": pd.Series(matrix[:, index], index=df.index, dtype=float)
            for index in range(self._hash_features)
        }

    def _target_frame(self, df: pd.DataFrame, *, oof: bool) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for column in self._target_encode:
            if oof:
                out[f"{column}__target"] = pd.Series(self._te_oof[column], index=df.index, dtype=float)
            else:
                values = _as_str_series(df[column])
                out[f"{column}__target"] = values.map(self._te_maps[column]).fillna(self._te_prior[column]).astype(float)
        return out

    def _row_statistics_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if not self._row_statistics or not self._numeric:
            return {}
        matrix = pd.concat([_safe_numeric(df[column]) for column in self._numeric], axis=1)
        matrix.columns = self._numeric
        mapping = {
            "mean": matrix.mean(axis=1),
            "std": matrix.std(axis=1, ddof=0),
            "min": matrix.min(axis=1),
            "max": matrix.max(axis=1),
            "median": matrix.median(axis=1),
        }
        return {
            f"row_{stat}": mapping[stat].fillna(0.0).astype(float)
            for stat in self._row_statistics_stats
        }

    def _modulo_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if not self._modulo_features:
            return {}
        out: dict[str, pd.Series] = {}
        for column in self._numeric:
            values = _safe_numeric(df[column])
            for modulus in self._modulo_moduli:
                out[f"{column}__mod_{modulus}"] = (values % modulus).fillna(0.0).astype(float)
            for step in self._modulo_steps:
                out[f"{column}__div_{step}"] = np.floor(values / step).fillna(0.0).astype(float)
        return out

    def _interaction_frame(self, df: pd.DataFrame, *, oof: bool) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for interaction in self._interactions:
            name = interaction_name(interaction.columns, interaction.encoding)
            if interaction.encoding == "target" and oof:
                out[name] = pd.Series(
                    self._interaction_oof[name], index=df.index, dtype=float
                )
                continue
            values = self._interaction_series(df, interaction.columns)
            if interaction.encoding == "frequency":
                out[name] = values.map(self._interaction_maps[name]).fillna(0.0).astype(float)
            else:
                out[name] = (
                    values.map(self._interaction_maps[name])
                    .fillna(self._interaction_prior[name])
                    .astype(float)
                )
        return out

    def _missingness_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if not self._missingness.enabled or not self._missingness_columns:
            return {}
        out: dict[str, pd.Series] = {}
        selected = self._missingness_columns
        if self._missingness.row_count:
            out["missing__count"] = df.loc[:, selected].isna().sum(axis=1).astype(float)
        if self._missingness.indicators:
            for column in selected:
                out[f"{column}__missing"] = df[column].isna().astype(float)
        if self._missingness.zero_count:
            numeric_selected = [column for column in selected if column in self._numeric]
            if numeric_selected:
                matrix = pd.concat(
                    [_safe_numeric(df[column]) for column in numeric_selected], axis=1
                )
                out["zero__count"] = (matrix == 0).sum(axis=1).astype(float)
        return out

    def _datetime_cyclical_frame(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if not self._datetime_cyclical.enabled or not self._datetime_cyclical_columns:
            return {}
        period_lengths = {
            "month": 12.0,
            "dayofweek": 7.0,
            "dayofyear": 365.25,
            "hour": 24.0,
        }
        out: dict[str, pd.Series] = {}
        for column in self._datetime_cyclical_columns:
            parsed = self._parse_datetime(df[column])
            for period in self._datetime_cyclical.periods:
                values = getattr(parsed.dt, period).astype(float)
                radians = 2.0 * np.pi * values / period_lengths[period]
                out[f"{column}__{period}_sin"] = pd.Series(
                    np.sin(radians), index=df.index, dtype=float
                ).fillna(0.0)
                out[f"{column}__{period}_cos"] = pd.Series(
                    np.cos(radians), index=df.index, dtype=float
                ).fillna(0.0)
        return out

    def _assemble(self, df: pd.DataFrame, *, oof: bool) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for frame in (
            self._numeric_frame(df),
            self._categorical_frame(df),
            self._datetime_frame(df),
            self._datetime_cyclical_frame(df),
            self._text_frame(df),
            self._frequency_frame(df),
            self._hash_frame(df),
            self._target_frame(df, oof=oof),
            self._interaction_frame(df, oof=oof),
            self._missingness_frame(df),
            self._row_statistics_frame(df),
            self._modulo_frame(df),
        ):
            if frame:
                parts.append(pd.DataFrame(frame, index=df.index))
        if self._fold_plan is not None:
            plan_frame = self._fold_plan.transform(df)
            if not plan_frame.empty:
                parts.append(plan_frame)
        if self._knn is not None:
            if oof:
                if self._knn_oof is None:
                    raise DataError(
                        "kNN training features were not fitted.",
                        "This is an internal error; fit FeatureBuilder before transforming.",
                    )
                parts.append(self._knn_oof)
            else:
                parts.append(self._knn.transform(df.loc[:, self._feature_columns]))
        if self._numeric_interaction_transformer is not None:
            parts.append(self._numeric_interaction_transformer.transform(df))
        if self._cluster_transformer is not None:
            parts.append(self._cluster_transformer.transform(df))
        if self._anomaly_transformer is not None:
            parts.append(self._anomaly_transformer.transform(df))
        if self._time_series_transformer is not None:
            if oof:
                if self._time_series_train_frame is None:
                    raise DataError(
                        "Time-series training features were not fitted.",
                        "This is an internal error; fit FeatureBuilder before transforming.",
                    )
                parts.append(self._time_series_train_frame)
            else:
                parts.append(self._time_series_transformer.transform(df))
        if not parts:
            raise DataError(
                "No usable features were produced from the dataset.",
                "Check column roles and data.ignore.",
            )
        matrix = pd.concat(parts, axis=1)
        if matrix.columns.duplicated().any():
            duplicates = sorted(set(matrix.columns[matrix.columns.duplicated()].astype(str)))
            raise DataError(
                f"Duplicate engineered feature names: {', '.join(duplicates)}.",
                "Rename fold-plan features so they do not collide with base features.",
            )
        # Stable order, duplicate names impossible, all numeric.
        matrix = matrix.reindex(sorted(matrix.columns), axis=1)
        return matrix

    def _select_features(self, matrix: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Apply the configured final selector to the assembled matrix."""
        selection = self._feature_selection
        if selection.method != "variance" and matrix.shape[1] <= selection.max_features:
            return matrix
        values = matrix.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataError(
                "Feature selection requires finite feature values.",
                "Check fold-plan expressions and engineered columns for NaN/inf.",
            )
        if selection.method == "variance":
            scores = matrix.var(axis=0, ddof=0)
            ranked = scores[scores > selection.min_variance].sort_values(ascending=False)
            selected = set(ranked.index[: selection.max_features])
        elif selection.method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

            target = np.asarray(y)
            if self.spec.task in {"binary", "multiclass"}:
                if target.dtype.kind not in "iu":
                    target = pd.factorize(target)[0]
                scores = mutual_info_classif(values, target, random_state=self.config.seed)
            elif self.spec.task == "regression":
                scores = mutual_info_regression(
                    values,
                    target.astype(float),
                    random_state=self.config.seed,
                )
            else:
                raise DataError(
                    f"Feature selection does not support task '{self.spec.task}'.",
                    "Use a binary, multiclass or regression task.",
                )
            ranked = pd.Series(np.asarray(scores, dtype=float), index=matrix.columns)
            ranked = ranked.sort_values(ascending=False)
            selected = set(ranked.index[: selection.max_features])
        elif selection.method == "model_importance":
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

            target = np.asarray(y)
            if self.spec.task in {"binary", "multiclass"}:
                if target.dtype.kind not in "iu":
                    target = pd.factorize(target)[0]
                estimator = ExtraTreesClassifier(
                    n_estimators=64,
                    random_state=self.config.seed,
                    n_jobs=1,
                )
            elif self.spec.task == "regression":
                estimator = ExtraTreesRegressor(
                    n_estimators=64,
                    random_state=self.config.seed,
                    n_jobs=1,
                )
                target = target.astype(float)
            else:
                raise DataError(
                    f"Feature selection does not support task '{self.spec.task}'.",
                    "Use a binary, multiclass or regression task.",
                )
            estimator.fit(values, target)
            ranked = pd.Series(
                np.asarray(estimator.feature_importances_, dtype=float),
                index=matrix.columns,
            ).sort_values(ascending=False)
            selected = set(ranked.index[: selection.max_features])
        else:
            raise DataError(
                f"Unknown feature selection method '{selection.method}'.",
                "Use variance, mutual_info or model_importance.",
            )
        if not selected:
            raise DataError(
                "Feature selection removed all features.",
                "Lower feature_selection.min_variance or increase max_features.",
            )
        return matrix.loc[:, [column for column in matrix.columns if column in selected]]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit_transform(self, df_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
        if len(df_train) != len(y_train):
            raise DataError("FeatureBuilder received mismatched X/y row counts.", "This is an internal error.")
        self._fit_numeric(df_train)
        self._fit_categorical(df_train)
        self._fit_datetime(df_train)
        self._fit_text(df_train)
        self._fit_frequency(df_train)
        self._fit_target_encoding(df_train, np.asarray(y_train))
        self._fit_interactions(df_train, np.asarray(y_train))
        self._fit_knn(df_train, np.asarray(y_train))
        self._fit_advanced(df_train)
        if self._fold_plan is not None:
            self._fold_plan.fit(df_train)
        matrix = self._assemble(df_train, oof=True)
        if self._features.drop_constant:
            constant = [c for c in matrix.columns if matrix[c].nunique(dropna=False) <= 1]
            if constant:
                matrix = matrix.drop(columns=constant)
        if matrix.shape[1] == 0:
            raise DataError("All engineered features are constant.", "Add more informative columns.")
        if self._feature_selection.enabled:
            matrix = self._select_features(matrix, np.asarray(y_train))
        self._keep = list(matrix.columns)
        self.feature_names = list(matrix.columns)
        self._fitted = True
        return matrix.reset_index(drop=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise DataError("FeatureBuilder.transform() called before fit_transform().", "This is an internal error.")
        missing = [column for column in self._feature_columns if column not in df.columns]
        if missing:
            raise DataError(
                "Test/validation data is missing feature columns: " + ", ".join(missing) + ".",
                "Fix the schema mismatch between train and test before running the pipeline.",
            )
        matrix = self._assemble(df, oof=False)
        matrix = matrix.reindex(columns=self._keep, fill_value=0.0)
        return matrix.reset_index(drop=True)
