"""Fit-in-fold unsupervised clustering features.

A MiniBatchKMeans model is fitted on the outer training fold only.  The
transformer emits distances to each centroid and/or the cluster label; it
never reads the target, so validation and test rows cannot influence the
clusters.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from kconv.errors import DataError


class ClusterFeatureTransformer:
    """Build MiniBatchKMeans cluster-distance and cluster-label features.

    Parameters
    ----------
    columns:
        Numeric columns to cluster.  ``None`` or an empty sequence selects
        every numeric column in the fitted frame.
    n_clusters:
        Requested number of clusters, in ``2..50``.  Tiny training frames are
        capped at their row count.
    max_iter:
        Maximum MiniBatchKMeans iterations, in ``10..1000``.
    random_state:
        Seed for deterministic subsampling and cluster initialisation.
    include_distances:
        Emit one distance column per fitted centroid.
    include_labels:
        Emit ``cluster__label`` as a float column.
    max_fit_rows:
        Deterministic row cap for fitting the centroids.
    """

    def __init__(
        self,
        columns: Sequence[object] | None = None,
        *,
        n_clusters: int = 8,
        max_iter: int = 100,
        random_state: int = 42,
        include_distances: bool = True,
        include_labels: bool = True,
        max_fit_rows: int = 50_000,
    ) -> None:
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or not 2 <= n_clusters <= 50:
            raise ValueError("n_clusters must be an integer in 2..50")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or not 10 <= max_iter <= 1000:
            raise ValueError("max_iter must be an integer in 10..1000")
        if (
            isinstance(max_fit_rows, bool)
            or not isinstance(max_fit_rows, int)
            or not 100 <= max_fit_rows <= 1_000_000
        ):
            raise ValueError("max_fit_rows must be an integer in 100..1000000")
        if not isinstance(include_distances, bool) or not isinstance(include_labels, bool):
            raise ValueError("include_distances and include_labels must be booleans")
        if not include_distances and not include_labels:
            raise ValueError("clustering must include distances, labels, or both")
        selected_columns = None if columns is None else list(columns)
        self.columns: list[object] | None = selected_columns if selected_columns else None
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.include_distances = include_distances
        self.include_labels = include_labels
        self.max_fit_rows = int(max_fit_rows)
        self.feature_names: list[str] = []
        self._model: MiniBatchKMeans | None = None
        self._fill = np.array([])
        self._mean = np.array([])
        self._scale = np.array([])
        self._n_clusters_fit = 0
        self._fitted = False

    # ------------------------------------------------------------------
    # input preparation
    # ------------------------------------------------------------------

    def _select_columns(self, frame: pd.DataFrame) -> list[object]:
        if frame.columns.duplicated().any():
            duplicates = sorted({str(column) for column in frame.columns[frame.columns.duplicated()]})
            raise DataError(
                f"Duplicate column(s) for clustering features: {', '.join(duplicates)}.",
                "Rename duplicate columns before enabling clustering.",
            )
        if self.columns is not None:
            columns = list(self.columns)
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise DataError(
                    f"Cluster feature column(s) not present: {', '.join(map(str, missing))}.",
                    "Check features.clustering.columns against the dataset schema.",
                )
        else:
            columns = [
                column
                for column in frame.columns
                if pd.api.types.is_numeric_dtype(frame[column])
                and not pd.api.types.is_bool_dtype(frame[column])
            ]
        if not columns:
            raise DataError(
                "Clustering features need at least one numeric column.",
                "Select numeric columns or disable the clustering block.",
            )
        return columns

    @staticmethod
    def _matrix(frame: pd.DataFrame, columns: Sequence[object]) -> np.ndarray:
        try:
            values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise DataError(
                "Clustering features could not convert the selected columns to numeric.",
                "Use numeric columns or encode categoricals first.",
            ) from exc
        return values.to_numpy(dtype="float64", copy=False)

    def _prepare(self, frame: pd.DataFrame, columns: Sequence[object]) -> np.ndarray:
        if not self._fitted:
            raise DataError(
                "ClusterFeatureTransformer.transform() called before fit().",
                "Fit the transformer on the training fold first.",
            )
        values = self._matrix(frame, columns)
        values = np.where(np.isfinite(values), values, self._fill)
        return (values - self._mean) / self._scale

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def fit(self, frame: pd.DataFrame, y: object = None) -> ClusterFeatureTransformer:
        """Fit MiniBatchKMeans on the training frame without a target."""
        del y  # sklearn-style signature; the transform is unsupervised
        selected = self._select_columns(frame)
        if len(frame) == 0:
            raise DataError(
                "Clustering features received an empty training frame.",
                "Disable the clustering block for empty folds.",
            )
        values = self._matrix(frame, selected)
        fill = np.nanmedian(values, axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        values = np.where(np.isfinite(values), values, fill)

        if len(values) > self.max_fit_rows:
            rng = np.random.default_rng(self.random_state)
            keep = np.sort(rng.choice(len(values), size=self.max_fit_rows, replace=False))
            fit_values = values[keep]
        else:
            fit_values = values
        n_clusters = min(self.n_clusters, len(fit_values))
        n_clusters = max(1, n_clusters)
        mean = np.mean(fit_values, axis=0)
        scale = np.std(fit_values, axis=0, ddof=0)
        scale = np.maximum(scale, 1e-12)
        scaled = (fit_values - mean) / scale
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            n_init=3,
            random_state=self.random_state,
            batch_size=min(1024, max(256, len(scaled))),
            max_iter=self.max_iter,
        )
        model.fit(scaled)

        self.columns = selected
        self._model = model
        self._fill = fill
        self._mean = mean
        self._scale = scale
        self._n_clusters_fit = n_clusters
        self.feature_names = (
            [f"cluster__dist_{index}" for index in range(n_clusters)]
            if self.include_distances
            else []
        )
        if self.include_labels:
            self.feature_names.append("cluster__label")
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # transformations
    # ------------------------------------------------------------------

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return cluster distances and/or labels for ``frame``."""
        if not self._fitted or self._model is None:
            raise DataError(
                "ClusterFeatureTransformer.transform() called before fit().",
                "Fit the transformer on the training fold first.",
            )
        if len(frame) == 0:
            return pd.DataFrame(columns=self.feature_names, index=frame.index, dtype="float64")
        selected = self._select_columns(frame)
        if list(selected) != list(self.columns or []):
            raise DataError(
                "Cluster transform frame has a different numeric column set than the fitted frame.",
                "Keep the dataset schema stable between fit and transform.",
            )
        scaled = self._prepare(frame, selected)
        output: dict[str, np.ndarray] = {}
        if self.include_distances:
            distances = np.asarray(self._model.transform(scaled), dtype="float64")
            distances = np.nan_to_num(distances, nan=0.0, posinf=1e12, neginf=0.0)
            for index in range(self._n_clusters_fit):
                output[f"cluster__dist_{index}"] = distances[:, index]
        if self.include_labels:
            labels = np.asarray(self._model.predict(scaled), dtype="float64")
            output["cluster__label"] = np.nan_to_num(labels, nan=0.0)
        return pd.DataFrame(output, index=frame.index)[self.feature_names]

    def fit_transform(self, frame: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """Fit on ``frame`` and return cluster features for its rows."""
        return self.fit(frame, y).transform(frame)


__all__ = ["ClusterFeatureTransformer"]
