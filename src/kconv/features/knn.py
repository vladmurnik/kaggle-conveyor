"""Class-conditional kNN distance features.

The method is adapted from ``upura/knnFeat`` (MIT License, Copyright (c) 2018
Shotaro Ishihara).  The original generates ``k * c`` distance features with a
shuffled KFold; this module is an original implementation that makes the
contract fit-in-fold explicit:

* the reference set is the outer training fold only;
* training rows get an inner-OOF value, so a row is never its own neighbour;
* validation/test rows are transformed with the mapping fitted on the complete
  outer training fold.

Only numeric columns are used.  Categorical columns should be represented by a
numeric encoding before they are passed here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from kconv.errors import DataError


class KNNClassDistanceTransformer:
    """Build per-class nearest-neighbour distance features.

    Parameters
    ----------
    columns:
        Numeric columns to use.  ``None`` selects every numeric column in the
        fitted frame.
    neighbors:
        Number of neighbours per class.  Output contains the cumulative
        distance to the first, second, ..., ``neighbors`` nearest neighbours.
    folds:
        Maximum number of inner folds used for OOF training features.
    metric:
        Distance metric passed to scikit-learn's ``NearestNeighbors``.
    max_reference_rows:
        Deterministic cap on reference rows per class.  It keeps memory and
        runtime bounded on large datasets.
    random_state:
        Seed for the inner split and deterministic reference subsampling.
    """

    def __init__(
        self,
        columns: Sequence[object] | None = None,
        *,
        neighbors: int = 3,
        folds: int = 5,
        metric: str = "euclidean",
        max_reference_rows: int = 50_000,
        random_state: int = 42,
    ) -> None:
        if neighbors < 1:
            raise ValueError("neighbors must be at least 1")
        if folds < 2:
            raise ValueError("folds must be at least 2")
        if max_reference_rows < 1:
            raise ValueError("max_reference_rows must be positive")
        self.columns = None if columns is None else list(columns)
        self.neighbors = int(neighbors)
        self.folds = int(folds)
        self.metric = metric
        self.max_reference_rows = int(max_reference_rows)
        self.random_state = int(random_state)
        self.feature_names: list[str] = []
        self._classes = np.array([])
        self._models: dict[int, tuple[NearestNeighbors, int, float]] = {}
        self._fill = np.array([])
        self._mean = np.array([])
        self._std = np.array([])
        self._fitted = False

    # ------------------------------------------------------------------
    # input preparation
    # ------------------------------------------------------------------

    def _select_columns(self, frame: pd.DataFrame) -> list[object]:
        if frame.columns.duplicated().any():
            duplicates = sorted({str(column) for column in frame.columns[frame.columns.duplicated()]})
            raise DataError(
                f"Duplicate column(s) for kNN features: {', '.join(duplicates)}.",
                "Rename duplicate columns before enabling kNN features.",
            )
        if self.columns is not None:
            columns = list(self.columns)
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise DataError(
                    f"kNN feature column(s) not present: {', '.join(map(str, missing))}.",
                    "Check features.knn_features.columns against the dataset schema.",
                )
        else:
            columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
        if not columns:
            raise DataError(
                "kNN features need at least one numeric column.",
                "Select numeric columns or disable features.knn_features.",
            )
        return columns

    @staticmethod
    def _matrix(frame: pd.DataFrame, columns: Sequence[object]) -> np.ndarray:
        try:
            values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        except (TypeError, ValueError) as exc:
            raise DataError(
                "kNN features could not convert the selected columns to numeric.",
                "Use numeric columns or encode categoricals first.",
            ) from exc
        return values.to_numpy(dtype="float64", copy=False)

    def _prepare(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise DataError(
                "KNNClassDistanceTransformer.transform() called before fit().",
                "Fit the transformer on the training fold first.",
            )
        selected = self._select_columns(frame)
        if list(selected) != list(self.columns or []):
            raise DataError(
                "kNN transform frame has a different numeric column set than the fitted frame.",
                "Keep the dataset schema stable between fit and transform.",
            )
        values = self._matrix(frame, selected)
        values = np.where(np.isfinite(values), values, self._fill)
        return (values - self._mean) / self._std

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def fit(self, frame: pd.DataFrame, y: object) -> KNNClassDistanceTransformer:
        """Fit one nearest-neighbour index per class on the training frame."""
        selected = self._select_columns(frame)
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(frame):
            raise DataError(
                "kNN features received mismatched X/y row counts.",
                "Pass one label per training row.",
            )
        if len(frame) == 0 or target.size == 0:
            raise DataError(
                "kNN features received an empty training frame.",
                "Disable features.knn_features for empty folds.",
            )
        values = self._matrix(frame, selected)
        fill = np.nanmedian(values, axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        values = np.where(np.isfinite(values), values, fill)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        scaled = (values - mean) / std

        classes = np.unique(target)
        if classes.size == 0:
            raise DataError(
                "kNN features need at least one target class.",
                "Check the target column and the task type.",
            )

        models: dict[int, tuple[NearestNeighbors, int, float]] = {}
        rng = np.random.default_rng(self.random_state)
        for index, class_value in enumerate(classes):
            class_rows = scaled[target == class_value]
            if len(class_rows) == 0:  # pragma: no cover - np.unique guarantees a match
                continue
            if len(class_rows) > self.max_reference_rows:
                keep = np.sort(
                    rng.choice(len(class_rows), size=self.max_reference_rows, replace=False)
                )
                class_rows = class_rows[keep]
            actual_k = min(self.neighbors, len(class_rows))
            model = NearestNeighbors(n_neighbors=actual_k, metric=self.metric, algorithm="auto")
            model.fit(class_rows)
            distances, _ = model.kneighbors(class_rows)
            fill_distance = float(np.nanmax(distances)) if distances.size else 1.0
            if not np.isfinite(fill_distance) or fill_distance <= 0.0:
                fill_distance = 1.0
            models[index] = (model, actual_k, fill_distance)

        self.columns = selected
        self._classes = classes
        self._models = models
        self._fill = fill
        self._mean = mean
        self._std = std
        self.feature_names = [
            f"knn_c{class_index}_k{k}"
            for class_index in range(len(classes))
            for k in range(1, self.neighbors + 1)
        ]
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # transformations
    # ------------------------------------------------------------------

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return one cumulative-distance column per class and neighbour rank."""
        if not self._fitted:
            raise DataError(
                "KNNClassDistanceTransformer.transform() called before fit().",
                "Fit the transformer on the training fold first.",
            )
        if len(frame) == 0:
            return pd.DataFrame(columns=self.feature_names, index=frame.index, dtype="float64")
        matrix = self._prepare(frame)
        output: dict[str, np.ndarray] = {}
        for class_index in range(len(self._classes)):
            model, actual_k, fill_distance = self._models[class_index]
            if len(matrix) == 0:
                distances = np.empty((0, actual_k), dtype="float64")
            else:
                distances, _ = model.kneighbors(matrix)
            if actual_k < self.neighbors:
                padding = np.full(
                    (len(matrix), self.neighbors - actual_k),
                    fill_distance,
                    dtype="float64",
                )
                distances = np.concatenate([distances, padding], axis=1)
            cumulative = np.cumsum(distances, axis=1)
            for k in range(1, self.neighbors + 1):
                output[f"knn_c{class_index}_k{k}"] = cumulative[:, k - 1]
        return pd.DataFrame(output, index=frame.index)

    def fit_transform_oof(self, frame: pd.DataFrame, y: object) -> pd.DataFrame:
        """Fit on ``frame`` and return inner-OOF training-row features.

        The fitted state is retained, so a subsequent :meth:`transform` uses
        the complete training frame for validation/test rows.
        """
        target = np.asarray(y)
        self.fit(frame, target)
        if len(frame) == 0:
            return pd.DataFrame(columns=self.feature_names, index=frame.index, dtype="float64")

        _, counts = np.unique(target, return_counts=True)
        minimum_class_rows = int(counts.min()) if counts.size else 0
        if minimum_class_rows < 2:
            raise DataError(
                "kNN OOF features need at least two rows per class.",
                "Disable features.knn_features or provide more rows per class.",
            )

        n_splits = min(self.folds, minimum_class_rows)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        oof = pd.DataFrame(
            np.nan,
            index=frame.index,
            columns=self.feature_names,
            dtype="float64",
        )
        for train_idx, valid_idx in splitter.split(np.zeros(len(frame)), target):
            fold = KNNClassDistanceTransformer(
                self.columns,
                neighbors=self.neighbors,
                folds=self.folds,
                metric=self.metric,
                max_reference_rows=self.max_reference_rows,
                random_state=self.random_state,
            )
            fold.fit(frame.iloc[train_idx], target[train_idx])
            part = fold.transform(frame.iloc[valid_idx])
            part = part.reindex(columns=self.feature_names)
            # A fold can theoretically miss a class only if the caller supplied
            # a degenerate target; use zero rather than leaking the full-fit
            # value into the missing column.
            part = part.fillna(0.0)
            oof.iloc[valid_idx, :] = part.to_numpy(dtype="float64")
        return oof

    def fit_transform(self, frame: pd.DataFrame, y: object) -> pd.DataFrame:
        """Fit and transform the same rows (useful for inference diagnostics)."""
        return self.fit(frame, y).transform(frame)


__all__ = ["KNNClassDistanceTransformer"]
