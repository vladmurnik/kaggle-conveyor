"""Unsupervised IsolationForest anomaly score features.

The transformer is label-free: it fits one IsolationForest on a training fold
with median imputation and emits the anomaly score plus an optional binary
outlier flag for validation/test rows.

Scikit-learn's ``score_samples`` is the opposite of the anomaly score from the
original isolation forest paper, so the emitted feature is ``-score_samples``
and is therefore higher for more anomalous rows.  The optional flag marks the
rows the fitted model calls outliers (``predict == -1``).

Like the other fold-safe transformers, the fitted imputation statistics and
model are used unchanged on later frames; a row never influences its own score.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from kconv.errors import DataError


class AnomalyFeatureTransformer:
    """Fit one IsolationForest and emit anomaly score/flag features.

    Parameters
    ----------
    columns:
        Numeric columns to use.  ``None`` or an empty sequence selects every
        numeric column in the frame; explicit names must exist and be numeric
        convertibles.
    n_estimators:
        Number of IsolationForest trees.  Must be an integer in 10..500.
    contamination:
        Expected outlier fraction.  ``"auto"`` or a float in (0, 0.5].
    random_state:
        Seed for the deterministic IsolationForest.
    include_flag:
        Also emit ``anomaly__flag`` with ``1.0`` for outliers and ``0.0``
        otherwise.
    """

    def __init__(
        self,
        columns: Sequence[object] | None = None,
        *,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        random_state: int = 42,
        include_flag: bool = True,
    ) -> None:
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or not 10 <= n_estimators <= 500:
            raise ValueError("n_estimators must be an integer in 10..500")
        if contamination != "auto":
            contamination_value = float(contamination)
            if not 0.0 < contamination_value <= 0.5:
                raise ValueError("contamination must be 'auto' or in (0, 0.5]")
        if not isinstance(include_flag, bool):
            raise ValueError("include_flag must be a boolean")
        selected_columns = None if columns is None else list(columns)
        self.columns: Sequence[object] | None = selected_columns if selected_columns else None
        self.n_estimators = int(n_estimators)
        self.contamination = "auto" if contamination == "auto" else float(contamination)
        self.random_state = int(random_state)
        self.include_flag = bool(include_flag)
        self.feature_names: list[str] = []
        self._model: IsolationForest | None = None
        self._fill = np.array([])
        self._fitted = False

    # ------------------------------------------------------------------
    # input preparation
    # ------------------------------------------------------------------

    def _select_columns(self, frame: pd.DataFrame) -> list[object]:
        if frame.columns.duplicated().any():
            duplicates = sorted({str(column) for column in frame.columns[frame.columns.duplicated()]})
            raise DataError(
                f"Duplicate column(s) for anomaly features: {', '.join(duplicates)}.",
                "Rename duplicate columns before enabling anomaly features.",
            )
        if self.columns is not None:
            columns = list(self.columns)
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise DataError(
                    f"Anomaly feature column(s) not present: {', '.join(map(str, missing))}.",
                    "Check features.anomaly.columns against the dataset schema.",
                )
        else:
            columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
        if not columns:
            raise DataError(
                "Anomaly features need at least one numeric column.",
                "Select numeric columns or disable the anomaly block.",
            )
        return columns

    @staticmethod
    def _matrix(frame: pd.DataFrame, columns: Sequence[object]) -> np.ndarray:
        try:
            values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise DataError(
                "Anomaly features could not convert the selected columns to numeric.",
                "Use numeric columns or encode categoricals first.",
            ) from exc
        return values.to_numpy(dtype="float64", copy=False)

    def _prepare(self, frame: pd.DataFrame, columns: list[object]) -> np.ndarray:
        if not self._fitted:
            raise DataError(
                "AnomalyFeatureTransformer.transform() called before fit().",
                "Fit the transformer on the training fold first.",
            )
        values = self._matrix(frame, columns)
        values = np.where(np.isfinite(values), values, self._fill)
        return values

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def fit(self, frame: pd.DataFrame, y: object = None) -> AnomalyFeatureTransformer:
        """Fit the IsolationForest on the training frame without a target."""
        del y  # sklearn-style signature; the transform is unsupervised
        selected = self._select_columns(frame)
        if len(frame) == 0:
            raise DataError(
                "Anomaly features received an empty training frame.",
                "Disable the anomaly block for empty folds.",
            )
        values = self._matrix(frame, selected)
        fill = np.nanmedian(values, axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        values = np.where(np.isfinite(values), values, fill)
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,
        )
        model.fit(values)
        self.columns = selected
        self._model = model
        self._fill = fill
        self.feature_names = ["anomaly__score"] + (["anomaly__flag"] if self.include_flag else [])
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # transformations
    # ------------------------------------------------------------------

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return anomaly scores and the optional outlier flag for ``frame``."""
        if not self._fitted or self._model is None:
            raise DataError(
                "AnomalyFeatureTransformer.transform() called before fit().",
                "Fit the transformer on the training fold first.",
            )
        if len(frame) == 0:
            return pd.DataFrame(columns=self.feature_names, index=frame.index, dtype="float64")
        selected = self._select_columns(frame)
        if list(selected) != list(self.columns or []):
            raise DataError(
                "Anomaly transform frame has a different numeric column set than the fitted frame.",
                "Keep the dataset schema stable between fit and transform.",
            )
        values = self._prepare(frame, selected)
        output: dict[str, np.ndarray] = {"anomaly__score": -self._model.score_samples(values)}
        if self.include_flag:
            output["anomaly__flag"] = (self._model.predict(values) == -1).astype("float64")
        return pd.DataFrame(output, index=frame.index)[list(self.feature_names)]

    def fit_transform(self, frame: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """Fit on ``frame`` and return the anomaly features for the same rows."""
        return self.fit(frame, y).transform(frame)


__all__ = ["AnomalyFeatureTransformer"]
