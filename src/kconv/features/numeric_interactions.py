"""Label-free pairwise numeric interaction features.

The transformer expands a small set of numeric columns into deterministic
products, ratios, differences and sums.  It never reads the target and needs
no fitted statistics beyond the selected column list, so it is safe inside an
outer cross-validation fold.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from kconv.errors import DataError

_OPERATIONS = ("product", "ratio", "difference", "sum")
_SUFFIX = {
    "product": "product",
    "ratio": "ratio",
    "difference": "diff",
    "sum": "sum",
}


class NumericInteractionTransformer:
    """Build deterministic pairwise numeric interaction columns.

    Parameters
    ----------
    columns:
        Numeric columns to interact.  ``None`` or an empty sequence selects
        every numeric column in the fitted frame.
    max_pairs:
        Maximum number of column pairs, in ``1..50``.
    operations:
        Non-empty subset of ``product``, ``ratio``, ``difference`` and
        ``sum``.  Duplicates are removed while preserving order.
    clip_value:
        Symmetric finite clip bound for every output.
    """

    def __init__(
        self,
        columns: Sequence[object] | None = None,
        *,
        max_pairs: int = 6,
        operations: Sequence[str] = ("product", "ratio"),
        clip_value: float = 1e12,
    ) -> None:
        if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or not 1 <= max_pairs <= 50:
            raise ValueError("max_pairs must be an integer in 1..50")
        if isinstance(clip_value, bool) or not np.isfinite(float(clip_value)) or float(clip_value) <= 0:
            raise ValueError("clip_value must be a finite positive number")
        if isinstance(operations, str):
            operations = [operations]
        selected_operations = list(dict.fromkeys(str(operation) for operation in operations))
        if not selected_operations:
            raise ValueError("operations must be a non-empty subset of product/ratio/difference/sum")
        unknown = [operation for operation in selected_operations if operation not in _OPERATIONS]
        if unknown:
            raise ValueError(
                "unknown numeric interaction operation(s): " + ", ".join(unknown)
            )
        selected_columns = None if columns is None else list(columns)
        self.columns: list[object] | None = selected_columns if selected_columns else None
        self.max_pairs = int(max_pairs)
        self.operations = selected_operations
        self.clip_value = float(clip_value)
        self.feature_names: list[str] = []
        self._pairs: list[tuple[object, object]] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # input preparation
    # ------------------------------------------------------------------

    def _select_columns(self, frame: pd.DataFrame) -> list[object]:
        if frame.columns.duplicated().any():
            duplicates = sorted({str(column) for column in frame.columns[frame.columns.duplicated()]})
            raise DataError(
                f"Duplicate column(s) for numeric interactions: {', '.join(duplicates)}.",
                "Rename duplicate columns before enabling numeric interactions.",
            )
        if self.columns is not None:
            columns = list(self.columns)
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise DataError(
                    f"Numeric interaction column(s) not present: {', '.join(map(str, missing))}.",
                    "Check features.numeric_interactions.columns against the dataset schema.",
                )
            for column in columns:
                try:
                    pd.to_numeric(frame[column], errors="raise")
                except (TypeError, ValueError) as exc:
                    raise DataError(
                        f"Numeric interaction column '{column}' is not numeric-convertible.",
                        "Use numeric columns or encode categoricals first.",
                    ) from exc
        else:
            columns = [
                column
                for column in frame.columns
                if pd.api.types.is_numeric_dtype(frame[column])
                and not pd.api.types.is_bool_dtype(frame[column])
            ]
        if len(columns) < 2:
            raise DataError(
                "Numeric interactions need at least two numeric columns.",
                "Select at least two numeric columns or disable numeric_interactions.",
            )
        return columns

    @staticmethod
    def _matrix(frame: pd.DataFrame, columns: Sequence[object]) -> np.ndarray:
        try:
            values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise DataError(
                "Numeric interactions could not convert the selected columns to numeric.",
                "Use numeric columns or encode categoricals first.",
            ) from exc
        return values.to_numpy(dtype="float64", copy=False)

    def _finite_values(self, values: np.ndarray) -> np.ndarray:
        return np.nan_to_num(
            values,
            nan=0.0,
            posinf=self.clip_value,
            neginf=-self.clip_value,
        )

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def fit(self, frame: pd.DataFrame, y: object = None) -> NumericInteractionTransformer:
        """Record the interaction pairs on a training frame without a target."""
        del y  # sklearn-style signature; the transform is label-free
        selected = self._select_columns(frame)
        if len(frame) == 0:
            raise DataError(
                "Numeric interactions received an empty training frame.",
                "Disable the numeric_interactions block for empty folds.",
            )
        self.columns = selected
        self._pairs = list(combinations(selected, 2))[: self.max_pairs]
        self.feature_names = [
            f"{left}__x__{right}__{_SUFFIX[operation]}"
            for left, right in self._pairs
            for operation in self.operations
        ]
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # transformations
    # ------------------------------------------------------------------

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return pairwise interaction columns for ``frame``."""
        if not self._fitted:
            raise DataError(
                "NumericInteractionTransformer.transform() called before fit().",
                "Fit the transformer on the training fold first.",
            )
        if len(frame) == 0:
            return pd.DataFrame(columns=self.feature_names, index=frame.index, dtype="float64")
        selected = self._select_columns(frame)
        if list(selected) != list(self.columns or []):
            raise DataError(
                "Numeric interaction transform frame has a different numeric column set "
                "than the fitted frame.",
                "Keep the dataset schema stable between fit and transform.",
            )
        values = self._finite_values(self._matrix(frame, selected))
        positions = {column: index for index, column in enumerate(selected)}
        output: dict[str, np.ndarray] = {}
        for left, right in self._pairs:
            left_values = values[:, positions[left]]
            right_values = values[:, positions[right]]
            for operation in self.operations:
                name = f"{left}__x__{right}__{_SUFFIX[operation]}"
                if operation == "product":
                    result = left_values * right_values
                elif operation == "ratio":
                    result = np.divide(
                        left_values,
                        right_values,
                        out=np.zeros_like(left_values),
                        where=np.abs(right_values) > 0,
                    )
                elif operation == "difference":
                    result = left_values - right_values
                else:  # sum
                    result = left_values + right_values
                result = self._finite_values(np.asarray(result, dtype="float64"))
                output[name] = np.clip(result, -self.clip_value, self.clip_value)
        return pd.DataFrame(output, index=frame.index)[self.feature_names]

    def fit_transform(self, frame: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """Fit on ``frame`` and return the interaction features for its rows."""
        return self.fit(frame, y).transform(frame)


__all__ = ["NumericInteractionTransformer"]
