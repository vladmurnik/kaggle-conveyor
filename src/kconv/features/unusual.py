"""Cheap, fold-safe feature transforms for triaged unusual patterns.

These helpers are deliberately standalone and dependency-light.  They are not
wired into :class:`kconv.features.pipeline.FeatureBuilder` yet; the experiment
catalog marks the ones with a reusable implementation as ``prototype`` and the
benchmark script exercises them on real UCI data.

Leakage rule: :class:`GaussianRankTransformer` is fitted on a training fold and
then applied unchanged to validation/test frames.  It never ranks train and
test jointly, unlike the source notebook that inspired UP-03.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from kconv.errors import KconvError

_ROW_STATS = {"mean", "std", "min", "max", "median"}


def _check_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise KconvError("Frame must be a pandas DataFrame.", "Pass the dataset frame to transform.")


def _numeric_series(frame: pd.DataFrame, column: object) -> pd.Series:
    if column not in frame.columns:
        raise KconvError(
            f"Column '{column}' is not present in the frame.",
            "Check the experiment config against the dataset schema.",
        )
    series = frame[column]
    if isinstance(series, pd.DataFrame):
        raise KconvError(f"Column '{column}' is duplicated.", "Rename duplicate columns first.")
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        raise KconvError(
            f"Column '{column}' is a date/time column.",
            "Derive numeric dates before adding modulo or rank features.",
        )
    try:
        return pd.to_numeric(series, errors="coerce").astype("float64")
    except (TypeError, ValueError) as exc:
        raise KconvError(
            f"Column '{column}' cannot be converted to numeric.",
            "Only numeric columns are supported by this prototype.",
        ) from exc


def _columns(frame: pd.DataFrame, columns: Iterable[object] | None) -> list[object]:
    selected = (
        list(columns)
        if columns is not None
        else [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    )
    if not selected:
        raise KconvError(
            "No numeric columns selected.",
            "Pass explicit numeric columns or include numeric features in the frame.",
        )
    return selected


class GaussianRankTransformer:
    """Map numeric columns to normal scores using a fitted training reference.

    Parameters
    ----------
    columns:
        Numeric columns to transform.  ``None`` selects all numeric columns.
    suffix:
        Suffix appended to each source column; the original column is retained.

    Notes
    -----
    Rank-to-uniform-to-normal is monotone, so trees are unchanged by it in
    theory; it mainly helps linear, distance-based and neural models.  The
    transformer stores only the sorted training values, one array per column.
    """

    def __init__(self, columns: Iterable[object] | None = None, *, suffix: str = "__gauss_rank") -> None:
        self.columns = None if columns is None else list(columns)
        self.suffix = suffix
        self._reference: dict[object, np.ndarray] = {}
        self._fitted = False

    def _rank(self, values: pd.Series, reference: np.ndarray) -> pd.Series:
        if reference.size == 0:
            return pd.Series(0.5, index=values.index, dtype="float64")
        raw = values.to_numpy(dtype="float64", copy=False)
        finite = np.isfinite(raw)
        ranks = np.full(len(raw), 0.5, dtype="float64")
        # searchsorted gives the number of reference values below each value;
        # the midpoint adjustment keeps ties and the endpoints away from 0/1.
        counts = np.searchsorted(reference, raw[finite], side="left").astype("float64")
        ranks[finite] = (counts + 0.5) / (reference.size + 1.0)
        ranks = np.clip(ranks, 1e-6, 1.0 - 1e-6)
        return pd.Series(norm.ppf(ranks), index=values.index, dtype="float64")

    def fit(self, frame: pd.DataFrame, y: object = None) -> GaussianRankTransformer:
        """Fit the empirical reference distribution on a training fold."""
        del y  # sklearn-style signature; the transform is label-free
        _check_frame(frame)
        columns = _columns(frame, self.columns)
        reference: dict[object, np.ndarray] = {}
        for column in columns:
            values = _numeric_series(frame, column)
            finite = values.to_numpy(dtype="float64", copy=False)
            finite = finite[np.isfinite(finite)]
            reference[column] = np.sort(finite)
        self.columns = columns
        self._reference = reference
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with one normal-score column per fitted source column."""
        _check_frame(frame)
        if not self._fitted:
            raise KconvError(
                "GaussianRankTransformer.transform() called before fit().",
                "Fit the transformer on the training fold first.",
            )
        new_columns: dict[str, pd.Series] = {}
        for column, reference in self._reference.items():
            values = _numeric_series(frame, column)
            new_columns[f"{column}{self.suffix}"] = self._rank(values, reference)
        return pd.concat([frame.copy(), pd.DataFrame(new_columns, index=frame.index)], axis=1)

    def fit_transform(self, frame: pd.DataFrame, y: object = None) -> pd.DataFrame:
        return self.fit(frame, y).transform(frame)


def add_modulo_features(
    frame: pd.DataFrame,
    columns: Iterable[object] | None = None,
    *,
    moduli: Sequence[int] = (10, 100),
    steps: Sequence[int] = (5,),
) -> pd.DataFrame:
    """Add ``x % m`` and ``x // step`` views of numeric ID-like columns.

    Modulo features are a memorization vector on high-cardinality identifiers.
    The caller must accept them on a fixed split; this helper does not decide
    whether the column is an identifier.
    """
    _check_frame(frame)
    if not moduli and not steps:
        raise KconvError("At least one modulus or step is required.")
    if any(int(value) <= 0 for value in moduli) or any(int(value) <= 0 for value in steps):
        raise KconvError("moduli and steps must be positive integers.")
    result = frame.copy()
    new_columns: dict[str, pd.Series] = {}
    for column in _columns(frame, columns):
        values = _numeric_series(frame, column)
        for modulus in moduli:
            new_columns[f"{column}__mod_{int(modulus)}"] = values % int(modulus)
        for step in steps:
            new_columns[f"{column}__div_{int(step)}"] = np.floor(values / int(step))
    return pd.concat([result, pd.DataFrame(new_columns, index=frame.index)], axis=1)


def add_row_statistics(
    frame: pd.DataFrame,
    columns: Iterable[object] | None = None,
    *,
    stats: Sequence[str] = ("mean", "std", "min", "max"),
    prefix: str = "row",
) -> pd.DataFrame:
    """Add row-wise aggregates over numeric feature columns.

    ``std`` uses ``ddof=0`` and is zero for one selected column.  Missing
    values are skipped by pandas reductions, exactly like the benchmark.
    """
    _check_frame(frame)
    selected = _columns(frame, columns)
    requested = [str(stat).lower() for stat in stats]
    unknown = set(requested) - _ROW_STATS
    if unknown:
        raise KconvError(
            f"Unknown row statistic(s): {', '.join(sorted(unknown))}.",
            f"Use one of: {', '.join(sorted(_ROW_STATS))}.",
        )
    if not requested:
        raise KconvError("At least one row statistic is required.")
    matrix = pd.concat([_numeric_series(frame, column) for column in selected], axis=1)
    matrix.columns = selected
    result = frame.copy()
    mapping = {
        "mean": matrix.mean(axis=1),
        "std": matrix.std(axis=1, ddof=0),
        "min": matrix.min(axis=1),
        "max": matrix.max(axis=1),
        "median": matrix.median(axis=1),
    }
    for stat in dict.fromkeys(requested):
        result[f"{prefix}_{stat}"] = mapping[stat].astype("float64")
    return result


__all__ = [
    "GaussianRankTransformer",
    "add_modulo_features",
    "add_row_statistics",
]
