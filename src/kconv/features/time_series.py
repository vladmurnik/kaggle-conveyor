"""Past-only grouped time-series features.

The transformer emits lags, trailing rolling statistics, elapsed time and
inter-row time differences.  ``fit`` stores a bounded per-group history tail;
``transform`` prepends that history before computing features, so validation
and test rows only see rows that occurred at or before their timestamp.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from kconv.errors import DataError

_STATS = ("mean", "std", "min", "max")


class TimeSeriesFeatureTransformer:
    """Build grouped lag/rolling/elapsed/diff features from a time column.

    Parameters
    ----------
    time_col:
        Column parsed with :func:`pandas.to_datetime`; unparseable values are
        rejected.
    group_col:
        Optional grouping column.  ``None`` treats the whole frame as one
        group.
    columns:
        Numeric value columns.  ``None`` or an empty sequence selects every
        numeric column except ``time_col`` and ``group_col``.
    lags:
        Non-empty lag list, each in ``1..100``.
    windows:
        Non-empty rolling-window list, each in ``2..100``.  Rolling statistics
        use past rows only.
    stats:
        Non-empty subset of ``mean``, ``std``, ``min`` and ``max``.
    include_elapsed:
        Emit ``ts__elapsed_seconds`` from the fitted per-group first timestamp.
    include_diff:
        Emit ``ts__diff_seconds`` from the previous timestamp in the group.
    max_history_rows:
        Global cap on the stored history rows used by later transforms.
    """

    def __init__(
        self,
        time_col: str,
        *,
        group_col: str | None = None,
        columns: Sequence[object] | None = None,
        lags: Sequence[int] = (1,),
        windows: Sequence[int] = (3,),
        stats: Sequence[str] = ("mean",),
        include_elapsed: bool = True,
        include_diff: bool = True,
        max_history_rows: int = 10_000,
    ) -> None:
        if not isinstance(time_col, str) or not time_col.strip():
            raise ValueError("time_col must be a non-empty column name")
        if group_col is not None and (not isinstance(group_col, str) or not group_col.strip()):
            raise ValueError("group_col must be None or a non-empty column name")
        selected_lags = list(dict.fromkeys(lags))
        if not selected_lags or any(
            isinstance(lag, bool) or not isinstance(lag, int) or not 1 <= lag <= 100
            for lag in selected_lags
        ):
            raise ValueError("lags must be a non-empty list of integers in 1..100")
        selected_windows = list(dict.fromkeys(windows))
        if not selected_windows or any(
            isinstance(window, bool) or not isinstance(window, int) or not 2 <= window <= 100
            for window in selected_windows
        ):
            raise ValueError("windows must be a non-empty list of integers in 2..100")
        selected_stats = list(dict.fromkeys(stats))
        if not selected_stats or any(stat not in _STATS for stat in selected_stats):
            raise ValueError("stats must be a non-empty subset of mean/std/min/max")
        if (
            isinstance(max_history_rows, bool)
            or not isinstance(max_history_rows, int)
            or not 100 <= max_history_rows <= 1_000_000
        ):
            raise ValueError("max_history_rows must be an integer in 100..1000000")
        if not isinstance(include_elapsed, bool) or not isinstance(include_diff, bool):
            raise ValueError("include_elapsed and include_diff must be booleans")
        selected_columns = None if columns is None else list(columns)
        self.time_col = time_col
        self.group_col = group_col
        self.columns: list[object] | None = selected_columns if selected_columns else None
        self.lags = selected_lags
        self.windows = selected_windows
        self.stats = selected_stats
        self.include_elapsed = include_elapsed
        self.include_diff = include_diff
        self.max_history_rows = int(max_history_rows)
        self.feature_names: list[str] = []
        self._history: pd.DataFrame | None = None
        self._group_first: dict[str, int] = {}
        self._global_first: int | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # input preparation
    # ------------------------------------------------------------------

    def _select_columns(self, frame: pd.DataFrame) -> list[object]:
        if frame.columns.duplicated().any():
            duplicates = sorted({str(column) for column in frame.columns[frame.columns.duplicated()]})
            raise DataError(
                f"Duplicate column(s) for time-series features: {', '.join(duplicates)}.",
                "Rename duplicate columns before enabling time_series.",
            )
        if self.time_col not in frame.columns:
            raise DataError(
                f"Time-series time column '{self.time_col}' is not present.",
                "Use time_col: auto or a datetime feature column.",
            )
        if self.group_col is not None:
            if self.group_col not in frame.columns:
                raise DataError(
                    f"Time-series group column '{self.group_col}' is not present.",
                    "Use group_col: auto or a feature column name.",
                )
            if self.group_col == self.time_col:
                raise DataError(
                    "Time-series group_col and time_col must be different columns.",
                    "Use a categorical group column or no group column.",
                )
        if self.columns is not None:
            columns = list(self.columns)
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise DataError(
                    f"Time-series feature column(s) not present: {', '.join(map(str, missing))}.",
                    "Check features.time_series.columns against the dataset schema.",
                )
            if self.time_col in columns or (self.group_col is not None and self.group_col in columns):
                raise DataError(
                    "Time-series value columns must not include time_col or group_col.",
                    "Use numeric value columns only.",
                )
            for column in columns:
                try:
                    pd.to_numeric(frame[column], errors="raise")
                except (TypeError, ValueError) as exc:
                    raise DataError(
                        f"Time-series column '{column}' is not numeric-convertible.",
                        "Use numeric value columns or encode categoricals first.",
                    ) from exc
        else:
            excluded = {self.time_col}
            if self.group_col is not None:
                excluded.add(self.group_col)
            columns = [
                column
                for column in frame.columns
                if column not in excluded
                and pd.api.types.is_numeric_dtype(frame[column])
                and not pd.api.types.is_bool_dtype(frame[column])
            ]
        if not columns:
            raise DataError(
                "Time-series features need at least one numeric value column.",
                "Add a numeric column or disable the time_series block.",
            )
        return columns

    def _parse_time(self, frame: pd.DataFrame) -> np.ndarray:
        parsed = pd.to_datetime(frame[self.time_col], errors="coerce", utc=True)
        if bool(pd.isna(parsed).any()):
            raise DataError(
                f"Time-series time column '{self.time_col}' contains invalid or missing timestamps.",
                "Clean the time column before enabling time_series.",
            )
        # pandas 3 may keep microsecond resolution; normalize explicitly so
        # elapsed/diff features are always expressed in seconds.
        parsed = parsed.dt.tz_localize(None).astype("datetime64[ns]")
        return parsed.astype("int64").to_numpy(dtype="int64")

    def _group_values(self, frame: pd.DataFrame) -> pd.Series:
        if self.group_col is None:
            return pd.Series("__all__", index=frame.index, dtype="object")
        return frame[self.group_col].astype("string").fillna("__missing__").astype(str)

    def _work(self, frame: pd.DataFrame) -> pd.DataFrame:
        time_values = self._parse_time(frame)
        group_values = self._group_values(frame)
        try:
            numeric = frame.loc[:, list(self.columns or [])].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise DataError(
                "Time-series features could not convert the selected columns to numeric.",
                "Use numeric value columns or encode categoricals first.",
            ) from exc
        work = pd.DataFrame(
            {
                "__group": group_values.to_numpy(),
                "__time": time_values,
                "__order": np.arange(len(frame), dtype=np.int64),
                "__row": np.arange(len(frame), dtype=np.int64),
            }
        )
        for column in self.columns or []:
            work[column] = numeric[column].to_numpy(dtype="float64")
        return work

    # ------------------------------------------------------------------
    # fitting and state
    # ------------------------------------------------------------------

    def fit(self, frame: pd.DataFrame, y: object = None) -> TimeSeriesFeatureTransformer:
        """Record the value columns, per-group history and first timestamps."""
        del y  # sklearn-style signature; the transform is label-free
        selected = self._select_columns(frame)
        if len(frame) == 0:
            raise DataError(
                "Time-series features received an empty training frame.",
                "Disable the time_series block for empty folds.",
            )
        self.columns = selected
        self.feature_names = self._feature_names()
        work = self._work(frame)
        self._store_history(work)
        self._fitted = True
        return self

    def _feature_names(self) -> list[str]:
        names = [
            f"ts__{column}__lag_{lag}"
            for column in self.columns or []
            for lag in self.lags
        ]
        names.extend(
            f"ts__{column}__{stat}_{window}"
            for column in self.columns or []
            for window in self.windows
            for stat in self.stats
        )
        if self.include_elapsed:
            names.append("ts__elapsed_seconds")
        if self.include_diff:
            names.append("ts__diff_seconds")
        return names

    def _store_history(self, work: pd.DataFrame) -> None:
        need = max(max(self.lags), max(self.windows), 1)
        ordered = work.sort_values(["__group", "__time", "__order"], kind="mergesort")
        history = ordered.groupby("__group", sort=False, group_keys=False).tail(need)
        if len(history) > self.max_history_rows:
            history = history.sort_values(
                ["__time", "__group", "__order"], kind="mergesort"
            ).tail(self.max_history_rows)
        self._history = history.copy()
        group_first = ordered.groupby("__group", sort=False)["__time"].min()
        self._group_first = {str(key): int(value) for key, value in group_first.items()}
        self._global_first = int(ordered["__time"].min())

    # ------------------------------------------------------------------
    # transformations
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling(values: pd.Series, window: int, stat: str) -> pd.Series:
        shifted = values.shift(1)
        rolling = shifted.rolling(window, min_periods=1)
        if stat == "mean":
            return rolling.mean()
        if stat == "std":
            return rolling.std()
        if stat == "min":
            return rolling.min()
        return rolling.max()

    def _compute(
        self,
        work: pd.DataFrame,
        history: pd.DataFrame | None,
        group_first: dict[str, int],
        global_first: int,
        index: pd.Index,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        if history is not None and not history.empty:
            stored = history.copy()
            stored["__phase"] = 0
            stored["__row"] = -1
            frames.append(stored)
        current = work.copy()
        current["__phase"] = 1
        frames.append(current)
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(
            ["__group", "__time", "__phase", "__order"], kind="mergesort"
        ).reset_index(drop=True)
        grouped = combined.groupby("__group", sort=False, group_keys=False)
        for column in self.columns or []:
            for lag in self.lags:
                name = f"ts__{column}__lag_{lag}"
                combined[name] = grouped[column].shift(lag).fillna(0.0)
            for window in self.windows:
                for stat in self.stats:
                    name = f"ts__{column}__{stat}_{window}"
                    combined[name] = grouped[column].transform(
                        lambda values, window=window, stat=stat: self._rolling(
                            values, window, stat
                        )
                    )
                    combined[name] = combined[name].fillna(0.0)
        if self.include_elapsed:
            first = combined["__group"].map(group_first).astype("float64")
            current_first = work.groupby("__group", sort=False)["__time"].min()
            first = first.fillna(
                combined["__group"].map(current_first).astype("float64")
            )
            first = first.fillna(float(global_first))
            elapsed = (combined["__time"].astype("float64") - first) / 1_000_000_000.0
            combined["ts__elapsed_seconds"] = elapsed.clip(lower=0.0).fillna(0.0)
        if self.include_diff:
            previous = combined.groupby("__group", sort=False)["__time"].shift(1)
            difference = (
                combined["__time"].astype("float64")
                - previous.astype("float64")
            ) / 1_000_000_000.0
            combined["ts__diff_seconds"] = difference.fillna(0.0).clip(lower=0.0)
        output = combined[combined["__phase"] == 1].copy()
        output = output.sort_values("__row", kind="mergesort")
        positions = output["__row"].to_numpy(dtype=np.int64)
        output.index = index[positions]
        return output[self.feature_names]

    def fit_transform(self, frame: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """Fit state on ``frame`` and return training-row time-series features."""
        self.fit(frame, y)
        work = self._work(frame)
        if self._global_first is None:
            raise DataError(
                "Time-series fit state is missing.",
                "This is an internal error; fit the transformer first.",
            )
        return self._compute(
            work,
            None,
            self._group_first,
            self._global_first,
            frame.index,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return time-series features for new rows using fitted history."""
        if not self._fitted:
            raise DataError(
                "TimeSeriesFeatureTransformer.transform() called before fit().",
                "Fit the transformer on the training fold first.",
            )
        if len(frame) == 0:
            return pd.DataFrame(columns=self.feature_names, index=frame.index, dtype="float64")
        selected = self._select_columns(frame)
        if list(selected) != list(self.columns or []):
            raise DataError(
                "Time-series transform frame has a different value column set than the fitted frame.",
                "Keep the dataset schema stable between fit and transform.",
            )
        work = self._work(frame)
        if self._global_first is None:
            raise DataError(
                "Time-series fit state is missing.",
                "This is an internal error; fit the transformer first.",
            )
        return self._compute(
            work,
            self._history,
            self._group_first,
            self._global_first,
            frame.index,
        )


__all__ = ["TimeSeriesFeatureTransformer"]
